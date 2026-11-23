# Build and toolchain notes

Verified on the dev machine (2026-11-23):
- nvcc 13.5 (`/usr/local/cuda`), driver 580, GPU RTX PRO 1000 Blackwell laptop (`sm_120`).
- conda env `tetvox-dev`: Python 3.12, nanobind 2.x, scikit-build-core 1.1.

## How the build works
- `pyproject.toml` → scikit-build-core → `CMakeLists.txt`.
- `tetvox_core` is a static library (`csrc/src/*.cu`, headers in `csrc/include/tetvox`). It is linked
  into the nanobind module `tetvox._core`, which is installed into the `tetvox` package.
- Editable install (`pip install --no-build-isolation -ve .`) uses `editable.rebuild = true`. Importing
  `tetvox` re-runs `cmake --build` in `build/<wheel_tag>/editable`, so edits to C++/CUDA are picked up on the
  next `import tetvox` without reinstalling. Reinstall only after changing `pyproject.toml` or
  adding/removing CMake targets in a way the existing build dir cannot handle.
- `--no-build-isolation` reuses nanobind and scikit-build-core from the env (fast, offline).
- CUDA architectures default to `native`. Pass
  `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES="..."` for portable builds. CUDA 13 supports `sm_75`
  and newer only.
- cudart is linked statically to avoid clashing with torch's bundled runtime (ADR 0001).

## Type stubs
`src/tetvox/_core.pyi` is generated and committed. Regenerate after changing bindings:

    python -m nanobind.stubgen -m tetvox._core -o src/tetvox/_core.pyi

`scripts/check.sh` fails if the committed stub is stale.

## Gotchas
- Settings under `[tool.scikit-build]` such as `editable.verbose` are baked in at install time, so
  reinstall after changing them.
- The base conda python has no numpy. Always work inside `tetvox-dev`.
- (Verified 2026-11-23, research R1.) A stand-alone `nvcc file.cu` without `-arch` builds for the
  default architecture and relies on PTX JIT. On this machine (nvcc 13.5, driver 580) the kernel
  launch then fails with `the provided PTX was compiled with an unsupported toolchain`, and
  `cudaDeviceSynchronize` still reports success. Always pass `-arch=native` (the CMake build already
  defaults to `native`) and check `cudaGetLastError()` right after a launch.
- (Verified 2026-11-23, research R1.) `__int128` works in device code with nvcc 13.5 (multiply,
  arithmetic shift, division), and so does `__mul64hi`.
- (Observed 2026-11-23, research R1; fixed the same day.) Editable and wheel builds used to share
  `build/<wheel_tag>`. A non-editable `pip install .` from another env then reconfigured that
  directory against its own interpreter, and `import tetvox` broke in `tetvox-dev`. The build dir is
  now `build/{wheel_tag}/{state}`, so `editable` and `wheel` builds are kept apart. If an editable
  import ever fails while re-running CMake, `rm -rf build` and reinstall with
  `pip install --no-build-isolation -ve .`.
