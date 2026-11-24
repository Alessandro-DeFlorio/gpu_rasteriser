# Build and toolchain notes

Verified on the dev machine (2026-11-23):
- nvcc 13.5 (`/usr/local/cuda`), driver 580, GPU RTX PRO 1000 Blackwell laptop (`sm_120`).
- conda env `tetvox-dev`: Python 3.12, nanobind 2.x, scikit-build-core 1.1.

## How the build works
- `CMakeLists.txt` builds two things (ADR 0004):
  - the C++ package. `tetvox_core` is a static library (`csrc/src/*.cu`, public headers in
    `csrc/include/tetvox`). It is exported as `tetvox::tetvox` (file `libtetvox.a`), with
    `tetvoxConfig.cmake` from `cmake/tetvoxConfig.cmake.in`. The install rules are in component
    `cpp`.
  - the nanobind module `tetvox._core`, only with `TETVOX_BUILD_PYTHON`. The option defaults to ON
    when `SKBUILD` is defined and to OFF in a plain CMake build, which then needs neither Python nor
    nanobind. The module links `tetvox_core` and is installed into the `tetvox` package, in
    component `python`.
- Python build: `pyproject.toml` → scikit-build-core → `CMakeLists.txt`.
  `install.components = ["python"]` keeps headers, the static lib and the CMake config out of the
  wheel.
- The version is set once, in `project(tetvox VERSION ...)`. `pyproject.toml` reads it from there
  with the scikit-build-core regex metadata provider.
- Public headers are globbed (`CONFIGURE_DEPENDS`) into a `FILE_SET HEADERS`, so a new header is
  installed without editing CMake. Sources are listed explicitly.
- C++20 reaches consumers through `target_compile_features(... PUBLIC cxx_std_20 cuda_std_20)`.
  There are no global `CMAKE_CXX_STANDARD` variables.
- Editable install (`pip install --no-build-isolation -ve .`) uses `editable.rebuild = true`. Importing
  `tetvox` re-runs `cmake --build` in `build/<wheel_tag>/editable`, so edits to C++/CUDA are picked up on the
  next `import tetvox` without reinstalling. Reinstall only after changing `pyproject.toml` or
  adding/removing CMake targets in a way the existing build dir cannot handle.
- `--no-build-isolation` reuses nanobind and scikit-build-core from the env (fast, offline).
- CUDA architectures default to `native`. Pass
  `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES="..."` for portable builds. CUDA 13 supports `sm_75`
  and newer only.
- cudart: only `_core` sets `CUDA_RUNTIME_LIBRARY Static`, to avoid clashing with torch's bundled
  runtime (ADR 0001). The C++ target leaves the choice to consumers (ADR 0004).

## CUDA or CPU-only (ADR 0007/0008)
- `TETVOX_ENABLE_CUDA` (default ON). `project()` lists only CXX; CUDA is enabled under the option.
  With ON and no CUDA compiler, configuration stops with CMake's own error (provisional; owner
  question pending, see ROADMAP). OFF builds `csrc/src/device_cpu_only.cpp` instead of
  `device.cu`, so `device_count()` returns 0.
- `cmake/config.hpp.in` → `<build>/include/tetvox/config.hpp`, installed with the other headers:
  `TETVOX_CUDA_ENABLED` (0/1, `#cmakedefine01`) and `tetvox::cuda_enabled`. Python reads
  `tetvox.cuda_enabled` from `_core`.
- CPU-only Python build: `pip install . -Ccmake.define.TETVOX_ENABLE_CUDA=OFF`. (Verified
  2026-11-24) the module then imports, reports `cuda_enabled False`, `device_count() 0`, and links
  no CUDA library.
- The committed `_core.pyi` records the default (CUDA) build: stubgen writes
  `cuda_enabled: bool = True`. A CPU-only build would generate `= False`; the stub gate runs only on
  the default build.

## C++ consumers
- `scripts/check.sh` builds the library with plain CMake in `build/cpp/core`, installs component
  `cpp` into `build/cpp/install`, then builds and runs `examples/cpp` against it
  (`-DCMAKE_PREFIX_PATH=<prefix>`). It repeats this CPU-only in `build/cpp-cpu-only`.
- `tetvoxConfig.cmake` sets `tetvox_CUDA_ENABLED` before anything else. A consumer that does not
  know how tetvox was built does what `examples/cpp` does: `find_package(tetvox QUIET)` to read the
  flag, `enable_language(CUDA)` if it is set, then `find_package(tetvox REQUIRED)`.
- (Verified 2026-11-23) A consumer of a CUDA build must enable the CUDA language. Then CMake links the CUDA runtime
  it asks for: static by default, shared with `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`. In a CXX-only
  project the link fails with `undefined reference to __cudaRegisterFatBinary`. `tetvoxConfig.cmake`
  therefore refuses, with a readable message, when CUDA is not enabled.
- (Verified 2026-11-23) `add_subdirectory(<tetvox>)` from a CUDA project works through the
  `tetvox::tetvox` alias. The parent's `CMAKE_CUDA_ARCHITECTURES` takes precedence.

## Type stubs
`src/tetvox/_core.pyi` is generated and committed. Regenerate after changing bindings:

    python -m nanobind.stubgen -m tetvox._core -o src/tetvox/_core.pyi

`scripts/check.sh` fails if the committed stub is stale.

## Gotchas
- (Verified 2026-11-23) `project(... CUDA)` replaces an unset `CMAKE_CUDA_ARCHITECTURES` with the
  compiler default (`75` for nvcc 13.5). The `native` default must therefore be set *before*
  `project()`. Set after it, as the bootstrap did, it only ever applied when scikit-build passed it.
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
