# 0004 — C++/CUDA is a first-class interface; Python is optional

Status: accepted (2026-11-23, proposed by the owner)

## Decision
tetvox is usable entirely from C++/CUDA. The Python package is one consumer of the C++ library, not
the only entry point.

- **CMake package.** The build exports a `tetvox::tetvox` target (public headers and the compiled
  kernels), consumable through `find_package(tetvox)`, `FetchContent` or `add_subdirectory`.
- **Python is optional in the build.** Python and nanobind sit behind `TETVOX_BUILD_PYTHON`. It is
  on by default under scikit-build (`SKBUILD` defined) and off for a plain CMake build, so a C++
  user needs neither Python nor nanobind.
- **Both paths exist in C++.**
  - Generic path: `rasterize`, `interpolate`, `interpolate_vjp`, `bary_vjp` are plain host
    functions on raw device pointers and a `cudaStream_t`. A C++ user can put their own kernels
    between them.
  - Fused path: `accumulate<Cost>` is a template in public `.cuh` headers. A C++ user writes a
    `Cost` struct and instantiates it in their own `.cu` file. The `Cost` interface is a C++20
    concept, so a malformed functor fails at compile time with a readable message.
- **Python gets prebuilt instantiations only** (for example the samseg data term), plus the
  generic path. Instantiating new fused costs from Python would need JIT (R2 addendum A3), which is
  not planned.
- **Input validation lives in the public C++ API.** It checks shapes, dtypes, device and value
  ranges once and throws C++ exceptions, which nanobind turns into Python exceptions. The Python
  layer does not repeat those checks; it only validates what does not exist in C++ (e.g. Python
  types, optional torch arguments). Internals still trust their invariants.
- **CUDA runtime linkage is the consumer's choice.** Only the Python module forces the static
  cudart (ADR 0001). The C++ target does not set `CMAKE_CUDA_RUNTIME_LIBRARY` for its consumers.
- **Packaging test.** A small C++ example project under `examples/cpp/` consumes the installed
  package and runs the generic path on the Kuhn-cube mesh. `scripts/check.sh` builds and runs it.
  Correctness tests stay in pytest, which goes through the same C++ entry points.

## Why
- A C++/CUDA user should not have to go through Python to use a C++/CUDA library.
- The fused path is a template. C++ users can instantiate their own costs directly, while Python
  users cannot without JIT.
- Nothing is implemented yet, so the change is cheap now and costly later.

## Consequences
- Two public APIs to document and keep stable. The Python layer stays thin: bindings, `TetMesh`
  and the atlas loader, and the optional torch `autograd.Function`s.
- The CLAUDE.md rule "validate once at the public Python API boundary" becomes "validate once at the
  public C++ API boundary".
- Public headers require C++20 from consumers (nvcc ≥ 12 supports it).
- The atlas `.npz` loader stays Python-only. C++ users pass their own arrays.
