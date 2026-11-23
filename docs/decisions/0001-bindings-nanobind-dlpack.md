# 0001 — Python bindings: nanobind + DLPack, torch optional

Status: accepted (2026-11-23)

## Decision
The C++/CUDA core takes raw (device) pointers and shapes. It is exposed through a nanobind module
`tetvox._core`, which receives arrays as `nb::ndarray` (DLPack). The core therefore works with numpy,
torch, cupy, jax or any other DLPack producer. Autograd integration (e.g. a `torch.autograd.Function`)
is a thin optional Python layer, and torch is an optional extra (`pip install tetvox[torch]`).

## Why
- No hard dependency on torch, and no build tied to torch's CUDA/ABI version.
- nanobind compiles faster and produces smaller binaries than pybind11. It has first-class ndarray support
  and generates type stubs (`_core.pyi`) for mypy.

## Consequences
- The CUDA runtime is linked statically (`CMAKE_CUDA_RUNTIME_LIBRARY Static`), so it does not clash with
  the runtime that ships in torch wheels.
- Streams and memory: the core takes the caller's CUDA stream and never owns framework tensors. The
  exact policy is still to be decided.
