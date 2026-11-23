# Architecture

Status: skeleton. Only the build chain and the atlas loader exist. The rasteriser design is pending
research R1/R2 (`docs/research/`).

## Layers
1. **C++/CUDA core** (`csrc/`). Framework-agnostic and operates on raw device pointers.
   - `csrc/include/tetvox/`: header-only `__host__ __device__` maths (barycentrics, fixed-point
     predicates), usable from kernels and from CPU reference code.
   - `csrc/src/`: kernels and host launchers, compiled into the static lib `tetvox_core`.
2. **Bindings** (`csrc/bindings/module.cpp`). A nanobind module `tetvox._core` that takes DLPack
   arrays (`nb::ndarray`), so numpy, torch, cupy and jax all work (ADR 0001).
3. **Python package** (`src/tetvox/`). Typed API: `TetMesh` and `load_atlas_npz` in `mesh.py`. An
   optional torch autograd wrapper will live here later.

## Planned data flow (to be confirmed by R1/R2)
mesh (vertices v, tets) → **rasterise** → per-voxel (tet id, barycentrics π) → **interpolate**
per-vertex fields (e.g. α) → user / pluggable **cost** → upstream ∂L/∂π or ∂L/∂f per voxel →
**scatter** gradient (and Hessian blocks) back to vertices using ∂π/∂v (samseg-math.md §2).
