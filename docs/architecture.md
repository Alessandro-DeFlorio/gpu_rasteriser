# Architecture

Status: skeleton. Only the build chain (C++ package plus optional Python module) and the atlas
loader exist. The op contracts (ADR 0005, R2 and
its addendum) and the rasterisation rule (ADR 0006) are decided.

## Layers
1. **C++/CUDA library** (`csrc/`), a first-class interface on its own (ADR 0004). It is
   framework-agnostic: raw device pointers, a caller-provided `cudaStream_t`, caller-allocated
   outputs. Public input validation happens here, once.
   - `csrc/include/tetvox/`: header-only `__host__ __device__` maths (barycentrics, exact
     predicates, moment accumulation) and the fused-path templates (`accumulate<Cost>`), usable from
     kernels, from the CPU reference, and from users' own `.cu` files.
   - `csrc/src/`: kernels and host launchers, compiled into `tetvox_core`, which is exported as the
     CMake target `tetvox::tetvox` (`find_package(tetvox)` or `add_subdirectory`). Consumers enable
     CUDA and choose the cudart flavour themselves.
   - `examples/cpp/`: a stand-alone consumer of the installed package, built and run by
     `scripts/check.sh`.
2. **Bindings** (`csrc/bindings/module.cpp`), built only with `TETVOX_BUILD_PYTHON` (ON under
   scikit-build, OFF in a plain CMake build). Links cudart statically (ADR 0001). A
   nanobind module `tetvox._core` taking DLPack arrays (`nb::ndarray`), so numpy, torch, cupy and
   others work (ADR 0001).
3. **Python package** (`src/tetvox/`). Thin: bindings re-exported with types, `TetMesh` and
   `load_atlas_npz` in `mesh.py`, and an optional torch autograd wrapper later.

## Data flow (ADR 0005, R2)
- Inputs are in **voxel index space**. Voxel centres are the integer points of
  `[0,X)×[0,Y)×[0,Z)`. Any affine is applied to the vertices by the caller.
- Mesh = positions `(N,3)`, tetrahedra `(T,4)`, per-vertex values `(N,C)`. Priors, stiffness and
  display metadata are not part of it.

```
positions, tets ──► per-tet geometry (A⁻¹, v₀, det A) ──► deformation prior (separate op, fold check)
        │
        └──► rasterise ──► raster buffer: tet_id (int32), π₁..π₃ (3×f32)
                              │
          generic path:       ├──► interpolate(values) ──► user cost (any framework) ──► ∂L/∂f
                              │        interpolate_vjp ──► ∂L/∂values, ∂L/∂π
                              │    or a user kernel on the raster buffer ──────────► ∂L/∂π
                              │                                     bary_vjp ──► ∂L/∂v
          fused path:         └──► accumulate<Cost, Curvature> ──► cost, ∂L/∂v, (tet blocks)
```

- Contract between cost and geometry: per-voxel `∂L/∂π`, reduced to `ḡ` (3 floats). Geometry is
  applied once per tet: `G_i = −A⁻ᵀ Σ_x π_i ḡ`.
- The caller sums the data term and `κ·prior`, then applies the boundary condition to the gradient.
- Invalid meshes give `+inf` cost and a count of offending tets; the evaluation path does not raise.
