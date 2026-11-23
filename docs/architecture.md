# Architecture

Status: skeleton. Only the build chain (C++ package plus optional Python module) and the atlas
loader exist. Decided but not built: the op contracts (ADR 0005, R2 and its addendum), the
rasterisation rule (ADR 0006), the CPU/CUDA backends and build options (ADR 0007, amended by ADR
0008: dtype-generic interface, window rule, validation) and the op inventory with the public shape
of the rasterisation ops (ADR 0009).

## Layers
1. **C++/CUDA library** (`csrc/`), a first-class interface on its own (ADR 0004). It is
   framework-agnostic: raw device pointers, a caller-provided `cudaStream_t`, caller-allocated
   outputs. Public input validation happens here, once.
   - `csrc/include/tetvox/`: header-only `__host__ __device__` procedures (ADR 0009: `snap`,
     `setup_tet`, `plane_values`, `owns`, `barycentric`, `row_span`; later moment accumulation) and
     the fused-path templates (`accumulate<Cost>`), usable from kernels, from the CPU backend, and
     from users' own `.cu` files. `types.hpp` holds the view structs (`MeshView<T>`,
     `TetSetupView<T>`), `GridShape`, `TetStatus` and `default_snap_bits`; `cpu.hpp` (and later
     `cuda.hpp`) declare the array ops.
   - `csrc/src/`: kernels and host launchers, compiled into `tetvox_core`, which is exported as the
     CMake target `tetvox::tetvox` (`find_package(tetvox)` or `add_subdirectory`). Consumers enable
     CUDA and choose the cudart flavour themselves.
   - Two backends with one contract (ADR 0007): `tetvox::cpu::*` (C++, OpenMP if found) and
     `tetvox::cuda::*` (kernels, `cudaStream_t`). Both call the same shared header, so their outputs
     are bit-identical. Build option `TETVOX_ENABLE_CUDA` (default ON; OFF gives a CPU-only build),
     reported to consumers by the generated `tetvox/config.hpp` (`tetvox::cuda_enabled`). Ops are
     dtype-generic: templates constrained by `Real` and explicitly instantiated for float and
     double, outputs follow the input dtype (ADR 0008/0009). The CUDA twin of each op adds only a
     trailing `cudaStream_t`. Both backends run the same per-row procedure (`row_span`): the CPU
     with OpenMP over tets, the GPU with K3 (one thread per box row). Planned, not built yet.
   - `examples/cpp/`: a stand-alone consumer of the installed package, built and run by
     `scripts/check.sh`.
2. **Bindings** (`csrc/bindings/module.cpp`), built only with `TETVOX_BUILD_PYTHON` (ON under
   scikit-build, OFF in a plain CMake build). Links cudart statically (ADR 0001). A
   nanobind module `tetvox._core` taking DLPack arrays (`nb::ndarray`), so numpy, torch, cupy and
   others work (ADR 0001).
3. **Python package** (`src/tetvox/`). Thin: one wrapper per C++ op with the same name, taking
   arrays (not `TetMesh`), with `out=` and allocation in the input's framework; dispatch on device
   and dtype is nanobind overload resolution (ADR 0009). Plus the generic container
   `TetMesh[F]` (`mesh.py`), `SamsegAtlas[F]` and `load_atlas_npz(dtype=float32)` (`samseg.py`,
   holding a `TetMesh` plus samseg model data), and an optional torch autograd wrapper later.

## Data flow (ADR 0005, R2)
- Inputs are in **voxel index space**. Voxel centres are the integer points of
  `[0,X)×[0,Y)×[0,Z)`. Any affine is applied to the vertices by the caller. Axes are numbered 0, 1
  and 2: position column `d` is array axis `d` (C order, axis 2 fastest), and the ownership
  tie-break puts axis 0 first (ADR 0007).
- Mesh (`TetMesh`) = positions `(N,3)`, tetrahedra `(T,4)` i32, per-vertex values `(N,C)`;
  positions and values share one dtype, float32 or float64 (ADR 0008/0009; today the code still uses
  f32), where `C = 0` is allowed. Priors, stiffness and display metadata are not part of it; for samseg
  they live in `SamsegAtlas`, which holds the mesh rather than subclassing it.

```
positions, tets ──► setup_tets ──► TetSetup (per tet: n_j, c_j, f_j, 2ᵏ/D, box, status)
                         │              │
                         │              └──► count_skipped ──► {inverted, outside_window}
                         │                   (deformation prior: separate op, fold check)
                         ▼
          generic path:  rasterize (= barycentrics ∘ locate) ──► tet_id (int32, −1 unowned),
                         │                                       π₁..π₃ (input dtype)
                         ├──► interpolate(values) ──► user cost (any framework) ──► ∂L/∂f
                         │      interpolate_vjp_values ──► ∂L/∂values
                         │      interpolate_vjp_bary ─────────────► ḡ = ∂L/∂π
                         │    or a user kernel on tet_id / π ─────► ḡ
                         │                         tet_moments ──► m (T,4,3) ──► vertex_grad ──► ∂L/∂v
          fused path:    └──► accumulate<Cost, Curvature> ──► cost, m (T,4,3) ──► vertex_grad ──► ∂L/∂v
          diagnostic:         count_claims ──► claims per voxel (exactly-once test)
```

- Contract between cost and geometry: per-voxel `∂L/∂π`, reduced to `ḡ` (3 floats). Geometry is
  applied once per tet by `vertex_grad`: `G_i = −A⁻ᵀ m_i` with `m_i = Σ_x π_i ḡ`, where the rows of
  `A⁻¹` come from the snapped record (`p_j = n_j · 2ᵏ/D`), so ownership, π and `∂π/∂v` share one
  geometry (ADR 0006).
- `snap_bits` (k, default 14) is an argument of `setup_tets` only; every later op reads the record.
- The caller sums the data term and `κ·prior`, then applies the boundary condition to the gradient.
- Invalid meshes: `setup_tets` marks each skipped tet in `status` (`count_skipped` gives the
  counts) and the fused cost is `+inf`; the evaluation path does not raise.
