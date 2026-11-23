# 0007 — CPU backend, `rasterize` contract, snapping and precision switch

Status: accepted (2026-11-23). The owner answered points 1–4 directly and accepted the session's
recommendations for the rest. Amended by ADR 0008: the `TETVOX_REAL` precision switch
is replaced by a dtype-generic interface (float32 and float64 at run time). Amended by ADR 0009: the
optional `visits` output becomes the op `count_claims`, the counts come from the per-tet `status`
written by `setup_tets`, and the Python call is `setup_tets` + `rasterize` instead of
`rasterize(mesh, shape)`.

Context: review of the earlier CPU rasteriser `meshing_cxx` against ADR 0005/0006
(`docs/knowledge/meshing-cxx-reference.md`). This ADR amends ADR 0005 (dtype of the stored π) and
ADR 0006 (axis naming, rounding mode, role of the CPU code).

## Decision

**Two public backends, one contract.** `tetvox::cpu::rasterize(...)` and
`tetvox::cuda::rasterize(..., cudaStream_t)` have identical signatures apart from the stream. The
CPU backend is a product, not only a test oracle. It is written first, so the contract can be
exercised and changed cheaply before the CUDA kernel exists.

**Shared maths.** Both backends call the shared `__host__ __device__` header of ADR 0006 (`snap`,
`setup_tet`, `inside`, barycentrics), so their outputs are bit-identical, π included. The CPU
backend takes its loop structure from `meshing_cxx`: a clipped bounding-box scan and a per-tet
cofactor setup. The arithmetic follows ADR 0006 (int64 per voxel, `__int128` setup, `c_j` folding).
For an independent check, the tests carry a small pure-Python exact-integer reference (Python ints),
run on small meshes.

**Licence and credit.** Code derived from `meshing_cxx` is MIT. Its file header credits Koen Van
Leemput and Alessandro De Florio and names the departures listed below.

**Snapping.** `q = round(v · 2¹⁴)`, rounding to nearest with ties to even: `llrint` on the CPU under
the default rounding mode, `__float2ll_rn` / `__double2ll_rn` on the GPU. Multiplying by a power of
two is exact, so this rounding is the only rounding step. *Departure from `meshing_cxx`*, which
truncates toward zero at scale 10⁴. Document this in the code header.

**Tie-break axis order.** Axes are named by number, not by physical direction. Priority is axis 0,
then 1, then 2: the sample is `X + (ε, ε², ε³)` with ε → 0⁺. Sample `X` passes face `j` if
`E_j(X) > 0`, or if `E_j(X) = 0` and the first non-zero component of `n_j`, in the order axis 0,
1, 2, is positive. This is a fixed convention, not a parameter. It is the rule ADR 0006 writes as
"x first" (position column `d` is array axis `d`), so R1's measurements still apply. *Departure from
`meshing_cxx`*, which scans axis 2 first.

**Build.** A `TETVOX_ENABLE_CUDA` CMake option, default ON. With OFF, the build is CPU-only (no
CUDA compiler needed). That allows CPU-only wheels, macOS, and CI that runs the full correctness
tests on ordinary runners.

**CPU parallelism.** OpenMP pragmas over tets. OpenMP is linked if `find_package(OpenMP)` finds it;
otherwise the same code runs serially. Results are identical either way: each voxel has exactly one
writer, and the optional visit counter uses `omp atomic`.

**Python API.** `tetvox.rasterize(mesh, shape, *, out=None)` picks the backend from the device of the
input arrays (their DLPack device). It takes no `device=` argument.

**Output contract.**
- `tet_id`: int32, shape `(X, Y, Z)`, `−1` where no tet owns the voxel centre.
- Barycentrics: π₁..π₃, shape `(X, Y, Z, 3)`, dtype `real` (amends ADR 0005's float32). Unowned
  voxels get 0. The layout is revisited in the raster-buffer benchmark (ROADMAP).
- Return value: a small struct with two counts, tets with `D ≤ 0` (inverted or degenerate) and tets
  outside the exact-arithmetic window (non-finite positions included). Both kinds are skipped, not
  rasterised. They never raise (ADR 0005).
- Optional `visits`: int32 `(X, Y, Z)`, the number of tets that claim each voxel centre. It is
  written only when the caller passes it (`atomicAdd` on the GPU). It exists so that both backends
  can be tested for exactly-once ownership.
- No mask in `rasterize`; masking happens at accumulation. Closed-box mode is deferred.

**Precision switch.**
- A CMake option `TETVOX_REAL` ∈ {`float32` (default), `float64`} generates `tetvox/config.hpp`,
  which defines `tetvox::real`. Python reads `tetvox.real_dtype` from `_core`.
- Positions, values and the stored π follow `real`. `tet_id` stays int32, and accumulators stay as
  in ADR 0005.
- The C++ API rejects inputs of the wrong dtype; there is no silent cast or copy.
  `load_atlas_npz` casts to `real_dtype`, and the `TetMesh` annotations become
  `float32 | float64`.
- Prebuilt wheels ship float32 only; float64 users build from source.

**Invalid input.**
- Tet indices outside `[0, N)` are a programming error. Both backends check them and throw; on the
  GPU this is a reduction kernel costing microseconds.
- A tet with non-finite positions counts as outside the window: it is counted and skipped.

## Why
- A CPU backend validates the contract before the costly CUDA work and serves users without a
  GPU. "Same interface" should mean the same results, hence the shared header.
- `meshing_cxx` is sound and was measured watertight within its safe range. But it departs from the
  spec on snapping, axis order, overflow, π geometry, output format and bad-tet reporting, and it is
  tied to pybind11. So we port its structure and replace its arithmetic.
- Ties-to-even is what the GPU rounding intrinsics do, so CPU and GPU snap identically.
- Axis 0 first is the standard SoS convention and plain lexicographic order on index tuples (numpy
  C-order iteration). GEMS uses it, and R1 measured it. It costs nothing at run time, because it is
  folded into `c_j` once per tet.

## Consequences
- `project()` can no longer list CUDA unconditionally; CUDA is enabled under `TETVOX_ENABLE_CUDA`.
- `__int128` (ADR 0006) restricts the CPU backend to GCC and Clang. MSVC is unsupported for now.
- The `TetMesh` dtypes of the ADR 0005 split (f32/i32/f32) become `real`/i32/`real`.
- Sub-decisions still open are listed under "Open questions" in `docs/ROADMAP.md`.
