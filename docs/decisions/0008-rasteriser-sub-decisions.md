# 0008 — Rasteriser sub-decisions: dtype-generic interface, window rule, validation

Status: accepted (2026-11-23). Owner answers to the sub-decisions that ADR 0007 left open, and to
new ones found while planning the build options and the CPU backend. Amends ADR 0007 (precision
switch replaced) and ADR 0006 (grid limit, status of k). Point B (the public shape of `rasterize`)
and the loader dtype were settled later in ADR 0009.

## Decision

**Precision: dtype-generic interface, chosen at run time (replaces ADR 0007's `TETVOX_REAL`).**
- Every op accepts float32 and float64 for positions, values and its real-valued outputs. Outputs
  take the dtype of the input. The C++ ops are templates instantiated for `float` and `double` in the
  library; Python dispatches on the array dtype.
- Within one call, positions, values and real-valued outputs share one dtype. A mismatch is
  rejected, never cast or copied (ADR 0007's no-silent-cast rule still holds).
- Internal precision choices (e.g. accumulators, ADR 0005) are compile-time constants inside the
  library, not part of the interface.
- There is no `TETVOX_REAL` build option, no `tetvox::real` and no `tetvox.real_dtype`. Both
  precisions are tested in one build. Wheels serve both.
- Why: rasterisation snaps positions to an integer lattice and is exact from then on, so the input
  dtype barely affects the result. It mainly affects storage, bandwidth and what the caller already
  holds (samseg's optimiser keeps positions in double).

**`device_count()` and the CUDA flag.** `device_count()` exists in every build and returns 0 in a
CPU-only build. A generated `tetvox/config.hpp` defines `tetvox::cuda_enabled` (`constexpr bool`)
and the macro `TETVOX_CUDA_ENABLED` (0 or 1). Python exposes `tetvox.cuda_enabled`.

**Exact window test.**
- k and L (ADR 0006): vertices are snapped to multiples of 2⁻ᵏ voxel, and every rasterised tet must
  fit in a window of 2ᴸ voxels per axis so that the per-voxel int64 test cannot overflow; the
  condition is `2k + 3L ≤ 59`.
- A tet is inside the window iff, per axis, the span of its snapped vertices is
  `≤ 2^(k+L)` lattice units (`≤`, not `<`). Re-derived worst case with `≤`:
  `|t_j| ≤ 12·2^(2k+3L) + 1 = 3·2⁶¹ + 1 < 2⁶³`. (Two cells of the bound table in R1 §2.2 are off by
  a factor of 2; its formula column is right.)
- **No grid-size limit.** The integer origin `O` is per tet, so only per-tet windows matter. This
  replaces ADR 0006's consequence "grids larger than 1024 voxels per axis need a smaller k".
- Every tet is classified, whether or not it touches the grid, in this order:
  1. `outside_window`: any coordinate non-finite, `|v| > 2^(62−k)` voxels (so that `v·2ᵏ` and the
     differences of snapped coordinates fit int64; 2⁴⁸ at k = 14), or a span above the window;
  2. otherwise `inverted`: `D ≤ 0` (inverted or degenerate);
  3. otherwise rasterised (clipped to the grid, possibly empty).
  Each skipped tet is counted once.

**k and L: run time unless compile time is measurably faster.** Owner rule: if fixing k at compile
time gives a real speed advantage, it is a compile-time constant; if the difference is negligible,
k is a run-time parameter, which is easier to use and tune. Analysis (inferred): k appears only in
the per-tet setup (snap, the floor shift in `c_j`, `f_j`, `S/D`) and in the window check, never in
the per-voxel test or the π formula. So the expected cost is negligible and the working plan is a
run-time `k` (default 14, ADR 0006) with L derived as the largest value satisfying
`2k + 3L ≤ 59` (10 at k = 14) and the `|v|` bound derived as above. Confirm with a benchmark of the
CPU backend (run-time vs `constexpr` k) and report it to the owner before the CUDA kernel. Where k
appears in the public API is part of point B.

**Barycentrics.** R1 §2.5's recipe in the call's dtype `T`: per tet `f_j ∈ [0, 1]` (exact) and
`inv = T(2ᵏ / double(D))`; per voxel `π_j = (T(t_j) + f_j) · inv`. `D` (`__int128`) is converted to
double by a correctly rounded helper (shift plus sticky bit) instead of the compiler runtime, so the
CPU and GPU agree bit for bit. Double is not forced per voxel: FP64 is slow on consumer GPUs.

**Voxels claimed by several tets** (possible only when the mesh is not a tiling, e.g. overlapping
positive tets around a fold whose inverted tets were skipped). The result is unspecified: any
claimant's `tet_id`, and `tet_id` and π may come from different claimants. Writes use relaxed
`std::atomic_ref` stores on the CPU (plain stores in machine code) so there is no C++ data race.

**Validation at the C++ boundary.** `std::invalid_argument` (Python `ValueError`) for a negative
shape component, `num_vertices < 0`, `num_tets < 0` or `> INT32_MAX` (`tet_id` is int32), and any
tet index outside `[0, N)`. It runs as a pre-pass, so outputs are untouched when it throws. An empty
mesh or grid is allowed. C++ does not check null pointers or buffer sizes; the binding checks array
shapes against the grid shape. Wrong dtype, non-contiguous or wrong-device arrays give nanobind's
`TypeError` (no implicit conversion). A claim counter, where requested, is zeroed by the call, not
accumulated.

**Packaging.** `tetvoxConfig.cmake` requires the CUDA language only if the package was built with
CUDA, and exports `tetvox_CUDA_ENABLED`. OpenMP, when found, is a link-only public dependency of the
static library, and the config calls `find_dependency(OpenMP)`. `examples/cpp` enables CUDA only if
tetvox was built with it.

**`scripts/check.sh`.** Besides the default build, it builds the plain-CMake library in the
CPU-only configuration and builds and runs the C++ example against it. Full pytest runs on the
default build only; CPU-only pytest moves to the CI roadmap item. (Both dtypes are tested in the
default build, since the interface is dtype-generic.)

**Tests (CPU backend).**
- π against an exact Python-rational reference on small meshes: absolute tolerance `1e-6` for
  float32 and `1e-12` for float64 (named constants in the tests).
- Only π₁..π₃ are stored, so "π ≥ 0 and |Σπ − 1| < 1e-6" is checked as `π₁..π₃ ≥ 0` and
  `1 − Σπ ≥ −1e-6`.
- Exhaustive exactly-once on the adult atlas and the four distinct infant meshes
  (`template_{1-2,3-5,6-8}mo_41`, `atlas_reference_41`), each on its template grid: the half-open box
  owned exactly once, the `x_d = max_d` layers unowned.
- Adversarial bit budget: a Kuhn cube spanning exactly 1024 voxels (the `≤` boundary) with a small
  grid near its far vertex, so `t` reaches about 2⁵⁷; and a copy with one vertex moved by one lattice
  unit, which must count as `outside_window`.

## Settled later (ADR 0009)
- **B. Public shape of `rasterize`.** Settled in ADR 0009: an exposed per-tet setup op
  (`setup_tets`, the only op taking `snap_bits`, writing a per-tet `status`), `count_skipped`,
  `locate`, `barycentrics`, `rasterize` (both in one pass) and the diagnostic `count_claims`, which
  replaces ADR 0007's optional `visits`. The proposal that was on the table (`rasterize` returning
  the counts) was not adopted: the counts come from the setup op.
- The dtype of `load_atlas_npz` (`dtype=` argument, default float32) and the generic `TetMesh[F]`
  annotations: also ADR 0009.

## Why
See each point. The common thread: the exact integer core makes precision a storage question, so
the interface follows the caller's dtype and only measured speed differences justify compile-time
constants.

## Consequences
- ADR 0007's "Precision switch" section is replaced; its other points stand.
- The ROADMAP item "Build options" shrinks to `TETVOX_ENABLE_CUDA` plus the generated
  `config.hpp`, and gains dtype-generic `TetMesh` and loader once the open point is answered.
- Every op is instantiated twice (float, double): longer compile, larger binary.
