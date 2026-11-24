# 0006 — Rasterisation rule, exact arithmetic and first kernel

Status: accepted (2026-11-23); amended by ADR 0007 (axes named 0, 1, 2 with axis 0 first; snapping
rounds to nearest, ties to even; the CPU code is a public backend sharing this header) and by ADR
0008 (no grid-size limit, windows are per tet; window test `span ≤ 2^(k+L)`; k becomes a run-time
parameter unless compile time proves measurably faster). Amended by ADR 0010: the rule, the
snapping and the arithmetic hold for simplices of dimension D = 1, 2, 3; `2k + 3L ≤ 59` is the D = 3
case of `(D−1)k + DL ≤ 62 − ⌊log₂(2·D!)⌋`; K3 rows run along the last axis; "no atomics" is open for
the K3 launcher until the CUDA item benchmarks the candidates.

Context: `docs/research/R1-watertight-tet-rasterisation.md` (§1–§3, "Impact on design"). Policies on
the boundary, invalid meshes and the raster buffer are in ADR 0005; this ADR does not repeat them.

## Decision
- **Ownership rule.** A voxel centre `X` belongs to the tet whose open interior contains
  `X + (ε, ε², ε³)` for all sufficiently small `ε > 0`. Per face `j` with plane function `E_j` and
  inward normal `n_j`, the sample passes if `E_j(X) > 0`, or if `E_j(X) = 0` and the first non-zero
  component of `n_j` is positive (Simulation of Simplicity; the D3D top-left rule extended to 3D).
  Every voxel centre in the domain is owned by exactly one tet.
- **Exact integer arithmetic.** Vertices are snapped once to a lattice of 2⁻ᵏ voxel with **k = 14**.
  The per-voxel test `n_j·(X − O) + c_j ≥ 0` runs in **int64**, with the tie-break folded into
  `c_j`. The per-tet setup (`c_j`, `D`) uses `__int128`. All vertices and samples of a rasterised
  tet must lie in a window of **2¹⁰ voxels** per axis. `static_assert(2k + 3L ≤ 59)`.
- **One geometry.** Ownership, barycentrics (`π_j = E_j / D` from the exact numerators) and `∂π/∂v`
  all use the snapped mesh. The backward pass treats snapping as the identity.
- **First kernel: K3.** `setup_tet` (per tet) → exclusive scan of row counts → `raster_row` (one
  thread per (tet, y, z) row, solving the x-interval). No atomics, no sorting, no persistent state.
  The walk from the previous owner (K4) is not built unless profiling inside a real pipeline shows
  that rasterisation is a significant share of an iteration (`docs/knowledge/samseg-runs.md`
  suggests it is not).
- **Shared header.** `snap`, `setup_tet`, `inside` and the barycentric formula are
  `__host__ __device__` in `csrc/include/tetvox/`. The CPU reference oracle and the kernels call the
  same functions, so their owner grids are bit-identical.
- **Changes from R1, per ADR 0005.** The raster buffer stores `π₁..π₃` (not four values). A tet
  that is inverted or leaves the window does not raise: the call reports the count and the cost is
  `+inf`.

## Why
- Floating-point tests cannot guarantee watertightness: emulating the GEMS test in float64 on the
  adult atlas leaves 23,166 of 6,998,400 voxels not owned exactly once (R1 §2.1). The exact rule owns
  each of them once.
- K3 took 3.9 ms + 0.3 ms setup for the adult atlas in R1's prototype, against about 19 ms per
  evaluation in the owner's current pipeline.

## Consequences
- Tests (R1 "Impact on design"): exhaustive exactly-once on the adult and infant atlases; a Kuhn grid
  with all vertices on voxel centres (maximal ties), and scaled; a rotated placement partly outside
  the grid; GPU owners bit-identical to the CPU reference; `π_j ≥ 0` and `|Σπ − 1| < 1e-6` for owned
  voxels; an adversarial bit-budget test with tets spanning the full window.
- (Superseded by ADR 0008: the origin is per tet, so only per-tet windows matter.) Grids larger
  than 1024 voxels per axis need a smaller k, or tiling with a per-tile origin. Not
  needed for 256³.
- k = 14 still needs confirming on a real registration (experiment in ROADMAP).
