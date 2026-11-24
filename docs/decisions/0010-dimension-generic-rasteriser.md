# 0010 — Dimension-generic rasteriser (1D, 2D, 3D) and the CUDA launcher candidates

Status: accepted (2026-11-24). Owner answers to the decision lists of R3 ("Impact on design",
points 1–15) and R4 (D1–D5), in the numbering of the session's proposal (points 1–18 below). Amends
ADRs 0005–0009 as stated under "Consequences".

Context: `docs/research/R3-dimension-generic-rasteriser.md` (a `template <int D>` header is bit for
bit equal to the 3D header and as fast on CPU and GPU; a hand-written 2D path is no faster than the
generic code with the same integer widths; 4D needs `__int128` per sample at k = 14) and
`docs/research/R4-nvdiffrast-lessons.md` (licence verdict; what to inherit for the CUDA backend).

## Decision

### Scope (points 1, 13)
- **Public for D ∈ {1, 2, 3}**, in C++ and Python, with one interface for every D. Internal
  strategies may differ by dimension behind it; the interface may not. 1D is public too, although
  its main use is testing.
- **Capped at 3** by the concept `SupportedDim<D>` (`1 ≤ D ≤ max_dim = 3`). The code is written
  for any D, so the cap could be lifted later, but nothing for 4D is instantiated, tested or
  guaranteed, and the integer types are not generalised for it (R3 §2, §5.5).

### Names (point 2)
Neutral names, renamed outright with no 3D aliases (nothing has been released):

| Before | After |
|---|---|
| `TetMesh`, field `tetrahedra` | `SimplexMesh`, field `simplices` |
| `TetSetup`, `TetSetupView` | `SimplexSetup`, `SimplexSetupView` |
| `setup_tets`, `setup_tet` | `setup_simplices`, `setup_simplex` |
| `TetRecord` | `SimplexRecord` |
| `tet_id` | `simplex_id` |
| `TetStatus`, `SkippedTets` | `SimplexStatus`, `SkippedSimplices` |
| `num_tets` | `num_simplices` |
| `tet_moments` (ADR 0009 op 10, not built yet) | `simplex_moments` |

Unchanged: `snap`, `plane_values`, `owns`, `barycentric`, `row_span`, `count_skipped`, `locate`,
`barycentrics`, `rasterize`, `count_claims`, `GridShape` (now `GridShape<D>`). The package keeps
the name tetvox, and "voxel" stays the word for a grid sample in any dimension. `SamsegAtlas` and
`load_atlas_npz` stay 3D; the `.npz` key `tetrahedra` is the file format and does not change.

### C++ form (point 3)
```cpp
namespace tetvox {                                    // tetvox/types.hpp
inline constexpr std::size_t max_dim = 3;
template <std::size_t D> concept SupportedDim = D >= 1 && D <= max_dim;
template <std::size_t D> using GridShape = std::array<int64_t, D>;   // C order, axis D−1 fastest
template <Real T, std::size_t D> requires SupportedDim<D> struct MeshView;          // (N,D), (T,D+1)
template <Real T, std::size_t D> requires SupportedDim<D> struct SimplexSetupView;
}
namespace tetvox::cpu {                               // tetvox/cpu.hpp
template <Real T, std::size_t D>
void setup_simplices(const MeshView<T, D>& mesh, int snap_bits, const SimplexSetupView<T, D>& out);
SkippedSimplices count_skipped(const SimplexStatus* status, int64_t num_simplices);
template <Real T, std::size_t D>
void locate(const SimplexSetupView<T, D>& s, GridShape<D> shape, int32_t* simplex_id);
// barycentrics, rasterize, count_claims likewise
}
```
- Scalar type first, then the dimension, as in `std::array` and Eigen. `std::size_t D`, so that D
  is deduced from a `std::array<…, D>` (R3 §4.3).
- The library explicitly instantiates every op for D ∈ {1, 2, 3} × {float, double}.

### Integer types (point 4)
Fixed for every D: int64 for `n_j`, `c_j`, `t_j` and boxes, `__int128` for the setup determinants.
**No CMake option.** The owner suggested a compile-time choice "like the other types", or else the
session's choice. The float dtype is no longer a build option (ADR 0008 replaced `TETVOX_REAL` with
run-time float/double); an integer-width option would make the dtype of the exposed record, and with
it the Python stub and the tests, depend on the build, and it would help only 2D, only at k ≤ 8 and
by at most 12 % (R3 §5.4).

### Bit budget (point 5): the proven R1-style bound
- `int64_budget_bits(D) = 62 − ⌊log₂(2·D!)⌋` = 61, 60, 59 for D = 1, 2, 3 (R3 §2.1: the per-sample
  bound `|t_j| ≤ 2·D!·2^((D−1)k + DL) + 1` must stay below 2⁶³).
- `window_bits(D, k) = min(⌊(int64_budget_bits(D) − (D−1)·k) / D⌋, 63 − k)`. The second term keeps
  the window `2^(k+L)` representable in the unsigned window test. It binds only in 1D (for k ≥ 2),
  where `|q| ≤ 2⁶²` already bounds every span by 2⁶³, so a segment is `outside_window` only through
  the snap range. 3D is unchanged (`L = 10` at k = 14); 2D gets `L = 23` at k = 14.
- 1D check of the bound: `n_0 = −1`, `n_1 = +1` (the 0×0 minor is 1), `D = q_1 − q_0`, the box has
  at most `2ᴸ` samples and `|c_j| ≤ 2ᴸ + 2`, so `|t_j| ≤ 2^(L+1) + 1`, the formula with D = 1.
- R3's tight bound (§2.3) is not adopted.

### Conventions (points 6, 7)
- `default_snap_bits = 14` and `max_snap_bits = 24` for every D.
- The tie-break puts axis 0 first in every D, and positive orientation is
  `det[v_1 − v_0, …, v_D − v_0] > 0` in axis order. For a 2D image indexed `(row, col)` this reads
  "top, then left", and a positive triangle is clockwise on screen. Results differ from a
  Direct3D-style rasteriser only at exact ties (R3 §1.4).

### Row axis (point 8)
Rows run along the **last axis, D − 1**, the contiguous one in C order, in both backends.
`row_span` solves along axis D − 1. No result changes (the rule lives in `c_j`).

### Shapes (point 9)
positions `(N, D)`, simplices `(T, D+1)` int32, values `(N, C)`; record `normals (T, D+1, D)`,
`offsets`/`fractions (T, D+1)`, `bary_scale (T,)`, `box_lo`/`box_hi (T, D)`, `status (T,)`;
`simplex_id (n_0, …, n_{D−1})` int32; `bary (n_0, …, n_{D−1}, D)` holding `π_1..π_D`, the trailing
axis kept for D = 1; moments `(T, D+1, D)` when they are built. The 3D layouts of ADR 0009 are the
D = 3 case.

### Python (point 10)
- One function per op for every D. D comes from the arrays (`positions.shape[1]`, the record's
  `box_lo.shape[1]`); grid shapes are `tuple[int, ...]`.
- `_core` has one binding per backend and dtype, which reads D from the array shapes and calls the
  matching instantiation. It raises `ValueError` when D is not 1, 2 or 3 or when the shapes disagree
  with D (ADR 0008: invalid values raise `ValueError`).

### What stays 3D (point 11)
All ops of ADR 0009 are generic in D, including `accumulate<Cost>` (the `Cost` concept sees D + 1
barycentrics). `SamsegAtlas` and the loader stay 3D. The deformation prior stays 3D until it is
analysed in its own item.

### Order of work (point 13) and CPU inlining
- The shared header and the CPU backend become generic first, on the feature branch, with 3D
  outputs bit-identical to the previous code; the CUDA kernels are then written once on the generic
  procedures.
- `row_span` is force-inlined in host and device code (R3 §5.2: about 5 % on the CPU, no result
  change).

### CUDA backend (points 14–18, from R4)
- **Launcher (D1): open until measured.** The CUDA item builds the candidates side by side and
  benchmarks them: (a) row-count scan, readback of the total and one thread per row; (b) persistent
  blocks taking chunks of rows through a device-side counter; (c) persistent blocks taking batches
  of simplices, with the row counts scanned in shared memory. Measure on the adult atlas at its
  template grid and at 256³, and on 2D meshes, including the cost of (a)'s synchronisation inside a
  loop. The winner is recorded in an ADR then. Until that ADR, ADR 0006's "no atomics" and
  ADR 0009's "only `count_skipped` synchronises" and `cudaMallocAsync` plan are open for the
  launcher; each voxel still has exactly one writer.
- **Empty-row reject (D2):** the division-free row test goes into the shared `row_span` if a
  measurement shows no CPU slowdown.
- **Atomics (D3):** scatters combine per (warp, simplex) first, then per block is measured;
  float32 inside a block, float64 across blocks (ADR 0005). Settled finally in the
  `simplex_moments` item.
- **Tunable constants (D4):** chunk size and threads per block are named compile-time constants;
  blocks per multiprocessor come from CUDA's occupancy query at run time. Tests run with two chunk
  sizes to show that results do not depend on them.
- **Attribution (D5):** no code is copied from nvdiffrast (its "Nvidia Source Code License (1-Way
  Commercial)" limits derivative works to non-commercial use, §3.3). Ideas taken from it are cited
  (Laine & Karras 2011; Laine et al. 2020) next to the code that uses them.

## Why
- Generic code costs nothing at run time for D ∈ {2, 3}: bit-identical records, owners and π;
  GPU row kernel 4.5 ms against 4.5 ms, setup 0.48 against 0.50 ms, 72 against 70 registers, no
  local memory; CPU within the noise of inlining (R3 §5.2–5.4).
- Written now, the CUDA kernels are written once; converting later would cost about twice as much
  (R3 recommendation 6). Only public names and shapes are expensive after a release, so they are
  settled here.
- The launcher choice changes accepted ADRs and has no measurement yet (R4 counts work items only),
  so the owner keeps it open until the candidates are timed.

## Consequences
- ADR 0005–0009 read with D in place of 3 and the names above. In particular: ADR 0006/0008's
  `2k + 3L ≤ 59` is the D = 3 case of the budget above; ADR 0007's and ADR 0009's rows along axis 0
  become rows along axis D − 1; ADR 0009's type and function names, shapes and the C++ form are
  replaced by the ones above.
- Tests: the exact reference and the Kuhn fixtures take D; exhaustive exactly-once tests run in 1D
  and 2D as well as 3D; the real-atlas tests stay 3D.
- The C++ example rasterises a Kuhn cube in 1D, 2D and 3D.
- R3's and R4's reports are the research behind this ADR; R4's other items (coalesced atomics,
  block-local redistribution, zero-gradient exit) are inputs to the CUDA and gradient items.
