# The shared header and the CPU backend (facts, 2026-11-24)

Code: `csrc/include/tetvox/{types,raster,cpu}.hpp`, `csrc/src/cpu.cpp`, `src/tetvox/raster.py`.
Tests: `tests/test_raster.py` against the pure-Python exact reference `tests/exact_reference.py`.
Since ADR 0010 every procedure and op is generic in the simplex dimension D ∈ {1, 2, 3}; facts
below that predate it were measured in 3D with the old names (`setup_tets`, `TetSetup`, `tet_id`).

## Range of f_j (derived; settles ROADMAP open question 4)
With `c_j = floor((E_O − 1 + own_j) / S)`, write `E_O − 1 + own_j = S·c_j + r` with `r ∈ [0, S)`.
Then `E_O − S·c_j = r + 1 − own_j`, which lies in `[0, S − 1]` when the face owns its ties and in
`[1, S]` when it does not. So `f_j ∈ [0, 1]`, and `f_j = 1` occurs. ADR 0008 (`[0, 1]`) is right;
ADR 0009's table said `[0, 1)` and was corrected. The numerator is an integer of at most `2^k`, so
`f_j` is exact in float32 for `k ≤ 24` (`max_snap_bits`). The derivation does not depend on D.

## Orientation convention (verified)
A simplex is positive when `det[v_1 − v_0, …, v_D − v_0] > 0` in axis order. The face normals of
`setup_simplex` (R3 §1.1: `(n_j)_m = (−1)^(j+m+1)` times the minor of the edges from the
lowest-numbered vertex other than j, with column m deleted; ±1 in 1D) then point inwards and
`E_j(v_j) = D`. In 3D this equals R1's face table `{{1,3,2},{0,2,3},{0,3,1},{0,1,2}}` (the integers
are exact, so any correct formula gives the same record bit for bit). All five distinct atlas
meshes satisfy the convention (no tet skipped).

## Bit budget per dimension (ADR 0010)
`int64_budget_bits<D> = 62 − ⌊log₂(2·D!)⌋` = 61, 60, 59 and
`window_bits<D>(k) = min(⌊(budget − (D−1)k)/D⌋, 63 − k)`. At k = 14: L = 49, 23, 10. In 1D the
second term binds (for k ≥ 2), and `|q| ≤ 2⁶²` already caps a span at 2⁶³ lattice units, so a
segment is `outside_window` only through the snap range `|v| ≤ 2^(62−k)`. Tested at the boundary:
a 2D and a 3D Kuhn cube spanning exactly `2^L` voxels are rasterised exactly and one lattice unit
more is `outside_window`; a 1D segment from `−2^(62−k)` is rasterised exactly and the next
representable value is `outside_window`.

## Verified results
- Exhaustive exactly-once ownership on the adult mesh and the four distinct infant meshes, each on
  its template grid (`max + 1` voxels per axis, corners at integers): every voxel centre of the
  half-open box is claimed once, and the layers `x_d = max_d` by none (`count_claims`). `rasterize`
  owns exactly those voxels.
- The same in 1D, 2D and 3D on Kuhn grids with every vertex on a voxel centre (the most ties) and
  on jittered Kuhn grids of 4,096 segments on 16,385 voxels (1D), 8,192 triangles on 257² (2D) and
  10,368 tets on 49³ (3D).
- `SimplexSetup` fields equal the exact reference bit for bit (both dtypes, `snap_bits` 0, 14 and
  24, D = 1, 2, 3).
- `rasterize` equals `barycentrics(setup, locate(setup, shape))` bit for bit.
- π against exact Fractions: within 1e-6 (float32) and 1e-12 (float64) on the small meshes and on
  2000 random owned voxels of the adult atlas. float32 and float64 positions of the adult atlas give
  the same owners.
- **The dimension-generic code (ADR 0010) reproduces the 3D-only code bit for bit** on the adult
  atlas: every record field, `simplex_id`, π and `claims`, float32 and float64 (scratch comparison
  against outputs saved from commit `d9e629c`).
- Host and device agree bit for bit: a scratch nvcc 13.5 (`sm_120`) kernel calling `snap`,
  `setup_simplex<float, D>`, `plane_values`, `owns`, `barycentric` and `row_span` on 10,000 random
  simplices for each of D = 1, 2, 3 matched the host results exactly, with no stack or local memory
  in any kernel. The header compiles as device code without `--expt-relaxed-constexpr` (it uses no
  `std::array` in device paths).

## Timings (measured, dev laptop, 14 OpenMP threads, adult atlas on 181×217×181, float32)
Ranges of the best of 40 runs, in five alternating runs of two stand-alone benchmarks built from the
3D-only code (commit `d9e629c`) and from the generic code:

| Op | 3D-only, ms | generic, ms |
|---|---|---|
| `setup_simplices` | 6.9–11.3 | 7.3–9.1 |
| `rasterize` | 38.8–39.6 | 35.4–36.3 |
| `locate` | 29.4–37.3 | 27.8–35.0 |
| `count_claims` | 25.2–32.2 | 32.1–33.1 |

`rasterize` is about 9 % faster (rows along the contiguous axis and `row_span` inlined). Setup is
within the noise. `count_claims` (diagnostic) is 2–3 % slower in its usual mode; the 3D-only code
also shows a second mode at 25 ms that the generic one did not in these runs (cause not found;
likely thread placement on this hybrid CPU).

**Inlining (g++ 13.3 `-O3`).** With the face index as a run-time loop variable, GCC calls the
generic face-normal helper out of line (4 calls per simplex) and setup is about 20 % slower than the
3D-only code. Making the face index a template parameter and force-inlining the helpers
(`TETVOX_FORCEINLINE`) gives 45 multiplications and no calls, as in the 3D-only code (566 against
545 instructions for `setup_simplex<float, 3>`). `row_span` is force-inlined as well (ADR 0010).

**k at run time vs compile time (ADR 0008 benchmark, 3D-only code).** The whole parallel setup op
takes the same time with `k` as a run-time value and as `std::integral_constant<int, 14>`: within
±1.5 % over three rounds of 41 runs. The op writes 67 MB of records, so memory bandwidth dominates.
A serial loop of the per-tet setup alone is about 56 % slower with a run-time k (the per-tet
arithmetic does not fold), and single-thread timings swing by ±50 % between rounds on this laptop.
k never reaches the per-voxel ops. Under the owner's rule this is "negligible", so k stays a
run-time parameter (owner to confirm).

## Departures from ADR 0009's sketch (internal, owner may veto)
- `row_span(rec, x, extent)` takes the extent of the row axis, not the whole `GridShape`: indexing
  a `std::array` in device code needs `--expt-relaxed-constexpr`, which every user `.cu` would then
  need too. `x[D−1]` is ignored.
- The snap-range test (`in_snap_range`: finite and `|v| ≤ 2^(62−k)`) runs on the unsnapped
  positions in the op, before `snap`; `setup_simplex(q, k)` does the rest of ADR 0008's
  classification.
- `types.hpp` also exports `max_snap_bits` (24), `snap_range_bits` (62), `int64_budget_bits<D>`,
  `window_bits<D>(k)`, `max_dim` (3) and the `TETVOX_HD` and `TETVOX_FORCEINLINE` macros;
  `raster.hpp` exports `SimplexRecord<T, D>`, `PlaneValues<D>`, `Barycentrics<T, D>`, `RowSpan`
  and `in_snap_range`, with `load_record`/`store_record`, `face_normal(s)`, `det` and the integer
  helpers in `tetvox::detail`. Python exports `MAX_SNAP_BITS`.
- `tetvox._core` takes a `SimplexSetup` as its seven arrays; the Python wrappers unpack it. The
  binding checks array ndim through nanobind (TypeError) and the agreement of shapes with D
  (ValueError).
- `barycentrics` and its validation pass run OpenMP over grid rows, so in 1D (one row) they are
  serial.

## Notes for the CUDA backend
- Rows run along the last axis (ADR 0010), the contiguous one, so the voxels of one row are
  adjacent in memory.
- `count_claims` uses `owns` at every box voxel, independently of `row_span`, so it also checks that
  the two agree.

## Measured in research R3 (2026-11-24; method in `docs/research/R3-dimension-generic-rasteriser.md` §5.1)
Throwaway code outside the repo, adult atlas on 181×217×181, float32, k = 14, dev laptop.
- **Minimal K3 on the GPU with this header and ADR 0009's record**: setup kernel 0.49–0.51 ms, row
  scan 0.10 ms (`thrust::exclusive_scan`: 0.064 ms), fill of `tet_id`/`bary` 0.30 ms, row kernel
  4.5 ms (10,572,440 rows, 6,998,400 voxels), about 5.4 ms in total. Its `tet_id` and π equal the
  CPU `rasterize` bit for bit. Row kernel: 904 SASS instructions, 70 registers, no local memory
  (frame size 0); setup kernel: 1,104 instructions, 92 registers, no local memory.
- **The row kernel is not bound by arithmetic.** Stepping `t_j` along the row (one addition per
  face and voxel instead of `plane_values`) gave 4.5 ms again. Per row it does a binary search and
  gathers about 200 bytes of record from seven arrays; per voxel it does three scattered stores.
- **Row direction.** Rows along axis 2 instead of axis 0 give the same owners and π (the rule lives
  in `c_j`) and the same time for this mesh: GPU 4.35 ms against 4.46–4.66 ms, CPU `rasterize`
  40.8 ms against 39.3–41.1 ms in the same run. The direction does matter on the CPU when spans are
  long: 2.8× for 512-pixel spans in a 2D test (27 ms against 76 ms on 4096²), 14–23 % for spans of
  a few pixels in 2D. (Settled by ADR 0010: the last axis.)
- **Inlining in `cpu.cpp`** (g++ 13.3 `-O3`, seen in the disassembly): `row_span` and
  `load_record` are out-of-line calls (once per row, once per tet) and `store_barycentrics` is a
  call per voxel. In a scratch copy of the same loops with `row_span` forced inline, `rasterize`
  and `locate` ran 4–5 % faster than `cpu.cpp` (32.6 against 34.3 ms, 24.2 against 25.3 ms).
- **Timing noise.** An identical second copy of `cpu.cpp` in the same binary differs from the
  first by at most 1 % on the minimum of 30 rounds when the variants share output buffers and
  rotate their order; whole-session drift on this laptop is 10–15 %.
