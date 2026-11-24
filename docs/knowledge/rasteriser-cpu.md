# The shared header and the CPU backend (facts, 2026-11-24)

Code: `csrc/include/tetvox/{types,raster,cpu}.hpp`, `csrc/src/cpu.cpp`, `src/tetvox/raster.py`.
Tests: `tests/test_raster.py` against the pure-Python exact reference `tests/exact_reference.py`.

## Range of f_j (derived; settles ROADMAP open question 4)
With `c_j = floor((E_O − 1 + own_j) / S)`, write `E_O − 1 + own_j = S·c_j + r` with `r ∈ [0, S)`.
Then `E_O − S·c_j = r + 1 − own_j`, which lies in `[0, S − 1]` when the face owns its ties and in
`[1, S]` when it does not. So `f_j ∈ [0, 1]`, and `f_j = 1` occurs. ADR 0008 (`[0, 1]`) is right;
ADR 0009's table said `[0, 1)` and was corrected. The numerator is an integer of at most `2^k`, so
`f_j` is exact in float32 for `k ≤ 24` (`max_snap_bits`).

## Orientation convention (verified)
A tet is positive when `det[v1 − v0, v2 − v0, v3 − v0] > 0`; `setup_tet`'s face table
`{{1,3,2},{0,2,3},{0,3,1},{0,1,2}}` (R1) then gives inward normals and `E_j(v_j) = D`. All five
distinct atlas meshes satisfy it (no tet skipped). The old `kuhn_cube()` test fixture had half
its tets negative; it now reorders them.

## Verified results
- Exhaustive exactly-once ownership on the adult mesh and the four distinct infant meshes, each on
  its template grid (`max + 1` voxels per axis, corners at integers): every voxel centre of the
  half-open box is claimed once, and the layers `x_d = max_d` by none (`count_claims`). `rasterize`
  owns exactly those voxels.
- `TetSetup` fields equal the exact reference bit for bit (both dtypes, `snap_bits` 0, 14 and 24),
  including a Kuhn cube spanning exactly `2^10` voxels sampled at its far vertex. One lattice unit
  more makes every tet `outside_window`.
- `rasterize` equals `barycentrics(setup, locate(setup, shape))` bit for bit.
- π against exact Fractions: within 1e-6 (float32) and 1e-12 (float64) on the small meshes and on
  2000 random owned voxels of the adult atlas. float32 and float64 positions of the adult atlas give
  the same `tet_id`.
- Host and device agree bit for bit: a scratch nvcc 13.5 (`sm_120`) kernel calling `snap`,
  `setup_tet<float>`, `plane_values`, `barycentric`, `row_span` and `to_double_rn` on 10,000 random
  tets matched the host results exactly. So the header compiles as device code without
  `--expt-relaxed-constexpr` (it uses no `std::array` in device paths).

## Timings (measured, dev laptop, 14 OpenMP threads, adult atlas on 181×217×181, float32)
| Op | ms |
|---|---|
| `setup_tets` | 8–12 |
| `rasterize` | 40 |
| `locate` | 31 |
| `count_claims` | 33 |

**k at run time vs compile time (ADR 0008 benchmark).** The whole parallel `setup_tets` op takes
the same time with `k` as a run-time value and as `std::integral_constant<int, 14>`: within ±1.5 %
over three rounds of 41 runs. The op writes 67 MB of records, so memory bandwidth dominates. A
serial loop of `setup_tet` alone is about 56 % slower with a run-time k (the per-tet arithmetic
does not fold), and single-thread timings swing by ±50 % between rounds on this laptop. k never
reaches the per-voxel ops. Under the owner's rule this is "negligible", so k stays a run-time
parameter (owner to confirm).

## Departures from ADR 0009's sketch (internal, owner may veto)
- `row_span(rec, i1, i2, extent0)` takes `shape[0]`, not the whole `GridShape`: indexing a
  `std::array` in device code needs `--expt-relaxed-constexpr`, which every user `.cu` would then
  need too.
- The snap-range test (`in_snap_range`: finite and `|v| ≤ 2^(62−k)`) runs on the unsnapped
  positions in the op, before `snap`; `setup_tet(q, k)` does the rest of ADR 0008's classification.
- `types.hpp` also exports `max_snap_bits` (24), `int64_budget_bits` (59), `snap_range_bits` (62),
  `window_bits(k)` and the `TETVOX_HD` macro; `raster.hpp` exports `TetRecord<T>`, `PlaneValues`,
  `Barycentrics<T>`, `RowSpan` and `in_snap_range`, with `load_record`/`store_record` and the
  integer helpers in `tetvox::detail`. Python exports `MAX_SNAP_BITS`.
- `tetvox._core` takes a `TetSetup` as its seven arrays; the Python wrappers unpack it.

## Notes for the CUDA backend
- Rows run along axis 0 (`row_span`, ADR 0009), but memory is C order with axis 2 fastest
  (ADR 0007). Consecutive voxels of one row are `shape[1]·shape[2]` elements apart: a strided walk
  on the CPU. Under K3 (inferred, not measured) neighbouring threads take neighbouring `i2` rows,
  so their writes at equal `i0` are adjacent and coalesce only where the row spans line up. Rows
  along axis 2 would make every row contiguous. The row direction changes no result (ownership and
  π are per voxel), so it can be measured and revisited before K3.
- `count_claims` uses `owns` at every box voxel, independently of `row_span`, so it also checks that
  the two agree.
