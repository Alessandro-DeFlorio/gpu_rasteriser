# Roadmap

Every agent session reads this file first and updates it last. Status: `todo` / `doing` / `done` /
`blocked(<why>)`.

## Done
- 2026-11-23 Repo bootstrap: build chain (scikit-build-core + CMake + CUDA + nanobind), conda dev
  env, check script, docs skeleton, ADRs 0001–0003, researcher agent and skills.
- 2026-11-23 Atlas loader (`TetMesh`, `load_atlas_npz`) and `scripts/inspect_atlas.py`; findings in
  `docs/knowledge/atlas-format.md`.
- 2026-11-23 Owner review of R2 and the open questions of R1/R2: decisions in ADR 0005, R2 addendum
  (run-time costs, prior split, curvature policies, float64, channel layout). ADR 0004: C++ is a
  first-class interface and Python is optional.
- 2026-11-23 One real samseg run copied to `data/samseg_runs/` (start and final mesh, timings);
  facts in `docs/knowledge/samseg-runs.md`.
- 2026-11-23 Owner approved R1's core proposal: ADR 0006 (exact SoS tie-break, k = 14 snapping with
  int64 tests, K3 row kernel first). Python `out=` convenience confirmed (ADR 0005).
- 2026-11-23 Build restructure for ADR 0004: `TETVOX_BUILD_PYTHON` option, exported
  `tetvox::tetvox` package (component `cpp`, kept out of the wheel), `examples/cpp/` consumer built
  and run by `scripts/check.sh`, static cudart only on `_core`, version set only in `project()`.
  The `native` CUDA-arch default now works in plain CMake builds too (`build.md`).
- 2026-11-23 `TetMesh` split (ADR 0005). `TetMesh` (positions, tetrahedra, `values`) is the
  generic mesh. `SamsegAtlas` (`samseg.py`) holds a `TetMesh` (composition) plus affine,
  `can_move`, names, colours, labels and stiffness. Owner decisions: generic name `values` (not
  `alphas`), `values` required (`(N, 0)` when there is no data), dtypes f32/i32/f32, no checks in
  the Python constructor, prior-normaliser keys ignored entirely. CLAUDE.md gained the
  "Decisions and assumptions" rule.
- 2026-11-23 Review of the earlier CPU rasteriser `meshing_cxx` (owner and supervisor), measured on
  the adult atlas: watertight up to ×4 scale, silent int64 overflow at ×5, several departures from
  ADR 0005/0006 (`docs/knowledge/meshing-cxx-reference.md`). Owner decisions in ADR 0007: public
  CPU backend with the same contract as CUDA, sharing ADR 0006's header; snapping rounds to nearest
  with ties to even; axes named 0, 1, 2 with axis 0 first in the tie-break; `TETVOX_ENABLE_CUDA`;
  OpenMP; `TETVOX_REAL` precision switch; output contract (`tet_id`, π, counts, optional
  `visits`); invalid-input policy.
- 2026-11-23 Owner answers to ADR 0007's open sub-decisions and to new ones (ADR 0008): the
  interface is dtype-generic (float32 and float64 at run time, replacing `TETVOX_REAL`);
  `device_count()` returns 0 in CPU-only builds plus `tetvox::cuda_enabled`; window test
  `span ≤ 2^(k+L)` per tet with no grid limit and grid-independent classification; k run-time unless
  compile time is measurably faster; R1 §2.5 barycentrics; multi-claimed voxels unspecified;
  validation rules; packaging; check.sh builds the CPU-only configuration; test tolerances. Open:
  the public shape of `rasterize` (point B).
- 2026-11-23 Owner approved the op inventory and the rasterisation API (ADR 0009, settles ADR 0008
  point B). Per-element `__host__ __device__` procedures (`snap`, `setup_tet`, `plane_values`,
  `owns`, `barycentric`, `row_span`) and one-thing array ops: `setup_tets` (exposed per-tet record,
  unclipped origin, per-tet `status`, the only op taking `snap_bits`), `count_skipped`, `locate`,
  `barycentrics`, `rasterize` (= both, bit for bit), `count_claims` (diagnostic, replaces
  `visits`), then `interpolate`, `interpolate_vjp_values`, `interpolate_vjp_bary`, `tet_moments`,
  `vertex_grad`, `accumulate<Cost>`. C++ view structs + `Real`-constrained templates instantiated
  for float/double; CUDA twin adds only a stream. Python mirrors the C++ names, takes arrays,
  `out=` all or nothing. `load_atlas_npz(dtype=float32)`, generic `TetMesh[F]`.

- 2026-11-24 ADRs 0007–0009 merged into `main` (fast-forward, local). Owner: build the next items
  as a prototype; the questions found while reviewing them are discussed once it exists (Open
  questions, "deferred to the prototype").
- 2026-11-24 Build options (ADR 0007/0008): `TETVOX_ENABLE_CUDA`, generated `tetvox/config.hpp`,
  `tetvox.cuda_enabled`, `device_count()` = 0 in CPU-only builds, `tetvoxConfig.cmake` exporting
  `tetvox_CUDA_ENABLED` and requiring CUDA only when built with it, `examples/cpp` working in both,
  `check.sh` building both. Dtype-generic `TetMesh[F]`, `SamsegAtlas[F]`,
  `load_atlas_npz(dtype=float32)` (ADR 0009). Details in `build.md`.

- 2026-11-24 Shared header + CPU backend (ADR 0006–0009), prototype: `types.hpp`, `raster.hpp`
  (`snap`, `setup_tet`, `plane_values`, `owns`, `barycentric`, `row_span`, `__host__ __device__`),
  `cpu.hpp`/`cpu.cpp` (ops 1–6, float and double, OpenMP), nanobind overloads, Python wrappers
  (`tetvox.setup_tets` … `count_claims`, `TetSetup`, `TetStatus`, `SkippedTets`). Every test of the
  item passes, including exhaustive exactly-once on the five distinct atlas meshes; the k
  benchmark shows no difference in the parallel op (run-time k kept, owner to confirm). The C++
  example rasterises a Kuhn cube in both build configurations. Facts and timings in
  `rasteriser-cpu.md`. Not done: allocation of outputs for torch inputs (torch is not in
  `tetvox-dev`; numpy only, other frameworks pass `out=`).

## In progress
- (none)

## Next
- todo — First CUDA backend for ops 1–6 of ADR 0009 (K3 for `locate`/`rasterize`, stream as the
  last parameter, row offsets via `cudaMallocAsync`; settle Python stream handling), bit-identical
  to the CPU backend; benchmark on the adult atlas at 256³. Extend `examples/cpp/` to run it on the
  Kuhn cube too (ADR 0004).
- todo — `interpolate`, `interpolate_vjp_values`, `interpolate_vjp_bary`, `tet_moments`,
  `vertex_grad` (ADR 0009 ops 7–11); `accumulate<Cost>` on `setup` + shape with the curvature policy
  parameter (only `None` implemented, R2 addendum A4); optional torch autograd wrapper (forward
  `setup_tets` + `rasterize`, backward `tet_moments` + `vertex_grad`).
- todo — Deformation prior as a separate per-tet op, including the fold check (R2 addendum A2).
  Decide whether it reads the snapped geometry of `TetSetup` (ADR 0009).
- todo — CI (lint, mypy, compile-only CUDA build in a CUDA container); prebuilt wheels
  (cibuildwheel) before first release.

## Experiments (benchmarks that settle open choices)
- Channel layout `(X, Y, Z, K)` vs `(K, X, Y, Z)` for K ∈ {8, 16, 44} (R2 addendum A6).
- Raster buffer: 3 stored barycentrics (current choice) vs 4 vs `tet_id` only and recompute
  (`locate` + `barycentrics`, ADR 0009).
- float32 vs two-level vs float64 per-tet moments: error against a float64 CPU reference, and time
  (R2 addendum A5).
- Tet-parallel vs voxel-parallel accumulation.
- (R1) Snapping precision k ∈ {8…16}: ownership and cost on the start and final meshes of
  `data/samseg_runs/`, on positions interpolated between them, and with random jitter.
- (R1) K3 vs K4: deferred. It needs a real per-evaluation trajectory, which we get for free once
  tetvox runs inside the owner's rebuilt pipeline. The owner's current POC is being rebuilt and is
  not to be modified. Existing data suggest K3 is enough (`samseg-runs.md`).
- (R1) int64 division speed on GPUs other than the dev laptop.

## Open questions
- `TetMesh.values` is required, so a mesh without data uses shape `(N, 0)`. Make it optional if
  that proves awkward (a small, local change).
- How far to take curvature after `None` (`GaussNewton` is the cheapest next step, R2 addendum A4).
- Run-time vs compile-time k (ADR 0008): the CPU benchmark shows no difference in the parallel
  `setup_tets` op (`rasteriser-cpu.md`); run time is kept. Owner to confirm.
- Torch: add torch to `tetvox-dev` (or a CI job) so the torch allocation path of the Python ops can
  be built and tested (ADR 0009 says numpy and torch).
- Row direction for K3: rows run along axis 0, while memory has axis 2 fastest (ADR 0007/0009).
  Measure rows along axis 2 before the CUDA kernel (`rasteriser-cpu.md`).
- Deferred to the prototype (owner, 2026-11-24). Each is built provisionally as noted and must be
  confirmed or changed:
  1. `TETVOX_ENABLE_CUDA=ON` without a CUDA compiler: provisionally CMake's own configure error
     (recommended; a silent CPU-only fallback is the alternative).
  2. `barycentrics` and the caller's `tet_id`: is a value outside `[−1, T)`, one naming a skipped
     tet, or one whose tet's box does not contain the voxel checked (pre-pass, `invalid_argument`;
     recommended, and built that way) or a documented precondition? The box condition was added
     while building: without it a wrong id can overflow int64 in `plane_values`.
  3. torch tensors with `requires_grad=True` passed to plain ops: provisionally they fail in the
     DLPack export (recommended; the alternative is a silent detach in the wrapper).
  4. Settled 2026-11-24: `f_j ∈ [0, 1]` (ADR 0008 is right; ADR 0009's table corrected). Exact in
     float32 for `snap_bits ≤ 24` (`rasteriser-cpu.md`).
