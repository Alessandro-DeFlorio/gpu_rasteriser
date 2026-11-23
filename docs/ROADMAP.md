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

## In progress
- (none)

## Next
- todo — Split `TetMesh` (ADR 0005): a minimal mesh (positions, tetrahedra, per-vertex values) and
  a samseg atlas wrapper for the rest (affine, `can_move`, stiffness, prior normaliser, names,
  colours, labels).
- todo — CPU reference rasteriser (exact, slow) in the shared headers, used as the test oracle for
  watertightness: every voxel centre in the domain is owned by exactly one tet.
- todo — First CUDA rasteriser → per-voxel (tet id, π₁..π₃); benchmark on the adult atlas at 256³.
  Extend `examples/cpp/` to run it on the Kuhn-cube mesh, as ADR 0004 asks (today the example only
  calls `device_count`).
- todo — Interpolate + VJP ops; `accumulate<Cost>` with the curvature policy parameter (only
  `None` implemented, R2 addendum A4); optional torch autograd wrapper.
- todo — Deformation prior as a separate per-tet op on the shared geometry table, including the fold
  check (R2 addendum A2).
- todo — CI (lint, mypy, compile-only CUDA build in a CUDA container); prebuilt wheels
  (cibuildwheel) before first release.

## Experiments (benchmarks that settle open choices)
- Channel layout `(X, Y, Z, K)` vs `(K, X, Y, Z)` for K ∈ {8, 16, 44} (R2 addendum A6).
- Raster buffer: 3 stored barycentrics (current choice) vs 4 vs `tet_id` only and recompute.
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
- How far to take curvature after `None` (`GaussNewton` is the cheapest next step, R2 addendum A4).
