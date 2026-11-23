# Roadmap

Every agent session reads this file first and updates it last. Status: `todo` / `doing` / `done` /
`blocked(<why>)`.

## Done
- 2026-11-23 Repo bootstrap: build chain (scikit-build-core + CMake + CUDA + nanobind), conda dev
  env, check script, docs skeleton, ADRs 0001–0003, researcher agent and skills.
- 2026-11-23 Atlas loader (`TetMesh`, `load_atlas_npz`) and `scripts/inspect_atlas.py`; findings in
  `docs/knowledge/atlas-format.md`.

## In progress
- doing — R1 research: watertight tet rasterisation, fixed point, load balancing
  (`docs/research/briefs/R1-watertight-tet-rasterisation.md`).
- doing — R2 research: decoupling cost/gradient from the rasteriser
  (`docs/research/briefs/R2-decoupling-cost-from-rasteriser.md`).

## Next
- todo — Owner reviews R1/R2 and turns the conclusions into ADRs (rasterisation rule, number
  representation, load-balancing strategy, op/API split).
- todo — CPU reference rasteriser (exact, slow) in the shared headers, used as the test oracle for
  watertightness: every voxel centre in the domain is owned by exactly one tet.
- todo — First CUDA rasteriser → per-voxel (tet id, π); benchmark on the adult atlas at 256³.
- todo — Interpolate + VJP ops; optional torch autograd wrapper.
- todo — CI (lint, mypy, compile-only CUDA build in a CUDA container); prebuilt wheels
  (cibuildwheel) before first release.

## Open questions
- Policy for voxel centres exactly on the outer domain boundary (see atlas-format.md).
- Meaning of `log_mass_laplace_kappa1` and `mass_laplace_n_free` in the atlases (ask owner).
- Output grid definition: template grid only, or arbitrary grid via the affine?
