# Roadmap

Every agent session reads this file first and updates it last. Status: `todo` / `doing` / `done` /
`blocked(<why>)`.

## Done
- 2026-11-23 Repo bootstrap: build chain (scikit-build-core + CMake + CUDA + nanobind), conda dev
  env, check script, docs skeleton, ADRs 0001–0003, researcher agent and skills.
- 2026-11-23 Atlas loader (`TetMesh`, `load_atlas_npz`) and `scripts/inspect_atlas.py`; findings in
  `docs/knowledge/atlas-format.md`.

## In progress
- doing — Owner review of R1 (`docs/research/R1-watertight-tet-rasterisation.md`, draft
  2026-11-23). Proposed: Simulation of Simplicity-style tie-break `X + (ε, ε², ε³)`; vertices snapped
  to 2⁻¹⁴ voxel with exact int64 plane tests (`__int128` in setup); a stateless row-parallel span
  kernel first, with a walk from the previous owner as the alternative.
- doing — Owner review of R2 (`docs/research/R2-decoupling-cost-from-rasteriser.md`, draft
  2026-11-23). Proposed contract: per-voxel `∂L/∂π` with a generic path (rasterise → interpolate →
  bary VJP) plus a fused C++ `accumulate<Cost>` functor for the samseg data term.

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
- Policy for voxel centres exactly on the outer domain boundary. Half-open leaves the 110,737
  centres on `x_d = max_d` unowned on the template grid. R1 describes an exact closed-box mode.
- (R1) Row spans vs walk from the previous owner: needs a replay of a real registration trajectory.
- (R1) Behaviour on `D ≤ 0` (inverted tets) and on vertices outside the grid window.
- Meaning of `log_mass_laplace_kappa1` and `mass_laplace_n_free` in the atlases (ask owner).
- Output grid definition: template grid only, or arbitrary grid via the affine?
- (R2) Curvature to support: clamped EM bound `diag(r/π)` or exact `r rᵀ`, with or without the
  indefinite terms. GEMS itself uses no Hessian (L-BFGS).
- (R2) Real target size: 256³ × K=44, or the template grid × 8 merged classes (samseg merges
  before deforming)?
