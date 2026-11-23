# Real samseg runs: local data and measured workload

The owner's torch port of samseg lives outside this repo (`~/git/samseg_standalone`, pipeline script
`experiments/poc_samseg_multiatlas.py`, GPU rasteriser `POC_samseg_torch/rasterisers.py`). Its batch
runs are in `~/isbi/{bobs,dhcp,mcrib,mcribtimer}/<mode>/<subject>/`. All of it is local only.

## What a run directory contains (verified 2026-11-23)

- `affine_registered_full_atlas.npz`: the mesh after affine registration, before deformation (all
  classes, e.g. K = 41).
- `final_atlas.npz`: the mesh after deformation, with `alphas` already merged to the GMM classes
  (K = 7 for the infant atlas).
- `timing.json`, `run_config.json`, `common_domain.json`, `registration.json`; segmentations,
  priors and posteriors as NIfTI; copies of the subject's input images.
- **Not saved:** vertex positions per iteration or per cost evaluation. The POC is being rebuilt
  and is not to be modified (owner, 2026-11-23), so a real trajectory will come from running tetvox
  inside the rebuilt pipeline.

## Local copy in this repo

`data/samseg_runs/mcrib_sub-001_ses-1mo/` (gitignored, ADR 0003): `mesh_start.npz`,
`mesh_final.npz`, the four JSON files and `run.log` (the log section of this subject). The subject
images were deliberately not copied. Source: `~/isbi/mcribtimer/single_atlas_both/sub-001_ses-1mo`.

## Facts measured on that subject (M-CRIB, T1+T2, infant reference atlas)

- Mesh: N = 33,882, T = 210,872, the same tetrahedra before and after deformation.
- Positions stay in **template voxel coordinates** (bbox `[0,140]×[0,168]×[0,143]` before and
  after). The registration result is stored as the separate `affine` field.
- Vertex displacement from start to final mesh, in template voxels: mean 1.54, median 1.41, p99
  4.39, max 8.53. No tet has `det A ≤ 0` before or after.
- Image domain: common crop 158 × 177 × 148 = 4.14 M voxels; 1,953,380 voxels in the evaluation mask.
- Optimiser for the deformation: plain gradient descent (`GradientDescent`), 100 steps in each of
  the first two outer iterations, then early stopping. In total about 456 cost-and-gradient
  evaluations.
- Time: main loop 42.5 s. EM took 33.0 s and deformation 8.6 s, i.e. about 19 ms per evaluation of
  the existing POC rasteriser (includes Python overhead). EM, not rasterisation, dominates.

Implication for R1 open question 2 (K3 vs K4) [inferred]: R1's K3 prototype rasterised the larger
adult atlas in 3.9 ms + 0.3 ms setup. Even at the full adult size that is well below the
current per-evaluation time, and deformation is a fifth of the main loop. K4 is unlikely to be
worth its fallback logic (ADR 0006: K3 first, K4 deferred).
