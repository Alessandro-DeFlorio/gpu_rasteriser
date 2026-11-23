# Brief R2 — Decoupling the cost function and its derivatives from the rasteriser

Output: `docs/research/R2-decoupling-cost-from-rasteriser.md` (follow `docs/research/_template.md`).

## Context
Read `docs/knowledge/samseg-math.md` first. If the PDF is available locally, also read
`literature/equations.pdf` (§2 data term, §3 prior). The rasteriser assigns each voxel to one tet
with barycentrics `π`. samseg then evaluates, per voxel, the following quantities:
- the interpolated priors `α_k = Σ_j α_{j,k} π_j`;
- the cost `D = −log Σ_k l_k α_k`;
- the gradient w.r.t. vertex positions, accumulated per tet;
- the EM-bound Hessian (sparse, 12×12 per tet).

The owner sees the tool as "a fancy interpolator". They want the cost and gradient computation to be
detached from the rasteriser if possible, and are unsure whether:
- (a) the cost can simply be encoded in the mesh (per-vertex values);
- (b) we should define cost and gradient functions in C++ that the rasteriser calls, but abstracted
  away from it;
- (c) something else is better.

Key observation to verify and build on:
- `∂π_j/∂v_i = −A⁻ᵀ e_j π_i` depends only on tet geometry and `π`.
- For the samseg data term, `∂D/∂v_i = Σ_m (∂D/∂π_m) · ∂π_m/∂v_i`, with `∂D/∂π_m = −r_m` (§2.3).
- So a generic backward pass needs only an upstream per-voxel `∂L/∂π` (4 numbers). Equivalently, for
  an interpolated field `f = Σ_j f_j π_j` (f_j = per-vertex values, e.g. α), it needs `∂L/∂f` (C
  numbers) per voxel.

Scale:
- K up to 44 classes on ~256³ voxels. Materialising `α_k` per voxel costs 44 × 16.7 M × 4 B ≈ 3 GB.
- The per-voxel likelihoods `l_k` come from an intensity model (GMM) that is updated outside the
  rasteriser.
- Vertices move every iteration.

## Questions
1. **Interpolation-operator view.** Formalise rasterisation as a sparse linear operator
   `W(v) ∈ ℝ^{voxels × vertices}`, with 4 non-zeros per row. Then `f = W(v) F` for any per-vertex
   field `F`. Work out:
   - the VJPs w.r.t. `F` (that is `Wᵀ g`) and w.r.t. `v` (geometric term);
   - what per-voxel data each needs;
   - what can be precomputed per tet (`A⁻¹`, `p_m`).
   Is the representation "per voxel: tet id + π (or + 3 floats)" sufficient and minimal?
2. **Where the cost lives.** Compare:
   - (i) cost fully in user code, operating on materialised interpolated fields (torch autograd
     friendly; memory heavy);
   - (ii) a templated C++/CUDA cost functor fused into the rasterisation/accumulation kernel
     (CUB / Thrust style policy, or nvdiffrast-style split kernels). The functor maps
     `(voxel data, per-vertex data, π) → (cost, ∂cost/∂π, optionally ∂²)`;
   - (iii) the "encode in the mesh" idea: the per-voxel vertex likelihoods `l_j = Σ_k l_k α_{j,k}`
     reduce K to 4 per voxel. Can the cost be expressed through per-vertex quantities alone, or not,
     given that `l_k` is per voxel?
   - (iv) a two-pass design: the rasteriser produces `(tet id, π)`; a separate, pluggable "voxel
     cost" kernel produces `∂L/∂π`; a generic "scatter" kernel accumulates vertex gradients (and
     Hessian blocks).
   For each option, cover memory traffic, kernel fusion, extensibility to other costs (e.g. other
   likelihood models, label-map losses, plain interpolation of images or deformation fields), and
   how cleanly it integrates with torch/JAX autograd through DLPack (see
   `docs/decisions/0001-bindings-nanobind-dlpack.md`).
3. **Second-order information.** How does each design support the samseg EM-bound Hessian and
   Gauss-Newton-style blocks? What per-voxel quantity does the generic scatter need in addition
   (e.g. `∂²L/∂π²`, or a diagonal approximation)?
4. **Correctness of the gradient.** The ownership of a voxel is piecewise constant in `v`. Confirm
   that the gradient of the interpolator ignores ownership changes, just as samseg does (no
   boundary/visibility terms as in nvdiffrast's antialiasing). When would that matter for
   registration, and is there any cheap remedy?
5. **Prior art.** Look at how nvdiffrast (rasterise → interpolate → texture as separate ops), Kaolin,
   PyTorch3D and the samseg/GEMS C++ code in the FreeSurfer repository (e.g.
   `kvlAtlasMeshDeformationCostAndGradientCalculator` and its subclasses, if the names still match) split these
   responsibilities. GEMS has a cost-and-gradient calculator class hierarchy layered on a rasteriser;
   describe it.

## Deliverable specifics
- A recommended architecture with an API sketch: C++ signatures and the Python-facing functions,
  showing which ops exist, their inputs and outputs (shapes and dtypes), and what is fused.
- An estimate of memory and bandwidth for the 256³ × K=44 case for each option.
- A clear yes/no on whether the gradient can be detached from the cost via `∂L/∂π` (or `∂L/∂f`),
  with the derivation.
- Cite everything. Mark each claim verified or inferred.
