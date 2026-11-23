# R2 — Decoupling the cost function and its derivatives from the rasteriser

Status: draft
Brief: `briefs/R2-decoupling-cost-from-rasteriser.md`

Every claim is tagged **[V]** (verified: read in the cited source, or checked numerically by the
author) or **[I]** (inferred: the author's reasoning, arithmetic or extrapolation).

## Question

1. Can rasterisation be formalised as a sparse interpolation operator `W(v)`, and what do its VJPs
   need per voxel and per tet? Is "tet id + π" sufficient and minimal?
2. Where should the cost live: (i) user code on materialised fields, (ii) a C++/CUDA functor fused
   into the kernel, (iii) "encoded in the mesh", or (iv) a two-pass design around `∂L/∂π`?
3. How does each design support the samseg EM-bound Hessian and Gauss-Newton blocks?
4. Is the gradient that ignores ownership changes correct? When does that matter?
5. How do nvdiffrast, PyTorch3D, Kaolin and samseg/GEMS split these responsibilities?

## Recommendation

- **Yes, the gradient can be detached from the cost through `∂L/∂π`.** For any per-voxel loss,
  `∂L/∂v_i = −π_i A⁻ᵀ ḡ` with `ḡ = (g₁−g₀, g₂−g₀, g₃−g₀)`, `g = ∂L/∂π`. The geometric factor
  depends only on `π` and the tet's `A⁻¹`. Three floats per voxel are sufficient and minimal.
  Derivation in Findings §1; finite-difference error 1e-10 [V].
- **Make `∂L/∂π` the contract, and build two paths on one shared accumulator** (answer "(c)" to the
  owner's question, combining options (ii) and (iv)):
  - a generic path: `rasterize → (tet_id, bary)`, `interpolate`, and `bary_vjp(grad_bary) → ∂L/∂v`.
    Any cost can be written in user code (torch autograd included). This is the "fancy
    interpolator";
  - a fused path: a C++ template `accumulate<Cost>` whose functor returns
    `(cost, ∂cost/∂π, optional curvature)` per voxel. It never materialises per-voxel fields.
    Use it for the samseg data term.
  Both paths call the same `__host__ __device__` code: per-tet moments, then a per-tet finaliser.
- **Accumulate geometry-free per-tet moments, then apply the geometry once per tet.** The voxel loop
  needs only `π` and the cost's derivatives: `m_i = Σ_x π_i ḡ` (12 floats per tet). The finaliser
  computes `G_i = −A⁻ᵀ m_i`. The cost functor never sees `A`.
- **Do not materialise `α_k` for K = 44.** One array is 2.95 GB at 256³, and option (i) needs three
  to four of them, on an 8 GB development GPU. Option (i) stays available for small C (images,
  deformation fields, merged classes).
- **The cost cannot be encoded in the mesh alone**, because `l_j = Σ_k l_k α_{j,k}` contracts
  per-voxel data with per-vertex data. What the mesh should carry is a reduced `α`: samseg already
  merges the 44 structures into 8 classes before deformation, and the adult atlas has on average
  2.4 non-zero `α` per vertex (at most 8). Store `α` sparsely per vertex.
- **Curvature uses the same contract, plus `∂²L/∂π²` per voxel.** The general per-tet block is
  `H_ij = A⁻ᵀ(Σ_x π_i π_j H̄)A⁻¹ − p_j G_iᵀ − G_j p_iᵀ`. The samseg EM bound is the special case
  `∂²/∂π² = diag(r_m/π_m)` (4 floats per voxel). The exact data-term curvature is `r rᵀ`, which
  needs no extra per-voxel data.
- **Clamp `π_m` in the EM-bound curvature.** `r_m/π_m` diverges as a voxel approaches a face, and
  atlas boundary faces lie exactly on voxel-centre planes. The owner should choose between the
  clamped EM bound and the bounded `r rᵀ` form.
- **Ignore ownership changes in the gradient, as samseg does.** The interpolated field is continuous
  across faces, so the total cost is continuous and the gradient is exact wherever no voxel centre
  lies on a face. No nvdiffrast-style antialiasing is needed. This argument requires
  watertightness and per-vertex (not per-tet) attributes.
- **Keep the raster buffer `(tet_id, bary)` as a first-class output** (268 MB at 256³). It is reused
  by priors, gradient and curvature at the same `v`, and by multi-pass costs.

## Findings

### 1. Interpolation-operator view

Notation follows `docs/knowledge/samseg-math.md` [1]: `x = Aπ + v₀`, `p_m = A⁻ᵀ s_m`,
`P = [p₀ p₁ p₂ p₃]ᵀ ∈ ℝ^{4×3}`, so `Σ_m p_m = 0` and `∂π_m/∂x = p_m`.

**Operator.** Let `t(x)` be the owner of voxel `x`. Define `W(v) ∈ ℝ^{X×N}` with
`W[x, n] = π_j(x; v)` if `n` is local vertex `j` of tet `t(x)`, else 0: four non-zeros per row, rows
sum to 1. Then `f = W(v) F` for any per-vertex field `F ∈ ℝ^{N×C}` (eq. 4 of [1] with `F = α`) [I].

**Derivative of `π`.** For every `i, j ∈ {0..3}`:

    ∂π_j/∂v_i = −π_i p_j                                                    (1)

[1] §2.1 derives this for `j = 1..3` as `−A⁻ᵀ e_j π_i`, including the chain rule for `i = 0`. The
case `j = 0` follows from `π₀ = 1 − Σπ_j` and `p₀ = −Σ p_j` [V: read in [1]; finite differences
agree to 1.5e-10].

**VJP w.r.t. vertex positions.** Let `L` depend on voxel `x` through `π`, with `g = ∂L/∂π ∈ ℝ⁴`
(the four `π` treated as free variables). By (1):

    ∂L/∂v_i = Σ_m g_m ∂π_m/∂v_i = −π_i Σ_m g_m p_m = −π_i Pᵀg = −π_i A⁻ᵀ ḡ   (2)

with `ḡ = (g₁−g₀, g₂−g₀, g₃−g₀)`.
- Only `ḡ` matters: adding a constant to `g` changes nothing, because `Σ_m p_m = 0`. Three numbers
  per voxel are sufficient and minimal [V: numerically, shift invariance to 6e-17].
- `q = Pᵀg` is the spatial gradient `∇_x L` of the per-voxel loss, so (2) reads
  `∂L/∂v_i = −π_i ∇_x L`: moving vertex `i` by `δ` moves the material under the voxel by `π_i δ` [I].
- For the samseg data term, `g_m = ∂D/∂π_m = −r_m`, so `ḡ = −r̄` and (2) becomes
  `∂D/∂v_i = A⁻ᵀ r̄ π_i`, which is [1] §2.1 and §2.3 [V: read in [1]; finite differences agree to
  1.1e-10 for both forms].
- For an interpolated field with upstream `g_f = ∂L/∂f ∈ ℝ^C`: `g = F_t g_f`, where `F_t ∈ ℝ^{4×C}`
  holds the four vertex rows. Equivalently `∂L/∂v_i = −π_i J_tᵀ g_f`, where `J_t = F_tᵀ P` is the
  Jacobian `∇_x f`, constant over the tet [V: finite differences agree to 5e-10].

**Per-tet factorisation.** Summing (2) over the voxels of tet `t`:

    G_{t,i} = −A_t⁻ᵀ m_{t,i},        m_{t,i} = Σ_{x∈t} π_i(x) ḡ(x) ∈ ℝ³       (3)

The moments `m` (12 floats per tet) contain no geometry. `A⁻¹` is applied once per tet. The global
gradient is the sum of `G_{t,i}` over the tets incident to each vertex. This is the structure of
`A⁻ᵀ[Σ_v r̄ᵛ π_iᵛ]` in [1] §2.1 [V: a whole-mesh implementation of (3) matches finite differences
to 1.3e-8 on a jittered Kuhn mesh with 384 tets and 2197 samples].

**VJP w.r.t. vertex values.** `∂L/∂F = W(v)ᵀ g_f`, that is `∂L/∂F_n = Σ_x W[x,n] g_f(x)`: a
scatter-add of `π_j g_f` to the four vertices. It needs no geometry [I]. This is the op that atlas
building ([1] §5) would use.

**Per-voxel and per-tet data.**

| Op | Per voxel in | Per tet / vertex in | Out |
|---|---|---|---|
| forward `f = W F` | `tet_id`, `π` | 4 rows of `F` | `f` (C) |
| `Wᵀ g_f` | `tet_id`, `π`, `g_f` (C) | — | `∂L/∂F` (N×C) |
| reduce to `ḡ` | `tet_id`, `g_f` (C) | 4 rows of `F` | `ḡ` (3) |
| `ḡ → ∂L/∂v` | `tet_id`, `π`, `ḡ` (3) | `A⁻¹` (9 floats) | `∂L/∂v` (N×3) |

Per tet, `A⁻¹` and `v₀` (12 floats, 16 MB for the adult atlas) are enough. The rows of `A⁻¹` are
`p₁..p₃`, and `det A` comes for free for the prior term ([1] §3) [I].

**Is "tet id + π" sufficient and minimal?**
- Sufficient: yes. No per-voxel Jacobian is needed, because (1) factorises into `π` and per-tet
  `P` [I, from (1)–(3)].
- Minimal: `tet_id` alone is enough, since `π = A⁻¹(x − v₀)` can be recomputed. Storing
  `π₁..π₃` is a cache: 16 B instead of 4 B per voxel, in exchange for one 12-float gather and nine
  multiply-adds per voxel per consumer [I]. Which is faster needs a benchmark (Open questions).
- nvdiffrast stores the same thing in 2D: `(u, v, z/w, triangle_id)` per pixel, with the third
  barycentric recomputed as `1 − u − v` [V: [3] `ops.py` docstring; `interpolate.cu` line 194].
- Consumers must tolerate `π_m` slightly outside `[0, 1]` if ownership is decided in fixed point
  (R1) while `π` is computed in float [I].

### 2. Where the cost lives

**Sizes used below** [I: arithmetic]. `V = 256³ = 16.78 M` voxels, so 4 B per voxel is 67 MB.
`T = 338,993` and `N = 54,598` for the adult atlas (`atlas-format.md`). The development GPU has
8151 MiB [V: `nvidia-smi`].

| Array | Size at 256³ |
|---|---|
| `tet_id` (i32) + `bary` (3×f32) | 268 MB |
| one K = 44 field (`α_k`, `l_k`, or `∂L/∂α_k`) | 2.95 GB |
| one 8-class field (merged classes) | 537 MB |
| `ḡ` (3×f32) / diagonal curvature (4×f32) / symmetric curvature (6×f32) | 201 / 268 / 403 MB |
| per-tet gradient moments (12×f32) / with EM curvature (52×f32) / blocks `[T,12,12]` | 16 / 70 / 195 MB |
| dense `α` table `[N,44]` / sparse, 8 entries per vertex | 9.6 / ~2.2 MB |

**Memory and traffic per cost-and-gradient evaluation** [I: arithmetic; per-voxel arrays only,
vertex-table gathers excluded; timings were not measured].

| Option | Extra device memory | Per-voxel bytes read + written |
|---|---|---|
| (i) materialised, K = 44 | 9–12 GB (`α`, `l`, product, `∂L/∂α`) | 15–21 GB |
| (i) materialised, 8 merged classes | 1.6–2.2 GB | 2.7–3.8 GB |
| (ii) fused functor, raster buffer kept | 16 MB (70 MB with curvature) | ~0.35 GB |
| (ii) fused into the rasteriser itself | 16 MB | ~0.08 GB (image only) |
| (iii) `l_j` buffer (4 floats) | 268 MB on top of (iv) | +0.54 GB on top of (iv) |
| (iv) two-pass through `ḡ` | 201 MB (469 MB with curvature) | ~1.0 GB (1.55 GB with curvature) |

**(i) Cost in user code on materialised fields.**
- Cleanest for autograd: `interpolate` is linear, and its backward is the two VJPs of §1.
- At K = 44 it does not fit the 8 GB development GPU [I].
- It is the right tool for small C: image or deformation-field interpolation (C = 1–3), label maps,
  or the 8 merged classes.

**(ii) Fused C++/CUDA functor.**
- No per-voxel intermediate at all. The functor gathers the four vertex rows, evaluates the
  likelihoods from the image and returns `(cost, g, curvature)`.
- Prior art: GEMS has an experimental CUDA kernel templated on a `VertexAction` functor called for
  every inside voxel as `action(tet, iz, iy, ix)`, with one thread block per tet. Only a visit
  counter and an alpha drawer exist; there is no CUDA cost calculator [V: [2]
  `gems/cuda/simplesharedtetrahedroninterior.hpp`, `visitcounteraction.hpp`,
  `alphadraweraction.hpp`]. An older GEMS header shows a `FragmentProcessor` policy with
  `operator()(pi0, pi1, pi2, pi3)`, `StartNewSpan` and `StartNewTetrahedron`; it is stale, since
  `AtlasMeshRasterizor` is no longer a template [V: [2]
  `gems/kvlAtlasMeshMinLogLikelihoodCalculator.h`].
- A new cost means a new C++ struct and a rebuild. It cannot be defined from Python [I].
- The gather of `α` dominates. Dense K = 44 is 704 B for four vertices. In a voxel-parallel kernel
  that is repeated per voxel; in a tet-parallel kernel it is loaded once per tet (0.24 GB per pass
  in total) [I].

**(iii) Encode the cost in the mesh.**
- Not possible in general. `D` depends on voxel and mesh only through the four numbers
  `l_j = Σ_k l_k α_{j,k}` ([1] eq. 6), but `l_j` is indexed by voxel *and* local vertex. It is a
  contraction of a per-voxel vector `l(x) ∈ ℝ^K` with a per-vertex vector `α_j ∈ ℝ^K`, so it is not
  a vertex field [I].
- A materialised `l_j` buffer is only 4 floats per voxel, but computing it is the same gather and
  K-reduction as in (ii). It saves nothing over (iv) with `ḡ` [I].
- What can be moved into the mesh is any voxel-independent reduction of `α`:
  - samseg merges the structures into "super-structure" classes before deformation
    (`mergedAlphas = alphas · fractionsTableᵀ`, then `downSampledMesh.alphas = mergedAlphas`)
    [V: [2] `samseg/merge_alphas.py`, `samseg/Samseg.py` lines 922–924]. The local
    `sharedGMMParameters_adult.txt` defines 8 classes with 10 Gaussians [V: local file];
  - `α` is sparse. In the adult atlas a vertex has on average 2.41 non-zero `α` (median 2, max 8).
    The union over a tet's four vertices is 3.15 on average (max 12), and 2.84 after merging to 8
    classes. 10.5% of the volume lies in tets with one active class, where the data gradient is
    exactly zero [V: measured on `atlas_adult.npz` as `(alphas > 0).sum(1)` and
    `(alphas[tetrahedra] > 0).any(1).sum(1)`];
  - GEMS exploits the sparsity: it skips classes whose four `α` are all zero [V: [2]
    `kvlAtlasMeshToIntensityImageCostAndGradientCalculator.cxx` lines 94–100 and 125].
- Special case: for one quantised intensity channel with B bins, `l_k(x) = Λ[k, bin(x)]`, so
  `l_j(x) = (αΛ)[j, bin(x)]` is a per-vertex table of B values and the voxel carries one index.
  GEMS's mutual-information calculator has this structure (64 bins) [V: [2]
  `kvlMutualInformationCostAndGradientCalculator.cxx` lines 61, 210, 245]. It does not extend to
  multi-contrast GMMs [I].

**(iv) Two-pass design through `∂L/∂π`.**
- Rasteriser → `(tet_id, bary)`; pluggable voxel-cost kernel → `ḡ` (and curvature); generic scatter
  → vertex gradient and Hessian blocks.
- Extra traffic over (ii) is about 0.65 GB per evaluation at 256³ [I]. This is small next to (i).
- The cost kernel can be written in torch, cupy or numba without rebuilding tetvox [I].
- The voxel-cost kernel still needs the mesh (tet id → vertices → `α`), so "pluggable" means "a
  functor that gathers vertex data", not "a function of the voxel alone" [I].

**Extensibility.**

| Use case | Per-voxel input | Per-vertex input | Fits `∂L/∂π`? |
|---|---|---|---|
| samseg GMM data term | intensities | sparse `α` | yes: `g = −l_j / Σ l_j π_j` |
| label-map loss | label | `α` | yes; GEMS's label calculator is `l_j = Σ_{k∈label} α_{j,k}` [V: [2] `kvlAtlasMeshToLabelImageCostAndGradientCalculator.cxx`] |
| mutual information | bin index | `α` | yes, with histogram passes over the same raster buffer first [V: [2], up to 10 `Histogrammer` passes plus one gradient pass] |
| interpolate an image or deformation field | — | `F` (C small) | yes, through `interpolate` and its VJPs |
| point-set / landmark cost | — | — | not a raster cost; GEMS adds it in a post-process hook [V: [2] `kvlAtlasMeshToPointSetCostAndGradientCalculator.cxx`] |
| deformation prior ([1] §3) | — | reference `A_ref⁻¹` | per-tet only; no rasterisation needed [I] |

**Autograd through DLPack (ADR 0001).**
- The generic path maps directly onto `torch.autograd.Function`: `rasterize` marks `tet_id`
  non-differentiable and its backward calls `bary_vjp`. nvdiffrast and PyTorch3D do exactly this
  [V: [3] `_rasterize_func.backward` calls `rasterize_grad(pos, tri, out, dy)`; [4]
  `_RasterizeFaceVerts` calls `ctx.mark_non_differentiable(pix_to_face)` and
  `rasterize_meshes_backward(face_verts, pix_to_face, grad_zbuf, grad_barycentric_coords, ...)`].
- With `bary` holding `(π₁, π₂, π₃)`, autograd delivers `ḡ` directly, because the user computes
  `π₀ = 1 − Σ` downstream [I].
- The fused path returns a scalar and a ready gradient. Its `Function` stores the gradient in
  `forward` and scales it in `backward` [I].
- Second derivatives through autograd (double backward) are not provided by this design. Curvature
  is an explicit op (§3) [I].
- JAX was not investigated beyond noting that it needs a `custom_vjp` around a host callback [I].

### 3. Second-order information

**Second derivative of `π`** (from [1] eq. 8 and (1)):

    ∂²π_m / ∂v_i ∂v_jᵀ = π_j p_m p_iᵀ + π_i p_j p_mᵀ                          (4)

**General Hessian block for one voxel.** With `H^π = ∂²L/∂π∂πᵀ ∈ ℝ^{4×4}`, `q = Pᵀg` and
`G_i = −π_i q` (the voxel's gradient (2)):

    ∂²L / ∂v_i ∂v_jᵀ = π_i π_j Pᵀ H^π P − p_j G_iᵀ − G_j p_iᵀ                 (5)

[V: finite differences of a generic non-linear `L(π)` agree to 1.6e-8]. Also
`Pᵀ H^π P = A⁻ᵀ H̄ A⁻¹` with the reduced `H̄ = S H^π Sᵀ ∈ ℝ^{3×3}`, `S = [−1 | I]`
[V: numerically, 3e-17]. So a symmetric 3×3 (6 floats) is the minimal general per-voxel payload.

**The samseg forms are special cases of (5).**
- EM bound ([1] §2.2): `Q = −Σ_m ω_m log π_m` with `ω` frozen. Then `∂Q/∂π_m = −r_m` and
  `∂²Q/∂π_m² = ω_m/π_m² = r_m/π_m`, so `H^π = diag(r_m/π_m)`. Inserting this in (5) gives the
  formula of [1] §2.3 term by term [V: (5) with this `H^π`, the literal §2.3 expression and finite
  differences of `Q` agree to 6e-8].
- Exact data term: `D = −log(l·π)` gives `H^π = r rᵀ`, hence `Pᵀ H^π P = q qᵀ`, a rank-1 PSD
  matrix built from the gradient alone [V: finite differences of `D` agree to 1.3e-8].
- The last two terms of (5) are the geometric part, from (4). They are indefinite. In one test
  voxel the EM-bound block had eigenvalues from −2.1e-4 to 0.76, and the exact block from −4.4e-3
  to 3.9e-3 [V: numerically]. Dropping them gives a Gauss-Newton matrix that is PSD whenever `H^π`
  is [V: numerically PSD for both forms]. [1] keeps them.

**Per-tet accumulation.** Summing (5) over the voxels of a tet:

    H_ij = A⁻ᵀ C_ij A⁻¹ − p_j G_iᵀ − G_j p_iᵀ,     C_ij = Σ_x π_i π_j H̄(x)     (6)

where `G_i` is now the tet's gradient contribution (3).
- General curvature: 10 pairs `i ≤ j` × 6 = 60 accumulators per tet [I].
- Diagonal `H^π` (EM bound): `A⁻ᵀ C_ij A⁻¹ = Σ_m c_{ij,m} p_m p_mᵀ` with
  `c_{ij,m} = Σ_x π_i π_j h_m`: 40 accumulators per tet, exactly the bracket of [1] §2.3 [V].
- Rank-1 (`H^π = g gᵀ`): `C_ij = Σ_x π_i π_j ḡ ḡᵀ`, no extra per-voxel input [I].
- The geometric terms cost nothing extra: they use the 12 gradient moments [I].

**What each option needs.**
- (ii) and (iv): the functor (or cost kernel) returns `h` next to `g`. The scatter adds the
  accumulators above. (iv) writes 4 or 6 more floats per voxel (268 or 403 MB at 256³) [I].
- (i): autograd does not give (5). The user would have to supply `H̄` per voxel and call the same
  scatter [I].
- Keeping the per-tet moments (52 floats, 70 MB) instead of dense `[T,12,12]` blocks (195 MB)
  allows a matrix-free Hessian-vector product in O(T) per product [I].

**Singularity of the EM-bound curvature.** `r_m/π_m → ∞` as `π_m → 0`. In (6) the factor
`π_i π_j / π_m` cancels only when `m ∈ {i, j}`.
- [V: numerically] with `π₃ = 1e-2, 1e-4, 1e-6` the largest eigenvalue of the EM-bound block is
  2.7, 2.6e2, 2.6e4; at `π₃ = 0` it is `inf`. The exact block stays at 0.063.
- Atlas boundary faces lie exactly on voxel-centre planes (`atlas-format.md`), so voxels with
  `π_m = 0` exist in every evaluation [I].
- Current GEMS never forms this Hessian. Its optimisers are L-BFGS, conjugate gradient and gradient
  descent. In the C++ sources "Hessian" appears only in the L-BFGS inverse-Hessian comment
  [V: [2] `grep -rli hessian gems`; the one other hit is an unrelated Matlab bias-field file]. There
  is no reference behaviour to copy.

### 4. Correctness of the gradient

**Claim.** If the per-voxel loss depends on the mesh only through fields interpolated from
per-vertex values, the gradient that ignores ownership changes is the exact gradient of the sampled
cost wherever no voxel centre lies on a tet face.

**Argument** [I].
- `f(x; v) = Σ_n F_n φ_n(x; v)` is continuous in `x` across faces, since both tets interpolate the
  same three vertex values on a shared face. When a voxel changes owner it sits on the shared face,
  where both tets give the same `f`. So `L(v) = Σ_x ℓ_x(f(x; v))` is continuous and piecewise
  smooth.
- Within one ownership configuration, `t(x)` is constant and (2) is the exact derivative.
- A change of owner changes the slope (a kink), not the value. There is no Dirac term to add.

**Numerical check** [V] on the jittered Kuhn mesh:
- along a path with 305 ownership changes, the cost never jumped (largest step 1.4× the median
  step), while a per-tet-constant attribute jumped by up to 22;
- per-coordinate central differences (step 1e-6) matched the analytic gradient to 1.3e-8, and the
  forward and backward directional differences at step 1e-6 both gave 12.5256, the analytic value;
- one sample that started 1.9e-6 (in barycentric units) from a face crossed it at a step between
  1e-6 and 1e-5. Beyond the crossing the forward slope dropped from 12.5 to about 11, while the
  backward slope stayed at 12.5. That is the size of one kink in a problem with 2197 samples.

**What samseg does.** GEMS adds `π_i × (∇_x Σ_k l_k α_k) / Σ_k l_k α_k` per voxel to vertex `i`,
which is (2), and nothing else. There is no boundary or visibility term [V: [2]
`kvlAtlasMeshToIntensityImageCostAndGradientCalculator.cxx` lines 118–171]. Its iterator exposes
the interpolated loadings and their per-axis increments, i.e. `∇_x` of the interpolated field
[V: [2] `kvlTetrahedronInteriorConstIterator.h`].

**Why nvdiffrast is different.** A rendered pixel shows the nearest surface. Moving a silhouette
edge reveals a different surface with different attributes, so the image is discontinuous in `v`.
nvdiffrast states that point-sampled coverage "does not produce vertex position gradients related
to occlusion and visibility effects" and generates them in a separate antialiasing op [V: [3]
`docs/index.html`]. A tet mesh that partitions the volume and shares vertex values has no interior
silhouettes [I].

**When ignoring ownership does matter** [I unless tagged].
- Per-tet (face-discontinuous) quantities: per-tet labels, `∇_x f`, Jacobian-determinant
  modulation. These jump across faces and their gradient would miss boundary terms.
- A non-watertight rasteriser: a voxel that is dropped or double-counted near a face adds or removes
  a whole `ℓ_x`, which is a jump in `L`. Watertightness (R1) is what makes the argument above hold.
- The outer boundary, if covered voxels can change. With boundary vertices sliding in the planes
  `x_d = 0` and `x_d = max_d` the covered set is fixed. GEMS is called with
  `boundaryCondition='Sliding'` [V: [2] `samseg/ProbabilisticAtlas.py`].
- Folded tets (`det A ≤ 0`): the mesh stops being a partition. GEMS aborts the evaluation and
  returns the maximum cost when the prior meets `detJ ≤ 0` [V: [2]
  `kvlAtlasMeshPositionCostAndGradientCalculator.cxx` lines 585–587 and the `m_Abort` path].
- Aliasing, which is a sampling issue and not a gradient error: a tet without a voxel centre gets no
  data gradient. The smallest adult tets are 0.14 voxel³ (`atlas-format.md`).

**Cheap remedies** [I].
- None is needed for exactness.
- Against aliasing: coarse-to-fine optimisation on smoothed, downsampled images, which samseg
  already does (`downSampledMesh`, `multiResolutionLevel` in [2] `samseg/Samseg.py` [V]).
- Supersampling comes for free if the sample lattice is defined by an affine instead of the voxel
  grid (the open roadmap question on the output grid).
- Exact integration over voxel ∩ tet is not worth its cost.

### 5. Prior art

**nvdiffrast** [V: [3] `nvdiffrast/torch/ops.py`, `csrc/common/interpolate.cu`, `rasterize.cu`].
- Four separate ops: `rasterize`, `interpolate`, `texture`, `antialias`.
- `rasterize(pos, tri, resolution) → (u, v, z/w, triangle_id)` per pixel. Its backward takes the
  upstream gradient of that buffer and returns `g_pos`.
- `interpolate(attr, rast, tri)`; its backward returns `g_attr` and `g_rast`. The kernel
  accumulates `gb0 += y·(s0 − s2)`, `gb1 += y·(s1 − s2)`, which is the reduced `ḡ` of (2), and
  scatters `g_attr` with atomics coalesced per triangle within a warp.
- This is option (iv) with the cost in user code (option (i) for the shading).

**PyTorch3D** [V: [4] `renderer/mesh/rasterize_meshes.py`, `ops/interp_face_attrs.py`].
- `rasterize_meshes → (pix_to_face, zbuf, barycentric_coords, dists)`; `pix_to_face` is marked
  non-differentiable; the backward maps `grad_barycentric_coords` to `grad_face_verts`.
- `interpolate_face_attributes(pix_to_face, barycentric_coords, face_attributes)` is a separate op.
  Shading is Python code that calls it and works on materialised per-pixel tensors
  (`renderer/mesh/shading.py`).

**Kaolin** [V: [5] `kaolin/render/mesh/rasterization.py`, `deftet.py`].
- `rasterize(..., face_features) → (interpolated_features, face_idx)` fuses rasterisation and
  interpolation. Its backward takes `grad_interpolated_features` and returns gradients for the
  vertex image positions and the features. This is the `∂L/∂f` (C numbers per pixel) contract.
- Its tet code (`deftet_sparse_render`, marching tetrahedra) renders tet faces along rays. It is not
  a volume voxeliser and nothing is directly reusable.

**samseg / GEMS** [V: [2] at commit `fdac5ad`, files named below]. The code is in
`github.com/freesurfer/samseg` under `gems/`. The main `freesurfer/freesurfer` tree keeps only an
old copy in `attic/gems` (commit `17d3ca6`).
- `AtlasMeshRasterizor` (`kvlAtlasMeshRasterizor.{h,cxx}`): `Rasterize(mesh)` starts the threads;
  thread `k` takes tets `k, k+n, k+2n, …` and calls the pure virtual
  `RasterizeTetrahedron(mesh, tetId, threadNumber)`. The static split is deliberate: it makes
  round-off reproducible for a given thread count, and a `__float128` accumulator can be enabled
  for reproducibility across thread counts.
- Direct subclasses: the drawers (`AlphaDrawer`, `MultiAlphaDrawer`, `ValueDrawer`,
  `JacobianDeterminantDrawer`, `SummaryDrawer`), `VisitCounter`, `Histogrammer`,
  `StatisticsCollector` (for atlas building), and `AtlasMeshPositionCostAndGradientCalculator`.
- `AtlasMeshPositionCostAndGradientCalculator::RasterizeTetrahedron` fetches the four positions and
  per-thread `double` accumulators (one full `[N,3]` gradient array per thread), then calls two
  virtuals:
  - `AddPriorContributionOfTetrahedron` (the Ashburner prior, implemented in this class);
  - `AddDataContributionOfTetrahedron(p0..p3, alphas0..3, cost&, gradient0..3&)` (empty by default).
  After the loop it sums the threads, calls the hook `PostProcessCostAndGradient` and applies the
  boundary condition (none, sliding, affine, translation).
- Data-term subclasses override `AddDataContributionOfTetrahedron` and run their own voxel loop with
  `TetrahedronInteriorConstIterator`:
  - `AtlasMeshToIntensityImageCostAndGradientCalculatorBase` owns a `LikelihoodImageFilterBase`
    whose output is a per-voxel array of class likelihoods (empty for masked voxels), updated before
    each rasterisation. Subclasses: GMM (`…ToIntensityImage…`), `…LogDomain…`, and the diffusion
    models (Wishart, Frobenius, DSWbeta);
  - `AtlasMeshToLabelImage…`, `MutualInformation…`, `ConditionalGaussianEntropy…`,
    `AtlasMeshToPointSet…` (post-process hook only), `AverageAtlasMeshPosition…`.
- So GEMS decouples by the template-method pattern: the tet loop is the base class, the cost is a
  virtual per-tet method. Cost and gradient are not detached from each other: every subclass
  hand-codes value and gradient together, and each consumer re-rasterises the mesh [V for the
  structure; I for the assessment].
- GEMS materialises the per-voxel likelihood vector, but only over the merged classes, since the
  mesh `α` were merged in Python first [V: [2] `kvlGMMLikelihoodImageFilter.hxx`,
  `samseg/Samseg.py`].

## Options compared

| Option | Pros | Cons | Fit for tetvox |
|---|---|---|---|
| (i) cost in user code on materialised fields | any cost; plain autograd; no C++ | 9–12 GB at 256³ × 44; 15–21 GB traffic per evaluation | Provide it (`interpolate` + VJPs). Use for small C only |
| (ii) fused C++/CUDA functor | no per-voxel intermediates; ~0.35 GB traffic; exploits sparse `α` | rebuild to add a cost; not definable from Python | Use for the samseg data term and other built-ins |
| (iii) encode in the mesh | reduces K before the kernel | cannot express the cost; `l_j` is per voxel and per vertex | Only as preprocessing: merged, sparse `α` |
| (iv) two-pass through `∂L/∂π` | one generic scatter; cost pluggable from Python; ~1 GB traffic | cost kernel still gathers vertex data; 0.2–0.5 GB of buffers | Make it the contract and the generic path |
| GEMS style (virtual per-tet method) | proven; simple on CPU | no virtual dispatch in kernels; value and gradient entangled; re-rasterises per consumer | Do not copy the structure; reuse the maths |

## Open questions

- **Tet-parallel or voxel-parallel accumulation?** Tet-parallel loads the vertex data once per tet,
  owns its moments without atomics and sums in a fixed order. Voxel-parallel needs atomics into
  per-tet moments. This depends on R1's rasteriser. Settle it with a benchmark of both on the adult
  atlas once the raster buffer exists.
- **Determinism and precision of the sums.** GEMS accumulates in `double` and offers `__float128`
  for reproducibility [V]. Are float32 per-tet moments enough when a tet holds up to ~10⁵ voxels?
  Compare float32 moments with a float64 CPU reference on the adult atlas. The total cost should be
  summed in double in any case [I].
- **Which curvature does the owner want?** Clamped EM bound (`r_m / max(π_m, ε)`), exact `r rᵀ`, and
  with or without the indefinite geometric terms. Ask the owner; compare convergence on a real
  registration.
- **Store `π` or recompute it from `tet_id`?** 16 B against 4 B per voxel. Benchmark.
- **Masked voxels.** samseg evaluates only voxels inside a mask [V: `downSampledMask` in [2]]. Should
  the generic ops take a mask or a compact voxel list? This ties in with R1's output layout.
- **Output allocation.** ADR 0001 leaves the memory policy open. The sketch below assumes
  caller-allocated outputs.
- **Is 256³ × 44 the real target?** The local atlases are 181×217×181 voxels, and deformation runs on
  8 merged classes. The 44-class field is needed for the final segmentation only. A fused
  "posterior arg-max and volumes" functor would avoid materialising it. Ask the owner.
- **JAX.** Not investigated. Decide whether it is a supported target before designing for it.
- Not read: the nvdiffrast paper (Laine et al. 2020). The findings above rest on its code and
  documentation only.

## Impact on design

**Ops.** Shapes use the grid `(X, Y, Z)`, `V = X·Y·Z`, `N` vertices, `T` tets, `C` channels. All
arrays are float32 unless noted, and live on one device.

Python-facing (`tetvox`, framework-agnostic through DLPack):

```python
def rasterize(positions, tetrahedra, shape) -> tuple[Array, Array]:
    """positions (N, 3); tetrahedra (T, 4) int32; shape (X, Y, Z).
    Returns tet_id (X, Y, Z) int32, -1 where no tet owns the voxel,
    and bary (X, Y, Z, 3) = (pi1, pi2, pi3); pi0 = 1 - sum."""

def interpolate(tet_id, bary, tetrahedra, values) -> Array:
    """values (N, C). Returns (X, Y, Z, C)."""

def interpolate_vjp(tet_id, bary, tetrahedra, values, grad_out) -> tuple[Array, Array]:
    """grad_out (X, Y, Z, C). Returns grad_values (N, C) and grad_bary (X, Y, Z, 3)."""

def bary_vjp(positions, tetrahedra, tet_id, bary, grad_bary) -> Array:
    """grad_bary (X, Y, Z, 3) = dL/d(pi1, pi2, pi3). Returns grad_positions (N, 3). Eq. (3)."""

def bary_curvature(positions, tetrahedra, tet_id, bary, grad_bary, hess_bary,
                   *, geometric_terms: bool) -> Array:
    """hess_bary (X, Y, Z, 4): diagonal of d2L/dpi2 over pi0..pi3, or (X, Y, Z, 6): reduced
    symmetric 3x3. Returns per-tet blocks (T, 12, 12). Eq. (6)."""

def samseg_data_term(positions, tetrahedra, tet_id, bary, alphas, images, gmm,
                     *, curvature: Literal["none", "em", "exact"]) -> DataTerm:
    """Fused. alphas: sparse per-vertex priors; images (X, Y, Z, n_contrasts).
    Returns cost (float), grad_positions (N, 3), tet_blocks (T, 12, 12) or None."""
```

`tetvox.torch` wraps `rasterize` and `interpolate` as `autograd.Function`s (`tet_id`
non-differentiable; backwards call `bary_vjp` and `interpolate_vjp`) and wraps `samseg_data_term`
as a scalar with a stored gradient.

C++ core (`namespace tetvox`, raw device pointers and a stream, as in ADR 0001):

```cpp
struct Mesh   { const float* positions; const int32_t* tets; int32_t num_vertices, num_tets; };
struct Raster { const int32_t* tet_id; const float* bary; int64_t num_voxels; };

// Shared maths, csrc/include/tetvox/, __host__ __device__ (tested against the CPU reference).
struct TetMoments { float m[4][3]; float c[10][4]; };       // gradient / diagonal curvature
void add_voxel(TetMoments&, const float pi[4], const float gbar[3], const float h[4]);
void finalize(const float a_inv[9], const TetMoments&, bool geometric_terms,
              float grad[4][3], float block[12][12]);

// Generic path.
void interpolate(Raster, const int32_t* tets, const float* values, int channels, float* out,
                 cudaStream_t);
void interpolate_vjp(Raster, const int32_t* tets, const float* values, const float* grad_out,
                     int channels, float* grad_values, float* grad_bary, cudaStream_t);
void bary_vjp(Mesh, Raster, const float* grad_bary, float* grad_positions, cudaStream_t);

// Fused path: one kernel instantiation per cost, no virtual calls on the device.
//   struct Cost {
//     struct TetData;                                      // e.g. active classes and their alphas
//     __device__ void load_tet(int32_t tet, const int32_t vertex[4], TetData&) const;
//     __device__ float operator()(int64_t voxel, const float pi[4], const TetData&,
//                                 float g[4], float h[4]) const;   // returns the voxel's cost
//   };
template <class Cost>
void accumulate(Mesh, Raster, const Cost&, double* cost, float* grad_positions,
                float* tet_blocks /* nullable */, cudaStream_t);
```

**What is fused.** In `accumulate<Cost>`: the vertex-data gather, the likelihood evaluation,
`∂cost/∂π`, the curvature and the moment accumulation run in one kernel. Nothing per-voxel is
written. Whether rasterisation is fused in as well is left to R1 and a benchmark; the API does not
expose the choice.

**Data structures.**
- Raster buffer: `tet_id` int32 and `bary` 3×float32, dense over the grid.
- Per-vertex priors: fixed-width sparse rows (class index and value, 8 entries cover the adult
  atlas), built once per atlas. Merge classes before building them.
- Per-tet table recomputed every iteration: `A⁻¹`, `v₀`, `det A` (shared with the prior term).
- Optional output of the per-tet moments, for matrix-free Hessian-vector products.

**Precision.** Total cost in double. Per-tet moments in float32 or float64 according to the open
question above. Costs must clamp wherever they divide by `π_m` or take `log` of a sum that can
reach zero (GEMS adds `1e-15` [V]).

**Tests.**
- `bary_vjp` and `finalize` against finite differences in float64 on the Kuhn-cube fixture (the
  author's throwaway checks reached 1e-8 to 1e-10).
- `samseg_data_term` gradient equals `rasterize → interpolate → cost in numpy → interpolate_vjp →
  bary_vjp` on a small mesh: the fused and generic paths must agree.
- EM-bound blocks equal the literal [1] §2.3 formula; exact blocks equal finite differences.
- Continuity: along a random vertex path with many ownership changes, the cost has no jumps. This
  doubles as a watertightness test.

## References

[1] K. Van Leemput, *Working with tetrahedral meshes in samseg*, note dated 25 July 2025. Local copy
`literature/equations.pdf` (not in git). §1 footnote 1, §2.1–2.3, §3.

[2] FreeSurfer, *samseg* repository (contains the GEMS C++ code under `gems/`), commit `fdac5ad`
(12 Nov 2026). https://github.com/freesurfer/samseg — also `freesurfer/freesurfer` commit `17d3ca6`
(17 Nov 2026), where only `attic/gems` remains. https://github.com/freesurfer/freesurfer

[3] NVlabs, *nvdiffrast*, commit `253ac4f` (5 Dec 2025): `nvdiffrast/torch/ops.py`,
`csrc/common/interpolate.cu`, `csrc/common/rasterize.cu`, `docs/index.html`.
https://github.com/NVlabs/nvdiffrast

[4] Meta, *PyTorch3D*, commit `ddfc551` (17 Nov 2026): `pytorch3d/renderer/mesh/rasterize_meshes.py`,
`pytorch3d/ops/interp_face_attrs.py`. https://github.com/facebookresearch/pytorch3d

[5] NVIDIA, *Kaolin*, commit `5292e99` (20 Nov 2026): `kaolin/render/mesh/rasterization.py`,
`kaolin/render/mesh/deftet.py`. https://github.com/NVIDIAGameWorks/kaolin

[6] Local sources: `docs/knowledge/samseg-math.md`, `docs/knowledge/atlas-format.md`,
`docs/decisions/0001-bindings-nanobind-dlpack.md`, `atlases/adult_template/atlas_adult.npz` and
`sharedGMMParameters_adult.txt` (not in git).

Numerical checks were throwaway numpy scripts run outside the repository (single-tet finite
differences of (1)–(6); a whole-mesh gradient and continuity test; `α` sparsity statistics; the
EM-bound curvature near a face). They are not kept; the test list above says how to reproduce them.
