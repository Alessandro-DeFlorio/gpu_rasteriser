# samseg tetrahedral-mesh maths

Source: K. Van Leemput, *"Working with tetrahedral meshes in samseg"* (25 Jul 2025), local copy
`literature/equations.pdf` (not in git). Section numbers below refer to that note. Use its notation
in code and docs.

## Notation

| Symbol | Meaning |
|---|---|
| `v_j`, j=0..3 | tet vertex positions (image/voxel grid coordinates) |
| `A = [v1−v0, v2−v0, v3−v0]` | 3×3 edge matrix; maps standard tet → actual tet: `x = Aπ + v0` |
| `π = (π0..π3)` | barycentric coords: `π_{1..3} = A⁻¹(x − v0)`, `π0 = 1 − π1 − π2 − π3`; inside ⇔ all `π_j ≥ 0` |
| `α_{j,k}` | prior probability of class k at vertex j (`alphas` in the npz), `Σ_k α_{j,k} = 1` |
| `l_k` | per-voxel likelihood of class k (from the intensity model, e.g. GMM) |
| `l_j = Σ_k l_k α_{j,k}` | per-voxel "vertex likelihood" |
| `s_m` | `e_m` for m=1..3, `−1 = −(1,1,1)ᵀ` for m=0 |
| `p_m = A⁻ᵀ s_m` | columns of `A⁻ᵀ` (m=1..3) and their negative sum (m=0); gradient of `π_m` w.r.t. `x` |
| `A_ref`, `J = A A_ref⁻¹` | reference (zero-penalty) shape and the affine deformation of the tet |

The closed-form 3×3 inverse (cofactors / `det`) is in §1. Shared faces/edges must be owned by exactly
one tet (top-left style rule, footnote 1 of §1). This is what our rasteriser has to guarantee.

## Data term (§2)

Per voxel inside a tet: `α_k = Σ_j α_{j,k} π_j` (piecewise-linear interpolation, eq. 4) and

    D = −log Σ_k l_k α_k = −log Σ_j l_j π_j          (eqs. 5–6)

Gradient w.r.t. vertex i (summed over the V voxels in the tet, then over tets):

    ∂D/∂v_i = A⁻ᵀ Σ_v r̄^v π_i^v,   r̄ = (r1−r0, r2−r0, r3−r0)ᵀ,   r_j = l_j / Σ_j' l_j' π_j'
            = Σ_v π_i^v Σ_m r_m^v p_m                                          (§2.3)

Note `r_j = −∂D/∂π_j`. The only voxel-dependent input to the geometric part of the gradient is
`∂D/∂π`, because `∂π_j/∂v_i = −A⁻ᵀ e_j π_i` (§2.1). This is the basis for decoupling the cost from the
rasteriser (see `docs/research/` R2).

Hessian (§2.2): ignore the derivative of the EM weights `ω_j = l_j π_j / Σ l π` (EM upper-bound
curvature). Its compact form is in §2.3. It is built from `π_i π_j`, `r_m/π_m`, the outer products
`p_m p_mᵀ` and the per-tet gradients. Copy it from the PDF when implementing: the text extraction of
that equation is unreliable. Each tet contributes a 4×4 grid of 3×3 blocks (12×12), so the global Hessian is very sparse.

## Prior term (§3, Ashburner)

    P = (|A_ref| + |A|)/6 · (‖J‖²_F + ‖J⁻¹‖²_F − 6)  =  V · S

No SVD is needed. Reuse `A⁻¹` from the data term and precompute `A_ref⁻¹` and `|A_ref|` once. With
`M = J Jᵀ`:
- `∂‖J‖²/∂v_i = 2 M p_i`
- `∂‖J⁻¹‖²/∂v_i = −2 M⁻¹ p_i`
- `∂V/∂v_i = |A| p_i / 6`
- `∂P/∂v_i = V ∂S/∂v_i + S ∂V/∂v_i`

The Hessian is in §3.3. Simplified variant: `V = |A_ref|/3`, so only the first terms are needed. The
full version is needed only for the longitudinal subject-specific warp. The total prior is the sum over
tets, scaled by `κ`.

## Deformation parametrisation (§4) and atlas building (§5)

Vertices can be a linear model `v = Φθ + δ`. Gradients and Hessians map via `Φᵀ(·)` and
`Φᵀ(·)Φ`. §5 covers estimating `α` and the reference mesh itself. This is not needed for the
rasteriser.
