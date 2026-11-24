# R3 — A dimension-generic rasteriser: cost, design and whether it is worth it

Status: reviewed (owner, 2026-11-24): its decision list was answered in ADR 0010.
Brief: `briefs/R3-dimension-generic-rasteriser.md`

Every claim is tagged:
- **[verified]**: read in the cited source;
- **[measured]**: result of a throwaway script, C++ program or CUDA kernel written for this report
  (method in §5.1; nothing was added to the repo);
- **[inferred]**: my own reasoning.

Notation follows R1 §2.2 and `raster.hpp`: `S = 2ᵏ`, `M = 2^(k+L)`, `q` the snapped vertices, `O` the
per-simplex integer origin, `E_j` the face function of the face opposite vertex `j`,
`t_j = n_j·(X − O) + c_j`. `d` is the dimension; a `d`-simplex has `d + 1` vertices and faces.

## Question
Can the rasteriser (ownership rule, exact arithmetic, per-element procedures, K3 traversal) be
written once for any dimension `d` with compile-time generic loops, at the speed of
dimension-specific code? Or does each dimension need its own optimised code? Is it worth doing, and
is now (before the CUDA backend) the moment? Seven sub-questions: (1) theory in `d` dimensions,
(2) bit budget as a function of `d`, (3) traversal per dimension, (4) implementation techniques and
their cost, (5) measurements, (6) scope beyond the rasteriser, (7) use cases.

## Recommendation
1. **It can be written once, and it costs nothing at run time for `d` = 2 and 3.** The rule, the
   snapping, the `c_j` folding, the barycentric formula and K3 contain nothing specific to
   `d = 3` (§1). A throwaway `template <int D>` header reproduces the current header **bit for bit**
   on the adult atlas (records, owners, π; host and device). On the GPU it runs in the same time
   (rows 4.5 ms against 4.5 ms, setup 0.48 against 0.50 ms), with nearly the same registers (72 against 70)
   and no local memory. On the CPU it is within the noise of the inliner (−5 % to +9 % depending on
   two coding details, §5.2).
2. **Recommended option: (c) in its lightest form.** One generic code path, `template <int D>`,
   plus a small per-`D` traits struct, instantiated for **`D` = 2 and 3 only**. No hand-written
   per-dimension specialisation of any procedure: none paid for itself (a hand-written int32 2D
   path equals the generic one instantiated with the same widths, §5.4).
3. **Integer types need not depend on `D` for `D` ∈ {2, 3}.** Use int64 per sample and `__int128`
   in the setup for both. Only the budget constant depends on `D`:
   `(D − 1)·k + D·L ≤ 60` for `D = 2`, `≤ 59` for `D = 3` (§2). In 2D this gives `L = 23` at
   `k = 14`. An int32 per-pixel 2D path (`k + 2L ≤ 28`, e.g. `k = 8`, `L = 10`) is at most 12 %
   faster on the GPU (0 % at 4096²) and would force a different default `k`: not worth a second
   record dtype.
4. **Keep K3 as the only traversal for both dimensions.** In 2D it was the best or within 36 % of
   the best stateless alternative on every mesh tested, while each alternative has a bad case (K1
   up to +70 % on small triangles, one thread per triangle 9× slower on large ones) (§3).
5. **Do not support `d ≥ 4`.** At `k = 14` the per-sample test needs `__int128` (4.3× slower rows,
   160 registers, 496-byte local-memory frames, and no numpy/DLPack dtype for the record), or
   `k ≤ 5` with int64. K3 needs 6.7 row set-ups per owned cell. I found no user (§2, §5.5, §7).
6. **When.** If the owner wants 2D at all, now is the cheapest moment: template the shared header
   and the CPU backend first (public 3D API unchanged, checked bit-identical against the current
   code), then write the CUDA K3 kernels once on the generic procedures. Do it only together with a
   tested `D = 2` instantiation; a template with one instantiation would be untested generality.
   If the owner does not want 2D, stay 3D-only (option (a)): converting the internals later is
   mechanical and protected by the bit-identity tests, roughly twice the work because two backends
   exist by then. What becomes expensive later is only renaming the public API (`tet…` names,
   `(X, Y, Z, 3)` shapes), so the naming decisions of "Impact on design" should be taken before the
   first release either way.
7. **Rules for the generic device code** (each one cost time in the prototype, §4.3): C arrays,
   never `std::array`, in device paths; `template <std::size_t D>` or a non-deduced context if a
   `std::array<…, D>` is a parameter; mark `constexpr` helpers `TETVOX_HD`; on the CPU pass loop
   indices as values (a parameter pack), not as an array shared by reference, and declare every
   helper template `inline`; solve the row relative to the box origin if the test type can differ
   from the index type. `#pragma unroll` is not needed.
8. **Kernel tuning should target memory traffic, not arithmetic.** Stepping `t_j` along the row
   instead of calling `plane_values` per voxel, int32 instead of int64, and rows along the
   contiguous axis all left the GPU time unchanged on atlas-like meshes (§5.3, §5.4). This is why
   genericity is free here, and it is the main lesson for the CUDA item.

## Findings

### 1. Theory in d dimensions

#### 1.1 Face functions, orientation, barycentrics
Setting: a simplex with vertices `v_0..v_d ∈ ℝᵈ`, `A = [v_1 − v_0, …, v_d − v_0]`, `D = det A > 0`
(the positive orientation; for `d = 3` this is the convention of `rasteriser-cpu.md`).
- Define `E_j(x)` as `det A` with `v_j` replaced by `x` (equivalently the determinant of the
  `(d+1)×(d+1)` matrix with rows `(1, v_i)`, row `j` replaced by `(1, x)`). It is affine in `x`,
  `E_j(v_i) = D·δ_ij`, `Σ_j E_j(x) = D`, and `π_j = E_j / D`. **[inferred]** (multilinearity of the
  determinant; this is how `tests/exact_reference.py` already defines the plane functions for
  `d = 3`.)
- `E_j(x) = n_j·(x − a)` for any vertex `a ≠ v_j`, with `n_j = ∇E_j = D·p_j` (`p_j` of
  `samseg-math.md`). Two equivalent closed forms: **[inferred]**
  - for `j = 1..d`, `n_j` is row `j` of `adj(A)`, and `n_0 = −Σ_{j≥1} n_j`;
  - for any `j`: let `a` be the lowest-numbered vertex other than `j` and `e_1..e_{d−1}` the edges
    from `a` to the other face vertices in increasing order. Then
    `(n_j)_m = (−1)^(j+m+1) · det(e with column m deleted)`, a `(d−1)×(d−1)` minor (0-based `j`,
    `m`). This is the generalised cross product with the sign that makes `n_j` point inwards. The
    prototype uses this form; for `d = 3` it is bit-identical to the current face table and cross
    product on all 338,993 tets of the adult atlas (§5.2). **[measured]**
  - `d = 2`: `n_j = (−e_1, e_0)` for the edge `e` opposite `j`, directed with the triangle on its
    left: the Pineda edge function of R1 §1.1. No multiplication.
  - `d = 3`: the face table `{{1,3,2},{0,2,3},{0,3,1},{0,1,2}}` of `raster.hpp` is this formula
    with the sign absorbed into the vertex order.
- `E_j(v_j) = D`, so `D = n_d·(v_d − v_0)`, as the header computes it. **[inferred]**

#### 1.2 Ownership rule and proof
**Rule.** `X` is owned by the simplex iff, for all `d + 1` faces,
`E_j(X) > 0`, or `E_j(X) = 0` and the first non-zero component of `n_j` (axis 0 first) is positive.

**Proof that it partitions the domain exactly once** (R1 §1.2, with `3` replaced by `d`).
**[inferred]**; **[measured]** exhaustively for `d` = 2, 3, 4 below.
1. `E_j(X + (ε, ε², …, εᵈ)) = E_j(X) + Σ_m n_{j,m} ε^(m+1)`. This is a polynomial in `ε` whose
   coefficients are not all zero (`n_j ≠ 0` because `E_j(v_j) = D ≠ 0 = E_j(a)`). For small
   `ε > 0` its sign is that of the lowest-order non-zero coefficient. So the rule says: the shifted
   point `X_ε` is strictly inside. The moment curve `(ε, ε², …, εᵈ)` is what makes this work in any
   `d`: it leaves every hyperplane through the origin.
2. Finitely many simplices and samples, so one `ε` serves all; `X_ε` then lies on no face
   hyperplane of any simplex.
3. In a tiling (interiors pairwise disjoint, union = domain) a point on no face hyperplane lies in
   the interior of exactly one simplex, or outside the domain. Hence one owner, or none.
4. Lower-dimensional faces (in `d` dimensions a sample can lie on up to `d` face hyperplanes of one
   simplex) need no special case: all the face tests must pass, which is again "`X_ε` inside".
5. Two simplices sharing a face see `E` and `−E`, `n` and `−n`: exact signs give opposite
   decisions.
- The tiling precondition is the same as in R1: every `D > 0` and a non-self-intersecting boundary
  (the signed count of simplices containing a generic point equals the winding number of the
  boundary, a degree argument that holds in any dimension). **[inferred]**
- Simulation of Simplicity is presented as "a general-purpose programming technique" for
  degenerate input of geometric algorithms. **[verified]** [2] (abstract only; that it is stated
  for determinants in arbitrary dimension is my recollection of the paper, **[inferred]**).

**Exhaustive check with Python integers** (`theory_check.py`, no rounding, the explicit rule and
the folded form `t_j ≥ 0`, Kuhn triangulations of a grid of `d`-cubes): **[measured]**

| d | mesh | simplices | samples | exact ties `E_j = 0` | half-open box owned once, rest unowned | folded = explicit | closed-box variant once | "boundary always inside": multiply owned |
|---|---|---|---|---|---|---|---|---|
| 2 | lattice-aligned | 72 | 625 | 1,080 | yes | yes | yes | 5 |
| 2 | jittered | 72 | 625 | 107 | yes | yes | yes | 0 |
| 3 | lattice-aligned | 162 | 1,000 | 10,368 | yes | yes | yes | 54 |
| 3 | jittered | 162 | 1,000 | 1,635 | yes | yes | yes | 0 |
| 4 | lattice-aligned | 384 | 625 | 51,840 | yes | yes | yes | 113 |
| 4 | jittered | 384 | 625 | 10,186 | yes | yes | yes | 0 |

Larger checks through the C++ prototype (`count_claims`, §5): 2D Delaunay meshes on 4096² (the
16,769,025 pixels of `[0, 4095)²` claimed once, the 8,191 on the far edges by none), the adult
atlas, and a jittered 4D Kuhn mesh (6,250,000 cells of `[0, 50)⁴` once, the rest of the 51⁴ grid by
none). **[measured]**

#### 1.3 The outer boundary
- Half-open rule: `X` is owned iff `X_ε` is inside the domain. For a box `Π [0, max_m]` this is
  `X_m ∈ [0, max_m)` on every axis, i.e. the half-open box, in any `d`. **[inferred]**,
  **[measured]** (table above, and §5).
- The exact closed-box variant of R1 §1.3 (shift `(s_0 ε, …, s_{d−1} εᵈ)` with `s_m = −1` where
  `X_m = max_m`) also generalises: every sample of the closed box was owned once for `d` = 2, 3, 4.
  **[measured]**
- "A boundary face always passes" double-counts in every dimension (5, 54 and 113 samples in the
  small lattice-aligned meshes). **[measured]** The reason given in R1 does not depend on `d`.

#### 1.4 What changes with d
Nothing qualitative in the rule, the snapping or the barycentric formula. In detail:
- **Integer identity and folding**: `E_j(S·X) = S·(n_j·(X − O)) + E_j(S·O)` and
  `c_j = floor((E_j(S·O) − 1 + own_j)/S)` hold as written; `f_j ∈ [0, 1]` with the same derivation
  as `rasteriser-cpu.md`, exact in float32 for `k ≤ 24`. **[inferred]**, **[measured]** (folded =
  explicit for `d` = 2, 3, 4).
- **Barycentrics** `π_j = (T(t_j) + f_j)·(2ᵏ/D)`: unchanged; the raster buffer holds `π_1..π_d`.
  The numerator is exact before one conversion, so the relative error is a few ulps for any `d`.
  **[inferred]** One practical change appears only when `t_j` is `__int128` (`d = 4`): the
  conversion `T(t_j)` then needs a helper like `to_double_rn` to be host/device identical. In my 4D
  run the plain cast agreed bit for bit, but all values were small. **[measured]** (agreement) /
  **[inferred]** (not a guarantee).
- **Quantitative changes**: the bit budget (§2); `d + 1` faces of `d` terms per test and `d + 1`
  divisions per row; the snap displacement bound `√d·2^(−k−1)`; the bounding-box fill, at most
  `h_d/d!` = 1/2, 1/3, 1/8 for `d` = 2, 3, 4 (`h_d` in §2.3). **[inferred]**
- **A convention to state for 2D.** "Axis 0 first" equals the Direct3D top-left rule only if axis 0
  is the horizontal image axis and axis 1 points down. For a numpy image indexed `[row, col]` with
  positions given as `(row, col)`, the rule is "top, then left", which decides slanted edges
  through pixel centres differently from D3D. Both are valid tilings; results differ from a
  graphics rasteriser only at exact ties. Likewise "positive" means `det[v_1−v_0, v_2−v_0] > 0` in
  `(axis 0, axis 1)` order, which is clockwise on screen for `(row, col)`. **[inferred]** (D3D rule
  as verified in R1 §1.1 [1])

### 2. Bit budget as a function of d

#### 2.1 Derivation (R1 §2.2 generalised)
Assume the window test: per axis, the span of the snapped vertices is at most `M = 2^(k+L)`, and
samples lie in the simplex's box, so `|X − O| ≤ 2ᴸ`. **[inferred]** throughout.

| quantity | how it arises | bound, R1 style (Leibniz) | tight bound (§2.3) |
|---|---|---|---|
| edge component | difference of two `q` | `M` | `M` |
| `n_j` component | `(d−1)×(d−1)` minor of edges | `(d−1)!·M^(d−1)` | `h_{d−1}·M^(d−1)` |
| `D`, `E_j(S·O)` | `d×d` determinant | `d!·Mᵈ` | `h_d·Mᵈ` |
| `c_j` | `floor((E_j(S·O) − 1 + own)/S)` | `d!·Mᵈ/S + 1` | `h_d·Mᵈ/S + 1` |
| `n_j·(X − O)` | `d` products | `d!·M^(d−1)·2ᴸ` | — |
| `t_j` | sum of the two | `2·d!·2^((d−1)k + dL) + 1` | `h_d·2^((d−1)k + dL) + 1` |

- The per-sample test fits a signed `W`-bit integer iff the `t_j` bound is `< 2^(W−1)`; the setup
  fits iff the `E_j(S·O)` bound is `< 2^(W−1)`.
- With the R1-style bound: `(d−1)·k + d·L ≤ W − 2 − floor(log2(2·d!))`. For `d = 3`, `W = 64` this
  is `2k + 3L ≤ 59`, the constant of ADR 0006/0008.
- Unclipped origin (ADR 0009): when a box is empty on an axis, `O` lies outside the hull by less
  than a voxel, so the setup bound becomes `d!·(M + S)ᵈ`. The prototype's window function uses the
  safe form `d!·2ᵈ·Mᵈ` for the setup width.

#### 2.2 The table
Largest exponent sum allowed, R1-style bound (tight bound in brackets). **[inferred]**; the
formulas are checked numerically in §2.4.

| d | per-sample condition | int32 | int64 | `__int128` | setup condition | int64 | `__int128` |
|---|---|---|---|---|---|---|---|
| 2 | `k + 2L ≤` | 28 (30) | 60 (62) | 124 (126) | `2M²`: `k + L ≤` | 30 | 62 |
| 3 | `2k + 3L ≤` | 27 (29) | **59** (61) | 123 (125) | `6M³`: `k + L ≤` | 20 | 41 |
| 4 | `3k + 4L ≤` | 25 (29) | 57 (61) | 121 (125) | `24M⁴`: `k + L ≤` | 14 | 30 |

Worked `(k, L)` pairs (window = `2ᴸ` samples per axis):

| d | per-sample type / setup type | feasible `(k, L)` | remark |
|---|---|---|---|
| 2 | int32 / int64 | (4, 12), (8, 10), (12, 8), (14, 7) | cudaraster uses 4 sub-pixel bits and a 2048² viewport to stay in 32 bits **[verified]** [5]; that is `k + 2L = 26` |
| 2 | int64 / int64 | (14, 16), (20, 10) | the setup (`k + L ≤ 30`) binds; no `__int128` anywhere |
| 2 | int64 / `__int128` | (14, 23), (24, 18) | same types as 3D |
| 3 | int64 / `__int128` | (14, 10), (16, 9), (24, 3) | ADR 0006 |
| 4 | int64 / `__int128` | (14, 3), (8, 8), (5, 10) | at `k = 14` the window is 8 voxels |
| 4 | `__int128` / `__int128` | (14, 16) | the setup (`k + L ≤ 30`) binds; `L = 15` with the safe setup form of §2.1, as in §5.5 |

Consequences: **[inferred]**
- The integer types are dimension-dependent only at the extremes: 2D *may* use int32 per pixel (at
  the price of `k ≤ 8` for a 1024-pixel window), 4D *must* use `__int128` per sample to keep
  `k = 14`. For `d` ∈ {2, 3}, int64 per sample with `__int128` setup serves both; only the budget
  constant (60 or 59) differs.
- A "write once" design therefore needs a traits struct per `D` that carries the budget constant,
  and it can carry the integer types if the owner ever wants an int32 2D path. The prototype does
  this (`Ints32`, `Ints64`, `Ints128`) and all three instantiate from the same source.
- `__int128` in the record is a hard obstacle beyond speed: numpy and DLPack have no 128-bit
  integer dtype, so ADR 0009's "transparent record" could not be exposed for 4D without splitting
  each value into two words.

#### 2.3 The tight bound (a by-product)
`E_j(x)` is the determinant of `d + 1` points of an axis-aligned cube of side `M`, and it is affine
in each coordinate of each point. Its maximum over the cube is therefore reached with every point
at a cube corner, where it equals `Mᵈ` times a `d×d` 0/1 determinant (reflect the axes to move one
point to the origin). The largest such determinant is `h_d` = 1, 1, 2, 3, 5, 9 for `d` = 1..6
(Hadamard's maximal determinant problem for {0,1} matrices). **[inferred]** (argument),
**[verified]** [7] (values). Partial sums of `t_j` taken in axis order starting from `c_j` are
plane values at other box points, so they obey the same bound. **[inferred]**
- For `d = 3` this gives `|t_j| ≤ 2·2^(2k+3L) + 1`, i.e. `2k + 3L ≤ 61` instead of 59; 60 if the
  products are summed before `c_j` is added, as `row_span` does today. The constant 59 is safe and
  conservative by one to two bits. I do not propose changing it here (see "Open questions").

#### 2.4 Numerical check, d = 2, 3, 4
Python integers; all simplices with vertices on the corners of the window cube (the extreme
points of §2.3) plus 3,000 random ones, samples at the box corners, partial sums included.
**[measured]** Every observed maximum is below the R1-style bound and equals the tight one.

| d, k, L | max `\|n\|` (bound) | max `\|D\|` (bound) | max `\|c\|` (bound) | max `\|t\|` incl. partial sums (bound) |
|---|---|---|---|---|
| 2, 8, 10 | 2¹⁸ (2¹⁸) | 2³⁶ (2³⁷) | 2²⁸ (2²⁹) | 2²⁸ (2³⁰) |
| 2, 14, 23 | 2³⁷ (2³⁷) | 2⁷⁴ (2⁷⁵) | 2⁶⁰ (2⁶¹) | 2⁶⁰ (2⁶²) |
| 3, 14, 10 | 2⁴⁸ (2⁴⁹) | 2⁷³ (2⁷⁴·⁵⁸) | 2⁵⁹ (2⁶⁰·⁵⁸) | 2⁵⁹ (2⁶¹·⁵⁸) |
| 3, 16, 9 | 2⁵⁰ (2⁵¹) | 2⁷⁶ (2⁷⁷·⁵⁸) | 2⁶⁰ (2⁶¹·⁵⁸) | 2⁶⁰ (2⁶²·⁵⁸) |
| 4, 5, 10 | 2⁴⁶ (2⁴⁷·⁵⁸) | 2⁶¹·⁵⁸ (2⁶⁴·⁵⁸) | 2⁵⁶·⁵⁸ (2⁵⁹·⁵⁸) | 2⁵⁶·⁵⁸ (2⁶⁰·⁵⁸) |
| 4, 14, 15 | 2⁸⁸ (2⁸⁹·⁵⁸) | 2¹¹⁷·⁵⁸ (2¹²⁰·⁵⁸) | 2¹⁰³·⁵⁸ (2¹⁰⁶·⁵⁸) | 2¹⁰³·⁵⁸ (2¹⁰⁷·⁵⁸) |

The C++ prototype derives `L` from `k` with the R1-style formula and passed the exhaustive
exactly-once checks in all three dimensions (§1.2, §5). **[measured]**

### 3. Traversal per dimension

#### 3.1 What is traversal-independent
`snap`, `setup` (normals, `c_j`, `f_j`, `2ᵏ/D`, box, status), `plane_values`, `owns`,
`barycentric`, and, from R2, `interpolate`, the moments `m_i = Σ π_i ḡ` and
`vertex_grad` (`G_i = −(2ᵏ/D) Σ_m m_{i,m} n_m`) are formulas over `d + 1` faces and `d` axes. They
depend on `d` only through loop bounds. **[inferred]** `row_span` is "solve `t_j ≥ 0` along one
axis" and is equally generic; the prototype has it for any axis as a template parameter.
**[measured]** (it compiles and gives identical owners along axis 0 and along the last axis.)

What depends on the traversal: flattening `(simplex, i_1, …, i_{d−1})` into a row index (the scan,
the binary search, `d − 2` divisions to unravel), the row axis, and the write pattern.

#### 3.2 Workload by dimension **[measured]**

| mesh | owned samples | box candidates per owned sample | rows per owned sample | row-kernel time per owned sample (GPU, int64) |
|---|---|---|---|---|
| 2D Delaunay, 341,650 triangles, 4096² | 16.77 M | 2.52 | 0.22 | 0.24 ns |
| 2D Delaunay, same triangles, 2048² | 4.19 M | 2.52 | 0.44 | 0.27 ns |
| 3D adult atlas, 338,993 tets | 7.00 M | 11.3 (R1) | 1.51 | 0.64 ns |
| 4D jittered Kuhn, 240,000 pentatopes | 6.25 M | — | 6.67 | 3.1 ns |

The trend is geometric: a `d`-simplex fills at most `h_d/d!` of its box, and the same number of
samples is spread over more and shorter rows as `d` grows. **[inferred]**

#### 3.3 2D: is K3 still right?
GPU, hand-written int32 kernels, `k = 8`, row kernel only, ms, best of 40 (setup about 0.25 ms and
scan 0.1 ms extra, the same for all). **[measured]**

| mesh | K3 rows along axis 0 | K3 rows along axis 1 | K1 (thread per box candidate) | K2 (thread per triangle) |
|---|---|---|---|---|
| 341,650 uniform triangles, 4096² (median area 37 px) | 3.88 | 3.97 | 5.21 | 4.46 |
| same triangles, 2048² (median area 9 px) | 0.98 | 1.00 | 1.66 | 1.02 |
| 341,638 clustered triangles, 4096² (areas 0.0003 to 3,570 px) | 3.13 | 3.15 | 4.88 | 3.39 |
| 128 large triangles (512 px legs), 4096² | 2.24 | 2.42 | **1.65** | 20.1 |

- K3 is the best on the three meshes of small triangles and 36 % behind K1 on the large ones. K1
  wins only for large triangles (box fill near 50 %, no loop, coalesced writes) and loses 35–70 %
  on small ones. K2 is 4–15 % behind K3 on small triangles and 9× slower on large ones.
  **[measured]**
- The margins are much smaller than in 3D, where R1 measured K1 2.3× and K2 9× slower than K3,
  because the 2D box fill is 40 % instead of 9 %. **[measured]** / **[inferred]**
- The 2D prior art agrees on the structure. FreePipe gives one thread a whole triangle; Laine and
  Karras report that its performance "is excellent when there are many small, fairly homogeneously
  sized triangles" but that a "highly variable number of pixels in each triangle leads to poor
  thread utilization". **[verified]** [5] That is K2 and its measured failure above.
- cudaraster's bins (128² px) and tiles (8² px) exist so that a framebuffer tile fits in shared
  memory and triangles are blended in input order; its fine rasteriser builds 8×8 coverage masks
  from look-up tables ("51 assembly instructions per edge"); it has "a relatively high constant
  per-triangle cost", and the authors suggest bypassing the two binning stages for small triangles
  "where the ordering requirements are not necessary". **[verified]** [5] We have no ordering and
  one writer per pixel, so, as in R1 §3.3(b), the hierarchy buys nothing. **[inferred]**
- A 2026 CUDA rasteriser for very large scenes (CuRast) also launches one thread per small
  triangle (box under 128 pixels) looping over the bounding box, one warp per medium triangle, and
  tiles only for large ones; it resolves visibility with a 64-bit `atomicMin` on depth and id, and
  states that Nanite's software rasteriser is about three times faster than the hardware pipeline
  on small triangles. **[verified]** [6] (I could not retrieve the Nanite slides themselves.) The
  balancing in these systems is by triangle size class; K3's prefix sum over rows does the same job
  without classes. **[inferred]**
- For scale only (different hardware and task, not a comparison): cudaraster reports 130 Mtris/s
  for 10-pixel triangles on a GTX 480 for its full pipeline **[verified]** [5]; the 2D K3 prototype
  here does 341,650 triangles of 12 pixels in 1.3–1.4 ms including setup and scan (about
  250 Mtris/s) on the dev laptop, writing the owner and two barycentrics, no depth.
  **[measured]**
- A limit of K3 that 2D exposes: its unit of work is the row, and a row is written by one thread.
  With few huge triangles a row can be thousands of pixels. On the 128-triangle mesh K3 was still
  fine (2.2 ms), but two triangles covering a 4096² image would give 4,096 threads 4,096 pixels
  each. Splitting rows into segments of bounded length would fix it in any dimension. Not
  measured. **[inferred]**

#### 3.4 Row direction
The ROADMAP asks whether rows should run along axis 0 (now) or the last axis (contiguous in C
order). The rule does not change (it lives in `c_j`), only the axis along which the row is solved;
results were bit-identical. **[measured]**

| case | rows along axis 0 | rows along the last axis |
|---|---|---|
| 3D adult atlas, GPU row kernel | 4.46–4.66 ms | 4.35 ms |
| 3D adult atlas, CPU `rasterize` | 39.3–41.1 ms | 40.8 ms |
| 2D small triangles 4096², GPU | 3.88 ms | 3.97 ms |
| 2D small triangles 4096², CPU | 24.9 ms | 21.5 ms |
| 2D clustered 4096², CPU | 24.9 ms | 19.1 ms |
| 2D 128 large triangles 4096², CPU | 76.5 ms | 27.0 ms |

It matters only on the CPU and only when spans are long (2.8× for 512-pixel spans); for the adult
atlas it makes no difference on either backend. (The CPU row of the atlas was taken late in the
session, when all timings had drifted up by about 15 %.)

#### 3.5 4D
K3 generalises mechanically (one thread per `(simplex, i_1, i_2, i_3)`), and the prototype ran it
correctly. But it needs 6.7 row set-ups per owned cell on the test mesh, each with five exact
divisions, and the row kernel took 19 ms for 6.25 M cells with int64 (`k = 5`) and 83 ms with
`__int128` (`k = 14`). A 4D traversal would want to solve a 2D slab per thread or walk between
neighbours instead. So the best traversal does depend on `d`, but the dependence is weak between 2
and 3 and strong only at 4. **[measured]** (numbers) / **[inferred]** (what 4D would want)

### 4. Implementation techniques and their cost

#### 4.1 How the prototype is written
One header (`gen/raster_nd.hpp`, about 330 lines) with
`template <int D, Real T, class Ints = DefaultInts<D>>` on `Record`, `setup_simplex`,
`plane_values`, `owns`, `barycentric`, `row_span`, `row_span_along<Axis>`, and the SoA view with
`load_record`/`store_record`. Techniques used:
- fixed-size C arrays `[D + 1][D]`, plain `for` loops with compile-time bounds;
- the face table replaced by the sign formula of §1.1, so nothing is generated or tabulated;
- determinants by recursive cofactor expansion with `if constexpr` (`det<N>`), used for the
  `(D−1)×(D−1)` minors. For `D = 3` this compiles to the same two multiplications per component as
  the hand-written cross product;
- a traits struct (`Ints32`, `Ints64`, `Ints128`) giving `test_t`, `wide_t`, `index_t`, and a
  `constexpr window_bits<D, Ints>(k)` that derives `L` from both the per-sample and the setup
  width;
- CPU loops over the box by a recursive function template that passes the indices as a parameter
  pack; the GPU row kernel unravels the row index with a loop of `D − 2` divisions.

No explicit specialisation was needed. On determinant methods **[inferred]**: for `n ≤ 4`
cofactor expansion is division-free and has the fewest operations; Bareiss elimination needs exact
divisions and pays off only for larger `n`; closed forms are what the expansion compiles to. Setup
is 0.5 ms of a 5.4 ms rasterisation, so sharing sub-minors (computing `adj(A)` once and
`n_0 = −Σ n_j`: 18 instead of 24 multiplications in 3D) is not worth obscuring the code.

#### 4.2 Prior art
- **deal.II** writes everything as `template <int dim>`. The tutorial calls this "dimension
  independent programming"; the compiler "has to compile each function for each dimension
  separately", "knows the size of the loop at compile time", and the result "will be as fast as if
  we had used the formulas above right away". **[verified]** [9] This is the model the prototype
  follows, and the measurements below confirm the claim for our code.
- **DUNE** makes the size a compile-time parameter of its small vectors
  (`template<class K, int SIZE> class FieldVector`, `static constexpr int dimension = SIZE`).
  **[verified]** [10]
- **CGAL** has a separate dD kernel. `Epick_d<DimensionTag>` takes `Dimension_tag<d>` when the
  dimension is "known at compile-time", or `Dynamic_dimension_tag`; its predicates are exact
  "thanks to the use of filters"; it requires Eigen. **[verified]** [8] CGAL also has separate 2D and 3D
  kernels (general knowledge, not checked here); the dD manual pages I read do not relate the two.
  The generic kernel is for float inputs with filtered exact predicates, a problem we do not have
  after snapping. **[inferred]**
- **Shewchuk's predicates** are one C file with "the 2D and 3D orientation and incircle tests",
  adaptive so that they "do only as much work as necessary". **[verified]** [11] They are written
  per predicate because each adaptive stage has an error bound derived for that specific
  expression. **[inferred]** (R1 [18] is the paper.)
- **Geogram** generates them instead: "The C++ code of the predicates is automatically generated
  from its formula, written in a simple specification language", with "arithmetic filters,
  expansion arithmetics and symbolic perturbation"; the generated files are per predicate and
  dimension (`orient2d.h`, `orient3d.h`, `orient4d.h`, `det3d.h`, `det4d.h`, `side1.h`…`side4.h`).
  **[verified]** [12]
  - Reading of the last two: predicate libraries are per-dimension because adaptive
    floating-point exactness is expression-specific. Our exactness comes from fixed-width integers
    after snapping, where a generic loop is as exact as a hand-written formula, so that reason
    does not apply. **[inferred]**
- **Eigen in CUDA**: fixed-size matrices, vectors and arrays work in kernels; the index type must
  be redefined to `int` (`EIGEN_DEFAULT_DENSE_INDEX_TYPE`) and host SIMD disabled in `.cu` files.
  **[verified]** [13] Nothing here needs it: the largest object is a 4×3 integer array, and Eigen
  would add a dependency to a header that users include in their own kernels (ADR 0004/0009).
  **[inferred]**
- **Dimension-generic GPU rasterisers or voxelisers for simplicial meshes**: I found none.
  "Higher-dimensional rasterization" in graphics means stochastic sampling of moving, defocused
  triangles over (x, y, lens, time). **[verified]** [15] (abstract and slides as summarised by a
  search; paper not read). 4D pentatope meshes are visualised by GPU tessellation of their
  tetrahedral facets. **[verified]** [17] (abstract). A search for lattice-point enumeration in
  4-simplices returned nothing usable.

#### 4.3 Pitfalls met while writing it **[measured]**
1. **`std::array` and `int D`.** A parameter `const std::array<int64_t, D>&` with
   `template <int D>` does not deduce (g++ 13.3: "mismatched types 'int' and 'long unsigned
   int'"), because the size of `std::array` is a `std::size_t`. Use `template <std::size_t D>`, a
   non-deduced context (`std::type_identity_t`), or an own struct.
2. **`constexpr` helpers in device code.** A `constexpr` function without `__host__ __device__`
   called from a host-device function gives nvcc warning #20013 (it wants
   `--expt-relaxed-constexpr`). Marking the helpers `TETVOX_HD` removes it. This is the same family
   as the `std::array` restriction already in `rasteriser-cpu.md`.
3. **Unrolling and local memory.** With int64, nvcc 13.5 unrolled every constant-trip loop for
   `D` = 2, 3 and 4 without any pragma: frame size 0 and no `__local_depot` in the PTX. Adding
   `#pragma unroll` to every loop changed nothing (generic row kernel 4.60 ms with it, 4.37–4.72
   without). The CUDA guide lists
   what goes to local memory: "Arrays for which it cannot determine that they are indexed with
   constant quantities", large structures, and register spills. **[verified]** [14] Compile-time
   bounds avoid the first.
4. **`__int128` per sample** (the 4D instantiation) does produce local memory: frames of 496 B
   (row kernel) and 688 B (setup), 160 and 148 registers. The same generic source with int64 has
   frame 0. So this cost comes from the wide type, not from genericity.
5. **GCC's inliner.** A helper function template that takes a lambda and is not declared `inline`
   was not inlined into the OpenMP region: `count_claims` took 41 ms instead of 23. Declaring it
   `inline` fixed it.
6. **Index arrays shared by reference.** A recursive loop template that keeps the indices in one
   array passed by reference forces them through memory: `rasterize` +8 %, `locate` +10 %. Passing
   them as a parameter pack brought `rasterize` to +2–5 %; `locate` stayed at +5–9 % until
   `row_span` was inlined (§5.2).
7. **Narrowing in `row_span`.** With `test_t ≠ index_t` (int32 or `__int128`) the span must be
   computed relative to `box_lo` and cast at the end; the absolute form of the current header
   overflows int32 and truncates `__int128`.
8. **Micro-choices outweigh genericity.** `owns` written as a loop of `&&` gives 20 more host
   instructions out of line than the hand-written chain, but lets the compiler stop at the first
   failing face inside the box scan; an OR of the sign bits gives the same 47 instructions as the
   hand code but made `count_claims` 15–20 % slower (26.8 against 23.4–23.9 ms).
9. **Compile time and size** are not an issue (table in §5.6).

### 5. Measurements

#### 5.1 Method
- Machine: the dev laptop of `build.md` (Intel Core Ultra 5 235H, 14 OpenMP threads; RTX PRO 1000
  Blackwell, `sm_120`). g++ 13.3 `-std=gnu++20 -O3 -DNDEBUG -fopenmp`; nvcc 13.5
  `-std=c++20 -O3 -arch=native`.
- "Hand" for 3D is the repository itself: `csrc/src/cpu.cpp` compiled unchanged and linked into
  the benchmark, and `tetvox/raster.hpp` called from a minimal K3 kernel pair. "Generic" is the
  scratch header of §4.1 with `D = 3`, `Ints64`.
- Data: adult atlas, float32 positions, grid 181×217×181, `k = 14`. 2D: Delaunay triangulations
  (scipy from another conda env, since `tetvox-dev` has none) of 170,000 random points plus border
  points tiling `[0, size − 1]²`, uniform or clustered; 341,650 triangles. 4D: Kuhn triangulation
  of a 10⁴ grid of cells of side 5 with interior vertices jittered by ±15 %; 240,000 pentatopes.
- CPU timing: every variant writes the same output buffers; the order of the variants rotates each
  round; 30 rounds; minimum and median. A second copy of `cpu.cpp` (namespace renamed by a macro)
  runs as a control: it differs from the original by at most 1 % on the minimum.
- GPU timing: wall clock around launch + `cudaDeviceSynchronize`, `cudaGetLastError` checked,
  40–60 rounds, variants alternating. The minimal K3 is: setup kernel (snap, classify, record, row
  count) → exclusive scan (three small kernels; `thrust::exclusive_scan` takes 0.064 ms for the
  same input) → fill with −1/0 → row kernel (binary search of the row offset, `load_record`,
  `row_span`, then `plane_values` + `barycentric` and three stores per voxel).
- Generated code: `objdump -d` per out-of-line wrapper for the host; `cuobjdump -sass` and
  `cuobjdump -elf` (`.nv.info`: `EIATTR_REGCOUNT`, `EIATTR_FRAME_SIZE`) for the device.
- Everything is in
  `/tmp/claude-3234968/-u-68-defloa2-unix-git-gpu-rasteriser/5b1132b0-d879-4ce4-b26d-07a7d4855cff/scratchpad/r3`
  (`gen/raster_nd.hpp`, `gen/ops_nd.hpp`, `hand2d/raster2d.hpp`, `gpu/k3_nd.cuh`,
  `gpu/k3_{3d,2d,4d}.cu`, `bench_cpu{2,3}.cpp`, `asm_procs.cpp`, `sass_stats.py`, `count_insn.py`,
  `theory_check.py`, `make_mesh2d.py`, `make_kuhn.py`). The directory is temporary.
- Limits: one laptop, thermally noisy (single runs drift by 10–15 % over a session; comparisons are
  always within one binary and one run); un-tuned kernels; one mesh per case.

#### 5.2 D = 3: identity and CPU **[measured]**
- **Bit identity.** Generic and repository code give identical records (all seven fields, all
  338,993 tets), identical `tet_id`, π and `claims`, on the host; the generic device kernels equal
  the hand device kernels, and the device output equals the host output (`cmp` on the dumps).
  6,998,400 voxels owned, each claimed once.
- **Host instructions** (x86-64, per procedure, hand / generic): `setup` 521 / 534 (45
  multiplications both); `plane_values` 60 / 60; `plane_values` + `owns` 47 / 47 (OR form) or 67
  (`&&` loop); `barycentric` 19 / 19; `row_span` 196 / 193 (8 multiplications and 8 divisions
  both).
- **CPU time**, ms, minimum of 30 rounds (medians are 1–3 ms higher and tell the same story):

| variant | `setup` | `rasterize` | `locate` | `count_claims` |
|---|---|---|---|---|
| repository `cpu.cpp` | 4.9–5.5 | 33.3–34.5 | 25.0–25.3 | 22.3–22.8 |
| control (identical copy) | 4.7–5.3 | 34.1–34.6 | 24.9–25.3 | 22.5–23.0 |
| generic, naive (index array by reference, helper not `inline`) | 4.8 | 37.1 | 27.4 | 40.8 |
| generic, indices as a pack, helpers `inline` | 4.9 | 35.8 | 27.3 | 23.2 |
| … and `row_span` in the header's absolute form | 4.9 | 35.6 | 26.4 | 23.4 |
| … and `row_span` forced inline | 5.1 | **32.6** | **24.2** | 23.2 |

- Reading: written with care the generic code is 3–5 % slower than `cpu.cpp`, written naively
  8–10 % (and 80 % in one op), and with `row_span` inlined 4–5 % *faster*. The differences are
  inlining and calling-convention effects, not a cost of genericity. In both versions GCC leaves
  `row_span` and `load_record` out of line, and `cpu.cpp` also calls `store_barycentrics` once per
  voxel (seen in the disassembly). Inlining `row_span` is a free 5 % for the CPU backend,
  independent of this report's question.

#### 5.3 D = 3: GPU **[measured]**
Minimal K3 on the adult atlas, 10,572,440 rows, ms over 60 rounds (ranges over several program
runs):

| kernel | hand (repo header), min | generic, min | hand, median | generic, median |
|---|---|---|---|---|
| setup (snap, record, row count) | 0.49–0.51 | 0.47–0.49 | 0.55–0.59 | 0.53–0.58 |
| scan (shared) | 0.10 | | | |
| fill −1 / 0 (shared) | 0.30 | | | |
| row kernel | 4.46–4.65 | 4.37–4.72 | 4.88–4.94 | 4.91–5.01 |
| row kernel, `t_j` stepped along the row | | 4.51–4.54 | | 4.95–4.98 |
| row kernel, rows along axis 2 | | 4.35 | | 4.98 |

Device code (SASS, `sm_120`):

| kernel | instructions | registers | frame (local memory) |
|---|---|---|---|
| setup, hand | 1,104 | 92 | 0 |
| setup, generic | 1,104 | 92 | 0 |
| rows, hand | 904 | 70 | 0 |
| rows, generic | 928 | 72 | 0 |

- Generic and hand-written are indistinguishable: within 1–2 % in either direction, well inside
  the run-to-run spread.
- The whole rasterisation is about 5.4 ms (0.5 + 0.1 + 0.3 + 4.5). R1's prototype measured
  3.9 + 0.3 ms with a smaller record (clipped int32 boxes), four stored π and its own layout; the
  present number is for ADR 0009's record as built.
- Stepping `t_j` (4 additions instead of 12 multiplications per voxel) gains nothing, and neither
  does the row direction. The kernel is bound by its memory traffic: per row a binary search and
  about 200 bytes of record gathered from seven arrays (27 global-load instructions in the SASS),
  per voxel three scattered stores. A pure fill of the same outputs takes 0.3 ms.

#### 5.4 D = 2 **[measured]**
All variants at the same `k` produce identical owners and π, and the hand-written int32 record
equals the generic `Ints32` record bit for bit; device equals host. `k = 8` so that int32 is legal
(`L = 10`).

GPU, ms, minimum of 40 (setup / scan / row kernel):

| variant | uniform, 4096² | uniform, 2048² | clustered, 4096² |
|---|---|---|---|
| generic, int64 / int128, `k = 14` | 0.25 / 0.10 / 3.94 | 0.23 / 0.10 / 1.11 | 0.24 / 0.10 / 3.07 |
| generic, int64 / int128, `k = 8` | 0.24 / 0.10 / 3.97 | 0.23 / 0.10 / 1.11 | 0.23 / 0.11 / 3.19 |
| generic, int32 / int64 (traits), `k = 8` | 0.21 / 0.10 / 3.89 | 0.20 / 0.10 / 0.98 | 0.21 / 0.10 / 3.11 |
| hand-written 2D, int32, stepped `t_j` | 0.26 / 0.09 / 3.88 | 0.24 / 0.10 / 0.98 | 0.25 / 0.09 / 3.13 |

Device code: row kernel 448 instructions and 40 registers (hand), 480 and 40 (generic int32), 656
and 56 (generic int64); frame 0 in all.

CPU `rasterize`, ms, minimum of 15 (setup in brackets):

| variant | uniform, 4096² | uniform, 2048² | clustered, 4096² |
|---|---|---|---|
| generic, int64 / int128, `k = 8` | 26.0 (4.7) | 8.6 (5.0) | 25.7 (5.2) |
| generic, int32 / int64 (traits) | 24.4 (3.5) | 7.9 (3.9) | 24.8 (3.6) |
| hand-written 2D, int32, stepped | 24.9 (4.2) | 8.2 (4.0) | 24.9 (3.8) |

- A hand-written 2D path has no advantage over the generic code instantiated with the same
  integer widths (GPU ±1 %, CPU ±2 %).
- The widths themselves are worth 0–12 % on the GPU (nothing at 4096², where the per-pixel stores
  dominate; 12 % at 2048²) and about 6 % on the CPU, plus 25 % of a 5 ms setup.
- int32 is also a restriction: at the default `k = 14` it allows triangles of at most 128 pixels.

#### 5.5 D = 4 **[measured]**
The generic source compiled for `D = 4` without change, for both `Ints128` and `Ints64`.

| | `__int128` per sample, `k = 14` (`L = 15`) | int64 per sample, `k = 5` (`L = 10`) |
|---|---|---|
| exactly-once on `[0, 50)⁴`, nothing outside | yes | yes |
| device = host (record, owners, π) | yes | yes |
| CPU setup / `rasterize` | 12.4 / 231 ms | 9.1 / 123 ms |
| GPU setup / row kernel | 1.07 / 83.0 ms | 0.60 / 19.1 ms |
| row kernel: instructions, registers, frame | 1,528, 160, 496 B | 1,024, 96, 0 |
| setup kernel: instructions, registers, frame | 2,072, 148, 688 B | 1,968, 168, 0 |

It still unrolls (frame 0 with int64); the loss comes from the integer width and from the 41.7 M
rows.

#### 5.6 Compile time and size **[measured]**

| translation unit | compile time | size |
|---|---|---|
| repository `cpu.cpp` (3D, float + double) | 0.66 s | 24.6 kB text |
| generic CPU ops, `D` = 3, float + double | 0.56 s | 18.2 kB |
| generic CPU ops, `D` = 2, 3 | 1.22 s | 30.7 kB |
| generic CPU ops, `D` = 2, 3, 4 | 2.30 s | 69.6 kB |
| generic K3 kernels (nvcc), `D` = 3 / 2, 3 / 2, 3, 4 | 5–7 s each | 139 / 219 / 394 kB object |

### 6. Scope beyond the rasteriser
What a generic design changes, per item. "Trivial" means the formula holds with `3 → D`,
`4 → D + 1`. **[inferred]** unless tagged.

| Item | In a `D`-generic design | Kind of change |
|---|---|---|
| `GridShape` | `std::array<int64_t, D>` (pitfall 1 of §4.3) | type parameter |
| `MeshView<T>` | positions `(N, D)`, cells `(T, D + 1)`; the field names `tetrahedra`, `num_tets` | type parameter + naming |
| `TetSetupView<T>` | normals `(T, D+1, D)`, offsets and fractions `(T, D+1)`, boxes `(T, D)`; dtype of normals/offsets fixed to int64 if recommendation 3 holds | type parameter + naming |
| `TetStatus`, `SkippedTets` | unchanged content | naming only |
| `snap`, `in_snap_range` | unchanged | none |
| 1 `setup_tets` | trivial; `D` from `positions.shape[1]`; budget constant per `D` | generic |
| 2 `count_skipped` | unchanged | none |
| 3–6 `locate`, `barycentrics`, `rasterize`, `count_claims` | trivial; outputs `(n_0, …, n_{D−1})` and `(…, D)`; row enumeration generic (**[measured]** in the prototype) | generic |
| 7 `interpolate` | `Σ_j π_j F[v_j]` over `D + 1` vertices | generic |
| 8 `interpolate_vjp_values` | scatter of `π_j g_f` | generic |
| 9 `interpolate_vjp_bary` | `ḡ_m = (F_m − F_0)·g_f`, `m = 1..D`; `grad_bary (…, D)` | generic |
| 10 `tet_moments` | `(T, D+1, D)` | generic |
| 11 `vertex_grad` | `G_i = −(2ᵏ/D) Σ_m m_{i,m} n_m` | generic |
| 12 `accumulate<Cost>` | the `Cost` concept sees `D + 1` barycentrics; samseg's `−log Σ_j l_j π_j` has the same form in 2D (the 2009 paper presents it for triangles **[verified]** [4]) | generic, concept parametrised |
| deformation prior | the Ashburner penalty has explicit dimension-dependent constants (`− 6`, `/6` in `samseg-math.md`) | per-`D` constants; could stay 3D-only |
| `SamsegAtlas`, `load_atlas_npz` | samseg atlases are 3D (affine 4×4, `can_move (N, 3)`) | stays 3D |
| `TetMesh` (Python) | `(N, D)`, `(T, D + 1)`; name and field `tetrahedra` | naming |
| bindings | nanobind overloads are {cpu, cuda} × {float, double} today; a shape constraint per `D` doubles them, or one overload reads `D` from the array shape and switches | owner decision |
| `.pyi` stub | grows with the overload count; `shape: tuple[int, int, int]` becomes `tuple[int, ...]` (arity no longer checked statically) or stays precise through per-`D` overloads | owner decision |
| tests | the Kuhn fixture and `tests/exact_reference.py` generalise (my `theory_check.py` has a generic determinant and Kuhn grid); 2D exhaustive exactly-once on 4096² takes tens of milliseconds; the real-atlas tests stay 3D | parametrise over `D` |
| docs | ADRs 0005–0009 say voxel, tet, `(X, Y, Z)`; one new ADR stating how each statement reads for `D = 2` is enough | one ADR |
| compile time, binary | §5.6: negligible | none |

### 7. Use cases
- **2D, real but secondary.**
  - The paper that introduced these atlases develops the method on triangles and evaluates it on
    2-D data first: "The techniques proposed in this paper apply equally well in the 2-D domain,
    using triangular atlas mesh representations, as in the 3-D domain, using tetrahedral meshes";
    Sections IV and V are experiments "on manually labeled datasets in 2-D". **[verified]** [4]
    `equations.pdf` likewise derives everything for a triangle and then says "In 3D, we're not
    dealing with triangles but with tetrahedra". **[verified]** [3] So 2D is the natural setting
    for teaching, figures and quick model experiments on slices.
  - GEMS itself is 3D only: its mesh type is `itk::Mesh<PointParameters, 3, AtlasMeshTraits>` with
    a `ReferenceTetrahedronInfo` per cell. **[verified]** [18] I found no 2D samseg.
  - For the project itself: exhaustive tests that run in milliseconds, images one can look at when
    debugging ownership, and a second instantiation that exercises the generic code. **[inferred]**
  - Outside samseg: piecewise-affine warps of triangulated landmarks (active appearance models)
    and mesh-based 2D registration use exactly "triangle id + barycentrics per pixel"; a search
    found the classical literature but no differentiable GPU tool dedicated to it, and found
    nvdiffrast used for per-pixel barycentrics instead. **[verified]** (search results only; none
    of those papers was read.) A watertight, tiling-aware 2D op with 14 sub-pixel bits would differ
    from a graphics rasteriser (4 bits in cudaraster **[verified]** [5]) mainly in precision and in
    having no depth; whether anyone needs that is unproven. **[inferred]**
  - Histology: the search returned 3D pipelines (samseg on ex vivo MRI, 3D histology
    reconstruction), no 2D triangle-mesh atlas. **[verified]** (search results only)
- **4D, none found.** Space-time finite elements do use pentatope meshes ("four-dimensional
  simplex elements (pentatopes)") **[verified]** [16] (abstract), but I found no demand for
  resampling them on a 4D grid, and §2 and §5.5 show what it would cost.

## Options compared
| Option | Pros | Cons | Fit for tetvox |
|---|---|---|---|
| (a) Stay 3D-only | No work; names and ADRs stay as accepted; nothing to explain | No 2D tests, figures or slice experiments; if 2D is wanted later, two backends and a released API must be converted | Right if the owner sees no use for 2D |
| (b) One fully generic `template <int D>`, same types for every `D` | Simplest generic form; measured as fast as hand code for `D = 3` | Wrong for `D = 4` (int64 forces `k ≤ 5`); the budget constant differs per `D` anyway, so some per-`D` data is unavoidable | Almost: needs the traits of (c) |
| **(c) Generic code path + per-`D` traits (budget, optionally integer widths), `D` ∈ {2, 3}** | Bit-identical and equally fast in 3D; 2D equals a hand-written path with the same widths; kernels written once; 2D gives cheap exhaustive tests | Renaming and shape generalisation in the public API; a second test matrix; one more ADR; slightly more abstract header (minors instead of a cross product) | **Recommended if 2D is wanted** |
| (c′) As (c) plus hand specialisations of hot procedures | Could squeeze a 2D int32 path (≤ 12 % on small images) | No measured need; two code paths to keep bit-identical; a second record dtype | Not justified by the measurements |
| (d) Separate 2D and 3D implementations | Each reads like its dimension | Duplicates every procedure, kernel and test for no speed gain (§5.4); the two can drift | Reject |
| `D = 4` in any option | — | `__int128` per sample or `k ≤ 5`; local memory; no numpy dtype; a different traversal; no user | Reject |

## Open questions
1. **Does the owner want 2D as a product feature**, as a C++-only/test-only instantiation, or not
   at all? Everything else follows from this.
2. **Other GPUs.** All timings are from one laptop GPU. int64 division and int64 multiply may weigh
   more elsewhere (already a ROADMAP experiment); an int32 2D path would matter only there.
   Settle by running `k3_2d`-like kernels on the target GPU once the CUDA backend exists.
3. **Long rows.** K3 balances by row; 2D images with a few huge triangles would give very long
   rows per thread (§3.3). Measure a two-triangle 4096² case when a 2D kernel exists; if it is
   slow, split rows into bounded segments (dimension-independent).
4. **The tight budget.** §2.3 suggests `2k + 3L ≤ 61` (or 60 with today's summation order in
   `row_span`) instead of 59 for 3D. Before relying on it: write the proof out in full, fix the
   summation order, and extend the adversarial test of ADR 0008 to the new boundary. Owner
   decision; it would allow `L = 11` at `k = 14` or `k = 15` at `L = 10`.
5. **Deformation prior in 2D.** Not analysed; the constants depend on `d`.
6. **Not verified.** The Nanite slides (over the fetch size limit; cited through [6]); the SoS
   paper beyond its abstract; why CGAL keeps separate 2D/3D kernels; the 2D registration and AAM
   literature (search results only). No tile-based 2D rasteriser was prototyped; that it would not
   help is inferred from R1 §3.3 and [5].
7. **CPU row direction.** Rows along the last axis are 2.8× faster on the CPU for long 2D spans and
   neutral on the adult atlas (§3.4). If 2D is adopted, the row axis should probably be the last
   one on the CPU; this is the existing ROADMAP question, now with data.

## Impact on design
If option (c) is adopted. Nothing below is decided; each item is a choice for the owner.

**Structure**
- `raster.hpp`: `template <int D, Real T>` on the record and procedures (`D` first or last is a
  choice); the face table replaced by the sign formula of §1.1; `row_span` generic in the row
  axis; a `Budget<D>`/traits struct with the constant of §2.2. The current functions can remain as
  the `D = 3` instantiation, so `tetvox::setup_tet<float>` keeps compiling.
- `cpu.cpp` and the future `cuda` kernels: one template per op, explicitly instantiated for
  `D` ∈ {2, 3} × {float, double}. K3: rows = product of the clipped extents of the non-row axes;
  unravel with `D − 2` divisions.
- Tests: Kuhn fixture and exact reference parametrised over `D`; exhaustive 2D exactly-once
  (lattice-aligned, jittered, partly outside the grid); the bit-budget boundary test per `D`; the
  existing 3D tests unchanged and required to stay bit-identical through the refactor.

**Decisions the owner would have to take** (options; a lean is given only where this report has
evidence)
1. *Which `D` are instantiated.* {3}; {2, 3}; {2, 3, 4}. Evidence favours {2, 3}.
2. *Visibility of 2D.* Public in C++ and Python; public in C++ only at first; internal and used
   only by tests. Affects how much naming must be settled now.
3. *Names.* `TetMesh`, `tetrahedra`, `TetSetup`, `setup_tets`, `tet_id`, `TetStatus`,
   `SkippedTets`, `num_tets`, `TetRecord`: (i) keep the `tet` names for every `D` (a triangle is
   then a "tet"); (ii) neutral names (`simplices` or `cells`, `SimplexSetup`, `setup_simplices`,
   `simplex_id`) with the `tet` names kept as 3D aliases; (iii) parallel names per dimension
   (`setup_triangles`, `tri_id`). Affects the whole public API of ADR 0009 and the package's
   vocabulary; costly to change after a release.
4. *C++ template form.* `template <int D, Real T>` or `std::size_t D`; order of the parameters;
   whether `GridShape` becomes `GridShape<D>`; which instantiations the library exports.
5. *Integer types per `D`.* Same for 2 and 3 (int64 / `__int128`), or an int32 / int64 2D path.
   Evidence favours the same types (≤ 12 % and a restricted `k` otherwise). Affects the dtype of
   the exposed record.
6. *Budget constant per `D`.* R1 style (60 for 2D, 59 for 3D) or the tight values of §2.3 (62 or
   61, 61 or 60). Affects `window_bits`, the `outside_window` classification and the adversarial
   test.
7. *Default `snap_bits` in 2D.* 14 as in 3D (window 2²³ pixels), or another value. `max_snap_bits`
   stays 24 by the `f_j` argument.
8. *Tie-break and orientation conventions in 2D.* Axis 0 first and `det > 0` in axis order, as in
   3D (consistent, but "top, then left" and clockwise for `(row, col)` images, §1.4), or a
   graphics-compatible convention for 2D. Affects whether 2D output can be compared bit for bit
   with D3D-style rasterisers at ties.
9. *Row axis.* Axis 0 (now) or the last axis, per backend (§3.4). No effect on results.
10. *Shapes.* `tet_id (n_0, …, n_{D−1})`, `bary (…, D)` holding `π_1..π_D`, record
    `(T, D+1, D)`, moments `(T, D+1, D)`; confirm that the 3D layouts are the `D = 3` case of
    these.
11. *Python dispatch.* One `rasterize` (and one of each op) dispatching on array shape, or
    per-dimension function names; `shape: tuple[int, ...]` or precise overloads; how the stub
    documents it.
12. *What stays 3D.* `SamsegAtlas`, the loader, and probably the deformation prior; whether
    `accumulate<Cost>` and the samseg cost are instantiated for 2D.
13. *4D.* State explicitly that it is out of scope, so that nobody generalises the integer types
    for it.
14. *ADR.* One new ADR that amends 0005–0009 for `D`, or edits to each.
15. *Sequencing.* Generic internals before the CUDA backend (recommended if 2D is wanted), or the
    3D CUDA backend first and the conversion afterwards.

**Independent of the decision** (by-products of the measurements)
- Force or encourage inlining of `row_span` in the CPU loops (about 5 % in the prototype).
- In the CUDA K3 kernel, spend the effort on the record gather and the write pattern; the
  arithmetic form does not show in the timings (§5.3).
- `thrust::exclusive_scan` is available with the toolkit and takes 0.064 ms for the row counts of
  the adult atlas.

## References
Sources [3], [4] and [5] were read as full text (local PDFs, `pdftotext`). The other web sources
were read through the fetch tool, which returns extracted passages; the quotes are as it returned
them and were not re-checked against the original pages. Search-only items are marked in the text.

[1] R1 — *Watertight solid rasterisation of tetrahedral meshes on the GPU*, this repository,
`docs/research/R1-watertight-tet-rasterisation.md` (its references [1]–[4] for Pineda, the Direct3D
rules and Giesen; [13] for the cudaraster constants in nvdiffrast). ADRs 0006–0009 and
`docs/knowledge/rasteriser-cpu.md` for the 3D rule, budget and header.
[2] H. Edelsbrunner, E. P. Mücke, *Simulation of Simplicity: A Technique to Cope with Degenerate
Cases in Geometric Algorithms*, ACM TOG 9(1), 1990. https://arxiv.org/abs/math/9410209 (abstract
read)
[3] K. Van Leemput, *Working with tetrahedral meshes in samseg*, 25 Jul 2025. Local copy
`literature/equations.pdf`.
[4] K. Van Leemput, *Encoding Probabilistic Brain Atlases Using Bayesian Inference*, IEEE TMI
28(6):822–837, 2009. Local copy `literature/VanLeemputTMI2009.pdf`;
https://pmc.ncbi.nlm.nih.gov/articles/PMC3274721/
[5] S. Laine, T. Karras, *High-Performance Software Rasterization on GPUs*, HPG 2011.
https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
[6] *CuRast: Cuda-Based Software Rasterization for Billions of Triangles*, arXiv:2604.21749, 2026.
https://arxiv.org/abs/2604.21749 (also the source for the statement about Nanite: B. Karis,
R. Stubbe, G. Wihlidal, *A Deep Dive into Nanite Virtualized Geometry*, SIGGRAPH 2021 course,
https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf, not
retrieved)
[7] OEIS A003432, *Hadamard maximal determinant problem: largest determinant of a (real)
{0,1}-matrix of order n*. https://oeis.org/A003432
[8] CGAL, *dD Geometry Kernel*, user manual and `Epick_d` reference.
https://doc.cgal.org/latest/Kernel_d/index.html,
https://doc.cgal.org/latest/Kernel_d/classCGAL_1_1Epick__d.html
[9] deal.II, *The step-4 tutorial program* (dimension independent programming).
https://www.dealii.org/current/doxygen/deal.II/step_4.html
[10] DUNE, `Dune::FieldVector< K, SIZE >` class reference.
https://dune-project.org/doxygen/master/classDune_1_1FieldVector.html
[11] J. R. Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Predicates for
Computational Geometry* (web page of the predicates). https://www.cs.cmu.edu/~quake/robust.html
[12] B. Lévy, *Robustness and Efficiency of Geometric Programs: The Predicate Construction Kit
(PCK)*, 34th Gocad Meeting, 2014 (abstract),
https://www.ring-team.org/research-publications/ring-meeting-papers?view=pub&id=2396; Geogram file
list, https://brunolevy.github.io/geogram/files.html
[13] Eigen, *Using Eigen in CUDA kernels*. https://libeigen.gitlab.io/eigen/docs-nightly/TopicCUDA.html
[14] NVIDIA, *CUDA Programming Guide*, "Writing SIMT Kernels", section Local Memory.
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html
[15] J. Nilsson et al., *Design and Novel Uses of Higher-Dimensional Rasterization*, HPG 2012.
https://diglib.eg.org/items/86923c5f-05ac-4fea-8dbf-19d272f39a1a (abstract and slides, through a
search summary)
[16] M. von Danwitz et al., *Four-Dimensional Elastically Deformed Simplex Space-Time Meshes for
Domains with Time Variant Topology*, arXiv:2012.14200. https://arxiv.org/abs/2012.14200 (abstract)
[17] *Tessellation and interactive visualization of four-dimensional spacetime geometries*,
arXiv:2403.19036. https://arxiv.org/abs/2403.19036 (abstract)
[18] FreeSurfer GEMS, `kvlAtlasMesh.h`.
https://github.com/freesurfer/samseg/blob/dev/gems/kvlAtlasMesh.h
