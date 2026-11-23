# R1 — Watertight solid rasterisation of tetrahedral meshes on the GPU

Status: draft (researcher agent, 2026-11-23; not yet reviewed by the owner)
Brief: `briefs/R1-watertight-tet-rasterisation.md`

## Question
How should tetvox assign every voxel centre inside a tetrahedral mesh to **exactly one** tet, with
its barycentric coordinates, on the GPU? Four sub-questions:
1. the theory of watertight rasterisation and its generalisation from triangles to tets, including
   the tie-breaking rule and the outer domain boundary;
2. exactness: fixed point versus floating point, the bit budget, vertex snapping and its effect on
   differentiability, and accurate barycentrics for slivers;
3. load balancing for tets whose sizes differ by five orders of magnitude, on a mesh that deforms
   every iteration;
4. what existing differentiable rasterisers do and whether anything is reusable.

## Recommendation
1. **Tie-breaking rule.** A voxel centre `X` belongs to the tet whose open interior contains the
   shifted point `X + (ε, ε², ε³)` for all sufficiently small `ε > 0`. Per face `j` with plane
   function `E_j` and inward normal `n_j`: the sample passes if `E_j(X) > 0`, or if `E_j(X) = 0` and
   the first non-zero component of `(n_x, n_y, n_z)` is positive. This is the D3D top-left rule
   extended by one axis, the rule GEMS already intends, and the precise form of the "imaginary sun"
   of `equations.pdf` footnote 1 (§1.2).
2. **Exact integer arithmetic.** Evaluate the rule in integers. Snap each vertex once to a lattice
   of 2⁻ᵏ voxel with **k = 14**, and test `n_j·(X − O) + c_j ≥ 0` in **int64**, where the tie-break is
   folded into the per-face constant `c_j`. The per-tet constants `c_j` and the determinant `D` need
   **`__int128`** during setup. This fits when all vertices and samples lie in a window of 2¹⁰
   voxels per axis (§2.2). Floating point cannot give the guarantee: emulating the GEMS test in
   float64 on the adult atlas in template space leaves 23,166 of 6,998,400 voxels not owned exactly
   once (§2.1).
3. **One geometry for everything.** Use the snapped mesh for ownership, for the barycentrics
   (`π_j = E_j / D`, from the exact integer numerators) and for `∂π/∂v`. Treat snapping as the
   identity in the backward pass. Do not mix snapped ownership with unsnapped float barycentrics
   (§2.4, §2.5).
4. **Domain boundary.** Default to the half-open rule everywhere: a sample on a boundary face is
   owned only if that face's inward normal is lexicographically positive, and unowned samples get
   tet id −1. On the template grid this drops the layers `x_d = max_d`. Offer an exact closed-box
   mode as an option. Do **not** implement "boundary faces always count as inside": it double-counts
   (146 samples on the adult atlas, §1.3).
5. **Load balancing.** Start with a stateless **row-parallel span rasteriser**: one thread per
   (tet, y, z) row, which solves the x-interval by integer division and writes the span. It needs no
   atomics, because each voxel has exactly one writer. A throwaway prototype took 3.9 ms for the
   adult atlas on the dev laptop GPU, plus 0.3 ms of per-tet setup (§3).
6. **Second candidate.** A voxel-parallel **walk from the previous owner** is 2.5× faster in the
   prototype (1.5 ms) but is stateful and needs a fallback. Decide between the two with the replay
   experiment in "Open questions". Do not build hierarchical tile binning.
7. **Output.** A dense per-voxel `(tet_id: int32, π: 4 × float32)` grid, 20 bytes per voxel.

## Findings

Each claim is tagged:
- **[verified]**: read in the cited source;
- **[measured]**: result of a throwaway script or CUDA prototype written for this report (adult
  atlas, 338,993 tets; see "Method of the measurements" at the end of this section);
- **[inferred]**: my own reasoning.

### 1. Theory of watertight rasterisation, from 2D to 3D

#### 1.1 Edge functions and the top-left rule
- Pineda defines the edge function `E(x, y) = (x − X)·dY − (y − Y)·dX`. It is positive on one side
  of the edge, negative on the other and zero on it. It is linear, so it can be stepped with one
  addition per pixel. A triangle is the set where all three edge functions have the same sign.
  **[verified]** [1]
- Pineda states that "a tie breaker rule must be applied to the points that lie exactly on any of
  the edges", and names bounding-box traversal as the simplest traversal. **[verified]** [1]
- Direct3D's rule: a sample exactly on an edge is inside if the edge is a *top* edge (exactly
  horizontal and above the other edges) or a *left* edge (not horizontal, on the left side). If two
  edges touch the sample, both must be top or left. Vertices are first snapped to n.8 fixed point,
  and attribute interpolation is set up from the snapped positions. **[verified]** [2][3]
- In integer arithmetic the rule costs nothing per pixel: add a bias of −1 to the edge functions of
  the non-owning edges and keep the test `E ≥ 0`, because for integers `E > 0` equals `E ≥ 1`.
  **[verified]** [4]. The software rasteriser bundled with nvdiffrast does exactly this
  (`if (dy > 0 || (dy == 0 && dx <= 0)) curr--; // exclusive`). **[verified]** [13]
- With y pointing down, "left edge" means the inward normal has `n_x > 0`, and "top edge" means
  `n_x = 0, n_y > 0`. So the top-left rule is the statement "the sample shifted by `(ε, ε²)` is
  strictly inside". **[inferred]**

#### 1.2 Generalisation to tetrahedra
Setting. Tet `t` has vertices `v_0..v_3` with `D = det[v_1−v_0, v_2−v_0, v_3−v_0] > 0`. For the face
opposite vertex `j`, take its three vertices `(a, b, c)` in the order (1,3,2), (0,2,3), (0,3,1),
(0,1,2) for `j = 0..3`. Then `n_j = (b − a) × (c − a)` points into the tet, the plane function is
`E_j(x) = n_j · (x − a)`, and
- `E_j(v_j) = D`, `Σ_j E_j(x) = D` for every `x`, and `π_j(x) = E_j(x) / D`;
- `n_j / D` is the gradient of `π_j`, i.e. `p_j` in the notation of `samseg-math.md`. **[inferred]**
  (elementary; checked numerically, see §2.5)

**Rule.** `X` is owned by `t` iff for all four faces
`E_j(X) > 0  or  (E_j(X) = 0 and lexsign(n_j) > 0)`,
where `lexsign(n)` is the sign of the first non-zero component of `(n_x, n_y, n_z)`.

**Why it is watertight.** **[inferred]**, and **[measured]** exhaustively below.
1. For `X_ε = X + (ε, ε², ε³)`: `E_j(X_ε) = E_j(X) + n_x ε + n_y ε² + n_z ε³`. For small `ε > 0`
   its sign is the sign of `E_j(X)` if that is non-zero, and `lexsign(n_j)` otherwise. So the rule
   is the same as "`X_ε` is strictly inside `t`".
2. `n_j ≠ 0` for a non-degenerate tet, so `lexsign(n_j) ≠ 0`: the shifted point lies on no face
   plane through `X`. Planes not through `X` are a positive distance away. There are finitely many
   tets and samples, so one `ε` works for all of them.
3. A point that lies on no face is in the interior of exactly one tet of a tiling, or outside the
   mesh. Hence each sample has exactly one owner, or none if `X_ε` is outside the mesh.
4. Edges and vertices need no special case: a sample on an edge has two zero plane functions, and
   requiring both faces to pass is again "`X_ε` strictly inside". D3D states the same for two
   edges. **[verified]** [3]
5. The two tets that share a face see `E` and `−E`, `n` and `−n`. Their decisions are opposite as
   long as both compute the sign of `E` and of `n` exactly. This is where exact arithmetic enters
   (§2).

Preconditions. The argument needs the (snapped) mesh to be a tiling: every `D > 0` and the boundary
surface not self-intersecting. Then the signed count `Σ_t sign(D_t)·[x ∈ t]` equals the winding
number of the boundary, which is 1 inside, so with all signs positive exactly one tet contains each
generic point. **[inferred]** An inverted tet (`D < 0`) breaks the guarantee locally and must be
reported, not silently rasterised.

Relation to the other formulations in the brief:
- *Imaginary sun* (`equations.pdf`, footnote 1): boundary points whose view of a sun at
  `(−∞, −∞)` is blocked by the triangle body are excluded. **[verified]** [5] A single sun direction
  is ambiguous for edges parallel to it. The lexicographic rule is the precise version: the sun is
  infinitely far along −x, then −y, then −z, and a tet owns its "lit" faces. **[inferred]**
- *Simulation of Simplicity* replaces each coordinate by a polynomial in `ε` and never computes the
  perturbation. **[verified]** [17] Our rule is SoS applied to the query point only; the mesh is
  assumed non-degenerate. **[inferred]**
- *GEMS* implements the same lexicographic rule (see §1.5).

#### 1.3 The outer domain boundary
- With the rule above, a sample exactly on a boundary face is owned iff that face's inward normal
  is lexicographically positive. For the atlas on its template grid, the faces `x_d = 0` own their
  samples and the faces `x_d = max_d` do not. The owned set is exactly the half-open box:
  6,998,400 centres of [0,180)×[0,216)×[0,180) are owned once, and all 110,737 centres with some
  `x_d = max_d` are unowned. **[measured]**
- This half-open rule needs no knowledge of the domain shape. It stays correct when the mesh is
  placed by an affine transform inside a larger image, which is the normal samseg case, and when two
  meshes abut. **[inferred]**
- The tempting alternative, "a zero plane function on a boundary face always passes", is **not**
  watertight: 146 samples of the adult atlas are then owned by 2 to 5 tets. **[measured]** Reason:
  around a boundary edge or vertex the shifted point is outside the mesh, and the interior-face
  tests of several tets of the fan can pass at once. **[inferred]**
- An exact closed variant exists for an axis-aligned box domain `[0, max_d]`: flip the sign of the
  shift on each axis where `X_d = max_d`, i.e. use `(s_x ε, s_y ε², s_z ε³)` with `s_d = −1` on the
  upper planes. The shifted point is then always inside the box. On the adult atlas all 7,109,137
  centres are owned exactly once, and the result equals the half-open one on the half-open box.
  **[measured]** It is valid only while the boundary vertices stay on the box faces. **[inferred]**

#### 1.4 Prior art on solid rasterisation and voxelisation
- Schwarz & Seidel voxelise closed triangle *surfaces*: a voxel centre is inside if a ray from it
  crosses the surface an odd number of times. Each triangle flips (atomic XOR) all voxels behind
  it. To count a crossing through a shared edge or vertex exactly once they use the top-left fill
  rule in the 2D projection, implemented in floating point by adding "the smallest floating-point
  number" to the edge functions of left/top edges, together with "a consistent ordering of the
  vertices". **[verified]** [9] This is a parity scheme for surfaces. It does not identify *which*
  tet owns a voxel, so it is not reusable here. **[inferred]**
- They report two weaknesses of the triangle-parallel variant: uneven work inside a warp, and
  memory traffic and serialised atomics from flipping whole columns. Their tile-based variant (4×4
  columns per tile, one warp per tile, sorted tile/triangle queue) avoids the atomics.
  **[verified]** [9]
- Eisemann & Décoret process all slices in one rendering pass with bitwise arithmetic. I could not
  retrieve the paper; this is the description given by [9] and by the abstract. **[verified in
  [9] only]** [10]
- Laine & Karras ("cudaraster") use four stages: triangle setup (cull, clip, snap to fixed point,
  plane equations), bin rasteriser (128×128 pixel bins), coarse rasteriser (8×8 pixel tiles), fine
  rasteriser (one warp per tile). They guarantee hole-free output by "correct rasterization rules".
  They limit the viewport to 2048×2048 with **4 sub-pixel bits** to stay in 32-bit arithmetic, and
  note that more would need 64-bit arithmetic. **[verified]** [11] The copy in nvdiffrast has the
  same constants (`CR_SUBPIXEL_LOG2 4`, `CR_MAXVIEWPORT_LOG2 11`, `CR_TILE_LOG2 3`).
  **[verified]** [13]
- The tiling exists so that a framebuffer tile can live in shared memory and so that triangles are
  blended in input order. **[verified]** [11] Neither reason applies to us: there is no depth test
  or blending, and with an exact rule each voxel has one writer. **[inferred]**
- GPU point location in tets: Georgii & Westermann resample tets in barycentric space and use the
  barycentrics as the inside test. **[verified]** [21] A recent tet-to-voxel tool enumerates each
  tet's bounding box with a point-in-tet test, in OpenMP and CUDA; I found no treatment of ties in
  it. **[verified, skimmed]** [22]
- Projected tetrahedra (Shirley & Tuchman) and tet-mesh ray casting (Weiler et al.) are volume
  *rendering* methods: they composite along view rays and do not produce a per-voxel owner. I did
  not read these two papers; this characterisation comes from secondary sources. **[inferred]**
  [24]
- Woop et al. show for ray/triangle tests that consistent float edge functions are watertight along
  edges "but edges do not meet exactly at the vertices", and fix this with a fallback to double
  precision. **[verified]** [25] The same weakness applies to float plane functions for tets (§2.1).

#### 1.5 How GEMS does it today
- `kvlTetrahedronInteriorConstIterator` clips the tet's bounding box to the image (`ceil` of the
  minimum, `floor` of the maximum), computes `M = A⁻¹` in double with the cofactor formula, and
  evaluates the four barycentrics at the first voxel. It then steps through the box and **adds**
  the columns of `M` per step (per-row, per-column and per-slice start values are kept).
  **[verified]** [6]
- Inside test (`IsOutsideTetrahdron`): outside if any `π_j < 0`. If `π_j == 0` exactly, the sample
  is outside when the x-increment of `π_j` is negative; if that is zero, when the y-increment is
  negative; if that is zero too, when the z-increment is negative. The code comments describe this
  as the top-left rule for tetrahedra, obtained by virtually shifting the voxel to the right.
  **[verified]** [6] This is the rule of §1.2, evaluated in floating point.
- Parallelism: tet `i` goes to CPU thread `i mod nThreads`. The comment says this "isn't a
  particularly good way of load-balancing" but makes the round-off reproducible. **[verified]** [7]
- An old CUDA prototype exists in `gems/cuda/`: one thread *block* per tet, the 2D thread block
  tiled over the bounding box in x-y, each thread looping over z, and the same test evaluated
  directly (not incrementally) in `float` or `double`. Its visit counter uses `atomicAdd`.
  **[verified]** [8]
- The current sources are in `github.com/freesurfer/samseg` (`gems/`); the FreeSurfer repository
  keeps an identical copy of the iterator under `attic/gems/`. **[verified]** [6]

### 2. Exactness and fixed point

#### 2.1 Floating point does not give the guarantee
- I emulated the GEMS test with direct evaluation, as in the GEMS CUDA prototype, on the adult
  atlas **in template space** (unsnapped float32 vertices). Counting visits per voxel centre of the
  half-open box (6,998,400 centres): **[measured]**

  | arithmetic | 0 visits | 1 | 2 | 3 | ≥ 4 | not exactly once |
  |---|---|---|---|---|---|---|
  | float64 | 21,663 | 6,975,234 | 1,462 | 39 | 2 | 23,166 (0.33%) |
  | float32 | 26,094 | 6,971,098 | 1,177 | 30 | 1 | 27,302 (0.39%) |
  | exact (§2.2) | 0 | 6,998,400 | 0 | 0 | 0 | 0 |

  The same emulation is exactly watertight on a Kuhn grid with integer vertices, so the failures
  come from rounding, not from the emulation. **[measured]**
- Cause: this atlas has many exact ties in template space (630,255 candidate–face pairs with
  `E_j = 0`; 15% of the vertices have an integer coordinate). In floating point a barycentric that
  should be exactly zero comes out as ±rounding noise, independently in the two tets.
  **[measured]** (tie count) / **[inferred]** (cause)
- After a generic placement (rotation, scale 1.137, translation into a 256³ grid) there are no
  exact ties. On identical geometry, float64 then agrees with the exact result on all 16.8 M
  samples, and float32 loses 1 interior voxel of 10,069,647. **[measured]** So float64 is "almost
  always right" in the usual samseg setting, but it is not a guarantee, and it fails badly for
  lattice-aligned meshes, which are exactly what tests use.
- Caveat: the GEMS CPU code evaluates incrementally, which I did not emulate. Accumulated additions
  cannot be more exact than direct evaluation. **[inferred]**
- Could floats be made consistent? Computing each face's plane function from a canonical vertex
  order makes the two tets of a face agree bit-for-bit. That is watertight across faces, but not at
  edges and vertices, where the rounding noise of several planes can produce a cycle (no owner) or
  two owners. **[inferred]**; the analogous effect is documented for ray/triangle tests.
  **[verified]** [25]
- Robust float predicates (Shewchuk) take arbitrary float inputs and return exact signs through
  adaptive-precision arithmetic, with a running time that depends on how close to degenerate the
  input is. **[verified]** [18] They would work, but they are branchy and data-dependent, which
  suits a GPU badly, and they are unnecessary once the vertices are snapped. **[inferred]**

#### 2.2 Recommended number representation and overflow argument
Notation: `S = 2ᵏ`; `q = round(v·S)` are integer lattice coordinates; voxel centres are the lattice
points `S·X` with `X` integer; `O` is an integer voxel origin per tet (the lower corner of its
clipped bounding box).

Key identity. For a voxel centre, `E_j(S·X) = S·(n_j·(X − O)) + E_j(S·O)`. The low `k` bits are the
same for every voxel, so they can be dropped once per tet **[inferred]**; this is the 3D analogue of
the observation that with whole-pixel steps the sub-pixel scale enters only once **[verified]**
[4]:

    owned by face j   ⟺   E_j ≥ 1 − own_j   ⟺   n_j·(X − O) + c_j ≥ 0,
    c_j = floor((E_j(S·O) − 1 + own_j) / S),        own_j = [lexsign(n_j) > 0].

The tie-break is inside `c_j`, so the per-voxel test is one comparison per face.

Bounds. Assume every vertex of a rasterised tet and every sample lies in a window of `2ᴸ` voxels per
axis, so coordinate differences are at most `M = 2^(k+L)` lattice units and `|X − O| ≤ 2ᴸ`.
**[inferred]**

| quantity | worst-case magnitude | k = 14, L = 10 | type |
|---|---|---|---|
| `q` (relative to the window) | `2^(k+L)` | 2²⁴ | int32 |
| `n_j` component (difference of two products) | `2·M²` | 2⁴⁹ | int64 |
| `n_j·(X − O)` | `3·2M²·2ᴸ` | 3·2⁵⁹ | int64 |
| `c_j` | `6M³/S + 1` | 3·2⁵⁹ + 1 | int64 |
| `t_j = n_j·(X − O) + c_j` | `12·2^(2k+3L) + 1` | 3·2⁶⁰ + 1 < 2⁶² | **int64** |
| `E_j(S·O)`, `D` | `6·M³` | 6·2⁷² < 2⁷⁵ | **`__int128`** (setup only) |

So int64 suffices per voxel iff `2k + 3L ≤ 59`: k = 14 for a 1024-voxel window, k = 16 for 512.
k = 15 at L = 10 would overflow (3·2⁶² > 2⁶³). The bound uses the window for `|X − O|`, so it also
covers a walk that tests a tet far from the sample.

Measured on the adult atlas at k = 14: `|n|` 41 bits, `|c|` 45, `|t|` 46, `|E_O|` and `|D|` 59.
**[measured]** They fit in int64 for this mesh, but the worst case does not, hence `__int128` for
the setup.

Other widths: **[inferred]**
- **int32** per voxel would need `2k + 3L ≤ 27`, e.g. tets of 128 voxels with 3 sub-voxel bits. Not
  viable. In 2D the edge function needs about `2p + 2` bits for `p`-bit coordinates
  **[verified]** [4]; in 3D the plane function has degree 3, which is why D3D-style 32-bit budgets
  do not carry over.
- **double used as an exact integer** (53 bits) would need `2k + 3L ≤ 49`: k = 9 at L = 10, five
  sub-voxel bits fewer than int64.
- **int64 only, no 128-bit setup**, evaluating `E_j = n_j·(S·X − a)` directly, needs
  `3(k + L) + 3 ≤ 63`, i.e. `k + L ≤ 20`: k = 10 for a 1024 window. Simpler, but 2⁻¹⁰ voxel equals
  samseg's convergence threshold (§2.4).

Feasibility on the toolchain: `__int128` multiply, shift and divide work in device code with nvcc
13.5. A GPU setup kernel using it reproduced the Python big-integer reference bit-for-bit for all
338,993 tets (`n`, `c`, bounding boxes, `S/D`), in 0.30 ms including the prefix sum.
**[measured]** CUDA documents `__int128` as supported since 11.5/11.6 when the host compiler
supports it. **[verified]** [23] (from the release-note text returned by a web search)

Verification of the whole scheme on the adult atlas, k = 14: **[measured]**
- template grid: 79,065,084 candidates tested, 6,998,400 owned, each sample of the half-open box
  exactly once;
- generic placement, 256³: 113,705,158 candidates, 10,069,647 owned, none twice;
- the single-comparison form `t_j ≥ 0` agreed with the explicit rule of §1.2 on every candidate;
- Kuhn grids with all vertices on voxel centres (the worst case for ties): exactly once.

#### 2.3 Incremental or direct evaluation?
- In integers both are exact, so this is purely a performance choice. **[inferred]** GEMS's
  incremental scheme in double accumulates rounding. **[verified]** [6] (that it accumulates) /
  **[inferred]** (that this matters only at ties)
- On the GPU the natural form is direct evaluation at the start of a row (two multiplies per face)
  and then either adds along the row or, better, solving the row's x-interval in closed form
  (§3). **[inferred]**; the closed-form version is what the prototype measures.

#### 2.4 Is snapping acceptable for a differentiable interpolator?
- At k = 14 a vertex moves by at most 2⁻¹⁵ ≈ 3.1e-5 voxel per axis. A float32 coordinate in
  [128, 256) already has a spacing of 2⁻¹⁶ ≈ 1.5e-5, and in [256, 512) of 3.1e-5. So the lattice is
  within a factor two of what float32 positions can represent on atlas-sized grids. **[inferred]**
- samseg stops deforming when the maximal vertex movement falls below 0.001 voxel, and uses the
  same number for its line search. **[verified]** [20] That is 16 lattice steps at k = 14, but
  only 1 step at k = 10 and a quarter step at k = 8. Sub-voxel bit counts in the D3D range (8) are
  therefore too coarse here; k = 14 is the reason for the reduced int64 form. **[inferred]**
- The function `v ↦ L(snap(v))` is piecewise constant, so strictly it has zero gradient. The
  straight-through treatment (`∂snap/∂v = I`) returns the exact gradient of `L` at the snapped
  mesh. The same is implicitly true of float32 rounding today. **[inferred]**
- D3D does the same in spirit: attributes are interpolated from the *snapped* positions.
  **[verified]** [3]
- The brief's mixed proposal (ownership from snapped vertices, barycentrics and gradients from
  unsnapped floats) is **not** consistent: a sample owned by the snapped tet can lie up to about
  one lattice step outside the unsnapped tet, where a barycentric is negative. In
  `−log Σ_j l_j π_j` a negative `π_j` can make the argument negative when the other likelihoods are
  near zero. **[inferred]** Using the snapped geometry throughout removes the problem and costs
  nothing (§2.5).
- Snapping did not invert any tet of the adult atlas for the tested k = 4, 6, …, 16; its smallest
  altitude is 0.179 voxel. **[measured]** During registration tets can get thinner; the exact sign of `D` after
  snapping is available for free and must be checked.

#### 2.5 Accurate float barycentrics, also for slivers
- `π_j = E_j / D`, and `E_j / S = t_j + f_j` with `t_j` the int64 already computed for the inside
  test and `f_j = (E_j(S·O) − S·c_j) / S ∈ [0, 1]` a per-face constant. So
  `π_j = (float(t_j) + f_j) · (S / D)`: one int-to-float conversion, one add, one multiply.
  **[inferred]**
- Because `t_j ≥ 0` for an owned sample, every `π_j ≥ 0` by construction. The numerator is exact
  before conversion, so the relative error of each `π_j` is a few float32 ulps, independent of the
  tet's shape. On all 6,998,400 owned samples of the adult atlas, in float32: max absolute error
  1.2e-7, max relative error 1.9e-7, minimum 0.0, `|Σπ − 1| ≤ 1.4e-7`. **[measured]**
- The usual route `π = A⁻¹(x − v_0)` in float32 has an absolute error of roughly
  `ε·|x| / h_j` (`h_j` = altitude), which grows as the tet flattens. On the adult atlas, which has
  benign altitudes, it gave a maximum error of 1.7e-6 and three samples with a negative barycentric
  (−1.2e-7). **[measured]** (numbers) / **[inferred]** (scaling law)
- The gradient needs `p_j = ∂π_j/∂x = float(n_j) · (S / D)` in voxel units, from the same exact
  integers. **[inferred]**
- Store all four `π_j`. Recomputing `π_0 = 1 − π_1 − π_2 − π_3` loses the relative accuracy of a
  small `π_0`, which matters for terms like `r_m / π_m` in the Hessian. **[inferred]**

### 3. Load balancing

#### 3.1 Workload (adult atlas, template grid) **[measured]**
- Bounding-box candidates: 79.07 M for 7.00 M owned voxels, i.e. 11.3 inside tests per owned voxel
  (fill 8.9%). Median 96 per tet, maximum 133,380. The top 0.1% of tets hold 17.4% of the
  candidates, the top 1% hold 35.4%, the top 10% hold 59.1%.
- (tet, y, z) rows: 10.57 M, i.e. 1.5 row set-ups per owned voxel.
- Tet–brick pairs by bounding box: 3.60 M for 4³ bricks, 1.30 M for 8³, 0.70 M for 16³; mean list
  length per brick 31, 88 and 348.
- Coherence between iterations (generic placement, smooth displacement with maximum amplitude `a`):

  | `a` (voxels) | owners changed | tet tests per sample in a walk | longest walk | walk failures |
  |---|---|---|---|---|
  | 0.02 | 0.53% | 1.005 | 5 | 0 |
  | 0.1 | 2.60% | 1.026 | 6 | 0 |
  | 0.5 | 12.1% | 1.13 | 15 | 0 |
  | 2.0 | 37.5% | 1.54 | > 200 | 1 loop, 31 wrong exits (of 10.07 M) |

  In addition 749 / 4,007 / 20,368 / 81,545 samples changed between inside and outside the mesh.

#### 3.2 Prototype timings **[measured]**
Throwaway CUDA kernels, exact int64 tests, k = 14, writing `tet_id` and four float32 barycentrics.
Dev machine: RTX PRO 1000 Blackwell laptop GPU, nvcc 13.5, `-O3 -arch=native`. Best of 7 runs. Mesh
constants already on the device. Every variant reproduced the reference owner grid exactly (0
mismatches).

| kernel | work items | template grid 181×217×181 | generic placement 256³ |
|---|---|---|---|
| per-tet setup (snap, normals, `__int128` constants, prefix sum) | 339 k tets | 0.30 ms | 0.29 ms |
| K1: one thread per bbox candidate (tet found by binary search in a prefix sum) | 79 M / 114 M | 9.1 ms | 13.7 ms |
| K2: one thread per tet, loop over its bbox (no balancing) | 339 k | 36 ms | 53 ms |
| K3: one thread per (tet, y, z) row, x-span by integer division | 10.6 M / 13.8 M | **3.9 ms** | **5.1 ms** |
| K4: one thread per voxel, walk from the previous owner (zero deformation) | 7.1 M / 16.8 M | 1.5 ms | 2.4 ms |

Limits of these numbers: one laptop GPU; un-tuned kernels; K4 measured only with an unchanged mesh
(the table in §3.1 gives its work growth); the K4 time excludes the fallback and the boundary pass
it needs.

#### 3.3 The four strategies of the brief
(a) **Tet-parallel bounding-box enumeration.**
- Unbalanced (K2), it is 4× slower than balanced (K1), as the heavy tail predicts. **[measured]**
- Balancing does not need size bins or sub-boxes: a prefix sum over the per-tet candidate counts
  turns the problem into a flat list, and each thread recovers its tet by binary search (K1).
  **[measured]** that this works. It is the count-then-offset pattern that [9] uses to build its
  tile/triangle queue. **[verified]** [9]
- Work is proportional to the bounding boxes: 11.3 tests per owned voxel, mostly rejections.
- No atomics: the exact rule gives each voxel one writer. **[inferred]**, consistent with the
  bit-identical results. Stateless, so mesh deformation costs nothing extra.

(d) **Scanline / slab traversal.**
- The light version is K3: per row, each face gives a bound `dx ≥ ceil(−r/a)` or
  `dx ≤ floor(r/(−a))` (exact integer division), and the row is accepted or rejected directly when
  `a = 0`. Only owned voxels are visited afterwards. Work: 1.5 row set-ups plus one write per owned
  voxel. 2.3–2.7× faster than K1. **[measured]**
- A full slab decomposition (z-slice → polygon → rows) would also remove the empty rows, but it
  adds clipping logic for a gain bounded by the row set-up share. Not worth it now. **[inferred]**

(c) **Voxel-parallel point location.**
- *Walk from the previous owner* (K4). Tet connectivity never changes, so the neighbour table is
  built once. Work is close to the minimum of one tet test per voxel for realistic steps
  (§3.1). No lists, no prefix sums, one writer per voxel by construction. **[measured]**
- Its weaknesses:
  - it is stateful: it needs a first full rasterisation (e.g. by K3), and again at each resolution
    level;
  - the plain visibility walk can loop in a non-Delaunay triangulation **[verified]** [19], which
    happened once at a 2-voxel step **[measured]**;
  - a walk can leave through a boundary face although the sample is inside elsewhere (31 cases at a
    2-voxel step) **[measured]**; the walking literature assumes a convex domain **[verified]**
    [19];
  - samples that were outside the mesh have no starting tet, so the boundary tets must be
    rasterised separately each iteration. **[inferred]**
- Remedies: the stochastic walk has zero probability of looping forever **[verified]** [19]; a
  step cap with a K3 fallback handles both failure modes. **[inferred]**
- *Uniform grid of tet lists.* The lists must be rebuilt every iteration (count, prefix sum, fill).
  With 8³ bricks a voxel would scan a list of 88 tets on average until it finds its owner, far more
  than the 11.3 tests of K1. Exact brick–tet culling and smaller bricks reduce this, at the cost of
  more pairs (3.6 M for 4³). **[inferred]** from the measured list lengths; not prototyped.
- *BVH.* Refitting per iteration is cheap because the topology is static, but traversal per voxel
  costs more than a walk that usually succeeds at the first test. **[inferred]**; not prototyped.

(b) **Hierarchical binning (cudaraster style).**
- Its purpose is ordered blending and depth testing in a shared-memory framebuffer tile.
  **[verified]** [11] We need neither.
- The fine stage would still test each tet against the voxels of the bricks it overlaps, i.e. about
  the K1 candidate count, plus the cost of building 1.3 M tet–brick pairs every iteration. It
  cannot beat K3. **[inferred]**; not prototyped.

Ranking: **1. K3 row-parallel spans; 2. K4 walk from the previous owner; 3. K1 candidate-parallel
(simplest, good first GPU oracle); 4. hierarchical binning and grid lists.** All measured variants
are in the 1–14 ms range, so the rasteriser is unlikely to be the bottleneck next to interpolating
K = 44 priors per voxel. That favours the simplest stateless design. **[inferred]**

#### 3.4 Output layout
- Dense per-voxel `(tet_id int32, π 4×float32)`: 20 B per voxel, 142 MB for the adult template
  grid and 336 MB for 256³. It is what a per-voxel cost kernel and a DLPack consumer want, and what
  nvdiffrast and PyTorch3D return in 2D (§4). **[inferred]**
- K3 visits voxels grouped by tet, so a compact tet-sorted list (CSR: tet → voxel indices) could be
  emitted with one extra counting pass. That would let the gradient scatter run per tet without
  per-voxel atomics. Whether this is wanted depends on R2. **[inferred]**

### 4. Differentiable rasterisers
- **nvdiffrast.** The rasteriser outputs per-pixel triangle id and barycentrics `(u, v)`; a separate
  interpolation op consumes them, and its backward pass scatter-adds into the vertex attributes with
  the barycentrics as weights. Point-sampled visibility "cannot produce visibility-related gradients
  for vertex positions"; a separate antialiasing op supplies those at silhouettes.
  **[verified]** [12] Its CUDA rasteriser snaps to integer sub-pixel coordinates and uses the −1
  bias fill rule. **[verified]** [13]
- **SoftRas.** Replaces the hard inside test by a per-triangle probability map (a sigmoid of the
  signed distance to the triangle) and an aggregation over all triangles, so that every triangle
  influences every pixel. **[verified]** [14]
- **PyTorch3D.** Returns, per pixel, the K nearest faces with face id, barycentrics and signed
  distances; uses a two-pass coarse-to-fine scheme over image tiles; decouples rasterisation from
  shading; softens face boundaries by blending as in SoftRas. **[verified]** [15]
- **Kaolin / DMTet / DefTet.** Kaolin offers marching tetrahedra (differentiable iso-surface
  extraction from an SDF on a deformable tet grid) and the DefTet volumetric renderer. Its
  documentation lists no op that voxelises tets onto a regular grid or locates points in tets.
  **[verified]** [16]
- **Consequence for tetvox.** The soft formulations exist because triangle *coverage* is
  discontinuous in the vertex positions and the silhouette carries the signal. Our interpolant is
  continuous across interior faces, so an ownership change causes a kink, not a jump, and no
  boundary term is needed in the interior. The outer mesh boundary is the one true discontinuity;
  samseg ignores it. This is R2's question 4. **[inferred]**
- **Reusable.** The op split (rasterise → `(id, barycentrics)` → interpolate → scatter) and the
  integer fill-rule technique. No code: all of these are 2D triangle rasterisers with depth.
  **[inferred]**

### Recommended algorithm (pseudocode)

    const k = 14;  S = 1 << k                    // lattice = 2^-14 voxel
    FACE = {{1,3,2}, {0,2,3}, {0,3,1}, {0,1,2}}  // (a,b,c) of the face opposite vertex j; inward normal

    setup_tet(t):                                // one thread per tet, every iteration
      for v in 0..3:  q[v] = llrint(double(pos[tets[t][v]]) * S)        // int64; identical for all tets
                                                                        // sharing the vertex
      lo[d] = clamp(ceil_div (min_v q[v][d], S), 0,  G[d])              // integer voxel bbox,
      hi[d] = clamp(floor_div(max_v q[v][d], S), -1, G[d] - 1)          // clipped to the grid
      ext   = max(hi - lo + 1, 0)
      for j in 0..3:
        (a, b, c) = q[FACE[j]]
        n[j]  = cross(b - a, c - a)                                     // int64, |.| <= 2^49
        E_O   = dot_i128(n[j], S * lo - a)                              // __int128, |.| < 2^75
        own   = n[j].x > 0 || (n[j].x == 0 && (n[j].y > 0 || (n[j].y == 0 && n[j].z > 0)))
        c[j]  = int64((E_O - 1 + own) >> k)                             // floor; tie-break folded in
        f[j]  = float(E_O - (i128(c[j]) << k)) / S                      // in [0, 1]
      D = dot_i128(n[3], q[3] - q[0])                                   // __int128
      if D <= 0:  ext = 0;  atomic_inc(n_bad_tets)                      // degenerate or inverted
      inv[t]  = float(double(S) / double(D))
      rows[t] = (ext.x > 0) ? ext.y * ext.z : 0
    row_offset = exclusive_scan(rows)

    inside(t, X):                                // the predicate; shared by CPU reference and kernels
      return for all j:  dot(n[j], X - lo) + c[j] >= 0                  // int64

    raster_row(g):                               // one thread per (tet, dy, dz) row
      t = upper_bound(row_offset, g) - 1;   r = g - row_offset[t]
      dy = r % ext.y;   dz = r / ext.y
      x0 = 0;   x1 = ext.x - 1
      for j in 0..3:
        b[j] = n[j].y * dy + n[j].z * dz + c[j]                         // int64
        a    = n[j].x
        if      a > 0:  x0 = max(x0, ceil_div(-b[j], a))
        else if a < 0:  x1 = min(x1, floor_div(b[j], -a))
        else if b[j] < 0:  return
      for dx in x0..x1:                          // every voxel here is owned by t, and by no other tet
        vox = linear_index(lo + (dx, dy, dz))
        tet_id[vox] = t
        for j in 0..3:  pi[vox][j] = (float(b[j] + n[j].x * dx) + f[j]) * inv[t]

    rasterise():  fill tet_id with -1;  setup_tet for all t;  raster_row for all g

Notes:
- `ceil_div` and `floor_div` are exact integer divisions rounding towards +∞ and −∞.
- The mesh is valid for this call iff `n_bad_tets == 0` and every vertex of a tet that touches the
  grid lies inside the 2¹⁰-voxel window; otherwise report an error.
- Closed-box mode (optional): keep `c_strict[j] = floor((E_O − 1) / S)` and a flag
  `tie[j] = (E_O mod S == 0)`. Then `E_j = 0 ⟺ t_j == −1 and tie[j]`, and such a face passes iff the
  first non-zero of `(s_x n_x, s_y n_y, s_z n_z)` is positive, with `s_d = −1` where `X_d = max_d`.

### Method of the measurements
- Reference: a numpy rasteriser with Python big integers for the per-tet constants, enumerating
  every bounding-box candidate, plus emulations of the GEMS test. Prototype: two stand-alone `.cu`
  programs. All of it is throwaway and lives outside the repo, in
  `/tmp/claude-3234968/-u-68-defloa2-unix-git-gpu-rasteriser/aa8a6d25-925a-49e5-9048-daff6300cfe5/scratchpad/r1`
  (`ref.py`, `exact_atlas.py`, `float_emul.py`, `affine_test2.py`, `misc_checks.py`, `walk_test.py`,
  `gpu/raster.cu`, `gpu/setup.cu`). That directory is temporary; the pseudocode above is sufficient
  to rebuild the reference.
- "Generic placement": the adult atlas rotated by 7°, −4°, 11° about x, y, z, scaled by 1.137 and
  translated into a 256³ grid, with part of the mesh outside the grid.
- The deformation test uses a sum of six sinusoids per component with wavelengths of at least 64
  voxels, scaled to the stated maximum displacement.

## Options compared

Number representation:

| Option | Pros | Cons | Fit for tetvox |
|---|---|---|---|
| Float32/float64 plane tests with the lexicographic tie-break (GEMS today) | No snapping; simplest | No guarantee; 0.33% of voxels wrong on the template grid in float64; negative barycentrics | Reject |
| Float with canonical per-face evaluation | Consistent across faces | Not watertight at edges and vertices | Reject |
| Adaptive exact predicates on raw floats (Shewchuk) | Exact, no snapping | Branchy, data-dependent cost, complex on GPU | Reject |
| int64 only, k = 10 (`k + L ≤ 20`) | Exact; no 128-bit arithmetic | Lattice 2⁻¹⁰ voxel equals samseg's stop criterion | Fallback if `__int128` is unwanted |
| **int64 per voxel + `__int128` setup, k = 14 (`2k + 3L ≤ 59`)** | Exact; lattice near float32 resolution; one compare per face; exact non-negative barycentrics | 128-bit arithmetic in the setup kernel; 1024-voxel window | **Recommended** |
| double as exact integer, k = 9 | No integer types | Five sub-voxel bits fewer than int64 | Reject |

Load balancing (adult atlas, dev laptop GPU):

| Option | Pros | Cons | Fit for tetvox |
|---|---|---|---|
| (a) K2 thread per tet | Trivial | 36 ms; heavy-tailed imbalance | Reject |
| (a) K1 thread per candidate, prefix-sum balanced | Simple, stateless, no atomics | 11.3 tests per owned voxel; 9.1 ms | First GPU oracle |
| **(d) K3 row-parallel spans** | Stateless, no atomics, 1.5 set-ups per owned voxel; 3.9 ms | int64 division per face and row | **Recommended baseline** |
| (c) K4 walk from previous owner | About 1 tet test per voxel; 1.5 ms; single writer by construction | Stateful; needs neighbour table, loop/exit fallback and a boundary pass | Second candidate; decide by experiment |
| (c) grid of tet lists / BVH | Stateless voxel-parallel | Lists rebuilt per iteration; long lists (88 at 8³) | Not competitive (inferred) |
| (b) hierarchical binning | Proven for triangles | Solves ordering/blending problems we do not have; rebuild per iteration | Reject (inferred) |

## Open questions
1. **Boundary policy (owner).** Is the half-open rule acceptable as the definition of "the domain",
   so that on the template grid the layers `x_d = max_d` are unowned? Or is the closed-box mode
   needed, and for which use (tests on the template grid, atlas building)?
2. **K3 or K4.** Experiment: record the vertex positions of a real samseg deformation run (all
   resolution levels, all iterations and line-search evaluations) and replay them through both
   kernels. Measure, per call, the wall time including setup, and for K4 the number of walk steps,
   capped walks, wrong exits and outside-to-inside samples. Run on the dev laptop and on the target
   GPU. Compare with the time of the interpolate and scatter kernels from R2. Adopt K4 only if
   rasterisation with K3 exceeds roughly 10–15% of an iteration and K4's fallback rate is
   negligible.
3. **Choice of k.** k = 14 is argued from the float32 spacing and samseg's 0.001-voxel criterion,
   not from a registration run. Experiment: run the same registration with k ∈ {8, 10, 12, 14, 16}
   (16 needs a 512-voxel window) and compare the cost trajectory and the final vertex positions.
4. **Invalid meshes.** What should happen when snapping or the optimiser produces `D ≤ 0`, or when
   a tet touching the grid has a vertex outside the 1024-voxel window: raise, or skip and report a
   count? The line search may deliberately probe inverted configurations.
5. **Output layout.** Dense grid only, or also the tet-sorted list (§3.4)? Depends on R2.
6. **Not verified.** The GEMS CPU path was emulated with direct, not incremental, evaluation. Grid
   lists, BVH and hierarchical binning were not prototyped. I could not retrieve Eisemann & Décoret,
   Shirley & Tuchman or Weiler et al. Timings are from one laptop GPU.
7. **FP64 use.** The setup kernel uses double for `S / D` and the snapping (339 k tets, negligible).
   The per-voxel path is int64 and float32 only. Whether int64 division in K3 is slow on other GPUs
   is untested.

## Impact on design
- **Shared header** (`csrc/include/tetvox/`, `__host__ __device__`): `snap`, `setup_tet`, `inside`,
  and the barycentric formula. The CPU reference oracle and every kernel call the same functions,
  so GPU and CPU owners are bit-identical by construction (the prototype achieved this).
- **Per-tet setup record** (structure of arrays, rebuilt every call): `n` 4×3 int64, `c` 4 int64,
  `f` 4 float32, `inv` float32, `lo` and `ext` 3 int32 each: 172 B per tet, 58 MB for the adult
  atlas. Plus one int64 row offset per tet.
- **Kernels**: `setup_tet` (per tet) → exclusive scan → `raster_row` (per row). No atomics, no
  sorting, no persistent state. Optional later: neighbour table (static, 4 int32 per tet) and a
  walk kernel.
- **API**: inputs are float32 positions in grid voxel coordinates, the tets and the grid shape.
  Outputs are `tet_id` (int32, −1 outside) and `π` (4 float32). Return or raise on
  `n_bad_tets > 0`. Document the half-open convention and the 2¹⁰-voxel window.
- **Gradients**: forward and backward both use the snapped mesh; `p_j = float(n_j)·(S/D)`. The
  backward pass treats snapping as the identity.
- **Precision constants**: k = 14, window 2¹⁰ voxels, condition `2k + 3L ≤ 59`; put a
  `static_assert` on it.
- **Tests**:
  - exhaustive exactly-once on the adult and infant atlases in template space (expected owned
    count: the half-open box);
  - a Kuhn grid with all vertices on voxel centres (maximal ties), and the same grid scaled;
  - a rotated placement with part of the mesh outside the grid;
  - GPU owner grid bit-identical to the CPU reference;
  - `π_j ≥ 0` and `|Σπ − 1| < 1e-6` for every owned voxel;
  - an adversarial bit-budget test with tets spanning the full window.

## References
[1] J. Pineda, *A Parallel Algorithm for Polygon Rasterization*, SIGGRAPH 1988, Computer Graphics
22(4):17–20. https://people.csail.mit.edu/ericchan/bib/pdf/p17-pineda.pdf
[2] Microsoft, *Rasterization Rules (Direct3D 10/11)*.
https://learn.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules
[3] Microsoft, *Direct3D 11.3 Functional Specification*, §3.4.1 Coordinate Snapping, §3.4.2 Triangle
Rasterization Rules. https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
[4] F. Giesen, *Triangle rasterization in practice*, blog, 2013.
https://fgiesen.wordpress.com/2013/02/08/triangle-rasterization-in-practice/
[5] K. Van Leemput, *Working with tetrahedral meshes in samseg*, 25 Jul 2025, footnote 1. Local copy
`literature/equations.pdf`.
[6] FreeSurfer GEMS, `kvlTetrahedronInteriorConstIterator.hxx`.
https://github.com/freesurfer/samseg/blob/dev/gems/kvlTetrahedronInteriorConstIterator.hxx
(identical copy: https://github.com/freesurfer/freesurfer/blob/dev/attic/gems/kvlTetrahedronInteriorConstIterator.hxx)
[7] FreeSurfer GEMS, `kvlAtlasMeshRasterizor.cxx`.
https://github.com/freesurfer/freesurfer/blob/dev/attic/gems/kvlAtlasMeshRasterizor.cxx
[8] FreeSurfer GEMS, `cuda/simplesharedtetrahedroninterior.hpp`, `cuda/simplesharedtetrahedron.hpp`,
`cuda/visitcounteraction.hpp`. https://github.com/freesurfer/freesurfer/tree/dev/attic/gems/cuda
[9] M. Schwarz, H.-P. Seidel, *Fast Parallel Surface and Solid Voxelization on GPUs*, ACM TOG 29(6)
(SIGGRAPH Asia), 2010. https://michael-schwarz.com/research/publ/files/vox-siga10.pdf
[10] E. Eisemann, X. Décoret, *Single-pass GPU Solid Voxelization for Real-time Applications*,
Graphics Interface 2008. https://inria.hal.science/inria-00345291 (not retrieved; see §1.4)
[11] S. Laine, T. Karras, *High-Performance Software Rasterization on GPUs*, HPG 2011.
https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
[12] S. Laine, J. Hellsten, T. Karras, Y. Seol, J. Lehtinen, T. Aila, *Modular Primitives for
High-Performance Differentiable Rendering*, ACM TOG 39(6), 2020. https://arxiv.org/abs/2011.03277
[13] nvdiffrast sources, `csrc/common/cudaraster/impl/Constants.hpp` and `Util.inl`.
https://github.com/NVlabs/nvdiffrast/tree/main/csrc/common/cudaraster/impl
[14] S. Liu, T. Li, W. Chen, H. Li, *Soft Rasterizer: A Differentiable Renderer for Image-based 3D
Reasoning*, ICCV 2019. https://arxiv.org/abs/1904.01786
[15] N. Ravi et al., *Accelerating 3D Deep Learning with PyTorch3D*, 2020.
https://arxiv.org/abs/2007.08501
[16] T. Shen, J. Gao, K. Yin, M.-Y. Liu, S. Fidler, *Deep Marching Tetrahedra*, NeurIPS 2021,
https://arxiv.org/abs/2111.04276; Kaolin documentation, *Volumetric Mesh Optimization*,
https://kaolin.readthedocs.io/en/latest/notes/volumetric_meshes.html
[17] H. Edelsbrunner, E. P. Mücke, *Simulation of Simplicity: A Technique to Cope with Degenerate
Cases in Geometric Algorithms*, ACM TOG 9(1):66–104, 1990. https://arxiv.org/abs/math/9410209
[18] J. R. Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
Predicates*, CMU-CS-96-140R, 1997. https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf
[19] O. Devillers, S. Pion, M. Teillaud, *Walking in a Triangulation*, INRIA RR-4120, 2001.
https://inria.hal.science/inria-00072509
[20] samseg, `samseg/SamsegUtility.py`, default optimisation options.
https://github.com/freesurfer/samseg/blob/dev/samseg/SamsegUtility.py
[21] J. Georgii, R. Westermann, *A Generic and Scalable Pipeline for GPU Tetrahedral Grid Rendering*,
IEEE Visualization 2006.
https://www.cs.cit.tum.de/fileadmin/w00cfj/cg/Research/Publications/2006/A_Generic_and_Scalable_Pipeline/vis06_tet.pdf
[22] *Robust Voxelization and Visualization by Improved Tetrahedral Mesh Generation*, arXiv
2106.01326, 2021. https://arxiv.org/abs/2106.01326
[23] NVIDIA, *CUDA Toolkit Release Notes* 11.5 and 11.6 (`__int128`).
https://docs.nvidia.com/cuda/archive/11.6.2/cuda-toolkit-release-notes/index.html
[24] P. Shirley, A. Tuchman, *A Polygonal Approximation to Direct Scalar Volume Rendering*, 1990;
M. Weiler, M. Kraus, M. Merz, T. Ertl, *Hardware-Based Ray Casting for Tetrahedral Meshes*, IEEE
Visualization 2003. (Not read; cited from secondary descriptions only.)
[25] S. Woop, C. Benthin, I. Wald, *Watertight Ray/Triangle Intersection*, JCGT 2(1), 2013.
https://jcgt.org/published/0002/01/05/paper.pdf
