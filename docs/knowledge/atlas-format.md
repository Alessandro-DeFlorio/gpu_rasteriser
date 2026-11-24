# samseg atlas `.npz` format and statistics

The atlases are local only (`atlases/`, gitignored, see ADR 0003). Regenerate all numbers below with:

    python scripts/inspect_atlas.py atlases/adult_template/atlas_adult.npz

Loader: `tetvox.load_atlas_npz(path) -> SamsegAtlas` (`src/tetvox/samseg.py`). Geometry and
priors go into the generic `atlas.mesh: TetMesh` (`src/tetvox/mesh.py`), the rest into the atlas
record (ADR 0005).

## Keys (verified 2026-11-23)

| Key | Stored dtype | Shape | `SamsegAtlas` field (canonical dtype) | Meaning |
|---|---|---|---|---|
| `positions` | f32 (f64 in `atlas_adult_7_classes`) | (N, 3) | `mesh.positions` (`F`) | vertex positions, **template voxel coordinates** |
| `tetrahedra` | i32 | (T, 4) | `mesh.tetrahedra` (i32) | vertex indices, 0-based |
| `alphas` | f32 / f64 | (N, K) | `mesh.values` (`F`) | per-vertex class priors α, rows sum to 1 (err ≤ 1.5e-6), min 0 |
| `affine` | f32 / f64 | (4, 4) | `affine` (f64) | template voxel → world (mm) |
| `can_move` | bool | (N, 3) | `can_move` | per-axis mobility mask |
| `names` | `<U..` | (K,) | `names` (tuple[str]) | class names |
| `colors` | i64 | (K, 4) | `colors` | RGBA |
| `FreeSurferLabels` | i64 | (L,) | `freesurfer_labels` | FreeSurfer label ids. **L ≠ K in reduced-class atlases** (7-class atlases keep the full 41/44 list) |
| `K` | f64 | () | `stiffness` | 0.1 in all atlases. Inferred: samseg mesh stiffness κ (`meshStiffness` default 0.1) |
| `log_mass_laplace_kappa1` | f64 | () | not loaded (owner, 2026-11-23: out of scope) | only in full-class atlases. Laplace approximation of the deformation prior's cost (log mass) at κ = 1 (owner, 2026-11-23). The POC logs it rescaled to the effective κ as `log Z_u (Laplace)` (100.0 adult, −186309 infant) |
| `mass_laplace_n_free` | i64 | () | not loaded | only in full-class atlases. Number of free coordinates in that Laplace approximation (162454 adult, 101437 infant; not 3·N). Prior-side data, not used by the rasteriser (ADR 0005) |

Canonical dtypes: `F` for geometry and priors, chosen by `load_atlas_npz(path, dtype=...)`
(float32 by default, or float64; ADR 0009), f64 for the affine. Only `atlas_adult_7_classes.npz`
stores f64, and its values are the same atlas mesh.

## Atlases present

| File(s) | N | T | K | Grid extent (voxels) | Voxel size (affine) |
|---|---|---|---|---|---|
| `adult_template/atlas_adult.npz` (+ `_backup`, `_bup`) | 54,598 | 338,993 | 44 | [0,180]×[0,216]×[0,180] | 0.63 mm, slightly rotated |
| `adult_template/atlas_adult_7_classes.npz` | 54,598 | 338,993 | 7 | same | 1 mm, LIA-style axis flip |
| `subtemplates_12/template_{1-2,3-5,6-8}mo_{7,41}.npz`, `atlas_reference_{7,41}.npz` | 33,882 | 210,872 | 7 / 41 | [0,140]×[0,168]×[0,143] | 1 mm |

The `atlas_config_*.json` files reference these as `./torch_atlases/...` (from the owner's
pipeline). `sharedGMMParameters*.txt` merge structures into GMM classes (format:
`mergedName numberOfComponents searchStrings`).

## Mesh structure (verified on adult and infant atlases)

- **Every tet is positively oriented** (`det(A) > 0`). There are no inverted or degenerate tets in the
  reference meshes.
- **The mesh tiles its bounding box exactly.** The sum of |volume| equals the bbox volume. All boundary
  vertices lie on the bbox faces, and the boundary faces are planes `x_d = 0` and `x_d = max_d`.
- Manifold: interior faces are shared by exactly 2 tets; no non-manifold faces; no unused vertices.
- `can_move`: all true in the full-class atlases. In `atlas_adult_7_classes`, exactly the 1256
  boundary vertices are fixed on all axes.
- Vertices are generally **not** on the voxel lattice. In the adult atlas, 0.25% of vertices have all
  coordinates integer and 15% have at least one. The latter are mostly boundary vertices, whose
  coordinates are 0 or max.

### Size distribution (matters for load balancing, research R1)

| | adult | infant template | infant reference |
|---|---|---|---|
| \|volume\| min / median / p99 / max (voxel³) | 0.14 / 8.3 / 164 / 20,100 | 0.056 / 4.8 / 201 / 14,500 | 0.15 / 4.9 / 199 / 10,400 |
| max/min volume ratio | 1.4e5 | 2.6e5 | 6.8e4 |
| longest edge min / median / max (voxels) | 2.1 / 7.4 / 99 | 2.4 / 5.5 / 92 | 2.4 / 5.6 / 89 |
| voxel centres in bbox: median / p99 / max | 96 / 1,870 / 133,000 | 45 / 2,400 / 90,000 | 48 / 2,300 / 81,000 |
| quality \|V\|/maxedge³ min / median (regular tet ≈ 0.118) | 0.0016 / 0.021 | 0.0006 / 0.029 | 0.0021 / 0.030 |

Implications for rasterisation:
- The median tet covers a few to a few tens of voxels, but the largest cover ~10⁴–10⁵. Work per tet
  is therefore heavy-tailed, so one-thread-per-tet will be badly load-imbalanced.
- Many tets are slivers (quality ≪ regular). Bounding-box enumeration wastes many tests on them,
  since tet volume / bbox volume is small.
- **Domain boundary:** boundary faces lie exactly on voxel-centre planes (`x_d = 0` and
  `x_d = max_d`, both integers). A strict half-open (top-left) rule would drop one whole layer of
  boundary voxels on the "non-owning" side. The rule needs a policy for samples on the outer
  boundary, e.g. treat them as inside, or define the grid so it excludes/includes them explicitly.

### Sparsity of `alphas` (adult atlas, measured 2026-11-23 for research R2)

Measured with `(alphas > 0).sum(1)` per vertex and `(alphas[tetrahedra] > 0).any(1).sum(1)` per
tet:
- non-zero `α` per vertex: mean 2.41, median 2, p99 4, max 8 (of K = 44);
- classes active in a tet (union over its 4 vertices): mean 3.15, median 3, p99 6, max 12;
  volume-weighted mean 2.75;
- 10.5% of the volume lies in tets with a single active class (identical `α` at all 4 vertices).
  The data-term gradient is exactly zero there;
- after merging to the 8 classes of `sharedGMMParameters_adult.txt`: 2.84 active classes per tet on
  average.

Implication: store per-vertex priors sparsely (8 entries per vertex cover this atlas).

### Rasterisation workload and exact-arithmetic facts (measured in research R1, 2026-11-23)

Adult atlas on its own template grid (181×217×181 voxel centres), vertices snapped to 2⁻¹⁴ voxel.
Details and method: `docs/research/R1-watertight-tet-rasterisation.md`.

- Smallest tet altitude: 0.179 voxel (p1 of the per-tet minimum: 0.56, median 1.9). Snapping the
  vertices to 2⁻ᵏ voxel inverts no tet for the tested k = 4, 6, 8, 10, 12, 14, 16.
- Bounding-box candidates: 79.07 M in total, i.e. 11.3 inside tests per owned voxel (fill 8.9%). The
  top 0.1% of tets hold 17% of the candidates, the top 1% hold 35%.
- (tet, y, z) rows: 10.57 M. Tet–brick pairs by bounding box: 3.60 M (4³ bricks), 1.30 M (8³),
  0.70 M (16³); mean list length per brick 31 / 88 / 348.
- Exact ties are common on the template grid: 630,255 (candidate, face) pairs have a plane function
  that is exactly zero. After a generic affine placement there were none.
- With the half-open tie-breaking rule (sample shifted by +ε, +ε², +ε³), exactly the 6,998,400
  centres of [0,180)×[0,216)×[0,180) are owned, each once. The 110,737 centres on the planes
  `x_d = max_d` are owned by no tet.
- The GEMS sources live in `github.com/freesurfer/samseg` (`gems/`, branch `dev`). The FreeSurfer
  repository keeps an identical copy of `kvlTetrahedronInteriorConstIterator.hxx` under
  `attic/gems/`.

### K3 row workload and scatter statistics (measured in research R4, 2026-11-24)

Adult atlas on its template grid (181×217×181), ownership from the CPU backend, rows along axis 0
(`row_span`). Counts of work items, not timings. Method: `docs/research/R4-nvdiffrast-lessons.md`
(§6.1 and "Method of the measurements").

- Of the 10,572,440 K3 rows only 3,183,490 (30.1%) own a voxel. A non-empty row spans 2.20 voxels
  on average (median 1, p99 13, max 95). 3,636 tets (1.1%) own no voxel.
- 4,717,158 rows (44.6%) lie under the tet's projection onto axes (1, 2); there are 1,795,028
  `(tet, i2)` slabs.
- A division-free test per face (the largest `t_j` over the row is negative) rejects 4,214,092 of
  the 7,388,950 empty rows and never a non-empty one. It cuts the int64 divisions of `row_span`
  from 42.0 M to 25.3 M.
- If every K3 lane loops over its own row, a warp runs for its longest row: 33.8 M lane-iterations
  for 7.0 M voxels (utilisation 20.7%). Pooling the rows of a 256-row block and dealing the voxels
  evenly gives 55.5%; one thread per tet gives 10.7%.
- Global atomics needed for a 12-float per-tet moment set: 84.0 M per voxel, 38.2 M per non-empty
  row, 6.5 M per group of same-tet rows in a warp of 32 consecutive rows, 4.3 M per group in a
  256-row block. In a voxel-parallel kernel a warp of 32 voxels along axis 2 holds 2.15 owned
  voxels per distinct tet (3.45 for a 4×8 block in axes 1, 2).
- Tet–brick pairs after an exact conservative plane test: 1.76 M (4³ bricks), 0.96 M (8³), 0.62 M
  (16³); 15 / 65 / 310 tets per brick. Running rows inside each brick would give 15.3 M / 15.0 M /
  13.6 M row pieces, more than the 10.6 M whole rows of K3.
