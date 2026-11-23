# samseg atlas `.npz` format and statistics

The atlases are local only (`atlases/`, gitignored, see ADR 0003). Regenerate all numbers below with:

    python scripts/inspect_atlas.py atlases/adult_template/atlas_adult.npz

Loader: `tetvox.load_atlas_npz(path) -> TetMesh` (`src/tetvox/mesh.py`).

## Keys (verified 2026-11-23)

| Key | Stored dtype | Shape | `TetMesh` field (canonical dtype) | Meaning |
|---|---|---|---|---|
| `positions` | f32 (f64 in `atlas_adult_7_classes`) | (N, 3) | `positions` (f32) | vertex positions, **template voxel coordinates** |
| `tetrahedra` | i32 | (T, 4) | `tetrahedra` (i32) | vertex indices, 0-based |
| `alphas` | f32 / f64 | (N, K) | `alphas` (f32) | per-vertex class priors α, rows sum to 1 (err ≤ 1.5e-6), min 0 |
| `affine` | f32 / f64 | (4, 4) | `affine` (f64) | template voxel → world (mm) |
| `can_move` | bool | (N, 3) | `can_move` | per-axis mobility mask |
| `names` | `<U..` | (K,) | `names` (tuple[str]) | class names |
| `colors` | i64 | (K, 4) | `colors` | RGBA |
| `FreeSurferLabels` | i64 | (L,) | `freesurfer_labels` | FreeSurfer label ids. **L ≠ K in reduced-class atlases** (7-class atlases keep the full 41/44 list) |
| `K` | f64 | () | `stiffness` | 0.1 in all atlases. Inferred: samseg mesh stiffness κ (`meshStiffness` default 0.1) |
| `log_mass_laplace_kappa1` | f64 | () | not loaded | only in full-class atlases; meaning unknown (100.0 adult, −186309 infant) |
| `mass_laplace_n_free` | i64 | () | not loaded | only in full-class atlases; meaning unknown (162454 adult, 101437 infant; not 3·N) |

Canonical dtypes: f32 for geometry and priors, f64 for the affine. Only `atlas_adult_7_classes.npz`
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
