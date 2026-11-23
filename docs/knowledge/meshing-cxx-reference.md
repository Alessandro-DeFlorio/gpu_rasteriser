# `meshing_cxx`: the earlier CPU rasteriser (review, 2026-11-23)

The owner and their supervisor (Koen Van Leemput) wrote a CPU rasteriser some time ago. It lives
outside this repo, in `~/git/samseg-rasterization/meshing_cxx/` (local only). The owner allows code
derived from it under MIT, with credit to both authors (ADR 0007).

## Files
- `wrap.cpp` (1,807 lines): pybind11 module built with `cppimport` (C++11, OpenMP, `-O3`).
  Rasteriser: `TriangleMapper` (lines 27–168), `to_be_erased` (172–201), `TriangleRasterizer`
  (482–639), `rasterize` (643–690). Pixel functors: `PixelVisitDrawer` (visit count),
  `SimplePiDrawer` (id + 1 and all four π, as double), `AlphaDrawer` (interpolation). Lines 695 and
  later hold the samseg costs, the prior and `fit_alphas`, which were not reviewed in detail.
- `wrap_ale.cpp`: the owner's copy. It adds a Hessian of the data term and
  `USE_SYMMETRIC_PRIOR_COST`. The rasteriser is identical.
- Prebuilt modules: `wrap.cpython-311-…so` and `…-314-…so`. They import in conda envs with Python
  3.11 (`babyseg_venv`) or 3.14 (`bda3`); `tetvox-dev` has 3.12.
- No licence file. Git authors: mostly Koen Van Leemput, some commits by the owner.

## Algorithm (verified by reading)
- **Per tet:** the vertices are snapped with `static_cast<int64_t>(v * 1e4)`, i.e. truncated toward
  zero at scale 10⁴. The bounding box is `ceil(min)`…`floor(max)`, clipped to `[0, size−1]`. The
  cofactors of `A` and `D = det A` are computed in int64 without normalisation.
- **Per voxel in the box:** all four plane functions are evaluated directly in int64,
  `E_j = n_j · (S·X − v₀)` with `E_0 = D − Σ_{j≥1} E_j`. The voxel is outside if any `E_j < 0`.
- **Tie-break (`to_be_erased`):** for each `j` with `E_j = 0`, the voxel is erased if the first
  non-zero component of `n_j`, scanning **axis 2, then 1, then 0**, is negative. This is a valid
  SoS rule with the opposite axis priority to ADR 0006 and GEMS.
- **π:** by default π is recomputed in double from the *unsnapped* vertices and clamped to
  `[0, 1]`. The path `#define GRID_SNAPPING_FOR_PIS` (off) uses `E_j / D` from the snapped integers
  instead.
- **Invalid tets:** `D < 0` can never pass, so those tets are silently skipped. `D = 0` is not
  excluded. Nothing is reported.
- **Parallelism:** OpenMP over tets, dynamic schedule. The per-thread functor copies share one
  output `py::array`, so the visit counter's `+= 1` races when two tets claim the same voxel. A race
  could hide a double owner, so use one thread when testing.
- **Layout:** output arrays are C-order with shape `size` (axis 2 fastest); position column `d` maps
  to array axis `d`.

## Measured (2026-11-23; adult atlas, prebuilt module, `count_pixel_visits_3d`)
| Placement | Result |
|---|---|
| Template space, grid `181×217×181` | All 6,998,400 centres of the half-open box owned exactly once; the layers `x_d = max_d` unowned (same as R1 §1.3) |
| Rotated, scale 1.137, into 256³ | No double owners. 11 voxels differ from a float box-hull test, all within 5·10⁻⁵ voxel of the hull (snapping shift) |
| Template ×2, ×3, ×4 (longest edge up to ~400 voxels) | Exactly once, no holes |
| Template ×5 / ×6 | int64 overflow: 1.79 M / 28.0 M holes, 0.82 M / 9.0 M double owners, silently |

- The worst-case int64 bound of this direct form is `6·M³ < 2⁶³` with `M = edge·10⁴`, i.e. about
  115 voxels per tet (R1 §2.2: direct int64 needs `3(k+L)+3 ≤ 63`). Real atlas tets stay well
  below the worst case, so the failure starts only between ×4 and ×5.
- Throughput: about 0.5 s single-threaded for the template grid.

## Compliance with ADR 0005/0006 (as of the review)
- **Compliant:** voxel centres at integer points; positions in voxel index space; half-open domain;
  exact SoS ownership.
- **Not compliant:**
  - axis priority (2 > 1 > 0 instead of 0 > 1 > 2);
  - snapping (10⁴ truncation instead of round-to-nearest at 2¹⁴);
  - no `__int128`, no window check, silent overflow;
  - π computed from the unsnapped geometry and clamped;
  - output format (double, id + 1, four π);
  - invalid tets neither excluded nor reported;
  - visit counter races under OpenMP;
  - tied to pybind11.
- What tetvox reuses (ADR 0007): the loop structure (clipped bounding-box scan, per-tet cofactor
  setup), not the arithmetic.
