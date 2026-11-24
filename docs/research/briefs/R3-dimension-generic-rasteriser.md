# Brief R3 — A dimension-generic rasteriser: cost, design and whether it is worth it

Output: `docs/research/R3-dimension-generic-rasteriser.md` (follow `docs/research/_template.md`).

## Context
tetvox is a watertight, differentiable voxeliser for tetrahedral meshes (3D only today). The rule,
the arithmetic and the op set are settled for 3D:
- ADR 0006: Simulation-of-Simplicity ownership (the sample shifted by `(ε, ε², ε³)` is strictly
  inside), vertices snapped to `2⁻ᵏ` voxel, int64 per-voxel test `n_j·(X − O) + c_j ≥ 0` with the
  tie-break folded into `c_j`, int128 per-tet setup, bit budget `2k + 3L ≤ 59`, K3 row kernel.
- ADR 0007–0009: CPU and CUDA backends share one header of `__host__ __device__` procedures
  (`snap`, `setup_tet`, `plane_values`, `owns`, `barycentric`, `row_span`) and are bit-identical;
  ops `setup_tets`, `locate`, `barycentrics`, `rasterize`, `count_claims`, then interpolation and
  gradient ops (`interpolate`, `interpolate_vjp_*`, `tet_moments`, `vertex_grad`,
  `accumulate<Cost>`).
- Code: `csrc/include/tetvox/raster.hpp` and `types.hpp` (the procedures, hand-written for 3D:
  `int64_t normal[4][3]`, a fixed face table, an explicit cross product, `row_span` over axis 0),
  `csrc/src/cpu.cpp` (OpenMP backend). The CUDA backend (K3) is the next ROADMAP item and not built
  yet, so the moment to make the header dimension-generic, if ever, is now.
- Measured CPU timings and the device/host bit-identity check are in
  `docs/knowledge/rasteriser-cpu.md`. R1 §3.2 has K3 GPU prototype timings (3.9 ms + 0.3 ms setup
  for the adult atlas).

Read first: `docs/research/R1-watertight-tet-rasterisation.md` (§1.1–1.3 rule and proof, §2.2 bit
budget, §3 load balancing), `docs/research/R2-decoupling-cost-from-rasteriser.md` (op split and
gradients), ADRs 0005–0009, `docs/knowledge/rasteriser-cpu.md`, `docs/knowledge/build.md`,
`docs/knowledge/atlas-format.md`. Do not re-derive what they already settle for 3D.

The owner's question: the generalised top-left rule (SoS) and the edge/plane functions hold in any
dimension. Can the rasteriser be written once, for any dimension `d` (2D triangles in pixels, 3D
tets in voxels, perhaps 4D), with compile-time generic loops, at the same speed as dimension-specific
code? Or does each dimension need its own optimised code, possibly behind a shared abstraction? Is
it worth doing at all? A parallel brief (R4) covers nvdiffrast in depth; here use 2D rasterisers
(including nvdiffrast's cudaraster) only as data points on what an optimised 2D path looks like.

## Questions
1. **Theory in d dimensions.**
   - State the ownership rule for a `d`-simplex in `ℝᵈ` (sample shifted by `(ε, ε², …, εᵈ)`), the
     face functions (cofactors / generalised cross product of `d − 1` edge vectors), the orientation
     convention (face opposite vertex `j`, sign pattern) and barycentrics `π_j = E_j / D`. Give or
     cite a proof that it partitions `ℝᵈ` exactly once for any `d` (R1 gave it for `d = 3`), and how
     the outer-boundary rule of R1 §1.3 generalises.
   - Does anything about the rule, the snapping or the barycentric formula of R1 §2.5 change
     qualitatively with `d`?
2. **Bit budget as a function of d.** Generalise R1 §2.2: the bits needed by the face normals
   (`(d−1)×(d−1)` minors), by `D` (a `d×d` determinant), by `c_j` and by the per-voxel test, as a
   function of `d`, `k` and `L`. For which `(d, k, L)` do int32, int64 and int128 suffice, for the
   setup and for the per-voxel test? Consequence: are the integer types themselves
   dimension-dependent (e.g. 2D fits int32 per pixel, 4D needs int128 per voxel), and what does that
   do to a "write once" design? Check the formula numerically for `d = 2, 3, 4`.
3. **Traversal per dimension.** K3 generalises to "one thread per box row of the `(d−1)`-dimensional
   cross-section, solve the span along one axis". Is that still the right traversal in 2D (images of
   up to ~4k², many tiny triangles; compare with tile/hierarchical rasterisers and with the 2D
   specific tricks of cudaraster and other GPU software rasterisers) and in 4D? Does the best
   traversal depend on `d` enough that a generic traversal loses materially? Which parts are
   traversal-independent (setup, plane values, inside test, barycentrics, moments, vertex gradient)?
4. **Implementation techniques and their cost.** Survey how dimension-generic geometric code is
   written in C++ and CUDA, and what it costs:
   - `template <int D>` with fixed-size arrays, `#pragma unroll`, `if constexpr`, compile-time face
     tables (generated combinatorially), determinants by cofactor expansion vs Bareiss vs closed
     forms; traits that pick integer widths per `D`; explicit specialisations of hot procedures
     behind a generic interface.
   - Prior art: CGAL's dD kernel (`Epick_d` / `Epeck_d`), deal.II and DUNE (`dim` templates),
     Shewchuk's and Geogram's predicates (hand-written or generated per dimension, and why), Eigen
     fixed-size types in CUDA, and any GPU rasteriser or voxeliser that is dimension-generic
     (search; e.g. space-time / 4D simplex rasterisation, lattice-point enumeration in simplices).
   - Known pitfalls in device code: register pressure and spills from generic arrays, local-memory
     arrays when indexing is not resolved at compile time, `std::array` in device code needing
     `--expt-relaxed-constexpr` (`rasteriser-cpu.md`), compile time and binary size of
     `D × dtype × backend` instantiations.
5. **Measure it.** Build a throwaway `template <int D>` version of the procedures
   (`setup_tet`, `plane_values`, `owns`, `barycentric`, `row_span`) in the scratch directory, never
   in the repo. Then:
   - For `D = 3`: check it gives bit-identical records and owners to the current header on the adult
     atlas (`atlases/adult_template/atlas_adult.npz`, template grid). Compare against the current
     hand-written header: (a) the generated code (host assembly and device SASS: instruction count,
     registers, spills, local memory), (b) CPU time of the setup and rasterize loops (OpenMP, as in
     `cpu.cpp`), (c) GPU time of a minimal K3 kernel (setup + row-offset scan + row kernel) on the
     dev GPU (RTX PRO 1000 Blackwell laptop, `sm_120`, nvcc from `tetvox-dev`).
   - For `D = 2`: compare the generic instantiation against a hand-written 2D version (with the
     integer widths 2D allows) on a 2D triangle mesh of comparable size (e.g. a Delaunay
     triangulation of random points, or a Kuhn grid, on a 2048² or 4096² image).
   - If cheap, compile `D = 4` to see whether it still unrolls and what integer width it needs.
   Report the numbers with the method, so that they can be repeated.
6. **Scope beyond the rasteriser.** For each ADR 0009 op and for the public types (`MeshView`,
   `TetSetupView`, `GridShape`, `TetMesh`/`SamsegAtlas` in Python, shapes such as `(T, 4)`,
   `(T, 4, 3)`, `(X, Y, Z, 3)`), say what changes in a dimension-generic design: what is trivially
   generic (formulas in `d`), what needs per-`d` code, and the impact on the Python API (one
   `rasterize` for all `d` dispatching on shapes, or per-dimension names), the `.pyi` stub, tests
   (exhaustive exactly-once in 2D is cheap), docs and compile time.
7. **Use cases.** Who would use 2D (and ≥ 4D)? Look for concrete needs: 2D slice-based or histology
   registration with triangle-mesh atlases, whether samseg/GEMS (FreeSurfer repository) supports 2D
   meshes, mesh-based image registration in 2D, differentiable 2D rasterisation of triangle meshes
   to pixel grids (e.g. as a solid, watertight alternative to graphics rasterisers), 4D space-time.
   Keep it short; this is to weigh the benefit against the cost.

## Deliverable specifics
- A clear recommendation among at least: (a) stay 3D-only; (b) one fully generic `template <int D>`
  code path; (c) a generic interface and generic non-hot code with per-`D` specialisations of the
  hot procedures and per-`D` traits (integer widths, bit budget, traversal); (d) separate 2D and 3D
  implementations. State whether it is worth doing now, before the CUDA backend, and the cost of
  doing it later instead.
- The bit-budget table as a function of `d`, with the derivation.
- The measurements of question 5, with the method.
- A list of the decisions the owner would have to take if the recommendation is adopted (names,
  shapes, integer types per `d`, which `d` are instantiated, API placement), each with options. Do
  not take them.
- Cite everything. Mark each claim verified or inferred.
