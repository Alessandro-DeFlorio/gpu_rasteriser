# Brief R1 — Watertight solid rasterisation of tetrahedral meshes on the GPU

Output: `docs/research/R1-watertight-tet-rasterisation.md` (follow `docs/research/_template.md`).

## Context
tetvox voxelises a tetrahedral mesh onto a regular 3D voxel grid. Every voxel centre inside the mesh
must be assigned to **exactly one** tet, with its barycentric coordinates. The output is a
piecewise-linear interpolator (see `docs/knowledge/samseg-math.md`).

Typical meshes are samseg brain atlases:
- 34k–55k vertices and 211k–339k tets.
- Grids of roughly 141×169×144 to 181×217×181 voxels. **Read `docs/knowledge/atlas-format.md`
  first.** It holds the measured tet-size distribution: volume ratio up to 2.6e5, longest edge
  2–99 voxels, median ~45–96 voxel centres per bbox, max ~1.3e5, many slivers. It also records that
  the mesh tiles its bounding box exactly, with boundary faces lying *on* voxel-centre planes. Your
  tie-breaking rule must state what happens to samples on the outer domain boundary.
- Tet sizes range from much smaller than a voxel to tens of voxels across, because the atlas is
  adaptively coarsened.
- Vertices move during registration, so the mesh changes every optimisation iteration. Rasterisation
  runs many times, and any precomputation has to be cheap or incremental.

The owner wants:
1. A watertight result with a generalised top-left tie-breaking rule. No voxel is lost or
   double-counted on shared faces, edges or vertices.
2. Possibly fixed-point arithmetic for the inside tests, to make this exact.
3. A procedure that copes with both very large and very small tets.

## Questions
1. **Theory of watertight rasterisation, from 2D to 3D.**
   - Edge functions (Pineda 1988) and the top-left rule (D3D10/11 rasterisation rules).
   - How the rule generalises to 3D tets, so that each sample on a shared face, edge or vertex belongs
     to exactly one tet. Cover face orientation, lexicographic and "imaginary sun" tie-breaking, and
     consistency across the two tets sharing a face (the owner's reference: footnote 1 of
     `equations.pdf`).
   - Prior art on 3D/solid rasterisation and voxelisation: Schwarz & Seidel 2010 "Fast parallel
     surface and solid voxelization on GPUs"; Eisemann & Décoret; projected tetrahedra / volume
     rendering (Shirley–Tuchman, Weiler et al.); GPU point-in-tet location.
   - How samseg/FreeSurfer GEMS does it today (`kvlAtlasMeshRasterizor`,
     `kvlTetrahedronInteriorConstIterator`, or similar in the FreeSurfer GitHub repository).
2. **Exactness and fixed point.**
   - Fixed-point (integer) plane functions vs float with robust predicates (Shewchuk) vs Simulation
     of Simplicity (Edelsbrunner–Mücke) for degenerate cases.
   - Bit budget: grid extent (e.g. ≤ 2¹⁰ voxels per axis) × sub-voxel precision bits. What fits in
     int32 and int64 for the 3×3 orientation determinants and the plane-function evaluations?
     Should the plane equations be evaluated incrementally (adds only) or directly?
   - Is fixed point needed at all if samples are voxel centres and the plane coefficients are computed
     exactly from snapped vertices? Is snapping vertices to a sub-voxel grid acceptable for a
     differentiable interpolator? (Gradients would still use float barycentrics computed from the
     unsnapped positions. Discuss whether this mixed approach is consistent.)
   - How to obtain float barycentrics for the owning tet that stay accurate near degenerate
     (sliver) tets.
3. **Load balancing for tets of very different sizes.** Compare:
   - (a) tet-parallel bounding-box enumeration with size binning (thread / warp / block per tet,
     possibly splitting large tets into sub-boxes);
   - (b) hierarchical binning (coarse tiles/bricks, then fine rasterisation), as in hardware and
     software rasterisers (Laine & Karras 2011 "High-performance software rasterization on GPUs",
     nvdiffrast, cudaraster);
   - (c) voxel-parallel point location (uniform grid of tet lists, BVH, or a walk from the previous
     iteration's owner, which suits our case since vertices move a little per iteration);
   - (d) scanline / slab-based traversal of each tet (z-slices → 2D polygons → rows → spans).
   For each: work efficiency, memory, atomics or writes needed, and how well it fits a mesh that
   deforms every iteration. Also: what should the output layout be? A dense per-voxel
   `(tet_id, π)` grid, or a compact list?
4. **Differentiable rasterisers.** Briefly, what nvdiffrast, SoftRas, PyTorch3D and Kaolin's DMTet /
   tet-based voxelisers do. How do they treat gradients at ownership boundaries? Is anything
   directly reusable?

## Deliverable specifics
- A recommended rasterisation algorithm, written as pseudocode, with the tie-breaking rule stated
  precisely and argued for watertightness.
- A recommended number representation (bit widths), with the overflow argument.
- A ranked list of load-balancing strategies, and the experiment needed to choose between the top two.
- Cite everything. Mark each claim verified or inferred.
