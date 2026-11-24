# 0005 — Scope, mesh contents and op contracts (owner answers to the R1/R2 open questions)

Status: accepted (2026-11-23); amended by ADR 0007 and ADR 0008 (stored π and mesh
positions/values are float32 or float64, following the caller's dtype at run time, not fixed
float32). Amended by ADR 0010: the contracts hold for meshes of dimension D = 1, 2, 3, with the
generic names (`SimplexMesh`, `simplex_id`, …) and D in place of 3 in every shape.

Context: `docs/research/R1-watertight-tet-rasterisation.md` and
`docs/research/R2-decoupling-cost-from-rasteriser.md` (including its addendum). This ADR records the
owner's answers. R1's tie-breaking rule and integer arithmetic are adopted in ADR 0006.

## Decision

**Scope.** The size is whatever the caller passes. Design for 256³ voxels with 8–16 classes (the
typical case), and keep 256³ × 44 channels working (memory-bound, fused path).

**Grid and affine.** The rasteriser works in voxel index space only: voxel centres are the integer
points of `[0, X) × [0, Y) × [0, Z)`, and the vertex positions are given in that space. Affines are
not part of tetvox. A caller that works in another space (warped, world, supersampled) transforms
the vertices once before calling, and applies the chain rule through that transform itself.

**Mesh contents.** The core mesh is vertices, tetrahedra and per-vertex values (`α`). Everything
else in a samseg atlas (affine, `can_move`, stiffness, prior normaliser, names, colours, labels)
belongs to the caller's model, not to the rasteriser's mesh type. In particular the deformation
prior and its Laplace normaliser (`log_mass_laplace_kappa1`) are a separate concern from
rasterisation.

**Frameworks.** torch gets a convenience `autograd` wrapper. Every other framework uses DLPack.
JAX is not a target.

**Domain boundary.** Half-open rule. A closed-box mode is an option, for tests and atlas building.

**Invalid meshes** (`det A ≤ 0`, or a vertex outside the exact-arithmetic window): never raise in
the evaluation path. Return `+inf` cost and report the number of offending tets, so a line search
can probe such configurations.

**Masked voxels.** Dense grid plus an optional mask. No compact voxel list and no tet-sorted list for
now.

**Raster buffer.** `tet_id` int32 plus 3 barycentrics `(π₁, π₂, π₃)` as float32, `π₀ = 1 − Σ`.
Whether 4 stored barycentrics or `tet_id` only (recomputing `π`) is faster stays an open benchmark.

**Curvature.** Gradient only for now. The accumulator is designed so that curvature can be added
without changing the cost interface (R2 addendum A4).

**Reproducibility.** Bitwise-reproducible sums are not required. Per-tet sums are float32 inside a
thread block and float64 across blocks and for the total cost (R2 addendum A5).

**Memory.** The caller allocates all outputs. The C++ API takes output pointers; the Python API takes
optional `out=` arrays and allocates when they are absent.

**Run-time-defined costs.** Supported through the generic path and through user kernels on the
raster buffer (R2 addendum A3, levels 1 and 2). No JIT now; the fused-cost concept must depend only
on the public headers, so that JIT stays possible.

## Consequences
- `TetMesh` is split: a minimal mesh (positions, tetrahedra, values) and a samseg atlas wrapper in
  Python for the rest. Not yet done (ROADMAP).
- Kernels that read per-voxel channel fields take a channel stride, so that both `(X, Y, Z, K)` and
  `(K, X, Y, Z)` work. Which layout is native is decided by benchmark (ROADMAP).
