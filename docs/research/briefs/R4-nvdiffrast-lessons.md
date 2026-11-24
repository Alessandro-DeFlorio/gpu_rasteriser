# Brief R4 — nvdiffrast in depth: what it does and what tetvox can inherit

Output: `docs/research/R4-nvdiffrast-lessons.md` (follow `docs/research/_template.md`).

## Context
tetvox is a watertight, differentiable voxeliser for tetrahedral meshes: every voxel centre is owned
by exactly one tet, with its barycentrics, so the mesh is a piecewise-linear interpolator,
differentiable in the vertex positions and per-vertex values. Main target: samseg brain atlases
(~340k tets, grids up to ~256³; `docs/knowledge/atlas-format.md`, `samseg-math.md`). The mesh
deforms every optimisation iteration, so rasterisation runs many times.

The owner's view: our use case is slightly different from nvdiffrast's, so we cannot use it
directly, but we could inherit its important parts and adapt them. Settled so far:
- R1 (`docs/research/R1-watertight-tet-rasterisation.md`) §1.1, §1.4, §4: nvdiffrast's op split
  (rasterise → `(id, barycentrics)` → interpolate → scatter), cudaraster's four stages and its
  fixed-point constants, the −1 bias fill rule. R1 chose K3 (one thread per tet box row, exact int64
  spans) over hierarchical binning.
- R2 (`docs/research/R2-decoupling-cost-from-rasteriser.md`): read nvdiffrast's `ops.py`,
  `interpolate.cu` and `rasterize.cu` at commit `253ac4f`; why its antialiasing op exists and why we
  do not need one in the interior. R2 lists the nvdiffrast paper (Laine et al. 2020) as not read.
- ADRs 0001 (nanobind + DLPack, no torch dependency), 0004 (C++ is a first-class interface), 0005–
  0009 (contracts, exact rule, CPU + CUDA backends sharing one header, op inventory). Read them, and
  `docs/architecture.md` and `docs/knowledge/rasteriser-cpu.md`. Do not repeat what R1 and R2
  already established; go deeper and cite those reports where they apply.

A parallel brief (R3) asks whether tetvox should become dimension-generic (2D triangles and 3D tets).
Note where nvdiffrast's design is inherently 2D and where it would carry over to a 2D tetvox path.

## Questions
1. **What nvdiffrast is now.** Latest release and commit, licence, supported frameworks (torch, and
   whether TensorFlow/JAX are still supported), how it is built and distributed (JIT via torch
   extensions or prebuilt), the two rasteriser backends (OpenGL and CUDA/cudaraster) and when each
   is used, and the public ops with their inputs, outputs, shapes and dtypes. Read the paper
   (Laine et al., "Modular Primitives for High-Performance Differentiable Rendering", SIGGRAPH Asia
   2020) and the documentation.
2. **The CUDA rasteriser in detail.** From the source (`csrc/common/cudaraster/`): triangle setup
   (clipping, culling, snapping, plane equations, what is stored per triangle), bin / coarse / fine
   stages, queue and memory management (how buffer sizes are chosen and grown, overflow handling),
   use of atomics, determinism, how large and tiny triangles are load-balanced, resolution and
   precision limits, depth handling, range and instanced modes, and how the output buffer is
   written. Relate each mechanism to our problem: no depth test, no blending, exactly one owner per
   voxel by construction, 3D grids, a mesh that changes every call, K3 as the current plan.
3. **Gradients and op structure.** How `rasterize`, `interpolate`, `texture` and `antialias` compute
   their backward passes: what each forward saves, scatter strategy (atomics, ordering,
   determinism), the barycentric screen-space derivatives (`rast_db`, `diff_attrs`), how gradients
   flow to vertex positions through barycentrics, and float precision choices. Compare with ADR
   0009's ops 7–12 (`interpolate`, `interpolate_vjp_values`, `interpolate_vjp_bary`, `tet_moments`,
   `vertex_grad`, `accumulate<Cost>`), and say whether nvdiffrast does anything we should copy or
   avoid (e.g. per-pixel vs per-triangle accumulation of position gradients).
4. **Engineering around the kernels.** Framework integration (torch custom ops, autograd functions,
   how it handles CUDA streams, devices, contiguity, dtypes), the GL/CUDA context object and its
   lifetime, buffer caching across calls, error handling and input validation, testing strategy,
   and packaging. Contrast with tetvox's choices (nanobind + DLPack, caller-allocated outputs,
   `out=`, a stream as the last parameter, validation once at the C++ boundary). Which of their
   choices would improve ours, and which conflict with ADRs 0001/0004?
5. **Licence and reuse.** What exactly does nvdiffrast's licence allow? Can code (or substantial
   adaptations of it) go into an MIT-licensed project? Are parts of it (e.g. the original
   cudaraster, released with Laine & Karras 2011) under a different licence? Conclude whether
   tetvox can copy code or only reimplement ideas, and what attribution either needs. Quote the
   licence text you rely on.
6. **What to inherit.** A ranked list of concrete ideas, each with: what nvdiffrast does (with file
   and line or paper section), how it would look in tetvox (which op, kernel or API), expected
   benefit, cost, and whether it changes an accepted ADR. Also list what we deliberately do not
   inherit and why. Include whether cudaraster's hierarchical binning could beat K3 for our tet
   size distribution (R1 §3.1, `atlas-format.md`), with an argument or a measurement.

## Deliverable specifics
- A clear answer to "can we use nvdiffrast, or which parts can we inherit", with the licence
  verdict up front.
- The ranked inherit / do-not-inherit list of question 6.
- Read the actual source, at a named commit, and cite files and lines. Mark each claim verified or
  inferred. Any throwaway code goes in the scratch directory, never in the repo.
