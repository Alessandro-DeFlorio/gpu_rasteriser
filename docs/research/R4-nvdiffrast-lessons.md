# R4 — nvdiffrast in depth: what it does and what tetvox can inherit

Status: reviewed (owner, 2026-11-24): its decision list was answered in ADR 0010.
Brief: `briefs/R4-nvdiffrast-lessons.md`

Tags: **[V]** verified (read in the cited source at the named commit), **[M]** measured by a
throwaway script written for this report (adult atlas, template grid; "Method" at the end of
Findings), **[I]** inferred (my reasoning). Source citations are `file:lines` in nvdiffrast at
commit `253ac4f` (tag `v0.4.0`, 5 Dec 2025) [1] unless another reference is named.

## Question
1. What is nvdiffrast now (release, licence, frameworks, build, backends, ops)?
2. How does its CUDA rasteriser (cudaraster) work in detail, and how does each mechanism relate to
   our problem (no depth, no blending, one owner per voxel, 3D, mesh changes every call, K3)?
3. How do its ops compute gradients, and how does that compare with ADR 0009's ops 7–12?
4. How is it engineered around the kernels, and which choices would improve or conflict with ours?
5. What does its licence allow? Can code go into an MIT project?
6. What should tetvox inherit, ranked, and what not? Could hierarchical binning beat K3?

## Recommendation

**Licence verdict first: copy no code from nvdiffrast. Reimplement ideas only.** nvdiffrast is
under the "Nvidia Source Code License (1-Way Commercial)": redistribution only under that licence,
and derivative works must keep its non-commercial use limitation. That cannot be combined with MIT
(§5). The 2012 release of the original cudaraster is New BSD and could legally be reused with its
notice, but it is a 2D depth-buffered pipeline written for Fermi-era warp semantics; nothing in it
is worth copying. Ideas and algorithms are free to reimplement; cite the two papers where we do.

**We cannot use nvdiffrast itself**, in 3D or in a 2D path: 4 sub-pixel bits, barycentrics computed
from unsnapped float positions and then clamped, a float32-packed id, clip-space/depth machinery, a
hard torch dependency, and the licence (§1, §2, §5).

What to inherit, in order (details and line references in §6):

1. **Sync-free work distribution with persistent threads and a device-side counter.** K3 as
   specified ("scan the row counts, then one thread per row") needs the total row count on the
   host to size its launch (inferred; no ADR says how the total is obtained). That is a
   device-to-host readback and a stream sync inside `locate`/`rasterize`, while ADR 0009 states
   that only `count_skipped` synchronises. nvdiffrast's antialias op and cudaraster show the
   pattern that avoids it: launch a fixed number of blocks, let each block take chunks of work
   with one `atomicAdd`, and keep the work count on the device. Build the CUDA K3 launcher this
   way. The variant that takes *tets* in batches and scans row counts in shared memory also
   removes the global scan, the `cudaMallocAsync` temporary and the binary search over `T`.
2. **Coalesced atomics for every scatter** (`tet_moments`, `interpolate_vjp_values`,
   `vertex_grad`). Group the lanes of a warp by tet id, sum inside the group, and issue one global
   atomic per group. On the adult atlas this cuts the global atomics of a 12-float moment set from
   84.0 M (per voxel) to 6.5 M (per warp-and-tet group of rows). It is also the natural
   implementation of ADR 0005's "float32 inside a block, float64 across blocks".
3. **Redistribute per-voxel work inside a block before running an expensive body.** In plain K3
   each lane loops over its own row, so a warp runs until its longest row ends. Measured lane
   utilisation is 20.7 %. Pooling the rows of a block and dealing voxels evenly to its threads (the
   cudaraster fine-stage scheme) gives 53–57 %; a global voxel-parallel second pass gives 100 %.
   This matters little for `rasterize` (cheap body) and a lot for `accumulate<Cost>`.
4. **Reject empty rows without dividing.** 69.9 % of K3 rows own no voxel. cudaraster's
   trivial-reject corner test, applied to a row, removes 57 % of those with two multiplies and a
   compare per face, and cuts the int64 divisions by 40 %. It changes no result.
5. **Skip voxels whose incoming gradient is exactly zero** in the backward ops (nvdiffrast's bit
   test). Free, and 10.5 % of the atlas volume has an exactly zero data gradient.
6. **Keep what we already do differently**: geometry-free per-tet moments instead of nvdiffrast's
   per-pixel position gradients; one snapped geometry for ownership, π and `∂π/∂v`; validation once
   at the boundary; exact sizes computed before launch instead of grow-and-retry; no context
   object; no framework linkage. nvdiffrast's source confirms each of these as the better choice
   for us (§3, §4).

**Do not build cudaraster's bin/coarse/fine hierarchy.** With exact conservative culling the adult
atlas gives 1.76 M tet–brick pairs for 4³ bricks, and the fine stage would process 15.3 M row
pieces against K3's 10.6 M whole rows, before any binning cost. Its 2D coverage look-up table has no
3D counterpart at our precision (§6.3). This confirms R1 §3.3(b) with counts; it is still not a GPU
timing.

Items 1 and 4 touch accepted ADRs (the wording "no atomics" in ADR 0006, the async statement and
the `cudaMallocAsync` plan in ADR 0009, the shared `row_span` procedure). They are listed as owner
decisions in "Impact on design"; none is taken here.

## Findings

### 1. What nvdiffrast is now

**Release and history.**
- Latest release and head of `main`: `v0.4.0`, commit `253ac4f`, 5 Dec 2025. `git ls-remote` on
  2026-11-24 shows no newer commit. **[V]** [1]
- v0.4.0 changed three things: the JIT build was replaced by an ahead-of-time build (commit
  `45e2e96`), the OpenGL rasteriser was removed (`96a79e0`, 1,564 lines deleted), and TensorFlow 1.x
  support was removed (`31e06ba`). **[V]** [1] (`git log`, `git show --stat`)
- The CUDA rasteriser was added in v0.3.0 (commit `a1ec436`, Aug 2022). Resolutions above 2048² came
  in v0.3.3 by tiling the viewport. **[V]** [1] (`git log`; v0.3.5 docs)
- R2 read the same commit. Its statement that nvdiffrast has four ops stands; the paper is read here
  for the first time.

**Frameworks.** PyTorch only. The README calls it "a PyTorch library". The package contains
`nvdiffrast.torch` and nothing else. There was never a JAX binding. **[V]** [1] (`README.md`,
`pyproject.toml`, `nvdiffrast/torch/__init__.py`) The 2020 paper describes both TensorFlow and
PyTorch. **[V]** [2] §3.1

**Build and distribution.**
- One pybind11 extension `_nvdiffrast_c`, compiled at install time with
  `torch.utils.cpp_extension.CUDAExtension` (`setup.py:26-55`). It links against the installed
  torch, so the install needs `pip install ... --no-build-isolation`. **[V]** [1] (`setup.py`,
  docs "Installation")
- Before v0.4.0 the plugin was compiled at first use with `torch.utils.cpp_extension.load`
  (`ops.py:124` at tag `v0.3.5`). **[V]** [1]
- No prebuilt wheels: the documented install is from GitHub, and `pypi.org/pypi/nvdiffrast/json`
  returns 404 (checked 2026-11-24). **[V]**
- GPU architectures come from torch's `TORCH_CUDA_ARCH_LIST`; by default only the installed GPU is
  targeted (docs, "Install into Docker container"). **[V]** [1]
- Requirements stated in the docs: Python ≥ 3.9, PyTorch ≥ 2.7, a CUDA toolkit matching torch.
  **[V]** [1]

**Rasteriser backends.** Since v0.4.0 only the CUDA rasteriser exists. `RasterizeGLContext` is a
deprecated stub that creates a `RasterizeCudaContext` (`ops.py:550-556`). **[V]** [1] The v0.3.5
docs explain the earlier choice: OpenGL uses the hardware pipeline and is likely faster for complex
meshes at high resolution but is hard to set up on Linux clusters; the CUDA rasteriser works
wherever CUDA does, with "subpixel precision ... limited to 4 bits" and an internal viewport capped
at 2048². They also state its origin: "Our code is based on the paper's publicly released CUDA
kernels, with considerable modifications". **[V]** [1] (docs at `v0.3.5`) The paper's rasteriser
was OpenGL only, chosen because "the hardware graphics pipeline ensures that the rasterization is
accurate and there are, e.g., no visibility leaks due to precision issues". **[V]** [2] §3.3

**Public ops** (all tensors float32 unless noted, contiguous, on one CUDA device; a leading
minibatch axis `B`). **[V]** [1] (`ops.py` docstrings, docs "API reference")

| Op | Inputs | Outputs |
|---|---|---|
| `RasterizeCudaContext(device)` | — | context holding cudaraster's buffers |
| `rasterize(ctx, pos, tri, resolution, ranges=None, grad_db=True)` | `pos` `[N,4]` or `[B,N,4]` clip space; `tri` `[T,3]` int32; `ranges` `[B,2]` int32 on the CPU | `rast` `[B,H,W,4]` = `(u, v, z/w, triangle_id + 1)`, 0 where empty; `rast_db` `[B,H,W,4]` = `(du/dX, du/dY, dv/dX, dv/dY)` |
| `DepthPeeler(...).rasterize_next_layer()` | as `rasterize` | next depth layer |
| `interpolate(attr, rast, tri, rast_db=None, diff_attrs=None)` | `attr` `[N,A]` or `[B,N,A]` | `[B,H,W,A]`, and `[B,H,W,2·D]` attribute pixel derivatives |
| `texture(tex, uv, uv_da, mip_level_bias, mip, filter_mode, boundary_mode, max_mip_level)` | `tex` `[B,h,w,C]` or cube `[B,6,h,w,C]`; `uv` `[B,H,W,2 or 3]` | `[B,H,W,C]` |
| `texture_construct_mip(tex)` | | opaque mip stack, reusable across calls |
| `antialias(color, rast, pos, tri, topology_hash, pos_gradient_boost)` | `color` `[B,H,W,C]` | `[B,H,W,C]` |
| `antialias_construct_topology_hash(tri)` | | opaque edge hash, reusable across calls |

- "Range mode" renders a different triangle range per image; "instanced mode" renders one topology
  with `B` position sets. **[V]** [1] (docs)
- The first vertex of a triangle has `(u, v) = (1, 0)`, the second `(0, 1)`, the third `(0, 0)`.
  `z/w` and the id carry no gradient. **[V]** [1] (docs "Rasterization")
- The design rules of the paper: modular primitives with a fixed interface (C1), positions given in
  clip space with all transforms left to the user (C3), deferred shading on image-space buffers so
  that shading is user code (C4). **[V]** [2] §3.2 These are the rules tetvox already follows
  (voxel index space, affines left to the caller, op split; ADR 0005, R2 §5). **[I]**
- The paper's reason for separating interpolation from rasterisation: "one can supply a different
  index buffer for attribute interpolation than was used for rasterization". **[V]** [2] §3.4,
  footnote 3 Our `interpolate(tet_id, bary, tetrahedra, values)` has the same freedom. **[I]**

### 2. The CUDA rasteriser in detail

`csrc/common/cudaraster/` is "a slimmed-down and modernized version of the original CudaRaster
codebase ... Modifications have been made to accommodate post-Volta execution model with warp
divergence. Support for shading, blending, quad rendering, and supersampling have been removed"
(`CudaRaster.hpp:12-17`). **[V]** The pipeline is setup → bin → coarse → fine, then a per-pixel
"shader" kernel in nvdiffrast proper. R1 §1.4 has the four stages and constants; what follows is
new.

#### 2.1 Triangle setup (`impl/TriangleSetup.inl`)
- One thread per triangle; output slot = input index, so order is kept without compaction
  (`:181-233`). **[V]**
- **Index checks in the kernel.** A triangle or vertex index out of range marks the triangle as
  having zero sub-triangles (`:228-248`). There is no host-side validation of indices. **[V]**
- **Frustum cull, then clip.** A triangle with all vertices outside one clip plane is dropped
  (`:269-283`). A triangle fully inside the frustum takes the fast path (`:329-352`). Otherwise it
  is clipped against the six frustum planes into a fan of up to 7 sub-triangles (a 9-vertex
  buffer, `:183`), whose vertices are computed in float from barycentrics and snapped again
  (`:355-434`; `Util.inl:101-160`). Space for more than one sub-triangle comes from a
  global `atomicAdd` on `numSubtris` (`:398-404`). **[V]**
- The disabled fast path carries this comment: "This test is a little sloppy and may result in a
  shared edge being clipped in one triangle and not in another. This increases the potential of
  pixel leaks at shared edges." (`:285-287`). **[V]** So clipping is where cudaraster's
  watertightness is weakest: a clipped shared edge gets new vertices, snapped per triangle. **[I]**
  tetvox never clips geometry; it clips only the integer box to the grid (ADR 0009 "unclipped
  origin"), so this failure mode does not exist. **[I]**
- **Snapping.** `p = cvt.rni.sat(v.x / v.w · W · 2³)`: round to nearest, saturating, 4 sub-pixel
  bits, coordinates relative to the viewport centre (`:11-24`; `Util.inl:31`;
  `Constants.hpp:13-14`). **[V]** Same rounding mode as ADR 0007 (nearest, ties to even). **[I]**
- **Culling of triangles that cannot cover a sample** (`prepareTriangle`, `:42-116`): zero area;
  negative area with back-face culling (nvdiffrast disables it and swaps two vertices instead,
  `:127-137`); bounding box falls between sample points; box holds one or two samples and neither
  is covered (tested with exact int32 edge functions). **[V]** The cudaraster paper: "This ensures
  that for densely tessellated surfaces, we output at most one triangle per sample". **[V]** [3]
  §5.1
- **Stored per triangle**: a 16-byte header (three snapped vertices as S16 pairs, and a word with
  20 bits of minimum depth and 3 × 4 "flip bits" that encode each edge's octant and its tie-break)
  and 16 bytes of data (a fixed-point depth plane and the triangle id) (`PrivateDefs.hpp:19-39`;
  `:160-176`). **[V]** No edge-function coefficients are stored; later stages recompute them from
  the S16 vertices. **[V]** (`BinRaster.inl:202-207`, `FineRaster.inl:80-97`) Ours is 197 B per tet
  because the int64 normals are stored; cudaraster's choice is possible only because 2D int32
  products are cheap to redo. **[I]**
- **Tie-break.** `if (dy > 0 || (dy == 0 && dx <= 0))` selects the complement flip, i.e. the
  exclusive side (`:28-38`; reference form `Util.inl:304-309`). R1 §1.1 already cites it. **[V]**

#### 2.2 Bin and coarse stages (`impl/BinRaster.inl`, `impl/CoarseRaster.inl`)
- **Bin**: 16 blocks of 512 threads (`Constants.hpp:26,35`; `RasterImpl.cpp:360`). Each block takes
  a batch of triangles with one global `atomicAdd` (`BinRaster.inl:56-66`), compacts away culled
  ones into a ring buffer, finds the 128×128-pixel bins each triangle touches, and appends the
  triangle to *its own* per-bin queue. **[V]** The reason, from the paper: "every CTA writes to its
  own set of per-bin output queues", which avoids synchronisation between blocks while each queue
  stays in input order. **[V]** [3] §4.2
- **Three paths by bounding-box size** in the bin stage: all triangles of the warp in one bin
  (`__match_any_sync`), at most 2×2 bins (four `atomicOr`), general case with conservative edge
  functions over the box (`BinRaster.inl:196-267`). The coarse stage has the same split: boxes of at
  most 2×2 tiles are recorded whole (case A), otherwise tiles are tested with edge functions biased
  to the trivial-reject corner (cases B and C, `CoarseRaster.inl:274-378`). **[V]** This is how tiny
  and large triangles are handled differently: tiny ones skip the edge tests.
- **Coarse**: blocks first sort the bins by descending triangle count in shared memory, then take
  bins one at a time with a global atomic (`CoarseRaster.inl:78-103`), merge the 16 input queues in
  triangle order (`:137-218`) and write per-tile (8×8 pixels) queues. **[V]** The paper adds that
  writes are spread evenly: "we calculate the total number of writes the CTA has to perform, and
  distribute these write tasks evenly to all 512 threads ... each of these is implemented as a
  binary search over a small domain". **[V]** [3] §5.3
- **Queues**: linked segments (512 entries per bin segment, 32 per tile segment,
  `Constants.hpp:27-28`) allocated with one global `atomicAdd` each
  (`BinRaster.inl:311-341`, `CoarseRaster.inl:476`). **[V]** "each allocation unavoidably requires
  one globally atomic operation". **[V]** [3] §3.3

#### 2.3 Fine stage (`impl/FineRaster.inl`)
- `numSMs × occupancy` blocks of up to 20 warps run as persistent threads; each *warp* takes one
  active tile with `atomicAdd(&atomics.fineCounter, 1)` and owns it exclusively (`:211-219`;
  `RasterImpl.cpp:63-68,366`). **[V]**
- The 8×8 tile lives in shared memory (colour = triangle id, and depth); it is cleared or read at
  the start and written back at the end (`:234-250`, `:370-381`). **[V]**
- **Coverage.** 32 lanes read 32 triangles of the tile's queue. Each computes a 64-bit coverage
  mask as the AND of three per-edge masks from a 6 KB look-up table (`:75-101`;
  `Util.inl:248-300`). The table lookup is documented as "Assumes viewport <= 2^11, subpixels <=
  2^4, no guardband" (`Util.inl:85`) and computes its index in float32. **[V]** The paper: 51
  instructions per edge. **[V]** [3] §5.4
- **Fragment distribution.** Masks are pop-counted, a warp prefix sum gives each triangle its
  fragment offset, empty triangles are compacted away, and then *each lane takes one fragment*:
  it finds its triangle with a ballot and its pixel with `findBit` (`:285-338`). **[V]** So the
  per-fragment work is balanced over lanes whatever the triangle sizes. **[I]**
- **Depth.** `depth = zx·X + zy·Y + zb` in U32, early-z against the tile maximum, then `atomicMin`
  on the shared-memory depth and a lane-id `atomicMax` to pick one winner among equal depths
  (`:152-172`, `:344-361`). **[V]** Not needed by us: there is no depth, and one owner per voxel by
  construction means no write conflict to resolve. **[I]**

#### 2.4 Memory management, overflow, synchronisation (`impl/RasterImpl.cpp`, `impl/Buffer.cpp`)
- The context owns all buffers and keeps them between calls. `Buffer::grow` reallocates only when
  more is needed (`Buffer.cpp:31-52`). **[V]**
- Sizes are worst-case guesses plus slack: `triCount + 4096` sub-triangles, bin and tile segments
  from the triangle count and the number of bins/tiles (`RasterImpl.cpp:138-169`). **[V]**
- **Overflow is found after the fact.** Every stage returns early if a counter exceeds its buffer
  (`BinRaster.inl:36-37`, `CoarseRaster.inl:66-67`, `FineRaster.inl:203-204`). The host copies the
  counters back, and if any overflowed it enlarges the buffers and *runs all stages again*
  (`RasterImpl.cpp:172-231`). At 2²⁴ sub-triangles it gives up and the op raises "subtriangle
  count overflow" (`:220-221`; `torch_rasterize.cpp:123`). **[V]**
- **Every rasterisation blocks.** `launchStages` ends with `cudaStreamSynchronize(stream)`
  (`RasterImpl.cpp:367`), after an async copy of the counters to pinned host memory (`:362`).
  **[V]**
- Launch sizes come from device queries: `cudaDevAttrMultiProcessorCount` and
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (`RasterImpl.cpp:59-68`). **[V]**

#### 2.5 Atomics and determinism
- Global atomics are used only for work counters and segment allocation (seven counters per image,
  `PrivateDefs.hpp:45-61`). Per-pixel conflicts are resolved inside one warp in shared memory.
  **[V]**
- The forward result is deterministic by design: queues stay in input order, and the paper states
  "we strictly enforce the rendering order ... producing deterministic output which is important
  for verification purposes". **[V]** [3] §1 Which block processes which batch is not
  deterministic, but the output does not depend on it. **[I]**
- The backward passes use float32 `atomicAdd` (§3), so the summation order, and with it the last
  bits of the gradient, can change between runs. **[I]** (the atomics are **[V]**; nvdiffrast makes
  no statement on reproducibility)

#### 2.6 Limits
- 4 sub-pixel bits, 2048² internal viewport, S16 vertex coordinates, int32 edge functions
  (`Constants.hpp:13-14`; `PrivateDefs.hpp:19-29`). R1 §1.4. **[V]**
- Larger outputs are rendered as several viewport tiles; each tile runs all four stages over all
  triangles (`torch_rasterize.cpp:98-124`). **[V]** The limit exists because coordinates are stored
  relative to the viewport centre in 16 bits. tetvox's per-tet origin (ADR 0008) removes the
  equivalent limit without tiling. **[I]**
- At most 2²⁴ sub-triangles per image (`Constants.hpp:30`). Triangle ids are stored as float32;
  above 2²⁴ a bit-cast encoding is used (`common.h:186-193`). **[V]**

#### 2.7 How the output buffer is written
- cudaraster's result is an internal U32 buffer of `triangle index + 1` (plus depth), padded to
  multiples of 8 pixels (`RasterImpl.cpp:87-99`). **[V]**
- A separate per-pixel kernel, `RasterizeCudaFwdShaderKernel`, then reads the id, fetches the three
  *unsnapped* float32 clip-space positions, evaluates float edge functions at the pixel, normalises
  them to `(u, v)`, and **clamps**: `__saturatef` on each and a rescale so that `u + v ≤ 1`
  (`rasterize.cu:28-94`; comment "Clamps to avoid NaNs"). It also writes `rast_db`
  (`:96-113`). **[V]**
- So nvdiffrast's CUDA path is exactly `barycentrics ∘ locate` in ADR 0009's terms: an id-only
  rasteriser followed by a per-pixel barycentric kernel. **[I]**
- It mixes two geometries: ownership from positions snapped to 1/16 pixel, barycentrics from
  unsnapped floats, reconciled by clamping. R1 §2.4 rejected this mix for tetvox, and the earlier
  `meshing_cxx` did the same thing (`meshing-cxx-reference.md`). nvdiffrast can afford it because a
  clamped barycentric only shifts a shading sample; in `−log Σ l_j π_j` it would bias the cost.
  **[I]**

#### 2.8 Range and instanced modes
- Range mode: setup runs one thread per triangle of all ranges and finds its image by a linear walk
  over the ranges (`TriangleSetup.inl:196-208`). Instanced mode: the launch has one grid layer per
  image and offsets the vertex buffer by `numVertices · image` (`:190-195`, `:253-254`;
  `RasterImpl.cpp:348-359`). All later stages use `blockIdx.z` as the image index. **[V]**
- tetvox has no batch axis. An instanced analogue (several position sets on one topology, for
  example the trial points of a line search) would be a leading axis on `positions` and on the
  record. Nothing in ADR 0009 prevents adding it later. **[I]**

#### 2.9 What is inherently 2D, and what would carry to a 2D tetvox path (for R3)
- Inherently 2D or rendering-specific: the 8×8 coverage table (an edge through a tile is a
  y-intercept plus a 7-bit step string; a plane through a 4³ brick has no such short code), the
  int32/S16 bit budget (degree-2 edge functions), depth and peeling, clip space and perspective,
  `rast_db` for mip selection, the antialias op. **[I]** from [1], [3] §5.4
- Dimension-independent: the op split, the `(id, barycentrics)` buffer, the interpolation formulas
  and their VJPs, coalesced atomics, persistent threads with atomic work counters, block-local scan
  and redistribution, the conservative corner test. **[I]**
- A 2D tetvox path could not call nvdiffrast either: 4 sub-pixel bits against our k = 14, clamped
  float barycentrics, clipping, and the licence. Whether a 2D path should use rows (K3) or tiles
  with a coverage mask is R3's question. Note only that cudaraster's table is exact because of its
  small bit budget; at k = 14 the table index would need exact int64 arithmetic. **[I]**

### 3. Gradients and op structure

#### 3.1 What each forward saves
All four ops are `torch.autograd.Function`s that save their *inputs and outputs*, no hidden state:
`rasterize` saves `(pos, tri, out)` (`ops.py:79`), `interpolate` saves `(attr, rast, tri[,
rast_db])` (`:215,231`), `texture` saves its inputs and the mip stack (`:309,329`), `antialias`
saves `(color, rast, pos, tri)` plus a work buffer (`:475-476`). **[V]** The antialias work buffer
is the one true saved intermediate: the forward pass stores, per detected discontinuity, the pixel,
the edge and the blend weight, "so that we do not have to repeat it in the backward pass". **[V]**
[2] §3.6; `antialias.cu:373-378,406-410`

#### 3.2 `rasterize` backward (`rasterize.cu:119-273`)
- One thread per pixel. It reads only two numbers of upstream gradient per pixel,
  `dy = ∂L/∂(u, v)`, and the triangle id from the saved output (`:136-138`). **[V]**
- **It recomputes the geometry per pixel**: fetches the three positions, re-evaluates the float edge
  functions, and forms the nine position-gradient components `(x, y, w)` of the three vertices
  (`:172-211`). It does not read the stored `(u, v)`. **[V]**
- An epsilon is added to the triangle area here, `ep = copysignf(1e-6f, at)`, but not in the
  forward kernel (`:190-193` against `:76`). **[V]**
- Pixels whose incoming gradients are all ±0 exit early by a bit test (`:143-148`). **[V]**
- **Scatter**: nine `atomicAdd`s per pixel into the `[N,4]` gradient, coalesced per triangle inside
  a warp (`:169-170`, `:269-272`). **[V]**
- With `grad_db=True` it also back-propagates through `rast_db`, about 50 more lines of algebra
  (`:213-267`). The docs say these gradients "are often not crucial" and can be disabled. **[V]**

#### 3.3 Coalesced atomics (`common.h:196-241`)
`CA_SET_GROUP(triIdx)` calls `__match_any_sync` to find the lanes of the warp that hold the same
triangle and elects the lowest as leader. `caAtomicAdd(ptr, v)` then adds `v` into the leader's
shared-memory slot and the leader issues one global `atomicAdd`. It needs compute capability 7.0;
below that the macro is a plain `atomicAdd`. **[V]** tetvox's toolchain floor is `sm_75`
(`build.md`), so the intrinsic is always available. **[I]**

#### 3.4 `interpolate` (`interpolate.cu`)
- Forward: `out[i] = b0·a0[i] + b1·a1[i] + b2·a2[i]` with `b2 = 1 − u − v`, channels last, one
  thread per pixel looping over channels; a warp with no geometry zeroes its outputs and exits
  (`:37-47`, `:73-80`). **[V]**
- Backward: **one kernel produces both gradients.** Per channel it accumulates
  `gb0 += y·(s0 − s2)`, `gb1 += y·(s1 − s2)` and scatter-adds `b_k·y` to the three vertices with
  coalesced atomics (`:198-210`), then writes `(gb0, gb1, 0, 0)` per pixel (`:212-213`). **[V]**
  R2 §5 identified `gb` as the reduced `ḡ`. The gradient array is `zeros_like(attr)`, the
  per-pixel one `empty_like(rast)` (`torch_interpolate.cpp:211-212`). **[V]**
- `rast_db` and `diff_attrs` serve one purpose: screen-space derivatives of texture coordinates for
  mip-level selection. **[V]** [2] §3.4–3.5; docs "Image-space derivatives" The 3D counterpart,
  `∂π_j/∂x = p_j`, is constant per tet and already in the record (`normals · bary_scale`, ADR 0009),
  so tetvox needs no per-voxel derivative buffer. **[I]**

#### 3.5 `texture` and `antialias`
- `texture` backward scatter-adds into the texture and the mip levels with atomics coalesced per
  texel (`texture_kernel.cu:627-637,992`; `common.h:237-241`), then a second kernel folds the mip
  pyramid's gradients back to the base level (`torch_texture.cpp:678-686`). **[V]** Not relevant to
  tetvox beyond the scatter pattern. **[I]**
- `antialias` is three kernels: build an edge hash from the topology with `atomicCAS`
  (`antialias.cu:82-96,140-160`, cacheable across calls); find pixel pairs with different ids and
  append them to a work list whose counter update is coalesced to once per block (`:165-214`);
  analyse the list with persistent threads (`:219-232`). The work list is sized for the worst case,
  two items per pixel, so its length never has to reach the host (`torch_antialias.cpp:123-151`).
  The gradient kernel walks the same list (`antialias.cu:387-410`). **[V]** R2 §4 explains why
  tetvox needs no such op. Its *engineering* is the useful part (§6, item 1).

#### 3.6 Comparison with ADR 0009's ops 7–12

| nvdiffrast | tetvox (ADR 0009) | Difference |
|---|---|---|
| cudaraster → id buffer | 3 `locate` | exact SoS rule at 2⁻¹⁴ voxel against top-left at 1/16 pixel with depth |
| shader kernel → `(u, v)` from unsnapped positions, clamped | 4 `barycentrics` (5 `rasterize` fused) | ours: `π_j = (t_j + f_j)·2ᵏ/D` from the exact integers, non-negative by construction |
| `rast_db` buffer | none; `p_j` per tet in the record | 3D derivative is per-tet constant |
| `interpolate` forward | 7 `interpolate` | same formula, one more vertex |
| `interpolate` backward, one kernel, two outputs | 8 `interpolate_vjp_values`, 9 `interpolate_vjp_bary` | ours are two ops; 9 reads no π |
| `rasterize` backward: per pixel, geometry recomputed, 9 atomics to vertices | 10 `tet_moments` (geometry-free, per voxel) + 11 `vertex_grad` (geometry once per tet) | see below |
| `texture`, `antialias`, depth peeling | none | not our problem (R2 §4) |
| none: everything is materialised ("deferred shading", [2] C4) | 12 `accumulate<Cost>` | nvdiffrast has no fused path; R2's memory argument is specific to K = 44 |

**Per-pixel or per-primitive accumulation of position gradients?** Keep ADR 0009. **[I]**, with
these reasons:
- nvdiffrast's backward gathers `tri` (12 B) and three positions (48 B) per pixel and redoes the
  edge functions. The tetvox equivalent would gather the tet record per voxel (197 B, int64
  normals) or recompute a 3×3 inverse per voxel. `tet_moments` reads only `tet_id`, π and `ḡ`.
- The number of accumulators is the same either way: 4 × 3 moments per tet, or 4 × 3 gradient
  components to its vertices. Applying `A⁻ᵀ` after the sum instead of before costs nothing extra
  and removes the geometry from the voxel loop (R2 eq. 3).
- Per-tet targets coalesce perfectly by tet id. Per-vertex targets are shared by about 25 tets
  each (`4T/N` = 24.8 on the adult atlas), so they collide across groups.
- nvdiffrast's forward and backward use slightly different geometry (the epsilon). Ours share the
  record.

**Things nvdiffrast does that we should copy:** coalesced atomics, the zero-gradient early exit,
explicit reusable objects for anything derived from static topology. **Things to avoid:** the
per-pixel geometry recomputation, clamped barycentrics, the area epsilon, float-packed ids, the
gradient through a derivative buffer. Details in §6.

**Float precision.** nvdiffrast is float32 throughout (`torch_common.inl:23-28` rejects anything
else); the rasteriser is int32/S16 fixed point with a U32 depth; gradients are accumulated in
float32 with atomics. There is no float64 anywhere. **[V]**

### 4. Engineering around the kernels

| Aspect | nvdiffrast | tetvox (ADRs) | Verdict |
|---|---|---|---|
| Framework | hard torch dependency; pybind11 module built against the installed torch (`setup.py`, `torch_bindings.cpp:43-71`) **[V]** | nanobind + DLPack, torch optional (ADR 0001) | keep ours; theirs is why there are no wheels and why TF was dropped |
| C++ API | none. Kernel parameter structs are internal; the only entry points take `torch::Tensor` **[V]** | first-class C++ API (ADR 0004) | keep ours |
| Device | `OptionalCUDAGuard(device_of(pos))`; all inputs must be on one device (`torch_rasterize.cpp:45,50`) **[V]** | backend from the DLPack device (ADR 0007) | equivalent |
| Stream | `at::cuda::getCurrentCUDAStream()` in every op (`:46`) **[V]** | explicit trailing `cudaStream_t` (ADR 0009); Python handling open | see below |
| Sync | every forward rasterisation synchronises the stream (`RasterImpl.cpp:367`); other ops are asynchronous **[V]** | only `count_skipped` synchronises (ADR 0009) | ours is stricter; needs item 1 of §6 to hold |
| Outputs | allocated inside C++ with `torch::empty` / `zeros_like` (`torch_rasterize.cpp:126-129,237`) **[V]** | caller-allocated, `out=` (ADR 0005) | keep ours; theirs relies on torch's caching allocator |
| Contiguity, dtype | inputs must be contiguous float32/int32, else an error; upstream gradients are made contiguous silently (`:52-54,216`) **[V]** | reject, never cast or copy (ADR 0007/0008) | equivalent, ours without the silent copy |
| Alignment | kernels cast buffers to `float4*`/`float2*`, so ops check 16-byte alignment and fail otherwise (`:151-154,240-243`) **[V]** | scalar loads | avoid theirs: a DLPack array may start at any offset |
| Validation | shape, dtype, device checks with `TORCH_CHECK` at the op boundary (`torch_common.inl:20-28`); **index validity is checked per pixel in every kernel** (`rasterize.cu:30,43-47,141,155-159`; `interpolate.cu:35,54-58,150,163-167`) **[V]** | once, in a pre-pass at the C++ boundary (ADR 0008) | keep ours |
| State | a context object per device holds cudaraster's buffers; setters then `drawTriangles`; a depth peeler reserves it (`ops.py:47-68,141-204`; `CudaRaster.hpp:42-51`) **[V]** | no state; `TetSetup` is an explicit caller-owned intermediate (ADR 0009) | keep ours |
| Caching across calls | internal buffers grow and persist; mip stacks and the topology hash can be built once and passed in, otherwise "constructed internally and discarded" (`ops.py:442-465,529-544`) **[V]** | `out=` reuse of `TetSetup` and outputs | same idea, already adopted |
| Errors | exceptions from `TORCH_CHECK`; CUDA errors checked at each launch (`framework.h:24-25`) **[V]** | C++ exceptions → Python (ADR 0004) | equivalent |
| Tests | none in the repository; five sample scripts, of which `triangle.py` is the suggested smoke test **[V]** [1] (file list, docs "Samples") | exhaustive exactly-once tests, exact reference (ADR 0006/0008) | keep ours |
| Packaging | sdist only, built on the user's machine against their torch **[V]** | wheels planned, static cudart (ADR 0001) | keep ours |

- A sign of the missing tests: `rasterize` *returns* instead of raising a `RuntimeError` when
  called during depth peeling (`ops.py:131-132`). **[V]**
- **Streams from Python.** nvdiffrast can ask torch for its current stream because it links torch.
  tetvox cannot (ADR 0001). The equivalents without linking are: the optional torch wrapper passes
  `torch.cuda.current_stream().cuda_stream` to the binding; a cupy caller passes
  `cupy.cuda.get_current_stream().ptr`; plain DLPack callers get the default stream or pass one.
  **[I]** (API names from memory, not re-read for this report; to be settled in the CUDA item as
  the ROADMAP says)
- **No choice of nvdiffrast's conflicts usefully with ADR 0001/0004 in our favour.** Adopting its
  integration style (torch tensors in C++, torch allocation, current-stream lookup) would break
  both ADRs for no gain. **[I]**
- **What would improve ours:** only patterns inside the kernels and launchers (§6), plus two
  documentation habits: every op docstring states device, contiguity, dtype and shape of each
  argument, and the docs explain the coordinate conventions in one place. **[I]**

### 5. Licence and reuse

**nvdiffrast** (`LICENSE.txt`, "Nvidia Source Code License (1-Way Commercial)"). **[V]** [1] The
clauses relied on:
- Grant, §2.1: "each Licensor grants to you a perpetual, worldwide, non-exclusive, royalty-free,
  copyright license to reproduce, prepare derivative works of, publicly display, publicly perform,
  sublicense and distribute its Work and any resulting derivative works in any form."
- §3.1: "Redistribution. You may reproduce or distribute the Work only if (a) you do so under this
  License, (b) you include a complete copy of this License with your distribution, and (c) you
  retain without modification any copyright, patent, trademark, or attribution notices that are
  present in the Work."
- §3.2: "Derivative Works. You may specify that additional or different terms apply to the use,
  reproduction, and distribution of your derivative works of the Work ("Your Terms") only if (a)
  Your Terms provide that the use limitation in Section 3.3 applies to your derivative works, and
  (b) you identify the specific derivative works that are subject to Your Terms."
- §3.3: "Use Limitation. The Work and any derivative works thereof only may be used or intended for
  use non-commercially. ... As used herein, "non-commercially" means for research or evaluation
  purposes only and not for any direct or indirect monetary gain."
- §1: "derivative works shall not include works that remain separable from, or merely link (or bind
  by name) to the interfaces of, the Work."
- There is no patent grant; §3.4 only terminates the licence of anyone who brings a patent claim.

Each source file additionally carries a header saying that use "without an express license
agreement from NVIDIA CORPORATION is strictly prohibited" (e.g. `ops.py:1-7`); `LICENSE.txt` is that
agreement. The README adds: "We do not currently accept outside code contributions". **[V]** [1]

Consequences. **[I]** (my reading; not legal advice)
- Code copied from nvdiffrast, or a substantial adaptation of it, is a derivative work. It could be
  distributed only with the §3.3 non-commercial limitation attached (§3.2 a). MIT grants use
  "without restriction", so such code cannot be MIT. Mixing it in would make tetvox as a whole
  non-commercial-only in practice and contradict ADR 0002.
- This covers nvdiffrast's copy of cudaraster too: its files are headed "Copyright (c) 2009-2022,
  NVIDIA CORPORATION" with the same notice (`impl/TriangleSetup.inl:1-7`), and the modernisation
  (the `__syncwarp` logic, viewport tiling, peeling) exists only under this licence. **[V]** for the
  header, **[I]** for the conclusion
- Merely *calling* nvdiffrast as an optional dependency would not make tetvox a derivative (§1),
  but there is no technical reason to do so.
- Algorithms and ideas are not what copyright protects. Reimplementing a technique from the papers
  or from having read the source is fine, provided the code is written fresh and not transcribed.
  Small things like the coalescing macros should be re-expressed in our own form (inline functions,
  our names), not copied.

**The original cudaraster** (Laine & Karras 2011) has a different licence. **[V]** [4]
- The Google Code archive record for `cudaraster` has `"license": "bsd"`, and its description of
  release 1.1 lists "Switch to New BSD License (previously Apache License 2.0)".
- A mirror whose first commit is "the 1.1 release, downloaded from
  http://code.google.com/p/cudaraster/" carries the 3-clause BSD text, "Copyright (c) 2009-2011,
  NVIDIA Corporation", in `LICENSE` and in each file header. Its constants equal nvdiffrast's
  (`CR_SUBPIXEL_LOG2 4`, `CR_MAXVIEWPORT_LOG2 11`, `CR_TILE_LOG2 3`, checked with `diff`).
- BSD-3-Clause code may be included in an MIT project if the copyright notice, the conditions and
  the disclaimer are kept in the source and reproduced with binary distributions, and NVIDIA's name
  is not used for endorsement. **[I]**
- It is not worth using: Windows/Visual Studio 2010 era, CUDA 4, written for warps that execute in
  lock-step without `__syncwarp` (the very thing nvdiffrast had to fix), 2D with depth, blending and
  a shader pipe. **[I]** from [4] README and `CudaRaster.hpp:12-17`

**Conclusion.** tetvox reimplements ideas and copies nothing. Attribution: cite Laine et al. 2020
and Laine & Karras 2011 in `docs/` and in a comment where a technique is used (courtesy and
traceability; no licence requires it). If BSD code from cudaraster 1.1 were ever adapted, that
file keeps the BSD notice and the wheel gains a third-party notices file. Patents were not searched
(Open questions).

### 6. What to inherit

#### 6.1 Measurements used below **[M]**
Adult atlas (T = 338,993) on its template grid 181×217×181, ownership from the tetvox CPU backend
(6,998,400 owned voxels). Rows run along axis 0 as in `row_span`.

| Quantity | Value |
|---|---|
| bounding-box candidates / K3 rows | 79,065,084 / 10,572,440 (as R1 §3.1) |
| non-empty rows | 3,183,490 (30.1 %); 69.9 % of row threads write nothing |
| span of a non-empty row: mean / p50 / p90 / p99 / max | 2.20 / 1 / 4 / 13 / 95 voxels |
| tets owning no voxel | 3,636 (1.1 %), 39,583 rows |
| rows whose `(i1, i2)` lies under the tet's projection ("shadow") | 4,717,158 (44.6 %) |
| `(tet, i2)` slabs | 1,795,028 |
| empty rows rejected by a division-free one-face test | 4,214,092 (57.0 % of the empty rows), none wrongly |
| int64 divisions in `row_span`: all rows / after that reject | 41,995,988 / 25,251,247 |
| lane-iterations of the voxel loop, plain K3 (32 × longest row per warp) | 33,773,280 → utilisation 20.7 % |
| same, rows pooled per block of 32 / 256 / 1024 rows, voxels dealt evenly | utilisation 52.9 % / 55.5 % / 56.7 % |
| same, one thread per owned voxel after a global scan | 6,998,400 → 100 % (plus the 10.57 M row threads) |
| same, one thread per tet (K2) | utilisation 10.7 % |
| global atomics for a 12-float moment set: per voxel / per non-empty row / per (warp, tet) group of rows / per (256-row block, tet) group | 84.0 M / 38.2 M / 6.5 M / 4.3 M |
| voxel-parallel kernel, owned voxels per (warp, tet) group: 32 voxels along axis 2 / 4×8 block | 2.15 / 3.45 |
| tet–brick pairs, bricks of 4³ / 8³ / 16³: by bounding box | 3.60 M / 1.30 M / 0.70 M (as R1) |
| … after an exact conservative plane test | 1.76 M / 0.96 M / 0.62 M |
| … that own at least one voxel | 1.23 M / 0.73 M / 0.52 M |
| row pieces if each surviving pair ran rows inside its brick | 15.3 M / 15.0 M / 13.6 M |
| mean tets per brick after the plane test | 15.1 / 65.1 / 309.5 |

These are counts of work items, not timings. No GPU kernel was run for this report (R3 is timing a
minimal K3 on the same GPU in parallel).

#### 6.2 Ranked list

**1. Persistent threads with a device-side work counter; no host readback.**
- nvdiffrast: `AntialiasFwdAnalysisKernel` launches `numCTA × numSM` blocks that loop "Persistent
  threads work fetcher": `s_base = atomicAdd(&p.workBuffer[0].y, THREADS_PER_BLOCK)` until the
  count stored *on the device* is reached (`antialias.cu:219-232`; launch at
  `torch_antialias.cpp:144-151`). cudaraster's bin, coarse and fine stages do the same with batch
  sizes of 512–16,384 triangles, one bin, and one tile (`BinRaster.inl:56-66`,
  `CoarseRaster.inl:93-103`, `FineRaster.inl:211-219`). **[V]**
- Why it matters for tetvox: R1's K3 is "exclusive scan of row counts, then one thread per row".
  The number of rows is known only on the device after the scan, and a normal launch needs it on
  the host to set the grid size. That is a readback and a `cudaStreamSynchronize` in the middle of
  `locate`, `rasterize` and `accumulate`. It is at odds with ADR 0009 ("every other op is
  asynchronous, so CUDA graph capture works"). **[I]** (ADR 0009 does not mention the row total;
  R1's prototype is gone, so how it obtained the total is not recorded)
- In tetvox, three ways to launch K3:
  - (a) read the total back and synchronise. Simplest. The op blocks, like nvdiffrast's
    rasteriser does.
  - (b) keep the global scan and row offsets, but run the row kernel as persistent blocks that take
    chunks of row indices from an atomic counter and read the total from device memory. Sync-free.
    Still needs the temporary (`cudaMallocAsync`) and the binary search over `T` per thread.
  - (c) persistent blocks take *batches of tets* from the counter (as the bin stage takes
    triangles), compute their row counts, prefix-sum them in shared memory, and deal the rows to
    the block's threads with a search over the batch only. Sync-free, no global scan, no temporary
    allocation, no search over `T`. Each block needs a loop over rounds for batches with many rows.
- Benefit: asynchronous ops as ADR 0009 promises; dynamic balancing between blocks for the
  heavy-tailed tets. Cost: one global atomic per block per chunk (thousands per call, not per
  voxel); a launch size taken from an occupancy query; a more complex kernel for (c). Output is
  unchanged and still deterministic, since each voxel has one writer. **[I]**
- ADR impact: ADR 0006 says K3 uses "No atomics". (b) and (c) add a work counter, so the wording
  needs the owner's amendment. ADR 0009's `cudaMallocAsync` plan becomes unnecessary under (c).

**2. Coalesced atomics for scatters.**
- nvdiffrast: `common.h:196-241`, used in `rasterize.cu:169-170,269-272`, `interpolate.cu:177-178,
  207-209`, `antialias.cu:548-554`, `texture_kernel.cu:627-637`. **[V]**
- In tetvox: `tet_moments` (and the moment part of `accumulate<Cost>`), `interpolate_vjp_values`,
  `vertex_grad`. In a row-parallel kernel, consecutive rows belong to the same tet, so grouping a
  warp by tet id reduces 84.0 M per-voxel atomics to 6.5 M; per 256-row block, 4.3 M. In a
  voxel-parallel kernel the gain is 2.2–3.5×. **[M]**
- This is also how to realise ADR 0005's rule "float32 inside a thread block and float64 across
  blocks": lanes sum their own row in float32 registers, the group combines in float32, the leader
  issues the float64 `atomicAdd`. **[I]**
- Cost: about 40 lines, written fresh. No ADR change. It gives input to the ROADMAP experiment
  "tet-parallel vs voxel-parallel accumulation": row-parallel with coalescing needs 4–6× fewer global
  atomics than voxel-parallel with coalescing (6.5 M against 24–39 M). **[M]**/**[I]**

**3. Block-local redistribution of per-voxel work.**
- cudaraster: the fine stage turns coverage masks into a fragment queue with a warp prefix sum and
  gives every lane one fragment (`FineRaster.inl:285-338`); the coarse stage "distribute[s] these
  write tasks evenly to all 512 threads" ([3] §5.3). **[V]**
- In tetvox: a K3 row thread that loops over its span keeps its warp busy until the longest span of
  the warp ends. Utilisation is 20.7 %; pooled per block it is 53–57 %; with a global second pass
  100 %. **[M]**
- Where it pays: `accumulate<Cost>`, whose per-voxel body gathers vertex data and evaluates the
  cost. For `rasterize` the body is four multiply-adds and a 16-byte write, and the row set-up
  probably dominates. **[I]**, not timed
- Cost: a shared-memory scan and a small search per lane per round; more kernel code. It combines
  naturally with 1(c), which already has the batch's rows in shared memory. No ADR change; float
  sums change order, which ADR 0005 allows.

**4. Division-free rejection of empty rows.**
- cudaraster: bin and coarse stages bias each edge function to the box corner where it is largest
  and reject the box when the value is negative (`BinRaster.inl:239-244`,
  `CoarseRaster.inl:317-322`). **[V]**
- In tetvox, inside or before `row_span`: for face `j`, `max over the row of t_j = b_j + max(a_j·0,
  a_j·(ext0 − 1))`; if negative, the row is empty. It removes 4.21 M of the 7.39 M empty rows and
  40 % of the int64 divisions, and never rejects a non-empty row. **[M]**
- Cost: two multiplies and a compare per face on values already computed. It answers part of R1's
  open question 7 (int64 division speed) by avoiding divisions. It touches the shared header, so
  CPU and GPU stay identical. Results are unchanged. **[I]** Owner decision, since `row_span` is a
  documented procedure (ADR 0009).

**5. Zero-gradient early exit in backward ops.**
- nvdiffrast: `rasterize.cu:143-148` ORs the bit patterns of the incoming gradients and returns if
  only sign bits are set. **[V]**
- In tetvox: `tet_moments`, `interpolate_vjp_values`. Masked voxels and the 10.5 % of the volume in
  single-class tets (`atlas-format.md`) carry `ḡ = 0` exactly. No ADR change; sums differ at most in
  the sign of a zero. **[I]**

**6. A coarser cull before the rows (two-level K3).**
- cudaraster: tiles are tested conservatively before any pixel work (§2.2). **[V]**
- In tetvox: one thread per `(tet, i2)` slab computes the `i1` range under the tet's projection,
  then rows are launched only there: 1.80 M slab threads plus 4.72 M rows instead of 10.57 M rows.
  **[M]** R1 §3.3(d) called this slab decomposition and judged it not worth the clipping logic.
  The count supports trying it only if profiling shows row set-up dominating after item 4. **[I]**

**7. Explicit, reusable objects for anything derived from static topology.**
- nvdiffrast: `antialias_construct_topology_hash(tri)` and `texture_construct_mip(tex)` return
  opaque objects the caller may keep (`ops.py:442-465,529-544`). **[V]**
- In tetvox: already the pattern of `TetSetup`. A further use: a vertex-to-tet incidence table
  (CSR), built once, would let `vertex_grad` gather instead of scatter: no atomics and a fixed
  summation order. Low priority; `vertex_grad` is 4.1 M atomics per call as it stands. **[I]**

**8. Launch-size and warp-exit helpers.** Block shapes derived from the buffer shape
(`common.cpp:14-58`), whole-warp early exit with `__all_sync` (`interpolate.cu:37-47`). Minor.
**[V]**/**[I]**

#### 6.3 Could hierarchical binning beat K3 on our size distribution?
No. **[I]** from the counts in §6.1; no binned kernel was run.
- **Work.** A 3D cudaraster needs a brick small enough for one coverage mask: 4³ = 64 voxels, the
  analogue of the 8×8 tile. That gives 1.76 M tet–brick pairs after exact culling (3.60 M by
  bounding box), 15 tets per brick. For each pair the fine stage must produce four half-space
  masks. If it does so by rows inside the brick, it handles 15.3 M row pieces, more than K3's
  10.6 M whole rows, because every long row is cut at brick borders. Larger bricks reduce the
  pairs but not the row pieces (15.0 M at 8³).
- **No table.** cudaraster's fine stage is fast because an edge through an 8×8 tile is looked up in
  a 6 KB table. Its index is computed in float32, valid for a 2¹¹ viewport with 4 sub-pixel bits
  (`Util.inl:85,248-275`). A plane through a 4³ brick is a 4×4 height field, with no comparably
  small code; decomposed into four 4×4 slices it needs four exact intercepts per face per pair, in
  int64 at k = 14. That is as many divisions as K3 spends on the same voxels. **[I]**
- **Overhead.** Binning adds queue writes with segment allocation for 1.76 M pairs on every call,
  buffers whose size is not known in advance (hence nvdiffrast's retry loop), and a stream sync.
  **[I]** from §2.4
- **Its two purposes do not apply.** Ordered blending and a depth-tested tile in shared memory
  ([3] §4.1) are not needed when each voxel has one writer (R1 §1.4).
- **What it does better** can be had without bins: balanced per-fragment work (item 3), dynamic
  scheduling and no global search (item 1c), rejection before fine work (items 4 and 6).
- One real advantage remains unmatched: a brick written back from shared memory is a coalesced
  write, while K3 writes short spans (2.2 voxels on average) at scattered addresses. R1 measured
  3.9 ms for the whole atlas with that pattern, so it is not a bottleneck today. Rows along axis 2
  (ROADMAP open question) address part of it. **[I]**

#### 6.4 Deliberately not inherited
| What | Where | Why not |
|---|---|---|
| any code | whole repository | licence (§5) |
| bin / coarse / fine hierarchy, linked-segment queues | `impl/*.inl` | §6.3 |
| 8×8 coverage look-up table | `Util.inl:214-300` | 2D; exact only at 4 sub-pixel bits |
| barycentrics from unsnapped floats, clamped | `rasterize.cu:57-91` | breaks "one geometry" (ADR 0006, R1 §2.4) |
| area epsilon in the backward only | `rasterize.cu:190-193` | we skip `D ≤ 0` exactly and need none |
| per-pixel geometry recomputation and per-vertex scatter of position gradients | `rasterize.cu:172-272` | §3.6 |
| id stored as float32 in the barycentric buffer, `id + 1`, 0 = empty | `common.h:186-193`, `rasterize.cu:94` | int32 `tet_id` with −1 is cleaner and has no 2²⁴ boundary |
| `float4`/`float2` casts and alignment checks | `torch_rasterize.cpp:151-154` | DLPack offsets; fragile |
| index checks in every kernel | `rasterize.cu:43-47` etc. | validate once (ADR 0008, CLAUDE.md) |
| geometric clipping | `TriangleSetup.inl:355-434` | not watertight along clipped edges; we clip the box only |
| worst-case buffers with grow-and-retry, sync per call | `RasterImpl.cpp:172-231,367` | our sizes are exact before launch |
| stateful context with setters | `CudaRaster.hpp:42-51`, `ops.py:47-68` | explicit intermediates (ADR 0009) |
| `rast_db`, `diff_attrs`, gradients through derivative buffers | `rasterize.cu:96-113,213-267` | `∂π/∂x` is per tet, in the record |
| `texture`, `antialias`, depth, peeling | — | not our problem (R2 §4) |
| torch tensors and allocation in C++, current-stream lookup, build against torch | `csrc/torch/*` | ADR 0001, ADR 0004 |
| one backward kernel for both interpolate gradients | `interpolate.cu:131-270` | ADR 0009 keeps ops single-purpose; typical use needs one of the two. A fused kernel can be added under ADR 0009's rule if a benchmark asks for it |

### Method of the measurements
- Scripts `workload.py`, `workload2.py`, `workload3.py`, `workload4.py` in the session scratch
  directory
  `/tmp/claude-3234968/-u-68-defloa2-unix-git-gpu-rasteriser/5b1132b0-d879-4ce4-b26d-07a7d4855cff/scratchpad/r4/scripts/`
  (temporary). They call `tetvox.setup_tets` and `tetvox.rasterize` (CPU backend, branch
  `cpu-backend`) on `atlases/adult_template/atlas_adult.npz`, grid `(181, 217, 181)`, and count
  with numpy.
- Rows are enumerated as in R1's pseudocode: tet-major, then `(i1, i2)` over the grid-clipped box.
  A row's span is the number of voxels of that tet in the row.
- "Lane-iterations" of plain K3: rows grouped 32 at a time in enumeration order, 32 × the longest
  span of each group. Pooled: `blk · ceil(voxels in the block / blk)` per block of `blk` rows.
- Coalescing groups: distinct tets with at least one non-empty row among 32 (or 256) consecutive
  rows; for the voxel-parallel case, distinct owners among 32 consecutive voxels in C order, or in
  a 4×8 block of axes (1, 2).
- Brick pairs: bricks overlapped by the clipped box; "plane test" rejects a pair when, for some
  face, the largest `t_j` over the voxels of box ∩ brick is negative (integers from the record).
- The shadow test uses float64 on the snapped positions (point in the hull of the four projected
  vertices); it is a statistic, not an ownership decision. No non-empty row fell outside it.
- nvdiffrast was cloned to the scratch directory and only read; the papers were converted with
  `pdftotext`.

## Options compared

Work distribution for the CUDA K3 (§6.2, item 1):

| Option | Pros | Cons | Fit for tetvox |
|---|---|---|---|
| (a) scan, read the row total back, launch one thread per row | simplest; R1's prototype shape | stream sync in every `locate`/`rasterize`/`accumulate`; contradicts ADR 0009's async statement; no graph capture | acceptable as a first oracle only |
| (b) scan, then persistent blocks taking row chunks; total stays on the device | sync-free; small change from (a); dynamic balance between blocks | still a temporary of `T + 1` offsets and a search over `T` per thread; one work atomic per chunk | good first production kernel |
| (c) persistent blocks taking tet batches; row counts scanned in shared memory | sync-free; no global scan, no temporary, no global search; ready for block-local redistribution (item 3) | most complex; rounds for heavy batches; tuning constants (batch size, block size) | best match; build after (b) if timing justifies |
| cudaraster-style bins and tiles | proven in 2D; coalesced tile writes | more row work than K3, queues rebuilt per call, unknown buffer sizes, no 3D mask table | reject (§6.3) |

Accumulation of the 12 per-tet moments (items 2 and 3):

| Option | Global atomics per call (adult atlas) | Notes |
|---|---|---|
| voxel-parallel, plain | 84.0 M | reads the raster buffer; simplest |
| voxel-parallel, coalesced per warp | 24–39 M | nvdiffrast's scheme as is |
| row-parallel, one set per non-empty row | 38.2 M | |
| row-parallel, coalesced per (warp, tet) | 6.5 M | recommended starting point |
| row-parallel, coalesced per (256-row block, tet) | 4.3 M | matches ADR 0005's two-level sum |
| nvdiffrast style: per-voxel geometry, scatter to vertices | 84.0 M, colliding across ~25 tets per vertex | reject (§3.6) |

## Open questions
1. **K3 launcher (owner).** Which of (a), (b), (c)? Settle with the first CUDA backend: time all
   three on the adult atlas at 181×217×181 and 256³, including the sync cost of (a) inside a torch
   loop.
2. **Does block-local redistribution pay for `rasterize`?** Only a timing can say; the count says
   2.7× fewer lane-iterations, but the body is cheap. For `accumulate<Cost>` time it with the
   samseg cost once that exists.
3. **Row reject in the shared header (owner).** Add the one-face test to `row_span`, or keep
   `row_span` minimal and put the test in the kernels? Measure on CPU and GPU.
4. **Cost of int64 division** on the target GPU (R1 open question 7) decides how much items 4 and 6
   are worth.
5. **Python stream handling.** The three mechanisms in §4 need to be checked against the torch,
   cupy and DLPack documentation and one chosen (ROADMAP, CUDA item).
6. **Patents.** nvdiffrast's licence grants no patent rights, and neither does BSD. Whether NVIDIA
   holds patents covering cudaraster's techniques (coverage tables, the queue scheme) was not
   searched. The techniques recommended here (persistent threads, warp-coalesced atomics,
   corner tests) are widely published; this is an assessment, not a search.
7. **Not done.** No GPU timing of any variant. `texture_kernel.cu` and `torch_texture.cpp` were
   only skimmed. The cudaraster paper was read in the sections cited (§1, §3.3, §4.1–4.2, §5.1–5.4),
   not in full. The nvdiffrast paper was read in §1–3.7 and §4.4–4.5.

## Impact on design

Decisions for the owner (none is taken here):
- **D1. K3 work distribution** (options (a)–(c) above). Affects ADR 0006 ("No atomics, no sorting,
  no persistent state": a work counter is an atomic), ADR 0009 (async statement; whether
  `cudaMallocAsync` or a workspace argument is needed at all), and the kernels of `locate`,
  `rasterize`, `count_claims`, `accumulate`. Recommendation: (b) first, (c) if timing justifies.
- **D2. Row pre-reject** in `row_span` or in the kernels. Affects the shared header and both
  backends; no result changes. Recommendation: in the shared procedure, so the CPU gains too.
- **D3. Coalescing granularity and accumulator types** for `tet_moments` and `accumulate`: per warp
  or per block; float32 inside, float64 across (ADR 0005 already says so). Recommendation: per
  (warp, tet) first, measure per block.
- **D4. Tunable constants** these introduce, each to be named and owned: tets or rows per work
  chunk, threads per block, blocks per SM (from the occupancy query or fixed).
- **D5. Attribution.** Cite [2] and [3] in the docs and next to each borrowed technique; copy no
  code. Recommendation: yes.

Consequences if adopted:
- **Kernels.** `tetvox::cuda::locate/rasterize` = `setup_tets` output → persistent row kernel using
  `row_span` unchanged → single-writer stores. No readback. `count_skipped` remains the only
  synchronising op.
- **Backward ops.** `tet_moments` row-parallel on the K3 distribution with coalesced atomics by tet
  id and a zero-`ḡ` early exit; `vertex_grad` per tet with plain atomics (4.1 M per call);
  `interpolate_vjp_values` voxel-parallel with coalescing by tet id.
- **No new public type.** No context object, no workspace argument under (c).
- **Precision.** Nothing changes: exact integers for ownership, π from the record, sums as ADR
  0005.
- **Tests.** Bit-identity of GPU and CPU `tet_id`/π must hold for every launcher variant and for
  any chunk size (run the test with two different chunk constants). Gradient ops: compare against
  the CPU backend within the float tolerances of ADR 0008, since atomics reorder sums.
- **Docs.** `architecture.md` already describes K3 as "one thread per box row"; with D1 it should
  say how rows reach threads.

## References
[1] NVlabs, *nvdiffrast*, commit `253ac4f` (tag `v0.4.0`, 5 Dec 2025).
https://github.com/NVlabs/nvdiffrast — files cited by path; also `docs/index.html` at this commit
and at tag `v0.3.5` (`20c4135`), `LICENSE.txt`, `README.md`, `setup.py`, `pyproject.toml`.
[2] S. Laine, J. Hellsten, T. Karras, Y. Seol, J. Lehtinen, T. Aila, *Modular Primitives for
High-Performance Differentiable Rendering*, ACM TOG 39(6), 2020. https://arxiv.org/abs/2011.03277
(read: §1–3.7, §4.4–4.5)
[3] S. Laine, T. Karras, *High-Performance Software Rasterization on GPUs*, HPG 2011.
https://research.nvidia.com/sites/default/files/pubs/2011-08_High-Performance-Software-Rasterization/laine2011hpg_paper.pdf
(read: §1, §3.3, §4.1–4.2, §5.1–5.4)
[4] Original cudaraster 1.1. Google Code archive record:
https://storage.googleapis.com/google-code-archive/v2/code.google.com/cudaraster/project.json
(`"license": "bsd"`; release notes in the description). Mirror of the 1.1 package:
https://github.com/ap1/cudaraster (commit `11f6516`, `LICENSE`, `README.md`,
`src/cudaraster/cuda/Constants.hpp`).
[5] Local: `docs/research/R1-watertight-tet-rasterisation.md`,
`docs/research/R2-decoupling-cost-from-rasteriser.md`, ADRs 0001 and 0004–0009,
`docs/knowledge/{atlas-format,rasteriser-cpu,build,meshing-cxx-reference}.md`.
