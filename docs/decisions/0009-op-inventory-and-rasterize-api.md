# 0009 — Op inventory and the public shape of the rasterisation ops

Status: accepted (2026-11-23). The owner approved the session's proposal (points B1–B14) unchanged.
It settles ADR 0008's open point B and the dtype of `load_atlas_npz`. It amends ADR 0007: the
optional `visits` output becomes the separate op `count_claims`, the counts move to the setup op,
and the Python call `rasterize(mesh, shape)` becomes the setup + rasterize chain below.

Context: ADR 0005–0008, `docs/research/R1-watertight-tet-rasterisation.md` ("Recommended
algorithm", "Impact on design"), `docs/research/R2-decoupling-cost-from-rasteriser.md` (§1, "Impact
on design", addendum). The owner's requirement: each op does one thing, and it is clear which
procedure each op runs and how the ops compose into the generic and fused paths.

## Decision

### Two levels: procedures and ops (B1, B12)
**Procedures** are `__host__ __device__` functions in a public header (`tetvox/raster.hpp`,
namespace `tetvox::`, documented building blocks, not `detail`). Every op is a loop over them:

| Procedure | Computes |
|---|---|
| `snap(v, k)` | `q = llrint(v·2ᵏ)`, ties to even (ADR 0007) |
| `setup_tet(q[4], k) → TetRecord<T>` | classification (ADR 0008 order), `n_j`, `c_j`, `f_j`, `2ᵏ/D`, box |
| `plane_values(rec, X) → t[4]` | `t_j = n_j·(X − O) + c_j`, int64 |
| `owns(t)` | the inside test: all `t_j ≥ 0` (the tie-break is folded into `c_j`) |
| `barycentric(rec, t) → π[4]` | `π_j = (T(t_j) + f_j)·(2ᵏ/D)` (R1 §2.5, ADR 0008) |
| `row_span(rec, i1, i2, shape) → [i0a, i0b]` | the axis-0 interval of one box row, by exact `ceil_div`/`floor_div` |

**Ops** work on whole arrays. Each exists as `tetvox::cpu::*` and `tetvox::cuda::*`, is bound to
Python, and does one thing:

| # | Op | Procedure | Inputs | Outputs |
|---|---|---|---|---|
| 1 | `setup_tets` | per tet: `snap`, classify, `setup_tet` | positions (N,3) T, tetrahedra (T,4) i32, `snap_bits` | `TetSetup` record |
| 2 | `count_skipped` | reduction of `status` | `status` | `{inverted, outside_window}` |
| 3 | `locate` | voxel → owning tet ("simplex of voxel"): `row_span` per box row | setup, grid shape | `tet_id` (X,Y,Z) i32, −1 unowned |
| 4 | `barycentrics` | `plane_values` + `barycentric` at each owned voxel, from `tet_id` | setup, `tet_id` | `bary` (X,Y,Z,3) T = π₁..π₃, 0 unowned |
| 5 | `rasterize` | 3 and 4 in one pass; equal to `barycentrics(setup, locate(setup, shape))` bit for bit | setup, grid shape | `tet_id`, `bary` |
| 6 | `count_claims` | `owns` at every voxel of every box (diagnostic) | setup, grid shape | `claims` (X,Y,Z) i32, zeroed by the call |
| 7 | `interpolate` | `f = Σ_j π_j F[v_j]` | `tet_id`, `bary`, tetrahedra, values (N,C) | field (X,Y,Z,C) |
| 8 | `interpolate_vjp_values` | `∂L/∂F = Wᵀ g_f` (scatter) | `tet_id`, `bary`, tetrahedra, `grad_field` | (N,C) |
| 9 | `interpolate_vjp_bary` | `ḡ_m = (F_m − F_0)·g_f` (reads no π) | `tet_id`, tetrahedra, values, `grad_field` | `grad_bary` (X,Y,Z,3) |
| 10 | `tet_moments` | `m_i = Σ_x π_i ḡ`, geometry-free (R2 eq. 3) | setup, `tet_id`, `bary`, `grad_bary` | (T,4,3) |
| 11 | `vertex_grad` | `G_i = −(2ᵏ/D) Σ_m m_{i,m} n_m`, scatter-add to vertices | setup, tetrahedra, moments | `grad_positions` (N,3) |
| 12 | `accumulate<Cost>` | fused: `row_span` → `barycentric` → `Cost` → moments, one kernel, nothing per voxel; +inf if any status ≠ ok | setup, grid shape, tetrahedra, `Cost` | total cost (double), moments (T,4,3) |

- Ops 1–6 are specified here in full and built in the CPU-backend item.
- Ops 7–12 are accepted in name and inputs. Their numerics (accumulator dtype, the `Cost` concept,
  curvature policies) are decided in their own ROADMAP items. `tet_moments` takes `setup` so that
  its signature survives the tet-parallel vs voxel-parallel experiment (a tet-parallel version
  walks the tet's rows). `accumulate` rasterises inside its kernel from `setup` + shape (R2's
  "fused into the rasteriser" option) rather than reading a stored raster buffer; this is
  confirmed when that item is built.
- R2's `bary_vjp` is `vertex_grad ∘ tet_moments`; R2's `interpolate_vjp` is ops 8 and 9. There is no
  composite op unless it is a genuine fused kernel with a defined equality to the chain (op 5).
- Internal only: K3's row-offset scan, the validation pre-pass, per-thread partial sums.
- Tests only: the pure-Python exact-integer reference in `tests/` (compared against `TetSetup`
  fields, `tet_id`, π and `claims`), and the bit-identity test `rasterize == barycentrics∘locate`.

Composition:

```
Generic:  setup_tets ─► rasterize ─► interpolate ─► [user cost] ─► interpolate_vjp_bary ─► tet_moments ─► vertex_grad
              │            (or a user kernel on tet_id / bary writing ḡ) ───────────────────►┘
              └─► count_skipped (caller's +inf policy)
Fused:    setup_tets ─► accumulate<Cost> ──────────────────────────────────────────────────────────► vertex_grad
```

### The setup record is an exposed intermediate (B2, B3)
- `setup_tets` is the only op that snaps. Every later op reads its record, so ownership, π and
  `∂π/∂v` use the same snapped integers (ADR 0006 "one geometry") by construction, k appears in one
  place, and the counts come from one place.
- **Unclipped origin.** `O = ceil(min q / 2ᵏ)` per axis, not clipped to the grid. The record
  depends only on `(positions, tetrahedra, k)`; clipping to the grid happens in ops 3–6 and 12.
  The bit budget is unchanged (inferred): samples lie in the tet's own box, so `|X − O| ≤ 2ᴸ`.
  When a box is empty along an axis, `O` lies outside the hull by less than one voxel; then
  `|c_j| ≤ 6(M + S)³/S + 1 < 2⁶³` still holds, and no sample is tested.
- **Transparent layout** (structure of arrays, documented fields):

  | Field | Shape | Type | Meaning |
  |---|---|---|---|
  | `normals` | (T,4,3) | int64 | inward face normals `n_j`, lattice units |
  | `offsets` | (T,4) | int64 | `c_j`, tie-break folded in |
  | `fractions` | (T,4) | T | `f_j ∈ [0, 1)` |
  | `bary_scale` | (T,) | T | `2ᵏ/D`; `p_j = n_j · bary_scale` is `∂π_j/∂x` in voxel units |
  | `box_lo`, `box_hi` | (T,3) | int64 | `O = ceil(min q/2ᵏ)`, `floor(max q/2ᵏ)`; int64 because `|v|` may reach `2^(62−k)` |
  | `status` | (T,) | uint8 | `TetStatus`: `ok = 0`, `inverted = 1`, `outside_window = 2` |

  For a skipped tet every field other than `status` is unspecified. Size: 197 B per tet with float,
  217 B with double (67 / 74 MB for the adult atlas), allocated by the caller.

### Raster ops (B5, B13, B14)
- `locate`, `barycentrics` and `rasterize` are all public. `rasterize` is the fused form used in
  practice; `locate` and `barycentrics` serve label-only uses and the "tet_id only, recompute π"
  raster-buffer benchmark.
- `count_claims` is a public diagnostic (does a mesh tile the grid; overlap around folds), not a
  test-only function.
- **How ownership is found fast.** No op searches per voxel: each tet writes the voxels it owns, and
  each voxel has exactly one writer, so there are no atomics and no sorting. The backends share the
  procedures and differ only in how work is distributed:
  - GPU: K3 (ADR 0006): exclusive scan of row counts flattens all (tet, i1, i2) rows, one thread per
    row solves `row_span` and writes the interval.
  - CPU: OpenMP over tets, each thread scans its tet's grid-clipped box (structure of
    `meshing_cxx`, ADR 0007), solving each row with `row_span` (B14). CPU and GPU thus run the same
    per-row procedure.
  - `count_claims` instead evaluates `owns` at every voxel of the clipped box on both backends. It is
    an independent path: it checks exact tiling and, at the same time, that `row_span` and `owns`
    agree.

### C++ form (B6, B7)
```cpp
namespace tetvox {                                   // tetvox/types.hpp
template <class T> concept Real = std::same_as<T, float> || std::same_as<T, double>;
using GridShape = std::array<int64_t, 3>;            // axes 0,1,2; C order, axis 2 fastest (ADR 0007)
inline constexpr int default_snap_bits = 14;         // k, ADR 0006
enum class TetStatus : uint8_t { ok = 0, inverted = 1, outside_window = 2 };
struct SkippedTets { int64_t inverted; int64_t outside_window; };

template <Real T> struct MeshView {                  // caller-owned, read-only
  const T* positions; const int32_t* tetrahedra; int64_t num_vertices, num_tets; };
template <Real T> struct TetSetupView {              // caller-owned; written only by setup_tets
  int64_t num_tets; int64_t* normals; int64_t* offsets; T* fractions; T* bary_scale;
  int64_t* box_lo; int64_t* box_hi; TetStatus* status; };
}
namespace tetvox::cpu {                              // tetvox/cpu.hpp
template <Real T> void setup_tets(const MeshView<T>& mesh, int snap_bits, const TetSetupView<T>& out);
SkippedTets count_skipped(const TetStatus* status, int64_t num_tets);
template <Real T> void locate      (const TetSetupView<T>& s, GridShape shape, int32_t* tet_id);
template <Real T> void barycentrics(const TetSetupView<T>& s, GridShape shape, const int32_t* tet_id, T* bary);
template <Real T> void rasterize   (const TetSetupView<T>& s, GridShape shape, int32_t* tet_id, T* bary);
template <Real T> void count_claims(const TetSetupView<T>& s, GridShape shape, int32_t* claims);
}
// tetvox::cuda: the same declarations with a required `cudaStream_t stream` as the last parameter.
```
- View structs for the mesh and the record; single outputs are raw pointers. Raw pointers with
  counts, not `std::span`: buffer sizes are checked by the binding (ADR 0008).
- Function templates constrained by `Real`, declared in the public header, defined and explicitly
  instantiated for `float` and `double` in the library. Not overloads, so user code generic over `T`
  can call `rasterize<T>`; not a type-erased view, so dtype errors are compile-time.
- One mutable `TetSetupView`; consumers take it by `const&` and only read it.
- Argument order: inputs, parameters, outputs, then `stream`.
- **CUDA twin:** identical apart from the required trailing stream. Only `count_skipped` synchronises
  (it returns host values); every other op is asynchronous, so CUDA graph capture works. K3's
  temporary row offsets use `cudaMallocAsync` on the stream inside the op (provisional until the
  CUDA item; the alternative is a cub-style workspace argument).
- Outputs are caller-allocated (ADR 0005). Each op writes every element of its outputs (−1 / 0 for
  unowned voxels); `count_claims` zeroes its buffer itself.
- Validation (ADR 0008) runs in the op that reads the data: `setup_tets` checks the tet indices,
  counts and `snap_bits`; ops taking a shape check it. A `TetSetupView` is trusted to come from
  `setup_tets`.

### `snap_bits` and the skipped counts (B8, B9)
- The run-time parameter k is called `snap_bits` in the API (documented as k of ADR 0006). It is an
  argument of `setup_tets` only: required in C++ (`tetvox::default_snap_bits = 14` is available),
  keyword with default `DEFAULT_SNAP_BITS = 14` in Python.
- Valid range `0 ≤ snap_bits ≤ 24`, otherwise `std::invalid_argument`. 24 keeps `f_j = r/2ᵏ` exact
  in float32; `2k + 3L ≤ 59` alone would allow 29. L is derived from k as in ADR 0008.
- `setup_tets` writes the per-tet `status` and returns nothing. `count_skipped` reduces it to the two
  counts. `accumulate` reads `status` on the device to return +inf. The caller can also see which
  tets were skipped.

### Python surface (B10, B11)
```python
F = TypeVar("F", np.float32, np.float64)
DEFAULT_SNAP_BITS: Final = 14
class TetStatus(IntEnum): OK = 0; INVERTED = 1; OUTSIDE_WINDOW = 2
class SkippedTets(NamedTuple): inverted: int; outside_window: int

@dataclass(frozen=True)
class TetSetup(Generic[F]):          # one array per record field, same framework/device as positions
    normals: NDArray[np.int64]; offsets: NDArray[np.int64]; fractions: NDArray[F]
    bary_scale: NDArray[F]; box_lo: NDArray[np.int64]; box_hi: NDArray[np.int64]
    status: NDArray[np.uint8]

def setup_tets(positions: NDArray[F], tetrahedra: NDArray[np.int32], *,
               snap_bits: int = DEFAULT_SNAP_BITS, out: TetSetup[F] | None = None) -> TetSetup[F]
def count_skipped(setup: TetSetup[F]) -> SkippedTets
def locate(setup: TetSetup[F], shape: tuple[int, int, int], *,
           out: NDArray[np.int32] | None = None) -> NDArray[np.int32]
def barycentrics(setup: TetSetup[F], tet_id: NDArray[np.int32], *,
                 out: NDArray[F] | None = None) -> NDArray[F]
def rasterize(setup: TetSetup[F], shape: tuple[int, int, int], *,
              out: tuple[NDArray[np.int32], NDArray[F]] | None = None
              ) -> tuple[NDArray[np.int32], NDArray[F]]
def count_claims(setup: TetSetup[F], shape: tuple[int, int, int], *,
                 out: NDArray[np.int32] | None = None) -> NDArray[np.int32]
```
- Python names equal the C++ names, at top level (`tetvox.rasterize`); no backend in the name.
- Ops take arrays, not `TetMesh`: each op lists exactly what it reads, the mapping to C++ is 1:1, and
  torch keeps `positions` as its own leaf tensor. `TetMesh` is the container the loader returns.
- Returns: a plain tuple for `(tet_id, bary)`, the `TetSetup` dataclass for the record, a
  `NamedTuple` for the two counts.
- `out=` mirrors the outputs (a tuple for two, a `TetSetup` for the record), all or nothing; the
  same objects are returned. Reusing a `TetSetup` across iterations allocates nothing.
- Dispatch on device and dtype is nanobind overload resolution in `_core` ({cpu, cuda} × {float,
  double}; cuda only when built). The Python wrappers have no branches on device or dtype. Wrong
  device or dtype gives nanobind's `TypeError` (ADR 0008).
- With `out=None`, outputs are allocated in the framework of the input: numpy and torch now, cupy
  when CUDA lands. Other frameworks must pass `out=` (`TypeError` otherwise). No `array-api-compat`
  dependency.
- Static types are precise for numpy, generic over `F`. Other DLPack frameworks work at run time;
  `tetvox.torch` provides torch-typed wrappers later.
- CUDA stream handling in Python is settled with the CUDA item.

### Loader dtype and `TetMesh` (B4; ADR 0008 open point)
- `load_atlas_npz(path, *, dtype: type[F] = np.float32) -> SamsegAtlas[F]`. The default matches 11 of
  the 12 stored files; samseg users who keep positions in double pass `np.float64`.
- `TetMesh(Generic[F])` with `positions: NDArray[F]`, `values: NDArray[F]`,
  `tetrahedra: NDArray[np.int32]`, so mypy checks "same dtype". `SamsegAtlas` is generic and holds a
  `TetMesh[F]`. `TypeVar` + `Generic`, since the floor is Python 3.10 (no PEP 695).

## Why
- The owner asked that every op do one thing and that its procedure be visible. Exposing the setup
  record makes the per-tet work an explicit op and leaves `rasterize` a pure function of
  `(setup, shape)`.
- Recomputing setup inside each op would repeat k on every op and could silently give the
  gradient a different geometry from the raster, and on CUDA it would force a host sync per op to
  return counts.
- Sharing the per-row procedure between CPU and GPU makes the owner grids identical by construction;
  keeping `count_claims` per-voxel gives the tests an independent path.

## Consequences
- ADR 0007's optional `visits` and its "Return value: a small struct with two counts" are replaced
  by `count_claims` and `status` + `count_skipped`. Its `rasterize(mesh, shape, *, out=None)` call
  is replaced by `setup_tets` + `rasterize`; "backend picked from the device of the input arrays,
  no `device=` argument" stands.
- R2's per-tet geometry table `(A⁻¹, v₀, det A)` is replaced by `TetSetup` for the data term.
  Whether the deformation prior also reads the snapped geometry is decided in the prior item.
- `architecture.md`: "`TetMesh` is the only mesh type ops take" no longer holds; ops take arrays.
- The C++ example and pytest exercise `setup_tets → rasterize` and `count_claims`.
