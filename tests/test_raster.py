"""CPU rasterisation ops (ADR 0006-0010) against the exact reference and invariants, in 1D-3D."""

from pathlib import Path
from typing import Any

import exact_reference as ref
import numpy as np
import numpy.typing as npt
import pytest
from conftest import kuhn_cube, kuhn_grid

import tetvox
from tetvox import SimplexSetup, SimplexStatus

# Absolute tolerance of pi against the exact value (ADR 0008).
PI_ATOL = {np.float32: 1e-6, np.float64: 1e-12}
# Owned voxels satisfy pi_1..pi_D >= 0 and 1 - sum(pi) >= -SUM_TOL (ADR 0008).
SUM_TOL = 1e-6
DTYPES = [np.float32, np.float64]
DIMS = [1, 2, 3]
K = tetvox.DEFAULT_SNAP_BITS

Mesh = tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], tuple[int, ...]]


def rotation(dim: int, seed: int) -> npt.NDArray[np.float64]:
    """A random rotation matrix (dim, dim)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dim, dim)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def small_meshes(dim: int) -> dict[str, Mesh]:
    """Small meshes for the reference comparison: positions (N, dim), simplices, grid shape."""
    positions, simplices = kuhn_grid(2, dim)
    rng = np.random.default_rng(0)
    jitter = positions * 2.0
    interior = np.all((positions > 0) & (positions < 2), axis=1)
    jitter[interior] += rng.uniform(-0.3, 0.3, size=(int(interior.sum()), dim))
    rotated = (positions - 1.0) * 2.7 @ rotation(dim, 1).T + 1.2  # partly outside the grid
    return {
        "kuhn_on_centres": (positions, simplices, (4,) * dim),
        "kuhn_scaled_3": (positions * 3.0, simplices, (8,) * dim),
        "jittered": (jitter, simplices, (6,) * dim),
        "rotated_partly_outside": (rotated, simplices, (6,) * dim),
    }


SMALL = {f"{dim}d_{name}": mesh for dim in DIMS for name, mesh in small_meshes(dim).items()}


@pytest.fixture(params=list(SMALL))
def small_mesh(request: pytest.FixtureRequest) -> Mesh:
    return SMALL[request.param]


def assert_setup_matches(
    setup: SimplexSetup[Any], expected: list[ref.RefSimplex], snap_bits: int = K
) -> None:
    """Every field of every OK simplex, and every status, bit for bit."""
    np.testing.assert_array_equal(setup.status, [s.status for s in expected])
    dtype = setup.fractions.dtype.type
    for i, simplex in enumerate(expected):
        if simplex.status != ref.OK:
            continue
        np.testing.assert_array_equal(setup.normals[i], simplex.normals)
        np.testing.assert_array_equal(setup.offsets[i], simplex.offsets)
        np.testing.assert_array_equal(setup.fractions[i], [dtype(f) for f in simplex.fractions])
        assert setup.bary_scale[i] == ref.bary_scale(simplex, snap_bits, dtype)
        np.testing.assert_array_equal(setup.box_lo[i], simplex.box_lo)
        np.testing.assert_array_equal(setup.box_hi[i], simplex.box_hi)


@pytest.mark.parametrize("dtype", DTYPES)
def test_setup_matches_reference(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, simplices, _ = small_mesh
    setup = tetvox.setup_simplices(positions.astype(dtype), simplices)
    assert_setup_matches(setup, ref.setup_simplices(positions.astype(dtype), simplices, K))


@pytest.mark.parametrize("dtype", DTYPES)
def test_raster_matches_reference(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, simplices, shape = small_mesh
    p = positions.astype(dtype)
    setup = tetvox.setup_simplices(p, simplices)
    expected = ref.rasterize(ref.setup_simplices(p, simplices, K), shape, K)
    claims = tetvox.count_claims(setup, shape)
    np.testing.assert_array_equal(claims, expected.claims)
    assert np.max(claims) <= 1  # these meshes tile their hull
    simplex_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(simplex_id, expected.owner)
    np.testing.assert_allclose(bary, expected.pi[..., 1:], rtol=0, atol=PI_ATOL[dtype])
    assert bary.dtype == dtype


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rasterize_equals_locate_then_barycentrics(dim: int, dtype: type[np.floating]) -> None:
    positions, simplices, shape = SMALL[f"{dim}d_rotated_partly_outside"]
    setup = tetvox.setup_simplices(positions.astype(dtype), simplices)
    simplex_id, bary = tetvox.rasterize(setup, shape)
    located = tetvox.locate(setup, shape)
    np.testing.assert_array_equal(simplex_id, located)
    chained = tetvox.barycentrics(setup, located)
    assert bary.tobytes() == chained.tobytes()


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("scale", [1, 2, 3])
def test_kuhn_grid_on_voxel_centres_owned_exactly_once(dim: int, scale: int) -> None:
    """Every vertex on a voxel centre: the most ties. The half-open box [0, 3·scale)^D is owned
    exactly once; the far layers and everything outside the mesh are unowned."""
    positions, simplices = kuhn_grid(3, dim)
    extent = 3 * scale
    shape = (extent + 2,) * dim
    claims = tetvox.count_claims(tetvox.setup_simplices(positions * scale, simplices), shape)
    expected = np.zeros(shape, np.int32)
    expected[(slice(0, extent),) * dim] = 1
    np.testing.assert_array_equal(claims, expected)


# Cells per axis of the jittered grids: 4,096 to 10,368 simplices, depending on the dimension.
JITTERED_CELLS = {1: 4096, 2: 64, 3: 12}


@pytest.mark.parametrize("dim", DIMS)
def test_jittered_kuhn_grid_owned_exactly_once(dim: int) -> None:
    """Exhaustive at a larger scale than the reference allows: a Kuhn grid of cells of 4 voxels
    with jittered interior vertices tiles [0, 4n]^D, so the half-open box is owned exactly once."""
    n = JITTERED_CELLS[dim]
    positions, simplices = kuhn_grid(n, dim)
    positions *= 4.0
    interior = np.all((positions > 0) & (positions < 4 * n), axis=1)
    rng = np.random.default_rng(dim)
    positions[interior] += rng.uniform(-0.6, 0.6, size=(int(interior.sum()), dim))
    setup = tetvox.setup_simplices(positions, simplices)
    assert tetvox.count_skipped(setup) == tetvox.SkippedSimplices(0, 0)
    shape = (4 * n + 1,) * dim
    claims = tetvox.count_claims(setup, shape)
    expected = np.zeros(shape, np.int32)
    expected[(slice(0, 4 * n),) * dim] = 1
    np.testing.assert_array_equal(claims, expected)
    simplex_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(simplex_id >= 0, claims == 1)
    assert_barycentrics_valid(simplex_id, bary)


def assert_barycentrics_valid(
    simplex_id: npt.NDArray[np.int32], bary: npt.NDArray[np.floating]
) -> None:
    owned = bary[simplex_id >= 0]
    assert np.min(owned) >= 0
    assert np.min(1 - owned.sum(axis=1)) >= -SUM_TOL
    assert not bary[simplex_id < 0].any()


@pytest.mark.parametrize("dtype", DTYPES)
def test_barycentric_bounds(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, simplices, shape = small_mesh
    simplex_id, bary = tetvox.rasterize(
        tetvox.setup_simplices(positions.astype(dtype), simplices), shape
    )
    assert_barycentrics_valid(simplex_id, bary)


@pytest.mark.parametrize("dim", DIMS)
def test_statuses_and_counts(dim: int) -> None:
    """One simplex of each kind; skipped simplices never claim a voxel."""
    cube, simplices = kuhn_cube(dim)
    base = cube.astype(np.float64) * 2.0
    far = (1 << dim) - 1  # the vertex at (1, ..., 1)
    blocks = [base]
    flat = base.copy()
    flat[:, -1] = 0.0  # every simplex degenerate
    blocks.append(flat)
    if dim > 1:  # in 1D at k = 14 the snap range binds before the window (ADR 0010)
        span_too_wide = base.copy()
        span_too_wide[far, 0] = 2.0 ** ref.window_bits(dim, K) + 1.0
        blocks.append(span_too_wide)
    for value in (np.nan, np.inf, 2.0 ** (ref.SNAP_RANGE_BITS - K + 1)):
        beyond = base.copy()
        beyond[far, -1] = value
        blocks.append(beyond)
    positions = np.concatenate(blocks)
    corners = 1 << dim
    first = np.concatenate([simplices[:1] + corners * i for i in range(len(blocks))])
    swapped = simplices[:1, [1, 0, *range(2, dim + 1)]]  # D < 0
    all_simplices = np.concatenate([swapped, first])
    setup = tetvox.setup_simplices(positions, all_simplices)
    expected = [s.status for s in ref.setup_simplices(positions, all_simplices, K)]
    assert expected == [1, 0, 1] + [2] * (len(blocks) - 2)
    np.testing.assert_array_equal(setup.status, expected)
    assert tetvox.count_skipped(setup) == tetvox.SkippedSimplices(
        inverted=2, outside_window=len(blocks) - 2
    )
    shape = (4,) * dim
    claims = tetvox.count_claims(setup, shape)
    only_ok = ref.rasterize(ref.setup_simplices(positions[:corners], simplices[:1], K), shape, K)
    assert claims.sum() > 0
    np.testing.assert_array_equal(claims, only_ok.claims)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
def test_bit_budget_boundary(dim: int, dtype: type[np.floating]) -> None:
    """A Kuhn cube spanning exactly 2^L voxels (the <= boundary), sampled near its far vertex,
    where t is largest; one lattice unit more is outside the window."""
    cube, simplices = kuhn_cube(dim)
    span = 2.0 ** ref.window_bits(dim, K)
    positions = (cube.astype(np.float64) * span - (span - 3.0)).astype(dtype)  # far vertex at 3
    shape = (5,) * dim
    setup = tetvox.setup_simplices(positions, simplices)
    expected_simplices = ref.setup_simplices(positions, simplices, K)
    assert_setup_matches(setup, expected_simplices)
    assert all(s.status == ref.OK for s in expected_simplices)
    expected = ref.rasterize(expected_simplices, shape, K)
    simplex_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(simplex_id, expected.owner)
    np.testing.assert_array_equal(tetvox.count_claims(setup, shape), expected.claims)
    np.testing.assert_allclose(bary, expected.pi[..., 1:], rtol=0, atol=PI_ATOL[dtype])

    beyond = positions.copy()
    beyond[(1 << dim) - 1, 0] += dtype(2.0**-K)
    assert (tetvox.setup_simplices(beyond, simplices).status == SimplexStatus.OUTSIDE_WINDOW).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_snap_range_boundary_1d(dtype: type[np.floating]) -> None:
    """In 1D the snap range |v| <= 2^(62 - k) binds before the window (ADR 0010): a segment from
    -2^(62 - k) is rasterised exactly near its far end; the next representable value is outside."""
    bound = dtype(2.0 ** (ref.SNAP_RANGE_BITS - K))
    positions = np.array([[-bound], [3.0]], dtype=dtype)
    segment = np.array([[0, 1]], np.int32)
    shape = (5,)
    setup = tetvox.setup_simplices(positions, segment)
    expected_simplices = ref.setup_simplices(positions, segment, K)
    assert_setup_matches(setup, expected_simplices)
    expected = ref.rasterize(expected_simplices, shape, K)
    simplex_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(simplex_id, [0, 0, 0, -1, -1])
    np.testing.assert_array_equal(simplex_id, expected.owner)
    np.testing.assert_allclose(bary, expected.pi[..., 1:], rtol=0, atol=PI_ATOL[dtype])

    beyond = positions.copy()
    beyond[0, 0] = np.nextafter(-bound, dtype(-np.inf))
    assert tetvox.setup_simplices(beyond, segment).status[0] == SimplexStatus.OUTSIDE_WINDOW


@pytest.mark.parametrize("snap_bits", [-1, tetvox.MAX_SNAP_BITS + 1])
def test_snap_bits_out_of_range_rejected(snap_bits: int) -> None:
    positions, simplices = kuhn_cube()
    with pytest.raises(ValueError, match="snap_bits"):
        tetvox.setup_simplices(positions, simplices, snap_bits=snap_bits)


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("snap_bits", [0, tetvox.MAX_SNAP_BITS])
def test_snap_bits_range_ends_accepted(dim: int, snap_bits: int) -> None:
    positions, simplices = kuhn_grid(2, dim)
    setup = tetvox.setup_simplices(positions, simplices, snap_bits=snap_bits)
    assert_setup_matches(setup, ref.setup_simplices(positions, simplices, snap_bits), snap_bits)
    claims = tetvox.count_claims(setup, (3,) * dim)
    assert np.min(claims[(slice(0, 2),) * dim]) == 1 and claims.sum() == 2**dim


@pytest.mark.parametrize("dim", DIMS)
def test_simplex_index_out_of_range_rejected(dim: int) -> None:
    positions, simplices = kuhn_cube(dim)
    bad = simplices.copy()
    bad[0, 0] = 1 << dim
    with pytest.raises(ValueError, match="simplex index"):
        tetvox.setup_simplices(positions, bad)


@pytest.mark.parametrize("dim", DIMS)
def test_barycentrics_rejects_invalid_simplex_id(dim: int) -> None:
    """Provisional (ROADMAP open question 2): ids outside [-1, T), or whose box misses the voxel."""
    positions, simplices = kuhn_cube(dim)
    setup = tetvox.setup_simplices(positions, simplices)
    shape = (3,) * dim
    simplex_id = tetvox.locate(setup, shape)
    origin, far = (0,) * dim, (2,) * dim
    for bad_value, voxel in [(len(simplices), origin), (-2, origin), (0, far)]:
        bad = simplex_id.copy()
        bad[voxel] = bad_value
        with pytest.raises(ValueError, match="simplex_id"):
            tetvox.barycentrics(setup, bad)


def test_dimension_mismatches_rejected() -> None:
    """D outside 1..3, or arrays that disagree with D, raise ValueError (ADR 0010)."""
    positions, simplices = kuhn_cube(3)
    with pytest.raises(ValueError, match="dimension"):
        tetvox.setup_simplices(np.zeros((16, 4), np.float32), np.zeros((1, 5), np.int32))
    with pytest.raises(ValueError, match="D \\+ 1 columns"):
        tetvox.setup_simplices(positions, np.ascontiguousarray(simplices[:, :3]))
    setup = tetvox.setup_simplices(positions, simplices)
    with pytest.raises(ValueError, match="D dimensions"):
        tetvox.rasterize(setup, (3, 3))
    simplex_id = np.zeros((3, 3, 3), np.int32)
    with pytest.raises(ValueError, match="bary"):
        tetvox.rasterize(setup, (3, 3, 3), out=(simplex_id, np.zeros((3, 3, 3, 2), np.float32)))


def test_mixed_dtypes_rejected() -> None:
    positions, simplices = kuhn_cube()
    setup32 = tetvox.setup_simplices(positions, simplices)
    positions64 = positions.astype(np.float64)
    with pytest.raises(TypeError):
        tetvox.setup_simplices(positions64, simplices, out=setup32)  # type: ignore[misc]
    with pytest.raises(TypeError):
        tetvox.setup_simplices(positions, simplices.astype(np.int64))


def test_out_arrays_are_reused() -> None:
    positions, simplices = kuhn_cube()
    setup = tetvox.setup_simplices(positions, simplices)
    assert tetvox.setup_simplices(positions, simplices, out=setup) is setup
    out = (np.full((3, 3, 3), 7, np.int32), np.full((3, 3, 3, 3), 7, np.float32))
    simplex_id, bary = tetvox.rasterize(setup, (3, 3, 3), out=out)
    assert simplex_id is out[0] and bary is out[1]
    assert (simplex_id[2] == -1).all() and not bary[2].any()  # every element is written


def atlas_grid_shape(positions: npt.NDArray[np.floating]) -> tuple[int, int, int]:
    """The template grid: the atlas box is [0, max]^3 with integer corners (atlas-format.md)."""
    assert np.min(positions) == 0
    top = positions.max(axis=0)
    assert np.array_equal(top, np.round(top))
    x, y, z = (int(v) + 1 for v in top)
    return x, y, z


def test_atlas_owned_exactly_once(distinct_atlas: Path) -> None:
    """Exhaustive: on its template grid, every voxel centre of the half-open box is owned by
    exactly one tet and the far layers x_d = max_d by none (ADR 0006, 0008)."""
    mesh = tetvox.load_atlas_npz(distinct_atlas).mesh
    shape = atlas_grid_shape(mesh.positions)
    setup = tetvox.setup_simplices(mesh.positions, mesh.simplices)
    assert tetvox.count_skipped(setup) == tetvox.SkippedSimplices(0, 0)
    claims = tetvox.count_claims(setup, shape)
    expected = np.zeros(shape, np.int32)
    expected[:-1, :-1, :-1] = 1
    np.testing.assert_array_equal(claims, expected)
    simplex_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(simplex_id >= 0, claims == 1)
    assert_barycentrics_valid(simplex_id, bary)


def test_adult_atlas_pi_against_reference(adult_atlas: Path) -> None:
    """pi at a random sample of owned voxels, both dtypes, against exact Fractions; ownership does
    not depend on the dtype (the positions are the same numbers)."""
    mesh32 = tetvox.load_atlas_npz(adult_atlas).mesh
    mesh64 = tetvox.load_atlas_npz(adult_atlas, dtype=np.float64).mesh
    shape = atlas_grid_shape(mesh32.positions)
    simplex_id, bary32 = tetvox.rasterize(
        tetvox.setup_simplices(mesh32.positions, mesh32.simplices), shape
    )
    simplex_id64, bary64 = tetvox.rasterize(
        tetvox.setup_simplices(mesh64.positions, mesh64.simplices), shape
    )
    np.testing.assert_array_equal(simplex_id, simplex_id64)
    owned = np.argwhere(simplex_id >= 0)
    sample = owned[np.random.default_rng(0).choice(len(owned), size=2000, replace=False)]
    for voxel in sample:
        i0, i1, i2 = (int(v) for v in voxel)
        t = int(simplex_id[i0, i1, i2])
        exact = ref.owned_pi(
            ref.setup_simplex(mesh64.positions[mesh64.simplices[t]], K), (i0, i1, i2), K
        )
        assert exact is not None
        expected = [float(x) for x in exact[1:]]
        np.testing.assert_allclose(bary32[i0, i1, i2], expected, rtol=0, atol=PI_ATOL[np.float32])
        np.testing.assert_allclose(bary64[i0, i1, i2], expected, rtol=0, atol=PI_ATOL[np.float64])
