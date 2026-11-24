"""CPU rasterisation ops (ADR 0006-0009) against the exact reference and invariants."""

from pathlib import Path
from typing import Any

import exact_reference as ref
import numpy as np
import numpy.typing as npt
import pytest
from conftest import kuhn_cube, kuhn_grid

import tetvox
from tetvox import TetSetup, TetStatus

# Absolute tolerance of pi against the exact value (ADR 0008).
PI_ATOL = {np.float32: 1e-6, np.float64: 1e-12}
# Owned voxels satisfy pi_1..pi_3 >= 0 and 1 - sum(pi) >= -SUM_TOL (ADR 0008).
SUM_TOL = 1e-6
DTYPES = [np.float32, np.float64]
K = tetvox.DEFAULT_SNAP_BITS

Mesh = tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], tuple[int, int, int]]


def rotation(seed: int) -> npt.NDArray[np.float64]:
    """A random rotation matrix (3, 3)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def small_meshes() -> dict[str, Mesh]:
    """Small meshes for the reference comparison: positions (N, 3), tetrahedra, grid shape."""
    positions, tets = kuhn_grid(2)
    rng = np.random.default_rng(0)
    jitter = positions * 2.0
    interior = np.all((positions > 0) & (positions < 2), axis=1)
    jitter[interior] += rng.uniform(-0.3, 0.3, size=(int(interior.sum()), 3))
    rotated = (positions - 1.0) * 2.7 @ rotation(1).T + 1.2  # partly outside the grid
    return {
        "kuhn_on_centres": (positions, tets, (4, 4, 4)),
        "kuhn_scaled_3": (positions * 3.0, tets, (8, 8, 8)),
        "jittered": (jitter, tets, (6, 6, 6)),
        "rotated_partly_outside": (rotated, tets, (6, 6, 6)),
    }


SMALL = small_meshes()


@pytest.fixture(params=list(SMALL))
def small_mesh(request: pytest.FixtureRequest) -> Mesh:
    return SMALL[request.param]


def assert_setup_matches(
    setup: TetSetup[Any], expected: list[ref.RefTet], snap_bits: int = K
) -> None:
    """Every field of every OK tet, and every status, bit for bit."""
    np.testing.assert_array_equal(setup.status, [t.status for t in expected])
    dtype = setup.fractions.dtype.type
    for i, tet in enumerate(expected):
        if tet.status != ref.OK:
            continue
        np.testing.assert_array_equal(setup.normals[i], tet.normals)
        np.testing.assert_array_equal(setup.offsets[i], tet.offsets)
        np.testing.assert_array_equal(setup.fractions[i], [dtype(f) for f in tet.fractions])
        assert setup.bary_scale[i] == ref.bary_scale(tet, snap_bits, dtype)
        np.testing.assert_array_equal(setup.box_lo[i], tet.box_lo)
        np.testing.assert_array_equal(setup.box_hi[i], tet.box_hi)


@pytest.mark.parametrize("dtype", DTYPES)
def test_setup_matches_reference(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, tets, _ = small_mesh
    setup = tetvox.setup_tets(positions.astype(dtype), tets)
    assert_setup_matches(setup, ref.setup_tets(positions.astype(dtype), tets, K))


@pytest.mark.parametrize("dtype", DTYPES)
def test_raster_matches_reference(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, tets, shape = small_mesh
    p = positions.astype(dtype)
    setup = tetvox.setup_tets(p, tets)
    expected = ref.rasterize(ref.setup_tets(p, tets, K), shape, K)
    claims = tetvox.count_claims(setup, shape)
    np.testing.assert_array_equal(claims, expected.claims)
    assert np.max(claims) <= 1  # these meshes tile their hull
    tet_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(tet_id, expected.owner)
    np.testing.assert_allclose(bary, expected.pi[..., 1:], rtol=0, atol=PI_ATOL[dtype])
    assert bary.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_rasterize_equals_locate_then_barycentrics(dtype: type[np.floating]) -> None:
    positions, tets, shape = SMALL["rotated_partly_outside"]
    setup = tetvox.setup_tets(positions.astype(dtype), tets)
    tet_id, bary = tetvox.rasterize(setup, shape)
    located = tetvox.locate(setup, shape)
    np.testing.assert_array_equal(tet_id, located)
    chained = tetvox.barycentrics(setup, located)
    assert bary.tobytes() == chained.tobytes()


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_kuhn_grid_on_voxel_centres_owned_exactly_once(scale: int) -> None:
    """Every vertex on a voxel centre: the most ties. The half-open box [0, 3·scale)^3 is owned
    exactly once; the far layers and everything outside the mesh are unowned."""
    positions, tets = kuhn_grid(3)
    extent = 3 * scale
    shape = (extent + 2, extent + 2, extent + 2)
    claims = tetvox.count_claims(tetvox.setup_tets(positions * scale, tets), shape)
    expected = np.zeros(shape, np.int32)
    expected[:extent, :extent, :extent] = 1
    np.testing.assert_array_equal(claims, expected)


def assert_barycentrics_valid(
    tet_id: npt.NDArray[np.int32], bary: npt.NDArray[np.floating]
) -> None:
    owned = bary[tet_id >= 0]
    assert np.min(owned) >= 0
    assert np.min(1 - owned.sum(axis=1)) >= -SUM_TOL
    assert not bary[tet_id < 0].any()


@pytest.mark.parametrize("dtype", DTYPES)
def test_barycentric_bounds(small_mesh: Mesh, dtype: type[np.floating]) -> None:
    positions, tets, shape = small_mesh
    tet_id, bary = tetvox.rasterize(tetvox.setup_tets(positions.astype(dtype), tets), shape)
    assert_barycentrics_valid(tet_id, bary)


def test_statuses_and_counts() -> None:
    """One tet of each kind; skipped tets never claim a voxel."""
    cube, tets = kuhn_cube()
    base = cube.astype(np.float64) * 2.0
    span_too_wide = base.copy()
    span_too_wide[7, 0] = 1025.0  # span 1025 voxels > 2^10 at k = 14
    huge = base.copy()
    huge[7] = 2.0**49  # |v| > 2^(62 - 14) = 2^48
    flat = base.copy()
    flat[:, 2] = 0.0  # every tet degenerate
    variants = {
        TetStatus.OK: base,
        TetStatus.INVERTED: flat,
        TetStatus.OUTSIDE_WINDOW: span_too_wide,
    }
    nonfinite = [base.copy(), base.copy(), huge]
    nonfinite[0][7, 1] = np.nan
    nonfinite[1][7, 2] = np.inf
    blocks = [*variants.values(), *nonfinite]
    positions = np.concatenate(blocks)
    tetrahedra = np.concatenate([tets[:1] + 8 * i for i in range(len(blocks))])
    inverted = np.concatenate([tets[:1, [0, 2, 1, 3]], tetrahedra])  # swapped: D < 0
    setup = tetvox.setup_tets(positions, inverted)
    expected = [1, 0, 1, 2, 2, 2, 2]
    np.testing.assert_array_equal(setup.status, expected)
    assert tetvox.count_skipped(setup) == tetvox.SkippedTets(inverted=2, outside_window=4)
    claims = tetvox.count_claims(setup, (4, 4, 4))
    only_ok = ref.rasterize(ref.setup_tets(positions[:8], tets[:1], K), (4, 4, 4), K)
    assert claims.sum() > 0
    np.testing.assert_array_equal(claims, only_ok.claims)


@pytest.mark.parametrize("dtype", DTYPES)
def test_bit_budget_boundary(dtype: type[np.floating]) -> None:
    """A Kuhn cube spanning exactly 2^10 voxels (the <= boundary), sampled near its far vertex,
    where t is largest; one lattice unit more is outside the window."""
    cube, tets = kuhn_cube()
    positions = (cube.astype(np.float64) * 1024.0 - 1021.0).astype(dtype)  # far vertex at 3
    shape = (5, 5, 5)
    setup = tetvox.setup_tets(positions, tets)
    expected_tets = ref.setup_tets(positions, tets, K)
    assert_setup_matches(setup, expected_tets)
    assert all(t.status == ref.OK for t in expected_tets)
    expected = ref.rasterize(expected_tets, shape, K)
    tet_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(tet_id, expected.owner)
    np.testing.assert_array_equal(tetvox.count_claims(setup, shape), expected.claims)
    np.testing.assert_allclose(bary, expected.pi[..., 1:], rtol=0, atol=PI_ATOL[dtype])

    beyond = positions.copy()
    beyond[7, 0] += dtype(2.0**-K)
    assert (tetvox.setup_tets(beyond, tets).status == TetStatus.OUTSIDE_WINDOW).all()


@pytest.mark.parametrize("snap_bits", [-1, tetvox.MAX_SNAP_BITS + 1])
def test_snap_bits_out_of_range_rejected(snap_bits: int) -> None:
    positions, tets = kuhn_cube()
    with pytest.raises(ValueError, match="snap_bits"):
        tetvox.setup_tets(positions, tets, snap_bits=snap_bits)


@pytest.mark.parametrize("snap_bits", [0, tetvox.MAX_SNAP_BITS])
def test_snap_bits_range_ends_accepted(snap_bits: int) -> None:
    positions, tets = kuhn_grid(2)
    setup = tetvox.setup_tets(positions, tets, snap_bits=snap_bits)
    assert_setup_matches(setup, ref.setup_tets(positions, tets, snap_bits), snap_bits)
    claims = tetvox.count_claims(setup, (3, 3, 3))
    assert np.min(claims[:2, :2, :2]) == 1 and claims.sum() == 8


def test_tet_index_out_of_range_rejected() -> None:
    positions, tets = kuhn_cube()
    bad = tets.copy()
    bad[0, 0] = 8
    with pytest.raises(ValueError, match="tet index"):
        tetvox.setup_tets(positions, bad)


def test_barycentrics_rejects_invalid_tet_id() -> None:
    """Provisional (ROADMAP open question 2): ids outside [-1, T), or whose box misses the voxel."""
    positions, tets = kuhn_cube()
    setup = tetvox.setup_tets(positions, tets)
    tet_id = tetvox.locate(setup, (3, 3, 3))
    for bad_value, voxel in [(6, (0, 0, 0)), (-2, (0, 0, 0)), (0, (2, 2, 2))]:
        bad = tet_id.copy()
        bad[voxel] = bad_value
        with pytest.raises(ValueError, match="tet_id"):
            tetvox.barycentrics(setup, bad)


def test_mixed_dtypes_rejected() -> None:
    positions, tets = kuhn_cube()
    setup32 = tetvox.setup_tets(positions, tets)
    with pytest.raises(TypeError):
        tetvox.setup_tets(positions.astype(np.float64), tets, out=setup32)  # type: ignore[misc]
    with pytest.raises(TypeError):
        tetvox.setup_tets(positions, tets.astype(np.int64))


def test_out_arrays_are_reused() -> None:
    positions, tets = kuhn_cube()
    setup = tetvox.setup_tets(positions, tets)
    assert tetvox.setup_tets(positions, tets, out=setup) is setup
    out = (np.full((3, 3, 3), 7, np.int32), np.full((3, 3, 3, 3), 7, np.float32))
    tet_id, bary = tetvox.rasterize(setup, (3, 3, 3), out=out)
    assert tet_id is out[0] and bary is out[1]
    assert (tet_id[2] == -1).all() and not bary[2].any()  # every element is written


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
    setup = tetvox.setup_tets(mesh.positions, mesh.tetrahedra)
    assert tetvox.count_skipped(setup) == tetvox.SkippedTets(0, 0)
    claims = tetvox.count_claims(setup, shape)
    expected = np.zeros(shape, np.int32)
    expected[:-1, :-1, :-1] = 1
    np.testing.assert_array_equal(claims, expected)
    tet_id, bary = tetvox.rasterize(setup, shape)
    np.testing.assert_array_equal(tet_id >= 0, claims == 1)
    assert_barycentrics_valid(tet_id, bary)


def test_adult_atlas_pi_against_reference(adult_atlas: Path) -> None:
    """pi at a random sample of owned voxels, both dtypes, against exact Fractions; ownership does
    not depend on the dtype (the positions are the same numbers)."""
    mesh32 = tetvox.load_atlas_npz(adult_atlas).mesh
    mesh64 = tetvox.load_atlas_npz(adult_atlas, dtype=np.float64).mesh
    shape = atlas_grid_shape(mesh32.positions)
    tet_id, bary32 = tetvox.rasterize(tetvox.setup_tets(mesh32.positions, mesh32.tetrahedra), shape)
    tet_id64, bary64 = tetvox.rasterize(
        tetvox.setup_tets(mesh64.positions, mesh64.tetrahedra), shape
    )
    np.testing.assert_array_equal(tet_id, tet_id64)
    owned = np.argwhere(tet_id >= 0)
    sample = owned[np.random.default_rng(0).choice(len(owned), size=2000, replace=False)]
    for voxel in sample:
        i0, i1, i2 = (int(v) for v in voxel)
        t = int(tet_id[i0, i1, i2])
        exact = ref.owned_pi(
            ref.setup_tet(mesh64.positions[mesh64.tetrahedra[t]], K), (i0, i1, i2), K
        )
        assert exact is not None
        expected = [float(x) for x in exact[1:]]
        np.testing.assert_allclose(bary32[i0, i1, i2], expected, rtol=0, atol=PI_ATOL[np.float32])
        np.testing.assert_allclose(bary64[i0, i1, i2], expected, rtol=0, atol=PI_ATOL[np.float64])
