"""Rasterisation ops (ADR 0009): per-tet setup, then ownership and barycentrics on a voxel grid.

Each function wraps the C++ op of the same name. Arrays are passed through without conversion; a
wrong dtype, layout or device raises `TypeError`, and invalid values raise `ValueError` (ADR 0008).
With `out=None` the outputs are allocated as numpy arrays; other frameworks pass `out=`.

Grids are in voxel index space: voxel (i0, i1, i2) has its centre at that integer point, and
position column d is grid axis d (C order, axis 2 fastest; ADR 0007).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Final, Generic, NamedTuple

import numpy as np
import numpy.typing as npt

from tetvox import _core
from tetvox._typing import F

DEFAULT_SNAP_BITS: Final[int] = _core.default_snap_bits
"""k of ADR 0006: vertices are snapped to multiples of 2**-k voxel."""

MAX_SNAP_BITS: Final[int] = _core.max_snap_bits
"""Largest accepted `snap_bits` (ADR 0009)."""


class TetStatus(IntEnum):
    """Classification of a tet by `setup_tets` (ADR 0008)."""

    OK = 0
    INVERTED = 1
    OUTSIDE_WINDOW = 2


class SkippedTets(NamedTuple):
    """Number of tets of each skipped status."""

    inverted: int
    outside_window: int


@dataclass(frozen=True)
class TetSetup(Generic[F]):
    """Per-tet record written by `setup_tets` (ADR 0009), one array per field, T tets.

    For a tet whose status is not OK, every field other than `status` is unspecified.

    Attributes:
        normals: (T, 4, 3) int64 inward face normals n_j, in lattice units of 2**-k voxel.
        offsets: (T, 4) int64 plane offsets c_j, with the ownership tie-break folded in.
        fractions: (T, 4) F, f_j in [0, 1].
        bary_scale: (T,) F, 2**k / D; n_j * bary_scale is the gradient of pi_j in voxel units.
        box_lo: (T, 3) int64 integer origin O = ceil(min q / 2**k), not clipped to any grid.
        box_hi: (T, 3) int64 floor(max q / 2**k).
        status: (T,) uint8 `TetStatus` values.
    """

    normals: npt.NDArray[np.int64]
    offsets: npt.NDArray[np.int64]
    fractions: npt.NDArray[F]
    bary_scale: npt.NDArray[F]
    box_lo: npt.NDArray[np.int64]
    box_hi: npt.NDArray[np.int64]
    status: npt.NDArray[np.uint8]


_SetupArrays = tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[F],
    npt.NDArray[F],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.uint8],
]


def _arrays(setup: TetSetup[F]) -> _SetupArrays[F]:
    """The fields of `setup` in declaration order, as the C++ ops take them."""
    s = setup
    return (s.normals, s.offsets, s.fractions, s.bary_scale, s.box_lo, s.box_hi, s.status)


def _empty(like: object, shape: tuple[int, ...], dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """An uninitialised output array in the framework of `like` (numpy only for now)."""
    if isinstance(like, np.ndarray):
        return np.empty(shape, dtype=dtype)
    raise TypeError(f"cannot allocate outputs for {type(like).__name__}; pass out=")


def setup_tets(
    positions: npt.NDArray[F],
    tetrahedra: npt.NDArray[np.int32],
    *,
    snap_bits: int = DEFAULT_SNAP_BITS,
    out: TetSetup[F] | None = None,
) -> TetSetup[F]:
    """Snap the vertices to 2**-snap_bits voxel, classify every tet and compute its record.

    Args:
        positions: (N, 3) float32 or float64 vertex positions in voxel index space.
        tetrahedra: (T, 4) int32 vertex indices; tets must be positively oriented.
        snap_bits: k, in [0, MAX_SNAP_BITS].
        out: a `TetSetup` for T tets, of the dtype of `positions`, to write into.

    Returns:
        The record (`out` if given). Count skipped tets with `count_skipped`.
    """
    if out is None:
        num_tets = tetrahedra.shape[0]
        out = TetSetup(
            normals=_empty(positions, (num_tets, 4, 3), np.int64),
            offsets=_empty(positions, (num_tets, 4), np.int64),
            fractions=_empty(positions, (num_tets, 4), positions.dtype),
            bary_scale=_empty(positions, (num_tets,), positions.dtype),
            box_lo=_empty(positions, (num_tets, 3), np.int64),
            box_hi=_empty(positions, (num_tets, 3), np.int64),
            status=_empty(positions, (num_tets,), np.uint8),
        )
    _core.setup_tets(positions, tetrahedra, snap_bits, *_arrays(out))
    return out


def count_skipped(setup: TetSetup[F]) -> SkippedTets:
    """Number of inverted (or degenerate) and out-of-window tets in `setup`."""
    inverted, outside_window = _core.count_skipped(setup.status)
    return SkippedTets(inverted, outside_window)


def locate(
    setup: TetSetup[F],
    shape: tuple[int, int, int],
    *,
    out: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32]:
    """The tet owning each voxel centre: (X, Y, Z) int32, -1 where no tet does."""
    tet_id = _empty(setup.status, shape, np.int32) if out is None else out
    _core.locate(*_arrays(setup), tet_id)
    return tet_id


def barycentrics(
    setup: TetSetup[F],
    tet_id: npt.NDArray[np.int32],
    *,
    out: npt.NDArray[F] | None = None,
) -> npt.NDArray[F]:
    """Barycentrics pi_1..pi_3 of each voxel centre in the tet `tet_id` (X, Y, Z) names.

    Returns:
        (X, Y, Z, 3) in the dtype of `setup`; 0 where `tet_id` is -1. Pass `tet_id` from `locate`.
    """
    bary = _empty(tet_id, (*tet_id.shape, 3), setup.fractions.dtype) if out is None else out
    _core.barycentrics(*_arrays(setup), tet_id, bary)
    return bary


def rasterize(
    setup: TetSetup[F],
    shape: tuple[int, int, int],
    *,
    out: tuple[npt.NDArray[np.int32], npt.NDArray[F]] | None = None,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[F]]:
    """`locate` and `barycentrics` in one pass, equal to them bit for bit.

    Returns:
        tet_id (X, Y, Z) int32, -1 where unowned, and bary (X, Y, Z, 3) pi_1..pi_3 in the dtype
        of `setup`, 0 where unowned.
    """
    if out is None:
        out = (
            _empty(setup.status, shape, np.int32),
            _empty(setup.status, (*shape, 3), setup.fractions.dtype),
        )
    _core.rasterize(*_arrays(setup), *out)
    return out


def count_claims(
    setup: TetSetup[F],
    shape: tuple[int, int, int],
    *,
    out: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32]:
    """Diagnostic: how many tets claim each voxel centre, (X, Y, Z) int32; 1 where a mesh tiles."""
    claims = _empty(setup.status, shape, np.int32) if out is None else out
    _core.count_claims(*_arrays(setup), claims)
    return claims
