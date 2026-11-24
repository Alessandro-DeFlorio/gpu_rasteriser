"""Rasterisation ops (ADR 0009, 0010): per-simplex setup, then ownership and barycentrics.

Each function wraps the C++ op of the same name. Arrays are passed through without conversion; a
wrong dtype, layout, device or number of array dimensions raises `TypeError`, and invalid values
(including shapes that disagree with the simplex dimension) raise `ValueError` (ADR 0008). With
`out=None` the outputs are allocated as numpy arrays; other frameworks pass `out=`.

Every op works for D = 1, 2 and 3 (segments, triangles, tetrahedra); D is read from the arrays.
Grids are in voxel index space: voxel (i_0, ..., i_{D-1}) has its centre at that integer point, and
position column d is grid axis d (C order, the last axis fastest; ADR 0007, 0010).
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


class SimplexStatus(IntEnum):
    """Classification of a simplex by `setup_simplices` (ADR 0008)."""

    OK = 0
    INVERTED = 1
    OUTSIDE_WINDOW = 2


class SkippedSimplices(NamedTuple):
    """Number of simplices of each skipped status."""

    inverted: int
    outside_window: int


@dataclass(frozen=True)
class SimplexSetup(Generic[F]):
    """Per-simplex record written by `setup_simplices` (ADR 0009, 0010): T simplices of dimension D.

    For a simplex whose status is not OK, every field other than `status` is unspecified.

    Attributes:
        normals: (T, D + 1, D) int64 inward face normals n_j, in lattice units of 2**-k voxel.
        offsets: (T, D + 1) int64 plane offsets c_j, with the ownership tie-break folded in.
        fractions: (T, D + 1) F, f_j in [0, 1].
        bary_scale: (T,) F, 2**k / D; n_j * bary_scale is the gradient of pi_j in voxel units.
        box_lo: (T, D) int64 integer origin O = ceil(min q / 2**k), not clipped to any grid.
        box_hi: (T, D) int64 floor(max q / 2**k).
        status: (T,) uint8 `SimplexStatus` values.
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


def _arrays(setup: SimplexSetup[F]) -> _SetupArrays[F]:
    """The fields of `setup` in declaration order, as the C++ ops take them."""
    s = setup
    return (s.normals, s.offsets, s.fractions, s.bary_scale, s.box_lo, s.box_hi, s.status)


def _empty(like: object, shape: tuple[int, ...], dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """An uninitialised output array in the framework of `like` (numpy only for now)."""
    if isinstance(like, np.ndarray):
        return np.empty(shape, dtype=dtype)
    raise TypeError(f"cannot allocate outputs for {type(like).__name__}; pass out=")


def _dim(setup: SimplexSetup[F]) -> int:
    """The simplex dimension D of `setup`."""
    return int(setup.box_lo.shape[1])


def setup_simplices(
    positions: npt.NDArray[F],
    simplices: npt.NDArray[np.int32],
    *,
    snap_bits: int = DEFAULT_SNAP_BITS,
    out: SimplexSetup[F] | None = None,
) -> SimplexSetup[F]:
    """Snap the vertices to 2**-snap_bits voxel, classify every simplex and compute its record.

    Args:
        positions: (N, D) float32 or float64 vertex positions in voxel index space, D in 1, 2, 3.
        simplices: (T, D + 1) int32 vertex indices; simplices must be positively oriented,
            det[v_1 - v_0, ..., v_D - v_0] > 0 in axis order.
        snap_bits: k, in [0, MAX_SNAP_BITS].
        out: a `SimplexSetup` for T simplices of dimension D, of the dtype of `positions`, to write
            into.

    Returns:
        The record (`out` if given). Count skipped simplices with `count_skipped`.
    """
    if out is None:
        num, dim = simplices.shape[0], positions.shape[1]
        out = SimplexSetup(
            normals=_empty(positions, (num, dim + 1, dim), np.int64),
            offsets=_empty(positions, (num, dim + 1), np.int64),
            fractions=_empty(positions, (num, dim + 1), positions.dtype),
            bary_scale=_empty(positions, (num,), positions.dtype),
            box_lo=_empty(positions, (num, dim), np.int64),
            box_hi=_empty(positions, (num, dim), np.int64),
            status=_empty(positions, (num,), np.uint8),
        )
    _core.setup_simplices(positions, simplices, snap_bits, *_arrays(out))
    return out


def count_skipped(setup: SimplexSetup[F]) -> SkippedSimplices:
    """Number of inverted (or degenerate) and out-of-window simplices in `setup`."""
    inverted, outside_window = _core.count_skipped(setup.status)
    return SkippedSimplices(inverted, outside_window)


def locate(
    setup: SimplexSetup[F],
    shape: tuple[int, ...],
    *,
    out: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32]:
    """The simplex owning each voxel centre: `shape` (D extents) int32, -1 where none does."""
    simplex_id = _empty(setup.status, shape, np.int32) if out is None else out
    _core.locate(*_arrays(setup), simplex_id)
    return simplex_id


def barycentrics(
    setup: SimplexSetup[F],
    simplex_id: npt.NDArray[np.int32],
    *,
    out: npt.NDArray[F] | None = None,
) -> npt.NDArray[F]:
    """Barycentrics pi_1..pi_D of each voxel centre in the simplex `simplex_id` names.

    Returns:
        (*simplex_id.shape, D) in the dtype of `setup`; 0 where `simplex_id` is -1. Pass
        `simplex_id` from `locate`.
    """
    shape = (*simplex_id.shape, _dim(setup))
    bary = _empty(simplex_id, shape, setup.fractions.dtype) if out is None else out
    _core.barycentrics(*_arrays(setup), simplex_id, bary)
    return bary


def rasterize(
    setup: SimplexSetup[F],
    shape: tuple[int, ...],
    *,
    out: tuple[npt.NDArray[np.int32], npt.NDArray[F]] | None = None,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[F]]:
    """`locate` and `barycentrics` in one pass, equal to them bit for bit.

    Returns:
        simplex_id `shape` int32, -1 where unowned, and bary (*shape, D) pi_1..pi_D in the dtype
        of `setup`, 0 where unowned.
    """
    if out is None:
        out = (
            _empty(setup.status, shape, np.int32),
            _empty(setup.status, (*shape, _dim(setup)), setup.fractions.dtype),
        )
    _core.rasterize(*_arrays(setup), *out)
    return out


def count_claims(
    setup: SimplexSetup[F],
    shape: tuple[int, ...],
    *,
    out: npt.NDArray[np.int32] | None = None,
) -> npt.NDArray[np.int32]:
    """Diagnostic: how many simplices claim each voxel centre, `shape` int32; 1 where a mesh
    tiles."""
    claims = _empty(setup.status, shape, np.int32) if out is None else out
    _core.count_claims(*_arrays(setup), claims)
    return claims
