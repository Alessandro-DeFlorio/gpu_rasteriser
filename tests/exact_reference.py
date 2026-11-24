"""Pure-Python exact reference of the rasterisation rule (ADR 0006-0010), for small meshes.

Python integers and Fractions, so nothing overflows or rounds. Works for any dimension D.
Independent of the C++ path where it matters: plane functions are signed volumes (determinants),
ownership is the explicit SoS rule of ADR 0006 (E_j > 0, or E_j = 0 and the first non-zero
component of n_j positive, axis 0 first), and pi_j = E_j / D is an exact Fraction. The record fields
(n_j, c_j, f_j, box) are the C++ contract itself, so they follow its formulas (R1 §2.2, §2.5).
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import numpy.typing as npt

SNAP_RANGE_BITS = 62  # |v| <= 2**(62 - k) (ADR 0008)
OK, INVERTED, OUTSIDE_WINDOW = 0, 1, 2

Point = tuple[int, ...]


def int64_budget_bits(dim: int) -> int:
    """(D - 1) k + D L <= 62 - floor(log2(2 D!)) (ADR 0010; 59 for D = 3 as in ADR 0006)."""
    floor_log2 = (2 * math.factorial(dim)).bit_length() - 1
    return 62 - floor_log2


def window_bits(dim: int, k: int) -> int:
    """L: the largest window exponent within the budget, with k + L <= 63 (ADR 0010)."""
    return min((int64_budget_bits(dim) - (dim - 1) * k) // dim, 63 - k)


def snap(v: float, k: int) -> int:
    """round(v * 2**k), ties to even (Python's round on a Fraction)."""
    return round(Fraction(v) * (1 << k))


def det(rows: list[list[int]]) -> int:
    """Determinant of a square integer matrix by cofactor expansion along row 0."""
    if len(rows) == 1:
        return rows[0][0]
    return sum(
        (-1) ** c * rows[0][c] * det([row[:c] + row[c + 1 :] for row in rows[1:]])
        for c in range(len(rows))
    )


def signed_volume(p: Sequence[Point]) -> int:
    """det[p_1 - p_0, ..., p_D - p_0] (D! times the signed volume)."""
    return det([[a - b for a, b in zip(p[i], p[0], strict=True)] for i in range(1, len(p))])


@dataclass
class RefSimplex:
    """Exact per-simplex data. Fields other than status are set only for OK simplices."""

    status: int
    q: list[Point] = field(default_factory=list)
    det: int = 0
    normals: list[Point] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    fractions: list[Fraction] = field(default_factory=list)
    box_lo: Point = ()
    box_hi: Point = ()

    def plane(self, j: int, p: Point) -> int:
        """E_j(p): the signed volume with vertex j replaced by the lattice point p."""
        return signed_volume([p if i == j else self.q[i] for i in range(len(self.q))])


def lex_positive(n: Point) -> bool:
    """First non-zero component positive, in axis order 0, 1, ..."""
    return next((c > 0 for c in n if c != 0), False)


def setup_simplex(vertices: npt.NDArray[np.floating], k: int) -> RefSimplex:
    """Classify one simplex (ADR 0008 order) and compute its record. vertices: (D + 1, D)."""
    dim = vertices.shape[1]
    s = 1 << k
    bound = 1 << (SNAP_RANGE_BITS - k)
    coords = [float(x) for x in vertices.ravel()]
    if not all(math.isfinite(x) and abs(x) <= bound for x in coords):
        return RefSimplex(OUTSIDE_WINDOW)
    q = [tuple(snap(c, k) for c in v) for v in vertices.tolist()]
    window = 1 << (k + window_bits(dim, k))
    lo = tuple(min(p[d] for p in q) for d in range(dim))
    hi = tuple(max(p[d] for p in q) for d in range(dim))
    if any(hi[d] - lo[d] > window for d in range(dim)):
        return RefSimplex(OUTSIDE_WINDOW)
    volume = signed_volume(q)
    if volume <= 0:
        return RefSimplex(INVERTED)
    simplex = RefSimplex(OK, q=q, det=volume)
    simplex.box_lo = tuple(-(-c // s) for c in lo)
    simplex.box_hi = tuple(c // s for c in hi)
    origin = tuple(c * s for c in simplex.box_lo)
    zero = (0,) * dim
    units = [tuple(int(a == b) for b in range(dim)) for a in range(dim)]
    for j in range(dim + 1):
        e0 = simplex.plane(j, zero)
        normal = tuple(simplex.plane(j, unit) - e0 for unit in units)
        at_origin = simplex.plane(j, origin)
        offset = (at_origin - 1 + int(lex_positive(normal))) // s
        simplex.normals.append(normal)
        simplex.offsets.append(offset)
        simplex.fractions.append(Fraction(at_origin - s * offset, s))
    return simplex


def setup_simplices(
    positions: npt.NDArray[np.floating], simplices: npt.NDArray[np.int32], k: int
) -> list[RefSimplex]:
    """setup_simplex for every simplex. positions (N, D), simplices (T, D + 1)."""
    return [setup_simplex(positions[t], k) for t in simplices]


def bary_scale(simplex: RefSimplex, k: int, dtype: type[np.floating]) -> np.floating:
    """2**k / D as the C++ computes it: correctly rounded to double, then cast to dtype."""
    scale: np.floating = dtype((1 << k) / float(simplex.det))
    return scale


@dataclass
class RefRaster:
    """Exact rasterisation of a grid: claims and, per claimed voxel, its last claimant and pi."""

    claims: npt.NDArray[np.int32]
    owner: npt.NDArray[np.int32]
    pi: npt.NDArray[np.float64]  # (*shape, D + 1), exact values rounded once to float64


def owned_pi(simplex: RefSimplex, voxel: Point, k: int) -> list[Fraction] | None:
    """Exact pi_0..pi_D if the SoS rule gives the voxel centre to the simplex, else None."""
    p = tuple(c * (1 << k) for c in voxel)
    planes = [simplex.plane(j, p) for j in range(len(simplex.q))]
    inside = all(
        e > 0 or (e == 0 and lex_positive(simplex.normals[j])) for j, e in enumerate(planes)
    )
    return [Fraction(e, simplex.det) for e in planes] if inside else None


def rasterize(simplices: list[RefSimplex], shape: tuple[int, ...], k: int) -> RefRaster:
    """Visit every voxel of every OK simplex's grid-clipped box and apply the explicit rule."""
    claims = np.zeros(shape, np.int32)
    owner = np.full(shape, -1, np.int32)
    pi = np.zeros((*shape, len(shape) + 1), np.float64)
    for t, simplex in enumerate(simplices):
        if simplex.status != OK:
            continue
        ranges = [
            range(max(lo, 0), min(hi, extent - 1) + 1)
            for lo, hi, extent in zip(simplex.box_lo, simplex.box_hi, shape, strict=True)
        ]
        for voxel in itertools.product(*ranges):
            exact = owned_pi(simplex, voxel, k)
            if exact is not None:
                claims[voxel] += 1
                owner[voxel] = t
                pi[voxel] = [float(x) for x in exact]
    return RefRaster(claims, owner, pi)
