"""Pure-Python exact reference of the rasterisation rule (ADR 0006-0009), for small meshes.

Python integers and Fractions, so nothing overflows or rounds. Independent of the C++ path where it
matters: plane functions are signed volumes (determinants), ownership is the explicit SoS rule of
ADR 0006 (E_j > 0, or E_j = 0 and the first non-zero component of n_j positive, axis 0 first), and
pi_j = E_j / D is an exact Fraction. The record fields (n_j, c_j, f_j, box) are the C++ contract
itself, so they follow its formulas (R1 §2.2, §2.5).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import numpy.typing as npt

INT64_BUDGET_BITS = 59  # 2k + 3L <= 59 (ADR 0006)
SNAP_RANGE_BITS = 62  # |v| <= 2**(62 - k) (ADR 0008)
OK, INVERTED, OUTSIDE_WINDOW = 0, 1, 2

Point = tuple[int, int, int]


def snap(v: float, k: int) -> int:
    """round(v * 2**k), ties to even (Python's round on a Fraction)."""
    return round(Fraction(v) * (1 << k))


def signed_volume(p: Sequence[Point]) -> int:
    """det[p1 - p0, p2 - p0, p3 - p0] (6 times the signed volume)."""
    a, b, c = ([p[i][d] - p[0][d] for d in range(3)] for i in (1, 2, 3))
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@dataclass
class RefTet:
    """Exact per-tet data. Fields other than status are set only for OK tets."""

    status: int
    q: list[Point] = field(default_factory=list)
    det: int = 0
    normals: list[Point] = field(default_factory=list)
    offsets: list[int] = field(default_factory=list)
    fractions: list[Fraction] = field(default_factory=list)
    box_lo: Point = (0, 0, 0)
    box_hi: Point = (0, 0, 0)

    def plane(self, j: int, p: Point) -> int:
        """E_j(p): the signed volume with vertex j replaced by the lattice point p."""
        return signed_volume([p if i == j else self.q[i] for i in range(4)])


def lex_positive(n: Point) -> bool:
    """First non-zero component positive, in axis order 0, 1, 2."""
    return next((c > 0 for c in n if c != 0), False)


def setup_tet(vertices: npt.NDArray[np.floating], k: int) -> RefTet:
    """Classify one tet (ADR 0008 order) and compute its record. vertices: (4, 3)."""
    s = 1 << k
    bound = 1 << (SNAP_RANGE_BITS - k)
    coords = [float(x) for x in vertices.ravel()]
    if not all(math.isfinite(x) and abs(x) <= bound for x in coords):
        return RefTet(OUTSIDE_WINDOW)
    q = [(snap(v[0], k), snap(v[1], k), snap(v[2], k)) for v in vertices.tolist()]
    window = 1 << (k + (INT64_BUDGET_BITS - 2 * k) // 3)
    lo = tuple(min(p[d] for p in q) for d in range(3))
    hi = tuple(max(p[d] for p in q) for d in range(3))
    if any(hi[d] - lo[d] > window for d in range(3)):
        return RefTet(OUTSIDE_WINDOW)
    det = signed_volume(q)
    if det <= 0:
        return RefTet(INVERTED)
    tet = RefTet(OK, q=q, det=det)
    tet.box_lo = (-(-lo[0] // s), -(-lo[1] // s), -(-lo[2] // s))
    tet.box_hi = (hi[0] // s, hi[1] // s, hi[2] // s)
    origin = (tet.box_lo[0] * s, tet.box_lo[1] * s, tet.box_lo[2] * s)
    for j in range(4):
        e0 = tet.plane(j, (0, 0, 0))
        units = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        normal = (
            tet.plane(j, units[0]) - e0,
            tet.plane(j, units[1]) - e0,
            tet.plane(j, units[2]) - e0,
        )
        at_origin = tet.plane(j, origin)
        offset = (at_origin - 1 + int(lex_positive(normal))) // s
        tet.normals.append(normal)
        tet.offsets.append(offset)
        tet.fractions.append(Fraction(at_origin - s * offset, s))
    return tet


def setup_tets(
    positions: npt.NDArray[np.floating], tetrahedra: npt.NDArray[np.int32], k: int
) -> list[RefTet]:
    """setup_tet for every tet. positions (N, 3), tetrahedra (T, 4)."""
    return [setup_tet(positions[t], k) for t in tetrahedra]


def bary_scale(tet: RefTet, k: int, dtype: type[np.floating]) -> np.floating:
    """2**k / D as the C++ computes it: correctly rounded to double, then cast to dtype."""
    scale: np.floating = dtype((1 << k) / float(tet.det))
    return scale


@dataclass
class RefRaster:
    """Exact rasterisation of a grid: claims and, per claimed voxel, its last claimant and pi."""

    claims: npt.NDArray[np.int32]
    owner: npt.NDArray[np.int32]
    pi: npt.NDArray[np.float64]  # (X, Y, Z, 4), exact values rounded once to float64


def owned_pi(tet: RefTet, voxel: Point, k: int) -> list[Fraction] | None:
    """Exact pi_0..pi_3 if the SoS rule gives the voxel centre to the tet, else None."""
    p = (voxel[0] * (1 << k), voxel[1] * (1 << k), voxel[2] * (1 << k))
    planes = [tet.plane(j, p) for j in range(4)]
    inside = all(e > 0 or (e == 0 and lex_positive(tet.normals[j])) for j, e in enumerate(planes))
    return [Fraction(e, tet.det) for e in planes] if inside else None


def rasterize(tets: list[RefTet], shape: tuple[int, int, int], k: int) -> RefRaster:
    """Visit every voxel of every OK tet's grid-clipped box and apply the explicit rule."""
    claims = np.zeros(shape, np.int32)
    owner = np.full(shape, -1, np.int32)
    pi = np.zeros((*shape, 4), np.float64)
    for t, tet in enumerate(tets):
        if tet.status != OK:
            continue
        ranges = [
            range(max(tet.box_lo[d], 0), min(tet.box_hi[d], shape[d] - 1) + 1) for d in range(3)
        ]
        for i0 in ranges[0]:
            for i1 in ranges[1]:
                for i2 in ranges[2]:
                    exact = owned_pi(tet, (i0, i1, i2), k)
                    if exact is not None:
                        claims[i0, i1, i2] += 1
                        owner[i0, i1, i2] = t
                        pi[i0, i1, i2] = [float(x) for x in exact]
    return RefRaster(claims, owner, pi)
