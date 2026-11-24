"""The generic simplicial mesh (ADR 0005, 0009, 0010)."""

from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt

from tetvox._typing import F


@dataclass(frozen=True)
class SimplexMesh(Generic[F]):
    """A mesh of D-simplices (segments, triangles or tetrahedra) with per-vertex values.

    Positions are in voxel index space. `positions` and `values` share one real dtype `F` (float32
    or float64, ADR 0008). Shapes and dtypes are checked by the C++ ops that consume the arrays
    (ADR 0004), not here.

    Attributes:
        positions: (N, D) vertex positions in voxel index space (voxel centres at integers), with D
            in 1, 2, 3.
        simplices: (T, D + 1) vertex indices of each simplex.
        values: (N, C) per-vertex values interpolated over the mesh; C may be 0.
    """

    positions: npt.NDArray[F]
    simplices: npt.NDArray[np.int32]
    values: npt.NDArray[F]
