"""The generic tetrahedral mesh (ADR 0005, ADR 0009)."""

from dataclasses import dataclass
from typing import Generic

import numpy as np
import numpy.typing as npt

from tetvox._typing import F


@dataclass(frozen=True)
class TetMesh(Generic[F]):
    """A tetrahedral mesh with per-vertex values, in voxel index space.

    `positions` and `values` share one real dtype `F` (float32 or float64, ADR 0008). Shapes and
    dtypes are checked by the C++ ops that consume the arrays (ADR 0004), not here.

    Attributes:
        positions: (N, 3) vertex positions in voxel index space (voxel centres at integers).
        tetrahedra: (T, 4) vertex indices of each tet.
        values: (N, C) per-vertex values interpolated over the mesh; C may be 0.
    """

    positions: npt.NDArray[F]
    tetrahedra: npt.NDArray[np.int32]
    values: npt.NDArray[F]
