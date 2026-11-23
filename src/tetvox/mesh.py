"""The generic tetrahedral mesh that every tetvox op takes (ADR 0005)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TetMesh:
    """A tetrahedral mesh with per-vertex values, in voxel index space.

    Shapes and dtypes are checked by the C++ ops that consume the mesh (ADR 0004), not here.

    Attributes:
        positions: (N, 3) vertex positions in voxel index space (voxel centres at integers).
        tetrahedra: (T, 4) vertex indices of each tet.
        values: (N, C) per-vertex values interpolated over the mesh; C may be 0.
    """

    positions: npt.NDArray[np.float32]
    tetrahedra: npt.NDArray[np.int32]
    values: npt.NDArray[np.float32]
