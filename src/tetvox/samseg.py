"""samseg atlas: a `TetMesh` plus the samseg model data (format: docs/knowledge/atlas-format.md)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tetvox.mesh import TetMesh


@dataclass(frozen=True)
class SamsegAtlas:
    """A samseg probabilistic atlas. Only `mesh` is used by tetvox ops.

    Attributes:
        mesh: the atlas mesh, in template voxel coordinates. `mesh.values` (N, K) holds the
            per-vertex class priors (samseg alphas); rows sum to 1.
        affine: (4, 4) template voxel -> world (mm) transform.
        can_move: (N, 3) per-vertex, per-axis mobility mask used during registration.
        names: K class names.
        colors: (K, 4) RGBA display colours.
        freesurfer_labels: FreeSurfer label ids (length is not always K, see atlas-format.md).
        stiffness: samseg mesh stiffness K (kappa) scaling the deformation prior.
    """

    mesh: TetMesh
    affine: npt.NDArray[np.float64]
    can_move: npt.NDArray[np.bool_]
    names: tuple[str, ...]
    colors: npt.NDArray[np.int64]
    freesurfer_labels: npt.NDArray[np.int64]
    stiffness: float


def load_atlas_npz(path: Path) -> SamsegAtlas:
    """Load a samseg atlas `.npz`, converting arrays to the canonical dtypes of `SamsegAtlas`."""
    with np.load(path) as data:
        mesh = TetMesh(
            positions=data["positions"].astype(np.float32),
            tetrahedra=data["tetrahedra"].astype(np.int32),
            values=data["alphas"].astype(np.float32),
        )
        return SamsegAtlas(
            mesh=mesh,
            affine=data["affine"].astype(np.float64),
            can_move=data["can_move"].astype(np.bool_),
            names=tuple(str(name) for name in data["names"]),
            colors=data["colors"].astype(np.int64),
            freesurfer_labels=data["FreeSurferLabels"].astype(np.int64),
            stiffness=float(data["K"]),
        )
