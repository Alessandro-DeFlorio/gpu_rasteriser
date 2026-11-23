"""Tetrahedral mesh container and samseg atlas loader (format: docs/knowledge/atlas-format.md)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TetMesh:
    """A tetrahedral mesh with per-vertex class priors, in template voxel coordinates.

    Attributes:
        positions: (N, 3) vertex positions in template voxel coordinates.
        tetrahedra: (T, 4) vertex indices of each tet.
        alphas: (N, K) per-vertex class prior probabilities; rows sum to 1.
        affine: (4, 4) template voxel -> world (mm) transform.
        can_move: (N, 3) per-vertex, per-axis mobility mask used during registration.
        names: K class names.
        colors: (K, 4) RGBA display colours.
        freesurfer_labels: FreeSurfer label ids (length is not always K, see atlas-format.md).
        stiffness: samseg mesh stiffness K (kappa) scaling the deformation prior.
    """

    positions: npt.NDArray[np.float32]
    tetrahedra: npt.NDArray[np.int32]
    alphas: npt.NDArray[np.float32]
    affine: npt.NDArray[np.float64]
    can_move: npt.NDArray[np.bool_]
    names: tuple[str, ...]
    colors: npt.NDArray[np.int64]
    freesurfer_labels: npt.NDArray[np.int64]
    stiffness: float


def load_atlas_npz(path: Path) -> TetMesh:
    """Load a samseg atlas `.npz`, converting arrays to the canonical dtypes of `TetMesh`."""
    with np.load(path) as data:
        return TetMesh(
            positions=data["positions"].astype(np.float32),
            tetrahedra=data["tetrahedra"].astype(np.int32),
            alphas=data["alphas"].astype(np.float32),
            affine=data["affine"].astype(np.float64),
            can_move=data["can_move"].astype(np.bool_),
            names=tuple(str(name) for name in data["names"]),
            colors=data["colors"].astype(np.int64),
            freesurfer_labels=data["FreeSurferLabels"].astype(np.int64),
            stiffness=float(data["K"]),
        )
