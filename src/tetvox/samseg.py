"""samseg atlas: a 3D `SimplexMesh` plus the samseg model data (format:
docs/knowledge/atlas-format.md)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, overload

import numpy as np
import numpy.typing as npt

from tetvox._typing import F
from tetvox.mesh import SimplexMesh


@dataclass(frozen=True)
class SamsegAtlas(Generic[F]):
    """A samseg probabilistic atlas (3D only, ADR 0010). Only `mesh` is used by tetvox ops.

    Attributes:
        mesh: the tetrahedral atlas mesh, in template voxel coordinates: positions (N, 3),
            simplices (T, 4). `mesh.values` (N, K) holds the per-vertex class priors (samseg
            alphas); rows sum to 1.
        affine: (4, 4) template voxel -> world (mm) transform.
        can_move: (N, 3) per-vertex, per-axis mobility mask used during registration.
        names: K class names.
        colors: (K, 4) RGBA display colours.
        freesurfer_labels: FreeSurfer label ids (length is not always K, see atlas-format.md).
        stiffness: samseg mesh stiffness K (kappa) scaling the deformation prior.
    """

    mesh: SimplexMesh[F]
    affine: npt.NDArray[np.float64]
    can_move: npt.NDArray[np.bool_]
    names: tuple[str, ...]
    colors: npt.NDArray[np.int64]
    freesurfer_labels: npt.NDArray[np.int64]
    stiffness: float


# Overloads because mypy rejects a default for a parameter typed with a constrained TypeVar.
@overload
def load_atlas_npz(path: Path) -> SamsegAtlas[np.float32]: ...
@overload
def load_atlas_npz(path: Path, *, dtype: type[F]) -> SamsegAtlas[F]: ...
def load_atlas_npz(path: Path, *, dtype: type[Any] = np.float32) -> SamsegAtlas[Any]:
    """Load a samseg atlas `.npz`, converting arrays to the canonical dtypes of `SamsegAtlas`.

    Args:
        path: the atlas file.
        dtype: real dtype of `mesh.positions` and `mesh.values` (ADR 0009). The default matches 11
            of the 12 stored atlases; pass `np.float64` to keep positions in double.
    """
    with np.load(path) as data:
        mesh = SimplexMesh(
            positions=data["positions"].astype(dtype),
            simplices=data["tetrahedra"].astype(np.int32),
            values=data["alphas"].astype(dtype),
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
