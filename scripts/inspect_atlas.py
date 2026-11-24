"""Print structural statistics of a samseg atlas `.npz` (see docs/knowledge/atlas-format.md).

Usage: python scripts/inspect_atlas.py atlases/adult_template/atlas_adult.npz [...]
"""

import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tetvox import TetMesh, load_atlas_npz

FloatArray = npt.NDArray[np.float64]
Mesh = TetMesh[np.float32]  # load_atlas_npz default dtype


def signed_volumes(mesh: Mesh) -> FloatArray:
    """(T,) signed tet volumes det(A)/6 in voxel^3, computed in float64."""
    v = mesh.positions.astype(np.float64)[mesh.tetrahedra]
    edges = v[:, 1:] - v[:, :1]
    return np.asarray(np.linalg.det(edges.transpose(0, 2, 1)) / 6.0)


def max_edge_lengths(mesh: Mesh) -> FloatArray:
    """(T,) length of the longest of the 6 edges of each tet, in voxels."""
    v = mesh.positions.astype(np.float64)[mesh.tetrahedra]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lengths = np.stack([np.linalg.norm(v[:, a] - v[:, b], axis=1) for a, b in pairs], axis=1)
    return np.asarray(lengths.max(axis=1))


def lattice_points_in_bbox(mesh: Mesh) -> npt.NDArray[np.int64]:
    """(T,) number of integer voxel centres inside each tet's axis-aligned bounding box."""
    v = mesh.positions.astype(np.float64)[mesh.tetrahedra]
    extent = np.floor(v.max(axis=1)) - np.ceil(v.min(axis=1)) + 1
    return np.asarray(np.clip(extent, 0, None).prod(axis=1).astype(np.int64))


def face_multiplicities(mesh: Mesh) -> npt.NDArray[np.int64]:
    """Number of tets sharing each distinct triangular face."""
    faces = np.concatenate([np.delete(mesh.tetrahedra, i, axis=1) for i in range(4)])
    _, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    return np.asarray(counts.astype(np.int64))


def boundary_vertices(mesh: Mesh) -> npt.NDArray[np.bool_]:
    """(N,) mask of vertices lying on a boundary face (face owned by a single tet)."""
    faces = np.sort(np.concatenate([np.delete(mesh.tetrahedra, i, axis=1) for i in range(4)]), 1)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    mask = np.zeros(mesh.positions.shape[0], dtype=np.bool_)
    mask[unique[counts == 1].ravel()] = True
    return mask


def percentiles(name: str, values: FloatArray) -> str:
    """One-line summary of min / p1 / median / p99 / max."""
    labels = ["min", "p1", "median", "p99", "max"]
    q = np.percentile(values, [0, 1, 50, 99, 100])
    stats = "  ".join(f"{label} {v:.3g}" for label, v in zip(labels, q, strict=True))
    return f"{name:<24} {stats}"


def report(path: Path) -> None:
    with np.load(path) as raw:
        stored = {key: (str(raw[key].dtype), raw[key].shape) for key in raw.files}
    atlas = load_atlas_npz(path)
    mesh = atlas.mesh
    n_vertices, n_classes = mesh.values.shape
    volumes = signed_volumes(mesh)
    abs_volumes = np.abs(volumes)
    edges = max_edge_lengths(mesh)
    lattice = lattice_points_in_bbox(mesh)
    multiplicity = face_multiplicities(mesh)
    on_boundary = boundary_vertices(mesh)
    fixed = ~atlas.can_move.all(axis=1)
    lo, hi = mesh.positions.min(axis=0), mesh.positions.max(axis=0)
    voxel_size = np.linalg.norm(atlas.affine[:3, :3], axis=0)

    print(f"== {path}")
    for key, (dtype, shape) in stored.items():
        print(f"  stored {key:<26} {dtype:<6} {shape}")
    print(f"  vertices {n_vertices}  tets {len(volumes)}  classes {n_classes}")
    print(f"  bbox (voxels) {lo} .. {hi}   voxel size (mm) {voxel_size.round(4)}")
    print(f"  stiffness K {atlas.stiffness}")
    print(
        f"  alphas: max |row sum - 1| {np.abs(mesh.values.sum(axis=1) - 1).max():.2e}"
        f"  min {np.min(mesh.values):.2e}"
    )
    print(
        f"  orientation: {np.sum(volumes > 0)} positive, {np.sum(volumes < 0)} negative,"
        f" {np.sum(volumes == 0)} zero"
    )
    print(f"  total |volume| {abs_volumes.sum():.6g}  vs bbox volume {np.prod(hi - lo):.6g}")
    print("  " + percentiles("|volume| (voxel^3)", abs_volumes))
    print("  " + percentiles("max edge (voxels)", edges))
    print("  " + percentiles("voxel centres in bbox", lattice.astype(np.float64)))
    print(f"  tets with 0 voxel centres in bbox: {np.mean(lattice == 0):.1%}")
    print(f"  volume ratio max/min {abs_volumes.max() / abs_volumes.min():.3g}")
    quality = abs_volumes / edges**3
    print("  " + percentiles("|volume| / max_edge^3", quality))
    print(
        f"  faces: {np.sum(multiplicity == 1)} boundary, {np.sum(multiplicity == 2)} interior,"
        f" {np.sum(multiplicity > 2)} non-manifold"
    )
    print(f"  unused vertices {n_vertices - np.unique(mesh.tetrahedra).size}")
    print(
        f"  boundary vertices {on_boundary.sum()}  fixed vertices {fixed.sum()}"
        f"  fixed & boundary {(fixed & on_boundary).sum()}"
    )
    print(
        f"  boundary vertices on bbox faces: "
        f"{np.mean(((mesh.positions == lo) | (mesh.positions == hi)).any(axis=1)[on_boundary]):.1%}"
    )
    print(f"  names ({len(atlas.names)}): {', '.join(atlas.names)}")
    labels = atlas.freesurfer_labels
    print(f"  FreeSurfer labels ({labels.size}): {labels.tolist()}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        report(Path(arg))
