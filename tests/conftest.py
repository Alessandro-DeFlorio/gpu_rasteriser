from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

ATLAS_ROOT = Path(__file__).resolve().parent.parent / "atlases"


def kuhn_cube() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Unit cube split into 6 tets sharing the main diagonal (Kuhn triangulation).

    Vertex i sits at (bit0, bit1, bit2) of i. Every tet is positively oriented,
    det[v1 - v0, v2 - v0, v3 - v0] > 0, as tetvox requires. Returns positions (8, 3) and
    tetrahedra (6, 4).
    """
    positions = np.array([[(i >> a) & 1 for a in range(3)] for i in range(8)], dtype=np.float32)
    axis_orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets = np.array([[0, 1 << a, (1 << a) | (1 << b), 7] for a, b, _ in axis_orders], np.int32)
    v = positions[tets]
    negative = np.linalg.det(v[:, 1:] - v[:, :1]) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return positions, tets


def kuhn_grid(n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """A block of n x n x n unit Kuhn cubes covering [0, n]^3, all cubes cut the same way.

    Returns positions ((n + 1)^3, 3) float64 on integer points and tetrahedra (6 n^3, 4), every
    tet positively oriented. Neighbouring cubes share faces, so the block is a conforming mesh.
    """
    cube_positions, cube_tets = kuhn_cube()
    axis = np.arange(n + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    corner_offsets = cube_positions.astype(np.int64)  # (8, 3) offsets of the cube's corners
    tets = []
    for corner in np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3):
        corners = corner + corner_offsets
        index = (corners[:, 0] * (n + 1) + corners[:, 1]) * (n + 1) + corners[:, 2]
        tets.append(index[cube_tets])
    return grid.astype(np.float64), np.concatenate(tets).astype(np.int32)


@pytest.fixture
def synthetic_atlas(tmp_path: Path) -> Path:
    """Kuhn cube written in the samseg atlas npz format, with 2 classes."""
    positions, tetrahedra = kuhn_cube()
    alphas = np.stack([positions[:, 0], 1.0 - positions[:, 0]], axis=1).astype(np.float32)
    path = tmp_path / "cube.npz"
    np.savez(
        path,
        positions=positions,
        tetrahedra=tetrahedra,
        alphas=alphas,
        affine=np.eye(4, dtype=np.float32),
        can_move=np.ones((8, 3), dtype=bool),
        names=np.array(["right", "left"]),
        colors=np.array([[255, 0, 0, 255], [0, 0, 255, 255]]),
        FreeSurferLabels=np.array([1, 2]),
        K=np.array(0.1),
    )
    return path


def local_atlas(relative: str) -> Path:
    """An atlas under atlases/, or skip the test if it is not present (atlases/ is not in git)."""
    path = ATLAS_ROOT / relative
    if not path.exists():
        pytest.skip(f"{path} not available")
    return path


@pytest.fixture
def adult_atlas() -> Path:
    """The adult samseg atlas."""
    return local_atlas("adult_template/atlas_adult.npz")


# The adult mesh and the four distinct infant meshes (docs/knowledge/atlas-format.md).
DISTINCT_ATLAS_MESHES = [
    "adult_template/atlas_adult.npz",
    "subtemplates_12/template_1-2mo_41.npz",
    "subtemplates_12/template_3-5mo_41.npz",
    "subtemplates_12/template_6-8mo_41.npz",
    "subtemplates_12/atlas_reference_41.npz",
]


@pytest.fixture(params=DISTINCT_ATLAS_MESHES)
def distinct_atlas(request: pytest.FixtureRequest) -> Path:
    """Each distinct atlas mesh in turn."""
    return local_atlas(request.param)
