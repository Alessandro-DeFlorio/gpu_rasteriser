import itertools
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

ATLAS_ROOT = Path(__file__).resolve().parent.parent / "atlases"


def kuhn_cube(dim: int = 3) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Unit dim-cube split into dim! simplices sharing the main diagonal (Kuhn triangulation).

    Vertex i sits at (bit 0, ..., bit dim-1) of i. Every simplex is positively oriented,
    det[v_1 - v_0, ..., v_dim - v_0] > 0, as tetvox requires. Returns positions (2^dim, dim) and
    simplices (dim!, dim + 1).
    """
    positions = np.array(
        [[(i >> a) & 1 for a in range(dim)] for i in range(1 << dim)], dtype=np.float32
    )
    simplices = np.array(
        [np.cumsum([0] + [1 << a for a in order]) for order in itertools.permutations(range(dim))],
        np.int32,
    )
    v = positions[simplices].astype(np.float64)
    negative = np.linalg.det(v[:, 1:] - v[:, :1]) < 0
    swapped = list(range(dim + 1))
    swapped[0], swapped[1] = 1, 0
    simplices[negative] = simplices[negative][:, swapped]
    return positions, simplices


def kuhn_grid(n: int, dim: int = 3) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """A block of n^dim unit Kuhn cubes covering [0, n]^dim, all cubes cut the same way.

    Returns positions ((n + 1)^dim, dim) float64 on integer points and simplices (dim! n^dim,
    dim + 1), every simplex positively oriented. Neighbouring cubes share faces, so the block is
    a conforming mesh.
    """
    cube_positions, cube_simplices = kuhn_cube(dim)
    points = np.stack(np.meshgrid(*[np.arange(n + 1)] * dim, indexing="ij"), axis=-1)
    corner_offsets = cube_positions.astype(np.int64)  # (2^dim, dim) offsets of the cube's corners
    cells = np.stack(np.meshgrid(*[np.arange(n)] * dim, indexing="ij"), -1).reshape(-1, dim)
    simplices = []
    for corner in cells:
        index = np.ravel_multi_index(tuple((corner + corner_offsets).T), (n + 1,) * dim)
        simplices.append(index[cube_simplices])
    return points.reshape(-1, dim).astype(np.float64), np.concatenate(simplices).astype(np.int32)


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
