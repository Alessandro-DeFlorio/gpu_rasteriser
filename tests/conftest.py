from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

ATLAS_ROOT = Path(__file__).resolve().parent.parent / "atlases"


def kuhn_cube() -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Unit cube split into 6 tets sharing the main diagonal (Kuhn triangulation).

    Vertex i sits at (bit0, bit1, bit2) of i. Returns positions (8, 3) and tetrahedra (6, 4).
    """
    positions = np.array([[(i >> a) & 1 for a in range(3)] for i in range(8)], dtype=np.float32)
    axis_orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    tets = [[0, 1 << a, (1 << a) | (1 << b), 7] for a, b, _ in axis_orders]
    return positions, np.array(tets, dtype=np.int32)


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


@pytest.fixture
def adult_atlas() -> Path:
    """The adult samseg atlas, if present locally (atlases/ is not in git)."""
    path = ATLAS_ROOT / "adult_template" / "atlas_adult.npz"
    if not path.exists():
        pytest.skip(f"{path} not available")
    return path
