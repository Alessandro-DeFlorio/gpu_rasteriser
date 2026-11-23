from pathlib import Path

import numpy as np

from tetvox import load_atlas_npz


def test_load_synthetic_atlas(synthetic_atlas: Path) -> None:
    mesh = load_atlas_npz(synthetic_atlas)
    assert mesh.positions.shape == (8, 3)
    assert mesh.tetrahedra.shape == (6, 4)
    assert mesh.alphas.shape == (8, 2)
    assert mesh.names == ("right", "left")
    assert mesh.stiffness == 0.1


def test_load_adult_atlas(adult_atlas: Path) -> None:
    mesh = load_atlas_npz(adult_atlas)
    n_vertices = mesh.positions.shape[0]
    assert mesh.positions.dtype == np.float32
    assert mesh.tetrahedra.dtype == np.int32
    assert mesh.alphas.shape == (n_vertices, len(mesh.names))
    assert int(np.max(mesh.tetrahedra)) == n_vertices - 1
    np.testing.assert_allclose(mesh.alphas.sum(axis=1), 1.0, atol=1e-5)
