from pathlib import Path

import numpy as np

from tetvox import load_atlas_npz


def test_load_synthetic_atlas(synthetic_atlas: Path) -> None:
    atlas = load_atlas_npz(synthetic_atlas)
    assert atlas.mesh.positions.shape == (8, 3)
    assert atlas.mesh.tetrahedra.shape == (6, 4)
    assert atlas.mesh.values.shape == (8, 2)
    assert atlas.names == ("right", "left")
    assert atlas.stiffness == 0.1


def test_load_dtype(synthetic_atlas: Path) -> None:
    single = load_atlas_npz(synthetic_atlas)
    double = load_atlas_npz(synthetic_atlas, dtype=np.float64)
    assert single.mesh.positions.dtype == single.mesh.values.dtype == np.float32
    assert double.mesh.positions.dtype == double.mesh.values.dtype == np.float64
    assert double.mesh.tetrahedra.dtype == np.int32
    assert double.affine.dtype == np.float64
    np.testing.assert_array_equal(double.mesh.positions, single.mesh.positions)


def test_load_adult_atlas(adult_atlas: Path) -> None:
    atlas = load_atlas_npz(adult_atlas)
    mesh = atlas.mesh
    n_vertices = mesh.positions.shape[0]
    assert mesh.positions.dtype == np.float32
    assert mesh.tetrahedra.dtype == np.int32
    assert mesh.values.shape == (n_vertices, len(atlas.names))
    assert int(np.max(mesh.tetrahedra)) == n_vertices - 1
    np.testing.assert_allclose(mesh.values.sum(axis=1), 1.0, atol=1e-5)
