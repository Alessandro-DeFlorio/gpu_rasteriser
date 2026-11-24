"""tetvox: fast, watertight, differentiable CUDA voxeliser for simplicial meshes in 1D, 2D and 3D
(segments, triangles, tetrahedra)."""

from tetvox._core import cuda_enabled, device_count
from tetvox.mesh import SimplexMesh
from tetvox.raster import (
    DEFAULT_SNAP_BITS,
    MAX_SNAP_BITS,
    SimplexSetup,
    SimplexStatus,
    SkippedSimplices,
    barycentrics,
    count_claims,
    count_skipped,
    locate,
    rasterize,
    setup_simplices,
)
from tetvox.samseg import SamsegAtlas, load_atlas_npz

__all__ = [
    "DEFAULT_SNAP_BITS",
    "MAX_SNAP_BITS",
    "SamsegAtlas",
    "SimplexMesh",
    "SimplexSetup",
    "SimplexStatus",
    "SkippedSimplices",
    "barycentrics",
    "count_claims",
    "count_skipped",
    "cuda_enabled",
    "device_count",
    "load_atlas_npz",
    "locate",
    "rasterize",
    "setup_simplices",
]
