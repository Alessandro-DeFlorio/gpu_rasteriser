"""tetvox: fast, watertight, differentiable CUDA voxeliser for tetrahedral meshes."""

from tetvox._core import cuda_enabled, device_count
from tetvox.mesh import TetMesh
from tetvox.raster import (
    DEFAULT_SNAP_BITS,
    MAX_SNAP_BITS,
    SkippedTets,
    TetSetup,
    TetStatus,
    barycentrics,
    count_claims,
    count_skipped,
    locate,
    rasterize,
    setup_tets,
)
from tetvox.samseg import SamsegAtlas, load_atlas_npz

__all__ = [
    "DEFAULT_SNAP_BITS",
    "MAX_SNAP_BITS",
    "SamsegAtlas",
    "SkippedTets",
    "TetMesh",
    "TetSetup",
    "TetStatus",
    "barycentrics",
    "count_claims",
    "count_skipped",
    "cuda_enabled",
    "device_count",
    "load_atlas_npz",
    "locate",
    "rasterize",
    "setup_tets",
]
