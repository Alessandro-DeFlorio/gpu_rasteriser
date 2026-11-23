# tetvox

Fast, watertight, differentiable CUDA voxeliser for tetrahedral meshes (e.g. samseg probabilistic
atlases). Every voxel is assigned to exactly one tetrahedron, together with its barycentric
coordinates. This turns the mesh into a piecewise-linear interpolator whose output is
differentiable with respect to vertex positions and vertex values.

Status: early development. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Install

Requirements: an NVIDIA GPU, a CUDA toolkit (nvcc ≥ 12) on `PATH`, CMake ≥ 3.26, Python ≥ 3.10.

```bash
pip install .
```

By default the extension is compiled for the GPU(s) in the build machine. For a portable build, run:

```bash
pip install . -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120"
```

## Development

```bash
conda env create -f environment.yml
conda activate tetvox-dev
pip install --no-build-isolation -ve .   # editable: C++/CUDA is rebuilt automatically on import
scripts/check.sh                          # ruff, mypy --strict, clang-format, pytest
```

## License

MIT
