# tetvox

Fast, watertight, differentiable CUDA voxeliser for simplicial meshes in 1D, 2D and 3D: segments,
triangles and tetrahedra (e.g. samseg probabilistic atlases). Every voxel (or pixel) is assigned to
exactly one simplex, together with its barycentric coordinates. This turns the mesh into a
piecewise-linear interpolator whose output is differentiable with respect to vertex positions and
vertex values.

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

### From C++/CUDA

The library can be used without Python. A plain CMake build skips the Python module:

```bash
cmake -S . -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp
cmake --install build/cpp --component cpp --prefix <prefix>
```

```cmake
project(my_app LANGUAGES CXX CUDA)   # CUDA must be enabled
find_package(tetvox REQUIRED)        # with -DCMAKE_PREFIX_PATH=<prefix>
target_link_libraries(my_app PRIVATE tetvox::tetvox)
```

`add_subdirectory` works too. Your project chooses the CUDA runtime (`CMAKE_CUDA_RUNTIME_LIBRARY`).
See [examples/cpp](examples/cpp).

## Development

```bash
conda env create -f environment.yml
conda activate tetvox-dev
pip install --no-build-isolation -ve .   # editable: C++/CUDA is rebuilt automatically on import
scripts/check.sh                          # ruff, mypy --strict, clang-format, C++ example, pytest
```

## License

MIT
