#!/usr/bin/env bash
# All quality gates in one command. Run inside the tetvox-dev env after `pip install -ve .`
set -euo pipefail
cd "$(dirname "$0")/.."

stubs="$(mktemp -d)"
python -m nanobind.stubgen -q -m tetvox._core -o "$stubs/_core.pyi"
diff -u src/tetvox/_core.pyi "$stubs/_core.pyi" || { echo "stale stub: regenerate _core.pyi (docs/knowledge/build.md)"; exit 1; }

ruff check .
ruff format --check .
mypy
find csrc examples \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror

# C++ packaging test (ADR 0004): plain CMake build without Python, install, then build and run the
# stand-alone consumer in examples/cpp against the installed package. Run for the default (CUDA)
# configuration and for the CPU-only one (ADR 0008).
cpp_package_test() {
  local cpp=$1; shift
  cmake -S . -B "$cpp/core" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/$cpp/install" "$@" >/dev/null
  cmake --build "$cpp/core"
  rm -rf "$cpp/install"  # no stale files from earlier installs
  cmake --install "$cpp/core" --component cpp >/dev/null
  cmake -S examples/cpp -B "$cpp/example" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PWD/$cpp/install" >/dev/null
  cmake --build "$cpp/example"
  "$cpp/example/tetvox_example"
}
cpp_package_test build/cpp
cpp_package_test build/cpp-cpu-only -DTETVOX_ENABLE_CUDA=OFF

pytest -q "$@"
