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
find csrc \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' -o -name '*.cuh' \) -print0 \
  | xargs -0 clang-format --dry-run --Werror
pytest -q "$@"
