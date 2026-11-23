---
name: check
description: Build tetvox and run every quality gate (stub freshness, ruff, mypy --strict, clang-format, pytest). Use before declaring any code change done.
---

Run inside the `tetvox-dev` conda env:

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate tetvox-dev && scripts/check.sh

Extra arguments are passed to pytest (e.g. `scripts/check.sh -k mesh`). Importing `tetvox` rebuilds
the C++/CUDA extension automatically (editable install). If the import fails because the package is
not installed, run `pip install --no-build-isolation -ve .` first.

Report failures with the relevant output. Fix formatting with `ruff format .` and
`clang-format -i <files>`. If the stub is stale, regenerate it (see `docs/knowledge/build.md`).
