# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

tetvox is an open-source (MIT), fast, **watertight, differentiable CUDA voxeliser for tetrahedral
meshes**. The main target is samseg probabilistic brain atlases. Each voxel centre is assigned to
exactly one tet, together with its barycentrics, so the mesh acts as a piecewise-linear interpolator
whose output is differentiable with respect to vertex positions and per-vertex values.

Stack: C++20 / CUDA core → nanobind (DLPack) bindings → typed Python package. Torch is optional.

## Commands

Always work inside the conda env. Base python has no numpy.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate tetvox-dev
conda env create -f environment.yml         # first time only
pip install --no-build-isolation -ve .      # once; `import tetvox` then rebuilds C++/CUDA automatically
scripts/check.sh                            # all gates: stub freshness, ruff, mypy --strict, clang-format, pytest
pytest tests/test_mesh.py::test_load_adult_atlas   # a single test
python -m nanobind.stubgen -m tetvox._core -o src/tetvox/_core.pyi   # after changing bindings
python scripts/inspect_atlas.py atlases/adult_template/atlas_adult.npz  # atlas statistics
```

Skills: `/check` runs the gates; `/research <question>` starts a cited literature investigation via
the `researcher` agent (Fable), which writes to `docs/research/`.

## Layout

- `csrc/include/tetvox/`: header-only `__host__ __device__` maths, shared by kernels and the CPU
  reference.
- `csrc/src/*.cu`: kernels, built into the static lib `tetvox_core`.
- `csrc/bindings/module.cpp`: the `tetvox._core` nanobind module.
- `src/tetvox/`: the Python API.
- `tests/`: `conftest.py` provides a synthetic Kuhn-cube mesh (always available) and real-atlas
  fixtures that skip when absent.
- `atlases/`, `data/` (meshes from real samseg runs) and `literature/*.pdf` are local only and
  gitignored (ADR 0003).

## Documentation protocol (mandatory)

The docs are the project's memory. Do not rediscover what is written there.
- **Before** a task: read `docs/ROADMAP.md`, `docs/architecture.md` and the relevant
  `docs/knowledge/` files. Key files: `samseg-math.md` (notation and equations from
  `literature/equations.pdf`; use that notation in code), `atlas-format.md` (atlas schema and mesh
  statistics) and `build.md` (toolchain gotchas).
- **After** a task:
  - add new verified facts to `docs/knowledge/`;
  - record design decisions as an ADR in `docs/decisions/NNNN-*.md` (owner approval needed for
    decisions coming out of research);
  - update `docs/ROADMAP.md` (status, next steps, open questions);
  - keep `docs/architecture.md` current when structure changes.
- Research goes to `docs/research/<id>-*.md` (template `_template.md`, briefs in `briefs/`). Every claim
  is marked verified or inferred, with citations.

## Code rules

- **Types everywhere.** Python passes `mypy --strict`. Annotate every argument and return value, e.g.
  `def f(x: npt.NDArray[np.float32]) -> None`, and put array shapes in the docstring. Bindings ship
  a committed `.pyi` stub that must stay in sync.
- **No needless defensive programming.** Validate inputs once, at the public C++ API boundary
  (ADR 0004: C++ is a first-class interface). The Python layer does not repeat those checks.
  Internals trust their invariants: no redundant checks, try/except, or fallback paths for states
  that cannot happen.
- Tidy, small functions; no dead code or commented-out code. C++ follows `.clang-format` (Google,
  100 columns); Python follows ruff (100 columns).
- Shared maths goes in headers as `__host__ __device__`, so it can be tested against a CPU reference.
- Watertightness is a hard requirement: every voxel centre in the domain is owned by exactly one tet.
  Tests should check it exhaustively on real atlases when they are available.
