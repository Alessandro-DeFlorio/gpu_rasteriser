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
scripts/check.sh                            # all gates: stub freshness, ruff, mypy --strict, clang-format, C++ example, pytest
pytest tests/test_samseg.py::test_load_adult_atlas   # a single test
python -m nanobind.stubgen -m tetvox._core -o src/tetvox/_core.pyi   # after changing bindings
python scripts/inspect_atlas.py atlases/adult_template/atlas_adult.npz  # atlas statistics
```

Skills: `/check` runs the gates; `/research <question>` starts a cited literature investigation via
the `researcher` agent (Fable), which writes to `docs/research/`.

## Layout

- `csrc/include/tetvox/`: header-only `__host__ __device__` maths, shared by kernels and the CPU
  reference.
- `csrc/src/*.cu`: kernels, built into the static lib `tetvox_core`.
- `csrc/bindings/module.cpp`: the `tetvox._core` nanobind module, built only with
  `TETVOX_BUILD_PYTHON` (ON under scikit-build, OFF in a plain CMake build).
- `CMakeLists.txt`, `cmake/`: export `tetvox_core` as the C++ package `tetvox::tetvox` (ADR 0004).
  The version is set only in `project()`.
- `examples/cpp/`: stand-alone consumer of the installed C++ package; `scripts/check.sh` builds and
  runs it.
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

## Decisions and assumptions (mandatory)

The owner must know about every choice they have not made explicitly.

- **Before writing code**, list every decision the task needs that the owner's request, an ADR or
  `docs/` has not settled, and wait for answers. For each give: the choice, the options, your
  recommendation and why, and what it affects.
- **What counts:** anything that can change results, numerics, behaviour or the public API. For
  example: epsilons and tolerances, guards against division by zero, clamping, dtypes and precision
  (float32/64, fixed-point scale), rounding and tie-breaking, sampling conventions (voxel centre or
  corner, half-open or closed), coordinate spaces, axis and memory order, default argument values,
  handling of degenerate or invalid input, and names and placement of public API.
- **Mid-task:** if a new decision comes up, stop and ask. Never pick silently, even when the choice
  looks obvious.
- **Tunable values are never bare literals.** Each is a named constant with a comment giving its
  origin, and the owner decides whether it is fixed, compile-time (template or constexpr) or a
  run-time parameter.
- **Record the answer:** an ADR for design-level choices, `docs/knowledge/` or a code comment for
  local ones, so it is never asked twice.
- **Purely internal details** that do not affect any of the above (private helper names, how a
  file is organised internally) need no prior question; list them in the final report so the owner
  can still veto them.

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
