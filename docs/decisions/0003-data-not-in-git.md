# 0003 — Atlases and literature PDFs are not committed

Status: accepted (2026-11-23)

## Decision
- `atlases/` (samseg `.npz` atlases, ~70 MB) and `literature/*.pdf` (copyrighted papers) stay local and
  gitignored.
- Tests use a synthetic mesh generated in `tests/conftest.py`.
- Tests that need a real atlas skip when it is absent.

## Consequences
- What we learn from these files must be written down in `docs/knowledge/`: atlas schema and
  statistics in `atlas-format.md`, equations in `samseg-math.md`. A fresh clone must not need them to
  understand the design.
- Expected local layout: `atlases/adult_template/*.npz` and `atlases/subtemplates_12/*.npz`.
