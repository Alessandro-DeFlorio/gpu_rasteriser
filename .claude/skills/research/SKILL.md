---
name: research
description: Start a tetvox research investigation. Writes a brief if needed, then runs the researcher agent (Fable) in the background. Use when the user asks to investigate, review literature or answer a design question.
---

Arguments: a research question, or the id of an existing brief in `docs/research/briefs/`.

1. If no brief exists for the question, write `docs/research/briefs/R<next id>-<slug>.md`. Look at the
   existing briefs for the format: Context, Questions, Deliverable specifics. Give it enough project
   context to stand alone, and link to the relevant `docs/knowledge/` files.
2. Launch the `researcher` subagent in the background with the prompt: "Answer the brief
   `docs/research/briefs/<file>`. Write the report to `docs/research/<id>-<slug>.md`."
3. Add or update an `In progress` entry for it in `docs/ROADMAP.md`.
4. When it finishes, relay its summary to the user and point to the report. Do not turn its
   recommendations into ADRs without the user's approval.
