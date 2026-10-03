# Pre-registration: Study G (context management)

**Status: PLACEHOLDER (BUILD_PLAN B1).** Package B11 writes this pre-registration: G-H1 to G-H3, the single-agent
reference for G-H1, the tuning grid of the context-management arms and the probes (BUILD_PLAN §4). Until then this file
exists so the study orchestrator (`python -m ape.run_study --study study_g`) has a pre-registration to check and hash:
its body holds `[USER: …]` markers, so a live freeze refuses, and an offline rehearsal freezes a filled copy.
- `[PILOT: …]` items are fixed from the micro-pilot and tuning (`runs/study_g/<id>/micro-pilot/micro_pilot.json`).
- `[USER: …]` items need a decision (see `DECISIONS.md`).
- This header is instructions and is never checked. From the first `## ` heading on, any remaining marker blocks the
  freeze.

**Freeze procedure** (as the gate's, GATE_PREREG.md): run the study through the tune
(`uv run --locked python -m ape.run_study tune --study study_g --run-id <id>`), fill every item, set the status to
FROZEN, commit, then `uv run --locked python -m ape.run_study freeze --study study_g --run-id <id>`. Commit
`PROVENANCE.md` afterwards. Only then is Study G's test split generated (`build-test`, seeds from the run's own block in
[23000, 29000)).

## 1. Hypotheses, estimands and tests

[USER: Study G pre-registration (BUILD_PLAN B11)]

## 2. Open design items

[USER: G-H1 single-agent reference]
