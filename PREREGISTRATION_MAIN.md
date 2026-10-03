# Pre-registration: main study (Studies A, B, C, F)

**Status: PLACEHOLDER (BUILD_PLAN B1).** Package B5 writes this pre-registration: the hypotheses as confirmatory or
descriptive (D-028), the estimands, tests, sizes, splits and the freeze. Until then this file exists so the study
orchestrator (`python -m ape.run_study --study main`) has a pre-registration to check and hash: its body holds
`[USER: …]` markers, so a live freeze refuses, and an offline rehearsal freezes a filled copy.
- `[PILOT: …]` items are fixed from the micro-pilot, tuning and pilot data (`runs/main/<id>/pilot/pilot.json`,
  `prereg_items`, keyed by label).
- `[USER: …]` items need a decision (see `DECISIONS.md`).
- This header is instructions and is never checked. From the first `## ` heading on, any remaining marker blocks the
  freeze.

**Freeze procedure** (as the gate's, GATE_PREREG.md): run the study through the pilot
(`uv run --locked python -m ape.run_study pilot --study main --run-id <id> --gate-run-id <gate run>`), fill every item,
set the status to FROZEN, commit, then `uv run --locked python -m ape.run_study freeze --study main --run-id <id>
--gate-run-id <gate run>`. Commit `PROVENANCE.md` afterwards. Only then is the main study's test split generated
(`build-test`, seeds from the run's own block in [13000, 19000)).

## 1. Hypotheses, estimands and tests

[USER: main-study pre-registration (BUILD_PLAN B5)]

## 2. Budget policy

Token caps per task cell: [PILOT: token caps]. The KG arm: [PILOT: KG arm]. S7 targets: [PILOT: S7 targets per cell].
