# agentic-patterns-eval

A pre-registered evaluation of agent architectures (single agents with different knowledge delivery, multi-agent teams, context management) on synthetic task families with programmatic scoring. It is built on [Inspect AI](https://inspect.aisi.org.uk/) and runs on OpenAI's GPT-6 models. Accuracy, tokens, cost and wall-clock are measured with world-clustered statistics.

## Studies

| Study | Question | Pre-registration | Budget (conservative) |
|---|---|---|---|
| **Gate** | Is APG (the Adaptive Prompt Graph, [EvolvingWisdomAgents](https://github.com/Erik-Koning/EvolvingWisdomAgents)) non-inferior to LightRAG? The winner becomes the main study's KG arm. | `GATE_PREREG.md` | $220 |
| **Main** | Which mechanism drives multi-agent gains (decomposition, isolation, ensembling, communication, specialization), and does KG delivery substitute for a team? | `PREREGISTRATION_MAIN.md` | $827 |
| **Study G** | Do context-management strategies (prune, summarise, todo, reset, native compaction) and topologies keep their value as models get stronger? | `PREREGISTRATION_G.md` | $1,950 |

The program costs $2,996 conservative ($2,047 expected) against a $5,000 budget (`BUDGET.md`).

**Status:** every arm, both study runners, the analyses and the pre-registrations are built, independently reviewed (`BUILD_REVIEW.md`) and tested offline. Nothing has run live. What remains is listed in `READINESS_AUDIT.md`.

## Setup

```bash
uv sync --locked                      # Python 3.14; installs apg-core from the APG repo (below)
echo "OPENAI_API_KEY=<key>" > .env    # gitignored; never commit it
export APE_BACKUP_DIR=/path/to/backup # recommended: run outputs are mirrored there
```

**APG** lives in its own repository, [Erik-Koning/EvolvingWisdomAgents](https://github.com/Erik-Koning/EvolvingWisdomAgents) (Python kernel `apg-core` plus TypeScript packages). This project installs `apg-core` from it, pinned to the tag `apg-eval-baseline` (commit `64edf5d`; `pyproject.toml`, `uv.lock`, `PROVENANCE.md`).

## Run

**Offline** (mock models, no network, no cost):

```bash
uv run pytest -q                                              # ~15 min
uv run python -m ape.run_gate all --run-id rehearsal --offline
uv run python -m ape.run_study all --study main --run-id rehearsal --offline
uv run python -m ape.run_study all --study study_g --run-id rehearsal --offline
uv run python readiness/smoke.py --dry
```

**Live.** Each step spends money. Run them in order:

```bash
uv run python readiness/probe_openai.py [--study study_g] [--pin]  # model capabilities, snapshots, CM-native support
uv run python readiness/smoke.py                                   # live checks, capped at $4; required by every runner
uv run python -m ape.run_gate all --run-id gate-1                  # stops at the freeze: fill and commit GATE_PREREG.md
uv run python -m ape.run_study all --study main --run-id main-1 --gate-run-id gate-1
uv run python -m ape.run_study all --study study_g --run-id g-1
```

Every runner is resumable: re-run the same command to continue. Each phase is guarded by the study's budget, the program's spend ledger and a fresh smoke check. The test split is generated only after the freeze. Cost projections come from `uv run python -m ape.budget`.

**Before a live run you need** a valid key, role names and sign-offs in the three `config/tuning_grid*.yaml` files, and the spot-check in `readiness/spotcheck.md`.

## Layout

| Path | Contents |
|---|---|
| `src/ape/worlds/` | Seeded task generators: F1 breadth, F2 dependency chains, F3 tool load, F5, F7 policies, F8 long sessions |
| `src/ape/agent/` | Agent loop (`kb_react`), delivery arms, multi-agent arms (`multi/`), Study G sessions and context policies |
| `src/ape/apg/`, `src/ape/lgr/`, `src/ape/kb/` | APG, LightRAG and flat-retrieval knowledge delivery |
| `src/ape/analysis/` | Statistics, power simulations and hypothesis tables for each study |
| `src/ape/run_gate.py`, `run_study.py` | Orchestrators: phases, freezes, budget guards |
| `config/` | Run plan, models, prices, tuning grids, budget assumptions |
| `readiness/` | Probe, smoke checks, spot-check sheet |
| `tests/` | Offline test suite |

## Documents

- **Plan and status:** `BUILD_PLAN.md`, `READINESS_AUDIT.md`, `BUILD_REVIEW.md`
- **Design:** `ORCHESTRATOR_BRIEF_v2.md`, `CONTEXT_MANAGEMENT_AUDIT.md`, `HYPOTHESES.md`
- **Decisions:** `DECISIONS.md`, a dated log (D-000 onward) of every design and statistics choice, with open items for the user
- **Money:** `BUDGET.md`
