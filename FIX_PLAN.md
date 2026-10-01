# Fix plan: from "harness works offline" to "gate runs unattended within budget"

**Date:** 2026-09-30.

**Source of the issues:** the review of model settings, cost, completion, recording, timing and smoke coverage (session notes, 2026-09-30).

**How we work:**
- Fixes are implemented **one at a time, in the order below**, each by a focused subagent.
- Each fix is reviewed (diff read, full test suite run, semantics checked against this plan) and committed before the next starts.
- Every fix lands with offline tests: `mockllm` models, fake embeddings, no network, no spend.

| ID | Fix | Why it matters |
|---|---|---|
| FX-1 | Model roles and reasoning effort wired from one config | Runs currently use the default effort (medium), not the decided high/low (D-015) |
| FX-2 | Price table applied to every run, plus a preflight check | Inspect logs carry no $; the smoke spend cap ignores Inspect-side spend |
| FX-3 | Error tolerance and resumable runs | One transient sample error currently fails a whole 768-task run |
| FX-4 | Build throughput (parallel builds) | LightRAG builds take about an hour per large world at default concurrency |
| FX-5 | Cost model and right-sized run matrices (≤ $5K program) | The current program estimate is $6K–65K, dominated by Study G on Astra |
| FX-6 | Gate orchestrator (`run_gate.py`) with phases, manifests, calibration and freeze guard | No one-command run; the pilot → calibration → freeze → test sequence is manual |
| FX-7 | Decision report (`analyze_gate.py`) | Preconditions, invariants and the verdict are not computed automatically |
| FX-8 | Smoke-test coverage | Pull mode, error recovery, concurrency, L5, anchor and tuning are not exercised live |

FX-5 comes before FX-6 because the orchestrator encodes the run matrices that FX-5 sets.

---

## FX-1: Model roles and reasoning effort from one config

**Problem.** D-015 fixes the effort per role (agent high, kg low, build high). Nothing in the code sets `reasoning_effort`. The CLIs pass bare model names, so every call runs at the model default (medium for GPT-6 Luna). The build client (OpenAI SDK) sets no effort either.

**Decision.**
- **One config, `config/models.yaml`,** defines each role: `agent`, `kg`, `build`, `judge` (anchor) and `embeddings`. For each it gives the model, `reasoning_effort` and optional `max_tokens`.
  - Named profiles: `gate` (default) and later `study_g_luna`, `study_g_sol`, `study_g_astra`.
- **A new module, `src/ape/models.py`,** provides:
  - `load_profile(name)`
  - `agent_model()`, `role_models()` returning Inspect `Model` objects built with `GenerateConfig(reasoning_effort=…)`
  - `build_settings()` returning the model and effort for `BuildLlm`
- **`BuildLlm` (OpenAI SDK)** passes `reasoning_effort` to `chat.completions.create`.
- **Per-call configs** (APG classify's `response_schema`, LightRAG keyword schema) must **merge with**, not replace, the role's effort. Verify that Inspect merges a per-call `GenerateConfig` into the model's base config. If it doesn't, merge explicitly.
- **Inspect logs** already record the main model's generate config. Make sure role configs are recorded too, in `eval.model_roles` or in task metadata.
- **Rejected alternative:** CLI flags only. They're easy to forget, and roles take effort through model args inconsistently.

**Acceptance tests.**
- A `mockllm` `custom_outputs` callable receives `config` and asserts `reasoning_effort == "high"` on agent calls and `"low"` on kg calls (APG classify and LightRAG keywords). This must hold through the strict-schema calls.
- `BuildLlm` sends `reasoning_effort` (via a fake client that records kwargs).
- Every role's effort is present in the eval log.

---

## FX-2: Price table everywhere, plus preflight

**Problem.**
- No `inspect eval` call passes `model_cost_config`, so `ModelUsage.total_cost` is `None`.
  - `load_results` correctly refuses such logs.
  - `readiness/smoke.py`'s spend cap counts Inspect spend as $0.
- Nothing checks that every configured model has a price before money is spent.

**Decision.**
- Every eval entry point passes `model_cost_config=config/model_costs.yaml`: gate runs, tuning, anchor, smoke.
- Ledger pricing uses the same table (already the case in `analysis/cost.py`).
- `preflight()` in `src/ape/models.py` checks, for the chosen profile:
  - every role model has a price entry
  - the price file parses
  - `OPENAI_API_KEY` is present (live runs only)

  It fails fast with a clear message.
- **Smoke spend cap:** fixed and tested, using a `mockllm` price entry in a temp cost file.

**Acceptance tests.**
- A mock eval with a cost config gives non-`None` `total_cost`.
- The smoke spend guard stops when a tiny cap is exceeded.
- Preflight fails on a missing price.

---

## FX-3: Error tolerance and resumable runs

**Problem.** `fail_on_error` defaults to failing the eval on the first sample error, and runs are single `eval()` calls with no resume. One transient API error can kill hours of work, and there is no clean continuation.

**Decision.** A wrapper, `src/ape/runner.py: run_evals(tasks, log_dir, profile, …)`, on top of Inspect's `eval_set`:
- `retry_attempts` (default 3) with `retry_wait` back-off, for whole-task failures.
- `retry_on_error = 2` for sample-level retries (transient API and tool errors).
- `fail_on_error = 0.02`: up to 2% errored samples are tolerated, which matches PC5. Above that the task fails, and the next invocation resumes it.
- **Resume:** re-invoking with the same `log_dir` completes only unfinished work (`eval_set` semantics).
- **Concurrency:** `max_connections` per model and `max_samples` come from the profile, so the limits are tuned once (from FX-8's concurrency probe).
- **Integration:** it also applies FX-1 models and FX-2 prices, so every caller gets the same behaviour.

**Acceptance tests.**
- A mock model that raises on the first attempt for some samples: the run completes and the retries show in the logs.
- A task that fails deterministically: the run fails, and with the same log directory a second invocation after the cause is fixed finishes without re-running completed tasks.
- More than 2% failing samples: the task is marked failed.

---

## FX-4: Build throughput

**Problem.**
- LightRAG defaults: `llm_model_max_async = 4`, `max_parallel_insert = 3`, and one gleaning pass (2 extraction calls per chunk). A large F7-1000 world (~550 chunks) takes about an hour, and builds run serially.
- APG authoring concurrency is fixed at 8.

**Decision.**
- **Configurable concurrency:** `APE_BUILD_LLM_CONCURRENCY` (default 16) maps to LightRAG `llm_model_max_async` and `max_parallel_insert`, and to the APG authoring semaphore. The OpenAI client in `BuildLlm` uses `max_retries = 6`.
- **Parallel worlds:** builds run across worlds in a **process pool** (one world per process; `APE_BUILD_WORKERS`, default 4). LightRAG keeps process-global shared state, so processes give clean isolation; threads or coroutines in one process would not.
- **Idempotent:** a world whose manifest exists with a matching world hash is skipped, so a crashed build resumes.
- **Unchanged:** the extraction protocol (D-003: one LightRAG document per shared chunk) and the gleaning setting. Both systems keep the same build model and effort.

**Acceptance tests.**
- Oracle-kind builds of 3 worlds in parallel processes produce valid manifests identical to serial builds.
- Re-running skips completed worlds.
- APG authoring with `perfect_author` in parallel matches serial output byte for byte.

---

## FX-5: Cost model and right-sized run matrices (target ≤ $5,000 for the whole program)

**Problem.** The current estimate is gate $0.3–0.8K + main study $0.5–3K + **Study G $5–60K**. Study G is dominated by Astra (GPT-6 Astra: $10 / $50 per M tokens, high effort) on 60-case sessions with a 64K nominal window.

**Decision 1: a parametric cost model** (`src/ape/budget.py`). Every planned run cell is described by:
- model, effort and price
- samples (tasks × epochs × modes)
- agent calls per sample
- mean input tokens per call, cached share and cached-input price
- output tokens per call (including reasoning)
- kg calls per call
- build calls

It prints the cost per study, per phase and per tier, and the total. The defaults come from measured context sizes (APG ≈ 120, LightRAG ≈ 2,300, S3s ≈ 2,000, F7-1000 monolith ≈ 83K) and conservative output assumptions. **After the pilot, it recalibrates from real logs** (actual tokens per call), and the orchestrator refuses a phase whose projected cost exceeds the remaining budget.

**Decision 2: right-sized matrices.**

| Area | Current plan | New plan | Effect on result quality |
|---|---|---|---|
| Gate primary | 16 worlds/cell × 12 tasks × 3 epochs, push+pull on F7 | **unchanged** | none (runs are cheap on Luna) |
| Gate diagnostics (S1, S5o, S6, S7) | 100 tasks/cell × 3 epochs | 100 tasks/cell × **2** epochs; S7 keeps 3 (it is in the GO rule) | small: diagnostics are descriptive |
| Gate secondaries (id_only, matched budget ×2, TE-all) | 100 tasks/cell × 3 epochs each | id_only and TE-all: 100 × **2**; matched budget: **one** budget (≈300 tokens, the informative one) × 100 × 2 | small: secondary, descriptive |
| Builds | Sol if Luna fails the D-017 check | unchanged; the cost model flags the Sol path (≈ +$1.2K) as needing approval | none |
| Main study | primary tier plus tier replication incl. Astra | primary **Luna**; tier replication **Luna/Sol only** (Study G owns the Astra tier) | mechanism results generalize to the Luna and Sol tiers, not Astra |
| Study G window | nominal W = 64K, compaction threshold 40K | **W = 32K, threshold 20K** | Context pressure is relative to W, so management effects still show at roughly half the tokens per call. Not testable: management beyond 32K absolute context. |
| Study G session length | F8 N = 60 (all tiers) | **N = 40** for Luna and Sol, **N = 30** for Astra | With the smaller W, sessions still cross W several times; the degradation curve comes from item position |
| Study G context arms on Astra | 7 arms × 12 sessions × 3–5 epochs | **4 arms** (CM0, CM-sum, CM-todo, O-state) × **5 sessions × 2 epochs** | Wider CIs for Astra cells. The capability trend rests on 5 points (Luna low/high, Sol low/high, Astra high). |
| Study G on Sol | 7 arms × 12 × 3 | **5 arms** (CM0, CM-prune, CM-sum, CM-todo, O-state) × **8 × 2** | moderate |
| Study G on Luna | 7 arms × 12 × 3 | **7 arms × 10 × 3**, plus a Luna-low effort point | small |
| Study G topology block | 4 arms × 3 tiers × 12 × 3 at N = 30 | **N = 20**; Luna and Sol: 4 arms × 8 × 2; **Astra: S1 and M2 only × 5 × 2** | G1 at the top tier estimated from 2 arms |
| Study G Tier B (lesson writing, consolidation, subagent isolation) | after the core | **only if budget remains** after the core, on Luna | none for the core hypotheses |

**Budget allocation** (the cost model must confirm before the plan is accepted):

| Bucket | Target |
|---|---|
| Gate phase (smoke, anchor, builds on Luna, tuning, pilot, test) | ≤ $700 |
| Main study | ≤ $1,000 |
| Study G | ≤ $2,700 |
| Contingency | ~$600 |
| **Total** | **≤ $5,000** |

**Not yet known, so the model treats them as parameters:**
- the GPT-6 cached-input price (defaults to the full input price: conservative)
- real output and reasoning tokens per call at high effort (defaults to 1,500)
- calls per sample

The pilot replaces all three defaults with measurements.

**Doc updates:**
- `GATE_PREREG.md` §4: diagnostic and secondary sizes.
- `CONTEXT_MANAGEMENT_AUDIT.md` §8: the new Study G matrix.
- `ORCHESTRATOR_BRIEF_v2.md` §7.2: main-study tiers.
- `DECISIONS.md`: a new D-entry.

**Acceptance tests.**
- The cost model reproduces hand-computed costs for two small cells.
- The printed total for the new plan is ≤ $5,000 with documented defaults.
- Recalibration from mock logs updates the per-call token parameters.

---

## FX-6: Gate orchestrator (`run_gate.py`)

**Problem.** The gate is ~10 manual steps. The pilot → S7 sizing → LightRAG budget calibration → pre-registration freeze → test-world generation sequence is error-prone, and nothing ties one run's logs together.

**Decision.** `python -m ape.run_gate <phase> --run-id <id>`, or `all`. Each phase is idempotent and resumable. Each writes `runs/<id>/<phase>/manifest.json` with its inputs, configs, git commit, log files, cost so far and a status.

| Phase | Does |
|---|---|
| `preflight` | FX-2 preflight; probe results present; APG pin matches `PROVENANCE.md`; warn if the git tree is dirty |
| `build-dev` | Generate dev worlds (gate cells, F5, the id_only and messy variants), embed, author APG, build LightRAG (FX-4) |
| `tune` | `ape.tuning` for APG, LightRAG and S3s → `config/selected.yaml` (APG*, LGR*, S3s settings) |
| `anchor` | PC1: anchor index, hybrid and naive runs, `pc1_from_logs` → `pc1.json` |
| `pilot` | Pilot worlds; the selected arms plus diagnostics at 1 epoch, push and pull. Writes `config/s7_targets.json` (APG* realized median per cell, read by S7 through `APE_S7_TARGETS`) and `config/budget_calibration.yaml` (LightRAG and S3s caps that land within ±25% of the matched budget, at most 3 calibration iterations; APG's fill settings). Re-runs the NI power simulation with pilot σ (`pilot/power.json`), recalibrates the cost model, and writes `pilot/pilot.json` with a value for every `[PILOT: …]` placeholder. Refused once the run is frozen. |
| `freeze` | Refuses while `GATE_PREREG.md`'s body (from §1; the header is instructions) still has a `[PILOT` or `[USER` marker, a tracked file or untracked code is uncommitted, or PC1 fails. Records sha256 of `GATE_PREREG.md`, `config/selected.yaml`, `config/s7_targets.json`, `config/budget_calibration.yaml`, `config/models.yaml`, `config/model_costs.yaml`, `config/run_plan.yaml` and `config/tuning_grid.yaml`, plus the commit, the analysis-code commit and the APG pin, in `runs/<id>/freeze.json` and `PROVENANCE.md`. A run is frozen once: afterwards `tune` and `pilot` refuse (even with `--force`). Offline it is a rehearsal on a filled copy. `require_frozen` is the guard for `build-test` and `test`. |
| `build-test` | Refuses unless `freeze.json` exists and every frozen file still matches its hash. Then generates the test worlds (`gate.build.test`, `gate.build.test-id-only`, `gate.build.test-f5`) and builds graphs and indices. It is the only way to generate the test split: `python -m ape.build --split test` refuses unless build-test set `APE_TEST_SPLIT_RUN`. |
| `test` | The same freeze guard. Through FX-3's runner: every run cell of the frozen plan's test, diagnostics, F5 and secondaries phases (arms × cells × deliveries × epochs), with FX-1 models and FX-2 prices. One eval set per cell and environment, in `runs/<id>/test/<cell>/<group>-<hash>/`: the selections' knobs together; the matched-budget cell adds `budget_calibration.yaml`'s knobs on top; LightRAG naive runs as LGR* with `APE_LGR_MODE=naive`. The GO rule's cells (`gate.test.f7`, `gate.test.f3`, `gate.diag.s7`) run first, and the FX-5 budget check runs before every group, so a budget stop leaves the primary complete. A failing group is recorded, the others still run, and a re-run resumes. `test/manifest.json` lists every cell's groups, logs and status for FX-7. |
| `analyze` | FX-7 report (`ape.analyze_gate`). Needs no other phase complete: it reports what exists, a missing input fails its precondition by name, and `all` runs it after a failed or budget-stopped test too. |

**Acceptance test.** An end-to-end `all` run in offline mode:
- tiny sizes
- `mockllm` agent and kg
- fake embeddings
- `perfect_author`
- oracle LightRAG indices

It must produce every manifest, freeze record and report, and the freeze guard must block test-world generation when a frozen file changes.

---

## FX-7: Decision report (`analyze_gate.py`)

**Problem.** `gate_stats.decide()` exists, but PC1–PC6, the invariants, per-mode verdicts, secondaries, costs and the error-label tables are not assembled.

**Decision.** `python -m ape.analyze_gate --run-id <id>` writes `runs/<id>/report/decision.json` and `report.md`, containing:

- **Preconditions:**
  - PC1 from `pc1.json`
  - PC2: F5 LGR* ≥ LightRAG naive
  - PC3: invariants
  - PC4: realized tokens vs configured budget, and ±25% matched-budget calibration
  - PC5: error rate < 2% and cap-hit rate < 10% per arm, where a cap hit is `turns_used == max_turns` with no answer
  - PC6: tuning log complete, every system within budget, selection recorded
- **Verdict:**
  - per delivery mode for the F7 cells (push, pull) and push for F3, via `decide(apg_arm=APG*, lgr_arm=LGR*)`
  - combined verdict labels: GO / GO_PUSH_ONLY / GO_PULL_ONLY / GO_WITH_COST_FLAG / INCONCLUSIVE / NO_GO / PRECONDITION_FAIL
- **Tables:**
  - per-cell Δ with CIs
  - success, partial credit, evidence recall (union, first, per-step), error-label counts per arm
  - cost per query and the cost ratio (`analysis/cost.py`)
  - latency (`total_time`, `working_time`, `compile_ms`)
  - determinism (epoch agreement)
- **Secondaries,** reported separately and never pooled: id_only, matched budget, TE-all, F5, messy (if run).

**Acceptance test.** Synthetic logs from mock evals with known outcomes produce the expected verdict and precondition flags. Missing inputs (e.g. no anchor) produce PRECONDITION_FAIL with a named reason, not a crash.

**Done** (`src/ape/analyze_gate.py`, the `analyze` phase of `ape.run_gate`; statistics in `analysis/gate_stats.py`):
- Each mode's verdict pools the four gate cells with equal weights: push = F7 push + F3; pull = F7 pull + the same F3 push cells. S7 (push) is the placebo for both. GATE_PREREG §3 now says so.
- Arms are labelled by their declared plan names (APG*, LGR*, LGR-naive, ...), taken from the test manifest. `gate_stats.task_means` refuses to average one arm's push and pull runs in a cell.
- The NO-GO diagnosis is always computed: S5o against LGR* on shared tasks, plus each failed sample's pipeline miss (`gate_stats.pipeline_miss`: not in graph, shortlist, classify, compose, evidence, tool exposure, or delivered but failed).
- Choices where the prereg is silent are in `analyze_gate.CHOICES`, and the report lists them too.
- Tests: `tests/test_analyze_gate.py` covers synthetic rows with known outcomes for every verdict label, PC1–PC6 passing and failing, pull pooling and the mode-mixing guard. `tests/test_run_gate.py` covers the offline end-to-end report, a budget-stopped run and a missing `pc1.json`.

---

## FX-8: Smoke-test coverage

**Problem.** The smoke tests miss parts the experiment depends on: high effort actually honoured, pull mode with a real model, recovery from errors, real concurrency, L5, anchor and tuning wiring, and retrieval quality with real embeddings.

**Decision.** Extend `readiness/probe_openai.py` and `readiness/smoke.py`, both on FX-1 profiles and FX-2 prices:

| Check | How | Pass criterion |
|---|---|---|
| Effort honoured | Probe `reasoning_effort` high and low per role model; compare reasoning tokens | Both accepted; high uses more reasoning tokens than low |
| Pull mode live | F7-100, S3s and APG-s with `delivery=pull`, 4 tasks | `search_kb` called in ≥ 75% of samples; errors 0 |
| Error recovery | One task through the FX-3 runner with an injected failing first attempt (harness-side fault, not the model) | Completes; retry recorded |
| Concurrency burst | 24 samples at the profile's `max_connections`; record 429s, retries and the latency distribution | No failed samples; recommended concurrency written to the report |
| L5 | LightRAG realized context vs `max_total_tokens` | Realized ≤ cap; median within the expected band |
| Anchor wiring | 2 questions per type, hybrid and naive, judge role | Completes; `pc1_check` runs (no reproduction claim) |
| Tuning wiring | 2 candidates × 1 dev cell × 1 world × 2 tasks | Log and selection written |
| Real-embedding retrieval | F7-100, no LLM: S3s policy recall at budget; APG shortlist gold rate | Reported; warn if < 0.8 |

The cap is raised to `--max-usd 3`, since the anchor index build on Luna is ≈ $0.5. Everything stays in `cache/smoke/`, and `--dry` must keep passing offline.

**Implemented (2026-10-01).**
- `readiness/smoke.py` runs each row as a named check (`--only`, `--skip`; a check's prerequisites are added). The pass/fail logic is in `readiness/smoke_checks.py`; the probe's effort check is in `readiness/probe_openai.py`. Error recovery uses a smoke-only task wrapper (`readiness/smoke_fault.py`); nothing in `src/ape` changed for it.
- Anchor and tuning wiring run through the real orchestrator. `python -m ape.run_gate --smoke` (`SMOKE_SCALE`) is OFFLINE_SCALE's sizes on F7-10 and F3-5, live, isolated under `cache/smoke/runs/`, and priced at those sizes. Smoke runs preflight → build-dev → tune → anchor → pilot, then checks that the freeze refuses. A smoke run never freezes. At smoke size, a budget calibration that does not converge is a warning.
- Spend:
  - The script prints a projected cost per check and refuses to start over `--max-usd`.
  - Before each check it stops if spend plus that check's projection would pass the cap.
  - The orchestrator also runs under run_gate's own guard (`budget_usd`).
  - The gate profile projects $1.38 (with the GPT-6 Luna anchor fallback, $1.25), so the $3 default stands.
- Output: `cache/smoke/report.json` and `report.md`.
- `--dry` passes every check, and `tests/test_smoke.py` runs it end to end (~20 s).
- Smoke projection prices F7-100 at its measured 58 chunks (`budget_assumptions.yaml`). No plan cell uses F7-100, so the totals are unchanged.
- L5's criterion is now "≤ `max_total_tokens`, median within 0.1–1.0 × the cap" (READINESS L5).

---

## Definition of done

1. All eight fixes are merged with offline tests; the full suite passes.
2. `python -m ape.run_gate all --offline` passes end to end.
3. `READINESS.md` is updated: H2 and the new smoke checks have status rows.
4. The cost model prints a ≤ $5,000 program total.
5. Once a working key exists (O-1), the sequence is `run_gate preflight` → smoke → `run_gate all`, with a stop at `freeze` for human sign-off.
