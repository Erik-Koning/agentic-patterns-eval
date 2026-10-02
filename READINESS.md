# Readiness audit

Status key: ✅ pass · ⚠️ pass with caveat or partial · ❌ fail · ⛔ blocked (waiting on another item or on the user) · ⏳ not started.

**(B)** marks an item that blocks the next phase. Every (B) item must be ✅ before the micro-pilot (checkpoint 6).

Last updated: 2026-09-30.

## Environment

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| E1 (B) | Venv imports all dependencies; lockfile committed | ✅ | Python 3.14.0. inspect-ai 0.3.273, lightrag-hku 1.5.7, apg-core 0.1.0, openai 3.22.1 import cleanly. `uv.lock` is committed. Python 3.12 is **not** usable; see G0 and D-001. | — |
| E2 (B) | `OPENAI_API_KEY` in gitignored `.env`; `models.list()` succeeds | ❌ | 2026-09-30: the shell's `OPENAI_KEY` is **rejected by OpenAI (401 invalid_api_key)**, meaning it was revoked or has expired. `.env` (gitignored, mode 600) still holds that rejected key; replace it with a new one (O-1). No spend occurred. | user |
| E3 (B) | Model IDs chosen per role (agent, kg, build, embeddings); honoured params probed, including reasoning effort | ⛔ | Blocked on E2. `readiness/probe_openai.py --profile gate` probes plain, temperature 0, seed, effort low and strict JSON per role model, and (FX-8) whether effort is honoured: one problem at effort high and low, passing when high uses more reasoning tokens than low (`readiness/smoke_checks.py`; ≈ $0.02 for the gate profile). `readiness/smoke.py`'s `effort` check reads the result. | — |
| E4 | Rate-limit tier / TPM fits the concurrency plan | ⛔ | Blocked on E2. Closed by `readiness/smoke.py`'s `burst` check (FX-8): 24 samples at the profile's `max_connections`, recording failed samples, retries, rate-limit signals and latency, and writing a recommended `max_connections` (capped by the probe's TPM header). Passes `--dry`. | — |
| E5 (B) | Price table (source URL + date) in Inspect's model-cost config | ⚠️ | `config/model_costs.yaml` filled with listed GPT-6 and embedding prices (2026-09-30). Cached-input prices are set equal to input (conservative) until confirmed. Model keys are confirmed at E3. | — |

## APG

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| A1 (B) | Commit or tree hash recorded | ✅ | Tag `apg-eval-baseline` → `d47f7f3` (first commit; secret scan clean). Installed as a git source pinned to the tag; tree hashes match (`PROVENANCE.md`). | — |
| A2 (B) | APG pytest passes, including 66 conformance fixtures | ✅ | `123 passed in 0.09s` on 3.14, run against the installed copy (`-p no:cacheprovider`, no writes to the APG repo). **Fails to import on 3.12** (G0). | — |
| A3 (B) | 1,000-leaf graph: validate < 5 s, route p95 < 1 s, compose < 50 ms, outline < 2.5k tokens | ✅ | `readiness/bench_apg_scale.py`: validate 0.004 s; route p50 0.155 s / p95 0.165 s; compose p95 0.31 ms; shortlisted outline ≤ 504 tokens. **Without embeddings the outline is 42,000 tokens** (G9). | — |
| A4 | Gap register with workarounds | ✅ | See the gap register below. | — |
| A5 | Ported classify prompt string-equal to TS `CLASSIFY_SYSTEM` | ✅ | `tests/test_apg.py::test_classify_prompt_is_string_equal_to_ts_driver` parses the TS source and compares. | — |

## LightRAG

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| L1 (B) | API present in 1.5.7 | ✅ | Present: `aquery_data`, `ainsert_custom_kg`, `initialize_storages`, `initialize_pipeline_status`. `QueryParam` has `mode` (default `mix`), `only_need_context`, `top_k` (40), `chunk_top_k` (20), `max_entity_tokens` (6000), `max_relation_tokens` (8000), `max_total_tokens` (30000). `ainsert(input, ids=, file_paths=)` is available for provenance. Default storages are JSON, NanoVectorDB and NetworkX. **`enable_rerank` defaults to True** (D-004). | — |
| L2 (B) | 5-document live build: entities non-empty, `source_id` maps to our chunk IDs | ⛔ | Needs E2. `readiness/smoke.py` check `L2` (an F7-100 world, ≈ $0.10). Passes `--dry`. | — |
| L3 (B) | 20 concurrent in-loop queries, no loop errors; workspaces disjoint | ⚠️ | **Offline half passes:** `tests/test_lgr.py` runs 20 concurrent samples over two worlds inside Inspect's loop with no loop errors, and neither world's workspace leaks the other's facts. **Still needed:** a repeat on a real extraction index after L2. | — |
| L4 | Query cache off gives two metered keyword calls for two identical queries | ⛔ | Needs E2. `readiness/smoke.py` check `L4_L5`: one metered keyword call per compile. Passes `--dry`. | — |
| L5 | Realized context ≤ `max_total_tokens`, median within the expected band | ⛔ | Needs E2. `readiness/smoke.py` check `L4_L5` (FX-8): fail if any compile exceeds the cap; warn if the median is outside 0.1–1.0 × the cap. This replaces the earlier "±10% of the cap": LightRAG's `max_total_tokens` also budgets its prompt template, so the realized context sits below it by design (offline: 0.29 × the cap). Passes `--dry`. | — |
| L6 | GraphRAG-Bench data, license, eval script and paper model recorded | ✅ | Repo `GraphRAG-Bench/GraphRAG-Benchmark` @ fdbab59: Medical corpus (1 document, ~218k tokens) and 2,062 questions in 4 types. Official accuracy is `Evaluation/metrics/answer_accuracy.py` (gpt-4o-mini judge, T=0, seed 42). LightRAG setup from paper App. H.2. Anchor: `ape.anchor`, `ape.tasks.anchor_graphragbench`, 11 offline tests. Deviations: D-009. | — |

## Design documents

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| D1 (B) | `ORCHESTRATOR_BRIEF_v2.md` §0 filled | ⚠️ | Filled: `kg_framework`, `kg_ingestion_pipeline`, provider. `compute_budget_usd` = 5000 (D-021). **Still TODO-BLOCKING:** `team` (including the skeptic; O-3). Model IDs wait on E3. | user |
| D2 | Claude-specific text adapted for OpenAI | ⏳ | Sampling rejection and preserved-thinking notes in §6.3 and §5.3 of the brief. | — |
| D3 | `EVAL_DESIGN.md` flagged for OpenAI re-pricing | ⏳ | | — |

## Harness (Phase 1b: after integration, before any pilot)

| ID | Check | Status |
|---|---|---|
| H1 (B) | API verification table with doc URLs | ✅ (see API verification table below) |
| H2 (B) | Mock dry run of every arm × family; required log fields present | ⚠️ Offline dry runs pass for S1, S3s, S6, S7, S5o, APGo-q, APG-s (authored, F7/F3), LGRo-q and LGRo-s on F7/F3/F5. LGR-q/s need a real extraction index (after L2). |
| H3 (B) | Classify/keyword calls appear as ModelEvents, one per compile | ✅ Asserted in `tests/test_gate_dry_run.py` (APG: one per compile, minus embedding bypasses) and `tests/test_lgr.py` (LightRAG: exactly one per compile). |
| H4 | Live check: OpenAI accepts tool sets that change across turns | ⛔ (needs E2). `readiness/smoke.py` check `H4`: S3s on F3-60, whose retrieved tool sets change between steps (warns if they never changed). Passes `--dry`. |
| H5 | Human spot-check of 20 tasks per family | ⏳ |
| H6 | One-command gate (FIX_PLAN FX-6): `python -m ape.run_gate all --run-id <id>` runs preflight → build-dev → tune → anchor → pilot → freeze → build-test → test → analyze, each phase resumable, budget-checked and recorded in `runs/<id>/` | ✅ offline end to end (`--offline`: mock models, fake embeddings, oracle indices; `tests/test_run_gate.py`), including the freeze guard on build-test/test, the test split's lock, primary-first budget stops and resume after a failed group. Live needs O-1 and the filled `GATE_PREREG.md`. |
| H8 | Smoke coverage (FIX_PLAN FX-8): live pull mode, error recovery, a concurrency burst, real-embedding retrieval, and the orchestrator's live wiring | ✅ `--dry` (every check passes offline; `tests/test_smoke.py` runs it end to end and tests each check's pass/fail logic). ⛔ live: needs E2/E3. Checks: `pull` (S3s and APG-s pull on F7-100; `search_kb` in ≥ 75% of samples, no errors), `recovery` (a first-attempt harness fault, retried and recorded), `burst` (E4), `retrieval` (S3s recall and APG shortlist gold rate, no LLM; warn < 0.8), `orchestrator` (`run_gate --smoke` preflight → pilot on F7-10 and F3-5, then the freeze refuses). |
| H7 | Decision report (FIX_PLAN FX-7): `python -m ape.analyze_gate --run-id <id>` → `runs/<id>/report/decision.json` and `report.md` with PC1–PC6, the verdict per delivery mode and combined, the NO-GO diagnosis, tables and secondaries | ✅ offline: `tests/test_analyze_gate.py` (every verdict label and PC2–PC6 on synthetic rows with known outcomes) and `tests/test_run_gate.py` (the end-to-end report: PRECONDITION_FAIL naming PC1, which the mock anchor fails; a budget-stopped run's missing cells; a missing `pc1.json`). |

---

## Gap register (A4)

Every APG gap, what it affects, and how the harness handles it. Upstream fixes change APG's tree hash and so require a new pin (`PROVENANCE.md`).

| ID | Gap | Evidence | Effect | Harness workaround |
|---|---|---|---|---|
| G0 | `apg_core` fails to import on Python < 3.14 even though it declares `>=3.11`. | `connectors.py:175-193`: the class-scope `def list(...)` shadows the builtin, so `list[dict]` annotations raise `TypeError` under eager annotation evaluation. | Portability only. Results are unaffected. | Use Python 3.14 (D-001). Suggested upstream fix: `from __future__ import annotations` in `connectors.py`. |
| G1 | `compose` never includes descendants of the target; only the root→leaf path contributes. | `compose.py:37`; determinism spec §5 | Content placed on a category node's children is invisible when the category is routed to. | Authoring rule: policy text goes on **routable leaves**, and category nodes are thin descriptors. |
| G2 | `toolAllowlist` is intersected over the primary path only; secondary matches' allowlists are ignored. | `compose.py:321-330` | Multi-domain F3 tasks can lose the second domain's tools. | Allowlists on leaves only. Measure it as an APG gap; don't patch it. |
| G3 | No exception or override relation between sibling policies. | Schema has only merge modes along the path + `composition.priority` | Relational F7 (precedence/exceptions) must be encoded indirectly. | Leaf-level `bring` edges with `recursiveBring`; precedence stated in the text. |
| G4 | `route` has no subtree or scope parameter. | `router.py:29` | Specialist agents can't route inside their own subtree. | Build a sub-graph document per specialist. |
| G5 | Module-global connector registry; stateful `ScriptedLlm` queues. | `connectors.py:13-51` | Cross-talk risk under Inspect's concurrent samples. | Pass a per-call connectors dict and a fresh `ScriptedLlm` per call. Never use `bind_connectors`. |
| G6 | `precompute_embeddings` fills only missing vectors, so vectors go stale after text edits. | `embed.py:20` | Wrong shortlist after re-authoring. | Always use `force=True` after a build. |
| G7 | The Python validator has no JSON-Schema check (only TS/Ajv does). | `tests/test_templates.py:3-4` | Malformed authored graphs could pass `validate_graph`. | Add a `jsonschema` check against `schema/apg.schema.json`. |
| G8 | No Python LLM or embeddings connector. | `connectors.py:35-67` (only `ScriptedLlm` / `MapEmbeddings`) | The kernel can't route with a real model. | Adapter-side classify through Inspect (record/replay) and cached OpenAI embeddings. |
| G9 | Without embeddings, the outline lists every eligible node. | A3: 42,000 tokens at 1,000 leaves vs 504 shortlisted | Routing cost grows linearly with the number of policies. | Embeddings are mandatory; tune `shortlistK` on dev. |
| G10 | Cosine scoring is pure Python, O(N·dim). | A3: route p50 155 ms at 1,000 × 1536 | Latency. | Acceptable. Run in a worker thread. |
| G11 | The compose token budget is not a hard cap: path constraints are never dropped and separators aren't counted. | determinism spec §5; `compose.py:211` | Realized context can exceed `maxPromptTokens`. | Pass a tiktoken `countTokens`; match arms on **realized** tokens. |
| G12 | `apply_changeset` deep-copies and rebuilds on every op, which is quadratic. | `mutation.py:19, 256-259` | Slow bulk authoring. | Build raw documents, then run one validation. |
| G13 | The TS `buildMonolith` differs from compose: it drops outputFormat and recursive brings and uses a different slot order. | `examples/wisdom-chat/src/lib/compare.ts:136-152` | Unequal content between the monolith and graph arms. | Port it with parity fixes (module 3). |
| G14 | APG has no license. | `README.md:146-147` | Blocks an open release. | Internal use only until the owner decides. |
| G15 | Neither repo has git history. | `git rev-parse HEAD` fails | Reproducibility. | Tree-hash pin (`PROVENANCE.md`) until a commit/tag exists. |

---

## API verification table (H1)

Each API was checked two ways: introspected in the installed **inspect-ai 0.3.273** (signatures and fields), and exercised by the offline test suite. Doc URLs come from the official index `https://inspect.aisi.org.uk/llms.txt` (fetched 2026-09-30).

| API used | Where in `ape` | Verified locally | Documentation |
|---|---|---|---|
| `get_model(role=...)`, `model_roles=` on `eval`/`Task`, `--model-role` | `apg/arm.py`, `lgr/adapter.py`, tests | ✅ signature; kg-role ModelEvents asserted in tests | https://inspect.aisi.org.uk/models.html.md (§ Model Roles) |
| `Model.generate(input, tools, config)` | `agent/kb_react.py` | ✅ | https://inspect.aisi.org.uk/reference/inspect_ai.model.html.md |
| `GenerateConfig(response_schema=ResponseSchema(name, json_schema, strict))`, `JSONSchema` | `apg/classify.py`, `lgr/adapter.py` | ✅ fields | https://inspect.aisi.org.uk/structured.html.md |
| `GenerateConfig.reasoning_effort`, `seed`, `temperature` | readiness probe (E3) | ✅ fields present | https://inspect.aisi.org.uk/reasoning.html.md |
| `ToolDef(fn, name, description, parameters=ToolParams(...))` | `worlds/env_tools.py` | ✅ kwargs callable with explicit schema | https://inspect.aisi.org.uk/tools-custom.html.md |
| `execute_tools(messages, tools)` (custom loop) | `agent/kb_react.py` | ✅ | https://inspect.aisi.org.uk/agent-custom.html.md |
| `store()` / `TaskState.store` | `worlds/env_tools.py`, scorers | ✅ | https://inspect.aisi.org.uk/agent-custom.html.md; reference `inspect_ai.util` (URL not checked) |
| `@solver`, `@scorer`, `Score`, metric dicts | `agent/kb_react.py`, `scorers/success.py` | ✅ | https://inspect.aisi.org.uk/metrics.html.md |
| Epochs | gate runs `--epochs 3` | ✅ used in loader test | https://inspect.aisi.org.uk/metrics.html.md |
| `set_model_cost`, `ModelCost(input, output, input_cache_write, input_cache_read)`, `--model-cost-config` | E5 (`config/model_costs.yaml`) | ✅ fields | https://inspect.aisi.org.uk/setting-limits.html.md (§ Model Cost) |
| `token_limit`, `cost_limit`, `message_limit`, `time_limit`, `working_limit`, `turn_limit` on `Task` | gate caps (PC5) | ✅ Task params | https://inspect.aisi.org.uk/setting-limits.html.md |
| `EvalSample.role_usage`, `model_usage`, `total_time`, `working_time` | `analysis/gate_stats.py` | ✅ fields; loader test on real logs | https://inspect.aisi.org.uk/eval-logs.html.md |
| `read_eval_log`, `samples_df` | `analysis/gate_stats.py` | ✅ | https://inspect.aisi.org.uk/eval-logs.html.md, https://inspect.aisi.org.uk/dataframe.html.md |
| `mockllm/model` with `custom_outputs` callable | dry-run tests | ✅ | provider list: https://inspect.aisi.org.uk/providers.html.md (mockllm not named there; behaviour verified by tests) |
| Caching (off for every arm) | gate runs | — (default off) | https://inspect.aisi.org.uk/caching.html.md |
| Batch mode (not used for agent loops) | — | — | https://inspect.aisi.org.uk/models-batch.html.md |

---

## One-command live readiness (after O-1)

```
uv run python readiness/probe_openai.py --list --profile gate      # E2-E4, effort honoured (≈ $0.02)
uv run python readiness/smoke.py                                   # every check below, cap --max-usd 3 (projected ≈ $1.4)
uv run python readiness/smoke.py --only pull,burst                 # re-run some checks (with the checks they need)
```

`readiness/smoke.py` (FX-8) runs, on the gate profile: `effort` (from the probe), `L2` (live LightRAG extraction and source mapping), `D017` (the build-quality check: authoring `id_coverage` and LightRAG ID coverage ≥ 0.95), `H4` (changing tool sets), `L4_L5` (one keyword call per compile; realized context against `max_total_tokens`), `APG`, `pull`, `recovery`, `burst`, `retrieval` and `orchestrator` (the real `run_gate` at SMOKE_SCALE through pilot, then a refused freeze). Before spending it prints each check's projected cost and refuses a total over `--max-usd`; before each check it stops if spend plus that check's projection would pass the cap. It writes `cache/smoke/report.json` and `report.md`, and exits non-zero when a check fails. Everything lives in `cache/smoke/`; nothing touches `config/`, `PROVENANCE.md` or `runs/`. `--dry` passes every check offline ($0, ~20 s; `tests/test_smoke.py`).

Projected live cost on the gate profile (conservative priors, 2026-10-01): L2 $0.10 · D017 $0.04 · H4 $0.01 · L4_L5 $0.01 · APG $0.01 · pull $0.05 · recovery $0.01 · burst $0.10 · retrieval ~$0 · orchestrator $1.06 (build-dev $0.08, tune $0.13, anchor $0.52, of which $0.47 is the GraphRAG-Bench index, pilot $0.32) = **$1.38**. With the Luna anchor fallback (no gpt-4o-mini in the probe) it is $1.25. The smoke's anchor index is isolated, so the real gate builds its own.

## Running the gate (after the smoke passes)

```
uv run python -m ape.run_gate all --run-id gate-1    # preflight → build-dev → tune → anchor → pilot, then stops at freeze
# fill GATE_PREREG.md's [PILOT: …] items from runs/gate-1/pilot/pilot.json and its [USER: …] items (D-017 builder, O-3 names);
# set Status to FROZEN; commit it with config/selected.yaml, config/s7_targets.json, config/budget_calibration.yaml
uv run python -m ape.run_gate freeze --run-id gate-1  # hashes the design; refuses on a marker, a dirty tree or a failing PC1
git commit PROVENANCE.md                              # the freeze record
uv run python -m ape.run_gate all --run-id gate-1    # build-test → test → analyze; finished phases are skipped
```

- **Progress:** each phase writes `runs/gate-1/<phase>/manifest.json` (status, spend, logs). Re-running the same command resumes after a crash, a failed group or a budget stop.
- **Result:** `runs/gate-1/report/report.md` (the verdict, PC1–PC6, tables) and `decision.json`.
- **Spend:** the guard refuses any phase projected over what is left of $5,000. The gate's projection is $201 conservative / $176 expected (`BUDGET.md`).
- **Before the freeze:** H5 (spot-check) and O-3 (names) must be done, and the D-017 builder decided.
