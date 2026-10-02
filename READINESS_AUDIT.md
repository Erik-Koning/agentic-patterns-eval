# Readiness audit (2026-10-02)

**Scope:** the whole program (the APG vs LightRAG gate, the main study and Study G) at commit `03e50dd`.

**Method:**
- An independent agent with no stake in the code audited every component, hypothesis and document.
- I spot-checked its consequential claims against the code; every one held.
- This document merges that audit with what this round built and fixed.

**Offline state:**
- 290 tests pass.
- `run_gate all --offline` runs all nine phases.
- `readiness/smoke.py --dry` passes.
- The cost model prints **$4,743 conservative / $3,223 expected**.
- Nothing has run live: the OpenAI key is invalid (O-1).

## 1. Bottom line

| Study | Readiness | What is true now |
|---|---|---|
| **Gate** | **≈ 80%** | Everything is built and runs end to end offline: arms, orchestrator, freeze and test-split lock, the decision report, and smoke coverage. What remains is user input (key, names, spot-check, builder decision), the first live checks, and a short hardening list (§3). |
| **Main study** | **≈ 25%** | The task families exist: F1, F2 (new) and F3, F7. So do the single-agent arms: S1, S3s, S5 (APG), S6, S7 and LightRAG. Still missing: every multi-agent arm (S8, S9, M1, M1s, M1k, M2, M7), the persona content, the analysis code, the runner, the freeze and the pre-registration. |
| **Study G** | **≈ 30%** | The F8 family exists (new), with the session runner for the full-history baseline (CM0), the oracle-state arm, state probes, session scoring and failure labels. Still missing: the six context-management arms, the best-stack arm, the topology arms in sessions, the analysis (GLMM, TOST, slopes), the runner and the pre-registration. |

**Biggest risks:**
1. **Budget headroom.** The plan fits $5,000 with **$257 contingency (5%)**, against the brief's ≥ 20% rule (`ORCHESTRATOR_BRIEF_v2.md:71`).
   - The total rests on priors no live call has tested yet.
   - Two known items would push it over:
     - the measured F1 record history: +$319, to $5,062;
     - the D-017 Sol builder fallback: +$957.
   - One prior points the other way. F8's reference solver needs ~2.3 generations per item; Study G is priced at 6. Study G is 71% of the plan, so it may be substantially over-priced.
2. **PC1 (the LightRAG anchor) is fragile.** The setup uses LightRAG 1.5.7 instead of the paper's 1.2.5, with rewritten judge prompts. If gpt-4o-mini is retired, Luna judges Luna at ±10 pp. A PC1 failure blocks the freeze, and there is no fix procedure yet.
3. **The multi-agent and context-management harness is the largest piece of unbuilt work.** It is L-sized, and Studies A, B, C, F and G all depend on it.

**Recommendation:** run the probe, the smoke test and the gate now, and build the main-study and Study G harness in parallel. They run after the gate result anyway.

## 2. Built and fixed in this round

| Item | What | Commit |
|---|---|---|
| **F1 breadth aggregation** (2, 8, 32) | Supplier-registry world, the same at every level so levels are paired. Each supplier needs its own lookup (~400-token records) plus its segment's rating rule from the KB. The decomposition for multi-agent arms is stored in task tags, never in the prompt. Primary score: all N exact. Secondary: item F1. | `6890a9a` |
| **F2 dependency chains** (2, 5, 10) | Each hop's escalation code exists only in the previous supplier's record. Primary score: the final supplier exact. Secondary: the correct-prefix length. Every task carries a Study D fault spec, off unless enabled. | `6890a9a` |
| **F8 shift sessions** (any N) | N cases on one F7+F3 world. Bulky tool files (~1.1K tokens). Mid-shift memos that are later withdrawn. A quota of 2 approvals per customer. Follow-ups that name only the earlier case ID. An end-of-shift report. `state_at(k)`, an independent replay solver (agrees on 36/36 sessions), CM0 and O-state runners, forked probes never appended to history, and the failure labels. | `e2740f1` |
| **Fixed: F3 tools crashed on every call** | An unannotated `**kwargs` made Inspect raise on every mutating F3 tool call. Every live F3 gate sample would have errored at its first procedure step; offline mocks never called those tools. New regression test: a mock agent that knows the gold makes exactly the gold calls through the real tool path, and must score 100% on F1, F2, F3, F5 and F7. | `70ed9ea` |
| **Freeze integrity** | The freeze now also hashes `uv.lock` and the analysis code (`analyze_gate.py`, `analysis/*`), so a post-freeze change blocks build-test and test. | `03e50dd` |
| **Stale records** | `.gitignore` covers `/.claude/`. Corrected: the pre-registration's `selected.yaml` path, READINESS D1, EXPERIMENT_AUDIT B5/tuning/messy, HYPOTHESES K1 (no multiplicity-corrected test exists yet), O-1's key limit. | `03e50dd` |

## 3. Gate: what stands between now and a verdict

**Needs you:**
1. **O-1:** a project key with staged hard limits (≈ $300 for probe + smoke + gate, raised per study later).
2. **O-3:** names for the APG owner, skeptic and analyst.
3. **H5:** review `readiness/spotcheck.md`, part A (30–45 min). It has been regenerated on the gate cells.
4. **D-017:** if Luna fails the build-quality check, the Sol fallback (+$957) does not fit without cuts. Build-dev now measures this; see `build-dev/build_quality.json`.
5. **Backup destination:** set `APE_BACKUP_DIR` to a disk or synced folder outside the repo before the first live run.

**Engineering: done (2026-10-02, hardening).** All verified offline; the live checks are noted.

| Item | What changed | Live check |
|---|---|---|
| Probe on the real path | `probe_openai.py` tests effort high vs low through Inspect's own provider (Responses API for GPT-6) and through the build client, per role, with each role's real configuration. Tested against a local fake OpenAI server, so the real SDK and provider code run. | Each role's `paths.inspect.api` is `responses` and its effort verdict passes |
| Model snapshot pins | The probe records each alias's served snapshot. `--pin` writes them to PROVENANCE.md. Live preflight refuses when a pinned alias now resolves to a different snapshot. | The snapshot is a dated ID, not the alias echoed back |
| Program-wide spend guard | A spend registry (`cache/spend/registry.jsonl`) lists every live log dir and ledger. The guard sums all of them per sample, once each, including flushed samples of killed runs; it no longer counts one run id. Failed attempts' logs are kept so their spend is counted. `python -m ape.budget spend` prints it. | `ape.budget spend` after the smoke shows the smoke's spend |
| Crash visibility | `runner_index.json` is written before `eval_set` starts and updated after. | — |
| Per-sample runaway guard | Inspect's `cost_limit`: 20× the cell's projected per-sample cost, at least $0.50 ($2 when unprojected). It is fixed per log dir so resume still pairs. Hits are flagged and count in PC5. | — |
| Error tolerance | `fail_on_error` per task: 2% from 150 samples up, otherwise a count of ≥ 3, so small tasks are not killed by one error. | — |
| Embedding cache | Float32 BLOBs: 6 KB per 1,536-d vector, was 34 KB. Old caches migrate in place, safely across processes. Live calls request base64. | — |
| D-017 measurement | `ape.build_quality`: authoring and LightRAG ID coverage per world and gate cell, with the verdict and recommendation. It is in build-dev's manifest and fills the pre-registration's builder item. | The verdict on the real F7-1000 dev worlds |
| H5 spot-check | Regenerated on the gate cells: part A is 20 F7 + 20 F3 tasks; part B covers F1, F2 and F8. It fixed an F3 text defect ("a credit code code"). A test fails if the sheet goes stale. | Your review |
| Backups | `python -m ape.backup`: incremental, consistent SQLite snapshots, never deletes. Runs after every live phase when `APE_BACKUP_DIR` is set. | — |
| Freeze integrity | The freeze also hashes `uv.lock` and the analysis code. | — |

**PC1 failure procedure (GATE_PREREG §7):**
1. Diagnose the anchor logs. A harness defect is fixed and the anchor re-run, as a logged deviation.
2. If no defect explains the miss, the analyst may freeze with `--accept-pc1-failure "<diagnosis>"`. The diagnosis is recorded in `freeze.json` and PROVENANCE.md, and the report carries the caveat that LightRAG's setup is not validated by the anchor.
3. Otherwise the gate stops.

## 4. Main study

**Ready:**
- the families F1, F2, F3 and F7, with programmatic scoring;
- the single-agent arms S1, S3s, S5 (APG), S6 and S7, plus LightRAG;
- the cost model;
- the shared runner (`ape.runner`).

**Missing** (blocks the main study):

| Component | Effort | Notes |
|---|---|---|
| Multi-agent arms: S9 (plan-then-execute), M1/M1s (orchestrator, parallel/serial), M1k/M2 (specialist workers), M7 (council), the S8 frontier (best-of-k at matched cost) | L | No `as_tool`, handoff or council code exists. F1 subtasks are ready for them. |
| Per-study test seeds and test lock | S | **The main study would reuse the gate's already-analysed F7/F3 test worlds**: seeds are fixed at 3000+ for every study. |
| Study runner, freeze, pre-registration (`PREREGISTRATION.md`) | M | Generalize `run_gate`'s phases by study. |
| Analysis: GLMM (lme4/pymer4/bambi; statsmodels has no crossed logistic GLMM), Holm, TOST, BCa, S8 interpolation per meter, pass^k, Kendall τ; power simulation extensions | L | None exists. |
| Tuning grids for S1, S5 and the multi-agent arms | M | The brief requires equal tuning budgets (`:485`); only APG, LightRAG and S3s are tuned today. |
| Build cells: F7-100 worlds and F1-2 KG builds | S | `main.F.sol` runs S5 and M2 on F7-100, and the micro-pilot runs S5 on F1-2, but neither has builds. |

**Design issues to decide** (§6):
- **H6 (tiers) is confounded.** Sol runs F7-100 while Luna runs F7-1000, so tier is mixed with KB size.
- **H1d (M1 ≈ M1s within 2 pp) is unpowered.** n = 50 gives a 90% CI of ±6–10 pp, and latency is confounded by shared rate limits.
- **K3 (persona) is untestable.** The worlds have no persona content, so S5-P0 is identical to S5.
- **K4:** the relational-vs-independent clause is untestable as planned (only relational cells).
- **F1's context pressure is moderate.** At N = 32 a single agent carries ~13K tokens of records against the main study's W = 128K. The record size is a generator knob, so the micro-pilot should confirm that N = 32 actually stresses a single context before the test split is fixed.

## 5. Study G

**Ready:**
- F8 generation and scoring;
- the CM0 and O-state runners;
- forked state probes (role `probe`, never appended);
- per-item records (position, view tokens, dependency flag) for the degradation curve;
- failure labels.

**Missing** (blocks Study G):

| Component | Effort |
|---|---|
| The CM arms: prune, trim, sum, todo, reset and native (Inspect already ships `compaction()`, `memory()` and `todo_write()`), plus S-CM* | M |
| The topology arms (M1, M2) inside sessions | L |
| The `cm` role (summarizer) | S |
| Session checkpoint and resume. A retried sample restarts the whole session: 240 calls, ≈ $30 on Astra. | M |
| Error tolerance for small tasks. `fail_on_error = 0.02` lets a task of under 50 samples tolerate **zero** errors, and every Study G task is that small. Use a count, not a proportion. | S |
| `cm_stats`: logistic GLMM with session intercepts, TOST, RER and headroom, degradation slopes; a session-clustered power simulation | M–L |
| Probe the API features for Sol, Astra and native compaction | S |
| Pre-registration and runner | M |

**Measured by the F8 build:**
- **Calls per item:** the reference solver needs **2.3 generations per item**, against the budget's prior of 6. A real agent needs more, but this is the strongest evidence that Study G is over-priced. Recalibrate at the micro-pilot.
- **Window pressure appears only at N = 40.** Those sessions cross W = 32K at items 25–33. **The N = 20 topology sessions and Astra's N = 24 sessions may never overflow.** That weakens G-H1 (does the multi-agent advantage shrink with capability?), which needs context pressure.

**Design issues to decide** (§6):
- **Overflow in the N = 20 and N = 24 cells:** raise the tool-output size knob for those cells, or accept no overflow.
- **The reference for G-H1:** today S1's overflow counts as failure for the rest of the session, so the harness sets the S1-vs-multi-agent gap.
- **CM-native runs at one capability point only,** so "native ≈ harness at high tiers" is untestable.
- **S1+KG and S-subiso** are in the hypotheses but not the plan.
- **The capability anchor** (F7-10, F3-5) is near the ceiling, so it measures capability poorly.

## 6. Decisions for you

| # | Decision | My recommendation |
|---|---|---|
| 1 | Key (O-1) | A project key with a ≈ $300 limit now, raised per study. |
| 2 | Names (O-3) | — |
| 3 | Contingency at 5% vs the brief's 20% | Accept for now with staged spending. Recalibrate after the gate pilot and Study G's micro-pilot (F8 suggests Study G is over-priced), and apply cuts C4/C5 then if needed. |
| 4 | D-017 Sol builder (+$957) | Decide only if Luna fails the dev build check. It would need cuts in Study G. |
| 5 | H6 tier confound | Add a Luna F7-100 cell (S1, S5, M1, M2; ≈ $15) plus F7-100 builds, so tier is compared at the same KB size. |
| 6 | Main-study scope | Drop K3/S5-P0 (no persona content) or fund the content. Report H1d and the H2c TOST as descriptive. Reduce H5 to D1, D2 and D6. |
| 7 | Study G overflow and G-H1 | Raise tool-output size for the N = 20 and N = 24 cells so they cross W. Choose the single-agent reference before the pre-registration. |
| 8 | Anchors (τ³-bench, AppWorld, BrowseComp-Plus, CoDA) | Formally defer to Tier B. Only GraphRAG-Bench is wired. |

## 7. Engineering plan (in order)

1. ~~**Gate hardening**~~ done (§3). Next: probe → review → `--pin` → smoke → gate.
2. **Per-study seeds, test lock and runner** (S + M). This prevents reusing the gate's test worlds.
3. **Multi-agent arms** S9, M1/M1s, M1k/M2, M7 and the S8 frontier, with their tuning grids (L). Study G's topology block reuses them.
4. **Study G harness:** the CM arms through Inspect's compaction, todo and memory tools, S-CM*, the `cm` role, session checkpointing, error tolerance by count (M–L).
5. **Analysis and power** for the main study and Study G, then **the two pre-registrations** (L).
6. **Consistency pass on the briefs** once decisions 3–7 are made: contingency, Study C size, Study F cells, session N, epochs and arm IDs disagree across the brief, the context-management audit, the plan and FIX_PLAN. The independent audit listed 17 disagreements. Five were stale records, fixed in `03e50dd`; the rest need the decisions above.

## 8. Live-run facts worth knowing

- **Wall clock:** about 270K agent and kg calls for the gate, so roughly 1–2 days at 16–32 concurrent calls. The whole program takes weeks of calendar time, dominated by Study G on Sol and Astra.
- **Disk:**
  - Logs: ~2 GB.
  - Embedding cache: 5–13 GB until the float32 fix.
  - Indices: 1–2 GB.
- **Crash behaviour:**
  - The gate resumes cleanly: manifests, `eval_set` log reuse, idempotent builds, an atomic ledger.
  - The main study and Study G have no runner yet.
- **Reproducibility gaps:**
  - APG is pinned through a local `file://` git source, which is not portable.
  - `inspect-ai` is pinned only in `uv.lock`, not in `pyproject.toml`.
  - The GraphRAG-Bench data hash is recorded but never checked.
  - Reasoning models are non-deterministic, so epochs are the only control.
