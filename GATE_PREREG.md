# Pre-registration: APG vs LightRAG suitability gate

**Status: DRAFT.** Items marked `[PILOT]` are fixed from pilot data. Items marked `[USER]` need a decision (see `DECISIONS.md`).

**Freeze procedure:**
1. Fill every `[PILOT]` and `[USER]` item.
2. Record the git commit of the analysis code (`src/ape/analysis/gate_stats.py`) and the sha256 of this file in `PROVENANCE.md`.
3. Only then generate the test split (`python -m ape.build --split test ...`).

After the freeze, any change to this file is a logged deviation (see the end of this document).

## 1. Question

Is APG (Adaptive Prompt Graph), with graphs built by the authoring pipeline `ape/apg/author.py`, **non-inferior** to LightRAG as a knowledge-delivery system for a single tool-using agent on the task families APG targets? Those families are policy compliance with a large KB (F7-relational) and tool-load procedures (F3).

## 2. Estimand and test

- **Estimand.** Δ = success(APG*) − success(LGR*), where APG* and LGR* are each system's configuration selected on dev under the same tuning budget (§5). APG* is the better of APG-q and APG-s; LGR* covers mode × per-query/per-step × budgets.
  - Success is the programmatic task outcome (`scorers/success.py`), averaged over epochs per task.
  - Δ is the mean paired task-level difference within each gate cell, then averaged over cells with equal weights.
- **Gate cells:** F7-10, F7-1000 (relational), F3-5, F3-60.
- **Test.**
  - One-sided non-inferiority at α = 0.025, margin **δ = 5 pp** (absolute).
  - Inference uses a world-clustered bootstrap (10,000 resamples, worlds resampled within cells). The lower bound is the 2.5th percentile.
  - Implementation: `ape.analysis.gate_stats.decide` at the frozen commit.
- **Secondary conditions (all required for GO):**
  - F7-1000 cell point Δ > −10 pp.
  - APG* beats the random-node placebo S7: world-level sign-flip test, one-sided p < 0.05.
- **Superiority.** If the lower bound is > 0, superiority is also reported (fixed-sequence; no α penalty).

## 3. Arms

Every arm uses the same agent model, agent loop (`agent/kb_react.py`), system prompt, tools, step query, context placement rule (D-005) and turn cap.

| Arm | Role | Delivery |
|---|---|---|
| APG* | gate | Authored APG graph; the better of per-query (APG-q) and per-step (APG-s) on dev |
| LGR* | gate | The LightRAG configuration selected on dev (§5): LightRAG's own extraction index, mode and budgets fixed at freeze |
| S3s | gate (reference) | Flat hybrid BM25 + dense retrieval, recompiled each step |
| APG-q, LGR-q/s | diagnostic | Per-query variants |
| S5o | diagnostic | Oracle APG graph from the world spec, per step |
| LGRo-* | diagnostic, optional | Oracle LightRAG custom KG |
| S1 | diagnostic | Whole corpus in the system prompt |
| S6 | diagnostic | Gold facts only |
| S7 | diagnostic / placebo | Random chunks, sized per cell to APG*'s realized median context on the pilot (`config/s7_targets.json`), capped at 50% of the corpus |

**Delivery modes** (EXPERIMENT_AUDIT B3; `[USER: O-6]` to confirm).
- **push:** the harness compiles context (per query or per step) and hands it to the agent.
- **pull:** the agent calls `search_kb(query)`, backed by the same arm's retriever, so it can follow references it has read.
- **Why both matter** (measured offline on F7-1000 relational, exception-applies tasks):
  - In push mode, flat retrieval delivered the exception 0/25, with ID-only *and* descriptive rendering.
  - After reading the policy, a pull query ("exceptions to Policy <id>") delivered it 25/25 in both renderings.
  - The delivery mode, not the rendering, decides relational outcomes, and both modes are realistic deployments.
- **F7 cells:** push and pull are **co-primary**. The verdict (§8) is computed and reported per mode. An unqualified GO requires GO in both; GO in one mode only is reported as **GO_PUSH_ONLY** or **GO_PULL_ONLY**, and the user decides.
- **F3 cells:** push only (tool procedures carry no cross-references to follow).
- **Cost:** the F7 half of the gate runs twice, about 1.5× the push-only cost.

**Exception rendering** (F7 relational). The gate cells use `descriptive` exceptions (they restate the policy's domain, region and band). `id_only` exceptions (policy ID only) are run as a paired secondary analysis of explicit cross-reference following; they are never pooled into the verdict. `exception_applies` tasks are reported separately in every table.

## 4. Samples

- **Splits.** `dev` (seeds 1000+) is for tuning, `pilot` (2000+) for calibration, `test` (3000+) for the gate. The test split is generated after the freeze and never inspected before the gate run.
- **Size `[USER: O-5]`.** Recommended: **12 worlds per cell × 12 tasks × 3 epochs** (576 tasks per gate arm). That gives power ≈ 0.79 at true Δ = 0 with σ_g = 0.3, and ≈ 0.88 with σ_g ≤ 0.15.
  - Diagnostic arms: 100 tasks × 3 epochs, drawn from the same worlds.
  - Re-simulate with pilot σ_w and σ_g (`power/power_sim.py --mode ni`) before freezing `[PILOT]`.
- **Models `[USER / E3]`.** OpenAI IDs for the four roles, with reasoning effort fixed per role:
  - agent: `…`
  - kg (APG classify and LightRAG keywords): `…`
  - build (APG authoring and LightRAG extraction): `…`
  - embeddings: `…`

## 5. Tuning (dev split only; equal budget)

- **Budget.** Each system gets **N = `[PILOT]`** configurations, evaluated on the same dev tasks. Every configuration tried is logged in `cache/tuning_log.jsonl`.
- **LightRAG** (tuned by the skeptic): mode ∈ {local, global, hybrid, mix, naive}, per-query vs per-step, `top_k`, `chunk_top_k`, `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens`. Rerank is off (D-004).
- **APG** (tuned by the APG side): per-query vs per-step, `shortlistK` ∈ {12, 24, 48} (`APE_APG_SHORTLIST_K`), `minConfidence` (`APE_APG_MIN_CONFIDENCE`), routable vs non-routable categories, `maxPromptTokens`. The authoring prompt may be revised on dev only.
- **S3s** (tuned by the skeptic): token budget and fusion depth.

## 6. Fairness rules

1. Both KGs consume the identical shared chunk set (`worlds/render.py`).
2. Context size (EXPERIMENT_AUDIT B1). The primary comparison uses each system's dev-selected configuration ("best foot forward"), with realized context tokens (tiktoken `o200k_base`) and cost reported per arm. APG natively delivers ~100-300 tokens where LightRAG and flat retrieval deliver thousands, so forcing parity would change what APG is. A **matched-budget secondary analysis** runs APG*, LGR* and S3s at two budgets, ≈300 and ≈2,000 realized tokens: APG fills up with further shortlisted nodes (`APE_APG_FILL=1`, larger `shortlistK`); LightRAG and S3s are capped (LightRAG's per-part budgets calibrated on dev so realized tokens land within ±25% of the budget).
3. Caching:
   - Query-time LLM caches are off for both KGs.
   - Inspect output caching is off for every arm.
   - Provider prompt caching is left at its default for every arm; cache-neutral tokens are reported alongside.
4. Tool exposure:
   - Primary policy **TE-retrieved**: APG uses its allowlists; the others expose tools named in their delivered context.
   - Sensitivity policy **TE-all**: every tool bound for every arm.
5. Cost accounting:
   - Per-query KG calls (APG classify, LightRAG keywords) run through the "kg" role and are metered by Inspect.
   - Build calls and embeddings are metered in `cache/ledger.jsonl`.
   - Both are always reported.

## 7. Preconditions

A failure here means fix and re-pilot, not NO-GO.

| ID | Condition |
|---|---|
| PC1 | The LightRAG setup reproduces its published GraphRAG-Bench Medical accuracy (arXiv 2506.05690 v3, Table 2: 63.32 / 61.32 / 63.14 / 67.91) for each question type: within ±5 pp if answer, judge and build models are all gpt-4o-mini, ±10 pp otherwise. Uses hybrid mode, `n_per_type` = 200 (all 166 Creative Generation questions), and the official accuracy metric (0.75 × LLM-judged statement F1 + 0.25 × embedding similarity; prompts verbatim). Hybrid must beat naive on the **macro mean** over types; per-type wins are reported but not required, because the paper's own LightRAG loses to vanilla RAG on two types. Implementation: `ape.anchor.graphragbench.pc1_from_logs`. |
| PC2 | On F5, LGR* ≥ LightRAG naive mode. |
| PC3 | Invariants: S6 ≥ S5o ≥ APG* ≥ S7, each allowing 3 pp tolerance; and S6 > S7 by a world-level sign-flip test, p < 0.05. |
| PC4 | Sanity bound: no arm's median realized context exceeds 4× its configured budget, and in the matched-budget analysis every capped arm's median lands within ±25% of the budget. (Replaces the earlier APG/LightRAG parity ratio, which APG's design makes unattainable: EXPERIMENT_AUDIT B1.) |
| PC5 | Harness errors < 2%, and every arm's cap-hit rate < 10%. |
| PC6 | The tuning log is complete for every system. |

## 8. Decision

| Verdict | Condition |
|---|---|
| GO | Lower bound > −5 pp, and all secondary conditions (§2) hold. |
| GO_WITH_COST_FLAG | GO, but APG* per-query cost (including classify and embeddings) is more than 2× LGR*. The user decides. |
| GO_PUSH_ONLY / GO_PULL_ONLY | GO in one delivery mode for the F7 cells but not the other (§3). The user decides. |
| INCONCLUSIVE | The CI spans both −5 pp and 0, or fewer than 4 worlds per cell. Allows **one** pre-registered extension on fresh test worlds, with α split 0.0125 / 0.0125. |
| NO_GO | Anything else. |

**NO-GO diagnosis:**
- If S5o is non-inferior to LGR*, the problem is **extraction**: fix `apg/author.py`.
- Otherwise it is the **runtime or representation**. Each miss is labelled from the logs as one of: shortlist miss, classify miss, compose miss (descendant / slot / truncation), or tool-exposure miss.
- At most **one** fix cycle is allowed: new APG pin and fresh test seeds. A second NO-GO makes LightRAG the KG arm of the main study.

## 9. Reporting (always, whatever the verdict)

- Δ with its CI, and per-cell Δ.
- F5 results, reported separately and never pooled.
- Evidence recall/precision per arm.
- Cost: per query, and per world build.
- Latency: wall-clock and working time.
- Determinism: outcome stability across epochs.
- Every precondition value and every deviation from this document.

## Deviations log

| Date | Change | Reason | Approved by |
|---|---|---|---|
