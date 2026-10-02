# Pre-registration: APG vs LightRAG suitability gate

**Status: DRAFT.** Unfilled items in the body are marked `[PILOT: <label>]` or `[USER: <label>]`.
- `[PILOT: …]` items are fixed from dev tuning and pilot data. `runs/<id>/pilot/pilot.json` gives each value under `prereg_items`, keyed by its label.
- `[USER: …]` items need a decision (see `DECISIONS.md`).
- This header is instructions and is never checked. From §1 on, any remaining marker blocks the freeze.

**Freeze procedure:**
1. Run the gate through the pilot: `python -m ape.run_gate pilot --run-id <id>`.
2. Fill every `[PILOT: …]` and `[USER: …]` item in the body, set the status to FROZEN, and commit.
3. Run `python -m ape.run_gate freeze --run-id <id>`.
   - It refuses while a marker remains in the body, a tracked file has uncommitted changes, or PC1 fails.
   - It also refuses unless build-dev, tune, anchor and pilot are current: a re-run of each would be a skip. Otherwise a re-tune after the pilot would freeze a new APG*/LGR* with S7 targets and caps sized for the old one.
   - It records in `runs/<id>/freeze.json` and `PROVENANCE.md`: the sha256 of this file, `config/selected.yaml`, `config/s7_targets.json`, `config/budget_calibration.yaml`, `config/models.yaml`, `config/model_costs.yaml`, `config/run_plan.yaml` and `config/tuning_grid.yaml`; the analysis code (`src/ape/analyze_gate.py`, `src/ape/analysis/`) and `uv.lock`; the commit; the analysis-code commit; the APG pin; and the run's test-seed block (§4).
   - Commit `PROVENANCE.md` afterwards.
4. Only then is the test split generated (`run_gate build-test`). It refuses unless every frozen file still matches its hash, and unless `src/`, `power/` and `uv.lock` are exactly the freeze commit's (scorers, generators, the agent loop and the adapters included). `test` refuses on the same conditions.

After the freeze, any change to this file, to a frozen file or to the code is a logged deviation (see the end of this document). `tune`, `pilot` and `anchor` refuse to re-run on a frozen run.

## 1. Question

Is APG (Adaptive Prompt Graph), with graphs built by the authoring pipeline `ape/apg/author.py`, **non-inferior** to LightRAG as a knowledge-delivery system for a single tool-using agent on the task families APG targets? Those families are policy compliance with a large KB (F7-relational) and tool-load procedures (F3).

## 2. Estimand and test

- **Estimand.** Δ = success(APG*) − success(LGR*), where APG* and LGR* are each system's configuration selected on dev under the same tuning budget (§5). APG* is the better of APG-q and APG-s; LGR* covers mode × per-query/per-step × budgets.
  - Success is the programmatic task outcome (`scorers/success.py`), averaged over epochs per task.
  - Δ is the mean paired task-level difference within each gate cell, then averaged over cells with equal weights.
- **Gate cells:** F7-10, F7-1000 (relational), F3-5, F3-60.
- **Test** (D-023).
  - One-sided non-inferiority, margin **δ = 5 pp** (absolute).
  - **α.** The whole procedure spends one-sided α = **0.025**, including the one extension (§8):
    - the gate run spends **0.020**;
    - the extension spends **0.005**, on disjoint test worlds.
    - The familywise error is at most their sum (Bonferroni).
  - **Interval: world-clustered t.**
    - Each cell's Δ is the mean paired task difference over its worlds: a ratio estimator, with the linearized cluster variance Σ_w e_w² / (m (m − 1)), where e_w = (y_w − Δ_c n_w) / n̄.
    - The pooled Δ averages the cells, with variance Σ_c var_c / 16.
    - The t quantile uses Satterthwaite's df over cells.
    - The lower bound at one-sided level a is Δ − t(1 − a, df) · SE.
    - Why it replaced the percentile bootstrap (RELIABILITY_REVIEW S2). In 20,000 simulated gates per scenario at true Δ = −5 pp, under power_sim's model with per-cell baselines 0.85 / 0.45 / 0.75 / 0.60:
      - the percentile cluster bootstrap pre-registered earlier passed 3.0–4.0% of the time at a nominal 2.5%, because with ≤ 16 worlds per cell it understates the spread;
      - the t-interval passes 2.6% (16 worlds), 2.3% (12) and 2.4% (F7-1000 with σ_g = 1.0, the other cells 0.2). It passes 2.8% in the extreme σ_w = 1.0, σ_g = 0.6 case, where skewed world effects remain.
    - The full procedure stays within its α in every case (see the extension rule in §8).
    - The bootstrap is still reported, for reference.
  - **Delivery modes (§3).** The push and pull hypotheses are tested with **Holm** at the stage's α:
    - the mode with the smaller p-value is tested at α/2;
    - the other at α, only if the first is rejected.
  - **Minimum data.** Every cell needs at least 4 **paired** APG*/LGR* worlds: worlds where both arms ran a task. Other arms' worlds never count.
  - **Implementation.** `ape.analysis.gate_stats.decide` and `holm_modes` at the frozen commit.
- **Secondary conditions (all required for GO; computed and reported whatever the verdict):**
  - F7-1000 cell point Δ > −10 pp.
  - APG* beats the random-node placebo S7: world-level sign-flip test, one-sided p < 0.05.
- **Superiority.** Tested only after GO in both delivery modes (serial gatekeeping), with Holm across the modes at the stage's α. It is then reported for the modes whose superiority hypothesis is rejected. In 20,000 simulated gates at true Δ = 0, false superiority was 1.8% at α = 0.020.
- **Errored samples.** A sample that still errors after its retries (tolerated up to the PC5 limit) counts as a **failure** in the primary analysis, so a system that errors more pays for it. A sensitivity analysis excludes errored samples.

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
| S7 | diagnostic / placebo | Random chunks, sized per cell to APG*'s realized median context per compile on the pilot (`config/s7_targets.json`: [PILOT: S7 targets per cell]), capped at 50% of the corpus. S7 follows APG*'s delivery schedule (D-024): once per task for a per-query APG*, and a fresh random draw at every step for a per-step APG*, so it matches APG* both in what the model sees at each call and in what a sample delivers in total. |

**Delivery modes** (EXPERIMENT_AUDIT B3; decided D-016).
- **push:** the harness compiles context (per query or per step) and hands it to the agent.
- **pull:** the agent calls `search_kb(query)`, backed by the same arm's retriever, so it can follow references it has read.
- **Why both matter** (measured offline on F7-1000 relational, exception-applies tasks):
  - In push mode, flat retrieval delivered the exception 0/25, with ID-only *and* descriptive rendering.
  - After reading the policy, a pull query ("exceptions to Policy <id>") delivered it 25/25 in both renderings.
  - The delivery mode, not the rendering, decides relational outcomes, and both modes are realistic deployments.
- **F7 cells:** push and pull are **co-primary**.
  - The verdict (§8) is computed and reported per mode.
  - An unqualified GO requires GO in both. GO in one mode only is reported as **GO_PUSH_ONLY** or **GO_PULL_ONLY**, and the user decides.
  - Each mode's verdict pools the same four gate cells with equal weights: the F7 cells in that mode plus the F3 cells, which are push only (push = F7 push + F3; pull = F7 pull + F3). Both are compared against the S7 placebo (push).
  - **Multiplicity** (D-023). The two modes' NI hypotheses are tested with Holm at the stage's familywise α (§2), and each mode's verdict uses its interval at its Holm level.
    - In simulation at the margin (both modes at Δ = −5 pp; 20,000 gates), separate per-mode tests at 0.025 would issue some GO-type label 4.3% of the time.
    - With Holm the rate is 2.2% at 0.025 and **1.8% at 0.020**.
    - With push at the margin and pull clearly non-inferior, the false push claim stays at 2.0% (α = 0.020).
- **F3 cells:** push only (tool procedures carry no cross-references to follow).
- **Cost:** the F7 half of the gate runs twice, about 1.5× the push-only cost.

**Exception rendering** (F7 relational). The gate cells use `descriptive` exceptions (they restate the policy's domain, region and band). `id_only` exceptions (policy ID only) are run as a paired secondary analysis of explicit cross-reference following; they are never pooled into the verdict. `exception_applies` tasks are reported separately in every table.

## 4. Samples

- **Splits.** `dev` (seeds 1000+) is for tuning, `pilot` (2000+) for calibration, `test` (3000+) for the gate. The test split is generated after the freeze and never inspected before the gate run.
  - Each gate run freezes its own block of 100 test seeds: the first run 3000–3015, and the INCONCLUSIVE extension or the NO-GO fix cycle the next unused block (3100+). A run never regenerates another run's test worlds.
  - Offline rehearsals use seeds 9000+, so they never generate the real test worlds.
- **Size** (D-017): **[PILOT: test worlds per cell] worlds per cell × 12 tasks × 3 epochs.**
  - Planned: 16 if the Luna builder passes the dev quality check, otherwise 12.
    - 16 worlds per cell gives 768 tasks per gate arm.
    - Power is P(unqualified GO in both modes, Holm at 0.020) at Δ = 0, under the prior σ and baselines 0.85 / 0.45 / 0.75 / 0.60 (D-023): **0.84** at 16 worlds per cell, rising to 0.885 with the extension, and 0.68 at 12.
    - The earlier 0.91 / 0.79 described one mode's percentile bootstrap at 0.025.
  - If 12, set the `gate.build.test` and `gate.test.*` worlds in `config/run_plan.yaml` before the freeze.
  - Diagnostic arms (push), drawn from the same test worlds: S1, S5o and S6 at 100 tasks per cell × 2 epochs; the S7 placebo (in the GO rule) at 100 × 3.
  - F5 (PC2; reported separately, never pooled): APG*, LGR* and LightRAG naive × F5-1hop, F5-2hop × 100 tasks × 2 epochs.
  - Secondaries (§3, §6; never pooled), each with APG*, LGR* and S3s at 2 epochs:
    - id_only: F7-10 and F7-1000, push and pull, 100 tasks per cell, on paired id_only renderings of 9 test seeds per cell.
    - Matched budget: one budget, ≈300 realized tokens, all 4 cells, 100 tasks per cell.
    - TE-all: F3-5 and F3-60, 100 tasks per cell.
    - Messy (dev diagnostic): F7-10 and F7-1000 on the 2 messy dev worlds, 50 tasks per cell.
  - Cells sized in tasks use whole worlds: the first ⌈n / 12⌉ worlds of the cell by seed. So "100 tasks" is 9 worlds × 12 = 108 tasks, and the messy cells are 1 world of 50 tasks each.
  - These sizes are set by the cost model (FIX_PLAN FX-5, D-021; `config/run_plan.yaml`, `BUDGET.md`).
  - Pilot re-simulation of the NI power: [PILOT: pilot σ_w, σ_g and power].
    - Source: `ape.run_gate pilot` → `runs/<id>/pilot/power.json`, via `power/power_sim.py`.
    - **Estimating σ** (D-023; `ape.analysis.pilot.variance_components`). APG* against LGR* on the pilot, by moment matching each cell's between-world variances to the power model at the cell's own baseline. 80% intervals come from the χ² distribution of those variances.
      - It replaces the delta-method conversion, which ignored task heterogeneity and underestimated σ by about 40%.
      - In 300 simulated 4-world pilots, the new estimator recovered σ_g of 0.30 / 0.80 as 0.28 / 0.77 (the old one: 0.50 for 0.80), and σ_w of 1.0 as 0.99. Interval coverage was 0.73–0.82.
      - **8 worlds per cell** (D-026). The pilot builds 8 worlds per cell. The main pilot cells use worlds 1–4; the σ cell (`gate.pilot.sigma`) runs APG* and LGR* (push, 1 epoch) on worlds 5–8, and σ is estimated from all 8. S7 targets and the matched-budget calibration still come from worlds 1–4.
      - When the worlds vary less than the model predicts even at σ = 0, the interval has no upper end. It is then reported as max(estimate, prior), flagged as not informative, and can never support "suffices" (D-026).
    - **Recommendation** (D-026). Two-sided, at the planned size, from the ends of the 80% intervals:
      - **suffices**: the power at the upper ends reaches 0.8 (both upper ends informative);
      - **insufficient**: even the power at the lower ends misses 0.8. Add test worlds within the budget, or rethink; `power.json` reports 20 and 24 worlds;
      - **ambiguous**: otherwise, or σ not estimable. Proceed with the planned 16 and rely on the pre-registered extension (§8).
      - In 300 simulated pilots per scenario, at a true power of about 0.4 (σ_g = 0.8), 8 worlds called "insufficient" 75–83% of the time and "suffices" never (4 worlds: 55–61% and 2–4%). At a true power of about 0.85, they said "suffices" 11–12%, "insufficient" 11–12% (the 80% intervals' expected misfires near the threshold) and "ambiguous" otherwise.
    - Where σ cannot be estimated, the priors are used, and the recommendation is "ambiguous".
- **Models** (D-015; exact IDs confirmed at E3):
  - agent: GPT-6 Luna, effort high
  - kg (APG classify and LightRAG keywords): GPT-6 Luna, effort low
  - build (APG authoring and LightRAG extraction): [USER: builder] (D-017: GPT-6 Luna, high, if it passes the dev quality check; otherwise GPT-6 Sol, medium, which needs approval)
  - embeddings: text-embedding-3-small

## 5. Tuning (dev split only; equal budget)

- **Budget.** Each system gets at most **N = 8** configurations, declared in advance in `config/tuning_grid.yaml` and evaluated on the same dev cells by `python -m ape.run_gate tune --run-id <id>`. Every configuration tried is logged in `runs/<id>/tune/tuning_log.jsonl`; a re-run archives the previous log beside it (`tuning_log.<UTC time>.jsonl`), and archived logs count toward PC6 and are reported. The selections go to `runs/<id>/tune/selected.yaml` and `config/selected.yaml`; the freeze hashes `config/selected.yaml`. Selection: highest mean dev success; candidates within 1 pp go to the cheaper one.
- **Selections** (`config/selected.yaml`, fixed at freeze): APG* = [PILOT: APG* selection]; LGR* = [PILOT: LGR* selection]; S3s = [PILOT: S3s selection].
- **Roles** (D-018, O-3):
  - APG owner: [USER: APG owner]
  - skeptic (not on the APG side; owns the LightRAG and S3s candidates): [USER: skeptic]
  - analyst (owns this freeze): [USER: analyst]
- **LightRAG** (tuned by the skeptic): mode ∈ {local, global, hybrid, mix, naive}, per-query vs per-step, `top_k`, `chunk_top_k`, `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens`. Rerank is off (D-004).
- **APG** (tuned by the APG side): per-query vs per-step, `shortlistK` ∈ {12, 24, 48} (`APE_APG_SHORTLIST_K`), `minConfidence` (`APE_APG_MIN_CONFIDENCE`), routable vs non-routable categories, `maxPromptTokens`. The authoring prompt may be revised on dev only.
- **S3s** (tuned by the skeptic): token budget and fusion depth.

## 6. Fairness rules

1. Both KGs consume the identical shared chunk set (`worlds/render.py`).
2. Context size (EXPERIMENT_AUDIT B1). The primary comparison uses each system's dev-selected configuration ("best foot forward"), with realized context tokens (tiktoken `o200k_base`) and cost reported per arm. APG natively delivers ~100-300 tokens where LightRAG and flat retrieval deliver thousands, so forcing parity would change what APG is. A **matched-budget secondary analysis** runs APG*, LGR* and S3s at one budget, ≈300 realized tokens (the ≈2,000 budget was dropped for cost, D-021). Its settings come from `ape.run_gate pilot` → `config/budget_calibration.yaml`:
   - APG fills up with further shortlisted nodes (`APE_APG_FILL=1`, larger `shortlistK`): [PILOT: APG* matched-budget settings].
   - LightRAG and S3s are capped. Their caps are calibrated on the pilot split so each arm's median realized context lands within ±25% of the budget.
   - LGR* caps: [PILOT: LGR* matched-budget caps].
   - S3s cap: [PILOT: S3s matched-budget cap].
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
| PC1 | The LightRAG setup reproduces its published GraphRAG-Bench Medical accuracy (arXiv 2506.05690 v3, Table 2: 63.32 / 61.32 / 63.14 / 67.91) for each question type: within ±5 pp if answer, judge and build models are all gpt-4o-mini, ±10 pp otherwise. Uses hybrid mode, `n_per_type` = 200 (all 166 Creative Generation questions), and the official accuracy metric (0.75 × LLM-judged statement F1 + 0.25 × embedding similarity; prompts verbatim). Hybrid must beat naive on the **macro mean** over types; per-type wins are reported but not required, because the paper's own LightRAG loses to vanilla RAG on two types. Implementation: `ape.anchor.graphragbench.pc1_from_logs`. **If PC1 fails:** first diagnose the anchor logs for a harness defect (errors, judge parse failures, index build failures); a defect is fixed and the anchor re-run, as a logged deviation. If no defect explains the miss, the analyst may accept the failure at the freeze (`run_gate freeze --accept-pc1-failure "<diagnosis>"`), which records the diagnosis in `freeze.json` and `PROVENANCE.md`. The verdict is then reported with the caveat that the LightRAG setup is not validated by the anchor, and PC2 stands as the remaining competence check. Otherwise the gate stops. |
| PC2 | On F5, no evidence that LGR* is worse than LightRAG naive mode by more than 3 pp (D-023). PC2 fails when the one-sided 97.5% upper bound of LGR* − naive is below −3 pp, with F5-1hop and F5-2hop equally weighted and the world-clustered t-interval. A true tie fails at most 2.5% of the time (simulated: 0.2%); the earlier point-estimate rule failed 18%. |
| PC3 | Invariants: S6 ≥ S5o ≥ APG* ≥ S7 (D-023). Each adjacent pair fails only on evidence of a shortfall larger than 3 pp: its one-sided upper bound at 0.025 / 3 per pair, world-clustered t on the shared tasks, is below −3 pp. And S6 > S7 by a world-level sign-flip test, p < 0.05. A chain of ties fails at most 2.5% of the time (simulated: 0 of 1,000); the earlier rule failed 16%. |
| PC4 | **Reported, not gated** (D-023). This is the median realized context against 4× each arm's configured budget. The bound holds by construction: LightRAG's `max_total_tokens` is 4× its budget, S7's budget is its own target, and APG and S3s pack under theirs. An arm over it is flagged as an anomaly. Also reported, not gated: in the matched-budget secondary every capped arm's median should land within ±25% of the budget. A miss or a missing secondary marks that secondary "not matched" in the report and never blocks the verdict (D-022). (This replaces the earlier APG/LightRAG parity ratio, which APG's design makes unattainable: EXPERIMENT_AUDIT B1.) |
| PC5 | Harness errors < 2%, and every arm's cap-hit rate < 10%. A cap hit is a sample that ran out of turns without answering, or that a sample limit cut short (including the runner's per-sample cost guard). The S7 placebo is gated on errors only; its cap hits are reported (D-023), since a random context that keeps the agent searching until the cap is the placebo working. The runner's abort threshold (`fail_on_error`: 2% of a task's samples, or a count of at least 3 for tasks under 150 samples) is only a circuit breaker; PC5 is judged from the logs. |
| PC6 | The tuning log is complete for every system. |

## 8. Decision

| Verdict | Condition |
|---|---|
| GO | In both delivery modes: Holm rejects the mode's NI hypothesis (its lower bound at its Holm level is > −5 pp), and all secondary conditions (§2) hold. |
| GO_WITH_COST_FLAG | GO, but APG* per-query cost (including classify and embeddings) is more than 2× LGR*. The user decides. |
| GO_PUSH_ONLY / GO_PULL_ONLY | GO, as above, in one delivery mode for the F7 cells but not the other (§3). The user decides. |
| INCONCLUSIVE | No mode is GO, and in some mode the interval at its Holm level spans both −5 pp and 0, or some cell has fewer than 4 paired APG*/LGR* worlds. This allows **one** pre-registered extension; see the extension rule below the table. |
| NO_GO | Anything else. |

**The extension** (D-023):
- **Data.** Fresh test worlds, a new run disjoint from stage 1's. `analyze_gate` checks the two runs' `build-test/worlds.json`.
- **Test.** The extension's worlds are analysed **alone**, at one-sided α = **0.005**, with Holm across the modes. Its label is the gate's final verdict.
- **Total α.** The gate run's 0.020 plus 0.005 keeps the total one-sided α at 0.025 (Bonferroni over disjoint data).
- **Simulated** (20,000 gates):
  - At the margin, stage 1 issues some GO-type label 1.77% of the time, it is INCONCLUSIVE 23.9% of the time, and the extension adds 0.11%, for **1.88% overall**.
  - At Δ = 0, P(unqualified GO) is 0.837 at stage 1 and 0.885 with the extension.
- **Command.** `python -m ape.analyze_gate --run-id <extension run> --extension-of <stage-1 run>`.

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
