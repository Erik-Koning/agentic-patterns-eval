# Pre-registration: main study (Studies A, B, C, F)

**Status: DRAFT.** Unfilled items in the body are marked `[PILOT: <label>]` or `[USER: <label>]`.
- `[PILOT: …]` items are fixed from the micro-pilot, tuning and pilot. `runs/main/<id>/pilot/pilot.json` gives each value under `prereg_items`, keyed by its label; an offline rehearsal fills its copy from there.
- `[USER: …]` items need a decision (see `DECISIONS.md`).
- This header is instructions and is never checked. From §1 on, any remaining marker blocks the freeze.
- The table in §2.3 is generated. It is `ape.analysis.main_hypotheses.render_hypothesis_table()` pasted verbatim between its marker comments, and `tests/test_prereg_main.py` fails if the two differ. To change a hypothesis, change `main_hypotheses.py` and paste the new output.
- The runner writes every `[PILOT: …]` label into `prereg_items`, including `cap multiple` (the pilot's cap gate, D-039/D-045) and `pilot σ and power` (the in-process power re-simulation at the gate run's pilot σ, §2.6).

**Freeze procedure** (as the gate's, GATE_PREREG.md):
1. Run the study up to the freeze: `uv run --locked python -m ape.run_study all --study main --run-id <id> --gate-run-id <gate run>`. It runs preflight, build-dev, micro-pilot, tune and pilot, then stops at the freeze while a marker remains. (`--locked`: uv never rewrites `uv.lock`, which the freeze hashes.)
2. Fill every `[PILOT: …]` and `[USER: …]` item in the body, set the status to FROZEN, and commit.
3. Run `uv run --locked python -m ape.run_study freeze --study main --run-id <id>`.
   - It refuses if a marker remains in the body, a tracked file has uncommitted changes, `ape.analyze_main` is missing, or a phase the freeze rests on (build-dev, micro-pilot, tune, pilot) is not current. It also refuses while any arm's pilot cap-hit rate is shown to be above 10% (its one-sided 97.5% Clopper–Pearson lower bound, pooled over the arm's pilot cells) at the largest cap multiple (§6; D-039, D-045, D-047).
   - It records the following in `runs/main/<id>/freeze.json` and `PROVENANCE.md`:
     - the sha256 of this file;
     - `config/models.yaml`, `config/model_costs.yaml`, `config/run_plan.yaml` and `config/tuning_grid_main.yaml`;
     - the run's own outputs `config/selected.yaml`, `config/token_caps.json`, `config/token_caps_pilot.json` and `config/s7_targets.json`;
     - `uv.lock`, `src/ape/analyze_main.py` and `src/ape/analysis/`;
     - the gate run's `selected.yaml`, `decision.json` and `freeze.json`;
     - the commit and the analysis-code commit, the APG pin, the `APE_*` design knobs, the KG-arm resolution, the token caps, the run's role and its test-seed block.
4. Commit `PROVENANCE.md`.
5. Only then is the test split generated (`run_study build-test`). It refuses unless every frozen file still matches its hash, `src/`, `power/` and `uv.lock` are exactly the freeze commit's, and the design knobs and the KG resolution are the frozen ones. `test` refuses on the same conditions.

After the freeze, any change to this file, to a frozen file or to the code is a logged deviation (§10). `micro-pilot`, `tune` and `pilot` refuse to re-run on a frozen run.

## 1. Questions

The Minimum Viable Study (brief §7.2) runs Studies A, B and C on the primary tier, GPT-6 Luna (high effort). Study F replicates part of it on GPT-6 Sol (high). Results generalise to these two tiers only; Study G owns Astra.

| Question | Study | Hypotheses |
|---|---|---|
| **RQ1, mechanism attribution.** Which mechanisms account for multi-agent gains over a single ReAct agent, compared at matched *realised* cost against a compute-matched single-agent frontier (S8)? The MVS mechanisms are decomposition, context isolation, concurrency, ensembling, peer communication and role specialization. | A | M1, M2, M3 (confirmatory); M4 and the §4.3 mechanism contrasts (descriptive) |
| **RQ2, KG-compiled context.** Does per-step KG-compiled context in a single agent recover the benefit of multi-agent specialization? Does graph structure beat equally engineered flat retrieval when the KG is built by the real pipeline? | B | K1, K2, K2-NI, M5 (confirmatory); K4 (descriptive) |
| **RQ3, meter sensitivity.** Do arm rankings change across tokens, cache-adjusted $ and wall-clock? | A, B | C1 (descriptive) |
| **RQ4, determinism.** At matched accuracy, do KG single-agent arms and multi-agent arms differ on empirical determinism? | C | A1 (descriptive) |
| **Tiers** (the brief's RQ5 tier clause). Does the coordination payoff shrink from Luna to Sol? | F | T1 (descriptive) |

Not addressed here:
- auditability (the auditability metrics are not produced);
- RQ5's routing question (task features predicting the best arm);
- fault propagation (Study D) and latency regimes (Study E);
- the Tier B arms.

## 2. Hypotheses, estimands and tests

### 2.1 Outcome, unit and estimand

- **Outcome.** Task success: binary and programmatic (`scorers/success.py`), per sample and epoch.
  - A sample that still errors after its retries (`retry_on_error` = 2) counts as a **failure**.
  - A **cap hit** also counts as a failure (§6), and casts no vote in the S8 frontier (§3.3), whatever the scorer read. Inspect scores a sample after a limit with its last state, so a capped F3 sample whose end state happened to equal the gold, or an answer recorded in the turn a limit fired, would otherwise count as a success; `main_load` sets its success to 0 and its answer key to none, and keeps the scorer's value for the record (D-047).
  - No LLM judge is used.
- **Task value.** An arm's success on a task is the mean over its epochs. Every arm has 3 epochs. S1's 3 are the first 3 runs of its 8-run pool (§4.3).
- **Contrast.** Each member of a hypothesis is a per-task contrast of task values, Σ coef × term (the table's "Contrast" column). It takes one of three forms:
  - a paired difference (M1 − S9);
  - a difference in differences on the same tasks (K2's 2×2, T1's arm × tier);
  - an arm against the S8 frontier at that arm's realised mean cost in the same cell and tier (`S8@M1`, §3.3).
- **Estimand.** The member's mean contrast:
  - within a cell, the mean over the tasks where every term ran (paired tasks);
  - over a member's cells, the equally weighted mean of the cells. A pooled claim is therefore a claim about the **equal-weight average over its cells**, not about every cell: a cell near the ceiling, where no arm can differ much, dilutes it. Each cell's estimate is reported beside every pooled member (§9).
- **Clusters.** Tasks within a world share its knowledge base (and its KG build), so worlds are the independent units.
  - F1 and F2 worlds with the same seed share one supplier registry, so they form **one cluster across Study A's cells**. A member pooled over Study A has 9 clusters, not 36.
  - F3 and F7 worlds are their own clusters. A member pooled over Study B has 36.

### 2.2 Tests

- **Decision test** (D-029, D-031): a **world-clustered sign-flip test** on the per-task contrast.
  - It is exact over all 2^G sign flips up to G = 20 clusters, and uses 10,000 Monte Carlo flips above that.
  - The non-inferiority and equivalence tests shift each cluster by the margin, weighted by its share of the estimate.
  - The smallest attainable one-sided p is 2^−G: 0.00195 at 9 worlds, below every Holm level in this design (the smallest is 0.0125).
  - **Its condition is symmetric world contributions.** For superiority with exchangeable arms the test is exact. For the shifted nulls (NI, TOST), the differences in differences and the frontier contrasts it assumes that each world's contribution is symmetric about the null. Worlds that are catastrophic for one arm only (a broken KG build, say) or arms near the ceiling skew the world sums, and the level drifts above nominal. Sensitivities, from the independent review (10,000 simulated studies each) and `main_power`:
    - one arm catastrophic in 15% of worlds (−3 logit): K1 0.075 and K2 0.036 (nominal 0.025), M5 0.075–0.078 (nominal 0.05), K2-NI 0.045 (0.025); M1 0.049–0.053 (0.025) with 10–15% of worlds at −3 to −4 logit;
    - milder failures (10–20% of worlds at −1.5 to −2 logit): 0.023–0.032 against 0.025;
    - ceiling cells at a TOST boundary: M3 at +6 pp, where M7 would succeed about 98% of the time in F1-2 and F2-2, 0.052–0.059 against 0.05 (0.047 with those two cells off the ceiling);
    - heavy-tailed (t3) world effects, correlated epochs, unequal clusters and arm-specific world spreads keep the level at or below nominal (0.017–0.022 against 0.025).
  - So every member reports its **per-world influence**: each world's contribution to the estimate, and the estimate and test without it. A supported claim that does not survive dropping one world, or that rests on a world whose KG build failed its quality check (`build-test/build_quality.json`), carries a §8 caveat.
- **Frontier contrasts carry the matched cost's noise** (M2.frontier, M3, and the descriptive M2-F2 and T1). S8 is interpolated at the arm's realised mean cost over S1's mean run cost, both measured on the same worlds, so the interpolation weight is an estimate. Each task's delta-method influence on it, the frontier's slope times (the arm's task cost − r̂ × S1's task run cost) / S1's mean run cost, is added to the task's contrast value, centred within the cell so the estimate is unchanged, and the test, the intervals and the bootstrap all carry it. Without it, when failed runs cost more than successful ones (κ = 1–3, plausible with caps at 8 × B0), M3's TOST rejected at 0.073–0.10 against 0.05 and M2.frontier at 0.027–0.036 against 0.025 (§2.6).
- **Test types**:
  - **superiority**: H0 Δ ≤ 0 against Δ > 0;
  - **less**: H0 Δ ≥ 0 against Δ < 0 (K2's interaction);
  - **non-inferiority**: H0 Δ ≤ −margin;
  - **equivalence (TOST)**: two one-sided tests at ±margin, with p the larger of the two.
- **α**:
  - one-sided **0.025** for the superiority, non-inferiority and `less` families;
  - **0.05** for each one-sided test of a TOST, so equivalence is shown when the 90% interval lies inside ±margin.
- **Multiplicity:**
  - **Holm within each family**, using the gate's stopping rule: members after the first non-rejection are "not tested".
  - **Serial gatekeeping** for M2: stage 2 is tested, at the full α, only after every stage-1 member is rejected.
  - **No correction across families** (brief §7.4). There are 7 confirmatory families.
- **Interval** (the headline). The sign-flip test inverted, two-sided 1 − 2α. It agrees with the decision by construction: a TOST rejects exactly when this interval lies inside ±margin.
  - Also reported, with no decision resting on them:
    - the world-clustered **BCa bootstrap** (10,000 resamples; the brief's interval; it covered 88% at 9 worlds in simulation);
    - the gate's **world-clustered t**.
- **Missing data.**
  - A cell where a term has no data, or where an arm is beyond the S8 frontier (§3.3), is left out of a pooled member and named in the report.
  - A member with no data in any of its cells is "not evaluable".
  - A contrast with fewer than 4 clusters is flagged `few_clusters`. The test stays valid, just weak.
- **Implementation.** `ape.analysis.main_stats.evaluate_family`, run by `ape.analyze_main.analyze`, which is the study runner's analyze phase. Both are used at the frozen commit.

### 2.3 Hypothesis table

The confirmatory families, their members, contrasts, cells, tests, margins, α, procedure and planned power, and the descriptive estimands. The table is generated from `ape.analysis.main_hypotheses`, the code the analysis runs. Its "Planned MDE / power" column is from `ape.analysis.main_power` at the planned sizes and the gate's σ priors (§2.6).

<!-- BEGIN GENERATED: main_hypotheses.render_hypothesis_table() -->
| ID | Member | Source | Contrast | Cells | Test | Margin | Holm family (α, procedure) | Planned MDE / power | Status |
|---|---|---|---|---|---|---|---|---|---|
| M1 | M1.F1-32 | brief H1a | M1 − S9 | F1-32 | one-sided superiority (H0 Δ ≤ 0) | – | M1 (0.025, holm) | MDE ≈ 15 pp; type I 0.015; power 0.50 at +10 pp, 0.86 at +15 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| M2 | M2.gate | brief H1b | M1 − S1 | F1-32 | one-sided superiority (H0 Δ ≤ 0) | – | M2 (0.025, serial, stage 1) | MDE ≈ 15 pp; type I 0.028 (M1 = S1); the same single-cell design as M1.F1-32 (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| M2 | M2.frontier | brief H1b | M1 − S8@M1 | F1-32 | one-sided superiority (H0 Δ ≤ 0) | – | M2 (0.025, serial, stage 2) | MDE ≈ 15 pp; false claims at the frontier boundary 0.019 / 0.029 / 0.023 with failed runs costing κ = 0 / 1 / 3 times more (κ 3, 3,000 studies: 0.027); power 0.56 at +10 pp, 0.92 at +15 pp (κ 3: 0.47 at +10 pp), after the gate (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| M3 | M3.pooled | brief H1c; D-033 | M7 − S8@M7 | F1-2, F1-32, F2-2, F2-10 | equivalence, TOST (H0 Δ ≤ −margin or Δ ≥ margin) | ±6 pp | M3 (0.05, holm) | type I at nominal except up to 0.059 at the +6 pp boundary (κ 0) and 0.055 at −6 pp (κ 1), from the near-ceiling cells F1-2 and F2-2 (0.047 off the ceiling); power at Δ = 0: 0.89 / 0.79 / 0.69 at κ = 0 / 1 / 3 (4,000 studies per boundary, 1,000 for power, 108 tasks per cell, gate σ priors; main_power.m3_pool_check, seeds 20261040 / 20261050 / 20261060) | confirmatory |
| M5 | M5.pooled | brief H1e; D-033 | M2 − M1k | F3-5, F3-60, F7-10, F7-1000 | equivalence, TOST (H0 Δ ≤ −margin or Δ ≥ margin) | ±6 pp | M5 (0.05, holm) | type I 0.052 / 0.052 at +6 / −6 pp (4,000 studies); power 0.88 at Δ = 0, 0.72 at +2 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| K1 | K1.F7-1000 | brief H2-struct | S5 − S3s | F7-1000 | one-sided superiority (H0 Δ ≤ 0) | – | K1 (0.025, holm) | MDE ≈ 15 pp; family type I 0.010; power 0.41 at +10 pp, 0.83 at +15 pp (Holm of 2) (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| K1 | K1.F3-60 | brief H2-struct | S5 − S3s | F3-60 | one-sided superiority (H0 Δ ≤ 0) | – | K1 (0.025, holm) | MDE ≈ 15 pp; family type I 0.010; power 0.49 at +10 pp, 0.90 at +15 pp (Holm of 2) (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| K2 | K2.interaction | brief H2; D-033 | M1k − S5 − M1 + S1 | F3-5, F3-60, F7-10, F7-1000 | one-sided (H0 Δ ≥ 0, H1 Δ < 0) | – | K2 (0.025, holm) | MDE ≈ 8 pp; type I 0.023; power 0.47 at −5 pp, 0.87 at −8 pp, 0.96 at −10 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| K2-NI | K2-NI.pooled | brief H2 (operational clause); D-033 | S5 − M2 | F3-5, F3-60, F7-10, F7-1000 | one-sided non-inferiority (H0 Δ ≤ −margin); cost S5/M2 ≤ 0.6 at 97.5% confidence, point estimate ≤ 0.5 | 5 pp | K2-NI (0.025, holm) | type I 0.022 at −5 pp; power 0.79 at Δ = 0, 0.38 at −2 pp, 0.98 at +3 pp, with the cost condition (1,000 simulated studies at 108 tasks per cell, gate σ priors) | confirmatory |
| T1 | T1.F1-32 | brief H6 (tier clause); D-028 #2; D-033 | M1[sol] − S8[pool 3]@M1[sol] − M1 + S8[pool 3]@M1 | F1-32 | estimate and intervals | – | – | – | descriptive |
| T1 | T1.F7-100 | brief H6 (tier clause); D-028 #2; D-033 | M1[sol] − S8[pool 3]@M1[sol] − M1 + S8[pool 3]@M1 | F7-100 | estimate and intervals | – | – | – | descriptive |
| T1 | T1.pooled | brief H6 (tier clause); D-028 #2; D-033 | M1[sol] − S8[pool 3]@M1[sol] − M1 + S8[pool 3]@M1 | F1-32, F7-100 | estimate and intervals | – | – | – | descriptive |
| M1-F2 | M1-F2.F2-10 | brief H1a (F2 clause) | M1 − S9 | F2-10 | estimate and intervals | – | – | – | descriptive |
| M2-F2 | M2-F2.F2-2 | brief H1b (F2 clause) | M1 − S8@M1 | F2-2 | estimate and intervals | – | – | – | descriptive |
| M2-F2 | M2-F2.F2-10 | brief H1b (F2 clause) | M1 − S8@M1 | F2-10 | estimate and intervals | – | – | – | descriptive |
| M4 | M4.F1-32 | brief H1d; D-028 #3 | M1 − M1s | F1-32 | estimate and intervals | – | – | – | descriptive |
| K4 | K4 | brief H2c; D-028 #3 | KB scaling: success slopes over log10 KB size (F7-10/100/1000) for S1, S5, S3s, M1, M2 and the paired slope differences S5 − S1 and S5 − S3s, with intervals. | – | k4 | – | – | – | descriptive |
| C1 | C1 | brief H3 | Cost-meter rank flips: Kendall τ between arm rankings (cost per solved task) under tokens, cache-adjusted $ and wall-clock is < 0.8 in ≥ 2 of the 4 MVS families. | – | c1 | – | – | – | descriptive |
| A1 | A1 | brief H5 | Determinism: pass^k, outcome stability (D1), answer agreement (D2) and resource predictability (D6) per arm and cell on the Study C subset (5 epochs), and KG single-agent vs multi-agent contrasts at matched accuracy. | – | a1 | – | – | – | descriptive |
<!-- END GENERATED: main_hypotheses.render_hypothesis_table() -->

How to read the table:
- `S8@X` is the S8 frontier interpolated at arm X's realised mean cost on tokens.
- `[pool 3]` means the frontier is built from the first 3 S1 runs only.
- `[sol]` is the Sol tier; every other term is Luna.
- `k4`, `c1` and `a1` name the descriptive estimators in `ape.analysis.main_descriptive`.

### 2.4 Confirmatory families

Each family gets one of four labels. "Not supported" never means "no effect": it means the claim was not shown (§9).

| Label | Meaning |
|---|---|
| **supported** | The member was rejected at its Holm level, and its cost condition (if any) holds. |
| **not supported** | The member was tested and not rejected. |
| **not tested** | Holm stopped before the member, or its gate stage failed. |
| **not evaluable** | The member's data are missing. |

**M1, context isolation (brief H1a).**
- **Test.** M1.F1-32: M1 − S9 > 0 on F1-32 (the breadth task at N = 32), one-sided at 0.025.
- **Claim if supported:** on F1-32, running subtasks in isolated worker contexts beats planning and executing them in one context.
- **M1 stands in for M1s**, as the MVS allows.
  - The stand-in rests on construction: M1 and M1s give the model identical inputs (D-034, tested). It does not rest on M4's estimate, which is descriptive (D-028).
  - M1s − S9 on M1s's 60 tasks is reported alongside.
  - If M4 shows evidence that concurrency changed accuracy, that is an invariant violation (§8, I1).
- The F2 clause ("≤ 0 on F2-high") has no margin in the brief, so it is descriptive (M1-F2).

**M2, ensembling against coordination (brief H1b).** Serial gatekeeping (brief §7.4: a multi-agent arm meets the S8 frontier only after beating S1).
- **Stage 1.** M2.gate: M1 − S1 > 0 on F1-32.
- **Stage 2.** M2.frontier: M1 − S8@M1 > 0 on F1-32. It is tested at the full α only if stage 1 is rejected; otherwise it is "not tested".
  - If M1 costs more than S8(8), stage 2 is "not evaluable" (beyond the frontier).
- **Claim if both are supported:** on F1-32, coordination beats compute-matched self-consistency at M1's realised cost.
- **The brief's other clauses:**
  - "Multi-agent keeps its advantage on F1-high" (brief H2) is stage 1's claim.
  - The F4 clause is out of scope.
  - The F2 clause ("S8 at M1's cost ≥ M1") is descriptive (M2-F2).

**M3, peer communication (brief H1c).**
- **Test.** M3.pooled: equivalence of M7 − S8@M7 within **±6 pp** (D-033; the brief's ±3 pp had power 0.13), averaged over Study A's four cells with equal weights, by TOST at 0.05 per side.
- **Claim if supported:** at matched realised cost, the council (3 members, 2 critique rounds, a chair) is within 6 pp of compute-matched self-consistency in either direction, **averaged over the four cells with equal weights**. Two of the four (F1-2 and F2-2) are near the ceiling, where neither arm can differ much, so the claim is read with the per-cell estimates reported beside it.
- **Size** (D-048). The test stays confirmatory on the four cells, with its residual size stated: type I is at nominal except up to **0.059 at the +6 pp boundary (κ = 0)** and **0.055 at −6 pp (κ = 1)**, against 0.05. The two near-ceiling cells cause it (0.047 with them off the ceiling); it is not the matched cost's noise, which the correction removes (§2.2). Testing F1-32 and F2-10 alone was rejected: type I 0.030–0.041, but power 0.12–0.20 at Δ = 0.
- **Planned power at Δ = 0:** 0.89 / 0.79 / 0.69 at κ = 0 / 1 / 3 (§2.6).
- **Not gated on M7 > S1.** Equivalence to the frontier is informative either way.
- Cells where M7 costs more than S8(8) are left out and named. If every cell is, the member is "not evaluable".

**M5, role specialization (brief H1e).**
- **Test.** M5.pooled: equivalence of M2 − M1k within **±6 pp** (D-033; ±3 pp had power 0.05), averaged over Study B's four cells with equal weights, by TOST at 0.05 per side.
- **Claim if supported:** giving the KG workers specialist roles changes success by less than 6 pp, **averaged over the four cells with equal weights**; each cell's estimate is reported beside it.

**K1, graph structure (brief H2-struct).**
- **Test.** Two members, Holm at 0.025: K1.F7-1000 and K1.F3-60, each S5 − S3s > 0.
  - S5 is the **extracted** KG (the gate-selected arm, §3.2).
  - S3s is flat hybrid retrieval per step, engineered by the skeptic.
- **Claim if supported** (per member): the graph beats equally engineered flat retrieval on that cell.
- The extraction-error gap (S5o − S5) is the gate's diagnostic. It is not re-run here.

**K2, substitution (brief H2).**
- **Test.** K2.interaction: (M1k − S5) − (M1 − S1) < 0, averaged over Study B's four cells with equal weights, one-sided at 0.025. It is its own family since D-033.
- **Claim if supported:** the multi-agent advantage shrinks under KG delivery, averaged over the four cells with equal weights.

**K2-NI, operational non-inferiority (brief H2, operational clause).**
- **Test.** K2-NI.pooled: S5 − M2 > −5 pp, averaged over Study B's four cells with equal weights, one-sided at 0.025. It is its own family since D-033 (the brief's 3 pp per family had power 0.12–0.16).
- **Cost condition.** The claim also requires S5/M2's realised token cost ratio on the paired tasks to be **≤ 0.6 with 97.5% confidence** (its one-sided upper bound: the upper end of the two-sided 95% world-clustered BCa interval) and its **point estimate ≤ 0.5**.
  - Realised cost is the logged final attempts' usage. An errored attempt that was retried is in no log (D-030), so its usage is not in the ratio; the report gives each arm's share of such usage from the usage ledgers.
  - This is an intersection-union condition, so it needs no extra α.
  - A rejection with a failed cost condition is "not supported", with the reason given.
- **Claim if supported:** a single agent with the KG is no more than 5 pp worse than the specialist team, averaged over the four cells with equal weights (each cell's estimate reported beside it), at a cost ratio below 0.6 with 97.5% confidence and about half or less at its point estimate.

### 2.5 Descriptive analyses

These are reported with intervals and labelled "descriptive" wherever they appear. No claim rests on them, and they carry no test.

| ID | What is reported | Notes |
|---|---|---|
| T1 (brief H6, tier clause) | The coordination payoff M1 − S8@M1, Sol minus Luna on the same tasks. Reported per cell (F1-32, F7-100) and pooled, with the sign-flip, BCa and cluster-t intervals. Also every arm contrast in each tier and its Sol − Luna change (`tier_contrasts`). | The brief predicts a decrease. Frontiers come from the first 3 S1 runs in both tiers, so both are built alike. Planned power 0.44–0.54 at a 15 pp drop (D-033). Only the agent model differs between tiers. |
| M1-F2, M2-F2 | M1 − S9 on F2-10; M1 − S8@M1 on F2-2 and F2-10. | The brief's F2 clauses have no margin. |
| M4 (brief H1d) | M1 − M1s accuracy on F1-32 (M1s's 60 tasks), with 95% and 90% intervals; the wall-clock ratio M1/M1s with its interval. | The brief's thresholds are \|Δ\| < 2 pp and a ratio < 0.6. At n = 60 the interval cannot reach ±2 pp (D-028). |
| K4 (brief H2c) | Success slope per decade of KB size over F7-10, F7-100 and F7-1000 for S1, S5, S3s, M1 and M2. Also the paired slope differences S5 − S1 and S5 − S3s, with intervals. | S3s has no F7-100 cell, so its slope rests on two levels. The relational-vs-independent clause is not tested: there are no independent-policy cells (D-028). |
| C1 (brief H3) | Per family (F1, F2, F3, F7; Study A and B cells), each arm's cost per solved task under tokens, cache-adjusted $ and wall-clock (Inspect working time); Kendall τ-b between the three rankings; the rule "the smallest pairwise τ < 0.8 in ≥ 2 of the 4 families", with world-clustered bootstrap support. LLM calls are reported, not ranked. | The brief asks for each predicted flip's direction to be pre-registered. None is, so the rule is reported on point estimates and is not a test. |
| A1 (brief H5) | On the Study C subset (§4.4), per arm and cell: unbiased pass^k and pass@k, D1 outcome stability, D2 modal-answer share and normalised answer entropy ("no answer" counts as an answer), and D6 (the CV of tokens, cache-adjusted $ and wall-clock: median and p90 over tasks). Also S5 − M1 and S5 − M2 on D1, D2 (modal share) and success, with "matched accuracy" meaning the 95% cluster-t interval of the success difference lies within ±3 pp. | The metrics use each arm's first 5 runs: its 3 main epochs, then Study C's 2; S8k3's 5 runs all come from Study C (D-044). The auditability metrics (A2, A4, A6, A8) and D5 are not produced, so H5's auditability clause is not addressed. |
| §4.3 mechanism contrasts | Each contrast per cell, where both arms ran, with intervals (table below). | Labels follow the brief: a mechanism claim must cite a single-switch row; anything else is a **bundle** contrast. |
| S8 frontier | Each condition's S8(k) curve, k = 1..8, on every meter. Also every arm against S8 at its own realised cost on every meter, with the match status (inside, below, beyond). | Tokens is the confirmatory meter. The other meters are descriptive. |
| Live S8k3 against post-hoc S8(3) | Live minus post hoc on the Study C tasks, per cell. | Their tie rules differ (§3.3). This stands in for the brief's M7-R0 invariant, which is not run. |
| Invariants | §8. | |
| Arms and harness | Per arm, cell and tier: success, every cost meter, the cap-hit and harness-error rates with Clopper–Pearson bounds, and the agents' stop reasons. | |
| GLMM | Per family, `success ~ arm * log_knob + (1 \| task) + (1 \| task:arm)`, fitted with statsmodels' variational-Bayes mixed GLM. | Descriptive only (D-029). A failed fit is reported, never fatal. |

Mechanism contrasts (`main_hypotheses.MECHANISM_CONTRASTS`; brief §4.3):

| Mechanism | Contrast | Cells | Single switch? |
|---|---|---|---|
| Decomposition | S9 − S1 | Study A | yes (DEC) |
| Isolation | M1 − S9 | Study A | M1 stands in for M1s (ISO; CONC changes timing only) |
| Isolation (M1s) | M1s − S9 | F1-32 | yes (ISO) |
| Concurrency | M1 − M1s | F1-32 | yes (CONC) |
| Multi-agent vs single (monolith) | M1 − S1 | Studies A and B | bundle (DEC, ISO, CONC) |
| Council vs single | M7 − S1 | Study A | bundle (ENS, COMM) |
| KG delivery | S5 − S1 | Study B, F7-100 | yes (DEL) |
| Graph structure | S5 − S3s | Study B | yes (DEL) |
| Flat retrieval | S3s − S1 | Study B | yes (DEL) |
| KG vs placebo | S5 − S7 | Study B | yes (DEL) |
| KG workers | M1k − M1 | Study B | yes (DEL, arm-wide) |
| Specialization | M2 − M1k | Study B | yes (SPEC) |
| Single KG vs specialists | S5 − M2 | Study B, F7-100 | bundle |

Not in this study: ensembling as S8(k) − S1 (it is the frontier table), extraction error (S5o − S5, the gate's), and the Tier B contrasts (M3 − M2, M4 − M2, M5 − M2, M6 − M5, M7 − M7-R0, M7 − M9a/b, M8 − M7).

### 2.6 Power

- **Planned power.** The table's planned power is from `ape.analysis.main_power`: 1,000 simulated studies per scenario (more at the null boundaries where stated) through the real analysis path, at the sizes the runner runs (§4.2: 9 worlds × 12 = 108 tasks per cell, 3 epochs, the 8-run S1 pool).
  - σ: the gate's priors (σ_w 0.5, σ_g 0.3, σ_u 1.5, σ_v 0.5).
  - Baselines: assumed S1 success per cell (F1-2 0.80, F1-32 0.40, F2-2 0.80, F2-10 0.50, F3-5 0.75, F3-60 0.60, F7-10 0.85, F7-100 0.65, F7-1000 0.45; Sol +0.5 on the logit scale).
  - Cost multiples: the budget priors (`config/budget_assumptions.yaml`: M1 2.5, M7 5.0, S9 1.3).
  - **Cost that depends on success.** Each run costs (1 + κ (1 − success)) times its base, so with κ > 0 a failed run costs more (with caps at 8 × B0, a failing run that wanders to its cap is the plausible case). M2.frontier and M3 are simulated at κ = 0, 1 and 3. Their frontier nulls are fixed points: the arm's success equals S8 at its own expected cost ratio, which then moves with its success. The population frontier is computed exactly (the vote's multinomial integrated over task difficulty), so a null boundary carries no Monte Carlo error of its own.
  - **Type I error** is at or below nominal within Monte Carlo error at every null boundary, with one exception, M3 (D-048): up to **0.059 at the +6 pp boundary (κ = 0)** and **0.055 at −6 pp (κ = 1)** against 0.05; 0.052 / 0.051 at +6 pp and 0.052 / 0.048 at −6 pp for κ = 0 / 3 otherwise. The excess comes from Study A's ceiling cells: at the +6 pp boundary M7 would succeed about 98% of the time in F1-2 and F2-2, which skews the world sums (§2.2's condition); with those two cells off the ceiling it is 0.047.
  - **M3's planned power** (four cells, ±6 pp): 0.89 / 0.79 / 0.69 at Δ = 0, 0.67 / 0.61 / 0.52 at +2 pp and 0.70 / 0.58 / 0.51 at −2 pp, at κ = 0 / 1 / 3. Reproduce with `ape.analysis.main_power.m3_pool_check(STUDY_A_CELLS, kappa, seed)` at seeds 20261040 / 20261050 / 20261060 for κ = 0 / 1 / 3 (4,000 studies per boundary, 1,000 for power; scenario i is seeded seed + i); the rejected two-cell design is `m3_pool_check(("F1-32", "F2-10"), kappa, seed)` at 20261010 / 20261020 / 20261030. Without the matched-cost correction (§2.2) it was 0.095 at κ = 3. M2.frontier: 0.019 / 0.029 / 0.023 at κ = 0 / 1 / 3 (0.036 at κ = 3 without the correction).
  - Power is below 0.8 at plausible effects for several members. D-033 records the owner's decisions on that: the TOST margins widened to ±6 pp, K2-NI separated out, T1 made descriptive, and the superiority members' planned MDE kept at about 15 pp. Power falls with κ for the frontier members (M3 at Δ = 0: 0.87, 0.78, 0.67 at κ = 0, 1, 3), because the correction makes them carry the matched cost's noise.
  - The sign flip's sensitivity to asymmetric world contributions is in §2.2.
- **Pilot re-simulation.** Before the freeze the power is re-simulated with σ_w and σ_g from the gate run's pilot (`runs/<gate run>/pilot/power.json`, `variance_components`: 8 worlds per F3/F7 cell, D-026). The main pilot's 2 worlds per cell cannot estimate them. The command is `uv run --locked python -m ape.analysis.main_power --reps 1000 --pilot <those variance components as JSON> --out runs/main/<id>/pilot/power.json`.
  - Result: [PILOT: pilot σ and power].
- **The re-simulation is reported, not a decision.** The sizes are fixed by the budget (D-035). A member keeps its test whatever its re-simulated power, and the report gives its achieved MDE.

## 3. Arms

### 3.1 What every arm shares

- **Models.** The same agent model in a tier: GPT-6 Luna, or GPT-6 Sol in Study F's Sol cells, both at high effort. The same `kg` role: GPT-6 Luna, low effort.
- **Loop and prompts.** The same per-turn agent rules: `kb_react`'s rules, applied to every agent of every arm by `multi.core.react_loop`, which gives single-agent arms byte-identical model inputs. Also the same base system prompt.
- **Tools.** The same tools and tool-exposure rule (the gate's TE-retrieved).
- **Delivery.** Push delivery (D-005 placement).
- **Turn cap per agent loop.** F1-N: N + 12 (44 at F1-32); F2-k: 2k + 8; F3 and F7: 12.
- **Token cap.** The same per task cell for every arm (§6).
- **Agent rules not used.** Inspect's `react()`, `as_tool()` and `handoff()` are not used, because each changes information flow (D-034).

### 3.2 Registry

Switches are the brief's §4.1 vector, given as differences from S1 (DEL=MONO, DEC=0, ISO=0, CONC=0, ENS=1, SPEC=0, STATE=none, CTRL=dynamic, COMM=none, HET=0, CMP=0). Multi-agent and ensemble arms log their vector with every sample (`mas_switches`). Single-agent arms are identified by their arm name.

| Arm | Switches | Protocol | Knowledge delivery (every agent of the arm) |
|---|---|---|---|
| S1 | — | One ReAct agent. | Monolith: the whole corpus in the system prompt, compiled once per task. |
| S3s | DEL=FLATs | One agent. | Hybrid BM25 + dense retrieval of the shared chunks, recompiled each step, as the gate run selected it (D-041, §5). |
| S5 | DEL=KGs (KGq if the KG arm is per-query) | One agent. | The **KG arm**, resolved from the gate's verdict (below). |
| S7 | DEL=RAND | One agent. | Random chunks sized per cell to S5's median realised context per compile on the pilot, capped at 50% of the corpus. Per step when S5 is per step, with a fresh seeded draw each step; once per task otherwise (D-024). Targets: [PILOT: S7 targets per cell]. |
| S8 | ENS=k, k = 1..8 | Post hoc, from S1's 8-run pool (§3.3). Not a run. | Monolith. |
| S8k3 | ENS=3 | 3 independent S1 attempts, then a plurality vote on the answer key, with an LLM aggregator on ties only. The winning attempt's end state becomes the sample's. | Monolith. |
| S9 | DEC=1, ISO=0, CTRL=dynamic | One planner-executor. `plan` is forced on the first turn; then it works through the plan with the task's tools in the same context. | Monolith. |
| M1 | DEC=1, ISO=1, CONC=1 | An orchestrator with three tools: `plan` (forced first, replanning allowed), `delegate` (1–3 subtasks per call; one call is one round) and the answer tool. It never gets environment tools. Each subtask runs in a fresh worker context with the task's non-terminal tools. Only the worker's result returns, clipped at 2,000 tokens. Workers in a round run concurrently. | Monolith. |
| M1s | DEC=1, ISO=1, CONC=0 | As M1, with workers run one after another. Workers in a round see the environment as it was when the round began, and their changes merge in subtask order. So M1 and M1s give the model identical inputs (D-034). | Monolith. |
| M1k | M1 + DEL=KG | As M1. | The KG arm. |
| M2 | M1k + SPEC=1 | As M1k. The 3 workers are specialists: the world's domains (F3 service domains, F7 policy domains) are dealt round-robin. Each specialist gets its domains named, a specialty prefix on its KG queries and, on F3, its domains' tool subset. The orchestrator routes each subtask to a specialist. Not run on F1 or F2 (D-034). | The KG arm. |
| M7 | ENS=3, COMM=real | 3 members each propose an answer with a rationale. Then come 2 critique rounds: each member's own history continues with the others' latest proposals, and it re-proposes. Each member keeps its own environment copy. Finally a chair in a fresh context, with only the answer tool, submits. Not run on F3, whose tools mutate the environment. | Monolith. |

**Delivery is arm-wide** (D-034). The orchestrator, chair and aggregator get the arm's delivery too. So each step along the chain changes exactly one switch:
- S9 → M1s changes ISO;
- M1 → M1k changes DEL for every agent;
- M1k → M2 changes SPEC only.

**Fixed structural parameters** (brief §4.2). These are fixed a priori, and never tuned or scaled to a budget:
- 3 workers (M1, M1s, M1k, M2);
- 2 critique rounds and council k = 3 (M7);
- ensemble k = 3 (S8k3).

**The KG arm** (S5, and the workers and orchestrator of M1k and M2) is resolved at the freeze from the gate run named by `--gate-run-id` (D-036):
- **GO family** (GO, GO_WITH_COST_FLAG, GO_PUSH_ONLY, GO_PULL_ONLY) gives the gate's APG*. GO_PULL_ONLY is noted: the main study runs the KG arm push.
- **NO_GO** gives the gate's LGR*.
- Either way it runs with that gate run's selected knobs.
- **Refused:** a live run refuses an INCONCLUSIVE or PRECONDITION_FAIL verdict; an unfrozen, unanalysed or offline gate run; and any oracle arm.
- **Builds.** The KG is built for every KG cell by the gate run's builder (`main.build.kg`, with its build-quality check).
- **Resolution:** [PILOT: KG arm].

### 3.3 The S8 frontier and the live S8k3

**S8 frontier** (brief §4.5; `ape.analysis.frontier`, D-031):
- **Pool.** S1 runs 8 times on every Study A and B task. S8(k), k = 1..8, is a k-run ensemble.
- **Answer.** The ensemble's answer is the **plurality of the runs' canonical answer keys**: the part of the answer the scorer reads, normalised as it normalises, so equal keys always score alike.
- **Success.** S8(k)'s success on a task is the mean over all C(8, k) subsets of its runs (exhaustive subsampling, unbiased for a fresh k-run ensemble).
- **Tie rule.** A tie counts as the expected success of a uniform random choice among the tied answers. No aggregator is run post hoc.
- **Abstentions.** A run with no answer (a cap hit or a harness error) casts no vote but still costs. A subset in which every run abstains fails.
- **Cost.** S8(k) costs the sum of its runs, k times the task's mean run cost, on every meter. Wall-clock may also be priced with members run in parallel (descriptive).
- **Interpolation.** The curve is linear between adjacent k. An arm is compared with S8 at its own realised mean cost per sample in the same cell and tier, on **tokens**, the cap-enforcement meter; every other meter is descriptive. Per task, S8's value is the same convex combination of S8(k) and S8(k + 1), so the contrast stays paired. Three match statuses:
  - **inside:** the arm's cost falls on the curve; the contrast is tested.
  - **below:** an arm cheaper than one S1 run is compared with S8(1) and flagged.
  - **beyond:** an arm costlier than S8(8) is not tested, and is named, never extrapolated.
- **T1.** T1 uses 3-run frontiers in both tiers. Sol's S1 has 3 epochs, so Luna's pool is cut to its first 3.

**Live S8k3** (Study C; D-034):
- **Vote.** The plurality on the same answer keys.
- **Tie rule.** With no unique plurality, an LLM aggregator chooses among the tied candidates. If it does not choose, the earliest tied attempt wins. The aggregator's call is part of S8k3's cost.
- **Use.** Because the tie rules differ, the live S8k3 and the post-hoc S8(3) are compared descriptively only (§2.5).

## 4. Samples

### 4.1 Splits and seeds (D-036)

| Split | Seeds | Used by |
|---|---|---|
| dev | 1000+, shared with the gate. The gate's F3/F7 dev worlds and their paid KG builds are reused, and never rebuilt with another builder. F1/F2 dev worlds are added. | tune |
| pilot | 12000+ | micro-pilot, pilot |
| test | The run's own block of 100 seeds in [13000, 19000): the first live run gets 13000–13008 (9 worlds per cell). A later run freezes the next unused block. | Studies A, B, C, F |
| offline rehearsal | 19000+ | offline runs only |

- **When the test split is generated.** Only after the freeze, by `build-test`, which alone unlocks the main study's test split (`main/<run id>`). Neither the gate's nor Study G's unlock opens it.
- **Never seen before the run.** The test worlds are never inspected before the test phase.
- **Isolation between studies and runs.** The three studies' seed ranges are disjoint, which is asserted at import. A run never regenerates another run's worlds.

### 4.2 Cells

A cell sized in tasks uses whole worlds: the first ⌈n / 12⌉ worlds of the block by seed, 12 tasks per world. So "100 tasks" is 9 worlds × 12 = 108 tasks.

| Plan cell | Study | Arms | Task cells | Worlds × tasks | Epochs | Tier |
|---|---|---|---|---|---|---|
| main.A.s1-pool | A | S1 | F1-2, F1-32, F2-2, F2-10 | 9 × 12 | 8 (the pool) | Luna |
| main.A.arms | A | S9, M1, M7 | F1-2, F1-32, F2-2, F2-10 | 9 × 12 | 3 | Luna |
| main.A.m1s | A | M1s | F1-32 | 5 × 12 | 3 | Luna |
| main.B.s1-pool | B | S1 | F3-5, F3-60, F7-10, F7-1000 | 9 × 12 | 8 (the pool) | Luna |
| main.B.arms | B | S3s, S5, S7, M1, M1k, M2 | F3-5, F3-60, F7-10, F7-1000 | 9 × 12 | 3 | Luna |
| main.C.a | C | S1, M1, M7 | Study A's cells | 2 × 12 | +2 | Luna |
| main.C.b | C | S1, S5, M1, M2 | Study B's cells | 2 × 12 | +2 | Luna |
| main.C.s8 | C | S8k3 | Study A's and B's cells | 2 × 12 | 5 | Luna |
| main.F.luna | F | S5 | F1-32 | 9 × 12 | 3 | Luna |
| main.F.luna-f7-100 | F | S1, S5, M1, M2 | F7-100 | 9 × 12 | 3 | Luna |
| main.F.sol | F | S1, S5, M1 | F1-32, F7-100 | 5 × 12 (D-052) | 3 | Sol |
| main.F.sol-m2 | F | M2 | F7-100 | 5 × 12 (D-052) | 3 | Sol |

- **Cell meanings.** F1-N is breadth aggregation over N suppliers. F2-k is a dependency chain of k hops. F3-n is tool load with n tools. F7-n is policy compliance over n policies (relational, descriptive exceptions).
- **Families.** Study A: F1, F2. Study B: F3, F7.
- **Run order.** The test phase runs Studies A and B first, then C and F (§6).
- **KG builds.** `main.build.kg` builds 11 worlds per KG cell (9 test + 2 pilot) for F7-10, F7-100, F7-1000, F3-5, F3-60, F1-2 and F1-32.
- **Before the test split**, all at 2 worlds × 12 tasks per cell:
  - **micro-pilot** (pilot split, uncapped): S1, S5, M1 on F1-2, F1-32, F7-10, F7-1000, 2 epochs;
  - **tuning** (dev split): §5;
  - **pilot** (pilot split, capped): Study A's arms on Study A's cells and Study B's arms on Study B's cells, 1 epoch.

### 4.3 The S1 pool

- S1 runs 8 epochs on every Study A and B task (`main.A.s1-pool`, `main.B.s1-pool`).
- Its 3 main epochs are the pool's first 3 runs, in run order (plan cell, log file, epoch). The S8 frontier uses all 8.
- Elsewhere S1 has 3 epochs (Study F's F7-100 and Sol cells), so those cells have 3-run frontiers only.

### 4.4 Study C subset

- **Tasks.** The first 2 test worlds (24 tasks) of every Study A and B cell. These are the same tasks the main cells run.
- **Extra epochs.** 2 more epochs for S1, M1 and M7 (Study A cells) and for S1, S5, M1 and M2 (Study B cells). Each of these arms then has 5 runs per task.
- **S8k3** runs only here (`main.C.s8`), with 5 epochs, so it also has 5 runs per task (D-044).
- **Use.** Study C's plan cells feed only the determinism section (A1) and the live S8k3 check. They are never in the confirmatory frame.

### 4.5 Study F tiers

- **Sol.** S1, S5 and M1 on F1-32 and F7-100, plus M2 on F7-100 (M2 is not defined on F1, D-034).
- **Luna.** S1, S5, M1 and M2 run on F7-100 (D-028 #2), so the tier contrast is not confounded with KB size. On F1-32, Luna reuses Study A's S1 and M1 and adds S5.
- **Matched tasks.** Sol runs the first 5 of the 9 test worlds per cell (60 tasks, D-052); Luna runs all 9. T1 compares the tiers on Sol's 60 tasks, which Luna also runs. T1 is descriptive (D-033), so the smaller Sol sample only widens its intervals (about 1.35×).
- **What differs.** Only the agent model. The `kg` role, the KG builds and the embeddings are the same. Sol cells use Luna's token caps (§6).

### 4.6 Models

From `config/models.yaml`, profiles `main_luna` and `main_sol` (frozen):

| Role | Model |
|---|---|
| agent | GPT-6 Luna, effort high; GPT-6 Sol, high, in Study F's Sol cells |
| kg | GPT-6 Luna, effort low |
| build | The gate run's builder (D-017) |
| embeddings | text-embedding-3-small |

Exact model IDs and snapshots are those `readiness/probe_openai.py` confirmed and `PROVENANCE.md` pins.

## 5. Tuning (dev split only; equal budget)

- **Grid.** `python -m ape.run_study tune --study main --run-id <id>` evaluates the candidates declared in `config/tuning_grid_main.yaml` on dev worlds only (seeds 1000+). The freeze hashes the whole grid file. Its rules (D-041):
  - **What is tuned.** The multi-agent arms M1 and M7, each on its own protocol text (S9 runs M1's variant, below): variants of the role notes the arm adds on top of the base prompt every agent gets. A variant never adds generic task advice that would help S1 as much.
  - **Dev cells.** Each tuned system is named for the plan arm it tunes and tunes on its own dev cells, which are cells of the run plan's tuning cells (`main.tune.a`: F1-32, F2-10; `main.tune.b`: F3-60, F7-1000, where M1 alone tunes). M1 tunes on all four, with one selection. Each cell uses 2 worlds × 12 tasks and 1 epoch.
  - **Equal budget:** every tuned system declares exactly `budget_per_system` candidates (the count the run plan prices), before any dev run.
  - **Own knobs only.** A candidate sets only its own arm's knobs; no knob is shared across systems. `tuning.study_grid_problems` checks the grid against the plan before any dev run.
  - **Not tuned.** The structural parameters (§3.2) are not knobs.
  - **S9, M1s, M1k and M2** have no system: they run as M1's selection (D-047), so the chain M1s → M1 → M1k → M2 shares one protocol variant and its single-switch steps (ISO/CONC, DEL, SPEC) differ in nothing else. Separate picks would have been near-arbitrary at the tune's resolution (below).
  - **Sol** cells run the Luna selections.
  - **Known limit** (D-041). With about 40 dev tasks per candidate, the standard error of a difference between two candidates is about 11 pp. So the tune catches broken variants, not small gains.
- **Records.**
  - Every configuration tried is logged in `runs/main/<id>/tune/tuning_log.jsonl`.
  - A re-run archives the previous log beside it, and archived logs count against the budget.
  - The tune records a PC6-style completeness check. A failure is reported, and the analyst resolves it before the freeze or records it as a deviation.
- **Selection.** The highest mean dev success over the system's dev cells. Candidates within 1 pp of the best go to the cheaper one.
  - Selections are written to `runs/main/<id>/config/selected.yaml`, which the freeze hashes.
  - A plan arm named by a system runs as its selection in every later phase.
  - Each arm's knobs are its own (`APE_MAS_<ARM>_*`), so a selection changes only its own arm.
- **Inherited, not re-tuned** (D-041; the grid's `inherited`). Re-tuning these on the same profile and dev worlds would repeat part of the gate's tune.
  - **S5** runs as the gate-selected KG arm with the gate's knobs (D-036, §3.2).
  - **S3s** runs as the gate run's S3s selection.
  - **S1** has no text of its own: its prompt is the base every agent of every arm gets, as the skeptic engineered it in the gate.
  - The gate tuned S5's systems and S3s under its own equal-budget rule (GATE_PREREG §5) on the gate's dev cells, which include this grid's F3-60 and F7-1000 dev worlds. It gave S5's systems 6 configurations each and S3s 3, against 4 per system here.
- **Owners and sign-off.** The grid's `owners` and `signed_off` name each system's owner and record their sign-off. A live tune refuses a placeholder grid, or any system without an owner and a sign-off (D-036). The roles (brief §7.3; D-018):
  - **M-arm prompt author** (an independent team member, not the skeptic). Writes the M-arm prompts and owns the M1 and M7 candidates (S9, M1s, M1k and M2 run as M1's selection): [USER: M-arm prompt author]
  - **Skeptic.** Engineers the baselines S1 and S3s, including caching, to be as strong as possible, and signs off that S1 runs as engineered: [USER: skeptic]
  - **Analyst.** Owns this freeze: [USER: analyst]

## 6. Budget policy

- **Token caps** (brief §4.5: caps, not matched budgets; D-036).
  - **B0** is the median realised total tokens of S1 per task cell on the micro-pilot. The micro-pilot runs uncapped, and errored samples are excluded.
  - **Where B0 comes from.**
    - The pilot measures B0 for the cells the micro-pilot did not run (F2-2, F2-10, F3-5, F3-60).
    - F7-100 borrows the cap of the nearest measured F7 level on a log scale; a tie takes the larger cap.
    - Sol cells use the Luna caps.
  - **One cap per task cell, for every arm:** C_max = ⌈m × B0⌉ total tokens, with m the cap multiple.
  - **Enforcement** is on total tokens over every model call of the sample: every agent and the `kg` role (Inspect's `token_limit`). All four meters are reported.
  - The final caps: [PILOT: token caps].
- **Cap multiple** (D-039, D-045, D-047). m starts at 8. After the pilot, each arm's token-limit hit rate (an Inspect token limit, or an agent stopped on `limit`) is computed over its pilot cells.
  - The decision uses the **one-sided 97.5% Clopper–Pearson lower bound, pooled over the arm's pilot cells** (the gate's PC5 rule), not point estimates per arm and cell, which would escalate or refuse by chance at true rates of 2–5%.
  - If any arm's lower bound exceeds 10%, m doubles for **all** arms (8 → 16, and at most once more, to 32). That arm's pilot cells re-run under the new caps, and the check repeats.
  - Turn-cap hits and other limits (the working-time and cost guards) are gated the same way but never re-run: the freeze refuses when an arm's lower bound for them exceeds 10%.
  - Arms the pilot does not run get projected rates: M1s takes M1's, S8k3 is projected from resampled triples of S1 pilot samples, and Sol cells take the Luna rates.
  - The freeze records the final multiple and every pilot rate (`config/cap_gate.json`). It refuses while any arm's lower bound is above 10% at 32 × B0.
  - The final multiple: [PILOT: cap multiple].
- **Cap hits are failures** (success 0) in every analysis, and cast no vote in the S8 frontier, whatever the scorer read (§2.1).
  - **What counts as a cap hit:** any Inspect sample limit; or an unanswered sample whose single agent reached its turn cap; or, for a multi-agent arm, an unanswered sample in which any agent stopped at its turn cap or on a limit.
  - Cap-hit rates are reported per arm and cell, with the token-cap share and Clopper–Pearson bounds.
- **Other limits.**
  - The per-agent turn caps of §3.1.
  - A per-sample runaway guard (Inspect `cost_limit`): 20 × the cell's conservative per-sample projection, at least $0.50. A hit is a cap hit.
  - No experimental wall-clock cap is applied; wall-clock is reported, never capped. A runaway guard stops a sample whose working time (which excludes waits for connections and rate limits) exceeds 30 minutes at a 12-turn cap, scaled with the turn cap (F1-32: 110 minutes; `budget.sample_working_limit`); a guard hit ends the sample with limit `working` and counts as a cap hit (a failure), and is reported per arm and cell.
- **Budget guard.** Before each phase, and before each test group, the conservative projection must fit what is left of both the program's $5,000 (D-021) and the main study's allocation of $1,100 (D-035).
  - The test phase runs Studies A and B first, so a budget stop loses only Studies C and F.
  - A stopped or failed test is still analysed. Its missing cells make members "not evaluable", named in the report.
- **Cost meters** (brief §6.2), reported for every arm.
  - **Tokens:** input including cache reads and writes, output including reasoning; each part separately and as a total.
  - **LLM calls.**
  - **Cache-adjusted $ and list $,** computed from token counts × `config/model_costs.yaml`, never from billed amounts.
  - **Wall-clock:** Inspect working time, which leaves out rate-limit waits; total time is kept too.
  - Every agent of a multi-agent arm is included, with per-agent accounting (`mas_accounting`).
  - **Realised cost is the logged final attempts' usage.** An errored attempt that Inspect retried is in no log (D-030); its usage is in the run's usage ledgers (`<log dir>/usage_ledger.jsonl`), counted in spend, and the report gives each arm's share of it. Realised cost, the frontier matching and K2-NI's cost ratio exclude it.
  - Build and embedding costs are metered in the ledger and reported per build, outside per-sample cost.

## 7. Fairness and validity controls

1. **One harness and one tool layer.** Every arm runs on the same Inspect harness, agent rules, tools and scorers. Arms differ only in their switches (§3.1).
2. **One corpus.** Every arm draws on the same rendered source corpus of the same world: S1 reads the whole corpus, S3s and S7 its shared chunks, and S5 the KG built from those chunks.
3. **M1 against M1s: identical inputs.** Concurrency changes only timing, by construction (§3.2). D-034's test checks it: identical multisets of model inputs and identical end states.
4. **Arm-wide delivery.** Each step in the chain S9 → M1s → M1 → M1k → M2 changes one switch (§3.2).
5. **Independent prompt author.** The M-arm prompts are written and owned by the M-arm prompt author (§5), not by the skeptic who engineers the baselines.
6. **Tuning parity.** The same budget for every tuned arm, on dev only, declared before any dev run (§5).
7. **One cap for every arm.** Arms are compared at realised cost through the S8 frontier, never at matched budgets. Cap-hit rates are reported (§6).
8. **Extracted KG.** The KG arm is the extracted KG chosen by the gate. Oracle arms are refused (§3.2).
9. **Caching.**
   - Inspect output caching is off for every arm.
   - Provider prompt caching is left at its default for every arm.
   - Cache-read and cache-write tokens are reported per arm.
   - **Per-run cache nonce** (brief §6.2): every agent prompt of a task starts with a short `Run reference` line derived from the run, phase, group and arm (`agent/cache_nonce.py`; recorded per sample as `cache_nonce`), so provider prompt caches never cross runs or arms that share the same corpus prefix. Epochs of one arm share their nonce, so a later epoch can read an earlier epoch's cache: cache-hit claims use first epochs. kg-role calls (APG classify, LightRAG keywords) carry no nonce.
10. **Blind analysis.** The analysis code (`analyze_main.py`, `analysis/`) is frozen with this design, before the test split exists.
11. **Fresh test split.** Generated after the freeze (§4.1).
12. **Programmatic scoring only.**
13. **Logged runs.** Provider, model, date and concurrency settings are logged with every eval. The concurrency settings matter because concurrent workers share the model's connection limit, so M1's wall-clock gain depends on it.

## 8. Preconditions and invariants

These are the brief's §4.4 invariants and the harness checks.

**A violation blocks the analysis until explained.** `analyze_main` computes and reports every check but does not stop on them. The rule is the analyst's:
- Results resting on a violated check are not reported as findings until the violation is diagnosed from the logs.
- The diagnosis, or the harness defect with its fix and re-run, is recorded in the Deviations log.

| ID | Check | Computed by | Rule |
|---|---|---|---|
| I1 | M1 ≈ M1s on accuracy (brief §4.4) | M4, `main_descriptive.m4` | A violation is a 95% interval of M1 − M1s that excludes 0. It calls into question M1 standing in for M1s in the M1 family (§2.4). |
| I2 | S5 ≥ S7, and S5 > S7 (brief §4.4) on Study B's cells | `main_descriptive.invariants` | S5 ≥ S7 fails only on evidence of a shortfall larger than 3 pp (the gate's `violation_test`). S5 > S7 uses a sign flip at one-sided 0.05. If S5 ≈ S7, the KG effect may be only a shorter-prompt effect; K1, K2 and K2-NI are then reported with that caveat. |
| I3 | The live S8k3 against the post-hoc S8(3) | `main_descriptive.s8k3_check` | Reported. It stands in for the M7-R0 invariant, which is not run. |
| H1 | Harness errors per arm and cell | `gate_stats.harness_rates` | A violation is when the one-sided 97.5% Clopper–Pearson lower bound exceeds 2% (the gate's PC5 rule, D-027). |
| H2 | Cap hits per arm and cell | `harness_rates`, `caps_table` | The pilot gate (§6) keeps these at or below 10%. At test, a lower bound above 10% is reported as a caveat on every member using that arm and cell. It does not block, since cap hits are pre-registered failures. |
| H3 | The frozen caps were the caps applied | `analyze_main.caps_table` (`consistent`) | Any inconsistency is a violation. |
| H4 | Coverage: every confirmatory plan cell ran, and every row was read | `analyze_main` (`coverage`, `problems`) | Missing cells make members "not evaluable", named. A cell missing for any reason other than a budget stop is a violation. |
| H5 | Rows are counted once | `main_stats.confirmatory_rows` | Duplicate sample-epochs are dropped and counted: rows with the same sample uuid and epoch, whichever log file holds them (a completed sample copied into a retry's log keeps its uuid). |
| W1 | No supported claim rests on one world, or on a world whose KG build failed its quality check (§2.2) | `main_stats.world_influence`, `caveats`; `analyze_main` reads `build-test/build_quality.json` | Not a block: the claim is reported with the caveat, the influential world and the result without it. The analyst diagnoses the world from its logs (a harness or build defect is a deviation, §10). |

**Not checked**, because the arms are not in this study:
- S5 ≈ M3 (handoff) and M1 with one worker ≈ S1: Tier B.
- S6 ≥ S5o ≥ S5: the gate's diagnostics on the gate's own test worlds (GATE_PREREG PC3).

## 9. Decision and reporting rules

- **Labels** (§2.4): supported, not supported, not tested, not evaluable.
- **Non-significance is never reported as "no effect".**
  - A superiority or `less` member that is not supported means the effect was not shown.
  - A TOST member that is not supported means equivalence was not shown. That is neither an effect nor its absence.
  - A supported TOST member means no effect beyond ±6 pp.
- **Every family is reported**, whatever its result. Each member gets:
  - its estimate;
  - the sign-flip, BCa and cluster-t intervals;
  - p and its Holm level;
  - clusters and tasks;
  - its label and the reason for it;
  - for K2-NI, the cost ratio with its interval;
  - for a pooled member, each cell's estimate and interval beside the pooled one;
  - its per-world influence (the most influential worlds, and the estimate and test without each) and any §8 caveat;
  - the cells left out.
- **Descriptive results are labelled "descriptive"** wherever they appear, and carry no claim (§2.5).
- **Mechanism claims cite a single-switch contrast.** Anything else is labelled a bundle contrast (§2.5).
- **Always reported:**
  - every cost meter per arm and cell;
  - cap-hit and error rates;
  - the frontier table;
  - the invariants (§8);
  - tuning parity (candidates per system and the selections);
  - the KG arm and the gate run it came from;
  - every deviation.
- **The report.** `runs/main/<id>/report/report.md` and `decision.json`, written by the analyze phase at the frozen commit.
- **Publication.** All logs and negative results are published (brief §8).

## 10. Deviations, extensions and the freeze

- **Freeze.** The procedure is in the header. The freeze records this file's hash and the frozen design (§4–§6), and the test split exists only after it.
- **Deviations.** Any change after the freeze, to this file, a frozen file or the code, is a deviation. It is logged below with its date, the change, the reason and who approved it.
  - Running with the change needs a new run id, which freezes a fresh test-seed block.
  - A harness defect found during the test is fixed the same way. The defective run is still reported.
- **No pre-registered extension.**
  - No α is reserved for one, and any further data on these hypotheses needs a new pre-registration.
  - A run frozen with `--extension-of` is analysed by `analyze_main` alone, as a primary-only analysis of its own worlds at the table's α. Its report says it is not a pre-registered extension and does not combine with the primary run.

## Deviations log

| Date | Change | Reason | Approved by |
|---|---|---|---|
