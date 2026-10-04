# Pre-registration: M5 add-on study (ledger orchestrator)

**Status: DRAFT.** Unfilled items in the body are marked `[USER: <label>]`; each needs a decision before the freeze.
- This study is an add-on to the main study (D-053). It has its own hypothesis families and α, and **adds no hypothesis to the main study's families**: PREREGISTRATION_MAIN.md, its table, its Holm families and its α are unchanged by it.
- The table in §2.3 is generated. It is `ape.analysis.m5_hypotheses.render_hypothesis_table()` pasted verbatim between its marker comments, and `tests/test_prereg_m5.py` fails if the two differ. To change a hypothesis, change `m5_hypotheses.py` and paste the new output.
- This header is instructions and is never checked. From §1 on, any remaining marker blocks the freeze.

**Freeze procedure.**
1. A main run is frozen (`runs/main/<main id>/freeze.json`). This study inherits its design and runs on its test worlds (§4).
2. The study's cells are enabled (`studies.m5` in `config/run_plan.yaml`; every cell is `enabled: false` until then), committed.
3. Run the study up to the freeze: `uv run --locked python -m ape.run_study all --study m5 --run-id <id> --main-run-id <main id>`. It runs preflight and the pilot (§4.3), then stops at the freeze while a marker remains.
4. Fill every `[USER: …]` item in the body, set the status to FROZEN, and commit.
5. Run `uv run --locked python -m ape.run_study freeze --study m5 --run-id <id>`. The freeze records this file's hash, the analysis code (`src/ape/analyze_m5.py`, `src/ape/analysis/`), the commit, and what the run inherits from the main run (`config/main_inheritance.json`: the main run's id and freeze hash, its test seeds, token caps and their multiple, its KG resolution, its selections and the limit rules).
6. Then `test` and `analyze`. After the freeze, any change to this file, to a frozen file or to the code is a logged deviation (§6).

## 1. Question

Does a shared text ledger help a multi-agent team? The ledger is a Magentic-style orchestrator state: a **task ledger** (facts, guesses, plan) and a **progress ledger** each round (is the request satisfied, is the team looping, is it progressing, the next step), with a **replan after repeated stalls**. The brief (§4.3) lists it as the mechanism contrast "text ledger" with the prediction "+ on F2" (dependency chains).

| Question | Hypothesis |
|---|---|
| Does the ledger help the orchestrator-and-identical-workers team, on dependency chains (F2, the brief's prediction) and on breadth tasks (F1)? | L1 (confirmatory: L1.F2, L1.F1) |
| Does it help the specialist team? | L2 (confirmatory: L2.pooled, on Study B's cells) |
| In which cells, at what price, and with what ledger behaviour? | L1-cells, L2-cells, LC, LD (descriptive) |

**Adapted from the brief** (D-053). The brief defines M5 as M2 + text ledger on F2 and F5. M2 is undefined on F1 and F2: routing to segment specialists is blind there (D-034). So the F2 prediction is tested on the M1 team (arm M5 = M1 + ledger), and the brief's own team (M2 + ledger, arm M5-spec) on F3 and F7, where M2 is defined. F5 is not in this study. The arm name M5 is not the main study's hypothesis M5 (role specialization, brief H1e); this study's IDs are L1, L2 and their descriptive companions.

## 2. Hypotheses, estimands and tests

### 2.1 Outcome, unit and estimand

As the main study's (PREREGISTRATION_MAIN.md §2.1), with the same code (`main_load`, `main_stats`):
- **Outcome.** Task success, binary and programmatic. An errored sample (after its retries) and a cap hit are failures.
- **Task value.** An arm's mean success over its 3 epochs on a task.
- **Contrast.** The paired difference ledger arm − control on each task both ran: M5 − M1, M5-spec − M2.
- **Estimand.** Within a cell, the mean contrast over the paired tasks; over a member's cells, the equal-weight mean of the cells. A pooled claim is a claim about that average, and each cell's estimate is reported beside it.
- **Clusters.** Worlds are the independent units. F1 and F2 worlds of one seed share one supplier registry, so they form one cluster across Study A's cells: L1's members have **9 clusters**, however many cells they pool. F3 and F7 worlds are their own clusters: L2 has **36**.
- **Concurrent controls only.** Each ledger arm is compared only with the control that ran beside it in this study's own test cells (`m5.test.*`), on the same tasks. The main study's own M1 and M2 runs are never used in these contrasts: they ran earlier, so model and provider drift between the runs would enter the contrast. The analysis drops any row from another plan cell and names it (`m5_report.m5_rows`).

### 2.2 Tests

- **Decision test.** The main study's world-clustered sign-flip test on the per-task contrast (PREREGISTRATION_MAIN.md §2.2): exact over all 2^G flips up to 20 clusters (L1: 9; the smallest attainable p is 2^−9 = 0.00195, below L1's smallest Holm level, 0.0125), 10,000 Monte Carlo flips above (L2: 36). Ledger arm and control are exchangeable under the null, so the test is exact for these superiority contrasts; each member still reports its per-world influence, and a supported claim that one world carries is flagged.
- **Superiority**, one-sided: H0 Δ ≤ 0 against Δ > 0.
- **α = 0.025 one-sided per family, Holm within a family**, the main study's convention (brief §7.4: Holm within each hypothesis family, no correction across families). This study's families are its own: they are not added to any of the main study's families, and the main study's are not corrected for them.
- **Intervals.** The sign-flip test inverted (two-sided 95%) is the headline; the world-clustered BCa bootstrap and the cluster-t interval are reported beside it.

### 2.3 Hypothesis table

Generated from `ape.analysis.m5_hypotheses`, the code the analysis runs. The planned power is from `ape.analysis.m5_power` at the planned sizes and the gate's σ priors (§2.6).

<!-- BEGIN GENERATED: m5_hypotheses.render_hypothesis_table() -->
| ID | Member | Source | Contrast | Cells | Test | Margin | Holm family (α, procedure) | Planned MDE / power | Status |
|---|---|---|---|---|---|---|---|---|---|
| L1 | L1.F2 | brief §4.3 (text ledger, + on F2); D-053 | M5 − M1 | F2-2, F2-10 | one-sided superiority (H0 Δ ≤ 0) | – | L1 (0.025, holm) | MDE ≈ 10 pp; family type I 0.019 (0.019 with F1 at +10 pp); power 0.85 at +10 pp with F1 null, 0.92 with both at +10 pp, 0.64 / 0.96 at +8 / +12 pp with F1 null, 0.84 if a +20 pp effect sits on F2-10 alone (Holm of 2; 10,000 studies per null, 2,000 per effect, 108 tasks per cell, gate σ priors; m5_power seed 20261053) | confirmatory |
| L1 | L1.F1 | brief §4.3 (text ledger, + on F2); D-053 | M5 − M1 | F1-2, F1-32 | one-sided superiority (H0 Δ ≤ 0) | – | L1 (0.025, holm) | MDE ≈ 10 pp; family type I 0.019 (0.018 with F2 at +10 pp); power 0.84 at +10 pp with F2 null, 0.91 with both at +10 pp (Holm of 2; the same simulation) | confirmatory |
| L2 | L2.pooled | D-053 | M5-spec − M2 | F3-5, F3-60, F7-10, F7-1000 | one-sided superiority (H0 Δ ≤ 0) | – | L2 (0.025, holm) | MDE ≈ 5.5 pp; type I 0.023; power 0.36 / 0.57 / 0.77 / 0.90 / 0.995 at +3 / +4 / +5 / +6 / +8 pp (10,000 studies at the null, 2,000 per effect, 108 tasks per cell, gate σ priors; m5_power seed 20262053) | confirmatory |
| L1-cells | L1-cells.F1-2 | D-053 | M5 − M1 | F1-2 | estimate and intervals | – | – | – | descriptive |
| L1-cells | L1-cells.F1-32 | D-053 | M5 − M1 | F1-32 | estimate and intervals | – | – | – | descriptive |
| L1-cells | L1-cells.F2-2 | D-053 | M5 − M1 | F2-2 | estimate and intervals | – | – | – | descriptive |
| L1-cells | L1-cells.F2-10 | D-053 | M5 − M1 | F2-10 | estimate and intervals | – | – | – | descriptive |
| L2-cells | L2-cells.F3-5 | D-053 | M5-spec − M2 | F3-5 | estimate and intervals | – | – | – | descriptive |
| L2-cells | L2-cells.F3-60 | D-053 | M5-spec − M2 | F3-60 | estimate and intervals | – | – | – | descriptive |
| L2-cells | L2-cells.F7-10 | D-053 | M5-spec − M2 | F7-10 | estimate and intervals | – | – | – | descriptive |
| L2-cells | L2-cells.F7-1000 | D-053 | M5-spec − M2 | F7-1000 | estimate and intervals | – | – | – | descriptive |
| LC | LC | D-053 | The ledger's price: the realised cost ratios M5/M1 (Study A's cells, F1, F2) and M5-spec/M2 (Study B) on tokens and $, per cell and pooled, with world-clustered 95% intervals. | – | cost | – | – | – | descriptive |
| LD | LD | D-053 | Ledger diagnostics per arm and cell: replans per task, the share of samples that replanned, the stall rate (stalled rounds over rounds) and the ledger's share of the sample's tokens. | – | ledger | – | – | – | descriptive |
<!-- END GENERATED: m5_hypotheses.render_hypothesis_table() -->

### 2.4 Confirmatory families

Labels as the main study's (PREREGISTRATION_MAIN.md §2.4): **supported** (rejected at its Holm level), **not supported** (tested, not rejected), **not tested** (Holm stopped before it), **not evaluable** (its data are missing). "Not supported" never means "no effect".

**L1, the ledger on the M1 team.** Holm of 2 at one-sided 0.025.
- **L1.F2:** M5 − M1 > 0 on dependency chains, averaged over F2-2 and F2-10 with equal weights. The brief's prediction (§4.3: "+ on F2").
- **L1.F1:** M5 − M1 > 0 on breadth tasks, averaged over F1-2 and F1-32 with equal weights (D-053: the ledger should matter on both families).
- **Claim if supported** (per member): on that family's two cells, averaged with equal weights, adding the ledger to the M1 team raises task success at the same token caps. Each cell's estimate is reported beside the pooled one; F1-2 and F2-2 are near the ceiling (M1 around 0.85 in the planning assumptions), so most of a pooled effect may come from F1-32 and F2-10.
- Both members rest on the same 9 registry clusters, so their tests are positively dependent; Holm holds the family-wise level under any dependence.

**L2, the ledger on the specialist team.** One member at one-sided 0.025.
- **L2.pooled:** M5-spec − M2 > 0, averaged over Study B's four cells (F3-5, F3-60, F7-10, F7-1000) with equal weights.
- **Claim if supported:** averaged over the four cells with equal weights, adding the ledger to the specialist team raises task success at the same token caps.

**How the design was chosen** (`ape.analysis.m5_power.design_check`, seed 20261153, case i seeded 20261153 + i; 10,000 simulated studies per null, 2,000 per effect; §2.6's model and sizes). Power is the rate at which a member is supported; Δ is the equal-weight mean effect over the member's cells.

| Design | Type I (family-wise) | Power |
|---|---|---|
| A. L1.F2 alone | 0.018 | 0.52 / 0.77 / 0.93 / 0.99 at +6 / +8 / +10 / +12 pp; 0.71 if +15 pp sits on F2-10 alone |
| B. F2-10 alone | 0.018 | 0.35 / 0.50 / 0.68 / 0.87 / 0.99 at +8 / +10 / +12 / +15 / +20 pp on F2-10 |
| **C. L1.F2 + L1.F1, Holm of 2 (chosen)** | 0.019; L1.F2 false claims 0.017 with F1 at +10 pp | L1.F2 with F1 null: 0.64 / 0.83 / 0.96 at +8 / +10 / +12 pp; L1.F1 with F2 null: 0.83 at +10 pp; both at +10 pp: 0.91 / 0.91 |
| **D. L2.pooled alone (chosen)** | 0.022 | 0.35 / 0.57 / 0.78 / 0.91 at +3 / +4 / +5 / +6 pp |
| E. One family over L1.F2, L1.F1 and L2.pooled (Holm of 3) | 0.020 | L1.F2 0.78 at +10 pp (others null); L2.pooled 0.60 at +5 pp (others null) |

- **F2 pooled over both cells, not F2-10 alone.** At a uniform effect the pooled member has nearly twice the power (0.93 against 0.50 at +10 pp). F2-10 alone does better only if the whole effect sits on F2-10 (0.87 against 0.71 at +15 pp there).
- **L1.F1 is confirmatory.** It is powered as well as L1.F2 (0.83–0.84 at +10 pp in the Holm of 2), so the rule "an underpowered member is descriptive" does not demote it. Its price is L1.F2's power: 0.93 alone against 0.83–0.85 in the Holm of 2 at +10 pp, an MDE of about 10 pp instead of about 8.5 pp.
- **L2 is confirmatory.** Its 36 clusters give an MDE of about 5.5 pp.
- **Two families, not one.** One Holm family over all three members would lower L2.pooled's power at +5 pp from 0.78 to 0.60 and L1.F2's at +10 pp from 0.83 to 0.78. Each hypothesis is its own family, as each of the main study's is.
- **Planned MDE** (power 0.8): **10 pp** for each L1 member, **5.5 pp** for L2.pooled.

### 2.5 Descriptive analyses

Reported with intervals and labelled "descriptive"; no claim rests on them.

| ID | What is reported |
|---|---|
| L1-cells, L2-cells | Each cell's contrast (M5 − M1 in Study A's four cells; M5-spec − M2 in Study B's four), with intervals. |
| LC | The ledger's price: the realised cost ratios M5/M1 (Study A, F1, F2 and each cell) and M5-spec/M2 (Study B and each cell) on tokens and cache-adjusted $, cells equally weighted, with world-clustered 95% BCa intervals. No cost condition is pre-registered. |
| LD | Ledger diagnostics per ledger arm, pooled and per cell, from each sample's `mas_ledger` and `mas_accounting` records (`m5_load`): replans per task, the share of samples that replanned, rounds per sample, the stall rate (stalled rounds over rounds), the share of samples with a stall, the ledger's share of the sample's tokens, and success with and without a replan (conditional on each run's own difficulty, so not an effect of replanning). A field the records lack is reported as not available. |

Also reported: per arm and cell, success, every cost meter, cap-hit and harness-error rates with Clopper–Pearson bounds, and the design (tasks, worlds, epochs, plan cells).

### 2.6 Power

- **The model and the path.** `ape.analysis.m5_power` simulates through the main study's model and analysis: `main_power.simulate` draws a study (logit success = arm location + world effect σ_w 0.5 + world × arm effect σ_g 0.3 + task difficulty σ_u 1.5 + task × arm effect σ_v 0.5, the gate's priors; F1 and F2 worlds of one seed share the world effect), and `main_stats.evaluate_family`, the function the report runs, tests it.
- **Sizes.** The runner's (`m5_power.m5_sizes`, from `studies.m5`): 108 tasks per cell over 9 worlds, 3 epochs per arm (§4.1).
- **Scenarios.** Each control at the main study's assumed S1 baseline for its cell plus 0.05 (M1) or 0.03 (M2), as `main_power`'s scenarios put them (F1-2 0.85, F1-32 0.45, F2-2 0.85, F2-10 0.55; F3-5 0.78, F3-60 0.63, F7-10 0.88, F7-1000 0.48); the ledger arm at the control + Δ, capped at 0.98.
- **Type I error** is at or below nominal at every null: L1 family-wise 0.019 (10,000 studies; 0.018–0.019 when the other member has a +10 pp effect), L2 0.023 (10,000 studies). Ledger arm and control are exchangeable under these nulls, so the sign flip is exact.
- **Planned power** (2,000 studies per effect): the table's column (§2.3) and §2.4. Reproduce with `uv run --locked python -m ape.analysis.m5_power --reps 2000 --null-reps 10000 [--design] [--sensitivity]`: the table's scenario j of family i (L1 = 0, L2 = 1) is seeded 20261053 + 1000 i + j; the design check's case i 20261153 + i; the sensitivity 20261653 + 10 j + i.
- **Sensitivity.** With a larger world × arm spread (σ_g 0.5 instead of 0.3), L1.F2's power at +10 pp (F1 null) falls to 0.69 and L2.pooled's at +5 pp to 0.61; with a larger world spread (σ_w 0.8), 0.84 and 0.80. The ledger's effect varying across worlds is what costs power, since 9 clusters carry L1.
- **The pilot does not re-simulate.** The sizes are fixed (D-053); a member keeps its test whatever its achieved power, and the report gives its interval.

## 3. Arms

Each ledger arm differs from its control in one switch, STATE = text ledger (brief §4.2).

| Arm | Definition | Cells |
|---|---|---|
| M5 | M1 + ledger: the orchestrator keeps the task ledger and the progress ledger, and replans after repeated stalls; workers, tools, prompts and everything else are M1's. | F1-2, F1-32, F2-2, F2-10 |
| M1 | The control: orchestrator with 3 identical workers, monolith delivery (the main study's M1, run again here). | as M5 |
| M5-spec | M2 + ledger, the same switch on the specialist team. | F3-5, F3-60, F7-10, F7-1000 |
| M2 | The control: orchestrator with specialist KG workers (the main study's M2, run again here). | as M5-spec |

- **What the four arms share with the main study:** the models (GPT-6 Luna, high effort; the `kg` role), the per-turn agent rules and base prompt, the tools and tool exposure, push delivery, the per-agent turn caps, the token caps (§4.2) and the limit rules.
- **Selections.** This study tunes nothing. M5 and M5-spec run under M1's prompt selection, as M1k and M2 do (D-047); M2 and M5-spec use the main run's KG arm (its gate resolution).
- **The ledger's own settings** (what a stall is, how many stalls trigger a replan) are fixed in the arm's code, which the freeze fixes; they are not tuned.

## 4. Samples

### 4.1 Worlds and tasks

- **The main study's test worlds.** This study builds no worlds. It runs on the frozen main run's test-seed block, the same 9 worlds per cell and 12 tasks per world as the main study's Studies A and B, so its arms are paired with each other on the same tasks.
- **Cells** (`studies.m5`, `config/run_plan.yaml`):

| Plan cell | Arms | Cells | Tasks per cell | Epochs |
|---|---|---|---|---|
| `m5.test.a` | M5, M1 | F1-2, F1-32, F2-2, F2-10 | 100 → 9 worlds × 12 = 108 | 3 |
| `m5.test.b` | M5-spec, M2 | F3-5, F3-60, F7-10, F7-1000 | 100 → 108 | 3 |

- The ledger arm and its control run in the same plan cell, at the same time, so they are concurrent (§2.1).

### 4.2 Inherited settings

From the frozen main run, unchanged (`config/main_inheritance.json`, which the freeze hashes): the test-seed block, the token caps per task cell and their multiple, the KG resolution, the selections, the design knobs and the working- and cost-limit rules. The caps are therefore the same for each ledger arm and its control: the ledger's own tokens count against the same cap (brief §4.5: caps, not matched budgets), and a cap hit is a failure.

### 4.3 Pilot

`m5.pilot.a` (M5 on F2-10) and `m5.pilot.b` (M5-spec on F3-60), 20 tasks, 1 epoch, on the main run's pilot worlds at its frozen caps. It checks the ledger arms' cap-hit and harness-error rates and that their records are written. No pilot result enters the analysis. If a ledger arm's pilot cap-hit rate is shown to be above 10% (its one-sided 97.5% Clopper–Pearson lower bound, the main study's rule), the study is not frozen as is: the caps cannot change without breaking the inheritance, so the owner decides between running at the inherited caps, with the cap-hit rate stated beside every result, and not running; the decision is logged in DECISIONS.md.

### 4.4 Budget

About $110 conservative on Luna when enabled (D-053), within the program budget; zero while the cells are off. A budget stop before both test cells finish leaves the missing members "not evaluable", named in the report.

## 5. Decision and reporting rules

- **Labels:** supported (rejected at its Holm level), not supported (tested, not rejected: the effect was not shown, never "no effect"), not tested (Holm stopped before it), not evaluable (its data are missing, named).
- **Every family is reported**, whatever its result, with each member's estimate, the three intervals, p and its Holm level, clusters and tasks, its label and reason, the per-cell estimates of a pooled member, and its most influential worlds.
- **Always reported:** the descriptive analyses (§2.5), cap-hit and error rates per arm and cell, the coverage of every plan cell and group, the inherited main run, and every deviation.
- **The report.** `runs/m5/<id>/report/report.md` and `decision.json`, written by the analyze phase (`ape.analyze_m5`) at the frozen commit.
- **Roles.** Owner: [USER: owner]. Analyst: [USER: analyst].

## 6. Deviations, extensions and the freeze

- **Freeze.** The procedure is in the header.
- **Deviations.** Any change after the freeze, to this file, a frozen file or the code, is a deviation, logged below with its date, the change, the reason and who approved it. Running with the change needs a new run id. A harness defect found during the test is fixed the same way; the defective run is still reported.
- **No pre-registered extension.** No α is reserved for one. A run frozen as an extension is analysed alone, as a primary-only analysis at the table's α, and its report says so (`analyze_m5.role_context`).
- **The main study is unaffected.** Running, not running or stopping this study changes nothing in PREREGISTRATION_MAIN.md: no hypothesis, family, α or frozen file of the main study.

## Deviations log

| Date | Change | Reason | Approved by |
|---|---|---|---|
