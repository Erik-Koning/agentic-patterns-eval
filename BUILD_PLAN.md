# Build plan: making the main study and Study G runnable

**Date:** 2026-10-03.

**Why this plan exists:** the gate is code-complete (READINESS_AUDIT.md). The main study and Study G are not runnable: their comparison arms, runner, analysis and pre-registrations don't exist yet. This plan records everything needed to make them runnable, in order.

**Process:** the same as FIX_PLAN and RELIABILITY_REVIEW.
- Work packages are built in isolated worktrees and reviewed, tested and merged.
- An independent review runs before any paid run.
- Nothing runs live for a study until its pre-registration is frozen.

---

## 0. Design decisions applied first (D-028)

These change the plan, the configuration and the documents before any build work, so the build starts from the corrected design.

| # | Decision | Change | Status |
|---|---|---|---|
| 1 | **Drop the persona test (K3 / brief H2b).** The worlds contain no persona content, so S5-P0 equals S5. | Remove S5-P0 from `main.pilot.b` and `main.B.arms` and from the budget arms. HYPOTHESES K3 becomes dropped. | ✅ |
| 2 | **Fix the tier confound (H6).** Sol runs F7-100 while Luna ran only F7-1000. | Add a Luna F7-100 cell (S1, S5, M1, M2 × 100 tasks × 3 epochs). Add the missing KG builds: F7-100 for both tiers, and F1-2 for the micro-pilot's S5. | ✅ |
| 3 | **Report the underpowered tests as descriptive.** H1d / M4 (concurrency) gives ±6–10 pp at n = 50 against a 2 pp claim. H2c / K4 (KB scaling) has 2 levels, and its relational-vs-independent clause has no independent cells. | HYPOTHESES M4 and K4, and the brief's hypothesis text, say "descriptive"; no confirmatory test is claimed. | ✅ |
| 4 | **Make Study G's short sessions overflow the window.** At the default ~1K-token tool files, 20- and 24-case sessions never cross W = 32K. | Measured on 20 reference sessions per setting: 20-case cells use 2,250-token files and the 24-case cell 1,750, so the median crossing lands at 0.70 of the session, matching the 40-case cells (0.70). The 10-case cell stays a no-overflow control. `make_world` passes F8 knobs through. | ✅ |
| 5 | **Defer the other external benchmarks** (τ³-bench, AppWorld, BrowseComp-Plus, CoDA) to Tier B. | The brief §5.4 and EVAL_DESIGN say deferred. GraphRAG-Bench (the gate's PC1) remains the only wired anchor. | ✅ |

---

## 1. What "runnable" means

A study is runnable when **all** of these hold:

1. **Arms:** every arm in its `config/run_plan.yaml` cells exists. A gold-knowing mock drives each arm through the real loop and must score 100%, and a misbehaving mock must not crash it.
2. **Runner:** a one-command orchestrator covers preflight, builds, tuning, pilot, freeze, test-world builds, test and analysis. It is resumable, budget-guarded and recorded per run. It has the study's own seed range and test-split lock, so it never touches another study's test worlds.
3. **Tuning:** declared grids with an equal budget per system and owner sign-off before a live tune.
4. **Analysis:** code computes every pre-registered test. Each test's type I error and power are simulated end to end through the real analysis path.
5. **Pre-registration:** a study pre-registration with `[PILOT: …]` / `[USER: …]` placeholders, frozen and hashed with the code, as the gate's is.
6. **Smoke:** live smoke checks cover the study's new arm types, and the runner refuses to start without a fresh passing smoke.
7. **Budget:** the study's projection fits its allocation, and the program fits $5,000.
8. **Review:** an independent offline review of the new code finds no open A or B findings.

---

## 2. Work packages

### Main study

| ID | Package | What | Depends on | Acceptance | Size |
|---|---|---|---|---|---|
| **B1** | Study-generic runner | Refactor `run_gate`'s phases into a study-parameterised orchestrator. The gate keeps its behaviour and tests. Each study has its own split seed range (proposed: main 4000+/2500+, Study G 5000+/2700+), its own test lock and its own freeze, all with per-run outputs. | — | Every gate test passes unchanged. `run_study all --study main --offline` runs end to end on the existing single-agent arms. | M |
| **B2** | Multi-agent framework and arms | A worker/orchestrator layer on Inspect, with isolated worker contexts and a concurrency switch. It provides message passing, plus per-agent cost and token accounting and switch-vector logging (DEC, ISO, CONC, COMM, ENS, SPEC). Arms: **S9** (plan, then execute in one context), **M1/M1s** (orchestrator plus identical workers, parallel/serial; F1 sub-questions come from task tags), **M1k/M2** (KG-scoped and specialist workers), **M7** (council: k members, 2 critique rounds, chair) and the **S8** frontier (best-of-k over the S1 pool, matched on realised cost). | — | The gold-mock gives 100% on F1/F2/F3/F7 for every arm, and the robustness suite passes. Cost accounting sums correctly across agents. | L |
| **B3** | Tuning grids | Equal budgets for S1, S5, S9, M1, M7, M1k and M2, declared in a study tuning grid with owners and sign-off. | B1, B2 | PC6-style completeness checks on the main grid. | M |
| **B4** | Main analysis | Mixed-effects logistic models (the tool is decided in B4: lme4 via R, or Bayesian bambi; statsmodels has no crossed logistic GLMM), the single-switch mechanism contrasts, Holm, TOST, cluster-robust intervals, S8 frontier interpolation per cost meter, cost-meter rank flips (Kendall τ) and pass^k. Plus a per-hypothesis power simulation and a decision report. | B1 | Simulated type I error and power for every confirmatory test, through the real path. | L |
| **B5** | Main pre-registration | `PREREGISTRATION_MAIN.md`: hypotheses as confirmatory or descriptive (D-028), estimands, tests, sizes, splits, freeze. | B2–B4 | The freeze refuses on placeholders and guards the code, as in the gate. | M |
| **B6** | Build cells | The F7-100 and F1-2 KG builds added by decision 2, wired into the runner. | B1 | The build-quality check covers them. | S |

### Study G

| ID | Package | What | Depends on | Acceptance | Size |
|---|---|---|---|---|---|
| **B7** | Session runner and context policies | A `ContextPolicy` layer, so each arm transforms the view the model sees. It enforces W, keeps view-token and per-item records, and adds **mid-session checkpoint/resume** so a retried session does not restart from item 1. The F8 knobs from decision 4 reach the generator. | B1 | CM0 and O-state behave as today, and a killed session resumes mid-way. | M |
| **B8** | Context-management arms | **CM-prune**, **CM-trim**, **CM-sum** (a new `cm` summariser role), **CM-todo** (Inspect `todo_write` / `memory`), **CM-reset** (a fresh context with a handoff note), **CM-native** (the provider's native compaction, if the probe shows it is supported) and **S-CM\*** (the best single-agent stack, chosen on dev). | B7 | The gold-mock gives a perfect session under every arm. Management tokens are metered separately. Probes are never appended. | M |
| **B9** | Topology arms in sessions | M1 and M2 from B2, adapted to run inside a session. S1 is CM0. | B2, B7 | Perfect sessions under the gold-mock; per-agent cost accounting. | L |
| **B10** | Study G analysis | A logistic GLMM with session random intercepts, degradation slopes (success ~ log view tokens × position) by arm × tier, TOST for strategy × capability, RER and headroom recovered, probe F1 per checkpoint, the failure taxonomy, and a session-clustered power simulation. | B1, B7 | Simulated operating characteristics for G-H1 to G-H3. | M–L |
| **B11** | Study G pre-registration, grid and probes | `PREREGISTRATION_G.md`. The tuning grid for the CM arms (equal budgets) with sign-off. The probe extended to Sol, Astra and native compaction. The open G-H1 single-agent reference (§4) is decided here. | B8–B10 | The freeze works and the probe output is checked. | M |

### Shared

| ID | Package | What | Depends on | Acceptance | Size |
|---|---|---|---|---|---|
| **B12** | Smoke and review | Live smoke checks for each new arm family (one small task or session each, within a cap). The runner requires them. Then an independent review round of B1–B11 and a budget re-projection. | B1–B11 | No open A/B findings; the smoke projection fits its cap. | M |

---

## 3. Order and parallelism

1. **First, in parallel:** B1 (runner), B2 (multi-agent framework and arms) and B7 (session runner). They touch different files.
2. **Then:** B3 and B6 (main) in parallel with B8 and B9 (Study G).
3. **Then:** B4 and B10 (analysis).
4. **Then:** B5 and B11 (pre-registrations).
5. **Last:** B12 (smoke and independent review). Then each study goes live: probe, smoke, `run_study … all`, which stops at the freeze for sign-off.

The gate can run live at any time during this build. The main study needs the gate's verdict only to fix its KG arm, which is APG if GO and LightRAG otherwise (B5).

---

## 4. Open questions for later (not blocking the build)

- ~~**GLMM tooling (B4)**~~ **Decided (D-029):** confirmatory tests are design-based (brief §7.4), clustered by world (main) or session (Study G); the GLMM is descriptive, via statsmodels' variational-Bayes mixed GLM. No R.
- **G-H1's single-agent reference (B11):** today S1's overflow counts as failure for the rest of the session, so the harness sets the S1-vs-multi-agent gap. Choose before the Study G pre-registration.
- **CM-native at one tier only,** **S1+KG and S-subiso** (in the hypotheses, not the plan), and the **capability anchor near the ceiling**: decide in B11.
- **Contingency** ($238, 4.8%, against the brief's 20%): recalibrate after the gate pilot and Study G's micro-pilot.

## 5. Out of scope (Tier B/C)

F4, F5 for the main study, F6, F9; fault-propagation Study D, beyond the F2 data hooks that exist; M3–M6, M7-R0, M8 and M9; and the external benchmarks deferred by decision 5.

## 6. Progress

| Package | Status | Notes |
|---|---|---|
| Seam | ✅ 4fbb68b, a85799f | `main_study` covers F1/F2/F3/F7; `agent/solvers.arm_solver` dispatch point; S5 = the gate's KG arm (`APE_KG_ARM`); `ape.build` takes seed_base and F8 knobs |
| B1 + B6 | ✅ merged dffa9b3; integration merged 61e4215 (877 passed) | D-036, D-045 |
| B2 | ✅ merged cb3516d | D-034 (M2 dropped on F1) |
| B4 | ✅ statistics core merged cc9dcb5; run-dir glue after B1 | D-029, D-031; power gaps P-1 |
| B7 | ✅ merged efe6158 | D-030 |
| B8 | ✅ merged b45f316 | D-037; overheads priced (D-035) |
| B10 | ✅ statistics core merged f413dd2; run-dir glue after B1 | D-029, D-032; power gaps P-2 |
| B3 | ✅ merged d2f12ab | D-041 |
| B9 | ✅ merged 83172ed | D-040 (incl. a B7 checkpoint fix) |
| Analysis glue | ✅ `analyze_main` f6eeac6, `analyze_g` merged | B1's `analyze(run)` interface |
| B5 | ✅ merged 2d29cb4 | D-044 |
| B11 | ✅ PREREGISTRATION_G.md d04e2d4; tuning grid + probe extension merged | D-042, D-043 |
| B12 | ✅ part 1 merged (smoke checks, cache nonce, runaway guard; D-046); 🔄 independent review of a576a2c (3 reviewers: runner and money; arms and records; statistics and preregs) | full suite 944 passed |

**Follow-ups found during the build** (must be done before any paid run):
✅ - **For B12:** a per-run cache nonce (brief §6.2: provider prefix caches must not carry across runs or arms sharing a monolith prefix) and a runaway wall-clock guard (brief §4.5); the runner writing the `cap multiple` and `pilot σ and power` pre-registration items.
✅ - **Pilot cap-hit gate (D-039):** M7 may outgrow the 8 × B0 token cap; the main runner doubles the cap multiple for all arms when any arm's pilot cap-hit rate exceeds 10% (with the B1 integration).
✅ - **Freeze scope:** the gate's freeze hashes all of `run_plan.yaml`, `models.yaml` and `model_costs.yaml`, so once the gate freezes, any later main/G plan edit would break it. Being changed to per-study resolved slices (with the B1 integration).
✅ - **`search_kb` limit handling:** a token limit tripped inside `search_kb` became a tool error (one extra model call). Being fixed with the B1 integration.
✅ - **Spend misses errored attempts** (D-030): Inspect drops the usage of a sample's errored attempts under `retry_on_error`, so `ape.budget`/`ape.spend` undercount. Fix (after B1 merges, since it touches runner/spend code): Inspect 0.3.273's `on_model_usage` hook fires for every successful generate call, errored attempts included, with eval/run/eval-set IDs; a hook that appends each call's usage and cost to a flushed per-run ledger makes spend complete and kill-safe, with the logs kept as a cross-check.

