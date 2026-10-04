# Independent review of the main-study and Study G build (BUILD_PLAN B12)

**Date:** 2026-10-03. **Reviewed commit:** a576a2c (full suite: 944 passed, 2 skipped).

## Method

Three reviewers who wrote none of the code, each in its own read-only copy, each proving its findings with offline reproductions (scripts and outputs in the session scratchpad: `review-runner/`, `review-arms/`, `review-stats/`):
1. **Study runner, money and integrity:** `run_study`, `run_gate` changes, `runner`, spend ledger, budget guard, freeze slices, seeds, smoke requirement.
2. **Arms, agent loops, records and scoring:** multi-agent and session arms driven through the real loops with scripted real-model misbehaviour.
3. **Statistics and pre-registrations:** independent Monte Carlo (own generative models, 10,000 replicates per scenario) through the real analysis functions; every pre-registration claim checked against the code.

**Severity:** A = wrong verdict or results, a crash of a live run, or a pre-registration violation; B = biases results or wastes significant money; C = minor.

**Bottom line.** Nothing found sinks the design, and the happy path computes and accounts correctly. Confirmed sound: no study can touch another's test worlds; each tuned arm runs exactly its own selection; the gate's behaviour is unchanged; the loops survive realistic misbehaviour with exact accounting; the generated pre-registration tables equal the code; the confirmatory tests hold α under exchangeable, heavy-tailed, correlated-epoch and unequal-cluster conditions. Four A-level problems block a live run or a freeze; all are small to medium fixes.

## Findings and fix assignments

| # | Sev | Area | Finding | Fix package |
|---|---|---|---|---|
| R-A1 | **A** | runner | The main freeze hashes the gate's `decision.json` byte for byte (it carries `generated_at`); a routine gate re-analysis (triggered because the gate's analyze lists the shared build ledger, which main's builds append to) rewrites it and permanently blocks a frozen main run. | F-runner |
| R-A2 | **A** | runner | The analyze phase never checks the freeze: a report can come from analysis code or config changed after the freeze, unrecorded (the gate has the same gap). | F-runner |
| S-1 | **A** | stats | A capped F3 sample still scores as a success and votes in the S8 frontier (the F3 scorer reads end state; Inspect scores after a limit), contradicting PREREGISTRATION_MAIN (cap hits are failures, cast no vote). Affects K1, K2, K2-NI, M5 and the F3 frontier. | F-stats-main |
| S-2 | **A** | stats | M3 and M2's frontier test ignore the uncertainty of the matched cost: when failed runs cost more than successful ones (κ = 1–3; plausible with caps at 8 × B0), M3's TOST rejects at 0.073–0.089 against 0.05 and M2.frontier at 0.027–0.031 against 0.025. | F-stats-main |
| A-1 | B (near A) | arms | M1k and M2 each pick their own prompt variant, so the single-switch steps M1 → M1k (DEL) and M1k → M2 (SPEC) can differ in prompt text too; the tune cannot tell variants apart (SE ≈ 11 pp). | F-arms + F-runner (M1k, M2 inherit M1's selection) |
| A-2 | B | arms | Provider errors inside workers, council members and S8k3 attempts are swallowed: a failure or lost vote where S1 would retry; biases every multi-agent arm against S1 and hides harness errors. | F-arms |
| A-3 | B | arms | The todo arms' instructions ("mark completed when done") assume a turn the loop never gives after the answer, so the list lags a case; handicaps CM-todo, CM-reset and S-CM*'s todo stacks. | F-session |
| R-B1 | B | runner | The smoke requirement can be bypassed after a code change by running paid phases one at a time; a smoke override persists across code changes. | F-runner |
| R-B2 | B | runner | Pre-freeze phases carry no code identity: a resumed micro-pilot, tune or pilot mixes code versions. | F-runner |
| R-B3 | B | runner | Allocations, `total_usd` and `concurrency` are frozen as if they were design, so a budget or rate-limit adjustment after the freeze blocks the test; the freeze never checks the test is affordable. | F-runner |
| R-B4 | B | runner | `ape.budget` prices `n_tasks` as written; the runner runs whole worlds of 12 (108, 60, 24): main projects ~$1,128–1,164 against its $1,100 allocation, and the Sol tier replication runs last. | F-runner (+ allocation) |
| R-B5 | B | runner | Wall-clock and cost-limit hits never block the freeze; Study G's micro-pilot computes no session error/limit/overflow rates, and its topology arms (the largest cost) are never piloted. | F-runner (+ a topology pilot cell) |
| S-3 | B | stats | The sign-flip holds α only under symmetric world contributions (catastrophic-world scenarios: K1 0.075, M5 0.075 at nominal 0.05/0.025); the pre-registration states no such condition. | F-stats-main (text, sensitivities) |
| S-4 | B | stats | Study G's G-H3 tests run at 0.026–0.040 against 0.025 under D-038's df rule (worse with persistent world × arm effects or a budget stop leaving 3 Sol sessions, which is not INCOMPLETE). | F-stats-g |
| S-5 | B | stats | The D-045 cap gate decides on point estimates over 44 arm-cells of 24 tasks: at true rates of 2–5% it escalates by chance with P = 0.42–0.996 and refuses the freeze with P = 0.39–0.99; the S8k3 projection overstates its rate. | F-runner |
| S-6 | B | stats | Pooled TOST/NI claims are equal-weight averages over four cells (ceiling cells dilute them) but are worded as overall facts. | F-stats-main (text, per-cell estimates) |
| A-4…A-9, R-C1…C6, S-7…S-12 | C | all | Answer recorded after a limit; session records miss the limit-tripping call; time/working-limited sessions keep their checkpoint; CM-prune counts no-op compactions; M7 proposal race; CM-native compaction usage bypasses the ledger; build spend attributed to the first label; stale task identities in the runner index; tune's flat cost limit; the gate keeps a shell cache nonce; unfrozen cost-limit inputs and CM-native support record; power inputs vs the pre-registration; M3's null placement; `g_report` section isolation and NaN-cost labels; document/code inconsistencies; K2-NI cost wording; G-H2a Astra type I. | per area |

## Status

Fix round dispatched 2026-10-03 (packages F-runner, F-arms, F-session, F-stats-main, F-stats-g). This section is updated when they merge.
