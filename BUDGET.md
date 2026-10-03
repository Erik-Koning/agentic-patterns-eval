# Program budget (FX-5)

**Date:** 2026-10-03 (D-034). **Snapshot** of `uv run python -m ape.budget` with priors only (no calibration yet). The numbers come from `config/run_plan.yaml` (what runs) × `config/budget_assumptions.yaml` (per-call priors) × `config/model_costs.yaml` (prices). Decision: D-021.

## Totals for the right-sized plan (USD)

| Study | Phases (conservative $) | Conservative | Expected | Allocation (D-035) |
|---|---|---|---|---|
| Gate | smoke 3 · anchor 11 · builds 57 · tuning 8 · pilot 13 · test 59 · diagnostics 24 · F5 3 · secondaries 37 | **214** | 188 | ≤ 300 |
| Main study | builds 5 · micro-pilot 10 · tuning 35 · pilot 15 · A 141 · B 136 · C 27 · F 642 | **1,012** | 776 | ≤ 1,100 |
| Study G | micro-pilot 11 · tuning 22 · capability anchor 51 · context management 1,764 · topology 1,638 | **3,486** | 2,196 | ≤ 3,600 |
| **Total** | | **4,712** | **3,161** | ≤ 5,000 |
| Contingency | $5,000 − conservative total | 288 | | ≥ 280 |

- **By tier** (conservative / expected): Astra-high 1,966 / 1,230 · Sol-high 1,846 / 1,280 · Luna-high 797 / 593 · Luna-low 89 / 45 · gpt-4o-mini (anchor) 11 / 10 · smoke cap 3.
- **Pilot σ cell (D-026, 2026-10-02):** +$8 conservative. The pilot builds 8 worlds per cell (was 4; +$6.19) and runs APG* and LGR* on the 4 new ones (`gate.pilot.sigma`, +$2.17), so σ is estimated from 8 worlds per cell.
- **PC1 fidelity (D-025, 2026-10-02):** the anchor grew about $5: answer contexts mapped to LightRAG 1.2.5's caps (26.5K tokens) and a second scorer's judge calls; its bge embeddings run locally ($0).
- **The total fits.** Each study is within its allocation (D-035: allocations reset to the right-sized plan plus headroom; the FX-5 targets were 700 / 1,000 / 2,700, and Study G's topology cells would have been stopped by the guard). The gate's underspend covers part of that. The conservative scenario prices every cached token at the full input price, so expected spend is ≈ $3.2K. The orchestrator's budget check (FX-6) stops any phase whose projected cost exceeds what remains.
- **Biggest drivers:** Study G on Astra ($1,966), Study G on Sol ($1,220), and Study F on Sol ($626).
- **D-017's Sol build path** would add **$1,074** (was $957; the 16 extra pilot worlds of D-026 are built by the same builder). That needs approval.
- **F1/F2 measured (2026-10-01).** The registry generators (`worlds/gen_registry.py`) replaced the assumed F1/F2 sizes: corpus 3,337 tokens (was 2,000) and 20 chunks per world at every level (`readiness/measure_registry.py`). Main study +$50.
- **Not yet priced: bulky F1/F2 tool results.** Each supplier record is ~400 tokens and stays in the history. The measured per-call history (F1-32: 1,258 tokens per prior call, against the 250 prior) is in `readiness/measure_registry.py`. Priced through `history_per_prior_call_by_cell` (commented out in `config/budget_assumptions.yaml`), it adds about $320, almost all of it Study F on Sol over F1-32, for a total of $5,062. That needs a decision; cut C5 (Study F Sol 100 → 60 tasks) would cover it.

## Plan changes and cuts

**Review change (2026-10-01).** Study F's Sol replication now runs its policy cell on **F7-100** at the full 100 tasks, instead of F7-1000 at 60 tasks (cut C5, now superseded).
- Why: Study F replicates the tier comparison. KB scaling is K4's question, measured on Luna. On F7-1000 the 90K-token monolith, times Sol's price and repeated per worker for M1, was the largest main-study cost.
- Effect: the Sol replication keeps its power (MDE ~13–15 pp) and costs less. With the saving, cut C4 is no longer needed and is reversed: Astra topology runs at 5 sessions.

Cuts are applied in order until the conservative total fits. The current plan needs C1–C3. C4–C6 are not applied. Gate primary sizes and epochs and the S7 placebo are never cut.

| Cut | Saves | Quality impact |
|---|---|---|
| C1: drop the Sol-low capability point | $471 | 4 capability points remain (Luna low/high, Sol high, Astra high). Capability-slope and TOST interactions lose power; the effort contrast exists at Luna only. |
| C2: Study G Sol CM sessions 8 → 6 | $202 | Sol Gap_T, R_x and slope CIs ~1.15× wider |
| C3: Astra CM sessions 5 → 4, N 30 → 24 | $552 | Astra CIs ~1.12–1.25× wider. Four sessions per arm is close to descriptive. The degradation curve covers positions 1–24, and the item-30 probe is lost. |
| C4: Astra topology sessions 5 → 4 | ($188) | **not applied** (reversed after the review change) |
| C5: Study F Sol tasks 100 → 60 | ($308) | **not applied** (superseded by the F7-100 change) |

After calibration, if the total drops, revert C3 first, then C2.

## Key assumptions (`config/budget_assumptions.yaml`)

| Assumption | Value | Status |
|---|---|---|
| Output tokens per call, including reasoning | high 1,500 · medium 800 · low 400 | **Unverified.** Dominates gate and main-study cost. |
| Agent calls per sample | F7 push 4 / pull 5, F3 5, F5 2, F8 6 per item; F1/F2 4–12 | **F1, F2 and F8 generators are not built** |
| Input per call | base 800 + context (APG/S5o/S6/S7 150, LightRAG 2,300, S3s 2,000, S1 = corpus: F7-10 0.9K, F7-100 9.3K, F7-1000 90.5K (measured, descriptive rendering), F3-60 9.5K) + 250 per earlier turn | Context sizes measured offline; history assumed |
| Study G view (W = 32K) | CM0 0.6 W, managed 0.4 W, O-state 4K; management calls +8% (CM-sum), +20% (CM-todo), +25% (CM-reset, S-CM*), +5% (CM-native); probes | Views **unverified** (dominate Study G). Management overheads for todo/reset/S-CM* **measured** with the mocks (B8), incl. `todo_write` traffic; T_abs-triggered ones (sum, native) are lower bounds there, so their priors stay |
| Multi-agent tokens | M1/M2 2.5×, M7 5×, S9 1.3×, S8k3 3× a single agent | M7 **measured** with the gold mock (B2: 3.4–5.8× S1 tokens; was 3×); the others assumed and consistent with the mock (S9 1.1–1.35×, M1 0.9–2.7×, M2 ≈ 2.75–3× S5) |
| Cached input | conservative: full input price; expected: 20–85% of input cached, at 0.1× the input price | **Unverified until E5** |
| Builds and kg calls | APG 600 in / 1,200 out per chunk; LightRAG 2.2 calls × 3,000 / 1,000; classify 600 / 350; keywords 1,000 / 350 | Given |

## Re-running

```bash
uv run python -m ape.budget                        # conservative check; exits 1 above $5,000
uv run python -m ape.budget --scenario expected --detail
uv run python -m ape.budget --uncut                # the plan before the cuts
uv run python -m ape.budget calibrate LOGS...      # writes config/budget_calibration_measured.yaml
uv run python -m ape.budget remaining --budget 5000 LOGS... --ledger cache/ledger.jsonl
```

After the gate pilot (and Study G's micro-pilot), run `calibrate` on its logs. The measured calls per sample, tokens per call and cache share then replace the priors wherever (arm, model, effort, cell, delivery) match, and lines priced that way are marked `m`.

If the total drops, revert cuts in reverse order (C3 first). To revert one, edit its cells and its `applied:` flag; `load_plan` checks that the two agree.

FX-6 calls `projected_cost()` and `remaining()` before each phase. It refuses the phase if the projection is larger than what is left.
- **D-028 (2026-10-03):** +$6 net. Dropping S5-P0 (no persona content) saves $6.5; the Luna F7-100 tier cell (+$12.25) and the F7-100 and F1-2 KG builds (+$0.6) make the tier contrast and the micro-pilot's S5 runnable. Study G's larger tool files for short sessions don't change the projection (session views are priced at 0.6 W).
- **D-033 / D-034 (2026-10-03):** −$84 net. Study G's topology gets a Luna-low cell and 16 Luna sessions per point (+$73, for G-H3's power); M7 is priced at its measured 5× a single agent (was 3×; +$33); M2 is dropped from F1-32 in Study F, where specialization is undefined (−$190). Contingency $238 → $322.
- **B8 overheads (2026-10-03):** +$32. CM-todo, CM-reset and S-CM* are priced at their measured management overhead including `todo_write` traffic (0.20, 0.25, 0.25 of agent calls; were 0.15, 0.05, 0.15). Contingency $322 → $290.
- **D-041 (2026-10-03):** −$0.8. S3s is no longer re-tuned in the main study (`main.tune.b`): S3s and S5 inherit the gate's selections.
- **D-043 (2026-10-03):** +$2. CM-native replaces CM-prune at Sol (+$6); the Study G tune prices CM-sum and CM-todo at their 2 real candidates (−$4).
