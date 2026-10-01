# Program budget (FX-5)

**Date:** 2026-10-01. **Snapshot** of `uv run python -m ape.budget` with priors only (no calibration yet). The numbers come from `config/run_plan.yaml` (what runs) × `config/budget_assumptions.yaml` (per-call priors) × `config/model_costs.yaml` (prices). Decision: D-021.

## Totals for the right-sized plan (USD)

| Study | Phases (conservative $) | Conservative | Expected | FX-5 target |
|---|---|---|---|---|
| Gate | smoke 3 · anchor 9 · builds 48 · tuning 8 · pilot 11 · test 59 · diagnostics 24 · F5 3 · secondaries 37 | **201** | 177 | ≤ 700 |
| Main study | builds 5 · micro-pilot 10 · tuning 29 · pilot 13 · A 105 · B 142 · C 23 · F 786 | **1,114** | 894 | ≤ 1,000 |
| Study G | micro-pilot 11 · tuning 24 · capability anchor 51 · context management 1,737 · topology 1,556 | **3,378** | 2,130 | ≤ 2,700 |
| **Total** | | **4,693** | **3,201** | ≤ 5,000 |
| Contingency | $5,000 − conservative total | 307 | | ~600 |

- **By tier** (conservative / expected): Sol-high 1,972 / 1,412 · Astra-high 1,955 / 1,222 · Luna-high 711 / 535 · Luna-low 44 / 21 · gpt-4o-mini (anchor) 9 / 8 · smoke cap 3.
- **The total fits.** Main study and Study G are over their FX-5 targets by $114 and $678. The gate's underspend covers part of that. The conservative scenario prices every cached token at the full input price, so expected spend is ≈ $3.2K. The orchestrator's budget check (FX-6) stops any phase whose projected cost exceeds what remains.
- **Biggest drivers:** Study G on Astra ($1,955), Study G on Sol ($1,201), and Study F on Sol ($771).
- **D-017's Sol build path** would add **$905**. That needs approval.

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
| Study G view (W = 32K) | CM0 0.6 W, managed 0.4 W, O-state 4K; +5–15% management calls; probes | **Unverified.** Dominates Study G. |
| Multi-agent tokens | M1/M2 2.5×, M7 3×, S9 1.3× a single agent | **Assumed** (brief: 2–3×) |
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
