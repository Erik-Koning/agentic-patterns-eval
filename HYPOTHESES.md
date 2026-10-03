# Hypothesis registry

This is the single ID space for every hypothesis in the program. Other documents should cite these IDs; the old IDs are kept in the "Source" column only.

**Status:**
- ✅ the harness produces the data today (offline-verified)
- 🟡 partly built
- ❌ not built
- ⏸ deferred

Families: F3 tool load · F5 relational/temporal · F7 policy compliance (levels 10/100/1000; renderings descriptive / id_only / messy) · F1/F2/F4/F8/F9 (planned).

Last updated 2026-10-03 (D-033, D-034; the main-study rows K1, K4, M1–M5, C1 and A1 match `analysis/main_hypotheses.py` and PREREGISTRATION_MAIN.md, B5).

## Gate (APG vs LightRAG): `GATE_PREREG.md`

| ID | Statement | Source | Arms | Cells | Primary measure | Test | Status |
|---|---|---|---|---|---|---|---|
| GATE | APG* is non-inferior to LGR* (margin 5 pp), reported per delivery mode for F7 | GATE_PREREG §2 | APG*, LGR*, S3s; diagnostics S1, S5o, S6, S7, LGRo-* | F7-10, F7-1000 (descriptive), F3-5, F3-60; F5 reported separately | Task success (programmatic) | One-sided NI, world-clustered bootstrap; secondary conditions; PC1–PC6 | ✅ harness; ⏳ live runs |

## K: knowledge delivery

| ID | Statement | Source | Arms | Cells | Primary measure | Test | Status |
|---|---|---|---|---|---|---|---|
| K1 | Graph structure beats equally engineered flat retrieval on relational policies and tool scoping | brief H2-struct | S5 (the gate-selected, extracted KG arm) vs S3s | F7-1000, F3-60 | Success | One-sided superiority per cell, Holm of 2 at 0.025 (planned MDE ≈ 15 pp). The gate's id_only and messy contrasts stay descriptive context | 🟡 built (B2, B4); ⏳ runs |
| K2 | KG delivery substitutes for multi-agent specialization (negative delivery × architecture interaction) | brief H2 | S1, S5, M1, M1k | F3, F7 | Success | 2×2 interaction, one-sided, Study B's four cells pooled | 🟡 arms and analysis built (B2, B4); ⏳ runs |
| K2-NI | Operationally, the KG single agent is non-inferior to the specialist team at half its cost | brief H2 | S5, M2 | F3, F7 | Success; realized cost ratio | **D-033:** one NI test pooled over Study B's four cells at **5 pp**, its own Holm family, plus S5/M2 cost ≤ 0.5 (upper 95% ≤ 0.6) | 🟡 built; ⏳ runs |
| K3 | Persona content adds ≈ 0 on objective tasks | brief H2b | S5 vs S5-P0 | F3, F7 | Success | TOST ±2 pp | ⏸ **dropped (D-028):** the worlds carry no persona content, so S5-P0 would equal S5; S5-P0 is removed from the run plan |
| K4 | Monolith degrades with KB size; KG stays flat (only when policies are relational) | brief H2c | S1, S5, S3s (M1, M2 also reported) | F7 levels 10/100/1000, relational only (S3s: 10 and 1000) | Success vs log(KB size) | **Descriptive (D-028):** slopes and their CIs are reported; no confirmatory TOST (two levels give a slope CI of about ±4.7 pp per decade against ±2 pp). The relational-vs-independent clause is not tested: the plan has no independent cells | 🟡 generators and arms exist (F7-10/100/1000) |
| K5 | Authored-graph quality survives realistic documents | EXPERIMENT_AUDIT §3 | APG*, LGR* (authored vs oracle) | F7 messy vs descriptive | Success gap and extraction coverage | Paired contrast messy vs clean | ✅ generator; ⏳ dev diagnostic |

## M: mechanism attribution (single-switch contrasts)

| ID | Statement | Source | Arms | Cells | Measure | Test | Status |
|---|---|---|---|---|---|---|---|
| M1 | Context isolation drives multi-agent gains on breadth tasks | brief H1a | M1 vs S9 (M1 stands in for M1s; M1s − S9 reported) | F1-32; F2-10 descriptive | Success | One-sided superiority on F1-32 (planned MDE ≈ 15 pp); the F2 clause has no margin, so it is descriptive | 🟡 built; ⏳ runs |
| M2 | Coordination beats compute-matched ensembling on breadth tasks (F1-high); ensembling matches it on dependency chains (F2) | brief H1b | M1 vs S8 at M1's realized cost (gate: M1 vs S1) | F1-32; F2-2, F2-10 descriptive | Success at matched realized cost | Frontier interpolation, serial gatekeeping after M1 > S1 (MDE ≈ 15 pp); the F2 clause has no margin, so it is descriptive | 🟡 built; ⏳ runs |
| M3 | Communication adds ≈ 0 at matched cost | brief H1c | M7 vs S8 (M9a/b Tier B) | F1-2, F1-32, F2-2, F2-10 pooled | Success | TOST **±6 pp** (D-033; ±3 pp had power 0.13) | 🟡 built; ⏳ runs |
| M4 | Concurrency affects latency only | brief H1d | M1 vs M1s | F1 | Success, wall-clock | **Descriptive (D-028):** the accuracy difference and the latency ratio are reported with CIs; no equivalence claim (n = 50 gives a 90% CI of ±6–10 pp against ±2 pp) | 🟡 built; ⏳ runs |
| M5 | Role specialization adds ≈ 0 | brief H1e | M2 vs M1k | F3-5, F3-60, F7-10, F7-1000 pooled | Success | TOST **±6 pp** (D-033; ±3 pp had power 0.05) | 🟡 built; ⏳ runs |
| T1 | The coordination payoff (M1 − S8 at matched cost) shrinks from Luna to Sol | brief H6 (tier clause) | M1, S8 on Luna and Sol | F1-32, F7-100 | Success at matched cost | **Descriptive (D-033):** estimate with interval (power 0.44–0.54 at 15 pp) | 🟡 built; ⏳ runs |

## C, F, A, P: cost meters, faults, auditability, predictability

| ID | Statement | Source | Status | Missing piece |
|---|---|---|---|---|
| C1 | Arm rankings flip across cost meters (tokens, cache-adjusted $, wall-clock; calls reported, not ranked) | brief H3 | 🟡 built (B4); ⏳ runs | **Descriptive (D-031):** the τ < 0.8 rule on point estimates, with bootstrap support. No flip directions are pre-registered, so there is no test. |
| F1 | Typed shared state reduces fault propagation versus text ledgers | brief H4 | ❌ | Fault hooks; M5/M6 arms |
| A1 | KG single-agent arms are more deterministic and auditable at matched accuracy | brief H5 | 🟡 | **Descriptive in the main study:** pass^k, D1, D2 and D6 on the Study C subset. D5, the probes, citations and the auditor scanner (A2–A8) are missing. |
| P1 | Task features predict the best arm better than family identity | brief H6 | ❌ | Many families × arms |

## G: Study G (context management across model tiers), `CONTEXT_MANAGEMENT_AUDIT.md`

| ID | Statement | Source | Arms | Measure | Status |
|---|---|---|---|---|---|
| G1 | The topology gap shrinks as capability rises | Study G H1 | S-CM* (reference), M1, M2 at Luna low/high and Sol; Astra runs S1 and M2 only; S1+KG and S-subiso are Tier B (D-043) | **Descriptive (D-033):** slope of the M2 − S-CM* logit gap on measured capability, with interval; S1 on pre-overflow items as sensitivity; S1 itself descriptive only (power 0.12; no affordable design reaches 0.8) | 🟡 analysis built; topology in sessions is B9 |
| G2 | Context-management gains persist across tiers and grow with length | Study G H2 | CM0, CM-sum, CM-todo, O-state at every capability point; CM-prune and CM-reset at Luna-high; CM-native at Luna-high and Sol (D-043, if the probe confirms Sol support) | Confirmatory: Gap_T > 0 at every capability point (G-H2a). **Descriptive (D-033):** headroom recovered R_x across capability (G-H2b; TOST power 0.01), degradation slopes | 🟡 arms and analysis built (B7, B8, B10); ⏳ runs |
| G3 | Isolation, not specialization, carries most multi-agent gain; a single agent with context management recovers it more cheaply | Study G H3 (overlaps M1, M5) | S1, M1, M2, S-CM* (S-subiso Tier B) | Isolation share ≥ 0.5 (G-H3a); S-CM* recovers ≥ 80% of M2 − S1 at ≤ 60% of its cost per solved item (G-H3b); D-033 adds a Luna-low point and 16 Luna sessions | 🟡 analysis built; topology in sessions is B9 |

## B: benchmark predictions, `EVAL_DESIGN.md`

| ID | Statement | Source | Status |
|---|---|---|---|
| B1 | Multi-agent gains shrink as the single-agent baseline improves (capability saturation) | EVAL_DESIGN H-M1 | ⏸ Tested within G1 on synthetic families. Public benchmarks are external anchors only. |
| B2 | Most multi-agent gains are recovered by compute-matched best-of-N | EVAL_DESIGN H-M2 | ⏸ Tested within M2 |
| B3 | Error amplification ranks topologies: independent > decentralized > centralized | EVAL_DESIGN H-M3 | ⏸ Tested within F1 |
| B4 | Parallel topologies reduce wall-clock only on breadth tasks | EVAL_DESIGN H-M4 | ⏸ Tested within M4 |
| B5 | Best-of-N and the evaluator raise pass^k; handoff and debate lower it | EVAL_DESIGN H-M5 | ⏸ |

**Platform decision.** Everything runs on the custom Inspect harness (`src/ape`), on OpenAI (D-000). The nine-pattern public-benchmark study in `EVAL_DESIGN.md` is deferred; its vetted benchmarks serve as external anchors (brief F0).
