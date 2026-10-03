# Hypothesis registry

This is the single ID space for every hypothesis in the program. Other documents should cite these IDs; the old IDs are kept in the "Source" column only.

**Status:**
- ✅ the harness produces the data today (offline-verified)
- 🟡 partly built
- ❌ not built
- ⏸ deferred

Families: F3 tool load · F5 relational/temporal · F7 policy compliance (levels 10/100/1000; renderings descriptive / id_only / messy) · F1/F2/F4/F8/F9 (planned).

Last updated 2026-09-30.

## Gate (APG vs LightRAG): `GATE_PREREG.md`

| ID | Statement | Source | Arms | Cells | Primary measure | Test | Status |
|---|---|---|---|---|---|---|---|
| GATE | APG* is non-inferior to LGR* (margin 5 pp), reported per delivery mode for F7 | GATE_PREREG §2 | APG*, LGR*, S3s; diagnostics S1, S5o, S6, S7, LGRo-* | F7-10, F7-1000 (descriptive), F3-5, F3-60; F5 reported separately | Task success (programmatic) | One-sided NI, world-clustered bootstrap; secondary conditions; PC1–PC6 | ✅ harness; ⏳ live runs |

## K: knowledge delivery

| ID | Statement | Source | Arms | Cells | Primary measure | Test | Status |
|---|---|---|---|---|---|---|---|
| K1 | Graph structure beats equally engineered flat retrieval on relational policies and tool scoping | brief H2-struct | APG*, LGR*, S3s | F7 (descriptive, id_only, messy), F3 | Success; error labels (missed_exception, wrong_tool); evidence recall | Paired contrasts per rendering and delivery mode | 🟡 data from the gate and its id_only/messy runs; the report shows the contrasts, but no multiplicity-corrected test across renderings and modes is implemented |
| K2 | KG delivery substitutes for multi-agent specialization (negative delivery × architecture interaction) | brief H2 | S1, S5, M1, M1k | F3, F7 | Success | 2×2 interaction | ❌ needs M1, M1k |
| K3 | Persona content adds ≈ 0 on objective tasks | brief H2b | S5 vs S5-P0 | F3, F7 | Success | TOST ±2 pp | ⏸ **dropped (D-028):** the worlds carry no persona content, so S5-P0 would equal S5; S5-P0 is removed from the run plan |
| K4 | Monolith degrades with KB size; KG stays flat (only when policies are relational) | brief H2c | S1, S5, S3s | F7 levels 10/100/1000 × relational/independent | Success vs log(KB size) | **Descriptive (D-028):** slopes and their CIs are reported; no confirmatory TOST (two levels give a slope CI of about ±4.7 pp per decade against ±2 pp). The relational-vs-independent clause is not tested: the plan has no independent cells | 🟡 generators and arms exist (F7-10/100/1000) |
| K5 | Authored-graph quality survives realistic documents | EXPERIMENT_AUDIT §3 | APG*, LGR* (authored vs oracle) | F7 messy vs descriptive | Success gap and extraction coverage | Paired contrast messy vs clean | ✅ generator; ⏳ dev diagnostic |

## M: mechanism attribution (single-switch contrasts)

| ID | Statement | Source | Arms | Cells | Measure | Test | Status |
|---|---|---|---|---|---|---|---|
| M1 | Context isolation drives multi-agent gains on breadth tasks | brief H1a | M1s vs S9 | F1, F2 | Success | Paired contrast | ❌ |
| M2 | Ensembling beats coordination at matched cost on dependency chains | brief H1b | S8 vs M1 | F1, F2 | Success at matched realized cost | Frontier interpolation | ❌ |
| M3 | Communication adds ≈ 0 at matched cost | brief H1c | M7 vs S8, M9a/b | F1, F2 | Success | TOST ±3 pp | ❌ |
| M4 | Concurrency affects latency only | brief H1d | M1 vs M1s | F1 | Success, wall-clock | **Descriptive (D-028):** the accuracy difference and the latency ratio are reported with CIs; no equivalence claim (n = 50 gives a 90% CI of ±6–10 pp against ±2 pp) | ❌ needs M1, M1s |
| M5 | Role specialization adds ≈ 0 | brief H1e | M2 vs M1k | F3, F7 | Success | TOST ±3 pp | ❌ |

## C, F, A, P: cost meters, faults, auditability, predictability

| ID | Statement | Source | Status | Missing piece |
|---|---|---|---|---|
| C1 | Arm rankings flip across cost meters (tokens, calls, cache-adjusted $, wall-clock) | brief H3 | 🟡 | The meters exist (cost report, compile latency). Multi-agent arms are needed for a meaningful ranking. |
| F1 | Typed shared state reduces fault propagation versus text ledgers | brief H4 | ❌ | Fault hooks; M5/M6 arms |
| A1 | KG single-agent arms are more deterministic and auditable at matched accuracy | brief H5 | 🟡 | D1/D2 come from epochs. Probes, citations and the auditor scanner are missing. |
| P1 | Task features predict the best arm better than family identity | brief H6 | ❌ | Many families × arms |

## G: Study G (context management across model tiers), `CONTEXT_MANAGEMENT_AUDIT.md`

| ID | Statement | Source | Arms | Measure | Status |
|---|---|---|---|---|---|
| G1 | The topology gap shrinks as capability rises | Study G H1 | S1, S1+KG, M1, M2 × Luna/Sol/Astra (± effort) | topology × capability interaction (logit); RER | ❌ needs M arms and F8 |
| G2 | Context-management gains persist across tiers and grow with length | Study G H2 | CM0, CM-prune, CM-sum, CM-native, CM-todo, CM-reset, O-state × tiers | Equivalence on strategy × capability; strategy × log(view tokens); headroom recovered | ❌ needs the ContextPolicy layer, F8, probes |
| G3 | Isolation, not specialization, carries most multi-agent gain; a single agent with context management recovers it more cheaply | Study G H3 (overlaps M1, M5) | S1, M1, M2, S-CM*, S-subiso | Isolation share; recovery at cost ratio | ❌ |

## B: benchmark predictions, `EVAL_DESIGN.md`

| ID | Statement | Source | Status |
|---|---|---|---|
| B1 | Multi-agent gains shrink as the single-agent baseline improves (capability saturation) | EVAL_DESIGN H-M1 | ⏸ Tested within G1 on synthetic families. Public benchmarks are external anchors only. |
| B2 | Most multi-agent gains are recovered by compute-matched best-of-N | EVAL_DESIGN H-M2 | ⏸ Tested within M2 |
| B3 | Error amplification ranks topologies: independent > decentralized > centralized | EVAL_DESIGN H-M3 | ⏸ Tested within F1 |
| B4 | Parallel topologies reduce wall-clock only on breadth tasks | EVAL_DESIGN H-M4 | ⏸ Tested within M4 |
| B5 | Best-of-N and the evaluator raise pass^k; handoff and debate lower it | EVAL_DESIGN H-M5 | ⏸ |

**Platform decision.** Everything runs on the custom Inspect harness (`src/ape`), on OpenAI (D-000). The nine-pattern public-benchmark study in `EVAL_DESIGN.md` is deferred; its vetted benchmarks serve as external anchors (brief F0).
