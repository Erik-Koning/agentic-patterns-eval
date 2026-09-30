# Experiment audit: fairness, data sufficiency, hypothesis coverage

**Date:** 2026-09-30.

**Scope:**
- the implemented APG-vs-LightRAG gate harness (`src/ape`, `GATE_PREREG.md`)
- the planned main study (`ORCHESTRATOR_BRIEF_v2.md`)
- Study G (`CONTEXT_MANAGEMENT_AUDIT.md`)
- the benchmark study (`EVAL_DESIGN.md`)

**Method:**
- a code review of the harness
- offline measurements (fake embeddings, mock models, no spend; scripts in the session scratchpad)
- tracing every hypothesis to the arms, families, logs and tests it needs

## Verdict

**Not yet.** The engineering is sound, but three things stand in the way:
1. **Four blocking fairness problems** in the gate, three of them found by measurement. As designed, the gate would either never reach a verdict, or reach one decided by a rendering choice in the generator rather than by the systems.
2. **Logging gaps** that would make the pre-registered NO-GO diagnosis and the cost comparison impossible.
3. **Most hypotheses have nothing to run yet.** The harness currently answers the gate question and parts of H2-struct and H2c. The other hypotheses (mechanism attribution, KG substitution, fault injection, determinism/auditability, and Study G's tier/topology/context questions) need multi-agent arms, long-horizon generators and context policies that do not exist yet.

The fixes in §1–§2 are small (≈3–4 days). The build-out in §4 is the larger program.

---

## 1. Blocking fairness issues (fix before any pilot)

### B1. The context-parity precondition (PC4) can never pass

**Measured** realized context per compile, with perfect routing for APG:

| Cell | Corpus | APG | LightRAG (default query params) | S3s |
|---|---|---|---|---|
| F7-10 | 863 tokens | ~129 | ~1,120 | 863 (entire KB) |
| F7-1000 | 83,048 tokens | ~116 | ~2,350 | 2,000 |

- **Why:** APG composes one policy leaf plus its brings, by design.
- **Consequence:** PC4 requires the APG-s / LGR* median to fall in [0.8, 1.25]. The measured ratio is ≈0.05–0.1, so every gate run would end PRECONDITION_FAIL.
- **Fix** (update `GATE_PREREG.md` §7):
  1. **Primary comparison:** each system at its dev-tuned configuration ("best foot forward"), with realized context tokens and cost reported. This is the product-relevant comparison.
  2. **Matched-budget comparison as a pre-registered secondary.** Two budgets (≈300 and ≈2,000 tokens):
     - LightRAG and S3s are capped down to the budget.
     - APG is allowed to fill up using secondary matches (`allowMulti`, larger `shortlistK`).
  3. **Replace PC4** with a sanity bound: no arm may exceed 4× the nominal budget.

### B2. The random placebo S7 is mis-sized

- **Problem:** S7 packs `APE_CONTEXT_BUDGET` = 2,000 tokens.
  - In F7-10 and F3-5 that is the **entire knowledge base**, so S7 equals the monolith.
  - Everywhere else it delivers ~15× more text than APG.
- **Consequence:** the GO condition "APG-s > S7" and invariant PC3 would fail for the wrong reason.
- **Fix:** token-match S7 **per cell to APG-s's realized median**, measured on the pilot and frozen in the pre-registration. Cap it at 50% of the corpus so it stays a placebo in small KBs.

### B3. F7-relational measures a generator rendering choice, not relational ability

**Measured** on F7-1000, exception-applies tasks, using the step query after customer lookup (fake lexical embeddings; indicative):

| Arm | Policy recall | Exception recall |
|---|---|---|
| S3s (flat hybrid) | 24/25 | **0/25** |
| LightRAG, oracle custom KG, default params | 14/25 | **0/25** |
| APG, oracle graph, perfect routing | 25/25 | 25/25 |

**Two causes**, both design choices rather than properties of the systems:
1. **ID-only links.** Exceptions are rendered in a separate register that names only the policy ID ("Exception X-424 (amends Policy P-9446) … platinum status …"), with no domain or region words.
2. **Push-only delivery.** No arm can issue its own follow-up retrieval. The step query is the task prompt plus tool observations, and the policy ID never appears in tool observations.

A retriever that doesn't pre-link IDs therefore **cannot** get the exception. Half of F7-relational tasks (exception applies) are decided by this. The gate pools F7 with F3, so an APG "GO" could be manufactured here.

**Fix:**
- **Exception-rendering knob:** `id_only` vs `descriptive` (the exception also restates domain, region and band). Pre-register both; the gate cell uses `descriptive`, and `id_only` is reported as the "explicit cross-reference" test.
- **Pull-mode delivery:** a `search_kb(query)` tool backed by each arm's own retriever (APG route+compose, LightRAG query, flat hybrid), so the agent can follow a reference it has read. Pre-register push and pull as co-primary delivery modes; real deployments use both.
- **Reporting:** report `exception_applies` tasks separately in every table.

### B4. The kg role silently falls back to the agent model

`get_model(role="kg")` returns the default (agent) model when `--model-role kg=…` is omitted (confirmed in `inspect_ai/model/_model.py`: `required=False` by default). Routing and keyword calls would then run on the wrong model without any error, changing their cost and possibly their accuracy.

**Fix:** use `required=True` in `apg/arm.py` and `lgr/adapter.py`, and record the kg model name in every compile record.

### B5. The gate is underpowered at the planned size

Already raised as O-5: 288 tasks gives ≈0.52 power at true Δ = 0; 12 worlds per cell (576 tasks) gives ≈0.79. Decide this before the pilot, since it sets the number of worlds to build.

### B6. Asymmetric configuration selection

LGR* is the best of LightRAG's per-query and per-step modes on dev, but APG is fixed to per-step (APG-s).

**Fix:** define APG* symmetrically (the better of APG-q and APG-s on dev, under the same tuning budget) and compare APG* vs LGR*.

---

## 2. Data gaps (the logs would not support the pre-registered analyses)

| Gap | Blocks | Fix |
|---|---|---|
| The tools actually bound at each step are not logged (only the arm's allowlist is) | NO-GO taxonomy "tool-exposure miss"; F3 analysis | Log `exposed_tools` per step in `compile_log` |
| APG route logs only the shortlist *size*, not its IDs, and not whether the gold node was in it | NO-GO taxonomy "shortlist miss vs classify miss" | Log shortlist IDs. For authored graphs, map gold facts → leaves via `sourceChunkIds` and log `gold_in_shortlist` / `gold_routed` |
| Compile latency (route, classify, retrieval) is not recorded | Latency claims; attributing time to KG work vs the model | Time each compile; store `compile_ms` in the record |
| Query-time embedding calls go to the ledger without arm or world context | "Per-query cost including embeddings" (pre-registered) | Pass `{arm, world, sample}` context to `embed()`; count query tokens in the compile record |
| `load_results` treats a missing `total_cost` as 0 | Cost ratio silently 0, so GO_WITH_COST_FLAG can never trigger | Fail loudly when cost is missing; require `--model-cost-config` |
| No tuning-log writer (`cache/tuning_log.jsonl` is referenced but no code writes it) | PC6 (tuning parity) cannot be verified | A small tuning runner that sweeps configs on dev and appends every config and score |
| No per-arm cost or build-cost report from logs + ledger | The decision's `cost_ratio` is entered by hand | `analysis/cost.py`: per-query $ per arm, build $ per world, amortized total |
| Evidence recall is unioned over steps, so arms that cause more turns get more chances | Evidence comparisons | Also log per-step recall and first-step recall |

---

## 3. Construct validity (not bugs; these limit what the results mean)

1. **APG is tested as a retriever, not as a prompt program.** Our graphs use only the knowledge slot plus brings and allowlists. APG's differentiators are unused: persona/task/constraint slot composition, constraints that are never truncated, walker flows and escalation. A GO or NO-GO describes "APG as a KG retriever". If the product claim is compiled prompts, add a family where policies map naturally onto the constraints slot and personas vary per task.
2. **Authoring is too easy.** F7 policies carry explicit IDs, so the authored graph is close to the oracle. Real documents reference each other loosely. Add a `messy` rendering (paraphrased references, no IDs, split paragraphs) so extraction quality actually varies.
3. **The gate result is specific to one model tier.** It runs one agent model (GPT-6 Luna, high), and KG benefits may shrink for stronger agents. Treat the gate verdict as holding at the Luna tier. Study G's S1+KG arm across tiers is what generalizes it.
4. **No persona content exists in any world,** so brief H2b (persona ≈ 0) cannot be tested with the current generators.
5. **Small-KB cells are near-trivial.** In F7-10 and F3-5 every arm except APG sees essentially the whole KB. Those cells test "does routing lose information", not KG quality. That is fine, but label it in reports.

---

## 4. Hypothesis coverage: will we get the data?

Status key: ✅ built and producing data · 🟡 partially built · ❌ not built.

| Hypothesis (source) | Needs | Status | Data we would get today |
|---|---|---|---|
| **Gate: APG non-inferior to LightRAG** (`GATE_PREREG.md`) | APG*, LGR*, S3s, S5o, S6, S7; F7/F3 (+F5); NI stats; PC1 anchor | 🟡 built, but B1–B6 and §2 must be fixed | A verdict that is invalid until B1–B3 are fixed |
| **H2-struct:** graph beats flat retrieval (brief) | APG-s vs S3s on F7-relational / F3 | 🟡 same harness | Yes, after B3 (both renderings, push and pull) |
| **H2c:** KB scaling (brief) | F7 levels 10/100/1000 × relational/independent; S1, S5, S3s | 🟡 generators and arms exist; the gate runs only 10 and 1000 | Yes, if the 100 level and the independent variant are added to the run plan |
| H1a–e: mechanism attribution (brief) | S8, S9, M1s, M1, M7 (± M9); F1/F2 generators | ❌ no multi-agent arms, no best-of-N, no F1/F2 | None |
| H2: KG substitution, 2×2 (brief) | S1, S5, M1, M1k | ❌ no M arms | None |
| H2b: persona ≈ 0 (brief) | Persona content in worlds; S5-P0 | ❌ | None |
| H3: meter rank flips (brief) | Four cost meters; multiple arms | 🟡 meters exist if cost config and latency logging are fixed | Only across single-agent delivery arms |
| H4: fault propagation (brief) | Fault hooks; M5/M6 | ❌ | None |
| H5: determinism and auditability (brief) | Epochs (D1/D2), probes, citations (A2), auditor (A4) | 🟡 epochs only | D1/D2 for single-agent arms |
| H6: predictability (brief) | Many families × arms | ❌ | None |
| G-H1: topology convergence (Study G) | 3 tiers; S1, S1+KG, M1, M2; capability anchor | ❌ no M arms; tiers chosen (Luna/Sol/Astra) | None |
| G-H2: context persistence (Study G) | ContextPolicy layer, F8, O-state, probes, tiers | ❌ design only | None |
| G-H3: isolation vs specialization (Study G) | M1, M2, S-CM*, S-subiso | ❌ | None |
| Benchmark study (`EVAL_DESIGN.md`) | 9 pattern implementations on public benchmarks | ❌ not integrated; priced for Claude | None |

**Program-level issue.** There are three overlapping designs with clashing hypothesis IDs (EVAL_DESIGN H-M1…, brief H1–H6, Study G G-H1…) and different platform assumptions (MAF vs our custom loop; Claude vs OpenAI).

**Recommendation:**
- **One hypothesis registry** (`HYPOTHESES.md`): every hypothesis, the arms, families, logs and tests that answer it, and its status. This is the brief's traceability matrix, produced now.
- **One platform:** the custom Inspect harness.
- **EVAL_DESIGN's role:** repurpose its public benchmarks as external anchors for the synthetic families (brief F0), or explicitly defer the nine-pattern benchmark study.

---

## 5. What is already solid

- **Worlds:** deterministic, seeded, split-disjoint, with fact-level provenance and reference solvers. Every gold answer is verified against the spec.
- **Metering and replay:** one metered kg call per compile for both KGs; record/replay routing; content-hash pins for graphs and indices; stale-index refusal.
- **Harness hygiene:** shared agent loop, step query and placement rule; per-world workspace isolation; concurrency checked.
- **Statistics:** world-clustered NI statistics with world-level placebo tests, a minimum-worlds guard, and a power simulation calibrated against the analytic formula.
- **Process:** PC1 anchor with the official metric; pre-registration draft; readiness audit; 92 offline tests.

---

## 6. Prioritized fixes

| Priority | Fix | Effort |
|---|---|---|
| P0 | B4 `required=True` for kg; kg model name in records | 0.5 h |
| P0 | §2 logging: exposed tools, shortlist IDs, gold-in-shortlist, compile_ms, embedding context, cost guard | 0.5 day |
| P0 | B2 S7 sized per cell to APG-s realized tokens (pilot-calibrated), capped at 50% of corpus | 2 h |
| P0 | B3 exception-rendering knob plus `search_kb` pull-mode tool for APG, LightRAG and flat | 1–1.5 days |
| P0 | B1 replace PC4 with best-config primary plus a matched-budget secondary (APG fill via multi-match; LightRAG and S3s capped) | 0.5 day plus a pre-registration edit |
| P0 | B6 APG* selection; B5 gate size decision (user) | 1 h plus a decision |
| P1 | Tuning runner and log (PC6); `analysis/cost.py` | 1 day |
| P1 | `messy` rendering knob; per-step evidence metrics | 1 day |
| P1 | `HYPOTHESES.md` registry (single ID space, status per hypothesis) | 0.5 day |
| P2 | Main-study build: S8/S9, M1s/M1/M1k/M2, M7; F1/F2 generators; fault hooks; persona content | 2–3 weeks |
| P2 | Study G build (`CONTEXT_MANAGEMENT_AUDIT.md` §10) | 2–3 weeks |
