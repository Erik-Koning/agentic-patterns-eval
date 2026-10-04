# Decisions

Each entry records the decision, why it was made, and who made it.

## Made

**D-000 (2026-09-29, user).** Comparator KG: **LightRAG**. Provider: **OpenAI only**, for chat and embeddings. Suitability result: **go/no-go gate**.
- GO: APG becomes the KG arm, with LightRAG as a control.
- NO-GO: one APG fix cycle, otherwise LightRAG becomes the KG arm.

**D-001 (2026-09-30).** The project runs on **Python 3.14**, not the 3.12 in the plan.
- Why: `apg_core` fails to import on 3.12 (gap G0), even though it declares `>=3.11`. Everything else installs and imports on 3.14.
- Alternative: patch APG upstream (`from __future__ import annotations` in `connectors.py`). Declined for now because it changes the user's repo and the pin.

**D-002 (2026-09-30).** APG is **pinned by content tree hash** (`PROVENANCE.md`) until the owner commits and tags `apg-eval-baseline`. It is installed non-editable.

**D-003 (2026-09-30).** LightRAG ingestion: each shared chunk is inserted as its own document, with `ids=[chunk_id]` and `file_paths=[chunk_id]`.
- Why: LightRAG then never re-chunks, both KGs see identical chunk boundaries, and `source_id`/`file_path` map straight to our chunk IDs, and from there to world-spec fact IDs.

**D-004 (2026-09-30).** LightRAG `enable_rerank=False`.
- Why: the rerank step needs a reranker model, and the OpenAI-only decision (D-000) rules out Cohere/Jina rerankers. APG has no rerank stage either.
- Revisit if the PC1 anchor reproduction fails, since the published numbers may assume reranking.

**D-005 (2026-09-30).** Context placement is part of the delivery definition.
- **Per-query arms** (S1, S6, S7, APG-q, LGR-q) put their context in the system prompt, compiled once, so it is part of the cached prefix.
- **Per-step arms** (S3s, APG-s, S5o, LGR-s) append a fresh context message after the history on every step. That message is replaced, never accumulated, so the history prefix stays cacheable.
- Why: putting a monolith at the end of every step would re-send ~83k uncached tokens per turn (unrealistic); putting per-step context in the system prompt would break prompt caching.
- Consequence: frequency and position are confounded between -q and -s variants, which is inherent to per-step delivery.

**D-006 (2026-09-30).** Evidence is scored at world-spec fact granularity.
- Mapping per system: chunks and flat retrieval map chunk → facts. APG oracle nodes carry `props.factIds`. Authored APG leaves and LightRAG records map through their source chunk.
- The chunk-based mapping over-approximates what a unit contains, equally for both KGs.

**D-007 (2026-09-30).** The APG authoring pipeline uses one category per source document. The LLM then splits each chunk into units that declare and reference identifiers; references become reciprocal `bring` edges, except toward tool-spec leaves. Allowlists may only name tools that some leaf actually defines, which hardens the pipeline against stray words in an author's `tools` list.

**D-008 (2026-09-30).** Gate inference is world-clustered throughout: bootstrap over worlds within cells, and world-level sign flips for the S7 placebo and the S6 > S7 invariant. With fewer than 4 worlds per cell the verdict is INCONCLUSIVE.
- Why: a task-level test was shown in synthetic data to be anti-conservative when world×arm effects exist.

**D-009 (2026-09-30; corrected 2026-10-02 with D-025).** The PC1 anchor (GraphRAG-Bench Medical) follows the paper's LightRAG setup, with these documented deviations:
- **LightRAG version and query caps.** 1.5.7 instead of 1.2.5, so the query caps are a mapping, derived from the 1.2.5 source (operate.py `_build_query_context`, `combine_contexts`).
  - **1.2.5:**
    - It caps entity descriptions, relation descriptions and chunk text at 4000 tokens each, *per side*.
    - In hybrid mode it then concatenates the local and global sides, so the context holds up to 8000 + 8000 description tokens and 2 × 3 chunks of 1200 tokens.
  - **1.5.7:**
    - It truncates the merged entity and relation lists once, measured on their JSON records.
    - Chunks get what is left of `max_total_tokens`.
  - **The mapping:**
    - `max_entity_tokens` 9000: 8000 plus about 1000 tokens of record keys for about 60 entities.
    - `max_relation_tokens` 9500: 8000 plus about 1500.
    - `chunk_top_k` 6.
    - `max_total_tokens` 26,500, so 6 chunks still fit after the knowledge-graph parts.
  - Before 2026-10-02 the caps were 4000 / 4000 / chunk_top_k 20 / 12,000: half of 1.2.5's entity and relation room and up to 3× its chunks. `top_k` 30, temperature 0.7, chunks 1200/100, hybrid mode, unchanged.
- **Judge prompts and scorers (corrected).** The earlier wording ("three of the four published numbers predate the rewrite") was wrong.
  - The three older numbers predate the benchmark's *custom scorer entirely*. They were scored with a vendored copy of RAGAS's `answer_correctness`: GraphRAG-Benchmark e6305f5, removed 2025-06-14.
  - Neither the 2025-06-14 custom code nor its 2025-07-21 rewrite produced them.
  - Creative Generation (2025-09-25) came from today's code.
  - PC1 now scores each number with its own scorer (D-025).
- **Embeddings (corrected; the retrieval deviation was undocumented).** The paper used BAAI/bge-large-en-v1.5 for retrieval (App. H.2, run_lightrag.py) and for similarity.
  - We had text-embedding-3-small for both. Since D-025 the anchor runs bge-large-en-v1.5 locally for both (`ape.anchor.bge`; the gate keeps OpenAI embeddings):
    - retrieval: mean-pooled as 1.2.5's `hf_embed`;
    - similarity: CLS-pooled and normalised, as sentence-transformers.
  - BAAI's own ONNX export is pinned at revision d4aa6901 and run with onnxruntime. Checked against sentence-transformers and transformers/torch: cosine 1.0000000, element differences ≤ 1e-5.
  - 1.2.5's `hf_embed` also averages padding positions within a batch. Embedding each text on its own (a batch of one) avoids that batching artefact; 1.5.7 batches differently anyway.
- **Keyword extraction** uses structured JSON output.
- **Corpus insertion.** The corpus is inserted whole and chunked by LightRAG, as in the paper, rather than per D-003. With per-chunk insertion, LightRAG 1.5.7 would list up to 75 chunk IDs on every entity line and use up the token caps.
- **Why:** PC1 validates *our LightRAG setup*, not the gate's shared-chunk protocol. If PC1 fails, these deviations are the first suspects.

**D-010 (2026-09-30).** The PC1 sample is 200 questions per type (all 166 Creative Generation), not 50.
- Why: at 50 per type the sampling error (≈4 pp) makes a faithful reproduction fail at least one of four ±5 pp checks more often than not; at 200 it is ≈2 pp.

**D-011 (2026-09-30, EXPERIMENT_AUDIT B1).** The gate's primary comparison uses each system's dev-selected configuration, with realized context tokens and cost reported. A matched-budget secondary analysis runs at ≈300 and ≈2,000 tokens (APG fills via `APE_APG_FILL=1`; LightRAG and S3s are capped). PC4 is now a sanity bound.
- Why: measured APG context is ≈120 tokens against LightRAG's 1,100–2,350, so the old parity ratio (0.8–1.25) could never pass.

**D-012 (2026-09-30, B2).** S7 is sized per cell to APG*'s realized median context on the pilot (`config/s7_targets.json`) and capped at 50% of the corpus.
- Why: at the old fixed 2,000 tokens, S7 was the entire KB in F7-10 and F3-5.

**D-013 (2026-09-30, B3).** F7 exceptions are rendered `descriptive` in gate cells and `id_only` as a paired secondary, both from the same random stream. `search_kb` pull delivery exists for every retrieval arm.
- Offline measurements (lexical fake embeddings, F7-1000):
  - push-mode flat retrieval finds the exception 0/25 in **both** renderings;
  - a pull follow-up query naming the policy ID finds it 25/25 in both.
- So the delivery mode decides relational results. Rendering alone does not fix single-shot retrieval at 1,000 policies, because the numeric band can't be matched lexically.

**D-014 (2026-09-30, B4 and B6).**
- The kg role is required (`get_model(role="kg", required=True)`) and its model name is logged per compile.
- APG* is selected on dev from APG-q and APG-s, symmetric with LGR*.
- APG's `shortlistK` and `minConfidence` are tunable through `APE_APG_SHORTLIST_K` and `APE_APG_MIN_CONFIDENCE`, and are folded into the logged graph version.

**D-015 (2026-09-30, user: "do all").** Models and roles (OpenAI, GPT-6 family, released 2026-09-03 / 09-22).

| Role | Model | Effort |
|---|---|---|
| agent (gate) | GPT-6 Luna | high |
| kg (APG classify, LightRAG keywords) | GPT-6 Luna | low |
| build (APG authoring, LightRAG extraction) | GPT-6 Luna, high; falls back to GPT-6 Sol, medium (rule in D-017) | — |
| embeddings | text-embedding-3-small | — |
| Study G tiers | Luna, Sol, Astra | all at high, plus low as extra capability points |
| summarizer and probe (Study G) | same model as the agent | — |

- Listed prices per 1M tokens, input / output: Luna $0.10 / $0.50; Sol $2 / $10; Astra $10 / $50 (input doubles above 272K context); embeddings $0.02.
- Exact API model IDs and the parameters each honours are confirmed by `readiness/probe_openai.py` (E3).

**D-016 (2026-09-30).** Gate delivery modes: push and pull **co-primary for F7** cells, with the verdict reported per mode; F3 push only. Was O-6.

**D-017 (2026-09-30).** Gate size and build model, decided together.
- Build model: run a dev check with Luna (high) as the builder for both systems. Pass threshold: authored APG graphs declare ≥ 95% of spec IDs (`id_coverage`) on descriptive worlds, and LightRAG extraction recovers ≥ 95% of policy/procedure IDs as entities.
- If Luna passes: build on Luna and use **16 worlds per cell** (power ≈ 0.91 at Δ = 0).
- If it fails: build on Sol (medium) and use **12 worlds per cell** (power ≈ 0.79).
- Re-simulate with pilot σ before freezing. Was O-5.

**D-018 (2026-09-30).** Budget and roles.
- Gate phase: `compute_budget_usd` = **1,500** (anchor + dev + pilot + gate, ~30% contingency).
- Study G is budgeted separately after its micro-pilot.
- Roles: APG owner, skeptic (not on the APG side; tunes LightRAG and S3s and owns their tuning candidates), analyst (owns the pre-registration freeze). Names are still open (O-3).

**D-019 (2026-09-30).** Single hypothesis registry (`HYPOTHESES.md`), and the custom Inspect harness is the only platform. The nine-pattern public-benchmark study in `EVAL_DESIGN.md` is deferred; its benchmarks are external anchors.

**D-020 (2026-09-30, FX-1).** Models and efforts come from one file, `config/models.yaml` (profiles `gate`, `study_g_*`, `anchor`, `anchor_luna`), through `ape.models`.
- Effort is set on each Inspect role model, so it survives per-call schema configs.
- Build effort is sent by `BuildLlm` and recorded in the ledger and manifests.
- Sampling parameters (temperature, top_p, seed) exist only in the paper-faithful `anchor` profile (gpt-4o-mini). If E3 shows gpt-4o-mini is retired, the anchor uses `anchor_luna`, without sampling parameters and with the ±10 pp tolerance.

**D-021 (2026-10-01, FX-5; user: whole program ≈ $5,000 "without much result quality loss").** The program is budgeted with one cost model, `ape.budget`. Inputs are the run plan (`config/run_plan.yaml`), per-call priors (`config/budget_assumptions.yaml`) and the price table. Details: `BUDGET.md`.
- **Totals for the right-sized plan, conservative / expected:**
  - gate $201 / 177
  - main study $1,114 / 894
  - Study G $3,378 / 2,130
  - **total $4,693 / 3,201**, leaving $307 of conservative contingency
- **Against the FX-5 targets:** main and Study G are over by $114 and $678. The program total fits. The program target ($5,000) replaces D-018's gate-only $1,500.
- **Scenarios:**
  - conservative: cached input at the full input price. This is the budget gate: `python -m ape.budget` exits 1 above $5,000.
  - expected: 20–85% of input cached at 0.1× the input price. This is an assumption until E5 confirms GPT-6 cached pricing.
- **Matrices:**
  - Gate: primary unchanged. Diagnostics run 100 × 2, with the S7 placebo at 100 × 3. There is one matched budget (≈300 tokens). See GATE_PREREG §4.
  - Main study: Studies A–C on Luna. Study F replicates on Luna and Sol (brief §7.2).
  - Study G: W = 32K, threshold 20K, N = 40 (Astra 24), Tier B excluded. See CONTEXT_MANAGEMENT_AUDIT §8.
  - Every pilot, tuning, build and capability-anchor run is budgeted.
- **Review change:** Study F's Sol replication uses F7-100 at the full 100 tasks, instead of F7-1000 at 60 (C5 superseded). The 90K-token monolith at Sol prices dominated the main study, and KB scaling is K4's question on Luna. The Sol MDE stays at ~13–15 pp.
- **Cuts,** applied in order until the total fit (uncut $6,530):
  - C1: Sol-low point dropped, −$471. The capability axis has 4 points.
  - C2: Study G Sol CM sessions 8 → 6, −$202. CIs ~1.15× wider.
  - C3: Astra CM sessions 5 → 4 and N 30 → 24, −$552. CIs ~1.12–1.25× wider; Astra close to descriptive; no item-30 probe.
  - C4 (Astra topology 5 → 4) is reversed: no longer needed after the review change.
  - C5 is superseded and C6 was not needed.
  - Never cut: gate primary sizes and epochs, the S7 placebo.
- **Most uncertain priors:**
  - output tokens per call at high effort (1,500)
  - Study G view sizes (0.6 W / 0.4 W) and calls per item (6)
  - multi-agent multipliers (2.5–3×)
  - the cached-input price and cache share
  - F1, F2 and F8 sizes (generators not built)
- **Not affordable as planned:** D-017's Sol build path (+$905; +$957 once FX-6a priced the id_only and F5 dev worlds). It needs approval.
- **Recalibration:** after the gate pilot and Study G's micro-pilot, `python -m ape.budget calibrate` replaces the priors with measured calls and tokens per (arm, model, effort, cell, delivery). Cuts are then revisited in reverse order (C5 first). FX-6 refuses a phase whose projected cost exceeds `remaining()`.

**D-022 (2026-10-01, FX-7; before any freeze).** Which preconditions can block the verdict, and which test cells run first.
- **PC4:** only the 4× sanity bound gates. The matched-budget ±25% clause is reported with the matched-budget secondary. A miss, or a secondary the budget stopped, marks that secondary "not matched" and never blocks the verdict, which does not depend on the matched caps. The offline rehearsal showed why: a cap calibrated on pilot worlds drifted outside ±25% on test worlds (303 → 386 tokens).
- **PC2:** LGR* ≥ LightRAG naive allows PC3's 3 pp tolerance on the point estimate. With no tolerance, a true tie would fail the precondition about half the time from noise.
- **Run order:** the test phase runs every cell the verdict needs first: the GO rule's cells (gate.test.f7, gate.test.f3, gate.diag.s7) and the cells for PC3 (gate.diag) and PC2 (gate.f5). A budget stop can then only cost secondaries.
- **Pull verdict:** pools F7 pull with F3 push. F3 runs push only, so both modes cover the same four cells (GATE_PREREG §3).

**D-023 (2026-10-02, RELIABILITY_REVIEW R1; before any freeze).** The gate's statistics, re-derived on simulated realistic data.

Simulations use `power/power_sim.py`'s model with per-cell baselines 0.85 / 0.45 / 0.75 / 0.60 and 20,000 gates per scenario unless stated. They supersede the PC2 point-estimate rule of D-022.

- **Interval: world-clustered t-interval** (ratio estimator per cell, linearized cluster variance, Satterthwaite df), replacing the percentile cluster bootstrap.
  - At the margin (Δ = −5 pp) the bootstrap passed 3.0–4.0% of the time at a nominal 2.5%.
  - The t-interval passes 2.6% (16 worlds), 2.3% (12 worlds), 2.4% (heterogeneous F7-1000) and 2.8% (σ_w = 1.0, σ_g = 0.6).
  - The bootstrap is still reported.
- **Minimum worlds:** at least 4 *paired* APG*/LGR* worlds per cell. Before, every arm's worlds were counted, so 2 paired worlds could yield GO.
- **α budget:** one-sided 0.025 for the whole procedure.
  - The gate run spends 0.020; the one extension spends 0.005, on fresh worlds analysed alone.
  - The extension needs a stage 1 that is INCONCLUSIVE and test worlds disjoint from stage 1's (`analyze_gate --extension-of`).
  - The bound is Bonferroni over disjoint data.
  - Simulated at the margin: 1.88% overall. P(unqualified GO) at Δ = 0: 0.837, or 0.885 with the extension.
- **Delivery modes: Holm** across push and pull at the stage's α.
  - Per-mode tests issued some GO-type label 4.3% of the time at the margin (both modes at α = 0.025, as before). With Holm at 0.020 the rate is 1.8%.
  - Superiority is tested only after GO in both modes (serial gatekeeping), again with Holm. False superiority at Δ = 0 is 1.8%.
- **PC2 and PC3 fail only on evidence of a violation.** The test is whether the one-sided upper bound is below −3 pp; PC3 tests each pair at 0.025 / 3.
  - At a tie: PC2 fails 0.2% of the time (was 18%), and a PC3 chain 0 of 1,000 (was 16%).
  - The opposite rule, passing only when the lower bound exceeds −3 pp, would fail 88% of ties at F5's size, so it was not used.
  - Detection of a real 8 pp violation: 26% (PC2) and 42% (PC3).
- **PC4's 4× bound is descriptive.** It holds by construction for every budgeted arm.
- **PC5:** S7's cap hits are reported, not gated.
- **Unpaired tasks** are counted per cell, and the §2 secondary conditions are reported whatever the verdict.
- **Pilot σ: moment matching** to the power model (Gauss–Hermite), with 80% χ² intervals, replacing the delta method.
  - Over 300 simulated 4-world pilots, σ_g = 0.8 is estimated at 0.77 (was 0.50), σ_g = 0.3 at 0.28, and σ_w = 1.0 at 0.99.
  - The recommendation uses the intervals' upper ends. At true σ_g = 0.8 it never claims 16 worlds reach 0.8 (true power 0.38). Under the prior σ it usually says "the analyst decides", because a 4-world pilot cannot bound σ_g tightly.
- **Power numbers (GATE_PREREG §4):** P(unqualified GO) at Δ = 0 under the prior σ is 0.84 at 16 worlds and 0.68 at 12. The earlier 0.91 / 0.79 were for one mode's bootstrap at 0.025.

**D-024 (2026-10-02, RELIABILITY_REVIEW S8; before any freeze).** The S7 placebo follows APG*'s delivery schedule.
- **The problem:** S7 delivered its random context once, sized to APG*'s median per compile. A per-step APG* (APG-s) delivers that much at every step, so per sample S7 delivered a fraction of APG*'s tokens. That weakened the placebo the GO rule tests APG* against (§2: APG* > S7).
- **The rule:** S7 runs per step when the selected APG* is per-step (`APE_S7_PER_STEP=1`, set by `ape.run_gate` from `selected.yaml`), and once per task when it is per-query.
  - Per step, each step is a fresh random draw, seeded by the task and the step query so a trajectory replays the same draws.
  - Each draw is sized to APG*'s per-compile median and capped at 50% of the corpus.
- **Why per-step mirroring, not one per-sample total:** a single draw of the per-sample total would show the model up to n_steps times more context in one call than APG* ever shows at once. Mirroring matches APG* both in what the model sees at each call and in what a sample delivers in total; only relevance differs, which is what the placebo isolates.

**D-025 (2026-10-02, PC1 investigation; before any freeze).** PC1 checks each published number with the scorer that produced it, under a macro-plus-type rule, and gates only on what has a published reference.
- **The evidence:** the PC1 investigation of 2026-10-02 read the benchmark's repository history, leaderboard data, paper versions, issues and pull requests.
  - **Leaderboard and paper.** The leaderboard data (`medical_data.csv`) and arXiv 2506.05690 v1–v3 carry the same LightRAG Medical numbers: 63.32 / 61.32 / 63.14, plus 67.91 from v2.
    - Fact Retrieval, Complex Reasoning and Contextual Summarize were on the leaderboard by 2025-05-28.
    - No post-rewrite LightRAG numbers exist.
  - **The scorer behind the three older numbers.** It was not the "pre-rewrite" custom code (first committed 2025-06-14) but a vendored RAGAS `answer_correctness` (e6305f5, uploaded 2025-06-09, removed 2025-06-14). It differs from today's code in:
    - prompt rendering: a JSON-Schema signature, JSON examples and inputs, an "Output:" cue;
    - tolerant parsing with one fix-format re-ask;
    - dropping failed samples;
    - raw cosine of bge *document* embeddings.

    Its instruction text and examples are byte-identical to today's.
  - **Creative Generation** (added 2025-09-25) came from today's code.
  - **A parse bug in today's official scorer.** It parses the classification strictly. GraphRAG-Benchmark PR #56 (open) reports gpt-4o-mini replying `Output: {...}` plus reasoning. That makes F1 = 0 for every sample and accuracy about 0.25 × similarity (≈ 22%). Our replica reproduced the bug silently.
  - **Estimated pass likelihoods for a faithful setup**, with gpt-4o-mini:
    - status quo: about 10–15% (about 20% without the parse collapse);
    - matched scorers per type, as adopted here: about 40%;
    - hybrid > naive alone: about 55–65%, but weak as a fidelity check.
    - Even with matched scorers, the remaining 1.5.7-vs-1.2.5 gap makes a ±5 pp miss on at least one of four types likelier than not. Hence the macro rule below.
- **The rule (GATE_PREREG §7, PC1):**
  - **Gating scorer:**
    - the three older types: the vendored-RAGAS scorer (`ape.anchor.ragas`; renderings byte-identical to RAGAS's own code on test inputs);
    - Creative Generation: today's official scorer with a tolerant classification parse.
  - **Pass:**
    - with gpt-4o-mini answering, judging and building: the macro mean over the four types within ±5 pp of the published macro (63.92), and each type within ±10 pp;
    - otherwise (e.g. the Luna fallback): ±10 pp and ±15 pp.
  - **Reported, not gated:**
    - Today's official scorer, parsed strictly, for every type. Strict parse failures are now flagged.
    - Hybrid > naive. Our naive mode is LightRAG's, not the paper's llama_index RAG baseline (256-token chunks, top 5), so it has no published anchor.
  - **Not evaluable:** if any type's gating-scorer parse-failure rate exceeds 5%, PC1 is *not evaluable*, with a diagnosis. That is a harness defect: the freeze refuses it even with `--accept-pc1-failure`.
- **Also changed:**
  - **Embeddings:** bge-large-en-v1.5 runs locally for the anchor's retrieval and similarity (D-009).
  - **Query caps:** mapped from 1.2.5 (D-009).
  - **Cost:** the anchor's projected cost rose from $5.56 to $10.42 conservative. The answer contexts roughly double and a second scorer's judge calls are added; the OpenAI embedding calls are gone. The program is now $4,748 conservative / $3,228 expected (+$5).
  - **Smoke:** the smoke run reports both scorers' judge parse-failure rates, and warns on any gating failure.
- **Why not simply gate on hybrid > naive:** that check has no published reference and little power to detect an unfaithful setup. The macro ±5 pp check keeps PC1 an anchor to the literature. The per-type ±10 pp bound still catches a type-specific defect.

**D-026 (2026-10-02, pre-run review; before any freeze).** The pilot estimates σ from 8 worlds per cell and makes a two-sided recommendation.
- **The problem:** a 4-world pilot bounds σ_g so loosely that the recommendation rarely decides anything. And in about 4% of 4-world pilots the χ² pivot had no upper solution and reported σ_g's upper end as 0. That made the "conservative" scenario optimistic, and it caused the wrong "16 suffices" calls at a true power near 0.4.
- **Simulation** (real estimator, power_sim's model, baselines 0.85/0.45/0.75/0.60, 300 pilots per row, conservative cost from `ape.budget`):

  | True (σ_w, σ_g), true power at 16 | Design | Detects an underpowered gate | Wrong "suffices" | Cost |
  |---|---|---|---|---|
  | (0.5, 0.8), 0.39 | 4 worlds, old rule | 0.62 (two-sided) | 0.03 | $14.55 |
  | | 4 worlds × 2 epochs | 0.71 | 0 | +$8.37 |
  | | 6 worlds | 0.76 | 0 | +$7.28 |
  | | **8 worlds for APG*/LGR* only (chosen)** | **0.83** | **0** | **+$8.35** |
  | (1.0, 0.8), 0.41 | 4 → 8 worlds | 0.55 → 0.75 | 0.04 → 0 | |

  - No affordable pilot makes "16 suffices" decisive near the target. At a true power of 0.83–0.86, "suffices" comes out only 8–16% of the time at any size up to 8 worlds. What extra worlds buy is catching an underpowered gate.
  - Worlds beat epochs at equal cost.
- **The rule** (`ape.analysis.pilot.power_report`; GATE_PREREG §4):
  - **suffices** when the power at the upper σ ends reaches 0.8 and both upper ends are informative;
  - **insufficient** when the power at the lower ends misses 0.8 (add test worlds within budget or rethink; 20 and 24 worlds are reported);
  - **ambiguous** otherwise, or when σ is not estimable: proceed with the planned 16 and rely on the extension.
  - An upper end with no pivot solution falls back to max(estimate, prior), is flagged, and never supports "suffices".
- **The pilot:** `gate.build.pilot` builds 8 worlds per cell, and `gate.pilot.sigma` runs APG* and LGR* (push, 1 epoch) on worlds 5–8. Everything else in the pilot keeps worlds 1–4.
- **Verified on the implementation** (300 pilots per scenario, with the zero-bound fallback):
  - At a true power of about 0.4, 8 worlds call "insufficient" 75–83% of the time and never "suffices" (4 worlds: 55–61% and 2–4%).
  - At about 0.85, they call "suffices" 11–12%, "insufficient" 11–12% (the 80% intervals' expected misfires near the threshold) and "ambiguous" otherwise.
  - The upper end is flagged in 4.7% of 8-world pilots at σ_g = 0.3 (8.0% at 4 worlds).
- **Rejected: a two-stage rule** (4 worlds, then 4 more only if not decisive). It saves about $3–6 expected, but optional stopping drops σ_g's interval coverage to 0.65–0.68 and gets close calls wrong 16–17% of the time.
- **Cost:** +$8 conservative. Program total: $4,751 / $3,231 expected. D-017's Sol build path rises to +$1,074, because the extra pilot worlds use the same builder.

**D-027 (2026-10-03, second statistics review; before any freeze).** Pilot corners, PC5 on evidence, Holm's worst case and stopping rule, and the extension's role fixed at the freeze.
- **Pilot σ corners.** Power falls with σ_g but rises with σ_w: a world effect moves both arms alike. At 16 worlds, σ_g 0.3 gives 0.826 / 0.838 / 0.854 / 0.931 at σ_w 0 / 0.5 / 1.0 / 2.0. So the least favourable corner of the 80% intervals is usually σ_w low with σ_g high, and taking both upper ends as "conservative" overstated power by 2–6 pp. In the review's 200 pilots per scenario, the corrected rule changed 3.5–9% of decisions.
  - **The rule:** "conservative" is the minimum power over the four σ_w × σ_g corners and "optimistic" the maximum (`ape.analysis.pilot.power_report`; `corners` in power.json). The flagged-upper-end rule is unchanged.
  - **Verified** (200 simulated 8-world pilots per scenario; 600 power simulations per corner). The implementation's decisions equal the corner rule in every pilot.
    - At σ_g 0.8 (true power ≈ 0.4): "insufficient" 81% (σ_w 0.5) and 70% (σ_w 1.0), "suffices" never.
    - At σ_g 0.3 (true power ≈ 0.85): "suffices" 6% and 6.5%, "insufficient" 9% and 6.5%, "ambiguous" otherwise.
  - **pilot.json** "pilot σ_w, σ_g and power" now carries the 80% intervals, any not-informative flags and the least-favourable-corner powers.
- **PC5 fails only on evidence.** The point rule failed 50% of gates at a true error rate of 1.5% and 26% at a true cap-hit rate of 8.5%: a finished, paid gate became PRECONDITION_FAIL on noise, the problem D-023 fixed for PC2 and PC3.
  - **The rule:** an arm and mode fails when the one-sided 97.5% Clopper–Pearson lower bound of its error rate exceeds 2%, or of its cap-hit rate exceeds 10% (`gate_stats.harness_rates`). A point estimate at or over a limit is a warning, shown as a caveat on the verdict. The pilot reports the same rates as an early warning (`pilot.json` `harness`).
  - **Re-simulated** (300 gates each, 10 gated groups):

    | True rates | Fails | Warns |
    |---|---|---|
    | 1.0% errors | 0% | 0.7% |
    | 1.5% errors | 0% | 58% |
    | 2.0% errors (at the limit) | 19% | 100% |
    | 8.5% cap hits | 0% | 25% |
    | 10% cap hits (at the limit) | 17% | 100% |
    | 4% errors in one arm and mode | 95% | — |
    | 15% cap hits in one arm and mode | 100% | — |

  - Samples are treated as independent. Errors that cluster by world would make the bound slightly high, so a failure stays strong evidence.
- **Holm's worst case** is one mode at the margin and the other clearly non-inferior. Holm then gives the marginal mode the full 0.020: stage-1 false claims 2.08–2.12% (20,000 gates, SE 0.10 pp), 2.17–2.33% overall with the extension, and 2.4% at 12 worlds. That is within the procedure's total 0.025, so no α change. GATE_PREREG §3 and §8 state it.
- **Holm's stopping rule.** When Holm stops at the first mode, the other mode is "not tested": it is classified and shown at the stopping level (α/2), never as GO. Before, it was shown at α with an interval that could exclude −5 pp while being labelled INCONCLUSIVE (review edge case 10).
  - A mode with a cell under 4 paired worlds leaves the family, so the other mode is tested alone at the full α. That is one test, so the familywise error is still controlled; GATE_PREREG §2 documents it.
- **The extension's role is fixed at the freeze.** Before, its α depended only on the analyst passing `--extension-of` to analyze_gate, and a second extension of the same stage 1 was accepted (two 0.005 spends).
  - `run_gate freeze --extension-of <run>` (or `--fix-cycle-of <run>`) records `role: {kind, of}` in freeze.json, outside the freeze fingerprint, like `--accept-pc1-failure`. analyze_gate derives the stage and α from that record; its `--extension-of` is only a consistency check.
  - The freeze refuses unless the named run is a frozen, analysed primary run with verdict INCONCLUSIVE (extension) or NO_GO (fix cycle). It also refuses a second extension or fix cycle of the same run, found by scanning runs/*/freeze.json.
- **Smaller.** Rows with NaN success now raise a counted `NaNOutcomeWarning` instead of being dropped silently. `load_results` never produces them.

**D-028 (2026-10-03, user: "do/fix #1–#5 before we enable the main and G to be runnable").** Five design decisions for the main study and Study G, applied to the plan, config and documents before the build (BUILD_PLAN.md §0).
1. **The persona test is dropped (K3 / brief H2b).** The worlds carry no persona content, so S5-P0 would equal S5. S5-P0 is removed from `main.pilot.b`, `main.B.arms` and the budget arms (−$6.5).
2. **The tier confound is fixed (H6).**
   - The problem: `main.F.sol` runs F7-100 while Luna ran only F7-1000.
   - New cell `main.F.luna-f7-100`: S1, S5, M1, M2 × 100 tasks × 3 epochs (+$12.25).
   - `main.build.kg` gains F7-100 (both tiers' S5) and F1-2 (the micro-pilot's S5) builds, 11 worlds each (+$0.6).
3. **Underpowered tests are descriptive.** H1d / M4 (concurrency: n = 50 gives ±6–10 pp against a 2 pp margin) and H2c / K4 (KB scaling: two levels, slope CI about ±4.7 pp per decade against ±2 pp) are reported with intervals, not as confirmatory equivalence tests. K4's relational-vs-independent clause is not tested; the plan has no independent cells.
4. **Study G's short sessions now cross the window.**
   - At the default ~1,000-token tool files, 20- and 24-case sessions never cross W = 32K.
   - 20 reference sessions per setting were measured with the default 40-case cells as the reference:

     | Cell | Tool-file tokens | Median crossing (fraction of the session) |
     |---|---|---|
     | 40-case (default) | 1,000 | 0.70 |
     | 20-case | 2,000 | 0.80 |
     | 20-case | **2,250** | **0.70** |
     | 24-case | 1,500 | 0.83 |
     | 24-case | **1,750** | **0.71** |
     | 24-case | 2,000 | 0.62 |

   - The run plan sets `knobs: {output_tokens: 2250}` on `g.topo.*` and `{output_tokens: 1750}` on `g.cm.astra-high`. `make_world` passes F8 knobs to the generator, and a test checks the plan's crossings.
   - The 10-case cell stays a no-overflow control.
   - Session views are priced at 0.6 W, so the projection is unchanged.
   - Real agents add text and reasoning to the view, so they cross earlier than the reference.
5. **External benchmarks are deferred to Tier B:** τ³-bench, AppWorld, BrowseComp-Plus and CoDA (brief §5.4, EVAL_DESIGN). GraphRAG-Bench Medical remains the gate's PC1 anchor only.

**Budget:** $4,762 conservative / $3,239 expected (main $1,170), contingency $238.

**Still open for the Study G pre-registration** (BUILD_PLAN §4): G-H1's single-agent reference, CM-native at one tier, S1+KG and S-subiso, and the capability anchor near the ceiling.

**D-029 (2026-10-03, BUILD_PLAN B4 tooling question).** Confirmatory inference in the main study and Study G is design-based; the mixed-effects logistic model is descriptive.
- **Why:** R is not installed, so lme4 would add a toolchain; bambi/PyMC is Bayesian and far too slow for the power simulations every confirmatory test needs (thousands of refits); statsmodels has no frequentist crossed-effects logistic GLMM. The brief already names design-based tests for every claim (§7.4: paired sign-flip permutation on task-level means, Holm within each hypothesis family, TOST or one-sided tests with pre-registered margins, clustered bootstrap CIs), and the gate validated this approach end to end.
- **Main study:** each contrast is paired on tasks (task-level epoch means), resampled or sign-flipped at the **world** level, since tasks within a world share their KB (the gate's finding). Interactions (H2's 2×2, arm × tier) are paired difference-in-differences per task. The S8 frontier is interpolated per cost meter from exhaustive subsampling of the S1 pool.
- **Study G:** sessions are the clusters. G-H1's slope is a session-clustered regression of per-session paired differences on measured capability; G-H2's equivalence and slopes and G-H3's shares likewise.
- **GLMM:** `statsmodels` `BinomialBayesMixedGLM` (variational Bayes; no new dependency) fits the brief's `success ~ arm * log_knob + (1 | task) + (1 | task:arm)` and Study G's `success ~ topology * capability + (1 | session) + (1 | item)` as descriptive models only. No claim rests on them.

**D-030 (2026-10-03, BUILD_PLAN B7: Study G session runner).** Choices made while building the ContextPolicy layer and mid-session resume (`agent/context_policy.py`, `agent/session_checkpoint.py`).
- **Thresholds and the window.**
  - T_abs is level-triggered: the policy's `on_threshold` hook runs before every call whose view exceeds it, measured on the harness meter (o200k over message text and tool calls; tool schemas excluded). Inspect-based arms force compaction from this hook instead of using Inspect's own threshold, which counts tokens differently.
  - W applies to management (`cm`) inputs as well as agent views. A management overflow at an item boundary fails from the next item; the item that just ended still counts. Probes are not W-checked.
  - A policy's extra tool schemas (e.g. `todo_write`, ~800 tokens) are not counted against W or T, like every other tool schema; their size is recorded (`policy_tool_tokens`).
- **Records.** Every call carries a kind (agent, cm, probe), its model and its usage. Management calls and probes at a boundary are attributed to the item that just ended, session-start calls to item 0 and report-phase calls to N+1. At a boundary the order is: the policy's `after_item`, then the probe (it sees the post-boundary view), then the checkpoint. Policy tool calls are logged with `policy: true` and ignored by the scorers.
- **Knobs.** Policy knobs are environment variables `APE_CM_<KNOB>`, recorded in task metadata and in the checkpoint key; each env group needs its own log directory (as for every APE_* knob). `threshold` is a task arg so tuning candidates can vary it. The knob variant stays `variant` (= `gen_f8.variant_tag(knobs)`).
- **Resume** (`APE_SESSION_CHECKPOINTS=<dir>`, one per run):
  - Inspect's own sample checkpointing was not used: it resumes only on task-level retries (not `retry_on_error`), needs a restic binary and changes the eval config.
  - A checkpoint is saved at session start and after every completed item. The interrupted item reruns from its first call, so a resumed session is not bit-identical to an uninterrupted one; `f8_resume` and the score metadata record every resume.
  - The key hashes the configuration and the source of the session modules, so any change, including a code edit between a crash and its retry, starts a fresh session.
  - Checkpoints are deleted when a session ends normally, overflows or hits an Inspect limit.
- **Inspect 0.3.273 facts that change the audit's plan** (CONTEXT_MANAGEMENT_AUDIT §6): `todo_write()` stores nothing; `memory()` keeps files in the sample store, which a resume does not restore unless the policy carries them; Inspect's Summary strategy makes an internal call our per-call meter cannot see, so CM-sum uses our own F8 prompt through the metered `cm` path; CM-native raises where the provider lacks support.
- **Known gap, to fix before any paid run (tracked in BUILD_PLAN §6):** Inspect logs a sample's usage only for its last attempt, so the usage of errored attempts under `retry_on_error` appears in no log, for every task. Spend accounting (`ape.budget`, `ape.spend`) undercounts by that amount. Sessions record the lost usage (`f8_resume.unlogged`); other tasks do not yet.

**D-031 (2026-10-03, BUILD_PLAN B4: main-study statistics; `src/ape/analysis/main_*.py`, `frontier.py`).** Choices made while implementing D-029. The hypothesis table (`main_hypotheses.HYPOTHESES`) is the single source PREREGISTRATION_MAIN.md copies.
- **Decision test:** world-clustered sign-flip on task-level epoch means, exact up to 20 clusters (else 10,000 Monte Carlo flips); NI and TOST shift each cluster by the margin. The gate's cluster-t is a cross-check only: at 9 worlds it ran at 0.031–0.033 against a nominal 0.025.
- **Headline interval:** the inverted sign-flip test (coverage 0.94–0.96 at 9 worlds). The world-clustered BCa bootstrap is reported as the brief asks but carries no decision (coverage 0.88).
- **Clusters:** F1/F2 worlds with the same seed share one supplier registry, so they form one cluster across Study A's cells (a pooled Study A test has 9 clusters, not 36).
- **α and pooling:** one-sided 0.025 for superiority, NI and interaction tests; 0.05 per side for TOST (a 90% interval); Holm within each hypothesis family, no correction across families. Pooled members weight cells equally.
- **S8 frontier:** confirmatory meter = tokens. S8(k) costs the sum of its k runs (wall-clock may use the parallel maximum). The vote is a plurality of canonical answer keys; ties count as expected success; errored runs abstain. An arm beyond S8(8) is not tested; one cheaper than one S1 run is compared with S8(1) and flagged. The live S8k3 arm is compared with the post-hoc S8(3) descriptively. S1's 3 main epochs are the first 3 pool runs.
- **Gatekeeping:** only M2 (S8 vs M1) is gated on M1 > S1.
- **H6** (no ID in HYPOTHESES.md yet): the coordination payoff M1 − S8(3)@M1, Sol − Luna on the same tasks (F1-32, F7-100), from 3-run frontiers in both tiers.
- **K2:** the 2×2 interaction and both NI members form one Holm family; the cost clause (S5/M2 tokens, upper 95% ≤ 0.6) is an intersection-union condition.
- **C1:** arms ranked by cost per solved task under tokens, cache-adjusted $ and working time; a family flips when its smallest pairwise Kendall τ-b is below 0.8 (Study A and B cells).
- **Outcomes:** wall-clock = Inspect working time; a cap hit is the eval's turn cap reached without an answer, or any Inspect limit; errored samples are failures.
- **Power** (1,000 replicates per scenario, gate priors, 9 worlds × 100 tasks, 3 epochs): type I error at or below nominal for every family. Power is below 0.8 for most members at plausible effects (decided in D-033).

**D-032 (2026-10-03, BUILD_PLAN B10: Study G statistics; `src/ape/analysis/g_*.py`).** Choices made while implementing D-029. The hypothesis table (`g_hypotheses.HYPOTHESES`) is the single source PREREGISTRATION_G.md copies.
- **Decision test:** a session-clustered t with Satterthwaite df (capped at clusters − 1; a world run at several capability points is one cluster), on weighted sums of per-point session means with epochs pooled first. The exact wild sign-flip (null-restricted residuals) is reported alongside. The t is primary here, unlike the main study, because 4 Astra sessions give a sign-flip floor of p = 0.0625; its type I error is 0.023–0.027 at nominal 0.025 in 10,000 simulated studies.
- **Ratios** (isolation share ≥ 0.5, recovery ≥ 0.8) are tested as linear forms; the ratios get delta, Fieller and bootstrap intervals, and are not estimated when their denominator is ≤ 2 pp.
- **Estimands:** session outcome = item success rate (binary whole-session success is reported with pass^k). G-H2's equivalence estimand is R_x (headroom recovered), not the logit gain, which drifts under a constant R. Degradation slopes exclude overflowed items. Cost meters exclude probes and include management calls.
- **Power** at the planned sizes: type I at nominal everywhere; G-H1, G-H2b and G-H3b are far below 0.8 (decided in D-033).

**D-033 (2026-10-03, user, on the power findings P-1 and P-2).** What the main study and Study G claim, given the power simulations of D-031 and D-032.
- **Main study:**
  - M3 (communication ≈ 0 at matched cost) and M5 (specialization ≈ 0): TOST margin widened from ±3 pp to **±6 pp** (power 0.13 / 0.05 → 0.84 / 0.87). The claim is now "no effect larger than 6 pp".
  - K2: the 2×2 interaction stays confirmatory (power 0.86 at −10 pp). Its operational non-inferiority becomes **K2-NI**: one test pooled over Study B's four cells at a **5 pp** margin, in its own Holm family, with the cost condition unchanged (power 0.79).
  - The tier clause (brief H6, now **T1**) is **descriptive**: an estimate with its interval (power 0.44–0.54 at 15 pp).
  - The superiority tests (M1, M2's frontier, K1) are unchanged; their planned MDE is about 15 pp, as the brief already states for within-condition contrasts.
- **Study G:**
  - **G-H1 is descriptive** (an estimate with its interval; power 0.12, and no affordable design reaches 0.8). The reference is S-CM*; S1 on items before S1's overflow is the sensitivity analysis; S1 itself is descriptive only, because its overflow rule makes its gap drift with capability.
  - **G-H2b (R_x equivalence across capability) is descriptive** (power 0.01 at ±0.20 R). G-H2a (Gap_T > 0 at every point) stays confirmatory.
  - **Design change:** a new `g.topo.luna-low` cell (effort low; S1, M1, M2, S-CM*; N 20; 16 sessions × 2 epochs) and `g.topo.luna` 8 → 16 sessions. +$73 conservative / $43 expected. Expected power: G-H3a ≥ 0.99, G-H3b ≈ 0.70.

**D-034 (2026-10-03, BUILD_PLAN B2: multi-agent arms; `src/ape/agent/multi/`).** Choices made while building S9, M1/M1s, M1k, M2, M7 and S8k3, and one plan change (user).
- **One loop.** Every agent runs `multi.core.react_loop`, which applies `kb_agent`'s per-turn rules (step context never stored, tool exposure, nudges); it gives byte-identical model inputs to `kb_agent` for S1, S3s and APG-s. Inspect's `react()`, `as_tool()` and `handoff()` are not used: each changes information flow (an added submit tool and continue message; one string in and the last message out; `content_only` filtering).
- **Delivery is arm-wide:** the orchestrator, chair and aggregator get the arm's delivery too, so S9 → M1s changes one switch (the cost: M1 on F7-1000 carries the monolith in the orchestrator as well).
- **Orchestrator arms** (M1, M1s, M1k, M2): a forced `plan` on turn 0 (as in S9), replanning allowed; the orchestrator's tools are `plan`, `delegate` (1–3 subtasks per call, one call = one round) and the answer tool, never environment tools; each subtask runs in a fresh worker context with the task's non-terminal tools and returns only its result (clipped at 2,000 tokens). Every worker, council phase, chair and aggregator gets the task's turn cap.
- **M1 = M1s in what the model sees:** workers in a round each get a copy of the environment as the round began, and their changes merge in subtask order afterwards, so concurrency changes only timing (tested: identical multisets of model inputs and identical end states).
- **M2's specialization:** a world's domains (F3 service domains, F7 policy domains) are dealt round-robin to 3 specialists, each with its domains' tools and a specialty prefix on its KG queries; the orchestrator routes subtasks by specialist. **M2 is dropped from F1-32 (user):** on F1 the orchestrator cannot know a supplier's segment before a lookup, so routing would be blind. `main.F.luna` keeps S5 on F1-32; `main.F.sol` runs S1, S5, M1 on F1-32 and F7-100, and `main.F.sol-m2` runs M2 on F7-100 (−$190).
- **M7:** 3 members propose (a proposal tool shaped like the answer tool, plus a rationale), then 2 critique rounds continue each member's own history with the others' latest proposals; a chair in a fresh context submits. Members run with separate environments; M7 refuses F3 (mutating tools). Priced at its measured 5× a single agent (was 3×; +$33).
- **S8k3:** 3 isolated `kb_agent` attempts; a plurality vote on the answer as the scorer compares it; an aggregator only on ties (earliest tied attempt as fallback); the winner's end state becomes the sample's. The post-hoc S8 frontier (D-031) counts ties as expected success instead, so the live S8k3 is compared with S8(3) descriptively.
- **Records:** `mas_accounting` (per agent and per model role, summing exactly to the sample's usage), `mas_switches`, `mas_params`, `mas_agents` (stop reason per agent: cap-hit analysis for these arms reads it), `mas_plan`, `mas_rounds`, `mas_specialists`, `mas_council`, `mas_ensemble`. `compile_log` holds every agent's compiles, so evidence scoring covers the union.
- **Worker failures:** ~~return a marked result~~ since D-047, a non-limit exception in a worker, council member, chair, ensemble attempt or aggregator is recorded on that agent and re-raised, so the sample errors and is retried like S1's; model misbehaviour that is not an exception (bad tool calls, text-only replies, a worker that never reports) still gives marked results. B12's smoke checks `mas_agents[*].error`.
- One private Inspect API is used (`inspect_ai.util._store.init_subtask_store`, pinned at 0.3.273).

**D-035 (2026-10-03, budget after the build's measurements).** The plan is priced with what B2 and B8 measured, and the per-study allocations now match it.
- **Measured overheads:** M7 at 5× a single agent (D-034); CM-todo, CM-reset and S-CM* at 0.20, 0.25 and 0.25 management calls per agent call (B8, gold and naive mocks, `todo_write` traffic included; were 0.15, 0.05, 0.15). CM-sum and CM-native keep their priors (0.08, 0.05), which exceed the mock rates; their T_abs triggers fire more often with real agents, so the micro-pilot recalibrates them.
- **Allocations** (`run_plan.yaml` `budget.allocations`, enforced by both orchestrators' budget guards): gate 300, main 1,100, Study G 3,600 (were the FX-5 targets 700 / 1,000 / 2,700, under which the guard would have stopped Study G's topology cells). Contingency target 280.
- **Totals:** $4,710 conservative / $3,160 expected (gate 214, main 1,013, Study G 3,483); contingency $290.

**D-036 (2026-10-03, BUILD_PLAN B1 + B6: the study runner `ape.run_study`).** Choices made while generalising the gate orchestrator.
- **Shared machinery stays in `run_gate.py`.** Nothing was moved out: gate tests monkeypatch its module-level names, and `run_study` calls `run_gate.X` at call time, so a patch applies to both runners. Additions there default to the gate's behaviour (`build_world_set(study=, profile=)`, `record_build_health(systems=)`, `check_live_smoke(extra_required=)`). Gate run ids `main` and `study_g` are refused (`runs/main/`, `runs/study_g/` hold those studies' runs).
- **Seed namespaces** (asserted disjoint at import): dev 1000+ shared by all studies (reusing the gate's dev worlds and their paid KG builds; a live build-dev refuses to rebuild them with a different builder); pilot gate 2000+, main 12000+, Study G 22000+; test blocks gate [3000, 9000), main [13000, 19000), Study G [23000, 29000); offline rehearsals 9000 / 19000 / 29000. BUILD_PLAN's proposed 4000+/5000+ fell inside the gate's test range. The test lock records which study unlocked it. `gate_samples` orders worlds by numeric seed and, without a `seed_base`, keeps only seeds below 10,000 (as strings, `s12000` sorted before `s2000`, so a gate cell could have read main-study worlds).
- **Phases:** main: preflight, build-dev, micro-pilot, tune, pilot, freeze, build-test, test, analyze; Study G the same without the pilot (its micro-pilot is `g.pilot.luna`). Micro-pilot and pilot run on the pilot split, tuning on dev, the rest on test; the test phase runs its primary groups first (main: Studies A and B; Study G: capability anchor and context management).
- **KG arm:** any GO label gives APG* (GO_PULL_ONLY noted), NO_GO gives LGR*, with that gate run's selected knobs; live refuses INCONCLUSIVE, PRECONDITION_FAIL, an unfrozen, unanalysed or offline gate run, and oracle arms. The gate run is kept in `run.json`. Builds per arm: S5, M1k, M2 get the resolved KG system; S3s gets chunk embeddings.
- **Token caps (brief §4.5):** the micro-pilot runs uncapped; cap = ceil(8 × the median of S1's per-sample total tokens), errored samples excluded; the pilot adds B0 for cells the micro-pilot lacks (`token_caps_pilot.json`); F7-100 borrows from the nearest measured level on a log scale (a tie takes the larger cap); Sol cells use the Luna caps; Study G is uncapped (its window rule bounds it).
- **S7 in the main study:** targets are S5's median realised context per cell in the pilot; S7 runs in its own group with `APE_S7_PER_STEP` mirroring the KG arm.
- **Freeze:** the pre-registration, the configs, the run's outputs (selected.yaml; main also both caps files and s7_targets.json), `uv.lock`, `src/ape/analyze_<study>.py` and all of `src/ape/analysis/`, and the gate run's selected.yaml, decision.json and freeze.json. Live, the analysis module must exist before the freeze. PROVENANCE markers carry the study (`<!-- ape:test-seeds study=… run=… -->`).
- **Budget guard:** the smaller of the program's remainder and the study's allocation remainder, spend counted by study label.
- **Tuning:** grid keys are plan arm names, systems may have their own `dev_cells`, session systems are scored by mean item success; the completeness check warns and does not block the freeze; live tune refuses a placeholder grid or a system without owner and sign-off.
- **Analysis interface:** `ape.analyze_<study>.analyze(run) -> dict` writes `report/report.md`; offline without a module writes a stub, live refuses.

**D-037 (2026-10-03, BUILD_PLAN B8: the context-management arms; `agent/cm_arms.py`, `cm_prompts.py`).**
- **One view layout for every arm:** system, start message (the task spec), notes if any, the completed cases still shown, the todo list if any, the current case. The current case is never compacted, summarised or dropped. At T_abs (level-triggered) a compactor acts first (prune or trim, no model call), then, if still over, a mover drops completed cases (sum, reset or todo).
- **Arms and defaults** (knobs `APE_CM_*`): CM-prune = Inspect's `CompactionEdit`, keeping the last 3 tool results among completed cases; CM-trim = `CompactionTrim`, preserving 0.8; CM-sum = our own incremental F8 summary through the metered `cm` path (Inspect's Summary strategy makes calls our meter cannot see), structured prompt; CM-todo = `todo_write` plus extraction at shift start and on each `MEMO` line, the list parsed from the calls and checkpointed, replacing completed cases at T_abs; CM-reset = todo plus a reset at T_abs and every 5 cases with a handoff note; S-CM* = any valid stack via `APE_CM_STACK` (at most one of prune/trim and one of sum/reset; 17 stacks), default prune+todo+reset. Inspect's strategies are applied directly, not through its `compaction()` handler, so they compose and their state is ours.
- **CM-native** compacts on the agent's own model and is gated: any real model needs a confirmed support record (`<APE_CACHE>/native_compaction.json`, written by the probe, keyed by model name); without one every sample fails at start. Until the next agent call an opaque block counts as everything it replaced; after it, as the provider's reported input minus our count of the rest, which over-counts it (conservative). A failed compact is logged and the session continues; after 3 failures the arm stops compacting.
- **Error handling:** bad content from a management call is logged and handled; provider exceptions on summary and extraction calls propagate, so the sample retries and resumes; only native compaction errors are absorbed.
- **Knobs share one namespace** (`APE_CM_PRUNE_KEEP` applies to CM-prune and S-CM*), so each tuning env group runs only its own arm.

**D-038 (2026-10-03, amends D-032's df rule; Study G statistics after D-033).** With the D-033 topology design, all 16 Luna worlds run at both Luna points but only 8 also at Sol, so worlds contribute unequally, and capping the t's df at clusters − 1 let G-H3's tests reject at 0.028–0.030 against a nominal 0.025 (10,000 simulated studies). The df are now capped at G_eff − 1, the model-based effective number of clusters (≈ 12.6 here), and the variance uses exactly unbiased cross-point terms for partially shared worlds; both reduce to the old rule for balanced designs. Type I is back at 0.026–0.027. Power at the D-033 design: G-H2a 0.99; G-H3a 0.997 at an isolation share of 0.75 (0.90 at 0.70); G-H3b 0.69 at recovery 0.95 and cost ratio 0.45 (24 Luna sessions per point would give 0.76 for about +$51). G-H1 (descriptive): its interval excludes 0 in 26% of studies under full convergence.

**D-039 (2026-10-03, main-study token caps).** In the offline rehearsal M7 hit the 8 × B0 cap on every Study A sample. That may be the naive mock, but a k=3 council with two critique rounds and a chair costs about 5× a single agent, so live, tasks above S1's median could hit the cap often, and cap-driven failures would bias M3 and C1. Rule (implemented in `ape.run_study` with the B1 integration): after the pilot, each arm's cap-hit rate is computed per cell (`main_load.cap_hit_of`). If any arm exceeds 10% (PC5's threshold), the cap multiple doubles for **all** arms (8 → 16, at most once more to 32), that arm's pilot cells re-run under the new caps, and the check repeats. The freeze records the final multiple and every pilot cap-hit rate, and refuses while any arm is above 10% at 32×. One cap applies to every arm, as the brief requires (§4.5).

**analyze_main (2026-10-03).** The main study's analyze phase reads the runner's test manifest and freeze, labels arms by their plan names (S5 = the frozen KG arm), takes the tier from each group's profile, and writes `report/decision.json` and `report/report.md` (coverage, caps and cap-hit rates, tuning parity, the full main report). PREREGISTRATION_MAIN.md defines no extension α yet, so an extension run is analysed alone as a primary-only analysis and the report says so; B5 decides whether the main study has an extension.

**D-040 (2026-10-03, BUILD_PLAN B9: topology arms in Study G sessions; `agent/multi/session_team.py`, `session_prompts.py`).**
- **S1 = CM0**, registered as S1 (identical records and score; only the arm name differs).
- **M1 and M2 are context policies on the unchanged session loop:** the session's agent is the orchestrator, so the history, view, W check, per-case cap and nudge, report phase, probes, records and checkpoints apply to it as to CM0. Its view follows **the CM0 rule (no context management)**, so the topology contrast is not confounded with a CM strategy. It holds the cases, its delegations and answers and the workers' results, never the bulky customer and order files. There is no `plan` tool in sessions: delegating each case is the decomposition (DEC=1 is still recorded).
- **Tools:** the orchestrator gets the case-answer and report tools plus `delegate` (1–3 subtasks); workers get the lookups and procedure tools plus `report` (M2: its specialist's subset). Workers see a fresh context per subtask: the session system prompt (corpus and shift rules) and one user message with the worker note and the subtask. M2's specialists are the world's policy domains, then its service domains, dealt round-robin to 3; M1 and M2 differ only in SPEC (tested).
- **W applies to every agent's input**; an overflow anywhere ends the session as for CM0. A limit hit in a worker ends the sample with its records; other worker errors end the attempt, and the retry resumes from the last item (unlike the main study, where a failed worker returns a marked result: sessions have resume).
- **Accounting** comes from per-call records (kind `agent`, with agent ID and role), so it stays exact after a resume; per-agent sums equal Inspect's usage once probes are added (tested). An item's `generations` counts the orchestrator only.
- **Bug fixed in B7's checkpoints:** the saved state held live references (the recorder's events, policies' state), so a failure mid-item leaked the failed attempt's partial tool events into the resumed session (a duplicated procedure call fails a ticket). The saved state is now the JSON written, read back; existing checkpoint files no longer resume (their code hash changed).
- **For the Study G pre-registration:** at N = 20 and W = 32K the orchestrator never overflows, so M1 − S1 is largely S1 overflowing (the harness's hard-window rule) while the team does not. That is the isolation mechanism G-H3 measures, but measured against a harness-enforced window; PREREGISTRATION_G.md must say so. Measured with the gold mock, M1/M2 cost 1.75× S1's calls and about 0.5× its tokens per item reached; the budget's 2.5×-generations prior (≈1.67× tokens) is conservative and stays until the micro-pilot recalibrates it.

**D-041 (2026-10-03, BUILD_PLAN B3: main-study tuning; `config/tuning_grid_main.yaml`, `agent/multi/knobs.py`).**
- **What is tuned:** each multi-agent arm's own protocol text, never generic task advice that would help S1 as much. `APE_MAS_<ARM>_PROMPT` picks one of four role-note variants for S9, M1, M1k, M2 and M7: `default` (B2's notes, byte for byte), `concise`, `structured` (IDs and produced values named at every hand-off) and `verify` (checks at each hand-off). Data formats stay fixed and the notes never mention scheduling. M1s runs as M1's selection (identical inputs). Structural parameters (3 workers, 2 rounds, k = 3) are not knobs; `APE_MAS_<ARM>_CLIP` exists but no candidate uses it. Knobs are validated when the solver is built and recorded in task metadata and fingerprints (`APE_MAS_` in `ARM_KNOB_PREFIXES`).
- **Equal budget:** 4 candidates per system (S9, M7 on F1-32 and F2-10; M1k and M2 on F3-60 and F7-1000; M1 on all four, one selection), `tie_pp` 1.0 as in the gate.
- **Inherited from the gate:** S5 (the KG arm) and S3s keep the gate's selections: same profile and the same dev worlds, so re-tuning would repeat part of the gate's tune. `main.tune.b` no longer runs S3s (−$0.8). S1 has no text of its own: its prompt is the base every agent of every arm gets, engineered by the skeptic in the gate; the skeptic signs off S1.
- **Owners:** the M arms belong to the independent M-arm prompt author (brief §7.3), proposed to be someone other than the skeptic; owner fields are TODO until named (O-3).
- **Completeness check** (`tuning.study_grid_problems`) before any dev run: candidate counts equal under `equal_budgets`, distinct ids, each candidate runs its own arm and sets only its own valid knobs, no knob shared across systems, and counts and dev cells match the run plan.
- **Known limits:** with about 40 dev tasks per candidate the standard error of a difference between candidates is about 11 pp, so the tune catches broken variants, not small gains. The gate gave S5's systems 6 configurations and S3s 3, not 4. Sol cells reuse the Luna selections.

**D-042 (2026-10-03, BUILD_PLAN B11: PREREGISTRATION_G.md drafted).** The Study G pre-registration is written; its hypothesis and design tables are generated from `analysis/g_hypotheses.py` and a test keeps them identical. Decisions made with it:
- **T_abs is fixed at 20K for every arm** (run_plan `study_g.threshold`), not tuned. It defines when management fires, so tuning it per arm would let arms differ in a structural parameter; the runner never passes it as a candidate.
- **Each tuned arm runs with exactly its own selection's knobs.** CM knobs share one namespace (D-037), and the runner applied every selection's env to every arm of a cell, so S-CM* would have run with CM-prune's and CM-sum's knobs; being fixed in the runner (per-arm env groups).
- **Topology is a primary test group** (G-H3 has been confirmatory since D-033), so a budget stop cannot drop it first; being fixed in the runner.
- **A missing planned capability point** makes a confirmatory row INCOMPLETE, never SUPPORTED (the claims are restricted to the planned points); being fixed in the analysis.
- M1 and M2 are not tuned in Study G (they inherit the main study's protocol defaults), while S-CM* is tuned; the pre-registration states the asymmetry.
- Study G's runner fills no `[PILOT: …]` items: tuned values come from the frozen `selected.yaml`.
- **Open for the user** (`[USER: …]` in the pre-registration): CM-native tiers, S1+KG and S-subiso, the capability anchor, the go-ahead after the micro-pilot review, and the CM-arm owner, skeptic, analyst and independent prompt author.

**D-043 (2026-10-03, user: Study G's open design questions; costs from the B11 probe/grid package).**
- **CM-native at two tiers:** it replaces CM-prune in `g.cm.sol-high` (+$6), so native compaction is measured at Luna-high and Sol, which the provider-absorption question needs. It runs only with the probe's confirmed Sol support record; if the probe finds Sol unsupported, CM-prune goes back into that cell before the freeze (a plan change, not a deviation). Rejected: adding it at Sol beside CM-prune (+$125, over Study G's allocation) or at Sol and Astra (+$373, over $5,000).
- **Capability anchor unchanged** (S1 on F7-10 + F3-5), with the **tier order (Luna-low < Luna-high < Sol < Astra) as a pre-registered sensitivity** for every analysis that uses measured capability (G-H1, G-H2b). Rejected: a harder anchor (F7-100 + F3-60, +$45). The anchor is near the ceiling (Sol and Astra about 1.1 SE apart under the priors), so the measured scale may not separate the top tiers; the sensitivity covers that.
- **S1+KG and S-subiso are Tier B:** F8's knowledge base is already in the system prompt (~2.3K tokens), so a KG arm has nothing to retrieve, and S-subiso is the audit's Tier B CM-subiso. The pre-registration lists both as not run.
- **Study G tuning grid** (`config/tuning_grid_study_g.yaml`): each candidate writes out its arm's complete `APE_CM_*` configuration, so an arm runs at test exactly as on dev; CM-prune keeps 3/6/12 tool results; CM-sum structured/plain and CM-todo extraction on/off (2 candidates each: with T_abs fixed that is their design space; `g.tune.luna` now prices 2, −$4); CM-reset every 5 cases with handoff, at T_abs only with handoff, every 5 with the todo list alone; S-CM* prune+todo+reset, prune+todo+sum, prune+sum. The skeptic owns S-CM* (audit §6.2), the CM-arm author the four CM arms. CM0, O-state, CM-trim, CM-native and the topology arms are untuned. Dev noise (SE ≥ 4 pp per candidate) catches broken configurations, not small gains.
- **Probe** (`readiness/probe_openai.py --study study_g`): every Study G profile (Luna, Sol, Astra) is probed for honoured parameters; the API mode each model runs on in Inspect is recorded (a model on two modes is a problem); a native-compaction probe per agent model writes the CM-native support record. About $0.6 extra.
- **Budget:** $4,712 conservative / $3,161 expected; contingency $288.

**D-044 (2026-10-03, BUILD_PLAN B5: PREREGISTRATION_MAIN.md drafted).** The main pre-registration is written; its hypothesis table is generated from `analysis/main_hypotheses.py` and a test keeps them identical. Decisions made with it:
- **No pre-registered extension** for the main study: any further data is a new pre-registration. `analyze_main` analyses an extension run alone as primary-only if one is ever frozen.
- **The live S8k3 gets 5 runs** in its own Study C cell (`main.C.s8`, +$13): it runs only in Study C, so with 2 epochs its determinism metrics rested on 2 runs, against the brief's 5 per arm.
- **Real sizes:** the runner builds whole worlds of 12 tasks, so "100 tasks" is 108 (9 worlds), M4's "n = 50" is 60 and Study C's 15 is 24; the pre-registration states the real sizes.
- **Power figures:** the code table's (after D-033) are the ones that count: M3 0.87, M5 0.89 at Δ = 0 (±6 pp); K2's interaction 0.94 at −10 pp (0.82 at −8); K2-NI 0.78 at Δ = 0. D-033 quoted the earlier simulation's 0.84 / 0.87 / 0.86 / 0.79.
- ~~Not implemented: a per-run cache nonce and a wall-clock cap.~~ Both were added in B12 (D-046); the pre-registration describes them.
- `HYPOTHESES.md` rows K1, M1, M2, M3, M5, K4, C1 and A1 were aligned with the code table; the GATE row now names the gate's actual test (cluster-t, D-023).
- **Open for the user** (`[USER: …]`): the M-arm prompt author, the skeptic and the analyst. Two `[PILOT: …]` items are not yet written by the runner (`cap multiple`, `pilot σ and power`).
- **Budget:** $4,725 conservative / $3,168 expected; contingency $275.

**D-045 (2026-10-03, BUILD_PLAN B1 integration, rounds 2–3).** Merged at 61e4215; the full suite passes (877 passed, 2 skipped) and both offline rehearsals run every arm (main 132 s, Study G 57 s).
- **Freeze scope:** the gate and each study hash their own resolved slice of `run_plan.yaml` (their study section, `budget.total_usd`, `sample_cost_limit`, their allocation, the cuts touching their cells, and for Study G the `study_g` block), of `models.yaml` (the profiles they use) and of `model_costs.yaml` (those models' prices); their tuning grid, pre-registration and run outputs whole. Whole-file hashes are recorded for information only. A main or Study G plan edit after the gate's freeze no longer breaks it.
- **Spend:** an Inspect `on_model_usage` hook appends every call's usage and cost to `<log_dir>/usage_ledger.jsonl`; the registry, `program_spend` and the guards count the ledger beyond what the logs hold, with a cross-check. In the retry test the logs held $12 and the ledger added $4 for two errored attempts. Sessions are not double-counted against `f8_resume.unlogged`.
- **Selections:** each tuned arm runs in its own eval set with exactly its own selection's knobs (M1s under M1's); untuned arms get none. The shared-knob rule in the grid check is replaced by per-study own-knob checks (main: `knobs.candidate_problems`; Study G: `run_study.cm_candidate_problems`). S3s inherits the gate run's `S3s` selection (in every fingerprint, the freeze and PROVENANCE). Every key in a grid's `owners` (S1 included) needs an owner and sign-off, and an M arm owned by the skeptic is refused.
- **D-039 gate refined:** the token-cap multiple doubles only on token-limit hits above 10% (an Inspect token limit, or an agent stopped on `limit`), re-running only the arms over it, each round budget-checked; turn-cap hits are reported and never re-run, and the freeze refuses when any arm's turn-cap-hit rate exceeds 10% (the turn caps then need a decision). Arms the pilot does not run get projected rates: M1s = M1's, S8k3 = the share of S1 pilot samples above cap/3, Sol cells = the Luna rates. Pilot rates are measured against the test's caps: the pilot ran F2 and F3 uncapped (the micro-pilot never measured them), so an uncapped sample over the cap counts as a token hit, and re-runs use the test's caps; F7-100 and S5 on F1-32, which no pilot cell runs, are projected from the nearest measured level and from the micro-pilot. The gate's record (`config/cap_gate.json`, token / turn / other rates, projections labelled) is frozen; the main pilot writes the `cap multiple` and `pilot σ and power` pre-registration items (the power re-simulation runs in-process with the gate run's pilot σ). The possible re-runs (up to 2 pilot-sized rounds, about $36) are projected by the runner, not in `ape.budget`'s plan total.
- **Other fixes:** CM-native cells (Luna and Sol) and grid candidates refuse at preflight without a confirmed support record; every session task gets the checkpoint dir, seed base and the plan's T_abs; test-manifest log paths are run-relative; `runner.run_evals` runs `eval_set` in a copied context (Inspect left the caller's model roles set after an in-process run, which made `get_model(role="kg", required=True)` silently return a mock); `kb_react` ends the sample when the kg classify inside `search_kb` trips a limit.

**D-046 (2026-10-03, BUILD_PLAN B12 part 1: smoke checks, cache nonce, runaway guard).**
- **Live smoke for every new arm family** (each with a dry version on mock models): main `mas_orchestrator` (M1, F1-2), `mas_council` (M7), `mas_ensemble` (S8k3, F7-10), `mas_kg_workers` (M1k, F7-10, under both KG arms the gate can resolve); Study G `g_cm_sum`, `g_cm_todo` (one 6-case session, window 16K, threshold 4K so management fires) and `g_team_session` (M1 in a session). Each checks the arm's records (per-agent accounting sums to the sample's usage, no worker errors, sane stops; call kinds, metered management, probes kept out of the history, W enforced), the configured effort on every call, the cache nonce, and realised cost against its projection. `Study.smoke_checks` requires main's four and Study G's three, so a live study run refuses without them; the gate's required list is unchanged. Smoke projection $3.36 (gate $2.95 + studies $0.41) under the default `--max-usd 4`; the plan's `gate.smoke` line is raised to $3.4.
- **Per-run cache nonce** (`APE_CACHE_NONCE`, `agent/cache_nonce.py`): the study runner sets one seed per eval set (study, run, phase, group; per candidate in tuning); each task derives its nonce from it and its arm (main: and delivery, exposure). A `Run reference: <12 hex>` line starts every agent, multi-agent role, session, probe and management prompt; it is in each sample's metadata and in the session checkpoint key. kg-role calls carry none (their prompts belong to the gate-tuned KG arms). The gate never reads it, and without a seed every prompt is byte-identical to before (tested). Epochs of one arm share a nonce, so cache-hit claims use first epochs.
- **Runaway guard** (`budget.sample_working_limit`): Inspect `working_limit` (working time excludes connection and rate-limit waits) of 30 minutes at a 12-turn cap, scaled with the turn cap (F1-32 110 minutes), and 30 minutes + 10 per case for F8 sessions (40 cases: 7.2 hours). Inspect's task identifier includes it, so only the study runners apply it, never the gate. A hit ends the sample with limit `working`, counted as a cap hit; the D-045 gate reports it under "other" and never escalates on it.

**D-047 (2026-10-03, the independent review of a576a2c; BUILD_REVIEW.md).** Four A-level and twelve B-level findings; nothing sinks the design. Decisions taken with the fix round:
- **S9, M1s, M1k and M2 run under M1's selection:** one protocol variant for the whole chain S1 → S9 (DEC) → M1s (ISO) → M1 (CONC) → M1k (DEL) → M2 (SPEC), so each single-switch step, and M1's confirmatory contrast M1 − S9, differs in nothing else; S9, M1k and M2 are no longer tuned (`main.tune.a` = M1, M7; `main.tune.b` = M1). Separate picks were near-arbitrary anyway (tune SE ≈ 11 pp). M7 (a different protocol, compared with the S8 frontier) keeps its own tune.
- **Provider errors inside multi-agent arms propagate** (the sample errors and is retried, as S1's would and as the session team's already did) instead of becoming a scored failure or a lost vote.
- **Cap hits are failures and cast no vote** in the main analysis, as PREREGISTRATION_MAIN states (a capped F3 sample scored a success before); in sessions, the item in progress when a limit fires fails.
- **The frontier tests (M3, M2's frontier) must carry the matched cost's uncertainty**; if no correction restores nominal size, they become descriptive (decided when the fix reports).
- **Study G's t gets a small-sample correction and a minimum sessions per planned point** (below it: INCOMPLETE), so G-H3 holds 0.025.
- **The D-045 gate decides on one-sided 97.5% Clopper–Pearson lower bounds pooled per arm** (as the gate's PC5), not point estimates per arm-cell, which would escalate or refuse by chance at true rates of 2–5%; other-limit (working, cost guard) rates are gated the same way; S8k3 is projected from resampled triples of S1 samples.
- **Study G pilots its topology arms** in a short `g.pilot.topo` cell and computes session error, limit and overflow rates before the freeze.
- **Integrity:** a study run freezes a copy of the gate resolution, not the gate's report bytes; analyze refuses live on post-freeze changes unless a deviation is recorded (gate too); paid phases re-check smoke currency; pre-freeze phases carry code identity; allocations and concurrency are not frozen; `ape.budget` prices whole worlds as the runner runs them.

**D-048 (2026-10-03, user: M3 after the review's fixes).** M3 (communication ≈ 0 at matched cost: M7 − S8@M7, TOST ±6 pp) **stays confirmatory on the four Study A cells**, with its residual size documented. With the matched-cost correction (D-047) its type I is at nominal except up to 0.059 at the +6 pp boundary (κ = 0) and 0.055 at −6 pp (κ = 1) against 0.05. The cause is the two near-ceiling cells F1-2 and F2-2 (0.047 off the ceiling), not cost noise. Planned power at Δ = 0: 0.89 / 0.79 / 0.69 at κ = 0 / 1 / 3 (κ: how much more a failed run costs). The claim is worded as an equal-weight average over four cells, two near the ceiling, and read with the per-cell estimates. Rejected: testing only F1-32 and F2-10 (type I 0.030–0.041 but power 0.12–0.20) and making M3 descriptive. M2's frontier test is back at nominal (0.019–0.029) and stays confirmatory.

**D-049 (2026-10-03, Study G statistics after the review; amends D-032/D-038).** G-H3's t-tests ran at 0.026–0.040 against 0.025. The variance is unbiased (rms(se)/sd(est) = 1.00); the statistic's tails are heavier than t_df's, so no df correction helped (Bell–McCaffrey, G_eff − 2 and kurtosis-aware df each removed ≈ 0.001; a skewness-corrected t made the cost clause worse). Instead, **every confirmatory t (G-H2a per point, G-H3-pre, G-H3a, both G-H3b clauses) rejects at a calibrated level of 0.013** (`g_hypotheses.T_LEVEL`), with matching 97.4% intervals; the family α stays 0.025 and the sign-flip is still read at α. Across 72 scenarios on the reviewer's generators (16,000–20,000 studies each) the worst rates are 0.024 (G-H3) and 0.021 (G-H2a); the reviewer's own scripts, rerun unchanged, give 0.012–0.023. **Minimum sessions per planned point:** max(3, ⌈0.6 × planned⌉) (topology Luna 10/16, Sol 5/8; CM Luna 6/10, Sol 4/6, Astra 3/4); below it the point counts as missing and the family is INCOMPLETE. **Power now:** G-H2a 0.94 (Astra-limited); G-H3a 0.99 / 0.83 / 0.55 at isolation shares 0.75 / 0.70 / 0.65; G-H3b 0.56 at recovery 0.95 and cost ratio 0.45 (24 Luna sessions per point would give 0.67 for about +$51, a sizing choice left to the micro-pilot review, PREREGISTRATION_G §4.5). Also: the item in progress at a sample limit fails, and so does the report; report sections are isolated; a NaN cost makes G-H3b NOT_TESTABLE; an unreadable plan makes every confirmatory row INCOMPLETE offline and is an error live; an extension run is labelled a replication.

**D-050 (2026-10-03, the runner fixes after the review).** Merged at fa8db46.
- **Gate resolution:** a study run copies the gate's verdict, KG arm, the KG and S3s selection entries and the hash of the gate's freeze.json into `config/gate_resolution.json` at first resolution, and fingerprints and freezes that copy, never the gate's report bytes; a later, different resolution is refused live. The shared build ledger is no longer an analyze input for the gate or the studies.
- **Analyze refuses live** when the run is not frozen or a frozen file changed, unless `--deviation "<reason>"` is given; the deviation is stamped into decision.json and report.md (gate and studies).
- **Smoke currency:** every live paid phase checks that the preflight smoke still matches the code; a `--skip-smoke-check` override is tied to the code it was given for and never covers build-test or test.
- **Code identity** is in the params and log-dir keys of the pre-freeze paid phases, so a code change re-runs them in fresh directories.
- **Budget is not frozen** (`total_usd`, allocations, each profile's concurrency); a live freeze refuses when the test's projection exceeds what is left. `ape.budget` prices whole worlds as the runner runs them (+$111): **$4,836 conservative / $3,249 expected**, contingency $164; main allocation $1,225 (the allocations, 300 / 1,225 / 3,600, add to more than $5,000 by design: the program guard binds).
- **Pilot gate** (cap hits, errors, limits): decided per arm pooled over its pilot cells on the one-sided 97.5% Clopper–Pearson lower bound (as PC5); token-cap evidence escalates the multiple, turn-cap or other-limit evidence refuses the freeze, a point estimate over 10% without evidence is a recorded warning; S8k3 is projected from all triples of S1 pilot samples. Study G's micro-pilot writes session error, limit and overflow rates (`config/session_health.json`, frozen); overflow is an expected outcome for CM0, S1, M1, M2, CM-prune and CM-native, and refuses the freeze for the other arms. A short topology pilot cell (`g.pilot.topo`: S1, M1, M2, S-CM*; N 20; 2 sessions; $3.7) runs in Study G's micro-pilot.
- **Selections and tuning:** S9, M1s, M1k and M2 run under M1's selection; `main.tune.a` = {M1, M7}, `main.tune.b` = {M1}.
- **Smaller:** build-ledger entries carry their run's spend label; the runner index is read for the latest run's tasks only, and the working-limit config is in each group's key; each tuning system gets its own per-sample cost limit; the gate clears a shell `APE_CACHE_NONCE`; the freeze records each test group's per-sample cost limit and, with CM-native cells, the native support record. A parsed-YAML cache cut the offline rehearsals to 113 s (main) and 54 s (Study G).

**D-051 (2026-10-04, user: reduce cost).** The Astra topology cell `g.topo.astra` (S1 and M2 at Astra, 5 sessions × 2 epochs) is **switched off by default** (`enabled: false` in `config/run_plan.yaml`); the cell definition and every code path stay, so setting `enabled: true` restores it. It fed only G-H1's descriptive S1-pre sensitivity at Astra: G-H1's S-CM* reference and G-H3's pooled points (Luna-low, Luna-high, Sol) never used Astra. −$939 conservative / −$586 expected: **$3,897 / $2,664**, contingency $1,103 (22%). Considered and not taken: Study F Sol at 60 tasks (−$300), dropping CM-todo at Astra (−$284, a core arm at the top tier), 24 Luna topology sessions (+$51), and probing a cheaper service tier.

**D-052 (2026-10-04, user: reduce cost further without losing much).** −$901 conservative / −$617 expected: **$2,996 / $2,047**, contingency $2,004.
- **Study G runs one epoch with more sessions at Sol and Astra:** context management at Sol 8 × 1 (was 6 × 2) and at Astra 6 × 1 (was 4 × 2), topology at Sol 12 × 1 (was 8 × 2); −$601 / −$380. Sessions are the clusters, and the variance model puts epoch-to-epoch noise (0.2 logit) well below between-world noise (0.5), so extra sessions buy more than a second epoch. Simulated through g_power's real path (800 studies per design, 2,400 at the nulls): G-H2a 0.94 → 0.99 (Astra 6 sessions also reach the sign-flip level, 1/64); G-H3a 0.99 → 0.98 at a share of 0.8, 0.81 → 0.79 at 0.7; G-H3b 0.58 → 0.58; type I unchanged (≤ 0.017 at the calibrated 0.013). With epoch noise 2.5× the prior: G-H2a 0.92 → 0.97, G-H3a 0.97 → 0.96, G-H3b 0.46 → 0.45. Lost: whole-session pass^k and epoch agreement at Sol and Astra (descriptive; still at Luna). Cuts C2 and C3 were rewritten to record this (C2: Sol epochs 2 → 1 at 8 sessions; C3: Astra 6 sessions × 1 epoch at N 24). Minimum sessions per point: topology Sol 8 of 12, CM Sol 5 of 8, Astra 4 of 6.
- **Study F's Sol replication runs 60 tasks** (the first 5 of the 9 test worlds; cut C5 applied, also for `main.F.sol-m2`); −$300 / −$237. T1 is descriptive since D-033, so its intervals widen about 1.35×. `main_power.simulate` now draws a cell's tasks once at the largest size any tier runs it and gives a smaller tier the first worlds.
- **Not taken:** dropping CM-todo at Astra (a core arm at the top tier), lowering reasoning effort (changes the capability points), a smaller window (weakens the long-context effect the study measures). The uncut plan now fits $5,000 on its own; cuts C1, C2, C3 and C5 are applied by choice. Reproduction: `scratchpad/costcut/designs.py` and `epochsens.py` (session scratchpad).

## Open (needs user input)

| ID | Decision | Blocks |
|---|---|---|
| O-1 | **Create a new OpenAI project key with a hard spend limit** (recommended: staged limits, about $300 for the probe, smoke and gate, then raised per study up to the $5,000 program budget, D-021), then replace the contents of `.env`: `! echo "OPENAI_API_KEY=<new key>" > .env`. The shell's `OPENAI_KEY` is rejected (401 invalid_api_key). | E2–E5, L2, L4, L5, H4, the Luna build-quality check, everything live |
| O-3 | Names for the three roles in D-018. The skeptic must not be on the APG side. | Dev tuning (the skeptic owns the LightRAG/S3s candidates) |
| H5 | A human reviews `readiness/spotcheck.md` (30–45 min). | Pilot |
