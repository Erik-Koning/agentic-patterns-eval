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

**D-009 (2026-09-30).** The PC1 anchor (GraphRAG-Bench Medical) follows the paper's LightRAG setup, with these documented deviations:
- LightRAG 1.5.7 instead of 1.2.5, so context caps are an approximate mapping.
- The judge prompts are the current repo versions (rewritten 2025-07-21). Three of the four published numbers predate the rewrite.
- Similarity uses our embedding model instead of bge-large-en-v1.5.
- Keyword extraction uses structured JSON output.
- The corpus is inserted whole and chunked by LightRAG, as in the paper, rather than per D-003. With per-chunk insertion, LightRAG 1.5.7 would list up to 75 chunk IDs on every entity line and use up the token caps.
- Why: PC1 validates *our LightRAG setup*, not the gate's shared-chunk protocol. If PC1 fails, these deviations are the first suspects.

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
- **Not affordable as planned:** D-017's Sol build path (+$905). It needs approval.
- **Recalibration:** after the gate pilot and Study G's micro-pilot, `python -m ape.budget calibrate` replaces the priors with measured calls and tokens per (arm, model, effort, cell, delivery). Cuts are then revisited in reverse order (C5 first). FX-6 refuses a phase whose projected cost exceeds `remaining()`.

## Open (needs user input)

| ID | Decision | Blocks |
|---|---|---|
| O-1 | **Create a new OpenAI project key with a hard spend limit** (recommended: the D-018 budget, $1,500), then replace the contents of `.env`: `! echo "OPENAI_API_KEY=<new key>" > .env`. The shell's `OPENAI_KEY` is rejected (401 invalid_api_key). | E2–E5, L2, L4, L5, H4, the Luna build-quality check, everything live |
| O-3 | Names for the three roles in D-018. The skeptic must not be on the APG side. | Dev tuning (the skeptic owns the LightRAG/S3s candidates) |
| H5 | A human reviews `readiness/spotcheck.md` (30–45 min). | Pilot |
