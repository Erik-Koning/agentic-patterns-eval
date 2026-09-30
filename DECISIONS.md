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

## Open (needs user input)

| ID | Decision | Blocks |
|---|---|---|
| O-1 | Put `OPENAI_API_KEY` in `.env`. The shell has `OPENAI_KEY`; one way is `echo "OPENAI_API_KEY=$OPENAI_KEY" > .env`, run by the user so the key is never displayed. | E2–E5, L2, L4, L5, H4 |
| O-2 | Commit and tag `apg-eval-baseline` in EvolvingWisdomAgents (optional; the tree hash covers it for now). | A1 → ✅ |
| O-3 | `team` (including a skeptic not on the APG team) and `compute_budget_usd` for the brief's §0. | D1 |
| O-4 | Approval for live smoke tests (checkpoint 3, < $2). | L2, H4 |
| O-5 | **Gate size.** The plan's starting point is 6 worlds per cell × 12 tasks × 3 epochs = 288 tasks per gate arm. `power_sim.py --mode ni` gives only **≈0.52 power at true Δ = 0** (σ_w = 0.5, σ_g = 0.3). For 80% power: **12 worlds/cell × 12 tasks** (576 tasks, 0.79; 0.88 if σ_g ≤ 0.15) or **16 × 12** (768 tasks, 0.91). Worlds matter far more than tasks per world. This is ≈1.8–2.5× the gate's run and build cost. Recommend 12 worlds/cell, re-simulated with pilot σ estimates before freezing GATE_PREREG. | GATE_PREREG, checkpoint 8 budget |
| O-6 | **Delivery modes.** Recommended (data-driven, see D-013): push and pull **co-primary for the F7 cells**, with the verdict reported per mode; F3 push only. That is ≈1.5× the push-only gate cost. Cheaper alternative: pull as a half-size secondary, which leaves the relational verdict dependent on the push-only assumption. | GATE_PREREG §3, gate budget |
