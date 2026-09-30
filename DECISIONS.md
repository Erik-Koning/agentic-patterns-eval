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

## Open (needs user input)

| ID | Decision | Blocks |
|---|---|---|
| O-1 | Put `OPENAI_API_KEY` in `.env`. The shell has `OPENAI_KEY`; one way is `echo "OPENAI_API_KEY=$OPENAI_KEY" > .env`, run by the user so the key is never displayed. | E2–E5, L2, L4, L5, H4 |
| O-2 | Commit and tag `apg-eval-baseline` in EvolvingWisdomAgents (optional; the tree hash covers it for now). | A1 → ✅ |
| O-3 | `team` (including a skeptic not on the APG team) and `compute_budget_usd` for the brief's §0. | D1 |
| O-4 | Approval for live smoke tests (checkpoint 3, < $2). | L2, H4 |
| O-5 | **Gate size.** The plan's starting point is 6 worlds per cell × 12 tasks × 3 epochs = 288 tasks per gate arm. `power_sim.py --mode ni` gives only **≈0.52 power at true Δ = 0** (σ_w = 0.5, σ_g = 0.3). For 80% power: **12 worlds/cell × 12 tasks** (576 tasks, 0.79; 0.88 if σ_g ≤ 0.15) or **16 × 12** (768 tasks, 0.91). Worlds matter far more than tasks per world. This is ≈1.8–2.5× the gate's run and build cost. Recommend 12 worlds/cell, re-simulated with pilot σ estimates before freezing GATE_PREREG. | GATE_PREREG, checkpoint 8 budget |
