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

## Open (needs user input)

| ID | Decision | Blocks |
|---|---|---|
| O-1 | Put `OPENAI_API_KEY` in `.env`. The shell has `OPENAI_KEY`; one way is `echo "OPENAI_API_KEY=$OPENAI_KEY" > .env`, run by the user so the key is never displayed. | E2–E5, L2, L4, L5, H4 |
| O-2 | Commit and tag `apg-eval-baseline` in EvolvingWisdomAgents (optional; the tree hash covers it for now). | A1 → ✅ |
| O-3 | `team` (including a skeptic not on the APG team) and `compute_budget_usd` for the brief's §0. | D1 |
| O-4 | Approval for live smoke tests (checkpoint 3, < $2). | L2, H4 |
