# Readiness audit

Status key: ✅ pass · ⚠️ pass with caveat or partial · ❌ fail · ⛔ blocked (waiting on another item or on the user) · ⏳ not started.

**(B)** marks an item that blocks the next phase. Every (B) item must be ✅ before the micro-pilot (checkpoint 6).

Last updated: 2026-09-30.

## Environment

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| E1 (B) | Venv imports all dependencies; lockfile committed | ✅ | Python 3.14.0. inspect-ai 0.3.273, lightrag-hku 1.5.7, apg-core 0.1.0, openai 3.22.1 import cleanly. `uv.lock` is committed. Python 3.12 is **not** usable; see G0 and D-001. | — |
| E2 (B) | `OPENAI_API_KEY` in gitignored `.env`; `models.list()` succeeds | ⛔ | Only `OPENAI_KEY` is set in the shell. The SDK and Inspect read `OPENAI_API_KEY`. | user |
| E3 (B) | Model IDs chosen per role (agent, kg, build, embeddings); honoured params probed | ⛔ | Blocked on E2. Probe script: `readiness/probe_openai.py` (to be written). | — |
| E4 | Rate-limit tier / TPM fits the concurrency plan | ⛔ | Blocked on E2. | — |
| E5 (B) | Price table (source URL + date) in Inspect's model-cost config | ⛔ | Blocked on E3. | — |

## APG

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| A1 (B) | Commit or tree hash recorded | ⚠️ | Tree hashes are in `PROVENANCE.md`. The repo has **no commits**, so a commit + `apg-eval-baseline` tag is still preferred. | user |
| A2 (B) | APG pytest passes, including 66 conformance fixtures | ✅ | `123 passed in 0.09s` on 3.14, run against the installed copy (`-p no:cacheprovider`, no writes to the APG repo). **Fails to import on 3.12** (G0). | — |
| A3 (B) | 1,000-leaf graph: validate < 5 s, route p95 < 1 s, compose < 50 ms, outline < 2.5k tokens | ✅ | `readiness/bench_apg_scale.py`: validate 0.004 s; route p50 0.155 s / p95 0.165 s; compose p95 0.31 ms; shortlisted outline ≤ 504 tokens. **Without embeddings the outline is 42,000 tokens** (G9). | — |
| A4 | Gap register with workarounds | ✅ | See the gap register below. | — |
| A5 | Ported classify prompt string-equal to TS `CLASSIFY_SYSTEM` | ⏳ | Done in Phase 2, module 5. | — |

## LightRAG

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| L1 (B) | API present in 1.5.7 | ✅ | Present: `aquery_data`, `ainsert_custom_kg`, `initialize_storages`, `initialize_pipeline_status`. `QueryParam` has `mode` (default `mix`), `only_need_context`, `top_k` (40), `chunk_top_k` (20), `max_entity_tokens` (6000), `max_relation_tokens` (8000), `max_total_tokens` (30000). `ainsert(input, ids=, file_paths=)` is available for provenance. Default storages are JSON, NanoVectorDB and NetworkX. **`enable_rerank` defaults to True** (D-004). | — |
| L2 (B) | 5-document live build: entities non-empty, `source_id` maps to our chunk IDs | ⛔ | Needs E2 and approval (< $1). | — |
| L3 (B) | 20 concurrent in-loop queries, no loop errors; workspaces disjoint | ⏳ | Can partly run with a fake LLM function in Phase 2, module 6. | — |
| L4 | Query cache off gives two metered keyword calls for two identical queries | ⛔ | Needs E2. | — |
| L5 | Realized context within ±10% of `max_total_tokens` | ⛔ | Needs E2. | — |

## Design documents

| ID | Check | Status | Evidence / notes | Owner |
|---|---|---|---|---|
| D1 (B) | `ORCHESTRATOR_BRIEF_v2.md` §0 filled | ⚠️ | Filled: `kg_framework`, `kg_ingestion_pipeline`, provider. **Still TODO-BLOCKING:** `team` (including the skeptic) and `compute_budget_usd`. Model IDs wait on E3. | user |
| D2 | Claude-specific text adapted for OpenAI | ⏳ | Sampling rejection and preserved-thinking notes in §6.3 and §5.3 of the brief. | — |
| D3 | `EVAL_DESIGN.md` flagged for OpenAI re-pricing | ⏳ | | — |

## Harness (Phase 1b: after integration, before any pilot)

| ID | Check | Status |
|---|---|---|
| H1 (B) | API verification table with doc URLs | ⏳ |
| H2 (B) | Mock dry run of every arm × family; required log fields present | ⏳ |
| H3 (B) | Classify/keyword calls appear as ModelEvents, one per compile | ⏳ |
| H4 | Live check: OpenAI accepts tool sets that change across turns | ⛔ (needs E2) |
| H5 | Human spot-check of 20 tasks per family | ⏳ |

---

## Gap register (A4)

Every APG gap, what it affects, and how the harness handles it. Upstream fixes change APG's tree hash and so require a new pin (`PROVENANCE.md`).

| ID | Gap | Evidence | Effect | Harness workaround |
|---|---|---|---|---|
| G0 | `apg_core` fails to import on Python < 3.14 even though it declares `>=3.11`. | `connectors.py:175-193`: the class-scope `def list(...)` shadows the builtin, so `list[dict]` annotations raise `TypeError` under eager annotation evaluation. | Portability only. Results are unaffected. | Use Python 3.14 (D-001). Suggested upstream fix: `from __future__ import annotations` in `connectors.py`. |
| G1 | `compose` never includes descendants of the target; only the root→leaf path contributes. | `compose.py:37`; determinism spec §5 | Content placed on a category node's children is invisible when the category is routed to. | Authoring rule: policy text goes on **routable leaves**, and category nodes are thin descriptors. |
| G2 | `toolAllowlist` is intersected over the primary path only; secondary matches' allowlists are ignored. | `compose.py:321-330` | Multi-domain F3 tasks can lose the second domain's tools. | Allowlists on leaves only. Measure it as an APG gap; don't patch it. |
| G3 | No exception or override relation between sibling policies. | Schema has only merge modes along the path + `composition.priority` | Relational F7 (precedence/exceptions) must be encoded indirectly. | Leaf-level `bring` edges with `recursiveBring`; precedence stated in the text. |
| G4 | `route` has no subtree or scope parameter. | `router.py:29` | Specialist agents can't route inside their own subtree. | Build a sub-graph document per specialist. |
| G5 | Module-global connector registry; stateful `ScriptedLlm` queues. | `connectors.py:13-51` | Cross-talk risk under Inspect's concurrent samples. | Pass a per-call connectors dict and a fresh `ScriptedLlm` per call. Never use `bind_connectors`. |
| G6 | `precompute_embeddings` fills only missing vectors, so vectors go stale after text edits. | `embed.py:20` | Wrong shortlist after re-authoring. | Always use `force=True` after a build. |
| G7 | The Python validator has no JSON-Schema check (only TS/Ajv does). | `tests/test_templates.py:3-4` | Malformed authored graphs could pass `validate_graph`. | Add a `jsonschema` check against `schema/apg.schema.json`. |
| G8 | No Python LLM or embeddings connector. | `connectors.py:35-67` (only `ScriptedLlm` / `MapEmbeddings`) | The kernel can't route with a real model. | Adapter-side classify through Inspect (record/replay) and cached OpenAI embeddings. |
| G9 | Without embeddings, the outline lists every eligible node. | A3: 42,000 tokens at 1,000 leaves vs 504 shortlisted | Routing cost grows linearly with the number of policies. | Embeddings are mandatory; tune `shortlistK` on dev. |
| G10 | Cosine scoring is pure Python, O(N·dim). | A3: route p50 155 ms at 1,000 × 1536 | Latency. | Acceptable. Run in a worker thread. |
| G11 | The compose token budget is not a hard cap: path constraints are never dropped and separators aren't counted. | determinism spec §5; `compose.py:211` | Realized context can exceed `maxPromptTokens`. | Pass a tiktoken `countTokens`; match arms on **realized** tokens. |
| G12 | `apply_changeset` deep-copies and rebuilds on every op, which is quadratic. | `mutation.py:19, 256-259` | Slow bulk authoring. | Build raw documents, then run one validation. |
| G13 | The TS `buildMonolith` differs from compose: it drops outputFormat and recursive brings and uses a different slot order. | `examples/wisdom-chat/src/lib/compare.ts:136-152` | Unequal content between the monolith and graph arms. | Port it with parity fixes (module 3). |
| G14 | APG has no license. | `README.md:146-147` | Blocks an open release. | Internal use only until the owner decides. |
| G15 | Neither repo has git history. | `git rev-parse HEAD` fails | Reproducibility. | Tree-hash pin (`PROVENANCE.md`) until a commit/tag exists. |
