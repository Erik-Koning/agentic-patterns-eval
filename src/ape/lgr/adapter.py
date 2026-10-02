"""LightRAG delivery arms, queried inside Inspect's event loop.

    LGR-q / LGR-s     index from LightRAG's own LLM extraction, per query / per step
    LGRo-q / LGRo-s   oracle custom-KG index (diagnostic)

Each compile makes exactly one keyword-extraction call, through Inspect's "kg" model
role so it is metered like APG's classify call. `aquery_data` supplies the delivered units;
the native context string is then fetched with those keywords pre-filled, which
skips a second extraction.

Provenance (RELIABILITY_REVIEW K3): a compile's `fact_ids` are the facts its context text carries under the shared
rule of `ape.kb.provenance`, not every fact of every chunk an entity or relation was extracted from (a generic
entity can name dozens of source chunks while its description restates a few facts).

Keyword failures: when the kg model's reply yields no keywords (unparseable, a refusal, empty lists), the compile
falls back to the query itself as the low-level keyword, as before, and its meta says so (`keyword_fallback`,
`keyword_error`); the units are then fetched with the fallback keywords, so they match the delivered text.

Freshness (K6): an index is used only if its manifest records this world version, kind, embedding model and, for
extract indices, the current build model and effort (`ape.lgr.build.build_key`).
"""

import os

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import JSONSchema
from lightrag import QueryParam

from ..config import Config, embedding_cache
from ..kb.context import ContextResult
from ..kb.provenance import FactMatcher, fact_matcher
from ..worlds.spec import TaskItem, World
from .common import SEP, index_dir, open_rag, read_manifest

ARMS = {"LGR-q": ("extract", False), "LGR-s": ("extract", True), "LGRo-q": ("oracle", False), "LGRo-s": ("oracle", True)}
KEYWORDS_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["high_level_keywords", "low_level_keywords"],
        "properties": {
            "high_level_keywords": {"type": "array", "items": {"type": "string"}},
            "low_level_keywords": {"type": "array", "items": {"type": "string"}},
        },
    }
)


def kg_llm_func():
    """LightRAG `llm_model_func` for query time, routed through Inspect (role "kg")."""
    model = get_model(role="kg", required=True)  # never silently fall back to the agent model

    async def llm(prompt: str, system_prompt: str | None = None, history_messages: list | None = None, **kwargs) -> str:
        messages = ([ChatMessageSystem(content=system_prompt)] if system_prompt else []) + [ChatMessageUser(content=prompt)]
        config = GenerateConfig()
        if kwargs.get("response_format") or kwargs.get("keyword_extraction"):
            config = GenerateConfig(response_schema=ResponseSchema(name="keywords", json_schema=KEYWORDS_SCHEMA, strict=True))
        return (await model.generate(messages, config=config)).completion

    llm.model_name = str(model)
    return llm


def query_params(budget: int) -> dict:
    """Defaults; the skeptic tunes these on dev (GATE_PREREG fairness rule 1). `budget` is the LightRAG arms'
    own knob, `Config.lgr_budget_tokens` (APE_LGR_BUDGET, else APE_CONTEXT_BUDGET); explicit APE_LGR_* caps win."""
    return {
        "mode": os.environ.get("APE_LGR_MODE", "mix"),
        "top_k": int(os.environ.get("APE_LGR_TOP_K", "20")),
        "chunk_top_k": int(os.environ.get("APE_LGR_CHUNK_TOP_K", "10")),
        "max_entity_tokens": int(os.environ.get("APE_LGR_ENTITY_TOKENS", str(budget // 3))),
        "max_relation_tokens": int(os.environ.get("APE_LGR_RELATION_TOKENS", str(budget // 3))),
        "max_total_tokens": int(os.environ.get("APE_LGR_TOTAL_TOKENS", str(budget * 4))),
        "enable_rerank": False,  # D-004
    }


def _paths(record: dict) -> list[str]:
    return [p for p in str(record.get("file_path", "")).split(SEP) if p]


class LgrArm:
    def __init__(self, name: str, per_step: bool, rag, matcher: FactMatcher, params: dict, manifest: dict, kg_model: str = ""):
        self.name = name
        self.kg_model = kg_model
        self.per_step = per_step
        self.rag = rag
        self.matcher = matcher
        self.params = params
        self.manifest = manifest

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        result = await self.rag.aquery_data(query, QueryParam(**self.params))
        data, meta = result.get("data", {}), result.get("metadata", {})
        kw = meta.get("keywords", {})
        hl, ll = kw.get("high_level") or [], kw.get("low_level") or []
        fallback = not hl and not ll
        if fallback:
            ll = [query]  # empty keywords would trigger a second, unmetered-by-design extraction
            # The units of the context actually delivered: keywords pre-filled, so no second kg call.
            result = await self.rag.aquery_data(query, QueryParam(**self.params, hl_keywords=hl, ll_keywords=ll))
            data = result.get("data", {})
        text = await self.rag.aquery(query, QueryParam(**self.params, only_need_context=True, hl_keywords=hl, ll_keywords=ll))
        text = str(text or "")
        entities, relations, chunks = data.get("entities", []), data.get("relationships", []), data.get("chunks", [])
        return ContextResult(
            text=text,
            unit_ids=[e["entity_name"] for e in entities] + [f"{r['src_id']}->{r['tgt_id']}" for r in relations] + [c["chunk_id"] for c in chunks],
            fact_ids=self.matcher.delivered(text),
            tools=None,
            meta={
                "lightrag": {
                    "mode": self.params["mode"],
                    "keywords": {"high": hl, "low": ll},
                    "keyword_fallback": fallback,
                    "keyword_error": "the kg model's reply yielded no keywords; the query itself was used" if fallback else None,
                    "counts": [len(entities), len(relations), len(chunks)],
                    "source_chunks": len({p for rec in (*entities, *relations, *chunks) for p in _paths(rec)}),
                },
                "index_hash": self.manifest.get("index_hash"),
                "kg_model": self.kg_model,
            },
        )


def manifest_problems(manifest: dict, world: World, kind: str, embedding_model: str) -> list[str]:
    """Why an index's manifest does not match what this run would build (empty: it matches)."""
    from .build import build_key

    want = build_key(world, kind, embedding_model)
    return [f"{k} is {manifest.get(k)!r}, expected {v!r}" for k, v in want.items() if manifest.get(k) != v]


async def build_lgr_arm(arm: str, world: World, cfg: Config) -> LgrArm:
    kind, per_step = ARMS[arm]
    wd = index_dir(cfg, world.id, kind)
    if not (wd / "ape_manifest.json").exists():
        raise FileNotFoundError(f"{wd} missing: run `python -m ape.lgr.build --kind {kind}` first")
    manifest = read_manifest(wd)
    if manifest.get("world_hash") != world.content_hash():
        raise RuntimeError(f"{wd} was built from a different world version")
    emb = embedding_cache(cfg)
    if problems := manifest_problems(manifest, world, kind, emb.model):
        raise RuntimeError(f"{wd} does not match this run's build settings ({'; '.join(problems)}): rebuild it, or set APE_BUILD_FALLBACK / the profile as for the build")
    llm = kg_llm_func()
    rag = await open_rag(wd, world.id, llm, emb, query_time=True)
    return LgrArm(arm, per_step, rag, fact_matcher(world), query_params(cfg.lgr_budget_tokens), manifest, kg_model=llm.model_name)
