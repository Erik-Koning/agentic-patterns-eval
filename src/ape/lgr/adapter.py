"""LightRAG delivery arms, queried inside Inspect's event loop.

    LGR-q / LGR-s     index from LightRAG's own LLM extraction, per query / per step
    LGRo-q / LGRo-s   oracle custom-KG index (diagnostic)

Each compile makes exactly one keyword-extraction call, through Inspect's "kg" model
role so it is metered like APG's classify call. `aquery_data` supplies provenance;
the native context string is then fetched with those keywords pre-filled, which
skips a second extraction.
"""

import json
import os

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ResponseSchema, get_model
from inspect_ai.util import JSONSchema
from lightrag import QueryParam

from ..config import Config, embedding_cache
from ..kb.context import ContextResult
from ..kb.provenance import ChunkIndex
from ..worlds.render import chunk_world
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
    """Defaults; the skeptic tunes these on dev (GATE_PREREG fairness rule 1)."""
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
    def __init__(self, name: str, per_step: bool, rag, chunk_index: ChunkIndex, params: dict, manifest: dict, kg_model: str = ""):
        self.name = name
        self.kg_model = kg_model
        self.per_step = per_step
        self.rag = rag
        self.chunk_index = chunk_index
        self.params = params
        self.manifest = manifest

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        result = await self.rag.aquery_data(query, QueryParam(**self.params))
        data, meta = result.get("data", {}), result.get("metadata", {})
        kw = meta.get("keywords", {})
        hl, ll = kw.get("high_level") or [], kw.get("low_level") or []
        if not hl and not ll:
            ll = [query]  # empty keywords would trigger a second, unmetered-by-design extraction
        text = await self.rag.aquery(query, QueryParam(**self.params, only_need_context=True, hl_keywords=hl, ll_keywords=ll))
        entities, relations, chunks = data.get("entities", []), data.get("relationships", []), data.get("chunks", [])
        sources = [p for rec in (*entities, *relations, *chunks) for p in _paths(rec) if p in self.chunk_index.by_id]
        return ContextResult(
            text=str(text or ""),
            unit_ids=[e["entity_name"] for e in entities] + [f"{r['src_id']}->{r['tgt_id']}" for r in relations] + [c["chunk_id"] for c in chunks],
            fact_ids=self.chunk_index.facts(list(dict.fromkeys(sources))),
            tools=None,
            meta={
                "lightrag": {"mode": self.params["mode"], "keywords": {"high": hl, "low": ll}, "counts": [len(entities), len(relations), len(chunks)]},
                "index_hash": self.manifest.get("index_hash"),
                "kg_model": self.kg_model,
            },
        )


async def build_lgr_arm(arm: str, world: World, cfg: Config) -> LgrArm:
    kind, per_step = ARMS[arm]
    wd = index_dir(cfg, world.id, kind)
    if not (wd / "ape_manifest.json").exists():
        raise FileNotFoundError(f"{wd} missing: run `python -m ape.lgr.build --kind {kind}` first")
    manifest = read_manifest(wd)
    if manifest["world_hash"] != world.content_hash():
        raise RuntimeError(f"{wd} was built from a different world version")
    llm = kg_llm_func()
    rag = await open_rag(wd, world.id, llm, embedding_cache(cfg), query_time=True)
    return LgrArm(arm, per_step, rag, ChunkIndex(chunk_world(world)), query_params(cfg.context_budget_tokens), manifest, kg_model=llm.model_name)
