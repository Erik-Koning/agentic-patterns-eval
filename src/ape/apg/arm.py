"""APG delivery arms.

    APG-q / APG-s   authored graph (apg/author.py), compiled per query / per step
    APGo-q / S5o    oracle graph (apg/oracle.py), compiled per query / per step

Graph documents live in `indices/apg/{world_id}.{kind}.apg.json` with node embeddings
precomputed. Oracle graphs are built on demand; authored graphs must be built first.
"""

import hashlib
import json
from pathlib import Path

from apg_core import Graph, load_graph, precompute_embeddings
from apg_core.outline import embed_text
from inspect_ai.model import get_model

from ..config import Config, embedding_cache
from ..kb.context import ContextResult
from ..kb.provenance import ChunkIndex
from ..llm.embeddings import CachedEmbeddingsConnector, EmbeddingCache
from ..worlds.render import chunk_world
from ..worlds.spec import TaskItem, World
from .adapter import compile_context
from .oracle import build_oracle, strip_embeddings

ARMS = {"APG-q": ("authored", False), "APG-s": ("authored", True), "APGo-q": ("oracle", False), "S5o": ("oracle", True)}


def graph_path(cfg: Config, world_id: str, kind: str) -> Path:
    return cfg.indices_dir / "apg" / f"{world_id}.{kind}.apg.json"


def graph_version(doc: dict) -> str:
    """Content hash of the graph without vectors (vectors are a pure function of text + model)."""
    return hashlib.sha256(json.dumps(strip_embeddings(doc), sort_keys=True).encode()).hexdigest()[:16]


async def embed_graph(doc: dict, emb: EmbeddingCache, context: dict) -> dict:
    """Embed every routable node's embedText and store vectors on the nodes (force: never stale, gap G6)."""
    g = Graph(doc)
    texts = [embed_text(g, n["id"]) for n in g.dfs() if n.get("routable") is not False]
    await emb.embed(texts, context=context)
    return precompute_embeddings(doc, CachedEmbeddingsConnector(emb), force=True)


async def ensure_graph(world: World, kind: str, cfg: Config, emb: EmbeddingCache) -> dict:
    path = graph_path(cfg, world.id, kind)
    if path.exists():
        return json.loads(path.read_text())
    if kind != "oracle":
        raise FileNotFoundError(f"{path} missing: build the authored graph first (ape.apg.author)")
    doc = await embed_graph(build_oracle(world, cfg.context_budget_tokens), emb, {"world": world.id, "system": "apg-oracle"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return doc


class ApgArm:
    def __init__(self, name: str, per_step: bool, doc: dict, emb: EmbeddingCache, chunk_index: ChunkIndex, budget: int):
        self.name = name
        self.per_step = per_step
        self.graph = load_graph(doc)
        self.version = graph_version(doc)
        self.emb = emb
        self.chunk_index = chunk_index
        self.budget = budget
        self.kg_model = get_model(role="kg")

    def _facts(self, node_ids: list[str]) -> list[str]:
        out: list[str] = []
        for nid in node_ids:
            props = self.graph.get(nid).get("props") or {}
            out.extend(props.get("factIds") or self.chunk_index.facts(props.get("sourceChunkIds") or []))
        return list(dict.fromkeys(out))

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        c = await compile_context(self.graph, query, self.kg_model, self.emb, self.budget)
        return ContextResult(
            text=c.text,
            unit_ids=c.contributors,
            fact_ids=self._facts(c.contributors),
            tools=c.tool_allowlist,
            meta={"route": c.route, "graph_version": self.version, "truncated": len(c.truncated), "classify_error": c.classify_error},
        )


async def build_apg_arm(arm: str, world: World, cfg: Config) -> ApgArm:
    kind, per_step = ARMS[arm]
    emb = embedding_cache(cfg)
    doc = await ensure_graph(world, kind, cfg, emb)
    return ApgArm(arm, per_step, doc, emb, ChunkIndex(chunk_world(world)), cfg.context_budget_tokens)
