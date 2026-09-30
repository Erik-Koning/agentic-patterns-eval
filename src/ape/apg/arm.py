"""APG delivery arms.

    APG-q / APG-s   authored graph (apg/author.py), compiled per query / per step
    APGo-q / S5o    oracle graph (apg/oracle.py), compiled per query / per step

Graph documents live in `indices/apg/{world_id}.{kind}.apg.json` with node embeddings
precomputed. Oracle graphs are built on demand; authored graphs must be built first.
"""

import copy
import hashlib
import json
import os
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


def tuned(doc: dict) -> dict:
    """Apply the dev-tuned APG knobs (GATE_PREREG §5) to a graph copy; they become part of its version."""
    d = copy.deepcopy(doc)
    routing = d.setdefault("defaults", {}).setdefault("routing", {})
    if os.environ.get("APE_APG_SHORTLIST_K"):
        routing["shortlistK"] = int(os.environ["APE_APG_SHORTLIST_K"])
    if os.environ.get("APE_APG_MIN_CONFIDENCE"):
        routing["minConfidence"] = float(os.environ["APE_APG_MIN_CONFIDENCE"])
    return d


class ApgArm:
    def __init__(self, name: str, per_step: bool, doc: dict, emb: EmbeddingCache, chunk_index: ChunkIndex, budget: int, fill: bool = False):
        self.name = name
        self.per_step = per_step
        doc = tuned(doc)
        self.graph = load_graph(doc)
        self.version = graph_version(doc)
        self.emb = emb
        self.chunk_index = chunk_index
        self.budget = budget
        self.fill = fill
        self.kg_model = get_model(role="kg", required=True)  # never silently fall back to the agent model
        self._node_facts = {n["id"]: self._facts([n["id"]]) for n in self.graph.dfs()}

    def _facts(self, node_ids: list[str]) -> list[str]:
        out: list[str] = []
        for nid in node_ids:
            props = self.graph.get(nid).get("props") or {}
            out.extend(props.get("factIds") or self.chunk_index.facts(props.get("sourceChunkIds") or []))
        return list(dict.fromkeys(out))

    def gold_nodes(self, task: TaskItem) -> set[str]:
        gold = set(task.gold_fact_ids)
        return {n for n, facts in self._node_facts.items() if gold & set(facts)}

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        c = await compile_context(self.graph, query, self.kg_model, self.emb, self.budget, fill=self.fill)
        gold = self.gold_nodes(task)
        matched = {m["nodeId"] for m in c.route["matches"]}
        return ContextResult(
            text=c.text,
            unit_ids=c.contributors,
            fact_ids=self._facts(c.contributors),
            tools=c.tool_allowlist,
            meta={
                "route": c.route,
                "graph_version": self.version,
                "truncated": len(c.truncated),
                "classify_error": c.classify_error,
                "kg_model": str(self.kg_model),
                "fill": self.fill,
                # NO-GO diagnosis (GATE_PREREG §8): where did the gold knowledge get lost?
                "gold": {
                    "nodes": sorted(gold),
                    "in_shortlist": bool(gold & set(c.route["shortlist"])),
                    "in_matches": bool(gold & matched),
                    "in_contributors": bool(gold & set(c.contributors)),
                },
            },
        )


async def build_apg_arm(arm: str, world: World, cfg: Config) -> ApgArm:
    kind, per_step = ARMS[arm]
    emb = embedding_cache(cfg)
    doc = await ensure_graph(world, kind, cfg, emb)
    fill = os.environ.get("APE_APG_FILL") == "1"
    return ApgArm(arm, per_step, doc, emb, ChunkIndex(chunk_world(world)), cfg.context_budget_tokens, fill=fill)
