"""APG delivery arms.

    APG-q / APG-s   authored graph (apg/author.py), compiled per query / per step
    APGo-q / S5o    oracle graph (apg/oracle.py), compiled per query / per step

Graph documents live in `indices/apg/{world_id}.{kind}.apg.json` with node embeddings
precomputed. Oracle graphs are built on demand; authored graphs must be built first.

Freshness (RELIABILITY_REVIEW K6). Every graph records in its `meta` the world version it was built from
(`worldHash`, tasks aside), the chunks (`chunksHash`), its author (`oracle`, the fake author or the build model and
effort), for authored graphs the authoring prompt and unit schema (`authorVersion`), the embedding model and the
vector dimension (`graph_meta`). `ensure_graph` checks a graph against the world and the embedding model in use
before an arm uses it (`graph_problems`): a stale or unlabelled oracle graph is rebuilt; a stale authored graph is
refused (it must be rebuilt by `ape.artifacts`, which pays for authoring); an authored graph whose LLM author is not
the current build model and effort is refused too. Query time checks the query vector against the graph's
dimension (`ape.apg.adapter`), so a vector mismatch fails loudly instead of routing on truncated cosines.

Provenance (K3). A compile's `fact_ids` are the facts its delivered text carries under the shared rule of
`ape.kb.provenance`, the same rule every arm is held to, not every fact of every source chunk of a contributing
leaf. The NO-GO diagnosis maps nodes to facts the same way, from each leaf's knowledge text.

Classify counters (K5) in each compile's meta: `classify_error` (the kg output could not be parsed),
`classify_fallback` (routing fell back for lack of a usable match), `classify_repaired` (repairs applied to the
output: code fences, an `id: title` node ID, a percentage confidence) and `classify_unknown_ids` (matches naming no
node of the graph).
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
from ..kb.provenance import fact_matcher
from ..llm.embeddings import CachedEmbeddingsConnector, EmbeddingCache
from ..worlds.spec import TaskItem, World
from .adapter import compile_context
from .oracle import build_oracle, strip_embeddings

ARMS = {"APG-q": ("authored", False), "APG-s": ("authored", True), "APGo-q": ("oracle", False), "S5o": ("oracle", True)}
ORACLE_AUTHOR = "oracle"


def graph_path(cfg: Config, world_id: str, kind: str) -> Path:
    return cfg.indices_dir / "apg" / f"{world_id}.{kind}.apg.json"


def graph_version(doc: dict) -> str:
    """Content hash of the graph without vectors (vectors are a pure function of text + model)."""
    return hashlib.sha256(json.dumps(strip_embeddings(doc), sort_keys=True).encode()).hexdigest()[:16]


def graph_meta(world: World, author: str | None, embedding_model: str) -> dict:
    """What a graph was built from; stored in its open `meta` mapping (the vector dimension is added once embedded).

    - `worldHash`: the world without its tasks (`World.artifact_hash`), so 20 or 40 tasks share a graph;
    - `chunksHash`: the shared chunks (`ape.worlds.render.chunks_hash`), so a renderer or chunker change marks it stale;
    - authored graphs also record `authorVersion`, the authoring prompt and unit schema (`author_prompt_version`)."""
    from ..worlds.render import chunk_world, chunks_hash

    meta = {"worldHash": world.artifact_hash(), "chunksHash": chunks_hash(chunk_world(world)), "author": author, "embeddingModel": embedding_model}
    if author != ORACLE_AUTHOR:
        from .author import author_prompt_version

        meta["authorVersion"] = author_prompt_version()
    return meta


def graph_dim(doc: dict) -> int | None:
    """The node vectors' dimension; ValueError when the graph mixes dimensions."""
    dims = {len(n["embedding"]) for n in doc.get("nodes", []) if n.get("embedding")}
    if len(dims) > 1:
        raise ValueError(f"graph {doc.get('graphId')} mixes vector dimensions {sorted(dims)}")
    return dims.pop() if dims else None


def _expected_llm_author() -> str | None:
    """The authored-graph author the current build settings would produce (None if they cannot be read)."""
    from ..models import build_settings
    from .author import llm_author_id

    try:
        return llm_author_id(*build_settings())
    except Exception:  # noqa: BLE001  (no profile: the author check is skipped, the other checks still run)
        return None


def graph_problems(doc: dict, world: World, emb: EmbeddingCache, kind: str) -> list[str]:
    """Why `doc` cannot serve `world` with this embedding model (empty: it can)."""
    meta = doc.get("meta") or {}
    problems = []
    if not meta:
        return ["no build meta (written before freshness checks)"]
    want = graph_meta(world, ORACLE_AUTHOR if kind == "oracle" else meta.get("author"), emb.model)
    if meta.get("worldHash") != want["worldHash"]:
        problems.append("built from another version of the world")
    if meta.get("chunksHash") != want["chunksHash"]:
        problems.append("built from other chunks (the renderer or chunker changed, or the graph predates this check)")
    if kind != "oracle" and meta.get("authorVersion") != want.get("authorVersion"):
        problems.append("authored with another authoring prompt or unit schema")
    if meta.get("embeddingModel") != emb.model:
        problems.append(f"embedded with {meta.get('embeddingModel')!r}, not {emb.model!r}")
    try:
        dim = graph_dim(doc)
    except ValueError as e:
        problems.append(str(e))
    else:
        if meta.get("embeddingDim") is not None and dim is not None and dim != meta["embeddingDim"]:
            problems.append(f"node vectors are {dim}-d but meta records {meta['embeddingDim']}-d")
    if kind == "authored":
        author = meta.get("author")
        llm_authored = author not in (None, ORACLE_AUTHOR) and not str(author).startswith("fake:")
        expected = _expected_llm_author() if llm_authored else None
        if llm_authored and expected is not None and author != expected:
            problems.append(f"authored by {author!r}, but the build settings now name {expected!r}")
    return problems


async def embed_graph(doc: dict, emb: EmbeddingCache, context: dict) -> dict:
    """Embed every routable node's embedText and store vectors on the nodes (force: never stale, gap G6)."""
    g = Graph(doc)
    texts = [embed_text(g, n["id"]) for n in g.dfs() if n.get("routable") is not False]
    await emb.embed(texts, context=context)
    return precompute_embeddings(doc, CachedEmbeddingsConnector(emb), force=True)


async def ensure_graph(world: World, kind: str, cfg: Config, emb: EmbeddingCache) -> dict:
    """The graph an arm uses, checked for freshness (module docstring); oracle graphs are (re)built on demand."""
    path = graph_path(cfg, world.id, kind)
    if path.exists():
        doc = json.loads(path.read_text())
        problems = graph_problems(doc, world, emb, kind)
        if not problems:
            return doc
        if kind != "oracle":
            raise RuntimeError(
                f"{path} is stale ({'; '.join(problems)}): rebuild it (python -m ape.artifacts --kinds apg), with the same "
                "build settings as the run: set APE_BUILD_FALLBACK=1 or the profile exactly as for the build (D-017)"
            )
    elif kind != "oracle":
        raise FileNotFoundError(f"{path} missing: build the authored graph first (ape.apg.author)")
    doc = await embed_graph(build_oracle(world, cfg.apg_budget_tokens), emb, {"world": world.id, "system": "apg-oracle"})
    doc["meta"] = graph_meta(world, ORACLE_AUTHOR, emb.model) | {"embeddingDim": graph_dim(doc)}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)  # atomic: a concurrent reader never sees half a graph
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
    def __init__(self, name: str, per_step: bool, doc: dict, emb: EmbeddingCache, world: World, budget: int, fill: bool = False):
        self.name = name
        self.per_step = per_step
        self.kg_model = get_model(role="kg", required=True)  # never silently fall back to the agent model
        doc = tuned(doc)
        self.graph = load_graph(doc)
        self.version = graph_version(doc)
        self.emb = emb
        self.matcher = fact_matcher(world)
        self.budget = budget
        self.fill = fill
        self._node_facts = {n["id"]: self._facts_of(n) for n in self.graph.dfs()}

    def _facts_of(self, node: dict) -> list[str]:
        """The facts a node's knowledge carries (the shared rule); an oracle node without text falls back to its factIds."""
        text = ((node.get("prompt") or {}).get("slots") or {}).get("knowledge") or ""
        return self.matcher.delivered(text) if text.strip() else list((node.get("props") or {}).get("factIds") or [])

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
            fact_ids=self.matcher.delivered(c.text),
            tools=c.tool_allowlist,
            meta={
                "route": c.route,
                "graph_version": self.version,
                "truncated": len(c.truncated),
                "classify_error": c.classify_error,
                "classify_fallback": c.classify_fallback,
                "classify_repaired": c.classify_repaired,
                "classify_unknown_ids": c.classify_unknown_ids,
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
    return ApgArm(arm, per_step, doc, emb, world, cfg.apg_budget_tokens, fill=fill)
