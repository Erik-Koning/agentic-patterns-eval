"""LLM authoring of APG graphs from the shared chunks: the realistic KG arm (APG-q / APG-s).

APG has no document ingestion; in production graphs are authored. This pipeline
plays the author: it sees only the rendered chunks (never the world spec).

1. One category per source document (the structure an author sees).
2. Per chunk, the build model splits the text into self-contained units (rule,
   exception, procedure, tool, event) and names the identifiers each unit declares
   and references, plus the tools it instructs the agent to call.
3. References become leaf-level `bring` edges; they are made reciprocal except
   toward tool-spec leaves (so an exception and its policy bring each other).
4. `toolAllowlist` goes on leaves only (gap G2).
5. The raw document is validated once (semantic + JSON-Schema) and embedded.
6. The graph's open `meta` mapping (and its report) records what it was built from: the
   world's content hash, the author, the embedding model and its dimension. `ape.artifacts` skips a
   world whose graph records the same (`authored_graph_current`); `ape.apg.arm` refuses a stale one.

Robustness (RELIABILITY_REVIEW K1, K2, K4):
- Each chunk's reply is checked (`ape.llm.build_client.BuildLlm.json` re-asks refusals, empty and truncated
  replies, and repairs invalid JSON) and validated against `UNIT_SCHEMA`.
- A chunk that still fails, or returns no knowledge for text that carries facts, is *lost*. A world tolerates
  `max_failed_chunks(n)` lost chunks (APE_BUILD_MAX_FAILED_SHARE, default 2%), listed in the report's `lost_chunks`.
  One more and the world fails with every reason, and the calls still running are cancelled. Any other error
  (authentication, the network after the client's own retries) is fatal at once.
- Units are cached per chunk (`apg_author_units.sqlite` in the cache dir), keyed by the author (model and effort),
  the prompt and the schema, so a re-run after a failure pays only for chunks without a usable reply. A lost
  chunk is never cached.
- A graph with no knowledge units is never written.
- Identifiers are normalised (`ape.build_quality.normalize_ident`) before linking and measuring, so "Policy P-1035"
  and "P-1035" are the same node; the prompt asks for bare IDs anyway.
- The report's `id_coverage` is None for worlds without spec IDs (F1/F2/F5/F8) instead of a vacuous 1.0, and
  `fact_coverage` (`ape.kb.provenance`) is the share of the knowledge base's facts that some leaf carries.

    python -m ape.apg.author --split dev --family F7     # build model and effort from config/models.yaml

The CLI delegates to `ape.artifacts --kinds apg` (parallel worlds, skips current graphs).
"""

import asyncio
import hashlib
import json
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

import jsonschema
from apg_core import validate_graph

from ..build_quality import apg_spec_ids, canon_id, id_coverage, normalize_ident, rendered_spec_ids
from ..config import BuildConcurrency, Config, embedding_cache
from ..kb.provenance import fact_matcher
from ..llm.build_client import BuildLlm, BuildResponseError, max_failed_chunks
from ..llm.embeddings import SQLITE_TIMEOUT_S, _enable_wal
from ..llm.ledger import Ledger
from ..worlds.render import Chunk, chunk_world
from ..worlds.spec import World
from .arm import embed_graph, graph_dim, graph_meta, graph_path
from .schema import schema_errors

AUTHOR_SYSTEM = (
    "You are authoring a knowledge graph for an operations assistant from company documents. "
    "Split the document excerpt into self-contained knowledge units: one rule, exception, procedure, "
    "tool description or dated event per unit. For each unit give:\n"
    "- knowledge: the unit's full content, preserving every condition, number, code, name and date exactly; "
    "it must be understandable on its own.\n"
    "- title: a short, specific title a router can match against.\n"
    "- description: one sentence saying when this unit applies.\n"
    "- declares: identifiers this unit defines (policy, exception, procedure or tool names/IDs); empty if none.\n"
    "- references: identifiers of other rules or tools this unit mentions but does not define.\n"
    "- tools: names of tools this unit instructs the agent to call; empty if none.\n"
    "Write identifiers and tool names exactly as they appear in the excerpt, without a word in front: "
    "`P-1035`, not `Policy P-1035`; `duplicate_charge_credit`, not `the duplicate_charge_credit tool`.\n"
    "Do not invent content that is not in the excerpt."
)
_STR_ARRAY = {"type": "array", "items": {"type": "string"}}
UNIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["units"],
    "properties": {
        "units": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["declares", "title", "description", "knowledge", "references", "tools"],
                "properties": {
                    "declares": _STR_ARRAY,
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "knowledge": {"type": "string"},
                    "references": _STR_ARRAY,
                    "tools": _STR_ARRAY,
                },
            },
        }
    },
}

AuthorFn = Callable[[str, str], Awaitable[dict]]  # (system, user) -> {"units": [...]}


FAKE_AUTHOR_ID = "fake:perfect_author"
UNIT_CACHE = "apg_author_units.sqlite"


class UnitShapeError(ValueError):
    """An author reply that is not `{"units": [...]}` under UNIT_SCHEMA."""


class AuthoringError(RuntimeError):
    """A world's authoring failed: too many lost chunks, or nothing to build a graph from."""


def check_units(out: object) -> list[dict]:
    """The reply's units, validated against UNIT_SCHEMA (strict structured output should already guarantee it)."""
    if not isinstance(out, dict) or not isinstance(out.get("units"), list):
        raise UnitShapeError(f"reply is not {{'units': [...]}}: {str(out)[:120]}")
    errors = [e.message for e in jsonschema.Draft202012Validator(UNIT_SCHEMA).iter_errors(out)]
    if errors:
        raise UnitShapeError(f"reply breaks the unit schema: {errors[:3]}")
    return out["units"]


def _has_knowledge(units: list[dict]) -> bool:
    return any(str(u.get("knowledge", "")).strip() for u in units)


class UnitCache:
    """Per-chunk authored units, shared by build processes (sqlite in WAL mode, like the embedding cache)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=SQLITE_TIMEOUT_S, isolation_level=None)
        _enable_wal(self._db)
        self._db.execute("CREATE TABLE IF NOT EXISTS units (k TEXT PRIMARY KEY, v TEXT NOT NULL)")

    @staticmethod
    def key(author_id: str, system: str, user: str) -> str:
        payload = json.dumps({"author": author_id, "system": system, "user": user, "schema": UNIT_SCHEMA}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> list[dict] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM units WHERE k = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, units: list[dict]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO units (k, v) VALUES (?, ?)", (key, json.dumps(units)))

    def close(self) -> None:
        self._db.close()


def llm_author_id(model: str, reasoning_effort: str | None = None) -> str:
    """How a graph records its LLM author (build model and effort): a different author means a rebuild."""
    return f"{model}@{reasoning_effort}" if reasoning_effort else model


def build_llm_author(model: str, ledger: Ledger, world_id: str, reasoning_effort: str | None = None) -> AuthorFn:
    llm = BuildLlm(model, ledger, {"world": world_id, "system": "apg-author"}, reasoning_effort=reasoning_effort)

    async def author(system: str, user: str) -> dict:
        return await llm.json(system, user, "units", UNIT_SCHEMA)

    return author


def _idents(values: list[str]) -> list[str]:
    return list(dict.fromkeys(i for i in (normalize_ident(v) for v in values) if i))


def assemble(world: World, chunks: list[Chunk], units_per_chunk: list[list[dict]], budget: int, lost: dict[str, str] | None = None) -> tuple[dict, dict]:
    """The graph document and its report from each chunk's units; `lost` maps chunks without a usable reply to why."""
    titles = {d.id: d.title for d in world.documents}
    nodes = [{"id": "root", "parentId": None, "routable": False, "title": "Company knowledge"}]
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        nodes.append({"id": f"doc-{doc_id}", "parentId": "root", "type": "category", "routable": False, "title": titles[doc_id]})

    leaves: list[dict] = []
    for chunk, units in zip(chunks, units_per_chunk, strict=True):
        for u in units:
            if not u.get("knowledge", "").strip():
                continue
            leaves.append(
                {
                    "id": f"u{len(leaves):05d}",
                    "parentId": f"doc-{chunk.doc_id}",
                    "type": "category",
                    "routable": True,
                    "title": u["title"].strip() or u["knowledge"][:80],
                    "description": u["description"].strip(),
                    "prompt": {"slots": {"knowledge": u["knowledge"].strip()}},
                    "props": {"sourceChunkIds": [chunk.id], "declares": _idents(u["declares"])},
                    "_refs": _idents(u["references"]),
                    "_tools": _idents(u["tools"]),
                }
            )

    # Identifiers link by their canonical key: "Policy P-1035", "p-1035" and "P-1035" are one ID.
    declared: dict[str, list[str]] = {}
    spelling: dict[str, str] = {}  # canonical key -> the first declared spelling (allowlists use it)
    for leaf in leaves:
        for ident in leaf["props"]["declares"]:
            declared.setdefault(canon_id(ident), []).append(leaf["id"])
            spelling.setdefault(canon_id(ident), ident)
    # Only tools some leaf actually defines count as tools; stray words an author puts in
    # `tools` (e.g. "any" from "do not call any other tool") never reach an allowlist.
    tool_keys = {canon_id(t) for leaf in leaves for t in leaf["_tools"] if canon_id(t) in declared}
    tool_leaves = {lid for k in tool_keys for lid in declared[k]}

    brings: dict[str, list[str]] = {leaf["id"]: [] for leaf in leaves}
    unresolved = 0
    for leaf in leaves:
        for ref in leaf["_refs"] + leaf["_tools"]:
            targets = [t for t in declared.get(canon_id(ref), []) if t != leaf["id"]]
            unresolved += not targets
            for t in targets:
                brings[leaf["id"]].append(t)
                if t not in tool_leaves:
                    brings[t].append(leaf["id"])
    has_allowlist = False
    for leaf in leaves:
        b = list(dict.fromkeys(brings[leaf["id"]]))
        if b:
            leaf["bring"] = b
        allow = list(dict.fromkeys(spelling[canon_id(t)] for t in leaf.pop("_tools") if canon_id(t) in tool_keys))
        if allow:
            leaf["toolAllowlist"] = allow
            has_allowlist = True
        leaf.pop("_refs")
    nodes.extend(leaves)

    doc = {
        "schemaVersion": "1.0",
        "graphId": f"{world.id}.authored",
        "version": "1",
        "profile": "L3" if has_allowlist else "L1",
        "defaults": {"routing": {"minConfidence": 0.55, "allowMulti": True, "shortlistK": 12}, "budget": {"maxPromptTokens": budget}},
        "nodes": nodes,
        "edges": [],
    }
    report = {
        "leaves": len(leaves),
        "chunks": len(chunks),
        "chunks_without_units": sum(not _has_knowledge(u) for u in units_per_chunk),
        "lost_chunks": dict(lost or {}),
        "declared_ids": len(declared),
        "unresolved_references": unresolved,
        # Share of the spec's IDs that some leaf declares (D-017 aggregates it per cell in `ape.build_quality`, which
        # owns the ID list); None for worlds without spec IDs, where it would be a vacuous 1.0.
        "id_coverage": id_coverage(rendered_spec_ids(world, apg_spec_ids(world)), spelling.values()),
        # Share of the knowledge base's facts that some leaf's knowledge carries (the `ape.kb.provenance` rule).
        "fact_coverage": fact_matcher(world).coverage(leaf["prompt"]["slots"]["knowledge"] for leaf in leaves),
    }
    return doc, report


def author_prompt_version() -> str:
    """Hash of the authoring prompt and unit schema: recorded in authored graphs' meta, so a prompt revision on dev
    (GATE_PREREG §5 allows it) marks graphs authored with the old prompt stale."""
    payload = json.dumps({"system": AUTHOR_SYSTEM, "schema": UNIT_SCHEMA}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_meta(world: World, author_id: str | None, embedding_model: str) -> dict:
    """What an authored graph was built from; stored in the graph's open `meta` mapping."""
    return graph_meta(world, author_id, embedding_model)


def authored_graph_current(world: World, cfg: Config, author_id: str | None) -> bool:
    """True when the authored graph and its report exist and were built from this world version and chunks, by this
    author with this authoring prompt, with the current embedding model. A crash before the report is written, a
    changed world, chunker or prompt, or a different author all mean a rebuild."""
    path = graph_path(cfg, world.id, "authored")
    try:
        meta = json.loads(path.read_text()).get("meta") or {}
        report = json.loads(path.with_suffix(".report.json").read_text())
    except (OSError, ValueError):
        return False
    want = build_meta(world, author_id, embedding_cache(cfg).model)
    in_report = {"worldHash": report.get("world_hash"), "author": report.get("author"), "embeddingModel": report.get("embedding_model")}
    return all(meta.get(k) == v for k, v in want.items()) and all(in_report[k] == want[k] for k in in_report)


CHUNK_ERRORS = (BuildResponseError, UnitShapeError)  # a chunk's own failure; anything else is fatal


async def _author_chunks(world: World, chunks: list[Chunk], one: Callable[[Chunk], Awaitable[list[dict]]]) -> tuple[list[list[dict]], dict[str, str]]:
    """Every chunk's units, tolerating up to `max_failed_chunks` lost chunks (module docstring). On a fatal error or
    one lost chunk too many, the calls still running are cancelled and the error is raised."""
    tolerated = max_failed_chunks(len(chunks))
    tasks = {asyncio.ensure_future(one(c)): c for c in chunks}
    results: dict[str, list[dict]] = {}
    lost: dict[str, str] = {}
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                chunk, exc = tasks[t], t.exception()
                if exc is None:
                    results[chunk.id] = t.result()
                    if chunk.fact_ids and not _has_knowledge(results[chunk.id]):
                        lost[chunk.id] = "no knowledge units for text that carries facts"
                elif isinstance(exc, CHUNK_ERRORS):
                    lost[chunk.id] = f"{type(exc).__name__}: {exc}"
                else:
                    raise exc
                if len(lost) > tolerated:
                    listing = "; ".join(f"{c}: {why}" for c, why in lost.items())
                    raise AuthoringError(
                        f"APG authoring of {world.id}: {len(lost)} of {len(chunks)} chunk(s) lost, more than the {tolerated} "
                        f"tolerated (APE_BUILD_MAX_FAILED_SHARE). Usable chunks are cached, so a re-run asks only the lost ones. {listing}"
                    )
    finally:
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return [results.get(c.id, []) for c in chunks], lost


async def author_world(world: World, cfg: Config, author: AuthorFn, concurrency: int | None = None, author_id: str | None = None) -> dict:
    """Author, validate, embed and write the graph (always; `authored_graph_current` decides skips).

    `concurrency` defaults to APE_BUILD_LLM_CONCURRENCY; `author_id` is recorded in the graph's meta and keys the
    per-chunk unit cache (no `author_id`, no cache)."""
    chunks = chunk_world(world)
    titles = {d.id: d.title for d in world.documents}
    sem = asyncio.Semaphore(concurrency or BuildConcurrency.from_env().llm)
    cache = UnitCache(cfg.cache_dir / UNIT_CACHE) if author_id else None

    async def one(chunk: Chunk) -> list[dict]:
        user = f"Document: {titles[chunk.doc_id]}\n\nExcerpt:\n{chunk.text}"
        key = UnitCache.key(author_id, AUTHOR_SYSTEM, user) if cache is not None else ""
        if cache is not None and (hit := cache.get(key)) is not None:
            return hit
        async with sem:
            out = await author(AUTHOR_SYSTEM, user)
        units = check_units(out)
        if cache is not None and (_has_knowledge(units) or not chunk.fact_ids):  # a lost chunk is asked again next time
            cache.put(key, units)
        return units

    try:
        units, lost = await _author_chunks(world, chunks, one)
    finally:
        if cache is not None:
            cache.close()
    doc, report = assemble(world, chunks, units, cfg.apg_budget_tokens, lost=lost)
    if report["leaves"] == 0:
        raise AuthoringError(f"APG authoring of {world.id} produced no knowledge units from {len(chunks)} chunk(s); no graph written")
    emb = embedding_cache(cfg)
    doc["meta"] = build_meta(world, author_id, emb.model)
    check = validate_graph(doc)
    errors = schema_errors(doc)
    if not check["valid"] or errors:
        raise ValueError(f"authored graph invalid: {check['errors'][:3]} {errors[:3]}")
    doc = await embed_graph(doc, emb, {"world": world.id, "system": "apg-authored"})
    meta = doc["meta"] = build_meta(world, author_id, emb.model) | {"embeddingDim": graph_dim(doc)}
    report.update(world_hash=meta["worldHash"], author=author_id, embedding_model=emb.model, embedding_dim=meta["embeddingDim"])
    path = graph_path(cfg, world.id, "authored")
    report_path = path.with_suffix(".report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The report goes last, after the old one is gone: a graph without its report (a crash in
    # between) counts as unbuilt.
    report_path.unlink(missing_ok=True)
    path.write_text(json.dumps(doc))
    report_path.write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    from ..artifacts import main as artifacts_main

    artifacts_main(fixed_kinds=("apg",))


if __name__ == "__main__":
    main()
