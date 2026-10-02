"""Offline LightRAG index builds, one per world, run in their own process (never inside Inspect).

    python -m ape.lgr.build --split dev --family F7 --kind extract   # LLM extraction (build model from config/models.yaml)
    python -m ape.lgr.build --split dev --family F7 --kind oracle    # custom KG from the world spec, no LLM

`extract` is the real comparator: LightRAG's own entity/relation extraction over the
shared chunks, with calls metered in the build ledger. `oracle` inserts a KG derived
from the world spec (diagnostic upper bound, and the offline test fixture).

The CLI delegates to `ape.artifacts --kinds lightrag` (parallel worlds, skips current indices).

Rebuilds and resumes. The manifest is written last and records the build key (world content
hash, kind, build model and effort, embedding model); `index_current` compares it, and an extract
build whose documents LightRAG left FAILED raises instead of writing one. `build_index` always
builds from a cleared working directory, since LightRAG ignores document IDs it already holds and
would keep stale chunks and entities of an older world version. One file survives the clearing:
LightRAG's LLM response cache, when the earlier (complete or interrupted) build used the same
build model and effort. Its entries are keyed by the full prompt, so a rebuild after a crash or
a world change pays only for chunks whose extraction is not cached yet.

LightRAG keeps each workspace's stores in process memory once loaded, so rebuilding a world that
this process already opened needs a fresh process; `ape.artifacts` gives every world its own.

Degraded extractions (RELIABILITY_REVIEW K2). LightRAG's parser never raises: a reply in the wrong shape yields a
document with no entities, silently. After an extract build every chunk is checked, and a chunk is *lost* when its
document is not PROCESSED (its extraction calls kept failing, `BuildLlm.lightrag_func` raised) or when it carries
world facts but produced no entity. A world tolerates `max_failed_chunks(n)` lost chunks (APE_BUILD_MAX_FAILED_SHARE,
default 2%); they are recorded in the manifest's `extraction` section. More than that and the build raises (no
manifest), after dropping the lost chunks' cached replies so the rebuild asks them again; replies of chunks that
were fine stay cached. Unusable replies never reach the cache in the first place (`ape.llm.build_client`).
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from lightrag.base import DocStatus

from ..config import Config, embedding_cache
from ..llm.build_client import BuildLlm, max_failed_chunks
from ..llm.ledger import Ledger
from ..models import build_settings
from ..worlds.render import Chunk, chunk_world
from ..worlds.spec import World
from .common import MANIFEST, index_dir, open_rag, write_manifest

BUILD_MARKER = "ape_build.json"  # the build key of a build in progress; removed once the manifest is written
LLM_CACHE = "kv_store_llm_response_cache.json"  # LightRAG's LLM response cache (per workspace)


def oracle_kg(world: World, chunks: list[Chunk]) -> dict:
    home = {f: c.id for c in chunks for f in c.fact_ids}

    def ent(name: str, etype: str, fact_id: str) -> dict:
        return {"entity_name": name, "entity_type": etype, "description": world.facts[fact_id].text, "source_id": home[fact_id], "file_path": home[fact_id]}

    def rel(src: str, tgt: str, keywords: str, fact_id: str) -> dict:
        return {"src_id": src, "tgt_id": tgt, "description": world.facts[fact_id].text, "keywords": keywords, "weight": 1.0, "source_id": home[fact_id], "file_path": home[fact_id]}

    entities, relationships = [], []
    if world.family == "F7":
        entities += [ent(p.id, "policy", p.fact_id) for p in world.policies]
        entities += [ent(x.id, "exception", x.fact_id) for x in world.exceptions]
        relationships += [rel(x.id, x.policy_id, "amends,overrides", x.fact_id) for x in world.exceptions]
    elif world.family == "F3":
        entities += [ent(t.name, "tool", t.fact_id) for t in world.tools]
        entities += [ent(p.id, "procedure", p.fact_id) for p in world.procedures]
        relationships += [rel(p.id, s["tool"], "calls", p.fact_id) for p in world.procedures for s in p.steps]
    elif world.family in ("F1", "F2"):
        entities += [ent(f"{r['segment']} rating rule", "rule", r["fact_id"]) for r in world.entities["rules"]]
        entities += [ent("escalation protocol", "procedure", "f-ESC-protocol")]
        entities += [ent(code, "escalation_code", f"f-{code}") for code in sorted(world.entities["routes"])]
        entities += [ent(target, "supplier", f"f-{code}") for code, target in sorted(world.entities["routes"].items())]
        relationships += [rel(code, target, "routes to,escalation", f"f-{code}") for code, target in sorted(world.entities["routes"].items())]
    else:
        for e in world.events:
            relationships.append(rel(e.value if e.relation == "manager" else e.subject, e.subject if e.relation == "manager" else e.value, e.relation, e.fact_id))
        names = {r["src_id"] for r in relationships} | {r["tgt_id"] for r in relationships}
        first = {}
        for r in relationships:
            first.setdefault(r["src_id"], r)
            first.setdefault(r["tgt_id"], r)
        entities += [{"entity_name": n, "entity_type": "entity", "description": first[n]["description"], "source_id": first[n]["source_id"], "file_path": first[n]["file_path"]} for n in sorted(names)]
    return {
        "chunks": [{"content": c.text, "source_id": c.id, "file_path": c.id, "chunk_order_index": i} for i, c in enumerate(chunks)],
        "entities": entities,
        "relationships": relationships,
    }


def _dir_hash(path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name not in (MANIFEST, BUILD_MARKER)):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def build_key(world: World, kind: str, embedding_model: str) -> dict:
    """What an index is built from; a manifest with the same key means the index is current."""
    if kind not in ("extract", "oracle"):
        raise ValueError(f"unknown LightRAG build kind {kind!r}; expected 'extract' or 'oracle'")
    model, effort = build_settings() if kind == "extract" else (None, None)
    return {
        "world_id": world.id,
        "world_hash": world.content_hash(),
        "kind": kind,
        "embedding_model": embedding_model,
        "build_model": model,
        "build_effort": effort,
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def index_current(world: World, kind: str, cfg: Config) -> bool:
    """True when the index's manifest exists and records this world version and the current build settings."""
    manifest = _read_json(index_dir(cfg, world.id, kind) / MANIFEST)
    key = build_key(world, kind, embedding_cache(cfg).model)
    return manifest is not None and all(manifest.get(k) == v for k, v in key.items())


def _prepare_dir(wd: Path, key: dict) -> bool:
    """Clear `wd` for a build, keeping LightRAG's LLM cache when the previous build (its manifest, or
    the marker of an interrupted one) used the same build model and effort. Writes the build marker.
    Returns whether the cache was kept."""
    prev = _read_json(wd / MANIFEST) or _read_json(wd / BUILD_MARKER)
    keep = key["kind"] == "extract" and prev is not None and (prev.get("build_model"), prev.get("build_effort")) == (key["build_model"], key["build_effort"])
    kept = False
    if wd.exists():
        for p in sorted(wd.rglob("*"), reverse=True):  # children before their directories
            if keep and p.name == LLM_CACHE and p.is_file():
                kept = True
            elif p.is_dir():
                if not any(p.iterdir()):
                    p.rmdir()
            else:
                p.unlink()
    wd.mkdir(parents=True, exist_ok=True)
    (wd / BUILD_MARKER).write_text(json.dumps(key, indent=1, sort_keys=True))
    return kept


def _store(wd: Path, name: str) -> Path | None:
    """One of LightRAG's JSON KV stores (`kv_store_{name}.json`) in the index's workspace."""
    return next(iter(sorted(wd.rglob(f"kv_store_{name}.json"))), None)


def _read_store(wd: Path, name: str) -> dict:
    path = _store(wd, name)
    return (_read_json(path) or {}) if path is not None else {}


def extraction_health(wd: Path, chunks: list[Chunk]) -> dict:
    """Lost chunks of an extract build (module docstring): FAILED documents, and chunks with facts but no entity."""
    status = _read_store(wd, "doc_status")
    entities = _read_store(wd, "full_entities")

    def entity_count(doc_id: str) -> int:
        rec = entities.get(doc_id) or {}
        return int(rec.get("count", len(rec.get("entity_names") or [])))

    failed = [c.id for c in chunks if (status.get(c.id) or {}).get("status") != DocStatus.PROCESSED.value]
    empty = [c.id for c in chunks if c.id not in failed and c.fact_ids and entity_count(c.id) == 0]
    return {"chunks": len(chunks), "failed_docs": failed, "empty_docs": empty, "lost": len(failed) + len(empty), "tolerated": max_failed_chunks(len(chunks))}


def drop_cached_replies(wd: Path, doc_ids: list[str]) -> int:
    """Remove the LLM-cache entries of these documents' chunks, so a rebuild asks them again; returns how many."""
    cache_path = _store(wd, "llm_response_cache")
    if not doc_ids or cache_path is None:
        return 0
    keys = {k for k, v in _read_store(wd, "text_chunks").items() if isinstance(v, dict) and v.get("full_doc_id") in set(doc_ids)}
    cache = _read_json(cache_path) or {}
    keep = {k: v for k, v in cache.items() if not (isinstance(v, dict) and v.get("chunk_id") in keys)}
    if len(keep) != len(cache):
        cache_path.write_text(json.dumps(keep, ensure_ascii=False, indent=2))
    return len(cache) - len(keep)


async def build_index(world: World, kind: str, cfg: Config) -> dict:
    chunks = chunk_world(world)
    wd = index_dir(cfg, world.id, kind)
    emb = embedding_cache(cfg)
    key = build_key(world, kind, emb.model)

    async def no_llm(*_a, **_k) -> str:
        raise RuntimeError("oracle builds must not call an LLM")

    if kind == "extract":
        llm = BuildLlm(key["build_model"], Ledger(cfg.ledger_path), {"world": world.id, "system": "lightrag"}, reasoning_effort=key["build_effort"]).lightrag_func()
    else:
        llm = no_llm
    _prepare_dir(wd, key)
    rag = await open_rag(wd, world.id, llm, emb, query_time=False, embed_context={"world": world.id, "system": "lightrag"})
    try:
        if kind == "oracle":
            await rag.ainsert_custom_kg(oracle_kg(world, chunks))
        else:
            await rag.ainsert([c.text for c in chunks], ids=[c.id for c in chunks], file_paths=[c.id for c in chunks])
    finally:
        await rag.finalize_storages()
    extraction = None
    if kind == "extract":
        # LightRAG marks a document FAILED (its extraction calls kept failing) without raising, and parses a reply in
        # the wrong shape into no entities. Too many lost chunks: no manifest, so the next run rebuilds the index
        # (from the LLM cache for every chunk that was extracted well).
        extraction = extraction_health(wd, chunks)
        if extraction["lost"] > extraction["tolerated"]:
            dropped = drop_cached_replies(wd, extraction["empty_docs"])
            raise RuntimeError(
                f"LightRAG extraction incomplete for {world.id}: {extraction['lost']} of {len(chunks)} chunk(s) lost, more than the "
                f"{extraction['tolerated']} tolerated (APE_BUILD_MAX_FAILED_SHARE); FAILED documents {extraction['failed_docs']}, "
                f"documents with facts but no entity {extraction['empty_docs']} ({dropped} cached repl{'y' if dropped == 1 else 'ies'} dropped)"
            )
    manifest = {**key, "lightrag_version": importlib.metadata.version("lightrag-hku"), "chunks": len(chunks)}
    if extraction is not None:
        manifest["extraction"] = extraction
    manifest["index_hash"] = _dir_hash(wd)
    write_manifest(wd, manifest)
    (wd / BUILD_MARKER).unlink(missing_ok=True)
    return manifest


def main() -> None:
    from ..artifacts import main as artifacts_main

    artifacts_main(fixed_kinds=("lightrag",))


if __name__ == "__main__":
    main()
