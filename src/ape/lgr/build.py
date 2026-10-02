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
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from lightrag.base import DocStatus

from ..config import Config, embedding_cache
from ..llm.build_client import BuildLlm
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
    rag = await open_rag(wd, world.id, llm, emb, query_time=False)
    try:
        if kind == "oracle":
            await rag.ainsert_custom_kg(oracle_kg(world, chunks))
        else:
            await rag.ainsert([c.text for c in chunks], ids=[c.id for c in chunks], file_paths=[c.id for c in chunks])
            # LightRAG marks a document FAILED (e.g. its extraction calls kept failing) without
            # raising. Such an index is incomplete: no manifest, so the next run rebuilds it
            # (from the LLM cache for every chunk that was extracted).
            counts = {s: n for s, n in (await rag.doc_status.get_status_counts()).items() if n}
            if counts != {DocStatus.PROCESSED.value: len(chunks)}:
                raise RuntimeError(f"LightRAG extraction incomplete for {world.id}: document statuses {counts}, expected {len(chunks)} processed")
    finally:
        await rag.finalize_storages()
    manifest = {**key, "lightrag_version": importlib.metadata.version("lightrag-hku"), "chunks": len(chunks)}
    manifest["index_hash"] = _dir_hash(wd)
    write_manifest(wd, manifest)
    (wd / BUILD_MARKER).unlink(missing_ok=True)
    return manifest


def main() -> None:
    from ..artifacts import main as artifacts_main

    artifacts_main(fixed_kinds=("lightrag",))


if __name__ == "__main__":
    main()
