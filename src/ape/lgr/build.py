"""Offline LightRAG index builds, one per world, run in their own process (never inside Inspect).

    python -m ape.lgr.build --split dev --family F7 --kind extract   # LLM extraction (needs the build model)
    python -m ape.lgr.build --split dev --family F7 --kind oracle    # custom KG from the world spec, no LLM

`extract` is the real comparator: LightRAG's own entity/relation extraction over the
shared chunks, with calls metered in the build ledger. `oracle` inserts a KG derived
from the world spec (diagnostic upper bound, and the offline test fixture).
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import os

from ..config import Config, embedding_cache
from ..llm.build_client import BuildLlm
from ..llm.ledger import Ledger
from ..worlds.render import Chunk, chunk_world
from ..worlds.spec import World
from .common import index_dir, open_rag, write_manifest


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
    for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name != "ape_manifest.json"):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


async def build_index(world: World, kind: str, cfg: Config) -> dict:
    chunks = chunk_world(world)
    wd = index_dir(cfg, world.id, kind)
    emb = embedding_cache(cfg)

    async def no_llm(*_a, **_k) -> str:
        raise RuntimeError("oracle builds must not call an LLM")

    if kind == "extract":
        model = os.environ.get("APE_BUILD_MODEL")
        if not model:
            raise RuntimeError("set APE_BUILD_MODEL (chosen at readiness E3) for extraction builds")
        llm = BuildLlm(model, Ledger(cfg.ledger_path), {"world": world.id, "system": "lightrag"}).lightrag_func()
    else:
        model, llm = None, no_llm
    rag = await open_rag(wd, world.id, llm, emb, query_time=False)
    try:
        if kind == "oracle":
            await rag.ainsert_custom_kg(oracle_kg(world, chunks))
        else:
            await rag.ainsert([c.text for c in chunks], ids=[c.id for c in chunks], file_paths=[c.id for c in chunks])
    finally:
        await rag.finalize_storages()
    manifest = {
        "world_id": world.id,
        "world_hash": world.content_hash(),
        "kind": kind,
        "lightrag_version": importlib.metadata.version("lightrag-hku"),
        "embedding_model": emb.model,
        "build_model": model,
        "chunks": len(chunks),
    }
    manifest["index_hash"] = _dir_hash(wd)
    write_manifest(wd, manifest)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kind", choices=["extract", "oracle"], required=True)
    args = ap.parse_args()
    cfg = Config()
    paths = sorted((cfg.worlds_dir / args.split).glob(f"{args.family}-*.json"))

    async def run() -> None:
        for p in paths:
            m = await build_index(World.load(p), args.kind, cfg)
            print(m["world_id"], m["kind"], m["index_hash"][:12])

    asyncio.run(run())


if __name__ == "__main__":
    main()
