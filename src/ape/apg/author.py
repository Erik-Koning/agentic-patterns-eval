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

    python -m ape.apg.author --split dev --family F7     # needs APE_BUILD_MODEL
"""

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable

from apg_core import validate_graph

from ..config import Config, embedding_cache
from ..llm.build_client import BuildLlm
from ..llm.ledger import Ledger
from ..worlds.render import Chunk, chunk_world
from ..worlds.spec import World
from .arm import embed_graph, graph_path
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


def build_llm_author(model: str, ledger: Ledger, world_id: str) -> AuthorFn:
    llm = BuildLlm(model, ledger, {"world": world_id, "system": "apg-author"})

    async def author(system: str, user: str) -> dict:
        return await llm.json(system, user, "units", UNIT_SCHEMA)

    return author


def assemble(world: World, chunks: list[Chunk], units_per_chunk: list[list[dict]], budget: int) -> tuple[dict, dict]:
    titles = {d.id: d.title for d in world.documents}
    nodes = [{"id": "root", "parentId": None, "routable": False, "title": "Company knowledge"}]
    for doc_id in dict.fromkeys(c.doc_id for c in chunks):
        nodes.append({"id": f"doc-{doc_id}", "parentId": "root", "type": "category", "routable": False, "title": titles[doc_id]})

    leaves: list[dict] = []
    for chunk, units in zip(chunks, units_per_chunk):
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
                    "props": {"sourceChunkIds": [chunk.id], "declares": list(dict.fromkeys(u["declares"]))},
                    "_refs": list(dict.fromkeys(u["references"])),
                    "_tools": list(dict.fromkeys(u["tools"])),
                }
            )

    declared: dict[str, list[str]] = {}
    for leaf in leaves:
        for ident in leaf["props"]["declares"]:
            declared.setdefault(ident, []).append(leaf["id"])
    # Only tools some leaf actually defines count as tools; stray words an author puts in
    # `tools` (e.g. "any" from "do not call any other tool") never reach an allowlist.
    tool_names = {t for leaf in leaves for t in leaf["_tools"] if t in declared}
    tool_leaves = {lid for t in tool_names for lid in declared.get(t, [])}

    brings: dict[str, list[str]] = {leaf["id"]: [] for leaf in leaves}
    unresolved = 0
    for leaf in leaves:
        for ref in leaf["_refs"] + leaf["_tools"]:
            targets = [t for t in declared.get(ref, []) if t != leaf["id"]]
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
        allow = [t for t in leaf.pop("_tools") if t in tool_names]
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
        "chunks_without_units": sum(not u for u in units_per_chunk),
        "declared_ids": len(declared),
        "unresolved_references": unresolved,
        "id_coverage": _id_coverage(world, set(declared)),
    }
    return doc, report


def _id_coverage(world: World, declared: set[str]) -> float:
    """Share of the spec's identifiers that some authored leaf declares (extraction diagnostic)."""
    spec_ids = [p.id for p in world.policies] + [x.id for x in world.exceptions] + [p.id for p in world.procedures] + [t.name for t in world.tools]
    return sum(i in declared for i in spec_ids) / len(spec_ids) if spec_ids else 1.0


async def author_world(world: World, cfg: Config, author: AuthorFn, concurrency: int = 8) -> dict:
    chunks = chunk_world(world)
    titles = {d.id: d.title for d in world.documents}
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: Chunk) -> list[dict]:
        async with sem:
            out = await author(AUTHOR_SYSTEM, f"Document: {titles[chunk.doc_id]}\n\nExcerpt:\n{chunk.text}")
        return out.get("units", [])

    units = await asyncio.gather(*(one(c) for c in chunks))
    doc, report = assemble(world, chunks, list(units), cfg.context_budget_tokens)
    check = validate_graph(doc)
    errors = schema_errors(doc)
    if not check["valid"] or errors:
        raise ValueError(f"authored graph invalid: {check['errors'][:3]} {errors[:3]}")
    doc = await embed_graph(doc, embedding_cache(cfg), {"world": world.id, "system": "apg-authored"})
    path = graph_path(cfg, world.id, "authored")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--family", required=True)
    args = ap.parse_args()
    model = os.environ.get("APE_BUILD_MODEL")
    if not model:
        raise SystemExit("set APE_BUILD_MODEL (chosen at readiness E3)")
    cfg = Config()

    async def run() -> None:
        for p in sorted((cfg.worlds_dir / args.split).glob(f"{args.family}-*.json")):
            w = World.load(p)
            print(w.id, await author_world(w, cfg, build_llm_author(model, Ledger(cfg.ledger_path), w.id)))

    asyncio.run(run())


if __name__ == "__main__":
    main()
