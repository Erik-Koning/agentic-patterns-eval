"""Build worlds (and, where available, their KB artifacts) for a split.

    python -m ape.build --split dev --family F7 --levels 10 1000 --worlds 3 --tasks 12

Embeddings for chunks are computed here so gate runs only ever read the cache.
With APE_EMBEDDINGS=fake everything is offline and free. KB artifacts (APG graphs, LightRAG
indices) are built per world, in parallel, by `python -m ape.artifacts`.
"""

import argparse
import asyncio

from .artifacts import embed_chunks
from .config import Config, embedding_cache
from .worlds.generate import make_world


async def build(split: str, family: str, levels: list[str], n_worlds: int, n_tasks: int, relational: bool, embed: bool, exception_style: str = "descriptive") -> list[str]:
    cfg = Config()
    ids = []
    emb = embedding_cache(cfg) if embed else None
    for level in levels:
        for i in range(n_worlds):
            w = make_world(family, level, split, i, n_tasks, relational, exception_style)
            w.save(cfg.world_path(w.id))
            ids.append(w.id)
            if emb is not None:
                await embed_chunks(w, emb)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "pilot", "test"])
    ap.add_argument("--family", required=True, choices=["F7", "F3", "F5"])
    ap.add_argument("--levels", nargs="+", required=True)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--independent", action="store_true", help="F7 without exceptions")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--exception-style", choices=["descriptive", "id_only", "messy"], default="descriptive", help="F7 relational only")
    args = ap.parse_args()
    ids = asyncio.run(build(args.split, args.family, args.levels, args.worlds, args.tasks, not args.independent, not args.no_embed, args.exception_style))
    print("\n".join(ids))


if __name__ == "__main__":
    main()
