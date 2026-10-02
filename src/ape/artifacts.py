"""Per-world build artifacts, built in parallel worker processes (FIX_PLAN FX-4).

    python -m ape.artifacts --split dev --family F7 --levels 1000 --kinds chunks,apg,lightrag --lightrag-kind extract
    python -m ape.artifacts --split dev --family F7 --kinds apg,lightrag --lightrag-kind oracle --fake-author   # offline dry run

Worlds must exist already (`python -m ape.build`). For each world, in this order:
- `chunks`: the shared chunks' embeddings are in the embedding cache (what `ape.build` does).
- `apg`: the authored APG graph (`ape.apg.author.author_world`), authored by the build model and effort
  from config/models.yaml, or by the scripted `perfect_author` with `--fake-author` (no LLM).
- `lightrag`: the LightRAG index of `--lightrag-kind` (`ape.lgr.build.build_index`).

Parallelism. Worlds are built at once in a process pool, at most `--workers` (APE_BUILD_WORKERS,
default 4), and each world gets a fresh process of its own. LightRAG keeps process-global state
(pipeline status, keyed locks, the default workspace), so processes isolate builds where threads or
coroutines in one process would not. Workers are spawned, not forked: each re-reads its configuration
from the environment and gets a world path, never a World object. Within a world, how many calls run
at once comes from `ape.config.BuildConcurrency`:

    APE_BUILD_LLM_CONCURRENCY   (16)  build-model calls in flight: LightRAG llm_model_max_async, APG authoring
    APE_BUILD_PARALLEL_INSERT   (8)   LightRAG max_parallel_insert (shared chunks extracted at once)
    APE_BUILD_EMBED_CONCURRENCY (16)  LightRAG embedding_func_max_async
    APE_BUILD_WORKERS           (4)   worlds at once (the --workers default)

The workers share the embedding cache (sqlite in WAL mode) and the build ledger (whole-line appends).

Idempotent and resumable. A world's artifact is skipped when it is current:
- chunks: every chunk text is already in the embedding cache;
- apg: the graph's `meta` and its report record this world's content hash, author and embedding
  model (`ape.apg.author.authored_graph_current`);
- lightrag: the manifest records this world's content hash, the kind, the build model and effort and
  the embedding model (`ape.lgr.build.index_current`).
Anything else (no manifest or graph after a crash, a regenerated world, another build model) is
rebuilt. A failure in one world, or in one kind, never stops the others; the exit status is non-zero
when anything failed, and re-running the same command builds only what is missing.
"""

import argparse
import asyncio
import multiprocessing
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .config import BuildConcurrency, Config, embedding_cache
from .llm.embeddings import EmbeddingCache, EmbeddingMiss
from .worlds.render import chunk_world
from .worlds.spec import World

KINDS = ("chunks", "apg", "lightrag")
LIGHTRAG_KINDS = ("extract", "oracle")
ERROR_CHARS = 500


@dataclass
class WorldResult:
    """One world's outcome: each kind is "built", "skipped" or "failed" (with its error)."""

    path: str
    world_id: str | None = None
    kinds: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # kind (or "world", when it cannot load) -> error
    seconds: float = 0.0

    @property
    def status(self) -> str:
        if self.errors:
            return "failed"
        return "built" if "built" in self.kinds.values() else "skipped"

    def summary(self) -> str:
        kinds = " ".join(f"{k}={v}" for k, v in self.kinds.items())
        errors = "".join(f"\n    {k}: {e}" for k, e in self.errors.items())
        return f"{self.status:7s} {self.world_id or self.path}  {kinds}  ({self.seconds:.1f}s){errors}"


async def embed_chunks(world: World, emb: EmbeddingCache) -> bool:
    """Make sure every shared chunk of `world` is embedded; False when they all were already."""
    texts = [c.text for c in chunk_world(world)]
    try:
        emb.lookup(texts)
        return False
    except EmbeddingMiss:
        await emb.embed(texts, context={"world": world.id, "system": "chunks"})
        return True


async def _chunks(world: World, cfg: Config, _opts: dict) -> str:
    return "built" if await embed_chunks(world, embedding_cache(cfg)) else "skipped"


async def _apg(world: World, cfg: Config, opts: dict) -> str:
    from .apg.author import (
        FAKE_AUTHOR_ID,
        author_world,
        authored_graph_current,
        build_llm_author,
        llm_author_id,
    )

    if opts["fake_author"]:
        from .llm.fake import perfect_author

        author, author_id = perfect_author, FAKE_AUTHOR_ID
    else:
        from .llm.ledger import Ledger
        from .models import build_settings

        model, effort = build_settings()
        author, author_id = build_llm_author(model, Ledger(cfg.ledger_path), world.id, effort), llm_author_id(model, effort)
    if authored_graph_current(world, cfg, author_id):
        return "skipped"
    await author_world(world, cfg, author, author_id=author_id)
    return "built"


async def _lightrag(world: World, cfg: Config, opts: dict) -> str:
    from .lgr.build import build_index, index_current

    if index_current(world, opts["lightrag_kind"], cfg):
        return "skipped"
    await build_index(world, opts["lightrag_kind"], cfg)
    return "built"


_BUILDERS: dict[str, Callable] = {"chunks": _chunks, "apg": _apg, "lightrag": _lightrag}


def _error(e: BaseException) -> str:
    text = f"{type(e).__name__}: {e}"
    return text if len(text) <= ERROR_CHARS else text[: ERROR_CHARS - 3] + "..."


def build_world(path: str, kinds: Sequence[str], lightrag_kind: str | None = None, fake_author: bool = False) -> WorldResult:
    """Build one world's artifacts. The process-pool entry point, so it is top level, takes a path and
    reads its configuration from the environment. Errors are caught per kind and reported, not raised."""
    t0 = time.monotonic()
    result = WorldResult(str(path))
    try:
        world = World.load(path)
        result.world_id = world.id
    except Exception as e:  # noqa: BLE001  (reported in the result: one world never stops the others)
        traceback.print_exc()
        result.errors["world"] = _error(e)
        result.kinds = {k: "failed" for k in kinds}
        result.seconds = time.monotonic() - t0
        return result
    cfg = Config()
    opts = {"lightrag_kind": lightrag_kind, "fake_author": fake_author}

    async def run() -> None:
        for kind in kinds:
            try:
                result.kinds[kind] = await _BUILDERS[kind](world, cfg, opts)
            except Exception as e:  # noqa: BLE001  (reported in the result: one kind never stops the others)
                print(f"[{world.id}] {kind} failed:", file=sys.stderr)
                traceback.print_exc()
                result.kinds[kind] = "failed"
                result.errors[kind] = _error(e)

    asyncio.run(run())
    result.seconds = time.monotonic() - t0
    return result


def _check(kinds: Sequence[str], lightrag_kind: str | None) -> tuple[str, ...]:
    kinds = tuple(dict.fromkeys(kinds))
    if bad := [k for k in kinds if k not in KINDS]:
        raise ValueError(f"unknown artifact kind(s) {bad}; kinds are {list(KINDS)}")
    if "lightrag" in kinds and lightrag_kind not in LIGHTRAG_KINDS:
        raise ValueError(f"lightrag builds need a LightRAG kind, one of {list(LIGHTRAG_KINDS)}; got {lightrag_kind!r}")
    return tuple(k for k in KINDS if k in kinds)  # always chunks, apg, lightrag order


def build_artifacts(
    paths: Sequence[str | Path],
    kinds: Sequence[str],
    lightrag_kind: str | None = None,
    fake_author: bool = False,
    workers: int | None = None,
    on_result: Callable[[WorldResult], None] | None = None,
) -> list[WorldResult]:
    """Build `kinds` for every world file in `paths`, one fresh spawned process per world, at most
    `workers` (default APE_BUILD_WORKERS) at once. Returns one result per path, in input order;
    `on_result` sees each as it finishes."""
    kinds = _check(kinds, lightrag_kind)
    paths = [str(p) for p in paths]
    if not paths:
        return []
    workers = min(workers or BuildConcurrency.from_env().workers, len(paths))
    results: dict[str, WorldResult] = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as pool:
        futures: dict[Future, str] = {pool.submit(build_world, p, kinds, lightrag_kind, fake_author): p for p in paths}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001  (the worker process itself died, e.g. killed)
                r = WorldResult(p, kinds={k: "failed" for k in kinds}, errors={"world": _error(e)})
            results[p] = r
            if on_result:
                on_result(r)
    return [results[p] for p in paths]


def world_paths(cfg: Config, split: str, family: str, levels: Sequence[str] | None = None) -> list[Path]:
    """World files of a split and family, optionally only some levels (world IDs are `{family}-{level}-...`)."""
    root = cfg.worlds_dir / split
    patterns = [f"{family}-{level}-*.json" for level in levels] if levels else [f"{family}-*.json"]
    return sorted({p for pat in patterns for p in root.glob(pat)})


def main(argv: Sequence[str] | None = None, fixed_kinds: tuple[str, ...] | None = None) -> None:
    """CLI. `fixed_kinds` serves `python -m ape.apg.author` and `python -m ape.lgr.build`, which build one kind."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--split", required=True, choices=["dev", "pilot", "test"])
    ap.add_argument("--family", required=True, choices=["F7", "F3", "F5", "F1", "F2"])
    ap.add_argument("--levels", nargs="+", help="only these levels (default: every world of the family)")
    if fixed_kinds is None:
        ap.add_argument("--kinds", required=True, help=f"comma-separated, from {','.join(KINDS)}")
    if fixed_kinds is None or "lightrag" in fixed_kinds:
        ap.add_argument("--lightrag-kind", "--kind", dest="lightrag_kind", choices=LIGHTRAG_KINDS, required=fixed_kinds is not None, help="needed for lightrag builds")
    if fixed_kinds is None or "apg" in fixed_kinds:
        ap.add_argument("--fake-author", action="store_true", help="author APG graphs with the scripted perfect_author (offline, no LLM)")
    ap.add_argument("--workers", type=int, help="worlds built at once (default APE_BUILD_WORKERS, else 4)")
    args = ap.parse_args(argv)
    kinds = fixed_kinds or tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    lightrag_kind = getattr(args, "lightrag_kind", None)
    fake_author = getattr(args, "fake_author", False)
    try:
        kinds = _check(kinds, lightrag_kind)
    except ValueError as e:
        ap.error(str(e))
    if args.workers is not None and args.workers <= 0:
        ap.error("--workers must be positive")
    cfg = Config()
    paths = world_paths(cfg, args.split, args.family, args.levels)
    if not paths:
        raise SystemExit(f"no worlds for {args.family} {args.levels or ''} in {cfg.worlds_dir / args.split}; run `python -m ape.build` first")
    llm_calls = ("apg" in kinds and not fake_author) or ("lightrag" in kinds and lightrag_kind == "extract")
    if llm_calls or cfg.embeddings_backend != "fake":
        from .models import require_preflight

        require_preflight(live=True)  # every model priced and an API key present, before any spend (FX-2)
    workers = min(args.workers or BuildConcurrency.from_env().workers, len(paths))
    print(f"building {','.join(kinds)} for {len(paths)} world(s), {workers} at a time", flush=True)
    results = build_artifacts(paths, kinds, lightrag_kind, fake_author, workers, on_result=lambda r: print(r.summary(), flush=True))
    counts = {s: sum(r.status == s for r in results) for s in ("built", "skipped", "failed")}
    print(", ".join(f"{n} {s}" for s, n in counts.items()))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
