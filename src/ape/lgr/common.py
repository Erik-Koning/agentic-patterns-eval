"""Shared LightRAG construction for build (offline) and query (inside Inspect) paths.

One LightRAG chunk per shared chunk (DECISIONS D-003): each shared chunk is inserted
as its own document with `file_path` = our chunk ID, and `chunk_token_size` is set
above the shared chunker's cap so LightRAG never re-splits it.
"""

import asyncio
import json
import weakref
from pathlib import Path

import numpy as np
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from ..config import BuildConcurrency, Config
from ..llm.embeddings import EmbeddingCache

CHUNK_TOKEN_SIZE = 4096  # far above worlds.render.DEFAULT_CHUNK_TOKENS
SEP = "<SEP>"  # LightRAG's separator when a node or edge has several sources
MANIFEST = "ape_manifest.json"  # written last by a build: its presence means the index is complete


def index_dir(cfg: Config, world_id: str, kind: str) -> Path:
    return cfg.indices_dir / "lightrag" / f"{world_id}.{kind}"


def workspace_name(world_id: str) -> str:
    return world_id.replace("-", "_").replace(".", "_")


def embedding_func(emb: EmbeddingCache, dim: int) -> EmbeddingFunc:
    async def embed(texts: list[str], **_kwargs) -> np.ndarray:
        return np.array(await emb.embed(list(texts)), dtype=np.float32)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embed)


async def embedding_dim(emb: EmbeddingCache) -> int:
    return len((await emb.embed(["dimension probe"]))[0])


def build_concurrency_kwargs() -> dict[str, int]:
    """LightRAG concurrency for build-time instances, from `BuildConcurrency` (FX-4).

    Only how many calls run at once changes: gleaning and the extraction protocol (D-003) are untouched.
    """
    c = BuildConcurrency.from_env()
    return {"llm_model_max_async": c.llm, "max_parallel_insert": c.parallel_insert, "embedding_func_max_async": c.embed}


# LightRAG's process-wide init lock (shared_storage `data_init_lock`) is a plain asyncio.Lock. The first time two
# instances initialise at once it binds to that event loop, and from then on any contended initialisation in
# another loop (every eval set runs its own) raises "is bound to a different event loop", failing the sample.
# `open_rag` opens one instance at a time per loop, so LightRAG's lock is never contended and never binds.
_open_locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def _open_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    if (lock := _open_locks.get(loop)) is None:
        lock = _open_locks[loop] = asyncio.Lock()
    return lock


async def open_rag(working_dir: Path, world_id: str, llm_func, emb: EmbeddingCache, query_time: bool) -> LightRAG:
    """Query time disables the LLM response cache so every compile pays for its keyword call
    (APG's classify is never cached either); build time keeps the extraction cache.

    Build time also raises LightRAG's concurrency (`build_concurrency_kwargs`); query time keeps
    LightRAG's defaults."""
    working_dir.mkdir(parents=True, exist_ok=True)
    func = embedding_func(emb, await embedding_dim(emb))
    async with _open_lock():  # see _open_locks
        rag = LightRAG(
            working_dir=str(working_dir),
            workspace=workspace_name(world_id),
            llm_model_func=llm_func,
            embedding_func=func,
            enable_llm_cache=not query_time,
            enable_llm_cache_for_entity_extract=True,
            chunk_token_size=CHUNK_TOKEN_SIZE,
            **({} if query_time else build_concurrency_kwargs()),
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()
    return rag


def write_manifest(working_dir: Path, manifest: dict) -> None:
    (working_dir / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))


def read_manifest(working_dir: Path) -> dict:
    return json.loads((working_dir / MANIFEST).read_text())
