"""Shared LightRAG construction for build (offline) and query (inside Inspect) paths.

One LightRAG chunk per shared chunk (DECISIONS D-003): each shared chunk is inserted
as its own document with `file_path` = our chunk ID, and `chunk_token_size` is set
above the shared chunker's cap so LightRAG never re-splits it.
"""

import json
from pathlib import Path

import numpy as np
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from ..config import Config
from ..llm.embeddings import EmbeddingCache

CHUNK_TOKEN_SIZE = 4096  # far above worlds.render.DEFAULT_CHUNK_TOKENS
SEP = "<SEP>"  # LightRAG's separator when a node or edge has several sources


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


async def open_rag(working_dir: Path, world_id: str, llm_func, emb: EmbeddingCache, query_time: bool) -> LightRAG:
    """Query time disables the LLM response cache so every compile pays for its keyword call
    (APG's classify is never cached either); build time keeps the extraction cache."""
    working_dir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace_name(world_id),
        llm_model_func=llm_func,
        embedding_func=embedding_func(emb, await embedding_dim(emb)),
        enable_llm_cache=not query_time,
        enable_llm_cache_for_entity_extract=True,
        chunk_token_size=CHUNK_TOKEN_SIZE,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def write_manifest(working_dir: Path, manifest: dict) -> None:
    (working_dir / "ape_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def read_manifest(working_dir: Path) -> dict:
    return json.loads((working_dir / "ape_manifest.json").read_text())
