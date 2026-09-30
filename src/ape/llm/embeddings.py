"""Cached OpenAI embeddings shared by APG's shortlist, S3s and LightRAG.

Vectors are cached in sqlite keyed by (model, sha256(text)), so a text is embedded
once per model and repeated runs are deterministic and free. `lookup()` is the
synchronous, network-free path used inside APG's sync `route()`; it raises on a
miss so a routing call can never silently hit the network.
"""

import asyncio
import hashlib
import json
import sqlite3
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .ledger import Ledger, LedgerEntry
from .tokens import count_tokens

BATCH = 256

# Attribution for ledger entries made deep inside an arm (APG routing, S3s, LightRAG's own
# embedding calls). The agent loop sets it around every compile: {"arm", "world", "sample", "epoch"}.
EMBED_CONTEXT: ContextVar[dict | None] = ContextVar("ape_embed_context", default=None)


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class EmbeddingMiss(KeyError):
    pass


class EmbeddingCache:
    def __init__(self, path: str | Path, model: str, client: Any = None, ledger: Ledger | None = None):
        """`client` is an `openai.AsyncOpenAI`-compatible object; created lazily if omitted."""
        self.model = model
        self._client = client
        self._ledger = ledger
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self._path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()
        self.network_calls = 0

    def _get(self, text: str) -> list[float] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM emb WHERE k = ?", (_key(self.model, text),)).fetchone()
        return json.loads(row[0]) if row else None

    def _put_many(self, pairs: list[tuple[str, list[float]]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)",
                [(_key(self.model, t), json.dumps(v)) for t, v in pairs],
            )
            self._db.commit()

    def lookup(self, texts: list[str]) -> list[list[float]]:
        """Cache-only; raises EmbeddingMiss for any text not embedded yet."""
        out = []
        for t in texts:
            v = self._get(t)
            if v is None:
                raise EmbeddingMiss(t[:80])
            out.append(v)
        return out

    async def embed(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self._get(t) is None))
        for i in range(0, len(missing), BATCH):
            batch = missing[i : i + BATCH]
            if self._client is None:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI()
            resp = await self._client.embeddings.create(model=self.model, input=batch)
            self.network_calls += 1
            self._put_many([(t, d.embedding) for t, d in zip(batch, resp.data)])
            if self._ledger is not None:
                used = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
                self._ledger.append(
                    LedgerEntry(
                        role="embeddings",
                        model=self.model,
                        kind="embed",
                        input_tokens=used if used is not None else sum(count_tokens(t) for t in batch),
                        context=context or EMBED_CONTEXT.get() or {},
                    )
                )
        return self.lookup(texts)

    def embed_sync(self, texts: list[str], context: dict | None = None) -> list[list[float]]:
        """For offline build scripts only (never inside a running event loop)."""
        return asyncio.run(self.embed(texts, context))


class CachedEmbeddingsConnector:
    """APG `embeddings` connector: cache lookups only, so `route()` stays offline and deterministic."""

    def __init__(self, cache: EmbeddingCache):
        self._cache = cache

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._cache.lookup(texts)
