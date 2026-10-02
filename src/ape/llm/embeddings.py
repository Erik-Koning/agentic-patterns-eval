"""Cached OpenAI embeddings shared by APG's shortlist, S3s and LightRAG.

Vectors are cached in sqlite keyed by (model, sha256(text)), so a text is embedded
once per model and repeated runs are deterministic and free. `lookup()` is the
synchronous, network-free path used inside APG's sync `route()`; it raises on a
miss so a routing call can never silently hit the network.

Storage (schema 2). Each vector is a little-endian float32 BLOB: 6 KB for a 1,536-d vector, against ~30 KB
as the JSON text of schema 1. OpenAI computes and returns embeddings as float32 (the wire format is
base64-encoded float32), so float32 storage is lossless: live calls ask for `encoding_format="base64"` and
the decoded bytes are stored as they arrived. Reads return `list[float]` (Python floats holding the float32
values), the type every caller used before.

Migration. A schema-1 cache (JSON text rows) is converted in place when it is opened, in one write
transaction (`BEGIN IMMEDIATE`) that re-checks the schema version after taking the lock. Several build
processes (ape.artifacts workers) may open the same file at once: the first converts it and the others
wait on the busy timeout, then find schema 2 and do nothing. Converting rounds float64 values to float32;
no live embeddings had been cached when schema 2 was introduced (2026-10-02), only offline fake vectors,
whose generator now emits float32 values (`ape.llm.fake`), so cached values and fresh ones agree exactly.
"""

import asyncio
import base64
import hashlib
import json
import sqlite3
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

from .ledger import Ledger, LedgerEntry
from .tokens import count_tokens

BATCH = 256
SQLITE_TIMEOUT_S = 60.0  # how long a write waits for another process's lock on the shared cache
SCHEMA_VERSION = 2  # 1: JSON text rows; 2: little-endian float32 BLOB rows
DTYPE = np.dtype("<f4")

# Attribution for ledger entries made deep inside an arm (APG routing, S3s, LightRAG's own
# embedding calls). The agent loop sets it around every compile: {"arm", "world", "sample", "epoch"}.
EMBED_CONTEXT: ContextVar[dict | None] = ContextVar("ape_embed_context", default=None)


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class EmbeddingMiss(KeyError):
    pass


def to_blob(vector: Any) -> bytes:
    """A vector as stored: little-endian float32 bytes. A base64 string (OpenAI's `encoding_format="base64"`)
    is already those bytes, so it is decoded without a round trip through Python floats."""
    if isinstance(vector, str):
        raw = base64.b64decode(vector)
        if len(raw) % DTYPE.itemsize:
            raise ValueError(f"base64 embedding of {len(raw)} bytes is not a float32 array")
        return raw
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=DTYPE).tolist()


def _migrate(db: sqlite3.Connection) -> None:
    """Bring the cache to SCHEMA_VERSION in one write transaction (see the module docstring)."""
    db.execute("BEGIN IMMEDIATE")  # takes the write lock now, waiting on the busy timeout if another process holds it
    try:
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB NOT NULL)")
        row = db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None or int(row[0]) < SCHEMA_VERSION:
            # Schema 1 declared `v TEXT`; a BLOB value keeps its storage class under TEXT affinity, so the rows are
            # rewritten in place and the table is not rebuilt.
            rows = db.execute("SELECT k, v FROM emb WHERE typeof(v) = 'text'").fetchall()
            db.executemany("UPDATE emb SET v = ? WHERE k = ?", [(to_blob(json.loads(v)), k) for k, v in rows])
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def _enable_wal(db: sqlite3.Connection) -> None:
    """Switch the file to WAL (a persistent, one-time change). While the file is still in rollback mode
    (new, or created before WAL) and another process has a write open, SQLite refuses the switch at once
    with "database is locked" without consulting the busy timeout, so retry until that write commits."""
    deadline = time.monotonic() + SQLITE_TIMEOUT_S
    while True:
        try:
            db.execute("PRAGMA journal_mode=WAL").fetchone()
            return
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or time.monotonic() > deadline:
                raise
            time.sleep(0.05)


class EmbeddingCache:
    def __init__(self, path: str | Path, model: str, client: Any = None, ledger: Ledger | None = None):
        """`client` is an `openai.AsyncOpenAI`-compatible object; created lazily if omitted."""
        self.model = model
        self._client = client
        self._ledger = ledger
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # one connection per instance, shared by this process's threads
        # Several build processes (ape.artifacts workers, FX-4) share one cache file. WAL lets readers
        # run during another process's write; the busy timeout makes a writer wait for the other
        # process's commit instead of failing with "database is locked".
        # isolation_level=None: autocommit, so `_migrate` controls its own transaction (BEGIN IMMEDIATE).
        self._db = sqlite3.connect(self._path, check_same_thread=False, timeout=SQLITE_TIMEOUT_S, isolation_level=None)
        _enable_wal(self._db)
        _migrate(self._db)
        self.network_calls = 0

    def _get(self, text: str) -> list[float] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM emb WHERE k = ?", (_key(self.model, text),)).fetchone()
        return from_blob(row[0]) if row else None

    def _put_many(self, pairs: list[tuple[str, Any]]) -> None:
        """`pairs` of (text, vector), the vector a float sequence or a base64 float32 string."""
        rows = [(_key(self.model, t), to_blob(v)) for t, v in pairs]
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.executemany("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)", rows)
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

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

                from .build_client import MAX_RETRIES

                # The build client's retry policy (backoff, Retry-After): a rate-limited embedding call after an
                # expensive authoring pass must not fail the world (RELIABILITY_REVIEW K7).
                self._client = AsyncOpenAI(max_retries=MAX_RETRIES)
            # base64 is OpenAI's float32 wire format; asking for it explicitly returns the raw string, which is
            # stored as is (`to_blob`). A client that returns float lists instead is handled the same way.
            resp = await self._client.embeddings.create(model=self.model, input=batch, encoding_format="base64")
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
