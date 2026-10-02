"""Embedding cache storage (READINESS_AUDIT §3): float32 BLOBs, migration from JSON text, base64 decoding."""

import asyncio
import base64
import json
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from types import SimpleNamespace

import numpy as np

from ape.llm.embeddings import SCHEMA_VERSION, EmbeddingCache, _key
from ape.llm.ledger import Ledger

DIM = 1536


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)  # unit vectors, like OpenAI's


class Base64Client:
    """Mimics `openai.AsyncOpenAI().embeddings.create(..., encoding_format="base64")`: raw base64 float32."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors, self.formats, self.calls = vectors, [], 0
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input, encoding_format=None):
        self.calls += 1
        self.formats.append(encoding_format)
        data = [SimpleNamespace(embedding=base64.b64encode(self.vectors[t].astype("<f4").tobytes()).decode()) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=3 * len(input)))


def _old_cache(path, rows: dict[str, list[float]], model: str = "m") -> None:
    """A schema-1 cache, exactly as the previous code wrote it: JSON text rows, no meta table."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    db.executemany("INSERT OR REPLACE INTO emb (k, v) VALUES (?, ?)", [(_key(model, t), json.dumps(v)) for t, v in rows.items()])
    db.commit()
    db.close()


def test_float32_round_trip_is_exact_and_returns_python_float_lists(tmp_path):
    vecs = _vectors(3)
    texts = ["a", "b", "c"]
    client = Base64Client(dict(zip(texts, vecs, strict=True)))
    cache = EmbeddingCache(tmp_path / "e.sqlite", "m", client=client)
    out = asyncio.run(cache.embed(texts))
    assert all(isinstance(v, list) and all(type(x) is float for x in v) for v in out)
    assert np.array_equal(np.array(out, dtype=np.float32), vecs)  # bit-exact
    assert cache.lookup(["b"]) == [vecs[1].tolist()]


def test_a_1536_d_vector_takes_6_kb_instead_of_about_30(tmp_path):
    vec = _vectors(1)[0]
    cache = EmbeddingCache(tmp_path / "e.sqlite", "m", client=Base64Client({"t": vec}))
    asyncio.run(cache.embed(["t"]))
    blob = cache._db.execute("SELECT v FROM emb").fetchone()[0]
    assert isinstance(blob, bytes) and len(blob) == DIM * 4 == 6144
    as_json = len(json.dumps(vec.tolist()).encode())  # what schema 1 stored for the same vector
    assert as_json > 4 * len(blob), as_json


def test_live_calls_ask_for_base64_and_count_calls_and_ledger_as_before(tmp_path):
    vecs = _vectors(4)
    texts = ["w", "x", "y", "z"]
    client = Base64Client(dict(zip(texts, vecs, strict=True)))
    ledger = Ledger(tmp_path / "ledger.jsonl")
    cache = EmbeddingCache(tmp_path / "e.sqlite", "m", client=client, ledger=ledger)
    asyncio.run(cache.embed(["w", "x", "w"]))
    asyncio.run(cache.embed(["x", "y", "z", "w"]))
    assert client.formats == ["base64", "base64"] and cache.network_calls == 2 == client.calls
    assert ledger.totals(role="embeddings") == {"input_tokens": 3 * 2 + 3 * 2, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0, "calls": 2}
    assert asyncio.run(cache.embed(texts)) == vecs.tolist() and client.calls == 2  # all cached now


def test_a_json_cache_is_migrated_in_place_once(tmp_path):
    path = tmp_path / "old.sqlite"
    rows = {"p": [0.1, 0.2, 0.3], "q": [1.0, -2.5, 3.0e-7], "empty": []}
    _old_cache(path, rows)
    cache = EmbeddingCache(path, "m", client=Base64Client({}))
    db = cache._db
    assert {t for (t,) in db.execute("SELECT typeof(v) FROM emb")} == {"blob"}
    assert db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    for text, v in rows.items():
        assert cache.lookup([text]) == [np.asarray(v, dtype=np.float32).tolist()]
    # Reopening finds schema 2 and leaves the rows alone.
    before = db.execute("SELECT k, v FROM emb ORDER BY k").fetchall()
    again = EmbeddingCache(path, "m", client=Base64Client({}))
    assert again._db.execute("SELECT k, v FROM emb ORDER BY k").fetchall() == before


def _open_and_read(path: str, texts: list[str], start_at: float) -> int:
    time.sleep(max(0.0, start_at - time.time()))
    cache = EmbeddingCache(path, "m", client=Base64Client({}))
    return len(cache.lookup(texts))


def test_concurrent_opens_migrate_once_without_errors(tmp_path):
    path = tmp_path / "shared.sqlite"
    vecs = _vectors(1500, seed=1)
    texts = [f"t{i}" for i in range(len(vecs))]
    _old_cache(path, {t: v.astype(np.float64).tolist() for t, v in zip(texts, vecs, strict=True)})
    start_at = time.time() + 3.0  # after the spawned workers have imported
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        assert list(pool.map(_open_and_read, [str(path)] * 3, [texts] * 3, [start_at] * 3)) == [len(texts)] * 3  # raises on "database is locked"
    db = sqlite3.connect(path)
    assert {t for (t,) in db.execute("SELECT typeof(v) FROM emb")} == {"blob"}
    assert db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
    cache = EmbeddingCache(path, "m", client=Base64Client({}))
    assert np.array_equal(np.array(cache.lookup(texts), dtype=np.float32), vecs)


def test_fake_embeddings_are_float32_so_fresh_and_cached_vectors_agree(tmp_path):
    from ape.llm.fake import FakeEmbeddingsClient, bow_vector

    v = bow_vector("refund policy for EU gold customers")
    assert np.asarray(v, dtype=np.float32).tolist() == v
    cache = EmbeddingCache(tmp_path / "e.sqlite", "fake-bow", client=FakeEmbeddingsClient())
    assert asyncio.run(cache.embed(["refund policy for EU gold customers"])) == [v]
