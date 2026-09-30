import asyncio
from types import SimpleNamespace

import pytest

from ape.llm.embeddings import CachedEmbeddingsConnector, EmbeddingCache, EmbeddingMiss
from ape.llm.ledger import Ledger, LedgerEntry
from ape.llm.tokens import count_tokens, truncate_to_tokens


def test_token_counts_are_stable():
    assert count_tokens("") == 0
    assert count_tokens("hello world") == 2
    text = "Refund requests over $250 in the EU require a Tier-2 supervisor."
    assert count_tokens(text) == count_tokens(text)
    assert count_tokens(truncate_to_tokens(text, 5)) == 5


class FakeEmbeddingsClient:
    def __init__(self):
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input):
        self.calls += 1
        data = [SimpleNamespace(embedding=[float(len(t)), 1.0]) for t in input]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7 * len(input)))


def test_embedding_cache_hits_skip_network(tmp_path):
    client = FakeEmbeddingsClient()
    ledger = Ledger(tmp_path / "ledger.jsonl")
    cache = EmbeddingCache(tmp_path / "emb.sqlite", "fake-embed", client=client, ledger=ledger)

    first = asyncio.run(cache.embed(["a", "bb", "a"]))
    assert first == [[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]
    assert client.calls == 1

    second = asyncio.run(cache.embed(["bb", "a"]))
    assert second == [[2.0, 1.0], [1.0, 1.0]]
    assert client.calls == 1  # served from cache

    assert ledger.totals(role="embeddings") == {
        "input_tokens": 14,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "calls": 1,
    }


def test_connector_lookup_never_calls_network(tmp_path):
    client = FakeEmbeddingsClient()
    cache = EmbeddingCache(tmp_path / "emb.sqlite", "fake-embed", client=client)
    asyncio.run(cache.embed(["known"]))
    conn = CachedEmbeddingsConnector(cache)
    assert conn.embed(["known"]) == [[5.0, 1.0]]
    with pytest.raises(EmbeddingMiss):
        conn.embed(["unknown"])
    assert client.calls == 1


def test_ledger_filters_on_context(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append(LedgerEntry(role="build", model="m", kind="chat", input_tokens=10, context={"system": "apg"}))
    ledger.append(LedgerEntry(role="build", model="m", kind="chat", input_tokens=5, context={"system": "lightrag"}))
    assert ledger.totals(system="apg")["input_tokens"] == 10
    assert ledger.totals(role="build")["calls"] == 2
