"""Shared offline fakes: no test in this suite may touch the network."""

import pytest

from ape.llm.fake import FakeEmbeddingsClient, bow_vector  # noqa: F401  (re-exported for tests)


@pytest.fixture
def fake_embeddings(tmp_path):
    from ape.llm.embeddings import EmbeddingCache

    return EmbeddingCache(tmp_path / "emb.sqlite", "fake-embed", client=FakeEmbeddingsClient())
