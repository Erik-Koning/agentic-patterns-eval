"""Shared offline fakes: no test in this suite may touch the network, or the program's spend registry."""

import os

import pytest

from ape.llm.fake import FakeEmbeddingsClient, bow_vector  # noqa: F401  (re-exported for tests)
from ape.spend import REGISTRY_ENV


@pytest.fixture(scope="session", autouse=True)
def isolated_spend_registry(tmp_path_factory):
    """No test writes spend registrations to cache/spend/registry.jsonl: the session gets a temporary registry
    (for module-scoped fixtures), and every test its own (`per_test_spend_registry`), so one test's ledgers and
    logs never enter another's program spend. Fixtures that clear APE_* variables keep it (ape.spend raises
    under pytest without it)."""
    saved = os.environ.get(REGISTRY_ENV)
    os.environ[REGISTRY_ENV] = str(tmp_path_factory.mktemp("spend") / "registry.jsonl")
    yield os.environ[REGISTRY_ENV]
    if saved is None:
        os.environ.pop(REGISTRY_ENV, None)
    else:
        os.environ[REGISTRY_ENV] = saved


@pytest.fixture(autouse=True)
def per_test_spend_registry(tmp_path_factory, monkeypatch):
    monkeypatch.setenv(REGISTRY_ENV, str(tmp_path_factory.mktemp("spend-test") / "registry.jsonl"))


@pytest.fixture
def fake_embeddings(tmp_path):
    from ape.llm.embeddings import EmbeddingCache

    return EmbeddingCache(tmp_path / "emb.sqlite", "fake-embed", client=FakeEmbeddingsClient())
