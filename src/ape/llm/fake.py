"""Offline stand-ins used by dry runs and tests (never by gate runs)."""

import hashlib
from types import SimpleNamespace

import numpy as np

DIM = 256


def bow_vector(text: str) -> list[float]:
    """Hashed bag-of-words: lexically similar texts get similar vectors."""
    v = np.zeros(DIM, dtype=np.float64)
    for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


class FakeEmbeddingsClient:
    """Mimics `openai.AsyncOpenAI().embeddings.create`."""

    def __init__(self):
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=bow_vector(t)) for t in input],
            usage=SimpleNamespace(prompt_tokens=len(input)),
        )
