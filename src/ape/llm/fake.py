"""Offline stand-ins used by dry runs and tests (never by gate runs)."""

import hashlib
import re
from types import SimpleNamespace

import numpy as np

DIM = 256


def bow_vector(text: str) -> list[float]:
    """Hashed bag-of-words: lexically similar texts get similar vectors."""
    v = np.zeros(DIM, dtype=np.float64)
    for tok in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
    n = np.linalg.norm(v)
    # float32 values, as OpenAI returns: the cache stores float32 (ape.llm.embeddings), so fresh and cached agree.
    return (v / n if n else v).astype(np.float32).tolist()


class FakeEmbeddingsClient:
    """Mimics `openai.AsyncOpenAI().embeddings.create`."""

    def __init__(self):
        self.calls = 0
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input, encoding_format=None):  # floats whatever the format, like many clients
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=bow_vector(t)) for t in input],
            usage=SimpleNamespace(prompt_tokens=len(input)),
        )


_ID = re.compile(r"(?:Policy|Exception|Procedure|Tool) ([A-Za-z0-9_-]+?)(?=[\s.(,:]|$)")


async def perfect_author(system: str, user: str) -> dict:
    """Scripted APG author: one unit per paragraph, IDs found by regex (the ceiling an LLM author can reach)."""
    excerpt = user.split("Excerpt:\n", 1)[1].split("\n", 1)[1]  # drop the "[Document title]" header line
    units = []
    for para in excerpt.split("\n\n"):
        if not para.strip():
            continue
        ids = _ID.findall(para)
        units.append(
            {
                "declares": ids[:1],
                "title": " ".join(para.split()[:14]),
                "description": para[:120],
                "knowledge": para,
                "references": ids[1:],
                "tools": re.findall(r"call (\w+)", para),
            }
        )
    units.append({"declares": [], "title": "empty", "description": "", "knowledge": "  ", "references": [], "tools": []})
    return {"units": units}
