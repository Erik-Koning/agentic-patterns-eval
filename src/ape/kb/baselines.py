"""Non-KG delivery arms: S1 monolith, S3s flat hybrid retrieval, S6 oracle context, S7 random units.

All of them read the same chunk set (`worlds.render.chunk_world`), so content is
identical across arms; only selection and delivery differ.
"""

import random

import numpy as np
from rank_bm25 import BM25Okapi

from ..llm.embeddings import EmbeddingCache
from ..llm.tokens import count_tokens
from ..worlds.render import Chunk, corpus_text
from ..worlds.spec import TaskItem, World
from .context import ContextResult

RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def _pack(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Greedy in rank order; always keeps at least one chunk."""
    out, used = [], 0
    for c in chunks:
        n = count_tokens(c.text)
        if out and used + n > budget:
            break
        out.append(c)
        used += n
    return out


def _pack_under(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Like `_pack` but never exceeds the budget (skips chunks that don't fit); the placebo must stay small."""
    out, used = [], 0
    for c in chunks:
        n = count_tokens(c.text)
        if used + n <= budget:
            out.append(c)
            used += n
    return out or [min(chunks, key=lambda c: count_tokens(c.text))]


def _result(chunks: list[Chunk], **meta) -> ContextResult:
    return ContextResult(
        text="\n\n".join(c.text for c in chunks),
        unit_ids=[c.id for c in chunks],
        fact_ids=list(dict.fromkeys(f for c in chunks for f in c.fact_ids)),
        meta=meta,
    )


class Monolith:
    """S1: the whole corpus, compiled once per task."""

    name = "S1"
    per_step = False

    def __init__(self, world: World, chunks: list[Chunk]):
        self._text = corpus_text(world)
        self._units = [c.id for c in chunks]
        self._facts = list(world.facts)

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        return ContextResult(self._text, self._units, self._facts)


class FlatHybrid:
    """S3s: BM25 + dense retrieval fused by reciprocal rank, packed to a token budget, refreshed every step."""

    name = "S3s"
    per_step = True

    def __init__(self, chunks: list[Chunk], embeddings: EmbeddingCache, budget_tokens: int, depth: int = 50):
        self.chunks = chunks
        self.embeddings = embeddings
        self.budget = budget_tokens
        self.depth = depth
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._matrix = np.array(embeddings.lookup([c.text for c in chunks]), dtype=np.float32)
        self._matrix /= np.linalg.norm(self._matrix, axis=1, keepdims=True) + 1e-12

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        q = np.array((await self.embeddings.embed([query]))[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        dense_rank = np.argsort(-(self._matrix @ q), kind="stable")[: self.depth]
        bm25_rank = np.argsort(-self._bm25.get_scores(_tokenize(query)), kind="stable")[: self.depth]
        score: dict[int, float] = {}
        for ranking in (dense_rank, bm25_rank):
            for r, i in enumerate(ranking):
                score[int(i)] = score.get(int(i), 0.0) + 1.0 / (RRF_K + r + 1)
        order = sorted(score, key=lambda i: (-score[i], i))
        return _result(_pack([self.chunks[i] for i in order], self.budget), budget=self.budget)


class OracleContext:
    """S6: exactly the gold facts' canonical text (upper bound on selection)."""

    name = "S6"
    per_step = False

    def __init__(self, world: World):
        self.world = world

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        facts = task.gold_fact_ids
        return ContextResult("\n\n".join(self.world.facts[f].text for f in facts), list(facts), list(facts))


class RandomUnits:
    """S7: random chunks, token-matched to APG-s's realized context, seeded per task (the placebo).

    The target is capped at `max_corpus_fraction` of the corpus: in small KBs an uncapped
    target would deliver the whole corpus and stop being a placebo (EXPERIMENT_AUDIT B2).
    """

    name = "S7"
    per_step = False

    def __init__(self, chunks: list[Chunk], target_tokens: int, seed: int = 0, max_corpus_fraction: float = 0.5):
        self.chunks = chunks
        corpus = sum(count_tokens(c.text) for c in chunks)
        self.target = max(1, min(target_tokens, int(max_corpus_fraction * corpus)))
        self.seed = seed

    async def compile(self, query: str, task: TaskItem) -> ContextResult:
        rng = random.Random(f"{self.seed}|{task.id}")
        order = rng.sample(self.chunks, len(self.chunks))
        return _result(_pack_under(order, self.target), target=self.target)
