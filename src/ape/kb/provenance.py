"""Map delivered units back to world-spec facts and score evidence at fact granularity."""

from ..worlds.render import Chunk


class ChunkIndex:
    def __init__(self, chunks: list[Chunk]):
        self.by_id = {c.id: c for c in chunks}

    def facts(self, chunk_ids: list[str]) -> list[str]:
        out: list[str] = []
        for cid in chunk_ids:
            out.extend(self.by_id[cid].fact_ids)
        return list(dict.fromkeys(out))


def evidence_pr(delivered: list[str], gold: list[str]) -> tuple[float, float]:
    """(precision, recall) of delivered fact IDs against gold fact IDs; empty delivery has precision 0."""
    d, g = set(delivered), set(gold)
    hit = len(d & g)
    precision = hit / len(d) if d else 0.0
    recall = hit / len(g) if g else 1.0
    return precision, recall
