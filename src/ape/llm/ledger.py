"""JSONL ledger for model calls made outside Inspect (index builds, authoring, embeddings).

Inspect meters calls made inside a sample; build-time calls happen offline, so
they are appended here and priced later from the same price table.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LedgerEntry:
    role: str  # "build" | "kg" | "embeddings" | ...
    model: str
    kind: str  # "chat" | "embed"
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    context: dict = field(default_factory=dict)  # e.g. {"world": "...", "system": "apg" | "lightrag"}
    ts: float = field(default_factory=time.time)


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        line = json.dumps(asdict(entry), sort_keys=True)
        with self._lock, self.path.open("a") as f:
            f.write(line + "\n")

    def read(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        return [LedgerEntry(**json.loads(line)) for line in self.path.read_text().splitlines() if line]

    def totals(self, **match: str) -> dict[str, int]:
        """Sum token fields over entries whose attributes or context keys equal `match`."""
        out = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0, "calls": 0}
        for e in self.read():
            fields = {**e.context, "role": e.role, "model": e.model, "kind": e.kind}
            if all(fields.get(k) == v for k, v in match.items()):
                out["calls"] += 1
                for k in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
                    out[k] += getattr(e, k)
        return out
