"""JSONL ledger for model calls made outside Inspect (index builds, authoring, embeddings).

Inspect meters calls made inside a sample; build-time calls happen offline, so
they are appended here and priced later from the same price table. Every ledger file
registers itself in the program-wide spend registry (`ape.spend`) on its first append.

What is metered (RELIABILITY_REVIEW K7):
- every completed build chat call, retries and unusable replies included, from the API's usage
  (`ape.llm.build_client`); every completed embedding request (`ape.llm.embeddings`), attributed to its world
  and system at build time and to its arm, sample and epoch at query time;
- a build chat call cancelled in flight or timed out client-side, which the server may still bill: recorded with
  its estimated input tokens, no output tokens and `context["status"]` ("cancelled" or "timeout").

Not metered: the output of a call we stopped waiting for (unknowable), embedding requests cancelled in flight (a
few hundred input tokens at most), and calls that fail with an API error (not billed). Nor are the OpenAI SDK's own
retries inside one call (`max_retries`, on timeouts, 429s and 5xx): only the attempt that returned is recorded, so
an earlier attempt the server completed and billed before the client timed out goes unrecorded. Inspect's calls
(agent, kg, judge) have the same blind spot. On a healthy API it is rare; the OpenAI dashboard is the backstop.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..spend import label as spend_label
from ..spend import register_ledger


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
    label: str | None = None  # the spend label (APE_SPEND_LABEL) of the run that made the call, set on append (R-C1)


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        # Several build processes (ape.artifacts workers, FX-4) append to one ledger, and the thread
        # lock only orders this process's writers. So each entry goes to the kernel as ONE write() of
        # one complete line on an O_APPEND descriptor: the kernel moves to end of file and writes the
        # bytes in a single step, so lines from different processes never interleave. (Buffered text
        # mode may split a line over several write()s; an unbuffered write leaves nothing to flush.)
        # Entries are a few hundred bytes, well under the 4 KB that stays whole on any filesystem.
        # The program-wide spend registry learns of this file before its first entry (once per process).
        register_ledger(self.path)
        if entry.label is None:  # attributed per entry: one cache/ledger.jsonl serves every study's builds (R-C1)
            entry.label = spend_label()
        data = (json.dumps(asdict(entry), sort_keys=True) + "\n").encode()
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                written = os.write(fd, data)
            finally:
                os.close(fd)
        if written != len(data):
            raise OSError(f"short write to ledger {self.path}: {written} of {len(data)} bytes")

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
