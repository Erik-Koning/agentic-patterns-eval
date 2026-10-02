"""Offline sizes of the registry families (F1, F2) for config/budget_assumptions.yaml. No model, no network.

    uv run python readiness/measure_registry.py

Per cell, over SEEDS dev worlds: corpus tokens (the S1 monolith), shared chunks per world, record tokens, and the
reference trajectory's history per prior agent call under the assumed calls_per_sample: every lookup's tool call
and record stay in the history, the lookups spread evenly over the calls before the answer call.
"""

import json
import statistics

from ape.llm.tokens import count_tokens
from ape.worlds.gen_registry import render_record
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world, corpus_text

SEEDS = range(5)
CELLS = [("F1", "2"), ("F1", "8"), ("F1", "32"), ("F2", "2"), ("F2", "5"), ("F2", "10")]
CALLS = {"F1-2": 4, "F1-32": 12, "F2-2": 4, "F2-10": 11}  # config/budget_assumptions.yaml calls_per_sample (push)
# One lookup's call in the history: the tool call's name and arguments plus message framing.
CALL_OVERHEAD = count_tokens(json.dumps({"name": "lookup_supplier", "arguments": {"supplier_id": "SUP-12345"}})) + 10


def measure() -> dict[str, dict]:
    out = {}
    for family, level in CELLS:
        worlds = [make_world(family, level, "dev", i, 12) for i in SEEDS]
        corpus = statistics.mean(count_tokens(corpus_text(w)) for w in worlds)
        chunks = statistics.mean(len(chunk_world(w)) for w in worlds)
        record = statistics.mean(count_tokens(render_record(r)) for w in worlds for r in w.entities["suppliers"].values())
        lookups = int(level)  # F1: one per supplier; F2: one per hop
        cell = f"{family}-{level}"
        row = {
            "corpus_tokens": round(corpus),
            "chunks_per_world": round(chunks),
            "record_tokens": round(record),
            "lookup_tokens": round(lookups * (record + CALL_OVERHEAD)),
        }
        if cell in CALLS:
            row["history_per_prior_call"] = round(lookups * (record + CALL_OVERHEAD) / (CALLS[cell] - 1))
        out[cell] = row
    return out


if __name__ == "__main__":
    print(json.dumps(measure(), indent=1))
