"""Compile APG context inside Inspect without calling back into the event loop from sync code.

`apg_core.route` is synchronous and calls `llm.classify` inline. We run it twice in
a worker thread (record/replay):

1. Pass 1 uses a recording LLM that captures (query, outline, multi) and returns no
   matches. If routing never calls the LLM (embedBypass fired), pass 1 is final.
2. The captured outline is classified asynchronously with the "kg" model.
3. Pass 2 replays the parsed matches through a fresh `ScriptedLlm` and asserts the
   outline is byte-identical, then `compose` runs on the resulting targets.

Connectors are passed per call; the global registry (`bind_connectors`) is never used.

`fill=True` (matched-budget analyses only) adds further shortlisted nodes, in order of
embedding similarity to the query, as secondary targets until the next one would exceed
the budget. Default APG behaviour (`fill=False`) composes only what routing selected.
"""

import asyncio
import hashlib
from dataclasses import dataclass, field

import numpy as np
from apg_core import Graph, ScriptedLlm, compose, route
from inspect_ai.model import Model

from ..llm.embeddings import CachedEmbeddingsConnector, EmbeddingCache
from ..llm.tokens import count_tokens
from .classify import classify


class _RecordingLlm:
    def __init__(self, replay: list[dict] | None = None):
        self.calls: list[dict] = []
        self._replay = ScriptedLlm({"classify": [{"matches": replay}]}) if replay is not None else None

    def classify(self, query, outline, schema, multi):
        self.calls.append({"query": query, "outline": outline, "multi": multi})
        return self._replay.classify() if self._replay else []


@dataclass
class Compiled:
    text: str
    contributors: list[str]
    tool_allowlist: list[str] | None
    truncated: list[dict]
    route: dict
    classify_error: str | None = None
    meta: dict = field(default_factory=dict)


async def compile_context(
    graph: Graph,
    query: str,
    kg_model: Model,
    embeddings: EmbeddingCache | None,
    max_prompt_tokens: int,
    fill: bool = False,
) -> Compiled:
    connectors: dict = {}
    if embeddings is not None:
        await embeddings.embed([query])  # the only network step; node vectors are precomputed
        connectors["embeddings"] = CachedEmbeddingsConnector(embeddings)

    first = _RecordingLlm()
    routed = await asyncio.to_thread(route, query, graph, {**connectors, "llm": first})
    error = None
    outline_sha = None
    if first.calls:
        call = first.calls[0]
        outline_sha = hashlib.sha256(call["outline"].encode()).hexdigest()
        matches, error = await classify(kg_model, call["query"], call["outline"], call["multi"])
        second = _RecordingLlm(replay=matches)
        routed = await asyncio.to_thread(route, query, graph, {**connectors, "llm": second})
        replay_sha = hashlib.sha256(second.calls[0]["outline"].encode()).hexdigest()
        assert replay_sha == outline_sha, "APG outline changed between record and replay passes"

    targets = [m["nodeId"] for m in routed["matches"]]
    opts = {"query": query, "maxPromptTokens": max_prompt_tokens, "countTokens": count_tokens}
    composed = compose(graph, targets, opts)
    filled: list[str] = []
    if fill and embeddings is not None:
        q = np.asarray(embeddings.lookup([query])[0], dtype=np.float64)
        ranked = sorted(
            (n for n in routed["shortlist"] if n not in targets and graph.get(n).get("embedding")),
            key=lambda n: (-float(np.dot(q, graph.get(n)["embedding"])), n),
        )
        for n in ranked:
            trial = compose(graph, [*targets, *filled, n], opts)
            if count_tokens(trial["text"]) > max_prompt_tokens or trial["truncated"]:
                break
            filled.append(n)
            composed = trial
    return Compiled(
        text=composed["text"],
        contributors=list(composed["contributors"]),
        tool_allowlist=composed.get("toolAllowlist"),
        truncated=list(composed["truncated"]),
        route={
            "matches": routed["matches"],
            "fallback": routed["fallbackUsed"],
            "shortlist": list(routed["shortlist"]),
            "bypass": not first.calls,
            "outline_sha": outline_sha,
            "filled": filled,
        },
        classify_error=error,
    )
