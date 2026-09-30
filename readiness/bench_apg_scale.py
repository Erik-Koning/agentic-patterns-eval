"""Readiness check A3: APG kernel latency and outline size on a synthetic 1,000-leaf graph.

No network, no model calls: the LLM connector is a stub that picks the first
shortlisted node, and embeddings are seeded random unit vectors.

Pass criteria (READINESS.md A3): validate < 5 s; route p95 < 1 s with precomputed
embeddings; compose < 50 ms; outline < 2.5k tokens with the embedding shortlist.
"""

import json
import random
import statistics
import time

import tiktoken
from apg_core import compose, load_graph, route, validate_graph
from apg_core.outline import serialize_outline

N_CATEGORIES = 20
LEAVES_PER_CATEGORY = 50
DIM = 1536
ROUTES = 50

rng = random.Random(20260929)
enc = tiktoken.get_encoding("o200k_base")


def unit_vector() -> list[float]:
    v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def build_doc() -> dict:
    nodes = [
        {
            "id": "root",
            "parentId": None,
            "routable": False,
            "title": "Synthetic policy handbook",
            "prompt": {"slots": {"persona": "You are a policy assistant.", "task": "Answer using the policies."}},
        }
    ]
    for c in range(N_CATEGORIES):
        cid = f"cat-{c:02d}"
        nodes.append(
            {
                "id": cid,
                "parentId": "root",
                "type": "category",
                "routable": False,
                "title": f"Policy domain {c}",
                "description": f"Policies governing domain {c}.",
            }
        )
        for p in range(LEAVES_PER_CATEGORY):
            pid = f"pol-{c:02d}-{p:02d}"
            nodes.append(
                {
                    "id": pid,
                    "parentId": cid,
                    "type": "category",
                    "routable": True,
                    "title": f"Policy {c}.{p}: handling case type {p} in domain {c}",
                    "description": f"When a request in domain {c} involves case type {p}, apply rule {c}-{p}.",
                    "embedding": unit_vector(),
                    "prompt": {
                        "slots": {
                            "knowledge": f"Rule {c}-{p}: requests of case type {p} in domain {c} require approval "
                            f"level {p % 4} and must be completed within {(p % 7) + 1} business days.",
                            "constraints": f"Never skip the approval step for rule {c}-{p}.",
                        }
                    },
                }
            )
    return {
        "schemaVersion": "1.0",
        "graphId": "bench-1000",
        "version": "1",
        "profile": "L1",
        "defaults": {"routing": {"minConfidence": 0.55, "allowMulti": True}, "budget": {"maxPromptTokens": 6000}},
        "nodes": nodes,
        "edges": [],
    }


class StubLlm:
    """Deterministic classify: picks the last node listed in the outline."""

    def __init__(self) -> None:
        self.outline_tokens: list[int] = []

    def classify(self, query, outline, schema, multi):
        self.outline_tokens.append(len(enc.encode(outline)))
        first = outline.strip().splitlines()[-1].strip().split(":", 1)[0]
        return [{"nodeId": first, "confidence": 0.9, "reason": "stub"}]


class StubEmbeddings:
    def embed(self, texts):
        return [unit_vector() for _ in texts]


def pctl(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> None:
    doc = build_doc()
    t0 = time.perf_counter()
    result = validate_graph(doc)
    validate_s = time.perf_counter() - t0
    graph = load_graph(doc)

    llm = StubLlm()
    connectors = {"llm": llm, "embeddings": StubEmbeddings()}
    route_s, compose_ms, last = [], [], None
    for i in range(ROUTES):
        t0 = time.perf_counter()
        r = route(f"question {i} about a policy case", graph, connectors)
        route_s.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        last = compose(graph, [m["nodeId"] for m in r["matches"]], {"query": "q"})
        compose_ms.append((time.perf_counter() - t0) * 1000)

    full_outline_tokens = len(enc.encode(serialize_outline(graph, None)))

    report = {
        "nodes": len(doc["nodes"]),
        "validate_valid": result["valid"],
        "validate_errors": result["errors"][:3],
        "validate_warnings": len(result["warnings"]),
        "validate_s": round(validate_s, 3),
        "route_p50_s": round(statistics.median(route_s), 4),
        "route_p95_s": round(pctl(route_s, 0.95), 4),
        "compose_p95_ms": round(pctl(compose_ms, 0.95), 2),
        "outline_tokens_shortlisted_max": max(llm.outline_tokens),
        "outline_tokens_without_embeddings": full_outline_tokens,
        "composed_contributors": len(last["contributors"]) if last else None,
    }
    checks = {
        "validate<5s": validate_s < 5,
        "route_p95<1s": report["route_p95_s"] < 1,
        "compose_p95<50ms": report["compose_p95_ms"] < 50,
        "outline<2500tok": report["outline_tokens_shortlisted_max"] < 2500,
        "valid": result["valid"],
    }
    print(json.dumps({"report": report, "checks": checks, "pass": all(checks.values())}, indent=2))


if __name__ == "__main__":
    main()
