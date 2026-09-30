"""APG adapter (module 5): classify port, record/replay routing, determinism, concurrency, fallback."""

import asyncio
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from apg_core import ScriptedLlm, compose, load_graph, route
from inspect_ai.model import ModelOutput, get_model

from ape.apg.adapter import compile_context
from ape.apg.arm import embed_graph
from ape.apg.classify import CLASSIFY_SYSTEM
from ape.apg.oracle import build_oracle
from ape.apg.schema import schema_errors
from ape.llm.embeddings import CachedEmbeddingsConnector
from ape.llm.mock_agent import MODEL, mock_classifier
from ape.llm.tokens import count_tokens
from ape.worlds.generate import make_world

APG_ROOT = Path.home() / "Documents/Work/Code/EvolvingWisdomAgents"
VET = APG_ROOT / "templates/vet-clinic.apg.json"
QUERIES = ["labrador puppy feeding", "my cat has hairballs", "is chocolate toxic to dogs", "book a dental cleaning", "what do you charge for boarding"]


def kg_model(fn=mock_classifier):
    return get_model("mockllm/model", custom_outputs=fn, memoize=False)


@pytest.mark.skipif(not APG_ROOT.exists(), reason="APG repo not on this machine")
def test_classify_prompt_is_string_equal_to_ts_driver():
    ts = (APG_ROOT / "typescript/packages/connectors/src/anthropic.ts").read_text()
    m = re.search(r'const CLASSIFY_SYSTEM =\s*"((?:[^"\\]|\\.)*)";', ts)
    assert m and json.loads(f'"{m.group(1)}"') == CLASSIFY_SYSTEM


async def _embedded(doc, fake_embeddings):
    return await embed_graph(doc, fake_embeddings, {"test": True})


@pytest.mark.skipif(not VET.exists(), reason="vet-clinic template not on this machine")
async def test_record_replay_equals_direct_route_on_vet_clinic(fake_embeddings):
    doc = await _embedded(json.loads(VET.read_text()), fake_embeddings)
    assert not schema_errors(doc)
    graph = load_graph(doc)
    conn = CachedEmbeddingsConnector(fake_embeddings)
    model = kg_model()
    for q in QUERIES:
        c = await compile_context(graph, q, model, fake_embeddings, 40000)
        # Direct route with the same classification, via the kernel's own ScriptedLlm.
        if c.route["bypass"]:
            direct = route(q, graph, {"llm": ScriptedLlm({}), "embeddings": conn})
        else:
            outline_first = c.route["matches"][0]["nodeId"] if not c.route["fallback"] else None
            scripted = [{"nodeId": outline_first, "confidence": 0.9, "reason": "mock"}] if outline_first else []
            direct = route(q, graph, {"llm": ScriptedLlm({"classify": [{"matches": scripted}]}), "embeddings": conn})
        assert direct["matches"] == c.route["matches"]
        composed = compose(graph, [m["nodeId"] for m in direct["matches"]], {"query": q, "maxPromptTokens": 40000, "countTokens": count_tokens})
        assert composed["text"] == c.text


def _graphs():
    return [build_oracle(make_world(f, lvl, "dev", 0, 5), 2000) for f, lvl in [("F7", "100"), ("F3", "20"), ("F5", "2hop")]]


async def test_compose_hash_is_stable_and_concurrency_is_clean(fake_embeddings):
    docs = [await _embedded(d, fake_embeddings) for d in _graphs()]
    for d in docs:
        assert not schema_errors(d), schema_errors(d)[:3]
    graphs = [load_graph(d) for d in docs]
    model = kg_model()
    queries = ["refund request EU gold", "lost parcel order EU", "who managed Atlas", "tax exemption UK", "damaged item APAC"]
    jobs = [(g, q) for g in graphs for q in queries]
    sequential = [hashlib.sha256((await compile_context(g, q, model, fake_embeddings, 2000)).text.encode()).hexdigest() for g, q in jobs]
    again = [hashlib.sha256((await compile_context(g, q, model, fake_embeddings, 2000)).text.encode()).hexdigest() for g, q in jobs]
    assert sequential == again
    concurrent = await asyncio.gather(*(compile_context(g, q, model, fake_embeddings, 2000) for g, q in jobs * 4))
    assert [hashlib.sha256(c.text.encode()).hexdigest() for c in concurrent] == sequential * 4


async def test_malformed_classification_falls_back_with_error(fake_embeddings):
    doc = await _embedded(_graphs()[0], fake_embeddings)
    bad = kg_model(lambda *a: ModelOutput.from_content(MODEL, "not json"))
    c = await compile_context(load_graph(doc), "refund request EU", bad, fake_embeddings, 2000)
    assert c.classify_error and c.route["fallback"]


def test_oracle_compile_is_identical_across_processes(tmp_path):
    script = (
        "import asyncio, hashlib, json, sys\n"
        "from apg_core import load_graph\n"
        "from inspect_ai.model import get_model\n"
        "from ape.apg.adapter import compile_context\n"
        "from ape.apg.arm import embed_graph\n"
        "from ape.apg.oracle import build_oracle\n"
        "from ape.llm.embeddings import EmbeddingCache\n"
        "from ape.llm.fake import FakeEmbeddingsClient\n"
        "from ape.llm.mock_agent import mock_classifier\n"
        "from ape.worlds.generate import make_world\n"
        "async def main():\n"
        f"    emb = EmbeddingCache(r'{tmp_path}/e.sqlite', 'fake', client=FakeEmbeddingsClient())\n"
        "    doc = await embed_graph(build_oracle(make_world('F7','100','dev',0,5), 2000), emb, {})\n"
        "    m = get_model('mockllm/model', custom_outputs=mock_classifier)\n"
        "    c = await compile_context(load_graph(doc), 'refund request EU gold', m, emb, 2000)\n"
        "    print(hashlib.sha256(c.text.encode()).hexdigest())\n"
        "asyncio.run(main())\n"
    )
    runs = {subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.strip() for _ in range(2)}
    assert len(runs) == 1 and len(next(iter(runs))) == 64
