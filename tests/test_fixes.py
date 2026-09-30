"""Regression tests for the EXPERIMENT_AUDIT blocking fixes (B1-B4, logging gaps)."""

import asyncio
import json

import pytest
from apg_core import load_graph
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.apg.adapter import compile_context
from ape.apg.arm import ApgArm, embed_graph
from ape.apg.oracle import build_oracle
from ape.build import build
from ape.kb.baselines import RandomUnits
from ape.kb.provenance import ChunkIndex
from ape.llm.embeddings import EMBED_CONTEXT, EmbeddingCache
from ape.llm.fake import FakeEmbeddingsClient
from ape.llm.ledger import Ledger
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.llm.tokens import count_tokens
from ape.tasks.gate import gate
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def _eval(offline_env, **gate_args):
    return inspect_eval(
        gate(**gate_args),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        model_roles={"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)},
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]


# ---- B3a: exception rendering styles are paired ----

def test_exception_styles_are_paired_and_differ_only_in_exception_text():
    d = make_world("F7", "100", "dev", 0, 30, exception_style="descriptive")
    i = make_world("F7", "100", "dev", 0, 30, exception_style="id_only")
    assert "-rel-desc-" in d.id and "-rel-idonly-" in i.id
    assert [t.gold for t in d.tasks] == [t.gold for t in i.tasks]
    assert [p.outcome for p in d.policies] == [p.outcome for p in i.policies]
    for xd, xi in zip(d.exceptions, i.exceptions):
        policy = next(p for p in d.policies if p.id == xd.policy_id)
        assert policy.region in d.facts[xd.fact_id].text and f"{policy.region} region" not in i.facts[xi.fact_id].text
    changed = {f for f in d.facts if d.facts[f].text != i.facts[f].text}
    assert changed == {x.fact_id for x in d.exceptions}


# ---- B3b: pull delivery ----

def test_pull_mode_logs_pulls_and_rejects_arms_without_a_retriever(offline_env):
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log = _eval(offline_env, family="F7", level="10", split="dev", arm="S3s", delivery="pull")
    assert log.status == "success", log.error
    for s in log.samples:
        sources = [r["source"] for r in s.store["compile_log"]]
        assert sources and set(sources) == {"pull"}, "pull-only delivery must not push"
        assert "search_kb" in s.store["step_log"][0]["exposed_tools"]
    with pytest.raises(ValueError, match="no retriever"):
        gate(family="F7", level="10", split="dev", arm="S1", delivery="pull")


# ---- logging gaps ----

def test_compile_and_step_logs_carry_the_diagnostic_fields(offline_env):
    asyncio.run(build("dev", "F3", ["5"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log = _eval(offline_env, family="F3", level="5", split="dev", arm="S5o")
    assert log.status == "success", log.error
    for s in log.samples:
        for rec in s.store["compile_log"]:
            assert rec["compile_ms"] >= 0 and rec["source"] == "push"
            assert rec["meta"]["kg_model"].endswith("mockllm/model") or "mockllm" in rec["meta"]["kg_model"]
            gold = rec["meta"]["gold"]
            assert set(gold) == {"nodes", "in_shortlist", "in_matches", "in_contributors"} and gold["nodes"]
            assert isinstance(rec["meta"]["route"]["shortlist"], list)
        assert all("exposed_tools" in step for step in s.store["step_log"])


def test_kg_role_is_required(tmp_path):
    w = make_world("F7", "10", "dev", 0, 1)
    emb = EmbeddingCache(tmp_path / "e.sqlite", "fake", client=FakeEmbeddingsClient())
    doc = asyncio.run(embed_graph(build_oracle(w, 2000), emb, {}))
    with pytest.raises(Exception, match="kg"):
        ApgArm("S5o", True, doc, emb, ChunkIndex(chunk_world(w)), 2000)


def test_embedding_ledger_entries_are_attributed(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    emb = EmbeddingCache(tmp_path / "e.sqlite", "fake", client=FakeEmbeddingsClient(), ledger=ledger)
    token = EMBED_CONTEXT.set({"arm": "S3s", "world": "w1", "sample": "t1"})
    try:
        asyncio.run(emb.embed(["a query"]))
    finally:
        EMBED_CONTEXT.reset(token)
    assert ledger.totals(arm="S3s", world="w1")["calls"] == 1


# ---- B2: placebo sizing ----

def test_random_placebo_never_exceeds_half_the_corpus():
    w = make_world("F7", "10", "dev", 0, 3)
    chunks = chunk_world(w)
    corpus = sum(count_tokens(c.text) for c in chunks)
    arm = RandomUnits(chunks, target_tokens=2000)
    assert arm.target <= corpus // 2
    ctx = asyncio.run(arm.compile("", w.tasks[0]))
    assert ctx.tokens <= arm.target + 50 and set(ctx.fact_ids) != set(w.facts)


# ---- B1: APG fill for matched-budget comparisons ----

def test_apg_fill_adds_context_up_to_the_budget(fake_embeddings):
    from ape.llm.mock_agent import mock_classifier

    w = make_world("F7", "1000", "dev", 0, 3)
    doc = asyncio.run(embed_graph(build_oracle(w, 2000), fake_embeddings, {}))
    doc["defaults"]["routing"]["shortlistK"] = 40
    g = load_graph(doc)
    kg = get_model("mockllm/model", custom_outputs=mock_classifier, memoize=False)
    q = w.tasks[0].prompt
    plain = asyncio.run(compile_context(g, q, kg, fake_embeddings, 1500))
    filled = asyncio.run(compile_context(g, q, kg, fake_embeddings, 1500, fill=True))
    assert count_tokens(filled.text) <= 1500
    assert count_tokens(filled.text) > count_tokens(plain.text) and filled.route["filled"]
    assert set(plain.contributors) <= set(filled.contributors)
