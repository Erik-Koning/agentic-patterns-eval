"""GraphRAG-Bench anchor (PC1): loader, official ACC scorer, pc1_check, and an offline end-to-end run."""

import asyncio
import json
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import resolve_sample_attachments

from ape.anchor import accuracy as acc
from ape.anchor import fixture
from ape.anchor.fixture import CORPUS, MODEL, answer_in, mock_answerer
from ape.anchor.fixture import all_questions as _questions
from ape.anchor.fixture import write_dataset as _write_dataset
from ape.anchor.graphragbench import QUESTION_TYPES, build_index, load_medical, pc1_check, pc1_from_logs, pc1_tolerance
from ape.config import Config
from ape.models import agent_model, load_profile, role_models
from ape.tasks.anchor_graphragbench import graphragbench_anchor


def _ids(bench) -> list[str]:
    return [q["id"] for q in bench.questions]


# --- loader ---


def test_loader_is_stratified_deterministic_and_prefix_stable(tmp_path):
    data = _write_dataset(tmp_path / "a")
    b2 = load_medical(data, n_per_type=2, seed=7)
    assert [q["question_type"] for q in b2.questions] == [t for t in QUESTION_TYPES for _ in range(2)]
    assert _ids(b2) == _ids(load_medical(data, 2, 7)), "same seed, same sample"
    assert set(_ids(b2)) <= set(_ids(load_medical(data, 3, 7))), "a larger n extends a smaller one"
    assert _ids(b2) != _ids(load_medical(data, 2, 8))
    assert len(load_medical(data, 50, 7).questions) == 20, "n above the pool size takes the whole pool"
    assert b2.documents == [{"corpus_name": "Medical", "context": CORPUS}]

    shuffled = _write_dataset(tmp_path / "b", questions=list(reversed(_questions())))
    assert _ids(load_medical(shuffled, 2, 7)) == _ids(b2), "file order must not matter"
    assert _ids(load_medical(data / "Datasets" / "Questions" / "medical_questions.json", 2, 7)) == _ids(b2)


# --- official ACC ---


def _judge(statements: dict[str, list[str]], classification: str):
    calls = []

    async def generate(prompt: str) -> str:
        calls.append(prompt)
        if "Current Analysis:" in prompt:
            return classification
        return json.dumps(statements[answer_in(prompt)])

    return generate, calls


async def _embed_same(texts):
    return [[1.0, 0.0]] * len(texts)


async def _embed_orthogonal(texts):
    return [[1.0, 0.0], [0.0, 1.0]]


def _cls(tp=(), fp=(), fn=()) -> str:
    return json.dumps({k: [{"statement": s, "reason": "r"} for s in v] for k, v in (("TP", tp), ("FP", fp), ("FN", fn))})


async def test_perfect_answer_scores_one():
    gen, calls = _judge({"A.": ["A."]}, _cls(tp=["A."]))
    r = await acc.answer_correctness("Q?", "A.", "A.", gen, _embed_same)
    assert r["accuracy"] == pytest.approx(1.0)
    assert "nuclear fission" in calls[-1], "classification prompt carries the official few-shot examples"


async def test_f1_and_weights_match_upstream():
    # The official sun example: TP=1, FP=1, FN=5 -> P=1/2, R=1/6, F1=1/4. Orthogonal embeddings -> SS=0.5.
    gen, _ = _judge({"ans": ["a1", "a2"], "gt": ["g1"] * 5}, _cls(tp=["a1"], fp=["a2"], fn=["g"] * 5))
    r = await acc.answer_correctness("Q?", "ans", "gt", gen, _embed_orthogonal)
    assert r["factual_correctness"] == pytest.approx(0.25)
    assert r["semantic_similarity"] == pytest.approx(0.5)
    assert r["accuracy"] == pytest.approx(0.75 * 0.25 + 0.25 * 0.5)


async def test_classification_parse_is_strict_like_upstream():
    fenced = f"```json\n{_cls(tp=['A.'])}\n```"
    gen, _ = _judge({"A.": ["A."]}, fenced)
    assert (await acc.answer_correctness("Q?", "A.", "A.", gen, _embed_same))["factual_correctness"] == 0.0
    assert acc.parse_classification('["not", "an", "object"]') == 0.0
    with pytest.raises(acc.Excluded):
        acc.parse_classification('{"TP": ["bare strings are ill-formed"]}')


async def test_empty_statement_lists_skip_classification():
    gen, calls = _judge({"": [], "x": []}, "unused")
    r = await acc.answer_correctness("Q?", "", "x", gen, _embed_same)
    assert r["factual_correctness"] == 1.0 and len(calls) == 2


def test_statement_parser_follows_upstream_fallbacks():
    assert acc.parse_statements('```json\n["a", "b"]\n```') == ["a", "b"]
    assert acc.parse_statements('{"statements": ["a"]}') == ["a"]
    assert acc.parse_statements('Output: {"x": "a", "y": "b"}') == ["a", "b"]
    assert acc.parse_statements("['a', 'b',]") == ["a", "b"]
    assert acc.parse_statements("no json here") == []


# --- PC1 ---

PUB = {"Fact Retrieval": 63.32, "Complex Reasoning": 61.32, "Contextual Summarize": 63.14, "Creative Generation": 67.91}


def test_pc1_passes_within_tolerance_and_above_naive():
    ours = {"Fact Retrieval": 60.0, "Complex Reasoning": 64.0, "Contextual Summarize": 59.0, "Creative Generation": 70.0}
    naive = {"Fact Retrieval": 64.0, "Complex Reasoning": 55.0, "Contextual Summarize": 60.0, "Creative Generation": 58.0}
    r = pc1_check(ours, PUB, 5.0, naive)
    assert r["pass"] and r["reproduces_published"] and r["beats_naive"]
    assert not r["per_type"]["Fact Retrieval"]["beats_naive"], "per-type naive wins are reported, not gating (as in the paper)"


def test_pc1_fails_outside_tolerance_or_below_naive():
    ours = {**PUB, "Complex Reasoning": PUB["Complex Reasoning"] - 6}
    naive = {t: v - 3 for t, v in PUB.items()}
    tight, loose = pc1_check(ours, PUB, 5.0, naive), pc1_check(ours, PUB, 10.0, naive)
    assert not tight["pass"] and not tight["per_type"]["Complex Reasoning"]["within_tolerance"]
    assert loose["pass"]
    assert not pc1_check(PUB, PUB, 5.0, {t: v + 1 for t, v in PUB.items()})["pass"], "naive above the graph mode"
    missing = pc1_check({k: v for k, v in PUB.items() if k != "Creative Generation"}, PUB, 5.0, PUB)
    assert not missing["pass"] and missing["macro"] is None


def test_pc1_tolerance_depends_on_the_papers_model():
    assert pc1_tolerance("openai/gpt-4o-mini", "gpt-4o-mini-2024-07-18") == 5.0
    assert pc1_tolerance("openai/gpt-4o-mini", "openai/gpt-4.1-mini") == 10.0


# --- end to end ---

extract_calls: list[str] = []
fake_extract = fixture.fake_extractor(extract_calls)


def mock_judge(messages, tools, tool_choice, config):
    assert config.temperature == 0.0 and config.seed == 42, "official judge settings"
    return fixture.mock_judge(messages, tools, tool_choice, config)


@pytest.fixture
def anchor_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("APE_INDICES", str(tmp_path / "indices"))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    data = _write_dataset(tmp_path / "data")
    extract_calls.clear()
    manifest = asyncio.run(build_index(load_medical(data, 0, 0), Config(), fake_extract, None))
    return tmp_path, data, manifest


def _run(mode, anchor_env):
    tmp_path, data, _ = anchor_env
    return inspect_eval(
        graphragbench_anchor(mode=mode, n_per_type=2, data=str(data)),
        model=agent_model(load_profile("anchor"), model=MODEL, custom_outputs=mock_answerer),
        model_roles=role_models(load_profile("anchor"), roles=("judge",), model=MODEL, custom_outputs=mock_judge),
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]


def test_anchor_runs_end_to_end_offline(anchor_env):
    _, _, manifest = anchor_env
    inputs = [re.search(r"---Input Text---\n```\n(.*?)\n```", p, re.S) for p in extract_calls]
    assert len({m.group(1) for m in inputs if m}) >= 2, "LightRAG chunked the corpus itself"
    assert manifest["chunking"] == [1200, 100] and manifest["index_hash"]

    mix = _run("mix", anchor_env)
    assert mix.status == "success", mix.error
    assert len(mix.samples) == 8
    m = {name: v.value for s in mix.results.scores for name, v in s.metrics.items()}
    assert m["Fact Retrieval"] == pytest.approx(100.0), "gold answers score 1.0 under the official ACC"
    assert all(m[t] < 30 for t in QUESTION_TYPES[1:]) and m["excluded"] == 0
    for s in map(resolve_sample_attachments, mix.samples):
        answer_calls = [e for e in s.events if e.event == "model" and e.role != "judge"]
        assert len(answer_calls) == 2, "one keyword call + one answer call, both metered in this sample"
        system = answer_calls[-1].input[0].text
        assert "---Knowledge Base---" in system and "{history}" not in system and "Imiquimod" in system
        assert len([e for e in s.events if e.event == "model" and e.role == "judge"]) == 3
        assert s.store["lightrag"]["llm_generated"]

    naive = _run("naive", anchor_env)
    assert naive.status == "success", naive.error
    for s in map(resolve_sample_attachments, naive.samples):
        (answer_call,) = [e for e in s.events if e.event == "model" and e.role != "judge"]
        assert "Imiquimod" in answer_call.input[0].text, "naive mode fills the same prompt's knowledge base"

    result = pc1_from_logs(mix, naive)
    assert result["mode"] == "mix" and result["tolerance_pp"] == 10.0
    assert result["per_type"]["Fact Retrieval"]["ours"] == pytest.approx(100.0)


def test_index_from_another_corpus_is_refused(anchor_env):
    _, data, _ = anchor_env
    _write_dataset(data, corpus=CORPUS + " Addendum.")
    with pytest.raises(RuntimeError, match="different corpus"):
        graphragbench_anchor(n_per_type=1, data=str(data))
