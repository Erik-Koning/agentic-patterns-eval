"""End-to-end dry run (readiness H2 for the baseline arms): mock agent, fake embeddings, no network."""

import asyncio

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.build import build
from ape.llm.mock_agent import mock_agent
from ape.tasks.gate import gate

REQUIRED_STORE = ["compile_log", "turns_used", "arm"]


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APE_WORLDS", str(tmp_path / "worlds"))
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5"), ("F5", "2hop")])
@pytest.mark.parametrize("arm", ["S1", "S3s", "S6", "S7"])
def test_baseline_arms_run_end_to_end(offline_env, family, level, arm):
    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    logs = inspect_eval(
        gate(family=family, level=level, split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        log_dir=str(offline_env / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert len(log.samples) == 2
    for s in log.samples:
        for key in REQUIRED_STORE:
            assert key in s.store, key
        rec = s.store["compile_log"][0]
        assert {"tokens", "prompt_hash", "unit_ids", "fact_ids"} <= set(rec)
        assert set(s.scores) == {"task_success", "delivered_evidence", "error_analysis"}
        assert s.scores["error_analysis"].metadata["error"]
    if arm == "S6":
        assert all(s.scores["delivered_evidence"].value["evidence_recall"] == 1.0 for s in log.samples)
    per_step = arm == "S3s"
    turns = log.samples[0].store["turns_used"]
    assert len(log.samples[0].store["compile_log"]) == (turns if per_step else 1)
    history = [m.text for m in log.samples[0].messages]
    assert not any("refreshed for this step" in t for t in history), "per-step context must never enter the history"
    assert ("## Knowledge base" in history[0]) == (not per_step), "per-query context lives in the system prompt"


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5"), ("F5", "2hop")])
@pytest.mark.parametrize("arm", ["S5o", "APGo-q"])
def test_apg_oracle_arms_run_end_to_end(offline_env, family, level, arm):
    from ape.llm.mock_agent import mock_classifier

    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    logs = inspect_eval(
        gate(family=family, level=level, split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        model_roles={"kg": get_model("mockllm/model", custom_outputs=mock_classifier, memoize=False)},
        log_dir=str(offline_env / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    for s in log.samples:
        recs = s.store["compile_log"]
        assert all(r["meta"]["graph_version"] and "route" in r["meta"] for r in recs)
        assert len(recs) == (s.store["turns_used"] if arm == "S5o" else 1)
        # The kg classify calls are metered as model events on the kg model.
        kg_calls = [e for e in s.events if e.event == "model" and e.role == "kg"]
        bypassed = sum(r["meta"]["route"]["bypass"] for r in recs)
        assert len(kg_calls) == len(recs) - bypassed
        if family == "F3":
            # APG scopes tools by the routed procedure's allowlist, never by name-matching.
            assert all(r["tools"] is None or isinstance(r["tools"], list) for r in recs)


def test_results_load_from_real_logs(offline_env):
    from ape.analysis.gate_stats import load_results, task_means

    asyncio.run(build("dev", "F7", ["10"], n_worlds=2, n_tasks=3, relational=True, embed=True))
    files = []
    for arm in ("S1", "S6"):
        log = inspect_eval(
            gate(family="F7", level="10", split="dev", arm=arm),
            model=get_model("mockllm/model", custom_outputs=mock_agent),
            log_dir=str(offline_env / "logs" / arm),
            display="none",
            epochs=2,
        )[0]
        files.append(log.location)
    df = load_results(files, require_cost=False)
    with pytest.raises(ValueError, match="no cost"):
        load_results(files)
    assert set(df["arm"]) == {"S1", "S6"} and len(df) == 2 * 6 * 2
    assert set(df["cell"]) == {"F7-10"} and df["world"].nunique() == 2
    tm = task_means(df)
    assert list(tm.columns) == ["S1", "S6"] and len(tm) == 6


def _gold_agent(worlds_dir):
    """A mock agent that knows every task's gold and makes exactly the gold tool calls, through the real
    tool path (`execute_tools`): F3 the procedure's mutating calls then `finish`, F7 `submit_decision`, F5
    `submit_answer`. Plumbing that drops, mangles or rejects a correct call fails the 100% check."""
    import json
    from pathlib import Path

    from inspect_ai.model import ChatMessageTool, ModelOutput

    from ape.worlds.spec import World

    gold = {t.prompt: t for p in Path(worlds_dir).rglob("*.json") for t in World.load(p).tasks}

    def agent(messages, tools, tool_choice, config) -> ModelOutput:
        task = gold[next(m.text for m in messages if m.role == "user" and m.text in gold)]
        done = [m for m in messages if isinstance(m, ChatMessageTool)]
        if task.family == "F3":
            calls = task.gold["calls"]
            if len(done) < len(calls):
                c = calls[len(done)]
                return ModelOutput.for_tool_call("mockllm/model", c["tool"], c["args"])
            return ModelOutput.for_tool_call("mockllm/model", "finish", {})
        if task.family == "F7":
            return ModelOutput.for_tool_call("mockllm/model", "submit_decision", dict(task.gold))
        if task.family == "F1":
            return ModelOutput.for_tool_call("mockllm/model", "submit_ratings", {"ratings": task.gold["ratings"]})
        if task.family == "F2":
            return ModelOutput.for_tool_call("mockllm/model", "submit_chain", {"final_supplier": task.gold["final"], "chain": task.gold["chain"]})
        return ModelOutput.for_tool_call("mockllm/model", "submit_answer", {"answer": task.gold["answer"]})

    return agent


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5"), ("F3", "60"), ("F5", "2hop")])
def test_gold_tool_calls_score_100_percent_through_the_real_tool_path(offline_env, family, level):
    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=4, relational=True, embed=True))
    logs = inspect_eval(
        gate(family=family, level=level, split="dev", arm="S6"),
        model=get_model("mockllm/model", custom_outputs=_gold_agent(offline_env / "worlds")),
        log_dir=str(offline_env / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert all(s.error is None for s in log.samples), [s.error for s in log.samples]
    assert [s.scores["task_success"].value for s in log.samples] == ["C"] * 4


@pytest.mark.parametrize("family,level", [("F1", "2"), ("F1", "32"), ("F2", "2"), ("F2", "10")])
def test_main_study_gold_answers_score_100_percent_through_the_real_tool_path(offline_env, family, level):
    from ape.tasks.main import main_study

    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=4, relational=True, embed=True))
    logs = inspect_eval(
        main_study(family=family, level=level, split="dev", arm="S6"),
        model=get_model("mockllm/model", custom_outputs=_gold_agent(offline_env / "worlds")),
        log_dir=str(offline_env / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert all(s.error is None for s in log.samples), [s.error for s in log.samples]
    assert [s.scores["task_success"].value for s in log.samples] == ["C"] * 4
