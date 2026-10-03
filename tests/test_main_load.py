"""The main-study loader (analysis/main_load.py) on real Inspect logs from offline mock runs, and its answer keys."""

import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from ape.analysis.gate_stats import load_results
from ape.analysis.main_load import COLUMNS, answer_key, load_main, load_plan_cells, tier_of
from ape.build import build
from ape.tasks.main import main_study
from ape.worlds.spec import World

MODEL = "mockllm/model"


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def test_answer_keys_follow_the_scorers_normalisation():
    assert answer_key("F1", {"ratings": {"SUP-1": "Approved ", " sup-2": "probation"}}, None) == answer_key("F1", {"ratings": {"sup-1": "approved", "SUP-2": "Probation"}}, None)
    assert answer_key("F1", {"ratings": {"SUP-1": "approved"}}, None) != answer_key("F1", {"ratings": {"SUP-1": "preferred"}}, None)
    assert answer_key("F2", {"final": "sup-7", "chain": ["SUP-3"]}, None) == answer_key("F2", {"final": "SUP-7 ", "chain": []}, None), "F2 scores the final supplier only"
    assert answer_key("F7", {"action": "a", "deadline_days": "5"}, None) == answer_key("F7", {"action": "a", "deadline_days": 5}, None)
    calls = [{"tool": "x", "args": {"a": 1}}, {"tool": "y", "args": {}}]
    assert answer_key("F3", None, calls) == answer_key("F3", None, calls[::-1]), "F3's end state ignores call order"
    assert answer_key("F3", None, []) is not None, "an empty F3 end state is still an answer"
    assert answer_key("F1", None, None) is None and answer_key("F7", "not a dict", None) is None
    assert answer_key("F5", {"answer": "the Kraków office"}, None) == answer_key("F5", {"answer": "Krakow, Poland"}, None)
    assert answer_key("F1", {"ratings": {}}, None) != answer_key("F2", {"final": ""}, None), "keys are per family"


def test_tier_from_the_eval_model():
    assert tier_of("openai/gpt-6-luna") == "luna" and tier_of("openai/gpt-6-sol") == "sol" and tier_of("mockllm/model") == "model"


def _gold(worlds_dir: Path):
    gold = {t.prompt: t for p in worlds_dir.rglob("*.json") for t in World.load(p).tasks}

    def agent(messages, tools, tool_choice, config) -> ModelOutput:
        task = gold[next(m.text for m in messages if m.role == "user" and m.text in gold)]
        return ModelOutput.for_tool_call(MODEL, "submit_ratings", {"ratings": task.gold["ratings"]})

    return agent, gold


def _stalling(messages, tools, tool_choice, config) -> ModelOutput:
    """11 lookups, then text without a tool call twice: the loop stops after 13 turns without an answer."""
    if sum(isinstance(m, ChatMessageTool) for m in messages) < 11:
        return ModelOutput.for_tool_call(MODEL, "lookup_supplier", {"supplier_id": "SUP-0"})
    return ModelOutput.from_content(MODEL, "still thinking")


def test_load_main_reads_a_main_study_log(offline_env):
    asyncio.run(build("dev", "F1", ["2"], n_worlds=1, n_tasks=2, relational=True, embed=False))
    agent, gold = _gold(offline_env / "worlds")
    log = inspect_eval(main_study(family="F1", level="2", split="dev", arm="S6", plan_cell="main.A.arms"), model=get_model(MODEL, custom_outputs=agent), epochs=2, log_dir=str(offline_env / "logs"), display="none")[0]
    assert log.status == "success", log.error
    df = load_main([log.location], require_cost=False)
    assert list(df.columns) == list(COLUMNS) and len(df) == 4
    assert (df["plan_cell"] == "main.A.arms").all() and (df["arm"] == "S6").all() and (df["cell"] == "F1-2").all()
    assert (df["success"] == 1.0).all() and not df["cap_hit"].any() and not df["error"].any() and not df["multi_agent"].any()
    assert (df["max_turns"] == 14).all(), "the eval's own turn cap (gen_registry.turn_cap), not the 12-turn knob default"
    assert (df["tier"] == "model").all() and (df["calls"] >= 1).all()
    by_task = df.groupby("task")["answer_key"].nunique()
    assert (by_task == 1).all(), "the same (normalised) answer gives one key across epochs"
    for t in df["task"].unique():
        g = next(x for x in gold.values() if x.id == t)
        assert df.loc[df["task"] == t, "answer_key"].iloc[0] == answer_key("F1", {"ratings": g.gold["ratings"]}, None)
    gate = load_results([log.location], require_cost=False)
    assert gate[["world", "task", "epoch", "success"]].values.tolist() == df[["world", "task", "epoch", "success"]].values.tolist()
    frame, coverage = load_plan_cells({"main.A.arms": [log.location], "main.A.m1s": [str(offline_env / "missing.eval")], "main.A.s1-pool": []}, require_cost=False)
    assert len(frame) == 4 and [c["reason"] is None for c in coverage] == [True, False, False]


def test_a_sample_that_stops_unanswered_under_the_eval_turn_cap_is_not_a_cap_hit(offline_env):
    """13 turns without an answer at F1-2 (cap 14): the gate's loader, reading the 12-turn knob default, calls it a cap
    hit; the main loader reads the eval's max_turns and does not."""
    asyncio.run(build("dev", "F1", ["2"], n_worlds=1, n_tasks=1, relational=True, embed=False))
    log = inspect_eval(main_study(family="F1", level="2", split="dev", arm="S6"), model=get_model(MODEL, custom_outputs=_stalling), log_dir=str(offline_env / "logs"), display="none")[0]
    df = load_main([log.location], require_cost=False)
    row = df.iloc[0]
    assert row["turns_used"] == 13 and not row["answered"] and row["answer_key"] is None and row["success"] == 0.0
    assert not row["cap_hit"] and row["calls"] == 13
    assert bool(load_results([log.location], require_cost=False)["cap_hit"].iloc[0])
    json.dumps(df.drop(columns=["per_agent"]).to_dict(orient="records"), default=str)
