"""The main-study loader (analysis/main_load.py) on real Inspect logs from offline mock runs, and its answer keys."""

import asyncio
import itertools
import json
from pathlib import Path

import pandas as pd
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from ape.analysis.gate_stats import load_results
from ape.analysis.main_load import COLUMNS, answer_key, cap_hit_of, load_main, load_plan_cells, tier_of
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
    assert answer_key("F3", {"finished": True}, []) is not None, "a finished F3 run with no calls is an answer"
    assert answer_key("F3", None, []) is None, "an F3 run that neither finished nor called abstains (as in the live S8k3 vote)"
    assert answer_key("F1", None, None) is None and answer_key("F7", "not a dict", None) is None
    assert answer_key("F5", {"answer": "the Kraków office"}, None) == answer_key("F5", {"answer": "Krakow, Poland"}, None)
    assert answer_key("F1", {"ratings": {}}, None) != answer_key("F2", {"final": ""}, None), "keys are per family"


def test_the_main_loaders_answer_key_agrees_with_the_live_s8k3_vote():
    """Post-hoc S8 and the live S8k3 (agent/multi) must group answers alike: equal keys there, equal keys here."""
    from ape.agent.multi.primitives import answer_key as live_key
    from ape.worlds.spec import TaskItem

    cases = {
        "F1": [{"ratings": {"SUP-1": "approved"}}, {"ratings": {"sup-1": "Approved "}}, {"ratings": {"SUP-1": "preferred"}}, None],
        "F2": [{"final": "SUP-2", "chain": ["a"]}, {"final": "sup-2 ", "chain": []}, {"final": "SUP-3"}, None],
        "F7": [{"action": "deny", "deadline_days": "3"}, {"action": "deny", "deadline_days": 3}, {"action": "approve", "deadline_days": 3}, None],
    }
    for fam, answers in cases.items():
        task = TaskItem(id="t", world_id="w", family=fam, level="2", prompt="", gold={}, gold_fact_ids=[], answer_tool="x")
        for a, b in itertools.combinations(answers, 2):
            assert (answer_key(fam, a, []) == answer_key(fam, b, [])) == (live_key(task, a, []) == live_key(task, b, [])), (fam, a, b)
    f3 = TaskItem(id="t", world_id="w", family="F3", level="5", prompt="", gold={}, gold_fact_ids=[], answer_tool="finish")
    x, y = {"tool": "x", "args": {"o": 1}}, {"tool": "y", "args": {}}
    for a, b in itertools.combinations([(None, [x, y]), (None, [y, x]), ({}, []), (None, []), ({}, [x])], 2):
        assert (answer_key("F3", *a) == answer_key("F3", *b)) == (live_key(f3, *a) == live_key(f3, *b)), (a, b)
        assert (answer_key("F3", *a) is None) == (live_key(f3, *a) is None)


def test_cap_hits_read_multi_agent_stop_reasons():
    capped = [{"id": "orch", "stop": "text"}, {"id": "w1", "stop": "turn_cap"}]
    clean = [{"id": "orch", "stop": "done"}, {"id": "w1", "stop": "done"}]
    assert cap_hit_of(False, None, False, 3, 12, capped), "a worker at its cap, sample unanswered: cut short"
    assert not cap_hit_of(False, None, True, 3, 12, capped), "answered anyway: not a cap hit (agent_cap_hits keeps it)"
    assert not cap_hit_of(False, None, False, 30, 12, clean), "the top agent's turns_used does not decide for a team"
    assert cap_hit_of(False, None, False, 12, 12, None) and not cap_hit_of(False, None, False, 11, 12, None)
    assert cap_hit_of(False, "token", True, 1, 12, clean) and not cap_hit_of(True, "token", False, 12, 12, capped)


def test_load_main_reads_a_multi_agent_log(offline_env):
    """B2's M1 on F1-2 with its gold mock: the team's records reach the frame, every agent's calls count."""
    from ape.llm.mock_multi import GoldMulti

    asyncio.run(build("dev", "F1", ["2"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    gold = GoldMulti(offline_env / "worlds")
    log = inspect_eval(main_study(family="F1", level="2", split="dev", arm="M1"), model=get_model(MODEL, custom_outputs=gold, memoize=False), model_roles={"kg": get_model(MODEL, custom_outputs=gold.kg, memoize=False)}, log_dir=str(offline_env / "logs"), display="none")[0]
    assert log.status == "success", log.error
    df = load_main([log.location], require_cost=False)
    assert df["multi_agent"].all() and (df["success"] == 1.0).all() and df["answer_key"].notna().all() and not df["cap_hit"].any()
    for (_, row), s in zip(df.iterrows(), log.samples, strict=True):
        acc = s.store["mas_accounting"]
        assert row["per_agent"] == acc and row["calls"] == acc["totals"]["calls"] and row["tokens"] == acc["totals"]["total_tokens"]
        assert row["agents"] == len(s.store["mas_agents"]) >= 2 and row["agent_stops"] and row["agent_cap_hits"] == 0
        assert row["switches"] == s.store["mas_switches"] and row["realized_parallelism"] == acc["realized_parallelism"]


def test_a_capped_sample_is_a_failure_and_casts_no_vote(monkeypatch):
    """BUILD_REVIEW S-1, D-047: Inspect scores a sample after a limit with its last state, so a capped F3 sample whose end
    state equals the gold scored a success and voted in the S8 frontier. A cap hit is a failure with no answer key."""
    from types import SimpleNamespace as NS

    import inspect_ai.log

    from ape.worlds.env_tools import ANSWER, CALLS

    gold = [{"tool": "refund_order", "args": {"order_id": "O-1"}}]

    def sample(sid, limit, turns, answered=False):
        score = NS(value="C", answer=json.dumps({"calls": gold}), metadata={"turns_used": turns, "answered": answered})
        usage = {"openai/gpt-6-luna": NS(input_tokens=100, input_tokens_cache_read=0, input_tokens_cache_write=0, output_tokens=10, reasoning_tokens=0, total_cost=0.01, total_tokens=110)}
        return NS(metadata={"family": "F3", "level": "60", "world_id": "F3-60-test-s13000"}, model_usage=usage, scores={"task_success": score},
                  store={ANSWER: {"finished": True} if answered else None, CALLS: gold, "turns_used": turns}, error=None, id=sid, epoch=1,
                  limit=NS(type=limit) if limit else None, events=[], working_time=1.0, total_time=1.0, uuid=f"u-{sid}")  # fmt: skip

    log = NS(eval=NS(task_args={"arm": "S1"}, metadata={"max_turns": 12, "family": "F3"}, model="openai/gpt-6-luna", model_generate_config=NS(reasoning_effort="high")),
             samples=[sample("token", "token", 7), sample("turns", None, 12), sample("ok", None, 5, answered=True), sample("late", "token", 6, answered=True)])  # fmt: skip
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda f: log)
    df = load_main(["fake.eval"], plan_cell="main.B.s1-pool", require_cost=False).set_index("task")
    for t in ("token", "turns", "late"):
        assert df.loc[t, "cap_hit"] and df.loc[t, "success"] == 0.0 and pd.isna(df.loc[t, "answer_key"]) and df.loc[t, "scored_success"] == 1.0, t
    assert not df.loc["ok", "cap_hit"] and df.loc["ok", "success"] == 1.0 and not pd.isna(df.loc["ok", "answer_key"])
    assert list(df["uuid"]) == ["u-token", "u-turns", "u-ok", "u-late"]


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
    assert df["uuid"].notna().all() and df["uuid"].nunique() == len(df), "one uuid per sample-epoch run"
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
