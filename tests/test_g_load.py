"""Study G loaders and report on real Inspect logs, offline: F8 sessions run by the session mocks (CM0, overflowing)
and a gold-knowing agent (O-state), the capability anchor's S1 runs on F7-10 and F3-5, then `report_from_cells`.
Also the loader's tolerance of the per-call, management and restored-usage records package B7 is adding."""

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from ape.analysis.g_load import load_g_cells, sample_rows
from ape.analysis.g_report import g_report, render, report_from_cells
from ape.build import build
from ape.llm.mock_agent import mock_agent
from ape.llm.mock_session import _current, mock_session_agent
from ape.tasks.gate import gate
from ape.tasks.study_g import f8_session

CASE = re.compile(r"C-\d{6}")
PRICES = {"mockllm/model": {"input": 1.0, "output": 2.0, "input_cache_read": 0.5}}


def gold_agent(worlds):
    """Answers every case and the report with its gold, in any of `worlds` (found by case ID); probes as the mock."""
    by_case = {t.tags["case_id"]: (w, t) for w in worlds for t in w.tasks}

    def agent(messages, tools, tool_choice, config):
        if config.response_schema is not None:
            return mock_session_agent(messages, tools, tool_choice, config)
        i, text = _current(messages)
        if "End of shift" in text:
            ids = [c for m in messages for c in CASE.findall(m.text or "") if c in by_case]
            world = by_case[ids[0]][0]
            return ModelOutput.for_tool_call("mockllm/model", "submit_shift_report", {"report": json.dumps(world.entities["session"]["report_gold"])})
        cid = re.search(r"(?:Case|Ticket) (C-\d{6})", text).group(1)
        _, t = by_case[cid]
        if t.tags["kind"] == "ticket":
            done = sum(isinstance(m, ChatMessageTool) for m in messages[i + 1 :])
            calls = t.gold["calls"]
            if done < len(calls):
                return ModelOutput.for_tool_call("mockllm/model", calls[done]["tool"], calls[done]["args"])
            return ModelOutput.for_tool_call("mockllm/model", "finish", {"case_id": cid})
        return ModelOutput.for_tool_call("mockllm/model", "submit_decision", {"case_id": cid, **t.gold})

    return agent


@pytest.fixture(scope="module")
def g_logs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("g")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    asyncio.run(build("dev", "F8", ["12"], n_worlds=2, n_tasks=0, relational=True, embed=False))
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=3, relational=True, embed=False))
    asyncio.run(build("dev", "F3", ["5"], n_worlds=1, n_tasks=3, relational=True, embed=False))
    from ape.agent.arms import load_world

    worlds = [load_world(f"F8-12-dev-s{s}") for s in (1000, 1001)]
    logs = {}

    def run(task, model, name):
        log = inspect_eval(task, model=model, log_dir=str(tmp / "logs" / name), display="none", epochs=2 if "f8" in name else 1)[0]
        assert log.status == "success", log.error
        logs[name] = log.location

    run(f8_session(level="12", split="dev", arm="CM0", checkpoints="5,10", window=8000), get_model("mockllm/model", custom_outputs=mock_session_agent), "f8-cm0")
    run(f8_session(level="12", split="dev", arm="O-state", checkpoints="5,10"), get_model("mockllm/model", custom_outputs=gold_agent(worlds)), "f8-ostate")
    run(gate(family="F7", level="10", split="dev", arm="S1"), get_model("mockllm/model", custom_outputs=mock_agent), "cap-f7")
    run(gate(family="F3", level="5", split="dev", arm="S1"), get_model("mockllm/model", custom_outputs=mock_agent), "cap-f3")
    yield logs
    mp.undo()


def _cells(logs):
    session = [logs["f8-cm0"], logs["f8-ostate"]]
    cap = [logs["cap-f7"], logs["cap-f3"]]
    return {"g.cm.luna-high": session, "g.cm.sol-high": session, "g.cap.luna-high": cap, "g.cap.sol-high": cap}


def test_loader_reads_session_and_anchor_logs(g_logs):
    from inspect_ai.log import read_eval_log

    data = load_g_cells(_cells(g_logs), prices=PRICES)
    it, ss = data.items, data.sessions
    # 2 cells x 2 arms x 2 worlds x 2 epochs session-epochs, every position 1..12 present.
    assert len(ss) == 16 and len(it) == 16 * 12
    assert set(ss["point"]) == {"luna-high", "sol-high"} and set(ss["block"]) == {"cm"} and set(ss["arm"]) == {"CM0", "O-state"}
    assert (it.groupby(["plan_cell", "arm", "session", "epoch"])["position"].apply(lambda p: sorted(p) == list(range(1, 13)))).all()
    # Outcomes agree with the scorer, item by item and per session.
    log = read_eval_log(g_logs["f8-cm0"])
    for s in log.samples:
        row = ss[(ss["plan_cell"] == "g.cm.luna-high") & (ss["arm"] == "CM0") & (ss["session"] == s.metadata["world_id"]) & (ss["epoch"] == s.epoch)].iloc[0]
        assert row["item_success"] == pytest.approx(s.scores["f8_session_score"].value["item_success"])
        assert row["overflow"] and row["overflow_at"] == s.store["f8_overflow_at"]
        assert row["tax_overflow"] == 12 - s.store["f8_overflow_at"] + 1
        assert sorted(row["probes"]) == [5, 10]
        assert row["tokens"] > 0 and row["cost_usd"] > 0 and row["calls"] == len(s.store["f8_views"])
    ostate = ss[ss["arm"] == "O-state"]
    assert (ostate["item_success"] == 1.0).all() and (ostate["report_exact"] == 1.0).all() and (ostate["session_success"] == 1.0).all()
    assert (ostate["probe_coverage"] == 1.0).all()
    lost = it[(it["arm"] == "CM0") & it["overflow"]]
    assert len(lost) and (lost["success"] == 0).all() and not lost["reached"].any()
    reached = it[it["reached"]]
    assert (reached["view_tokens"] > 0).all()
    # Capability from the anchor logs: both cells, equal weights, a world-clustered SE where there are worlds to cluster.
    cap = data.capability.set_index("point")
    assert set(cap.index) == {"luna-high", "sol-high"}
    assert set(cap.loc["luna-high", "by_cell"]) == {"F7-10", "F3-5"} and 0.0 <= cap.loc["luna-high", "capability"] <= 1.0
    assert data.cells["g.cm.luna-high"]["point"] == "luna-high" and data.cells["g.cap.sol-high"]["block"] == "cap"
    assert any("planned sessions" in p for p in data.problems)


def test_report_end_to_end_from_cells(g_logs):
    d = report_from_cells(_cells(g_logs), prices=PRICES, capability={"luna-high": 0.78, "sol-high": 0.86}, reps=500, boot=200)
    json.dumps(d)
    dec = {r["id"]: r for r in d["decisions"]}
    # O-state solves everything and CM0 overflows: Gap_T = 1 − CM0's item success, shown at both points.
    assert dec["G-H2a"]["decision"] == "SUPPORTED"
    ss = load_g_cells(_cells(g_logs), prices=PRICES).sessions
    cm0 = ss[(ss["arm"] == "CM0") & (ss["point"] == "luna-high")].groupby("session")["item_success"].mean().mean()
    assert d["gh2"]["gap"]["points"]["luna-high"]["est"] == pytest.approx(1.0 - cm0)
    assert d["capability"]["points"][0]["source"] == "override"
    assert dec["G-H1"]["decision"] == "NOT_TESTABLE" and dec["G-H3-pre"]["decision"] == "NOT_TESTABLE"
    assert {c["arm"] for c in d["costs"]} == {"CM0", "O-state"} and all(c["cost_usd_per_solved"] is None or c["cost_usd_per_solved"] > 0 for c in d["costs"])
    assert {p["k"] for p in d["probes"]["by_checkpoint"]} == {5, 10}
    assert {t["arm"] for t in d["taxonomy"]} == {"CM0", "O-state"}
    assert any(r.get("status") == "ok" for r in d["gh2"]["degradation"]) or all("status" in r for r in d["gh2"]["degradation"])
    md = render(d)
    assert "## Decisions" in md and "G-H2a" in md and "Probes" in md


def _usage(i, o, cost=None):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o, **({"total_cost": cost} if cost is not None else {})}


def test_loader_tolerates_b7_records_and_missing_scores():
    """Per-call records with kind and usage, role usage for cm and probe, usage restored after a resume, and an errored
    sample without a score: tokens split by kind, probes excluded from the meters, unreached items fail."""
    store = {
        "f8_items": [{"position": 1, "case_id": "C-000001", "kind": "policy", "dependency": False, "dependency_kinds": [], "generations": 2, "view_tokens_first": 100, "view_tokens_decision": 150, "answered": True, "success": True}],
        "f8_calls": [
            {"item": 1, "kind": "agent", "view_tokens": 100, "usage": _usage(100, 10)},
            {"item": 1, "kind": "agent", "view_tokens": 150, "usage": _usage(150, 10)},
            {"item": 1, "kind": "cm", "usage": _usage(300, 50)},
            {"item": 1, "kind": "probe", "usage": _usage(200, 20)},
        ],
        "f8_probes": [{"k": 1, "answer": {}, "error": None, "scores": {"mean_f1": 0.5}, "usage": {"input_tokens": 200, "output_tokens": 20}}],
        "f8_overflow_at": None,
        "f8_restored_usage": {"agent": _usage(1000, 100, 0.002), "probe": _usage(50, 5, 0.0001)},
        "arm": {"checkpoints": [1, 3]},
    }
    sample = SimpleNamespace(
        id="F8-3-dev-s1", epoch=1, metadata={"world_id": "F8-3-dev-s1", "N": 3}, store=store, scores=None,
        model_usage={"openai/gpt-6-luna": _usage(970, 110)}, role_usage={"cm": _usage(300, 50)},
        error="RuntimeError: boom", total_time=12.0, working_time=10.0, limit=None,
    )  # fmt: skip
    items, row = sample_rows(sample, {"plan_cell": "g.cm.luna-high", "block": "cm", "point": "luna-high", "arm": "CM-sum"}, prices={"gpt-6-luna": {"input": 0.1, "output": 0.5}})
    assert [i["success"] for i in items] == [1.0, 0.0, 0.0] and [i["reached"] for i in items] == [True, False, False]
    assert items[0]["tokens_agent"] == 270 and items[0]["tokens_cm"] == 350 and items[0]["tokens_probe"] == 220  # the probe call, not again its record
    assert row["tokens_total"] == 970 + 110 + 1100 + 55
    assert row["tokens_cm"] == 350 and row["tokens_probe"] == 220 + 55
    assert row["tokens"] == row["tokens_total"] - row["tokens_probe"]
    assert row["calls_agent"] == 2 and row["calls_cm"] == 1 and row["calls"] == 3 and row["calls_probe"] == 1
    assert row["error"] and not row["scored"] and row["probes"] == {1: {"f1": 0.5, "taken": True}, 3: {"f1": 0.0, "taken": False}}
    assert row["cost_usd_total"] > row["cost_usd"] > 0 and row["usage_sources"]["restored"] == ["agent", "probe"]
    # Nothing at all recorded: every item fails, nothing raises.
    bare = SimpleNamespace(id="w", epoch=2, metadata={"world_id": "w", "N": 4}, store={}, scores={}, model_usage={}, role_usage={}, error=None, total_time=None, working_time=None, limit=None)
    items, row = sample_rows(bare, {"arm": "CM0"})
    assert len(items) == 4 and not any(i["success"] for i in items) and row["tokens"] == 0 and row["probes"] == {}


def test_report_from_tables_matches_report_from_cells(g_logs):
    data = load_g_cells(_cells(g_logs), prices=PRICES, capability={"luna-high": 0.78, "sol-high": 0.86})
    a = g_report(data.items, data.sessions, data.capability, reps=300, boot=0, glmm=False)
    b = report_from_cells(_cells(g_logs), prices=PRICES, capability={"luna-high": 0.78, "sol-high": 0.86}, reps=300, boot=0, glmm=False)
    assert a["decisions"] == b["decisions"] and a["gh2"]["gap"] == b["gh2"]["gap"]
