"""Study G loaders and report on real Inspect logs, offline: F8 sessions run by the session mocks (CM0, overflowing)
and a gold-knowing agent (O-state), the capability anchor's S1 runs on F7-10 and F3-5, then `report_from_cells`.
Also the loader's tolerance of the per-call, management and restored-usage records package B7 is adding."""

import asyncio
import json
import re
from types import SimpleNamespace

import numpy as np
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
    assert set(it["tier"]) == {"luna", "sol"} and set(it["effort"]) == {"high"} and set(ss["profile"]) == {"study_g_luna", "study_g_sol"}
    assert (ss["knobs"].map(lambda k: k.get("output_tokens")) == 1000).all() and (ss["cps_tokens"].dropna() > 0).all()
    assert ss.loc[ss["items_solved"] == 0, "cps_cost_usd"].isna().all()
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
    assert dec["G-H1"]["decision"] == "descriptive" and dec["G-H1"]["estimate_span"] is None and dec["G-H3-pre"]["decision"] == "NOT_TESTABLE"
    assert {c["arm"] for c in d["costs"]} == {"CM0", "O-state"} and all(c["cost_usd_per_solved"] is None or c["cost_usd_per_solved"] > 0 for c in d["costs"])
    assert {p["k"] for p in d["probes"]["by_checkpoint"]} == {5, 10}
    assert {t["arm"] for t in d["taxonomy"]} == {"CM0", "O-state"}
    assert any(r.get("status") == "ok" for r in d["gh2"]["degradation"]) or all("status" in r for r in d["gh2"]["degradation"])
    md = render(d)
    assert "## Decisions" in md and "G-H2a" in md and "Probes" in md


def _usage(i, o, cost=None):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o, **({"total_cost": cost} if cost is not None else {})}


LUNA = "openai/gpt-6-luna"
LUNA_PRICE = {"gpt-6-luna": {"input": 0.1, "output": 0.5}}
ITEM = {"position": 1, "case_id": "C-000001", "kind": "policy", "dependency": False, "dependency_kinds": [], "generations": 2, "view_tokens_first": 100, "view_tokens_decision": 150, "answered": True, "success": True}


def _sample(store, **kw):
    base = {"id": "F8-3-dev-s1", "epoch": 1, "metadata": {"world_id": "F8-3-dev-s1", "N": 3}, "store": store, "scores": None, "model_usage": {}, "role_usage": {}, "error": None, "total_time": 12.0, "working_time": 10.0, "limit": None}
    return SimpleNamespace(**(base | kw))


def test_loader_reads_b7_usage_by_kind_and_resumes():
    """B7's records: f8_usage (every call of the session, restored ones included) is authoritative over Inspect's
    sample usage (the last attempt only); calls by kind from f8_views; an item's tokens from its record; a resume's
    unlogged usage is reported apart from the arm's cost; probes stay out of the meters."""
    store = {
        "f8_items": [ITEM | {"cm_calls": 1, "usage": {"agent": _usage(250, 20), "cm": _usage(300, 50), "probe": _usage(200, 20)}}],
        "f8_views": [
            {"item": 1, "view_tokens": 100, "kind": "agent", "model": LUNA, "usage": _usage(100, 10)},
            {"item": 1, "view_tokens": 150, "kind": "agent", "model": LUNA, "usage": _usage(150, 10)},
            {"item": 1, "view_tokens": 900, "kind": "cm", "model": LUNA, "usage": _usage(300, 50), "purpose": "summary"},
            {"item": 2, "view_tokens": 300, "kind": "agent", "model": LUNA, "usage": _usage(1000, 100)},
        ],
        "f8_probes": [{"k": 1, "answer": {}, "error": None, "view_tokens": 120, "scores": {"mean_f1": 0.5}, "usage": _usage(200, 20), "item": 1, "kind": "probe", "model": LUNA}],
        "f8_usage": {"by_kind": {"agent": _usage(1250, 120), "cm": _usage(300, 50), "probe": _usage(200, 20)}, "by_model": {LUNA: _usage(1750, 190)}},
        "f8_resume": {"resumes": [{"resume": 1, "after_item": 1}], "count": 1, "after_items": [1], "failures": [], "unlogged": {"calls": 3, "by_kind": {"agent": _usage(500, 50)}, "by_model": {LUNA: _usage(500, 50)}}, "logged_elsewhere": None},
        "f8_overflow_at": None,
        "arm": {"checkpoints": [1, 3]},
        # B2's per-agent accounting (B9 topology sessions will carry it; worker calls are kind "agent"): ignored here.
        "mas_accounting": {"agents": {"orchestrator": {"calls": 2}, "worker-1": {"calls": 3}}, "error": None},
    }
    items, row = sample_rows(_sample(store, model_usage={LUNA: _usage(1000, 100)}), {"arm": "CM-sum"}, prices=LUNA_PRICE)
    assert row["tokens_total"] == 1940 and row["tokens_cm"] == 350 and row["tokens_probe"] == 220 and row["tokens"] == 1720
    assert row["calls_agent"] == 3 and row["calls_cm"] == 1 and row["calls"] == 4 and row["calls_probe"] == 1
    assert row["usage_sources"]["usage"] == "f8_usage" and row["resumes"] == 1 and row["tokens_unlogged"] == 550
    assert row["cost_usd_unlogged"] == pytest.approx((500 * 0.1 + 50 * 0.5) / 1e6)
    assert row["cost_usd"] == pytest.approx(((1250 + 300) * 0.1 + (120 + 50) * 0.5) / 1e6) and row["cost_usd_probe"] == pytest.approx((200 * 0.1 + 20 * 0.5) / 1e6)
    assert items[0]["tokens_agent"] == 270 and items[0]["tokens_cm"] == 350 and items[0]["tokens_probe"] == 220
    assert items[1]["tokens_agent"] == 1100 and items[1]["tokens_cm"] == 0 and items[1]["tokens_probe"] == 0  # from the call records
    assert not any(np.isnan(items[2][f"tokens_{k}"]) for k in ("agent", "cm", "probe"))  # recorded kinds: 0, not NaN
    # The score's compact copies serve when the store lacks them.
    sc = SimpleNamespace(value={"session_success": 0.0}, metadata={"usage": store["f8_usage"]["by_kind"], "resume": {"count": 2, "unlogged": {LUNA: _usage(10, 1)}}})
    lean = {k: v for k, v in store.items() if k not in ("f8_usage", "f8_resume")}
    _, row = sample_rows(_sample(lean, scores={"f8_session_score": sc}), {"arm": "CM-sum"}, prices=LUNA_PRICE)
    assert row["tokens_total"] == 1940 and row["usage_sources"]["usage"] == "score" and row["resumes"] == 2 and row["tokens_unlogged"] == 11


def test_loader_falls_back_without_b7_usage_and_tolerates_missing_scores():
    """Earlier sessions: Inspect's usage, cm from role usage, probes from their own records; an errored sample without a
    score: its unreached items fail; nothing recorded at all: every item fails, nothing raises."""
    store = {
        "f8_items": [ITEM],
        "f8_views": [{"item": 1, "view_tokens": 100}, {"item": 1, "view_tokens": 150}],
        "f8_probes": [{"k": 1, "answer": {}, "error": None, "scores": {"mean_f1": 0.5}, "usage": {"input_tokens": 200, "output_tokens": 20}}],
        "f8_overflow_at": None,
        "arm": {"checkpoints": [1, 3]},
    }
    sample = _sample(store, model_usage={LUNA: _usage(970, 110)}, role_usage={"cm": _usage(300, 50)}, error="RuntimeError: boom")
    items, row = sample_rows(sample, {"plan_cell": "g.cm.luna-high", "block": "cm", "point": "luna-high", "arm": "CM-sum"}, prices=LUNA_PRICE)
    assert [i["success"] for i in items] == [1.0, 0.0, 0.0] and [i["reached"] for i in items] == [True, False, False]
    assert items[0]["tokens_probe"] == 220 and np.isnan(items[0]["tokens_agent"])  # agent calls carry no usage here
    assert row["tokens_total"] == 1080 and row["tokens_cm"] == 350 and row["tokens_probe"] == 220 and row["tokens"] == 1080 - 220
    assert row["calls_agent"] == 2 and row["calls_cm"] == 0 and row["calls_probe"] == 1 and row["resumes"] == 0
    assert row["usage_sources"] == {"usage": "model_usage", "probe": "probes", "cm": "role_usage"}
    assert row["error"] and not row["scored"] and row["probes"] == {1: {"f1": 0.5, "taken": True}, 3: {"f1": 0.0, "taken": False}}
    assert row["cost_usd_total"] > row["cost_usd"] > 0
    bare = SimpleNamespace(id="w", epoch=2, metadata={"world_id": "w", "N": 4}, store={}, scores={}, model_usage={}, role_usage={}, error=None, total_time=None, working_time=None, limit=None)
    items, row = sample_rows(bare, {"arm": "CM0"})
    assert len(items) == 4 and not any(i["success"] for i in items) and row["tokens"] == 0 and row["probes"] == {}


def test_cell_info_resolves_the_d033_luna_low_topology_cell():
    """D-033's g.topo.luna-low (effort low on the Luna profile) maps to block topo and point luna-low, from the plan the
    caller passes (the repo's run_plan.yaml carries it once the config change lands)."""
    from dataclasses import replace

    from ape.analysis.g_load import cell_info
    from ape.budget import PlanCell, load_plan

    plan = load_plan()
    luna = plan.cell("g.topo.luna").spec
    new = PlanCell("g.topo.luna-low", "study_g", "topology", {**luna, "id": "g.topo.luna-low", "effort": "low", "sessions": 16})
    plan = replace(plan, cells=(*(c for c in plan.cells if c.id != "g.topo.luna-low"), new))
    ci = cell_info("g.topo.luna-low", plan)
    assert (ci["block"], ci["point"], ci["effort"], ci["sessions"], ci["N"]) == ("topo", "luna-low", "low", 16, 20)
    assert cell_info("g.topo.luna", plan)["point"] == "luna-high"


def test_report_from_tables_matches_report_from_cells(g_logs):
    data = load_g_cells(_cells(g_logs), prices=PRICES, capability={"luna-high": 0.78, "sol-high": 0.86})
    a = g_report(data.items, data.sessions, data.capability, reps=300, boot=0, glmm=False)
    b = report_from_cells(_cells(g_logs), prices=PRICES, capability={"luna-high": 0.78, "sol-high": 0.86}, reps=300, boot=0, glmm=False)
    assert a["decisions"] == b["decisions"] and a["gh2"]["gap"] == b["gh2"]["gap"]
