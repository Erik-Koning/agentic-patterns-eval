"""F8 "shift" sessions (Study G): generator, gold, state, scoring, taxonomy and the CM0 / O-state runner."""

import asyncio
import json
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from ape.build import build
from ape.llm.mock_session import mock_session_agent
from ape.scorers.session import external_state_score, probe_score, report_score, score_session, summary_loss, taxonomy
from ape.tasks.study_g import f8_session
from ape.worlds import gen_f8
from ape.worlds.gen_f8 import QUOTA, decisions_equal, solve_session, state_at

SEEDS = range(1000, 1006)


@pytest.fixture(scope="module")
def sessions():
    return {(n, s): gen_f8.generate(str(n), "dev", s) for n in (10, 24, 40) for s in SEEDS}


def _events(world, answers, skip=()) -> list[dict]:
    """Tool events of an agent that submits `answers` (one per item) in each item's window."""
    out = []
    for t, a in zip(world.tasks, answers, strict=True):
        pos, cid = t.tags["position"], t.tags["case_id"]
        if pos in skip:
            continue
        if t.tags["kind"] == "ticket":
            out += [{"item": pos, "tool": c["tool"], "args": c["args"]} for c in a["calls"]]
            out.append({"item": pos, "tool": "finish", "args": {"case_id": cid}})
        else:
            out.append({"item": pos, "tool": "submit_decision", "args": {"case_id": cid, **a}})
    return out


def test_sessions_are_byte_identical_per_seed():
    a, b = gen_f8.generate("24", "dev", 1001), gen_f8.generate("24", "dev", 1001)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    assert a.content_hash() != gen_f8.generate("24", "dev", 1002).content_hash()
    assert gen_f8.customer_file(a, a.tasks[0].tags.get("customer_id", "CU-0")) == gen_f8.customer_file(b, b.tasks[0].tags.get("customer_id", "CU-0"))
    # Knob variants are separate worlds with their own IDs.
    v = gen_f8.generate("24", "dev", 1001, memo_density=0.25)
    assert v.id == "F8-24-m0.25-dev-s1001" and a.id == "F8-24-dev-s1001"


def test_reference_solver_scores_100_percent(sessions):
    for w in sessions.values():
        answers, report = solve_session(w)
        res = score_session(w, _events(w, answers), report)
        assert res["item_success"] == 1.0 and res["report"]["exact"] and res["session_success"], w.id
        assert res["dependency_items"] > 0


def test_state_at_is_consistent_with_the_item_golds(sessions):
    for w in sessions.values():
        s, n = w.entities["session"], len(w.tasks)
        for k in range(n + 1):
            st = state_at(w, k)
            assert st["completed"] + st["pending"] == s["queue"]
            assert st["escalated"] == [t.tags["case_id"] for t in w.tasks[:k] if t.tags["kind"] != "ticket" and t.gold["action"] == "escalate"]
            for t in w.tasks[:k]:
                waiting = t.tags["awaiting"] and (t.tags["followed_up_at"] is None or t.tags["followed_up_at"] > k)
                assert (t.tags["case_id"] in st["open_followups"]) == waiting
            for m in s["memos"]:
                assert (m["id"] in st["memos_in_force"]) == (m["position"] <= k and (m["withdrawn_at"] is None or m["withdrawn_at"] > k))
        assert sorted(state_at(w, n)["open_followups"]) == s["report_gold"]["open_followups"]


def test_memos_change_the_gold_exactly_where_they_are_in_force(sessions):
    for w in sessions.values():
        for m in w.entities["session"]["memos"]:
            changed = []
            for t in w.tasks:
                pos, cf = t.tags["position"], t.tags["counterfactuals"]
                if m["id"] in cf["without_memo"]:
                    assert gen_f8.memo_active(m, pos), (w.id, m["id"], pos)
                    changed.append(pos)
                    if m["type"] == "escalate":
                        assert t.gold["action"] == "escalate" and t.gold["approver"] == "compliance_officer"
                    elif m["type"] == "deadline":
                        assert t.gold["deadline_days"] == m["days"]
                    elif m["type"] == "document":
                        assert t.gold["document"] == m["document"]
                    else:
                        assert any(c["tool"] == m["tool"] and c["args"][m["param"]] == m["new"] for c in t.gold["calls"])
                if m["id"] in cf["with_withdrawn"]:
                    assert m["withdrawn_at"] is not None and pos >= m["withdrawn_at"]
            assert any(p > m["position"] for p in changed), (w.id, m["id"])  # every memo matters later on
            # The memo is announced in exactly one case message, and its withdrawal in another.
            assert sum(m["text"] in t.prompt for t in w.tasks) == 1
            if m["withdrawn_at"]:
                assert f"{m['id']} is withdrawn" in w.tasks[m["withdrawn_at"] - 1].prompt


def test_quota_follow_ups_and_deferred_work(sessions):
    triggered = followups = 0
    for w in sessions.values():
        approvals: dict[str, int] = {}
        for t in w.tasks:
            tg = t.tags
            if tg["kind"] in ("policy", "repeat"):
                if tg["quota_triggered"]:
                    triggered += 1
                    assert approvals.get(tg["customer_id"], 0) >= QUOTA
                    assert t.gold["action"] == "escalate" and t.gold["approver"] == "team_lead"
                    assert tg["counterfactuals"]["without_quota"]["action"] == "approve"
                if t.gold["action"] == "approve":
                    approvals[tg["customer_id"]] = approvals.get(tg["customer_id"], 0) + 1
            if tg["kind"] == "followup":
                followups += 1
                orig = next(x for x in w.tasks if x.tags["case_id"] == tg["original"])
                assert orig.tags["awaiting"] and orig.tags["followed_up_at"] == tg["position"] < len(w.tasks) + 1
                # The follow-up names only the earlier case: the request itself must come from memory.
                assert orig.tags["customer_id"] not in t.prompt and f"${orig.tags['amount']:,}" not in t.prompt
                assert "followup" in tg["dependency_kinds"]
        rg = w.entities["session"]["report_gold"]
        assert rg["pending_recheck"] == sorted(c for c, d in rg["dispositions"].items() if d == "escalate")
        assert set(rg["dispositions"]) == set(w.entities["session"]["queue"])
    assert triggered > 0 and followups > 0


def test_report_scoring():
    w = gen_f8.generate("24", "dev", 1000)
    gold = w.entities["session"]["report_gold"]
    assert report_score(gold, gold)["exact"]
    missing = {**gold, "dispositions": dict(list(gold["dispositions"].items())[1:])}
    r = report_score(missing, gold)
    assert not r["exact"] and r["dispositions_accuracy"] == pytest.approx(23 / 24)
    wrong = {**gold, "pending_recheck": [*gold["pending_recheck"], w.entities["session"]["queue"][0]]}
    assert report_score(wrong, gold)["pending_recheck_f1"] < 1.0
    assert report_score(None, gold) == {"exact": False, "dispositions_accuracy": 0.0, "pending_recheck_f1": 0.0, "open_followups_f1": 0.0, "submitted": False}


def test_probe_and_external_state_scoring():
    w = gen_f8.generate("40", "dev", 1000)
    st = state_at(w, 15)
    assert probe_score(st, st)["mean_f1"] == 1.0
    half = {**st, "completed": st["completed"][:5] + ["C-000000"]}
    s = probe_score(half, st)
    assert s["completed"] == pytest.approx(2 * (5 / 6) * (5 / 15) / ((5 / 6) + (5 / 15))) and s["pending"] == 1.0
    assert probe_score(None, st)["mean_f1"] == 0.0
    todos = [{"content": f"Case {c}", "status": "completed"} for c in st["completed"]]
    todos += [{"content": f"Case {c}", "status": "pending"} for c in st["pending"]]
    todos += [{"content": f"Apply memo {m}", "status": "in_progress"} for m in st["memos_in_force"]]
    todos += [{"content": f"{c} awaits a follow-up", "status": "pending"} for c in st["open_followups"]]
    todos += [{"content": f"{c} escalated, recheck", "status": "completed"} for c in st["escalated"]]
    ext = external_state_score(st, todos=todos)
    assert ext["completed"] < 1.0 or not st["escalated"]  # escalation todos are 'completed' too: extra IDs cost precision
    assert ext["memos_in_force"] == 1.0 and ext["open_followups"] == 1.0
    notes = "Done: " + ", ".join(st["completed"]) + ". Memos: " + ", ".join(st["memos_in_force"])
    n = external_state_score(st, notes=notes)
    assert n["kind"] == "notes" and n["completed"] == 1.0 and n["memos_in_force"] == 1.0


def _find(sessions, pred):
    for w in sessions.values():
        for t in w.tasks:
            if pred(t):
                return w, t
    raise AssertionError("no session has such an item")


def test_failure_taxonomy_labels(sessions):
    # forgot constraint: an earlier memo in force is ignored
    w, t = _find(sessions, lambda t: t.tags["kind"] != "ticket" and "memo" in t.tags["dependency_kinds"])
    answers, report = solve_session(w)
    pos = t.tags["position"]
    mid = next(iter(t.tags["counterfactuals"]["without_memo"]))
    answers[pos - 1] = t.tags["counterfactuals"]["without_memo"][mid]
    assert "forgot_constraint" in taxonomy(w, _events(w, answers), report)["per_item"][pos]
    # stale state: a withdrawn memo still applied
    w, t = _find(sessions, lambda t: t.tags["counterfactuals"]["with_withdrawn"])
    answers, report = solve_session(w)
    pos = t.tags["position"]
    answers[pos - 1] = next(iter(t.tags["counterfactuals"]["with_withdrawn"].values()))
    assert "stale_state" in taxonomy(w, _events(w, answers), report)["per_item"][pos]
    # stale state: a follow-up repeats the original decision although policy changed since
    w, t = _find(sessions, lambda t: t.tags["kind"] == "followup" and not decisions_equal(t.tags["counterfactuals"]["original_gold"], t.gold))
    answers, report = solve_session(w)
    answers[t.tags["position"] - 1] = t.tags["counterfactuals"]["original_gold"]
    assert "stale_state" in taxonomy(w, _events(w, answers), report)["per_item"][t.tags["position"]]
    # resurrected, dropped and hallucinated, on one session
    w = sessions[(24, 1000)]
    answers, report = solve_session(w)
    events = _events(w, answers, skip=(5, 12, 13))
    deny = {"action": "deny", "approver": "none", "deadline_days": 1, "document": "none"}
    events.append({"item": 12, "tool": "submit_decision", "args": {"case_id": w.tasks[1].tags["case_id"], **deny}})
    events.append({"item": 13, "tool": "submit_decision", "args": {"case_id": "C-000000", **deny}, "rejected": True})
    # The same incidents in the window of an item that still succeeded are not failure labels (L10).
    events.append({"item": 9, "tool": "submit_decision", "args": {"case_id": w.tasks[1].tags["case_id"], **deny}})
    events.append({"item": 10, "tool": "submit_decision", "args": {"case_id": "C-000000", **deny}, "rejected": True})
    bad_report = {**report, "open_followups": [*report["open_followups"], "C-999999"]}
    tax = taxonomy(w, events, bad_report, probes=[{"answer": {"memos_in_force": ["M-99"]}}])
    assert tax["per_item"][5] == ["dropped_item"]
    assert "resurrected_done_item" in tax["per_item"][12] and "hallucinated_state" in tax["per_item"][13]
    assert 9 not in tax["per_item"] and 10 not in tax["per_item"]
    assert tax["incidents_on_success"] == {9: ["resurrected_done_item"], 10: ["hallucinated_state"]}
    assert tax["report_hallucinated"] == ["C-999999"] and tax["probe_hallucinated"] == ["M-99"]
    # overflow: every case from the overflow on
    tax = taxonomy(w, _events(w, answers), report, overflow_at=20)
    assert all(tax["per_item"][p] == ["overflow"] for p in range(20, 25)) and tax["counts"]["overflow"] == 5
    res = score_session(w, _events(w, answers), report, overflow_at=20)
    assert res["item_success"] == pytest.approx(19 / 24) and not res["session_success"] and not res["report"]["exact"]


def test_summary_loss_names_what_a_case_needs():
    w = gen_f8.generate("40", "dev", 1000)
    t = next(t for t in w.tasks if t.tags["kind"] == "followup")
    full = "\n".join(x.prompt for x in w.tasks) + "\n" + gen_f8.render_oracle_state(w, t.tags["position"] - 1)
    assert summary_loss(w, t.tags["position"], full) == []
    assert summary_loss(w, t.tags["position"], t.prompt) == [f"original request of {t.tags['original']}"]


def test_long_sessions_cross_the_window():
    stats = {n: gen_f8.generate(str(n), "dev", 1000).entities["session"]["reference"] for n in (10, 40, 60)}
    assert stats[10]["w_crossing_item"] is None
    assert 15 < stats[40]["w_crossing_item"] < 40 and stats[60]["w_crossing_item"] < 60
    assert 600 < stats[40]["tool_file_tokens_mean"] < 1400
    # Bigger tool files cross earlier.
    big = gen_f8.generate("40", "dev", 1000, output_tokens=2000).entities["session"]["reference"]
    assert big["w_crossing_item"] < stats[40]["w_crossing_item"]


# --- The runner, end to end on mockllm --------------------------------------------------------------------


@pytest.fixture(scope="module")
def session_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f8")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


def _run(tmp, arm, model, **kw):
    log = inspect_eval(f8_session(level="12", split="dev", arm=arm, checkpoints="5,10", **kw), model=model, log_dir=str(tmp / "logs"), display="none")[0]
    assert log.status == "success", log.error
    return log.samples[0]


def perfect_agent(world):
    """Answers every case with its gold, the report with its gold, and probes with the true state."""
    by_case = {t.tags["case_id"]: t for t in world.tasks}
    answered: list[str] = []

    def agent(messages, tools, tool_choice, config):
        if config.response_schema is not None:
            return ModelOutput.from_content("mockllm/model", json.dumps(state_at(world, len(answered))))
        last = next(m for m in reversed(messages) if m.role == "user")
        if "End of shift" in last.text:
            return ModelOutput.for_tool_call("mockllm/model", "submit_shift_report", {"report": json.dumps(world.entities["session"]["report_gold"])})
        t = by_case[re.findall(r"(?:Case|Ticket) (C-\d{6})", last.text)[-1]]
        cid = t.tags["case_id"]
        if t.tags["kind"] == "ticket":
            done = sum(isinstance(m, ChatMessageTool) for m in messages[messages.index(last) + 1 :])
            calls = t.gold["calls"]
            if done < len(calls):
                return ModelOutput.for_tool_call("mockllm/model", calls[done]["tool"], calls[done]["args"])
            answered.append(cid)
            return ModelOutput.for_tool_call("mockllm/model", "finish", {"case_id": cid})
        answered.append(cid)
        return ModelOutput.for_tool_call("mockllm/model", "submit_decision", {"case_id": cid, **t.gold})

    return agent


@pytest.mark.parametrize("arm", ["CM0", "O-state"])
def test_mock_session_runs_end_to_end(session_env, arm):
    s = _run(session_env, arm, get_model("mockllm/model", custom_outputs=mock_session_agent))
    items = s.store["f8_items"]
    assert [i["position"] for i in items] == list(range(1, 13))
    assert all(i["answered"] and i["view_tokens_decision"] > 0 for i in items)
    assert {"position", "view_tokens_first", "view_tokens_decision", "success", "dependency", "generations"} <= set(items[0])
    # One answer call per case, in its own window.
    events = s.store["f8_events"]
    for i in items:
        tool = "finish" if i["kind"] == "ticket" else "submit_decision"
        assert sum(e["item"] == i["position"] and e["tool"] == tool and e["args"]["case_id"] == i["case_id"] for e in events) == 1
    # Probes at the checkpoints, scored, and never appended to the history.
    assert [p["k"] for p in s.store["f8_probes"]] == [5, 10]
    assert not any("state check" in m.text for m in s.messages)
    assert s.store["f8_report"] is not None and s.store["f8_overflow_at"] is None
    assert set(s.scores["f8_session_score"].value) == {"item_success", "dependency_success", "report_exact", "session_success", "probe_f1", "probe_coverage", "overflow"}
    if arm == "O-state":
        assert max(v["view_tokens"] for v in s.store["f8_views"]) < 8000  # the view never accumulates the history


@pytest.mark.parametrize("arm", ["CM0", "O-state"])
def test_a_perfect_agent_scores_one(session_env, arm):
    from ape.agent.arms import load_world

    world = load_world("F8-12-dev-s1000")
    s = _run(session_env, arm, get_model("mockllm/model", custom_outputs=perfect_agent(world)))
    v = s.scores["f8_session_score"].value
    assert v == {"item_success": 1.0, "dependency_success": 1.0, "report_exact": 1.0, "session_success": 1.0, "probe_f1": 1.0, "probe_coverage": 1.0, "overflow": 0.0}


def test_cm0_overflows_past_the_window(session_env):
    s = _run(session_env, "CM0", get_model("mockllm/model", custom_outputs=mock_session_agent), window=8000)
    v, at = s.scores["f8_session_score"].value, s.store["f8_overflow_at"]
    assert at is not None and len(s.store["f8_items"]) == at - 1
    assert v["overflow"] == 1.0 and v["report_exact"] == 0.0 and s.store["f8_report"] is None
    assert max(x["view_tokens"] for x in s.store["f8_views"]) <= 8000


def test_probes_use_the_probe_role_when_defined(session_env):
    probe = get_model("mockllm/model", custom_outputs=lambda *a: ModelOutput.from_content("mockllm/model", json.dumps({c: ["C-111111"] for c in gen_f8.PROBE_CATEGORIES})))
    log = inspect_eval(
        f8_session(level="12", split="dev", arm="CM0", checkpoints="5"),
        model=get_model("mockllm/model", custom_outputs=mock_session_agent),
        model_roles={"probe": probe},
        log_dir=str(session_env / "logs"),
        display="none",
    )[0]
    p = log.samples[0].store["f8_probes"][0]
    assert p["answer"]["completed"] == ["C-111111"] and p["scores"]["completed"] == 0.0


def test_unbuilt_arms_are_refused():
    from ape.agent.session import f8_session_agent

    with pytest.raises(ValueError, match="not built yet"):
        f8_session_agent(arm="CM-lesson")  # Tier B: not built (CM-sum was, until B8)


def test_plan_knobs_make_short_sessions_cross_the_window_like_long_ones():
    """D-028: the run plan's F8 knobs put the reference session's W crossing near 0.70 of the session for the
    20- and 24-case cells, as the default does for 40 cases; knobs reach the generator through make_world."""
    import statistics

    import pytest

    from ape.budget import load_plan
    from ape.worlds import gen_f8
    from ape.worlds.generate import make_world

    plan = load_plan()
    cells = [c for c in plan.cells if c.study == "study_g" and c.spec.get("kind") == "session" and c.spec.get("knobs")]
    assert {c.id for c in cells} >= {"g.topo.luna", "g.topo.sol", "g.topo.astra", "g.cm.astra-high"}
    for cell in cells:
        n = int(cell.spec["N"])
        fractions = []
        for i in range(5):
            w = make_world("F8", str(n), "dev", i, 0, knobs=cell.spec["knobs"])
            assert w.entities["session"]["knobs"]["output_tokens"] == cell.spec["knobs"]["output_tokens"]
            crossing = gen_f8.reference_stats(w)["w_crossing_item"]
            assert crossing is not None, f"{cell.id}: session {i} never crosses W"
            fractions.append(crossing / n)
        assert 0.55 <= statistics.median(fractions) <= 0.85, (cell.id, fractions)
    with pytest.raises(ValueError, match="F8 only"):
        make_world("F7", "10", "dev", 0, 2, knobs={"output_tokens": 2000})
    with pytest.raises(ValueError, match="unknown F8 knobs"):
        make_world("F8", "10", "dev", 0, 0, knobs={"tokens": 2000})


def test_build_passes_knobs_and_seed_base_to_the_generator(session_env):
    """`ape.build` (and its --knob/--seed-base flags) reach make_world; the world ID carries the variant tag that
    `tasks.study_g` selects worlds by."""
    import asyncio

    from ape.build import build
    from ape.worlds.gen_f8 import variant_tag

    assert variant_tag(None) == "" and variant_tag({"output_tokens": 1000}) == "" and variant_tag({"output_tokens": 2250}) == "o2250"
    ids = asyncio.run(build("dev", "F8", ["20"], n_worlds=1, n_tasks=0, relational=True, embed=False, seed_base=1500, knobs={"output_tokens": 2250}))
    assert ids == ["F8-20-o2250-dev-s1500"]
