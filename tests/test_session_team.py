"""Topology arms in Study G sessions (BUILD_PLAN B9): S1 (= CM0), M1 and M2 on the ContextPolicy layer.

The gold mock runs perfect sessions under M1 and M2; S1 is exactly CM0; the orchestrator never holds the bulky files
(isolation), so M1/M2 finish sessions S1 overflows; W is enforced on workers too; per-agent accounting sums exactly
to the session's usage and the loader counts worker calls; probes go to the orchestrator's view; M1 and M2 differ
only in SPEC; every arm resumes mid-session exactly; misbehaving models never crash a session. Offline: mockllm."""

import asyncio
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model

from ape.agent.arms import load_world
from ape.agent.context_policy import FullHistory, policy_class
from ape.agent.multi import session_prompts as SP
from ape.agent.multi.session_team import ORCHESTRATOR_TOOLS, S1_SWITCHES, TOPOLOGY_ARMS, SessionTeam, SpecialistTeam
from ape.agent.multi.specialists import specialists
from ape.agent.session import SESSION_ARMS
from ape.agent.session_checkpoint import ENV
from ape.build import build
from ape.llm.mock_session import _current, gold_session_agent
from ape.tasks.study_g import f8_session
from ape.worlds import gen_f8

M = "mockllm/model"
WORLD12 = "F8-12-dev-s1000"
KNOBS20 = {"output_tokens": 2250}  # the topology cells' knob (run_plan.yaml g.topo.*)
RECORD_KEYS = ("f8_items", "f8_views", "f8_probes", "f8_events", "f8_report", "f8_overflow_at", "f8_cm_events", "f8_usage")
MAS_KEYS = ("mas_switches", "mas_params", "mas_rounds", "mas_accounting")


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f8-team")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv(ENV, raising=False)
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    (wid20,) = asyncio.run(build("dev", "F8", ["20"], n_worlds=1, n_tasks=0, relational=True, embed=False, knobs=KNOBS20))
    yield {"dir": tmp, "w20": wid20}
    mp.undo()


TASK_ARGS = ("window", "max_turns_per_item")


def _eval(worlds, arm, agent, level="12", variant="", **kw):
    task_kw = {k: kw.pop(k) for k in TASK_ARGS if k in kw}
    log = inspect_eval(
        f8_session(level=level, split="dev", arm=arm, checkpoints="5,10", variant=variant, **task_kw),
        model=get_model(M, custom_outputs=agent, memoize=False),
        log_dir=str(worlds["dir"] / "logs"),
        display="none",
        **kw,
    )[0]
    assert log.status == "success", log.error
    return log, log.samples[0]


def _gold(world_id=WORLD12):
    return gold_session_agent(load_world(world_id))


def _score(s):
    return s.scores["f8_session_score"].value


def _usage_sums_exactly(s) -> None:
    acc, by_kind = s.store["mas_accounting"], s.store["f8_usage"]["by_kind"]
    agent_views = [v for v in s.store["f8_views"] if v["kind"] == "agent"]
    for f in ("input_tokens", "output_tokens", "total_tokens"):
        per_agent = sum(a.get(f, 0) for a in acc["agents"].values())
        assert per_agent == sum(r.get(f, 0) for r in acc["roles"].values()) == acc["totals"].get(f, 0) == by_kind["agent"][f], f
        assert per_agent + acc["probes"].get(f, 0) == sum(getattr(u, f) for u in s.model_usage.values()), f
    assert sum(a["calls"] for a in acc["agents"].values()) == len(agent_views) and acc["probes"]["calls"] == len(s.store["f8_probes"])
    workers = {v["agent"] for v in agent_views if "agent" in v}
    assert workers == {w["id"] for w in s.store["mas_agents"] if w["role"] != "orchestrator"}


# --- registration and S1 ------------------------------------------------------------------------------------------


def test_topology_arms_are_registered_and_s1_is_cm0():
    assert set(TOPOLOGY_ARMS) == {"S1", "M1", "M2"} and set(TOPOLOGY_ARMS) <= set(SESSION_ARMS)
    assert policy_class("S1") is FullHistory is policy_class("CM0")
    assert policy_class("M1") is SessionTeam and policy_class("M2") is SpecialistTeam


def test_s1_runs_exactly_cm0_and_is_recorded_as_s1(worlds):
    _, cm0 = _eval(worlds, "CM0", _gold())
    _, s1 = _eval(worlds, "S1", _gold())
    assert {k: s1.store.get(k) for k in RECORD_KEYS} == {k: cm0.store.get(k) for k in RECORD_KEYS}
    assert [m.text for m in s1.messages] == [m.text for m in cm0.messages]
    assert s1.store["arm"] == {**cm0.store["arm"], "name": "S1"} and s1.store["arm"]["policy"] == "FullHistory"
    assert _score(s1) == _score(cm0) and "mas_accounting" not in s1.store


# --- M1 and M2 -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["M1", "M2"])
def test_gold_team_runs_perfect_sessions_with_every_record(worlds, arm):
    _, s = _eval(worlds, arm, _gold())
    sc = _score(s)
    assert s.error is None and s.store["f8_overflow_at"] is None
    assert sc["item_success"] == 1.0 and sc["report_exact"] == 1.0 and sc["session_success"] == 1.0
    for key in MAS_KEYS:
        assert s.store[key], key
    # The orchestrator answers and reports; it has no lookup or procedure tools, and its history never holds a file.
    assert s.store["arm"]["agent_tools"] == list(ORCHESTRATOR_TOOLS) and s.store["arm"]["policy_tools"] == ["delegate"]
    tool_msgs = [m for m in s.messages if isinstance(m, ChatMessageTool)]
    assert {m.function for m in tool_msgs} <= {*ORCHESTRATOR_TOOLS, "delegate"}
    world = load_world(WORLD12)
    answers = [e for e in s.store["f8_events"] if e["tool"] in ("submit_decision", "finish", "submit_shift_report")]
    lookups = [e for e in s.store["f8_events"] if e["tool"] in ("lookup_customer", "order_lookup")]
    assert len(answers) == len(world.tasks) + 1 and lookups
    # Workers: one fresh context per subtask, tagged calls, results back in the orchestrator's history.
    workers = [a for a in s.store["mas_agents"] if a["role"] != "orchestrator"]
    assert workers and all(w["status"] == "reported" and w["stop"] == "reported" for w in workers)
    assert all(not ({"submit_decision", "finish", "submit_shift_report"} & set(w["tool_calls"])) for w in workers)
    assert [r["round"] for r in s.store["mas_rounds"]] == list(range(1, len(s.store["mas_rounds"]) + 1))
    assert all(r["item"] in range(1, 13) for r in s.store["mas_rounds"])
    _usage_sums_exactly(s)
    # Items: the orchestrator's generations; worker calls count toward the item's usage.
    for item in s.store["f8_items"]:
        per_item = sum(v["usage"]["total_tokens"] for v in s.store["f8_views"] if v["item"] == item["position"] and v["kind"] == "agent")
        assert item["usage"]["agent"]["total_tokens"] == per_item


def test_isolation_finishes_the_sessions_s1_overflows(worlds):
    """At the topology cells' knobs (N = 20, 2,250-token files) S1 = CM0 crosses W; the orchestrator never holds a
    file, so M1 and M2 finish. Every agent's input, workers' included, stays under W."""
    world_id, variant = worlds["w20"], gen_f8.variant_tag(KNOBS20)
    _, s1 = _eval(worlds, "S1", _gold(world_id), level="20", variant=variant)
    assert s1.store["f8_overflow_at"] is not None and _score(s1)["item_success"] < 1.0
    for arm in ("M1", "M2"):
        _, s = _eval(worlds, arm, _gold(world_id), level="20", variant=variant)
        assert s.store["f8_overflow_at"] is None and _score(s)["session_success"] == 1.0
        views = s.store["f8_views"]
        orch = max(v["view_tokens"] for v in views if "agent" not in v)
        work = max(v["view_tokens"] for v in views if "agent" in v)
        assert max(orch, work) <= gen_f8.WINDOW and orch < max(v["view_tokens"] for v in s1.store["f8_views"])


def test_w_is_enforced_on_every_worker_call(worlds):
    """A window every earlier input fits but the first worker view that is the largest so far (one holding a file)
    does not: the session overflows at that item, from inside the worker, as CM0 would."""
    _, clean = _eval(worlds, "M1", _gold())
    views = clean.store["f8_views"]
    j = next(j for j, v in enumerate(views) if j and "agent" in v and v["view_tokens"] > max(x["view_tokens"] for x in views[:j]))
    window = max(x["view_tokens"] for x in views[:j])
    _, s = _eval(worlds, "M1", _gold(), window=window)
    pos = views[j]["item"]
    assert s.store["f8_overflow_at"] == pos and _score(s)["item_success"] == (pos - 1) / 12
    worker = next(a for a in s.store["mas_agents"] if a["id"] == views[j]["agent"])
    assert worker["stop"] == "overflow" and s.store["f8_views"] == views[:j]


class ProbeRecorder:
    """The gold agent, recording each probe's input."""

    def __init__(self, agent):
        self.agent, self.probes = agent, []

    def __call__(self, messages, tools, tool_choice, config):
        if config.response_schema is not None:
            self.probes.append(list(messages))
        return self.agent(messages, tools, tool_choice, config)


def test_probes_go_to_the_orchestrators_view(worlds):
    rec = ProbeRecorder(_gold())
    _, s = _eval(worlds, "M1", rec)
    assert len(rec.probes) == 2 and [p["k"] for p in s.store["f8_probes"]] == [5, 10]
    for probe in rec.probes:
        functions = {m.function for m in probe if isinstance(m, ChatMessageTool)}
        assert "delegate" in functions and not functions & {"lookup_customer", "order_lookup"}
        assert SP.ORCHESTRATOR_ADDENDUM.split("\n")[0] in probe[0].text  # the orchestrator's own system prompt
    assert all(p["scores"]["mean_f1"] == 1.0 for p in s.store["f8_probes"])
    # Never appended: the orchestrator's history holds no probe prompt.
    assert not any("Pause the shift" in (m.text or "") for m in s.messages)


def test_m1_and_m2_differ_only_in_specialization(worlds):
    _, m1 = _eval(worlds, "M1", _gold())
    _, m2 = _eval(worlds, "M2", _gold())
    s1, s2 = m1.store["mas_switches"], m2.store["mas_switches"]
    assert [k for k in S1_SWITCHES if s1[k] != s2[k]] == ["SPEC"]
    assert [k for k in S1_SWITCHES if s1[k] != S1_SWITCHES[k]] == ["DEC", "ISO", "CONC"]
    world = load_world(WORLD12)
    specs = {sp.name: sp for sp in specialists(world)}
    assert m2.store["mas_specialists"] == [{"name": n, "covers": list(sp.covers), "tools": list(sp.tools)} for n, sp in specs.items()]
    for w in m2.store["mas_agents"]:
        if w["role"] != "orchestrator":
            assert set(w["tool_calls"]) <= set(specs[w["role"]].tools) | {"report"}
    # The system prompts differ only in the team line of the orchestrator's addendum (M2: the roster).
    a, b = m1.messages[0].text.splitlines(), m2.messages[0].text.splitlines()
    only_a, only_b = [x for x in a if x not in set(b)], [y for y in b if y not in set(a)]
    assert len(only_a) == 1 and only_a[0].startswith("Every worker has the same knowledge")
    assert only_b[0].startswith("Your workers are specialists") and len(only_b) == 1 + len(specs)
    assert [x for x in a if x in set(b)] == [y for y in b if y in set(a)]


def test_f8_specialists_cover_both_kinds_of_domain():
    w = load_world(WORLD12)
    specs = specialists(w)
    policy, service = {p.domain for p in w.policies}, {t.domain for t in w.tools}
    covered = [d for sp in specs for d in sp.covers]
    assert set(covered) == policy | service and len(specs) == 3
    for sp in specs:
        want = (["lookup_customer"] if set(sp.covers) & policy else []) + (["order_lookup", *sorted(t.name for t in w.tools if t.domain in sp.covers)] if set(sp.covers) & service else [])
        assert list(sp.tools) == want


def test_the_g_loader_counts_worker_calls(worlds):
    from ape.analysis.g_load import load_g_cells

    log, s = _eval(worlds, "M1", _gold())
    data = load_g_cells({"g.topo.luna": [log.location]})
    row = data.sessions.iloc[0]
    assert row["arm"] == "M1" and row["calls"] == len(s.store["f8_views"]) > sum(1 for v in s.store["f8_views"] if "agent" not in v)
    agent = s.store["f8_usage"]["by_kind"]["agent"]
    assert row["tokens"] == agent["input_tokens"] + agent["output_tokens"]


# --- resume ----------------------------------------------------------------------------------------------------------


K = 7


def _position(text: str) -> int | None:
    world = load_world(WORLD12)
    hit = re.search(r"(?:Case|Ticket) (C-\d{6})", text or "")
    return next((t.tags["position"] for t in world.tasks if hit and t.tags["case_id"] == hit.group(1)), None)


class Flaky:
    """A call for item K by `role` (orchestrator or worker) raises once, a transient provider error: its first call,
    or with `mid` its first call after a tool result (a tool event of item K is recorded by then)."""

    def __init__(self, agent, role: str, mid: bool = False):
        self.agent, self.role, self.mid, self.left, self.positions = agent, role, mid, 1, []

    def __call__(self, messages, tools, tool_choice, config):
        names = {t.name for t in tools}
        if names:
            is_worker = "report" in names
            start, text = (1, next(m.text for m in messages if m.role == "user")) if is_worker else _current(messages)
            pos = _position(text)
            role = "worker" if is_worker else "orchestrator"
            self.positions.append((role, pos))
            acted = any(isinstance(m, ChatMessageTool) for m in messages[start + 1 :])
            if self.left and pos == K and role == self.role and acted == self.mid:
                self.left -= 1
                raise RuntimeError("transient API error (test)")
        return self.agent(messages, tools, tool_choice, config)


@pytest.mark.parametrize("arm,role,mid", [
    ("S1", "orchestrator", False), ("S1", "orchestrator", True), ("M1", "worker", False), ("M1", "worker", True),
    ("M1", "orchestrator", False), ("M1", "orchestrator", True), ("M2", "worker", True),
])
def test_every_topology_arm_resumes_mid_session_exactly(worlds, tmp_path, monkeypatch, arm, role, mid):
    """Including a failure mid-item, after a tool event of the item was recorded: the retry reruns the item from its
    first call and none of the failed attempt's events, rounds or workers survive."""
    _, clean = _eval(worlds, arm, _gold())
    monkeypatch.setenv(ENV, str(tmp_path / "ckpt"))
    flaky = Flaky(_gold(), role, mid)
    _, s = _eval(worlds, arm, flaky, retry_on_error=2, fail_on_error=False)
    assert s.error is None and len(s.error_retries) == 1 and "transient API error" in s.error_retries[0].message
    # Items 1..K-1 were not generated again: after the failure, every call works on item K or later (or the report).
    fail_at = next(i for i, (r, p) in enumerate(flaky.positions) if p == K and r == role)  # the item's first call by role
    assert all(p is None or p >= K for _, p in flaky.positions[fail_at:])
    assert {k: s.store.get(k) for k in RECORD_KEYS} == {k: clean.store.get(k) for k in RECORD_KEYS}
    assert [m.text for m in s.messages] == [m.text for m in clean.messages]
    if arm != "S1":
        strip = lambda agents: [{k: v for k, v in a.items() if k != "wall_s"} for a in agents]  # noqa: E731
        assert strip(s.store["mas_agents"]) == strip(clean.store["mas_agents"])
        for key in MAS_KEYS:
            assert s.store[key] == clean.store[key], key
        _usage_sums_exactly(clean)
    r = s.store["f8_resume"]
    assert r["count"] == 1 and r["after_items"] == [K - 1]
    assert not list((tmp_path / "ckpt").rglob("*.json"))


# --- misbehaving models ----------------------------------------------------------------------------------------------


def _agent_role(tools) -> str:
    names = {t.name for t in tools}
    return "worker" if "report" in names else "orchestrator" if "delegate" in names else "other"


class Override:
    """The gold agent with one role replaced by `fn(messages, tools)` (returning None defers to the gold)."""

    def __init__(self, role: str, fn, world_id=WORLD12):
        self.gold, self.role, self.fn = _gold(world_id), role, fn

    def __call__(self, messages, tools, tool_choice, config):
        if config.response_schema is None and tools and _agent_role(tools) == self.role:
            out = self.fn(messages, tools)
            if out is not None:
                return out
        return self.gold(messages, tools, tool_choice, config)


def _text(_m, _t):
    return ModelOutput.from_content(M, "Let me think about this shift.")


def _errors(s):
    return [m.error.message for m in s.messages if isinstance(m, ChatMessageTool) and m.error]


def test_an_orchestrator_that_only_chats_answers_nothing_and_never_crashes(worlds):
    _, s = _eval(worlds, "M1", Override("orchestrator", _text))
    assert s.error is None and _score(s)["item_success"] == 0.0 and s.store["mas_rounds"] == []
    assert all(i["generations"] == 2 for i in s.store["f8_items"])  # one nudge per case, then the next case


def test_malformed_and_unknown_orchestrator_calls_are_tool_errors(worlds):
    bad = [("frobnicate", {}), ("lookup_customer", {"customer_id": "CU-1"}), ("delegate", {"subtasks": "look it up"}),
           ("delegate", {"subtasks": []}), ("delegate", {"subtasks": ["a", "b", "c", "d"]}), ("delegate", {"subtasks": [" "]})]

    def fumble(messages, tools):
        i, text = _current(messages)
        if _position(text) != 1:
            return None
        errors = [m for m in messages[i + 1 :] if isinstance(m, ChatMessageTool) and m.error]
        return ModelOutput.for_tool_call(M, *bad[len(errors)]) if len(errors) < len(bad) else None

    _, s = _eval(worlds, "M1", Override("orchestrator", fumble), max_turns_per_item=10)
    errors = _errors(s)
    assert len(errors) == 6
    assert "frobnicate not found" in errors[0] and "lookup_customer not found" in errors[1]
    assert "you passed 0" in errors[3] and "you passed 4" in errors[4] and "Subtask 1 is empty" in errors[5]
    assert _score(s)["item_success"] == 1.0 and s.store["mas_rounds"][0]["item"] == 1


def test_workers_that_loop_chat_or_call_tools_they_lack_return_marked_results(worlds):
    def loop(messages, tools):
        return ModelOutput.for_tool_call(M, "lookup_customer", {"customer_id": "CU-00000"}) if "lookup_customer" in {t.name for t in tools} else None

    _, s = _eval(worlds, "M1", Override("worker", loop))
    workers = [a for a in s.store["mas_agents"] if a["role"] != "orchestrator"]
    looped = [w for w in workers if w["status"] == "turn_cap"]
    assert looped and all(w["turns"] == s.store["mas_params"]["worker_turns"] for w in looped)
    assert any("reached its turn limit" in m.text for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "delegate")
    assert s.error is None and s.store["f8_report"] is not None

    _, s = _eval(worlds, "M1", Override("worker", _text))
    workers = [a for a in s.store["mas_agents"] if a["role"] != "orchestrator"]
    assert workers and all(w["status"] == "text" and w["nudged"] for w in workers)
    assert s.error is None and _score(s)["item_success"] < 1.0

    def confused(messages, tools):
        errors = [m for m in messages if isinstance(m, ChatMessageTool) and m.error]
        if not errors:
            return ModelOutput.for_tool_call(M, "submit_decision", {"case_id": "C-000000", "action": "approve", "approver": "none", "deadline_days": 1, "document": "none"})
        return None

    _, s = _eval(worlds, "M2", Override("worker", confused))
    assert _score(s)["item_success"] == 1.0
    assert not any(e["tool"] == "submit_decision" and e["args"]["case_id"] == "C-000000" for e in s.store["f8_events"])


def test_a_token_limit_hit_inside_a_worker_ends_the_session_with_records(worlds):
    log, clean = _eval(worlds, "M1", _gold())
    calls = [e for e in clean.events if e.event == "model"]
    first_worker = next(i for i, v in enumerate(clean.store["f8_views"]) if "agent" in v)
    limit = sum(e.output.usage.total_tokens for e in calls[:first_worker]) + 1
    _, s = _eval(worlds, "M1", _gold(), token_limit=limit)
    assert s.limit is not None and s.limit.type == "token" and s.error is None
    (w,) = [a for a in s.store["mas_agents"] if a["role"] != "orchestrator"]
    assert w["stop"] == "limit" and s.store["mas_rounds"] and s.store["f8_items"] == []
    assert s.store["mas_accounting"]["agents"]["orchestrator"]["calls"] == first_worker
