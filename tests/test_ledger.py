"""M5 and M5-spec: the Magentic-One-style ledger orchestrators (brief §4.2 M5, STATE=text-ledger; `agent/multi/ledger.py`).

The gold mock scores 100% through the real loop (M5 on F1, F2, F3, F7; M5-spec on F3, F7); M5 gives every agent
exactly M1's inputs but for the ledger block at the end of the orchestrator's latest message and the ledger side
calls (M5-spec likewise against M2); the ledger's parse, view, stall and replan rules; misbehaving ledger replies never
crash a sample and the turn cap bounds every loop; ledger calls are the orchestrator's in `mas_accounting` (purpose
`ledger`), summing exactly; a limit inside a ledger call ends the sample with its records; the stall knob."""

import asyncio
import json
import os

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentText, ModelOutput, get_model

from ape.agent.multi import knobs as K
from ape.agent.multi import prompts as P
from ape.agent.multi.ledger import (
    PROGRESS_LEDGER,
    TASK_LEDGER,
    block_view,
    ledger_sha,
    parse_progress,
    parse_task,
    render_task,
    strip_ledger,
)
from ape.agent.multi.solvers import REGISTRY, S1_SWITCHES
from ape.apg.author import author_world
from ape.build import build
from ape.config import Config
from ape.llm.fake import perfect_author
from ape.llm.mock_multi import GoldMulti
from ape.tasks.main import main_study
from ape.worlds.env_tools import ANSWER
from ape.worlds.spec import World

M = "mockllm/model"
CELLS = {"F1": "8", "F2": "5", "F3": "5", "F7": "10"}
N_TASKS = 2


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ledger")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for k in [k for k in os.environ if k.startswith(K.PREFIX) or k == "APE_KG_ARM"]:
        mp.delenv(k)
    cfg = Config()
    for fam, level in CELLS.items():
        for wid in asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=N_TASKS, relational=True, embed=True)):
            if fam in ("F3", "F7"):
                asyncio.run(author_world(World.load(cfg.world_path(wid)), cfg, perfect_author))
    yield tmp
    mp.undo()


def _eval(env, family, arm, model, **kw):
    gold = GoldMulti(env / "worlds")
    log = inspect_eval(main_study(family=family, level=CELLS[family], split="dev", arm=arm), model=get_model(M, custom_outputs=model, memoize=False),
                       model_roles={"kg": get_model(M, custom_outputs=gold.kg, memoize=False)}, log_dir=str(env / "logs"), display="none", **kw)[0]
    assert log.status == "success", log.error
    return log


def _ledger_calls(s) -> int:
    led = s.store["mas_ledger"]
    return len(led["task_ledgers"]) + len(led["progress"])


def _sums_exactly(s) -> None:
    acc = s.store["mas_accounting"]
    assert acc["unattributed"] == {}
    for f in ("input_tokens", "output_tokens", "total_tokens"):
        assert sum(b[f] for a in acc["agents"].values() for b in a["models"].values()) == sum(getattr(u, f) for u in s.model_usage.values()), f
    orch = acc["agents"]["orchestrator"]
    ledger = orch["purposes"].get("ledger", {"calls": 0})
    assert ledger["calls"] == _ledger_calls(s), "every ledger call is an orchestrator call with purpose ledger"
    assert orch["calls"] == orch["models"]["agent"]["calls"] >= ledger["calls"]
    assert not any(a["purposes"] for aid, a in acc["agents"].items() if aid != "orchestrator"), "workers make no ledger call"


# --- the gold mock through the real loop ----------------------------------------------------------------------------


GOLD = [("M5", f) for f in CELLS] + [("M5-spec", f) for f in ("F3", "F7")]


@pytest.mark.parametrize("arm,family", GOLD, ids=[f"{a}-{f}" for a, f in GOLD])
def test_gold_mock_scores_100_percent_with_the_ledger(env, arm, family):
    log = _eval(env, family, arm, GoldMulti(env / "worlds"))
    assert all(s.error is None and s.limit is None for s in log.samples)
    assert [s.scores["task_success"].value for s in log.samples] == ["C"] * N_TASKS
    for s in log.samples:
        sw = s.store["mas_switches"]
        base = {**S1_SWITCHES, **REGISTRY["M1" if arm == "M5" else "M2"]} | {"DEL": "MONO" if arm == "M5" else "KGs"}
        assert [k for k in S1_SWITCHES if sw[k] != base[k]] == ["STATE"] and sw["STATE"] == "text-ledger"
        params = s.store["mas_params"]
        assert params["ledger_stall"] == 2 and params["ledger_sha"] == ledger_sha() and params["prompt_variant"] == "default"
        led = s.store["mas_ledger"]
        orch = next(a for a in s.store["mas_agents"] if a["role"] == "orchestrator")
        assert led["task_ledgers"][0] | {"ledger": None} == {"turn": 0, "reason": "start", "version": 1, "ledger": None}
        assert [p["turn"] for p in led["progress"]] == list(range(1, orch["turns"])), "one progress ledger before every later turn"
        assert led["replans"] == 0 and led["stalls_max"] == 0 and all(not p["stalled"] for p in led["progress"])
        _sums_exactly(s)


# --- M5 = M1 + the ledger --------------------------------------------------------------------------------------------


def _canon(messages) -> str:
    def one(m):
        rec = {"role": m.role, "text": m.text}
        if isinstance(m, ChatMessageAssistant):
            rec["calls"] = [(c.id, c.function, c.arguments) for c in m.tool_calls or []]
        if isinstance(m, ChatMessageTool):
            rec |= {"id": m.tool_call_id, "fn": m.function}
        return rec

    return json.dumps([one(m) for m in messages], sort_keys=True)


class Recorder(GoldMulti):
    """The gold mock, keeping every input: agent calls (the ledger block stripped, and whether it was there) apart from
    ledger calls."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.agent_inputs: list[str] = []
        self.blocks: list[tuple[str, bool]] = []
        self.ledger_inputs: list[list] = []

    def __call__(self, messages, tools, tool_choice, config):
        names = {t.name for t in tools}
        if config.response_schema is not None:
            self.ledger_inputs.append(list(messages))
        else:
            stripped = strip_ledger(messages)
            role = "worker" if "report" in names else "orchestrator"
            self.blocks.append((role, stripped != messages))
            self.agent_inputs.append(_canon(stripped) + "|" + ",".join(sorted(names)) + f"|{tool_choice}")
        return super().__call__(messages, tools, tool_choice, config)


@pytest.mark.parametrize("base,arm,family", [("M1", "M5", "F1"), ("M1", "M5", "F7"), ("M2", "M5-spec", "F3"), ("M2", "M5-spec", "F7")])
def test_the_ledger_arm_gives_every_agent_its_base_arms_inputs_but_the_ledger(env, base, arm, family):
    runs = {}
    for a in (base, arm):
        rec = Recorder(env / "worlds")
        log = _eval(env, family, a, rec, max_samples=1)
        runs[a] = (rec, log)
    (rb, lb), (rl, ll) = runs[base], runs[arm]
    assert sorted(rb.agent_inputs) == sorted(rl.agent_inputs), "byte-identical inputs once the ledger block is removed"
    assert not any(had for _, had in rb.blocks) and rb.ledger_inputs == []
    assert all(had == (role == "orchestrator") for role, had in rl.blocks), "every orchestrator call carries the ledger, no worker call does"
    assert len(rl.ledger_inputs) == sum(_ledger_calls(s) for s in ll.samples)
    for msgs in rl.ledger_inputs:  # a side call: the orchestrator's view (no ledger block) plus the ledger prompt
        assert isinstance(msgs[-1], ChatMessageUser) and msgs[-1].text.split("\n")[0] in {p.split("\n")[0] for p in (P.TASK_LEDGER_PROMPT, P.PROGRESS_LEDGER_PROMPT)}
        assert not any(P.LEDGER_HEADER in (m.text or "") for m in msgs)
    for sb, sl in zip(lb.samples, ll.samples, strict=True):
        assert sb.store[ANSWER] == sl.store[ANSWER] and sb.store["mas_rounds"] == sl.store["mas_rounds"] and sb.store["mas_plan"] == sl.store["mas_plan"]
        assert [m.text for m in sb.messages] == [m.text for m in sl.messages], "the ledger never enters the history"
        assert sb.store["mas_params"]["prompt_sha"] == sl.store["mas_params"]["prompt_sha"]


def test_m5_reads_m1s_prompt_variant_and_its_own_stall_knob(env, monkeypatch):
    monkeypatch.setenv("APE_MAS_M1_PROMPT", "concise")
    monkeypatch.setenv("APE_MAS_M5_STALL", "4")
    s = _eval(env, "F2", "M5", GoldMulti(env / "worlds"), limit=1).samples[0]
    assert s.store["mas_params"]["prompt_variant"] == "concise" and s.store["mas_params"]["prompt_sha"] == P.VARIANTS["concise"].sha()
    assert s.store["mas_params"]["ledger_stall"] == 4 == s.store["mas_ledger"]["stall_threshold"]
    assert K.arm_knobs("M5") == K.arm_knobs("M5-spec") == ["APE_MAS_M1_PROMPT", "APE_MAS_M1_CLIP", "APE_MAS_M5_STALL"]
    assert K.STALL not in K.arm_knobs("M1")
    for bad in ("0", "11", "two"):
        with pytest.raises(ValueError, match="stalled turns from 1 to 10"):
            K.resolve("M5", {"APE_MAS_M5_STALL": bad})
    assert K.candidate_problems("M1", {"APE_MAS_M5_STALL": "3"}) == ["M1 candidates set only M1's knobs (APE_MAS_M1_PROMPT, APE_MAS_M1_CLIP), not APE_MAS_M5_STALL"]


# --- the ledger's rules ----------------------------------------------------------------------------------------------


def test_ledger_replies_are_parsed_strictly_and_rendered_small():
    good = {"given_facts": ["a", " ", 3], "facts_to_look_up": ["b"], "facts_to_derive": [], "educated_guesses": ["x" * 1000], "plan": ["one", "two"]}
    t = parse_task(json.dumps(good))
    assert t["given_facts"] == ["a"] and len(t["educated_guesses"][0]) == 400 and t["plan"] == ["one", "two"]
    assert "  1. one\n  2. two" in render_task(t) and render_task(None) == P.LEDGER_NONE
    for bad in ("not json", "[]", json.dumps({**good, "plan": "one"}), json.dumps({k: v for k, v in good.items() if k != "plan"})):
        with pytest.raises(ValueError):
            parse_task(bad)
    p = {"request_satisfied": False, "in_loop": True, "progress_being_made": False, "next_action": "a", "instruction": "b", "reason": "c"}
    assert parse_progress(json.dumps(p)) == p
    for bad in ("{}", json.dumps({**p, "in_loop": "yes"}), "nope"):
        with pytest.raises(ValueError):
            parse_progress(bad)


def test_the_block_rides_at_the_end_of_a_copy_and_strips_back_exactly():
    tool = ChatMessageTool(content="result", tool_call_id="c1", function="delegate")
    history = [ChatMessageUser(content="task"), ChatMessageAssistant(content=""), tool]
    view = block_view(history, f"{P.LEDGER_HEADER}\nbody")
    assert view[-1].text == f"result\n\n{P.LEDGER_HEADER}\nbody" and history[-1].text == "result" and len(view) == 3
    assert strip_ledger(view)[-1].text == "result" and strip_ledger(history) == history
    listed = ChatMessageTool(content=[ContentText(text="part")], tool_call_id="c1", function="delegate")
    v = block_view([listed], f"{P.LEDGER_HEADER}\nbody")
    assert v[-1].text.endswith("body") and strip_ledger(v)[-1].content == listed.content
    first = block_view([ChatMessageUser(content="task")], f"{P.LEDGER_HEADER}\nbody")  # the first turn: a message of its own
    assert [m.role for m in first] == ["user", "user"] and strip_ledger(first) == first[:1]


class Scripted(GoldMulti):
    """The gold mock with the ledger replies replaced: `task(n)` and `progress(n)` give the n-th reply's text."""

    def __init__(self, worlds, task=None, progress=None, orchestrator=None):
        super().__init__(worlds)
        self.task_fn, self.progress_fn, self.orch_fn = task, progress, orchestrator
        self.n = {TASK_LEDGER: 0, PROGRESS_LEDGER: 0}

    def __call__(self, messages, tools, tool_choice, config):
        if config.response_schema is not None and config.response_schema.name in self.n:
            name = config.response_schema.name
            self.n[name] += 1
            fn = self.task_fn if name == TASK_LEDGER else self.progress_fn
            if fn is not None:
                return ModelOutput.from_content(M, fn(self.n[name]))
        if self.orch_fn is not None and "delegate" in {t.name for t in tools}:
            return self.orch_fn(strip_ledger(messages))
        return super().__call__(messages, tools, tool_choice, config)


STALLED = json.dumps({"request_satisfied": False, "in_loop": True, "progress_being_made": False, "next_action": "x", "instruction": "y", "reason": "z"})


def test_stalls_force_a_replan_at_the_threshold_and_restart_the_count(env, monkeypatch):
    seq = [True, True, False, True, True, True]  # stalled? per progress reply

    def progress(n):
        stalled = seq[n - 1] if n <= len(seq) else False
        return json.dumps({"request_satisfied": False, "in_loop": stalled, "progress_being_made": not stalled, "next_action": "a", "instruction": "b", "reason": "c"})

    s = _eval(env, "F2", "M5", Scripted(env / "worlds", progress=progress), limit=1).samples[0]
    led = s.store["mas_ledger"]
    assert s.scores["task_success"].value == "C"
    stalls = [p["stalls"] for p in led["progress"]]
    # Stalled, stalled (2: the replan, then the count restarts), progress (stays 0), stalled, stalled (replan), stalled.
    # An entry records the count it reached, the one a replan fires at.
    assert stalls[:6] == [1, 2, 0, 1, 2, 1] and led["replans"] == 2
    reasons = [(t["reason"], t["turn"], t["version"]) for t in led["task_ledgers"]]
    assert reasons == [("start", 0, 1), ("replan", 2, 2), ("replan", 5, 3)]
    monkeypatch.setenv("APE_MAS_M5_STALL", "1")  # every stalled turn replans
    s = _eval(env, "F2", "M5", Scripted(env / "worlds", progress=lambda n: STALLED), limit=1).samples[0]
    led = s.store["mas_ledger"]
    assert led["replans"] == len(led["progress"]) and s.scores["task_success"].value == "C"
    _sums_exactly(s)


# --- misbehaving ledger replies, bounded by the turn cap -------------------------------------------------------------


@pytest.mark.parametrize("reply", ["not json at all", json.dumps({"plan": "x"}), ""], ids=["text", "wrong-shape", "empty"])
def test_malformed_ledger_replies_are_recorded_and_change_nothing(env, reply):
    s = _eval(env, "F7", "M5", Scripted(env / "worlds", task=lambda n: reply, progress=lambda n: reply), limit=1).samples[0]
    assert s.error is None and s.scores["task_success"].value == "C"
    led = s.store["mas_ledger"]
    assert led["task_ledgers"][0]["version"] is None and led["task_ledgers"][0]["error"]
    assert all(p.get("error") and p["stalls"] == 0 for p in led["progress"]) and led["replans"] == 0
    _sums_exactly(s)


def test_a_perpetual_stall_and_an_orchestrator_that_never_answers_stop_at_the_turn_cap(env):
    def chatter(messages):
        return ModelOutput.for_tool_call(M, "plan", {"subtasks": ["think again"]})

    log = _eval(env, "F7", "M5", Scripted(env / "worlds", progress=lambda n: STALLED, orchestrator=chatter), limit=1)
    s = log.samples[0]
    turns = s.store["mas_params"]["max_turns"]
    orch = next(a for a in s.store["mas_agents"] if a["role"] == "orchestrator")
    led = s.store["mas_ledger"]
    assert orch["stop"] == "turn_cap" and orch["turns"] == turns and s.store.get(ANSWER) is None
    assert len(led["progress"]) == turns - 1 and led["replans"] == (turns - 1) // 2
    assert _ledger_calls(s) == 1 + (turns - 1) + led["replans"] <= 2 * turns, "at most one progress call and one replan per turn"
    _sums_exactly(s)


class LedgerBreaks(GoldMulti):
    """The first ledger call raises once: a provider error."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.fired = False

    def __call__(self, messages, tools, tool_choice, config):
        if not self.fired and config.response_schema is not None:
            self.fired = True
            raise RuntimeError("Error code: 500 - upstream connect error (simulated)")
        return super().__call__(messages, tools, tool_choice, config)


def test_a_provider_error_in_a_ledger_call_is_retried_like_any_agent_call(env):
    s = _eval(env, "F2", "M5", LedgerBreaks(env / "worlds"), limit=1, retry_on_error=1).samples[0]
    assert s.error is None and len(s.error_retries) == 1 and s.scores["task_success"].value == "C"
    s = _eval(env, "F2", "M5", LedgerBreaks(env / "worlds"), limit=1, fail_on_error=False).samples[0]
    assert s.error is not None and "upstream" in s.error.message and "mas_ledger" in s.store and "mas_accounting" in s.store


def test_a_token_limit_hit_inside_a_ledger_call_ends_the_sample_with_records(env):
    clean = _eval(env, "F2", "M5", GoldMulti(env / "worlds"), limit=1).samples[0]
    first = next(e for e in clean.events if e.event == "model")  # the task-ledger call comes first
    s = _eval(env, "F2", "M5", GoldMulti(env / "worlds"), limit=1, token_limit=first.output.usage.total_tokens - 1).samples[0]
    assert s.limit is not None and s.limit.type == "token" and s.error is None
    orch = next(a for a in s.store["mas_agents"] if a["role"] == "orchestrator")
    assert orch["stop"] == "limit" and s.store["mas_accounting"]["agents"]["orchestrator"]["purposes"]["ledger"]["calls"] == 1
    assert s.store.get(ANSWER) is None and "mas_ledger" in s.store
