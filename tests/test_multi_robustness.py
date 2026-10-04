"""The multi-agent and ensemble arms under a misbehaving model (the B2 robustness suite, after
`tests/test_agent_robustness.py`): text-only replies, malformed and unknown tool calls, a worker that never reports or
never returns, an orchestrator that never answers, `delegate` with 0 or more than 3 subtasks, exceptions inside a
worker, and limits hit inside a worker, a council member or an ensemble attempt. None may crash the sample or lose
its records."""

import asyncio

import anyio
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, get_model
from inspect_ai.tool import ToolCall

from ape.agent.multi import core
from ape.apg.author import author_world
from ape.build import build
from ape.config import Config
from ape.llm.fake import perfect_author
from ape.llm.mock_multi import GoldMulti, calls_made, out
from ape.tasks.main import main_study
from ape.worlds.env_tools import ANSWER
from ape.worlds.spec import World

M = "mockllm/model"
REQUIRED = ("mas_switches", "mas_params", "mas_agents", "mas_accounting", "compile_log", "step_log", "turns_used", "arm")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi-robust")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv("APE_KG_ARM", raising=False)
    for fam, level in (("F1", "8"), ("F2", "2"), ("F3", "5"), ("F7", "10")):
        ids = asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    cfg = Config()
    asyncio.run(author_world(World.load(cfg.world_path(ids[0])), cfg, perfect_author))
    yield tmp
    mp.undo()


def _run(env, family, level, arm, model, **kw):
    gold = GoldMulti(env / "worlds")
    log = inspect_eval(main_study(family=family, level=level, split="dev", arm=arm), model=get_model(M, custom_outputs=model, memoize=False),
                       model_roles={"kg": get_model(M, custom_outputs=gold.kg, memoize=False)}, log_dir=str(env / "logs"), display="none", limit=1, **kw)[0]
    assert log.status == "success", log.error
    s = log.samples[0]
    assert s.error is None, s.error
    if arm == "S1":
        return s
    for key in REQUIRED:
        assert key in s.store, key
    acc = s.store["mas_accounting"]
    assert acc["unattributed"] == {}
    assert acc["totals"]["total_tokens"] == sum(u.total_tokens for u in s.model_usage.values())
    return s


def _agents(s, role=None):
    return [a for a in s.store["mas_agents"] if role is None or a["role"] == role]


# --- the orchestrator ------------------------------------------------------------------------------------------------


class ChattyOrchestrator(GoldMulti):
    def orchestrator(self, messages, world, task, delegate):
        return out(messages, text="I would rather discuss this.")


def test_an_orchestrator_that_only_chats_stops_after_one_nudge(env):
    s = _run(env, "F7", "10", "M1", ChattyOrchestrator(env / "worlds"))
    (orch,) = _agents(s, "orchestrator")
    # The forced planning turn offers `plan` alone (the gold mock plans); then it only chats.
    assert orch["tool_calls"] == ["plan"] and orch["stop"] == "text" and orch["turns"] == 3 and orch["nudges"] == 1
    assert s.scores["task_success"].value == "I" and s.store.get(ANSWER) is None and s.store["mas_rounds"] == []


class FumblingOrchestrator(GoldMulti):
    """An unknown tool, `delegate` with a string, with no subtasks, with four, with an empty one; then the gold."""

    BAD = [("frobnicate", {}), ("delegate", {"subtasks": "look up the customer"}), ("delegate", {"subtasks": []}),
           ("delegate", {"subtasks": ["a", "b", "c", "d"]}), ("delegate", {"subtasks": ["   "]})]

    def orchestrator(self, messages, world, task, delegate):
        if any(f == "plan" for f, _ in calls_made(messages)):
            errors = [m for m in messages if isinstance(m, ChatMessageTool) and m.error]
            if len(errors) < len(self.BAD):
                return out(messages, self.BAD[len(errors)])
        return super().orchestrator(messages, world, task, delegate)


@pytest.mark.parametrize("arm", ["M1", "M1s"])
def test_malformed_and_unknown_orchestrator_calls_are_tool_errors_and_run_no_worker(env, arm):
    s = _run(env, "F7", "10", arm, FumblingOrchestrator(env / "worlds"))
    errors = [m.error.message for m in s.messages if isinstance(m, ChatMessageTool) and m.error]
    assert len(errors) == 5
    assert "frobnicate not found" in errors[0]
    assert "you passed 0" in errors[2] and "you passed 4" in errors[3] and "Subtask 1 is empty" in errors[4]
    # Only the one valid round ran a worker; the task still succeeds.
    assert len(s.store["mas_rounds"]) == 1 and len(_agents(s, "worker")) == 1
    assert s.scores["task_success"].value == "C"


class EndlessOrchestrator(GoldMulti):
    def orchestrator(self, messages, world, task, delegate):
        return out(messages, ("plan", {"subtasks": ["think again"]}))


def test_an_orchestrator_that_never_answers_stops_at_its_turn_cap(env):
    s = _run(env, "F7", "10", "M1", EndlessOrchestrator(env / "worlds"))
    (orch,) = _agents(s, "orchestrator")
    assert orch["stop"] == "turn_cap" and orch["turns"] == s.store["mas_params"]["max_turns"] == s.store["turns_used"]
    assert s.scores["task_success"].value == "I" and len(s.store["mas_plan"]) == orch["turns"]


# --- workers ---------------------------------------------------------------------------------------------------------


class LoopingWorker(GoldMulti):
    def worker(self, messages, names):
        return out(messages, ("lookup_customer", {"customer_id": "CU-00000"}))


class ChattyWorker(GoldMulti):
    def worker(self, messages, names):
        return out(messages, text="The customer is gold tier; escalate it.")


class ConfusedWorker(GoldMulti):
    """Calls a tool it does not have and then the answer tool, which workers never get; then works properly."""

    def worker(self, messages, names):
        errors = [m for m in messages if isinstance(m, ChatMessageTool) and m.error]
        if len(errors) == 0:
            return out(messages, ("submit_decision", {"action": "approve", "approver": "none", "deadline_days": 1, "document": "none"}))
        if len(errors) == 1:
            return out(messages, ("lookup_customer", {"customer": 5}))
        return super().worker(messages, names)


def test_a_worker_that_never_reports_hits_its_cap_and_the_orchestrator_goes_on(env):
    s = _run(env, "F7", "10", "M1", LoopingWorker(env / "worlds"))
    (w,) = _agents(s, "worker")
    assert w["stop"] == "turn_cap" and w["status"] == "turn_cap" and w["turns"] == s.store["mas_params"]["worker_turns"]
    result = next(m.text for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "delegate")
    assert "reached its turn limit" in result
    assert s.scores["task_success"].value == "I" and _agents(s, "orchestrator")[0]["stop"] == "text"


def test_a_worker_that_only_chats_returns_its_text_marked(env):
    s = _run(env, "F7", "10", "M1", ChattyWorker(env / "worlds"))
    (w,) = _agents(s, "worker")
    assert w["stop"] == "text" and w["nudges"] == 1 and w["status"] == "text"
    result = next(m.text for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "delegate")
    assert "no report" in result and "escalate it" in result


def test_a_worker_calling_tools_it_lacks_gets_tool_errors_and_recovers(env):
    s = _run(env, "F7", "10", "M1", ConfusedWorker(env / "worlds"))
    (w,) = _agents(s, "worker")
    assert w["tool_calls"][:2] == ["submit_decision", "lookup_customer"] and w["status"] == "reported"
    assert s.store[ANSWER] is not None and s.scores["task_success"].value == "C"


class OnceBroken(GoldMulti):
    """One transient provider error: the first call `when(tool names)` selects raises once (BUILD_REVIEW r06)."""

    def __init__(self, worlds, when, exc: type[Exception] = RuntimeError):
        super().__init__(worlds)
        self.when, self.exc, self.fired = when, exc, False

    def __call__(self, messages, tools, tool_choice, config):
        if not self.fired and self.when({t.name for t in tools}):
            self.fired = True
            raise self.exc("Error code: 500 - upstream connect error (simulated)")
        return super().__call__(messages, tools, tool_choice, config)


def _member(names):
    return "lookup_supplier" in names and not names & {"delegate", "report", "plan"}


ERROR_CASES = [
    ("S1", "F7", "10", lambda n: True),
    ("M1", "F7", "10", lambda n: "report" in n),  # a worker's call
    ("M7", "F2", "2", _member),  # a council member's call
    ("S8k3", "F2", "2", lambda n: True),  # an ensemble attempt's call
]


@pytest.mark.parametrize("arm,family,level,when", ERROR_CASES, ids=[c[0] for c in ERROR_CASES])
def test_a_provider_error_inside_any_agent_is_retried_and_scored_like_s1(env, arm, family, level, when):
    """BUILD_REVIEW A-2: a provider error in a worker, a council member or an ensemble attempt ends the sample as it does
    S1's, so Inspect's retry_on_error re-runs it, instead of a scored failure or a lost vote the harness never sees."""
    s = _run(env, family, level, arm, OnceBroken(env / "worlds", when), retry_on_error=2)
    assert len(s.error_retries) == 1 and "upstream connect error" in s.error_retries[0].message
    assert s.scores["task_success"].value == "C"
    if arm != "S1":
        assert not any(a["error"] for a in s.store["mas_agents"]), "the attempt that was scored ran clean"
    if arm == "S8k3":
        assert all(v is not None for v in s.store["mas_ensemble"]["votes"]), "no vote lost"


@pytest.mark.parametrize("exc", [RuntimeError, TimeoutError])
def test_an_unretried_worker_error_is_a_harness_error_with_records(env, monkeypatch, exc):
    """Without retries the sample errors (a counted harness error), with its records. A TimeoutError, which
    execute_tools would turn into a tool error the orchestrator works around, is held by delegate and re-raised too;
    Inspect then treats it as it treats one from S1's own call (a warning, the state scored, no retry)."""
    orig = core.Delivery.compile

    async def flaky(self, agent, query, turn, source="push"):
        if agent.id == "w1.2":
            raise exc("kg exploded")
        return await orig(self, agent, query, turn, source)

    monkeypatch.setattr(core.Delivery, "compile", flaky)
    gold = GoldMulti(env / "worlds")
    log = inspect_eval(main_study(family="F1", level="8", split="dev", arm="M1"), model=get_model(M, custom_outputs=gold, memoize=False),
                       log_dir=str(env / "logs"), display="none", limit=1, fail_on_error=False)[0]
    s = log.samples[0]
    if exc is TimeoutError:
        assert s.error is None and s.scores["task_success"].value == "I"
    else:
        assert s.error is not None and "kg exploded" in s.error.message
    orch = next(a for a in s.store["mas_agents"] if a["role"] == "orchestrator")
    assert orch["tool_calls"] == ["plan", "delegate"] and orch["stop"] == "interrupted", "the orchestrator never worked around it"
    bad = next(a for a in s.store["mas_agents"] if a["id"] == "w1.2")
    assert bad["stop"] == "error" and "kg exploded" in bad["error"]
    assert s.store["mas_rounds"] and s.store.get(ANSWER) is None


class HangingWorker(GoldMulti):
    async def __call__(self, messages, tools, tool_choice, config):
        if "report" in {t.name for t in tools}:
            await anyio.sleep(3600)
        return super().__call__(messages, tools, tool_choice, config)


def test_a_worker_that_never_returns_is_cut_by_the_sample_time_limit_with_records_kept(env):
    s = _run(env, "F7", "10", "M1", HangingWorker(env / "worlds"), time_limit=3)
    assert s.limit is not None and s.limit.type == "time"
    assert [a["id"] for a in s.store["mas_agents"]] == ["orchestrator", "w1.1"]
    assert s.store["mas_rounds"][0]["subtasks"][0]["worker"] == "w1.1"


# --- limits inside agents ---------------------------------------------------------------------------------------------


def _first_calls_tokens(env, family, level, arm, n):
    """Total tokens of the sample's first `n` model calls under the gold mock (to place a token limit)."""
    gold = GoldMulti(env / "worlds")
    log = inspect_eval(main_study(family=family, level=level, split="dev", arm=arm), model=get_model(M, custom_outputs=gold, memoize=False),
                       model_roles={"kg": get_model(M, custom_outputs=gold.kg, memoize=False)}, log_dir=str(env / "logs"), display="none", limit=1)[0]
    assert log.samples[0].scores["task_success"].value == "C"
    calls = [e for e in log.samples[0].events if e.event == "model"]
    return sum(e.output.usage.total_tokens for e in calls[:n])


@pytest.mark.parametrize("arm", ["M1", "M1s"])
def test_a_token_limit_hit_inside_a_worker_ends_the_sample_with_records(env, arm):
    # The orchestrator's plan and first delegate calls fit; the first worker call trips the sample's limit.
    limit = _first_calls_tokens(env, "F1", "8", arm, 2) + 1
    s = _run(env, "F1", "8", arm, GoldMulti(env / "worlds"), token_limit=limit)
    assert s.limit is not None and s.limit.type == "token"
    acc = s.store["mas_accounting"]["agents"]
    assert acc["orchestrator"]["calls"] == 2, "the orchestrator never generated after the limit"
    tripped = [aid for aid, a in acc.items() if a["role"] == "worker" and a["calls"]]
    assert tripped and all(aid.startswith("w1.") for aid in tripped)
    stops = {a["id"]: a["stop"] for a in s.store["mas_agents"]}
    assert stops["orchestrator"] == "limit" and stops[tripped[0]] == "limit"
    assert len(s.store["mas_rounds"]) == 1 and s.store["compile_log"] and s.store.get(ANSWER) is None
    if arm == "M1s":
        assert tripped == ["w1.1"], "serial workers after the tripping one never start"


@pytest.mark.parametrize("arm,family,level", [("S8k3", "F3", "5"), ("M7", "F2", "2"), ("S9", "F2", "2"), ("M1k", "F7", "10")])
def test_a_token_limit_hit_inside_any_agent_ends_the_sample_with_records(env, arm, family, level):
    limit = _first_calls_tokens(env, family, level, arm, 3) + 1
    s = _run(env, family, level, arm, GoldMulti(env / "worlds"), token_limit=limit)
    assert s.limit is not None and s.limit.type == "token" and s.store.get(ANSWER) is None
    assert s.store["mas_agents"] and s.store["compile_log"]


# --- council and ensemble --------------------------------------------------------------------------------------------


class SilentCouncil(GoldMulti):
    """Members never propose (they only chat); the chair, shown no proposals, chats too."""

    def __call__(self, messages, tools, tool_choice, config):
        names = {t.name for t in tools}
        if "select_answer" in names or "delegate" in names or "report" in names:
            return super().__call__(messages, tools, tool_choice, config)
        return out(messages, text="I am not sure.")


def test_a_council_without_proposals_ends_unanswered_without_crashing(env):
    s = _run(env, "F2", "2", "M7", SilentCouncil(env / "worlds"))
    council = s.store["mas_council"]
    assert all(p["answer"] is None for phase in council["phases"] for p in phase) and council["chair"] is None
    assert all(a["stop"] == "text" for a in s.store["mas_agents"])
    assert s.scores["task_success"].value == "I"
    chair_prompt = next(m.text for m in s.messages if m.role == "user")
    assert chair_prompt.count("no proposal") == 3


class DoubleProposal(GoldMulti):
    """Council members propose twice in one turn (two parallel calls of the proposal tool): a wrong answer first."""

    def __call__(self, messages, tools, tool_choice, config):
        o = super().__call__(messages, tools, tool_choice, config)
        calls = o.message.tool_calls or []
        if len(calls) == 1 and "rationale" in calls[0].arguments:
            wrong = ToolCall(id=calls[0].id + "x", function=calls[0].function, arguments={**calls[0].arguments, "final_supplier": "SUP-00000", "chain": ["SUP-00000"]})
            o.message.tool_calls = [wrong, calls[0]]
        return o


def test_two_proposals_in_one_turn_record_the_first_and_say_so_to_the_second(env):
    """BUILD_REVIEW A-8: both used to be told "Proposal recorded." while only the first counted."""
    s = _run(env, "F2", "2", "M7", DoubleProposal(env / "worlds"))
    for phase in s.store["mas_council"]["phases"]:
        assert all(p["answer"]["final"] == "SUP-00000" for p in phase), "the first proposal of the turn is the member's"
    chair = next(a for a in s.store["mas_agents"] if a["role"] == "chair")
    assert chair["tool_calls"] == ["submit_chain"]


def test_the_proposal_slot_is_taken_before_the_parse_awaits():
    """The race itself (BUILD_REVIEW r11): two concurrent calls; one records, the other is told it was ignored."""
    import anyio
    from inspect_ai.tool import ToolDef, ToolParam, ToolParams
    from inspect_ai.util import store

    from ape.agent.multi import prompts as P
    from ape.agent.multi.primitives import proposal_tool

    async def submit_chain(final_supplier: str, chain: list[str]) -> str:
        await anyio.sleep(0.01)
        store().set(ANSWER, {"final": final_supplier, "chain": chain})
        return "ok"

    answer = ToolDef(submit_chain, name="submit_chain", description="Submit.", parameters=ToolParams(
        properties={"final_supplier": ToolParam(type="string"), "chain": ToolParam(type="array", items=ToolParam(type="string"))}, required=["final_supplier", "chain"]))
    sink: list = []
    prop = proposal_tool(None, answer, sink)
    assert prop.parallel is False
    results = {}

    async def main():
        async def call(name, final):
            results[name] = await prop.tool(final_supplier=final, chain=[final], rationale=name)

        async with anyio.create_task_group() as tg:
            tg.start_soon(call, "first", "SUP-00000")
            tg.start_soon(call, "second", "SUP-55741")

    anyio.run(main)
    assert sorted(results.values()) == sorted(["Proposal recorded.", P.PROPOSAL_AGAIN])
    (recorded,) = sink
    winner = next(k for k, v in results.items() if v == "Proposal recorded.")
    assert recorded["rationale"] == winner
