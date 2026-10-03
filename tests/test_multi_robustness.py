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


def test_an_exception_inside_a_worker_is_a_failed_result_not_a_failed_sample(env, monkeypatch):
    orig = core.Delivery.compile

    async def flaky(self, agent, query, turn, source="push"):
        if agent.role == "worker" and agent.id == "w1.2":
            raise RuntimeError("kg exploded")
        return await orig(self, agent, query, turn, source)

    monkeypatch.setattr(core.Delivery, "compile", flaky)
    s = _run(env, "F1", "8", "M1", GoldMulti(env / "worlds"))
    bad = next(a for a in s.store["mas_agents"] if a["id"] == "w1.2")
    assert bad["status"] == "error" and "kg exploded" in bad["error"]
    result = next(m.text for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "delegate")
    assert "Subtask 2 (worker 2): [no result: the worker failed with an error]" in result
    assert s.scores["task_success"].value == "I" and s.scores["error_analysis"].metadata["error"] == "missing_items"
    assert sum(1 for a in _agents(s, "worker") if a["status"] == "reported") == 7


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


class OneBrokenAttempt(GoldMulti):
    """The first S8k3 attempt to start fails with an exception from the model; the others work."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.first = True

    def __call__(self, messages, tools, tool_choice, config):
        if self.first and len(messages) == 2:
            self.first = False
            raise RuntimeError("provider exploded")
        return super().__call__(messages, tools, tool_choice, config)


def test_an_ensemble_attempt_that_fails_leaves_the_others_to_vote(env):
    s = _run(env, "F7", "10", "S8k3", OneBrokenAttempt(env / "worlds"))
    attempts = _agents(s, "attempt")
    assert sum(a["stop"] == "error" for a in attempts) == 1 and "provider exploded" in next(a["error"] for a in attempts if a["error"])
    ens = s.store["mas_ensemble"]
    assert sum(v is not None for v in ens["votes"]) == 2 and ens["aggregation"] == "majority"
    assert s.scores["task_success"].value == "C"
