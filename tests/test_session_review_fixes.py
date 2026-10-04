"""The independent review's session findings (BUILD_REVIEW.md, package F-session), each with its fix. Offline: mockllm.

- A-3: the todo arms ask for the list update at the start of each case (the loop gives no turn after an answer), the
  same text for every todo arm; the gold mock and the extraction prompt follow it.
- A-5: the call that trips a token limit is recorded (from the transcript), so `f8_usage` and `mas_accounting` equal
  Inspect's usage: agent calls, management calls, native compaction calls, probes and team workers.
- A-6: a session ended by a time limit (a cancellation, never a LimitExceededError in the solver) deletes its checkpoint.
- A-7: CM-prune logs and counts a compaction only when it changed something (IDs aside).
- A-9 / R-C3: CM-native's provider compactions reach the run's usage ledger.
- r09: CM-native calibrates its opaque block on a case's first call, so the case's re-read reasoning is not counted.
- CM-trim: a trimmed window starts at a case message.
"""

import asyncio
import json

import anyio
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentData, GenerateConfig, ModelUsage, get_model, modelapi
from inspect_ai.model._providers.mockllm import MockLLM
from inspect_ai.tool import ToolCall
from inspect_ai.util import LimitExceededError
from test_cm_arms import FakeNative  # noqa: F401  (registers the fakenative provider)

from ape import usage_ledger
from ape.agent import cm_arms, context_policy
from ape.agent.arms import load_world
from ape.agent.cm_arms import NATIVE_RECORD_ENV, CMTrim, record_native_support
from ape.agent.cm_prompts import TODO_ADDENDUM, TODO_EXTRACT_PROMPT, management_request
from ape.agent.context_policy import SessionContext, SessionRecords, inspect_limit_hit
from ape.agent.session_checkpoint import ENV as CHECKPOINTS_ENV
from ape.build import build
from ape.llm.mock_session import gold_session_agent
from ape.tasks.study_g import f8_session

M = "mockllm/model"
W = "F8-12-dev-s1000"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("review-fixes")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for k in (CHECKPOINTS_ENV, NATIVE_RECORD_ENV, "APE_CM_STACK", "APE_CACHE_NONCE"):
        mp.delenv(k, raising=False)
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


def _run(env, arm, agent, model=M, eval_kw=None, **kw):
    log = inspect_eval(
        f8_session(level="12", split="dev", arm=arm, **kw),
        model=get_model(model, custom_outputs=agent, memoize=False),
        log_dir=str(env / "logs"),
        display="none",
        fail_on_error=False,
        **(eval_kw or {}),
    )[0]
    assert log.status == "success", log.error
    return log.samples[0]


def _model_events(sample):
    return [e for e in sample.events if e.event == "model" and e.output is not None and e.output.usage is not None]


def _recorded_total(sample) -> int:
    return sum((u or {}).get("total_tokens", 0) for u in sample.store["f8_usage"]["by_kind"].values())


def _inspect_total(sample) -> int:
    return sum(u.total_tokens for u in sample.model_usage.values())


# --- A-3 ----------------------------------------------------------------------------------------------------------


def test_the_todo_instruction_matches_the_loop_and_is_the_same_for_every_todo_arm(env, monkeypatch):
    assert "A case ends as soon as you submit its answer" in TODO_ADDENDUM
    assert "at the start of each case: first mark the previous case completed and this one in_progress" in TODO_ADDENDUM
    assert "mark any case still in_progress as completed (a case ends with its answer)" in TODO_EXTRACT_PROMPT
    prompts = []
    for arm, stack in (("CM-todo", None), ("CM-reset", None), ("S-CM*", "prune+todo"), ("S-CM*", "todo+sum"), ("S-CM*", "trim+todo+reset")):
        if stack:
            monkeypatch.setenv("APE_CM_STACK", stack)
        s = _run(env, arm, gold_session_agent(load_world(W)), threshold=10**6)
        prompts.append(s.messages[0].text)
        assert s.messages[0].text.endswith(TODO_ADDENDUM), (arm, stack)
    assert len(set(prompts)) == 1


def test_the_gold_mock_updates_its_list_at_the_start_of_each_case(env):
    world = load_world(W)
    s = _run(env, "CM-todo", gold_session_agent(world), threshold=10**6)
    writes = [e for e in s.store["f8_events"] if e["tool"] == "todo_write"]
    queue = world.entities["session"]["queue"]
    assert [e["item"] for e in writes] == list(range(1, 13))
    for e in writes:
        k = e["item"]
        status = {t["content"]: t["status"] for t in e["args"]["todos"]}
        assert all(status[f"Case {c}"] == "completed" for c in queue[: k - 1])  # the previous cases closed first
        assert status[f"Case {queue[k - 1]}"] == "in_progress" and all(status[f"Case {c}"] == "pending" for c in queue[k:])
    # Its extraction answer follows the reworded prompt: the case the message opens is in progress.
    case = world.tasks[3]
    req = management_request(TODO_EXTRACT_PROMPT, [("Current todo list", "[]"), ("Message", case.prompt)])
    out = gold_session_agent(world)([ChatMessageUser(content=req)], [], "none", GenerateConfig())
    assert {t["content"]: t["status"] for t in json.loads(out.completion)}[f"Case {case.tags['case_id']}"] == "in_progress"


# --- A-5 ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["CM0", "CM-sum", "CM-todo", "CM-native", "M1"])
def test_records_equal_inspects_usage_when_a_token_limit_trips(env, arm):
    clean = _run(env, arm, gold_session_agent(load_world(W)), threshold=6000, checkpoints="5,10")
    limit = sum(e.output.usage.total_tokens for e in _model_events(clean)[:9]) + 1
    s = _run(env, arm, gold_session_agent(load_world(W)), threshold=6000, checkpoints="5,10", eval_kw={"token_limit": limit})
    assert s.limit is not None and s.limit.type == "token"
    assert _recorded_total(s) == _inspect_total(s)
    tripped = [c for c in s.store["f8_views"] + s.store["f8_probes"] if c.get("limit")]
    assert len(tripped) == 1 and len(s.store["f8_views"]) + len(s.store["f8_probes"]) == len(_model_events(s))
    if arm == "M1":
        acc = s.store["mas_accounting"]
        assert acc["totals"]["total_tokens"] + acc["probes"].get("total_tokens", 0) == _inspect_total(s)


def test_a_probe_that_trips_the_limit_is_recorded_but_not_taken(env):
    clean = _run(env, "CM0", gold_session_agent(load_world(W)), checkpoints="5,10")
    events = _model_events(clean)
    first_probe = next(i for i, e in enumerate(events) if e.config.response_schema is not None)
    s = _run(env, "CM0", gold_session_agent(load_world(W)), checkpoints="5,10", eval_kw={"token_limit": sum(e.output.usage.total_tokens for e in events[:first_probe]) + 1})
    (probe,) = s.store["f8_probes"]
    assert probe["limit"] and probe["k"] == 5 and probe["answer"] is None and probe["usage"]["total_tokens"] > 0
    assert _recorded_total(s) == _inspect_total(s)
    by_k = s.scores["f8_session_score"].metadata["probes_by_checkpoint"]
    assert by_k["5"]["taken"] is False and s.scores["f8_session_score"].value["probe_coverage"] == 0.0


def test_a_worker_call_that_trips_the_limit_is_in_the_team_accounting(env):
    clean = _run(env, "M1", gold_session_agent(load_world(W)), checkpoints="5,10")
    events = _model_events(clean)
    first_worker = next(i for i, e in enumerate(events) if any(t.name == "report" for t in e.tools))
    s = _run(env, "M1", gold_session_agent(load_world(W)), checkpoints="5,10", eval_kw={"token_limit": sum(e.output.usage.total_tokens for e in events[:first_worker]) + 1})
    (tripped,) = [v for v in s.store["f8_views"] if v.get("limit")]
    assert tripped["agent"].startswith("w") and tripped["role"] == "worker"
    acc = s.store["mas_accounting"]
    assert acc["totals"]["total_tokens"] + acc["probes"].get("total_tokens", 0) == _inspect_total(s) == _recorded_total(s)
    assert acc["agents"][tripped["agent"]]["calls"] == 1


# --- A-6 ----------------------------------------------------------------------------------------------------------


class Hangs:
    """The gold agent, which stops answering (sleeps) after `n` calls: the sample's time limit cancels it."""

    def __init__(self, agent, n: int):
        self.agent, self.n = agent, n

    async def __call__(self, messages, tools, tool_choice, config):
        self.n -= 1
        if self.n < 0:
            await anyio.sleep(3600)
        return self.agent(messages, tools, tool_choice, config)


def test_a_time_limited_session_deletes_its_checkpoint(env, tmp_path, monkeypatch):
    monkeypatch.setenv(CHECKPOINTS_ENV, str(tmp_path / "ckpt"))
    s = _run(env, "CM0", Hangs(gold_session_agent(load_world(W)), 12), checkpoints="5,10", eval_kw={"time_limit": 3})
    assert s.limit is not None and s.limit.type == "time" and 0 < len(s.store["f8_items"]) < 12
    assert not list((tmp_path / "ckpt").rglob("*.json")), "a limit-ended session is finished: its checkpoint is gone"
    again = _run(env, "CM0", gold_session_agent(load_world(W)), checkpoints="5,10")
    assert again.store["f8_resume"] is None and len(again.store["f8_items"]) == 12


def test_a_working_limit_recorded_on_the_active_sample_is_a_limit(monkeypatch):
    import inspect_ai.log._samples as samples

    class Active:
        limit_exceeded_error = LimitExceededError("working", value=11.0, limit=10.0)

    monkeypatch.setattr(samples, "sample_active", lambda: Active())
    assert inspect_limit_hit() == "working"
    Active.limit_exceeded_error = None
    assert inspect_limit_hit() is None  # no limit recorded, no time limit in force: an interrupt keeps the checkpoint


# --- A-7 ----------------------------------------------------------------------------------------------------------


def test_cm_prune_logs_a_compaction_only_when_it_changes_something(env):
    s = _run(env, "CM-prune", gold_session_agent(load_world(W)), threshold=6000)
    events = s.store["f8_cm_events"]
    reactions = [e for e in events if e["event"] == "threshold"]
    assert reactions and all(e["view_tokens_after"] < e["view_tokens"] for e in reactions)
    assert sum(e["event"] == "prune" for e in events) == len(reactions)
    a = ChatMessageTool(content="(Tool result removed)", tool_call_id="c1", function="lookup_customer")
    assert cm_arms._content([a]) == cm_arms._content([a.model_copy(update={"id": "another"})])
    assert cm_arms._content([a]) != cm_arms._content([a.model_copy(update={"content": "file"})])


# --- A-9 / R-C3 and r09: CM-native -----------------------------------------------------------------------------------


@modelapi(name="fakenative_reasoning")
class FakeNativeReasoning(MockLLM):
    """test_cm_arms' FakeNative, whose reported input also counts 1,500 tokens of re-sent reasoning per assistant turn
    of the current case (as the Responses API re-reads a case's encrypted reasoning items)."""

    async def compact(self, input, tools, config, instructions=None):
        return [ChatMessageUser(content=[ContentData(data={"compaction": f"{len(input)} messages"})])], ModelUsage(input_tokens=1000, output_tokens=50, total_tokens=1050)

    async def generate(self, input, tools, tool_choice, config):
        out = await super().generate(input, tools, tool_choice, config)
        o = out[0] if isinstance(out, tuple) else out
        last_user = max((i for i, m in enumerate(input) if m.role == "user"), default=0)
        carry = 1500 * sum(isinstance(m, ChatMessageAssistant) for m in input[last_user:])
        o.usage = ModelUsage(input_tokens=o.usage.input_tokens + carry, output_tokens=o.usage.output_tokens, total_tokens=o.usage.total_tokens + carry)
        return out


def test_native_compactions_reach_the_usage_ledger(env, tmp_path, monkeypatch):
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(tmp_path / "native.json"))
    record_native_support("fakenative/model", True, evidence="offline test provider")
    ledger = tmp_path / "usage_ledger.jsonl"
    with usage_ledger.recording(ledger):
        s = _run(env, "CM-native", gold_session_agent(load_world(W)), model="fakenative/model", threshold=6000)
    entries = usage_ledger.read_entries([ledger])
    compactions = [v for v in s.store["f8_views"] if v.get("purpose") == "native"]
    compact_entries = [e for e in entries if (e["input_tokens"], e["output_tokens"]) == (1000, 50)]
    assert compactions and len(compact_entries) == len(compactions)  # one ledger line per compaction, no more
    assert len(entries) == len(_model_events(s)) + len(compactions)
    assert sum(e["input_tokens"] for e in entries) == sum(u.input_tokens for u in s.model_usage.values())
    assert {e["sample_uuid"] for e in entries} == {s.uuid}


def test_native_calibration_is_not_inflated_by_the_current_cases_reasoning(env, tmp_path, monkeypatch):
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(tmp_path / "native.json"))
    runs, order = {}, {}
    for model in ("fakenative/model", "fakenative_reasoning/model"):
        record_native_support(model, True, evidence="offline test provider")
        s = _run(env, "CM-native", gold_session_agent(load_world(W)), model=model, threshold=6000)
        ev = s.store["f8_cm_events"]
        runs[model] = ([e["item"] for e in ev if e["event"] == "native"], [e["opaque_tokens"] for e in ev if e["event"] == "native_calibrated"])
        order[model] = [e["event"] for e in ev if e["event"] in ("native", "native_calibrated")]
        assert s.scores["f8_session_score"].value["item_success"] == 1.0
    assert runs["fakenative/model"] == runs["fakenative_reasoning/model"] and runs["fakenative/model"][1]
    # No compaction while the last block still waits for its calibration: they alternate.
    seq = order["fakenative/model"]
    assert seq[::2] == ["native"] * len(seq[::2]) and seq[1::2] == ["native_calibrated"] * len(seq[1::2])


# --- CM-trim --------------------------------------------------------------------------------------------------------


def test_a_trimmed_window_starts_at_a_case_message():
    history = [ChatMessageSystem(content="sys"), ChatMessageUser(content="Shift start.")]
    for i in range(5):
        history += [
            ChatMessageUser(content=f"Case C-00000{i}: x"),
            ChatMessageAssistant(content="", tool_calls=[ToolCall(id=f"c{i}", function="lookup_customer", arguments={"customer_id": f"CU-{i}"})]),
            ChatMessageTool(content="file " * 20, tool_call_id=f"c{i}", function="lookup_customer"),
        ]
    ctx = SessionContext(world=None, window=10**6, threshold=1, agent_model=get_model(M), cm_model=get_model(M), records=SessionRecords())
    policy = CMTrim(ctx, trim_preserve=0.7)  # 15 messages: trim_messages keeps the last 11, which starts mid-case
    out = asyncio.run(policy._compact(history, len(history)))
    assert out[0].role == "user" and out[0].text == "Case C-000002: x" and len(out) == 9
    assert context_policy.POLICIES["CM-trim"] is CMTrim
