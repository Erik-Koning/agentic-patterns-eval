"""Study G's context-management arms (BUILD_PLAN B8) through the real session loop, offline (mockllm).

- Under the gold-knowing session mock every arm gives a perfect session, also where management fires (a low T_abs
  on 12 cases, and the plan's T_abs and W on 40 cases, where CM0 overflows).
- Management calls are metered separately (kind cm) and add up; probes are never appended.
- Each arm resumes exactly after a crash that follows its first management action.
- A misbehaving model (empty or garbled notes, unparseable todo lists, malformed todo_write calls, failing
  compaction) never crashes a session.
- CM-native refuses a provider model without a support record and runs one with it (a test provider with a real
  `compact`); its opaque block counts as what it replaced until the next call's provider input calibrates it.
- S-CM* stacks and every APE_CM_* knob are validated when the task is created.
"""

import asyncio
import json
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ChatMessageUser, ContentData, GenerateConfig, ModelOutput, ModelUsage, get_model, modelapi
from inspect_ai.model._providers.mockllm import MockLLM
from test_session_resume import Flaky

from ape.agent import cm_prompts
from ape.agent.arms import load_world
from ape.agent.cm_arms import (
    CM_ARMS,
    NATIVE_RECORD_ENV,
    NativeCompactionUnsupported,
    native_route,
    parse_stack,
    parse_todo_reply,
    parse_todos,
    record_native_support,
    valid_stacks,
)
from ape.agent.session import PROBE_PROMPT, SESSION_ARMS
from ape.agent.session_checkpoint import ENV as CHECKPOINTS_ENV
from ape.build import build
from ape.llm.mock_session import gold_session_agent, mock_session_agent
from ape.tasks.study_g import f8_session

M = "mockllm/model"
PERFECT = {"item_success": 1.0, "dependency_success": 1.0, "report_exact": 1.0, "session_success": 1.0, "probe_f1": 1.0, "probe_coverage": 1.0, "overflow": 0.0}
LOW_T = 6000  # on 12 cases every arm's management fires (the system prompt alone is ~2.5K tokens)
# What each arm's management leaves in f8_cm_events once it has fired.
FIRES = {
    "CM-prune": {"prune"},
    "CM-trim": {"trim"},
    "CM-sum": {"summary", "sum_drop"},
    "CM-todo": {"todo_extract", "todo_drop"},
    "CM-reset": {"todo_extract", "handoff", "reset_drop"},
    "CM-native": {"native", "native_calibrated"},
    "S-CM*": {"todo_extract", "handoff", "reset_drop"},
}
RECORD_KEYS = ("f8_items", "f8_views", "f8_probes", "f8_events", "f8_report", "f8_overflow_at", "f8_cm_events", "f8_usage")


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """One 12-case and one 40-case dev session (seed 1000)."""
    tmp = tmp_path_factory.mktemp("cm-arms")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for k in (CHECKPOINTS_ENV, NATIVE_RECORD_ENV, "APE_CM_STACK"):
        mp.delenv(k, raising=False)
    asyncio.run(build("dev", "F8", ["12", "40"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()




def _run(env: Path, arm: str, agent, level: str = "12", model: str = M, retries: int = 0, **kw):
    log = inspect_eval(
        f8_session(level=level, split="dev", arm=arm, **kw),
        model=get_model(model, custom_outputs=agent),
        log_dir=str(env / "logs"),
        display="none",
        retry_on_error=retries,
        fail_on_error=False,
    )[0]
    assert log.status == "success", log.error
    return log.samples[0]


def _events(sample) -> set[str]:
    return {e["event"] for e in sample.store["f8_cm_events"]}


def _gold(level: str = "12"):
    return gold_session_agent(load_world(f"F8-{level}-dev-s1000"))


def test_every_cm_arm_is_registered():
    assert set(CM_ARMS) == {"CM-prune", "CM-trim", "CM-sum", "CM-todo", "CM-reset", "CM-native", "S-CM*"}
    assert SESSION_ARMS == ("CM0", "O-state", *CM_ARMS)


@pytest.mark.parametrize("arm", list(CM_ARMS))
def test_every_arm_gives_a_perfect_session_while_its_management_fires(worlds, arm):
    s = _run(worlds, arm, _gold(), threshold=LOW_T, checkpoints="5,10")
    st = s.store
    assert s.error is None and s.scores["f8_session_score"].value == PERFECT
    assert FIRES[arm] <= _events(s), _events(s)
    # The history is complete and append-only; probes are never in it.
    texts = [m.text for m in s.messages]
    assert all(t.prompt in texts for t in load_world("F8-12-dev-s1000").tasks) and not any(PROBE_PROMPT in t for t in texts)
    assert max(v["view_tokens"] for v in st["f8_views"]) <= st["arm"]["window"]
    # Management calls are metered on their own and add up: per call, per item (+ the start and report phases), per session.
    cm = [v for v in st["f8_views"] if v["kind"] == "cm"]
    total = st["f8_usage"]["by_kind"].get("cm")
    assert (total is None) == (not cm)
    if cm:
        assert total["input_tokens"] == sum(v["usage"]["input_tokens"] for v in cm)
        per_item = sum(i["usage"].get("cm", {}).get("input_tokens", 0) for i in st["f8_items"])
        outside = sum(v["usage"]["input_tokens"] for v in cm if v["item"] in (0, 13))
        assert per_item + outside == total["input_tokens"]
        assert sum(i["cm_calls"] for i in st["f8_items"]) + sum(v["item"] in (0, 13) for v in cm) == len(cm)
        if arm != "CM-native":  # cm_generate runs on the cm role; native compaction is the agent model's own call
            assert s.role_usage["cm"].input_tokens == total["input_tokens"]
    assert s.scores["f8_session_score"].metadata["usage"].keys() >= {"agent", "probe"}


def test_at_plan_scale_cm0_overflows(worlds):
    s = _run(worlds, "CM0", _gold("40"), level="40")
    assert s.store["f8_overflow_at"] is not None and s.scores["f8_session_score"].value["session_success"] == 0.0


@pytest.mark.parametrize("arm", list(CM_ARMS))
def test_at_plan_scale_every_arm_stays_perfect(worlds, arm):
    """40 cases at the plan's T_abs (20K) and W (32K): CM0 overflows near case 27, every managed arm completes."""
    s = _run(worlds, arm, _gold("40"), level="40")
    assert s.scores["f8_session_score"].value == PERFECT
    assert FIRES[arm] & _events(s)
    assert s.store["arm"]["threshold"] == 20_000 and max(v["view_tokens"] for v in s.store["f8_views"]) <= 32_000


@pytest.mark.parametrize("arm", list(CM_ARMS))
def test_each_arm_resumes_exactly_after_its_management_fired(worlds, arm, tmp_path, monkeypatch):
    k = 9
    monkeypatch.setenv(CHECKPOINTS_ENV, str(tmp_path / "ckpt"))
    clean = _run(worlds, arm, _gold(), threshold=LOW_T, checkpoints="5,10")
    assert any(e["item"] < k for e in clean.store["f8_cm_events"]), "management fires before the crash"
    s = _run(worlds, arm, Flaky(_gold(), k=k), retries=2, threshold=LOW_T, checkpoints="5,10")
    assert s.error is None and len(s.error_retries) == 1 and s.store["f8_resume"]["after_items"] == [k - 1]
    assert {key: s.store[key] for key in RECORD_KEYS} == {key: clean.store[key] for key in RECORD_KEYS}
    assert [m.text for m in s.messages] == [m.text for m in clean.messages]
    assert s.scores["f8_session_score"].value == PERFECT
    assert not list((tmp_path / "ckpt").rglob("*.json"))


class Misbehaving:
    """The gold agent, but its management answers are bad (empty, prose, wrong JSON shapes, failing compaction) and
    its first todo_write call of each case is malformed (an empty entry, then a wrong type)."""

    BAD_NOTES = ("", "  ", "I cannot help with that.", '[{"content": 5}]', '{"todos": "x"}', "[not json")

    def __init__(self, agent):
        self.agent, self.n = agent, 0

    def __call__(self, messages, tools, tool_choice, config):
        if not tools and config.response_schema is None:
            self.n += 1
            if cm_prompts.purpose_of(messages[-1].text) == "native_mock":
                raise RuntimeError("compaction endpoint failed (test)")
            return ModelOutput.from_content(M, self.BAD_NOTES[self.n % len(self.BAD_NOTES)])
        out = self.agent(messages, tools, tool_choice, config)
        calls = out.message.tool_calls or []
        if calls and calls[0].function == "todo_write":
            case_start = max(i for i, m in enumerate(messages) if m.role == "user" and m.text.startswith(("Case ", "Ticket ", "MEMO")))
            if not any(isinstance(m, ChatMessageTool) and m.function == "todo_write" for m in messages[case_start:]):
                self.n += 1
                bad = [{"content": "", "status": "pending"}] if self.n % 2 else "oops"
                return ModelOutput.for_tool_call(M, "todo_write", {"todos": bad})
        return out


@pytest.mark.parametrize("arm", list(CM_ARMS))
def test_a_misbehaving_model_never_crashes_a_session(worlds, arm):
    s = _run(worlds, arm, Misbehaving(_gold()), threshold=LOW_T, checkpoints="5,10")
    assert s.error is None
    v = s.scores["f8_session_score"].value
    assert v["item_success"] == 1.0 and v["report_exact"] == 1.0  # the agent itself is still the gold one
    events = _events(s)
    if arm in ("CM-sum", "CM-reset", "S-CM*"):
        assert "empty_notes" in events or arm != "CM-sum"
    if arm in ("CM-todo", "CM-reset", "S-CM*"):
        sources = {e.get("source") for e in s.store["f8_cm_events"] if e["event"] == "todo_unparsed"}
        assert sources == {"extraction", "todo_write"}, sources
    if arm == "CM-native":
        assert "native_error" in events and "native_disabled" in events and "native" not in events


# --- CM-native: the support gate and the opaque block --------------------------------------------------------------


@modelapi(name="fakenative")
class FakeNative(MockLLM):
    """A provider with a real `compact` (offline): the compacted context is one opaque ContentData block."""

    async def compact(self, input, tools, config, instructions=None):
        return [ChatMessageUser(content=[ContentData(data={"compaction": f"{len(input)} messages"})])], ModelUsage(input_tokens=1000, output_tokens=50, total_tokens=1050)


def test_native_compaction_is_refused_without_a_support_record(worlds, tmp_path, monkeypatch):
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(tmp_path / "native.json"))
    assert native_route(M) == "mock"
    with pytest.raises(NativeCompactionUnsupported, match="Run the B11 probe"):
        native_route("openai/gpt-6-luna")
    s = _run(worlds, "CM-native", _gold(), model="fakenative/model", threshold=LOW_T)
    assert s.error is not None and "needs confirmed native compaction for fakenative/model" in s.error.message
    record_native_support("fakenative/model", False, evidence="compact returned 404 (test)")
    with pytest.raises(NativeCompactionUnsupported):
        native_route("fakenative/model")


def test_native_compaction_runs_on_a_confirmed_provider_with_the_conservative_w_rule(worlds, tmp_path, monkeypatch):
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(tmp_path / "native.json"))
    path = record_native_support("fakenative/model", True, evidence="compact + generate accepted (test)", api="test")
    assert json.loads(path.read_text())["models"]["fakenative/model"]["supported"] is True and native_route("fakenative/model") == "provider"
    s = _run(worlds, "CM-native", _gold(), model="fakenative/model", threshold=LOW_T)
    assert s.error is None and s.scores["f8_session_score"].value == PERFECT
    st = s.store
    native = [v for v in st["f8_views"] if v["kind"] == "cm"]
    assert native and all(v["purpose"] == "native" and v["usage"] == {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050} for v in native)
    events = st["f8_cm_events"]
    for e in (e for e in events if e["event"] == "native"):
        # No credit before calibration: the view right after a compaction counts exactly what it counted before.
        (shrunk,) = [x for x in events if x["event"] == "threshold" and x["item"] == e["item"]]
        assert shrunk["view_tokens_after"] == shrunk["view_tokens"] == e["view_tokens"]
        assert e["visible_tokens"] == 0  # the block's text is empty: opaque to the meter
    calibrated = [e for e in events if e["event"] == "native_calibrated"]
    assert calibrated and all(e["opaque_tokens"] < e["opaque_tokens_before"] for e in calibrated)
    # After calibration the views are smaller than CM0's.
    full = _run(worlds, "CM0", _gold(), model="fakenative/model")
    assert max(v["view_tokens"] for v in st["f8_views"] if v["kind"] == "agent" and v["item"] > 8) < max(v["view_tokens"] for v in full.store["f8_views"] if v["item"] > 8)


# --- Stacks, knobs, parsing, prompts ---------------------------------------------------------------------------------


def test_stacks():
    assert parse_stack("reset+todo+prune") == ("prune", "todo", "reset") and parse_stack(" Sum ") == ("sum",)
    assert len(valid_stacks()) == 17 and "prune+todo+reset" in valid_stacks()
    for bad in ("", "prune+trim", "sum+reset", "todo+todo", "native", "prune+lesson"):
        with pytest.raises(ValueError):
            parse_stack(bad)


def test_s_cm_star_takes_its_stack_and_knobs_from_the_environment(worlds, monkeypatch):
    monkeypatch.setenv("APE_CM_STACK", "trim+sum")
    monkeypatch.setenv("APE_CM_TRIM_PRESERVE", "0.9")  # too gentle to get under T_abs alone: the summary follows
    monkeypatch.setenv("APE_CM_SUM_PROMPT", "plain")
    s = _run(worlds, "S-CM*", _gold(), threshold=LOW_T)
    assert s.scores["f8_session_score"].value == PERFECT
    knobs = s.store["arm"]["knobs"]
    assert (knobs["stack"], knobs["trim_preserve"], knobs["sum_prompt"]) == ("trim+sum", 0.9, "plain")
    assert {"trim", "summary"} <= _events(s) and not {"todo_extract", "handoff"} & _events(s)
    assert s.store["arm"]["policy_tools"] == []


@pytest.mark.parametrize(
    "arm,env",
    [("S-CM*", {"APE_CM_STACK": "prune+trim"}), ("CM-sum", {"APE_CM_SUM_PROMPT": "fancy"}), ("CM-trim", {"APE_CM_TRIM_PRESERVE": "1.5"}), ("CM-prune", {"APE_CM_PRUNE_KEEP": "-1"}), ("CM-reset", {"APE_CM_RESET_SUMMARY": "maybe"})],
)
def test_bad_knobs_are_refused_when_the_task_is_created(worlds, monkeypatch, arm, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValueError):
        f8_session(level="12", split="dev", arm=arm)


def test_knobs_reach_the_arms(worlds, monkeypatch):
    monkeypatch.setenv("APE_CM_RESET_EVERY", "2")
    monkeypatch.setenv("APE_CM_RESET_SUMMARY", "false")
    monkeypatch.setenv("APE_CM_TODO_EXTRACT", "0")
    s = _run(worlds, "CM-reset", _gold(), threshold=10**6)  # resets only on the item count
    assert s.store["arm"]["knobs"] == {"todo_extract": False, "reset_every": 2, "reset_summary": False}
    drops = [e for e in s.store["f8_cm_events"] if e["event"] == "reset_drop"]
    assert [e["item"] for e in drops] == [2, 4, 6, 8, 10, 12] and {e["trigger"] for e in drops} == {"every 2 items"}
    assert not [v for v in s.store["f8_views"] if v["kind"] == "cm"]  # no handoff notes, no extraction
    assert s.scores["f8_session_score"].value == PERFECT


def test_todo_parsing_is_tolerant():
    good = [{"content": "Case C-1", "status": "completed"}, {"content": "Memo M-2", "status": "in_progress"}]
    assert parse_todos(good) == good and parse_todos(json.dumps(good)) == good and parse_todos({"todos": good}) == good
    assert parse_todos([{"content": " x ", "status": "done"}]) == [{"content": "x", "status": "pending"}]
    assert parse_todos([]) == [] and parse_todos("oops") is None and parse_todos([{"content": ""}]) is None and parse_todos(5) is None
    assert len(parse_todos([{"content": "a" * 900, "status": "pending"}] * 400)) == 300
    assert parse_todo_reply("Here is the list:\n" + json.dumps(good) + "\nDone.") == good
    assert parse_todo_reply("no list here") is None


def test_management_prompts_are_told_apart_and_transcripts_are_plain_text():
    for purpose, prompt in (("summary", cm_prompts.SUMMARY_PROMPTS["structured"]), ("summary", cm_prompts.SUMMARY_PROMPTS["plain"]), ("handoff", cm_prompts.HANDOFF_PROMPT), ("todo_extract", cm_prompts.TODO_EXTRACT_PROMPT), ("native_mock", cm_prompts.NATIVE_MOCK_PROMPT)):
        assert cm_prompts.purpose_of(cm_prompts.management_request(prompt, [("Transcript", "x")])) == purpose
    assert cm_prompts.purpose_of(PROBE_PROMPT) is None
    from inspect_ai.model import ChatMessageAssistant
    from inspect_ai.tool import ToolCall

    text = cm_prompts.render_transcript([
        ChatMessageUser(content="Case C-1: hello"),
        ChatMessageAssistant(content="", tool_calls=[ToolCall(id="a", function="lookup_customer", arguments={"customer_id": "CU-1"})]),
        ChatMessageTool(content="file", tool_call_id="a", function="lookup_customer"),
    ])
    assert text == 'USER: Case C-1: hello\nASSISTANT called lookup_customer({"customer_id": "CU-1"})\nTOOL lookup_customer result: file'


def test_the_naive_mock_plays_the_management_roles():
    req = cm_prompts.management_request(cm_prompts.TODO_EXTRACT_PROMPT, [("Current todo list", "[]"), ("Message", "Shift start. Your queue has 2 cases, in this order: C-000001, C-000002.")])
    todos = json.loads(mock_session_agent([ChatMessageUser(content=req)], [], "none", GenerateConfig()).completion)
    assert [t["content"] for t in todos] == ["Case C-000001", "Case C-000002"]
    memo = cm_prompts.management_request(cm_prompts.TODO_EXTRACT_PROMPT, [("Current todo list", json.dumps(todos)), ("Message", "MEMO M-1 (in force ...): x\nCase C-000001: ...")])
    assert [t["content"] for t in json.loads(mock_session_agent([ChatMessageUser(content=memo)], [], "none", GenerateConfig()).completion)][-1] == "Memo M-1"
    note = cm_prompts.management_request(cm_prompts.HANDOFF_PROMPT, [("Transcript", "TOOL submit_decision result: Decision recorded for C-000001.")])
    assert "C-000001" in mock_session_agent([ChatMessageUser(content=note)], [], "none", GenerateConfig()).completion
