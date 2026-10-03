"""The ContextPolicy layer and the session harness (BUILD_PLAN B7): CM0 and O-state unchanged, every policy hook,
W and T_abs, call kinds and usage, probes, and the F8 task's knobs, seed block and world selection.

Mid-session checkpoint/resume has its own module (`test_session_resume`). Offline: mockllm, fake embeddings."""

import asyncio
import json
from typing import ClassVar

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ChatMessageUser, ModelOutput, get_model
from inspect_ai.tool import ToolDef, ToolParam, ToolParams
from session_trace import CASES, GOLDEN, trace_session

from ape.agent import context_policy
from ape.agent.arms import load_world
from ape.agent.context_policy import ContextPolicy, SessionContext, resolve_knobs, view_tokens
from ape.agent.session import f8_session_agent, plan_threshold
from ape.build import build
from ape.llm.mock_session import MODEL, mock_session_agent
from ape.tasks.study_g import f8_session
from ape.worlds import gen_f8


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f8-policy")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv("APE_SESSION_CHECKPOINTS", raising=False)
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


# --- CM0 and O-state are unchanged ------------------------------------------------------------------------------


@pytest.mark.parametrize("name,nudges,kwargs", CASES, ids=[c[0] for c in CASES])
def test_cm0_and_o_state_views_and_records_are_unchanged(env, name, nudges, kwargs):
    """Every model input (agent calls and probes) and every pre-B7 record field equal the golden trace written by the
    loop before the ContextPolicy layer (tests/session_trace.py)."""
    golden = json.loads(GOLDEN.read_text())[name]
    trace = trace_session(env / "logs", nudges, **kwargs)
    assert trace["calls"] == golden["calls"]  # the views, call by call
    for key in ("items", "views", "probes", "events", "report", "overflow_at", "arm", "history", "score"):
        assert trace[key] == golden[key], key


def test_cm0_and_o_state_record_kinds_and_usage(env):
    for arm in ("CM0", "O-state"):
        log = inspect_eval(f8_session(level="12", split="dev", arm=arm, checkpoints="5,10"), model=get_model("mockllm/model", custom_outputs=mock_session_agent), log_dir=str(env / "logs"), display="none")[0]
        st = log.samples[0].store
        assert {v["kind"] for v in st["f8_views"]} == {"agent"} and all(v["usage"]["input_tokens"] > 0 for v in st["f8_views"])
        assert {p["kind"] for p in st["f8_probes"]} == {"probe"} and [p["item"] for p in st["f8_probes"]] == [5, 10]
        assert all(i["cm_calls"] == 0 and set(i["usage"]) <= {"agent", "probe"} for i in st["f8_items"])
        assert set(st["f8_usage"]["by_kind"]) == {"agent", "probe"} and st["f8_cm_events"] == [] and st["f8_resume"] is None
        assert st["arm"]["policy"] in ("FullHistory", "OracleState") and st["arm"]["threshold"] == plan_threshold() == 20_000
        assert st["arm"]["policy_tools"] == [] and st["arm"]["knobs"] == {}
        # Per-item usage adds up to the session's (report calls are outside every item).
        total = sum(i["usage"]["agent"]["input_tokens"] for i in st["f8_items"])
        report = sum(v["usage"]["input_tokens"] for v in st["f8_views"] if v["item"] == 13)
        assert total + report == st["f8_usage"]["by_kind"]["agent"]["input_tokens"]


# --- A managed test policy exercising every hook --------------------------------------------------------------------


class Ledger(ContextPolicy):
    """A test policy with every hook: keeps the system and start messages plus the last `keep_items` cases (state:
    the cases' start indices), writes a cm "summary" every `every` items, replaces the view by system + start +
    summary + current case when it crosses T_abs, offers a `note` tool whose notes it keeps, appends an addendum,
    counts its calls, and checkpoints all of it."""

    name = "ledger"
    KNOBS: ClassVar[dict] = {"keep_items": 2, "every": 3}

    def __init__(self, session, **knobs):
        super().__init__(session, **knobs)
        self.starts: list[int] = []
        self.summaries: list[str] = []
        self.notes: list[str] = []
        self.generations = 0
        self.started = False

    def tools(self):
        async def note(text: str) -> str:
            self.notes.append(text)
            return f"Noted ({len(self.notes)})."

        return [ToolDef(note, name="note", description="Write a note to your ledger.", parameters=ToolParams(properties={"text": ToolParam(type="string", description="The note.")}, required=["text"]))]

    def system_addendum(self):
        return "Ledger: write a note after each case."

    async def start(self, history):
        self.started = True
        out = await self.session.cm_generate([self.session.system, ChatMessageUser(content="Summarise the shift start.")], purpose="start")
        self.summaries.append(out.completion)

    def _keep(self, history, item_start):
        if not self.starts or self.starts[-1] != item_start:
            self.starts.append(item_start)
        cut = self.starts[-self.knobs["keep_items"]] if len(self.starts) >= self.knobs["keep_items"] else item_start
        return [*history[:2], *history[cut:]]

    async def view(self, history, item_start, done):
        return self._keep(history, item_start)

    async def on_threshold(self, history, view, tokens):
        self.session.log("compact", summaries=len(self.summaries))
        return [*history[:2], ChatMessageUser(content=f"Summary: {self.summaries[-1]}"), *history[self.starts[-1] :]]

    async def after_generate(self, history, view, output, appended):
        self.generations += 1

    async def after_item(self, history, done):
        if done % self.knobs["every"] == 0:
            out = await self.session.cm_generate([*history[:2], ChatMessageUser(content=f"Summarise cases 1-{done}.")], purpose="summary")
            self.summaries.append(out.completion)
            self.session.log("summary", done=done)

    async def probe_view(self, history, done):
        return [*history[:2], *history[self.starts[-1] :]] if self.starts else list(history)

    def state_dict(self):
        return {"starts": self.starts, "summaries": self.summaries, "notes": self.notes, "generations": self.generations, "started": self.started}

    def load_state(self, state, history):
        self.starts, self.summaries, self.notes = list(state["starts"]), list(state["summaries"]), list(state["notes"])
        self.generations, self.started = state["generations"], state["started"]


def ledger_agent(messages, tools, tool_choice, config):
    """mock_session_agent, plus a note after each lookup and a canned answer to management prompts."""
    last = messages[-1]
    if not tools and config.response_schema is None:  # a management call (no tools, no probe schema)
        return ModelOutput.from_content(MODEL, f"summary of {len(messages)} messages")
    if isinstance(last, ChatMessageTool) and last.function in ("lookup_customer", "order_lookup") and not last.error:
        return ModelOutput.for_tool_call(MODEL, "note", {"text": f"looked up {len(last.text)} chars"})
    return mock_session_agent(messages, tools, tool_choice, config)


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setitem(context_policy.POLICIES, "CM-ledger", Ledger)
    return "CM-ledger"


def _run(env, model, **kw):
    log = inspect_eval(f8_session(level="12", split="dev", checkpoints="5,10", **kw), model=model, log_dir=str(env / "logs"), display="none")[0]
    assert log.status == "success", log.error
    return log.samples[0]


def test_a_managed_policy_exercises_every_hook(env, ledger, monkeypatch):
    monkeypatch.setenv("APE_CM_EVERY", "4")  # a knob from the environment
    s = _run(env, get_model("mockllm/model", custom_outputs=ledger_agent), arm=ledger, threshold=3000)
    st = s.store
    arm = st["arm"]
    assert arm["policy"] == "Ledger" and arm["knobs"] == {"keep_items": 2, "every": 4} and arm["threshold"] == 3000
    assert arm["policy_tools"] == ["note"] and arm["policy_tool_tokens"] > 0
    # The addendum is part of the shared system prompt; the history stays complete and append-only.
    assert s.messages[0].text.endswith("Ledger: write a note after each case.")
    texts = [m.text for m in s.messages]
    assert all(t.prompt in texts for t in load_world("F8-12-dev-s1000").tasks) and gen_f8.REPORT_REQUEST in texts
    # Management calls: on the cm role, metered separately, attributed to the start (item 0) and the boundaries.
    cm = [v for v in st["f8_views"] if v["kind"] == "cm"]
    assert [(v["item"], v["purpose"]) for v in cm] == [(0, "start"), (4, "summary"), (8, "summary"), (12, "summary")]
    assert all(v["usage"]["output_tokens"] > 0 for v in cm)
    assert s.role_usage["cm"].input_tokens == sum(v["usage"]["input_tokens"] for v in cm)
    assert st["f8_usage"]["by_kind"]["cm"]["input_tokens"] == s.role_usage["cm"].input_tokens
    assert {i["position"]: i["cm_calls"] for i in st["f8_items"] if i["cm_calls"]} == {4: 1, 8: 1, 12: 1}
    assert st["f8_items"][3]["usage"]["cm"] == cm[1]["usage"]
    # The policy's tool calls are tool events flagged as policy traffic; scoring ignores them.
    notes = [e for e in st["f8_events"] if e["tool"] == "note"]
    lookups = [e for e in st["f8_events"] if e["tool"] in ("lookup_customer", "order_lookup")]
    assert notes and all(e.get("policy") for e in notes) and [e["item"] for e in notes] == [e["item"] for e in lookups]
    assert not any(e.get("policy") for e in lookups)
    # Thresholds: each view over T_abs went to on_threshold, and its replacement is what the model got (logged).
    events = st["f8_cm_events"]
    shrunk = [e for e in events if e["event"] == "threshold"]
    sent = {(v["item"], v["view_tokens"]) for v in st["f8_views"] if v["kind"] == "agent"}
    assert shrunk and all(e["view_tokens"] > 3000 and (e["item"], e["view_tokens_after"]) in sent for e in shrunk)
    assert len(shrunk) == sum(1 for e in events if e["event"] == "compact")
    assert any(e["event"] == "compact" for e in events) and [e["done"] for e in events if e["event"] == "summary"] == [4, 8, 12]
    # The trimmed view is smaller than CM0's.
    full = _run(env, get_model("mockllm/model", custom_outputs=mock_session_agent), arm="CM0")
    agent_views = lambda store: [v["view_tokens"] for v in store["f8_views"] if v["kind"] == "agent"]  # noqa: E731
    assert max(agent_views(st)) < max(agent_views(full.store))
    # Probes: from the policy's probe view, never appended.
    assert [p["k"] for p in st["f8_probes"]] == [5, 10] and not any("state check" in m.text for m in s.messages)


class Hoarder(ContextPolicy):
    """Shows the model only the current case, but summarises the whole history after case 4: a management input
    over W while every agent view stays under it."""

    name = "hoarder"

    async def view(self, history, item_start, done):
        return [*history[:2], *history[item_start:]]

    async def after_item(self, history, done):
        if done == 4:
            await self.session.cm_generate([*history, ChatMessageUser(content="Summarise the shift so far.")], purpose="summary")


def test_a_management_call_over_the_window_is_an_overflow(env, monkeypatch):
    """W applies to management calls too: the boundary summary after item 4 is over W, so items 5.. fail (item 4
    itself is complete)."""
    monkeypatch.setitem(context_policy.POLICIES, "CM-hoarder", Hoarder)
    agent = get_model("mockllm/model", custom_outputs=ledger_agent)
    wide = _run(env, agent, arm="CM-hoarder", window=10**6)
    over = next(v["view_tokens"] for v in wide.store["f8_views"] if v["kind"] == "cm")
    agent_max = max(v["view_tokens"] for v in wide.store["f8_views"] if v["kind"] == "agent")
    assert agent_max < over, "the test needs the summary input to be the only input over W"
    s = _run(env, agent, arm="CM-hoarder", window=over - 1)
    assert s.store["f8_overflow_at"] == 5 and [i["position"] for i in s.store["f8_items"]] == [1, 2, 3, 4]
    assert not any(v["kind"] == "cm" for v in s.store["f8_views"])
    last = s.store["f8_items"][-1]  # its boundary ended the session: its record is still complete
    assert last["cm_calls"] == 0 and set(last["usage"]) == {"agent"}
    v = s.scores["f8_session_score"].value
    assert v["overflow"] == 1.0 and s.scores["f8_session_score"].metadata["items"][3]["overflow"] is False


class Meddler(ContextPolicy):
    """Changes its state when probed: a bug the loop must refuse."""

    name = "meddler"

    def __init__(self, session, **knobs):
        super().__init__(session, **knobs)
        self.probed = 0

    async def probe_view(self, history, done):
        self.probed += 1
        return list(history)

    def state_dict(self):
        return {"probed": self.probed}


def test_a_probe_view_must_not_change_the_policy(env, monkeypatch):
    monkeypatch.setitem(context_policy.POLICIES, "CM-meddler", Meddler)
    log = inspect_eval(f8_session(level="12", split="dev", arm="CM-meddler", checkpoints="5"), model=get_model("mockllm/model", custom_outputs=mock_session_agent), log_dir=str(env / "logs"), display="none")[0]
    assert "probe_view changed the policy's state" in log.samples[0].error.message


def test_policy_registry_and_knobs(monkeypatch):
    with pytest.raises(ValueError, match="not built yet"):
        f8_session_agent(arm="CM-sum")
    with pytest.raises(TypeError):
        context_policy.register_policy("CM-x", object)
    monkeypatch.setenv("APE_CM_KEEP_ITEMS", "5")
    assert resolve_knobs(Ledger) == {"keep_items": 5, "every": 3}
    monkeypatch.setenv("APE_CM_EVERY", "x")
    with pytest.raises(ValueError):
        resolve_knobs(Ledger)
    with pytest.raises(ValueError, match="unknown knobs"):
        Ledger(SessionContext(world=None, window=1, threshold=1, agent_model=None, cm_model=None, records=None), stride=2)


def test_the_naive_mock_answers_management_calls():
    from inspect_ai.model import GenerateConfig

    out = mock_session_agent([ChatMessageUser(content="Summarise the shift so far.")], [], "none", GenerateConfig())
    assert out.completion.startswith("Shift notes:") and not out.message.tool_calls


def test_view_tokens_count_texts_and_tool_calls():
    assert view_tokens([ChatMessageUser(content="hello world")]) == 2
    assert view_tokens([]) == 0

