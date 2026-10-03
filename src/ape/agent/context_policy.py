"""Context policies for Study G's F8 sessions (CONTEXT_MANAGEMENT_AUDIT §6.1): what the model sees at each call.

The session loop (`ape.agent.session`) keeps the **full history** append-only in `state.messages`: it is what the
log shows, and the tool events the scorer reads come from it. At every call the loop asks the arm's policy for the
**view**, the messages the model is given. All arms share the loop, the base system prompt, the tools, the turn
caps, the probes and the nominal window W; a policy adds only its mechanism, through these hooks (each optional;
the base class is CM0, the full history):

- `tools()` / `system_addendum()`: extra tools (e.g. Inspect's `todo_write()`, `memory()`) and a short, fixed
  instruction appended to the shared system prompt. The loop records every call of a policy tool as a tool event
  flagged `policy: True` (the scorers skip them; external-state analysis reads them).
- `start(history)`: once, when a fresh session starts (a resumed one restores its state instead).
- `view(history, item_start, done)`: before every agent call, the model's input. May keep state and make
  management calls.
- `on_threshold(history, view, tokens)`: the view about to be sent exceeds T_abs (run_plan.yaml
  `study_g.threshold`, on the harness's token meter): a chance to return a smaller one (compaction, reset). It is
  level-triggered: called before every call whose view is over T_abs, never for one under it.
- `after_generate(history, view, output, appended)`: after every agent call and its tool results (calibrating a
  compaction handler, reading `todo_write` calls, reacting to an environment error).
- `after_item(history, done)`: at each item boundary, after the item's record and before its probe and checkpoint.
- `probe_view(history, done)`: what the model would see next, for the forked state probes (audit §5.3). It must
  neither change the policy's state nor make calls: the loop compares `state_dict()` before and after.
- `state_dict()` / `load_state(state, history)`: the policy's management state as JSON, saved with every
  mid-session checkpoint and restored on resume (`ape.agent.session_checkpoint`). State that lives elsewhere (e.g.
  Inspect's `memory()` files in the sample store) must be carried here too, or it is lost on resume.

Management model calls go through `SessionContext.cm_generate` on the `cm` role (the agent's own model when the run
defines none: self-summarisation, audit §6.3). The harness W-checks each like an agent call and records it per
call with kind `cm` and its usage, so management cost separates from agent cost (kind `agent`) and probe cost
(`probe`). Management events (a compaction, a reset, a summary text) go to `SessionContext.log`.

Knobs. A policy declares its tunable knobs with defaults in `KNOBS`; the environment overrides them as
`APE_CM_<KNOB>` (e.g. `APE_CM_KEEP_ITEMS=4`), like the gate's arm knobs, so a tuning candidate is an env group.
The names share one namespace across arms, so B8 prefixes them per mechanism where they must differ. The resolved
knobs are recorded with the arm and are part of the checkpoint key.

Inspect's `compaction()` handlers keep their state in a closure that only a `Checkpointer` can reach;
`TrackedState` is a minimal one, so a policy built on them checkpoints and resumes like any other.

Built here (B7): CM0 (`FullHistory`) and O-state (`OracleState`). B8 adds the managed arms with `register_policy`.
"""

import dataclasses
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig, Model, ModelOutput, ModelUsage
from inspect_ai.tool import Tool, ToolDef
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from ..llm.tokens import count_tokens
from ..worlds import gen_f8
from ..worlds.spec import World

KNOB_ENV_PREFIX = "APE_CM_"
KINDS = ("agent", "cm", "probe")  # call kinds: the arm's agent, its management calls, the forked probes


# --- Token meter, messages and usage ----------------------------------------------------------------------------


def message_tokens(m: ChatMessage) -> int:
    n = count_tokens(m.text or "")
    if isinstance(m, ChatMessageAssistant) and m.tool_calls:
        n += sum(count_tokens(json.dumps({"function": c.function, "arguments": c.arguments})) for c in m.tool_calls)
    return n


def view_tokens(messages: Sequence[ChatMessage]) -> int:
    """Tokens of a view (o200k_base over message texts and tool calls; tool schemas excluded, they are constant)."""
    return sum(message_tokens(m) for m in messages)


_MESSAGES = TypeAdapter(list[ChatMessage])


def dump_messages(messages: Sequence[ChatMessage]) -> list[dict]:
    """Messages as JSON (IDs kept: a restored history keeps the IDs a policy's state may refer to)."""
    return [m.model_dump(mode="json") for m in messages]


def load_messages(data: list[dict]) -> list[ChatMessage]:
    return _MESSAGES.validate_python(data)


def usage_record(u: ModelUsage | None) -> dict | None:
    """One call's Inspect usage as a JSON dict (ModelUsage fields; unset ones left out). `total_cost` is there when
    the run has prices (ape.models.eval_cost_kwargs)."""
    return None if u is None else {k: v for k, v in u.model_dump().items() if v is not None}


def add_usage(total: dict | None, u: Mapping | None) -> dict | None:
    if not u:
        return total
    out = dict(total or {})
    for k, v in u.items():
        if isinstance(v, int | float):
            out[k] = out.get(k, 0) + v
    return out


def usage_by(records: Sequence[Mapping], key: str) -> dict[str, dict]:
    """Usage summed per value of `key` ("kind" or "model") over call records."""
    out: dict[str, dict] = {}
    for r in records:
        if r.get("usage"):
            out[str(r.get(key))] = add_usage(out.get(str(r.get(key))), r["usage"])
    return out


# --- What the harness gives a policy -----------------------------------------------------------------------------


class SessionOverflow(Exception):
    """A call's input exceeded the nominal window W: the session ends, and the item at `position` and every later
    one fail (the report too)."""

    def __init__(self, position: int):
        super().__init__(position)
        self.position = position


@dataclass
class SessionRecords:
    """What a session records besides its history and tool events (store keys in `ape.worlds.env_f8`)."""

    items: list[dict] = field(default_factory=list)  # one per completed case
    views: list[dict] = field(default_factory=list)  # one per agent or management call: item, view tokens, kind, model, usage
    probes: list[dict] = field(default_factory=list)  # one per forked probe (kind probe)
    cm_events: list[dict] = field(default_factory=list)  # management events: policy logs, threshold reactions
    resumes: list[dict] = field(default_factory=list)  # one per resumed attempt (session_checkpoint)

    def state_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, d: Mapping) -> SessionRecords:
        return cls(**{f.name: list(d.get(f.name) or []) for f in dataclasses.fields(cls)})

    def calls(self) -> list[dict]:
        """Every model call of the session: agent and management calls, then probes."""
        return [*self.views, *self.probes]


@dataclass
class SessionContext:
    """The session as a policy sees it: read-only facts, the position, and metered management calls.

    `position` is the item being worked on (0 before the first case, N + 1 for the report) and `stage` where the
    loop is: "start", "item", "boundary" (after an item: its after_item hook and probe) or "report". `system` is
    the history's system message and `tools` every tool the agent has; both are set before `start` / `load_state`,
    so a policy's constructor must not use them."""

    world: World
    window: int
    threshold: int
    agent_model: Model
    cm_model: Model
    records: SessionRecords
    system: ChatMessageSystem | None = None
    tools: list = field(default_factory=list)
    position: int = 0
    stage: str = "start"

    @staticmethod
    def tokens(messages: Sequence[ChatMessage]) -> int:
        return view_tokens(messages)

    def overflow_position(self) -> int:
        """The item an overflow now fails from: the one being worked on (or the report), but the next one at the
        start or at a boundary (the item that just ended is complete)."""
        return self.position + 1 if self.stage in ("start", "boundary") else self.position

    def record_call(self, kind: str, model: Model, tokens: int, output: ModelOutput, purpose: str | None = None) -> dict:
        entry = {"item": self.position, "view_tokens": tokens, "kind": kind, "model": str(model), "usage": usage_record(output.usage)}
        if purpose:
            entry["purpose"] = purpose
        self.records.views.append(entry)
        return entry

    async def cm_generate(self, input: Sequence[ChatMessage], *, purpose: str, tools: Sequence[Tool | ToolDef] = (), config: GenerateConfig | None = None) -> ModelOutput:
        """A management call on the `cm` role: W-checked like an agent call (an input over W is an overflow),
        recorded with kind `cm`, its purpose and its usage."""
        tokens = view_tokens(input)
        if tokens > self.window:
            raise SessionOverflow(self.overflow_position())
        output = await self.cm_model.generate(list(input), tools=list(tools), config=config or GenerateConfig())
        self.record_call("cm", self.cm_model, tokens, output, purpose)
        return output

    def log(self, event: str, **data: Any) -> None:
        """A management event (JSON-able data), attributed to the current item and stage."""
        self.records.cm_events.append({"item": self.position, "stage": self.stage, "event": event, **data})


# --- Policies ------------------------------------------------------------------------------------------------------


class ContextPolicy:
    """Base policy: the append-only history is the view (CM0). Subclasses override the hooks their mechanism needs
    (module docstring); `KNOBS` declares tunable knobs and their defaults."""

    name: ClassVar[str] = "full-history"
    KNOBS: ClassVar[dict[str, Any]] = {}

    def __init__(self, session: SessionContext, **knobs: Any):
        if unknown := sorted(set(knobs) - set(self.KNOBS)):
            raise ValueError(f"{type(self).__name__}: unknown knobs {unknown}; known: {sorted(self.KNOBS)}")
        self.session = session
        self.knobs = {**self.KNOBS, **knobs}

    def tools(self) -> list[Tool | ToolDef]:
        return []

    def system_addendum(self) -> str:
        return ""

    async def start(self, history: list[ChatMessage]) -> None:
        return None

    async def view(self, history: list[ChatMessage], item_start: int, done: int) -> list[ChatMessage]:
        """The model's input for the next call. `item_start` indexes the current case's (or the report request's)
        user message in `history`; `done` cases are complete."""
        return list(history)

    async def on_threshold(self, history: list[ChatMessage], view: list[ChatMessage], tokens: int) -> list[ChatMessage] | None:
        return None

    async def after_generate(self, history: list[ChatMessage], view: list[ChatMessage], output: ModelOutput, appended: list[ChatMessage]) -> None:
        return None

    async def after_item(self, history: list[ChatMessage], done: int) -> None:
        return None

    async def probe_view(self, history: list[ChatMessage], done: int) -> list[ChatMessage]:
        """What the model would see next, after `done` cases (the probe prompt is added by the loop)."""
        return list(history)

    def state_dict(self) -> dict:
        return {}

    def load_state(self, state: dict, history: list[ChatMessage]) -> None:
        return None


class FullHistory(ContextPolicy):
    """CM0: the full, append-only history. Past W the session overflows."""

    name = "CM0"


class OracleState(ContextPolicy):
    """O-state: system + the true shift state after `done` cases (`gen_f8.render_oracle_state`) + the current
    case's messages. Not a real system: the upper bound that defines Gap_T and R_x."""

    name = "O-state"

    def _state(self, done: int, then: str = "") -> ChatMessageUser:
        text = gen_f8.render_oracle_state(self.session.world, done)
        return ChatMessageUser(content=f"{text}\n\n{then}".strip())

    async def view(self, history, item_start, done):
        return [self.session.system, self._state(done, history[item_start].text), *history[item_start + 1 :]]

    async def probe_view(self, history, done):
        return [self.session.system, self._state(done)]


POLICIES: dict[str, type[ContextPolicy]] = {"CM0": FullHistory, "O-state": OracleState}


def register_policy(arm: str, cls: type[ContextPolicy]) -> None:
    """Make `arm` runnable by `f8_session_agent` (B8's arms; tests register their own)."""
    if not (isinstance(cls, type) and issubclass(cls, ContextPolicy)):
        raise TypeError(f"{cls!r} is not a ContextPolicy")
    POLICIES[arm] = cls


def policy_class(arm: str) -> type[ContextPolicy]:
    if arm not in POLICIES:
        raise ValueError(f"session arm {arm!r} is not built yet; built: {tuple(POLICIES)} (the managed arms are B8, CONTEXT_MANAGEMENT_AUDIT §6)")
    return POLICIES[arm]


def _parse_knob(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def resolve_knobs(cls: type[ContextPolicy], environ: Mapping[str, str] = os.environ) -> dict:
    """The policy's knobs: its defaults, each overridden by APE_CM_<KNOB> when set (parsed to the default's type)."""
    out = dict(cls.KNOBS)
    for k, default in cls.KNOBS.items():
        raw = environ.get(f"{KNOB_ENV_PREFIX}{k.upper()}", "").strip()
        if raw:
            out[k] = _parse_knob(raw, default)
    return out


def knob_env(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """Every APE_CM_* variable set: recorded in the task metadata and part of the checkpoint key."""
    return {k: v for k, v in sorted(environ.items()) if k.startswith(KNOB_ENV_PREFIX)}


def tool_name(t: Tool | ToolDef) -> str:
    return t.name if isinstance(t, ToolDef) else ToolDef(t).name


def tool_schema_tokens(tools: Sequence[Tool | ToolDef]) -> int:
    """Tokens of tool schemas (name, description, parameters): view tokens exclude them, so a policy's extra tools are
    recorded with the arm (`policy_tool_tokens`) for analysis."""
    total = 0
    for t in tools:
        td = t if isinstance(t, ToolDef) else ToolDef(t)
        total += count_tokens(json.dumps({"name": td.name, "description": td.description, "parameters": td.parameters.model_dump(exclude_none=True)}))
    return total


class TrackedState:
    """A minimal Inspect `Checkpointer` (only `track`, the one method `inspect_ai.model.compaction()` calls), so a
    policy built on Inspect's compaction handlers checkpoints their closure state.

    Build the handler in `start` with `TrackedState()` and in `load_state` with `TrackedState(state[...])`, passing
    it as `compaction(..., checkpointer=tracked)`; save `tracked.state_dict()` in the policy's `state_dict`. On a
    resume, `track` hands back the saved value, so the handler continues from its compacted view (message IDs are
    kept by the restored history)."""

    def __init__(self, restored: Mapping[str, Any] | None = None):
        self._restored = dict(restored or {})
        self._callbacks: dict[str, Callable[[], Any]] = {}

    def track(self, key: str, callback: Callable[[], Any], initial_value: Any, *, value_type: type | None = None) -> Any:
        if key in self._callbacks:
            raise ValueError(f"checkpoint key {key!r} is already tracked")
        self._callbacks[key] = callback
        if key not in self._restored:
            return initial_value
        raw = self._restored[key]
        if value_type is not None:
            return TypeAdapter(value_type).validate_python(raw)
        if isinstance(initial_value, BaseModel):
            return type(initial_value).model_validate(raw)
        return raw

    def state_dict(self) -> dict:
        return {k: to_jsonable_python(cb()) for k, cb in self._callbacks.items()}
