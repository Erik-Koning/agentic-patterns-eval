"""Study G's context-management arms (CONTEXT_MANAGEMENT_AUDIT §6.2, BUILD_PLAN B8) on the ContextPolicy layer.

Every arm shares the loop, the base prompt, the session tools, the window W and the threshold T_abs (both on the
harness's token meter); it adds only its mechanism. The managed arms are presets of one composable policy
(`ManagedContext`), built from five components:

- **prune**: Inspect's `CompactionEdit` (keep_tool_uses = `prune_keep`, tool calls kept, memory off): the results
  of all but the last k tool uses become "(Tool result removed)". No model call.
- **trim**: Inspect's `CompactionTrim` (preserve = `trim_preserve`): the oldest messages are dropped, keeping that
  fraction (the system and start messages always stay). No model call.
- **sum**: our own summary call (`cm_prompts.SUMMARY_PROMPTS[sum_prompt]` through `cm_generate`): earlier notes
  plus the shown transcript become new notes, and the summarised messages leave the view.
- **reset**: a fresh context with rehydration: when it fires, the completed messages leave the view and, with
  `reset_summary`, a handoff note (`cm_prompts.HANDOFF_PROMPT`) takes their place. It fires at T_abs and after
  every `reset_every` items (0: only at T_abs).
- **todo**: external task state: Inspect's `todo_write()` and a short addendum; the list is parsed from the
  agent's `todo_write` calls (and from an extraction call, `cm_prompts.TODO_EXTRACT_PROMPT`, at the start of the
  shift and on every case message that announces or withdraws a memo, with `todo_extract`) and kept as policy
  state, so it is checkpointed. Without sum or reset in the stack, todo also drops the completed messages at
  T_abs: the list stands in for them.

The view is always: system, start message (the task spec), the notes (summary or handoff) if any, the shown
completed messages (a compacted window of them after prune or trim, then the uncompacted rest), the todo list,
then the current case's messages. The current case is never compacted, summarised or dropped. When the view
about to be sent exceeds T_abs (level-triggered, `on_threshold`), the components act in a fixed order: the
compactor (prune or trim) first; if the view is still over T_abs, the mover (sum, reset or todo) drops the
completed messages. A component with nothing to act on does nothing. Inspect's strategies are applied directly
(`strategy.compact`), not through `compaction()`: our threshold fires them, and the handler would count tokens
on every call (a provider round trip for OpenAI's Responses API).

Arms (each registered for `f8_session(arm=...)`; knobs are APE_CM_<KNOB>, defaults below):

| Arm | Components | Knobs (default) |
|---|---|---|
| CM-prune | prune | prune_keep (3, Inspect's default) |
| CM-trim | trim | trim_preserve (0.8, Inspect's default) |
| CM-sum | sum | sum_prompt (structured; or plain) |
| CM-todo | todo | todo_extract (true) |
| CM-reset | todo + reset | todo_extract (true), reset_every (5), reset_summary (true) |
| S-CM* | `stack` (prune+todo+reset) | stack, and every knob above |
| CM-native | provider compaction | native_max_failures (3) |

S-CM* stacks are written `a+b+c` (any order): a non-empty set of components with at most one compactor (prune,
trim) and at most one note writer (sum, reset): 17 valid stacks.

**CM-native** compacts with the provider (`Model.compact`, OpenAI's Responses compaction), on the agent's own
model, metered per call with `record_call` (kind cm). At T_abs the earlier context (system excluded, the current
case kept verbatim) goes to the provider; its compacted window replaces it. Gate: it refuses a real provider's
model unless the support record (`native_support`; written by the B11 probe with `record_native_support`) says
the model supports native compaction; mockllm models take a mock path (one model call whose answer becomes an
opaque block) so the plumbing is tested offline. W over an opaque block: the meter cannot read it, so the block
counts the tokens of everything it replaced (no credit) until the first agent call after the compaction reports
the provider's input tokens; from then on it counts that input minus the meter's tokens for the rest of the view.
That includes the tool schemas and per-message overhead, so it over-counts the block (conservative). A compact
call that fails is logged and the session continues uncompacted; after `native_max_failures` failures the arm
stops compacting (logged).

Misbehaving models never crash a session: empty notes keep the previous notes (logged), unparseable todo lists
and malformed `todo_write` calls leave the list as it was (logged), and native compaction failures fall back as
above. Every management event is in `f8_cm_events`, with the summary and handoff texts.
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from inspect_ai.model import ChatMessage, ChatMessageTool, ChatMessageUser, CompactionEdit, CompactionTrim, ContentData, Model, ModelUsage
from inspect_ai.model._call_tools import get_tools_info
from inspect_ai.tool import todo_write
from inspect_ai.util import LimitExceededError

from ..config import Config
from ..llm.tokens import count_tokens
from . import cm_prompts as P
from .context_policy import OPAQUE_TOKENS, ContextPolicy, SessionOverflow, dump_messages, load_messages, register_policy, view_tokens

COMPONENTS = ("prune", "trim", "todo", "sum", "reset")  # canonical order
COMPACTORS = ("prune", "trim")
NOTE_WRITERS = ("sum", "reset")
ALL_KNOBS: dict[str, Any] = {"prune_keep": 3, "trim_preserve": 0.8, "sum_prompt": "structured", "todo_extract": True, "reset_every": 5, "reset_summary": True}
DEFAULT_STACK = "prune+todo+reset"
TODO_STATUSES = ("pending", "in_progress", "completed")
MAX_TODOS, MAX_TODO_CHARS = 300, 500
MEMO_LINE = re.compile(r"^MEMO\b", re.M)
UNREACHABLE = 10**12  # Inspect strategies' own threshold: our T_abs fires them


def parse_stack(text: str) -> tuple[str, ...]:
    """An S-CM* stack ("prune+todo+reset") as components in canonical order; ValueError if it is not valid."""
    parts = [p.strip().lower() for p in str(text).split("+") if p.strip()]
    if not parts:
        raise ValueError("an S-CM* stack needs at least one component")
    if unknown := sorted(set(parts) - set(COMPONENTS)):
        raise ValueError(f"unknown stack components {unknown}; known: {list(COMPONENTS)}")
    if len(parts) != len(set(parts)):
        raise ValueError(f"stack {text!r} repeats a component")
    if len(set(parts) & set(COMPACTORS)) > 1 or len(set(parts) & set(NOTE_WRITERS)) > 1:
        raise ValueError(f"stack {text!r}: at most one of {list(COMPACTORS)} and one of {list(NOTE_WRITERS)}")
    return tuple(c for c in COMPONENTS if c in parts)


def valid_stacks() -> list[str]:
    from itertools import product

    out = []
    for comp, todo, note in product((None, *COMPACTORS), (None, "todo"), (None, *NOTE_WRITERS)):
        parts = [p for p in (comp, todo, note) if p]
        if parts:
            out.append("+".join(c for c in COMPONENTS if c in parts))
    return out


def parse_todos(raw: Any) -> list[dict] | None:
    """A todo list from a todo_write call or an extraction reply: [{content, status}], or None when it is not one.
    Items without text are skipped, an unknown status reads as pending, and long lists and texts are capped. An
    empty list is valid (the list was cleared); a non-empty list with no usable item is not."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, dict) and "todos" in raw:
        raw = raw["todos"]
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw[:MAX_TODOS]:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str) or not item["content"].strip():
            continue
        status = item.get("status") if item.get("status") in TODO_STATUSES else "pending"
        out.append({"content": item["content"].strip()[:MAX_TODO_CHARS], "status": status})
    return out if out or not raw else None


def parse_todo_reply(text: str) -> list[dict] | None:
    """The todo list in a model's reply: the whole reply as JSON, else its outermost [...] span."""
    parsed = parse_todos(text)
    if parsed is None and "[" in text and "]" in text:
        parsed = parse_todos(text[text.index("[") : text.rindex("]") + 1])
    return parsed


def render_todos(todos: list[dict]) -> str:
    lines = [f"- [{t['status']}] {t['content']}" for t in todos]
    return P.TODO_HEADER + "\n" + ("\n".join(lines) if lines else "(empty)")


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


# --- The composable managed policy ---------------------------------------------------------------------------------


class ManagedContext(ContextPolicy):
    """Compaction, summaries, resets and external task state as components (module docstring)."""

    name = "managed"
    STACK: ClassVar[tuple[str, ...]] = ()
    KNOBS: ClassVar[dict[str, Any]] = {}

    @classmethod
    def components(cls, knobs: dict) -> tuple[str, ...]:
        return cls.STACK

    @classmethod
    def validate(cls, knobs) -> None:
        k = {**ALL_KNOBS, **knobs}
        cls.components(k)
        if int(k["prune_keep"]) < 0:
            raise ValueError(f"prune_keep must be >= 0, got {k['prune_keep']}")
        if not 0.0 <= float(k["trim_preserve"]) <= 1.0:
            raise ValueError(f"trim_preserve must be in [0, 1], got {k['trim_preserve']}")
        if k["sum_prompt"] not in P.SUMMARY_PROMPTS:
            raise ValueError(f"sum_prompt must be one of {sorted(P.SUMMARY_PROMPTS)}, got {k['sum_prompt']!r}")
        if int(k["reset_every"]) < 0:
            raise ValueError(f"reset_every must be >= 0, got {k['reset_every']}")

    def __init__(self, session, **knobs):
        super().__init__(session, **knobs)
        self.k = {**ALL_KNOBS, **self.knobs}
        self.stack = self.components(self.k)
        self.compactor = next((c for c in self.stack if c in COMPACTORS), None)
        self.mover = next((c for c in self.stack if c in NOTE_WRITERS), "todo" if "todo" in self.stack else None)
        # State (checkpointed): messages before `covered` (from index 2: after system and start) are shown as
        # `window`, a compacted copy, or not at all once a mover dropped them; `notes` replace dropped messages.
        self.covered = 2
        self.window: list[ChatMessage] = []
        self.notes: str | None = None
        self.notes_kind: str | None = None
        self.todos: list[dict] = []
        self.seen_item = -1
        self.counts = {"compactions": 0, "moves": 0, "notes": 0, "extractions": 0, "todo_writes": 0, "unparsed": 0}
        self._item_start: int | None = None  # the current call's, set by view (not state)

    # -- tools and prompt
    def tools(self):
        return [todo_write()] if "todo" in self.stack else []

    def system_addendum(self) -> str:
        return P.TODO_ADDENDUM if "todo" in self.stack else ""

    # -- the view
    def _assemble(self, history: list[ChatMessage], item_start: int | None) -> list[ChatMessage]:
        end = len(history) if item_start is None else item_start
        head: list[ChatMessage] = [history[0], history[1]]
        if self.notes is not None:
            header = P.HANDOFF_HEADER if self.notes_kind == "handoff" else P.SUMMARY_HEADER
            head.append(ChatMessageUser(content=f"{header}\n\n{self.notes}"))
        todo = [ChatMessageUser(content=render_todos(self.todos))] if "todo" in self.stack else []
        return [*head, *self.window, *history[self.covered : end], *todo, *(history[item_start:] if item_start is not None else [])]

    def _completed(self, history: list[ChatMessage], item_start: int) -> list[ChatMessage]:
        """The shown messages of completed items (what a compactor or mover acts on)."""
        return [*self.window, *history[self.covered : item_start]]

    async def view(self, history, item_start, done):
        self._item_start = item_start
        if item_start != self.seen_item:
            self.seen_item = item_start
            if "todo" in self.stack and self.k["todo_extract"] and MEMO_LINE.search(history[item_start].text or ""):
                await self._extract(history[item_start].text, "memo")
        return self._assemble(history, item_start)

    async def probe_view(self, history, done):
        return self._assemble(history, None)

    # -- hooks
    async def start(self, history):
        if "todo" in self.stack and self.k["todo_extract"]:
            await self._extract(history[1].text, "start")

    async def on_threshold(self, history, view, tokens):
        item_start = self._item_start if self._item_start is not None else len(history)
        changed = False
        if self.compactor and self._completed(history, item_start):
            before = json.dumps(dump_messages(self._completed(history, item_start)))
            window = await self._compact(history, item_start)
            if json.dumps(dump_messages(window)) != before:
                self.window, self.covered, changed = window, item_start, True
                self.counts["compactions"] += 1
                self.session.log(self.compactor, view_tokens=tokens, window_messages=len(window))
            if changed and view_tokens(self._assemble(history, item_start)) <= self.session.threshold:
                return self._assemble(history, item_start)
        if self.mover and self._completed(history, item_start):
            await self._move(history, item_start, "threshold")
            changed = True
        return self._assemble(history, item_start) if changed else None

    async def after_item(self, history, done):
        every = int(self.k["reset_every"])
        if "reset" in self.stack and every > 0 and done % every == 0 and self._completed(history, len(history)):
            await self._move(history, len(history), f"every {every} items")

    async def after_generate(self, history, view, output, appended):
        if "todo" not in self.stack:
            return
        results = {m.tool_call_id: m for m in appended if isinstance(m, ChatMessageTool)}
        for call in output.message.tool_calls or []:
            if call.function != "todo_write":
                continue
            result = results.get(call.id)
            parsed = None if result is None or result.error else parse_todos(call.arguments.get("todos"))
            if parsed is None:
                self.counts["unparsed"] += 1
                self.session.log("todo_unparsed", source="todo_write", error=result.error.message if result is not None and result.error else None)
            else:
                self.todos = parsed
                self.counts["todo_writes"] += 1

    # -- mechanisms
    async def _compact(self, history: list[ChatMessage], item_start: int) -> list[ChatMessage]:
        prefix = [history[0], history[1]]
        if self.compactor == "prune":
            strategy = CompactionEdit(threshold=UNREACHABLE, memory=False, keep_tool_uses=int(self.k["prune_keep"]))
        else:
            strategy = CompactionTrim(threshold=UNREACHABLE, memory=False, preserve=float(self.k["trim_preserve"]))
        out, _ = await strategy.compact(self.session.agent_model, [*prefix, *self._completed(history, item_start)], [])
        ids = {m.id for m in prefix}
        return [m for m in out if m.id not in ids]

    async def _move(self, history: list[ChatMessage], item_start: int, trigger: str) -> None:
        """Drop the completed messages from the view, writing notes in their place (sum, reset with a summary)."""
        dropped = self._completed(history, item_start)
        if self.mover == "sum":
            await self._write_notes(P.SUMMARY_PROMPTS[self.k["sum_prompt"]], "summary", history, dropped)
        elif self.mover == "reset":
            if self.k["reset_summary"]:
                await self._write_notes(P.HANDOFF_PROMPT, "handoff", history, dropped)
            else:
                self.notes = self.notes_kind = None
        self.window, self.covered = [], item_start
        self.counts["moves"] += 1
        self.session.log(f"{self.mover}_drop", trigger=trigger, dropped_messages=len(dropped), dropped_tokens=view_tokens(dropped))

    async def _write_notes(self, prompt: str, purpose: str, history: list[ChatMessage], dropped: list[ChatMessage]) -> None:
        request = P.management_request(prompt, [("Shift start", history[1].text), ("Earlier notes", self.notes or ""), ("Transcript", P.render_transcript(dropped))])
        out = await self.session.cm_generate([ChatMessageUser(content=request)], purpose=purpose)
        text = (out.completion or "").strip()
        if not text:
            self.session.log("empty_notes", purpose=purpose)
            text = self.notes or P.NO_NOTES
        self.notes, self.notes_kind = text, purpose
        self.counts["notes"] += 1
        self.session.log(purpose, text=text, tokens=count_tokens(text))

    async def _extract(self, message: str, trigger: str) -> None:
        request = P.management_request(P.TODO_EXTRACT_PROMPT, [("Current todo list", json.dumps(self.todos)), ("Message", message)])
        out = await self.session.cm_generate([ChatMessageUser(content=request)], purpose="todo_extract")
        parsed = parse_todo_reply(out.completion or "")
        if parsed is None:
            self.counts["unparsed"] += 1
            self.session.log("todo_unparsed", source="extraction", trigger=trigger)
            return
        self.todos = parsed
        self.counts["extractions"] += 1
        self.session.log("todo_extract", trigger=trigger, entries=len(parsed))

    # -- checkpoint
    def state_dict(self):
        return {
            "covered": self.covered,
            "window": dump_messages(self.window),
            "notes": self.notes,
            "notes_kind": self.notes_kind,
            "todos": self.todos,
            "seen_item": self.seen_item,
            "counts": self.counts,
        }

    def load_state(self, state, history):
        self.covered, self.window = int(state["covered"]), load_messages(state["window"])
        self.notes, self.notes_kind, self.todos = state["notes"], state["notes_kind"], list(state["todos"])
        self.seen_item, self.counts = int(state["seen_item"]), dict(state["counts"])


class CMPrune(ManagedContext):
    name = "CM-prune"
    STACK = ("prune",)
    KNOBS: ClassVar[dict[str, Any]] = {"prune_keep": ALL_KNOBS["prune_keep"]}


class CMTrim(ManagedContext):
    name = "CM-trim"
    STACK = ("trim",)
    KNOBS: ClassVar[dict[str, Any]] = {"trim_preserve": ALL_KNOBS["trim_preserve"]}


class CMSum(ManagedContext):
    name = "CM-sum"
    STACK = ("sum",)
    KNOBS: ClassVar[dict[str, Any]] = {"sum_prompt": ALL_KNOBS["sum_prompt"]}


class CMTodo(ManagedContext):
    name = "CM-todo"
    STACK = ("todo",)
    KNOBS: ClassVar[dict[str, Any]] = {"todo_extract": ALL_KNOBS["todo_extract"]}


class CMReset(ManagedContext):
    name = "CM-reset"
    STACK = ("todo", "reset")
    KNOBS: ClassVar[dict[str, Any]] = {k: ALL_KNOBS[k] for k in ("todo_extract", "reset_every", "reset_summary")}


class StackCM(ManagedContext):
    """S-CM*: the best single-agent stack, chosen on dev; its components are the `stack` knob (APE_CM_STACK)."""

    name = "S-CM*"
    KNOBS: ClassVar[dict[str, Any]] = {"stack": DEFAULT_STACK, **ALL_KNOBS}

    @classmethod
    def components(cls, knobs: dict) -> tuple[str, ...]:
        return parse_stack(knobs["stack"])


# --- Native compaction ------------------------------------------------------------------------------------------

NATIVE_RECORD_ENV = "APE_NATIVE_COMPACTION_RECORD"


class NativeCompactionUnsupported(RuntimeError):
    """CM-native was asked to run on a provider model whose native compaction support is not confirmed."""


def native_record_path() -> Path:
    """The per-model support record: $APE_NATIVE_COMPACTION_RECORD, else <APE_CACHE>/native_compaction.json (next to
    the E3 probe's output)."""
    import os

    raw = os.environ.get(NATIVE_RECORD_ENV, "").strip()
    return Path(raw) if raw else Config().cache_dir / "native_compaction.json"


def native_support(model: str, path: Path | None = None) -> dict | None:
    p = path or native_record_path()
    if not p.is_file():
        return None
    try:
        return (json.loads(p.read_text()).get("models") or {}).get(model)
    except (OSError, ValueError):
        return None


def record_native_support(model: str, supported: bool, *, evidence: str, path: Path | None = None, **extra: Any) -> Path:
    """Write one model's native-compaction verdict (the B11 probe calls this after a real compact call: `supported`
    only when it returned a compacted context and the next generate accepted it)."""
    p = path or native_record_path()
    data = json.loads(p.read_text()) if p.is_file() else {"format": 1, "models": {}}
    data["models"][model] = {"supported": bool(supported), "evidence": evidence, "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"), **extra}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, sort_keys=True))
    return p


def native_route(model: Model | str) -> str:
    """"mock" for mockllm models (the offline path); "provider" for a model the support record confirms; otherwise
    NativeCompactionUnsupported. The study runner's preflight can call it before any sample runs."""
    name = str(model)
    if name.startswith("mockllm/"):
        return "mock"
    rec = native_support(name)
    if not rec or rec.get("supported") is not True:
        raise NativeCompactionUnsupported(
            f"CM-native needs confirmed native compaction for {name}: none in {native_record_path()}. Run the B11 probe, "
            "which records it with ape.agent.cm_arms.record_native_support, or leave CM-native out of this run."
        )
    return "provider"


class NativeCompaction(ContextPolicy):
    """CM-native: provider-side compaction of the earlier context at T_abs (module docstring)."""

    name = "CM-native"
    KNOBS: ClassVar[dict[str, Any]] = {"native_max_failures": 3}

    @classmethod
    def validate(cls, knobs) -> None:
        if int(knobs.get("native_max_failures", 3)) < 1:
            raise ValueError("native_max_failures must be >= 1")

    def __init__(self, session, **knobs):
        super().__init__(session, **knobs)
        self.route = native_route(session.agent_model)
        self.covered = 1  # the first compaction takes the start message too
        self.block: list[ChatMessage] = []
        self.calibrated = True
        self.failures = 0
        self.disabled = False
        self.counts = {"compactions": 0, "failures": 0}
        self._item_start: int | None = None

    def _assemble(self, history, item_start: int | None) -> list[ChatMessage]:
        if not self.block:
            return list(history)
        return [history[0], *self.block, *history[self.covered :]]

    async def view(self, history, item_start, done):
        self._item_start = item_start
        return self._assemble(history, item_start)

    async def probe_view(self, history, done):
        return self._assemble(history, None)

    async def _compact(self, input: list[ChatMessage]) -> tuple[list[ChatMessage], ModelUsage | None]:
        model = self.session.agent_model
        if self.route == "provider":
            return await model.compact(input, get_tools_info(self.session.tools))
        out = await model.generate([*input, ChatMessageUser(content=P.NATIVE_MOCK_PROMPT)])
        return [ChatMessageUser(content=[ContentData(data={"compaction": out.completion or ""})])], out.usage

    async def on_threshold(self, history, view, tokens):
        item_start = self._item_start if self._item_start is not None else len(history)
        if self.disabled or not history[self.covered : item_start]:
            return None
        input = [history[0], *self.block, *history[self.covered : item_start]]
        size = view_tokens(input)
        if size > self.session.window:
            raise SessionOverflow(self.session.overflow_position())
        try:
            compacted, usage = await self._compact(input)
            compacted = [m for m in compacted if m.role != "system"]
            if not compacted:
                raise ValueError("the provider returned an empty compacted context")
        except (SessionOverflow, LimitExceededError):
            raise
        except Exception as e:  # noqa: BLE001  (a failed compaction is logged and the session goes on uncompacted)
            self.failures += 1
            self.counts["failures"] += 1
            self.session.log("native_error", error=_short(e), failures=self.failures)
            if self.failures >= int(self.knobs["native_max_failures"]):
                self.disabled = True
                self.session.log("native_disabled", failures=self.failures)
            return None
        self.session.record_call("cm", self.session.agent_model, size, usage, "native")
        replaced = view_tokens(input[1:])
        visible = view_tokens(compacted)
        compacted[0] = compacted[0].model_copy(update={"metadata": {**(compacted[0].metadata or {}), OPAQUE_TOKENS: max(0, replaced - visible)}})
        self.block, self.covered, self.calibrated = compacted, item_start, False
        self.failures = 0
        self.counts["compactions"] += 1
        self.session.log("native", view_tokens=tokens, replaced_tokens=replaced, visible_tokens=visible)
        return self._assemble(history, item_start)

    async def after_generate(self, history, view, output, appended):
        """Calibrate the block from the first agent call after it: the provider's input tokens minus the meter's
        tokens for the rest of the view (tool schemas and overhead included: an over-count, conservative)."""
        if not self.block or self.calibrated or output.usage is None:
            return
        u = output.usage
        provider = u.input_tokens + (u.input_tokens_cache_read or 0) + (u.input_tokens_cache_write or 0)
        surcharge = int((self.block[0].metadata or {}).get(OPAQUE_TOKENS) or 0)
        rest = view_tokens(view) - surcharge
        new = max(0, provider - rest)
        self.block[0] = self.block[0].model_copy(update={"metadata": {**(self.block[0].metadata or {}), OPAQUE_TOKENS: new}})
        self.calibrated = True
        self.session.log("native_calibrated", provider_input_tokens=provider, opaque_tokens_before=surcharge, opaque_tokens=new)

    def state_dict(self):
        return {"covered": self.covered, "block": dump_messages(self.block), "calibrated": self.calibrated, "failures": self.failures, "disabled": self.disabled, "counts": self.counts}

    def load_state(self, state, history):
        self.covered, self.block = int(state["covered"]), load_messages(state["block"])
        self.calibrated, self.failures, self.disabled, self.counts = bool(state["calibrated"]), int(state["failures"]), bool(state["disabled"]), dict(state["counts"])


CM_ARMS: dict[str, type[ContextPolicy]] = {
    "CM-prune": CMPrune,
    "CM-trim": CMTrim,
    "CM-sum": CMSum,
    "CM-todo": CMTodo,
    "CM-reset": CMReset,
    "CM-native": NativeCompaction,
    "S-CM*": StackCM,
}
for _arm, _cls in CM_ARMS.items():
    register_policy(_arm, _cls)
