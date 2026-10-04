"""M5's text ledger (brief §4.2: M5, the text-ledger orchestrator, STATE=text-ledger), after Magentic-One's orchestrator.

M5 is M1 and M5-spec is M2, each with a ledger the harness keeps for the orchestrator; nothing else differs (the same
orchestrator, notes, `plan` and `delegate` tools, workers, delivery, prompt variant and clip):
- **Task ledger:** given facts, facts to look up, facts to derive, educated guesses and the plan. The orchestrator
  writes it before its first turn and rewrites it on a replan.
- **Progress ledger:** before each later turn: whether the request is satisfied, whether the work is in a loop, whether
  progress is being made, the next action and its instruction, and a reason.
- **Stalls and replans:** a progress ledger that reports no progress or a loop counts a stall, one that reports
  progress outside a loop takes one off (Magentic-One's rule); at `APE_MAS_M5_STALL` stalls (default 2) the task ledger
  is rewritten (`REPLAN_PROMPT`, with the stall count and the current ledger) and the count restarts.

**Mechanics.**
- Each ledger update is one side call by the orchestrator, on the agent model at the arm's effort, on its current
  view (its history and, for a per-step arm, this turn's knowledge) plus the ledger prompt, with strict JSON output
  (`TASK_SCHEMA`, `PROGRESS_SCHEMA`). It is never appended to the history: a side call cannot end the reasoning turn
  of the orchestrator's own calls.
- The current task ledger and the latest progress ledger then go to the orchestrator at the end of its latest
  message (`block_view`: on the last tool result in a copy, or in a user message of their own at the first turn), as
  per-step knowledge does (`kb_react.step_view`), so they are replaced every turn and never accumulate.
- The ledger is advisory: the orchestrator still acts only through its tools. The harness acts on the stall count
  alone, and the turn cap bounds every loop (at most one progress call and one replan per orchestrator turn).

**Records and cost.** Ledger calls are the orchestrator's calls, counted in `mas_accounting` (in its agent-model
bucket, and under `purposes.ledger`). `mas_ledger` holds every task-ledger version (start or replan, with the turn,
the parsed ledger or the parse error), every progress entry (turn, ledger or error, whether it counted as a stall,
the stall count it reached: a replan fires at the threshold and then restarts the count) and the replans. A reply that is not the schema's JSON is recorded as an error and changes
nothing (the previous ledger stays shown; the stall count is unchanged). An exception in a ledger call is a provider
error like any other agent's: the sample ends and Inspect retries it (BUILD_REVIEW A-2).
"""

import hashlib
import json
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageTool, ChatMessageUser, ContentText, GenerateConfig, ResponseSchema
from inspect_ai.util import JSONSchema

from . import prompts as P
from .core import AgentRecord, Team

TASK_KEYS = ("given_facts", "facts_to_look_up", "facts_to_derive", "educated_guesses", "plan")
PROGRESS_FLAGS = ("request_satisfied", "in_loop", "progress_being_made")
PROGRESS_TEXTS = ("next_action", "instruction", "reason")
TASK_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": list(TASK_KEYS),
        "properties": {k: {"type": "array", "items": {"type": "string"}} for k in TASK_KEYS},
    }
)
PROGRESS_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": [*PROGRESS_FLAGS, *PROGRESS_TEXTS],
        "properties": {**{k: {"type": "boolean"} for k in PROGRESS_FLAGS}, **{k: {"type": "string"} for k in PROGRESS_TEXTS}},
    }
)
TASK_LEDGER, PROGRESS_LEDGER = "task_ledger", "progress_ledger"  # the response schemas' names
MAX_ITEMS, MAX_CHARS = 20, 400  # per list and per string: the block stays small whatever the model writes


def ledger_sha() -> str:
    """A short hash of the ledger's prompts and labels (recorded in `mas_params.ledger_sha`)."""
    texts = [P.TASK_LEDGER_PROMPT, P.REPLAN_PROMPT, P.PROGRESS_LEDGER_PROMPT, P.LEDGER_HEADER, P.LEDGER_TASK, P.LEDGER_PROGRESS, json.dumps(P.LEDGER_FIELDS)]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()[:12]


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError) as e:
        raise ValueError(f"not JSON ({e})") from None


def parse_task(text: str) -> dict:
    """A task ledger from a reply, or ValueError. Strings are kept (trimmed, capped); anything else in a list is dropped."""
    raw = _json(text)
    if not isinstance(raw, dict) or (missing := [k for k in TASK_KEYS if k not in raw]):
        raise ValueError(f"not a task ledger (missing {missing if isinstance(raw, dict) else list(TASK_KEYS)})")
    out = {}
    for k in TASK_KEYS:
        if not isinstance(raw[k], list):
            raise ValueError(f"{k} is not a list")
        out[k] = [str(x).strip()[:MAX_CHARS] for x in raw[k] if isinstance(x, str) and x.strip()][:MAX_ITEMS]
    return out


def parse_progress(text: str) -> dict:
    """A progress ledger from a reply, or ValueError."""
    raw = _json(text)
    if not isinstance(raw, dict) or (missing := [k for k in (*PROGRESS_FLAGS, *PROGRESS_TEXTS) if k not in raw]):
        raise ValueError(f"not a progress ledger (missing {missing if isinstance(raw, dict) else [*PROGRESS_FLAGS, *PROGRESS_TEXTS]})")
    if bad := [k for k in PROGRESS_FLAGS if not isinstance(raw[k], bool)]:
        raise ValueError(f"{bad} are not true or false")
    return {**{k: raw[k] for k in PROGRESS_FLAGS}, **{k: str(raw[k]).strip()[:MAX_CHARS] for k in PROGRESS_TEXTS}}


def render_task(ledger: dict | None) -> str:
    if ledger is None:
        return P.LEDGER_NONE
    lines = [f"{P.LEDGER_FIELDS[k]}: {'; '.join(ledger[k]) or 'none'}" for k in TASK_KEYS if k != "plan"]
    plan = [f"  {i}. {step}" for i, step in enumerate(ledger["plan"], 1)]
    return "\n".join([*lines, f"{P.LEDGER_FIELDS['plan']}:" + ("" if plan else " none"), *plan])


def render_progress(ledger: dict | None) -> str:
    if ledger is None:
        return P.LEDGER_NONE
    flags = [f"{P.LEDGER_FIELDS[k]}: {'yes' if ledger[k] else 'no'}" for k in PROGRESS_FLAGS]
    return "\n".join([*flags, *(f"{P.LEDGER_FIELDS[k]}: {ledger[k] or '-'}" for k in PROGRESS_TEXTS)])


def block_view(view: list[ChatMessage], block: str) -> list[ChatMessage]:
    """`view` with `block` at its very end, in a copy: appended to the last tool result, or (when the last message is
    not a tool result: the first turn, after a nudge) in a user message of its own. As `kb_react.step_view`."""
    last = view[-1] if view else None
    if isinstance(last, ChatMessageTool):
        content = f"{last.content}\n\n{block}" if isinstance(last.content, str) else [*last.content, ContentText(text=f"\n\n{block}")]
        return [*view[:-1], last.model_copy(update={"content": content})]
    return [*view, ChatMessageUser(content=block)]


def strip_ledger(messages: list[ChatMessage]) -> list[ChatMessage]:
    """`messages` without a ledger block at their end (what the orchestrator would see without the ledger)."""
    last = messages[-1] if messages else None
    if last is None:
        return messages
    if isinstance(last, ChatMessageUser) and last.text.startswith(P.LEDGER_HEADER):
        return messages[:-1]
    marker = f"\n\n{P.LEDGER_HEADER}"
    if isinstance(last, ChatMessageTool):
        if isinstance(last.content, str) and marker in last.content:
            return [*messages[:-1], last.model_copy(update={"content": last.content[: last.content.index(marker)]})]
        if isinstance(last.content, list) and last.content and isinstance(last.content[-1], ContentText) and last.content[-1].text.startswith(marker):
            return [*messages[:-1], last.model_copy(update={"content": last.content[:-1]})]
    return messages


class Ledger:
    """The ledger of one orchestrator: `before_generate` is its `react_loop` hook (module docstring)."""

    def __init__(self, team: Team, agent: AgentRecord, stall_threshold: int):
        self.team, self.agent, self.threshold = team, agent, int(stall_threshold)
        self.task: dict | None = None
        self.version = 0
        self.progress: dict | None = None
        self.progress_turn: int | None = None
        self.stalls = 0
        self.record = team.records.setdefault("mas_ledger", {"stall_threshold": self.threshold, "task_ledgers": [], "progress": [], "replans": 0, "stalls_max": 0})

    async def _call(self, view: list[ChatMessage], prompt: str, name: str) -> tuple[str, str]:
        """One ledger call on the orchestrator's view: (reply text, schema name)."""
        schema = TASK_SCHEMA if name == TASK_LEDGER else PROGRESS_SCHEMA
        config = GenerateConfig(response_schema=ResponseSchema(name=name, json_schema=schema, strict=True))
        async with self.team.purpose("ledger"):
            out = await self.team.model.generate([*view, ChatMessageUser(content=prompt)], config=config)
        return out.completion, name

    async def _task_ledger(self, view: list[ChatMessage], turn: int, reason: str) -> None:
        prompt = P.TASK_LEDGER_PROMPT if reason == "start" else P.REPLAN_PROMPT.format(stalls=self.stalls, ledger=render_task(self.task))
        text, _ = await self._call(view, prompt, TASK_LEDGER)
        entry: dict = {"turn": turn, "reason": reason}
        try:
            self.task = parse_task(text)
            self.version += 1
            entry |= {"version": self.version, "ledger": self.task}
        except ValueError as e:
            entry |= {"version": None, "error": str(e)[:300], "reply": text[:300]}
        self.record["task_ledgers"].append(entry)

    async def _progress(self, view: list[ChatMessage], turn: int) -> None:
        text, _ = await self._call(view, P.PROGRESS_LEDGER_PROMPT.format(ledger=render_task(self.task)), PROGRESS_LEDGER)
        entry: dict = {"turn": turn}
        try:
            progress = parse_progress(text)
        except ValueError as e:
            self.record["progress"].append(entry | {"error": str(e)[:300], "reply": text[:300], "stalls": self.stalls})
            return
        self.progress, self.progress_turn = progress, turn
        stalled = progress["in_loop"] or not progress["progress_being_made"]
        self.stalls = self.stalls + 1 if stalled else max(0, self.stalls - 1)
        self.record["stalls_max"] = max(self.record["stalls_max"], self.stalls)
        self.record["progress"].append(entry | {"ledger": progress, "stalled": stalled, "stalls": self.stalls})
        if self.stalls >= self.threshold:
            await self._task_ledger(view, turn, "replan")
            self.record["replans"] += 1
            self.stalls = 0

    def block(self) -> str:
        parts = [P.LEDGER_HEADER, P.LEDGER_TASK.format(version=self.version, body=render_task(self.task))]
        if self.progress_turn is not None:
            parts.append(P.LEDGER_PROGRESS.format(turn=self.progress_turn, body=render_progress(self.progress)))
        return "\n".join(parts)

    async def before_generate(self, view: list[ChatMessage], turn: int) -> list[ChatMessage]:
        """The ledger calls this turn needs (the task ledger before the first turn, a progress ledger before each later
        one, a replan when the stalls reach the threshold), then the view with the ledger at its end."""
        if turn == 0:
            await self._task_ledger(view, turn, "start")
        else:
            await self._progress(view, turn)
        return block_view(view, self.block())
