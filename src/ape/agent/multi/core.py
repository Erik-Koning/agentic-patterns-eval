"""The pieces every multi-agent and ensemble arm shares: the agent registry and its per-agent accounting, knowledge
delivery per agent, the turn loop, and environment isolation.

**The turn loop (`react_loop`)** applies `kb_react.kb_agent`'s per-turn rules to any agent's history, so a worker, an
orchestrator or a council member sees knowledge exactly as a single agent of the same delivery arm does:
- Per-query arms put the knowledge in the system prompt (`kb_react.system_prompt`); per-step arms recompile from
  `step.step_query(<the agent's own task or subtask>, history)` every turn and deliver it through `kb_react.step_view`
  (appended to the last tool result in a copy, never to the history).
- Tool exposure per turn is `step.exposed_tools` over the delivered context, with the agent's always-on tools.
- A text-only reply gets one nudge per text-only streak (at most `kb_react.MAX_NUDGES` per loop call); a second
  text-only reply in a row ends the loop. The loop also ends when the agent's `done()` holds after a tool turn, or at
  its turn cap.
`tests/test_multi_agent.py` checks that a loop configured as S1 gives the model byte-identical inputs to `kb_agent`.

**Accounting (`Team.accounting`)** is read from the sample transcript, not counted by hand: every model call is a
`ModelEvent` carrying its span, and every agent runs inside its own `span(type="agent")`, so each call (the agent's own
generations and the `kg` role's classify calls made while compiling its context) belongs to the nearest enclosing
agent span. Inspect completes the event, with its usage, before it records the usage and checks the token limit, so
the call that trips a limit is counted too. Per agent: calls, input / output / reasoning / cache-read / cache-write /
total tokens and model seconds per model role, plus the agent's wall-clock, and per purpose (`Team.purpose`: a nested
span marking an agent's side calls, e.g. M5's ledger calls) the calls made for it, a subset of the agent's own. Cache-hit
events (`cache="read"`) are left
out, as Inspect leaves them out of the sample's usage. Calls outside every agent span would land in `unattributed`
(none do: the solvers open an agent span before any model call); the per-agent totals plus `unattributed` equal the
sample's `model_usage` exactly.

**Limits.** The sample's `token_limit` is a node in Inspect's limit tree, a ContextVar that every child task inherits, so
every agent's calls count against it (and against nothing else: the arms set no per-agent limits; turn caps are the
loops' own counters). When it trips inside a worker, the worker's `LimitExceededError` must end the sample.

**Errors.** Any other exception inside an agent (a provider error after the client's retries, a failed knowledge
compile) ends the sample too, as it would S1's: Inspect's `retry_on_error` then re-runs it, and an exhausted retry is a
counted harness error (BUILD_REVIEW A-2). Only model misbehaviour that is not an exception (text instead of a call, a bad
or unknown tool call, a worker that never reports) gives a marked result or a tool error the agents work around.
Inspect's `execute_tools` maps a `LimitExceededError`, and some other exceptions (a `TimeoutError`, a
`PermissionError`, ...), raised inside a tool into a tool error and carries on (`tool_call_error`); the workers run
inside the orchestrator's `delegate` tool, so the delegate tool holds any worker failure on the team (`note_error`)
and the loop re-raises it right after `execute_tools` (`raise_pending_error`). Every solver writes its records in a
`finally`, so a sample cut short keeps them.

**Isolation (`run_isolated`).** `inspect_ai.util.store()` is a ContextVar; each isolated unit (a worker, an ensemble
attempt, a council member's phase) runs in its own anyio task with its own `Store` (`init_subtask_store`, as Inspect's
own `subtask` does). Only the environment's keys (`env_*`, `worlds/env_tools.py`) are copied in; `merge_env` folds a
unit's environment changes back into the sample's store. Units run concurrently or one after another; either way each
unit is its own task, so concurrency changes scheduling only.
"""

import json
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
from inspect_ai.event import ModelEvent, SpanBeginEvent
from inspect_ai.log import transcript
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput, execute_tools, get_model
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef, ToolFunction
from inspect_ai.util import LimitExceededError, Store, current_span_id, span
from inspect_ai.util._store import init_subtask_store

from ...kb.context import ContextResult, DeliveryArm
from ...llm.embeddings import EMBED_CONTEXT
from ...llm.tokens import count_tokens, truncate_to_tokens
from ...worlds.env_tools import ANSWER
from ...worlds.spec import TaskItem, World
from ..arms import load_world
from ..cache_nonce import of as nonce_of
from ..cache_nonce import with_nonce
from ..kb_react import COMPILE_LOG, EMPTY, MAX_NUDGES, SEARCH_ERRORS, STEP_LOG, step_view, system_prompt
from ..step import exposed_tools, step_query

T = TypeVar("T")
ENV_PREFIX = "env_"  # the environment's store keys (worlds/env_tools.py); everything else is a log
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "input_tokens_cache_read", "input_tokens_cache_write")
AGENT_MODEL = "agent"  # the bucket for the agent model's own calls (ModelEvent.role None); other roles keep their name
TEXT_MAX_TOKENS = 2000  # cap on any one agent's text that another agent reads (a worker result, a proposal rationale)


def clip(text: str, max_tokens: int = TEXT_MAX_TOKENS) -> str:
    """`text` cut to `max_tokens` with a visible marker when cut (one agent's output in another's input)."""
    text = str(text or "")
    if count_tokens(text) <= max_tokens:
        return text
    return truncate_to_tokens(text, max_tokens) + f"\n[... cut to {max_tokens} tokens]"


# --- agents and the team --------------------------------------------------------------------------------------------


@dataclass
class AgentRecord:
    """One agent of a sample: who it is, what it did (a transcript summary) and what it was delivered."""

    id: str
    role: str
    parent: str | None = None
    info: dict = field(default_factory=dict)
    turns: int = 0
    nudges: int = 0
    stop: str | None = None  # how its last loop ended: done | text | turn_cap | limit | interrupted | error
    tool_calls: list[str] = field(default_factory=list)  # function names, in call order
    compile_log: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    error: str | None = None
    last_text: str = ""
    output: ModelOutput | None = None  # the agent's latest generation (not logged; the sample's output for a top agent)

    def summary(self) -> dict:
        return {
            "id": self.id,
            "role": self.role,
            "parent": self.parent,
            **self.info,
            "turns": self.turns,
            "nudges": self.nudges,
            "stop": self.stop,
            "tool_calls": self.tool_calls,
            "compiles": len(self.compile_log),
            "wall_s": round(self.wall_s, 3),
            "error": self.error,
            "last_text": clip(self.last_text, 200),
        }


class Team:
    """The agents of one sample, their spans, the arm's records and the per-agent accounting."""

    def __init__(self, state: TaskState, arm: str, switches: dict, params: dict, world: World | None = None, task: TaskItem | None = None):
        """`world` and `task` default to the sample's (`metadata["world_id"]`, `metadata["task"]`)."""
        self.state = state
        self.arm = arm
        self.switches = switches
        self.params = params
        self.world: World = world if world is not None else load_world(state.metadata["world_id"])
        self.task = task if task is not None else TaskItem(**state.metadata["task"])
        self.model = get_model()
        self.agents: dict[str, AgentRecord] = {}
        self.records: dict[str, Any] = {}
        self.top: AgentRecord | None = None  # the agent whose turns are the sample's `turns_used`
        self._spans: dict[str, str] = {}
        self._purposes: dict[str, str] = {}
        self._pending_error: Exception | None = None
        self._t0 = time.perf_counter()

    def agent(self, id: str, role: str, parent: str | None = None, **info) -> AgentRecord:
        if id in self.agents:
            raise ValueError(f"agent {id} already exists")
        self.agents[id] = rec = AgentRecord(id, role, parent, info)
        return rec

    @asynccontextmanager
    async def running(self, agent: AgentRecord):
        """The agent's span: every model call inside it (and inside nested non-agent spans) is the agent's."""
        t0 = time.perf_counter()
        try:
            async with span(name=f"{agent.role}:{agent.id}", type="agent", id=f"mas-{agent.id}-{uuid.uuid4().hex[:8]}"):
                self._spans[current_span_id()] = agent.id
                yield agent
        finally:
            agent.wall_s += time.perf_counter() - t0

    @asynccontextmanager
    async def purpose(self, name: str):
        """A span inside the running agent's: its calls stay the agent's, and are also counted under `purposes[name]`
        (M5's ledger calls are the orchestrator's calls, with purpose `ledger`)."""
        async with span(name=f"purpose:{name}", type="purpose"):
            self._purposes[current_span_id()] = name
            yield

    def note_error(self, error: Exception) -> None:
        """A limit or an error inside a tool (a worker run by `delegate`): `execute_tools` may turn it into a tool error,
        so it is kept here and re-raised once the tool call returns."""
        if self._pending_error is None:
            self._pending_error = error

    def raise_pending_error(self) -> None:
        if self._pending_error is not None:
            raise self._pending_error

    def answered(self) -> bool:
        return self.state.store.get(ANSWER) is not None

    # -- accounting ----------------------------------------------------------------------------------------------

    def accounting(self) -> dict:
        events = list(transcript().events)
        parents = {e.id: e.parent_id for e in events if isinstance(e, SpanBeginEvent)}
        owner_of: dict[str | None, tuple[str | None, str | None]] = {}

        def owner(span_id: str | None) -> tuple[str | None, str | None]:
            """The nearest enclosing agent span's agent, and the nearest purpose span below it (if any)."""
            if span_id not in owner_of:
                seen, sid, purpose = set(), span_id, None
                while sid is not None and sid not in self._spans and sid not in seen:
                    seen.add(sid)
                    purpose = purpose or self._purposes.get(sid)
                    sid = parents.get(sid)
                owner_of[span_id] = (self._spans.get(sid) if sid is not None else None, purpose)
            return owner_of[span_id]

        per: dict[str, dict] = {aid: {"role": a.role, "parent": a.parent, "models": {}, "purposes": {}, "wall_s": round(a.wall_s, 3)} for aid, a in self.agents.items()}
        unattributed: dict[str, dict] = {}

        def add(buckets: dict, key: str, e: ModelEvent) -> None:
            bucket = buckets.setdefault(key, {"calls": 0, **dict.fromkeys(TOKEN_FIELDS, 0), "model_seconds": 0.0})
            bucket["calls"] += 1
            for f in TOKEN_FIELDS:
                bucket[f] += getattr(e.output.usage, f) or 0
            bucket["model_seconds"] += e.working_time or 0.0

        for e in events:
            if not isinstance(e, ModelEvent) or e.pending or e.cache == "read" or e.output is None or e.output.usage is None:
                continue
            aid, purpose = owner(e.span_id)
            add(per[aid]["models"] if aid is not None else unattributed, e.role or AGENT_MODEL, e)
            if aid is not None and purpose is not None:
                add(per[aid]["purposes"], purpose, e)
        for rec in [*per.values(), {"models": unattributed}]:
            for b in [*rec["models"].values(), *rec.get("purposes", {}).values()]:
                b["model_seconds"] = round(b["model_seconds"], 3)
        for rec in per.values():
            mine = rec["models"].get(AGENT_MODEL, {})
            rec.update({"calls": mine.get("calls", 0), **{f: mine.get(f, 0) for f in TOKEN_FIELDS}})
            rec["all_total_tokens"] = sum(b["total_tokens"] for b in rec["models"].values())
        totals = {f: sum(b[f] for rec in [*per.values(), {"models": unattributed}] for b in rec["models"].values()) for f in ("calls", *TOKEN_FIELDS)}
        summed = sum(b["model_seconds"] for rec in per.values() for b in rec["models"].values())
        wall = time.perf_counter() - self._t0
        return {
            "agents": per,
            "unattributed": unattributed,
            "totals": totals,
            "sample_wall_s": round(wall, 3),
            "summed_model_s": round(summed, 3),
            "realized_parallelism": round(summed / wall, 3) if wall > 0 else None,
        }

    # -- records -------------------------------------------------------------------------------------------------

    def write(self, delivery: "Delivery | None", exposure: str) -> None:
        """Every record of the sample, into the sample's store (called from the solver's `finally`; no awaits)."""
        st = self.state.store
        agents = list(self.agents.values())
        st.set("mas_switches", self.switches)
        st.set("mas_params", self.params)
        st.set("mas_agents", [a.summary() for a in agents])
        try:
            st.set("mas_accounting", self.accounting())
        except Exception as e:  # never mask the sample's own exception from inside its `finally`
            st.set("mas_accounting", {"error": f"{type(e).__name__}: {e}"[:500]})
        for k, v in self.records.items():
            st.set(k, v)
        # The single-agent keys, so the gate's scorers and readers work unchanged: the compile log is every agent's
        # compiles (agent order, then turn), the step log every agent's turns.
        st.set(COMPILE_LOG, [rec for a in agents for rec in a.compile_log])
        st.set(STEP_LOG, [{**s, "agent": a.id} for a in agents for s in a.steps])
        st.set(SEARCH_ERRORS, [])
        top = self.top or (agents[0] if agents else None)
        st.set("turns_used", top.turns if top else 0)
        st.set("nudges", top.nudges if top else 0)
        arm = delivery.arm if delivery else None
        st.set("arm", {"name": self.arm, "per_step": bool(arm and arm.per_step), "exposure": exposure, "delivery": "push",
                       "delivery_arm": arm.name if arm else None, "multi_agent": True})

    def finish(self, delivery: "Delivery | None", exposure: str) -> None:
        """The solver's `finally`: records written, and the top agent's latest generation as the sample's output."""
        self.write(delivery, exposure)
        if self.top is not None and self.top.output is not None:
            self.state.output = self.top.output


# --- knowledge delivery -----------------------------------------------------------------------------------------------


class Delivery:
    """One delivery arm serving every agent of a sample; each compile is logged on the agent that asked for it."""

    def __init__(self, arm: DeliveryArm, team: Team):
        self.arm, self.team = arm, team
        self.per_step = arm.per_step

    async def compile(self, agent: AgentRecord, query: str, turn: int, source: str = "push") -> ContextResult:
        team = self.team
        ctx_vars = {"arm": self.arm.name, "world": team.world.id, "sample": str(team.state.sample_id), "epoch": team.state.epoch, "source": source, "agent": agent.id}
        token = EMBED_CONTEXT.set(ctx_vars)
        t0 = time.perf_counter()
        try:
            ctx = await self.arm.compile(query, team.task)
        finally:
            EMBED_CONTEXT.reset(token)
        agent.compile_log.append({**ctx.log_record(turn), "source": source, "compile_ms": round((time.perf_counter() - t0) * 1000, 1), "agent": agent.id, "role": agent.role})
        return ctx

    def system(self, ctx: ContextResult) -> ChatMessageSystem:
        """The single-agent arms' system prompt (push delivery): the knowledge itself for per-query arms, after the
        sample's cache nonce when the run sets one (`agent.cache_nonce`; every role of the arm shares it)."""
        return ChatMessageSystem(content=with_nonce(system_prompt(True, False, self.per_step, ctx.text), nonce_of(self.team.state.metadata)))


# --- the turn loop ----------------------------------------------------------------------------------------------------


async def react_loop(
    team: Team,
    agent: AgentRecord,
    messages: list[ChatMessage],
    *,
    delivery: Delivery,
    ctx: ContextResult | None,
    query: str,
    tools: dict[str, ToolDef],
    always: Sequence[str],
    exposure: str,
    max_turns: int,
    done: Callable[[], bool],
    nudge: str,
    first_tool: str | None = None,
    before_generate: Callable[[list[ChatMessage], int], Awaitable[list[ChatMessage]]] | None = None,
) -> ModelOutput | None:
    """Turns of one agent over `messages` (appended in place) until `done()`, a second text-only reply in a row, or
    `max_turns` (module docstring). `ctx` is the delivery for the first turn; with a per-step arm and `ctx=None` the
    first turn compiles too (a history that continues, e.g. a council member's next round). `first_tool` forces the
    first turn's tool choice (the planning step). `before_generate(view, turn)` may make side calls and returns the
    view to send instead (M5's ledger: the view with the current ledger at its end, never stored in the history)."""
    nudges, streak, output, turn, stop = 0, False, None, -1, "turn_cap"
    base = agent.turns
    try:
        for turn in range(max_turns):
            if delivery.per_step and (turn > 0 or ctx is None):
                ctx = await delivery.compile(agent, step_query(query, messages), base + turn)
            cur = ctx or EMPTY
            exposed = exposed_tools(tools, cur, exposure, list(always))
            agent.steps.append({"turn": base + turn, "exposed_tools": [t.name for t in exposed]})
            view = step_view(messages, cur.text) if delivery.per_step else messages
            if before_generate is not None:
                view = await before_generate(view, turn)
            choice = ToolFunction(name=first_tool) if turn == 0 and first_tool and any(t.name == first_tool for t in exposed) else None
            output = agent.output = await team.model.generate(view, tools=exposed, tool_choice=choice)
            messages.append(output.message)
            if output.message.text:
                agent.last_text = output.message.text
            if output.message.tool_calls:
                streak = False
                agent.tool_calls += [c.function for c in output.message.tool_calls]
                result = await execute_tools(messages, exposed)
                messages.extend(result.messages)
                team.raise_pending_error()
                if done():
                    stop = "done"
                    break
            elif streak or nudges >= MAX_NUDGES:
                stop = "text"
                break
            else:
                messages.append(ChatMessageUser(content=nudge))
                nudges, streak = nudges + 1, True
    except LimitExceededError:
        stop = "limit"
        raise
    except BaseException:  # an exception (the caller decides what it means) or a cancellation (e.g. the time limit)
        stop = "interrupted"
        raise
    finally:
        agent.turns = base + turn + 1
        agent.nudges += nudges
        agent.stop = stop
    return output


def last_assistant_text(messages: Sequence[ChatMessage]) -> str:
    return next((m.text for m in reversed(messages) if isinstance(m, ChatMessageAssistant) and m.text), "")


# --- isolation --------------------------------------------------------------------------------------------------------


def env_data(store: Store) -> dict:
    """A copy of the environment's keys in `store` (the start state of an isolated unit)."""
    return {k: deepcopy(v) for k, v in store.items() if k.startswith(ENV_PREFIX)}


def merge_env(target: Store, base: dict, branch: Store) -> None:
    """Fold one isolated unit's environment changes (relative to `base`, its start state) into `target`: list logs
    (calls, faults fired) append their new entries, counters add their deltas, an answer never replaces one already
    recorded (the first answer wins, as in `env_tools`), and anything else takes the unit's value."""
    for k, v in branch.items():
        if not k.startswith(ENV_PREFIX):
            continue
        b = base.get(k)
        if isinstance(v, list):
            new = v[len(b):] if isinstance(b, list) else v
            if new:
                target.set(k, [*target.get(k, []), *deepcopy(new)])
        elif isinstance(v, dict) and all(isinstance(x, int) for x in v.values()) and k != ANSWER:
            merged = dict(target.get(k, {}))
            for key, n in v.items():
                delta = n - ((b or {}).get(key, 0))
                if delta:
                    merged[key] = merged.get(key, 0) + delta
            target.set(k, merged)
        elif v != b:
            if k == ANSWER and target.get(ANSWER) is not None:
                continue
            target.set(k, deepcopy(v))


def _leaves(e: BaseException) -> list[BaseException]:
    if isinstance(e, BaseExceptionGroup):
        return [x for sub in e.exceptions for x in _leaves(sub)]
    return [e]


async def _task_group(fns: list[Callable[[], Awaitable[None]]]) -> None:
    """Run `fns` as sibling tasks (a failure cancels the rest). A limit error wins over any other failure, and a lone
    failure is re-raised bare, so Inspect's sample runner sees the `LimitExceededError` it handles."""
    try:
        async with anyio.create_task_group() as tg:
            for fn in fns:
                tg.start_soon(fn)
    except BaseExceptionGroup as group:
        leaves = _leaves(group)
        limit = next((x for x in leaves if isinstance(x, LimitExceededError)), None)
        if limit is not None:
            raise limit from None
        if len(leaves) == 1:
            raise leaves[0] from None
        raise


async def run_isolated(units: Sequence[Callable[[], Awaitable[T]]], stores: Sequence[Store], concurrent: bool) -> list[T]:
    """Run each unit in its own task with its own store, all at once (`concurrent`) or one after another; results in
    unit order."""
    results: list[Any] = [None] * len(units)

    def runner(i: int) -> Callable[[], Awaitable[None]]:
        async def run() -> None:
            init_subtask_store(stores[i])
            results[i] = await units[i]()

        return run

    if concurrent:
        await _task_group([runner(i) for i in range(len(units))])
    else:
        for i in range(len(units)):
            await _task_group([runner(i)])
    return results


def as_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
