"""The F8 session loop for Study G (CONTEXT_MANAGEMENT_AUDIT §6.1), with the two arms built so far.

The loop keeps the **full history** (logged as the sample's messages) and asks a view policy for what the model
sees at each call. Both arms share one system prompt (base prompt, the whole corpus, shift rules), one tool
set and one turn cap per case; only the view differs:

- **CM0**: the full, append-only history. The harness enforces the nominal window W: a call whose view would
  exceed W is an **overflow**, and that case and every later one fail (the report too).
- **O-state**: system prompt + the generator's true state after the previous case (`render_oracle_state`) +
  the current case's messages. The upper bound that defines Gap_T and R_x.

Each case arrives as a user message; the loop moves on once the case's answer call is made (submit_decision or
finish with its case ID), after one nudge, or at `max_turns_per_item`. After the last case it asks for the
end-of-shift report.

**Forked state probes** (§5.3): after each checkpoint case k <= N, the probe question goes to the policy's
current view (what the model would see next) in a side call with role `probe` and a strict JSON schema. The
answer is never appended to the history. Without a `probe` role the agent's model answers, as the budget
assumes ("at the agent's model and effort"); probe tokens are recorded separately so cost analysis can
exclude them.

Recorded in the store: per-case records (position, view tokens at the first call and at the decision call,
generations, answered, success, dependency), every call's view tokens, the probes, the tool events, the report
and the overflow position. Other arms (prune, trim, summary, native, notes, todo, reset) plug in as further
view policies later.
"""

import json
from typing import Protocol

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig, ResponseSchema, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import JSONSchema, store

from ..llm.tokens import count_tokens
from ..scorers.session import item_success, probe_score
from ..worlds import gen_f8
from ..worlds.env_f8 import EVENTS, ITEMS, OVERFLOW, PROBES, REPORT, VIEWS, SessionRecorder, build_session_tools
from ..worlds.spec import World

SESSION_ARMS = ("CM0", "O-state")
PROBE_ROLE = "probe"
PROBE_PROMPT = (
    "Pause the shift for a quick state check. Without using tools, list as JSON: the case IDs you have completed "
    "(completed), the case IDs still pending (pending), the memo IDs in force now (memos_in_force), the case IDs "
    "waiting for a follow-up (open_followups) and the case IDs you escalated this shift (escalated)."
)
PROBE_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": list(gen_f8.PROBE_CATEGORIES),
        "properties": {c: {"type": "array", "items": {"type": "string"}} for c in gen_f8.PROBE_CATEGORIES},
    }
)


def message_tokens(m: ChatMessage) -> int:
    n = count_tokens(m.text or "")
    if isinstance(m, ChatMessageAssistant) and m.tool_calls:
        n += sum(count_tokens(json.dumps({"function": c.function, "arguments": c.arguments})) for c in m.tool_calls)
    return n


def view_tokens(messages: list[ChatMessage]) -> int:
    """Tokens of a view (o200k_base over message texts and tool calls; tool schemas excluded, they are constant)."""
    return sum(message_tokens(m) for m in messages)


class ViewPolicy(Protocol):
    name: str

    def view(self, history: list[ChatMessage], item_start: int, done: int) -> list[ChatMessage]:
        """The model's input for the next call. `item_start` indexes the current case's (or the report request's)
        user message in `history`; `done` cases are complete."""
        ...

    def probe_view(self, history: list[ChatMessage], done: int) -> list[ChatMessage]: ...


class FullHistory:
    """CM0: the append-only history is the view."""

    name = "CM0"

    def view(self, history, item_start, done):
        return list(history)

    def probe_view(self, history, done):
        return list(history)


class OracleState:
    """O-state: system + true state after `done` cases + the current case's messages."""

    name = "O-state"

    def __init__(self, world: World, system: ChatMessageSystem):
        self.world, self.system = world, system

    def _state(self, done: int, then: str = "") -> ChatMessageUser:
        text = gen_f8.render_oracle_state(self.world, done)
        return ChatMessageUser(content=f"{text}\n\n{then}".strip())

    def view(self, history, item_start, done):
        return [self.system, self._state(done, history[item_start].text), *history[item_start + 1 :]]

    def probe_view(self, history, done):
        return [self.system, self._state(done)]


def view_policy(arm: str, world: World, system: ChatMessageSystem) -> ViewPolicy:
    if arm == "CM0":
        return FullHistory()
    if arm == "O-state":
        return OracleState(world, system)
    raise ValueError(f"session arm {arm!r} is not built yet ({SESSION_ARMS} are; the others need the ContextPolicy layer, CONTEXT_MANAGEMENT_AUDIT §6)")


class _Overflow(Exception):
    pass


@solver
def f8_session_agent(arm: str = "CM0", window: int = gen_f8.WINDOW, max_turns_per_item: int = 8, checkpoints: tuple[int, ...] = gen_f8.CHECKPOINTS) -> Solver:
    if arm not in SESSION_ARMS:
        raise ValueError(f"session arm {arm!r} is not built yet; built: {SESSION_ARMS}")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from .arms import load_world

        world = load_world(state.metadata["world_id"])
        n = len(world.tasks)
        rec = SessionRecorder(world)
        tools = list(build_session_tools(world, rec).values())
        system = ChatMessageSystem(content=gen_f8.system_prompt(world))
        history: list[ChatMessage] = [system, ChatMessageUser(content=gen_f8.start_message(world))]
        policy = view_policy(arm, world, system)
        model = get_model()
        probe_model = get_model(role=PROBE_ROLE, default=model)
        items, views, probes = [], [], []
        overflow_at = None

        async def call(item_start: int, done: int, position: int) -> tuple[int, bool]:
            """One generation on the policy's view; returns (view tokens, made tool calls)."""
            v = policy.view(history, item_start, done)
            vt = view_tokens(v)
            if vt > window:
                raise _Overflow(position)
            output = await model.generate(v, tools=tools)
            history.append(output.message)
            views.append({"item": position, "view_tokens": vt})
            if output.message.tool_calls:
                result = await execute_tools(history, tools)
                history.extend(result.messages)
                return vt, True
            return vt, False

        async def probe(k: int) -> None:
            msgs = [*policy.probe_view(history, k), ChatMessageUser(content=PROBE_PROMPT)]
            out = await probe_model.generate(msgs, config=GenerateConfig(response_schema=ResponseSchema(name="state_probe", json_schema=PROBE_SCHEMA, strict=True)))
            try:
                answer, error = json.loads(out.completion), None
            except (json.JSONDecodeError, TypeError) as e:
                answer, error = None, f"{type(e).__name__}: {e}"
            usage = out.usage
            probes.append({
                "k": k,
                "answer": answer,
                "error": error,
                "view_tokens": view_tokens(msgs),
                "scores": probe_score(answer if isinstance(answer, dict) else None, gen_f8.state_at(world, k)),
                "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens} if usage else None,
            })

        try:
            for task in world.tasks:
                pos, cid = task.tags["position"], task.tags["case_id"]
                rec.item = pos
                start = len(history)
                history.append(ChatMessageUser(content=task.prompt))
                first = decision = None
                generations, nudged = 0, False
                for _ in range(max_turns_per_item):
                    vt, acted = await call(start, pos - 1, pos)
                    generations += 1
                    first = vt if first is None else first
                    if rec.answered(task):
                        decision = vt
                        break
                    if not acted:
                        if nudged:
                            break
                        tool = "finish" if task.tags["kind"] == "ticket" else "submit_decision"
                        history.append(ChatMessageUser(content=f"Complete case {cid} with the tools, then call {tool} with case_id {cid}."))
                        nudged = True
                items.append({
                    "position": pos,
                    "case_id": cid,
                    "kind": task.tags["kind"],
                    "dependency": task.tags["dependency"],
                    "dependency_kinds": task.tags["dependency_kinds"],
                    "generations": generations,
                    "view_tokens_first": first,
                    "view_tokens_decision": decision,
                    "answered": decision is not None,
                    "success": item_success(world, task, rec.events),
                })
                if pos in checkpoints:
                    await probe(pos)
            rec.item = n + 1
            start = len(history)
            history.append(ChatMessageUser(content=gen_f8.REPORT_REQUEST))
            for _ in range(max_turns_per_item):
                _, acted = await call(start, n, n + 1)
                if rec.report is not None or not acted:
                    break
        except _Overflow as e:
            overflow_at = e.args[0]

        store().set(ITEMS, items)
        store().set(VIEWS, views)
        store().set(PROBES, probes)
        store().set(EVENTS, rec.events)
        store().set(REPORT, rec.report)
        store().set(OVERFLOW, overflow_at)
        store().set("arm", {"name": arm, "window": window, "max_turns_per_item": max_turns_per_item, "checkpoints": list(checkpoints)})
        state.messages = history
        return state

    return solve
