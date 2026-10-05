"""The context-length sweep's solver (CONTEXT_SWEEP.md): replay a shift's history, then one decision.

`sweep_probe` puts the world's replayed history (`gen_sweep.history`: system prompt, start message, every case before
the probe with the reference trajectory's calls and tool results, then the probe's message and, for `fresh` and
`memo`, its lookup) into the sample's messages and lets the model act on the probe with the F8 session tools, for at
most `max_generations` generations (default 2: one optional tool call, such as a lookup, then the decision; every
generation re-sends the whole context, so the cap bounds the cost of the largest sizes). It stops as soon as the probe
has its answer call (`SessionRecorder.answered`), or when a generation makes no tool call. There is no nudge.

Recorded in the store: the tool events (`env_f8.EVENTS`, all at the probe's position, so `scorers.session` reads
them) and `f8s_probe`: per generation the metered view tokens (the same meter as the world's `context_tokens`, plus
the nonce line), the tools called, the stop reason and Inspect's usage (input, output, reasoning and cached tokens as
the provider reports them, and the cost at the price table's flat rate). A call cut by a sample limit is recorded
with its error; the limit then ends the sample as Inspect decides.
"""

from inspect_ai.model import execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, store

from ..worlds import gen_sweep
from ..worlds.env_f8 import EVENTS, SessionRecorder, build_session_tools
from .cache_nonce import of as nonce_of
from .context_policy import tool_schema_tokens, usage_record, view_tokens

PROBE = "f8s_probe"
MAX_GENERATIONS = 2


@solver
def sweep_probe(max_generations: int = MAX_GENERATIONS) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from .arms import load_world

        world = load_world(state.metadata["world_id"])
        task = world.tasks[-1]
        rec = SessionRecorder(world)
        rec.item = task.tags["position"]
        tools = list(build_session_tools(world, rec).values())
        schema = tool_schema_tokens(tools)
        state.messages = gen_sweep.history(world, nonce_of(state.metadata))
        model = get_model()
        calls: list[dict] = []
        try:
            for _ in range(int(max_generations)):
                vt = view_tokens(state.messages) + schema
                try:
                    output = await model.generate(state.messages, tools=tools)
                except LimitExceededError as e:
                    calls.append({"view_tokens": vt, "error": f"LimitExceededError: {e}"[:300]})
                    raise
                called = [c.function for c in output.message.tool_calls or []]
                calls.append({"view_tokens": vt, "tools": called, "stop_reason": output.stop_reason, "usage": usage_record(output.usage)})
                state.messages.append(output.message)
                if called:
                    result = await execute_tools(state.messages, tools)
                    state.messages.extend(result.messages)
                if rec.answered(task) or not called:
                    break
        finally:
            store().set(EVENTS, rec.events)
            store().set(PROBE, {"calls": calls, "answered": rec.answered(task), "max_generations": int(max_generations)})
        return state

    return solve
