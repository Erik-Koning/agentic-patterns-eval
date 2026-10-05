"""The context-length sweep's solver (CONTEXT_SWEEP.md): replay a shift's history, then one decision.

`sweep_probe` puts the world's replayed history (`gen_sweep.history`: system prompt, start message, every case before
the probe with the reference trajectory's calls and tool results, then the probe's message and, for `fresh` and
`memo`, its lookup) into the sample's messages and lets the model act on the probe with the F8 session tools, for at
most `max_generations` generations (default 2: one optional tool call, such as a lookup, then the decision; every
generation re-sends the whole context, so the cap bounds the cost of the largest sizes). It stops as soon as the probe
has its answer call (`SessionRecorder.answered`), or when a generation makes no tool call. There is no nudge.

**Output cap that fits the window.** The model's window holds input and output together, so each generation asks for
at most `output_cap`: the task's ceiling (`max_output_tokens`), lowered so that the input as the provider may count it
(the metered tokens x SAFETY, plus PER_MESSAGE per message and a MARGIN) plus the output fit `window_tokens`, and never
below MIN_OUTPUT_TOKENS. At 960K (about 4,000 messages) that is about 43K instead of 64K.

**A request the provider will not take is "not measured", never a wrong answer.**
- `over_limit`: the request did not fit. Inspect turns OpenAI's `context_length_exceeded` into an empty output with
  stop reason `model_length`, which is caught here (`refused_for_size`), and so is any other 400 whose message names a
  size or limit (too many input items, too long, maximum, tokens).
- `rejected`: any other 400 (a request the provider refuses for another reason: a harness problem to look at).
Both end the sample at once without an Inspect error, so a deterministic refusal is not retried and never fails the
tier's eval set (the other sizes still run); the scorer labels them and the analysis keeps them out of every rate.
Transient errors (rate limits, connection, 5xx) still raise, so Inspect retries the sample.

Recorded in the store: the tool events (`env_f8.EVENTS`, all at the probe's position, so `scorers.session` reads
them) and `f8s_probe`: per generation the metered view tokens (the same meter as the world's `context_tokens`, plus
the nonce line), the output cap sent, the tools called, the stop reason and Inspect's usage (input, output, reasoning
and cached tokens as the provider reports them, and the cost at the price table's flat rate), plus `over_limit` and
`rejected`. A call cut by a sample limit is recorded with its error; the limit then ends the sample as Inspect decides.
"""

import math

import openai
from inspect_ai.model import GenerateConfig, ModelOutput, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, store

from ..worlds import gen_sweep
from ..worlds.env_f8 import EVENTS, SessionRecorder, build_session_tools
from .cache_nonce import of as nonce_of
from .context_policy import tool_schema_tokens, usage_record, view_tokens

PROBE = "f8s_probe"
MAX_GENERATIONS = 2
WINDOW_TOKENS = 1_050_000  # GPT-6 (Luna, Sol, Astra): input and output share it
MAX_OUTPUT_TOKENS = 64_000  # the per-generation ceiling (reasoning included)
MIN_OUTPUT_TOKENS = 8_000
SAFETY = 1.03  # the provider's count may run a little above the o200k meter
PER_MESSAGE = 4  # chat-format tokens per message the meter does not count
MARGIN = 2_000
SIZE_HINTS = ("context", "too long", "too large", "too many", "maximum", "exceed", "tokens", "input items", "limit")


def output_cap(view: int, n_messages: int, window: int = WINDOW_TOKENS, ceiling: int = MAX_OUTPUT_TOKENS) -> int:
    """The output tokens to ask for so that input (as the provider may count it) plus output fit the window."""
    room = window - math.ceil(view * SAFETY) - PER_MESSAGE * n_messages - MARGIN
    return max(MIN_OUTPUT_TOKENS, min(int(ceiling), room))


def refused_for_size(output: ModelOutput) -> bool:
    """The request did not fit the window: stop reason `model_length` with no tool call. Inspect renders OpenAI's
    `context_length_exceeded` so (an empty output whose text is the error); a generation that ran out of window while
    writing ends the same way, which with the output cap means the provider counted the input above our estimate.
    Either way the model could not answer at this size."""
    return output.stop_reason == "model_length" and not output.message.tool_calls


@solver
def sweep_probe(max_generations: int = MAX_GENERATIONS, max_output_tokens: int = MAX_OUTPUT_TOKENS, window_tokens: int = WINDOW_TOKENS) -> Solver:
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
        over_limit, rejected = False, None
        try:
            for _ in range(int(max_generations)):
                vt = view_tokens(state.messages) + schema
                cap = output_cap(vt, len(state.messages), int(window_tokens), int(max_output_tokens))
                try:
                    output = await model.generate(state.messages, tools=tools, config=GenerateConfig(max_tokens=cap))
                except LimitExceededError as e:
                    calls.append({"view_tokens": vt, "max_output_tokens": cap, "error": f"LimitExceededError: {e}"[:300]})
                    raise
                except openai.BadRequestError as e:  # deterministic: retrying the same request cannot help
                    text = f"{type(e).__name__}: {e}"[:300]
                    over_limit = any(h in str(e).lower() for h in SIZE_HINTS)
                    rejected = None if over_limit else text
                    calls.append({"view_tokens": vt, "max_output_tokens": cap, "error": text})
                    break
                if refused_for_size(output):
                    over_limit = True
                    calls.append({"view_tokens": vt, "max_output_tokens": cap, "stop_reason": output.stop_reason, "error": output.completion[:300], "usage": usage_record(output.usage)})
                    break
                called = [c.function for c in output.message.tool_calls or []]
                calls.append({"view_tokens": vt, "max_output_tokens": cap, "tools": called, "stop_reason": output.stop_reason, "usage": usage_record(output.usage)})
                state.messages.append(output.message)
                if called:
                    result = await execute_tools(state.messages, tools)
                    state.messages.extend(result.messages)
                if rec.answered(task) or not called:
                    break
        finally:
            store().set(EVENTS, rec.events)
            store().set(PROBE, {"calls": calls, "answered": rec.answered(task), "max_generations": int(max_generations), "over_limit": over_limit, "rejected": rejected})
        return state

    return solve
