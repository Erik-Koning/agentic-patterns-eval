"""The single agent loop shared by every delivery arm.

Delivery modes (GATE_PREREG §3):
- push: the harness compiles context and hands it to the model.
  Per-query arms put it in the system prompt once (cached prefix). Per-step arms deliver a fresh
  context each step at the end of the model's input; it is replaced, never accumulated (DECISIONS D-005).
  After turn 0 it is appended to the last tool result in the copy sent to the model (`step_view`), not
  added as a new user message: a user message after the model's tool calls would end its reasoning turn,
  and OpenAI drops the reasoning items that precede the last user message, so every step would start
  cold. The logged history never contains it.
- pull: the agent gets a `search_kb(query)` tool backed by the same arm's retriever, so it
  can follow references it has read (e.g. a policy ID named in a retrieved rule).
- both: push and pull together.

Tool exposure for a step uses every context delivered since the previous step: this
step's push plus any pulls made during the previous turn.

Robustness rules for real models:
- `search_kb` strips its query and caps it at SEARCH_QUERY_MAX_TOKENS; an empty query or a failed
  compile comes back to the model as a tool error and never kills the sample, except a sample limit (tokens,
  cost) that the compile's own kg call hits: that ends the sample at once, as anywhere else. Its output limit
  (`search_kb_max_output`) is sized from the largest configured context budget, so the model sees the
  whole delivery; the compile log records the bytes the model saw and any truncation.
- A text-only reply gets one nudge per text-only streak (at most MAX_NUDGES per sample); a second
  text-only reply in a row ends the sample.
- The compile and step logs, the turn count and the messages are written even when a sample limit
  (tokens, cost, messages) cuts the loop short.
"""

import os
import time
from collections.abc import Awaitable, Callable

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentText, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolError, ToolParam, ToolParams
from inspect_ai.util import LimitExceededError, store

from ..config import Config
from ..kb.context import ContextResult, DeliveryArm
from ..llm.embeddings import EMBED_CONTEXT
from ..llm.tokens import truncate_to_tokens
from ..worlds.env_tools import ANSWER, always_on, build_tools
from ..worlds.spec import TaskItem, World
from .cache_nonce import of as nonce_of
from .cache_nonce import with_nonce
from .step import exposed_tools, step_query

BASE_SYSTEM = (
    "You are an operations assistant at a retail company. "
    "Follow policies and procedures exactly as written, use the tools to gather facts you are not given, "
    "and finish by calling the answer tool named in the task."
)
# What the system prompt says about where company knowledge comes from, per delivery (L11).
KNOWLEDGE_NOTE = {
    "system": "Company knowledge is provided below.",
    "per_step": "Company knowledge relevant to the current step is provided at the end of the latest message and is refreshed every step.",
    "pull": "Use search_kb to look up company policies, procedures or anything a retrieved rule refers to.",
}
STEP_KB_HEADER = "## Knowledge base (refreshed for this step)"
COMPILE_LOG = "compile_log"
SEARCH_ERRORS = "search_errors"  # failed search_kb calls (empty query, compile error); never in the compile log
STEP_LOG = "step_log"
DELIVERY_MODES = ("push", "pull", "both")
EMPTY = ContextResult(text="", unit_ids=[], fact_ids=[])
MAX_NUDGES = 3  # per sample; one per text-only streak
SEARCH_QUERY_MAX_TOKENS = 1000  # well under the embedding model's 8,192-token input limit
KB_BYTES_PER_TOKEN = 6  # o200k English runs ~4-5 bytes per token; 6 leaves room for IDs and punctuation
KB_OUTPUT_MARGIN = 1.25
INSPECT_DEFAULT_TOOL_OUTPUT = 16 * 1024

ArmProvider = Callable[[World], Awaitable[DeliveryArm]]
WorldLoader = Callable[[str], World]


def search_kb_max_output(cfg: Config | None = None) -> int:
    """`search_kb`'s output limit in bytes: the largest context any arm can deliver per compile (S3s budget,
    APG budget, or LightRAG's max_total_tokens, which defaults to 4x its budget), at KB_BYTES_PER_TOKEN, with a
    margin; never below Inspect's 16 KiB default."""
    cfg = cfg or Config()
    lgr_total = int(os.environ.get("APE_LGR_TOTAL_TOKENS", "").strip() or cfg.lgr_budget_tokens * 4)
    tokens = max(cfg.s3s_budget_tokens, cfg.apg_budget_tokens, lgr_total)
    return max(INSPECT_DEFAULT_TOOL_OUTPUT, int(tokens * KB_BYTES_PER_TOKEN * KB_OUTPUT_MARGIN))


def _clip_bytes(text: str, limit: int) -> tuple[str, int, bool]:
    """`text` cut to at most `limit` UTF-8 bytes, with a visible marker when cut; (text, raw bytes, truncated)."""
    raw = text.encode()
    if len(raw) <= limit:
        return text, len(raw), False
    marker = f"\n[... knowledge base result truncated: {limit} of {len(raw)} bytes shown]"
    keep = raw[: max(0, limit - len(marker.encode()))].decode("utf-8", "ignore")
    return keep + marker, len(raw), True


def merged(ctxs: list[ContextResult]) -> ContextResult:
    """One view of several deliveries, for tool exposure: texts joined, allowlists unioned."""
    ctxs = [c for c in ctxs if c is not None]
    if not ctxs:
        return EMPTY
    scopes = [c.tools for c in ctxs if c.tools is not None]
    tools = list(dict.fromkeys(t for s in scopes for t in s)) if scopes else None
    return ContextResult(text="\n\n".join(c.text for c in ctxs), unit_ids=[], fact_ids=[], tools=tools)


def system_prompt(push: bool, pull: bool, per_step: bool, kb_text: str = "") -> str:
    """The system prompt for a delivery mode: what the model is told about where knowledge comes from, plus the
    knowledge itself for per-query push arms."""
    notes = ([KNOWLEDGE_NOTE["per_step" if per_step else "system"]] if push else []) + ([KNOWLEDGE_NOTE["pull"]] if pull else [])
    system = " ".join([BASE_SYSTEM, *notes])
    if push and not per_step:
        system += f"\n\n## Knowledge base\n{kb_text}"
    return system


def step_view(messages: list[ChatMessage], kb_text: str) -> list[ChatMessage]:
    """The model's input for a per-step call: the history plus this step's knowledge at the very end.

    After a tool step the knowledge is appended to the last tool result in a copy (the history stays clean), so
    no user message follows the model's reasoning and tool calls within a turn. At turn 0, or after a nudge
    (the last message is the user's), it goes in a user message of its own, as before."""
    block = f"{STEP_KB_HEADER}\n{kb_text}"
    last = messages[-1] if messages else None
    if isinstance(last, ChatMessageTool):
        if isinstance(last.content, str):
            content = f"{last.content}\n\n{block}"
        else:
            content = [*last.content, ContentText(text=f"\n\n{block}")]
        return [*messages[:-1], last.model_copy(update={"content": content})]
    return [*messages, ChatMessageUser(content=block)]


@solver
def kb_agent(
    arm_provider: ArmProvider,
    world_loader: WorldLoader,
    exposure: str = "retrieved",
    max_turns: int = 12,
    delivery: str = "push",
) -> Solver:
    if delivery not in DELIVERY_MODES:
        raise ValueError(f"delivery must be one of {DELIVERY_MODES}")
    push, pull = delivery in ("push", "both"), delivery in ("pull", "both")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        world = world_loader(state.metadata["world_id"])
        task = TaskItem(**state.metadata["task"])
        arm = await arm_provider(world)
        all_tools = build_tools(world, task)
        always = always_on(world) + (["search_kb"] if pull else [])
        log: list[dict] = []
        steps: list[dict] = []
        pulled: list[ContextResult] = []
        errors: list[dict] = []
        limit_hit: list[LimitExceededError] = []  # a sample limit a search_kb compile hit (its kg call), re-raised below
        turn = 0
        max_output = search_kb_max_output()

        async def compile_(query: str, source: str) -> ContextResult:
            token = EMBED_CONTEXT.set({"arm": arm.name, "world": world.id, "sample": str(state.sample_id), "epoch": state.epoch, "source": source})
            t0 = time.perf_counter()
            try:
                ctx = await arm.compile(query, task)
            finally:
                EMBED_CONTEXT.reset(token)
            log.append({**ctx.log_record(turn), "source": source, "compile_ms": round((time.perf_counter() - t0) * 1000, 1)})
            return ctx

        if pull:
            async def search_kb(query: str) -> str:
                q = truncate_to_tokens(str(query or "").strip(), SEARCH_QUERY_MAX_TOKENS)
                if not q:
                    errors.append({"step": turn, "error": "empty query"})
                    raise ToolError("Empty query: pass a non-empty search query.")
                try:
                    ctx = await compile_(q, "pull")
                except LimitExceededError as e:  # Inspect turns a tool's limit error into a tool error: the loop re-raises it
                    limit_hit.append(e)
                    raise
                except Exception as e:  # a retrieval failure is the model's problem to route around, not a harness error
                    errors.append({"step": turn, "error": f"{type(e).__name__}: {e}"[:500]})
                    raise ToolError(f"search_kb failed ({type(e).__name__}); try a different query.") from e
                pulled.append(ctx)
                text, raw_bytes, truncated = _clip_bytes(ctx.text or "No results.", max_output)
                log[-1].update({"query_tokens_capped": q != str(query or "").strip(), "raw_bytes": raw_bytes, "seen_bytes": len(text.encode()), "truncated": truncated, "max_output": max_output})
                return text

            all_tools["search_kb"] = ToolDef(
                search_kb,
                name="search_kb",
                description="Search the company knowledge base (policies, exceptions, procedures, tools, announcements).",
                parameters=ToolParams(properties={"query": ToolParam(type="string", description="What to look up.")}, required=["query"]),
                max_output=max_output,
            )

        ctx = await compile_(task.prompt, "push") if push else EMPTY
        # The history lives in state.messages itself: Inspect enforces the sample's message limit on its appends, and
        # when a limit stops the loop the logged conversation is exactly what ran.
        # A study run's per-run cache nonce (`cache_nonce`) leads the system prompt; without one it is unchanged.
        system = with_nonce(system_prompt(push, pull, arm.per_step, ctx.text), nonce_of(state.metadata))
        state.messages = [ChatMessageSystem(content=system), ChatMessageUser(content=task.prompt)]
        messages = state.messages
        model = get_model()
        nudges, streak, output = 0, False, None
        try:
            for turn in range(max_turns):
                if push and arm.per_step and turn > 0:
                    ctx = await compile_(step_query(task.prompt, messages), "push")
                recent_pulls, pulled[:] = list(pulled), []
                tools = exposed_tools(all_tools, merged([ctx, *recent_pulls]), exposure, always)
                steps.append({"turn": turn, "exposed_tools": [t.name for t in tools]})
                view = step_view(messages, ctx.text) if push and arm.per_step else messages
                output = await model.generate(view, tools=tools)
                messages.append(output.message)
                if output.message.tool_calls:
                    streak = False
                    result = await execute_tools(messages, tools)
                    if limit_hit:  # the sample's limit was reached inside a tool: end the sample, as Inspect does elsewhere
                        raise limit_hit[0]
                    messages.extend(result.messages)
                    if store().get(ANSWER) is not None:
                        break
                elif streak or nudges >= MAX_NUDGES:
                    break
                else:
                    messages.append(ChatMessageUser(content=f"Use the tools to complete the task, then call {task.answer_tool}."))
                    nudges, streak = nudges + 1, True
        finally:
            # Written even when a sample limit cuts the loop short, so the logs show what happened.
            store().set(COMPILE_LOG, log)
            store().set(STEP_LOG, steps)
            store().set(SEARCH_ERRORS, errors)
            store().set("turns_used", turn + 1)
            store().set("nudges", nudges)
            store().set("arm", {"name": arm.name, "per_step": arm.per_step, "exposure": exposure, "delivery": delivery})
            if output is not None:
                state.output = output
        return state

    return solve
