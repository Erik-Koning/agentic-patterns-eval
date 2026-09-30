"""The single agent loop shared by every delivery arm.

Delivery modes (GATE_PREREG §3):
- push: the harness compiles context and hands it to the model.
  Per-query arms put it in the system prompt once (cached prefix). Per-step arms append
  a fresh context message after the history each step; it is replaced, never accumulated
  (DECISIONS D-005).
- pull: the agent gets a `search_kb(query)` tool backed by the same arm's retriever, so it
  can follow references it has read (e.g. a policy ID named in a retrieved rule).
- both: push and pull together.

Tool exposure for a step uses every context delivered since the previous step: this
step's push plus any pulls made during the previous turn.
"""

import time
from collections.abc import Awaitable, Callable

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef, ToolParam, ToolParams
from inspect_ai.util import store

from ..kb.context import ContextResult, DeliveryArm
from ..llm.embeddings import EMBED_CONTEXT
from ..worlds.env_tools import ANSWER, always_on, build_tools
from ..worlds.spec import TaskItem, World
from .step import exposed_tools, step_query

BASE_SYSTEM = (
    "You are an operations assistant at a retail company. Company knowledge is provided below. "
    "Follow policies and procedures exactly as written, use the tools to gather facts you are not given, "
    "and finish by calling the answer tool named in the task."
)
PULL_ADDENDUM = "Use search_kb to look up company policies, procedures or anything a retrieved rule refers to."
COMPILE_LOG = "compile_log"
STEP_LOG = "step_log"
DELIVERY_MODES = ("push", "pull", "both")
EMPTY = ContextResult(text="", unit_ids=[], fact_ids=[])

ArmProvider = Callable[[World], Awaitable[DeliveryArm]]
WorldLoader = Callable[[str], World]


def merged(ctxs: list[ContextResult]) -> ContextResult:
    """One view of several deliveries, for tool exposure: texts joined, allowlists unioned."""
    ctxs = [c for c in ctxs if c is not None]
    if not ctxs:
        return EMPTY
    scopes = [c.tools for c in ctxs if c.tools is not None]
    tools = list(dict.fromkeys(t for s in scopes for t in s)) if scopes else None
    return ContextResult(text="\n\n".join(c.text for c in ctxs), unit_ids=[], fact_ids=[], tools=tools)


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
        turn = 0

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
                ctx = await compile_(query, "pull")
                pulled.append(ctx)
                return ctx.text or "No results."

            all_tools["search_kb"] = ToolDef(
                search_kb,
                name="search_kb",
                description="Search the company knowledge base (policies, exceptions, procedures, tools, announcements).",
                parameters=ToolParams(properties={"query": ToolParam(type="string", description="What to look up.")}, required=["query"]),
            )

        ctx = await compile_(task.prompt, "push") if push else EMPTY
        system = BASE_SYSTEM
        if push and not arm.per_step:
            system += f"\n\n## Knowledge base\n{ctx.text}"
        if pull:
            system += f"\n\n{PULL_ADDENDUM}"
        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=task.prompt)]
        model = get_model()
        nudged, output = False, None
        for turn in range(max_turns):
            if push and arm.per_step and turn > 0:
                ctx = await compile_(step_query(task.prompt, messages), "push")
            recent_pulls, pulled[:] = list(pulled), []
            tools = exposed_tools(all_tools, merged([ctx, *recent_pulls]), exposure, always)
            steps.append({"turn": turn, "exposed_tools": [t.name for t in tools]})
            extra = [ChatMessageUser(content=f"## Knowledge base (refreshed for this step)\n{ctx.text}")] if push and arm.per_step else []
            output = await model.generate(messages + extra, tools=tools)
            messages.append(output.message)
            if output.message.tool_calls:
                result = await execute_tools(messages, tools)
                messages.extend(result.messages)
                if store().get(ANSWER) is not None:
                    break
            elif nudged:
                break
            else:
                messages.append(ChatMessageUser(content=f"Use the tools to complete the task, then call {task.answer_tool}."))
                nudged = True

        store().set(COMPILE_LOG, log)
        store().set(STEP_LOG, steps)
        store().set("turns_used", turn + 1)
        store().set("arm", {"name": arm.name, "per_step": arm.per_step, "exposure": exposure, "delivery": delivery})
        state.messages = messages
        if output is not None:
            state.output = output
        return state

    return solve
