"""The single agent loop shared by every delivery arm.

Placement rule (DECISIONS D-005): per-query arms put their context in the system
prompt, compiled once, so it is part of the cached prefix. Per-step arms append a
fresh context message after the history on every step; it is replaced, never
accumulated, so the history prefix stays cacheable.
"""

from collections.abc import Awaitable, Callable

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import store

from ..kb.context import DeliveryArm
from ..worlds.env_tools import ANSWER, always_on, build_tools
from ..worlds.spec import TaskItem, World
from .step import exposed_tools, step_query

BASE_SYSTEM = (
    "You are an operations assistant at a retail company. Company knowledge is provided below. "
    "Follow policies and procedures exactly as written, use the tools to gather facts you are not given, "
    "and finish by calling the answer tool named in the task."
)
COMPILE_LOG = "compile_log"

ArmProvider = Callable[[World], Awaitable[DeliveryArm]]
WorldLoader = Callable[[str], World]


@solver
def kb_agent(arm_provider: ArmProvider, world_loader: WorldLoader, exposure: str = "retrieved", max_turns: int = 12) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        world = world_loader(state.metadata["world_id"])
        task = TaskItem(**state.metadata["task"])
        arm = await arm_provider(world)
        all_tools = build_tools(world, task)
        always = always_on(world)
        log: list[dict] = []

        async def compile_(query: str, step: int):
            ctx = await arm.compile(query, task)
            log.append(ctx.log_record(step))
            return ctx

        ctx = await compile_(task.prompt, 0)
        system = BASE_SYSTEM if arm.per_step else f"{BASE_SYSTEM}\n\n## Knowledge base\n{ctx.text}"
        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=task.prompt)]
        model = get_model()
        nudged, output = False, None
        for turn in range(max_turns):
            if arm.per_step and turn > 0:
                ctx = await compile_(step_query(task.prompt, messages), turn)
            tools = exposed_tools(all_tools, ctx, exposure, always)
            extra = [ChatMessageUser(content=f"## Knowledge base (refreshed for this step)\n{ctx.text}")] if arm.per_step else []
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
        store().set("turns_used", turn + 1)
        store().set("arm", {"name": arm.name, "per_step": arm.per_step, "exposure": exposure})
        state.messages = messages
        if output is not None:
            state.output = output
        return state

    return solve
