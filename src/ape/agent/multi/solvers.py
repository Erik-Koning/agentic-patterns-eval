"""The main study's multi-agent and ensemble arms as Inspect solvers, one factory per arm id (`ARM_FACTORIES`, registered
in `agent.solvers.MULTI_AGENT_ARMS`). Each factory takes `kb_agent`'s keywords (exposure, max_turns, delivery); every
arm runs push delivery only, as `tasks.main` already requires for arms without a retriever.

| Arm | Switches (differences from S1) | Agents | Delivery (every agent) |
|---|---|---|---|
| S9 | DEC=1, ISO=0, CTRL=dynamic | one planner-executor: `plan` (forced first) + the task's tools | S1 (MONO) |
| M1 | DEC=1, ISO=1, CONC=1 | orchestrator (`plan`, `delegate`, answer) + 3 identical workers, concurrent | S1 (MONO) |
| M1s | DEC=1, ISO=1, CONC=0 | as M1, workers one after another | S1 (MONO) |
| M1k | M1 + DEL=KGs | as M1 | S5: the gate-selected KG arm (`arms.kg_arm_name`) |
| M2 | M1k + SPEC=1 | as M1k with 3 specialists (`specialists.py`) | S5 |
| M7 | ENS=3, COMM=real | 3 council members, 2 critique rounds, a chair | S1 (MONO) |
| S8k3 | ENS=3 | 3 independent S1 attempts (`kb_agent`), vote, aggregator on ties | S1 (MONO) |

**DEL is arm-wide.** Every agent of an arm, the orchestrator, chair and aggregator included, gets knowledge through the
arm's delivery, so DEL stays one switch: M1 to M1k changes it for all agents at once, and M1k to M2 changes only SPEC.
(The alternative, an orchestrator without knowledge, would make S9 to M1s change two things: where subtasks execute
and whether the planner holds the knowledge.)

**Tuning knobs** (`knobs.py`, BUILD_PLAN B3): each factory reads its arm's `APE_MAS_<ARM>_*` variables when it is built
(the prompt variant of the role notes, and the clip on a worker's result or a member's rationale); M1s, M1k and M2 read
M1's, so the orchestrator chain runs one protocol variant and each of its steps stays one switch (BUILD_REVIEW A-1).

**Records** (sample store): `mas_switches` (the brief's full switch vector, §4.1), `mas_params` (fixed structure:
workers, rounds, k, turn caps, and how parallel units are scheduled; and the arm's knobs: `prompt_variant`, the hash
of its texts `prompt_sha`, and `result_clip_tokens` where one agent reads another's text), `mas_agents` (per agent:
id, role, parent, turns, nudges, how it stopped, its tool calls in order, its result or error), `mas_accounting` (per
agent and role: calls, input / output / reasoning / cache-read / cache-write / total tokens, model seconds, wall-clock;
`core`), and per arm `mas_plan` and `mas_rounds` (planner and orchestrator arms), `mas_council` (M7) or
`mas_ensemble` (S8k3).
The single-agent keys are written too (`compile_log` over every agent, so `delivered_evidence` scores the union of
what any agent was delivered; `step_log`; `turns_used` and `nudges` of the top agent: the planner, the orchestrator,
the chair or the winning attempt).

**Where each arm applies.** Every arm runs on F1, F2, F3 and F7, except M2 (no specialization on F2: `specialists`) and
M7 (its members attempt the task in parallel, and F3's tools change the environment). A sample outside these fails
with a clear error.
"""

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ...kb.context import DeliveryArm
from ...worlds.env_tools import always_on, build_tools
from ..arms import arm_provider, kg_arm_name, load_world
from ..kb_react import kb_agent
from . import knobs as K
from . import prompts as P
from .core import Delivery, Team, react_loop
from .primitives import COUNCIL_K, CRITIQUE_ROUNDS, ENSEMBLE_K, TeamConfig, plan_tool, run_council, run_ensemble, run_orchestrator, worker_tools
from .specialists import N_WORKERS, specialists

S1_SWITCHES = {"DEL": "MONO", "DEC": 0, "ISO": 0, "CONC": 0, "ENS": 1, "SPEC": 0, "STATE": "none", "CTRL": "dynamic", "COMM": "none", "HET": 0, "CMP": 0}
REGISTRY = {  # differences from S1 (brief §4.2); "KG" resolves to KGs or KGq by the KG arm's schedule
    "S9": {"DEC": 1, "ISO": 0, "CTRL": "dynamic"},
    "M1": {"DEC": 1, "ISO": 1, "CONC": 1},
    "M1s": {"DEC": 1, "ISO": 1, "CONC": 0},
    "M1k": {"DEC": 1, "ISO": 1, "CONC": 1, "DEL": "KG"},
    "M2": {"DEC": 1, "ISO": 1, "CONC": 1, "DEL": "KG", "SPEC": 1},
    "M7": {"ENS": COUNCIL_K, "COMM": "real"},
    "S8k3": {"ENS": ENSEMBLE_K},
}
DELIVERY_ARM = {"S9": "S1", "M1": "S1", "M1s": "S1", "M1k": "S5", "M2": "S5", "M7": "S1", "S8k3": "S1"}
NOT_ON = {"M2": ("F2", "F5"), "M7": ("F3",)}


def switch_vector(arm: str, delivery: DeliveryArm) -> dict:
    sw = {**S1_SWITCHES, **REGISTRY[arm]}
    if sw["DEL"] == "KG":
        sw["DEL"] = "KGs" if delivery.per_step else "KGq"
    return sw | {"delivery_arm": delivery.name}


def _params(arm: str, max_turns: int, knobs: K.Knobs) -> dict:
    p: dict = {"max_turns": max_turns}
    if arm in K.KNOB_ARM:
        p |= {"prompt_variant": knobs.prompt, "prompt_sha": knobs.notes.sha()}
        if K.KNOB_ARM[arm] in K.CLIP_ARMS:
            p["result_clip_tokens"] = knobs.clip
    if REGISTRY[arm].get("ISO"):
        p |= {"n_workers": N_WORKERS, "worker_turns": max_turns, "workers_scheduled": "concurrent" if REGISTRY[arm]["CONC"] else "serial"}
    if arm == "M7":
        p |= {"council_k": COUNCIL_K, "critique_rounds": CRITIQUE_ROUNDS, "members_scheduled": "concurrent"}
    if arm == "S8k3":
        p |= {"ensemble_k": ENSEMBLE_K, "attempts_scheduled": "concurrent", "vote": "plurality; LLM aggregator on ties"}
    if arm in DELIVERY_ARM and DELIVERY_ARM[arm] == "S5":
        p["kg_arm"] = kg_arm_name()
    return p


def _check(arm: str, delivery: str) -> K.Knobs:
    """The arm and delivery, checked when the solver is built, and the arm's knobs as the environment sets them then."""
    if arm not in REGISTRY:
        raise ValueError(f"no multi-agent arm {arm!r}; arms: {sorted(REGISTRY)}")
    if delivery != "push":
        raise ValueError(f"{arm} runs push delivery only (got {delivery!r})")
    return K.resolve(arm)


async def _start(team: Team) -> Delivery:
    arm = team.arm
    if team.world.family in NOT_ON.get(arm, ()):
        raise ValueError(f"{arm} is not defined on {team.world.family} (agent/multi/solvers.py: where each arm applies)")
    delivery = Delivery(await arm_provider(DELIVERY_ARM[arm])(team.world), team)
    team.switches = switch_vector(arm, delivery.arm)
    return delivery


@solver
def plan_execute(arm: str = "S9", exposure: str = "retrieved", max_turns: int = 12, delivery: str = "push") -> Solver:
    """S9: plan, then execute in the same context, replanning allowed."""
    knobs = _check(arm, delivery)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        team = Team(state, arm, {}, _params(arm, max_turns, knobs))
        dlv = None
        try:
            dlv = await _start(team)
            task = team.task
            agent = team.top = team.agent("agent", "planner")
            tools = build_tools(team.world, task)
            tools["plan"] = plan_tool(agent, team.records.setdefault("mas_plan", []))
            async with team.running(agent):
                ctx = await dlv.compile(agent, task.prompt, 0)
                note = knobs.notes.s9.format(tools=", ".join(worker_tools(team)[0]), answer_tool=task.answer_tool)
                state.messages = [dlv.system(ctx), ChatMessageUser(content=f"{task.prompt}\n\n{note}")]
                await react_loop(team, agent, state.messages, delivery=dlv, ctx=ctx, query=task.prompt, tools=tools, always=[*always_on(team.world), "plan"],
                                 exposure=exposure, max_turns=max_turns, done=team.answered, nudge=P.NUDGE_TASK.format(answer_tool=task.answer_tool),
                                 first_tool="plan")
        finally:
            team.finish(dlv, exposure)
        return state

    return solve


@solver
def orchestrated(arm: str = "M1", exposure: str = "retrieved", max_turns: int = 12, delivery: str = "push") -> Solver:
    """M1, M1s, M1k, M2: an orchestrator delegating to three workers in fresh contexts."""
    knobs = _check(arm, delivery)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        team = Team(state, arm, {}, _params(arm, max_turns, knobs))
        dlv = None
        try:
            dlv = await _start(team)
            specs = specialists(team.world) if REGISTRY[arm].get("SPEC") else None
            if specs is not None:
                team.records["mas_specialists"] = [{"name": s.name, "covers": list(s.covers), "tools": list(s.tools) if s.tools else None} for s in specs]
            cfg = TeamConfig(dlv, exposure, max_turns, worker_turns=max_turns, concurrent=bool(REGISTRY[arm]["CONC"]), specialists=specs,
                             notes=knobs.notes, clip_tokens=knobs.clip)
            await run_orchestrator(team, cfg)
        finally:
            team.finish(dlv, exposure)
        return state

    return solve


@solver
def council(arm: str = "M7", exposure: str = "retrieved", max_turns: int = 12, delivery: str = "push") -> Solver:
    """M7: k members, critique rounds, a chair."""
    knobs = _check(arm, delivery)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        team = Team(state, arm, {}, _params(arm, max_turns, knobs))
        dlv = None
        try:
            dlv = await _start(team)
            await run_council(team, TeamConfig(dlv, exposure, max_turns, worker_turns=max_turns, notes=knobs.notes, clip_tokens=knobs.clip))
        finally:
            team.finish(dlv, exposure)
        return state

    return solve


@solver
def self_consistency(arm: str = "S8k3", exposure: str = "retrieved", max_turns: int = 12, delivery: str = "push") -> Solver:
    """S8k3: k independent S1 attempts (the gate's own loop), majority vote, an LLM aggregator on ties."""
    knobs = _check(arm, delivery)
    attempt = kb_agent(arm_provider("S1"), load_world, exposure=exposure, max_turns=max_turns, delivery="push")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        team = Team(state, arm, {}, _params(arm, max_turns, knobs))
        dlv = None
        try:
            dlv = await _start(team)
            await run_ensemble(team, TeamConfig(dlv, exposure, max_turns, worker_turns=max_turns), attempt, generate)
        finally:
            team.finish(dlv, exposure)
        return state

    return solve


ARM_FACTORIES = {"S9": plan_execute, "M1": orchestrated, "M1s": orchestrated, "M1k": orchestrated, "M2": orchestrated, "M7": council, "S8k3": self_consistency}
