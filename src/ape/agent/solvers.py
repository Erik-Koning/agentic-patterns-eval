"""One solver per arm id, for the main study's task (`tasks/main.py`): the seam between the task and the arms.

Single-agent delivery arms (S1, S3s, S5, S6, S7, the APG/LightRAG arms) all run the one `kb_agent` loop; only the
delivery arm differs (`agent/arms.py`). Multi-agent and ensemble arms (BUILD_PLAN B2: S8k3, S9, M1, M1s, M1k, M2, M7)
register their solver factory in MULTI_AGENT_ARMS (`agent/multi/solvers.py`); every factory takes the same keywords
as `kb_agent` (exposure, max_turns, delivery).
"""

from collections.abc import Callable

from inspect_ai.solver import Solver

from .arms import arm_provider, load_world
from .kb_react import kb_agent
from .multi.solvers import ARM_FACTORIES

# arm id -> factory(arm, *, exposure, max_turns, delivery) -> Solver.
MULTI_AGENT_ARMS: dict[str, Callable[..., Solver]] = dict(ARM_FACTORIES)
# Arms in run_plan.yaml that are not single-agent delivery arms; until their factory is registered they are refused.
PLANNED_MULTI_AGENT_ARMS = ("S8k3", "S9", "M1", "M1s", "M1k", "M2", "M7")


def is_multi_agent(arm: str) -> bool:
    return arm in MULTI_AGENT_ARMS or arm in PLANNED_MULTI_AGENT_ARMS


def arm_solver(arm: str, *, exposure: str = "retrieved", max_turns: int, delivery: str = "push") -> Solver:
    """The solver that runs `arm`. Single-agent delivery arms get `kb_agent` over the arm's delivery."""
    if arm in MULTI_AGENT_ARMS:
        return MULTI_AGENT_ARMS[arm](arm, exposure=exposure, max_turns=max_turns, delivery=delivery)
    if arm in PLANNED_MULTI_AGENT_ARMS:
        raise NotImplementedError(f"arm {arm} is a multi-agent arm that is not built yet (BUILD_PLAN B2)")
    return kb_agent(arm_provider(arm), load_world, exposure=exposure, max_turns=max_turns, delivery=delivery)
