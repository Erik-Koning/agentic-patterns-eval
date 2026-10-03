"""Main-study task: one Inspect task per (family, level, arm) for F1 breadth aggregation and F2 dependency chains
(worlds/gen_registry.py), and F3 tool load and F7 policies (the gate's families).

    inspect eval src/ape/tasks/main.py@main_study -T family=F1 -T level=32 -T arm=S1 --model ...

The same scorers and tool-exposure rule as the gate (`tasks/gate.py`). The solver comes from `agent.solvers.arm_solver`:
single-agent delivery arms run the gate's agent loop, multi-agent and ensemble arms (S8k3, S9, M1, M1s, M1k, M2, M7)
their own solver. S5 is the KG arm the gate selected (`agent.arms.kg_arm_name`, APE_KG_ARM). The turn cap scales with
the registry knob (`gen_registry.turn_cap`, never below APE_MAX_TURNS): at N = 32 or k = 10 the gate's 12 turns would
make cap hits, not the arm, decide. F3 and F7 keep the gate's cap.

`faults=True` (Study D, Tier B) enables each task's fault (`tags["fault"]`) by setting `setup["faults_enabled"]` in
the sample's task; `env_tools` applies a fault only then. F1's `tags["subtasks"]` holds the decomposition a gold-knowing
mock orchestrator uses.

`plan_cell`, `group`, `seed_base` and `skip_worlds` are as in the gate task: a study orchestrator passes them, and they
are task args only when passed.
"""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from ..agent.arms import KG_ARM_ENV
from ..agent.solvers import arm_solver, is_multi_agent
from ..config import Config
from ..scorers.success import delivered_evidence, task_success
from ..scorers.taxonomy import error_analysis
from ..worlds import gen_f3, gen_f7
from ..worlds.gen_registry import LEVELS as REGISTRY_LEVELS
from ..worlds.gen_registry import turn_cap
from .gate import ARM_KNOB_PREFIXES, PULLABLE, gate_samples

LEVELS = {**REGISTRY_LEVELS, "F3": tuple(gen_f3.LEVELS), "F7": tuple(gen_f7.LEVELS)}


@task
def main_study(
    family: str = "F1",
    level: str = "2",
    split: str = "dev",
    arm: str = "S1",
    exposure: str = "retrieved",
    delivery: str = "push",
    limit_worlds: int | None = None,
    faults: bool = False,
    plan_cell: str | None = None,
    group: str | None = None,
    seed_base: int | None = None,
    skip_worlds: int | None = None,
) -> Task:
    if family not in LEVELS or level not in LEVELS[family]:
        raise ValueError(f"main_study covers {LEVELS}; got {family}-{level}")
    if delivery != "push" and arm not in PULLABLE and arm != "S5":
        raise ValueError(f"{arm} has no retriever for delivery={delivery}; pull applies to {sorted(PULLABLE | {'S5'})}")
    if faults and family != "F2":
        raise ValueError("environment faults are defined for F2 only")
    cfg = Config()
    samples = gate_samples(family, level, split, limit_worlds=limit_worlds, seed_base=seed_base, skip_worlds=skip_worlds)
    if faults:
        for s in samples:
            s.metadata["task"]["setup"] = {**s.metadata["task"]["setup"], "faults_enabled": True}
    max_turns = max(cfg.max_turns, turn_cap(family, level)) if family in REGISTRY_LEVELS else cfg.max_turns
    knobs = {k: v for k, v in sorted(os.environ.items()) if k.startswith(ARM_KNOB_PREFIXES) or k == KG_ARM_ENV}
    return Task(
        dataset=MemoryDataset(samples, name=f"{family}-{level}-{split}"),
        solver=arm_solver(arm, exposure=exposure, max_turns=max_turns, delivery=delivery),
        scorer=[task_success(), delivered_evidence(), error_analysis()],
        metadata={
            "arm": arm,
            "multi_agent": is_multi_agent(arm),
            "exposure": exposure,
            "delivery": delivery,
            "family": family,
            "level": level,
            "split": split,
            "faults": faults,
            "max_turns": max_turns,
            "knobs": knobs,
        }
        | ({"plan_cell": plan_cell, "group": group} if plan_cell or group else {}),
    )
