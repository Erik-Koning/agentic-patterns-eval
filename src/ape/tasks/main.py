"""Main-study task for the registry families (F1 breadth aggregation, F2 dependency chains; worlds/gen_registry.py).

    inspect eval src/ape/tasks/main.py@main_study -T family=F1 -T level=32 -T arm=S1 --model ...

The same single agent loop, arms, tool-exposure rule and scorers as the gate (`tasks/gate.py`); only the dataset,
the turn cap and the optional environment fault differ. The turn cap scales with the knob (`gen_registry.turn_cap`,
never below APE_MAX_TURNS): at N = 32 or k = 10 the gate's 12 turns would make cap hits, not the arm, decide.

`faults=True` (Study D, Tier B) enables each task's fault (`tags["fault"]`) by setting `setup["faults_enabled"]` in
the sample's task; `env_tools` applies a fault only then. Multi-agent arms (S8, S9, M1, M7, ...) are not built yet:
F1's `tags["subtasks"]` holds the decomposition they will use.
"""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from ..agent.arms import arm_provider, load_world
from ..agent.kb_react import kb_agent
from ..config import Config
from ..scorers.success import delivered_evidence, task_success
from ..scorers.taxonomy import error_analysis
from ..worlds.gen_registry import LEVELS, turn_cap
from .gate import ARM_KNOB_PREFIXES, PULLABLE, gate_samples


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
) -> Task:
    if family not in LEVELS or level not in LEVELS[family]:
        raise ValueError(f"main_study covers {LEVELS}; got {family}-{level}")
    if delivery != "push" and arm not in PULLABLE:
        raise ValueError(f"{arm} has no retriever for delivery={delivery}; pull applies to {sorted(PULLABLE)}")
    if faults and family != "F2":
        raise ValueError("environment faults are defined for F2 only")
    cfg = Config()
    samples = gate_samples(family, level, split, limit_worlds=limit_worlds)
    if faults:
        for s in samples:
            s.metadata["task"]["setup"] = {**s.metadata["task"]["setup"], "faults_enabled": True}
    max_turns = max(cfg.max_turns, turn_cap(family, level))
    return Task(
        dataset=MemoryDataset(samples, name=f"{family}-{level}-{split}"),
        solver=kb_agent(arm_provider(arm), load_world, exposure=exposure, max_turns=max_turns, delivery=delivery),
        scorer=[task_success(), delivered_evidence(), error_analysis()],
        metadata={
            "arm": arm,
            "exposure": exposure,
            "delivery": delivery,
            "family": family,
            "level": level,
            "split": split,
            "faults": faults,
            "max_turns": max_turns,
            "knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ARM_KNOB_PREFIXES)},
        },
    )
