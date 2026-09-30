"""Entry point for world generation, plus the split seed ranges.

Seeds are split-disjoint so dev tuning can never see a pilot or test world. The
test split is generated only after GATE_PREREG.md is frozen and hashed.
"""

from . import gen_f3, gen_f5, gen_f7
from .spec import World

SPLIT_SEED_BASE = {"dev": 1000, "pilot": 2000, "test": 3000}


def make_world(family: str, level: str, split: str, index: int, n_tasks: int, relational: bool = True) -> World:
    seed = SPLIT_SEED_BASE[split] + index
    if family == "F7":
        return gen_f7.generate(level, relational, split, seed, n_tasks)
    if family == "F3":
        return gen_f3.generate(level, split, seed, n_tasks)
    if family == "F5":
        return gen_f5.generate(level, split, seed, n_tasks)
    raise ValueError(family)


def solve(world: World, task) -> dict:
    """Reference answer computed from the structured spec (F5 gold is computed during generation)."""
    if world.family == "F7":
        return gen_f7.solve(world, task)
    if world.family == "F3":
        return gen_f3.solve(world, task)
    return task.gold
