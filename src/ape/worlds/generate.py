"""Entry point for world generation, plus the split seed ranges.

Seeds are split-disjoint so dev tuning can never see a pilot or test world. The
test split is generated only after GATE_PREREG.md is frozen and hashed: world-set
builders call `require_test_split_unlocked`, and only the gate orchestrator's
build-test phase (`python -m ape.run_gate build-test`, after its freeze guard) sets
TEST_SPLIT_ENV. `make_world` itself stays unguarded, for tests and readers.
"""

import os

from . import gen_f3, gen_f5, gen_f7
from .spec import World

SPLIT_SEED_BASE = {"dev": 1000, "pilot": 2000, "test": 3000}
TEST_SPLIT_ENV = "APE_TEST_SPLIT_RUN"  # the run id of the frozen gate run building the test split


class TestSplitLocked(RuntimeError):
    """The test split was requested outside the gate's build-test phase."""


def require_test_split_unlocked(split: str) -> None:
    """Refuse to build test-split worlds unless the gate orchestrator's build-test phase is running."""
    if split == "test" and not os.environ.get(TEST_SPLIT_ENV):
        raise TestSplitLocked(
            "the test split is generated only after the freeze, by `python -m ape.run_gate build-test --run-id <id>` "
            f"(GATE_PREREG §4; it sets {TEST_SPLIT_ENV} once the frozen files match their hashes)"
        )


def make_world(family: str, level: str, split: str, index: int, n_tasks: int, relational: bool = True, exception_style: str = "descriptive") -> World:
    seed = SPLIT_SEED_BASE[split] + index
    if family == "F7":
        return gen_f7.generate(level, relational, split, seed, n_tasks, exception_style)
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
