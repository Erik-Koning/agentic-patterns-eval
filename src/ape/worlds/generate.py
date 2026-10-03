"""Entry point for world generation, plus the split seed ranges.

Seeds are split-disjoint so dev tuning can never see a pilot or test world. The
test split is generated only after GATE_PREREG.md is frozen and hashed: world-set
builders call `require_test_split_unlocked`, and only the gate orchestrator's
build-test phase (`python -m ape.run_gate build-test`, after its freeze guard) sets
TEST_SPLIT_ENV. `make_world` itself stays unguarded, for tests and readers.

Seed namespaces. World i of a split has seed `base + i`, and the seed is part of the world ID, so two bases
never share a world. SPLIT_SEED_BASE holds the gate's bases. A gate run's test worlds start at the run's
frozen test-seed base (freeze.json `test_seeds`; 3000 for the first run, a fresh block for an extension or a
fix cycle, so already-analysed worlds are never reused); `ape.run_gate` passes it through TEST_SEED_BASE_ENV
while it builds the test split. Other studies (the main study, Study G) pass their own `seed_base` to
`make_world`, so their worlds never coincide with the gate's.
"""

import os

from . import gen_f3, gen_f5, gen_f7, gen_f8, gen_registry
from .spec import World

SPLIT_SEED_BASE = {"dev": 1000, "pilot": 2000, "test": 3000}
TEST_SPLIT_ENV = "APE_TEST_SPLIT_RUN"  # the run id of the frozen gate run building the test split
TEST_SEED_BASE_ENV = "APE_TEST_SEED_BASE"  # the frozen gate run's test-seed base, while its build-test runs
SEED_BLOCK = 100  # seeds a gate run's test block spans (base .. base + 99): more than any cell's world count


def split_seed_base(split: str) -> int:
    """The first seed of `split`: SPLIT_SEED_BASE, except the test split while a gate run's build-test sets
    TEST_SEED_BASE_ENV to its frozen base."""
    if split == "test" and (raw := os.environ.get(TEST_SEED_BASE_ENV, "").strip()):
        return int(raw)
    return SPLIT_SEED_BASE[split]


class TestSplitLocked(RuntimeError):
    """The test split was requested outside the gate's build-test phase."""


def require_test_split_unlocked(split: str) -> None:
    """Refuse to build test-split worlds unless the gate orchestrator's build-test phase is running."""
    if split == "test" and not os.environ.get(TEST_SPLIT_ENV):
        raise TestSplitLocked(
            "the test split is generated only after the freeze, by `python -m ape.run_gate build-test --run-id <id>` "
            f"(GATE_PREREG §4; it sets {TEST_SPLIT_ENV} once the frozen files match their hashes)"
        )


def make_world(
    family: str,
    level: str,
    split: str,
    index: int,
    n_tasks: int,
    relational: bool = True,
    exception_style: str = "descriptive",
    seed_base: int | None = None,
    knobs: dict | None = None,
) -> World:
    """World `index` of `split`: seed `seed_base + index` (default `split_seed_base(split)`).

    `knobs` are F8 generator settings from a run_plan session cell (`output_tokens`, `memo_density`,
    `dependency_density`; D-028); other families take none."""
    seed = (split_seed_base(split) if seed_base is None else int(seed_base)) + index
    if knobs and family != "F8":
        raise ValueError(f"generator knobs {sorted(knobs)} apply to F8 only, not {family}")
    if family == "F7":
        return gen_f7.generate(level, relational, split, seed, n_tasks, exception_style)
    if family == "F3":
        return gen_f3.generate(level, split, seed, n_tasks)
    if family == "F5":
        return gen_f5.generate(level, split, seed, n_tasks)
    if family == "F8":
        allowed = {"output_tokens", "memo_density", "dependency_density"}
        if unknown := set(knobs or {}) - allowed:
            raise ValueError(f"unknown F8 knobs {sorted(unknown)}; allowed: {sorted(allowed)}")
        return gen_f8.generate(level, split, seed, n_tasks, **(knobs or {}))  # one session; the level is N, n_tasks is ignored
    if family in ("F1", "F2"):
        return gen_registry.generate(family, level, split, seed, n_tasks)
    raise ValueError(family)


def solve(world: World, task) -> dict:
    """Reference answer computed from the structured spec (F5 gold is computed during generation)."""
    if world.family == "F7":
        return gen_f7.solve(world, task)
    if world.family == "F3":
        return gen_f3.solve(world, task)
    if world.family in ("F1", "F2"):
        return gen_registry.solve(world, task)
    return task.gold
