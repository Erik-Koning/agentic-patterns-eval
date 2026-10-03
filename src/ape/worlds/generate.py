"""Entry point for world generation, plus the split seed ranges and the test split's lock.

Seeds are split-disjoint so dev tuning can never see a pilot or test world. The
test split is generated only after a study's pre-registration is frozen and hashed:
world-set builders call `require_test_split_unlocked`, and only a study's own
build-test phase (`python -m ape.run_gate build-test` for the gate,
`python -m ape.run_study build-test --study <study>` for the others, after their
freeze guards) sets TEST_SPLIT_ENV. `make_world` itself stays unguarded, for tests and readers.

Seed namespaces. World i of a split has seed `base + i`, and the seed is part of the world ID, so two bases
never share a world. SPLIT_SEED_BASE holds the gate's bases. A gate run's test worlds start at the run's
frozen test-seed base (freeze.json `test_seeds`; 3000 for the first run, a fresh block for an extension or a
fix cycle, so already-analysed worlds are never reused); `ape.run_gate` passes it through TEST_SEED_BASE_ENV
while it builds the test split. Other studies pass their own `seed_base` to `make_world` (STUDY_SEEDS):

    study     dev     pilot    test blocks       offline rehearsal (test)
    gate      1000+   2000+    [3000, 9000)      9000+
    main      1000+   12000+   [13000, 19000)    19000+
    study_g   1000+   22000+   [23000, 29000)    29000+

- **dev is shared on purpose.** Dev worlds only tune, so the main study and Study G reuse the gate's dev
  namespace: a main dev world of a gate dev cell (F7-1000, F3-60) is the gate's own world, with its already-built
  (paid) KG artifacts. `ape.run_study` builds those with the gate's exact parameters and refuses to overwrite a
  shared world with other content.
- **pilot and test are a study's own.** Every gate seed is below GATE_SEED_LIMIT and every other study's own seeds
  are at or above it, in disjoint ranges (`_check_namespaces` asserts this at import). Each live run of a study
  freezes a fresh block of SEED_BLOCK test seeds inside its study's test range; offline rehearsals use the
  study's offline base, at the range's end.
- **Readers select their block.** World files of every study share `worlds/<split>/`. `tasks.gate.gate_samples`
  keeps only the seed block it is given, and without one only the gate's namespace, so a gate task never reads
  another study's worlds (as strings, `s12000` sorts before `s2000`).

The lock is per study: the gate's build-test sets TEST_SPLIT_ENV to its run id, another study's to
`<study>/<run id>` (`split_lock_value`), and `require_test_split_unlocked(split, study)` accepts only its own
study's unlock, so the gate's unlock never opens the main study's test worlds, nor the other way round.
"""

import os

from . import gen_f3, gen_f5, gen_f7, gen_f8, gen_registry
from .spec import World

SPLIT_SEED_BASE = {"dev": 1000, "pilot": 2000, "test": 3000}
TEST_SPLIT_ENV = "APE_TEST_SPLIT_RUN"  # the frozen run building the test split: a gate run id, or `<study>/<run id>`
TEST_SEED_BASE_ENV = "APE_TEST_SEED_BASE"  # the frozen gate run's test-seed base, while its build-test runs
SEED_BLOCK = 100  # seeds a run's test block spans (base .. base + 99): more than any cell's world count
GATE_SEED_LIMIT = 10_000  # every gate seed is below this; other studies' own (pilot, test) seeds are at or above it
# Per study: dev and pilot bases, the test range [first block, end) a live run's block lies in, and the offline
# rehearsals' test base (the range's end). The gate's entry restates SPLIT_SEED_BASE and run_gate's test constants.
STUDY_SEEDS: dict[str, dict] = {
    "gate": {"dev": 1000, "pilot": 2000, "test": (3000, 9000), "offline_test": 9000},
    "main": {"dev": 1000, "pilot": 12000, "test": (13000, 19000), "offline_test": 19000},
    "study_g": {"dev": 1000, "pilot": 22000, "test": (23000, 29000), "offline_test": 29000},
}


def _check_namespaces() -> None:
    """Import-time guard: no two studies' pilot or test seeds (offline blocks included) can coincide, every gate seed is
    below GATE_SEED_LIMIT and every other study's own seeds are at or above it."""
    spans: list[tuple[str, int, int]] = []
    for study, s in STUDY_SEEDS.items():
        lo, hi = s["test"]
        if lo >= hi or (hi - lo) % SEED_BLOCK or s["offline_test"] != hi:
            raise ValueError(f"STUDY_SEEDS[{study!r}]: the test range must be whole seed blocks, with the offline base at its end")
        own = [(f"{study} pilot", s["pilot"], s["pilot"] + SEED_BLOCK), (f"{study} test", lo, hi), (f"{study} offline", hi, hi + SEED_BLOCK)]
        below = [end <= GATE_SEED_LIMIT for _, _, end in own]
        if (study == "gate" and not all(below)) or (study != "gate" and any(start < GATE_SEED_LIMIT for _, start, _ in own)):
            raise ValueError(f"STUDY_SEEDS[{study!r}]: gate seeds lie below {GATE_SEED_LIMIT}, other studies' own seeds at or above it")
        spans += own
    spans.sort(key=lambda x: x[1])
    for (a, _, a_end), (b, b_start, _) in zip(spans, spans[1:], strict=False):
        if b_start < a_end:
            raise ValueError(f"seed namespaces overlap: {a} and {b}")
    gate = STUDY_SEEDS["gate"]
    if (gate["dev"], gate["pilot"], gate["test"][0]) != (SPLIT_SEED_BASE["dev"], SPLIT_SEED_BASE["pilot"], SPLIT_SEED_BASE["test"]):
        raise ValueError("STUDY_SEEDS['gate'] must restate SPLIT_SEED_BASE")


_check_namespaces()


def split_seed_base(split: str) -> int:
    """The first seed of `split`: SPLIT_SEED_BASE, except the test split while a gate run's build-test sets
    TEST_SEED_BASE_ENV to its frozen base."""
    if split == "test" and (raw := os.environ.get(TEST_SEED_BASE_ENV, "").strip()):
        return int(raw)
    return SPLIT_SEED_BASE[split]


class TestSplitLocked(RuntimeError):
    """The test split was requested outside the requesting study's build-test phase."""


def split_lock_value(study: str, run_id: str) -> str:
    """What a study's build-test sets TEST_SPLIT_ENV to: the gate's run id (as it always has), else `<study>/<run id>`
    (run ids never contain "/")."""
    return run_id if study == "gate" else f"{study}/{run_id}"


def unlocked_study() -> str | None:
    """The study whose build-test unlocked the test split (TEST_SPLIT_ENV), or None while it is locked."""
    raw = os.environ.get(TEST_SPLIT_ENV, "").strip()
    if not raw:
        return None
    return raw.split("/", 1)[0] if "/" in raw else "gate"


def require_test_split_unlocked(split: str, study: str = "gate") -> None:
    """Refuse to build test-split worlds unless `study`'s own build-test phase is running (TEST_SPLIT_ENV holds its
    unlock); another study's unlock does not count."""
    if split != "test":
        return
    owner = unlocked_study()
    if owner == study:
        return
    how = (
        "`python -m ape.run_gate build-test --run-id <id>` (GATE_PREREG §4"
        if study == "gate"
        else f"`python -m ape.run_study build-test --study {study} --run-id <id>` (after its pre-registration's freeze"
    )
    if owner is None:
        raise TestSplitLocked(f"the test split is generated only after the freeze, by {how}; it sets {TEST_SPLIT_ENV} once the frozen files match their hashes)")
    raise TestSplitLocked(
        f"the test split is unlocked for {owner} ({TEST_SPLIT_ENV}={os.environ[TEST_SPLIT_ENV]!r}), not for {study}: "
        f"{study}'s test worlds are generated only by {how}; another study's unlock never opens them)"
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
