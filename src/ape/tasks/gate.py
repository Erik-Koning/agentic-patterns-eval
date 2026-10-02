"""Inspect task for the APG-vs-LightRAG suitability gate.

    inspect eval src/ape/tasks/gate.py -T family=F7 -T level=1000 -T split=dev -T arm=APG-s --epochs 3 \
        --model openai/gpt-6-luna --reasoning-effort high \
        --model-role 'kg={model: openai/gpt-6-luna, reasoning_effort: low}'

Worlds must already be built (`python -m ape.build`). Bare `inspect eval` runs at the efforts its
flags give (the model default if none). Python callers build the models from `config/models.yaml`
with `ape.models.agent_model` and `role_models`. Either way the log records each model's config.
"""

import json
import os
import re
from dataclasses import asdict

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from ape.agent.arms import arm_provider, load_world
from ape.agent.kb_react import kb_agent
from ape.config import Config
from ape.scorers.success import delivered_evidence, task_success
from ape.scorers.taxonomy import error_analysis
from ape.worlds.gen_f7 import EXCEPTION_STYLES
from ape.worlds.generate import SEED_BLOCK
from ape.worlds.spec import World

# Arms with a retriever a search_kb tool can call (monolith and oracle context have none).
PULLABLE = {"S3s", "APG-q", "APG-s", "APGo-q", "S5o", "LGR-q", "LGR-s", "LGRo-q", "LGRo-s"}
# Environment knobs that change what an arm delivers; the task records them (metadata `knobs`) as they are when
# it is created, since they are not task args. Callers create the task under the knobs it runs with.
ARM_KNOB_PREFIXES = ("APE_APG_", "APE_LGR_", "APE_S3S_", "APE_S7_", "APE_CONTEXT_BUDGET", "APE_MAX_TURNS")


def _variant(family: str, relational: bool, exception_style: str) -> str:
    if family != "F7":
        return "-"
    return f"-rel-{EXCEPTION_STYLES[exception_style]}-" if relational else "-ind-"


def _seed(path) -> int:
    m = re.search(r"-s(\d+)\.json$", str(path))
    return int(m.group(1)) if m else -1


def gate_samples(
    family: str,
    level: str,
    split: str,
    relational: bool = True,
    limit_worlds: int | None = None,
    exception_style: str = "descriptive",
    seed_base: int | None = None,
) -> list[Sample]:
    """The tasks of the cell's worlds, the first `limit_worlds` by seed. `seed_base` keeps only one run's seed block
    (`seed_base` .. `seed_base` + SEED_BLOCK - 1): a later gate run's test worlds share the directory with an
    earlier run's."""
    cfg = Config()
    variant = _variant(family, relational, exception_style)
    paths = sorted((cfg.worlds_dir / split).glob(f"{family}-{level}{variant}{split}-*.json"))
    if seed_base is not None:
        paths = [p for p in paths if seed_base <= _seed(p) < seed_base + SEED_BLOCK]
    if limit_worlds:
        paths = paths[:limit_worlds]
    samples = []
    for path in paths:
        w = World.load(path)
        for t in w.tasks:
            samples.append(
                Sample(
                    id=t.id,
                    input=t.prompt,
                    target=json.dumps(t.gold, sort_keys=True),
                    metadata={
                        "world_id": w.id,
                        "family": family,
                        "level": level,
                        "split": split,
                        "exception_style": w.entities.get("exception_style"),
                        "task": asdict(t),
                        "case": t.tags.get("case"),
                    },
                )
            )
    if not samples:
        raise FileNotFoundError(f"no worlds for {family}-{level} in {cfg.worlds_dir / split}; run `python -m ape.build`")
    return samples


@task
def gate(
    family: str = "F7",
    level: str = "10",
    split: str = "dev",
    arm: str = "S1",
    exposure: str = "retrieved",
    relational: bool = True,
    limit_worlds: int | None = None,
    exception_style: str = "descriptive",
    delivery: str = "push",
    plan_cell: str | None = None,
    group: str | None = None,
    seed_base: int | None = None,
) -> Task:
    """`plan_cell` and `group` label a gate orchestrator run (the run_plan.yaml cell and its env group) in the
    task metadata; they are task args only when passed, so other callers' task identities are unchanged.
    `seed_base` selects one gate run's test-seed block (`gate_samples`); the orchestrator passes it only for a
    block other than the default 3000, so the first run's task identities are unchanged too."""
    if delivery != "push" and arm not in PULLABLE:
        raise ValueError(f"{arm} has no retriever for delivery={delivery}; pull applies to {sorted(PULLABLE)}")
    cfg = Config()
    return Task(
        dataset=MemoryDataset(gate_samples(family, level, split, relational, limit_worlds, exception_style, seed_base), name=f"{family}-{level}-{split}"),
        # The arm reads its knobs (budgets included) when it is built inside the run, under the run's environment.
        solver=kb_agent(arm_provider(arm), load_world, exposure=exposure, max_turns=cfg.max_turns, delivery=delivery),
        scorer=[task_success(), delivered_evidence(), error_analysis()],
        metadata={
            "arm": arm,
            "exposure": exposure,
            "delivery": delivery,
            "family": family,
            "level": level,
            "split": split,
            "exception_style": exception_style if family == "F7" and relational else None,
            "knobs": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ARM_KNOB_PREFIXES)},
        }
        | ({"plan_cell": plan_cell, "group": group} if plan_cell or group else {}),
    )
