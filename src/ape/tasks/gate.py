"""Inspect task for the APG-vs-LightRAG suitability gate.

    inspect eval src/ape/tasks/gate.py -T family=F7 -T level=1000 -T split=dev -T arm=APG-s \
        --model openai/<agent-model> --model-role kg=openai/<kg-model> --epochs 3

Worlds must already be built (`python -m ape.build`).
"""

import json
from dataclasses import asdict

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from ape.agent.arms import arm_provider, load_world
from ape.agent.kb_react import kb_agent
from ape.config import Config
from ape.scorers.success import delivered_evidence, task_success
from ape.worlds.spec import World


def gate_samples(family: str, level: str, split: str, relational: bool = True, limit_worlds: int | None = None) -> list[Sample]:
    cfg = Config()
    variant = {"F7": ("-rel-" if relational else "-ind-")}.get(family, "-")
    paths = sorted((cfg.worlds_dir / split).glob(f"{family}-{level}{variant}*.json"))
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
                    metadata={"world_id": w.id, "family": family, "level": level, "split": split, "task": asdict(t), "case": t.tags.get("case")},
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
) -> Task:
    cfg = Config()
    return Task(
        dataset=MemoryDataset(gate_samples(family, level, split, relational, limit_worlds), name=f"{family}-{level}-{split}"),
        solver=kb_agent(arm_provider(arm, cfg), load_world, exposure=exposure, max_turns=cfg.max_turns),
        scorer=[task_success(), delivered_evidence()],
        metadata={"arm": arm, "exposure": exposure, "family": family, "level": level, "split": split},
    )
