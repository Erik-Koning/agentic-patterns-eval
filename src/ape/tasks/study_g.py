"""Inspect task for Study G's F8 sessions: one sample is one session (CONTEXT_MANAGEMENT_AUDIT §4.1).

    inspect eval src/ape/tasks/study_g.py@f8_session -T level=40 -T split=dev -T arm=CM0 \
        --model openai/gpt-6-luna --reasoning-effort high

Worlds must already be built (`python -m ape.build --family F8 --levels 40 --worlds 10 --split dev`).
Arms built so far: CM0 and O-state (`ape.agent.session`). Probes use the `probe` role when the run defines
one, else the agent's model.
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from ape.agent.session import f8_session_agent
from ape.config import Config
from ape.scorers.session import f8_session_score
from ape.worlds import gen_f8
from ape.worlds.spec import World


def session_samples(level: str, split: str, limit_worlds: int | None = None, variant: str = "") -> list[Sample]:
    """One sample per F8 world of this length (and knob variant, default: the default knobs)."""
    cfg = Config()
    paths = sorted(Path(cfg.worlds_dir / split).glob(f"F8-{level}{'-' + variant if variant else ''}-{split}-s*.json"))
    if limit_worlds:
        paths = paths[:limit_worlds]
    samples = []
    for path in paths:
        w = World.load(path)
        s = w.entities["session"]
        samples.append(
            Sample(
                id=w.id,
                input=gen_f8.start_message(w),
                metadata={"world_id": w.id, "family": "F8", "level": level, "split": split, "N": s["N"], "knobs": s["knobs"], "w_crossing_item": s["reference"]["w_crossing_item"]},
            )
        )
    if not samples:
        raise FileNotFoundError(f"no F8-{level} sessions in {cfg.worlds_dir / split}; run `python -m ape.build --family F8`")
    return samples


@task
def f8_session(
    level: str = "40",
    split: str = "dev",
    arm: str = "CM0",
    limit_worlds: int | None = None,
    window: int = gen_f8.WINDOW,
    max_turns_per_item: int = 8,
    checkpoints: str = ",".join(map(str, gen_f8.CHECKPOINTS)),
    variant: str = "",
) -> Task:
    points = tuple(int(x) for x in str(checkpoints).split(",") if str(x).strip())
    return Task(
        dataset=MemoryDataset(session_samples(level, split, limit_worlds, variant), name=f"F8-{level}-{split}"),
        solver=f8_session_agent(arm=arm, window=window, max_turns_per_item=max_turns_per_item, checkpoints=points),
        scorer=f8_session_score(),
        metadata={"arm": arm, "family": "F8", "level": level, "split": split, "window": window, "checkpoints": list(points)},
    )
