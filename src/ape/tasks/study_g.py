"""Inspect task for Study G's F8 sessions: one sample is one session (CONTEXT_MANAGEMENT_AUDIT §4.1).

    inspect eval src/ape/tasks/study_g.py@f8_session -T level=40 -T split=dev -T arm=CM0 \
        --model openai/gpt-6-luna --reasoning-effort high

Worlds must already be built (`python -m ape.build --family F8 --levels 40 --worlds 10 --split dev`, with
`--knob output_tokens=2250` for a run_plan cell's knobs and `--seed-base` for a study's seed range). Arms are the
registered context policies (`ape.agent.context_policy`; CM0 and O-state so far). Probes use the `probe` role and
management calls the `cm` role when the run defines them, else the agent's model.

World selection, as `tasks.gate.gate_samples`:
- `variant` is the knob tag of the cell's worlds (`gen_f8.variant_tag(knobs)`, "" at the default knobs); every
  selected world must carry exactly those knobs.
- `seed_base` keeps one seed block (`seed_base` .. `seed_base` + SEED_BLOCK - 1), so a study's worlds never mix
  with another's in the same directory; `skip_worlds` drops the first worlds of the block and `limit_worlds` keeps
  the next ones, by seed.

`plan_cell`, `group`, `seed_base`, `skip_worlds` and `threshold` are task args only when passed (the study runner
passes them), so other callers' task identities are unchanged; `threshold` (T_abs) reaches the solver only then,
and run_plan.yaml's `study_g.threshold` applies otherwise. The task metadata records the variant's knobs, the
resolved threshold and every APE_CM_* variable (policy knobs: they are not task args, so a run that varies them
needs its own log directory, as with the gate's APE_* knobs).

Cache nonce (`agent.cache_nonce`): when APE_CACHE_NONCE is set as the task is created (the study runner sets it per eval
set), the task's nonce, derived from it and the arm, is in every sample's metadata and the task's (`cache_nonce`), and
the session's system prompt (so every view, probe and compaction input), the team's workers' and the management calls
start with it. Unset, nothing changes.
"""

import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample

from ape.agent import cache_nonce
from ape.agent.context_policy import knob_env
from ape.agent.session import f8_session_agent, plan_threshold
from ape.config import Config
from ape.scorers.session import f8_session_score
from ape.worlds import gen_f8
from ape.worlds.generate import SEED_BLOCK
from ape.worlds.spec import World


def _seed(path: Path) -> int:
    m = re.search(r"-s(\d+)\.json$", str(path))
    return int(m.group(1)) if m else -1


def session_samples(level: str, split: str, limit_worlds: int | None = None, variant: str = "", seed_base: int | None = None, skip_worlds: int | None = None) -> list[Sample]:
    """One sample per F8 world of this length and knob variant (default: the default knobs), optionally one seed
    block, after skipping `skip_worlds` and up to `limit_worlds`."""
    cfg = Config()
    paths = sorted(Path(cfg.worlds_dir / split).glob(f"F8-{level}{'-' + variant if variant else ''}-{split}-s*.json"))
    if seed_base is not None:
        paths = [p for p in paths if int(seed_base) <= _seed(p) < int(seed_base) + SEED_BLOCK]
    if skip_worlds:
        paths = paths[int(skip_worlds) :]
    if limit_worlds:
        paths = paths[:limit_worlds]
    samples = []
    for path in paths:
        w = World.load(path)
        s = w.entities["session"]
        if gen_f8.variant_tag(s["knobs"]) != variant:
            raise ValueError(f"{w.id}: knobs {s['knobs']} are not variant {variant!r}")
        samples.append(
            Sample(
                id=w.id,
                input=gen_f8.start_message(w),
                metadata={"world_id": w.id, "family": "F8", "level": level, "split": split, "N": s["N"], "knobs": s["knobs"], "w_crossing_item": s["reference"]["w_crossing_item"]},
            )
        )
    if not samples:
        block = f", seeds {seed_base}..{int(seed_base) + SEED_BLOCK - 1}" if seed_base is not None else ""
        raise FileNotFoundError(f"no F8-{level}{'-' + variant if variant else ''} sessions in {cfg.worlds_dir / split}{block}; run `python -m ape.build --family F8`")
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
    threshold: int | None = None,
    plan_cell: str | None = None,
    group: str | None = None,
    seed_base: int | None = None,
    skip_worlds: int | None = None,
) -> Task:
    points = tuple(int(x) for x in str(checkpoints).split(",") if str(x).strip())
    samples = session_samples(level, split, limit_worlds, variant, seed_base, skip_worlds)
    nonce = cache_nonce.task_nonce(arm)  # None unless the run sets APE_CACHE_NONCE
    cache_nonce.stamp(samples, nonce)
    run = {k: v for k, v in {"plan_cell": plan_cell, "group": group, "seed_base": seed_base, "skip_worlds": skip_worlds}.items() if v is not None}
    return Task(
        dataset=MemoryDataset(samples, name=f"F8-{level}-{split}"),
        solver=f8_session_agent(arm=arm, window=window, max_turns_per_item=max_turns_per_item, checkpoints=points, **({"threshold": threshold} if threshold is not None else {})),
        scorer=f8_session_score(),
        metadata={
            "arm": arm,
            "family": "F8",
            "level": level,
            "split": split,
            "window": window,
            "checkpoints": list(points),
            "variant": variant,
            "knobs": samples[0].metadata["knobs"],
            "threshold": plan_threshold() if threshold is None else int(threshold),
            "policy_env": knob_env(),
        }
        | run
        | cache_nonce.task_metadata(nonce),
    )
