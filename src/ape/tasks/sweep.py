"""Inspect task for the context-length sweep (CONTEXT_SWEEP.md): one sample is one probe decision at one size.

    inspect eval src/ape/tasks/sweep.py@context_sweep -T sizes=8000,64000 --model openai/gpt-6-luna --reasoning-effort high

Worlds must already be built (`python -m ape.run_sweep build`, or `ape.worlds.gen_sweep.generate_set` saved under
worlds/sweep/). The dataset is one sample per (task, kind, size): `kinds` and `sizes` choose them (comma-separated),
`tasks_per_kind` how many tasks (world seeds, `gen_sweep.task_seed`) each kind has, `seed_base` where the seeds start
and `output_tokens` the tool-file knob (part of the world ID).

`max_output_tokens` is the ceiling on each generation's output (reasoning included); the solver lowers it per request so
input plus output fit `window_tokens` (the GPT-6 window, 1,050,000: `agent.sweep.output_cap`). `working_limit` is the per-sample
runaway wall-clock guard (seconds). `group` labels the replicate (the runner passes it), so each replicate is its own
task identity.

Cache nonce (`agent.cache_nonce`): when APE_CACHE_NONCE is set as the task is created, every sample's system prompt
starts with the task's nonce, so provider prompt caches never cross runs or replicates.
"""

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from ape.agent import cache_nonce
from ape.agent.sweep import MAX_GENERATIONS, MAX_OUTPUT_TOKENS, WINDOW_TOKENS, sweep_probe
from ape.config import Config
from ape.scorers.sweep import sweep_score
from ape.worlds import gen_sweep
from ape.worlds.spec import World

WORKING_LIMIT_S = 1800


def _ints(x: str | int | list | tuple) -> list[int]:
    if isinstance(x, list | tuple):
        return [int(v) for v in x]
    return [int(v) for v in str(x).split(",") if str(v).strip()]


def _strs(x: str | list | tuple) -> list[str]:
    if isinstance(x, list | tuple):
        return [str(v) for v in x]
    return [v.strip() for v in str(x).split(",") if v.strip()]


def sweep_samples(kinds: list[str], sizes: list[int], seed_base: int, output_tokens: int, tasks_per_kind: int = 1) -> list[Sample]:
    """One sample per task, kind and size (task-major, then kind, then size ascending)."""
    cfg = Config()
    samples = []
    for j in range(int(tasks_per_kind)):
        for kind in kinds:
            for size in sorted(sizes):
                wid = gen_sweep.world_id(kind, size, gen_sweep.task_seed(kind, j, seed_base), output_tokens)
                path = cfg.world_path(wid)
                if not path.exists():
                    raise FileNotFoundError(f"no sweep world {wid} in {path.parent}; run `python -m ape.run_sweep build`")
                w = World.load(path)
                d = w.entities["sweep"]
                samples.append(
                    Sample(
                        id=w.id,
                        input=gen_sweep.START,
                        metadata={"world_id": w.id, "family": "F8S", "level": w.level, "split": gen_sweep.SPLIT, "kind": kind, "task": j,
                                  "seed": w.seed, "target_tokens": size, "context_tokens": d["context_tokens"], "middle_cases": d["middle_cases"]},
                    )
                )  # fmt: skip
    return samples


@task
def context_sweep(
    kinds: str = ",".join(gen_sweep.KINDS),
    sizes: str = ",".join(str(s) for s in gen_sweep.SIZES),
    seed_base: int = gen_sweep.SEED_BASE,
    output_tokens: int = gen_sweep.DEFAULT_OUTPUT_TOKENS,
    max_generations: int = MAX_GENERATIONS,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    working_limit: int = WORKING_LIMIT_S,
    group: str | None = None,
    tasks_per_kind: int = 1,
    window_tokens: int = WINDOW_TOKENS,
) -> Task:
    ks, ss = _strs(kinds), _ints(sizes)
    if bad := [k for k in ks if k not in gen_sweep.KINDS]:
        raise ValueError(f"unknown sweep kind(s) {bad}; one of {gen_sweep.KINDS}")
    if int(tasks_per_kind) < 1:
        raise ValueError("tasks_per_kind must be at least 1")
    samples = sweep_samples(ks, ss, int(seed_base), int(output_tokens), int(tasks_per_kind))
    nonce = cache_nonce.task_nonce("context_sweep")  # None unless the run sets APE_CACHE_NONCE
    cache_nonce.stamp(samples, nonce)
    return Task(
        dataset=MemoryDataset(samples, name="F8S"),
        solver=sweep_probe(max_generations=int(max_generations), max_output_tokens=int(max_output_tokens), window_tokens=int(window_tokens)),
        scorer=sweep_score(),
        config=GenerateConfig(max_tokens=int(max_output_tokens)),
        working_limit=int(working_limit),
        metadata={"family": "F8S", "kinds": ks, "sizes": ss, "seed_base": int(seed_base), "output_tokens": int(output_tokens),
                  "tasks_per_kind": int(tasks_per_kind), "max_generations": int(max_generations), "window_tokens": int(window_tokens)}
        | ({"group": group} if group is not None else {}) | cache_nonce.task_metadata(nonce),
    )  # fmt: skip
