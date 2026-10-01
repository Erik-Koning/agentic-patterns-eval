"""Smoke-only fault injection (FIX_PLAN FX-8, error recovery): a gate task whose solver raises a harness-side
error on each sample's first attempt, before any model call, then runs the real solver.

Nothing in `src/ape` imports this; it exists so `readiness/smoke.py` can show that the FX-3 runner's
sample retry (`retry_on_error`) re-runs a failed sample and records the failure in `EvalSample.error_retries`.
"""

from inspect_ai import Task, task, task_with
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ape.tasks.gate import gate

FAULT = "smoke: injected harness fault on the first attempt (not a model error)"


@solver
def fail_first_attempt(inner: Solver) -> Solver:
    attempts: dict[tuple, int] = {}  # per task instance: (sample id, epoch) -> attempts so far

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        key = (state.sample_id, state.epoch)
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] == 1:
            raise RuntimeError(FAULT)
        return await inner(state, generate)

    return solve


@task
def faulted_gate(family: str = "F3", level: str = "5", split: str = "dev", arm: str = "S3s") -> Task:
    base = gate(family=family, level=level, split=split, arm=arm)
    return task_with(base, solver=fail_first_attempt(base.solver), name="smoke_faulted_gate")
