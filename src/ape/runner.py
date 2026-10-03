"""Error-tolerant, resumable eval runs (FIX_PLAN FX-3): one wrapper over Inspect's `eval_set`.

    success, logs = run_evals([gate(arm="S1"), gate(arm="APG-s")], "cache/runs/pilot", profile="gate")

Every entry point (gate runs, tuning, anchor, smoke) runs through `run_evals`, so all get the same behaviour:

- **Models (FX-1).** The agent and each role are built from the profile, with their efforts and the
  profile's `max_connections`. `model_override` with `model_args` swaps the model and keeps the
  efforts (mockllm in tests). Pre-built `model` and `model_roles` (from `agent_model` / `role_models`)
  take precedence over both.
- **Prices (FX-2).** `model_cost_config` is always passed, and `require_preflight` runs before anything
  else. `live` defaults to True unless the agent is a mockllm model.
- **Sample retries.** `retry_on_error=2`: Inspect re-runs an errored sample from scratch up to twice
  before counting it as an error. Each retried error is kept in `EvalSample.error_retries`, and
  `EvalSampleSummary.retries` counts them.
- **Error budget (circuit breaker), per task.** Inspect marks a task's log `error` once its errored samples
  reach the task's `fail_on_error`. A task of at least 150 planned samples (dataset x epochs) gets the
  proportion 0.02, matching PC5's "< 2%". A smaller task gets a count, `max(3, ceil(0.02 x planned))`. With a
  proportion, a task under 50 samples would abort on its first persistent error: the gate's pilot cells
  (48 samples) and every Study G session task. Below the threshold the log is `success` and still holds the
  errored samples, unscored; they count as failures in the analysis (GATE_PREREG §2), and PC5 is judged from
  the logs, not from this threshold. A caller's `fail_on_error=` overrides every task, and so does a value
  the task set itself.
- **Runaway guard.** Every task gets Inspect's per-sample `cost_limit` (`ape.budget.sample_cost_limit`):
  20x the caller's conservative per-sample projection (`sample_cost_usd=`), at least $0.50, else $2.00.
  The values are in run_plan.yaml `budget.sample_cost_limit`. Inspect needs a price for every model to
  enforce it, so a run with an unpriced model (offline mockllm) gets none. A sample that hits it ends with
  `limit` "cost" (`gate_stats.load_results`: `limit_hit`, counted as a cap hit in PC5). A caller's
  `cost_limit=` overrides.
- **Task retries.** `retry_attempts=3` with `retry_immediate=False`, so the `retry_wait` back-off applies
  (30 s, then 60 s, ...). Inspect's retry loop finishes the whole set before each retry. A retry reuses
  the failed log's completed samples (same uuid, same usage) and re-runs only the rest. In this mode
  `retry_attempts` counts attempts, not retries. `retry_cleanup=False` keeps the failed attempts' logs:
  Inspect would delete them after a successful retry, and with them the spend of their errored samples.
  Spend counts each sample once by uuid (`ape.budget.log_samples`), so nothing is double counted, and
  `eval_set` itself always reads the newest log per task.
- **Spend registry.** Before `eval_set` starts, the log dir is recorded in the program's spend registry
  (`ape.spend`, label from APE_SPEND_LABEL), and the finish after it. So `ape.budget.program_spend`, and the
  orchestrator's guard, see a run's flushed spend even if the process is killed. Offline runs (mock agent)
  are recorded only where APE_SPEND_REGISTRY points.
- **Usage ledger.** Inspect logs a sample's last attempt only. While `eval_set` runs, every model call's usage and
  cost is appended to `<log_dir>/usage_ledger.jsonl` (`ape.usage_ledger`, an Inspect hook), errored attempts and
  killed runs included, so the spend guards also count what no log holds (D-030).
- **No leaked model context.** `eval_set` runs in a copy of the caller's context: Inspect sets its active model
  and model roles there and never resets them, so without the copy a later `get_model(role="kg")` in the same
  process (another run, a test) would silently get this run's model.
- **Resume.** `eval_set` pairs each task with the logs in `log_dir` by `inspect_ai.task_identifier`:
  `task#hash(task args)/model/hash(plan, generate config, model args, roles, limits)`. Concurrency and
  retry settings are not part of it. Calling again with the same `log_dir` reuses every successful log
  and resumes the rest. Gate tasks that differ only in a task arg (arm, cell, delivery, ...) are
  distinct tasks. Anything that is not a task arg, such as an APE_* environment knob, is NOT part of
  the identity, so variants that differ only that way need separate log dirs. A log dir must hold only
  this set's logs unless `log_dir_allow_dirty=True` is passed.
- **Concurrency.** The profile's `max_samples` and `max_tasks` go to `eval_set`. Its `max_connections`
  is on each model's GenerateConfig, because eval-level generate args reach only the agent.
- **Index.** `<log_dir>/runner_index.json` maps each task identifier to the task's name, args, log,
  status and sample counts, so the orchestrator (FX-6) can find logs without scanning. A run entry is
  written *before* `eval_set` starts (`status: running`, its tasks, pid, per-task `fail_on_error` and
  `cost_limit`) and completed after it (`done` or `failed`, with the error if `eval_set` raised). A killed
  run's entry stays `running`.

Every default can be overridden by keyword (`fail_on_error=`, `retry_wait=`, `max_tasks=`, ...); any other
keyword goes to `eval_set` unchanged. Returns `eval_set`'s `(success, logs)`. The logs are headers
without samples; read one in full with `inspect_ai.log.read_eval_log(log.location)`.
"""

import contextvars
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval_set, task_identifier
from inspect_ai.log import EvalLog, read_eval_log_sample_summaries
from inspect_ai.model import Model

from . import spend, usage_ledger
from .models import COSTS_PATH, Profile, agent_model, eval_cost_kwargs, load_costs, load_profile, require_preflight, role_models

INDEX_NAME = "runner_index.json"
ERROR_RATE = 0.02  # PC5: harness errors < 2%
SMALL_TASK_SAMPLES = 150  # below this, a proportion would abort on the first few errors: use a count
MIN_ERROR_COUNT = 3
ERROR_DEFAULTS: dict[str, Any] = {
    "retry_on_error": 2,  # per sample, for transient API and tool errors
    "retry_attempts": 3,  # per task (attempts, with retry_immediate=False)
    "retry_wait": 30,  # seconds, doubled after each failed attempt
    "retry_immediate": False,  # True would retry at once, ignoring retry_wait
    "retry_cleanup": False,  # keep failed attempts' logs, so their samples' spend stays countable
}


def task_fail_on_error(planned: int) -> float:
    """Inspect's `fail_on_error` for a task of `planned` samples: the proportion ERROR_RATE from
    SMALL_TASK_SAMPLES up, else a count max(MIN_ERROR_COUNT, ceil(ERROR_RATE x planned)) (a float, as Inspect's
    config field is typed; any value >= 1 is a count)."""
    if planned >= SMALL_TASK_SAMPLES:
        return ERROR_RATE
    return float(max(MIN_ERROR_COUNT, math.ceil(ERROR_RATE * planned)))


def planned_samples(task: Task, epochs: int | None = None, limit: Any = None) -> int:
    """Dataset size (after an eval-level `limit`) x epochs (`epochs` overrides the task's)."""
    n = len(task.dataset)
    if isinstance(limit, int):
        n = min(n, limit)
    elif isinstance(limit, tuple | list) and len(limit) == 2:
        n = max(0, min(n, limit[1]) - limit[0])
    return n * int(epochs or task.epochs or 1)


def _all_priced(models: Mapping[str, str], costs_path: Path) -> bool:
    """Whether the price table covers every model of the run (Inspect enforces `cost_limit` only then)."""
    table = load_costs(costs_path)
    return all(m in table for m in models.values())


def run_evals(
    tasks: Task | Sequence[Task],
    log_dir: str | Path,
    *,
    profile: Profile | str | None = None,
    roles: tuple[str, ...] = ("kg",),
    model_override: str | None = None,
    model_args: Mapping[str, Any] | None = None,
    model: Model | None = None,
    model_roles: Mapping[str, Model] | None = None,
    epochs: int | None = None,
    live: bool | None = None,
    costs_path: Path = COSTS_PATH,
    sample_cost_usd: float | None = None,
    **eval_kwargs: Any,
) -> tuple[bool, list[EvalLog]]:
    """Run `tasks` as one resumable eval set in `log_dir` (see the module docstring).

    `profile` is a name or a loaded `Profile` (default: $APE_MODEL_PROFILE, else "gate"). `roles` are the
    Inspect roles built from it (ignored when `model_roles` is given). `costs_path` is the price table
    for both Inspect and the preflight. `sample_cost_usd` is the cost model's conservative $ per sample for
    these tasks, which sets the per-sample `cost_limit` (`ape.budget.sample_cost_limit`).
    """
    from .budget import sample_cost_limit

    p = profile if isinstance(profile, Profile) else load_profile(profile)
    # Preflight on the model names first: building a provider's Model can itself need its API key.
    used = {"agent": str(model) if model is not None else model_override or p.role("agent").model}
    if model_roles is not None:
        used |= {r: str(m) for r, m in model_roles.items()}
    else:
        used |= {r: model_override or p.role(r).model for r in roles}
    if live is None:
        live = not used["agent"].startswith("mockllm/")
    overrides = {r: m for r, m in used.items() if r not in p.roles or p.roles[r].model != m}
    require_preflight(p, live=live, overrides=overrides, costs_path=costs_path)

    margs = dict(model_args or {})
    agent = model if model is not None else agent_model(p, model=model_override, **margs)
    role_map = dict(model_roles) if model_roles is not None else role_models(p, roles, model=model_override, **margs)
    options = {
        **ERROR_DEFAULTS,
        "max_samples": p.concurrency.max_samples,
        "max_tasks": p.concurrency.max_tasks,
        **eval_cost_kwargs(costs_path),
        **eval_kwargs,
    }
    task_list = [tasks] if isinstance(tasks, Task) else list(tasks)
    limit = None
    if "cost_limit" not in eval_kwargs and _all_priced(used, costs_path):
        # The limit is part of Inspect's task identity: a log dir keeps its first run's limit, or a resume after a
        # recalibration would not pair with the earlier logs and would re-run every sample.
        limit = read_index(log_dir).get("cost_limit_usd") or sample_cost_limit(sample_cost_usd)
    per_task = []
    for t in task_list:
        if not isinstance(t, Task):
            continue
        n = planned_samples(t, epochs, eval_kwargs.get("limit"))
        if "fail_on_error" not in eval_kwargs and t.fail_on_error is None:
            t.fail_on_error = task_fail_on_error(n)
        if limit is not None and t.cost_limit is None:
            t.cost_limit = limit
        per_task.append({"task": t.name, "planned": n, "fail_on_error": t.fail_on_error, "cost_limit": t.cost_limit})

    run = {"key": uuid.uuid4().hex, "started": _now(), "status": "running", "pid": os.getpid(), "profile": p.name, "live": live, "cost_limit_usd": limit, "tasks_planned": per_task}
    registry = spend.register_logs(log_dir, live=live, tasks=[x["task"] for x in per_task])
    run["usage_ledger"] = str(Path(log_dir) / usage_ledger.LEDGER_NAME)
    write_index(log_dir, [], run)
    try:
        with usage_ledger.recording(Path(log_dir) / usage_ledger.LEDGER_NAME):
            # In a copy of the caller's context: Inspect sets its active model and model roles (context variables)
            # in the context it is called from and never resets them, so a later `get_model(role=...)` in this
            # process would resolve to this run's models instead of failing.
            success, logs = contextvars.copy_context().run(
                eval_set, task_list, log_dir=str(log_dir), model=agent, model_roles=role_map or None, epochs=epochs, **options
            )
    except BaseException as e:
        error = f"{type(e).__name__}: {e}"[:500]
        write_index(log_dir, [], {**run, "finished": _now(), "status": "failed", "success": False, "error": error})
        spend.finish_logs(registry, log_dir, False, error)
        raise
    write_index(log_dir, logs, {**run, "finished": _now(), "status": "done", "success": success})
    spend.finish_logs(registry, log_dir, success)
    return success, logs


# --- Index ------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def log_path(log: EvalLog) -> str:
    """The log's location as a plain path (Inspect returns reused logs as file:// URIs)."""
    return (log.location or "").removeprefix("file://")


def index_entry(log: EvalLog) -> tuple[str, dict]:
    """(task identifier, entry) for one log header. Sample counts come from the log's sample summaries."""
    try:
        summaries = read_eval_log_sample_summaries(log.location) if log.location else None
    except (OSError, ValueError):
        summaries = None
    errored = sum(s.error is not None for s in summaries) if summaries is not None else None
    res = log.results
    # Planned = dataset x epochs (the denominator of fail_on_error); an errored log may have no results.
    planned = res.total_samples if res else (log.eval.dataset.samples or 0) * (log.eval.config.epochs or 1)
    entry = {
        "task": log.eval.task,
        "task_args": log.eval.task_args_passed,
        "model": log.eval.model,
        "model_roles": {r: c.model for r, c in (log.eval.model_roles or {}).items()},
        "log": log_path(log),
        "status": log.status,
        "samples": {
            "planned": planned,
            "logged": len(summaries) if summaries is not None else None,
            "completed": res.completed_samples if res else None,
            "errored": errored,
            "retries": sum(s.retries or 0 for s in summaries) if summaries is not None else None,
        },
        "error": log.error.message if log.error else None,
        "task_id": log.eval.task_id,
        "eval_id": log.eval.eval_id,
        "updated": _now(),
    }
    return task_identifier(log, None), entry


def read_index(log_dir: str | Path) -> dict:
    path = Path(log_dir) / INDEX_NAME
    return json.loads(path.read_text()) if path.exists() else {"log_dir": str(log_dir), "runs": [], "tasks": {}}


def write_index(log_dir: str | Path, logs: list[EvalLog], run: dict) -> Path:
    """Upsert one entry per log (keyed by task identifier) and `run` into `<log_dir>/runner_index.json`: a run with
    a `key` replaces its earlier entry (the `running` one written at the start), otherwise it is appended."""
    index = read_index(log_dir)
    entries = dict(index_entry(log) for log in logs)
    index["tasks"].update(entries)
    run = {**run, "tasks": sorted(entries)}
    at = next((i for i, r in enumerate(index["runs"]) if run.get("key") and r.get("key") == run["key"]), None)
    if at is None:
        index["runs"].append(run)
    else:
        index["runs"][at] = run
    if run.get("cost_limit_usd") is not None:
        index.setdefault("cost_limit_usd", run["cost_limit_usd"])
    index["updated"] = _now()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    path = Path(log_dir) / INDEX_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path
