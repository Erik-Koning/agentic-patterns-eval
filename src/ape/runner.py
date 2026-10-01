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
- **Error budget.** `fail_on_error=0.02`: Inspect marks a task's log `error` once its errored samples
  reach 2% of its planned samples (dataset x epochs), which is PC5's "< 2%". Below that, the log is
  `success` and still holds the errored samples, unscored. A task of fewer than 50 samples tolerates none.
- **Task retries.** `retry_attempts=3` with `retry_immediate=False`, so the `retry_wait` back-off applies
  (30 s, then 60 s, ...). Inspect's retry loop finishes the whole set before each retry. A retry reuses
  the failed log's completed samples and re-runs only the rest. In this mode `retry_attempts` counts
  attempts, not retries.
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
  status and sample counts, so the orchestrator (FX-6) can find logs without scanning.

Every default can be overridden by keyword (`fail_on_error=`, `retry_wait=`, `max_tasks=`, ...); any other
keyword goes to `eval_set` unchanged. Returns `eval_set`'s `(success, logs)`. The logs are headers
without samples; read one in full with `inspect_ai.log.read_eval_log(log.location)`.
"""

import json
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval_set, task_identifier
from inspect_ai.log import EvalLog, read_eval_log_sample_summaries
from inspect_ai.model import Model

from .models import COSTS_PATH, Profile, agent_model, eval_cost_kwargs, load_profile, require_preflight, role_models

INDEX_NAME = "runner_index.json"
ERROR_DEFAULTS: dict[str, Any] = {
    "fail_on_error": 0.02,  # PC5: harness errors < 2% (Inspect fails the task at errors >= 2% of planned samples)
    "retry_on_error": 2,  # per sample, for transient API and tool errors
    "retry_attempts": 3,  # per task (attempts, with retry_immediate=False)
    "retry_wait": 30,  # seconds, doubled after each failed attempt
    "retry_immediate": False,  # True would retry at once, ignoring retry_wait
}


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
    **eval_kwargs: Any,
) -> tuple[bool, list[EvalLog]]:
    """Run `tasks` as one resumable eval set in `log_dir` (see the module docstring).

    `profile` is a name or a loaded `Profile` (default: $APE_MODEL_PROFILE, else "gate"). `roles` are the
    Inspect roles built from it (ignored when `model_roles` is given). `costs_path` is the price table
    for both Inspect and the preflight.
    """
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
    started = _now()
    success, logs = eval_set(tasks, log_dir=str(log_dir), model=agent, model_roles=role_map or None, epochs=epochs, **options)
    write_index(log_dir, logs, {"started": started, "finished": _now(), "success": success, "profile": p.name, "live": live})
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
    """Upsert one entry per log (keyed by task identifier) and append `run` to `<log_dir>/runner_index.json`."""
    index = read_index(log_dir)
    entries = dict(index_entry(log) for log in logs)
    index["tasks"].update(entries)
    index["runs"].append({**run, "tasks": sorted(entries)})
    index["updated"] = _now()
    path = Path(log_dir) / INDEX_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)
    return path
