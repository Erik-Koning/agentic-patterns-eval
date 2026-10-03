"""Dev-split tuning with an equal, logged budget per system (GATE_PREREG §5, precondition PC6).

Each system's tuner declares its candidate configurations in advance in
`config/tuning_grid.yaml`: the skeptic for LightRAG and S3s, the APG owner for APG.
The runner refuses more than `budget_per_system` candidates. It runs every candidate on
the same dev cells and appends each result to `cache/tuning_log.jsonl`. Selection rule
(pre-registered): highest mean dev success; candidates within `tie_pp` of the best go to
the cheaper one. Cost is Inspect's, priced from `config/model_costs.yaml`; an unpriced call
fails the candidate rather than reading as $0 and winning the tie-break (mockllm is exempt).

Each candidate runs as one `ape.runner.run_evals` set over all dev cells (FX-3: retries, the 2%
error budget, resume), in its own log dir `<log_dir>/<system>/<id>-<spec hash>` (default log_dir
`cache/tuning_logs`; the gate orchestrator passes its run's): its env knobs are not part of Inspect's
task identity, so candidates must not share a dir. Re-running `tune` reuses each candidate's finished
logs. A sample that errored within the error budget counts as a failure.

    uv run python -m ape.tuning --system LightRAG [--profile gate]   # agent + kg models from config/models.yaml
"""

import argparse
import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT, Config

GRID_PATH = ROOT / "config" / "tuning_grid.yaml"


@contextlib.contextmanager
def env(overrides: dict[str, str]) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in overrides}
    os.environ.update({k: str(v) for k, v in overrides.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load_grid(path: Path = GRID_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def candidates(grid: dict, system: str) -> list[dict]:
    cands = grid["systems"][system]["candidates"]
    if len(cands) > grid["budget_per_system"]:
        raise ValueError(f"{system}: {len(cands)} candidates exceed the equal budget of {grid['budget_per_system']}")
    return cands


def planned_tuning(cells: Iterable[tuple[str, dict]]) -> dict[str, dict]:
    """What a study's run_plan.yaml tuning cells price, per arm: {arm: {"candidates": {cell id: count}, "cells": [its
    task cells, in plan order]}}. `cells` are (cell id, spec) pairs; a spec's `arms` is {arm: candidates} or a list."""
    out: dict[str, dict] = {}
    for cell_id, spec in cells:
        arms = spec["arms"] if isinstance(spec["arms"], dict) else dict.fromkeys(spec["arms"], 1)
        for arm, n in arms.items():
            p = out.setdefault(arm, {"candidates": {}, "cells": []})
            p["candidates"][cell_id] = int(n)
            p["cells"] += [c for c in spec["cells"] if c not in p["cells"]]
    return out


def study_grid_problems(grid: dict, planned: dict[str, dict] | None = None, env_problems: Callable[[str, dict[str, str]], list[str]] | None = None) -> list[str]:
    """What a study grid (`config/tuning_grid_<study>.yaml`) breaks of the tuning rules, found before any dev run: the
    PC6 rules that can be checked up front, so a broken candidate is found before it is paid for. A study system is
    keyed by the plan arm it tunes (`run_study`: that arm runs as its selection, under every selection's knobs at once):
    - every system declares 1 to `budget_per_system` candidates with distinct ids, and exactly that many when the grid
      sets `equal_budgets`;
    - every candidate runs as the system's own arm;
    - no two systems' candidates set the same variable (`run_study.selection_env` would refuse the selections at the
      pilot, after the tune is paid for);
    - each candidate's env passes `env_problems(system, env)`, the arm's own check (`agent.multi.knobs.candidate_problems`:
      a multi-agent arm's candidates set only its own knobs, with values they take);
    - an arm the grid lists under `inherited` (configured elsewhere, e.g. by the gate) is not also a system;
    - with `planned` (`planned_tuning` over the study's run_plan tuning cells): each system is priced there, with its
      candidate count in every cell that names it and its dev cells exactly those cells', and every arm the plan prices
      is a system; an inherited arm the plan still prices is a redundant spend."""
    problems: list[str] = []
    budget = int(grid.get("budget_per_system") or 0)
    systems, inherited = grid.get("systems") or {}, grid.get("inherited") or {}
    setters: dict[str, set[str]] = {}
    for name, sdef in systems.items():
        cands = sdef.get("candidates") or []
        ids = [c.get("id") for c in cands]
        if not 1 <= len(cands) <= budget:
            problems.append(f"{name}: {len(cands)} candidates; a system declares 1 to budget_per_system ({budget})")
        elif grid.get("equal_budgets") and len(cands) != budget:
            problems.append(f"{name}: {len(cands)} candidates; equal budgets give every system budget_per_system ({budget})")
        if dup := sorted({str(i) for i in ids if ids.count(i) > 1}):
            problems.append(f"{name}: candidate ids {dup} are declared more than once")
        for c in cands:
            if c.get("arm") != name:
                problems.append(f"{name}: candidate {c.get('id')} runs {c.get('arm')}, not {name} (a system's candidates run as the plan arm it is named for)")
            env = {str(k): str(v) for k, v in (c.get("env") or {}).items()}
            for k in env:
                setters.setdefault(k, set()).add(name)
            if env_problems is not None:
                problems += [f"{name}: candidate {c.get('id')}: {p}" for p in env_problems(name, env)]
        if name in inherited:
            problems.append(f"{name} is both tuned here and inherited ({inherited[name]})")
    for k, names in sorted(setters.items()):
        if len(names) > 1:
            problems.append(f"{k} is set by the candidates of {', '.join(sorted(names))}: every selection's knobs apply together, so each system sets only its own")
    if planned is not None:
        for arm, p in planned.items():
            where = ", ".join(p["candidates"])
            if arm in inherited:
                source = (inherited[arm] or {}).get("source") if isinstance(inherited[arm], dict) else inherited[arm]
                problems.append(f"{arm} is inherited ({source}), yet {where} prices tuning it: a redundant spend; drop it from the plan")
            elif arm not in systems:
                problems.append(f"{where} prices tuning {arm}, which the grid does not declare")
        for name, sdef in systems.items():
            if (p := planned.get(name)) is None:
                problems.append(f"{name}: no run_plan.yaml tuning cell prices it")
                continue
            n = len(sdef.get("candidates") or [])
            if bad := {c: k for c, k in p["candidates"].items() if k != n}:
                problems.append(f"{name}: {n} candidates, but run_plan.yaml prices {bad}")
            cells = list(sdef.get("dev_cells") or grid.get("dev_cells") or [])
            if set(cells) != set(p["cells"]):
                problems.append(f"{name}: dev cells {cells}, but run_plan.yaml prices it on {p['cells']}")
    return problems


def _priced_cost(log, label: str) -> float:
    """Inspect $ for one log. A non-mock call without a price would read as $0 and break the
    cost tie-break, so it is an error; mockllm (offline tests) has no price and counts as $0."""
    usage = [(m, u) for s in log.samples for m, u in (s.model_usage or {}).items()]
    if unpriced := sorted({m for m, u in usage if u.total_cost is None and not m.startswith("mockllm/")}):
        raise RuntimeError(f"{label}: no Inspect cost for {unpriced}; price them in config/model_costs.yaml")
    return sum(u.total_cost or 0.0 for _, u in usage)


def candidate_dir(log_dir: str | Path, cand: dict) -> Path:
    """`<log_dir>/<id>-<hash of the candidate spec>`: a changed spec never resumes another's logs."""
    digest = hashlib.sha256(json.dumps(cand, sort_keys=True).encode()).hexdigest()[:10]
    return Path(log_dir) / f"{cand['id']}-{digest}"


def _cell(log) -> str:
    """A log's task cell: `family-level` from its task args (a task without a `family` arg, e.g. an F8 session, records
    its family in the eval metadata)."""
    args = log.eval.task_args
    return f"{args.get('family') or (log.eval.metadata or {}).get('family')}-{args['level']}"


def _succeeded(sample) -> bool:
    """An errored sample (tolerated under fail_on_error) has no scores and counts as a failure."""
    score = (sample.scores or {}).get("task_success")
    return score is not None and score.value == "C"


def _gate_task(cell: str, cand: dict, limit_worlds: int | None):
    from .tasks.gate import gate

    family, level = cell.split("-", 1)
    return gate(family=family, level=level, split="dev", arm=cand["arm"], delivery=cand.get("delivery", "push"), limit_worlds=limit_worlds)


def run_candidate(
    cand: dict,
    cells: list[str],
    model,
    model_roles: dict,
    limit_worlds: int | None,
    epochs: int,
    log_dir: str | Path,
    profile=None,
    costs_path: Path | None = None,
    make_task: Callable[[str, dict, int | None], Any] | None = None,
    sample_success: Callable[[Any], float] | None = None,
) -> dict:
    """One candidate on every dev cell. `make_task(cell, cand, limit_worlds)` builds a cell's task (default: the gate
    task on the dev split; a study orchestrator passes its own, e.g. the main study's task or an F8 session);
    `sample_success(sample)` scores one sample in [0, 1] (default: 1 when `task_success` is C)."""
    from inspect_ai.log import read_eval_log

    from .runner import log_path, run_evals

    make_task = make_task or _gate_task
    sample_success = sample_success or _succeeded
    with env(cand.get("env", {})):
        tasks = [make_task(cell, cand, limit_worlds) for cell in cells]
        # allow_dirty: logs of an earlier run with other models or --limit-worlds may share the dir.
        prices = {"costs_path": costs_path} if costs_path is not None else {}
        success, logs = run_evals(
            tasks, candidate_dir(log_dir, cand), profile=profile, model=model, model_roles=model_roles, epochs=epochs, display="none", log_dir_allow_dirty=True, **prices
        )
    if not success:
        failed = [f"{_cell(h)}: {h.error.message if h.error else h.status}" for h in logs if h.status != "success"]
        raise RuntimeError(f"{cand['id']}: " + "; ".join(failed))
    headers = {_cell(h): h for h in logs}
    per_cell, cost, files = {}, 0.0, []
    for cell in cells:
        log = read_eval_log(headers[cell].location)
        scores = [float(sample_success(s)) for s in log.samples]
        per_cell[cell] = sum(scores) / len(scores)
        cost += _priced_cost(log, f"{cand['id']} {cell}")
        files.append(log_path(log))
    return {"per_cell": per_cell, "mean_success": sum(per_cell.values()) / len(per_cell), "cost_usd": cost, "log_files": files}


def select(records: list[dict], tie_pp: float) -> dict:
    """Pre-registered rule over candidates that ran: max mean dev success; ties within `tie_pp` -> lower cost.
    A candidate whose run failed (errors beyond the budget after retries) cannot be selected."""
    ran = [r for r in records if r.get("status", "ok") == "ok"]
    if not ran:
        raise RuntimeError("no tuning candidate completed; see the tuning log for each failure")
    best = max(r["mean_success"] for r in ran)
    tied = [r for r in ran if r["mean_success"] >= best - tie_pp / 100.0]
    return min(tied, key=lambda r: (r["cost_usd"], r["candidate"]["id"]))


def tune(
    system: str,
    model,
    model_roles: dict,
    limit_worlds: int | None = None,
    epochs: int = 1,
    grid: dict | None = None,
    log_path: Path | None = None,
    profile=None,
    log_dir: Path | None = None,
    costs_path: Path | None = None,
    make_task: Callable[[str, dict, int | None], Any] | None = None,
    sample_success: Callable[[Any], float] | None = None,
) -> dict:
    """Run every candidate of `system`, log each to `log_path` (default cache/tuning_log.jsonl), and return
    the selected record. Eval logs go to `<log_dir>/<system>/...` (default cache/tuning_logs); `costs_path`
    is the price table for Inspect (default config/model_costs.yaml). Candidates run on the system's own
    `dev_cells` when the grid gives it some (a study grid whose systems tune on different cells), else on the
    grid's; `make_task` and `sample_success` as in `run_candidate`."""
    grid = grid or load_grid()
    cfg = Config()
    log_path = log_path or cfg.cache_dir / "tuning_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    system_dir = Path(log_dir or cfg.cache_dir / "tuning_logs") / system
    records = []
    for cand in candidates(grid, system):
        try:
            cells = grid["systems"][system].get("dev_cells") or grid["dev_cells"]
            result = {"status": "ok", **run_candidate(cand, cells, model, model_roles, limit_worlds, epochs, system_dir, profile, costs_path, make_task, sample_success)}
        except RuntimeError as e:  # run failure: logged (PC6 evidence) and unselectable; config errors still raise
            result = {"status": "failed", "error": str(e)[:500], "mean_success": None, "cost_usd": None}
        rec = {"system": system, "candidate": cand, **result, "ts": time.time()}
        records.append(rec)
        with log_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    chosen = select(records, grid.get("tie_pp", 1.0))
    with log_path.open("a") as f:
        f.write(json.dumps({"system": system, "selected": chosen["candidate"]["id"], "rule": "max mean dev success; ties -> lower cost", "ts": time.time()}) + "\n")
    return chosen


def main() -> None:
    from .models import agent_model, embedding_model, load_profile, require_preflight, role_models

    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--profile", help="config/models.yaml profile (default: $APE_MODEL_PROFILE or gate)")
    ap.add_argument("--model", help="override the profile's agent model (keeps its effort)")
    ap.add_argument("--kg", help="override the profile's kg model (keeps its effort)")
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    profile = load_profile(args.profile)
    os.environ.setdefault("APE_EMBEDDING_MODEL", embedding_model(profile))
    require_preflight(profile, live=True, overrides={"agent": args.model, "kg": args.kg})
    agent, roles = agent_model(profile, model=args.model), role_models(profile, ("kg",), model=args.kg)
    chosen = tune(args.system, agent, roles, args.limit_worlds, args.epochs, profile=profile)
    print(json.dumps({"selected": chosen["candidate"], "mean_success": chosen["mean_success"], "cost_usd": chosen["cost_usd"]}, indent=1))


if __name__ == "__main__":
    main()
