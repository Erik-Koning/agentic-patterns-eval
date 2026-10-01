"""Dev-split tuning with an equal, logged budget per system (GATE_PREREG §5, precondition PC6).

Each system's tuner declares its candidate configurations in advance in
`config/tuning_grid.yaml`: the skeptic for LightRAG and S3s, the APG owner for APG.
The runner refuses more than `budget_per_system` candidates. It runs every candidate on
the same dev cells and appends each result to `cache/tuning_log.jsonl`. Selection rule
(pre-registered): highest mean dev success; candidates within `tie_pp` of the best go to
the cheaper one. Cost is Inspect's, priced from `config/model_costs.yaml`; an unpriced call
fails the candidate rather than reading as $0 and winning the tie-break (mockllm is exempt).

    uv run python -m ape.tuning --system LightRAG [--profile gate]   # agent + kg models from config/models.yaml
"""

import argparse
import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

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


def _priced_cost(log, label: str) -> float:
    """Inspect $ for one log. A non-mock call without a price would read as $0 and break the
    cost tie-break, so it is an error; mockllm (offline tests) has no price and counts as $0."""
    usage = [(m, u) for s in log.samples for m, u in (s.model_usage or {}).items()]
    if unpriced := sorted({m for m, u in usage if u.total_cost is None and not m.startswith("mockllm/")}):
        raise RuntimeError(f"{label}: no Inspect cost for {unpriced}; price them in config/model_costs.yaml")
    return sum(u.total_cost or 0.0 for _, u in usage)


def run_candidate(cand: dict, cells: list[str], model, model_roles: dict, limit_worlds: int | None, epochs: int, log_dir: str) -> dict:
    from inspect_ai import eval as inspect_eval

    from .models import eval_cost_kwargs
    from .tasks.gate import gate

    per_cell, cost, files = {}, 0.0, []
    with env(cand.get("env", {})):
        for cell in cells:
            family, level = cell.split("-", 1)
            log = inspect_eval(
                gate(family=family, level=level, split="dev", arm=cand["arm"], delivery=cand.get("delivery", "push"), limit_worlds=limit_worlds),
                model=model,
                model_roles=model_roles,
                epochs=epochs,
                log_dir=log_dir,
                display="none",
                **eval_cost_kwargs(),
            )[0]
            if log.status != "success":
                raise RuntimeError(f"{cand['id']} {cell}: {log.error}")
            scores = [s.scores["task_success"].value == "C" for s in log.samples]
            per_cell[cell] = sum(scores) / len(scores)
            cost += _priced_cost(log, f"{cand['id']} {cell}")
            files.append(log.location)
    return {"per_cell": per_cell, "mean_success": sum(per_cell.values()) / len(per_cell), "cost_usd": cost, "log_files": files}


def select(records: list[dict], tie_pp: float) -> dict:
    best = max(r["mean_success"] for r in records)
    tied = [r for r in records if r["mean_success"] >= best - tie_pp / 100.0]
    return min(tied, key=lambda r: (r["cost_usd"], r["candidate"]["id"]))


def tune(system: str, model, model_roles: dict, limit_worlds: int | None = None, epochs: int = 1, grid: dict | None = None, log_path: Path | None = None) -> dict:
    grid = grid or load_grid()
    cfg = Config()
    log_path = log_path or cfg.cache_dir / "tuning_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for cand in candidates(grid, system):
        result = run_candidate(cand, grid["dev_cells"], model, model_roles, limit_worlds, epochs, str(cfg.cache_dir / "tuning_logs"))
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
    chosen = tune(args.system, agent, roles, args.limit_worlds, args.epochs)
    print(json.dumps({"selected": chosen["candidate"], "mean_success": chosen["mean_success"], "cost_usd": chosen["cost_usd"]}, indent=1))


if __name__ == "__main__":
    main()
