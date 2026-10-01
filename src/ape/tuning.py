"""Dev-split tuning with an equal, logged budget per system (GATE_PREREG §5, precondition PC6).

Each system's tuner declares its candidate configurations in advance in
`config/tuning_grid.yaml`: the skeptic for LightRAG and S3s, the APG owner for APG.
The runner refuses more than `budget_per_system` candidates. It runs every candidate on
the same dev cells and appends each result to `cache/tuning_log.jsonl`. Selection rule
(pre-registered): highest mean dev success; candidates within `tie_pp` of the best go to
the cheaper one.

    uv run python -m ape.tuning --system LightRAG --model openai/<agent> --kg openai/<kg>
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


def run_candidate(cand: dict, cells: list[str], model, model_roles: dict, limit_worlds: int | None, epochs: int, log_dir: str) -> dict:
    from inspect_ai import eval as inspect_eval

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
            )[0]
            if log.status != "success":
                raise RuntimeError(f"{cand['id']} {cell}: {log.error}")
            scores = [s.scores["task_success"].value == "C" for s in log.samples]
            per_cell[cell] = sum(scores) / len(scores)
            cost += sum((u.total_cost or 0.0) for s in log.samples for u in (s.model_usage or {}).values())
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kg", required=True)
    ap.add_argument("--limit-worlds", type=int)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    chosen = tune(args.system, args.model, {"kg": args.kg}, args.limit_worlds, args.epochs)
    print(json.dumps({"selected": chosen["candidate"], "mean_success": chosen["mean_success"], "cost_usd": chosen["cost_usd"]}, indent=1))


if __name__ == "__main__":
    main()
