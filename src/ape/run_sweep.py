"""The context-length sweep's runner (CONTEXT_SWEEP.md, D-054): runs on its own, resumable, re-runnable.

    uv run --locked python -m ape.run_sweep <build|run|analyze|all> --run-id <id> [--offline]
        [--models luna,sol,astra] [--kinds fresh,memo,followup] [--sizes 8000,64000,...] [--rerun | --replicate N]
        [--tasks-per-kind 1] [--max-samples 6] [--budget-usd X] [--mock gold|naive|refuse-largest]

It needs no other study's run. Steps (`all` runs the three in order):

    build    the worlds: per kind, `gen_sweep.generate_set` at every size (deterministic; a world already on disk is
             kept when it is identical and refused when it differs). Live: worlds/sweep/; offline: the run's work/.
    run      one Inspect eval set per tier (Luna, Sol, Astra at high effort, the plan cells' profiles) and replicate,
             in runs/context_sweep/<run id>/rep-<n>/<tier>/, through `ape.runner.run_evals` (retries, the usage
             ledger, the spend registry, the per-sample cost guard). Re-running the same command resumes the latest
             replicate: completed samples are kept. `--rerun` starts the next replicate: the same tasks, sampled
             again. `--replicate N` resumes replicate N.
    analyze  `ape.analyze_sweep.analyze`: report/report.md, results.csv and results.json over every replicate.

**The design** (kinds, sizes, tiers, tasks per kind, world seeds, tool-file size) defaults to the plan cells
(config/run_plan.yaml `studies.context_sweep`) and is fixed in run.json at a run's first step, so its replicates stay
comparable; a later flag that changes it is refused (start another run id). `--models` may name a subset of the run's
tiers for one invocation. `--tasks-per-kind N` gives each kind N tasks (seeds `gen_sweep.task_seed`): more tasks, not
replicates, are what widen the evidence beyond the particular tasks.

**Money (live only).** The plan cells are off by default (`enabled: false`, as the M5 add-on's), so the program total
does not count the sweep until it is switched on: a live run refuses a tier whose cell is off. Before each tier the
conservative projection of its remaining samples (`budget.sweep_sample_usd`: 2 generations of the whole context, at
the long-context rate above a model's threshold) must fit both what is left of the program's budget
(budget.total_usd or `--budget-usd`, less the registry's spend and the sweep's long-context surcharge, which
Inspect's flat prices miss) and of the sweep's allocation (budget.allocations.context_sweep, less the sweep's own
spend from its usage ledgers at long-context rates). `run_evals` adds its preflight (prices, the API key, the pinned
snapshots).

**Offline** (`--offline`, zero spend): a mockllm agent (`--mock gold`, the default, knows every probe's gold;
`naive` decides every probe the same way; `refuse-largest` is gold but refuses the largest size as the provider would,
to rehearse the not-measured path), worlds and a spend registry under the run's work/, and an OpenAI key and base URL
that reach nothing. Offline and live runs never share a run id.

Every eval set runs under its own cache nonce (run, replicate, tier), so provider prompt caches never cross
replicates; within a set, the sizes of a kind share their history's prefix and may hit the cache.

**A size the provider refuses** (`over_limit`, `agent.sweep`) ends that sample at once, without an Inspect error, so
the tier's other sizes still run and nothing is retried; the sample is "not measured", never wrong. After each tier
the runner lists the refused sizes (run.json `over_limit_sizes`) and prints how to recover: a new run at every size,
with `--output-tokens 2000` (the same sizes in about half as many messages; 2000 is the most that still fits 8K) or
without the refused size. It never
re-renders one size on its own, since every size must share one rendering for the comparison to hold.
"""

import argparse
import contextlib
import json
import os
import secrets
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .analyze_sweep import RECOVERY
from .budget import BudgetError, Prices, load_assumptions, load_plan, program_spend, require_affordable, sweep_sample_usd
from .config import ROOT, Config
from .models import load_profile
from .worlds import gen_sweep
from .worlds.spec import World

STUDY = "context_sweep"
RUNS = ROOT / "runs" / STUDY
OFFLINE_KEY = "sk-ape-offline-run-no-network"
OFFLINE_BASE_URL = "http://127.0.0.1:9/v1"  # the discard port: a stray OpenAI call fails at once, locally
MAX_SAMPLES = 6  # concurrent samples per tier: a 960K request is heavy on tokens-per-minute limits
STEPS = ("build", "run", "analyze")


class SweepError(RuntimeError):
    """A step refuses to run (design change, cells off, budget)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- Plan and run design ------------------------------------------------------------------------------------


def plan_cells(plan=None) -> dict:
    """tier -> the plan cell (`sweep.<tier>`) of the context_sweep study."""
    plan = plan or load_plan()
    cells = {c.id.split(".", 1)[1]: c for c in plan.cells if c.study == STUDY and c.kind == "sweep"}
    if not cells:
        raise SweepError(f"config/run_plan.yaml has no `{STUDY}` sweep cells")
    return cells


def default_design(plan=None) -> dict:
    cells = plan_cells(plan)
    first = next(iter(cells.values())).spec
    return {
        "tiers": list(cells),
        "kinds": list(first["kinds"]),
        "sizes": sorted(int(s) for s in first["sizes"]),
        "tasks_per_kind": int(first.get("tasks_per_kind", 1)),
        "seed_base": gen_sweep.SEED_BASE,
        "output_tokens": int((first.get("knobs") or {}).get("output_tokens", gen_sweep.DEFAULT_OUTPUT_TOKENS)),
    }


def tasks_per_kind(design: dict) -> int:
    return int(design.get("tasks_per_kind", 1))  # runs created before the field ran one task per kind


def run_dir(run_id: str) -> Path:
    if not run_id or "/" in run_id or run_id.startswith("."):
        raise SweepError(f"bad run id {run_id!r}")
    return RUNS / run_id


def load_run(run_id: str, offline: bool, overrides: dict | None = None) -> dict:
    """The run's record (run.json), created at its first step from the plan's design and `overrides`; a later call
    whose overrides differ from the fixed design is refused, as is mixing offline and live."""
    d = run_dir(run_id)
    path = d / "run.json"
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if path.exists():
        run = json.loads(path.read_text())
        if run["offline"] != offline:
            raise SweepError(f"run {run_id} is {'offline' if run['offline'] else 'live'}; use another run id")
        if clash := {k: v for k, v in overrides.items() if k != "tiers" and run["design"].get(k) != v}:
            raise SweepError(f"run {run_id} fixed its design at its first step; {sorted(clash)} differ from it (start another run id)")
        if extra := sorted(set(overrides.get("tiers") or []) - set(run["design"]["tiers"])):
            raise SweepError(f"run {run_id} has no tier(s) {extra}; its tiers are {run['design']['tiers']}")
        return run
    design = default_design() | {k: v for k, v in overrides.items() if k != "tiers"}
    if bad := sorted(set(design["kinds"]) - set(gen_sweep.KINDS)):
        raise SweepError(f"unknown kind(s) {bad}; one of {gen_sweep.KINDS}")
    if tasks_per_kind(design) < 1:
        raise SweepError("--tasks-per-kind must be at least 1")
    known = plan_cells()
    if bad := sorted(set(overrides.get("tiers") or []) - set(known)):
        raise SweepError(f"unknown tier(s) {bad}; the plan has {sorted(known)}")
    run = {"run_id": run_id, "study": STUDY, "offline": offline, "created": _now(), "nonce_seed": secrets.token_hex(8), "design": design, "replicates": {}, "worlds": {}}
    save_run(run)
    return run


def save_run(run: dict) -> None:
    path = run_dir(run["run_id"]) / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run, indent=1, sort_keys=True))
    os.replace(tmp, path)


def label(run: dict) -> str:
    return f"{STUDY}{'-offline' if run['offline'] else ''}/{run['run_id']}"


@contextlib.contextmanager
def environment(run: dict, extra: dict | None = None) -> Iterator[None]:
    """The spend label; offline: worlds, cache and spend registry under the run's work/, and a key and base URL that
    reach nothing. Restored afterwards."""
    work = run_dir(run["run_id"]) / "work"
    updates: dict[str, str | None] = {"APE_SPEND_LABEL": label(run)}
    if run["offline"]:
        updates |= {"APE_WORLDS": str(work / "worlds"), "APE_CACHE": str(work / "cache"), "APE_SPEND_REGISTRY": str(work / "spend_registry.jsonl"),
                    "OPENAI_API_KEY": OFFLINE_KEY, "OPENAI_BASE_URL": OFFLINE_BASE_URL}  # fmt: skip
    updates |= extra or {}
    before = {k: os.environ.get(k) for k in updates}
    try:
        for k, v in updates.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- Steps -------------------------------------------------------------------------------------------------


def build(run: dict) -> dict:
    """Every world of the design, saved (or checked identical) under the run's worlds directory."""
    d = run["design"]
    built = {}
    with environment(run):
        cfg = Config()
        for j, kind in ((j, k) for j in range(tasks_per_kind(d)) for k in d["kinds"]):
            try:
                worlds = gen_sweep.generate_set(kind, d["sizes"], gen_sweep.task_seed(kind, j, d["seed_base"]), d["output_tokens"])
            except ValueError as e:
                raise SweepError(f"{e}: use --output-tokens 2000 or less, or drop the smallest size (a new run id)") from e
            for w in worlds:
                path = cfg.world_path(w.id)
                h = w.content_hash()
                if path.exists():
                    if World.load(path).content_hash() != h:
                        raise SweepError(f"{path} exists and differs from the generator's world {w.id}: refusing to overwrite it")
                else:
                    w.save(path)
                built[w.id] = {"hash": h, "context_tokens": w.entities["sweep"]["context_tokens"], "middle_cases": w.entities["sweep"]["middle_cases"]}
    run["worlds"] = built
    save_run(run)
    return {"worlds": len(built)}


def _completed(log_dir: Path) -> int:
    from .runner import latest_tasks, read_index

    entries = latest_tasks(read_index(log_dir)).values() if log_dir.exists() else []
    return sum(int((e.get("samples") or {}).get("completed") or 0) - int((e.get("samples") or {}).get("errored") or 0) for e in entries)


def projection(run: dict, tier: str, log_dir: Path | None = None) -> float:
    """Conservative $ for the tier's remaining samples (whole remaining fraction of the replicate)."""
    d = run["design"]
    cell = plan_cells()[tier]
    model = load_profile(cell.spec["profile"]).role("agent").model
    A, P = load_assumptions(), Prices.load()
    full = sum(sweep_sample_usd(model, s, A, P) for s in d["sizes"]) * len(d["kinds"]) * tasks_per_kind(d)
    planned = len(d["sizes"]) * len(d["kinds"]) * tasks_per_kind(d)
    done = min(planned, _completed(log_dir)) if log_dir is not None else 0
    return full * (planned - done) / planned


def sweep_spend(registry: Path | None = None) -> dict:
    """The sweep's live spend from the registry's context_sweep dirs: their usage ledgers priced at long-context
    rates (`usd`) and at Inspect's flat prices (`flat_usd`)."""
    from . import spend as reg
    from .analyze_sweep import ledger_spend

    registry = registry if registry is not None else reg.registry_path(live=True)
    dirs, _ = reg.registered(registry)
    usd = flat = 0.0
    for d, info in dirs.items():
        if info["study"] == STUDY and Path(d).is_dir():
            s = ledger_spend(Path(d))
            usd, flat = usd + s["usd"], flat + s["flat_usd"]
    return {"usd": usd, "flat_usd": flat, "surcharge_usd": max(0.0, usd - flat)}


def guard(run: dict, projected: float, budget_usd: float | None = None) -> dict:
    """Refuse (BudgetError) unless `projected` fits what is left of the program and of the sweep's allocation."""
    plan = load_plan()
    total = float(budget_usd if budget_usd is not None else plan.budget["total_usd"])
    allocation = float((plan.budget.get("allocations") or {}).get(STUDY, total))
    program = program_spend()
    mine = sweep_spend()
    program_left = total - program["spent_usd"] - mine["surcharge_usd"]
    sweep_left = allocation - mine["usd"]
    require_affordable(projected, min(program_left, sweep_left), "the sweep's next tier")
    return {"projected_usd": projected, "program_left_usd": program_left, "sweep_left_usd": sweep_left}


def replicate_for(run: dict, rerun: bool, replicate: int | None) -> int:
    reps = sorted(int(k) for k in run["replicates"])
    if replicate is not None:
        if rerun:
            raise SweepError("--rerun starts the next replicate; it cannot be combined with --replicate")
        if replicate < 1 or (replicate not in reps and replicate != (reps[-1] + 1 if reps else 1)):
            raise SweepError(f"replicate {replicate} does not exist (replicates: {reps or 'none'})")
        return replicate
    if not reps:
        return 1
    return reps[-1] + 1 if rerun else reps[-1]


def _mock(run: dict, kind: str):
    from inspect_ai.model import get_model

    from .llm import mock_sweep

    if kind == "naive":
        return get_model(mock_sweep.MODEL, custom_outputs=mock_sweep.naive_sweep_agent)
    cfg = Config()
    worlds = [World.load(cfg.world_path(wid)) for wid in run["worlds"]]
    refuse = max(run["design"]["sizes"]) if kind == "refuse-largest" else None
    return get_model(mock_sweep.MODEL, custom_outputs=mock_sweep.gold_sweep_agent(worlds, refuse_from=refuse))


def run_tiers(run: dict, tiers: Sequence[str] | None = None, *, rerun: bool = False, replicate: int | None = None, max_samples: int = MAX_SAMPLES,
              budget_usd: float | None = None, mock: str = "gold") -> dict:  # fmt: skip
    """One eval set per tier for the chosen replicate (module docstring)."""
    from .runner import run_evals
    from .tasks.sweep import context_sweep

    d = run["design"]
    tiers = list(tiers or d["tiers"])
    cells = plan_cells()
    if not run["offline"] and (off := [t for t in tiers if not cells[t].enabled]):
        raise SweepError(
            f"the sweep's plan cell(s) {[cells[t].id for t in off]} are off: set `enabled: true` on them in config/run_plan.yaml "
            f"(studies.{STUDY}) so the program budget counts the sweep, then run again"
        )
    if missing := [w for w in _world_ids(run) if w not in run["worlds"]]:
        raise SweepError(f"{len(missing)} world(s) of the design are not built (run the build step first), e.g. {missing[0]}")
    rep = replicate_for(run, rerun, replicate)
    run["replicates"].setdefault(str(rep), {"started": _now(), "tiers": {}})  # recorded first: a killed replicate is resumed, not skipped
    save_run(run)
    results = {}
    for tier in tiers:
        log_dir = run_dir(run["run_id"]) / f"rep-{rep}" / tier
        cell = cells[tier]
        model = load_profile(cell.spec["profile"]).role("agent").model
        projected = projection(run, tier, log_dir)
        if not run["offline"] and projected > 0:
            guard(run, projected, budget_usd)
        nonce = f"{run['nonce_seed']}/rep-{rep}/{tier}"
        with environment(run, {"APE_CACHE_NONCE": nonce}):
            task = context_sweep(kinds=",".join(d["kinds"]), sizes=",".join(map(str, d["sizes"])), seed_base=d["seed_base"], output_tokens=d["output_tokens"],
                                 group=f"rep-{rep}", tasks_per_kind=tasks_per_kind(d))  # fmt: skip
            kwargs = {"model": _mock(run, mock)} if run["offline"] else {}
            ok, _ = run_evals(task, log_dir, profile=cell.spec["profile"], roles=(), sample_cost_usd=max(sweep_sample_usd(model, s) for s in d["sizes"]), max_samples=max_samples, max_tasks=1, **kwargs)
        refused = refused_sizes(rep, tier, log_dir)
        results[tier] = {"success": ok, "log_dir": str(log_dir), "projected_usd": projected, "over_limit_sizes": refused}
        run["replicates"][str(rep)]["tiers"][tier] = {"success": ok, "updated": _now(), "model": model, "over_limit_sizes": refused}
        save_run(run)
        if refused:
            print(f"warning: the provider refused {tier}'s requests at {', '.join(f'{s // 1000}K' for s in refused)} for size; "
                  f"those samples are not measured (not counted as wrong). {RECOVERY}")  # fmt: skip
        if not ok:
            raise SweepError(f"tier {tier} (replicate {rep}) did not complete; re-run the same command to resume it")
    return {"replicate": rep, "tiers": results}


def refused_sizes(rep: int, tier: str, log_dir: Path) -> list[int]:
    """The sizes at which the provider refused one of the tier's requests for size (`over_limit`) in this replicate."""
    from .analyze_sweep import _rows, over_limit_sizes

    if not log_dir.exists():
        return []
    return over_limit_sizes(_rows(rep, tier, log_dir, load_assumptions(), Prices.load()))


def _world_ids(run: dict) -> list[str]:
    d = run["design"]
    return [gen_sweep.world_id(k, s, gen_sweep.task_seed(k, j, d["seed_base"]), d["output_tokens"]) for j in range(tasks_per_kind(d)) for k in d["kinds"] for s in d["sizes"]]


def analyze(run: dict) -> dict:
    from .analyze_sweep import analyze as analyze_run

    return analyze_run(run_dir(run["run_id"]))


# --- CLI ---------------------------------------------------------------------------------------------------


def _csv(x: str | None, cast=str) -> list | None:
    return None if x is None else [cast(v.strip()) for v in x.split(",") if v.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ape.run_sweep", description="The context-length sweep (CONTEXT_SWEEP.md): build, run, analyze.")
    ap.add_argument("step", choices=(*STEPS, "all"))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--offline", action="store_true", help="mock agent, no network, no spend")
    ap.add_argument("--models", help="tiers to run this time (default: the run's): luna, sol, astra")
    ap.add_argument("--kinds", help="design (first step only): fresh,memo,followup")
    ap.add_argument("--sizes", help="design (first step only): target context sizes in tokens, e.g. 8000,64000")
    ap.add_argument("--seed-base", type=int, help=f"design (first step only): world seed base (default {gen_sweep.SEED_BASE}); a new base gives new tasks")
    ap.add_argument("--output-tokens", type=int, help="design (first step only): tool-file size knob")
    ap.add_argument("--tasks-per-kind", type=int, help="design (first step only): tasks (world seeds) per kind (default 1: 3 tasks per size)")
    ap.add_argument("--rerun", action="store_true", help="run the next replicate (the same tasks, sampled again)")
    ap.add_argument("--replicate", type=int, help="resume this replicate")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--budget-usd", type=float, help="a lower program budget for the guard")
    ap.add_argument("--mock", choices=("gold", "naive", "refuse-largest"), default="gold",
                    help="offline agent: gold answers, naive answers, or gold with the largest size refused as over the limit")  # fmt: skip
    a = ap.parse_args(argv)
    try:
        tiers = _csv(a.models)
        run = load_run(a.run_id, a.offline, {"kinds": _csv(a.kinds), "sizes": sorted(_csv(a.sizes, int)) if a.sizes else None, "seed_base": a.seed_base,
                                             "output_tokens": a.output_tokens, "tasks_per_kind": a.tasks_per_kind, "tiers": tiers})  # fmt: skip
        steps = STEPS if a.step == "all" else (a.step,)
        if "build" in steps:
            print(f"build: {build(run)['worlds']} worlds")
        if "run" in steps:
            res = run_tiers(run, tiers, rerun=a.rerun, replicate=a.replicate, max_samples=a.max_samples, budget_usd=a.budget_usd, mock=a.mock)
            print(f"run: replicate {res['replicate']}: " + ", ".join(f"{t} {'ok' if r['success'] else 'incomplete'}" for t, r in res["tiers"].items()))
        if "analyze" in steps:
            out = analyze(run)
            print(f"analyze: {out['rows']} samples, spend ${out['spend_usd']:.2f} -> {out['report']}")
    except (SweepError, BudgetError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
