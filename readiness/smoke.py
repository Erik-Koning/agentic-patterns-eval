"""Live smoke tests (readiness E3-E5, L2-L5, H4, H6 live, D-017; FIX_PLAN FX-8).

    uv run python readiness/smoke.py                      # every check, live, on config/models.yaml's gate profile
    uv run python readiness/smoke.py --dry                # offline wiring check (mock models, fake embeddings), $0
    uv run python readiness/smoke.py --only pull,burst    # some checks (with the checks they need)
    uv run python readiness/smoke.py --skip orchestrator  # all but some
    uv run python readiness/smoke.py --profile gate --agent openai/gpt-6-sol   # a flag swaps a role's model, keeps its effort

Checks, in run order (`STEPS`; pass criteria in `smoke_checks.py`):

    effort        reasoning effort honoured per role model: high uses more reasoning tokens than low
                  (read from cache/openai_probe.json; run readiness/probe_openai.py first)
    L2            LightRAG extraction on an F7-100 world; source ids map to our chunks
    D017          the build model's authoring ID coverage and LightRAG's ID coverage >= 0.95 on one F7-100 world
                  (`ape.build_quality`, needs L2): an early warning; the decisive D-017 check is run_gate
                  build-dev's, on the gate's four dev cells (build-dev/build_quality.json)
    H4            S3s on F3-60: OpenAI accepts tool sets that change across turns (warn if they never changed)
    L4_L5         LGR-s on F7-100: one keyword call per compile (L4); realized context <= max_total_tokens with
                  the median in the expected band (L5) (needs L2)
    APG           APG-s on F7-100 with the authored graph (needs D017)
    pull          pull delivery: S3s and APG-s on F7-100, 4 tasks; search_kb in >= 75% of samples, no errors
                  (needs D017)
    recovery      a harness-side fault on a sample's first attempt (smoke-only solver wrapper) is retried by
                  the FX-3 runner and recorded
    burst         24 samples at the profile's max_connections: failed samples, retries, rate-limit signals,
                  latency distribution, and a recommended max_connections
    retrieval     no LLM: S3s evidence recall at its budget and APG's embedding-shortlist gold rate on F7-100
                  (warn below 0.8; the authored graph when D017 built it, else the oracle graph)
    orchestrator  the real gate orchestrator at SMOKE_SCALE (`python -m ape.run_gate --smoke`): preflight,
                  build-dev, tune, anchor and pilot on F7-10 and F3-5 (1 world, 2 tasks, 2 candidates per
                  system, anchor 2 questions per type), then the freeze must refuse (a smoke run never freezes,
                  and the draft pre-registration's open items are listed)

Cost. Before any spend the script prints each selected check's projected cost (conservative, from
`ape.budget`'s priors and config/model_costs.yaml) and refuses to start if the total exceeds --max-usd
(default $3). Before each check it stops if spend so far plus that check's projection would exceed the cap;
the orchestrator also runs under run_gate's own budget guard with what is left. The gate profile's
projection (GPT-6 Luna, high; 2026-10-01 priors): L2 $0.10, D017 $0.04, H4 $0.01, L4_L5 $0.01, APG $0.01,
pull $0.05, recovery $0.01, burst $0.10, orchestrator $1.06 (of which $0.47 is the GraphRAG-Bench anchor index,
built with gpt-4o-mini; $0.93 in all with the Luna fallback), total ≈ $1.38. The effort check reads the probe,
which costs ≈ $0.02 on its own. `--dry` spends $0. Spend is this invocation's: Inspect-metered calls, plus the
growth of the build/embedding ledger and of the orchestrator run.

The orchestrator check runs run_gate's gate profile (run_plan.yaml), not the --agent/--kg/--build overrides.

Isolation. Worlds, indices and cache live in cache/smoke/; the orchestrator's run in cache/smoke/runs/
(smoke-live, or smoke-dry; re-runs resume it). Nothing is written to config/, PROVENANCE.md or runs/.
Eval logs go to a fresh cache/smoke/logs/<timestamp>/. Writes cache/smoke/report.json and report.md; exits
non-zero when a check fails.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_checks as sc  # noqa: E402

from ape import build_quality as bq  # noqa: E402  (D-017 coverage: shared with run_gate's build-dev check)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "smoke"
SPEC_THRESHOLD = bq.THRESHOLD  # D-017
F7_TASKS = 20  # the F7-100 world's tasks: enough for the retrieval rates; runs take the first n (Inspect `limit`)
RUN_TASKS = {"H4": 2, "L4_L5": 2, "APG": 2, "pull": 4, "recovery": 1}
BURST = {"tasks": 12, "epochs": 2}  # 24 samples
DEFAULT_MAX_USD = 3.0
# name -> (checks it needs, what it checks)
STEPS: dict[str, tuple[tuple[str, ...], str]] = {
    "effort": ((), "reasoning effort honoured per role model (from the probe)"),
    "L2": ((), "LightRAG extraction and source mapping on F7-100"),
    "D017": (("L2",), "build-quality check: authoring and LightRAG ID coverage >= 0.95"),
    "H4": ((), "S3s on F3-60: changing tool sets accepted"),
    "L4_L5": (("L2",), "LGR-s on F7-100: one keyword call per compile; realized context vs max_total_tokens"),
    "APG": (("D017",), "APG-s on F7-100"),
    "pull": (("D017",), "pull mode: S3s and APG-s on F7-100"),
    "recovery": ((), "a first-attempt harness fault is retried and recorded"),
    "burst": ((), "24 samples at the profile's max_connections"),
    "retrieval": ((), "real-embedding retrieval, no LLM"),
    "orchestrator": ((), "run_gate preflight..pilot at SMOKE_SCALE; freeze refuses"),
}


def _isolate(dry: bool) -> None:
    os.environ["APE_WORLDS"] = str(OUT / "worlds")
    os.environ["APE_INDICES"] = str(OUT / "indices")
    os.environ["APE_CACHE"] = str(OUT / "cache")
    if dry:
        os.environ["APE_EMBEDDINGS"] = "fake"


def _inspect_spend(logs, dry: bool) -> float:
    """Inspect-metered $ (agent and role calls), priced via `eval_cost_kwargs()`. An unpriced call is an
    error on a live run, since counting it as $0 would blind the --max-usd cap. Dry runs use mockllm,
    which has no price: those count as $0, with a note."""
    unpriced = sorted({m for log in logs for s in (log.samples or []) for m, u in (s.model_usage or {}).items() if u.total_cost is None})
    if unpriced and not dry:
        raise SystemExit(f"no Inspect cost for {unpriced}; refusing to count it as $0 (is the model in config/model_costs.yaml?)")
    if unpriced:
        print(f"note: --dry: no price for {unpriced}; counted as $0")
    return sum(u.total_cost or 0.0 for log in logs for s in (log.samples or []) for u in (s.model_usage or {}).values())


def _ledger_spend() -> float:
    """Ledger-metered $ (build and embedding calls outside Inspect) in this scratch area."""
    from ape.analysis.cost import ledger_frame, load_prices
    from ape.config import Config
    from ape.llm.ledger import Ledger

    led = Ledger(Config().ledger_path)
    if not led.read():
        return 0.0
    try:
        return float(ledger_frame(led, load_prices())["usd"].sum())
    except KeyError as e:  # an unpriced build/embedding model: never let it read as $0 (or NaN, which passes the cap)
        raise SystemExit(f"ledger spend cannot be priced: {e}") from e


def _spend(logs, dry: bool = False) -> float:
    return _inspect_spend(logs, dry) + _ledger_spend()


def _guard(report: dict, logs: list, max_usd: float, dry: bool = False) -> None:
    report["spend_usd"] = _spend(logs, dry)
    if report["spend_usd"] > max_usd:
        raise SystemExit(f"spend {report['spend_usd']:.4f} exceeds --max-usd {max_usd}; stopping")


class Spend:
    """This invocation's spend: its Inspect logs, the smoke ledger's growth, and the orchestrator run's growth."""

    def __init__(self, dry: bool):
        self.dry = dry
        self.logs: list = []
        self.ledger0 = _ledger_spend()
        self.orchestrator = 0.0

    def total(self) -> float:
        return _inspect_spend(self.logs, self.dry) + (_ledger_spend() - self.ledger0) + self.orchestrator


# --- selection and projection ----------------------------------------------------------------------


def select_steps(only: Sequence[str] | None, skip: Sequence[str] | None) -> list[str]:
    """The checks to run, in STEPS order: `only` (default all) minus `skip`, plus every check they need. A
    needed check that `skip` names is an error, since the dependent check cannot run without it."""
    for name in [*(only or []), *(skip or [])]:
        if name not in STEPS:
            raise SystemExit(f"unknown check {name!r}; checks are {', '.join(STEPS)}")
    chosen = set(only or STEPS) - set(skip or [])
    todo = list(chosen)
    while todo:
        for need in STEPS[todo.pop()][0]:
            if need in (skip or []):
                raise SystemExit(f"--skip {need}, but a selected check needs it ({', '.join(c for c in chosen if need in STEPS[c][0])})")
            if need not in chosen:
                chosen.add(need)
                todo.append(need)
    return [s for s in STEPS if s in chosen]


def _cell(step: str, profile: str, **spec) -> "object":
    from ape.budget import PlanCell

    return PlanCell(f"smoke.{step}", "smoke", "smoke", {"profile": profile, **spec})


def step_cells(profile: str) -> dict[str, list]:
    """The checks that call models, as ad-hoc plan cells for `ape.budget` (F7-100 is priced at its measured
    chunk count). effort reads the probe and retrieval embeds a few queries: both ~$0."""
    agent = lambda step, arms, cell, n, **kw: _cell(step, profile, arms=arms, cells=[cell], n_tasks=n, epochs=kw.pop("epochs", 1), **kw)  # noqa: E731
    return {
        "L2": [_cell("L2", profile, kind="build", systems=["lightrag"], worlds={"F7-100": 1})],
        "D017": [_cell("D017", profile, kind="build", systems=["apg"], worlds={"F7-100": 1})],
        "H4": [agent("H4", ["S3s"], "F3-60", RUN_TASKS["H4"])],
        "L4_L5": [agent("L4_L5", ["LGR-s"], "F7-100", RUN_TASKS["L4_L5"])],
        "APG": [agent("APG", ["APG-s"], "F7-100", RUN_TASKS["APG"])],
        "pull": [agent("pull", ["S3s", "APG-s"], "F7-100", RUN_TASKS["pull"], deliveries=["pull"])],
        "recovery": [agent("recovery", ["S3s"], "F3-5", RUN_TASKS["recovery"])],
        "burst": [agent("burst", ["S3s"], "F7-100", BURST["tasks"], epochs=BURST["epochs"])],
    }


def projections(steps: Sequence[str], profile_name: str, dry: bool, orchestrator_run: Callable[[], object] | None = None) -> dict[str, float]:
    """Conservative projected $ per selected check (live prices; a dry run spends $0 but shows the same numbers).
    The orchestrator's is run_gate's own per-phase projection at SMOKE_SCALE."""
    from ape.budget import Plan, projected_cost

    cells = step_cells(profile_name)
    out: dict[str, float] = {}
    for step in steps:
        if step in cells:
            out[step] = float(projected_cost(plan=Plan(tuple(cells[step]), {}, {}, ())))
        elif step == "orchestrator":
            from ape import run_gate as rg

            run = orchestrator_run() if orchestrator_run else None
            out[step] = float(sum(rg.PHASE_DEFS[p].projected(run) for p in sc.SMOKE_PHASES)) if run else 0.0
        else:
            out[step] = 0.0
    return out


def print_projection(proj: dict[str, float], max_usd: float) -> None:
    print("Projected cost per check (conservative, live prices):")
    for step, usd in proj.items():
        print(f"  {step:<13} ${usd:7.3f}   {STEPS[step][1]}")
    print(f"  {'total':<13} ${sum(proj.values()):7.3f}   cap --max-usd {max_usd}")


# --- helpers over logs -----------------------------------------------------------------------------


def _search_kb_calls(sample) -> int:
    return sum(1 for m in sample.messages if m.role == "assistant" for tc in (m.tool_calls or []) if tc.function == "search_kb")


def _model_events(sample) -> list[dict]:
    out = []
    for e in sample.events:
        if e.event != "model":
            continue
        usage = getattr(getattr(e, "output", None), "usage", None)
        out.append({"role": getattr(e, "role", None), "retries": e.retries or 0, "error": e.error, "working_time": e.working_time, "tokens": usage.total_tokens if usage else None})
    return out


def _logger_messages(sample) -> list[str]:
    return [getattr(e.message, "message", None) for e in sample.events if e.event == "logger"]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# --- main ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="config/models.yaml profile (default: $APE_MODEL_PROFILE or gate)")
    ap.add_argument("--agent", help="override the profile's agent model (keeps its effort)")
    ap.add_argument("--kg", help="override the profile's kg model (keeps its effort)")
    ap.add_argument("--build", help="override the build model (sets APE_BUILD_MODEL)")
    ap.add_argument("--embed", help="override the embedding model")
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", type=lambda s: [x for x in s.split(",") if x], help=f"comma-separated checks: {','.join(STEPS)}")
    ap.add_argument("--skip", type=lambda s: [x for x in s.split(",") if x], help="comma-separated checks to leave out")
    args = ap.parse_args(argv)
    steps = select_steps(args.only, args.skip)
    _isolate(args.dry)
    if args.profile:
        os.environ["APE_MODEL_PROFILE"] = args.profile
    if args.build:
        os.environ["APE_BUILD_MODEL"] = args.build

    from inspect_ai.log import read_eval_log

    from ape import run_gate as rg
    from ape.build import build
    from ape.config import Config
    from ape.models import agent_model, build_settings, embedding_model, load_profile, require_preflight, role_models
    from ape.runner import run_evals
    from ape.worlds.spec import World

    profile = load_profile()
    if args.embed:
        os.environ["APE_EMBEDDING_MODEL"] = args.embed
    os.environ.setdefault("APE_EMBEDDING_MODEL", embedding_model(profile))
    build_model, build_effort = build_settings(profile)
    report: dict = {
        "dry": args.dry,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": {"profile": profile.name, "roles": profile.summary(), "overrides": {k: v for k, v in vars(args).items() if k in ("agent", "kg", "build", "embed")}, "build": [build_model, build_effort]},
        "max_usd": args.max_usd,
        "steps": steps,
        "checks": {},
    }

    def orchestrator_run(budget_usd: float | None = None):
        return rg.GateRun("smoke-dry" if args.dry else "smoke-live", offline=args.dry, smoke=True, runs_root=OUT / "runs", budget_usd=budget_usd)

    # 1. Projection: refuse to start a run the cap cannot cover.
    try:
        proj = projections(steps, profile.name, args.dry, orchestrator_run)
    except (rg.PhaseError, rg.BudgetError) as e:  # e.g. the anchor projection needs the probe (E3)
        raise SystemExit(f"cannot project the smoke cost: {e}") from None
    report["projection_usd"] = proj | {"total": sum(proj.values())}
    print_projection(proj, args.max_usd)
    ok, total = sc.plan_fits(proj, args.max_usd)
    if not ok:
        raise SystemExit(f"projected ${total:.3f} exceeds --max-usd {args.max_usd}: raise the cap or select fewer checks (--only/--skip)")
    # 2. Preflight: prices for every model; live, the API key. Dry runs swap in mockllm agent/kg models.
    require_preflight(profile, live=not args.dry, overrides={} if args.dry else {"agent": args.agent, "kg": args.kg})
    if args.dry:  # mock models, still built from the profile so the effort wiring is exercised
        from ape.llm.mock_agent import mock_agent, mock_kg

        agent = agent_model(profile, model="mockllm/model", custom_outputs=mock_agent)
        roles = role_models(profile, ("kg",), model="mockllm/model", custom_outputs=mock_kg)
    else:
        agent = agent_model(profile, model=args.agent)
        roles = role_models(profile, ("kg",), model=args.kg)
    spend = Spend(args.dry)
    cfg = Config()
    log_root = OUT / "logs" / time.strftime("%Y%m%dT%H%M%S")
    provenance0, config0 = _sha(ROOT / "PROVENANCE.md"), _tree_hash(ROOT / "config")

    # 3. Worlds: F7-100 (relational, descriptive) with F7_TASKS tasks; F3-60 (H4) and F3-5 (recovery) with 2 each.
    #    Chunk embeddings are cached here, so the runs only read them.
    needs_worlds = set(steps) - {"effort", "orchestrator"}
    f7 = f3 = None
    if needs_worlds:
        f7 = World.load(cfg.world_path(asyncio.run(build("dev", "F7", ["100"], n_worlds=1, n_tasks=F7_TASKS, relational=True, embed=True))[0]))
        f3 = World.load(cfg.world_path(asyncio.run(build("dev", "F3", ["5", "60"], n_worlds=1, n_tasks=2, relational=True, embed=True))[0]))

    def run_gate_check(name: str, task_or_tasks, **kw):
        """One check's eval set through the FX-3 runner, in its own fresh log dir; the full logs."""
        _, headers = run_evals(task_or_tasks, log_root / name, profile=profile, model=agent, model_roles=roles, display="none", message_limit=40, **kw)
        logs = [read_eval_log(h.location) for h in headers]
        spend.logs += logs
        return logs

    def record(name: str, res: dict, t0: float, spent0: float, logs: Sequence = ()) -> None:
        res |= {"spend_usd": round(spend.total() - spent0, 6), "seconds": round(time.time() - t0, 1), "logs": [str(getattr(lg, "location", lg)) for lg in logs]}
        report["checks"][name] = res
        print(f"[{name}] {res['status'].upper()}{': ' + res['reason'] if res.get('reason') else ''}", flush=True)

    ctx = {
        "args": args, "profile": profile, "cfg": cfg, "f7": f7, "f3": f3, "report": report, "spend": spend,
        "run_gate_check": run_gate_check, "orchestrator_run": orchestrator_run, "build_model": build_model, "build_effort": build_effort,
        "provenance0": provenance0, "config0": config0,
    }
    status = 0
    try:
        for step in steps:
            spent = spend.total()
            sc.require_step_fits(step, proj[step], spent, args.max_usd)
            t0 = time.time()
            print(f"[{step}] {STEPS[step][1]} (projected ${proj[step]:.3f}; spent so far ${spent:.4f})", flush=True)
            res, logs = STEP_FUNCS[step](ctx)
            record(step, res, t0, spent, logs)
    except SystemExit as e:  # a spend stop: record it and write the report
        report["stopped"] = str(e)
        print(f"STOPPED: {e}", flush=True)
        status = 2
    report["spend_usd"] = round(spend.total(), 6)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    failed = [n for n, r in report["checks"].items() if r["status"] == sc.FAIL]
    report["status"] = "stopped" if report.get("stopped") else (sc.FAIL if failed else sc.WARN if any(r["status"] == sc.WARN for r in report["checks"].values()) else sc.PASS)
    write_report(report)
    print(json.dumps({"status": report["status"], "failed": failed, "spend_usd": report["spend_usd"], "report": str(OUT / "report.md")}, indent=1))
    return status or (1 if failed else 0)


# --- the checks ------------------------------------------------------------------------------------
# Each takes the run context and returns (result, logs).


def check_effort(c: dict) -> tuple[dict, list]:
    probe_path = ROOT / "cache" / "openai_probe.json"
    if c["args"].dry:
        synthetic = {r: {"model": "mockllm/model", "effort": {"verdict": sc.effort_verdict(
            {"accepted": True, "usage": {"completion_tokens_details": {"reasoning_tokens": 900}}},
            {"accepted": True, "usage": {"completion_tokens_details": {"reasoning_tokens": 120}}},
        )}} for r in sc.EFFORT_ROLES}
        res = sc.effort_from_probe(synthetic)
        res["measured"]["source"] = "dry: a synthetic probe record (the probe is live-only)"
        return res, []
    if not probe_path.is_file():
        return sc.result(sc.FAIL, {}, reason=f"{probe_path.relative_to(ROOT)} missing: run readiness/probe_openai.py --profile {c['profile'].name}"), []
    res = sc.effort_from_probe(json.loads(probe_path.read_text()))
    res["measured"]["source"] = str(probe_path.relative_to(ROOT))
    return res, []


def check_l2(c: dict) -> tuple[dict, list]:
    from ape.lgr.build import build_index

    world, cfg = c["f7"], c["cfg"]
    kind = "oracle" if c["args"].dry else "extract"
    manifest = asyncio.run(build_index(world, kind, cfg))
    names = bq.lightrag_entity_ids(bq.lightrag_entity_file(cfg, world.id, kind).read_text())
    from ape.worlds.render import chunk_world

    measured = {"entities": len(names), "policy_ids_as_entities": bq.lightrag_id_coverage(world, names), "index_hash": manifest["index_hash"][:12], "chunks": len(chunk_world(world)), "kind": kind}
    return sc.result(sc.PASS if names else sc.FAIL, measured, reason=None if names else "no entities extracted"), []


def check_d017(c: dict) -> tuple[dict, list]:
    """An early warning on one F7-100 world with the real builder, using `ape.build_quality` (the same code as
    run_gate's build-dev check). The decisive D-017 check is build-dev's, on the gate's four dev cells."""
    from ape.apg.author import author_world, build_llm_author
    from ape.llm.ledger import Ledger

    world, cfg = c["f7"], c["cfg"]
    if c["args"].dry:
        from ape.llm.fake import perfect_author as author
    else:
        author = build_llm_author(c["build_model"], Ledger(cfg.ledger_path), world.id, c["build_effort"])
    authoring = asyncio.run(author_world(world, cfg, author))
    row = bq.world_quality(world, cfg, "oracle" if c["args"].dry else "extract")  # L2 built the index
    agg = bq.aggregate([row], expected_cells=[row["cell"]])
    cell = agg["cells"].get(row["cell"], {})
    measured = {
        "authoring": authoring,
        "lightrag_id_coverage": row["lightrag"]["coverage"] if row["lightrag"] else None,
        "threshold": bq.THRESHOLD,
        "world": row,
        "verdict": agg["verdict"],
        "offline": bool(c["args"].dry),
        "note": bq.OFFLINE_NOTE if c["args"].dry else "one F7-100 world; the decisive check is run_gate build-dev's (build-dev/build_quality.json)",
    }
    reasons = cell.get("reasons", []) if agg["verdict"] != "builder_passes" else []
    # D-017: failing it means the Sol builder (needs approval), so it is a finding, not a broken harness.
    return sc.result(sc.WARN if reasons else sc.PASS, measured, reason="; ".join(reasons) + " (D-017: the Sol fallback builder needs approval)" if reasons else None), []


def _agent_run(c: dict, name: str, family: str, level: str, arm: str, n: int, **gate_kw) -> tuple[list, list[dict]]:
    from ape.tasks.gate import gate

    logs = c["run_gate_check"](name, gate(family=family, level=level, split="dev", arm=arm, **gate_kw), limit=n)
    rows = []
    for log in logs:
        for s in log.samples or []:
            recs = s.store.get("compile_log", [])
            rows.append({
                "arm": log.eval.task_args.get("arm"),
                "error": str(s.error.message)[:200] if s.error else None,
                "success": s.scores["task_success"].value if s.scores and "task_success" in s.scores else None,
                "compiles": len(recs),
                "ctx_tokens": [r["tokens"] for r in recs],
                "kg_calls": sum(1 for e in s.events if e.event == "model" and getattr(e, "role", None) == "kg"),
                "search_kb_calls": _search_kb_calls(s),
                "exposed_tools_per_step": [st["exposed_tools"] for st in s.store.get("step_log", [])],
            })
    return logs, rows


def _run_status(logs: list, rows: list[dict]) -> tuple[str, str | None]:
    bad = [f"{lg.eval.task_args.get('arm')}: {lg.status}" for lg in logs if lg.status != "success"] + [f"sample error: {r['error']}" for r in rows if r["error"]]
    return (sc.FAIL, "; ".join(bad)) if bad else (sc.PASS, None)


def check_h4(c: dict) -> tuple[dict, list]:
    logs, rows = _agent_run(c, "H4_S3s_F3", "F3", "60", "S3s", RUN_TASKS["H4"])
    status, reason = _run_status(logs, rows)
    exposed = [e for r in rows for e in r["exposed_tools_per_step"]]
    changed = len({tuple(sorted(e)) for e in exposed}) > 1
    if status == sc.PASS and not changed:
        status, reason = sc.WARN, "the exposed tool set never changed across steps, so H4 was not exercised"
    return sc.result(status, {"samples": len(rows), "tool_sets_changed": changed, "exposed_tools_per_step": exposed[:6], "success": [r["success"] for r in rows]}, reason=reason), logs


def check_l4_l5(c: dict) -> tuple[dict, list]:
    from ape.lgr.adapter import query_params

    arm = "LGRo-s" if c["args"].dry else "LGR-s"
    logs, rows = _agent_run(c, "L4_L5_LGR_F7", "F7", "100", arm, RUN_TASKS["L4_L5"])
    status, reason = _run_status(logs, rows)
    compiles = sum(r["compiles"] for r in rows)
    kg = sum(r["kg_calls"] for r in rows)
    l4 = sc.result(sc.PASS if status == sc.PASS and compiles and kg == compiles else sc.FAIL, {"compiles": compiles, "kg_calls": kg}, reason=reason or (None if kg == compiles else f"{kg} keyword calls for {compiles} compiles"))
    cap = int(query_params(c["cfg"].lgr_budget_tokens)["max_total_tokens"])
    l5 = sc.l5_verdict([t for r in rows for t in r["ctx_tokens"]], cap)
    worst = sc.FAIL if sc.FAIL in (l4["status"], l5["status"]) else sc.WARN if sc.WARN in (l4["status"], l5["status"]) else sc.PASS
    return sc.result(worst, {"L4": l4, "L5": l5}, reason="; ".join(f"{k}: {v['reason']}" for k, v in (("L4", l4), ("L5", l5)) if v.get("reason")) or None), logs


def check_apg(c: dict) -> tuple[dict, list]:
    logs, rows = _agent_run(c, "APG_F7", "F7", "100", "APG-s", RUN_TASKS["APG"])
    status, reason = _run_status(logs, rows)
    return sc.result(status, {"samples": len(rows), "compiles": sum(r["compiles"] for r in rows), "kg_calls": sum(r["kg_calls"] for r in rows), "ctx_tokens": [t for r in rows for t in r["ctx_tokens"]][:8], "success": [r["success"] for r in rows]}, reason=reason), logs


def check_pull(c: dict) -> tuple[dict, list]:
    from ape.tasks.gate import gate

    tasks = [gate(family="F7", level="100", split="dev", arm=arm, delivery="pull") for arm in ("S3s", "APG-s")]
    logs = c["run_gate_check"]("pull_F7", tasks, limit=RUN_TASKS["pull"])
    samples = [{"arm": lg.eval.task_args.get("arm"), "search_kb_calls": _search_kb_calls(s), "error": str(s.error.message)[:200] if s.error else None} for lg in logs for s in (lg.samples or [])]
    res = sc.pull_verdict(samples)
    if bad := [f"{lg.eval.task_args.get('arm')}: {lg.status}" for lg in logs if lg.status != "success"]:
        res = sc.result(sc.FAIL, res["measured"], reason="; ".join(filter(None, [res.get("reason"), *bad])))
    return res, logs


def check_recovery(c: dict) -> tuple[dict, list]:
    from smoke_fault import faulted_gate

    task = faulted_gate(family="F3", level="5", split="dev", arm="S3s")
    logs = c["run_gate_check"]("recovery_F3", task, limit=RUN_TASKS["recovery"], retry_wait=1)
    samples = [{"id": s.id, "error": str(s.error.message)[:200] if s.error else None, "retries": len(s.error_retries or []), "retry_errors": [str(e.message)[:120] for e in (s.error_retries or [])]} for lg in logs for s in (lg.samples or [])]
    return sc.recovery_verdict(logs[0].status if logs else "missing", samples), logs


def check_burst(c: dict) -> tuple[dict, list]:
    from ape.tasks.gate import gate

    logs = c["run_gate_check"]("burst_F7", gate(family="F7", level="100", split="dev", arm="S3s"), limit=BURST["tasks"], epochs=BURST["epochs"])
    samples = [s for lg in logs for s in (lg.samples or [])]
    probe = ROOT / "cache" / "openai_probe.json"
    headers = None
    if not c["args"].dry and probe.is_file():
        headers = ((json.loads(probe.read_text()).get("agent") or {}).get("plain") or {}).get("ratelimit")
    res = sc.burst_verdict(
        int(c["profile"].concurrency.max_connections or 16),
        [{"error": s.error.message if s.error else None, "total_time": s.total_time, "working_time": s.working_time} for s in samples],
        [e for s in samples for e in _model_events(s)],
        [m for s in samples for m in _logger_messages(s)],
        headers,
    )
    if bad := [lg.status for lg in logs if lg.status != "success"]:
        res = sc.result(sc.FAIL, res["measured"], reason="; ".join(filter(None, [res.get("reason"), f"log status {bad}"])))
    return res, logs


def check_retrieval(c: dict) -> tuple[dict, list]:
    return asyncio.run(_retrieval(c["f7"], c["cfg"], c["args"].dry)), []


async def _retrieval(world, cfg, dry: bool) -> dict:
    from apg_core import load_graph, route

    from ape.apg.arm import ensure_graph, graph_path, tuned
    from ape.config import embedding_cache
    from ape.kb.baselines import FlatHybrid
    from ape.kb.provenance import ChunkIndex, evidence_pr
    from ape.llm.embeddings import CachedEmbeddingsConnector
    from ape.worlds.render import chunk_world

    class NoClassify:  # routing's embedding shortlist only: the LLM classify step is never reached
        def classify(self, query, outline, schema, multi):
            return []

    emb = embedding_cache(cfg)
    chunks = chunk_world(world)
    s3s = FlatHybrid(chunks, emb, cfg.s3s_budget_tokens)
    kind = "authored" if graph_path(cfg, world.id, "authored").is_file() else "oracle"
    graph = load_graph(tuned(await ensure_graph(world, kind, cfg, emb)))
    index = ChunkIndex(chunks)
    node_facts = {}
    for n in graph.dfs():
        props = n.get("props") or {}
        node_facts[n["id"]] = set(props.get("factIds") or index.facts(props.get("sourceChunkIds") or []))
    recalls, hits = [], []
    for t in world.tasks:
        recalls.append(evidence_pr((await s3s.compile(t.prompt, t)).fact_ids, t.gold_fact_ids)[1])
        await emb.embed([t.prompt])
        routed = route(t.prompt, graph, {"embeddings": CachedEmbeddingsConnector(emb), "llm": NoClassify()})
        gold = {nid for nid, facts in node_facts.items() if facts & set(t.gold_fact_ids)}
        hits.append(bool(gold & set(routed["shortlist"])))
    res = sc.retrieval_verdict(recalls, hits, graph=kind, dry=dry)
    res["measured"]["s3s_budget_tokens"] = cfg.s3s_budget_tokens
    return res


def check_orchestrator(c: dict) -> tuple[dict, list]:
    from ape import run_gate as rg

    spend: Spend = c["spend"]
    left = c["args"].max_usd - spend.total()
    probe_run = c["orchestrator_run"]()
    with rg.run_environment(probe_run):
        before = rg.spend(probe_run)["spent_usd"] if probe_run.dir.is_dir() else 0.0
    run = c["orchestrator_run"](budget_usd=before + left)  # run_gate's own guard gets what the cap leaves
    statuses: dict[str, str | None] = {}
    error = None
    for phase in sc.SMOKE_PHASES:
        try:
            statuses[phase] = rg.run_phases(run, phase)[phase]
        except Exception as e:  # noqa: BLE001  (recorded in the phase manifest and in the check)
            statuses[phase] = (rg.read_manifest(run, phase) or {}).get("status") or "failed"
            error = f"{phase}: {type(e).__name__}: {str(e)[:400]}"
            break
    freeze_error = None
    if error is None:
        try:
            rg.run_phases(run, "freeze")
        except rg.PhaseError as e:
            freeze_error = str(e)
    with rg.run_environment(run):
        after = rg.spend(run)["spent_usd"]
    spend.orchestrator += after - before
    text = rg.ROOT.joinpath("GATE_PREREG.md").read_text()
    files = {
        "tune/tuning_log.jsonl": (run.phase_dir("tune") / "tuning_log.jsonl").is_file(),
        "selected.yaml": run.selected_path.is_file(),
        "anchor/pc1.json": (run.phase_dir("anchor") / "pc1.json").is_file(),
    }
    res = sc.orchestrator_verdict(
        statuses, files, freeze_error, placeholders=len(rg.prereg_placeholders(text)),
        provenance_unchanged=_sha(ROOT / "PROVENANCE.md") == c["provenance0"], config_unchanged=_tree_hash(ROOT / "config") == c["config0"],
    )
    if error:
        res["reason"] = "; ".join(filter(None, [error, res.get("reason")]))
    warnings = {p: (rg.read_manifest(run, p) or {}).get("warnings") for p in sc.SMOKE_PHASES}
    pc1 = run.phase_dir("anchor") / "pc1.json"
    res["measured"] |= {
        "run_dir": str(run.dir),
        "warnings": {p: w for p, w in warnings.items() if w},
        "pc1_pass": json.loads(pc1.read_text()).get("pass") if pc1.is_file() else None,
        "pc1_note": "2 questions per type: wiring only, no reproduction claim",
        "run_spend_usd": round(after - before, 6),
    }
    return res, []


STEP_FUNCS: dict[str, Callable[[dict], tuple[dict, list]]] = {
    "effort": check_effort,
    "L2": check_l2,
    "D017": check_d017,
    "H4": check_h4,
    "L4_L5": check_l4_l5,
    "APG": check_apg,
    "pull": check_pull,
    "recovery": check_recovery,
    "burst": check_burst,
    "retrieval": check_retrieval,
    "orchestrator": check_orchestrator,
}
assert tuple(STEP_FUNCS) == tuple(STEPS)


# --- report ----------------------------------------------------------------------------------------


def _brief(measured: dict) -> str:
    """A one-line digest of a check's measured values for report.md (nested checks by status, pull by rate)."""
    parts = []
    for k, v in measured.items():
        if isinstance(v, dict) and "status" in v:
            parts.append(f"{k} {v['status']}" + (f" ({_brief(v.get('measured') or {})})" if v.get("measured") else ""))
        elif k == "arms" and isinstance(v, dict):
            parts += [f"{arm} search_kb {a['rate']:.0%}" for arm, a in v.items() if isinstance(a, dict) and "rate" in a]
        elif isinstance(v, dict | list) or v is None:
            continue
        else:
            parts.append(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {' '.join(str(v).split())}")
    return ", ".join(parts)[:300].replace("|", "/")


def write_report(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    lines = [
        f"# Smoke report ({'DRY: mock models, fake embeddings' if report['dry'] else 'live'})",
        "",
        f"- Status: **{report['status']}**{' (' + report['stopped'] + ')' if report.get('stopped') else ''}",
        f"- Profile: {report['models']['profile']}; spend ${report.get('spend_usd', 0):.4f} of --max-usd {report['max_usd']} (projected ${report.get('projection_usd', {}).get('total', 0):.3f})",
        f"- {report['started']} to {report.get('finished')}",
        "",
        "| Check | Status | Measured | Spend | Projected | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["checks"].items():
        reason = " ".join((r.get("reason") or "").split()).replace("|", "/")[:300]
        lines.append(f"| {name} | {r['status']} | {_brief(r.get('measured') or {})} | ${r.get('spend_usd', 0):.4f} | ${report['projection_usd'].get(name, 0):.3f} | {reason} |")
    if burst := report["checks"].get("burst"):
        m = burst["measured"]
        lines += ["", f"Recommended max_connections: **{m.get('recommended_max_connections')}** ({m.get('recommendation')}).", f"Model-call time (s): {m.get('model_call_time_s')}; sample time (s): {m.get('sample_total_time_s')}."]
    lines += ["", "Logs and full values: `cache/smoke/report.json`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
