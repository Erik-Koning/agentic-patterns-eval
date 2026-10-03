"""Live smoke tests (readiness E3-E5, L2-L5, H4, H6 live, D-017; FIX_PLAN FX-8).

    uv run --locked python readiness/smoke.py                      # every check, live, on config/models.yaml's gate profile
    uv run --locked python readiness/smoke.py --dry                # offline wiring check (mock models, fake embeddings), $0
    uv run --locked python readiness/smoke.py --only pull,burst    # some checks (with the checks they need)
    uv run --locked python readiness/smoke.py --skip orchestrator  # all but some
    uv run --locked python readiness/smoke.py --profile gate --agent openai/gpt-6-sol   # a flag swaps a role's model, keeps its effort

Checks, in run order (`STEPS`; pass criteria in `smoke_checks.py`):

    effort        per role, on the call path the run uses (Inspect's Responses API for agent/kg/judge, BuildLlm
                  for build): the role's config and strict JSON are accepted, and effort is honoured (high uses
                  more reasoning tokens than low). Read from cache/openai_probe.json (run
                  readiness/probe_openai.py first); --dry runs the probe's own code on mock models
    L2            LightRAG extraction on an F7-100 world; source ids map to our chunks
    D017          the build model's authoring ID coverage and LightRAG's ID coverage >= 0.95 on one F7-100 world
                  (`ape.build_quality`, needs L2): an early warning; the decisive D-017 check is run_gate
                  build-dev's, on the gate's four dev cells (build-dev/build_quality.json)
    H4            S3s on F3-60: OpenAI accepts tool sets that change across turns (warn if they never changed)
    L4_L5         LGR-s on F7-100: one keyword call per compile (L4); realized context <= max_total_tokens with
                  the median in the expected band (L5) (needs L2)
    APG           APG-s on F7-100 with the authored graph (needs D017)
    perstep_reasoning
                  S3s, a per-step arm, at the profile's effort on 2 F7-10 tasks (RELIABILITY_REVIEW L2, live): every
                  call after a tool step carries the step's knowledge on the last tool result, with no user message
                  after the model's turn, so OpenAI keeps the earlier reasoning (judged). The reasoning items that
                  reach each call (Inspect's own Responses-API conversion of the logged input) and the reasoning
                  tokens per call are reported, not judged
    pull          pull delivery: S3s and APG-s on F7-100, 4 tasks; search_kb in >= 75% of samples, no errors
                  (needs D017)
    recovery      a harness-side fault on a sample's first attempt (smoke-only solver wrapper) is retried by
                  the FX-3 runner and recorded
    burst         24 samples at the profile's max_connections: failed samples, retries, rate-limit signals,
                  latency distribution, and a recommended max_connections
    f8_session    one F8 session of N=5 cases per session arm (CM0, O-state) through the real Study G task
                  (RELIABILITY_REVIEW L12): no sample error, a per-item record with view tokens for every case,
                  the report submitted (or its nudge fired), a forked probe at k=5 with schema-valid JSON that
                  never enters the history; reasoning-only assistant turns are counted and must not end a session
    retrieval     no LLM: S3s evidence recall at its budget and APG's embedding-shortlist gold rate on F7-100
                  (warn below 0.8; the authored graph when D017 built it, else the oracle graph), plus LightRAG's
                  context recall with the query as its keywords when L2 built the index (reported only). Facts
                  are credited by the arms' delivered-text rule (`ape.kb.provenance`), never by source chunks
    extract_f7_1000
                  the real builder on ONE F7-1000 dev world (the gate's hard cell) through `ape.artifacts`: APG
                  authoring and LightRAG extraction, then D-017's coverage (`ape.build_quality`), build health
                  (lost chunks), APG retries/repairs (authoring calls beyond one per chunk), wall-clock, and the
                  ledger's calls and $ per system. Warns, never fails, on low coverage (the Sol fallback needs
                  approval); the decisive D-017 check is run_gate build-dev's
    orchestrator  the real gate orchestrator at SMOKE_SCALE (`python -m ape.run_gate --smoke`): preflight,
                  build-dev, tune, anchor and pilot on F7-10 and F3-5 (1 world, 2 tasks, 2 candidates per
                  system, anchor 2 questions per type), then the freeze must refuse (a smoke run never freezes,
                  and the draft pre-registration's open items are listed)

Cost. Before any spend the script prints each selected check's projected cost (conservative, from
`ape.budget`'s priors and config/model_costs.yaml) and refuses to start if the total exceeds --max-usd
(default $4). Before each check it stops if spend so far plus that check's projection would exceed the cap;
the orchestrator also runs under run_gate's own budget guard with exactly what the cap leaves (its budget is the
program's spend so far, this invocation's earlier checks included, plus that remainder). The gate profile's
projection (GPT-6 Luna, high; 2026-10-03 priors): L2 $0.10, D017 $0.04, H4 $0.01, L4_L5 $0.01, APG $0.01,
perstep_reasoning $0.01, pull $0.05, recovery $0.01, burst $0.10, f8_session $0.12 (two N=5 sessions priced as
Study G sessions: full-history views at the window's share, so conservative), extract_f7_1000 $1.34 (one F7-1000
world, 547 chunks, both systems), orchestrator $1.15 (build-dev $0.08, tune $0.13, anchor $0.58 with gpt-4o-mini,
pilot $0.37; $1.01 with the Luna anchor fallback), total ≈ $2.95 (≈ $2.81 with the Luna fallback) under the $4
default cap: the $1 of headroom covers checks that spend somewhat over their projection. The effort check reads the
probe, which costs ≈ $0.02 on its own. `--dry` spends $0. Spend is this invocation's: Inspect-metered calls, plus
the growth of the build/embedding ledger and of the orchestrator run. A live smoke's log dirs and ledger also
register in the program spend registry (`ape.spend`, label `smoke`), so they count toward the program's
$5,000 in run_gate's guard and in `python -m ape.budget spend`.

Robustness. A check that raises is recorded as a FAIL with the exception, and the next check runs; a spend stop or
an interrupt (Ctrl-C) ends the run, and in every case report.json, report.md and (live) checks.json are written, so
checks already paid for are never lost.

The orchestrator check runs run_gate's gate profile (run_plan.yaml), not the --agent/--kg/--build overrides.

Isolation. Dry and live smokes never share state: a dry run uses cache/smoke/dry/, a live run cache/smoke/live/
(`smoke_dir`), each with its own worlds, indices, cache, logs (a fresh logs/<timestamp>/ per invocation), the
orchestrator's runs (a fresh runs/smoke-<mode>-<timestamp> per invocation, `orchestrator_run_id`: a re-smoke after
a code change really re-runs build-dev through pilot instead of skipping them as done) and report.json / report.md
(which record the mode). A live run never reads or resumes anything a dry run made; it warns, and never deletes, if it
finds the older flat layout directly under cache/smoke/. Nothing is written to config/, PROVENANCE.md or runs/.
Exits non-zero when a check fails.

Live record and the gate. A live run also updates cache/smoke/live/checks.json: every check's latest live result
(status, finish time, commit and its uncommitted code changes `code_dirty`, profile, model overrides, the probe's
snapshots), so a check re-run alone with --only replaces its own entry. run_gate's live preflight
(`ape.run_gate.check_live_smoke`) refuses a live gate run unless the latest live report passed (warn allowed) on the
gate profile and every check here has a passing result no older than 7 days, of committed code that is exactly the
code that would run (src/, power/, uv.lock), on PROVENANCE.md's pinned snapshots; or the run passes
--skip-smoke-check "<reason>". A run settles this once, at its first preflight (run.json).

Cost model check. After a live run, its own Inspect logs go through `ape.budget.calibrate` into
cache/smoke/live/calibration_measured.yaml (never config/), and report.md's "Cost model check" compares the
measured output tokens per call (reasoning included) per role and effort, calls per sample per cell, and build
calls per system on the gate's worlds (the PC1 anchor's index, another model on another corpus, is left out) with
the priors, then re-projects the whole program both ways (measured cells, and the priors
adjusted by the measured per-call figures). It warns when either exceeds $5,000 or puts the gate over its $700
allocation, and prints the command that promotes the measurements into config/budget_calibration_measured.yaml.
A dry run shows the section as "not measured". Live runs also print storage warnings (APE_BACKUP_DIR, free disk).
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_checks as sc  # noqa: E402

from ape import build_quality as bq  # noqa: E402  (D-017 coverage: shared with run_gate's build-dev check)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "cache" / "smoke"  # dry state in dry/, live state in live/ (`smoke_dir`); never shared
RECORD_NAME = "checks.json"  # live only: every check's latest live result, read by run_gate's preflight
# The flat layout before dry and live were separated (2026-10-02): a live run ignores it and says so.
LEGACY_ENTRIES = ("worlds", "indices", "cache", "logs", "runs", "report.json", "report.md")
PROBE_PATH = ROOT / "cache" / "openai_probe.json"  # readiness/probe_openai.py's report (effort, burst's TPM header)
SPEC_THRESHOLD = bq.THRESHOLD  # D-017
F7_TASKS = 20  # the F7-100 world's tasks: enough for the retrieval rates; runs take the first n (Inspect `limit`)
RUN_TASKS = {"H4": 2, "L4_L5": 2, "APG": 2, "perstep_reasoning": 2, "pull": 4, "recovery": 1}
BURST = {"tasks": 12, "epochs": 2}  # 24 samples
F8 = {"N": 5, "arms": ("CM0", "O-state"), "message_limit": 400}  # one short session per arm; the probe at k = N
F7_1000_TASKS = 2  # the extraction check builds one F7-1000 dev world; its tasks are not run
DEFAULT_MAX_USD = 4.0  # the full live smoke projects ≈ $2.9; $3 left no headroom for checks that run near their projection
# name -> (checks it needs, what it checks), in run order: the costly extraction runs late, the orchestrator last.
STEPS: dict[str, tuple[tuple[str, ...], str]] = {
    "effort": ((), "per role and call path: config accepted, effort honoured (from the probe)"),
    "L2": ((), "LightRAG extraction and source mapping on F7-100"),
    "D017": (("L2",), "build-quality check: authoring and LightRAG ID coverage >= 0.95"),
    "H4": ((), "S3s on F3-60: changing tool sets accepted"),
    "L4_L5": (("L2",), "LGR-s on F7-100: one keyword call per compile; realized context vs max_total_tokens"),
    "APG": (("D017",), "APG-s on F7-100"),
    "perstep_reasoning": ((), "S3s (per step) on F7-10: knowledge rides on the last tool result; reasoning carried"),
    "pull": (("D017",), "pull mode: S3s and APG-s on F7-100"),
    "recovery": ((), "a first-attempt harness fault is retried and recorded"),
    "burst": ((), "24 samples at the profile's max_connections"),
    "f8_session": ((), "one F8 session (N=5) per arm, CM0 and O-state, with a forked probe"),
    "retrieval": ((), "real-embedding retrieval, no LLM"),
    "extract_f7_1000": ((), "the real builder on one F7-1000 world: APG authoring and LightRAG extraction"),
    "orchestrator": ((), "run_gate preflight..pilot at SMOKE_SCALE; freeze refuses"),
}
# The checks that run on the shared F7-100 / F3 worlds (built once, up front); the others build their own.
SHARED_WORLD_STEPS = {"L2", "D017", "H4", "L4_L5", "APG", "pull", "recovery", "burst", "retrieval"}


def smoke_dir(dry: bool) -> Path:
    """Where a dry or a live smoke keeps everything: worlds, indices, cache, logs, the orchestrator's runs and the
    report. Separate, so a live smoke never reads or resumes anything a dry run (fake embeddings, mock models) made."""
    return SMOKE_ROOT / ("dry" if dry else "live")


def orchestrator_run_id(dry: bool) -> str:
    """A fresh run id per invocation for the orchestrator check: a re-smoke (after a code change, or a week later)
    really re-runs build-dev through pilot, where a fixed id would skip its finished phases and pass on old work."""
    from datetime import UTC, datetime

    return f"smoke-{'dry' if dry else 'live'}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}"


def legacy_layout() -> list[str]:
    """Entries of the pre-split flat layout directly under cache/smoke/ (left by older dry runs)."""
    return [name for name in LEGACY_ENTRIES if (SMOKE_ROOT / name).exists()]


def _isolate(dry: bool) -> None:
    out = smoke_dir(dry)
    os.environ["APE_WORLDS"] = str(out / "worlds")
    os.environ["APE_INDICES"] = str(out / "indices")
    os.environ["APE_CACHE"] = str(out / "cache")
    # Smoke spend is the program's spend too: live log dirs and ledgers register in the program spend registry
    # (ape.spend) under this label, so run_gate's guard and `python -m ape.budget spend` count them. The
    # --max-usd cap stays this invocation's own. --dry (mock agent) registers nothing in the real registry.
    os.environ["APE_SPEND_LABEL"] = "smoke"
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
        "perstep_reasoning": [agent("perstep_reasoning", ["S3s"], "F7-10", RUN_TASKS["perstep_reasoning"])],
        "pull": [agent("pull", ["S3s", "APG-s"], "F7-100", RUN_TASKS["pull"], deliveries=["pull"])],
        "recovery": [agent("recovery", ["S3s"], "F3-5", RUN_TASKS["recovery"])],
        "burst": [agent("burst", ["S3s"], "F7-100", BURST["tasks"], epochs=BURST["epochs"])],
        # Study G's session pricing: full-history views at the window's share, probes at the agent's prices.
        "f8_session": [_cell("f8_session", profile, kind="session", arms=list(F8["arms"]), N=F8["N"], sessions=1, epochs=1)],
        "extract_f7_1000": [_cell("extract_f7_1000", profile, kind="build", systems=["apg", "lightrag"], worlds={"F7-1000": 1})],
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
    out = smoke_dir(args.dry)
    t_start = time.time()
    report: dict = {
        "dry": args.dry,
        "mode": "dry" if args.dry else "live",
        "dir": str(out),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "started_utc": _utc_now(),
        # code_dirty: uncommitted changes under src/, power/ and uv.lock (`run_gate.code_drift` of HEAD); run_gate's
        # live preflight refuses a smoke of uncommitted code, since what it smoked is no commit.
        "git": rg.git_state() | {"code_dirty": rg.code_drift("HEAD")},
        "models": {"profile": profile.name, "roles": profile.summary(), "overrides": {k: v for k, v in vars(args).items() if k in ("agent", "kg", "build", "embed")}, "build": [build_model, build_effort]},
        "snapshots": _probe_snapshots() if not args.dry else {},
        "max_usd": args.max_usd,
        "steps": steps,
        "checks": {},
        "warnings": [],
    }
    if not args.dry:
        from ape.models import storage_warnings

        if legacy := legacy_layout():
            report["warnings"].append(
                f"{SMOKE_ROOT} still holds the old flat smoke layout ({', '.join(legacy)}), from dry runs before dry and live "
                f"were separated; the live smoke ignores it and never deletes it: remove it yourself when convenient"
            )
        report["warnings"] += storage_warnings()
        for w in report["warnings"]:
            print(f"WARNING: {w}", flush=True)

    run_id = orchestrator_run_id(args.dry)  # one per invocation: never resumes (and so never skips) an earlier smoke's run

    def orchestrator_run(budget_usd: float | None = None):
        return rg.GateRun(run_id, offline=args.dry, smoke=True, runs_root=out / "runs", budget_usd=budget_usd)

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
    log_root = out / "logs" / time.strftime("%Y%m%dT%H%M%S")
    provenance0, config0 = _sha(ROOT / "PROVENANCE.md"), _tree_hash(ROOT / "config")

    # 3. Worlds: F7-100 (relational, descriptive) with F7_TASKS tasks; F3-60 (H4) and F3-5 (recovery) with 2 each.
    #    Chunk embeddings are cached here, so the runs only read them.
    f7 = f3 = None
    if set(steps) & SHARED_WORLD_STEPS:
        f7 = World.load(cfg.world_path(asyncio.run(build("dev", "F7", ["100"], n_worlds=1, n_tasks=F7_TASKS, relational=True, embed=True))[0]))
        f3 = World.load(cfg.world_path(asyncio.run(build("dev", "F3", ["5", "60"], n_worlds=1, n_tasks=2, relational=True, embed=True))[0]))

    def run_gate_check(name: str, task_or_tasks, *, model=None, message_limit: int = 40, **kw):
        """One check's eval set through the FX-3 runner, in its own fresh log dir; the full logs. `model` swaps the
        agent (the F8 session's dry run needs the session mock); `message_limit` caps a sample's messages."""
        _, headers = run_evals(task_or_tasks, log_root / name, profile=profile, model=model or agent, model_roles=roles, display="none", message_limit=message_limit, **kw)
        logs = [read_eval_log(h.location) for h in headers]
        spend.logs += logs
        return logs

    def record(name: str, res: dict, t0: float, spent0: float, logs: Sequence = ()) -> None:
        res |= {"spend_usd": round(spend.total() - spent0, 6), "seconds": round(time.time() - t0, 1), "finished_utc": _utc_now(), "logs": [str(getattr(lg, "location", lg)) for lg in logs]}
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
            try:
                res, logs = STEP_FUNCS[step](ctx)
            except Exception as e:  # noqa: BLE001  (a check that raises fails; the checks already paid for stay recorded)
                traceback.print_exc()
                res, logs = sc.result(sc.FAIL, {}, reason=f"the check raised {type(e).__name__}: {' '.join(str(e).split())[:400]}"), []
            record(step, res, t0, spent, logs)
    except SystemExit as e:  # a spend stop: record it and write the report
        report["stopped"] = str(e)
        print(f"STOPPED: {e}", flush=True)
        status = 2
    except BaseException as e:  # e.g. KeyboardInterrupt: the report and the live record are still written, then it re-raises
        report["stopped"] = f"interrupted: {type(e).__name__}"
        raise
    finally:
        failed = _finish(report, args, out, spend, log_root, orchestrator_run, t_start, profile)
    print(json.dumps({"status": report["status"], "failed": failed, "spend_usd": report["spend_usd"], "report": str(out / "report.md")}, indent=1))
    return status or (1 if failed else 0)


def _finish(report: dict, args, out: Path, spend: "Spend", log_root: Path, orchestrator_run: Callable, t_start: float, profile) -> list[str]:
    """Close the report whatever happened (a spend stop, a check that raised, an interrupt): spend, status, the
    cost-model check and, live, the record of every check's latest result that run_gate's preflight reads, then
    report.json and report.md. Returns the failed checks."""
    try:
        report["spend_usd"] = round(spend.total(), 6)
    except BaseException as e:  # noqa: BLE001  (an unpriced call: the report still gets written, with the reason)
        report["spend_usd"] = None
        report.setdefault("stopped", f"spend could not be priced: {e}")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["finished_utc"] = _utc_now()
    failed = [n for n, r in report["checks"].items() if r["status"] == sc.FAIL]
    report["status"] = "stopped" if report.get("stopped") else (sc.FAIL if failed else sc.WARN if any(r["status"] == sc.WARN for r in report["checks"].values()) else sc.PASS)
    # Live: the cost model against what the smoke measured (indicative; never written to config/), and the record of
    # every check's latest live result that run_gate's preflight reads.
    if args.dry:
        report["cost_model_check"] = {"status": "not measured", "reason": "dry run: mock models report no real usage"}
    else:
        orchestrated = "orchestrator" in report["checks"]
        try:
            report["cost_model_check"] = cost_model_check(out, log_root, orchestrator_run() if orchestrated else None, t_start, profile)
        except Exception as e:  # noqa: BLE001  (the check is informational: a failure here must not lose the report)
            report["cost_model_check"] = {"status": sc.WARN, "reason": f"cost-model check failed: {type(e).__name__}: {e}"}
        out.mkdir(parents=True, exist_ok=True)
        record_path = out / RECORD_NAME
        previous = json.loads(record_path.read_text()) if record_path.is_file() else None
        record_path.write_text(json.dumps(sc.update_live_record(previous, report, list(STEPS)), indent=1, default=str))
    write_report(report)
    return failed


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _probe_snapshots() -> dict[str, str | None]:
    """alias -> the snapshot the readiness probe saw serve it (cache/openai_probe.json); {} without a probe."""
    if not PROBE_PATH.is_file():
        return {}
    from ape.snapshots import probe_snapshots

    try:
        return probe_snapshots(json.loads(PROBE_PATH.read_text()))
    except (ValueError, KeyError, TypeError):
        return {}


def _reasoning_by_role(log_files: Sequence[Path]) -> dict[str, tuple[float, float]]:
    """role -> (reasoning tokens, output tokens) over the logs' samples: `role_usage` per role, and `model_usage`
    minus the roles for the agent."""
    from inspect_ai.log import read_eval_log

    acc: dict[str, list[float]] = {}
    for f in log_files:
        for s in read_eval_log(str(f)).samples or []:
            roles = dict(s.role_usage or {})
            for role, u in roles.items():
                a = acc.setdefault(role, [0.0, 0.0])
                a[0] += float(u.reasoning_tokens or 0)
                a[1] += float(u.output_tokens or 0)
            tot = [sum(float(getattr(u, k) or 0) for u in (s.model_usage or {}).values()) for k in ("reasoning_tokens", "output_tokens")]
            a = acc.setdefault("agent", [0.0, 0.0])
            a[0] += tot[0] - sum(float(u.reasoning_tokens or 0) for u in roles.values())
            a[1] += tot[1] - sum(float(u.output_tokens or 0) for u in roles.values())
    return {k: (v[0], v[1]) for k, v in acc.items()}


def _ledger_rows(root: Path, since: float) -> list[dict]:
    """Every build/embedding ledger entry under `root` (the smoke's cache and its orchestrator run's work dir) written
    from `since` on, as dicts."""
    from dataclasses import asdict

    from ape.llm.ledger import Ledger

    rows = []
    for path in sorted(root.rglob("ledger.jsonl")):
        rows += [asdict(e) for e in Ledger(path).read() if e.ts >= since]
    return rows


def cost_model_check(out: Path, log_root: Path, run, since: float, profile) -> dict:
    """What a live smoke measured against the cost model's priors, and the program re-projected with it.

    - Inspect logs (this invocation's checks, plus the orchestrator run's when it ran) go through
      `ape.budget.calibrate` into `<out>/calibration_measured.yaml`; config/ is never written. The exact command to
      promote them into config/budget_calibration_measured.yaml is returned (and printed in report.md).
    - Output tokens per call (reasoning included) per role and effort, and calls per sample per measured cell,
      against the priors; build calls per system from the ledgers.
    - Two re-projections of the whole plan, conservative, against the current one: `measured_cells` (the
      measured entries applied where (arm, model, effort, cell, delivery) match a plan cell) and
      `adjusted_priors` (the priors with the measured per-call figures in place everywhere). WARN when either
      exceeds the program budget or puts the gate over its allocation.
    """
    from ape.budget import calibrate, estimate, load_assumptions, load_measured, load_plan

    logs = sorted(log_root.rglob("*.eval")) if log_root.is_dir() else []
    if run is not None and run.dir.is_dir():
        logs += sorted(run.dir.rglob("*.eval"))
    out.mkdir(parents=True, exist_ok=True)
    measured_path = out / "calibration_measured.yaml"
    entries = calibrate(logs, measured_path, merge=False)["entries"] if logs else []
    (out / "calibration_logs.txt").write_text("".join(f"{p}\n" for p in logs))
    A, plan = load_assumptions(), load_plan()
    role_efforts = {r: s.reasoning_effort for r, s in profile.roles.items()}
    outputs = sc.output_vs_prior(sc.measured_per_call(entries, role_efforts), A, _reasoning_by_role(logs) if logs else None)
    builds = sc.build_per_call(_ledger_rows(out, since), A)
    config_measured = load_measured()

    def totals(est) -> dict[str, float]:
        return {"total": round(est.total("conservative"), 2), "gate": round(est.total("conservative", study="gate"), 2)}

    projections = {
        "baseline": totals(estimate(plan, A, measured=config_measured)),
        "measured_cells": totals(estimate(plan, A, measured=[*config_measured, *entries])),
        "adjusted_priors": totals(estimate(plan, sc.adjusted_assumptions(A, outputs, builds), measured=config_measured)),
    }
    verdict = sc.cost_verdict(projections, float(plan.budget["total_usd"]), float((plan.budget.get("allocations") or {}).get("gate", float("inf"))))
    rel = lambda p: os.path.relpath(p, ROOT)  # noqa: E731
    return {
        "status": verdict["status"],
        "reasons": verdict["reasons"],
        "logs": len(logs),
        "entries": len(entries),
        "measured_file": str(measured_path),
        "output_per_call": outputs,
        "calls_per_sample": sc.calls_vs_prior(entries, A),
        "builds": builds,
        "projections": projections,
        "note": "indicative: a smoke makes few calls per cell; the gate pilot's calibration is the one to adopt",
        "promote_command": f"uv run python -m ape.budget calibrate $(cat {rel(out / 'calibration_logs.txt')}) --out config/budget_calibration_measured.yaml",
        "promote_note": "merges with the entries already there; build priors (budget_assumptions.yaml build_calls) are edited by hand",
    }


# --- the checks ------------------------------------------------------------------------------------
# Each takes the run context and returns (result, logs).


def _probe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("readiness_probe_openai", Path(__file__).resolve().parent / "probe_openai.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_effort(c: dict) -> tuple[dict, list]:
    if c["args"].dry:  # the probe's own code on both call paths, with mock models: no network, no ledger
        from types import SimpleNamespace

        probe = _probe_module()
        report = asyncio.run(probe.run_probe(c["profile"], factory=probe.offline_factory, build_client=probe.OfflineBuildClient(), ledger=SimpleNamespace(append=lambda entry: None)))
        res = sc.effort_from_probe(report)
        res["measured"]["source"] = "dry: readiness/probe_openai.py on mock models (the probe itself is live-only)"
        return res, []
    if not PROBE_PATH.is_file():
        return sc.result(sc.FAIL, {}, reason=f"{PROBE_PATH} missing: run readiness/probe_openai.py --profile {c['profile'].name}"), []
    res = sc.effort_from_probe(json.loads(PROBE_PATH.read_text()))
    res["measured"]["source"] = str(PROBE_PATH)
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


def _resolved(logs: list) -> list:
    """The logs re-read with attachments resolved: Inspect stores long message texts in event inputs as
    `attachment://` references, which the structural checks must read in full."""
    from inspect_ai.log import read_eval_log

    return [read_eval_log(lg.location, resolve_attachments=True) for lg in logs]


def _perstep_calls(sample, kb_header: str) -> list[dict]:
    """Per agent call of a per-step sample: did the step's knowledge ride on the last tool result with no user
    message after the model's last turn (RELIABILITY_REVIEW L2), and what reasoning reached the call.

    A call "follows a tool step" when its input, without a trailing knowledge-only user message (the placement
    the fix replaced), ends with a tool result; a call after a nudge or at turn 0 does not. Carried reasoning is
    counted on Inspect's own Responses-API conversion of the logged input: reasoning items after the last user
    message are the ones OpenAI keeps."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentReasoning
    from inspect_ai.model._openai_responses import openai_responses_inputs

    out = []
    agent_calls = [e for e in sample.events if e.event == "model" and getattr(e, "role", None) in (None, "agent")]
    for turn, e in enumerate(agent_calls):
        msgs = list(e.input or [])
        base = msgs[:-1] if msgs and isinstance(msgs[-1], ChatMessageUser) and (msgs[-1].text or "").startswith(kb_header) else msgs
        last_assistant = max((i for i, m in enumerate(msgs) if isinstance(m, ChatMessageAssistant)), default=None)
        rec = {
            "sample": sample.id,
            "turn": turn,
            "after_tool_step": bool(base) and isinstance(base[-1], ChatMessageTool),
            "kb_in_last_tool": bool(msgs) and isinstance(msgs[-1], ChatMessageTool) and kb_header in (msgs[-1].text or ""),
            "user_after_assistant": last_assistant is not None and any(isinstance(m, ChatMessageUser) for m in msgs[last_assistant + 1 :]),
            "reasoning_in_input": sum(1 for m in msgs if isinstance(m, ChatMessageAssistant) and isinstance(m.content, list) for part in m.content if isinstance(part, ContentReasoning)),
            "reasoning_items": None,
            "reasoning_after_last_user": None,
            "reasoning_tokens": getattr(getattr(getattr(e, "output", None), "usage", None), "reasoning_tokens", None),
            "error": e.error,
        }
        try:
            items = asyncio.run(openai_responses_inputs(msgs))
            last_user = max((i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "user"), default=-1)
            rec["reasoning_items"] = sum(1 for it in items if it.get("type") == "reasoning")
            rec["reasoning_after_last_user"] = sum(1 for i, it in enumerate(items) if it.get("type") == "reasoning" and i > last_user)
        except Exception as ex:  # noqa: BLE001  (the conversion is Inspect-internal: report what it says, never fail on it)
            rec["conversion_error"] = f"{type(ex).__name__}: {ex}"[:160]
        out.append(rec)
    return out


def check_perstep_reasoning(c: dict) -> tuple[dict, list]:
    """S3s (a per-step arm) at the profile's effort on 2 F7-10 tasks: every call after a tool step carries the step's
    knowledge on the last tool result with no user message after the model's turn, so OpenAI keeps the earlier
    reasoning (the structure is judged); the reasoning items and tokens that reach each call are reported."""
    from ape.agent.kb_react import STEP_KB_HEADER
    from ape.build import build
    from ape.tasks.gate import gate

    n = RUN_TASKS["perstep_reasoning"]
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=n, relational=True, embed=True))
    logs = _resolved(c["run_gate_check"]("perstep_S3s_F7_10", gate(family="F7", level="10", split="dev", arm="S3s"), limit=n))
    res = sc.perstep_verdict([rec for lg in logs for s in (lg.samples or []) for rec in _perstep_calls(s, STEP_KB_HEADER)])
    if bad := [f"{lg.eval.task_args.get('arm')}: {lg.status}" for lg in logs if lg.status != "success"]:
        res = sc.result(sc.FAIL, res["measured"], reason="; ".join(filter(None, [res.get("reason"), *bad])))
    return res, logs


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
    headers = None
    if not c["args"].dry and PROBE_PATH.is_file():  # the agent model's x-ratelimit headers (E4)
        headers = sc.ratelimit_headers(json.loads(PROBE_PATH.read_text()), c["profile"].role("agent").model)
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


def _session_record(log, sample, categories) -> dict:
    """What the F8 session verdict needs from one session's log (`sc.f8_session_verdict`)."""
    from inspect_ai.model import ChatMessageAssistant, ContentReasoning

    from ape.agent.session import PROBE_PROMPT, REPORT_NUDGE
    from ape.worlds.env_f8 import ITEMS, PROBES, REPORT, VIEWS

    msgs = list(sample.messages or [])
    report = sample.store.get(REPORT)

    def reasoning_only(m) -> bool:
        return (
            isinstance(m, ChatMessageAssistant)
            and not m.tool_calls
            and not (m.text or "").strip()
            and isinstance(m.content, list)
            and any(isinstance(p, ContentReasoning) for p in m.content)
        )

    idx = [i for i, m in enumerate(msgs) if reasoning_only(m)]
    last_assistant = max((i for i, m in enumerate(msgs) if isinstance(m, ChatMessageAssistant)), default=-1)
    score = next(iter(sample.scores.values())).value if sample.scores else {}
    return {
        "log_status": log.status,
        "sample_error": sample.error.message if sample.error else None,
        "n_cases": int(sample.metadata.get("N") or 0),
        "items": sample.store.get(ITEMS, []),
        "views": sample.store.get(VIEWS, []),
        "probes": sample.store.get(PROBES, []),
        "report_submitted": report is not None,
        "report_nudged": any(m.role == "user" and (m.text or "").strip() == REPORT_NUDGE for m in msgs),
        "probe_in_history": any(PROBE_PROMPT in (m.text or "") for m in msgs),
        "reasoning_only_turns": len(idx),
        # A reasoning-only turn that is the session's last word, with no report: it ended the session.
        "reasoning_only_unanswered": bool(idx) and idx[-1] == last_assistant and report is None,
        "scores": score if isinstance(score, dict) else {},
    }


def check_f8_session(c: dict) -> tuple[dict, list]:
    """One F8 session of N=5 cases per session arm (CM0, O-state) through the real Study G task, with a forked
    probe after the last case (RELIABILITY_REVIEW L12): the session loop, the probe's strict schema, the report
    and the per-item records on the live API."""
    from ape.build import build
    from ape.tasks.study_g import f8_session
    from ape.worlds import gen_f8

    asyncio.run(build("dev", "F8", [str(F8["N"])], n_worlds=1, n_tasks=0, relational=True, embed=True))
    model = None
    if c["args"].dry:  # the session needs its own scripted agent (cases, report and probes), not the gate's
        from ape.llm.mock_session import mock_session_agent
        from ape.models import agent_model

        model = agent_model(c["profile"], model="mockllm/model", custom_outputs=mock_session_agent)
    tasks = [f8_session(level=str(F8["N"]), split="dev", arm=arm, limit_worlds=1, checkpoints=str(F8["N"])) for arm in F8["arms"]]
    logs = _resolved(c["run_gate_check"]("f8_session", tasks, model=model, message_limit=F8["message_limit"]))
    arms = {lg.eval.task_args.get("arm"): _session_record(lg, s, gen_f8.PROBE_CATEGORIES) for lg in logs for s in (lg.samples or [])[:1]}
    for lg in logs:  # a task that produced no sample still counts
        arms.setdefault(lg.eval.task_args.get("arm"), {"log_status": lg.status, "sample_error": "no sample", "n_cases": F8["N"]})
    return sc.f8_session_verdict(arms, gen_f8.PROBE_CATEGORIES), logs


def check_extract_f7_1000(c: dict) -> tuple[dict, list]:
    """The real builder on one F7-1000 dev world (the gate's hard cell): APG authoring and LightRAG extraction through
    `ape.artifacts`, then D-017's coverage (`ape.build_quality`) and build health (lost chunks), wall-clock and the
    ledger's calls and cost per system. Warns, never fails, on low coverage: the decisive check is build-dev's."""
    from ape.analysis.cost import load_prices, price_entry
    from ape.artifacts import build_world
    from ape.build import build
    from ape.llm.ledger import Ledger
    from ape.worlds.render import chunk_world
    from ape.worlds.spec import World

    cfg, dry = c["cfg"], c["args"].dry
    kind = "oracle" if dry else "extract"
    world_id = asyncio.run(build("dev", "F7", ["1000"], n_worlds=1, n_tasks=F7_1000_TASKS, relational=True, embed=False))[0]
    path = cfg.world_path(world_id)
    world = World.load(path)
    t0 = time.time()
    built = build_world(str(path), ["chunks", "apg", "lightrag"], lightrag_kind=kind, fake_author=dry)
    seconds = round(time.time() - t0, 1)
    row = bq.world_quality(world, cfg, kind)
    health = bq.build_health([world], cfg, kind, offline=dry)
    wh = health["worlds"][0] if health["worlds"] else {}
    prices = load_prices()
    per_system: dict[str, dict] = {}
    for e in Ledger(cfg.ledger_path).read():  # this world's build and embedding calls, by system
        if e.context.get("world") != world.id:
            continue
        s = per_system.setdefault(str(e.context.get("system") or "?"), {"calls": 0, "embed_calls": 0, "usd": 0.0})
        s["calls" if e.kind == "chat" else "embed_calls"] += 1
        try:
            s["usd"] = round(s["usd"] + price_entry(e, prices), 6)
        except KeyError:  # an unpriced model: the spend guard refuses it; here it is reported, not summed
            s["unpriced"] = e.model
    chunks = len(chunk_world(world))
    apg_calls = (per_system.get("apg-author") or {}).get("calls")
    quality = {
        "world": world.id,
        "kind": kind,
        "offline": dry,
        "chunks": chunks,
        "kinds": built.kinds,
        "seconds": seconds,
        "apg_id_coverage": (row.get("apg") or {}).get("coverage"),
        "apg_fact_coverage": (wh.get("apg") or {}).get("fact_coverage"),
        "apg_lost_chunks": len((wh.get("apg") or {}).get("lost_chunks") or {}),
        "apg_chunks_without_units": (wh.get("apg") or {}).get("chunks_without_units"),
        # One authoring call per chunk when every reply is usable: the excess is retries and repairs.
        "apg_extra_attempts": max(0, apg_calls - chunks) if apg_calls is not None else None,
        "lightrag_id_coverage": (row.get("lightrag") or {}).get("coverage"),
        "lightrag_extraction": ((wh.get("lightrag") or {}).get("extraction")),
        "ledger": per_system,
        "verdict": bq.aggregate([row], expected_cells=[row["cell"]])["verdict"],
        "note": bq.OFFLINE_NOTE if dry else "one F7-1000 dev world, the gate's hard cell; the decisive D-017 check is run_gate build-dev's",
    }
    return sc.extraction_verdict(dict(built.errors), quality, health["warnings"], bq.THRESHOLD), []


def check_retrieval(c: dict) -> tuple[dict, list]:
    return asyncio.run(_retrieval(c["f7"], c["cfg"], c["args"].dry)), []


def node_facts(matcher, node: dict) -> set[str]:
    """The facts an APG node's knowledge carries, by the arms' delivered-text rule (`ape.apg.arm.ApgArm._facts_of`):
    the matcher over the knowledge slot's text; an oracle node without text falls back to its factIds."""
    text = ((node.get("prompt") or {}).get("slots") or {}).get("knowledge") or ""
    return set(matcher.delivered(text)) if text.strip() else set((node.get("props") or {}).get("factIds") or [])


async def _lightrag_recall(world, cfg, dry: bool, matcher) -> tuple[list[float] | None, str]:
    """LightRAG's context for each task with the query itself as its keywords (no kg call; the live arm extracts
    keywords first), credited by the delivered-text rule. Needs L2's index; reported only."""
    from lightrag import QueryParam

    from ape.config import embedding_cache
    from ape.kb.provenance import evidence_pr
    from ape.lgr.adapter import query_params
    from ape.lgr.common import MANIFEST, index_dir, open_rag

    kind = "oracle" if dry else "extract"
    wd = index_dir(cfg, world.id, kind)
    if not (wd / MANIFEST).is_file():
        return None, f"no {kind} index (run L2): LightRAG not measured"

    async def no_llm(*_args, **_kwargs) -> str:  # keywords are given, so LightRAG must never call a model here
        raise RuntimeError("the retrieval check calls no model")

    rag = await open_rag(wd, world.id, no_llm, embedding_cache(cfg), query_time=True)
    params = query_params(cfg.lgr_budget_tokens)
    recalls = []
    for t in world.tasks:
        text = await rag.aquery(t.prompt, QueryParam(**params, only_need_context=True, hl_keywords=[], ll_keywords=[t.prompt]))
        recalls.append(evidence_pr(matcher.delivered(str(text or "")), t.gold_fact_ids)[1])
    return recalls, f"{kind} index, mode {params['mode']}, the query as keywords (no kg call)"


async def _retrieval(world, cfg, dry: bool) -> dict:
    """No LLM: S3s evidence recall at its budget, APG's embedding-shortlist gold rate, and LightRAG's context
    recall with the query as keywords. Facts are credited by the delivered-text rule (`ape.kb.provenance`), as
    the arms credit them: an APG node is "gold" when its knowledge text carries a gold fact."""
    from apg_core import load_graph, route

    from ape.apg.arm import ensure_graph, graph_path, tuned
    from ape.config import embedding_cache
    from ape.kb.baselines import FlatHybrid
    from ape.kb.provenance import evidence_pr, fact_matcher
    from ape.llm.embeddings import CachedEmbeddingsConnector
    from ape.worlds.render import chunk_world

    class NoClassify:  # routing's embedding shortlist only: the LLM classify step is never reached
        def classify(self, query, outline, schema, multi):
            return []

    emb = embedding_cache(cfg)
    matcher = fact_matcher(world)
    s3s = FlatHybrid(chunk_world(world), emb, cfg.s3s_budget_tokens)
    kind = "authored" if graph_path(cfg, world.id, "authored").is_file() else "oracle"
    graph = load_graph(tuned(await ensure_graph(world, kind, cfg, emb)))
    facts_by_node = {n["id"]: node_facts(matcher, n) for n in graph.dfs()}
    recalls, hits = [], []
    for t in world.tasks:
        recalls.append(evidence_pr((await s3s.compile(t.prompt, t)).fact_ids, t.gold_fact_ids)[1])
        await emb.embed([t.prompt])
        routed = route(t.prompt, graph, {"embeddings": CachedEmbeddingsConnector(emb), "llm": NoClassify()})
        gold = {nid for nid, facts in facts_by_node.items() if facts & set(t.gold_fact_ids)}
        hits.append(bool(gold & set(routed["shortlist"])))
    try:
        lgr, lgr_note = await _lightrag_recall(world, cfg, dry, matcher)
    except Exception as e:  # noqa: BLE001  (reported, never judged)
        lgr, lgr_note = None, f"LightRAG not measured: {type(e).__name__}: {str(e)[:160]}"
    res = sc.retrieval_verdict(recalls, hits, graph=kind, dry=dry, lightrag_recall=lgr, lightrag_note=lgr_note)
    res["measured"] |= {"s3s_budget_tokens": cfg.s3s_budget_tokens, "provenance": "delivered-text rule (ape.kb.provenance), as the arms"}
    return res


def check_orchestrator(c: dict) -> tuple[dict, list]:
    from ape import run_gate as rg

    spend: Spend = c["spend"]
    left = c["args"].max_usd - spend.total()
    probe_run = c["orchestrator_run"]()
    # run_gate's guard counts the whole program's spend (the registry), this invocation's earlier checks included, so
    # its budget is that spend plus what the cap leaves: its `remaining` is then exactly `left`. Always measured, run
    # dir or not: on a first smoke the earlier checks' spend is already in the registry.
    with rg.run_environment(probe_run):
        before = rg.spend(probe_run)["spent_usd"]
    run = c["orchestrator_run"](budget_usd=before + left)
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
        except (rg.PhaseError, rg.BudgetError) as e:  # a smoke run's freeze refuses before its budget guard; either way, recorded
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
    pc1_data = json.loads(pc1.read_text()) if pc1.is_file() else None
    parse = sc.anchor_parse_rates(pc1_data)
    res["measured"] |= {
        "run_dir": str(run.dir),
        "warnings": {p: w for p, w in warnings.items() if w},
        "pc1_pass": (pc1_data or {}).get("pass"),
        "pc1_note": "2 questions per type: wiring only, no reproduction claim",
        "anchor_judge_parse": parse,  # D-025: both scorers' parse-failure rates; any gating failure warns
        "run_spend_usd": round(after - before, 6),
    }
    if parse["warn"] and res["status"] == sc.PASS:
        res = sc.result(sc.WARN, res["measured"], reason=f"anchor judge replies did not parse ({parse['warn']}): PC1 would be not evaluable above 5%")
    return res, []


STEP_FUNCS: dict[str, Callable[[dict], tuple[dict, list]]] = {
    "effort": check_effort,
    "L2": check_l2,
    "D017": check_d017,
    "H4": check_h4,
    "L4_L5": check_l4_l5,
    "APG": check_apg,
    "perstep_reasoning": check_perstep_reasoning,
    "pull": check_pull,
    "recovery": check_recovery,
    "burst": check_burst,
    "f8_session": check_f8_session,
    "retrieval": check_retrieval,
    "extract_f7_1000": check_extract_f7_1000,
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


def _cost_lines(c: dict) -> list[str]:
    """report.md's "Cost model check" section."""
    lines = ["", "## Cost model check", ""]
    if c.get("status") == "not measured" or "projections" not in c:
        return lines + [f"Not measured: {c.get('reason')}."]
    lines += [f"Status: **{c['status']}**" + (f" ({'; '.join(c['reasons'])})" if c.get("reasons") else "") + f". {c['note'].capitalize()}.", ""]
    p = c["projections"]
    lines += ["| Projection (conservative) | Program | Gate |", "|---|---|---|"]
    lines += [f"| {name} | ${v['total']:,.0f} | ${v['gate']:,.0f} |" for name, v in p.items()]
    if c.get("output_per_call"):
        lines += ["", "| Role | Effort | Calls | Output/call | Prior | Ratio | Reasoning share | Input/call |", "|---|---|---|---|---|---|---|---|"]
        for r in c["output_per_call"]:
            lines.append(f"| {r['role']} | {r['effort'] or 'default'} | {r['calls']:g} | {r['output_per_call']:g} | {r['prior_output_per_call'] if r['prior_output_per_call'] is not None else '–'} | {r['ratio'] if r['ratio'] is not None else '–'} | {r['reasoning_share'] if r['reasoning_share'] is not None else '–'} | {r['input_per_call']:g} |")
    if c.get("calls_per_sample"):
        lines += ["", "| Arm | Cell | Delivery | Samples | Calls/sample | Prior | Ratio | Input/call |", "|---|---|---|---|---|---|---|---|"]
        for r in c["calls_per_sample"]:
            lines.append(f"| {r['arm']} | {r['cell']} | {r['delivery']} | {r['samples']} | {r['calls_per_sample']:g} | {r['prior_calls_per_sample'] if r['prior_calls_per_sample'] is not None else '–'} | {r['ratio'] if r['ratio'] is not None else '–'} | {r['input_per_call']:g} |")
    if c.get("builds"):
        lines += ["", "| Build system | Calls | Input/call (prior) | Output/call (prior) | Reasoning/call | Calls/chunk (prior) |", "|---|---|---|---|---|---|"]
        for k, b in c["builds"].items():
            pr = b["prior"]
            lines.append(f"| {k} | {b['calls']} | {b['input_per_call']:g} ({pr.get('input')}) | {b['output_per_call']:g} ({pr.get('output')}) | {b['reasoning_per_call']:g} | {b['calls_per_chunk']} ({pr.get('calls_per_chunk')}) |")
    lines += ["", f"Measured entries: `{c['measured_file']}` (not in config/). To adopt them ({c['promote_note']}):", "", f"    {c['promote_command']}"]
    return lines


def write_report(report: dict) -> None:
    out = smoke_dir(report["dry"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    lines = [
        f"# Smoke report ({'DRY: mock models, fake embeddings' if report['dry'] else 'live'})",
        "",
        f"- Status: **{report['status']}**{' (' + report['stopped'] + ')' if report.get('stopped') else ''}",
        f"- Profile: {report['models']['profile']}; spend ${report.get('spend_usd') or 0:.4f} of --max-usd {report['max_usd']} (projected ${report.get('projection_usd', {}).get('total', 0):.3f})",
        f"- {report['started']} to {report.get('finished')}; commit `{(report.get('git') or {}).get('commit')}`{' (dirty)' if (report.get('git') or {}).get('dirty') else ''}",
        *[f"- WARNING: {w}" for w in report.get("warnings") or []],
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
    if "cost_model_check" in report:
        lines += _cost_lines(report["cost_model_check"])
    if not report["dry"]:
        lines += ["", f"Every check's latest live result: `{out / RECORD_NAME}` (run_gate's live preflight requires all of them to pass)."]
    lines += ["", f"Logs and full values: `{out / 'report.json'}`."]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
