"""Study G's analysis phase (BUILD_PLAN B10; the interface is D-036's): `analyze(run) -> dict`.

    uv run python -m ape.run_study analyze --study study_g --run-id <id> [--offline]

`ape.run_study`'s analyze phase calls `analyze(run)` with the `StudyRun`. It writes

    runs/study_g/<id>/report/decision.json   every number of the report, machine-readable
    runs/study_g/<id>/report/report.md       the same, for people

and returns a compact JSON-able summary, which the phase records under `analysis` in the analyze manifest.

**Inputs** (each may be missing; a missing one is reported, never raised on):
- `test/manifest.json`: per plan cell, its status and groups. A group carries its arms as declared in the plan and as
  run (a tuned arm runs as its selection), the arms skipped because they are not built yet, its profile and effort,
  and its log files.
- `config/run_plan.yaml` (the run's config dir): each cell's block, design and planned arms.
- `config/models.yaml`: each group's capability point, `<tier>-<effort>` from its profile's agent model and its effort.
- `config/model_costs.yaml`: prices for usage Inspect left unpriced.
- `freeze.json`: the role (primary or extension), the frozen test-seed block, the rehearsal flag and the commit.
- The selections (`config/selected.yaml`) and the tuning log (`tune/tuning_log.jsonl`): what each tuned arm ran as.

**What it does.** Session cells (`g.cm.*`, `g.topo.*`) and the capability-anchor cells (`g.cap.*`) go to
`g_report.report_from_cells`-equivalent loading (`g_load.load_g_cells` with the plan, the points and the prices).
Arms are relabelled from the arm that ran to the arm the plan declares, per cell (the logs record the run arm, and the
analysis compares plan arms). `g_report.g_report` then computes every hypothesis. On top of that the report adds:
- the run: freeze, role, seed block and test status;
- the selections and tuning summary;
- a **coverage table** per plan cell and arm: planned, run, skipped (with the runner's reason; S1, M1 and M2 session
  arms are BUILD_PLAN B9's and are skipped offline until it lands), missing (planned and not skipped, but no data),
  and sessions present against planned.

The summary's `status` is `ok` (every planned arm of every test cell has data), `partial` (some cells or arms are
missing, failed, stopped or skipped) or `no_data` (no session was analysed).
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis.g_load import cell_info, load_g_cells, point_label
from .analysis.g_report import _clean, _table, g_report, render
from .config import ROOT

REPORT_DIR = "report"
TEST_PHASES = ("capability_anchor", "context_management", "topology")
SUMMARY_PROBLEMS = 40  # problems listed in the returned summary (all of them are in decision.json)


# --- inputs ------------------------------------------------------------------------------------------


def _read_json(path: Path, what: str, problems: list[str]) -> dict | None:
    if not path.is_file():
        problems.append(f"{what} missing ({path})")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        problems.append(f"{what} unreadable ({path}: {type(e).__name__}: {e})")
        return None


def _plan(run, problems: list[str]):
    from .budget import load_plan

    path = run.config("run_plan.yaml")
    try:
        return load_plan(path)
    except Exception as e:  # noqa: BLE001 - a missing or broken plan is a reported problem
        problems.append(f"run plan unreadable ({path}: {type(e).__name__}: {e}); cells fall back to their id prefix")
        return None


def _prices(run, problems: list[str]) -> dict:
    from .analysis.cost import load_prices

    try:
        return load_prices(run.costs_path)
    except Exception as e:  # noqa: BLE001
        problems.append(f"model costs unreadable ({run.costs_path}: {type(e).__name__}: {e}); unpriced usage stays unpriced")
        return {}


def _log_path(f: str, run) -> Path | None:
    p = Path(f)
    for cand in (p, ROOT / p, run.dir / p):
        if cand.is_file():
            return cand
    return None


def group_point(group: Mapping, models_path: Path) -> str | None:
    """A group's capability point: its profile's agent model and the group's effort (else the profile's)."""
    from .models import load_profile

    try:
        agent = load_profile(group.get("profile"), path=models_path).role("agent")
    except Exception:  # noqa: BLE001 - an unknown profile: the plan's point is used instead
        return None
    return point_label(agent.model, group.get("effort") or agent.reasoning_effort)


def plan_test_cells(plan) -> list:
    """The enabled Study G cells of the test plan phases, in plan order."""
    if plan is None:
        return []
    return [c for c in plan.cells if c.study == "study_g" and c.phase in TEST_PHASES and c.enabled]


def collect(run, test: dict | None, plan, problems: list[str]) -> tuple[dict, dict, dict]:
    """From the test manifest: {plan_cell: [log files]}, {plan_cell: {run arm: [declared arms]}} and
    {plan_cell: capability point}."""
    cells: dict[str, list[str]] = {}
    labels: dict[str, dict[str, list[str]]] = {}
    points: dict[str, str] = {}
    for cid, c in ((test or {}).get("cells") or {}).items():
        files: list[str] = []
        for g in c.get("groups") or []:
            for a in g.get("arms") or []:
                declared, ran = a.get("declared"), a.get("run") or a.get("declared")
                if declared:
                    names = labels.setdefault(cid, {}).setdefault(str(ran), [])
                    if declared not in names:
                        names.append(declared)
            if g.get("status") not in (None, "done", "skipped"):
                problems.append(f"{cid} ({g.get('name')}): group {g.get('status')}" + (f": {g['error']}" if g.get("error") else ""))
            for f in g.get("log_files") or []:
                p = _log_path(str(f), run)
                if p is None:
                    problems.append(f"{cid}: log file not found ({f})")
                else:
                    files.append(str(p))
            if cid not in points and (pt := group_point(g, run.models_path)):
                points[cid] = pt
        if files:
            cells[cid] = files
    for cid, names in labels.items():
        for ran, declared in names.items():
            if len(declared) > 1:
                problems.append(f"{cid}: arms {declared} all ran as {ran}; their logs are the same runs and are reported under each name")
    return cells, labels, points


def relabel(df: pd.DataFrame, labels: Mapping[str, Mapping[str, Sequence[str]]]) -> pd.DataFrame:
    """Rows labelled with the arm that ran, relabelled with the plan's declared arm per plan cell (a run arm two declared
    arms share is repeated under each). Rows of cells or arms the manifest does not map are left as they are."""
    if df is None or not len(df) or "arm" not in df or "plan_cell" not in df:
        return df
    parts = []
    for (cell, arm), g in df.groupby(["plan_cell", "arm"], dropna=False, sort=False):
        declared = list((labels.get(cell) or {}).get(str(arm)) or [arm])
        parts += [g.assign(arm=d, run_arm=arm) for d in declared]
    return pd.concat(parts, ignore_index=True)


def selections(run, problems: list[str]) -> dict:
    """The selections (what each tuned arm ran as) and a tuning summary per system."""
    import yaml

    out: dict[str, Any] = {"selected": {}, "tuning": {}}
    if run.selected_path.is_file():
        try:
            out["selected"] = {k: {f: v.get(f) for f in ("arm", "candidate", "mean_success", "cost_usd", "env")} for k, v in (yaml.safe_load(run.selected_path.read_text()) or {}).items() if isinstance(v, Mapping)}
        except Exception as e:  # noqa: BLE001
            problems.append(f"selections unreadable ({run.selected_path}: {type(e).__name__}: {e})")
    else:
        problems.append(f"no selections ({run.selected_path}): tuned arms are reported under their plan names only")
    log = run.phase_dir("tune") / "tuning_log.jsonl"
    if log.is_file():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            s = out["tuning"].setdefault(str(r.get("system")), {"candidates": 0, "selected": None, "rule": None})
            if "candidate" in r:
                s["candidates"] += 1
            if "selected" in r:
                s["selected"], s["rule"] = r.get("selected"), r.get("rule")
    tm = _read_json(run.manifest_path("tune"), "tune manifest", []) or {}
    if tm.get("skipped_systems"):
        out["skipped_systems"] = tm["skipped_systems"]
    if tm.get("tuning_completeness") is not None:
        out["completeness"] = tm["tuning_completeness"]
    return out


# --- coverage ----------------------------------------------------------------------------------------


def coverage(plan, test: dict | None, sessions: pd.DataFrame, capability: pd.DataFrame, points: Mapping[str, str]) -> list[dict]:
    """One row per plan test cell and arm: planned, run, skipped (with the reason), missing (planned, not skipped, no
    data), and sessions (or anchor tasks) present against planned."""
    from .run_gate import _arm_names

    mcells = (test or {}).get("cells") or {}
    present: dict[tuple[str, str], int] = {}
    if sessions is not None and len(sessions) and {"plan_cell", "arm", "session"} <= set(sessions.columns):
        present = {k: int(v) for k, v in sessions.groupby(["plan_cell", "arm"])["session"].nunique().items()}
    cap_tasks = {}
    if capability is not None and len(capability) and "point" in capability:
        cap_tasks = {r["point"]: int(r.get("n_tasks") or 0) for r in capability.to_dict("records")}
    rows = []
    for cell in plan_test_cells(plan):
        m = mcells.get(cell.id) or {}
        groups = m.get("groups") or []
        ran = {a.get("declared") for g in groups for a in g.get("arms") or [] if g.get("status") == "done"}
        skipped = {a.get("declared"): a.get("reason") for g in groups for a in g.get("skipped") or []}
        info = cell_info(cell.id, plan)
        point = points.get(cell.id) or info.get("point")
        for arm in _arm_names(cell.spec.get("arms") or []):
            if cell.kind == "agent":
                n = cap_tasks.get(point, 0) if arm in ran else 0
                planned = int(cell.spec.get("n_tasks") or 0) * len(cell.spec.get("cells") or [1])
                unit = "tasks"
            else:
                n = present.get((cell.id, arm), 0)
                planned = int(cell.spec.get("sessions") or 0)
                unit = "sessions"
            state = "skipped" if arm in skipped else ("present" if n else ("missing" if m else "not run"))
            rows.append({
                "cell": cell.id, "plan_phase": cell.phase, "block": info.get("block"), "point": point, "arm": arm,
                "cell_status": m.get("status") or "not run", "state": state, "reason": skipped.get(arm), "present": n,
                "planned": planned, "unit": unit,
            })  # fmt: skip
    return rows


def _summary(cov: list[dict], d: dict, test: dict | None, sessions: pd.DataFrame, problems: list[str], out_dir: Path) -> dict:
    states: dict[str, int] = {}
    for r in cov:
        states[r["state"]] = states.get(r["state"], 0) + 1
    if sessions is None or not len(sessions):
        status = "no_data"
    elif states.get("missing") or states.get("not run") or states.get("skipped") or (test or {}).get("status") != "done":
        status = "partial"
    else:
        status = "ok"
    return _clean({
        "status": status,
        "report": str(out_dir / "report.md"),
        "decision": str(out_dir / "decision.json"),
        "test_status": (test or {}).get("status"),
        "primary_complete": (test or {}).get("primary_complete"),
        "decisions": {r["id"]: r["decision"] for r in d.get("decisions") or []},
        "coverage": states,
        "skipped": [f"{r['cell']}: {r['arm']}" for r in cov if r["state"] == "skipped"],
        "missing": [f"{r['cell']}: {r['arm']}" for r in cov if r["state"] in ("missing", "not run")],
        "session_epochs": int(len(sessions)) if sessions is not None else 0,
        "problems": problems[:SUMMARY_PROBLEMS] + ([f"... {len(problems) - SUMMARY_PROBLEMS} more in decision.json"] if len(problems) > SUMMARY_PROBLEMS else []),
    })  # fmt: skip


# --- the phase ---------------------------------------------------------------------------------------


def analyze(run, **report_kw) -> dict:
    """The analyze phase for Study G: loads the run's test logs, reports every hypothesis, writes report/decision.json
    and report/report.md, and returns the summary (module docstring). `report_kw` goes to `g_report`."""
    problems: list[str] = []
    out_dir = run.dir / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    test = _read_json(run.manifest_path("test"), "test manifest (the test phase has not run)", problems)
    freeze = _read_json(run.freeze_path, "freeze.json", problems)
    plan = _plan(run, problems)
    prices = _prices(run, problems)
    if test is not None and test.get("status") != "done":
        problems.append(f"test phase {test.get('status')}; primary cells complete: {test.get('primary_complete')}")
    cells, labels, points = collect(run, test, plan, problems)
    data = load_g_cells(cells, plan=plan, points=points, prices=prices)
    items, sessions = relabel(data.items, labels), relabel(data.sessions, labels)
    d = g_report(items, sessions, data.capability, cells=data.cells, problems=[*problems, *data.problems], **report_kw)
    cov = coverage(plan, test, sessions, data.capability, points)
    sel = selections(run, problems)
    d["run"] = _clean({
        "study": run.study,
        "run_id": run.run_id,
        "offline": bool(run.offline),
        "freeze": {k: (freeze or {}).get(k) for k in ("frozen_at", "rehearsal", "role", "test_seeds", "code_commit")} if freeze else None,
        "test": {"status": (test or {}).get("status"), "primary_complete": (test or {}).get("primary_complete"), "cells": {c: x.get("status") for c, x in ((test or {}).get("cells") or {}).items()}},
    })  # fmt: skip
    d["coverage"] = _clean(cov)
    d["selections"] = _clean(sel)
    d["problems"] = list(problems)
    (out_dir / "decision.json").write_text(json.dumps(d, indent=1, default=str) + "\n")
    (out_dir / "report.md").write_text(render_run(d))
    return _summary(cov, d, test, sessions, [*problems, *data.problems], out_dir)


def render_run(d: dict) -> str:
    """The run report: the run, coverage and selections, then `g_report.render`'s sections."""
    r = d["run"]
    fz = r.get("freeze") or {}
    seeds = fz.get("test_seeds") or {}
    role = fz.get("role") or {}
    L = [f"# Study G report: run `{r['run_id']}`{' (OFFLINE: mock models)' if r['offline'] else ''}", ""]
    L += [
        f"- Freeze: {fz.get('frozen_at') or 'NOT FROZEN'}{' (offline rehearsal)' if fz.get('rehearsal') else ''}; role {role.get('kind') or '–'}{' of ' + role['of'] if role.get('of') else ''}; "
        f"test seeds {seeds.get('base', '–')}–{seeds.get('last', '–')}; commit `{fz.get('code_commit') or '–'}`.",
        f"- Test phase: {r['test']['status'] or 'not run'}; primary cells complete: {r['test']['primary_complete']}.",
        "",
    ]
    if d.get("problems"):
        L += ["**Inputs and runs:**", ""] + [f"- {p}" for p in d["problems"]] + [""]
    scale = " Offline runs are scaled down (one session or world per cell, one epoch): *Present* is against the live plan's *Planned*." if r["offline"] else ""
    L += ["## Coverage", "", f"Per plan test cell and arm. *skipped*: the runner skipped an arm not built yet (S1, M1 and M2 session arms are BUILD_PLAN B9's); *missing*: planned and run, but no data; *not run*: the cell is not in the test manifest.{scale}", ""]
    L += [_table(["Cell", "Point", "Arm", "Cell status", "State", "Present", "Planned", "Reason"], [[c["cell"], c["point"] or "–", c["arm"], c["cell_status"], c["state"], f"{c['present']} {c['unit']}", c["planned"], c["reason"] or ""] for c in d["coverage"]])]
    sel = d.get("selections") or {}
    if sel.get("selected"):
        L += ["## Selections (dev tuning)", "", _table(["Plan arm", "Runs as", "Candidate", "Dev success", "Dev $"], [[k, v.get("arm"), v.get("candidate"), _num(v.get("mean_success")), _num(v.get("cost_usd"), 4)] for k, v in sel["selected"].items()])]
    if sel.get("tuning"):
        L += [_table(["System", "Candidates", "Selected", "Rule"], [[k, v["candidates"], v["selected"] or "–", v["rule"] or ""] for k, v in sel["tuning"].items()])]
    body = render(d).split("\n", 1)
    L += [body[1] if len(body) > 1 else ""]
    return "\n".join(L)


def _num(x: Any, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.{nd}f}"
