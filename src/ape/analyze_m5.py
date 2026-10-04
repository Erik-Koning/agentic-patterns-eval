"""M5 add-on study report: the analyze phase of `ape.run_study --study m5` (D-053; the statistics are
`ape.analysis.m5_report`, which reuses `main_stats`; the hypothesis table `analysis.m5_hypotheses`).

    uv run python -m ape.analyze_m5 --run-id <id> [--runs-dir runs] [--offline]

`ape.run_study`'s analyze phase calls `analyze(run)` with its `StudyRun`, as `ape.analyze_main`'s. It writes

    runs/m5/<id>/report/decision.json   every number below, machine-readable
    runs/m5/<id>/report/report.md       the same, for people

and returns a short JSON-able summary (the analyze manifest records it under `analysis`).

Inputs, each optional (a missing one is reported, never raised):
- `test/manifest.json`: the m5 study's test cells (`m5.test.a`: M5 with its concurrent M1; `m5.test.b`: M5-spec with
  its concurrent M2), read as `analyze_main.load_test_rows` reads the main study's: every finished group's logs through
  `main_load.load_main`, labelled with plan cell, group, declared arm and tier; then each sample's ledger record
  (`analysis.m5_load.attach_ledgers`);
- `freeze.json`: the m5 run's freeze, and what it inherits from its frozen main run (the main run id, its test seeds,
  token caps, KG resolution and selections; the freeze's `inherited`, else config/main_inheritance.json).
Only this run's own test cells enter the contrasts: the main study's M1 and M2 runs are never read (concurrent
controls; `m5_report.m5_rows`).

Contents: the run (freeze, the main run it inherits, role, test status), coverage (every plan cell and group), token
caps and cap-hit rates per task cell (`analyze_main.caps_table`), errored attempts' usage, then the whole
`m5_report` (the confirmatory families L1 and L2, the descriptive contrasts, the ledger's price and diagnostics,
harness rates, the design and the analysis choices).
"""

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .analysis.m5_load import LEDGER_COLUMNS, attach_ledgers
from .analysis.m5_report import m5_report, render_m5
from .analysis.main_load import COLUMNS
from .analysis.main_report import _clean, _f, _table
from .analyze_main import _read_json, _usage_or_error, caps_table, load_test_rows, profile_tiers

REPORT_DIR = "report"
REPS = 10_000
OFFLINE_REPS = 1_000
EXTENSION_ALPHA: float | None = None  # PREREGISTRATION_M5.md defines no extension α
INHERITED = ("main_run", "main_offline", "freeze_sha256", "frozen_at", "code_commit", "test_seeds", "token_caps", "cap_multiple", "kg")


def inherited(run, freeze: dict | None) -> dict:
    """What this run inherits from its frozen main run: the freeze's record of it (`inherited` / `inheritance`), else
    the run's config/main_inheritance.json; {} when neither exists."""
    rec = (freeze or {}).get("inherited") or (freeze or {}).get("inheritance")
    if not isinstance(rec, dict):
        path = getattr(run, "inheritance_path", None)
        rec = _read_json(Path(path)) if path is not None else None
    return {k: rec.get(k) for k in INHERITED if k in rec} if isinstance(rec, dict) else {}


def role_context(freeze: dict | None) -> dict:
    """A primary run: the table's α. An extension: PREREGISTRATION_M5.md defines no extension α, so it is analysed alone,
    primary-only, at the table's α, and flagged."""
    role = (freeze or {}).get("role") or {"kind": "primary", "of": None}
    if role.get("kind") != "extension":
        return {"kind": "primary", "of": None, "alpha": None, "note": "primary run: each family at the hypothesis table's α"}
    if EXTENSION_ALPHA is not None:
        return {"kind": "extension", "of": role.get("of"), "alpha": EXTENSION_ALPHA, "note": f"extension of run {role.get('of')} at the pre-registered extension α = {EXTENSION_ALPHA}"}
    return {
        "kind": "extension", "of": role.get("of"), "alpha": None, "primary_only": True,
        "note": f"extension of run {role.get('of')}: PREREGISTRATION_M5.md defines no extension α, so this run is analysed alone as a primary-only analysis at the table's α; it does not combine with the earlier run",
    }  # fmt: skip


def analyze(run, reps: int | None = None) -> dict:
    """The analyze phase (module docstring): report/decision.json and report/report.md, and a JSON-able summary.
    Never raises on missing or broken inputs: they are reported (`problems`), and a failing step is recorded."""
    out_dir = run.dir / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    freeze = _read_json(run.freeze_path)
    if freeze is None:
        problems.append(f"{run.freeze_path} missing: the run is not frozen, so this report rests on no frozen design")
    inh = inherited(run, freeze)
    if not inh.get("main_run"):
        problems.append("no inherited main run recorded: the test worlds, caps and KG arm this run should share with a frozen main run are unverified")
    test_path = run.phase_dir("test") / "manifest.json"
    test = _read_json(test_path)
    if test is None:
        problems.append(f"{test_path} missing: the test phase has not run")
    elif test.get("status") != "done":
        problems.append(f"test phase {test.get('status')}" + ("; the primary cells are complete" if test.get("primary_complete") else "; the primary cells are NOT complete"))
    role = role_context(freeze)
    if role.get("primary_only"):
        problems.append(role["note"])
    tiers = profile_tiers(run.models_path)
    try:
        frame, coverage = load_test_rows(test, require_cost=not run.offline, tiers=tiers, run_dir=run.dir)
    except Exception as e:  # noqa: BLE001  (reported; the report still stands)
        frame, coverage = pd.DataFrame(columns=COLUMNS), []
        problems.append(f"test rows unreadable: {type(e).__name__}: {e}")
    problems += [f"{c['plan_cell']} ({c['group']}): {c['reason']}" for c in coverage if c["reason"] and not str(c["reason"]).startswith("group skipped")]
    try:
        frame = attach_ledgers(frame)
    except Exception as e:  # noqa: BLE001  (diagnostics only)
        frame = frame.assign(**dict.fromkeys(LEDGER_COLUMNS))
        problems.append(f"ledger records unreadable: {type(e).__name__}: {e}")
    try:
        report = m5_report(frame, alpha=role.get("alpha"), reps=reps or (OFFLINE_REPS if run.offline else REPS), coverage=coverage)
    except Exception as e:  # noqa: BLE001
        report = {"errors": [{"section": "m5_report", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=6)}], "summary": []}
    kg = inh.get("kg") or (freeze or {}).get("kg") or (test or {}).get("kg") or {}
    caps_freeze = {"token_caps": (freeze or {}).get("token_caps") or inh.get("token_caps") or {}}
    header = {
        "study": run.study, "run_id": run.run_id, "offline": run.offline, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frozen_at": (freeze or {}).get("frozen_at"), "rehearsal": (freeze or {}).get("rehearsal"), "code_commit": (freeze or {}).get("code_commit"),
        "inherited": {k: v for k, v in inh.items() if k not in ("token_caps", "kg")}, "test_seeds": (freeze or {}).get("test_seeds") or inh.get("test_seeds"), "role": role,
        "kg": {k: kg.get(k) for k in ("source", "gate_run", "verdict", "arm", "system") if k in kg},
        "test_status": (test or {}).get("status"), "primary_complete": (test or {}).get("primary_complete"),
        "skipped_arms": (test or {}).get("skipped_arms") or [], "tiers": tiers, "cap_multiple": (freeze or {}).get("cap_multiple") or inh.get("cap_multiple"),
    }  # fmt: skip
    decision = _clean({
        "header": header, "problems": problems, "coverage": coverage,
        "caps": caps_table(frame, test, caps_freeze) if len(coverage) or freeze else [],
        "errored_attempt_usage": _usage_or_error(test, frame, run.dir), "analysis": report,
    })  # fmt: skip
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=1, allow_nan=False))
    (out_dir / "report.md").write_text(render(decision))
    labels = {s["member"]: s.get("label") for s in report.get("summary") or []}
    return _clean({
        "status": "done" if test is not None and freeze is not None else "partial",
        "role": role["kind"], "primary_only": bool(role.get("primary_only")), "primary_complete": header["primary_complete"],
        "main_run": inh.get("main_run"), "samples": int(len(frame)),
        "supported": sorted(m for m, lab in labels.items() if lab == "supported"), "labels": labels,
        "problems": problems, "section_errors": [e.get("section") for e in report.get("errors") or []],
        "decision": str(out_dir / "decision.json"), "report": str(out_dir / "report.md"),
    })  # fmt: skip


# --- markdown --------------------------------------------------------------------------------------


def render(d: dict) -> str:
    h = d["header"]
    L = [f"# M5 add-on study report: run `{h['run_id']}`{' (OFFLINE)' if h['offline'] else ''}", ""]
    L += [f"> **{p}**" for p in d.get("problems") or []] + ([""] if d.get("problems") else [])
    inh, seeds, kg = h.get("inherited") or {}, h.get("test_seeds") or {}, h.get("kg") or {}
    L += ["## Run", ""]
    L += [
        f"- Generated {h['generated_at']}; frozen {h.get('frozen_at') or 'NOT FROZEN'}" + (" (offline rehearsal)" if h.get("rehearsal") else "") + (f"; code `{h['code_commit']}`" if h.get("code_commit") else "") + ".",
        f"- Inherits main run `{inh.get('main_run') or '–'}`" + (f" (frozen {inh['frozen_at']}" + (f", code `{inh['code_commit']}`" if inh.get("code_commit") else "") + ")" if inh.get("frozen_at") else "") + ": its test worlds (paired on the same tasks), token caps, KG arm and M1's selection. The controls ran fresh, beside the ledger arms.",
        f"- Role: {h['role']['note']}.",
        f"- KG arm (M2's and M5-spec's workers): {kg.get('arm') or '–'} ({kg.get('system') or '–'})" + (f", from {kg['source']}" if kg.get("source") else "") + ".",
        f"- Test seeds: {seeds.get('base', '–')}–{seeds.get('last', '–')}; test phase {h.get('test_status') or 'not run'}; primary cells complete: {_f(h.get('primary_complete'))}.",
        "",
    ]  # fmt: skip
    L += ["## Coverage", "", _table(["Plan cell", "Group", "Status", "Arms", "Skipped", "Cells", "Samples", "Missing"], [[c["plan_cell"], c["group"], c["status"] or "–", ", ".join(c["arms"]) or "–", ", ".join(f"{s['arm']} ({s['reason']})" for s in c["skipped"]) or "–", ", ".join(c.get("cells") or []) or "–", c["samples"], c["reason"] or ""] for c in d.get("coverage") or []])]
    L += ["## Token caps and cap hits", "", "The main run's frozen cap per task cell (inherited), the caps the test groups applied, and the share of samples cut short.", ""]
    L += [_table(["Task cell", "Frozen cap", "Applied", "Samples", "Cap hits", "Rate", "Token-cap hits", "Errors"], [[c["cell"], _f(c.get("frozen_cap")), ", ".join(f"{x:,}" for x in c.get("applied_caps") or []) or "–", c["samples"], _f(c.get("cap_hits")), _f(c.get("cap_hit_rate")), _f(c.get("token_limit_hits")), _f(c.get("errors"))] for c in d.get("caps") or []])]
    u = d.get("errored_attempt_usage") or {}
    L += ["## Errored attempts' usage (not in realised cost)", ""]
    if u.get("available"):
        L += [_table(["Arm", "Logged $", "Errored attempts' $", "Share", "Retried samples"], [[a, _f(x.get("logged_usd"), 4), _f(x.get("unlogged_usd"), 4), _f(x.get("unlogged_share")), x.get("retried_samples", x.get("reason", "–"))] for a, x in sorted((u.get("arms") or {}).items())])]
    else:
        L += [f"_not available: {u.get('reason')}_", ""]
    a = d.get("analysis") or {}
    body = render_m5(a) if a.get("header") else "\n".join(f"- {e['section']}: {e['error']}" for e in a.get("errors") or [])
    L += [body.split("\n", 1)[1] if body.startswith("# ") else body]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from .run_study import StudyRun

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (("runs_root", a.runs_dir), ("config_dir", a.config_dir)) if v is not None}
    result = analyze(StudyRun("m5", a.run_id, offline=a.offline, **kw), reps=a.reps)
    sys.stdout.write(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
