"""Main-study decision report: the analyze phase of `ape.run_study` (BUILD_PLAN B4; the statistics are
`ape.analysis.main_*`, the hypothesis table `analysis.main_hypotheses`).

    uv run python -m ape.analyze_main --run-id <id> [--runs-dir runs] [--offline]

`ape.run_study`'s analyze phase calls `analyze(run)` with its `StudyRun`. It writes

    runs/main/<id>/report/decision.json   every number below, machine-readable
    runs/main/<id>/report/report.md       the same, for people

and returns a short JSON-able summary (the analyze manifest records it under `analysis`).

Inputs, each optional (a missing one is reported, never raised):
- `test/manifest.json`: per plan cell, its groups with their arms (`{declared, run}`), the arms skipped as not built,
  the token caps per task cell, the status and the final log files (relative to the run directory; older manifests
  recorded them absolute or relative to the repo, `run_study.resolve_log`). Every finished group's logs are read with
  `analysis.main_load.load_main`, labelled with their plan cell and group; each row's arm is the group's declared plan
  name (S5 = the KG arm the freeze resolved, which the task itself runs as S5), and its tier is the group's profile's
  agent model (`config/models.yaml`; offline, mock logs carry no tier of their own).
- `freeze.json`: the role (primary, or the extension of an earlier run), the KG resolution, the token caps and the
  test seeds. PREREGISTRATION_MAIN.md defines no extension α (EXTENSION_ALPHA, B5), so an extension run is analysed
  alone, as a primary-only analysis of its own worlds at the table's α, and the report says so.
- `tune/tuning_log.jsonl` and `config/selected.yaml`: the candidates each system tried and the selection.

Contents: the run (freeze, role, KG arm, test status), coverage (every plan cell and group: what ran, what was skipped
and why, what is missing), token caps and cap-hit rates per task cell (and per arm), tuning, then the whole
`analysis.main_report` (confirmatory families, descriptive sections, frontier, rank flips, determinism, invariants,
harness rates, GLMM, the analysis choices).
"""

import argparse
import json
import sys
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .analysis.main_load import COLUMNS, load_main, tier_of
from .analysis.main_report import _clean, _f, _table, main_report, render_main

REPORT_DIR = "report"
REPS = 10_000
OFFLINE_REPS = 1_000  # offline rehearsals: one world per cell, so the resamples only need to run, not to converge
EXTENSION_ALPHA: float | None = None  # PREREGISTRATION_MAIN.md defines no extension α (BUILD_PLAN B5); see role_context


# --- inputs ----------------------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text()) if path.is_file() else None
    except (OSError, ValueError):
        return None


def _resolve(shown: str) -> Path:
    from .run_gate import _resolve as resolve

    return resolve(shown)


def profile_tiers(models_path: Path) -> dict[str, str]:
    """profile -> tier of its agent model (config/models.yaml), e.g. main_sol -> sol."""
    try:
        profiles = (yaml.safe_load(models_path.read_text()) or {}).get("profiles") or {}
    except (OSError, yaml.YAMLError):
        return {}
    return {name: tier_of(((p or {}).get("agent") or {}).get("model")) for name, p in profiles.items()}


def load_test_rows(test: dict | None, require_cost: bool, tiers: dict[str, str] | None = None, run_dir: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Every sample-epoch of every finished agent group of the test manifest, labelled with its plan cell, group,
    declared arm (`arm`; the arm as run is `run_arm`) and the tier of the group's profile; and one coverage entry per
    group (status, arms, skipped arms, samples, and the reason when its rows are missing). `run_dir` resolves the
    run-relative log paths (without it, relative paths are the repo's)."""
    from .run_study import resolve_log

    frames, coverage = [], []
    for cell_id, cell in ((test or {}).get("cells") or {}).items():
        for g in cell.get("groups") or []:
            entry = {
                "plan_cell": cell_id, "group": g.get("name"), "plan_phase": cell.get("plan_phase"), "primary": cell.get("primary"), "status": g.get("status"),
                "arms": [a.get("declared") for a in g.get("arms") or []], "skipped": [{"arm": s.get("declared"), "reason": s.get("reason")} for s in g.get("skipped") or []],
                "cells": g.get("cells"), "profile": g.get("profile"), "samples": 0, "reason": None,
            }  # fmt: skip
            files = [resolve_log(run_dir, f) if run_dir is not None else _resolve(f) for f in g.get("log_files") or []]
            labels: dict[str, str] = {}
            clash = [a for a in g.get("arms") or [] if labels.setdefault(a.get("run"), a.get("declared")) != a.get("declared")]
            if g.get("kind", "agent") != "agent":
                entry["reason"] = f"{g.get('kind')} group (not a main-study cell)"
            elif g.get("status") != "done":
                entry["reason"] = f"group {g.get('status')}" + (f": {g['error']}" if g.get("error") else "")
            elif not files:
                entry["reason"] = "no log files"
            elif missing := [str(f) for f in files if not f.is_file()]:
                entry["reason"] = f"log files missing: {missing}"
            elif clash:
                entry["reason"] = f"two declared arms ran as one: {clash}"
            else:
                try:
                    df = load_main(files, plan_cell=cell_id, require_cost=require_cost)
                except (ValueError, OSError, KeyError) as e:
                    entry["reason"] = f"logs unreadable: {type(e).__name__}: {e}"
                else:
                    df = df.assign(run_arm=df["arm"], group=g.get("name"))
                    df["arm"] = df["run_arm"].map(labels).fillna(df["run_arm"])
                    if tiers and g.get("profile") in tiers:
                        df["tier"] = tiers[g["profile"]]
                    frames.append(df)
                    entry["samples"] = int(len(df))
            coverage.append(entry)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[*COLUMNS, "run_arm", "group"])
    return frame, coverage


def role_context(freeze: dict | None) -> dict:
    """The run's role and the α its families are tested at. A primary run: the hypothesis table's α. An extension: its
    own α if PREREGISTRATION_MAIN.md defines one (EXTENSION_ALPHA); it defines none, so the extension is analysed alone
    as a primary-only analysis of its own worlds, at the table's α, and flagged."""
    role = (freeze or {}).get("role") or {"kind": "primary", "of": None}
    if role.get("kind") != "extension":
        return {"kind": "primary", "of": None, "alpha": None, "note": "primary run: each family at the hypothesis table's α"}
    if EXTENSION_ALPHA is not None:
        return {"kind": "extension", "of": role.get("of"), "alpha": EXTENSION_ALPHA, "note": f"extension of run {role.get('of')}: its fresh worlds analysed alone at the pre-registered extension α = {EXTENSION_ALPHA}"}
    return {
        "kind": "extension", "of": role.get("of"), "alpha": None, "primary_only": True,
        "note": f"extension of run {role.get('of')}: PREREGISTRATION_MAIN.md defines no extension α, so this run's fresh worlds are analysed alone as a "
        "primary-only analysis at the table's α; it is not a pre-registered extension and does not combine with the primary run",
    }  # fmt: skip


def caps_table(frame: pd.DataFrame, test: dict | None, freeze: dict | None) -> list[dict]:
    """Per task cell: the frozen token cap, the caps the test groups applied, and the cap-hit rate (any cap: the turn cap
    without an answer, the token cap or another Inspect limit) with the token-limit share; per arm in `arms`."""
    frozen = (freeze or {}).get("token_caps") or {}
    applied: dict[str, set] = {}
    for cell in ((test or {}).get("cells") or {}).values():
        for g in cell.get("groups") or []:
            for tc, cap in (g.get("caps") or {}).items():
                applied.setdefault(tc, set()).add(int(cap))
    cells = sorted(set(frozen) | set(applied) | (set(frame["cell"].dropna()) if len(frame) else set()))
    out = []
    for tc in cells:
        sub = frame[frame["cell"] == tc] if len(frame) else frame
        row: dict[str, Any] = {"cell": tc, "frozen_cap": frozen.get(tc), "applied_caps": sorted(applied.get(tc, ())), "samples": int(len(sub))}
        if len(sub):
            cap = sub["cap_hit"].fillna(False).astype(bool)
            tok = sub["limit_hit"].eq("token")
            row |= {"cap_hits": int(cap.sum()), "cap_hit_rate": float(cap.mean()), "token_limit_hits": int(tok.sum()), "errors": int(sub["error"].fillna(False).astype(bool).sum())}
            row["arms"] = {a: {"samples": int(len(s)), "cap_hit_rate": float(s["cap_hit"].fillna(False).astype(bool).mean()), "token_limit_hits": int(s["limit_hit"].eq("token").sum())} for a, s in sub.groupby("arm")}
        row["consistent"] = frozen.get(tc) is None or not applied.get(tc) or applied[tc] == {int(frozen[tc])}
        out.append(row)
    return out


def tuning_summary(log_path: Path) -> dict:
    """Per system: the candidates tried (ok / failed) and the selection (tune/tuning_log.jsonl)."""
    if not log_path.is_file():
        return {"available": False, "reason": f"{log_path} missing"}
    systems: dict[str, dict] = {}
    for line in log_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        s = systems.setdefault(str(rec.get("system")), {"candidates": 0, "failed": 0, "selected": None, "results": {}})
        if "selected" in rec:
            s["selected"] = rec["selected"]
        elif rec.get("candidate"):
            s["candidates"] += 1
            s["failed"] += rec.get("status") == "failed"
            s["results"][str(rec["candidate"].get("id"))] = rec.get("mean_success")
    counts = {k: v["candidates"] for k, v in systems.items()}
    return {"available": True, "systems": systems, "equal_candidates": len(set(counts.values())) <= 1, "candidates": counts}


# --- the analysis ----------------------------------------------------------------------------------


def analyze(run, reps: int | None = None, glmm: bool = True) -> dict:
    """The analyze phase (module docstring): report/decision.json and report/report.md, and a JSON-able summary.
    Never raises on missing or broken inputs: they are reported (`problems`), and a failing step is recorded."""
    out_dir = run.dir / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    freeze = _read_json(run.freeze_path)
    if freeze is None:
        problems.append(f"{run.freeze_path} missing: the run is not frozen, so this report rests on no frozen design")
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
        report = main_report(frame, alpha=role.get("alpha"), reps=reps or (OFFLINE_REPS if run.offline else REPS), coverage=coverage, glmm=glmm)
    except Exception as e:  # noqa: BLE001
        report = {"errors": [{"section": "main_report", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=6)}], "summary": []}
    kg = (freeze or {}).get("kg") or (test or {}).get("kg") or {}
    header = {
        "study": run.study, "run_id": run.run_id, "offline": run.offline, "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "frozen_at": (freeze or {}).get("frozen_at"), "rehearsal": (freeze or {}).get("rehearsal"), "code_commit": (freeze or {}).get("code_commit"),
        "test_seeds": (freeze or {}).get("test_seeds"), "role": role,
        "kg": {k: kg.get(k) for k in ("needed", "source", "gate_run", "verdict", "key", "declared", "arm", "system", "notes") if k in kg},
        "test_status": (test or {}).get("status"), "primary_complete": (test or {}).get("primary_complete"),
        "skipped_arms": (test or {}).get("skipped_arms") or [], "tiers": tiers, "cap_multiple": (freeze or {}).get("cap_multiple"),
    }  # fmt: skip
    decision = _clean({
        "header": header, "problems": problems, "coverage": coverage,
        "caps": caps_table(frame, test, freeze) if len(coverage) or freeze else [],
        "tuning": tuning_summary(run.phase_dir("tune") / "tuning_log.jsonl"), "analysis": report,
    })  # fmt: skip
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=1, allow_nan=False))
    (out_dir / "report.md").write_text(render(decision))
    labels = {s["member"]: s.get("label") for s in report.get("summary") or []}
    return _clean({
        "status": "done" if test is not None and freeze is not None else "partial",
        "role": role["kind"], "primary_only": bool(role.get("primary_only")), "primary_complete": header["primary_complete"],
        "samples": int(len(frame)), "supported": sorted(m for m, lab in labels.items() if lab == "supported"), "labels": labels,
        "problems": problems, "section_errors": [e.get("section") for e in report.get("errors") or []],
        "decision": str(out_dir / "decision.json"), "report": str(out_dir / "report.md"),
    })  # fmt: skip


# --- markdown --------------------------------------------------------------------------------------


def render(d: dict) -> str:
    h = d["header"]
    L = [f"# Main-study report: run `{h['run_id']}`{' (OFFLINE)' if h['offline'] else ''}", ""]
    L += [f"> **{p}**" for p in d.get("problems") or []] + ([""] if d.get("problems") else [])
    kg = h.get("kg") or {}
    seeds = h.get("test_seeds") or {}
    L += ["## Run", ""]
    L += [
        f"- Generated {h['generated_at']}; frozen {h.get('frozen_at') or 'NOT FROZEN'}" + (" (offline rehearsal)" if h.get("rehearsal") else "") + (f"; code `{h['code_commit']}`" if h.get("code_commit") else "") + ".",
        f"- Role: {h['role']['note']}.",
        f"- KG arm (S5, and M1k/M2's workers): {kg.get('arm') or '–'} ({kg.get('system') or '–'}), from {kg.get('source') or '–'}" + (f", gate verdict {kg['verdict']}" if kg.get("verdict") else "") + ".",
        f"- Test seeds: {seeds.get('base', '–')}–{seeds.get('last', '–')}; test phase {h.get('test_status') or 'not run'}; primary cells complete: {_f(h.get('primary_complete'))}.",
        f"- Tiers by profile: {', '.join(f'{p} → {t}' for p, t in sorted((h.get('tiers') or {}).items()) if p.startswith('main'))}.",
        "",
    ]  # fmt: skip
    L += ["## Coverage", "", "Every plan cell and group of the test phase: the arms that ran, the arms skipped (not built) and the rows read.", ""]
    L += [_table(["Plan cell", "Group", "Status", "Arms", "Skipped", "Cells", "Samples", "Missing"], [[c["plan_cell"], c["group"], c["status"] or "–", ", ".join(c["arms"]) or "–", ", ".join(f"{s['arm']} ({s['reason']})" for s in c["skipped"]) or "–", ", ".join(c.get("cells") or []) or "–", c["samples"], c["reason"] or ""] for c in d.get("coverage") or []])]
    m = d["header"].get("cap_multiple")
    multiple = f"{m} × S1's B0, the pilot cap-hit gate's multiple (D-039)" if m is not None else "the frozen multiple × S1's B0 (not in this freeze)"
    L += ["## Token caps and cap hits", "", f"The frozen cap per task cell ({multiple}; brief §4.5), the caps the test groups applied, and the share of samples cut short (turn cap without an answer, token cap or another Inspect limit).", ""]
    L += [_table(["Task cell", "Frozen cap", "Applied", "Samples", "Cap hits", "Rate", "Token-cap hits", "Errors", "Highest arm rate"], [[c["cell"], _f(c.get("frozen_cap")), ", ".join(f"{x:,}" for x in c.get("applied_caps") or []) or "–", c["samples"], _f(c.get("cap_hits")), _f(c.get("cap_hit_rate")), _f(c.get("token_limit_hits")), _f(c.get("errors")), (lambda a: f"{a[0]} {a[1]['cap_hit_rate']:.3f}" if a else "–")(max((c.get("arms") or {}).items(), key=lambda kv: kv[1]["cap_hit_rate"], default=None))] for c in d.get("caps") or []])]
    t = d.get("tuning") or {}
    L += ["## Tuning", ""]
    if t.get("available"):
        L += [f"Equal candidates per system: {_f(t.get('equal_candidates'))}.", ""]
        L += [_table(["System", "Candidates", "Failed", "Selected"], [[s, x["candidates"], x["failed"], x["selected"] or "–"] for s, x in sorted((t.get("systems") or {}).items())])]
    else:
        L += [f"_not available: {t.get('reason')}_", ""]
    body = render_main(d["analysis"]) if d.get("analysis", {}).get("header") else "\n".join(f"- {e['section']}: {e['error']}" for e in d.get("analysis", {}).get("errors") or [])
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
    result = analyze(StudyRun("main", a.run_id, offline=a.offline, **kw), reps=a.reps)
    sys.stdout.write(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
