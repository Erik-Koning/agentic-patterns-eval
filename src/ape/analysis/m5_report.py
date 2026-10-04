"""The M5 add-on study's report (D-053): its pre-registered tests (`m5_hypotheses`), the ledger's price and diagnostics,
and the data checks, from the tidy frame, as one JSON-serialisable dict and a markdown rendering.

    frame = m5_load.attach_ledgers(main_load.load_main(paths, plan_cell="m5.test.a"))   # or ape.analyze_m5 on a run
    d = m5_report(frame)
    md = render_m5(d)

**Reuse.** The main study's machinery throughout: `main_stats.build_tables` (task means over epochs, paired tasks,
world clusters with F1/F2 registry sharing), `evaluate_family` (the sign-flip tests, Holm, labels, per-cell estimates,
per-world influence) and `ratio_ci` (world-clustered BCa ratios).

**Concurrent controls only.** The contrasts compare each ledger arm with the control that ran beside it in the m5
study's own test cells (`m5.test.*`): rows of any other plan cell, the main study's own (older) M1 and M2 runs above
all, are dropped and named (`excluded_plan_cells`). Comparing with the main study's runs would put model and provider
drift between the runs into the contrast.

Every section is computed whatever the others do: a missing arm or cell makes a member "not evaluable" with the reason,
and a failure in one section is recorded under `errors` while the rest stands.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from .gate_stats import harness_rates
from .m5_hypotheses import F1_CELLS, F2_CELLS, HYPOTHESES, LEDGER_ARMS, PLAN_PREFIX, STUDY_A_CELLS, STUDY_B_CELLS, hypothesis_table
from .m5_load import LEDGER_COLUMNS
from .main_hypotheses import PRIMARY_TIER, TEST_TEXT
from .main_load import METERS
from .main_report import _clean, _ci, _design, _f, _per_cell, _pp, _safe, _table
from .main_stats import EXACT_MAX_CLUSTERS, REPS, Contrast, build_tables, confirmatory_rows, evaluate_family, ratio_ci

COST_GROUPS = {  # ledger arm -> {label: cells}
    "M5": {"Study A": STUDY_A_CELLS, "F1": F1_CELLS, "F2": F2_CELLS, **{c: (c,) for c in STUDY_A_CELLS}},
    "M5-spec": {"Study B": STUDY_B_CELLS, **{c: (c,) for c in STUDY_B_CELLS}},
}
COST_METERS = ("tokens", "usd")

CHOICES = (
    "Decision: the main study's world-clustered sign-flip test on the per-task contrast M5 − M1 (M5-spec − M2), exact "
    f"over all 2^G flips up to {EXACT_MAX_CLUSTERS} clusters (L1: 9), else 10,000 Monte Carlo flips (L2: 36 clusters); "
    "one-sided superiority at α = 0.025; Holm within each family (L1, L2), none across families and none with the main "
    "study's families, to which this study adds no hypothesis.",
    "Concurrent controls: each ledger arm is compared only with the control that ran beside it in this study's test cells "
    f"({PLAN_PREFIX}*), on the same tasks; the main study's own M1 and M2 runs are never used (drift).",
    "Task value: the mean over an arm's 3 epochs; a cell's estimate is the mean over the tasks both arms ran; a pooled "
    "member is the equal-weight mean of its cells, each cell's estimate reported beside it.",
    "Clusters: F1 and F2 worlds of one seed share one supplier registry, so L1's members have 9 clusters; F3 and F7 "
    "worlds are their own (L2: 36).",
    "Cap hits and errored samples are failures (success 0), as in the main study (D-047).",
    "Interval: the sign-flip test inverted (two-sided 1 − 2α); the world-clustered BCa bootstrap and cluster-t are "
    "reported beside it.",
    "The ledger's price: realised cost ratios (ledger arm / control) on tokens and $, on the paired tasks, cells equally "
    "weighted, with world-clustered 95% BCa intervals. Descriptive: no cost condition is pre-registered.",
    "Ledger diagnostics (descriptive): per sample from its mas_ledger and mas_accounting records (m5_load); a field a "
    "record lacks is reported as not available. Success with and without a replan is conditional on the run's own "
    "difficulty, not an effect of replanning.",
)


def m5_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The rows of this study's test cells (`m5.test.*`); every other plan cell's rows are dropped and named. Rows
    without a plan cell (a frame built outside a run) are kept and counted."""
    if "plan_cell" not in frame.columns:
        return frame, {"excluded_plan_cells": [], "excluded_rows": 0, "unlabelled_rows": int(len(frame))}
    pc = frame["plan_cell"]
    unlabelled = pc.isna()
    keep = unlabelled | pc.fillna("").astype(str).str.startswith(PLAN_PREFIX)
    return frame[keep], {
        "excluded_plan_cells": sorted({str(p) for p in pc[~keep].dropna()}),
        "excluded_rows": int((~keep).sum()),
        "unlabelled_rows": int(unlabelled.sum()),
    }


def concurrency(rows: pd.DataFrame) -> list[dict]:
    """Per task cell: the plan cells each ledger arm and its control came from, and whether they are the same."""
    out = []
    if rows.empty or "plan_cell" not in rows.columns:
        return out
    for arm, control in LEDGER_ARMS.items():
        for cell in sorted(set(rows.loc[rows["arm"].isin([arm, control]), "cell"].dropna())):
            sub = rows[rows["cell"] == cell]
            pa = sorted({str(p) for p in sub.loc[sub["arm"] == arm, "plan_cell"].dropna()})
            pc = sorted({str(p) for p in sub.loc[sub["arm"] == control, "plan_cell"].dropna()})
            out.append({"arm": arm, "control": control, "cell": cell, "arm_plan_cells": pa, "control_plan_cells": pc, "concurrent": bool(pa) and pa == pc})
    return out


def cost_ratios(tables, alpha: float = 0.025, reps: int = REPS, seed: int = 0) -> list[dict]:
    """Ledger arm / control mean cost per meter, per cell and pooled (cells equally weighted), on the paired tasks."""
    out = []
    for arm, groups in COST_GROUPS.items():
        control = LEDGER_ARMS[arm]
        for meter in COST_METERS:
            tab = (tables.cost.get(PRIMARY_TIER) or {}).get(meter)
            for label, cells in groups.items():
                row = {"arm": arm, "control": control, "meter": meter, "cells": label}
                if tab is None or arm not in tab.columns or control not in tab.columns:
                    out.append(row | {"ratio": None, "ci": [None, None], "reason": f"no {meter} cost for {arm} or {control}"})
                    continue
                pair = tab[[arm, control]][tab.index.get_level_values("cell").isin(cells)].dropna()
                if pair.empty:
                    out.append(row | {"ratio": None, "ci": [None, None], "reason": "no paired tasks"})
                    continue
                num, den = Contrast.from_series(pair[arm], tables.cluster_by), Contrast.from_series(pair[control], tables.cluster_by)
                r = ratio_ci(num, den, alpha, reps, seed)
                out.append(row | r | {"tasks": int(len(pair)), "clusters": num.G, "arm_mean": num.est, "control_mean": den.est})
    return out


def _ratio(num: pd.Series, den: pd.Series) -> float | None:
    ok = num.notna() & den.notna()
    d = float(den[ok].sum())
    return float(num[ok].sum()) / d if ok.any() and d > 0 else None


def _ledger_group(sub: pd.DataFrame) -> dict:
    rec = sub[sub["ledger_present"].fillna(False).astype(bool)] if "ledger_present" in sub.columns else sub.iloc[0:0]
    row: dict[str, Any] = {"samples": int(len(sub)), "with_record": int(len(rec))}
    if rec.empty:
        return row | {"reason": "no mas_ledger record"}
    rep = pd.to_numeric(rec["ledger_replans"], errors="coerce")
    rounds = pd.to_numeric(rec["ledger_rounds"], errors="coerce")
    stalls = pd.to_numeric(rec["ledger_stalls"], errors="coerce")
    if rep.notna().any():
        per_task = rec.assign(_r=rep).dropna(subset=["_r"]).groupby(["world", "task"])["_r"].mean()
        row |= {"replans_per_task": float(per_task.mean()), "replanned_share": float((rep.dropna() >= 1).mean()), "max_replans": int(rep.max())}
        succ = rec.assign(_r=rep).dropna(subset=["_r"])
        for name, part in (("success_replanned", succ[succ["_r"] >= 1]), ("success_not_replanned", succ[succ["_r"] < 1])):
            row[name] = float(part["success"].mean()) if len(part) else None
    if rounds.notna().any():
        row["rounds_per_sample"] = float(rounds.mean())
    if stalls.notna().any():
        row["stalled_share"] = float((stalls.dropna() >= 1).mean())
        row["stall_rate"] = _ratio(stalls, rounds) if rounds.notna().any() else None
    lt = pd.to_numeric(rec["ledger_tokens"], errors="coerce")
    st = pd.to_numeric(rec["sample_tokens"], errors="coerce")
    if lt.notna().any() and st.notna().any():
        row["ledger_token_share"] = _ratio(lt, st)
    if rec["success"].notna().any():
        row["success"] = float(rec["success"].mean())
    return row


def ledger_diagnostics(rows: pd.DataFrame) -> dict:
    """Per ledger arm, pooled and per cell: replans per task, the share of samples that replanned, rounds per sample,
    the stall rate (stalled rounds over rounds), the share of samples with a stall, the ledger's share of the tokens,
    and success with and without a replan (descriptive)."""
    if rows.empty or "ledger_present" not in rows.columns:
        return {"available": False, "reason": "no ledger columns (m5_load.attach_ledgers not run)"}
    out: dict = {"available": bool(rows["ledger_present"].fillna(False).astype(bool).any()), "arms": {}}
    for arm in LEDGER_ARMS:
        sub = rows[rows["arm"] == arm]
        if sub.empty:
            continue
        out["arms"][arm] = {"pooled": _ledger_group(sub), "cells": {c: _ledger_group(s) for c, s in sub.groupby("cell")}}
    if not out["available"]:
        out["reason"] = "no sample carries a mas_ledger record"
    return out


def m5_report(frame: pd.DataFrame, hypotheses=HYPOTHESES, alpha: float | None = None, reps: int = REPS, seed: int = 0, cluster_by: str = "kb", coverage: Sequence[dict] | None = None, flagged_worlds: Sequence[str] | None = None) -> dict:
    """Every section of the m5 report as one JSON-serialisable dict (module docstring). `alpha` overrides every family's
    one-sided α (default: the table's)."""
    errors: list = []
    frame = frame.copy()
    for col, default in (("tier", PRIMARY_TIER), ("error", False), ("cap_hit", False), ("answer_key", None), ("plan_cell", None)):
        if col not in frame.columns:
            frame[col] = default
    for col in LEDGER_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    mine, scope = m5_rows(frame)
    rows, notes = confirmatory_rows(mine, exclude_prefix="")
    meters = {m: c for m, c in METERS.items() if c in mine.columns}
    tables = build_tables(mine, meters, cluster_by=cluster_by, exclude_prefix="")
    flagged = set(flagged_worlds or ())
    d: dict = {
        "header": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": int(len(frame)),
            "scope": scope,
            "confirmatory_rows": notes | {"nan_success_rows": tables.notes.get("nan_success_rows", 0)},
            "plan_cells": sorted({str(p) for p in mine["plan_cell"].dropna()}),
            "tiers": sorted(tables.tm),
            "settings": {"alpha_override": alpha, "reps": reps, "seed": seed, "cluster_by": cluster_by, "primary_tier": PRIMARY_TIER},
            "coverage": list(coverage or []),
            "design": _safe(errors, "design", _design, rows) or [],
            "concurrency": _safe(errors, "concurrency", concurrency, rows) or [],
        },
        "hypothesis_table": hypothesis_table(hypotheses),
    }
    fams = {}
    for i, h in enumerate(hypotheses):
        if h.members:
            fams[h.id] = _safe(errors, f"hypothesis {h.id}", evaluate_family, tables, h, alpha, reps, seed + 101 * i, flagged_worlds=flagged)
    d["hypotheses"] = fams

    def _summary(status: str) -> list[dict]:
        return [
            {"family": fid, "member": mid, "contrast": m.get("contrast"), "cells": m.get("cells") or m.get("cells_planned"), "test": m.get("test"), "est": m.get("est"), "ci": m.get("ci"), "p": m.get("p"), "holm_level": m.get("holm_level"), "label": m.get("label") or ("estimated" if m.get("evaluable") else "not evaluable"), "reason": m.get("reason") or m.get("why"), "caveats": m.get("caveats") or [], "per_cell": m.get("per_cell"), "tasks": m.get("tasks"), "clusters": m.get("clusters")}
            for fid, f in fams.items()
            if f and f.get("status") == status
            for mid, m in f["members"].items()
        ]

    d["summary"] = _summary("confirmatory")
    d["descriptive"] = _summary("descriptive")
    d["caveats"] = [f"{s['member']}: {c}" for s in d["summary"] for c in s["caveats"]]
    d["cost"] = _safe(errors, "cost ratios", cost_ratios, tables, 0.025, min(reps, 4000), seed) or []
    d["ledger"] = _safe(errors, "ledger diagnostics", ledger_diagnostics, rows) or {"available": False, "reason": "failed"}
    d["harness"] = _safe(errors, "harness", harness_rates, rows.assign(error=rows["error"].fillna(False), cap_hit=rows["cap_hit"].fillna(False)), ("arm", "cell")) if len(rows) else []
    d["choices"] = list(CHOICES)
    d["errors"] = errors
    return _clean(d)


# --- markdown --------------------------------------------------------------------------------------


def _ratio_cell(r: dict) -> str:
    if r.get("ratio") is None:
        return f"– ({r.get('reason')})" if r.get("reason") else "–"
    lo, hi = (r.get("ci") or [None, None])[:2]
    return f"{_f(r['ratio'], 2)} [{_f(lo, 2)}, {_f(hi, 2)}]"


def render_m5(d: dict) -> str:
    h = d["header"]
    L = ["# M5 add-on study report", "", f"Generated {h['generated_at']}; {h['rows']:,} sample-epochs, {h['confirmatory_rows'].get('used', 0):,} in this study's test cells ({', '.join(h['plan_cells']) or '–'}).", ""]
    sc = h.get("scope") or {}
    if sc.get("excluded_rows"):
        L += [f"> Rows of other plan cells left out ({sc['excluded_rows']:,}): {', '.join(sc['excluded_plan_cells'])}. Only this study's concurrent controls enter its contrasts.", ""]
    if d.get("errors"):
        L += [f"> **Section failed:** {e['section']}: {e['error']}" for e in d["errors"]] + [""]
    conc = [c for c in h.get("concurrency") or [] if not c.get("concurrent")]
    if conc:
        L += [f"> **Not concurrent:** {c['arm']} vs {c['control']} on {c['cell']}: {c['arm_plan_cells'] or '–'} against {c['control_plan_cells'] or '–'}" for c in conc] + [""]
    L += ["## Confirmatory tests", "", "Interval: the sign-flip test inverted (two-sided 1 − 2α). Labels: supported / not supported (never 'no effect') / not tested / not evaluable (data missing). A pooled member is the equal-weight average over its cells; its per-cell estimates follow.", ""]
    L += [_table(["Member", "Contrast", "Cells", "Test", "Δ", "Interval", "p", "Holm level", "Clusters / tasks", "Result", "Per cell"], [[s["member"], s["contrast"], ", ".join(s["cells"] or []), TEST_TEXT.get(s["test"], s["test"]), _pp(s["est"]), _ci(s["ci"]), _f(s["p"], 4), _f(s["holm_level"], 4), f"{s.get('clusters') or '–'} / {s.get('tasks') or '–'}", (s["label"] or "–") + (f" ({s['reason']})" if s.get("reason") else "") + (" ⚠ caveat" if s.get("caveats") else ""), _per_cell(s.get("per_cell"))] for s in d["summary"]])]
    if d.get("caveats"):
        L += ["**Caveats on supported claims** (the sign flip assumes symmetric world contributions):", ""] + [f"- {c}" for c in d["caveats"]] + [""]
    for fid, f in (d.get("hypotheses") or {}).items():
        if not f or f.get("status") != "confirmatory":
            continue
        for mid, m in f["members"].items():
            infl = m.get("influence") or []
            if infl:
                top = "; ".join(f"{r['cluster']} {_pp(r['contribution'])} (without it Δ {_pp(r['est_without'])}" + (f", p {_f(r['p_without'], 4)}" if r.get("p_without") is not None else "") + ")" for r in infl[:3])
                L += [f"- {mid} most influential worlds: {top}"]
            for c in (m.get("coverage") or {}).get("missing_cells") or []:
                L += [f"- {mid}: {c} left out ({m['coverage']['cells'][c].get('reason')})"]
    L += ["", "## Descriptive contrasts", ""]
    L += [_table(["Member", "Contrast", "Cells", "Δ", "Interval (95%)", "Clusters / tasks", "Per cell"], [[s["member"], s["contrast"], ", ".join(s["cells"] or []), _pp(s["est"]), _ci(s["ci"]), f"{s.get('clusters') or '–'} / {s.get('tasks') or '–'}", _per_cell(s.get("per_cell")) if s.get("per_cell") else ("–" if s.get("est") is not None else s.get("reason") or "–")] for s in d.get("descriptive") or []])]
    L += ["## The ledger's price (cost ratio, ledger arm / control)", "", "Realised mean cost per sample on the paired tasks, cells equally weighted; world-clustered 95% BCa interval.", ""]
    by = {}
    for r in d.get("cost") or []:
        by.setdefault((r["arm"], r["cells"]), {})[r["meter"]] = r
    L += [_table(["Arm / control", "Cells", "Tokens", "$", "Tasks"], [[f"{a} / {next(iter(x.values()))['control']}", cells, _ratio_cell(x.get("tokens", {})), _ratio_cell(x.get("usd", {})), _f(next(iter(x.values())).get("tasks"))] for (a, cells), x in by.items()])]
    led = d.get("ledger") or {}
    L += ["## Ledger diagnostics", ""]
    if not led.get("available"):
        L += [f"_not available: {led.get('reason')}_", ""]
    rows = []
    for arm, x in (led.get("arms") or {}).items():
        for name, g in [("pooled", x.get("pooled") or {}), *sorted((x.get("cells") or {}).items())]:
            rows.append([arm, name, g.get("samples"), g.get("with_record"), _f(g.get("replans_per_task"), 2), _f(g.get("replanned_share")), _f(g.get("rounds_per_sample"), 1), _f(g.get("stall_rate")), _f(g.get("stalled_share")), _f(g.get("ledger_token_share")), f"{_f(g.get('success_replanned'))} / {_f(g.get('success_not_replanned'))}"])
    if rows:
        L += [_table(["Arm", "Cells", "Samples", "With record", "Replans per task", "Replanned share", "Rounds per sample", "Stall rate", "Samples with a stall", "Ledger token share", "Success replanned / not"], rows)]
    hr = d.get("harness") or []
    L += ["## Harness", "", _table(["Arm", "Cell", "Samples", "Errors", "Cap hits"], [[r["arm"], r["cell"], r["samples"], f"{_f(r['error_rate'])}{' ⚠' if r['error_fails'] else ''}", f"{_f(r['cap_hit_rate'])}{' ⚠' if r['cap_hit_fails'] else ''}"] for r in hr])]
    L += ["## Design", "", _table(["Cell", "Arm", "Samples", "Tasks", "Worlds", "Epochs", "Errors", "Success", "Plan cells"], [[r["cell"], r["arm"], r["samples"], r["tasks"], r["worlds"], r["epochs"], _f(r["errors"]), _f(r["success"]), ", ".join(r.get("plan_cells") or [])] for r in h.get("design") or []])]
    L += ["## Analysis choices", ""] + [f"- {c}" for c in d.get("choices") or []]
    return "\n".join(L) + "\n"


__all__ = ["CHOICES", "concurrency", "cost_ratios", "ledger_diagnostics", "m5_report", "m5_rows", "render_m5"]
