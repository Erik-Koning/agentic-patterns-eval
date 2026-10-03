"""Study G report (BUILD_PLAN B10): every row of `g_hypotheses` with its decision, and the descriptive tables, as a
JSON-serialisable dict and as markdown, in the style of `ape.analyze_gate`.

    from ape.analysis.g_report import report_from_cells, render
    d = report_from_cells({"g.cm.luna-high": [log, ...], "g.topo.sol": [...], "g.cap.sol-high": [...], ...})
    Path("report.md").write_text(render(d))

`g_report(items, sessions, capability, ...)` works on the tidy tables (`g_load`, or `g_power`'s simulations);
`report_from_cells` loads {plan_cell: [log paths]} first. The run-directory glue (reading a study run's manifests)
belongs to the study runner, which calls `report_from_cells`.

Contents:
- **Decisions** per confirmatory hypothesis (G-H2a; G-H3-pre → G-H3a → G-H3b in fixed sequence): SUPPORTED,
  NOT_SUPPORTED, NOT_TESTED (an earlier step of the sequence failed) or NOT_TESTABLE (with the reason: missing arms,
  points, capability or sessions). Descriptive rows are never decided; G-H1 and G-H2b (descriptive since D-033)
  carry their estimate and 95% interval in the decision rows and a table of their own.
- **G-H1** (descriptive) for the reference S-CM*, the sensitivity S1-pre and S1 as scored, on the logit and the
  probability scale, M1 alongside M2, with the per-point gaps.
- **G-H2:** Gap_T per point (the IUT), R_x per strategy and point with three intervals, the change in R_x over the
  capability span per strategy (descriptive) with the gain on the logit scale as a sensitivity, degradation slopes,
  the gain before and after the W crossing, and the N = 10 control.
- **G-H3:** the gatekeeper, H3a, H3b pooled over every point where all four topology arms ran (Luna-low, Luna-high
  and Sol-high in the D-033 design), the shares with Fieller and bootstrap intervals, the cost ratio per meter.
- **Descriptive:** outcomes per arm × point (item, dependency, report, session success and pass^k, overflow), cost per
  solved item per meter, probe F1 per checkpoint and the probe-to-behaviour correlation, the failure taxonomy, and the
  mixed models (D-029).
- **Caveats:** capability points the anchor cannot separate, tests whose exact sign-flip cannot reach α, ratios whose
  denominator is not bounded away from 0, missing sessions, errors and loader problems.

Nothing raises on missing data: every section reports what is there.
"""

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import g_stats as gs
from .g_hypotheses import ALPHA, DEFAULT_REFERENCE, HYPOTHESES, OPEN_CHOICES, REFERENCES, TOST_ESTIMAND, TOST_MARGIN, GHypothesis

CHOICES = (
    "Sessions are the clusters (D-029): epochs of a session are pooled before any statistic; a world run at several "
    "capability points is one cluster across them.",
    "The primary test of every confirmatory row is the session-clustered t (unbiased variance with shared worlds as one "
    "cluster; Satterthwaite df, capped at the effective number of clusters − 1 when worlds are shared across points). "
    "The wild sign-flip (exact up to 16 clusters) is reported beside it; where it cannot reach α (fewer than 6 sessions "
    "at α = 0.025) the report says so.",
    "A session's outcome is its item success rate; overflowed and never-reached items fail, and an errored sample's "
    "unscored items fail (as in the gate). The binary all-items-and-report session success is reported with pass^k.",
    "G-H1's slope is on the empirical-logit scale of the epoch-pooled item success (log((k + 0.5) / (n − k + 0.5))); "
    "the probability-scale slope is reported. Gaps, shares, headroom and cost ratios are on the probability scale.",
    "Ratios are not estimated when their denominator is ≤ 2 pp; Fieller's interval is reported unbounded when the "
    "denominator is not significantly positive.",
    "Cost per solved item is Σ cost / Σ items solved over the sessions and epochs of an arm and point; probe calls are "
    "excluded from every meter, management (cm) calls included. Wall-clock includes probe time.",
    "Degradation slopes leave out items lost to overflow (the harness failed them) and items with no view tokens.",
    "D-033: G-H1 (reference S-CM*, sensitivity S1-pre, S1 descriptive only) and G-H2b (the change in R_x over the "
    "capability span) are estimates with 95% intervals; no decision rests on them. The confirmatory rows are G-H2a and "
    "G-H3-pre → G-H3a → G-H3b.",
)


# ---------- helpers ----------


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN/inf to None, tuples and sets to lists, keys to strings."""
    if isinstance(x, Mapping):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.ndarray, pd.Series)):
        return _clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{float(x):.{nd}f}"


def _pp(x: Any) -> str:
    return "–" if x is None else f"{100 * float(x):+.1f} pp"


def _ci(ci: Sequence | None, fmt=_f) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "–"
    return f"[{fmt(ci[0])}, {fmt(ci[1])}]"


def _p(x: Any) -> str:
    return "–" if x is None else (f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ---------- sections ----------


def capability_section(capability, sessions: pd.DataFrame) -> dict:
    """Measured capability per point, which session points lack one, and adjacent points the anchor cannot separate
    (their difference within 2 standard errors)."""
    if isinstance(capability, pd.DataFrame):
        rows = capability.to_dict("records") if len(capability) else []
    else:
        rows = [{"point": p, "capability": v, "se": None, "source": "given"} for p, v in (capability or {}).items()]
    x = gs.capability_map(capability)
    pts = sorted(set(sessions["point"].dropna())) if sessions is not None and len(sessions) and "point" in sessions else []
    missing = [p for p in pts if p not in x]
    se = {r["point"]: r.get("se") for r in rows}
    order = sorted(x, key=x.get)
    close = []
    for a, b in zip(order, order[1:], strict=False):
        sa, sb = se.get(a), se.get(b)
        if sa is not None and sb is not None and np.isfinite(sa) and np.isfinite(sb):
            if x[b] - x[a] < 2 * math.hypot(sa, sb):
                close.append({"points": [a, b], "difference": x[b] - x[a], "se": math.hypot(sa, sb)})
    return {"points": rows, "order": order, "missing": missing, "not_separated": close}


def header_section(items: pd.DataFrame, sessions: pd.DataFrame, cells: Mapping | None, problems: Sequence[str]) -> dict:
    counts = []
    if sessions is not None and len(sessions):
        for key, g in sessions.groupby(["block", "point", "arm", "N"], dropna=False):
            counts.append({"block": key[0], "point": key[1], "arm": key[2], "N": key[3], "sessions": int(g["session"].nunique()), "session_epochs": len(g), "errors": int(g["error"].sum()) if "error" in g else 0})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "session_epochs": len(sessions) if sessions is not None else 0,
        "items": len(items) if items is not None else 0,
        "points": sorted(set(sessions["point"].dropna())) if sessions is not None and len(sessions) else [],
        "arms": sorted(set(sessions["arm"].dropna())) if sessions is not None and len(sessions) else [],
        "counts": counts,
        "cells": dict(cells or {}),
        "problems": list(problems),
    }


def _label(test: dict, *, claim_key: str = "reject") -> str:
    if not test or not test.get("testable", False):
        return "NOT_TESTABLE"
    return "SUPPORTED" if test.get(claim_key) else "NOT_SUPPORTED"


def decisions(d: dict, hypotheses: Sequence[GHypothesis]) -> list[dict]:
    """One row per hypothesis: its decision (confirmatory) or `descriptive`, with the numbers behind it. G-H1 and
    G-H2b (descriptive since D-033) carry their estimate and 95% interval; were a table to make them confirmatory again,
    they would be decided here as before."""
    out = []
    g1, g2, g3 = d["gh1"]["primary"], d["gh2"], d["gh3"]
    for h in hypotheses:
        row = {"id": h.id, "role": h.role, "family": h.family, "step": h.step}
        conf = h.role == "confirmatory"
        if h.id == "G-H1":
            sens = d["gh1"]["references"].get("S1-pre", {})
            row |= {
                "decision": _label(g1) if conf else "descriptive",
                "reference": g1.get("reference"),
                "estimate_span": g1.get("est_span"),
                "ci_span": g1.get("ci_span"),
                "estimate": g1.get("est"),
                "ci": g1.get("ci"),
                "sensitivity_S1_pre": {"estimate_span": sens.get("est_span"), "ci_span": sens.get("ci_span")},
                "reason": g1.get("reason"),
            }
            if conf:
                row |= {"p": g1.get("p_t"), "p_flip": g1.get("p_flip")}
            out.append(row)
            continue
        if h.id == "G-H2b":
            for s, r in g2["tost"].items():
                label = ("NOT_TESTABLE" if not r["testable"] else ("EQUIVALENT" if r.get("holm_equivalent") else "NOT_SHOWN")) if conf else "descriptive"
                out.append(row | {"id": f"G-H2b[{s}]", "decision": label, "estimate": r.get("est"), "ci": r.get("ci"), "reason": r.get("reason"), **({"margin": r.get("margin"), "p": r.get("p_t"), "holm_level": r.get("holm_level")} if conf else {})})
            continue
        if not conf:
            out.append(row | {"decision": "descriptive"})
            continue
        if h.id == "G-H2a":
            gap = g2["gap"]
            failed = [c for c, r in gap["points"].items() if not r.get("reject")]
            row |= {"decision": "NOT_TESTABLE" if not gap["testable"] else ("SUPPORTED" if gap["all_positive"] else "NOT_SUPPORTED"), "points_not_shown": failed, "flip_unreachable": gap["flip_unreachable"], "reason": gap.get("reason")}
        elif h.id.startswith("G-H3"):
            if not g3.get("testable"):
                row |= {"decision": "NOT_TESTABLE", "reason": g3.get("reason")}
            elif h.id == "G-H3-pre":
                row |= {"decision": _label(g3["pre"]), "estimate": g3["pre"]["est"], "ci": g3["pre"]["ci"], "p": g3["pre"]["p_t"]}
            elif h.id == "G-H3a":
                a = g3["h3a"]
                row |= {"decision": "NOT_TESTED" if not a["tested"] else ("SUPPORTED" if a["claim"] else "NOT_SUPPORTED"), "estimate": g3["isolation_share"].get("est"), "ci": g3["isolation_share"].get("ci_fieller"), "p": a["p_t"]}
            elif h.id == "G-H3b":
                b = g3["h3b"]
                row |= {"decision": "NOT_TESTED" if not b["tested"] else ("SUPPORTED" if b["claim"] else "NOT_SUPPORTED"), "recovery": g3["recovery_share"].get("est"), "recovery_p": b["recovery"].get("p_t"), "cost_ratio": b["cost"].get("ratio"), "cost_ratio_ci": b["cost"].get("ratio_ci"), "cost_p": b["cost"].get("p_t"), "reason": b["cost"].get("reason")}
        out.append(row)
    return out


def caveats(d: dict) -> list[str]:
    out = []
    cap = d["capability"]
    if cap["missing"]:
        out.append(f"No measured capability for {cap['missing']}: those points are left out of every slope.")
    for c in cap["not_separated"]:
        out.append(f"The capability anchor does not separate {c['points'][0]} and {c['points'][1]} (difference {c['difference']:.3f}, SE {c['se']:.3f}).")
    gap = d["gh2"]["gap"]
    if gap.get("flip_unreachable"):
        out.append(f"G-H2a: the exact sign-flip cannot reach α at {gap['flip_unreachable']} (too few sessions); the decision rests on the t-test there.")
    for s, h in d["gh2"]["headroom"].items():
        for c, r in h["points"].items():
            if r.get("testable") and r.get("den_significant") is False:
                out.append(f"R_{s} at {c}: the headroom is not significantly positive; its Fieller interval is unbounded.")
    for p in d["header"]["problems"]:
        out.append(f"Loader: {p}")
    errs = [c for c in d["header"]["counts"] if c["errors"]]
    if errs:
        out.append("Errored session-epochs (their unscored items count as failures): " + ", ".join(f"{c['arm']} at {c['point']} ({c['errors']})" for c in errs))
    return out


def g_report(
    items: pd.DataFrame,
    sessions: pd.DataFrame,
    capability,
    *,
    hypotheses: Sequence[GHypothesis] = HYPOTHESES,
    reference: str = DEFAULT_REFERENCE,
    tost_estimand: str = TOST_ESTIMAND,
    tost_margin: float | None = None,
    alpha: float = ALPHA,
    reps: int = gs.REPS,
    boot: int = 2000,
    seed: int = gs.SEED,
    primary: str = "t",
    glmm: bool = True,
    cells: Mapping | None = None,
    problems: Sequence[str] = (),
) -> dict:
    """The Study G report as a JSON-serialisable dict (see the module docstring). `primary` decides every confirmatory
    test by the session-clustered t ("t", default) or the sign-flip ("flip"); both p-values are always reported."""
    sessions = sessions if sessions is not None else pd.DataFrame()
    items = items if items is not None else pd.DataFrame()
    kw = {"alpha": alpha, "reps": reps, "seed": seed, "primary": primary}
    d: dict = {"header": header_section(items, sessions, cells, problems), "alpha": alpha}
    d["capability"] = capability_section(capability, sessions)
    d["outcomes"] = gs.outcome_table(sessions)
    d["gh1"] = {
        "primary": gs.gh1(sessions, capability, reference=reference, boot=boot, **kw),
        "references": {r: gs.gh1(sessions, capability, reference=r, boot=boot, **kw) for r in REFERENCES},
        "references_prob": {r: gs.gh1(sessions, capability, reference=r, scale="prob", **kw) for r in REFERENCES},
        "M1": gs.gh1(sessions, capability, topology="M1", reference=reference, **kw),
    }
    margin = tost_margin if tost_margin is not None else TOST_MARGIN[tost_estimand]
    d["gh2"] = gs.gh2(sessions, capability, estimand=tost_estimand, margin=margin, boot=boot, **kw)
    other = "gain" if tost_estimand == "R" else "R"
    d["gh2"]["tost_sensitivity"] = {s: gs.tost(sessions, capability, s, estimand=other, **kw) for s in d["gh2"]["tost"]}
    d["gh2"]["short_control"] = gs.short_control(sessions, alpha=alpha, reps=reps, seed=seed)
    d["gh2"]["crossing_split"] = gs.crossing_split(sessions, alpha=alpha, seed=seed)
    d["gh2"]["degradation"] = gs.degradation(items) if len(items) and "block" in items else []
    d["gh3"] = gs.gh3(sessions, boot=boot, **kw)
    d["costs"] = gs.cost_table(sessions)
    d["probes"] = {"by_checkpoint": gs.probe_table(sessions), "behaviour": gs.probe_behaviour(sessions, items)}
    d["taxonomy"] = gs.taxonomy_table(sessions)
    d["glmm"] = {"topology": gs.glmm(items, capability, block="topo", reference="S1"), "context_management": gs.glmm(items, capability, block="cm", arms=("CM0", "CM-sum", "CM-todo", "O-state"), reference="CM0")} if glmm else {}
    d["decisions"] = decisions(d, hypotheses)
    d["hypotheses"] = [h.to_dict() for h in hypotheses]
    d["choices"] = list(CHOICES)
    d["open_choices"] = list(OPEN_CHOICES)
    d["caveats"] = caveats(d)
    return _clean(d)


def report_from_cells(cells: Mapping[str, Sequence], *, plan=None, points: Mapping[str, str] | None = None, prices: Mapping | None = None, capability: Mapping[str, float] | None = None, **kw) -> dict:
    """`g_report` on {plan_cell: [log paths]} (session cells and g.cap.* anchor cells)."""
    from .g_load import load_g_cells

    data = load_g_cells(cells, plan=plan, points=points, prices=prices, capability=capability)
    return g_report(data.items, data.sessions, data.capability, cells=data.cells, problems=data.problems, **kw)


# ---------- markdown ----------


def _test_row(name: str, r: dict, fmt=_f) -> list:
    return [name, fmt(r.get("est")), _ci(r.get("ci"), fmt), _p(r.get("p_t")), _p(r.get("p_flip")), _f(r.get("flip_min_p"), 4), r.get("reason") or ""]


def render(d: dict) -> str:
    h = d["header"]
    L = ["# Study G report", "", f"Generated {h['generated_at']}. {h['session_epochs']} session-epochs, {h['items']} items; points {', '.join(h['points']) or '–'}; arms {', '.join(h['arms']) or '–'}. One-sided α = {d['alpha']} per family.", ""]
    if d["caveats"]:
        L += [f"> **Caveat:** {c}" for c in d["caveats"]] + [""]

    L += ["## Decisions", ""]
    rows = []
    for r in d["decisions"]:
        if r["role"] != "confirmatory":
            continue
        est = r.get("estimate")
        extra = ""
        if r["id"] == "G-H3b":
            extra = f"recovery {_f(r.get('recovery'))}, cost ratio {_f(r.get('cost_ratio'))} {_ci(r.get('cost_ratio_ci'))}"
        elif r["id"] == "G-H2a":
            extra = f"not shown at {r.get('points_not_shown')}" if r.get("points_not_shown") else ""
        rows.append([r["id"], f"**{r['decision']}**", _f(est), _ci(r.get("ci")), _p(r.get("p")), extra or (r.get("reason") or "")])
    L += [_table(["Hypothesis", "Decision", "Estimate", "Interval", "p (t)", "Notes"], rows)]
    rows = []
    for r in d["decisions"]:
        if r["role"] == "confirmatory":
            continue
        if r["id"] == "G-H1":
            s = r.get("sensitivity_S1_pre") or {}
            rows.append([f"G-H1 (M2 − {r.get('reference')})", "change in the logit gap over the capability span", _f(r.get("estimate_span")), _ci(r.get("ci_span")), r.get("reason") or ""])
            rows.append(["G-H1 (M2 − S1-pre, sensitivity)", "same, on items before S1's overflow", _f(s.get("estimate_span")), _ci(s.get("ci_span")), ""])
        elif r["id"].startswith("G-H2b"):
            rows.append([r["id"], "change in R_x over the capability span", _f(r.get("estimate")), _ci(r.get("ci")), r.get("reason") or ""])
    if rows:
        L += ["Descriptive estimates (D-033; 95% intervals, no decision):", "", _table(["Row", "Estimand", "Estimate", "Interval", "Note"], rows)]

    cap = d["capability"]
    L += ["## Capability (S1 on F7-10 + F3-5)", ""]
    L += [_table(["Point", "Capability", "SE", "Tasks", "Source"], [[p.get("point"), _f(p.get("capability")), _f(p.get("se")), p.get("n_tasks", "–"), p.get("source", "anchor")] for p in cap["points"]])]

    L += ["## Outcomes by arm and point", ""]
    L += [_table(["Block", "Point", "Arm", "N", "Sessions", "Item success", "95% CI", "Dependency", "Report exact", "Session success", "pass^k", "Overflow", "Errors"],
                 [[o["block"], o["point"], o["arm"], o["N"], o["sessions"], _f(o["item_success"]), _ci(o["item_success_ci"]), _f(o["dependency_success"]), _f(o["report_exact"]), _f(o["session_success"]), _f(o["pass_k"]), _f(o["overflow_rate"]), o["errors"]] for o in d["outcomes"]])]

    g1 = d["gh1"]
    L += [
        "## G-H1: topology gap vs capability (descriptive, D-033)",
        "",
        f"Reference **{g1['primary'].get('reference')}**; S1-pre (both arms on items before S1's overflow) is the sensitivity analysis; S1 as scored is "
        "reported only descriptively (its gap carries the overflow rule, which drifts with capability). Slope per unit measured capability and the "
        "change along the fitted line over the capability span, with 95% session-clustered and bootstrap intervals.",
        "",
    ]

    def g1_row(name: str, x: dict, boot: bool = True) -> list:
        return [name, _f(x.get("est")), _ci(x.get("ci")), _f(x.get("est_span")), _ci(x.get("ci_span")), _ci(x.get("boot_ci_span")) if boot else "–", ", ".join(x.get("points_used") or []), x.get("reason") or ""]

    rows = [g1_row(f"M2 − {r} (logit)", x) for r, x in g1["references"].items()]
    rows += [g1_row(f"M2 − {r} (prob)", x, boot=False) for r, x in g1["references_prob"].items()]
    rows += [g1_row(f"M1 − {g1['M1'].get('reference')} (logit)", g1["M1"], boot=False)]
    L += [_table(["Contrast", "Slope", "Interval", "Change over span", "Interval", "Bootstrap", "Points", "Note"], rows)]
    gap_rows = [[r, c, _f(x["est"]), _ci(x["ci"]), x["n"]] for r, xx in g1["references"].items() for c, x in xx.get("gaps", {}).items()]
    L += ["Per-point gaps (logit):", "", _table(["Reference", "Point", "Gap", "Interval", "Sessions"], gap_rows)]

    g2 = d["gh2"]
    L += ["## G-H2: context management", "", "### Gap_T = s(O-state) − s(CM0) (G-H2a, intersection-union)", ""]
    L += [_table(["Point", "Gap", "Interval", "p (t)", "p (flip)", "min flip p", "Shown"], [[c, _pp(r["est"]), _ci(r["ci"], _pp), _p(r["p_t"]), _p(r["p_flip"]), _f(r["flip_min_p"], 4), _f(r["reject"])] for c, r in g2["gap"]["points"].items()])]
    L += ["### Headroom recovered R_x (G-H2c)", ""]
    rows = []
    for s, hh in g2["headroom"].items():
        for c, r in hh["points"].items():
            rows.append([s, c, _f(r.get("est")), _ci(r.get("ci_delta")), _ci(r.get("ci_fieller")) if r.get("fieller_bounded") else ("unbounded" if r.get("testable") else "–"), _ci(r.get("ci_boot")), r.get("reason") or ""])
    L += [_table(["Strategy", "Point", "R", "Delta CI", "Fieller CI", "Bootstrap CI", "Note"], rows)]
    L += ["### Persistence across capability (G-H2b, descriptive, D-033)", "", "The change in each strategy's headroom recovered R_x along the fitted line over the capability span, with its 95% session-clustered interval; the audit's gain on the logit scale is shown as a sensitivity (it drifts with capability under CM0's overflow rule).", ""]
    rows = [[s, r.get("estimand"), _f(r.get("est")), _ci(r.get("ci")), ", ".join(r.get("points_used") or []), r.get("reason") or ""] for s, r in g2["tost"].items()]
    rows += [[f"{s} (sensitivity)", r.get("estimand"), _f(r.get("est")), _ci(r.get("ci")), ", ".join(r.get("points_used") or []), r.get("reason") or ""] for s, r in g2.get("tost_sensitivity", {}).items()]
    L += [_table(["Strategy", "Estimand", "Change over span", "Interval", "Points", "Note"], rows)]
    sc = g2.get("short_control") or {}
    if sc.get("testable"):
        L += ["### Short-session control (N = 10)", "", f"At {sc['point']}: Gap(N long) {_pp(sc['gap_long']['est'])} vs Gap(N = 10) {_pp(sc['gap_short']['est'])}; difference {_pp(sc['difference']['est'])} {_ci(sc['difference']['ci'], _pp)}.", ""]
    rows = []
    for c, arms in (g2.get("crossing_split") or {}).items():
        for a, r in arms.items():
            rows.append([c, a, _pp(r["gain_pre"]["est"]), _pp(r["gain_post"]["est"]), _pp(r["post_minus_pre"]["est"]), _ci(r["post_minus_pre"]["ci"], _pp)])
    if rows:
        L += ["### Gain over CM0 before and after the W crossing", "", _table(["Point", "Arm", "Before", "After", "After − before", "Interval"], rows)]
    rows = []
    for r in g2.get("degradation") or []:
        lv, pos = r.get("log_view") or {}, r.get("position") or {}
        rows.append([r.get("block"), r.get("point"), r.get("arm"), r.get("items"), _f((lv.get("log_view") or {}).get("est")), _ci((lv.get("log_view") or {}).get("ci")), _f((pos.get("position") or {}).get("est")), _ci((pos.get("position") or {}).get("ci")), r.get("status")])
    L += ["### Degradation slopes (logit per log view token / per session fraction; overflowed items excluded)", "", _table(["Block", "Point", "Arm", "Items", "log view", "CI", "Position", "CI", "Status"], rows)]

    g3 = d["gh3"]
    L += ["## G-H3: decomposition", ""]
    if g3.get("testable"):
        L += [_table(["Test", "Estimate", "Interval", "p (t)", "p (flip)", "min flip p", "Reason"], [
            _test_row("M2 − S1 > 0", g3["pre"], _pp),
            _test_row(f"(M1 − S1) − {g3['thresholds']['isolation']} (M2 − S1) > 0", g3["h3a"], _pp),
            _test_row(f"(S-CM* − S1) − {g3['thresholds']['recovery']} (M2 − S1) > 0", g3["h3b"]["recovery"], _pp),
        ])]
        L += [_table(["Share", "Estimate", "Fieller CI", "Bootstrap CI"], [[n, _f(g3[k].get("est")), _ci(g3[k].get("ci_fieller")) if g3[k].get("fieller_bounded") else "unbounded", _ci(g3[k].get("ci_boot"))] for n, k in (("isolation", "isolation_share"), ("specialisation", "specialization_share"), ("S-CM* recovery", "recovery_share"))])]
        L += [_table(["Meter", "CPS ratio S-CM* / M2", "Interval", "Bootstrap", "p (≤ threshold)", "Reason"], [[m, _f(c.get("ratio")), _ci(c.get("ratio_ci")), _ci(c.get("ratio_boot_ci")), _p(c.get("p_t")), c.get("reason") or ""] for m, c in g3["costs"].items()])]
    else:
        L += [f"Not testable: {g3.get('reason')}", ""]

    L += ["## Cost per solved item (probes excluded)", ""]
    L += [_table(["Block", "Point", "Arm", "Solved", "$ / solved", "tokens / solved", "calls / solved", "s / solved", "probe tokens", "cm tokens"], [[c["block"], c["point"], c["arm"], _f(c["solved"], 0), _f(c.get("cost_usd_per_solved"), 4), _f(c.get("tokens_per_solved"), 0), _f(c.get("calls_per_solved"), 2), _f(c.get("wall_clock_per_solved"), 2), _f(c.get("tokens_probe"), 0), _f(c.get("tokens_cm"), 0)] for c in d["costs"]])]
    L += ["## Probes (F1 per checkpoint; a missed checkpoint scores 0)", ""]
    L += [_table(["Block", "Point", "Arm", "k", "Sessions", "F1", "95% CI", "Coverage"], [[p["block"], p["point"], p["arm"], p["k"], p["sessions"], _f(p["f1"]), _ci(p["f1_ci"]), _f(p["coverage"])] for p in d["probes"]["by_checkpoint"]])]
    if d["probes"]["behaviour"]:
        L += ["Probe F1 vs next dependency-item success:", "", _table(["Block", "Point", "Arm", "Pairs", "r"], [[p["block"], p["point"], p["arm"], p["pairs"], _f(p["r"])] for p in d["probes"]["behaviour"]])]
    L += ["## Failure taxonomy (labels per item)", ""]
    labs = gs.TAX_LABELS
    L += [_table(["Block", "Point", "Arm", "Items", "Failed", *labs], [[t["block"], t["point"], t["arm"], _f(t["items"], 0), _f(t["failed"], 0), *[_f(t.get(f"{lab}_per_item")) for lab in labs]] for t in d["taxonomy"]])]
    L += ["## Mixed models (descriptive, D-029)", ""]
    for name, m in (d.get("glmm") or {}).items():
        if m.get("status") == "ok":
            L += [f"**{name}:** `{m['formula']}` ({m['rows']} items).", "", _table(["Term", "Posterior mean", "SD"], [[k, _f(v["mean"]), _f(v["sd"])] for k, v in m["fixed"].items()])]
        else:
            L += [f"**{name}:** {m.get('status')} {m.get('reason') or ''}", ""]
    L += ["## Hypotheses", "", _table(["ID", "Role", "Family", "Estimand", "Test"], [[x["id"], x["role"], x["family"], x["estimand"], x["test"]] for x in d["hypotheses"]])]
    L += ["## Choices", ""] + [f"- {c}" for c in d["choices"]] + ["", "## Open for the pre-registration (B11)", ""] + [f"- {c}" for c in d["open_choices"]] + [""]
    return "\n".join(L)
