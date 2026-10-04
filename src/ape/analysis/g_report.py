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
  NOT_SUPPORTED, NOT_TESTED (an earlier step of the sequence failed), INCOMPLETE (a planned point missing or below its
  minimum sessions, or the plan unreadable: S-4, S-9) or NOT_TESTABLE (with the reason: missing arms, points,
  capability or sessions, or a cost that cannot be computed). The t-tests decide at the calibrated level
  (`g_hypotheses.t_level`, S-4). Descriptive rows are never decided; G-H1 and G-H2b (descriptive since D-033)
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

Nothing raises on missing data: every section reports what is there. Each section is also isolated (S-9): one that
raises is listed under `errors` (and in the report's Section errors), and the others, and the decisions, stand.
"""

import math
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from . import g_stats as gs
from .g_hypotheses import ALPHA, DEFAULT_REFERENCE, HYPOTHESES, MIN_SESSIONS_FLOOR, MIN_SESSIONS_SHARE, OPEN_CHOICES, REFERENCES, T_LEVEL, TIER_ORDER, TOST_ESTIMAND, TOST_MARGIN, GHypothesis, t_level
from .g_hypotheses import min_sessions as min_sessions_for

CHOICES = (
    "Sessions are the clusters (D-029): epochs of a session are pooled before any statistic; a world run at several "
    "capability points is one cluster across them.",
    "The primary test of every confirmatory row is the session-clustered t (unbiased variance with shared worlds as one "
    "cluster; Satterthwaite df, capped at the effective number of clusters − 1 when worlds are shared across points). "
    "The wild sign-flip (exact up to 16 clusters) is reported beside it; where it cannot reach α (fewer than 6 sessions "
    "at α = 0.025) the report says so.",
    f"S-4: the confirmatory t-tests decide at the calibrated one-sided level {T_LEVEL} (family α {ALPHA}), set so that "
    "every simulated null (the review's stress generators and g_power's) rejects at most α; a planned point needs "
    f"at least max({MIN_SESSIONS_FLOOR}, ⌈{MIN_SESSIONS_SHARE} × planned⌉) usable sessions (never more than planned), "
    "else it is missing and the family INCOMPLETE.",
    "S-9: each report section is isolated (a failure is listed, the decisions stand); a G-H3b clause that cannot be "
    "computed (a missing cost) is NOT_TESTABLE; an unreadable run plan makes G-H2a and G-H3 INCOMPLETE.",
    "A session's outcome is its item success rate; overflowed and never-reached items fail, an errored sample's "
    "unscored items fail (as in the gate), and the item in progress when a sample limit fired fails, as does the report "
    "(D-047). The binary all-items-and-report session success is reported with pass^k.",
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
    "D-043: G-H1 and G-H2b are also reported with measured capability replaced by the tier rank (Luna-low < Luna-high "
    "< Sol < Astra, equally spaced), a pre-registered sensitivity for an anchor near the ceiling.",
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
            tier = d["gh1"].get("tier_order") or {}
            row |= {
                "decision": _label(g1) if conf else "descriptive",
                "reference": g1.get("reference"),
                "estimate_span": g1.get("est_span"),
                "ci_span": g1.get("ci_span"),
                "estimate": g1.get("est"),
                "ci": g1.get("ci"),
                "sensitivity_S1_pre": {"estimate_span": sens.get("est_span"), "ci_span": sens.get("ci_span")},
                "tier_order": {r: {"estimate_span": x.get("est_span"), "ci_span": x.get("ci_span"), "points": x.get("points_used")} for r, x in tier.items() if r in (g1.get("reference"), "S1-pre")},
                "reason": g1.get("reason"),
            }
            if conf:
                row |= {"p": g1.get("p_t"), "p_flip": g1.get("p_flip")}
            out.append(row)
            continue
        if h.id == "G-H2b":
            for s, r in g2["tost"].items():
                label = ("NOT_TESTABLE" if not r["testable"] else ("EQUIVALENT" if r.get("holm_equivalent") else "NOT_SHOWN")) if conf else "descriptive"
                t = (g2.get("tost_tier_order") or {}).get(s) or {}
                out.append(row | {"id": f"G-H2b[{s}]", "decision": label, "estimate": r.get("est"), "ci": r.get("ci"), "tier_order": {"estimate": t.get("est"), "ci": t.get("ci")}, "reason": r.get("reason"), **({"margin": r.get("margin"), "p": r.get("p_t"), "holm_level": r.get("holm_level")} if conf else {})})
            continue
        if not conf:
            out.append(row | {"decision": "descriptive"})
            continue
        if d.get("planned_unknown"):
            row["planned_unknown"] = True
        if h.id == "G-H2a":
            gap = g2["gap"]
            failed = [c for c, r in gap["points"].items() if not r.get("reject")]
            label = "NOT_TESTABLE" if not gap["testable"] else ("INCOMPLETE" if gap.get("incomplete") else ("SUPPORTED" if gap["all_positive"] else "NOT_SUPPORTED"))
            row |= {"decision": label, "points_not_shown": failed, "missing_points": gap.get("missing_points") or [], "short_points": gap.get("short_points") or {}, "present_positive": gap.get("present_positive"), "flip_unreachable": gap["flip_unreachable"], "level": gap.get("level"), "reason": gap.get("reason")}
        elif h.id.startswith("G-H3"):
            if not g3.get("testable"):
                short = g3.get("short_points") or {}
                row |= {"decision": "INCOMPLETE" if short else "NOT_TESTABLE", "reason": g3.get("reason"), "missing_points": g3.get("missing_points") or [], "short_points": short}
            elif h.id == "G-H3-pre":
                row |= {"decision": _label(g3["pre"]), "estimate": g3["pre"]["est"], "ci": g3["pre"]["ci"], "p": g3["pre"]["p_t"]}
            elif h.id == "G-H3a":
                a = g3["h3a"]
                row |= {"decision": "NOT_TESTED" if not a["tested"] else ("SUPPORTED" if a["claim"] else "NOT_SUPPORTED"), "estimate": g3["isolation_share"].get("est"), "ci": g3["isolation_share"].get("ci_fieller"), "p": a["p_t"]}
            elif h.id == "G-H3b":
                b = g3["h3b"]
                # S-9: a clause that cannot be computed (a NaN cost) is NOT_TESTABLE, never NOT_SUPPORTED
                label = "NOT_TESTED" if not b["tested"] else ("NOT_TESTABLE" if b.get("testable") is False else ("SUPPORTED" if b["claim"] else "NOT_SUPPORTED"))
                row |= {"decision": label, "recovery": g3["recovery_share"].get("est"), "recovery_p": b["recovery"].get("p_t"), "cost_ratio": b["cost"].get("ratio"), "cost_ratio_ci": b["cost"].get("ratio_ci"), "cost_p": b["cost"].get("p_t"), "reason": b.get("reason") or b["cost"].get("reason")}
            if g3.get("testable") and g3.get("incomplete"):
                # A planned point without data: the decision is INCOMPLETE whatever the pooled tests show (the
                # estimates above, over the planned points present, are kept; PREREGISTRATION_G.md calls it a deviation).
                row |= {"decision": "INCOMPLETE", "decision_on_present_points": row["decision"], "missing_points": g3["missing_points"], "short_points": g3.get("short_points") or {}, "points": g3.get("points")}
            if g3.get("level") is not None:
                row["level"] = g3["level"]
        if d.get("planned_unknown") and row.get("decision") != "INCOMPLETE":
            # S-9: without the plan the points each family must cover are unknown; never decide over the points present
            row |= {"decision": "INCOMPLETE", "decision_on_present_points": row.get("decision"), "reason": "planned points unknown (the run plan could not be read)"}
        out.append(row)
    return out


def planned_points_and_sessions(hypotheses: Sequence[GHypothesis] = HYPOTHESES, plan=None) -> tuple[tuple[dict[str, list[str]] | None, dict[str, dict[str, int]] | None], str | None]:
    """((planned, minimum), problem). `planned`: the capability points each confirmatory family must cover
    (PREREGISTRATION_G.md restricts each claim to them): G-H2a's context-management cells and G-H3's topology cells,
    the enabled long-session ones, as `g_load.cell_info` resolves them (profile + effort). `minimum`: per family and
    point, the fewest usable sessions (S-4: `g_hypotheses.min_sessions` of the plan's sessions; a point below it is
    missing). Both None, with the problem, when the plan cannot be read: the report then labels the confirmatory rows
    INCOMPLETE rather than decide over the points present (S-9)."""
    from .g_load import cell_info

    if plan is None:
        from ..budget import load_plan

        try:
            plan = load_plan()
        except Exception as e:  # noqa: BLE001 - reported; the confirmatory rows are then INCOMPLETE
            return (None, None), f"planned points unknown (run plan unreadable: {type(e).__name__}: {e}); G-H2a and G-H3 are INCOMPLETE"
    out: dict[str, list[str]] = {}
    mins: dict[str, dict[str, int]] = {}
    for key, hid in (("G-H2a", "G-H2a"), ("G-H3", "G-H3-pre")):
        h = next((x for x in hypotheses if x.id == hid), None)
        if h is None:
            continue
        pts: list[str] = []
        for c in h.cells:
            ci = cell_info(c, plan)
            if ci.get("known") and ci.get("enabled", True) and ci.get("block") in ("cm", "topo") and ci.get("point") and int(ci.get("N") or 0) > gs.SHORT_N:
                pts.append(ci["point"])
                k = min_sessions_for(ci.get("sessions"))
                if k is not None:
                    mins.setdefault(key, {})[ci["point"]] = max(k, mins.get(key, {}).get(ci["point"], 0))
        out[key] = list(dict.fromkeys(pts))
    return (out, mins), None


def planned_points(hypotheses: Sequence[GHypothesis] = HYPOTHESES, plan=None) -> tuple[dict[str, list[str]], str | None]:
    """(planned, problem) of `planned_points_and_sessions`; {} when the plan cannot be read."""
    (planned, _), problem = planned_points_and_sessions(hypotheses, plan)
    return planned or {}, problem


def caveats(d: dict) -> list[str]:
    out = []
    if d.get("errors"):
        out.append(f"{len(d['errors'])} report section(s) failed ({', '.join(e['section'] for e in d['errors'])}); see Section errors. The other sections, and the decisions, stand.")
    if d.get("planned_unknown"):
        out.append("The run plan could not be read, so the planned points are unknown: G-H2a and G-H3 are INCOMPLETE (the tests over the points present are shown, nothing is SUPPORTED).")
    for name, x in (("G-H2a", d["gh2"]["gap"]), ("G-H3", d["gh3"])):
        short = x.get("short_points") or {}
        none = [c for c in x.get("missing_points") or [] if c not in short]
        if none:
            out.append(f"{name} is INCOMPLETE: planned point(s) {', '.join(none)} have no usable sessions (a deviation from PREREGISTRATION_G.md); the estimates over the planned points present are kept, but nothing is SUPPORTED.")
        if short:
            out.append(f"{name} is INCOMPLETE: planned point(s) " + ", ".join(f"{c} ({v['sessions']} usable sessions, minimum {v['minimum']})" for c, v in short.items()) + " are below the minimum sessions per planned point (S-4); they are left out of the decision, and nothing is SUPPORTED.")
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


PLANNED_UNKNOWN = {"unknown": True}  # `planned` when the plan cannot be read: the confirmatory rows are INCOMPLETE


def _safe(errors: list, name: str, fallback, fn, *args, **kw):
    """One section of the report; its exception is recorded (section, error, trace) and `fallback` stands in, so one
    section's failure never takes out the decisions (as `main_report._safe`)."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001  (one section's failure must not take the report down)
        errors.append({"section": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=4)})
        return fallback(f"section failed: {type(e).__name__}: {e}") if callable(fallback) else fallback


def _untestable(reason: str) -> dict:
    return {"testable": False, "reason": reason}


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
    plan=None,
    planned: Mapping[str, Sequence[str]] | None = None,
    min_sessions: Mapping[str, Mapping[str, int]] | None = None,
    t_level_: float | None = None,
) -> dict:
    """The Study G report as a JSON-serialisable dict (see the module docstring). `primary` decides every confirmatory
    test by the session-clustered t ("t", default) or the sign-flip ("flip"); both p-values are always reported.
    The confirmatory t-tests are decided at `t_level_` (default `g_hypotheses.t_level(alpha)`, the S-4 calibrated
    level). `planned` ({"G-H2a": points, "G-H3": points}) are the points each confirmatory family must cover and
    `min_sessions` ({family: {point: sessions}}) the fewest usable sessions each needs; default: both from `plan`
    (default the repository's run plan) via `planned_points_and_sessions`. `PLANNED_UNKNOWN` marks a plan that could
    not be read: the confirmatory rows are then INCOMPLETE (S-9). Each section is isolated: a failing one is listed
    under `errors` and leaves the others, and the decisions, standing (S-9)."""
    sessions = sessions if sessions is not None else pd.DataFrame()
    items = items if items is not None else pd.DataFrame()
    errors: list[dict] = []
    kw = {"alpha": alpha, "reps": reps, "seed": seed, "primary": primary}
    lvl = t_level(alpha) if t_level_ is None else float(t_level_)
    header_fallback = lambda r: {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "session_epochs": len(sessions), "items": len(items), "points": [], "arms": [], "counts": [], "cells": dict(cells or {}), "problems": list(problems)}  # noqa: E731
    d: dict = {"header": _safe(errors, "header", header_fallback, header_section, items, sessions, cells, problems), "alpha": alpha, "t_level": lvl, "errors": errors}
    unknown = bool(planned and planned.get("unknown"))
    if planned is None:
        (planned, min_sessions_plan), plan_problem = _safe(errors, "planned points", lambda r: ((None, None), f"planned points unknown ({r})"), planned_points_and_sessions, hypotheses, plan)
        min_sessions = min_sessions if min_sessions is not None else min_sessions_plan
        if plan_problem:
            d["header"]["problems"] = [*d["header"]["problems"], plan_problem]
            unknown = planned is None
    if unknown:
        planned = {}
        if not any("planned points unknown" in p for p in d["header"]["problems"]):
            d["header"]["problems"] = [*d["header"]["problems"], "planned points unknown (the run plan could not be read): G-H2a and G-H3 are INCOMPLETE"]
    d["planned_points"] = dict(planned or {})
    d["planned_unknown"] = unknown
    d["min_sessions"] = {k: dict(v) for k, v in (min_sessions or {}).items()}
    d["capability"] = _safe(errors, "capability", lambda r: {"points": [], "order": [], "missing": [], "not_separated": [], "error": r}, capability_section, capability, sessions)
    d["outcomes"] = _safe(errors, "outcomes", [], gs.outcome_table, sessions)

    def gh1_all() -> dict:
        return {
            "primary": gs.gh1(sessions, capability, reference=reference, boot=boot, **kw),
            "references": {r: gs.gh1(sessions, capability, reference=r, boot=boot, **kw) for r in REFERENCES},
            "references_prob": {r: gs.gh1(sessions, capability, reference=r, scale="prob", **kw) for r in REFERENCES},
            "M1": gs.gh1(sessions, capability, topology="M1", reference=reference, **kw),
            # D-043: capability replaced by the tier rank (equally spaced), every reference
            "tier_order": {r: gs.gh1(sessions, TIER_ORDER, reference=r, boot=boot, **kw) for r in REFERENCES},
        }

    d["gh1"] = _safe(errors, "G-H1", lambda r: {"primary": _untestable(r) | {"reference": reference}, "references": {}, "references_prob": {}, "M1": _untestable(r), "tier_order": {}}, gh1_all)
    margin = tost_margin if tost_margin is not None else TOST_MARGIN[tost_estimand]
    gap_fallback = lambda r: {"gap": {"points": {}, "testable": False, "reason": r, "flip_unreachable": [], "missing_points": [], "incomplete": False, "all_positive": False}, "headroom": {}, "tost": {}}  # noqa: E731
    d["gh2"] = _safe(errors, "G-H2", gap_fallback, gs.gh2, sessions, capability, estimand=tost_estimand, margin=margin, boot=boot, planned_points=(planned or {}).get("G-H2a"), min_sessions=(min_sessions or {}).get("G-H2a"), level=lvl, **kw)
    other = "gain" if tost_estimand == "R" else "R"
    d["gh2"]["tost_sensitivity"] = _safe(errors, "G-H2b sensitivity", {}, lambda: {s: gs.tost(sessions, capability, s, estimand=other, **kw) for s in d["gh2"]["tost"]})
    d["gh2"]["tost_tier_order"] = _safe(errors, "G-H2b tier order", {}, lambda: {s: gs.tost(sessions, TIER_ORDER, s, estimand=tost_estimand, margin=margin, **kw) for s in d["gh2"]["tost"]})  # D-043
    d["gh2"]["short_control"] = _safe(errors, "short-session control", _untestable, gs.short_control, sessions, alpha=alpha, reps=reps, seed=seed)
    d["gh2"]["crossing_split"] = _safe(errors, "crossing split", {}, gs.crossing_split, sessions, alpha=alpha, seed=seed)
    d["gh2"]["degradation"] = _safe(errors, "degradation", [], lambda: gs.degradation(items) if len(items) and "block" in items else [])
    d["gh3"] = _safe(errors, "G-H3", _untestable, gs.gh3, sessions, boot=boot, planned_points=(planned or {}).get("G-H3"), min_sessions=(min_sessions or {}).get("G-H3"), level=lvl, **kw)
    d["costs"] = _safe(errors, "costs", [], gs.cost_table, sessions)
    d["probes"] = {"by_checkpoint": _safe(errors, "probes", [], gs.probe_table, sessions), "behaviour": _safe(errors, "probe behaviour", [], gs.probe_behaviour, sessions, items)}
    d["taxonomy"] = _safe(errors, "taxonomy", [], gs.taxonomy_table, sessions)
    d["glmm"] = {
        "topology": _safe(errors, "GLMM topology", lambda r: {"status": "failed", "reason": r}, gs.glmm, items, capability, block="topo", reference="S1"),
        "context_management": _safe(errors, "GLMM context management", lambda r: {"status": "failed", "reason": r}, gs.glmm, items, capability, block="cm", arms=("CM0", "CM-sum", "CM-todo", "O-state"), reference="CM0"),
    } if glmm else {}
    d["decisions"] = _safe(errors, "decisions", lambda r: [{"id": h.id, "role": h.role, "family": h.family, "step": h.step, "decision": "NOT_TESTABLE" if h.role == "confirmatory" else "descriptive", "reason": r} for h in hypotheses], decisions, d, hypotheses)
    d["hypotheses"] = [h.to_dict() for h in hypotheses]
    d["choices"] = list(CHOICES)
    d["open_choices"] = list(OPEN_CHOICES)
    d["caveats"] = _safe(errors, "caveats", [], caveats, d)
    return _clean(d)


def report_from_cells(cells: Mapping[str, Sequence], *, plan=None, points: Mapping[str, str] | None = None, prices: Mapping | None = None, capability: Mapping[str, float] | None = None, **kw) -> dict:
    """`g_report` on {plan_cell: [log paths]} (session cells and g.cap.* anchor cells)."""
    from .g_load import load_g_cells

    data = load_g_cells(cells, plan=plan, points=points, prices=prices, capability=capability)
    return g_report(data.items, data.sessions, data.capability, cells=data.cells, problems=data.problems, **kw)


# ---------- markdown ----------


def _test_row(name: str, r: dict, fmt=_f) -> list:
    return [name, fmt(r.get("est")), _ci(r.get("ci"), fmt), _p(r.get("p_t")), _p(r.get("p_flip")), _f(r.get("flip_min_p"), 4), r.get("reason") or ""]


def _render_safe(L: list, name: str, fn) -> None:
    """Render one section; a failure leaves a note in its place instead of taking the report down (S-9)."""
    try:
        fn(L)
    except Exception as e:  # noqa: BLE001
        L += [f"*Section {name} could not be rendered: {type(e).__name__}: {e}*", ""]


def render(d: dict) -> str:
    h = d["header"]
    L = ["# Study G report", "", f"Generated {h['generated_at']}. {h['session_epochs']} session-epochs, {h['items']} items; points {', '.join(h['points']) or '–'}; arms {', '.join(h['arms']) or '–'}. One-sided α = {d['alpha']} per family; confirmatory t-tests at the calibrated level {d.get('t_level', d['alpha'])} (S-4).", ""]
    if d["caveats"]:
        L += [f"> **Caveat:** {c}" for c in d["caveats"]] + [""]

    if d.get("errors"):
        L += ["## Section errors", "", "Each failed section was skipped; the others, and the decisions, stand.", "", _table(["Section", "Error"], [[e["section"], e["error"]] for e in d["errors"]])]

    def _s0(L: list) -> None:
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
            short = r.get("short_points") or {}
            if r.get("planned_unknown"):
                extra = "planned points unknown (run plan unreadable)" + (f"; on the points present: {r['decision_on_present_points']}" if r.get("decision_on_present_points") else "") + (f"; {extra}" if extra else "")
            elif r.get("missing_points"):
                none = [c for c in r["missing_points"] if c not in short]
                parts = ([f"planned point(s) without data: {', '.join(none)}"] if none else []) + ([f"below the minimum sessions: {', '.join(f'{c} ({v['sessions']} of {v['minimum']})' for c, v in short.items())}"] if short else [])
                extra = "; ".join(parts) + (f"; on the points present: {r['decision_on_present_points']}" if r.get("decision_on_present_points") else (f"; positive at every point present: {_f(r.get('present_positive'))}" if r.get("present_positive") is not None else "")) + (f"; {extra}" if extra else "")
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
                for ref, x in (r.get("tier_order") or {}).items():
                    rows.append([f"G-H1 (M2 − {ref}), tier order (D-043)", "change in the logit gap over the tier ranks spanned", _f(x.get("estimate_span")), _ci(x.get("ci_span")), ", ".join(x.get("points") or [])])
            elif r["id"].startswith("G-H2b"):
                rows.append([r["id"], "change in R_x over the capability span", _f(r.get("estimate")), _ci(r.get("ci")), r.get("reason") or ""])
                t = r.get("tier_order") or {}
                rows.append([f"{r['id']}, tier order (D-043)", "change in R_x over the tier ranks spanned", _f(t.get("estimate")), _ci(t.get("ci")), ""])
        if rows:
            L += ["Descriptive estimates (D-033; 95% intervals, no decision):", "", _table(["Row", "Estimand", "Estimate", "Interval", "Note"], rows)]

    _render_safe(L, 'decisions', _s0)

    def _s1(L: list) -> None:
        cap = d["capability"]
        L += ["## Capability (S1 on F7-10 + F3-5)", ""]
        L += [_table(["Point", "Capability", "SE", "Tasks", "Source"], [[p.get("point"), _f(p.get("capability")), _f(p.get("se")), p.get("n_tasks", "–"), p.get("source", "anchor")] for p in cap["points"]])]

    _render_safe(L, 'capability', _s1)

    def _s2(L: list) -> None:
        L += ["## Outcomes by arm and point", ""]
        L += [_table(["Block", "Point", "Arm", "N", "Sessions", "Item success", "95% CI", "Dependency", "Report exact", "Session success", "pass^k", "Overflow", "Errors"],
                     [[o["block"], o["point"], o["arm"], o["N"], o["sessions"], _f(o["item_success"]), _ci(o["item_success_ci"]), _f(o["dependency_success"]), _f(o["report_exact"]), _f(o["session_success"]), _f(o["pass_k"]), _f(o["overflow_rate"]), o["errors"]] for o in d["outcomes"]])]

    _render_safe(L, 'outcomes', _s2)

    def _s3(L: list) -> None:
        g1 = d["gh1"]
        L += [
            "## G-H1: topology gap vs capability (descriptive, D-033)",
            "",
            f"Reference **{g1['primary'].get('reference')}**; S1-pre (both arms on items before S1's overflow) is the sensitivity analysis; S1 as scored is "
            "reported only descriptively (its gap carries the overflow rule, which drifts with capability). Slope per unit measured capability and the "
            "change along the fitted line over the capability span, with 95% session-clustered and bootstrap intervals; the D-043 sensitivity "
            "repeats each with capability replaced by the tier rank (Luna-low < Luna-high < Sol < Astra, equally spaced), so its slope is per tier step.",
            "",
        ]

        def g1_row(name: str, x: dict, boot: bool = True) -> list:
            return [name, _f(x.get("est")), _ci(x.get("ci")), _f(x.get("est_span")), _ci(x.get("ci_span")), _ci(x.get("boot_ci_span")) if boot else "–", ", ".join(x.get("points_used") or []), x.get("reason") or ""]

        rows = [g1_row(f"M2 − {r} (logit)", x) for r, x in g1["references"].items()]
        rows += [g1_row(f"M2 − {r} (prob)", x, boot=False) for r, x in g1["references_prob"].items()]
        rows += [g1_row(f"M1 − {g1['M1'].get('reference')} (logit)", g1["M1"], boot=False)]
        rows += [g1_row(f"M2 − {r} (logit), tier order (D-043)", x) for r, x in (g1.get("tier_order") or {}).items()]
        L += [_table(["Contrast", "Slope", "Interval", "Change over span", "Interval", "Bootstrap", "Points", "Note"], rows)]
        gap_rows = [[r, c, _f(x["est"]), _ci(x["ci"]), x["n"]] for r, xx in g1["references"].items() for c, x in xx.get("gaps", {}).items()]
        L += ["Per-point gaps (logit):", "", _table(["Reference", "Point", "Gap", "Interval", "Sessions"], gap_rows)]

    _render_safe(L, 'G-H1', _s3)

    def _s4(L: list) -> None:
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
        rows += [[f"{s}, tier order (D-043)", r.get("estimand"), _f(r.get("est")), _ci(r.get("ci")), ", ".join(r.get("points_used") or []), r.get("reason") or ""] for s, r in g2.get("tost_tier_order", {}).items()]
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

    _render_safe(L, 'G-H2', _s4)

    def _s5(L: list) -> None:
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

    _render_safe(L, 'G-H3', _s5)

    def _s6(L: list) -> None:
        L += ["## Cost per solved item (probes excluded)", ""]
        L += [_table(["Block", "Point", "Arm", "N", "Solved", "$ / solved", "tokens / solved", "calls / solved", "s / solved", "probe tokens", "cm tokens"], [[c["block"], c["point"], c["arm"], c.get("N", "–"), _f(c["solved"], 0), _f(c.get("cost_usd_per_solved"), 4), _f(c.get("tokens_per_solved"), 0), _f(c.get("calls_per_solved"), 2), _f(c.get("wall_clock_per_solved"), 2), _f(c.get("tokens_probe"), 0), _f(c.get("tokens_cm"), 0)] for c in d["costs"]])]
    _render_safe(L, 'costs', _s6)

    def _s7(L: list) -> None:
        L += ["## Probes (F1 per checkpoint; a missed checkpoint scores 0)", ""]
        L += [_table(["Block", "Point", "Arm", "N", "k", "Sessions", "F1", "95% CI", "Coverage"], [[p["block"], p["point"], p["arm"], p.get("N", "–"), p["k"], p["sessions"], _f(p["f1"]), _ci(p["f1_ci"]), _f(p["coverage"])] for p in d["probes"]["by_checkpoint"]])]
        if d["probes"]["behaviour"]:
            L += ["Probe F1 vs next dependency-item success:", "", _table(["Block", "Point", "Arm", "N", "Pairs", "r"], [[p["block"], p["point"], p["arm"], p.get("N", "–"), p["pairs"], _f(p["r"])] for p in d["probes"]["behaviour"]])]
    _render_safe(L, 'probes', _s7)

    def _s8(L: list) -> None:
        L += ["## Failure taxonomy (labels per item)", ""]
        labs = gs.TAX_LABELS
        L += [_table(["Block", "Point", "Arm", "N", "Items", "Failed", *labs], [[t["block"], t["point"], t["arm"], t.get("N", "–"), _f(t["items"], 0), _f(t["failed"], 0), *[_f(t.get(f"{lab}_per_item")) for lab in labs]] for t in d["taxonomy"]])]
    _render_safe(L, 'taxonomy', _s8)

    def _s9(L: list) -> None:
        L += ["## Mixed models (descriptive, D-029)", ""]
        for name, m in (d.get("glmm") or {}).items():
            if m.get("status") == "ok":
                L += [f"**{name}:** `{m['formula']}` ({m['rows']} items).", "", _table(["Term", "Posterior mean", "SD"], [[k, _f(v["mean"]), _f(v["sd"])] for k, v in m["fixed"].items()])]
            else:
                L += [f"**{name}:** {m.get('status')} {m.get('reason') or ''}", ""]
    _render_safe(L, 'GLMM', _s9)

    def _s10(L: list) -> None:
        L += ["## Hypotheses", "", _table(["ID", "Role", "Family", "Estimand", "Test"], [[x["id"], x["role"], x["family"], x["estimand"], x["test"]] for x in d["hypotheses"]])]
        L += ["## Choices", ""] + [f"- {c}" for c in d["choices"]] + ["", "## Open for the pre-registration (B11)", ""] + [f"- {c}" for c in d["open_choices"]] + [""]
    _render_safe(L, 'hypotheses', _s10)

    return "\n".join(L)
