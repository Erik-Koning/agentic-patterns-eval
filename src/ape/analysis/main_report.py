"""Main-study report (BUILD_PLAN B4): every pre-registered test, the descriptive estimators and the data checks, from
the tidy frame, as one JSON-serialisable dict and a markdown rendering (the style of `analyze_gate`).

    frame, coverage = main_load.load_plan_cells({plan_cell: [log paths]})
    d = main_report(frame, coverage=coverage)
    md = render_main(d)

or `report_from_logs({plan_cell: [paths]})`. The run-directory glue (an `ape.analyze_main` reading a study run's
manifests) lands with the study runner (B1); this module only needs the logs per plan cell.

Every section is computed whatever the others do: a missing arm or cell makes a member "not evaluable" with the
reason, and an unexpected failure in one section is recorded under `errors` while the rest of the report stands.
Where the brief and D-029 leave a choice open, it is listed in CHOICES (and in the report).
"""

import json
import math
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import main_descriptive as desc
from .gate_stats import harness_rates
from .main_hypotheses import HYPOTHESES, PRIMARY_METER, PRIMARY_TIER, TEST_TEXT, hypothesis_table
from .main_load import METERS, load_plan_cells
from .main_stats import DETERMINISM_PREFIX, EXACT_MAX_CLUSTERS, REPS, S1_EPOCHS, build_tables, confirmatory_rows, evaluate_family

CHOICES = (
    "Decision: the world-clustered sign-flip test on the per-task contrast (exact over all 2^G flips up to "
    f"{EXACT_MAX_CLUSTERS} clusters, else 10,000 Monte Carlo flips); NI and TOST shift each cluster by the margin. "
    "Simulated at the planned 9 worlds per cell it holds α (single-cell superiority 2.6% at 2.5%; every family's null "
    "boundary in `main_power`), where the gate's cluster-t ran at 3.1-3.3%; cluster-t is reported as a cross-check.",
    "Headline interval: the sign-flip test inverted (consistent with the decision; 94-96% coverage at a nominal 95% with 9 "
    "worlds in simulation). The brief's world-clustered BCa bootstrap is reported too, but with 9 clusters it covered "
    "only 88% (percentile: 88%), so no decision rests on it. BCa over percentile because skewed success rates near 0/1 "
    "and ratio estimators bias the bootstrap distribution, which BCa corrects; neither fixes the few-cluster undercoverage.",
    "Clusters: F1 and F2 worlds of one seed share one supplier registry (one KB), so with cluster_by='kb' they are one "
    "cluster across Study A's cells; F3 and F7 worlds are their own clusters. Pooled members weight their cells equally.",
    "S1's task means use its first 3 pool epochs (the 3 epochs every arm has); the S8 frontier uses all 8. Study C plan "
    f"cells ({DETERMINISM_PREFIX}*) feed only the determinism section.",
    "S8 frontier: plurality of canonical answer keys over each k-subset of the S1 pool (exhaustive), abstentions cast no "
    "vote, ties count as the expected success of a random tied answer; S8(k) costs k mean runs on every meter "
    "('sum'; wall-clock may be priced with parallel members, 'max'). An arm is matched at its realised mean cost per "
    "sample in the cell, linear between adjacent k; below one run it meets S8(1) (flagged), above S8(K) it is beyond the "
    "frontier (named, not tested; a pooled member leaves that cell out). The live S8k3 may break ties differently.",
    f"Frontier meter for the confirmatory tests: {PRIMARY_METER} (the cap-enforcement meter, brief §4.5); every meter is "
    "reported in the frontier table.",
    "Holm within each family (the gate's holm_test, stopping at the first non-rejection); M2 is serial gatekeeping (M1 > S1 "
    "first, then M1 > S8 at matched cost). Superiority, NI and `less` families at one-sided α = 0.025; TOST families at "
    "0.05 per one-sided test (the 90% interval). No correction across families (brief §7.4).",
    "H2's NI members (S5 ≥ M2 − 3 pp, per family F3 and F7, each pooling its two cells) also need S5/M2's realised cost "
    "ratio ≤ 0.5 with the upper end of its 95% world-clustered BCa interval ≤ 0.6 (intersection-union: no extra α).",
    "H1a's and H1b's F2 clauses have no margin in the brief: reported descriptively.",
    "H6: the brief's coordination payoff M1 − S8 (matched cost) from 3-run frontiers in both tiers, Sol minus Luna, "
    "one-sided `less`, per cell (F1-32, F7-100) with Holm.",
    "Rank flips (H3): cost per solved task (mean cost / mean success, cells equally weighted, on the tasks every ranked arm "
    "ran) under tokens, cache-adjusted $ and wall-clock (Inspect working time); a family flips when its smallest pairwise "
    "Kendall τ-b is below 0.8; the rule needs 2 of the 4 families. World-clustered bootstrap support is reported.",
    "Determinism (H5): each arm's first 5 runs on Study C's tasks (main epochs, then Study C's 2); pass^k by the unbiased "
    "estimator; D2 counts 'no answer' as an answer. Descriptive: the auditability metrics are not produced.",
    "Errored samples are failures (success 0) and abstentions in the S8 vote; harness error and cap-hit rates are reported "
    "per arm and cell with the gate's Clopper–Pearson bounds.",
    "The GLMM is descriptive (D-029): statsmodels BinomialBayesMixedGLM (variational Bayes); a failed or unconverged fit is "
    "reported, never fatal.",
)


# --- helpers ---------------------------------------------------------------------------------------


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN/inf to None, tuples to lists, keys to strings."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return _clean(x.tolist())
    if isinstance(x, pd.DataFrame):
        return _clean(x.to_dict(orient="records"))
    return x


def _safe(errors: list, name: str, fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001  (one section's failure must not take the report down)
        errors.append({"section": name, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=4)})
        return None


def _design(rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []
    g = rows.groupby(["tier", "cell", "arm"])
    out = g.agg(samples=("success", "size"), tasks=("task", "nunique"), worlds=("world", "nunique"), epochs=("epoch", "nunique"), errors=("error", "sum"), success=("success", "mean")).reset_index()
    if "plan_cell" in rows.columns:
        pcs = g["plan_cell"].agg(lambda s: sorted({str(x) for x in s.dropna()})).reset_index(drop=True)
        out["plan_cells"] = pcs
    return out.to_dict(orient="records")


# --- the report ------------------------------------------------------------------------------------


def main_report(
    frame: pd.DataFrame,
    hypotheses=HYPOTHESES,
    alpha: float | None = None,
    reps: int = REPS,
    seed: int = 0,
    meters: dict | None = None,
    s1_epochs: int = S1_EPOCHS,
    cluster_by: str = "kb",
    wall_aggregate: str = "sum",
    rank_by: str = "cost_per_success",
    glmm: bool = True,
    coverage: Sequence[dict] | None = None,
    frontier_reps: int = 2000,
) -> dict:
    """Every section of the main-study report as one JSON-serialisable dict (module docstring). `alpha` overrides every
    family's one-sided α (default: the table's); `reps` sets the bootstrap and Monte Carlo sign-flip resamples."""
    meters = dict(METERS if meters is None else meters)
    errors: list = []
    frame = frame.copy()
    if "tier" not in frame.columns:
        frame["tier"] = PRIMARY_TIER
    for col, default in (("error", False), ("cap_hit", False), ("answer_key", None), ("plan_cell", None)):
        if col not in frame.columns:
            frame[col] = default
    rows, notes = confirmatory_rows(frame)
    tables = build_tables(frame, meters, s1_epochs, cluster_by, wall_aggregate)
    d: dict = {
        "header": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": int(len(frame)),
            "confirmatory_rows": notes | {"nan_success_rows": tables.notes.get("nan_success_rows", 0)},
            "plan_cells": sorted({str(p) for p in frame["plan_cell"].dropna()}),
            "tiers": sorted(tables.tm),
            "settings": {"alpha_override": alpha, "reps": reps, "seed": seed, "meters": meters, "s1_epochs": s1_epochs, "cluster_by": cluster_by, "wall_aggregate": wall_aggregate, "rank_by": rank_by, "primary_meter": PRIMARY_METER, "primary_tier": PRIMARY_TIER},
            "coverage": list(coverage or []),
            "design": _safe(errors, "design", _design, rows) or [],
        },
        "hypothesis_table": hypothesis_table(hypotheses),
    }
    fams = {}
    for i, h in enumerate(hypotheses):
        if h.members:
            fams[h.id] = _safe(errors, f"hypothesis {h.id}", evaluate_family, tables, h, alpha, reps, seed + 101 * i)
    d["hypotheses"] = fams
    d["summary"] = [
        {"family": fid, "member": mid, "contrast": m.get("contrast"), "cells": m.get("cells") or m.get("cells_planned"), "test": m.get("test"), "margin": m.get("margin"), "est": m.get("est"), "ci": m.get("ci"), "p": m.get("p"), "holm_level": m.get("holm_level"), "label": m.get("label"), "reason": m.get("reason") or m.get("why")}
        for fid, f in fams.items()
        if f and f.get("status") == "confirmatory"
        for mid, m in f["members"].items()
    ]
    d["descriptive"] = {
        "K4": _safe(errors, "K4", desc.k4_slopes, tables, reps=reps, seed=seed),
        "M4": _safe(errors, "M4", desc.m4, tables, reps=reps, seed=seed),
        "mechanisms": _safe(errors, "mechanisms", desc.mechanism_contrasts, tables, reps=reps, seed=seed),
        "tiers": _safe(errors, "tier contrasts", desc.tier_contrasts, tables, reps=reps, seed=seed),
    }
    d["arms"] = _safe(errors, "arms", desc.arm_summary, tables)
    d["frontier"] = _safe(errors, "frontier", desc.frontier_table, tables, reps=frontier_reps, seed=seed)
    d["rank_flips"] = _safe(errors, "rank flips", desc.rank_flips, tables, rank_by=rank_by, reps=min(reps, 2000), seed=seed)
    d["determinism"] = _safe(errors, "determinism", desc.determinism, frame, meters=tuple(m for m in ("tokens", "usd", "wall") if m in meters), cluster_by=cluster_by, reps=reps, seed=seed)
    d["invariants"] = _safe(errors, "invariants", desc.invariants, tables, reps=reps, seed=seed)
    d["s8k3"] = _safe(errors, "S8k3", desc.s8k3_check, frame, tables, reps=reps, seed=seed)
    d["harness"] = _safe(errors, "harness", harness_rates, rows.assign(error=rows["error"].fillna(False), cap_hit=rows["cap_hit"].fillna(False)), ("tier", "arm", "cell")) if len(rows) else []
    d["glmm"] = _safe(errors, "GLMM", desc.glmm, frame, s1_epochs=s1_epochs) if glmm else {"available": False, "reason": "not requested"}
    d["choices"] = list(CHOICES)
    d["errors"] = errors
    return _clean(d)


def report_from_logs(cells: Mapping[str, Sequence[str | Path]], require_cost: bool = True, **kw) -> dict:
    """{plan_cell: [log paths]} -> `main_report`, with each plan cell's coverage in the header."""
    frame, coverage = load_plan_cells(cells, require_cost=require_cost)
    return main_report(frame, coverage=coverage, **kw)


# --- markdown --------------------------------------------------------------------------------------


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


def _ci(ci: Any) -> str:
    return "–" if not ci or ci[0] is None or ci[1] is None else f"[{_pp(ci[0])}, {_pp(ci[1])}]"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def render_main(d: dict) -> str:
    h = d["header"]
    L = ["# Main-study report", "", f"Generated {h['generated_at']}; {h['rows']:,} sample-epochs ({h['confirmatory_rows'].get('used', 0):,} in the confirmatory frame); tiers {', '.join(h['tiers']) or '–'}.", ""]
    if d.get("errors"):
        L += [f"> **Section failed:** {e['section']}: {e['error']}" for e in d["errors"]] + [""]
    if h.get("coverage"):
        bad = [c for c in h["coverage"] if c.get("reason")]
        L += [f"- Plan cells: {len(h['coverage'])}, missing or unreadable: {len(bad)}" + (": " + "; ".join(f"{c['plan_cell']} ({c['reason']})" for c in bad) if bad else ".")]
    L += ["", "## Confirmatory tests", "", "Interval: the sign-flip test inverted (two-sided 1 − 2α). Labels: supported / not supported (never 'no effect') / not tested (Holm stopped or the gate stage failed) / not evaluable (data missing).", ""]
    L += [_table(["Member", "Contrast", "Cells", "Test", "Δ", "Interval", "p", "Holm level", "Result"], [[s["member"], s["contrast"], ", ".join(s["cells"] or []), TEST_TEXT.get(s["test"], s["test"]) + (f", margin {100 * s['margin']:g} pp" if s.get("margin") else ""), _pp(s["est"]), _ci(s["ci"]), _f(s["p"], 4), _f(s["holm_level"], 4), (s["label"] or "–") + (f" ({s['reason']})" if s.get("reason") else "")] for s in d["summary"]])]

    L += ["## Hypothesis families", ""]
    for fid, f in (d.get("hypotheses") or {}).items():
        if not f:
            continue
        L += [f"### {fid} ({f['source']}; {f['status']})", "", f["statement"], ""]
        if f.get("status") == "confirmatory":
            L += [f"α = {f['alpha']} one-sided, {f['procedure']}; frontier meter {f['meter']}." + (f" {f['note']}" if f.get("note") else ""), ""]
        rows = []
        for mid, m in f["members"].items():
            cr = m.get("cost_ratio")
            rows.append([mid, _pp(m.get("est")), _ci(m.get("ci")), _ci(m.get("ci_boot")), _ci(m.get("ci_t")), _f(m.get("p"), 4), _f(m.get("p_t"), 4), f"{m.get('clusters', '–')} / {m.get('tasks', '–')}", m.get("method", "–"), (f"{_f(cr.get('ratio'), 2)} [{_f(cr['ci'][0], 2)}, {_f(cr['ci'][1], 2)}] {'ok' if cr.get('ok') else 'fails'}" if cr else "–"), m.get("label") or ("–" if m.get("evaluable") else f"not evaluable: {m.get('reason')}")])
        L += [_table(["Member", "Δ", "Interval (flip)", "BCa bootstrap", "Cluster-t", "p (flip)", "p (t)", "Clusters / tasks", "Method", "Cost ratio", "Result"], rows)]
        for mid, m in f["members"].items():
            miss = (m.get("coverage") or {}).get("missing_cells") or []
            if miss:
                L += [f"- {mid}: cells left out: " + "; ".join(f"{c} ({m['coverage']['cells'][c].get('reason')})" for c in miss)]
        L += [""]

    desc_ = d.get("descriptive") or {}
    k4 = desc_.get("K4") or {}
    L += ["## Descriptive", "", "### K4: KB scaling (slope per decade of KB size)", ""]
    rows = [[name, _pp(x.get("slope")), _ci(x.get("ci_t")), _ci(x.get("ci_boot")), ", ".join(x.get("cells") or [])] if x.get("estimable") else [name, "–", "–", "–", x.get("reason")] for part in ("arms", "differences") for name, x in (k4.get(part) or {}).items()]
    L += [_table(["Arm / difference", "Slope", "Cluster-t 95%", "BCa 95%", "Cells"], rows)]
    m4 = desc_.get("M4") or {}
    if m4:
        acc, w = m4.get("accuracy") or {}, m4.get("wall_ratio") or {}
        L += ["### M4: concurrency (F1-32)", "", f"- M1 − M1s accuracy {_pp(acc.get('est'))}, 95% {_ci(acc.get('ci'))}, 90% {_ci((m4.get('accuracy_90') or {}).get('ci'))}.", f"- Wall-clock ratio M1/M1s {_f(w.get('ratio'), 2)} [{_f((w.get('ci') or [None])[0], 2)}, {_f((w.get('ci') or [None, None])[1], 2)}] (brief: < 0.6)." + (f" {w.get('reason')}" if w.get("reason") else ""), ""]
    mech = desc_.get("mechanisms") or []
    L += ["### Mechanism contrasts (§4.3)", "", _table(["Mechanism", "Contrast", "Cell", "Δ", "Interval (flip)", "BCa 95%", "Tasks"], [[m["mechanism"], m["contrast"], m["cell"], _pp(m.get("est")), _ci(m.get("ci")), _ci(m.get("ci_boot")), m.get("tasks")] for m in mech])]
    tiers = desc_.get("tiers") or []

    def _cell(x: Any) -> str:
        return "–" if not x else f"{_pp(x.get('est'))} {_ci(x.get('ci'))}"

    L += ["### Arm × tier (Study F, same tasks)", "", _table(["Cell", "Contrast", "Luna", "Sol", "Sol − Luna"], [[t["cell"], t["contrast"], _cell(t.get("luna")), _cell(t.get("sol")), _cell(t.get("sol − luna"))] for t in tiers])]

    L += ["## S8 frontier (primary meter)", ""]
    for row in d.get("frontier") or []:
        if row["meter"] != PRIMARY_METER:
            continue
        pts = ", ".join(f"k={p['k']}: {_f(p['success'])} @ {_f(p['cost'], 1)}" for p in row["points"])
        L += [f"**{row['cell']} ({row['tier']})**, K = {row['K']}{', inconsistent keys: ' + str(row['inconsistent_keys']) if row.get('inconsistent_keys') else ''}: {pts}", ""]
        L += [_table(["Arm", "Cost", "× S1 run", "Match", "Arm − S8", "Interval (flip)"], [[a, _f(x.get("cost"), 1), _f(x.get("cost_ratio_to_s1_run"), 2), ((x.get("match") or {}).get("status") or x.get("reason") or "–") + (f" k={x['match']['k']}, λ={x['match']['lambda']:.2f}" if (x.get("match") or {}).get("status") in ("inside", "below") else ""), _pp(x.get("est")), _ci(x.get("ci"))] for a, x in row["arms"].items()])]

    rf = d.get("rank_flips") or {}
    L += ["## Cost-meter rank flips (H3)", "", f"Ranking by {rf.get('rank_by')}; rule: min pairwise Kendall τ < {rf.get('threshold')} in ≥ {rf.get('min_families')} families → **{rf.get('rule_holds')}** ({rf.get('families_flipping')} of {rf.get('families_ranked')}; bootstrap support {_f(rf.get('rule_bootstrap_support'))}).", ""]
    L += [_table(["Family", "Arms", "τ by meter pair", "min τ", "Flip", "P(flip) bootstrap"], [[fam, ", ".join(x.get("arms") or []), "; ".join(f"{k} {_f(v, 2)}" for k, v in (x.get("tau") or {}).items()), _f(x.get("min_tau"), 2), _f(x.get("flip")), _f(x.get("p_flip_bootstrap"))] for fam, x in (rf.get("families") or {}).items()])]

    det = d.get("determinism") or {}
    L += ["## Determinism (H5, Study C subset)", ""]
    if det.get("available"):
        L += [_table(["Arm / cell / tier", "Tasks", "Runs", "Success", "pass^k (k = runs)", "D1", "D2 modal", "D2 entropy"], [[k, x["tasks"], x["runs_min"], _f(x["success"]), _f((x.get("pass_hat_k") or {}).get(str(x["runs_min"]), (x.get("pass_hat_k") or {}).get(x["runs_min"]))), _f(x["d1_outcome_stability"]), _f(x["d2_modal_share"]), _f(x["d2_entropy"])] for k, x in det["arms"].items()])]
        L += [_table(["Tier", "Contrast", "Metric", "Δ", "Cluster-t 95%", "Matched accuracy"], [[c["tier"], c["contrast"], c["metric"], _pp(c["est"]), _ci(c["ci_t"]), _f(c.get("matched_accuracy")) if c["metric"] == "success" else ""] for c in det.get("contrasts") or []])]
    else:
        L += [f"_not available: {det.get('reason')}_", ""]

    inv = d.get("invariants") or {}
    L += ["## Invariants (§4.4)", "", f"Overall: {_f(inv.get('pass'))}.", ""]
    L += [f"- {k}: pass {_f(t.get('pass'))}, Δ {_pp(t.get('delta'))}, upper {_pp(t.get('upper'))}" for k, t in (inv.get("tests") or {}).items()]
    if inv.get("s5_gt_s7"):
        s = inv["s5_gt_s7"]
        L += [f"- S5 > S7: Δ {_pp(s['est'])}, p {_f(s['p'], 4)} (< 0.05: {_f(s['pass'])})"]
    L += [""]
    s8 = d.get("s8k3") or []
    if s8:
        L += ["## Live S8k3 vs post-hoc S8(3)", "", _table(["Cell", "Tier", "Live − post hoc", "Cluster-t 95%", "Tasks"], [[x["cell"], x["tier"], _pp(x.get("live_minus_posthoc")), _ci(x.get("ci_t")), x.get("tasks", x.get("reason"))] for x in s8])]

    hr = d.get("harness") or []
    L += ["## Harness", "", _table(["Tier", "Arm", "Cell", "Samples", "Errors", "Cap hits"], [[r["tier"], r["arm"], r["cell"], r["samples"], f"{_f(r['error_rate'])}{' ⚠' if r['error_fails'] else ''}", f"{_f(r['cap_hit_rate'])}{' ⚠' if r['cap_hit_fails'] else ''}"] for r in hr])]

    gl = d.get("glmm") or {}
    L += ["## GLMM (descriptive)", "", gl.get("model") or gl.get("reason") or "", ""]
    for fam, x in (gl.get("families") or {}).items():
        if not x.get("fitted"):
            L += [f"- {fam}: not fitted ({x.get('reason')})"]
            continue
        top = sorted(((n, v) for n, v in x["fixed"].items() if n != "Intercept"), key=lambda kv: -abs(kv[1]["mean"]))[:6]
        L += [f"- {fam}: converged {_f(x['converged'])}; random SDs " + ", ".join(f"{k} {_f(v, 2)}" for k, v in x["random_sd"].items()) + "; largest fixed effects (logit): " + "; ".join(f"{n} {_f(v['mean'], 2)} ± {_f(v['sd'], 2)}" for n, v in top)]
    L += ["", "## Analysis choices", ""] + [f"- {c}" for c in d.get("choices") or []]
    return "\n".join(L) + "\n"


def write_report(d: dict, out_dir: Path) -> tuple[Path, Path]:
    """report.json and report.md in `out_dir` (for the runner's glue)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    j, m = out_dir / "report.json", out_dir / "report.md"
    j.write_text(json.dumps(d, indent=1, allow_nan=False))
    m.write_text(render_main(d))
    return j, m
