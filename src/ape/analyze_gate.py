"""Gate decision report (FIX_PLAN FX-7; GATE_PREREG.md §2, §3, §7, §8 and §9 are the normative spec).

    uv run python -m ape.analyze_gate --run-id <id> [--runs-dir runs] [--force]

This runs the gate orchestrator's final phase, `analyze` (`ape.run_gate`), and writes

    runs/<id>/report/decision.json   every number below, machine-readable
    runs/<id>/report/report.md       the same, for people

Input: what the test phase recorded in `test/manifest.json` (its `cells` -> `groups` -> `log_files`; see
`ape.run_gate`), plus `anchor/pc1.json` (PC1), `tune/tuning_log.jsonl` and the selections (PC6),
`freeze.json`, `build-test/worlds.json` and the build/embedding ledger (costs). The report covers whatever
exists: a missing, failed or budget-stopped cell is reported as missing with its reason, and a missing input
a precondition needs makes that precondition fail with a named reason (so the verdict is PRECONDITION_FAIL),
never a crash.

Contents:
- Preconditions PC1-PC6 (§7), each with its value, threshold, pass/fail and reason.
- The verdict per delivery mode (§3): `gate_stats.decide(apg_arm="APG*", lgr_arm="LGR*")` on the four gate cells,
  equally weighted, with S7 from `gate.diag.s7`, by the world-clustered t-interval. Push pools F7 push with F3
  (push); pull pools F7 pull with the same F3 push cells, since F3 is push only. The modes are tested with Holm at
  the gate run's familywise α (`gate_stats.holm_modes`; D-023), then one label (`gate_stats.combine_modes`): GO,
  GO_PUSH_ONLY, GO_PULL_ONLY, GO_WITH_COST_FLAG, INCONCLUSIVE, NO_GO or PRECONDITION_FAIL. Errored samples count as
  failures (§2); a sensitivity analysis excludes them.
- The one pre-registered extension (§8): `--extension-of <stage-1 run>` marks this run as the extension of an
  INCONCLUSIVE gate run; its fresh worlds are analysed alone at the extension's α (`extension_context`).
- The NO-GO diagnosis (§8), always computed: S5o against LGR* on their shared tasks, and every failed
  sample's pipeline miss (`gate_stats.pipeline_miss`).
- Tables: per-cell Δ with 95% CIs; per arm success, partial credit, evidence recall, error labels; cost per
  query and the APG*/LGR* ratio; build cost per world; latency; determinism; `exception_applies` tasks.
- Secondaries, each on its own and never pooled: id_only, matched budget, TE-all, F5 and messy.

Arms are labelled by their declared plan names (APG*, LGR*, S3s, S7, LGR-naive, ...; the test manifest maps
each to the arm that ran), and every frame selects one delivery mode per arm, so push and pull results are
never averaged together (`gate_stats.task_means` refuses that).

Where the pre-registration is silent, the choices are in CHOICES; the report lists them too.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import cost as costs
from .analysis.gate_stats import (
    ALPHA_TOTAL,
    CELL_FLOOR,
    COST_FLAG_RATIO,
    GATE_CELLS,
    INVARIANT_TOL,
    MARGIN,
    MIN_WORLDS_PER_CELL,
    MISS_LABELS,
    STAGE_ALPHA,
    Decision,
    combine_modes,
    decide,
    delta_ci,
    epoch_agreement,
    holm_modes,
    invariants,
    load_results,
    pooled_success,
    shared_tasks,
    task_means,
    violation_test,
)

REPORT_DIR = "report"
F7_CELL, F3_CELL, PLACEBO_CELL, DIAG_CELL, F5_CELL = "gate.test.f7", "gate.test.f3", "gate.diag.s7", "gate.diag", "gate.f5"
SECONDARIES = (  # plan cell, title, comparisons (a, b): Δ = a − b
    ("gate.sec.id-only", "id_only exceptions", (("APG*", "LGR*"), ("APG*", "S3s"))),
    ("gate.sec.matched-300", "Matched budget", (("APG*", "LGR*"), ("APG*", "S3s"))),
    ("gate.sec.te-all", "TE-all tool exposure", (("APG*", "LGR*"), ("APG*", "S3s"))),
    (F5_CELL, "F5 (relational / temporal)", (("APG*", "LGR*"), ("LGR*", "LGR-naive"))),
    ("gate.sec.messy", "Messy exceptions (dev diagnostic)", (("APG*", "LGR*"), ("APG*", "S3s"))),
)
APG, LGR, S3S, PLACEBO, NAIVE = "APG*", "LGR*", "S3s", "S7", "LGR-naive"
CHAIN = ("S6", "S5o", APG, PLACEBO)  # PC3, best to worst
CAPPED = (LGR, S3S)  # PC4: the arms the matched-budget calibration caps (APG fills instead)
F5_CELLS = ("F5-1hop", "F5-2hop")
MAX_ERROR_RATE = 0.02  # PC5
MAX_CAP_HIT_RATE = 0.10  # PC5
CONTEXT_FACTOR = 4.0  # PC4 (descriptive): median realized context vs 4x the configured budget
MATCHED_TOLERANCE = 0.25  # PC4: capped arms within ±25% of the matched budget
ALPHA = 0.025  # one-sided level of the reported 95% intervals and of the PC2 / PC3 violation tests
GATE_ALPHA = STAGE_ALPHA["stage1"]  # the gate run's familywise one-sided α across delivery modes (§2, D-023)
EXTENSION_ALPHA = STAGE_ALPHA["extension"]  # the one pre-registered extension's α (§8)
EXTENSION_FILE = "extension.json"  # in a run dir: this run is the extension of the named stage-1 run
REPS = 10_000  # bootstrap and sign-flip resamples (§2)
SEED = 0

CHOICES = (
    "PC2 fails only on evidence that LGR* is worse than LightRAG naive by more than 3 pp (PC3's tolerance): the "
    "one-sided 97.5% upper bound of LGR* − naive, pooled over F5-1hop and F5-2hop with equal weights (world-clustered "
    "t), is below −3 pp (D-023). A tie fails at most 2.5% of the time; the earlier point-estimate rule failed 18%.",
    "PC3 compares the four arms on the tasks all of them ran (the diagnostics run on the first worlds of each "
    "cell), in push mode, the only mode the diagnostic arms run. Each adjacent pair fails only on evidence of a "
    "violation larger than 3 pp, at one-sided 0.025 / 3 per pair (D-023).",
    "PC4 is descriptive (D-023): its 4× bound holds by construction for every budgeted arm (LightRAG's "
    "max_total_tokens is 4× its budget, S7's budget is its own target, APG and S3s pack under theirs), so it is "
    "reported, and an arm over it is flagged as an anomaly, never gated. The ±25% clause uses the median over every "
    "compile of each capped arm in the matched-budget cell, pooled over cells, as the calibration did; APG*'s "
    "matched median is reported, not gated. A miss or a missing matched-budget secondary is reported there and "
    "never blocks the verdict (D-022).",
    "A precondition whose data is missing (cell not run, failed or budget-stopped) fails with that reason: "
    "§7's conditions must be shown, not assumed.",
    "PC5 is gated on the cells the verdict uses (gate.test.f7, gate.test.f3, gate.diag.s7), per arm and delivery "
    "mode; every other cell's rates are reported. The S7 placebo is gated on errors only: its cap hits are reported, "
    "since a random context that leaves the agent searching until the turn cap is the placebo working (D-023).",
    "PC6 counts every configuration in the current and archived tuning logs against budget_per_system, requires "
    "a record for every candidate this run's grid declares, and a selection per system that matches selected.yaml.",
    "The pull verdict compares APG* (pull) with S7 (push): S7 is a placebo context and has no pull mode.",
    f"The delivery modes' NI hypotheses are tested with Holm at the gate run's familywise one-sided α = {GATE_ALPHA} "
    "(D-023): each mode is classified at its Holm level (GO, INCONCLUSIVE or NO_GO by its interval at that level). "
    "One mode GO and the other INCONCLUSIVE is GO_<MODE>_ONLY; no mode GO and either INCONCLUSIVE is INCONCLUSIVE. "
    "Superiority is tested only after GO in every mode (serial gatekeeping), with Holm at the same α.",
    f"The one pre-registered extension (§8) analyses its fresh test worlds alone at α = {EXTENSION_ALPHA}, Holm across "
    f"modes; with the gate run's {GATE_ALPHA} the total one-sided α is {ALPHA_TOTAL} by Bonferroni (D-023).",
    "Every primary interval is the world-clustered t-interval with Satterthwaite df (D-023); the percentile "
    "bootstrap is reported alongside it for reference.",
    "The cost ratio is per mode, the mean per-query cost (Inspect-metered agent and kg calls plus query-time "
    "embeddings) of APG* over LGR* on that mode's rows. Query-time embeddings are cached across runs, so the "
    "ledger records only first embeddings; their cost is spread evenly over the arm's test samples.",
    "The NO-GO diagnosis compares S5o (gate.diag) with LGR* (push) on their shared tasks, non-inferior when the "
    "lower bound of the 95% CI is above −5 pp.",
)


# --- small helpers ---------------------------------------------------------------------------------


def _clean(x: Any) -> Any:
    """JSON-safe: numpy scalars to Python, NaN/inf to None, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        return _clean(x.tolist())
    return x


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x: Any) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * float(x):+.1f} pp"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _arm_key(label: str, delivery: str) -> str:
    return f"{label} ({delivery})"


def _iso_ts(s: str | None) -> float | None:
    return datetime.fromisoformat(s).timestamp() if s else None


ROW_COLUMNS = (
    "plan_cell", "group", "label", "arm", "run_arm", "delivery", "cell", "world", "task", "epoch", "success", "error",
    "partial_credit", "evidence_recall", "evidence_recall_first", "evidence_recall_step_mean", "error_label", "case",
    "ctx_tokens", "compile_tokens", "compile_ms", "budget", "cap_hit", "limit_hit", "pipeline_miss", "cost_usd", "usd",
    "total_time", "working_time", "exposure", "split", "exception_style",
)  # fmt: skip


# --- loading ---------------------------------------------------------------------------------------


def load_test_rows(test: dict | None, require_cost: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Every sample-epoch of every finished test group, labelled with its plan cell, group and declared arm
    (`label`; `arm` is set to it too, for `gate_stats`), and the coverage of every group (status, samples,
    reason when missing)."""
    frames, coverage = [], []
    for cell_id, cell in ((test or {}).get("cells") or {}).items():
        for g in cell.get("groups") or []:
            entry = {"plan_cell": cell_id, "group": g["name"], "status": g.get("status"), "arms": [a["declared"] for a in g.get("arms", [])], "samples": 0, "reason": None}
            files = g.get("log_files") or []
            if g.get("status") != "done":
                entry["reason"] = f"group {g.get('status')}" + (f": {g['error']}" if g.get("error") else "")
            elif not files or not all(Path(f).is_file() for f in files):
                entry["reason"] = "log files missing"
            else:
                labels: dict[str, str] = {}
                for a in g.get("arms", []):
                    if a["run"] in labels and labels[a["run"]] != a["declared"]:
                        raise ValueError(f"{cell_id} ({g['name']}): {labels[a['run']]} and {a['declared']} both ran as {a['run']}")
                    labels[a["run"]] = a["declared"]
                try:
                    df = load_results(files, require_cost=require_cost)
                except (ValueError, OSError, KeyError) as e:
                    entry["reason"] = f"logs unreadable: {type(e).__name__}: {e}"
                else:
                    df = df.rename(columns={"arm": "run_arm"})
                    df["label"] = df["run_arm"].map(labels)
                    df["arm"] = df["label"]
                    df["plan_cell"], df["group"] = cell_id, g["name"]
                    frames.append(df)
                    entry["samples"] = int(len(df))
            coverage.append(entry)
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c for c in ROW_COLUMNS if c != "usd"])
    rows["usd"] = rows["cost_usd"].astype(float) if len(rows) else pd.Series(dtype=float)
    return rows, coverage


def _sel(rows: pd.DataFrame, plan_cell: str, group: str | None = None, delivery: str | None = None, labels: Sequence[str] | None = None) -> pd.DataFrame:
    m = rows["plan_cell"] == plan_cell
    if group is not None:
        m &= rows["group"] == group
    if delivery is not None:
        m &= rows["delivery"] == delivery
    if labels is not None:
        m &= rows["label"].isin(labels)
    return rows[m]


def mode_rows(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    """The rows of one delivery mode's verdict: APG*, LGR* and S3s on the F7 cells in `mode` and on the F3 cells
    in push (F3 is push only), and the S7 placebo (push)."""
    return pd.concat(
        [
            _sel(rows, F7_CELL, "selected", mode, (APG, LGR, S3S)),
            _sel(rows, F3_CELL, "selected", "push", (APG, LGR, S3S)),
            _sel(rows, PLACEBO_CELL, delivery="push", labels=(PLACEBO,)),
        ],
        ignore_index=True,
    )


def diag_push_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Push rows of the primary arms and the diagnostics (S1, S5o, S6, S7): PC3 and the NO-GO diagnosis."""
    return pd.concat([mode_rows(rows, "push"), _sel(rows, DIAG_CELL, delivery="push")], ignore_index=True)


# --- costs -----------------------------------------------------------------------------------------


def attach_costs(rows: pd.DataFrame, ledger_df: pd.DataFrame, window: tuple[float | None, float | None]) -> dict:
    """Add the query-time embedding cost to each row (`usd` = Inspect-metered cost + its share): the ledger's
    embedding calls attributed to an arm during the test phase, spread evenly over that arm's test rows."""
    info: dict[str, Any] = {"query_embedding_usd": {}, "window": list(window)}
    rows["query_embed_usd"] = 0.0
    if len(rows) and not ledger_df.empty and {"arm", "kind", "usd"} <= set(ledger_df.columns):
        emb = ledger_df[(ledger_df["kind"] == "embed") & ledger_df["arm"].notna()]
        lo, hi = window
        if "ts" in emb.columns and lo is not None:
            emb = emb[(emb["ts"] >= lo) & (emb["ts"] <= (hi or float("inf")))]
        counts = rows.groupby("run_arm").size()
        for run_arm, usd in emb.groupby("arm")["usd"].sum().items():
            if run_arm in counts.index:
                rows.loc[rows["run_arm"] == run_arm, "query_embed_usd"] = float(usd) / int(counts[run_arm])
            info["query_embedding_usd"][run_arm] = float(usd)
    rows["usd"] = rows["cost_usd"].astype(float) + rows["query_embed_usd"].astype(float)
    return info


def cost_ratio(frame: pd.DataFrame, a: str = APG, b: str = LGR) -> tuple[float | None, str | None]:
    """Mean per-query cost of `a` over `b` in `frame`; None (with a reason) when it cannot be computed."""
    ua, ub = frame.loc[frame["label"] == a, "usd"], frame.loc[frame["label"] == b, "usd"]
    if ua.empty or ub.empty:
        return None, f"no rows for {a if ua.empty else b}"
    if float(ub.mean()) <= 0:
        return None, f"{b} cost is $0 (offline, or prices missing)"
    return float(ua.mean()) / float(ub.mean()), None


def build_costs(ledger_df: pd.DataFrame, test_worlds: Sequence[str]) -> dict:
    """Mean build cost per test world, by system (APG authoring, LightRAG extraction, chunk embeddings)."""
    bc = costs.build_cost(ledger_df)
    if bc.empty:
        return {"per_system": {}, "note": "no build calls in the ledger (offline runs make none)"}
    bc = bc[bc["world"].isin(set(test_worlds))] if test_worlds else bc
    out = {}
    for system, g in bc.groupby("system", dropna=False):
        out[str(system)] = {"worlds": int(g["world"].nunique()), "usd_total": float(g["usd"].sum()), "usd_per_world": float(g.groupby("world")["usd"].sum().mean())}
    return {"per_system": out}


# --- preconditions ---------------------------------------------------------------------------------


def _pc(pid: str, ok: bool, value: Any, threshold: str, reason: str | None = None, gated: bool = True, **details: Any) -> dict:
    """A precondition result. `gated` False: reported only; it always passes and never blocks the verdict."""
    return {"id": pid, "pass": bool(ok) or not gated, "gated": gated, "value": value, "threshold": threshold, "reason": reason, "details": details}


def pc1(pc1_path: Path) -> dict:
    thr = "gating-scorer macro within ±macro tolerance of the published LightRAG macro and every type within ±type tolerance; judge parse failures ≤ 5% (D-025)"
    if not pc1_path.is_file():
        return _pc("PC1", False, None, thr, f"{pc1_path} missing: run the anchor phase")
    r = json.loads(pc1_path.read_text())
    tol = r.get("tolerance_pp") or {}
    reasons = []
    if r.get("status") == "not_evaluable":
        reasons.append(f"not evaluable: {r.get('diagnosis')}")
    else:
        if not r.get("macro_within_tolerance"):
            reasons.append(f"macro {_f(r.get('macro'), 2)} not within ±{tol.get('macro')} pp of the published {_f(r.get('published_macro'), 2)}")
        if not r.get("types_within_tolerance"):
            off = [t for t, x in (r.get("per_type") or {}).items() if not x.get("within_tolerance")]
            reasons.append(f"outside ±{tol.get('per_type')} pp: {', '.join(off)}")
    ok = bool(r.get("pass"))
    value = {
        "status": r.get("status"),
        "macro": r.get("macro"),
        "published_macro": r.get("published_macro"),
        "current_scorer_macro": r.get("current_scorer_macro"),
        "naive_macro": r.get("naive_macro"),
        "tolerance_pp": tol,
    }
    return _pc("PC1", ok, value, thr, None if ok else "; ".join(reasons) or "pc1.json says it fails", per_type=r.get("per_type"), offline=r.get("offline"), beats_naive=r.get("beats_naive"))


def pc2(rows: pd.DataFrame, reps: int = REPS) -> dict:
    tol = f"{INVARIANT_TOL * 100:.0f} pp"
    thr = f"no evidence that LGR* < LightRAG naive − {tol} on F5: fails when the one-sided {100 * (1 - ALPHA):.1f}% upper bound of LGR* − naive (F5 cells equally weighted, world-clustered t) is below −{tol}"
    f5 = pd.concat([_sel(rows, F5_CELL, "selected", "push", (LGR,)), _sel(rows, F5_CELL, NAIVE.lower(), "push", (NAIVE,))], ignore_index=True)
    have = set(f5["label"])
    if not {LGR, NAIVE} <= have:
        return _pc("PC2", False, None, thr, f"{F5_CELL}: no results for {sorted({LGR, NAIVE} - have)} (cell not run, failed or stopped)")
    tm = task_means(f5)
    t = violation_test(tm, LGR, NAIVE, cells=F5_CELLS, tol=INVARIANT_TOL, alpha=ALPHA)
    if t["delta"] is None:
        return _pc("PC2", False, None, thr, f"{F5_CELL}: {LGR} and {NAIVE} share no tasks")
    d = delta_ci(tm, LGR, NAIVE, cells=F5_CELLS, alpha=ALPHA, reps=reps, seed=SEED)
    reason = None if t["pass"] else f"LGR* − naive = {_pp(t['delta'])}, upper bound {_pp(t['upper'])} < −{tol} (p = {t['p_violation']:.4f})"
    return _pc("PC2", t["pass"], t["delta"], thr, reason, test=t, ci=[d["lo"], d["hi"]], missing_cells=d.get("missing_cells"))


def pc3(rows: pd.DataFrame, reps: int = REPS) -> dict:
    thr = f"S6 ≥ S5o ≥ APG* ≥ S7, each pair failing only on evidence of a shortfall > {INVARIANT_TOL * 100:.0f} pp (one-sided, {ALPHA:g} / 3 per pair), and S6 > S7 (world sign-flip p < 0.05), push"
    frame = diag_push_rows(rows)
    have = set(frame["label"])
    if missing := [a for a in CHAIN if a not in have]:
        return _pc("PC3", False, None, thr, f"no results for {missing} (cell not run, failed or stopped)")
    tm = shared_tasks(task_means(frame), CHAIN)
    if tm.empty:
        return _pc("PC3", False, None, thr, f"{list(CHAIN)} share no tasks")
    cells = tuple(c for c in GATE_CELLS if c in set(tm.index.get_level_values("cell")))
    inv = invariants(tm, chain=CHAIN, alpha=ALPHA)
    failed = [f"{k} (Δ {_pp(inv['tests'][k]['delta'])}, upper bound {_pp(inv['tests'][k]['upper'])})" for k, v in inv["pairs"].items() if not v]
    failed += [] if inv["s6_gt_s7_significant"] else ["S6>S7 not significant"]
    return _pc("PC3", inv["pass"], inv["success"], thr, "; ".join(failed) or None, pairs=inv["pairs"], tests=inv["tests"], s6_gt_s7_significant=inv["s6_gt_s7_significant"], shared_tasks=int(len(tm)), cells=list(cells))


def context_medians(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (plan cell, group, label, delivery): median realized tokens per compile, median budget, and the median
    over compiles of realized / budget (PC4)."""
    if rows.empty:
        return pd.DataFrame(columns=["plan_cell", "group", "label", "delivery", "compiles", "median_tokens", "budget", "median_ratio"])
    x = rows[["plan_cell", "group", "label", "delivery", "cell", "compile_tokens", "budget"]].explode("compile_tokens").dropna(subset=["compile_tokens"])
    x["compile_tokens"], x["budget"] = x["compile_tokens"].astype(float), x["budget"].astype(float)
    x["ratio"] = x["compile_tokens"] / x["budget"].astype(float)
    g = x.groupby(["plan_cell", "group", "label", "delivery"])
    return pd.DataFrame(
        {"compiles": g.size(), "median_tokens": g["compile_tokens"].median(), "budget": g["budget"].median(), "median_ratio": g["ratio"].median()}
    ).reset_index()


def pc4(rows: pd.DataFrame, matched_cell: str, context: int | None, matched_present: bool) -> dict:
    """Descriptive (D-023): the 4× bound holds by construction for every budgeted arm (CHOICES), so it is reported,
    with any arm over it flagged as an anomaly, and never gates; the matched-budget ±25% clause likewise (D-022)."""
    thr = f"reported, not gated (D-023): median realized context against {CONTEXT_FACTOR:g}× the configured budget (every budgeted arm); matched budget: capped arms within ±{MATCHED_TOLERANCE:.0%} of {context}"
    med = context_medians(rows)
    budgeted = med[med["budget"].notna()]
    over = budgeted[budgeted["median_ratio"] > CONTEXT_FACTOR]
    bound = [
        {"plan_cell": r.plan_cell, "group": r.group, "arm": _arm_key(r.label, r.delivery), "median_tokens": r.median_tokens, "budget": r.budget, "median_ratio": r.median_ratio, "pass": r.median_ratio <= CONTEXT_FACTOR}
        for r in budgeted.itertuples()
    ]
    anomalies = [f"{b['plan_cell']} {b['arm']}: median {b['median_tokens']:.0f} tokens = {b['median_ratio']:.1f}× its budget {b['budget']:.0f}" for b in bound if not b["pass"]]
    matched: dict[str, Any] = {"context": context, "notes": []}
    if not matched_present or context is None:
        matched["notes"].append(f"{matched_cell}: the matched-budget secondary was not run (or failed); its ±{MATCHED_TOLERANCE:.0%} clause cannot be checked")
        matched["status"] = "missing"
    else:
        m = _sel(rows, matched_cell, "matched")
        lo, hi = context * (1 - MATCHED_TOLERANCE), context * (1 + MATCHED_TOLERANCE)
        for label in (*CAPPED, APG):
            toks = [t for ts in m.loc[m["label"] == label, "compile_tokens"] for t in ts]
            per_cell = {c: float(np.median([t for ts in g["compile_tokens"] for t in ts])) for c, g in m[m["label"] == label].groupby("cell") if any(len(ts) for ts in g["compile_tokens"])}
            median = float(np.median(toks)) if toks else None
            gated = label in CAPPED
            ok = median is not None and lo <= median <= hi
            matched[label] = {"median_tokens": median, "per_cell": per_cell, "gated": gated, "pass": ok if gated else None}
            if gated and not ok:
                matched["notes"].append(f"matched budget {label}: median {_f(median, 0)} tokens outside [{lo:.0f}, {hi:.0f}]")
        matched["status"] = "checked"
        matched["interpretable"] = not matched["notes"]
    reason = ("anomaly (not gated): " + "; ".join(anomalies)) if anomalies else None
    return _pc("PC4", over.empty, {"arms_over_bound": int(len(over))}, thr, reason, gated=False, bound=bound, anomalies=anomalies, matched=matched)


def pc5(rows: pd.DataFrame) -> dict:
    thr = f"harness error rate < {MAX_ERROR_RATE:.0%} and cap-hit rate < {MAX_CAP_HIT_RATE:.0%}, per arm and mode, in the verdict's cells (S7: errors only)"
    if rows.empty:
        return _pc("PC5", False, None, thr, "no test results")
    rates = (
        rows.assign(cap_hit=rows["cap_hit"].astype(bool), error=rows["error"].astype(bool))
        .groupby(["plan_cell", "label", "delivery"])
        .agg(samples=("success", "size"), error_rate=("error", "mean"), cap_hit_rate=("cap_hit", "mean"))
        .reset_index()
    )
    table = []
    reasons = []
    for r in rates.itertuples():
        gated = r.plan_cell in (F7_CELL, F3_CELL, PLACEBO_CELL)
        cap_gated = r.label != PLACEBO  # the placebo's cap hits are the placebo working: reported, not gated (D-023)
        ok = r.error_rate < MAX_ERROR_RATE and (r.cap_hit_rate < MAX_CAP_HIT_RATE or not cap_gated)
        table.append({"plan_cell": r.plan_cell, "arm": _arm_key(r.label, r.delivery), "samples": r.samples, "error_rate": r.error_rate, "cap_hit_rate": r.cap_hit_rate, "gated": gated, "cap_hits_gated": cap_gated, "pass": ok})
        if gated and not ok:
            reasons.append(f"{r.plan_cell} {_arm_key(r.label, r.delivery)}: errors {r.error_rate:.1%}, cap hits {r.cap_hit_rate:.1%}")
    if not any(t["gated"] for t in table):
        reasons.append("no results in the verdict's cells")
    worst = max((t["error_rate"] for t in table if t["gated"]), default=None), max((t["cap_hit_rate"] for t in table if t["gated"]), default=None)
    return _pc("PC5", not reasons, {"max_error_rate": worst[0], "max_cap_hit_rate": worst[1]}, thr, "; ".join(reasons) or None, rates=table)


def pc6(tune_dir: Path, grid: dict | None, full_grid: dict | None, selected: dict | None, systems: Sequence[tuple[str, str]]) -> dict:
    thr = "a record for every declared candidate; ≤ budget_per_system configurations per system (archived logs included); each selection logged and equal to selected.yaml"
    log = tune_dir / "tuning_log.jsonl"
    if not log.is_file() or grid is None or full_grid is None:
        return _pc("PC6", False, None, thr, f"{log} missing: run the tune phase" if not log.is_file() else "tuning_grid.yaml unreadable")
    archived = sorted(p for p in tune_dir.glob("tuning_log.*.jsonl"))

    def read(p: Path) -> list[dict]:
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    current, old = read(log), [r for p in archived for r in read(p)]
    budget = int(full_grid.get("budget_per_system", 0))
    reasons, per_system = [], {}
    for system, key in systems:
        declared = [c["id"] for c in grid["systems"].get(system, {}).get("candidates", [])]
        allowed = {c["id"] for c in full_grid["systems"].get(system, {}).get("candidates", [])}
        recs = [r for r in current if r.get("system") == system and "candidate" in r]
        logged = {r["candidate"]["id"] for r in recs}
        tried = logged | {r["candidate"]["id"] for r in old if r.get("system") == system and "candidate" in r}
        sel = [r["selected"] for r in current if r.get("system") == system and "selected" in r]
        want = ((selected or {}).get(key) or {}).get("candidate")
        s = {
            "declared": declared,
            "logged": sorted(logged),
            "failed": sorted(r["candidate"]["id"] for r in recs if r.get("status") == "failed"),
            "configurations_tried": len(tried),
            "budget": budget,
            "undeclared": sorted(tried - allowed),
            "selection_logged": sel[-1] if sel else None,
            "selected_yaml": want,
        }
        per_system[system] = s
        if missing := [c for c in declared if c not in logged]:
            reasons.append(f"{system}: no record for {missing}")
        if len(tried) > budget:
            reasons.append(f"{system}: {len(tried)} configurations tried > budget {budget}")
        if s["undeclared"]:
            reasons.append(f"{system}: undeclared configurations {s['undeclared']}")
        if not sel:
            reasons.append(f"{system}: no selection logged")
        elif sel[-1] != want:
            reasons.append(f"{system}: logged selection {sel[-1]} != selected.yaml {want}")
    return _pc("PC6", not reasons, {s: v["configurations_tried"] for s, v in per_system.items()}, thr, "; ".join(reasons) or None, systems=per_system, archived_logs=[str(p) for p in archived])


# --- verdict ---------------------------------------------------------------------------------------


def _missing_primary(frame: pd.DataFrame) -> list[str]:
    out = []
    for c in GATE_CELLS:
        for a in (APG, LGR):
            if not ((frame["cell"] == c) & (frame["label"] == a)).any():
                out.append(f"{a} on {c}")
    if not (frame["label"] == PLACEBO).any():
        out.append("S7 placebo")
    return out


def _decision(frame: pd.DataFrame, preconditions: dict[str, bool], ratio: float | None, alpha: float, reps: int, seed: int) -> Decision:
    """`gate_stats.decide` on one mode's rows, or PRECONDITION_FAIL (with Δ on what exists) when primary data is missing."""
    tm = task_means(frame)
    if missing := _missing_primary(frame):
        d = delta_ci(tm, APG, LGR, cells=GATE_CELLS, alpha=ALPHA, reps=reps, seed=seed)
        return Decision("PRECONDITION_FAIL", d["delta"], (d["lo"], d["hi"]), [f"primary data missing: {', '.join(missing)}"], details={"dropped_tasks": d.get("dropped_tasks", {})})
    return decide(tm, LGR, preconditions, cost_ratio=ratio, alpha=alpha, reps=reps, seed=seed, apg_arm=APG)


def mode_verdict(rows: pd.DataFrame, mode: str, preconditions: dict[str, bool], reps: int = REPS, seed: int = SEED, alpha: float = GATE_ALPHA) -> tuple[dict, Decision | None, Decision | None]:
    """One delivery mode's unadjusted decision at `alpha` (`gate_stats.decide`; `verdict` then applies Holm across
    the modes), its per-cell Δs (95% world-clustered t), cost ratio, and the error-excluding sensitivity decision.
    Missing primary data makes it PRECONDITION_FAIL, with Δ reported on what exists."""
    frame = mode_rows(rows, mode)
    ratio, ratio_note = cost_ratio(frame)
    out: dict[str, Any] = {"mode": mode, "pooling": f"F7-10, F7-1000 ({mode}) + F3-5, F3-60 (push), equal weights", "cost_ratio": ratio, "cost_ratio_note": ratio_note}
    if frame.empty:
        return out | {"verdict": "PRECONDITION_FAIL", "reasons": [f"no primary results for {mode}"], "delta": None, "ci": [None, None], "cells": {}}, None, None
    tm = task_means(frame)
    out["cells"] = {c: delta_ci(tm, APG, LGR, cells=(c,), alpha=ALPHA, reps=reps, seed=seed) for c in GATE_CELLS}
    out["worlds_per_cell"] = {c: int(frame.loc[(frame["cell"] == c) & (frame["label"] == APG), "world"].nunique()) for c in GATE_CELLS}
    dec = _decision(frame, preconditions, ratio, alpha, reps, seed)
    sens = frame[~frame["error"].astype(bool)]
    sdec = _decision(sens, preconditions, ratio, alpha, reps, seed) if len(sens) and not _missing_primary(sens) else None
    out["excluded_samples"] = int(len(frame) - len(sens))
    return out, dec, sdec


def _mode_fields(dec: Decision) -> dict:
    d = {k: v for k, v in dec.details.items() if k != "_stats"}
    return {"verdict": dec.verdict, "reasons": dec.reasons, "delta": dec.delta, "ci": list(dec.ci), "superiority": dec.superiority, "details": d}


def verdict(rows: pd.DataFrame, pcs: list[dict], modes: Sequence[str], reps: int = REPS, seed: int = SEED, alpha: float = GATE_ALPHA, stage: str = "gate run") -> dict:
    """The decision (§8): every mode decided at `alpha`, then Holm across the modes (`gate_stats.holm_modes`, with
    superiority by serial gatekeeping), then one label (`gate_stats.combine_modes`). The error-excluding
    sensitivity analysis goes through the same steps."""
    # A precondition failure accepted at the freeze (only PC1 can be: GATE_PREREG §7) does not block the verdict;
    # the decision carries it as a caveat instead.
    preconditions = {p["id"]: bool(p["pass"] or p.get("accepted")) for p in pcs}
    raw = {m: mode_verdict(rows, m, preconditions, reps, seed, alpha) for m in modes}
    per_mode = {m: r[0] for m, r in raw.items()}
    decs = {m: r[1] for m, r in raw.items() if r[1] is not None}
    adjusted, holm = holm_modes(decs, alpha)
    for m, d in adjusted.items():
        per_mode[m] |= {"unadjusted_verdict": decs[m].verdict} | _mode_fields(d)
    sens = {m: r[2] for m, r in raw.items() if r[2] is not None}
    sens_adj, _ = holm_modes(sens, alpha) if sens else ({}, None)
    for m in per_mode:
        s = sens_adj.get(m)
        per_mode[m]["sensitivity_excluding_errors"] = (
            {"verdict": s.verdict, "delta": s.delta, "ci": list(s.ci), "excluded_samples": per_mode[m].get("excluded_samples", 0)}
            if s is not None
            else {"verdict": None, "note": "not computable: excluding errored samples leaves a primary cell empty"}
        )
    label, reasons = combine_modes({m: v["verdict"] for m, v in per_mode.items()})
    sens_label = combine_modes({m: s.verdict for m, s in sens_adj.items()})[0] if len(sens_adj) == len(per_mode) else None
    if label == "PRECONDITION_FAIL":
        reasons = [f"{p['id']}: {p['reason']}" for p in pcs if not (p["pass"] or p.get("accepted"))] + [r for v in per_mode.values() for r in v["reasons"] if r.startswith("primary data missing")]
    notes = [f"{stage}: familywise one-sided α = {alpha} across delivery modes (Holm); per-mode levels {', '.join(f'{m} {lvl:g}' for m, lvl in holm['levels'].items()) or 'n/a'}."]
    if label == "INCONCLUSIVE" and stage == "gate run":
        notes.append(f"§8 allows ONE pre-registered extension on fresh test worlds, analysed alone at α = {EXTENSION_ALPHA} (Holm across modes): `python -m ape.analyze_gate --run-id <extension run> --extension-of <this run>`.")
    if label in ("GO_PUSH_ONLY", "GO_PULL_ONLY", "GO_WITH_COST_FLAG"):
        notes.append("§8: the user decides.")
    sup = [m for m, v in per_mode.items() if v.get("superiority")]
    if sup:
        notes.append(f"Superiority (after GO in every mode; Holm at {alpha}) in: {', '.join(sup)}.")
    return {"label": label, "reasons": reasons, "notes": notes, "modes": per_mode, "alpha": alpha, "holm": holm, "sensitivity_label": sens_label}


def diagnosis(rows: pd.DataFrame, reps: int = REPS) -> dict:
    """§8 NO-GO diagnosis: S5o against LGR* (extraction vs runtime/representation), and the miss labels."""
    frame = diag_push_rows(rows)
    out: dict[str, Any] = {"rule": "S5o non-inferior to LGR* (lower bound > −5 pp) → extraction (fix apg/author.py); otherwise runtime or representation"}
    if {"S5o", LGR} <= set(frame["label"]):
        tm = shared_tasks(task_means(frame), ("S5o", LGR))
        d = delta_ci(tm, "S5o", LGR, cells=GATE_CELLS, alpha=ALPHA, reps=reps, seed=SEED)
        if d["lo"] is None:  # no clustered variance (fewer than 2 paired worlds in a cell): no diagnosis
            out |= {"s5o_vs_lgr": d, "s5o_non_inferior": None, "label": None, "reason": d.get("reason") or "S5o and LGR* share too few worlds for an interval"}
        else:
            ni = d["lo"] > -MARGIN
            out |= {"s5o_vs_lgr": d, "s5o_non_inferior": ni, "label": "extraction" if ni else "runtime or representation"}
    else:
        out |= {"s5o_vs_lgr": None, "label": None, "reason": f"no results for {sorted({'S5o', LGR} - set(frame['label']))}"}
    failed = rows[(rows["success"] == 0) & ~rows["error"].astype(bool) & rows["plan_cell"].isin((F7_CELL, F3_CELL, PLACEBO_CELL, DIAG_CELL))] if len(rows) else rows
    misses: dict[str, dict[str, int]] = {}
    for (label, delivery), g in failed.groupby(["label", "delivery"]) if len(failed) else []:
        counts = g["pipeline_miss"].value_counts()
        misses[_arm_key(label, delivery)] = {m: int(counts.get(m, 0)) for m in MISS_LABELS if counts.get(m, 0)}
    out["pipeline_misses"] = misses
    return out


# --- tables ----------------------------------------------------------------------------------------


def arm_table(rows: pd.DataFrame) -> dict[str, dict]:
    """Per (label, delivery): samples, worlds, success (cells equally weighted) and per cell, partial credit,
    evidence recall (union, first compile, per-step mean), realized context, error labels, cap hits, errors."""
    out = {}
    if rows.empty:
        return out
    for (label, delivery), g in rows.groupby(["label", "delivery"]):
        tm = task_means(g)
        cells = tuple(c for c in sorted(set(g["cell"])))
        per_cell = tm[label].groupby(level="cell").mean().to_dict()
        out[_arm_key(label, delivery)] = {
            "samples": int(len(g)),
            "worlds": int(g["world"].nunique()),
            "success": pooled_success(tm, label, cells),
            "success_per_cell": per_cell,
            "partial_credit": float(g["partial_credit"].mean()),
            "evidence_recall": float(g["evidence_recall"].mean()),
            "evidence_recall_first": float(g["evidence_recall_first"].mean()),
            "evidence_recall_step_mean": float(g["evidence_recall_step_mean"].mean()),
            "median_context_tokens": float(np.median([t for ts in g["compile_tokens"] for t in ts])) if any(len(ts) for ts in g["compile_tokens"]) else None,
            "error_labels": {str(k): int(v) for k, v in g["error_label"].value_counts().items()},
            "errors": int(g["error"].astype(bool).sum()),
            "cap_hits": int(g["cap_hit"].astype(bool).sum()),
        }
    return out


def _q(values: Sequence[float], q: float) -> float | None:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.quantile(v, q)) if v else None


def latency_table(rows: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for (label, delivery), g in rows.groupby(["label", "delivery"]) if len(rows) else []:
        ms = [m for ms_ in g["compile_ms"] for m in ms_]
        out[_arm_key(label, delivery)] = {
            "total_time_median_s": _q(list(g["total_time"].astype(float)), 0.5),
            "total_time_p95_s": _q(list(g["total_time"].astype(float)), 0.95),
            "working_time_median_s": _q(list(g["working_time"].astype(float)), 0.5),
            "working_time_p95_s": _q(list(g["working_time"].astype(float)), 0.95),
            "compile_ms_median": _q(ms, 0.5),
            "compile_ms_p95": _q(ms, 0.95),
        }
    return out


def determinism_table(rows: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for (label, delivery), g in rows.groupby(["label", "delivery"]) if len(rows) else []:
        epochs = int(g["epoch"].nunique())
        out[_arm_key(label, delivery)] = {"epochs": epochs, "epoch_agreement": epoch_agreement(g) if epochs > 1 else None}
    return out


def exception_table(rows: pd.DataFrame) -> dict[str, dict]:
    """§3: `exception_applies` tasks reported separately (F7 cells)."""
    f7 = rows[rows["cell"].astype(str).str.startswith("F7")] if len(rows) else rows
    out = {}
    for (label, delivery), g in f7.groupby(["label", "delivery"]) if len(f7) else []:
        ex, other = g[g["case"] == "exception_applies"], g[g["case"] != "exception_applies"]
        out[_arm_key(label, delivery)] = {"exception_applies": {"samples": int(len(ex)), "success": float(ex["success"].mean()) if len(ex) else None}, "other": {"samples": int(len(other)), "success": float(other["success"].mean()) if len(other) else None}}
    return out


def cost_table(rows: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for (label, delivery), g in rows.groupby(["label", "delivery"]) if len(rows) else []:
        out[_arm_key(label, delivery)] = {"samples": int(len(g)), "usd_per_query": float(g["usd"].mean()), "inspect_usd_per_query": float(g["cost_usd"].mean()), "embedding_usd_per_query": float(g["query_embed_usd"].mean()) if "query_embed_usd" in g else 0.0}
    return out


# --- secondaries -----------------------------------------------------------------------------------


def _seed_keys(df: pd.DataFrame) -> pd.Series:
    """cell:seed of each row's world (`F7-10-rel-idonly-test-s3000` -> `F7-10:s3000`): pairs renderings of a seed."""
    return df["cell"].astype(str) + ":" + df["world"].astype(str).str.rsplit("-", n=1).str[-1]


def secondary(rows: pd.DataFrame, coverage: list[dict], plan_cell: str, title: str, comparisons, reps: int = REPS) -> dict:
    groups = [c for c in coverage if c["plan_cell"] == plan_cell]
    sub = rows[rows["plan_cell"] == plan_cell] if len(rows) else rows
    out: dict[str, Any] = {"plan_cell": plan_cell, "title": title, "groups": groups}
    missing = [f"{g['group']}: {g['reason']}" for g in groups if g["reason"]]
    if not groups:
        return out | {"status": "missing", "reason": "not in the test manifest (test phase not run, or the cell is not in the plan)"}
    if sub.empty:
        return out | {"status": "missing", "reason": "; ".join(missing) or "no results"}
    out |= {"status": "partial" if missing else "done", "reason": "; ".join(missing) or None, "arms": arm_table(sub), "comparisons": {}}
    cells = tuple(sorted(set(sub["cell"])))
    for delivery, g in sub.groupby("delivery"):
        tm = task_means(g)
        for a, b in comparisons:
            out["comparisons"][f"{a} − {b} ({delivery})"] = delta_ci(tm, a, b, cells=cells, alpha=ALPHA, reps=reps, seed=SEED)
    return out


def id_only_pairing(rows: pd.DataFrame) -> dict:
    """id_only against the descriptive rendering of the same seeds (world-level success), per arm and mode."""
    ido = rows[rows["plan_cell"] == "gate.sec.id-only"] if len(rows) else rows
    desc = rows[(rows["plan_cell"] == F7_CELL) & (rows["group"] == "selected")] if len(rows) else rows
    out = {}
    for (label, delivery), g in ido.groupby(["label", "delivery"]) if len(ido) else []:
        d = desc[(desc["label"] == label) & (desc["delivery"] == delivery)]
        wi = g.groupby(_seed_keys(g))["success"].mean()
        wd = d.groupby(_seed_keys(d))["success"].mean() if len(d) else pd.Series(dtype=float)
        shared = sorted(set(wi.index) & set(wd.index))
        out[_arm_key(label, delivery)] = {
            "shared_worlds": len(shared),
            "descriptive": float(wd[shared].mean()) if shared else None,
            "id_only": float(wi[shared].mean()) if shared else None,
            "id_only_minus_descriptive": float((wi[shared] - wd[shared]).mean()) if shared else None,
        }
    return out


def te_pairing(rows: pd.DataFrame) -> dict:
    """TE-all against TE-retrieved (the primary F3 push runs) on shared tasks, per arm."""
    te = rows[rows["plan_cell"] == "gate.sec.te-all"] if len(rows) else rows
    base = _sel(rows, F3_CELL, "selected", "push") if len(rows) else rows
    out = {}
    for label, g in te.groupby("label") if len(te) else []:
        b = base[base["label"] == label]
        tm_all = g.groupby(["cell", "world", "task"])["success"].mean()
        tm_ret = b.groupby(["cell", "world", "task"])["success"].mean() if len(b) else pd.Series(dtype=float)
        shared = tm_all.index.intersection(tm_ret.index)
        out[label] = {"shared_tasks": int(len(shared)), "retrieved": float(tm_ret[shared].mean()) if len(shared) else None, "all": float(tm_all[shared].mean()) if len(shared) else None}
    return out


# --- the report ------------------------------------------------------------------------------------


def _world_ids(run_dir: Path) -> set[str]:
    f = run_dir / "build-test" / "worlds.json"
    return {w["world_id"] for w in json.loads(f.read_text())["worlds"]} if f.is_file() else set()


def extension_context(run) -> dict | None:
    """If `run` is the one pre-registered extension (§8; `extension.json` in its run dir, written by
    `analyze_gate --extension-of`), the stage-1 facts it rests on, with `problems` when it is not admissible:
    stage 1 must be analysed and INCONCLUSIVE, and the two runs' test worlds (`build-test/worlds.json`) must
    be disjoint, so the extension's worlds are fresh (no seed arithmetic is assumed)."""
    f = run.dir / EXTENSION_FILE
    if not f.is_file():
        return None
    spec = json.loads(f.read_text())
    stage1_dir = Path(spec["stage1_dir"])
    problems = []
    dec_path = stage1_dir / REPORT_DIR / "decision.json"
    stage1 = json.loads(dec_path.read_text()) if dec_path.is_file() else None
    if stage1 is None:
        problems.append(f"stage 1 ({stage1_dir}) has no decision.json: analyse it first")
    elif stage1["verdict"]["label"] != "INCONCLUSIVE":
        problems.append(f"stage 1's verdict is {stage1['verdict']['label']}, not INCONCLUSIVE: §8 allows the extension only after INCONCLUSIVE")
    if stage1 is not None and stage1.get("extension"):
        problems.append("stage 1 is itself an extension: §8 allows ONE extension")
    w1, w2 = _world_ids(stage1_dir), _world_ids(run.dir)
    if not w1 or not w2:
        problems.append("build-test/worlds.json missing in " + " and ".join(str(d) for d, w in ((stage1_dir, w1), (run.dir, w2)) if not w))
    elif shared := sorted(w1 & w2):
        problems.append(f"the extension reuses {len(shared)} stage-1 test world(s) (e.g. {shared[:3]}): it needs fresh worlds")
    return {
        "stage1_run_id": spec["stage1_run_id"],
        "stage1_dir": str(stage1_dir),
        "stage1_label": stage1["verdict"]["label"] if stage1 else None,
        "stage1_alpha": (stage1 or {}).get("constants", {}).get("alpha_gate_run", GATE_ALPHA),
        "stage1_worlds": len(w1),
        "extension_worlds": len(w2),
        "alpha": EXTENSION_ALPHA,
        "total_alpha": ALPHA_TOTAL,
        "rule": f"the extension's fresh test worlds are analysed alone at one-sided α = {EXTENSION_ALPHA} (Holm across modes); with the gate run's {GATE_ALPHA}, the familywise α is at most {ALPHA_TOTAL} (Bonferroni over disjoint data)",
        "problems": problems,
    }


def _test_window(test: dict | None) -> tuple[float | None, float | None]:
    if not test:
        return None, None
    runs = [h["at"] for h in test.get("history") or [] if h.get("action") == "run"]
    start = min(runs) if runs else test.get("started")
    return _iso_ts(start), _iso_ts(test.get("finished"))


def analyze(run) -> dict:
    """Assemble the decision for gate run `run` (an `ape.run_gate.GateRun`, inside its `run_environment`), write
    report/decision.json and report/report.md, and return the decision."""
    from . import run_gate as rg
    from .config import Config
    from .llm.ledger import Ledger
    from .tuning import load_grid

    test = rg.read_manifest(run, "test")
    rows, coverage = load_test_rows(test, require_cost=not run.offline)
    p = rg.plan(run)

    # Costs: Inspect-metered per sample, plus query-time embeddings and builds from the ledger.
    ledger_path = Config().ledger_path
    cost_notes: list[str] = []
    try:
        ledger_df = costs.ledger_frame(Ledger(ledger_path), costs.load_prices(run.costs_path)) if ledger_path.is_file() else pd.DataFrame()
    except KeyError as e:
        ledger_df = pd.DataFrame()
        cost_notes.append(f"ledger not priced: {e}")
    embed_info = attach_costs(rows, ledger_df, _test_window(test))
    worlds_file = run.phase_dir("build-test") / "worlds.json"
    test_worlds = [w["world_id"] for w in json.loads(worlds_file.read_text())["worlds"]] if worlds_file.is_file() else []

    # Preconditions.
    try:
        grid, full_grid = rg.tuning_grid(run), load_grid(run.config("tuning_grid.yaml"))
    except Exception:  # noqa: BLE001  (PC6 fails with the reason)
        grid = full_grid = None
    selected = rg.read_selected(run.selected_path) if run.selected_path.is_file() else None
    matched_spec = next((c.spec for c in rg._test_cells(p) if c.id == "gate.sec.matched-300"), {})
    matched_present = any(c["plan_cell"] == "gate.sec.matched-300" and c["group"] == "matched" and c["reason"] is None for c in coverage)
    pcs = [
        pc1(run.phase_dir("anchor") / "pc1.json"),
        pc2(rows),
        pc3(rows),
        pc4(rows, "gate.sec.matched-300", matched_spec.get("context"), matched_present),
        pc5(rows),
        pc6(run.phase_dir("tune"), grid, full_grid, selected, rg.TUNED_SYSTEMS),
    ]
    freeze = rg.read_freeze(run) or {}
    caveats = []
    if (acc := freeze.get("pc1_accepted")) and not pcs[0]["pass"]:
        pcs[0]["accepted"] = acc
        caveats.append(f"PC1 (the LightRAG anchor) failed and was accepted at the freeze: {acc['reason']}. The LightRAG setup is not validated by the anchor; PC2 is the remaining competence check.")
    f7_groups = [g for g in ((test or {}).get("cells", {}).get(F7_CELL, {}) or {}).get("groups", [])]
    modes = tuple(dict.fromkeys(d for g in f7_groups for d in g.get("deliveries", []))) or ("push", "pull")
    ext = extension_context(run)
    if ext is None:
        v = verdict(rows, pcs, modes)
    else:
        v = verdict(rows, pcs, modes, alpha=EXTENSION_ALPHA, stage="extension")
        if ext["problems"]:
            v |= {"label": "PRECONDITION_FAIL", "reasons": [f"extension not admissible: {p}" for p in ext["problems"]] + v["reasons"]}
        else:
            v["notes"].insert(0, f"Extension of {ext['stage1_run_id']} (stage 1 INCONCLUSIVE): {ext['rule']}. This label is the gate's final verdict.")
    primary_rows = pd.concat([mode_rows(rows, m) for m in modes], ignore_index=True).drop_duplicates(subset=["plan_cell", "label", "delivery", "task", "epoch"]) if len(rows) else rows
    diag_rows = _sel(rows, DIAG_CELL) if len(rows) else rows
    tables_rows = pd.concat([primary_rows, diag_rows], ignore_index=True) if len(rows) else rows
    decision = {
        "header": {
            "run_id": run.run_id,
            "offline": run.offline,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git": rg.git_state(),
            "freeze": {k: freeze.get(k) for k in ("frozen_at", "rehearsal", "git", "analysis_commit", "apg_core")} if freeze else None,
            "selected": {k: {"arm": v_.get("arm"), "candidate": v_.get("candidate")} for k, v_ in (selected or {}).items()},
            "spend": rg.spend(run),
            "test_status": (test or {}).get("status"),
            "primary_complete": (test or {}).get("primary_complete"),
            "cells_not_done": {c: x["status"] for c, x in ((test or {}).get("cells") or {}).items() if x.get("status") != "done"},
        },
        "verdict": v,
        "extension": ext,
        "caveats": caveats,
        "preconditions": pcs,
        "diagnosis": diagnosis(rows),
        "tables": {
            "arms": arm_table(tables_rows),
            "costs": cost_table(tables_rows),
            "cost_notes": cost_notes,
            "query_embeddings": embed_info,
            "build_costs": build_costs(ledger_df, test_worlds),
            "latency": latency_table(tables_rows),
            "determinism": determinism_table(tables_rows),
            "exception_applies": exception_table(primary_rows),
        },
        "secondaries": {cell: secondary(rows, coverage, cell, title, comps) for cell, title, comps in SECONDARIES},
        "secondary_pairings": {"id_only_vs_descriptive": id_only_pairing(rows), "te_all_vs_retrieved": te_pairing(rows)},
        "coverage": coverage,
        "choices": list(CHOICES),
        "constants": {
            "margin": MARGIN,
            "alpha_gate_run": GATE_ALPHA,
            "alpha_extension": EXTENSION_ALPHA,
            "alpha_total": ALPHA_TOTAL,
            "alpha_reported_intervals": ALPHA,
            "interval": "world-clustered t, Satterthwaite df",
            "cell_floor": CELL_FLOOR,
            "cost_flag_ratio": COST_FLAG_RATIO,
            "invariant_tol": INVARIANT_TOL,
            "min_paired_worlds_per_cell": MIN_WORLDS_PER_CELL,
            "bootstrap_reps": REPS,
        },
    }
    decision = _clean(decision)
    out = run.dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps(decision, indent=1, sort_keys=False))
    (out / "report.md").write_text(render(decision))
    return decision


# --- markdown --------------------------------------------------------------------------------------

SECTIONS = (
    "Verdict",
    "Run",
    "Preconditions",
    "Verdict by delivery mode",
    "Per-cell Δ",
    "NO-GO diagnosis",
    "Arms",
    "Error labels",
    "Cost",
    "Latency",
    "Determinism",
    "exception_applies tasks",
    "Secondaries",
    "Coverage",
    "Analysis choices",
)


def _ci(lo: Any, hi: Any) -> str:
    return "–" if lo is None or hi is None else f"[{_pp(lo)}, {_pp(hi)}]"


def render(d: dict) -> str:
    h, v = d["header"], d["verdict"]
    L: list[str] = [f"# Gate decision report: run `{h['run_id']}`{' (OFFLINE)' if h['offline'] else ''}", ""]
    L += ["## Verdict", "", f"**{v['label']}**", ""]
    L += [f"> **Caveat:** {c}" for c in d.get("caveats") or []] + ([""] if d.get("caveats") else [])
    L += [f"- {r}" for r in v["reasons"]] + [f"- {n}" for n in v["notes"]]
    if v["label"] == "NO_GO" and (dg := d["diagnosis"]).get("label"):
        L += [f"- NO-GO diagnosis: **{dg['label']}** (see below)."]
    L += [""]

    fz = h.get("freeze") or {}
    L += ["## Run", ""]
    L += [
        f"- Generated {h['generated_at']}; commit `{(h['git'] or {}).get('commit')}`{' (dirty)' if (h['git'] or {}).get('dirty') else ''}.",
        f"- Freeze: {fz.get('frozen_at') or 'NOT FROZEN'}" + (" (offline rehearsal)" if fz.get("rehearsal") else "") + (f"; analysis code `{fz.get('analysis_commit')}`" if fz else "") + ".",
        "- Selections: " + (", ".join(f"{k} = {x['arm']} ({x['candidate']})" for k, x in h["selected"].items()) + "." if h["selected"] else "none recorded."),
        f"- Spend: ${(h['spend'] or {}).get('spent_usd', 0):,.2f} of ${(h['spend'] or {}).get('budget_usd', 0):,.0f}.",
        f"- Test phase: {h['test_status'] or 'not run'}; primary complete: {_f(h['primary_complete']) if h['primary_complete'] is not None else '–'}"
        + (f"; cells not done: {', '.join(f'{c} ({s})' for c, s in h['cells_not_done'].items())}" if h["cells_not_done"] else "")
        + ".",
        "",
    ]

    L += ["## Preconditions", "", "A failure means fix and re-pilot, not NO-GO (§7).", ""]

    def mark(p: dict) -> str:
        if not p.get("gated", True):
            return "ℹ️ reported"
        return "✅" if p["pass"] else ("⚠️ accepted" if p.get("accepted") else "❌")

    L += [_table(["", "Pass", "Value", "Threshold", "Reason"], [[p["id"], mark(p), _short(p["value"]), p["threshold"], p["reason"] or ""] for p in d["preconditions"]])]

    ext = d.get("extension")
    if ext:
        L += ["## Extension (§8)", ""]
        L += [f"- Stage 1: run `{ext['stage1_run_id']}` ({ext['stage1_label']}, {ext['stage1_worlds']} test worlds, α = {ext['stage1_alpha']}).", f"- This run: {ext['extension_worlds']} fresh test worlds, analysed alone. Rule: {ext['rule']}."]
        L += [f"- Not admissible: {p}" for p in ext["problems"]] + [""]

    L += ["## Verdict by delivery mode", ""]
    L += [
        f"Both modes pool the four gate cells with equal weights: push = F7 push + F3; pull = F7 pull + the same F3 push cells (F3 is push only). Δ = success(APG*) − success(LGR*); non-inferiority margin −5 pp. "
        f"Inference: world-clustered t-interval (Satterthwaite df). The modes are tested with Holm at familywise one-sided α = {v.get('alpha', GATE_ALPHA)}; each mode's interval below is at its Holm level. Superiority only after GO in every mode.",
        "",
    ]
    rows = []
    for m, x in v["modes"].items():
        det = x.get("details") or {}
        inf = det.get("inference") or {}
        sec = det.get("secondary") or {}
        sens = x.get("sensitivity_excluding_errors") or {}
        rows.append([
            m, x["verdict"], x.get("unadjusted_verdict") or "–", _pp(x["delta"]), _f(inf.get("holm_level"), 4) if inf.get("holm_level") else "–", _ci(*x["ci"]), _ci(*(inf.get("ci95") or [None, None])),
            _f(inf.get("p_noninferiority"), 4), _f(x.get("superiority")) if x.get("superiority") is not None else "–",
            _pp(sec.get("f7_1000_delta")), _f(sec.get("p_apg_gt_s7")), _f(x["cost_ratio"], 2), f"{sens.get('verdict') or '–'} ({_pp(sens.get('delta'))})",
        ])  # fmt: skip
    L += [_table(["Mode", "Verdict", "Unadjusted", "Δ", "Holm level", "CI at level", "95% CI", "p (NI)", "Superiority", "F7-1000 Δ", "p(APG* > S7)", "Cost ratio", "Excl. errors"], rows)]
    for m, x in v["modes"].items():
        det = x.get("details") or {}
        if x["reasons"]:
            L += [f"- {m}: " + "; ".join(x["reasons"])]
        if det.get("bootstrap_ci95"):
            L += [f"- {m}: percentile bootstrap 95% CI (reference) {_ci(*det['bootstrap_ci95'])}."]
        if det.get("dropped_tasks"):
            L += [f"- {m}: unpaired tasks left out (one arm missing): {det['dropped_tasks']}."]
        if x.get("cost_ratio_note"):
            L += [f"- {m} cost ratio: {x['cost_ratio_note']}"]
    if v.get("sensitivity_label"):
        L += [f"- Sensitivity (errored samples excluded), combined: {v['sensitivity_label']}."]
    L += [""]

    L += ["## Per-cell Δ", "", "World-clustered t-interval per cell (df = paired worlds − 1); worlds and tasks are paired APG*/LGR* ones.", ""]
    rows = [
        [m, c, _pp(ci.get("delta")), _ci(ci.get("lo"), ci.get("hi")), (ci.get("worlds_per_cell") or {}).get(c, 0), ci.get("tasks", 0), sum((ci.get("dropped_tasks") or {}).values()), ("few worlds" if ci.get("few_worlds") else "") or (ci.get("reason") or "")]
        for m, x in v["modes"].items()
        for c, ci in (x.get("cells") or {}).items()
    ]
    L += [_table(["Mode", "Cell", "Δ", "95% CI", "Worlds", "Tasks", "Unpaired", "Note"], rows)]

    dg = d["diagnosis"]
    L += ["## NO-GO diagnosis", "", f"Rule: {dg['rule']}.", ""]
    if dg.get("s5o_vs_lgr") and dg.get("label"):
        s = dg["s5o_vs_lgr"]
        L += [f"- S5o − LGR* = {_pp(s['delta'])} {_ci(s['lo'], s['hi'])} on {s.get('tasks', 0)} shared tasks → **{dg['label']}**.", ""]
    elif dg.get("s5o_vs_lgr"):
        s = dg["s5o_vs_lgr"]
        L += [f"- S5o − LGR* = {_pp(s['delta'])} on {s.get('tasks', 0)} shared tasks; no diagnosis: {dg.get('reason')}.", ""]
    else:
        L += [f"- Not computable: {dg.get('reason')}.", ""]
    labels = [m for m in MISS_LABELS if any(m in c for c in dg["pipeline_misses"].values())]
    L += ["Where failed samples lost what they needed (`gate_stats.pipeline_miss`):", ""]
    L += [_table(["Arm", *labels], [[a, *(c.get(m, 0) for m in labels)] for a, c in dg["pipeline_misses"].items()])]

    t = d["tables"]
    L += ["## Arms", "", "Primary and diagnostic cells. Success weights cells equally; other columns are means over sample-epochs.", ""]
    cells_all = sorted({c for a in t["arms"].values() for c in a["success_per_cell"]})
    L += [_table(["Arm", "Samples", "Worlds", "Success", *cells_all, "Partial", "Recall", "Recall first", "Recall/step", "Median ctx", "Errors", "Cap hits"],
                 [[a, x["samples"], x["worlds"], _f(x["success"]), *(_f(x["success_per_cell"].get(c)) for c in cells_all), _f(x["partial_credit"]), _f(x["evidence_recall"]), _f(x["evidence_recall_first"]), _f(x["evidence_recall_step_mean"]), _f(x["median_context_tokens"], 0), x["errors"], x["cap_hits"]] for a, x in t["arms"].items()])]  # fmt: skip
    L += ["## Error labels", "", "`scorers/taxonomy.error_analysis`, counts per arm.", ""]
    el = sorted({k for x in t["arms"].values() for k in x["error_labels"]})
    L += [_table(["Arm", *el], [[a, *(x["error_labels"].get(k, 0) for k in el)] for a, x in t["arms"].items()])]

    L += ["## Cost", ""]
    L += [_table(["Arm", "Samples", "$/query", "Inspect $/query", "Embeddings $/query"], [[a, x["samples"], _f(x["usd_per_query"], 5), _f(x["inspect_usd_per_query"], 5), _f(x["embedding_usd_per_query"], 6)] for a, x in t["costs"].items()])]
    L += ["- APG*/LGR* cost ratio by mode: " + ", ".join(f"{m} {_f(x['cost_ratio'], 2)}" for m, x in v["modes"].items()) + f" (flag above {COST_FLAG_RATIO:g}×)."]
    L += [f"- {n}" for n in t["cost_notes"]]
    bc = t["build_costs"]
    if bc.get("per_system"):
        L += ["", _table(["System", "Test worlds", "$ total", "$ per world"], [[s, x["worlds"], _f(x["usd_total"], 2), _f(x["usd_per_world"], 3)] for s, x in bc["per_system"].items()])]
    else:
        L += [f"- Build cost per world: {bc.get('note')}."]
    L += [""]

    L += ["## Latency", ""]
    L += [_table(["Arm", "Total s (median)", "Total s (p95)", "Working s (median)", "Working s (p95)", "Compile ms (median)", "Compile ms (p95)"],
                 [[a, _f(x["total_time_median_s"], 2), _f(x["total_time_p95_s"], 2), _f(x["working_time_median_s"], 2), _f(x["working_time_p95_s"], 2), _f(x["compile_ms_median"], 1), _f(x["compile_ms_p95"], 1)] for a, x in t["latency"].items()])]  # fmt: skip
    L += ["## Determinism", "", "Share of tasks whose epochs all agree on success.", ""]
    L += [_table(["Arm", "Epochs", "Agreement"], [[a, x["epochs"], _f(x["epoch_agreement"]) if x["epoch_agreement"] is not None else "– (one epoch)"] for a, x in t["determinism"].items()])]
    L += ["## exception_applies tasks", "", "F7 cells; reported separately (§3).", ""]
    L += [_table(["Arm", "exception_applies n", "Success", "Other n", "Success"], [[a, x["exception_applies"]["samples"], _f(x["exception_applies"]["success"]), x["other"]["samples"], _f(x["other"]["success"])] for a, x in t["exception_applies"].items()])]

    L += ["## Secondaries", "", "Each reported on its own; never pooled into the verdict.", ""]
    pairs = d["secondary_pairings"]
    for cell, s in d["secondaries"].items():
        L += [f"### {s['title']} (`{cell}`)", ""]
        if s["status"] == "missing":
            L += [f"Missing: {s['reason']}.", ""]
            continue
        if s.get("reason"):
            L += [f"Partial: {s['reason']}.", ""]
        L += [_table(["Arm", "Samples", "Worlds", "Success", "Partial", "Recall", "Median ctx"], [[a, x["samples"], x["worlds"], _f(x["success"]), _f(x["partial_credit"]), _f(x["evidence_recall"]), _f(x["median_context_tokens"], 0)] for a, x in s["arms"].items()])]
        L += [_table(["Comparison", "Δ", "95% CI", "Note"], [[k, _pp(c.get("delta")), _ci(c.get("lo"), c.get("hi")), ("few worlds" if c.get("few_worlds") else "") or (c.get("reason") or "")] for k, c in s["comparisons"].items()])]
        if cell == "gate.sec.id-only" and pairs["id_only_vs_descriptive"]:
            L += ["Against the descriptive rendering of the same seeds:", ""]
            L += [_table(["Arm", "Shared worlds", "Descriptive", "id_only", "Difference"], [[a, x["shared_worlds"], _f(x["descriptive"]), _f(x["id_only"]), _pp(x["id_only_minus_descriptive"])] for a, x in pairs["id_only_vs_descriptive"].items()])]
        if cell == "gate.sec.te-all" and pairs["te_all_vs_retrieved"]:
            L += ["Against TE-retrieved (the primary F3 runs) on shared tasks:", ""]
            L += [_table(["Arm", "Shared tasks", "TE-retrieved", "TE-all"], [[a, x["shared_tasks"], _f(x["retrieved"]), _f(x["all"])] for a, x in pairs["te_all_vs_retrieved"].items()])]
        if cell == "gate.sec.matched-300":
            mt = next((p for p in d["preconditions"] if p["id"] == "PC4"), {}).get("details", {}).get("matched", {})
            arms_ = [a for a in (*CAPPED, APG) if isinstance(mt.get(a), dict)]
            if arms_:
                L += [f"Realized context against the {mt.get('context')}-token budget (reported, not gated; APG* fills and is not capped):", ""]
                L += [_table(["Arm", "Median tokens", "Capped", "Within ±25%"], [[a, _f(mt[a]["median_tokens"], 0), _f(mt[a]["gated"]), _f(mt[a]["pass"]) if mt[a]["pass"] is not None else "–"] for a in arms_])]
            if mt.get("notes"):
                L += ["**Not read as matched** (a capped arm missed the ±25% window; the contrasts above are at unequal context):", ""] + [f"- {n}" for n in mt["notes"]] + [""]

    L += ["## Coverage", "", "Every test group the manifest lists.", ""]
    L += [_table(["Plan cell", "Group", "Status", "Arms", "Samples", "Reason"], [[c["plan_cell"], c["group"], c["status"], ", ".join(c["arms"]), c["samples"], c["reason"] or ""] for c in d["coverage"]])]
    L += ["## Analysis choices", "", "Where GATE_PREREG.md is silent (`ape.analyze_gate.CHOICES`):", ""]
    L += [f"{i}. {c}" for i, c in enumerate(d["choices"], 1)] + [""]
    return "\n".join(L)


def _short(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, dict):
        return ", ".join(f"{k} {_short(v)}" for k, v in value.items())
    if isinstance(value, float):
        return _f(value)
    return str(value)


# --- CLI -------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from . import run_gate as rg
    from .budget import BudgetError
    from .config import ROOT
    from .models import PreflightError

    ap = argparse.ArgumentParser(prog="python -m ape.analyze_gate", description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--config-dir", type=Path, help="default: the config dir the run was created with (run.json)")
    ap.add_argument("--force", action="store_true", help="re-analyze even when the inputs are unchanged")
    ap.add_argument("--extension-of", metavar="STAGE1_RUN_ID", help=f"§8: this run is the one extension of an INCONCLUSIVE gate run; analysed alone at α = {EXTENSION_ALPHA}")
    ap.add_argument("--stage1-runs-dir", type=Path, help="where the stage-1 run lives (default: --runs-dir)")
    a = ap.parse_args(argv)
    info_path = a.runs_dir / a.run_id / "run.json"
    if not info_path.is_file():
        print(f"analyze_gate: {info_path} missing: no such gate run", file=sys.stderr)
        return 1
    info = json.loads(info_path.read_text())
    config_dir = a.config_dir or rg._resolve(info.get("config_dir", "config"))
    if a.extension_of:
        stage1_dir = (a.stage1_runs_dir or a.runs_dir) / a.extension_of
        if a.extension_of == a.run_id or not (stage1_dir / "run.json").is_file():
            print(f"analyze_gate: --extension-of {a.extension_of}: no such stage-1 run, or it is this run", file=sys.stderr)
            return 1
        (a.runs_dir / a.run_id / EXTENSION_FILE).write_text(json.dumps({"stage1_run_id": a.extension_of, "stage1_dir": str(stage1_dir.resolve())}, indent=1))
        a.force = True  # the analysis changes with the extension record
    try:
        run = rg.GateRun(a.run_id, offline=bool(info.get("offline")), force=a.force, runs_root=a.runs_dir, config_dir=config_dir)
        rg.run_phases(run, "analyze")
    except (rg.PhaseError, PreflightError, BudgetError) as e:
        print(f"analyze_gate: {e}", file=sys.stderr)
        return 1
    decision = json.loads((run.dir / REPORT_DIR / "decision.json").read_text())
    print(json.dumps({"run_id": run.run_id, "verdict": decision["verdict"]["label"], "reasons": decision["verdict"]["reasons"], "report": rg._show(run.dir / REPORT_DIR / "report.md")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
