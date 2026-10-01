"""Gate statistics and the GO / NO-GO decision (GATE_PREREG.md is the normative spec).

Estimand: Δ = success(APG-s) − success(LGR*), the mean over gate cells (equal weights)
of the mean paired task-level difference, where a task's success is its mean over
epochs. Inference resamples worlds within cells (world-clustered bootstrap), because
KG build quality varies per world.
"""

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

GATE_CELLS = ("F7-10", "F7-1000", "F3-5", "F3-60")
MARGIN = 0.05  # δ, absolute
CELL_FLOOR = -0.10  # F7-1000 point estimate must exceed this
INVARIANT_TOL = 0.03
COST_FLAG_RATIO = 2.0
MIN_WORLDS_PER_CELL = 4  # below this a cluster bootstrap understates variance
DEFAULT_MAX_TURNS = 12  # Config.max_turns without APE_MAX_TURNS
DEFAULT_BUDGET = 2000  # config.DEFAULT_CONTEXT_BUDGET


# ---------- loading ----------

def load_results(log_files: list[str | Path], require_cost: bool = True) -> pd.DataFrame:
    """One row per (sample, epoch) from Inspect eval logs.

    Cost comes from Inspect's model-cost config. A missing price would otherwise read as $0
    and silently disable the cost flag, so it is an error unless `require_cost=False`
    (offline dry runs only).

    Besides the outcome columns, each row carries what the decision report (FX-7) needs: the turn cap and
    whether it was hit (PC5), the arm's configured context budget from the task's recorded knobs and the
    realized tokens and latency of every compile (PC4, latency), and `pipeline_miss`, where a failed
    sample lost the gold knowledge (§8 NO-GO diagnosis; `pipeline_miss`).
    """
    from inspect_ai.log import read_eval_log

    rows = []
    for f in log_files:
        log = read_eval_log(str(f))
        args, emd = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or emd.get("arm")
        knobs = emd.get("knobs") or {}
        max_turns = int(knobs.get("APE_MAX_TURNS", DEFAULT_MAX_TURNS))
        for s in log.samples or []:
            md = s.metadata
            scores = s.scores or {}
            ev = scores.get("delivered_evidence")
            ts = scores.get("task_success")
            ts_md = (ts.metadata or {}) if ts is not None else {}
            usage = {role: u for role, u in (s.role_usage or {}).items()}
            missing = [m for m, u in (s.model_usage or {}).items() if u.total_cost is None]
            if missing and require_cost:
                raise ValueError(f"{f}: no cost for {missing}; run eval() with **ape.models.eval_cost_kwargs() (config/model_costs.yaml)")
            cell = f"{md['family']}-{md['level']}"
            store = s.store or {}
            compiles = store.get("compile_log", [])
            success = 1.0 if ts is not None and ts.value == "C" else 0.0
            turns, answered = ts_md.get("turns_used"), ts_md.get("answered")
            rows.append(
                {
                    "arm": arm,
                    "cell": cell,
                    "world": md["world_id"],
                    "task": s.id,
                    "epoch": s.epoch,
                    # An errored sample (tolerated under fail_on_error, PC5) has no scores. Primary analysis counts
                    # it as a failure (GATE_PREREG §2); the `error` column lets the sensitivity analysis exclude it.
                    "success": success,
                    "evidence_recall": ev.value["evidence_recall"] if ev else np.nan,
                    "evidence_recall_first": ev.value.get("evidence_recall_first", np.nan) if ev else np.nan,
                    "evidence_recall_step_mean": ev.value.get("evidence_recall_step_mean", np.nan) if ev else np.nan,
                    "partial_credit": scores["error_analysis"].value["partial_credit"] if "error_analysis" in scores else np.nan,
                    "error_label": scores["error_analysis"].metadata.get("error") if "error_analysis" in scores else ("harness_error" if s.error else None),
                    "case": md.get("case"),
                    "delivery": args.get("delivery", "push"),
                    "exposure": args.get("exposure", emd.get("exposure", "retrieved")),
                    "split": md.get("split", args.get("split")),
                    "exception_style": md.get("exception_style"),
                    "ctx_tokens": ev.value["ctx_tokens_mean"] if ev else np.nan,
                    "compile_tokens": [int(r["tokens"]) for r in compiles],
                    "compile_ms": [float(r["compile_ms"]) for r in compiles if r.get("compile_ms") is not None],
                    "budget": configured_budget(arm, knobs, cell),
                    "turns_used": turns,
                    "max_turns": max_turns,
                    # PC5: the agent ran out of turns without answering (GATE_PREREG §7).
                    "cap_hit": s.error is None and turns is not None and turns >= max_turns and not answered,
                    "pipeline_miss": None if success or s.error else pipeline_miss(compiles, store.get("step_log", []), md.get("task") or {}),
                    "cost_usd": sum((u.total_cost or 0.0) for u in (s.model_usage or {}).values()),
                    "kg_input_tokens": sum(u.input_tokens for r, u in usage.items() if r == "kg"),
                    "total_time": s.total_time,
                    "working_time": s.working_time,
                    "error": s.error is not None,
                }
            )
    return pd.DataFrame(rows)


def _s7_targets(path: str) -> dict:
    p = Path(path)
    return _read_targets(str(p), p.stat().st_mtime_ns) if p.is_file() else {}


@cache
def _read_targets(path: str, _mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text())


def _knob(knobs: dict, *names: str, default: int = DEFAULT_BUDGET) -> int:
    for name in names:
        raw = str(knobs.get(name, "")).strip()
        if raw:
            return int(raw)
    return default


def configured_budget(arm: str | None, knobs: dict, cell: str) -> int | None:
    """The context budget an arm was configured with, from the knobs its task recorded (`tasks.gate`), with the
    precedence of `ape.config` (per-arm knob, then APE_CONTEXT_BUDGET, then 2000). S7's budget is its frozen
    per-cell target (`agent.arms.s7_target`). S1 (whole corpus) and S6 (gold facts) have none."""
    arm = arm or ""
    if arm == "S3s":
        return _knob(knobs, "APE_S3S_BUDGET", "APE_CONTEXT_BUDGET")
    if arm.startswith("APG") or arm == "S5o":
        return _knob(knobs, "APE_APG_BUDGET", "APE_CONTEXT_BUDGET")
    if arm.startswith("LGR"):
        return _knob(knobs, "APE_LGR_BUDGET", "APE_CONTEXT_BUDGET")
    if arm == "S7":
        targets = _s7_targets(knobs["APE_S7_TARGETS"]) if knobs.get("APE_S7_TARGETS") else {}
        return int(targets[cell]) if cell in targets else _knob(knobs, "APE_S7_TARGET", "APE_CONTEXT_BUDGET")
    return None


# Where a failed sample lost the gold knowledge, in pipeline order (GATE_PREREG §8 NO-GO diagnosis). APG's
# compiles record the gold nodes' fate (`apg.arm`: in_shortlist, in_matches, in_contributors); other arms only
# whether every gold fact was delivered.
MISS_LABELS = (
    "not_in_graph",  # APG: no node of the graph carries a gold fact (extraction)
    "shortlist_miss",  # APG: no gold node in the embedding shortlist
    "classify_miss",  # APG: shortlisted, but the classify call did not match it
    "compose_miss_truncation",  # APG: matched, but cut by the token budget
    "compose_miss",  # APG: matched, but compose left it out (descendant / slot)
    "evidence_miss",  # other arms: some gold fact never delivered
    "tool_exposure_miss",  # F3: a tool the procedure needs was never exposed
    "delivered_but_failed",  # the knowledge and tools were there; the agent failed
)


def pipeline_miss(compiles: list[dict], steps: list[dict], task: dict) -> str:
    """The first stage at which a failed sample lost what it needed (MISS_LABELS), from its compile and step logs."""
    golds = [r["meta"]["gold"] for r in compiles if isinstance((r.get("meta") or {}).get("gold"), dict)]
    if golds:
        if not any(g.get("nodes") for g in golds):
            return "not_in_graph"
        if not any(g.get("in_shortlist") for g in golds):
            return "shortlist_miss"
        if not any(g.get("in_matches") for g in golds):
            return "classify_miss"
        if not any(g.get("in_contributors") for g in golds):
            return "compose_miss_truncation" if any((r.get("meta") or {}).get("truncated") for r in compiles) else "compose_miss"
    else:
        gold_facts = set(task.get("gold_fact_ids") or [])
        delivered = {f for r in compiles for f in r.get("fact_ids") or []}
        if gold_facts and not gold_facts <= delivered:
            return "evidence_miss"
    needed = {c["tool"] for c in (task.get("gold") or {}).get("calls", []) if isinstance(c, dict) and "tool" in c}
    exposed = {t for st in steps for t in st.get("exposed_tools") or []}
    if needed and steps and not needed <= exposed:
        return "tool_exposure_miss"
    return "delivered_but_failed"


def task_means(df: pd.DataFrame, value: str = "success") -> pd.DataFrame:
    """Rows (cell, world, task); one column per arm; epoch-averaged.

    An arm's runs in one cell must share one delivery mode: push and pull are co-primary and analysed separately
    (GATE_PREREG §3), so averaging them into one task mean is refused. Select a mode, or label arms by mode."""
    if "delivery" in df.columns and len(df):
        modes = df.groupby(["arm", "cell"])["delivery"].nunique()
        if (mixed := modes[modes > 1]).size:
            raise ValueError(f"task_means would mix delivery modes for {sorted(map(tuple, mixed.index))}: select one mode per arm and cell")
    return df.groupby(["cell", "world", "task", "arm"])[value].mean().unstack("arm")


# ---------- estimation ----------

def cell_deltas(tm: pd.DataFrame, a: str, b: str) -> pd.Series:
    d = (tm[a] - tm[b]).dropna()
    return d.groupby(level="cell").mean()


def pooled_delta(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> float:
    return float(cell_deltas(tm, a, b).reindex(cells).mean())


def cluster_bootstrap(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS, reps: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the pooled Δ, resampling worlds within each cell (vectorized)."""
    rng = np.random.default_rng(seed)
    d = (tm[a] - tm[b]).dropna()
    per_cell = []
    for c in cells:
        dc = d.xs(c, level="cell")
        g = dc.groupby(level="world")
        sums, counts = g.sum().to_numpy(), g.count().to_numpy()
        w = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=reps)
        per_cell.append((w @ sums) / (w @ counts))
    return np.mean(per_cell, axis=0)


def sign_flip_p(d: np.ndarray, reps: int = 10_000, seed: int = 0) -> float:
    """One-sided p-value for mean(d) > 0 by random sign flips of independent units."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    null = (flips * d).mean(axis=1)
    return float((1 + np.sum(null >= d.mean())) / (reps + 1))


def world_diffs(tm: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Per-world mean paired difference: the independent units for clustered sign-flip tests."""
    return (tm[a] - tm[b]).dropna().groupby(level=["cell", "world"]).mean().to_numpy()


def pooled_success(tm: pd.DataFrame, arm: str, cells=GATE_CELLS) -> float:
    return float(tm[arm].dropna().groupby(level="cell").mean().reindex(cells).mean())


# ---------- decision ----------

@dataclass
class Decision:
    verdict: str  # GO | GO_WITH_COST_FLAG | INCONCLUSIVE | NO_GO | PRECONDITION_FAIL
    delta: float
    ci: tuple[float, float]
    reasons: list[str] = field(default_factory=list)
    superiority: bool = False
    details: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=1, default=float)


def decide(
    tm: pd.DataFrame,
    lgr_arm: str,
    preconditions: dict[str, bool],
    cost_ratio: float | None = None,
    alpha: float = 0.025,
    reps: int = 10_000,
    seed: int = 0,
    apg_arm: str = "APG-s",
) -> Decision:
    """`apg_arm` / `lgr_arm` are the dev-selected configurations (APG*, LGR*; GATE_PREREG §5)."""
    failed = [k for k, ok in preconditions.items() if not ok]
    delta = pooled_delta(tm, apg_arm, lgr_arm)
    boot = cluster_bootstrap(tm, apg_arm, lgr_arm, reps=reps, seed=seed)
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    if failed:
        return Decision("PRECONDITION_FAIL", delta, (lo, hi), [f"precondition failed: {k}" for k in failed])

    cell = cell_deltas(tm, apg_arm, lgr_arm)
    s7 = world_diffs(tm, apg_arm, "S7") if "S7" in tm else np.array([])
    p_s7 = sign_flip_p(s7, reps, seed) if len(s7) else 1.0
    details = {"cell_deltas": cell.to_dict(), "p_apg_gt_s7": p_s7, "cost_ratio": cost_ratio}
    idx = tm.index.to_frame(index=False)
    worlds = idx.groupby("cell")["world"].nunique().reindex(GATE_CELLS).fillna(0)
    if worlds.min() < MIN_WORLDS_PER_CELL:
        return Decision("INCONCLUSIVE", delta, (lo, hi), [f"too few worlds per cell for clustered inference ({int(worlds.min())} < {MIN_WORLDS_PER_CELL})"], details=details)

    reasons = []
    if lo <= -MARGIN:
        if hi >= 0:
            return Decision("INCONCLUSIVE", delta, (lo, hi), ["CI spans both -margin and 0"], details=details)
        reasons.append(f"non-inferiority not shown: lower bound {lo:+.3f} <= -{MARGIN}")
    if cell.get("F7-1000", 0.0) <= CELL_FLOOR:
        reasons.append(f"F7-1000 cell Δ {cell['F7-1000']:+.3f} <= {CELL_FLOOR}")
    if p_s7 >= 0.05:
        reasons.append(f"{apg_arm} not better than random-node placebo S7 (p={p_s7:.3f})")
    if reasons:
        return Decision("NO_GO", delta, (lo, hi), reasons, details=details)
    verdict = "GO_WITH_COST_FLAG" if cost_ratio is not None and cost_ratio > COST_FLAG_RATIO else "GO"
    return Decision(verdict, delta, (lo, hi), [], superiority=lo > 0, details=details)


def invariants(tm: pd.DataFrame, chain=("S6", "S5o", "APG-s", "S7"), tol: float = INVARIANT_TOL) -> dict:
    """PC3: each arm in the chain may exceed its predecessor by at most `tol`; S6 > S7 significantly."""
    present = [a for a in chain if a in tm]
    succ = {a: pooled_success(tm, a) for a in present}
    pairs = {f"{hi}>={lo}": succ[hi] + tol >= succ[lo] for hi, lo in zip(present, present[1:])}
    ok_s6 = True
    if {"S6", "S7"} <= set(present):
        ok_s6 = sign_flip_p(world_diffs(tm, "S6", "S7")) < 0.05
    return {"success": succ, "pairs": pairs, "s6_gt_s7_significant": ok_s6, "pass": all(pairs.values()) and ok_s6}


# ---------- report helpers (FX-7) ----------


def delta_ci(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS, alpha: float = 0.025, reps: int = 10_000, seed: int = 0) -> dict:
    """Δ = a − b pooled over `cells` (equal weights) with its world-clustered bootstrap CI (the 100α and
    100(1−α) percentiles), on the tasks both arms have. Cells without paired data are left out and named;
    with none, the result has `delta` None and a `reason`."""
    out: dict = {"a": a, "b": b, "cells": list(cells)}
    if a not in tm or b not in tm:
        return out | {"delta": None, "lo": None, "hi": None, "reason": f"no data for {[x for x in (a, b) if x not in tm]}"}
    d = (tm[a] - tm[b]).dropna()
    d = d[d.index.get_level_values("cell").isin(list(cells))]
    present = [c for c in cells if c in set(d.index.get_level_values("cell"))]
    worlds = d.groupby(level="cell").apply(lambda x: x.index.get_level_values("world").nunique()).to_dict()
    out |= {"missing_cells": [c for c in cells if c not in present], "worlds_per_cell": {c: int(worlds.get(c, 0)) for c in cells}, "tasks": int(len(d))}
    if not present:
        return out | {"delta": None, "lo": None, "hi": None, "reason": f"no paired tasks of {a} and {b} in {list(cells)}"}
    boot = cluster_bootstrap(tm, a, b, cells=present, reps=reps, seed=seed)
    lo, hi = (float(x) for x in np.quantile(boot, [alpha, 1 - alpha]))
    return out | {"delta": pooled_delta(tm, a, b, present), "lo": lo, "hi": hi, "few_worlds": min(worlds.get(c, 0) for c in present) < MIN_WORLDS_PER_CELL}


def shared_tasks(tm: pd.DataFrame, arms) -> pd.DataFrame:
    """The task means of `arms` on the tasks every one of them has (so their successes compare like with like)."""
    cols = [a for a in arms if a in tm]
    return tm[cols].dropna() if cols else tm.iloc[0:0]


def epoch_agreement(df: pd.DataFrame) -> float | None:
    """Determinism: the share of (task, arm) pairs with two or more epochs whose epochs all agree on success."""
    multi = df.groupby(["arm", "cell", "world", "task"])["success"].agg(["count", "nunique"])
    multi = multi[multi["count"] > 1]
    return float((multi["nunique"] == 1).mean()) if len(multi) else None


GO_VERDICTS = ("GO", "GO_WITH_COST_FLAG")


def combine_modes(verdicts: dict[str, str]) -> tuple[str, list[str]]:
    """The gate's label from per-delivery-mode verdicts (GATE_PREREG §3, §8), {"push": ..., "pull": ...}:

    - PRECONDITION_FAIL in any mode → PRECONDITION_FAIL;
    - GO (or GO_WITH_COST_FLAG) in both → GO, or GO_WITH_COST_FLAG if either mode is cost-flagged;
    - GO in one mode only → GO_PUSH_ONLY / GO_PULL_ONLY (the user decides; the other mode's verdict is a reason);
    - otherwise INCONCLUSIVE if any mode is INCONCLUSIVE (the one pre-registered extension applies), else NO_GO.
    """
    if any(v == "PRECONDITION_FAIL" for v in verdicts.values()):
        return "PRECONDITION_FAIL", [f"{m}: {v}" for m, v in verdicts.items() if v == "PRECONDITION_FAIL"]
    go = [m for m, v in verdicts.items() if v in GO_VERDICTS]
    if len(go) == len(verdicts):
        flagged = [m for m, v in verdicts.items() if v == "GO_WITH_COST_FLAG"]
        return ("GO_WITH_COST_FLAG", [f"cost flag in {flagged}"]) if flagged else ("GO", [])
    if go:
        other = {m: v for m, v in verdicts.items() if m not in go}
        label = f"GO_{go[0].upper()}_ONLY" if len(go) == 1 else "GO"
        reasons = [f"{m}: {v}" for m, v in other.items()] + [f"{m}: cost flag" for m in go if verdicts[m] == "GO_WITH_COST_FLAG"]
        return label, reasons
    if any(v == "INCONCLUSIVE" for v in verdicts.values()):
        return "INCONCLUSIVE", [f"{m}: {v}" for m, v in verdicts.items()]
    return "NO_GO", [f"{m}: {v}" for m, v in verdicts.items()]
