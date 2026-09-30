"""Gate statistics and the GO / NO-GO decision (GATE_PREREG.md is the normative spec).

Estimand: Δ = success(APG-s) − success(LGR*), the mean over gate cells (equal weights)
of the mean paired task-level difference, where a task's success is its mean over
epochs. Inference resamples worlds within cells (world-clustered bootstrap), because
KG build quality varies per world.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GATE_CELLS = ("F7-10", "F7-1000", "F3-5", "F3-60")
MARGIN = 0.05  # δ, absolute
CELL_FLOOR = -0.10  # F7-1000 point estimate must exceed this
INVARIANT_TOL = 0.03
COST_FLAG_RATIO = 2.0
MIN_WORLDS_PER_CELL = 4  # below this a cluster bootstrap understates variance


# ---------- loading ----------

def load_results(log_files: list[str | Path]) -> pd.DataFrame:
    """One row per (sample, epoch) from Inspect eval logs."""
    from inspect_ai.log import read_eval_log

    rows = []
    for f in log_files:
        log = read_eval_log(str(f))
        arm = log.eval.task_args.get("arm") or (log.eval.metadata or {}).get("arm")
        for s in log.samples or []:
            md = s.metadata
            ev = s.scores.get("delivered_evidence")
            usage = {role: u for role, u in (s.role_usage or {}).items()}
            rows.append(
                {
                    "arm": arm,
                    "cell": f"{md['family']}-{md['level']}",
                    "world": md["world_id"],
                    "task": s.id,
                    "epoch": s.epoch,
                    "success": 1.0 if s.scores["task_success"].value == "C" else 0.0,
                    "evidence_recall": ev.value["evidence_recall"] if ev else np.nan,
                    "ctx_tokens": ev.value["ctx_tokens_mean"] if ev else np.nan,
                    "cost_usd": sum((u.total_cost or 0.0) for u in (s.model_usage or {}).values()),
                    "kg_input_tokens": sum(u.input_tokens for r, u in usage.items() if r == "kg"),
                    "total_time": s.total_time,
                    "working_time": s.working_time,
                    "error": s.error is not None,
                }
            )
    return pd.DataFrame(rows)


def task_means(df: pd.DataFrame, value: str = "success") -> pd.DataFrame:
    """Rows (cell, world, task); one column per arm; epoch-averaged."""
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
) -> Decision:
    failed = [k for k, ok in preconditions.items() if not ok]
    delta = pooled_delta(tm, "APG-s", lgr_arm)
    boot = cluster_bootstrap(tm, "APG-s", lgr_arm, reps=reps, seed=seed)
    lo, hi = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    if failed:
        return Decision("PRECONDITION_FAIL", delta, (lo, hi), [f"precondition failed: {k}" for k in failed])

    cell = cell_deltas(tm, "APG-s", lgr_arm)
    s7 = world_diffs(tm, "APG-s", "S7") if "S7" in tm else np.array([])
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
        reasons.append(f"APG-s not better than random-node placebo S7 (p={p_s7:.3f})")
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
