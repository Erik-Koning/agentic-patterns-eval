"""Main-study confirmatory statistics (D-029; ORCHESTRATOR_BRIEF_v2 §7.4): design-based, paired on tasks, clustered by
world. `main_hypotheses` says what is tested; this module says how.

**Unit and estimand.** Every contrast is a per-task value built from task-level epoch means (S1's first `s1_epochs`
pool runs, so it has the 3 epochs every arm has): a paired difference (M1 − S9), a difference in differences (H2's
2×2, T1's arm × tier) or an arm against the S8 frontier at that arm's realised cost (`frontier`). A member pools its
cells with equal weights; within a cell the estimate is the mean over its paired tasks (the gate's ratio estimator).

**Clusters.** Tasks within a world share their KB and its KG build (the gate's finding), so worlds are the
independent units. F1 and F2 worlds of one seed share one supplier registry (`gen_registry`: same KB at every F1/F2
level), so with `cluster_by="kb"` (default) they are one cluster across Study A's cells; F3 and F7 worlds are their
own clusters. A cluster contributes s_g = Σ_c λ_c Σ_{t ∈ g, c} d_t / N_c to the estimate (λ_c the cell weight, N_c
the cell's paired tasks), so the estimate is Σ_g s_g.

**Test** (the decision): a sign-flip permutation test over clusters, as the brief prescribes (§7.4): under H0 each
cluster's contribution is symmetric about the null, so flipping signs gives the null distribution of Σ_g s_g.
Exact (all 2^G flips) for G ≤ EXACT_MAX_CLUSTERS, else Monte Carlo with a fixed sign matrix. Non-zero nulls (NI,
TOST) shift each cluster by δ·w_g, w_g = Σ_c λ_c N_gc / N_c (Σ_g w_g = 1), so the shifted statistic is est − δ. The
smallest attainable one-sided p is 2^−G: 0.00195 at 9 worlds (`min_p`). Superiority under the sharp null of
exchangeable arms is exact; shifted nulls, DiD and frontier contrasts rest on approximate symmetry of the cluster
sums, which `main_power` checks by simulating type I error at each null boundary. **The condition is symmetric world
contributions** (BUILD_REVIEW S-3): worlds that are catastrophic for one arm only (a broken KG build, say) skew the
cluster sums, and the level then drifts above nominal (independent review: 0.075 for K1 and M5 with 15% of worlds at
−3 logit for one arm). So every member reports its per-world influence (`world_influence`: contribution and the
member without each world), and a supported claim that one world carries, or that rests on a world whose KG build
failed its quality check, is a §8 caveat (`caveats`).

**Frontier contrasts** carry the noise of the matched cost (`_matched_cost_influence`, BUILD_REVIEW S-2): S8 is
interpolated at the arm's realised mean cost over S1's mean run cost, an estimate from the same worlds, so each task's
delta-method influence on that weight is added to its contrast value (centred: the estimate is unchanged).

**Intervals** (two-sided 1 − 2α, so one-sided level-α bounds; reported, never the decision):
- `ci`: the sign-flip test inverted in δ (p is monotone in δ, so a bisection finds the bounds). It agrees with the
  decision by construction: a TOST rejects exactly when this interval lies inside ±margin;
- `ci_boot`: the world-clustered bootstrap, BCa (brief §7.4: "10k, BCa"). Worlds are resampled within cells when
  every cluster sits in one cell, else across. BCa over percentile: success rates near 0 or 1 and ratio estimators
  make the bootstrap distribution skewed and biased, which BCa's bias and acceleration terms correct; neither fixes the
  small-cluster undercoverage (the gate's percentile bootstrap gave 3-4% at a nominal 2.5% with 16 worlds, D-023),
  which is why it is not the decision;
- `ci_t`: the gate's world-clustered t (ratio estimator, linearised cluster variance, Satterthwaite df over cells;
  with clusters spanning cells, CR1 with G − 1 df), as a cross-check, with its p-value `p_t`.

**Multiplicity:** Holm within each family (`gate_stats.holm_test`: the same stopping rule as the gate), serial
gatekeeping across stages.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

from . import frontier as fr
from .gate_stats import holm_test
from .main_hypotheses import ALPHA, PRIMARY_METER, PRIMARY_TIER, Hypothesis, Member, Term
from .main_load import METERS

EXACT_MAX_CLUSTERS = 20  # 2^20 sums is ~8 MB: exact enumeration up to here, Monte Carlo above
MIN_CLUSTERS = 4  # below this a contrast is flagged `few_clusters` (the test stays valid, just weak)
DETERMINISM_PREFIX = "main.C."  # Study C plan cells: determinism only, never in the confirmatory frame
S1_EPOCHS = 3  # S1's main epochs are the first 3 of its 8-run pool (brief §4.5)
REPS = 10_000
BOUND_SPAN = 4.0  # |contrast| <= 4 for any 4-term contrast of success rates: the bisection bracket


# --------------------------------------------------------------------------- clusters and contrasts


def cluster_of(world: str, by: str = "kb") -> str:
    """The independent unit of a world: with `by="kb"`, F1/F2 worlds of one seed share one cluster
    ("F1-32-test-s4001" -> "REG-test-s4001"); every other world is its own. `by="world"`: the world itself."""
    if by == "world":
        return str(world)
    m = re.match(r"^F[12]-\d+-(.+)$", str(world))
    return f"REG-{m.group(1)}" if m else str(world)


@dataclass
class Contrast:
    """Per-task contrast values with their cell and cluster, and the cell weights λ_c."""

    d: np.ndarray
    cell: np.ndarray
    cluster: np.ndarray
    cells: list
    clusters: list
    weights: np.ndarray
    index: pd.MultiIndex | None = None

    def __post_init__(self):
        G, K = len(self.clusters), len(self.cells)
        self.Y = np.zeros((G, K))
        self.N = np.zeros((G, K))
        np.add.at(self.Y, (self.cluster, self.cell), self.d)
        np.add.at(self.N, (self.cluster, self.cell), 1.0)
        self.Nc = self.N.sum(axis=0)
        self.nested = bool(np.all((self.N > 0).sum(axis=1) <= 1))

    @classmethod
    def from_series(cls, d: pd.Series, cluster_by: str = "kb", weights: dict | None = None) -> "Contrast":
        """From a Series indexed (cell, world, task). `weights` {cell: λ}; default equal weights over the cells present."""
        d = d.dropna()
        cells_v = d.index.get_level_values("cell").astype(str)
        clus_v = np.array([cluster_of(w, cluster_by) for w in d.index.get_level_values("world")])
        cells = list(dict.fromkeys(cells_v))
        clusters = list(dict.fromkeys(clus_v))
        ci = {c: i for i, c in enumerate(cells)}
        gi = {g: i for i, g in enumerate(clusters)}
        lam = np.array([weights[c] for c in cells], dtype=float) if weights else np.full(len(cells), 1.0 / max(1, len(cells)))
        return cls(d.to_numpy(dtype=float), np.array([ci[c] for c in cells_v], dtype=int), np.array([gi[g] for g in clus_v], dtype=int), cells, clusters, lam, d.index)

    @property
    def G(self) -> int:
        return len(self.clusters)

    @property
    def est(self) -> float:
        return float(np.sum(self.weights * self.Y.sum(axis=0) / self.Nc)) if len(self.d) else float("nan")

    def contributions(self) -> tuple[np.ndarray, np.ndarray]:
        """(s_g, w_g): each cluster's share of the estimate and of a unit shift (module docstring)."""
        return (self.Y / self.Nc) @ self.weights, (self.N / self.Nc) @ self.weights

    def clusters_per_cell(self) -> dict:
        return {c: int((self.N[:, i] > 0).sum()) for i, c in enumerate(self.cells)}


# --------------------------------------------------------------------------- sign flip


def _flip_sums(a: np.ndarray) -> np.ndarray:
    """All 2^G sums Σ ε_g a_g, ε ∈ {±1}^G (iterative doubling)."""
    dist = np.zeros(1)
    for x in a:
        dist = np.concatenate([dist + x, dist - x])
    return dist


class SignFlip:
    """The clustered sign-flip test of one contrast; exact up to `exact_max` clusters, else Monte Carlo with one fixed
    sign matrix (so p is monotone in the shift δ and the inverted interval is well defined)."""

    def __init__(self, c: Contrast, reps: int = REPS, seed: int = 0, exact_max: int = EXACT_MAX_CLUSTERS):
        self.s, self.w = c.contributions()
        self.G, self.reps = c.G, reps
        self.exact = self.G <= exact_max
        self.signs = None if self.exact else np.random.default_rng(seed).choice([-1.0, 1.0], size=(reps, self.G))
        self.W = float(self.w.sum())

    @property
    def method(self) -> str:
        return f"sign-flip, exact (2^{self.G} = {2**self.G:,} flips)" if self.exact else f"sign-flip, Monte Carlo ({self.reps:,} flips of {self.G} clusters)"

    @property
    def min_p(self) -> float:
        return 2.0**-self.G if self.exact else 1.0 / (self.reps + 1)

    def p(self, delta: float, alternative: str) -> float:
        """One-sided p for H0: Δ ≤ δ (`greater`) or H0: Δ ≥ δ (`less`)."""
        a = self.s - delta * self.w
        T = float(a.sum())
        tol = 1e-12 * max(1.0, float(np.abs(a).sum()))
        if self.exact:
            dist = _flip_sums(a)
            hits = np.count_nonzero(dist >= T - tol) if alternative == "greater" else np.count_nonzero(dist <= T + tol)
            return float(hits / dist.size)
        dist = self.signs @ a
        hits = np.count_nonzero(dist >= T - tol) if alternative == "greater" else np.count_nonzero(dist <= T + tol)
        return float((1 + hits) / (self.reps + 1))

    def bound(self, alpha: float, side: str, est: float) -> float | None:
        """The level-α one-sided bound by test inversion: lower = sup{δ: p_greater(δ) ≤ α}, upper = inf{δ: p_less(δ) ≤ α}.
        None when no δ in the bracket is rejected (too few clusters for the level) or there is no shift (W = 0)."""
        if self.W <= 1e-12:
            return None
        if side == "lower":
            lo, hi = est - BOUND_SPAN, est
            if self.p(lo, "greater") > alpha:
                return None
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if self.p(mid, "greater") <= alpha else (lo, mid)
            return lo
        lo, hi = est, est + BOUND_SPAN
        if self.p(hi, "less") > alpha:
            return None
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if self.p(mid, "less") <= alpha else (mid, hi)
        return hi


# --------------------------------------------------------------------------- cluster-t and bootstrap


def cluster_t(c: Contrast) -> dict:
    """est, se and df of the world-clustered t (module docstring); se None when a cell has fewer than 2 clusters."""
    out = {"est": c.est, "se": None, "df": None}
    if not len(c.d):
        return out
    r = c.Y.sum(axis=0) / c.Nc
    resid = c.weights * (c.Y - r * c.N) / c.Nc  # (G, K)
    if c.nested:
        m = (c.N > 0).sum(axis=0)
        if np.any(m < 2):
            return out | {"reason": "fewer than 2 clusters in a cell"}
        var_c = m / (m - 1) * (resid**2).sum(axis=0)
        total = float(var_c.sum())
        df = total**2 / float(np.sum(var_c[var_c > 0] ** 2 / (m[var_c > 0] - 1))) if total > 0 else None
    else:
        G = c.G
        if G < 2:
            return out | {"reason": "fewer than 2 clusters"}
        total = float(G / (G - 1) * np.sum(resid.sum(axis=1) ** 2))
        df = float(G - 1)
    return out | {"se": math.sqrt(total), "df": df}


def t_p(st: dict, delta: float, alternative: str) -> float | None:
    if st.get("se") is None:
        return None
    if st["se"] == 0 or st["df"] is None:
        hit = st["est"] > delta if alternative == "greater" else st["est"] < delta
        return 0.0 if hit else 1.0
    z = (st["est"] - delta) / st["se"]
    return float(student_t.sf(z, st["df"]) if alternative == "greater" else student_t.cdf(z, st["df"]))


def t_interval(st: dict, alpha: float) -> list:
    if st.get("se") is None:
        return [None, None]
    if st["se"] == 0 or st["df"] is None:
        return [st["est"], st["est"]]
    q = float(student_t.ppf(1 - alpha, st["df"]))
    return [st["est"] - q * st["se"], st["est"] + q * st["se"]]


def _resample_weights(c: Contrast, reps: int, rng: np.random.Generator) -> np.ndarray:
    """(reps, G) cluster multiplicities: worlds resampled within each cell when clusters are nested in cells, else
    resampled across all clusters."""
    G = c.G
    if not c.nested:
        return rng.multinomial(G, np.full(G, 1 / G), size=reps).astype(float)
    W = np.zeros((reps, G))
    home = (c.N > 0).argmax(axis=1)
    for k in range(len(c.cells)):
        idx = np.flatnonzero(home == k)
        if len(idx):
            W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=reps)
    return W


def _estimates(c: Contrast, W: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return ((W @ c.Y) / (W @ c.N)) @ c.weights


def _jackknife(c: Contrast) -> np.ndarray:
    """Leave-one-cluster-out estimates (NaN where a cell would lose every cluster)."""
    W = 1.0 - np.eye(c.G)
    return _estimates(c, W)


def bca(theta: float, boot: np.ndarray, jack: np.ndarray, alpha: float) -> list:
    """The BCa interval (two-sided 1 − 2α) from bootstrap replicates and jackknife estimates."""
    boot, jack = boot[np.isfinite(boot)], jack[np.isfinite(jack)]
    if not len(boot) or not np.isfinite(theta):
        return [None, None]
    if np.ptp(boot) == 0:
        return [float(boot[0]), float(boot[0])]
    prop = (np.count_nonzero(boot < theta) + 0.5 * np.count_nonzero(boot == theta)) / len(boot)
    z0 = float(norm.ppf(min(max(prop, 1e-6), 1 - 1e-6)))
    a = 0.0
    if len(jack) >= 3:
        dev = jack.mean() - jack
        den = 6.0 * float(np.sum(dev**2)) ** 1.5
        a = float(np.sum(dev**3) / den) if den > 0 else 0.0
    out = []
    for q in (alpha, 1 - alpha):
        z = norm.ppf(q)
        adj = float(norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))
        out.append(float(np.quantile(boot, min(max(adj, 0.0), 1.0))))
    return out


def bootstrap_ci(c: Contrast, alpha: float, reps: int = REPS, seed: int = 0, kind: str = "bca") -> list:
    """The world-clustered bootstrap interval of the contrast (BCa, or `kind="percentile"`)."""
    if c.G < 2:
        return [None, None]
    boot = _estimates(c, _resample_weights(c, reps, np.random.default_rng(seed)))
    if kind == "percentile":
        b = boot[np.isfinite(boot)]
        return [float(np.quantile(b, alpha)), float(np.quantile(b, 1 - alpha))] if len(b) else [None, None]
    return bca(c.est, boot, _jackknife(c), alpha)


def ratio_ci(num: Contrast, den: Contrast, alpha: float, reps: int = REPS, seed: int = 0) -> dict:
    """est(num) / est(den) for two contrasts on the same tasks (e.g. two arms' mean costs), with its world-clustered
    BCa interval (both resampled with the same cluster draws)."""
    ratio = num.est / den.est if den.est else float("nan")
    if num.G < 2:
        return {"ratio": ratio, "ci": [None, None]}
    W = _resample_weights(num, reps, np.random.default_rng(seed))
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = _estimates(num, W) / _estimates(den, W)
        J = 1.0 - np.eye(num.G)
        jack = _estimates(num, J) / _estimates(den, J)
    return {"ratio": float(ratio), "ci": bca(ratio, boot, jack, alpha)}


# --------------------------------------------------------------------------- inference on one contrast


def p_value(flip: SignFlip, test: str, margin: float) -> tuple[float | None, dict]:
    """The decision p of `test` from the sign-flip test, with its one-sided parts."""
    if test == "superiority":
        p = flip.p(0.0, "greater")
        return p, {"greater_0": p}
    if test == "less":
        p = flip.p(0.0, "less")
        return p, {"less_0": p}
    if test == "ni":
        p = flip.p(-margin, "greater")
        return p, {f"greater_-{margin:g}": p}
    if test == "tost":
        lo, hi = flip.p(-margin, "greater"), flip.p(margin, "less")
        return max(lo, hi), {f"greater_-{margin:g}": lo, f"less_+{margin:g}": hi}
    return None, {}


def _t_decision_p(st: dict, test: str, margin: float) -> float | None:
    if test == "superiority":
        return t_p(st, 0.0, "greater")
    if test == "less":
        return t_p(st, 0.0, "less")
    if test == "ni":
        return t_p(st, -margin, "greater")
    if test == "tost":
        a, b = t_p(st, -margin, "greater"), t_p(st, margin, "less")
        return None if a is None or b is None else max(a, b)
    return None


def infer(c: Contrast, test: str, margin: float = 0.0, alpha: float = ALPHA, reps: int = REPS, seed: int = 0, ci: bool = True, exact_max: int = EXACT_MAX_CLUSTERS) -> dict:
    """Estimate, decision p (sign flip), cross-check p (cluster-t) and the three intervals of one contrast.
    `ci=False` skips the intervals (the power simulation needs only the decision)."""
    flip = SignFlip(c, reps=reps, seed=seed, exact_max=exact_max)
    est = c.est
    p, parts = p_value(flip, test, margin)
    st = cluster_t(c)
    out = {
        "est": est,
        "tasks": int(len(c.d)),
        "clusters": c.G,
        "clusters_per_cell": c.clusters_per_cell(),
        "cells": list(c.cells),
        "few_clusters": c.G < MIN_CLUSTERS,
        "method": flip.method,
        "min_p": flip.min_p,
        "p": p,
        "p_parts": parts,
        "p_t": _t_decision_p(st, test, margin),
        "se_t": st.get("se"),
        "df_t": st.get("df"),
        "level": alpha,
    }
    if ci:
        out["ci"] = [flip.bound(alpha, "lower", est), flip.bound(alpha, "upper", est)]
        out["ci_boot"] = bootstrap_ci(c, alpha, reps=reps, seed=seed)
        out["ci_t"] = t_interval(st, alpha)
    return out


# --------------------------------------------------------------------------- tables from the tidy frame


@dataclass
class Pool:
    """One condition's S1 pool: per task (rows of `index`) the first K runs' answer codes, success and costs."""

    index: pd.MultiIndex
    codes: np.ndarray
    success: np.ndarray
    cost: dict
    K: int
    short_tasks: int = 0

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, meters: dict, K: int | None = None) -> "Pool | None":
        """From one tier's and cell's S1 rows (all epochs). K defaults to the most common run count per task; tasks
        with fewer runs are left out (`short_tasks`); tasks with more keep their first K runs (`run_order`)."""
        if rows.empty:
            return None
        r = run_order(rows)
        r["_run"] = r.groupby(["world", "task"]).cumcount()
        counts = r.groupby(["world", "task"]).size()
        K_mode = int(counts.mode().max())
        K = K_mode if K is None else min(int(K), int(counts.max()))
        r = r[r["_run"] < K]
        full = counts[counts >= K].index
        r = r.set_index(["world", "task"]).loc[lambda x: x.index.isin(full)].reset_index()
        if r.empty:
            return None
        piv = lambda col: r.pivot(index=["cell", "world", "task"], columns="_run", values=col)  # noqa: E731  (keys are unique)
        succ = piv("success")
        keys = piv("answer_key").reindex(succ.index) if "answer_key" in r.columns else pd.DataFrame(index=succ.index, columns=succ.columns)
        cost = {m: piv(col).reindex(succ.index).to_numpy(dtype=float) for m, col in meters.items() if col in r.columns}
        return cls(succ.index, fr.key_codes(keys.to_numpy().tolist()), succ.to_numpy(dtype=float), cost, K, int((counts < K).sum()))


@dataclass
class Tables:
    """What every hypothesis reads, per tier: task means of success (`tm`) and of each meter's cost (`cost`), and the
    S1 pool rows for the S8 frontiers. The power simulation builds one directly (with `pools` filled in)."""

    tm: dict
    cost: dict
    pool_rows: dict = field(default_factory=dict)
    meters: dict = field(default_factory=lambda: dict(METERS))
    cluster_by: str = "kb"
    wall_aggregate: str = "sum"
    pools: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    _frontiers: dict = field(default_factory=dict)

    def pool(self, tier: str, cell: str, k: int | None = None) -> Pool | None:
        key = (tier, cell, k)
        if key not in self.pools:
            rows = self.pool_rows.get(tier)
            sub = rows[rows["cell"] == cell] if rows is not None and len(rows) else pd.DataFrame()
            self.pools[key] = Pool.from_rows(sub, self.meters, k) if len(sub) else None
        return self.pools[key]

    def frontier(self, tier: str, cell: str, meter: str, k: int | None = None) -> fr.Frontier | None:
        key = (tier, cell, meter, k)
        if key not in self._frontiers:
            pool = self.pool(tier, cell, k)
            if pool is None or meter not in pool.cost:
                self._frontiers[key] = None
            else:
                agg = self.wall_aggregate if meter == "wall" else "sum"
                self._frontiers[key] = fr.build(pool.codes, pool.success, pool.cost[meter], agg, {"K": pool.K, "short_tasks": pool.short_tasks, "tasks": len(pool.index), "aggregate": agg})
        return self._frontiers[key]

    def arm_cost(self, tier: str, cell: str, arm: str, meter: str) -> float | None:
        """The arm's realised mean cost per sample in the cell (mean of its task means), on `meter`."""
        tab = (self.cost.get(tier) or {}).get(meter)
        if tab is None or arm not in tab.columns:
            return None
        v = tab[arm][tab.index.get_level_values("cell") == cell].dropna()
        return float(v.mean()) if len(v) else None


def _task_mean(rows: pd.DataFrame, col: str) -> pd.DataFrame:
    return rows.groupby(["cell", "world", "task", "arm"])[col].mean().unstack("arm")


def run_order(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows sorted into run order (plan cell, log file, epoch) with `run` = 0, 1, ... per (tier, arm, cell, task).
    A pool split over several logs numbers its epochs from 1 in each, so epochs alone cannot order (or identify) runs;
    the first `s1_epochs` runs are S1's main epochs, and the pool's first K runs its frontier."""
    keys = [k for k in ("plan_cell", "log_file", "epoch") if k in rows.columns]
    r = rows.sort_values(keys, kind="stable", na_position="first").copy() if keys else rows.copy()
    group = [k for k in ("tier", "arm", "cell", "world", "task") if k in r.columns]
    r["run"] = r.groupby(group).cumcount() if len(r) else pd.Series(dtype=int)
    return r


def confirmatory_rows(frame: pd.DataFrame, exclude_prefix: str = DETERMINISM_PREFIX) -> tuple[pd.DataFrame, dict]:
    """The rows the hypotheses use: every plan cell except Study C's (determinism only), each sample-epoch once:
    - a row with a sample `uuid` (the loader's: one per sample-epoch run, kept when Inspect copies a completed sample
      into a retry's log) is a duplicate of an earlier row with the same uuid and epoch, whichever log file holds it;
    - a row without one is a duplicate of an earlier row with the same (tier, arm, cell, task, epoch) from the same log
      file (e.g. a log passed under two plan cells; without a log_file column, the same epoch twice).
    Duplicates are dropped and counted."""
    rows = frame
    if "plan_cell" in rows.columns and exclude_prefix:
        rows = rows[~rows["plan_cell"].fillna("").astype(str).str.startswith(exclude_prefix)]
    keys = [k for k in ("tier", "arm", "cell", "world", "task", "epoch", "log_file") if k in rows.columns]
    dup = rows.duplicated(subset=keys, keep="first")
    if "uuid" in rows.columns and rows["uuid"].notna().any():
        has = rows["uuid"].notna()
        by_uuid = rows[has].duplicated(subset=["uuid", "epoch"] if "epoch" in rows.columns else ["uuid"], keep="first")
        dup = dup | by_uuid.reindex(rows.index, fill_value=False)
    return rows[~dup], {"rows": int(len(frame)), "used": int((~dup).sum()), "duplicates_dropped": int(dup.sum()), "excluded_plan_cells": sorted({str(p) for p in frame.get("plan_cell", pd.Series(dtype=str)).dropna() if str(p).startswith(exclude_prefix)}) if exclude_prefix else []}


def build_tables(frame: pd.DataFrame, meters: dict | None = None, s1_epochs: int = S1_EPOCHS, cluster_by: str = "kb", wall_aggregate: str = "sum", exclude_prefix: str = DETERMINISM_PREFIX) -> Tables:
    """Tables from the tidy frame (`main_load`). S1 keeps its first `s1_epochs` epochs for its task means; the frontier
    uses all its pool runs. Rows with NaN success are left out and counted (the loader never produces them)."""
    meters = dict(METERS if meters is None else meters)
    rows, notes = confirmatory_rows(frame, exclude_prefix)
    if "tier" not in rows.columns:
        rows = rows.assign(tier=PRIMARY_TIER)
    nan = rows["success"].isna()
    notes["nan_success_rows"] = int(nan.sum())
    rows = rows[~nan]
    tm, cost, pool_rows = {}, {}, {}
    for tier, r in rows.groupby("tier"):
        r = run_order(r)
        pool_rows[tier] = r[r["arm"] == "S1"]
        main = r[~((r["arm"] == "S1") & (r["run"] >= s1_epochs))]
        tm[tier] = _task_mean(main, "success")
        cost[tier] = {m: _task_mean(main, col) for m, col in meters.items() if col in main.columns}
    return Tables(tm, cost, pool_rows, meters, cluster_by, wall_aggregate, notes=notes)


# --------------------------------------------------------------------------- members and families


def _matched_cost_influence(tables: Tables, term: Term, cell: str, meter: str, f: fr.Frontier, pool: "Pool", target: float, m: dict) -> tuple[pd.Series | None, dict]:
    """Each task's influence on the interpolation weight λ of `S8@arm` (BUILD_REVIEW S-2), times the frontier's slope.

    S8@arm is interpolated at r̂ = C̄_arm / C̄_S1 (the arm's realised mean cost over S1's mean run cost, both measured on
    the same worlds), so λ̂ = r̂ − k is an estimate. Treating the per-task S8 values as fixed ignores its noise, and when
    cost depends on success (failed runs costing more) that noise is correlated with the arm's own success: the frontier
    tests then ran at 0.073–0.089 against 0.05 (M3) and 0.027–0.031 against 0.025 (M2.frontier). By the delta method the
    estimate moves by slope × (λ̂ − λ), and λ̂ − λ ≈ mean_t (c_arm,t − r̂ c_S1,t) / C̄_S1 (`sum` costs: S8(k) costs k runs),
    so task t contributes ψ_t = slope × (c_arm,t − r̂ c_S1,t) / C̄_S1, slope = S8(k+1) − S8(k) on the condition curve. For
    `max`-priced wall-clock only the arm's cost is linearised: ψ_t = slope × (c_arm,t − C̄_arm) / (cost(k+1) − cost(k)).
    `below` (flat at S8(1)) has no influence."""
    if m.get("status") != "inside" or f.K < 2:
        return None, {"applied": False, "why": f"status {m.get('status')}" if m.get("status") != "inside" else "a 1-run pool has no slope"}
    j = int(m["k"]) - 1
    seg = j if j < f.K - 1 else f.K - 2  # at S8(K) exactly: the segment it closes
    slope = float(f.success[seg + 1] - f.success[seg])
    tab = (tables.cost.get(term.tier) or {}).get(meter)
    if tab is None or term.match not in tab.columns:
        return None, {"applied": False, "why": f"no per-task {meter} cost for {term.match}"}
    arm_cost = tab[term.match][tab.index.get_level_values("cell") == cell].dropna()
    agg = tables.wall_aggregate if meter == "wall" else "sum"
    if agg == "sum":
        s1_cost = pd.Series(np.nanmean(pool.cost[meter], axis=1), index=pool.index)
        both = pd.concat([arm_cost, s1_cost], axis=1).dropna()
        psi = slope * (both.iloc[:, 0] - (target / f.run_cost) * both.iloc[:, 1]) / f.run_cost
    else:
        dcost = float(f.cost[seg + 1] - f.cost[seg])
        psi = slope * (arm_cost - target) / dcost if dcost > 0 else None
    return psi, {"applied": psi is not None, "slope": slope, "aggregate": agg}


def _term(tables: Tables, term: Term, cell: str, meter: str) -> tuple[pd.Series | None, dict, pd.Series | None]:
    """One term's per-task values in one cell (index cell, world, task) or None with the reason, and for a matched S8
    term the per-task matched-cost influence ψ (`_matched_cost_influence`)."""
    if term.arm != "S8":
        tm = tables.tm.get(term.tier)
        if tm is None or term.arm not in tm.columns:
            return None, {"reason": f"no {term.arm} in tier {term.tier}"}, None
        s = tm[term.arm][tm.index.get_level_values("cell") == cell].dropna()
        return (s, {}, None) if len(s) else (None, {"reason": f"no {term.arm} tasks in {cell} ({term.tier})"}, None)
    f = tables.frontier(term.tier, cell, meter, term.pool)
    if f is None:
        return None, {"reason": f"no S1 pool for S8 in {cell} ({term.tier}, {meter})"}, None
    pool = tables.pool(term.tier, cell, term.pool)
    info = {"frontier": {"K": f.K, "tasks": len(pool.index), "short_tasks": pool.short_tasks, "inconsistent_keys": f.inconsistent_keys, "points": f.points()}}
    if term.k:
        if term.k > f.K:
            return None, info | {"reason": f"S8({term.k}) needs {term.k} pool runs, the pool has {f.K}"}, None
        return pd.Series(f.task_success[:, term.k - 1], index=pool.index), info, None
    target = tables.arm_cost(term.tier, cell, term.match, meter)
    if target is None:
        return None, info | {"reason": f"no {meter} cost for {term.match} in {cell} ({term.tier})"}, None
    vals, m = fr.interpolate(f, target)
    info["match"] = m
    if vals is None:
        return None, info | {"reason": f"{term.match} is {m['status']} the S8 frontier in {cell} ({term.tier}, {meter})"}, None
    psi, corr = _matched_cost_influence(tables, term, cell, meter, f, pool, target, m)
    info["matched_cost_correction"] = corr
    return pd.Series(vals, index=pool.index), info, psi


def term_series(tables: Tables, term: Term, cell: str, meter: str) -> tuple[pd.Series | None, dict]:
    """One term's per-task values in one cell (index cell, world, task), or None with the reason."""
    s, info, _ = _term(tables, term, cell, meter)
    return s, info


def member_series(tables: Tables, member: Member, meter: str = PRIMARY_METER) -> tuple[pd.Series, dict]:
    """The member's per-task contrast over its cells (cells without paired tasks are left out and named).

    A matched S8 term adds coef × ψ_t, its matched-cost influence centred over the cell's paired tasks: the point
    estimate is unchanged (the cell mean of the addition is 0), but every per-task value, so every cluster's
    contribution to the sign flip, the cluster-t and the bootstrap, now carries the noise of the matched cost."""
    parts, cells = [], {}
    for cell in member.cells:
        cols, info, psis = [], {}, []
        for term in member.terms:
            s, tinfo, psi = _term(tables, term, cell, meter)
            if tinfo.get("frontier") or tinfo.get("match"):
                info.setdefault("frontier", {})[term.label()] = {k: v for k, v in tinfo.items() if k != "reason"}
            if s is None:
                info["reason"] = tinfo["reason"]
                break
            cols.append(term.coef * s)
            if psi is not None:
                psis.append(term.coef * psi)
        if "reason" in info:
            cells[cell] = info | {"tasks": 0}
            continue
        df = pd.concat(cols, axis=1)
        full = df.dropna()
        cells[cell] = info | {"tasks": int(len(full)), "unpaired_tasks": int(len(df) - len(full))}
        if len(full):
            d_cell = full.sum(axis=1)
            if psis:
                corr = pd.concat(psis, axis=1).reindex(full.index).fillna(0.0).sum(axis=1)
                d_cell = d_cell + (corr - corr.mean())
            parts.append(d_cell)
        else:
            cells[cell]["reason"] = "no task has every term"
    d = pd.concat(parts) if parts else pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["cell", "world", "task"]))
    return d, {"cells": cells, "missing_cells": [c for c, i in cells.items() if i.get("reason")]}


def cost_ratio_check(tables: Tables, member: Member, alpha: float = 0.025, reps: int = REPS, seed: int = 0) -> dict | None:
    """A member's cost-ratio co-condition (`CostRatio`) on the paired tasks of its cells."""
    cr = member.cost_ratio
    if cr is None:
        return None
    tab = (tables.cost.get(PRIMARY_TIER) or {}).get(cr.meter)
    if tab is None or cr.num not in tab.columns or cr.den not in tab.columns:
        return {"ok": False, "reason": f"no {cr.meter} cost for {cr.num} or {cr.den}", "ratio": None, "ci": [None, None]}
    pair = tab[[cr.num, cr.den]][tab.index.get_level_values("cell").isin(member.cells)].dropna()
    if pair.empty:
        return {"ok": False, "reason": "no paired costs", "ratio": None, "ci": [None, None]}
    num, den = Contrast.from_series(pair[cr.num], tables.cluster_by), Contrast.from_series(pair[cr.den], tables.cluster_by)
    r = ratio_ci(num, den, alpha, reps, seed)
    ok = r["ratio"] is not None and np.isfinite(r["ratio"]) and r["ratio"] <= cr.max_point and r["ci"][1] is not None and r["ci"][1] <= cr.max_upper
    rule = f"{cr.num}/{cr.den} ≤ {cr.max_upper} with 97.5% confidence (its one-sided upper bound, the upper end of the two-sided 95% BCa interval) and point estimate ≤ {cr.max_point}"
    return r | {"ok": bool(ok), "meter": cr.meter, "rule": rule, "note": "realised cost is the logged final attempts' usage; errored attempts' usage (retried, D-030) is not in it"}


def per_cell_estimates(d: pd.Series, cluster_by: str, alpha: float, reps: int = REPS, seed: int = 0) -> dict:
    """A pooled member's estimate in each of its cells (BUILD_REVIEW S-6: a pooled claim is an equal-weight average over
    cells, and a ceiling cell dilutes it), with the inverted sign-flip and cluster-t intervals."""
    out = {}
    for cell in dict.fromkeys(d.index.get_level_values("cell").astype(str)):
        sub = d[d.index.get_level_values("cell") == cell]
        r = infer(Contrast.from_series(sub, cluster_by), "descriptive", alpha=alpha, reps=reps, seed=seed)
        out[cell] = {k: r.get(k) for k in ("est", "ci", "ci_t", "tasks", "clusters")}
    return out


def world_influence(d: pd.Series, cluster_by: str, test: str, margin: float, reps: int = REPS, seed: int = 0, exact_max: int = EXACT_MAX_CLUSTERS, flagged: set | None = None) -> list[dict]:
    """Each cluster's share of the estimate and the member re-estimated (and, for a test, re-tested) without it, most
    influential first (BUILD_REVIEW S-3: the sign flip holds α only when worlds contribute symmetrically, so a claim one
    world carries is a caveat). `flagged`: world ids that failed their KG build-quality check."""
    c = Contrast.from_series(d, cluster_by)
    if c.G < 2:
        return []
    s, _ = c.contributions()
    clus = np.array([cluster_of(w, cluster_by) for w in d.index.get_level_values("world")])
    worlds = d.index.get_level_values("world").astype(str)
    rows = []
    for g, name in enumerate(c.clusters):
        keep = clus != name
        sub = Contrast.from_series(d[keep], cluster_by)
        row = {"cluster": name, "contribution": float(s[g]), "tasks": int((~keep).sum()), "cells": sorted({str(x) for x in d.index.get_level_values("cell")[~keep]}), "est_without": sub.est}
        if test != "descriptive":
            row["p_without"] = p_value(SignFlip(sub, reps=reps, seed=seed, exact_max=exact_max), test, margin)[0]
        if flagged:
            row["kg_build_failed"] = sorted({w for w in worlds[~keep] if w in flagged})
        rows.append(row)
    return sorted(rows, key=lambda r: -abs(r["contribution"]))


def evaluate_member(tables: Tables, member: Member, alpha: float, meter: str = PRIMARY_METER, reps: int = REPS, seed: int = 0, ci: bool = True, exact_max: int = EXACT_MAX_CLUSTERS, flagged_worlds: set | None = None) -> dict:
    """One member: its contrast, inference, and cost condition; with `ci`, also its per-cell estimates (a pooled member)
    and its per-world influence (`world_influence`). Missing arms or cells are reported, never raised."""
    out = {"id": member.id, "contrast": member.contrast(), "test": member.test, "margin": member.margin if member.test in ("ni", "tost") else None, "cells_planned": list(member.cells)}
    d, info = member_series(tables, member, meter)
    out["coverage"] = info
    if d.empty:
        reasons = sorted({i.get("reason") for i in info["cells"].values() if i.get("reason")})
        return out | {"evaluable": False, "reason": "; ".join(reasons) or "no data", "p": None, "est": None}
    c = Contrast.from_series(d, tables.cluster_by)
    res = infer(c, member.test, member.margin, alpha, reps=reps, seed=seed, ci=ci, exact_max=exact_max)
    out |= {"evaluable": True} | res
    if ci:
        if len(c.cells) > 1:
            out["per_cell"] = per_cell_estimates(d, tables.cluster_by, alpha, reps, seed)
        out["influence"] = world_influence(d, tables.cluster_by, member.test, member.margin, reps, seed, exact_max, flagged_worlds)
        out["kg_build_failed_worlds"] = sorted({str(w) for w in d.index.get_level_values("world")} & set(flagged_worlds or ()))
    if member.cost_ratio is not None:
        out["cost_ratio"] = cost_ratio_check(tables, member, 0.025, reps if ci else min(reps, 2000), seed)
    return out


def caveats(member: dict) -> list[str]:
    """§8 caveats on a supported member: the claim does not survive dropping one world, or it rests on worlds whose KG
    build failed its quality check."""
    if member.get("label") != "supported":
        return []
    out = []
    level = member.get("holm_level")
    carried = [r["cluster"] for r in member.get("influence") or [] if r.get("p_without") is not None and level is not None and r["p_without"] > level]
    if carried:
        out.append(f"driven by one world: without {carried[0]} the test is not rejected at {level:g}" + (f" (also {len(carried) - 1} more)" if len(carried) > 1 else ""))
    if member.get("kg_build_failed_worlds"):
        out.append(f"rests on worlds whose KG build failed its quality check: {member['kg_build_failed_worlds']}")
    return out


def evaluate_family(tables: Tables, hyp: Hypothesis, alpha: float | None = None, reps: int = REPS, seed: int = 0, ci: bool = True, exact_max: int = EXACT_MAX_CLUSTERS, flagged_worlds: set | None = None) -> dict:
    """A hypothesis family: every member evaluated, then Holm within the family (or serial gatekeeping over stages), and
    each member's label:
    - `supported`: rejected at its Holm level (and its cost condition holds);
    - `not supported`: tested, not rejected (never "no effect");
    - `not tested`: Holm stopped before it, or its gate stage did not pass;
    - `not evaluable`: its data are missing (named).
    A supported member carries its §8 `caveats` (`caveats`: one world carries it, or a KG-build-failed world is in it)."""
    a = hyp.alpha if alpha is None else alpha
    members = {m.id: evaluate_member(tables, m, a, hyp.meter, reps, seed + i, ci, exact_max, flagged_worlds) for i, m in enumerate(hyp.members)}
    out = {"id": hyp.id, "source": hyp.source, "statement": hyp.statement, "status": hyp.status, "alpha": a, "procedure": hyp.procedure, "meter": hyp.meter, "members": members, "note": hyp.note}
    if hyp.status != "confirmatory":
        return out
    stages = sorted({m.stage for m in hyp.members}) if hyp.procedure == "serial" else [1]
    open_gate = True
    for st in stages:
        ids = [m.id for m in hyp.members if (m.stage == st or hyp.procedure != "serial")]
        if not open_gate:
            for i in ids:
                label = "not tested" if members[i].get("evaluable") else "not evaluable"
                members[i] |= {"holm_level": None, "rejected": False, "tested": False, "label": label, "why": f"stage {st}: the gate (stage {st - 1}) did not pass"}
            continue
        levels, rejected, tested = holm_test({i: members[i].get("p") for i in ids}, a)
        for i in ids:
            m = members[i]
            m |= {"holm_level": levels[i], "rejected": bool(rejected[i]), "tested": bool(tested[i])}
            cond = m.get("cost_ratio")
            if not m.get("evaluable"):
                m["label"] = "not evaluable"
            elif not tested[i]:
                m["label"] = "not tested"
                m["why"] = f"Holm stopped before it (level {levels[i]:g})"
            elif rejected[i] and (cond is None or cond["ok"]):
                m["label"] = "supported"
            else:
                m["label"] = "not supported"
                if rejected[i] and cond is not None:
                    m["why"] = f"rejected, but the cost condition fails: {cond.get('reason') or cond['rule']}"
        open_gate = all(members[i]["rejected"] for i in ids)
    for m in members.values():
        m["caveats"] = caveats(m)
    out["supported"] = [i for i, m in members.items() if m.get("label") == "supported"]
    return out


def describe(tables: Tables, terms: tuple[Term, ...], cells: tuple[str, ...], alpha: float = 0.025, meter: str = PRIMARY_METER, reps: int = REPS, seed: int = 0, weights: dict | None = None) -> dict:
    """A descriptive contrast (estimate and intervals, no test), pooled over `cells` (or weighted by `weights`)."""
    m = Member("descriptive", terms, cells, "descriptive")
    d, info = member_series(tables, m, meter)
    out = {"contrast": m.contrast(), "cells_planned": list(cells), "coverage": info}
    if d.empty:
        return out | {"evaluable": False, "est": None}
    w = None
    if weights:
        present = list(dict.fromkeys(d.index.get_level_values("cell").astype(str)))
        w = {c: weights[c] for c in present}
    c = Contrast.from_series(d, tables.cluster_by, w)
    return out | {"evaluable": True} | infer(c, "descriptive", 0.0, alpha, reps=reps, seed=seed, ci=True)
