"""Descriptive main-study estimators: what the report shows with intervals but tests nothing (D-028, D-029).

- `k4_slopes` (K4 / H2c): success slope per decade of KB size over F7-10/100/1000, per arm, and paired slope
  differences. The slope is a weighted contrast of cell means, λ_c = (x_c − x̄) / Σ(x − x̄)², x = log10(level), with the
  same world-clustered intervals as every contrast (worlds of different levels are independent clusters).
- `m4` (M4 / H1d): M1 − M1s accuracy on F1-32 (95% and 90% intervals) and the wall-clock ratio M1 / M1s.
- `mechanism_contrasts` (§4.3): each single-switch contrast per cell where both arms ran.
- `tier_contrasts` (Study F): each arm contrast in Luna and in Sol on the same tasks, and its Sol − Luna change.
- `frontier_table` (§4.5): each condition's S8 curve on every meter, and every arm against S8 at its own realised
  cost, per meter (beyond-the-frontier arms named, not extrapolated).
- `rank_flips` (C1 / H3): per family, each arm's ranking quantity (default cost per solved task: mean cost / mean
  success, cells equally weighted, on the tasks every ranked arm ran) under each meter; Kendall τ-b between meters; the
  rule "min pairwise τ < 0.8 in ≥ 2 of the 4 MVS families"; world-clustered bootstrap support for each.
- `determinism` (A1 / H5): on Study C's task subset, each arm's first 5 runs (its main epochs first, then Study C's 2;
  S1's main epochs are its pool's first 3): unbiased pass^k and pass@k, D1 outcome stability, D2 modal-answer share
  and normalised answer entropy, D6 cost CVs; and KG single-agent vs multi-agent contrasts with intervals.
- `invariants` (§4.4): S6 ≥ S5 ≥ S7 within 3 pp (`gate_stats.violation_test`: fails only on evidence) and S5 > S7
  (sign flip, 0.05), where those arms ran; M1 ≈ M1s is M4.
- `s8k3_check`: the live S8k3 arm (Study C) against the post-hoc S8(3) on the same tasks (their tie rules differ).
- `glmm` (D-029): `success ~ arm * log_knob + (1 | task) + (1 | task:arm)` per family with statsmodels'
  BinomialBayesMixedGLM (variational Bayes; its posterior SDs run small). Never raises: a failed fit is reported.
"""

import math
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import kendalltau

from .gate_stats import violation_test
from .main_hypotheses import KB_CELLS, MECHANISM_CONTRASTS, PRIMARY_TIER, STUDY_A_CELLS, STUDY_B_CELLS, TIER_CELLS, Member, Term
from .main_stats import (
    DETERMINISM_PREFIX,
    REPS,
    S1_EPOCHS,
    Contrast,
    SignFlip,
    Tables,
    _estimates,
    _resample_weights,
    cluster_t,
    confirmatory_rows,
    describe,
    infer,
    member_series,
    ratio_ci,
    run_order,
)

FAMILIES = ("F1", "F2", "F3", "F7")
MVS_CELLS = STUDY_A_CELLS + STUDY_B_CELLS
RANK_METERS = ("tokens", "usd", "wall")
TAU_THRESHOLD = 0.8
MIN_FLIP_FAMILIES = 2
DETERMINISM_EPOCHS = 5
MATCHED_ACCURACY = 0.03  # H5: "at matched accuracy (|Δ| < 3 pp)"


def _level(cell: str) -> float:
    return float(str(cell).split("-", 1)[1])


# --------------------------------------------------------------------------- K4, M4, §4.3


def k4_slopes(tables: Tables, arms=("S1", "S5", "S3s", "M1", "M2"), diffs=(("S5", "S1"), ("S5", "S3s")), cells=KB_CELLS, alpha: float = 0.025, reps: int = REPS, seed: int = 0) -> dict:
    """Slopes (success per decade of KB size) per arm and per paired difference, with intervals (module docstring)."""
    out = {"cells": list(cells), "x": {c: math.log10(_level(c)) for c in cells}, "arms": {}, "differences": {}, "unit": "success per decade (×100 = pp per decade)"}
    for name, terms in [(a, (Term(a),)) for a in arms] + [(f"{a} − {b}", (Term(a), Term(b, -1))) for a, b in diffs]:
        d, info = member_series(tables, Member("k4", terms, cells, "descriptive"))
        present = list(dict.fromkeys(d.index.get_level_values("cell").astype(str))) if len(d) else []
        target = out["arms" if len(terms) == 1 else "differences"]
        if len(present) < 2:
            target[name] = {"estimable": False, "cells": present, "reason": "fewer than 2 KB sizes"}
            continue
        x = np.array([math.log10(_level(c)) for c in present])
        lam = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
        c = Contrast.from_series(d, tables.cluster_by, dict(zip(present, lam, strict=True)))
        res = infer(c, "descriptive", alpha=alpha, reps=reps, seed=seed)
        per_cell = d.groupby(level="cell").mean().to_dict()
        target[name] = {"estimable": True, "slope": res["est"], "ci_t": res["ci_t"], "ci_boot": res["ci_boot"], "cells": present, "cell_means": per_cell, "clusters": res["clusters"], "tasks": res["tasks"]}
    return out


def m4(tables: Tables, cell: str = "F1-32", reps: int = REPS, seed: int = 0) -> dict:
    """M1 − M1s accuracy (95% and 90% intervals) and the wall-clock ratio M1 / M1s on their paired tasks."""
    acc95 = describe(tables, (Term("M1"), Term("M1s", -1)), (cell,), alpha=0.025, reps=reps, seed=seed)
    acc90 = describe(tables, (Term("M1"), Term("M1s", -1)), (cell,), alpha=0.05, reps=reps, seed=seed)
    out = {"cell": cell, "accuracy": acc95, "accuracy_90": {k: acc90.get(k) for k in ("est", "ci", "ci_boot", "ci_t")}}
    wall = (tables.cost.get(PRIMARY_TIER) or {}).get("wall")
    if wall is None or not {"M1", "M1s"} <= set(wall.columns):
        return out | {"wall_ratio": {"ratio": None, "ci": [None, None], "reason": "no wall-clock for M1 or M1s"}}
    pair = wall[["M1", "M1s"]][wall.index.get_level_values("cell") == cell].dropna()
    if pair.empty:
        return out | {"wall_ratio": {"ratio": None, "ci": [None, None], "reason": "no paired tasks"}}
    r = ratio_ci(Contrast.from_series(pair["M1"], tables.cluster_by), Contrast.from_series(pair["M1s"], tables.cluster_by), 0.025, reps, seed)
    return out | {"wall_ratio": r | {"tasks": int(len(pair)), "brief_threshold": 0.6}}


def mechanism_contrasts(tables: Tables, alpha: float = 0.025, reps: int = REPS, seed: int = 0, contrasts=MECHANISM_CONTRASTS) -> list[dict]:
    """§4.3's single-switch contrasts, per cell where both arms ran (descriptive; no test)."""
    out = []
    for name, a, b, cells in contrasts:
        for cell in cells:
            r = describe(tables, (Term(a), Term(b, -1)), (cell,), alpha=alpha, reps=reps, seed=seed)
            if r.get("evaluable"):
                out.append({"mechanism": name, "contrast": f"{a} − {b}", "cell": cell} | {k: r.get(k) for k in ("est", "ci", "ci_boot", "ci_t", "tasks", "clusters")})
    return out


# Study F's arm contrasts per cell: S1, S5 and M1 run both tiers on F1-32 and F7-100; M2 only on F7-100 (D-034).
TIER_CONTRASTS = {
    "F1-32": (("M1", "S1"), ("S5", "S1"), ("S5", "M1")),
    "F7-100": (("M1", "S1"), ("S5", "S1"), ("S5", "M1"), ("M2", "S1"), ("M2", "S5"), ("M2", "M1")),
}


def tier_contrasts(tables: Tables, cells=TIER_CELLS, contrasts: dict = TIER_CONTRASTS, tiers=(PRIMARY_TIER, "sol"), alpha: float = 0.025, reps: int = REPS, seed: int = 0) -> list[dict]:
    """Study F's arm × tier model as paired contrasts (brief §7.4): each arm contrast in each tier, and its change from
    the first tier to the second, a difference in differences on the same tasks (descriptive; T1 reports the
    cost-matched coordination payoff). A contrast whose arms did not run is left out."""
    base, other = tiers
    out = []
    for cell in cells:
        for a, b in contrasts.get(cell, ()):
            row = {"cell": cell, "contrast": f"{a} − {b}"}
            for tier in tiers:
                r = describe(tables, (Term(a, 1, tier), Term(b, -1, tier)), (cell,), alpha=alpha, reps=reps, seed=seed)
                row[tier] = {k: r.get(k) for k in ("est", "ci", "ci_t", "tasks")} if r.get("evaluable") else None
            r = describe(tables, (Term(a, 1, other), Term(b, -1, other), Term(a, -1, base), Term(b, 1, base)), (cell,), alpha=alpha, reps=reps, seed=seed)
            row[f"{other} − {base}"] = {k: r.get(k) for k in ("est", "ci", "ci_t", "tasks")} if r.get("evaluable") else None
            if any(row[t] for t in tiers):
                out.append(row)
    return out


# --------------------------------------------------------------------------- frontier


def frontier_table(tables: Tables, meters=None, alpha: float = 0.025, reps: int = 2000, seed: int = 0, skip=("S1",)) -> list[dict]:
    """Per tier, cell and meter: the S8 curve and every other arm against S8 at its realised mean cost."""
    out = []
    meters = list(meters or tables.meters)
    for tier, tm in tables.tm.items():
        for cell in sorted(set(tm.index.get_level_values("cell").astype(str))):
            if tables.pool(tier, cell) is None:
                continue
            arms = [a for a in tm.columns if a not in skip and tm[a][tm.index.get_level_values("cell") == cell].notna().any()]
            for meter in meters:
                f = tables.frontier(tier, cell, meter)
                if f is None:
                    continue
                row = {"tier": tier, "cell": cell, "meter": meter, "K": f.K, "points": f.points(), "inconsistent_keys": f.inconsistent_keys, "arms": {}}
                for a in arms:
                    cost = tables.arm_cost(tier, cell, a, meter)
                    r = describe(tables, (Term(a, 1, tier), Term("S8", -1, tier, match=a)), (cell,), alpha=alpha, meter=meter, reps=reps, seed=seed)
                    match = (((r.get("coverage") or {}).get("cells") or {}).get(cell) or {}).get("frontier", {})
                    m = next(iter(match.values()), {}).get("match") if match else None
                    row["arms"][a] = {"cost": cost, "cost_ratio_to_s1_run": cost / f.run_cost if cost is not None and f.run_cost else None, "match": m} | {k: r.get(k) for k in ("est", "ci", "ci_boot", "ci_t", "tasks")} | ({} if r.get("evaluable") else {"reason": ((r.get("coverage") or {}).get("cells") or {}).get(cell, {}).get("reason")})
                out.append(row)
    return out


def s8k3_check(frame: pd.DataFrame, tables: Tables, alpha: float = 0.025, reps: int = REPS, seed: int = 0, prefix: str = DETERMINISM_PREFIX, meter: str = "tokens") -> list[dict]:
    """The live S8k3 (Study C) against the post-hoc S8(3) from the pool, per cell, on the same tasks."""
    if "plan_cell" not in frame.columns:
        return []
    live = frame[(frame["arm"] == "S8k3") & frame["plan_cell"].fillna("").astype(str).str.startswith(prefix)]
    out = []
    for (tier, cell), g in live.groupby(["tier", "cell"]):
        f = tables.frontier(tier, cell, meter)
        pool = tables.pool(tier, cell)
        if f is None or f.K < 3:
            out.append({"tier": tier, "cell": cell, "reason": "no S1 pool with 3 runs"})
            continue
        s8 = pd.Series(f.task_success[:, 2], index=pool.index)
        tm = g.groupby(["cell", "world", "task"])["success"].mean()
        d = (tm - s8).dropna()
        if d.empty:
            out.append({"tier": tier, "cell": cell, "reason": "no shared tasks"})
            continue
        r = infer(Contrast.from_series(d, tables.cluster_by), "descriptive", alpha=alpha, reps=reps, seed=seed)
        out.append({"tier": tier, "cell": cell, "live_minus_posthoc": r["est"], "ci_t": r["ci_t"], "ci_boot": r["ci_boot"], "tasks": r["tasks"]})
    return out


# --------------------------------------------------------------------------- C1 rank flips


def _rank_quantity(cost: np.ndarray, succ: np.ndarray, rank_by: str) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if rank_by == "cost_per_success":
            q = cost / succ
            return np.where(np.isfinite(q), q, 1e300)
        if rank_by == "cost":
            return cost
        if rank_by == "success_per_cost":
            return -(succ / cost)
    raise ValueError(f"rank_by must be cost_per_success, cost or success_per_cost, not {rank_by!r}")


def _min_finite(values) -> float:
    v = [x for x in values if np.isfinite(x)]
    return float(min(v)) if v else float("nan")


def _tau(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def rank_flips(tables: Tables, families=FAMILIES, meters=RANK_METERS, rank_by: str = "cost_per_success", threshold: float = TAU_THRESHOLD, min_families: int = MIN_FLIP_FAMILIES, reps: int = 2000, seed: int = 0, tier: str = PRIMARY_TIER, cells=MVS_CELLS) -> dict:
    """H3's rule with bootstrap support (module docstring). Rankings use the arms that ran in every MVS cell of the
    family (`cells`: Studies A and B; Study F's F7-100 runs only four arms), on the tasks all of them ran (S1: its main
    epochs), cells equally weighted."""
    tm, cost = tables.tm.get(tier), tables.cost.get(tier) or {}
    meters = [m for m in meters if m in cost]
    out = {"rank_by": rank_by, "meters": meters, "threshold": threshold, "min_families": min_families, "families": {}, "pairs": [f"{a}~{b}" for a, b in combinations(meters, 2)]}
    if tm is None or len(meters) < 2:
        return out | {"rule_holds": None, "reason": "fewer than two meters with data"}
    rng = np.random.default_rng(seed)
    boots_flip = []
    for fam in families:
        fam_cells = sorted({c for c in tm.index.get_level_values("cell").astype(str) if c.split("-")[0] == fam and c in cells})
        if not fam_cells:
            continue
        sub = tm[tm.index.get_level_values("cell").isin(fam_cells)]
        arms = [a for a in sub.columns if all(sub[a][sub.index.get_level_values("cell") == c].notna().any() for c in fam_cells)]
        frames = [sub[arms]] + [cost[m].reindex(sub.index)[arms].add_suffix(f"|{m}") for m in meters]
        shared = pd.concat(frames, axis=1).dropna()
        if len(arms) < 2 or shared.empty:
            out["families"][fam] = {"arms": arms, "reason": "fewer than 2 arms with shared tasks"}
            continue
        contrasts = {(a, q): Contrast.from_series(shared[a if q == "success" else f"{a}|{q}"], tables.cluster_by) for a in arms for q in ["success", *meters]}
        est = {k: c.est for k, c in contrasts.items()}
        W = _resample_weights(next(iter(contrasts.values())), reps, rng)
        boot = {k: _estimates(c, W) for k, c in contrasts.items()}
        succ = np.array([est[(a, "success")] for a in arms])
        quant = {m: _rank_quantity(np.array([est[(a, m)] for a in arms]), succ, rank_by) for m in meters}
        taus = {f"{x}~{y}": _tau(quant[x], quant[y]) for x, y in combinations(meters, 2)}
        bsucc = np.stack([boot[(a, "success")] for a in arms], axis=1)
        bq = {m: _rank_quantity(np.stack([boot[(a, m)] for a in arms], axis=1), bsucc, rank_by) for m in meters}
        bmin = np.array([_min_finite([_tau(bq[x][r], bq[y][r]) for x, y in combinations(meters, 2)]) for r in range(reps)])
        min_tau = _min_finite(list(taus.values()))
        boots_flip.append(bmin < threshold)
        out["families"][fam] = {
            "cells": fam_cells, "arms": arms, "tasks": int(len(shared)),
            "success": dict(zip(arms, succ.tolist(), strict=True)),
            "quantity": {m: dict(zip(arms, quant[m].tolist(), strict=True)) for m in meters},
            "ranking": {m: [arms[i] for i in np.argsort(quant[m], kind="stable")] for m in meters},
            "tau": taus, "min_tau": min_tau, "flip": bool(np.isfinite(min_tau) and min_tau < threshold),
            "p_flip_bootstrap": float(np.mean(bmin < threshold)),
        }  # fmt: skip
    flips = sum(1 for f in out["families"].values() if f.get("flip"))
    support = float(np.mean(np.sum(boots_flip, axis=0) >= min_families)) if boots_flip else None
    return out | {"families_flipping": flips, "families_ranked": sum(1 for f in out["families"].values() if "tau" in f), "rule_holds": flips >= min_families, "rule_bootstrap_support": support}


# --------------------------------------------------------------------------- A1 determinism


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Unbiased pass^k for one task: C(c, k) / C(n, k) (all k of k fresh epochs succeed)."""
    return float(comb(c, k, exact=True) / comb(n, k, exact=True)) if n >= k else float("nan")


def pass_at_k(c: int, n: int, k: int) -> float:
    return float(1 - comb(n - c, k, exact=True) / comb(n, k, exact=True)) if n >= k else float("nan")


def determinism_runs(frame: pd.DataFrame, prefix: str = DETERMINISM_PREFIX, epochs: int = DETERMINISM_EPOCHS, s1_main: int = S1_EPOCHS) -> pd.DataFrame:
    """Rows of the Study C subset: each (tier, arm, cell, task) of a Study C plan cell with its main-study runs first
    (S1: its pool's first `s1_main`) then Study C's, the first `epochs` of them, numbered `run`."""
    if "plan_cell" not in frame.columns:
        return frame.iloc[0:0]
    pc = frame["plan_cell"].fillna("").astype(str)
    c_rows = frame[pc.str.startswith(prefix)]
    if c_rows.empty:
        return c_rows
    keys = ["tier", "arm", "cell", "task"]
    want = c_rows[keys].drop_duplicates()
    main, _ = confirmatory_rows(frame, prefix)
    main = run_order(main.merge(want, on=keys))
    main = main[~((main["arm"] == "S1") & (main["run"] >= s1_main))]
    c_rows = run_order(c_rows)
    both = pd.concat([main.assign(_src=0), c_rows.assign(_src=1)], ignore_index=True).sort_values(["_src", "run"], kind="stable")
    both["run"] = both.groupby(keys).cumcount()
    return both[both["run"] < epochs].drop(columns="_src")


def _entropy_norm(keys: list) -> float:
    n = len(keys)
    if n < 2:
        return float("nan")
    _, counts = np.unique(["∅" if k is None else str(k) for k in keys], return_counts=True)
    p = counts / n
    return float(-(p * np.log(p)).sum() / math.log(n))


def determinism(frame: pd.DataFrame, meters=("tokens", "usd", "wall"), prefix: str = DETERMINISM_PREFIX, epochs: int = DETERMINISM_EPOCHS, cluster_by: str = "kb", reps: int = REPS, seed: int = 0, kg_single=("S5",), multi=("M1", "M2", "M7"), alpha: float = 0.025) -> dict:
    """Per (tier, arm, cell) on the Study C subset (module docstring), and KG single-agent vs multi-agent contrasts."""
    runs = determinism_runs(frame, prefix, epochs)
    if runs.empty:
        return {"available": False, "reason": f"no Study C rows ({prefix}*)"}
    per_task = []
    for (tier, arm, cell, world, task), g in runs.groupby(["tier", "arm", "cell", "world", "task"]):
        s = g["success"].to_numpy(dtype=float)
        keys = g["answer_key"].tolist()
        _, counts = np.unique(["∅" if k is None else str(k) for k in keys], return_counts=True)
        row = {"tier": tier, "arm": arm, "cell": cell, "world": world, "task": task, "n": len(s), "c": int(s.sum()), "success": float(s.mean()), "d1": float(len(set(s)) == 1), "modal_share": float(counts.max() / len(keys)), "entropy": _entropy_norm(keys)}
        for m in meters:
            v = g[m].to_numpy(dtype=float) if m in g else np.array([])
            v = v[np.isfinite(v)]
            row[f"cv_{m}"] = float(v.std(ddof=1) / v.mean()) if len(v) > 1 and v.mean() > 0 else float("nan")
        per_task.append(row)
    pt = pd.DataFrame(per_task)
    out: dict = {"available": True, "epochs_target": epochs, "arms": {}, "contrasts": []}
    for (tier, arm, cell), g in pt.groupby(["tier", "arm", "cell"]):
        n = int(g["n"].min())
        out["arms"][f"{arm} / {cell} / {tier}"] = {
            "tier": tier, "arm": arm, "cell": cell, "tasks": int(len(g)), "runs_min": n, "runs_max": int(g["n"].max()),
            "success": float(g["success"].mean()),
            "pass_hat_k": {k: float(np.mean([pass_hat_k(c, nn, k) for c, nn in zip(g["c"], g["n"], strict=True)])) for k in range(1, n + 1)},
            "pass_at_k": {k: float(np.mean([pass_at_k(c, nn, k) for c, nn in zip(g["c"], g["n"], strict=True)])) for k in range(1, n + 1)},
            "d1_outcome_stability": float(g["d1"].mean()),
            "d2_modal_share": float(g["modal_share"].mean()),
            "d2_entropy": float(g["entropy"].mean()),
            "d6": {m: {"median": float(np.nanmedian(g[f"cv_{m}"])) if g[f"cv_{m}"].notna().any() else None, "p90": float(np.nanquantile(g[f"cv_{m}"], 0.9)) if g[f"cv_{m}"].notna().any() else None} for m in meters},
        }  # fmt: skip
    idx = pt.set_index(["tier", "arm", "cell", "world", "task"])
    for tier in sorted(pt["tier"].unique()):
        for a in kg_single:
            for b in multi:
                for metric in ("d1", "modal_share", "success"):
                    try:
                        x, y = idx.loc[(tier, a)][metric], idx.loc[(tier, b)][metric]
                    except KeyError:
                        continue
                    d = (x - y).dropna()
                    if d.empty:
                        continue
                    d.index = d.index.set_names(["cell", "world", "task"])
                    r = infer(Contrast.from_series(d, cluster_by), "descriptive", alpha=alpha, reps=reps, seed=seed)
                    row = {"tier": tier, "contrast": f"{a} − {b}", "metric": metric, "est": r["est"], "ci_t": r["ci_t"], "ci_boot": r["ci_boot"], "tasks": r["tasks"], "clusters": r["clusters"]}
                    if metric == "success":
                        row["matched_accuracy"] = bool(r["ci_t"][0] is not None and -MATCHED_ACCURACY < r["ci_t"][0] and r["ci_t"][1] < MATCHED_ACCURACY)
                    out["contrasts"].append(row)
    return out


# --------------------------------------------------------------------------- invariants


def invariants(tables: Tables, cells=STUDY_B_CELLS, tol: float = 0.03, alpha: float = 0.025, reps: int = REPS, seed: int = 0, tier: str = PRIMARY_TIER) -> dict:
    """§4.4 where the arms exist: the chain S6 ≥ S5 ≥ S7 (each pair within `tol`, failed only on evidence, at
    alpha / pairs) and S5 > S7 (sign flip, one-sided 0.05)."""
    tm = tables.tm.get(tier)
    if tm is None:
        return {"checked": False, "reason": f"no {tier} rows"}
    present = [a for a in ("S6", "S5", "S7") if a in tm.columns]
    out: dict = {"chain": present, "tests": {}, "checked": len(present) >= 2}
    level = alpha / max(1, len(present) - 1)
    for hi, lo in zip(present, present[1:], strict=False):
        cs = tuple(c for c in cells if tm[[hi, lo]][tm.index.get_level_values("cell") == c].dropna().shape[0])
        out["tests"][f"{hi}>={lo}"] = violation_test(tm, hi, lo, cells=cs, tol=tol, alpha=level) if cs else {"pass": None, "reason": "no paired tasks"}
    if {"S5", "S7"} <= set(tm.columns):
        d, _ = member_series(tables, Member("inv", (Term("S5"), Term("S7", -1)), cells, "superiority"))
        if len(d):
            c = Contrast.from_series(d, tables.cluster_by)
            p = SignFlip(c, reps=reps, seed=seed).p(0.0, "greater")
            out["s5_gt_s7"] = {"est": c.est, "p": p, "level": 0.05, "pass": p < 0.05, "clusters": c.G}
    passes = [t.get("pass") for t in out["tests"].values() if t.get("pass") is not None] + ([out["s5_gt_s7"]["pass"]] if "s5_gt_s7" in out else [])
    out["pass"] = all(passes) if passes else None
    return out


# --------------------------------------------------------------------------- GLMM (descriptive)


def _term_name(name: str) -> str:
    """"C(arm, Treatment('S1'))[T.M1]:log_knob" -> "arm[M1]:log_knob"."""
    return re.sub(r"C\(arm, Treatment\('[^']*'\)\)\[T\.([^\]]+)\]", r"arm[\1]", name)


def glmm_rows(frame: pd.DataFrame, s1_epochs: int = S1_EPOCHS, tier: str = PRIMARY_TIER, prefix: str = DETERMINISM_PREFIX) -> pd.DataFrame:
    rows, _ = confirmatory_rows(frame, prefix)
    if "tier" in rows.columns:
        rows = rows[rows["tier"] == tier]
    rows = run_order(rows)
    rows = rows[~((rows["arm"] == "S1") & (rows["run"] >= s1_epochs))]
    return rows[rows["success"].notna()]


def glmm(frame: pd.DataFrame, families=FAMILIES, s1_epochs: int = S1_EPOCHS, tier: str = PRIMARY_TIER, reference: str = "S1") -> dict:
    """The brief's mixed logistic model per family, descriptive (D-029). Every failure is caught and reported."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as e:  # pragma: no cover
        return {"available": False, "reason": f"statsmodels: {e}"}
    rows = glmm_rows(frame, s1_epochs, tier)
    out = {"available": True, "model": "success ~ arm * log_knob + (1 | task) + (1 | task:arm), BinomialBayesMixedGLM.fit_vb (variational Bayes; posterior SDs are typically too small)", "families": {}}
    for fam in families:
        g = rows[rows["cell"].astype(str).str.split("-").str[0] == fam].copy()
        if g.empty or g["arm"].nunique() < 2:
            out["families"][fam] = {"fitted": False, "reason": "fewer than 2 arms"}
            continue
        g["log_knob"] = np.log10(g["cell"].map(_level))
        g["task_arm"] = g["task"].astype(str) + ":" + g["arm"].astype(str)
        g["success"] = g["success"].astype(float)
        ref = reference if reference in set(g["arm"]) else sorted(g["arm"].unique())[0]
        arm_term = f"C(arm, Treatment('{ref}'))"
        formula = f"success ~ {arm_term} * log_knob" if g["log_knob"].nunique() > 1 else f"success ~ {arm_term}"
        entry = {"formula": formula, "rows": int(len(g)), "tasks": int(g["task"].nunique()), "arms": sorted(g["arm"].unique()), "reference": ref}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = BinomialBayesMixedGLM.from_formula(formula, {"task": "0 + C(task)", "task_arm": "0 + C(task_arm)"}, g)
                res = model.fit_vb()
            conv = getattr(res, "optim_retvals", None)
            converged = bool(conv.get("success")) if isinstance(conv, dict) else bool(getattr(conv, "success", False))
            entry |= {
                "fitted": True, "converged": converged,
                "fixed": {_term_name(n): {"mean": float(m), "sd": float(s)} for n, m, s in zip(model.exog_names, res.fe_mean, res.fe_sd, strict=True)},
                "random_sd": {n: float(np.exp(v)) for n, v in zip(model.vcp_names, res.vcp_mean, strict=True)},
                "warnings": sorted({str(w.message)[:200] for w in caught}),
            }  # fmt: skip
        except Exception as e:  # noqa: BLE001  (descriptive: report, never crash the report)
            entry |= {"fitted": False, "reason": f"{type(e).__name__}: {e}"[:300]}
        out["families"][fam] = entry
    return out


# exported for the report: one place for the per-tier, per-arm success and cost summary
def arm_summary(tables: Tables) -> list[dict]:
    out = []
    for tier, tm in tables.tm.items():
        for cell in sorted(set(tm.index.get_level_values("cell").astype(str))):
            sub = tm[tm.index.get_level_values("cell") == cell]
            for arm in sub.columns:
                v = sub[arm].dropna()
                if v.empty:
                    continue
                c = Contrast.from_series(v, tables.cluster_by)
                st = cluster_t(c)
                row = {"tier": tier, "cell": cell, "arm": arm, "tasks": int(len(v)), "clusters": c.G, "success": float(v.mean()), "se": st.get("se")}
                for m in tables.cost.get(tier, {}):
                    row[f"cost_{m}"] = tables.arm_cost(tier, cell, arm, m)
                out.append(row)
    return out
