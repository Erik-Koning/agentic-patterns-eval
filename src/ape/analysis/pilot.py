"""Pilot measurements the gate freezes (FIX_PLAN FX-6, GATE_PREREG §4 and §6): realized context per arm and
cell, the world variance components of the NI power model, and the re-simulated power.

Realized context is the tiktoken count of each compiled context (`compile_log` in the sample store, one record
per compile), so a per-step arm contributes one value per step. Medians are over compile records.

Variance components (`variance_components`; D-023). The power model (`power/power_sim.py`) is on the logit scale:
logit p = μ_c + u_w (σ_w) + g_w,arm (σ_g) + v_t (σ_u) + e_t,arm (σ_v). The pilot observes, per cell, two
probability-scale moments of its world means (noise-corrected by the within-world task variance):
- D_c: the between-world variance of the paired difference d_w = m_a,w − m_b,w (σ_g's signature);
- S_c: the between-world variance of s_w = (m_a,w + m_b,w) / 2 less D_c / 4 (σ_w's signature).
The model's population values of the same moments, D*_c(σ_w, σ_g) and S*_c(σ_w, σ_g), come from Gauss–Hermite
quadrature at the cell's own baseline μ_c (solved so the model's mean success is the cell's observed one) with
σ_u, σ_v at their priors. σ_g² and σ_w² are the (m_c − 1)-weighted means over cells of D_c / a_c and S_c / b_c,
where a_c = D*_c / σ_g² and b_c = S*_c / σ_w² are the model's secant slopes at the current estimate (a fixed
point, a few iterations). This replaces the delta-method conversion σ_p / (p̄(1 − p̄)), which ignored task
heterogeneity (the mean slope E[p(1 − p)] is only 0.67–0.71 of p̄(1 − p̄)) and so underestimated σ by about
40% (RELIABILITY_REVIEW S3).

Uncertainty: under normal world means, Σ_c (m_c − 1) V_c / E[V_c] is χ² with Σ(m_c − 1) df, where V_c is the raw
between-world variance and E[V_c] = D*_c(σ) + noise_c. Inverting that pivot in σ gives an 80% interval
(`ci80`). When the worlds vary less than the model predicts even at σ = 0, the pivot has no upper solution: the
upper end is then *not informative* (`ci80_upper_informative`), is shown as max(estimate, prior), and can never
support a "suffices" call (D-026; it used to collapse to 0, which made the conservative scenario optimistic).
Cells with fewer than 2 worlds contribute nothing; with none, the components are not estimable and the priors stay.

Recommendation (`power_report`, D-026): two-sided, at the planned test size. *suffices* when the power at the upper
ends of the intervals reaches the target; *insufficient* when even the lower ends miss it (add test worlds within
budget, or rethink; the powers at 20 and 24 worlds are reported); *ambiguous* otherwise (proceed with the planned
size and rely on the pre-registered extension).
"""

import importlib.util
import math
import statistics
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2

from ..config import ROOT

POWER_SIM = ROOT / "power" / "power_sim.py"
PRIOR_SIGMA_W = 0.5  # power_sim.py defaults (logit SD), used where the pilot cannot estimate
PRIOR_SIGMA_G = 0.3
SIGMA_U, SIGMA_V = 1.5, 0.5  # power_sim.py's task-difficulty and task x arm priors (not estimated by the pilot)
P_CLAMP = (0.05, 0.95)
CI_LEVEL = 0.80  # the σ interval; the conservative scenario takes its upper ends
SIGMA_MAX = 5.0
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(16)  # probabilists' Gauss–Hermite: E[f(Z)] = Σ w f(x) / √(2π)
_GH_W = _GH_W / _GH_W.sum()


def _h(x: np.ndarray) -> np.ndarray:
    """Population mean success over tasks given the world-and-arm logit `x`: E_z[expit(x + s z)], s² = σ_u² + σ_v²."""
    s = math.hypot(SIGMA_U, SIGMA_V)
    return expit(np.asarray(x)[..., None] + s * _GH_X) @ _GH_W


def _mu_for(p: float, sigma_w: float, sigma_g: float) -> float:
    """The cell baseline μ whose model mean success is `p` (bisection)."""
    sd = math.hypot(sigma_w, sigma_g)
    lo, hi = -12.0, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if float(_GH_W @ _h(mid + sd * _GH_X)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def model_moments(p: float, sigma_w: float, sigma_g: float) -> tuple[float, float]:
    """(D*, S*): the model's population between-world variance of the paired difference, and of the world mean
    less D*/4, at mean success `p` (3-D Gauss–Hermite quadrature over u, g_a, g_b)."""
    mu = _mu_for(p, sigma_w, sigma_g)
    u = sigma_w * _GH_X[:, None, None]
    ga, gb = sigma_g * _GH_X[None, :, None], sigma_g * _GH_X[None, None, :]
    a, b = _h(mu + u + ga), _h(mu + u + gb)
    w = _GH_W[:, None, None] * _GH_W[None, :, None] * _GH_W[None, None, :]
    d, s = a - b, (a + b) / 2
    var = lambda x: float(np.sum(w * x**2) - np.sum(w * x) ** 2)  # noqa: E731
    dd = var(d)
    return dd, var(s) - dd / 4


def _fit(cells: list[dict], sigma_w: float, sigma_g: float, iterations: int = 8) -> tuple[float, float]:
    """The fixed point of the secant-slope moment equations (module docstring), from a starting (σ_w, σ_g)."""
    floor = 0.05
    for _ in range(iterations):
        num_g = num_w = den = 0.0
        for c in cells:
            dg, sg_ = model_moments(c["p"], max(sigma_w, floor), max(sigma_g, floor))
            a = dg / max(sigma_g, floor) ** 2
            b = sg_ / max(sigma_w, floor) ** 2
            num_g += c["weight"] * c["D"] / a
            num_w += c["weight"] * c["S"] / b
            den += c["weight"]
        sigma_g, sigma_w = math.sqrt(max(0.0, num_g / den)), math.sqrt(max(0.0, num_w / den))
    return sigma_w, sigma_g


def _pivot_bounds(cells: list[dict], key: str, fixed: float, est: float, level: float) -> tuple[list[float], bool]:
    """The CI for σ_g (key "D", σ_w fixed) or σ_w (key "S", σ_g fixed) from the χ² pivot on raw world variances, and
    whether its upper end is informative. The pivot statistic falls as σ rises; when even σ = 0 leaves it at or below
    the lower χ² quantile (the worlds vary less than the model predicts at σ = 0), no σ solves the upper end: it is
    reported as max(estimate, prior) and flagged (module docstring)."""
    prior = PRIOR_SIGMA_G if key == "D" else PRIOR_SIGMA_W
    df = sum(c["weight"] for c in cells)

    def q(sig: float) -> float:
        tot = 0.0
        for c in cells:
            sw, sg = (fixed, sig) if key == "D" else (sig, fixed)
            dd, ss = model_moments(c["p"], sw, sg)
            expect = dd + c["noise_d"] if key == "D" else ss + dd / 4 + c["noise_s"]
            raw = c["raw_d"] if key == "D" else c["raw_s"]
            tot += c["weight"] * raw / max(expect, 1e-12)
        return tot

    def solve(target: float) -> float | None:
        if q(0.0) <= target:
            return None  # no σ ≥ 0 reaches the target
        if q(SIGMA_MAX) >= target:
            return SIGMA_MAX
        lo, hi = 0.0, SIGMA_MAX
        for _ in range(25):
            mid = (lo + hi) / 2
            if q(mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    tail = (1 - level) / 2
    lower, upper = solve(float(chi2.ppf(1 - tail, df))), solve(float(chi2.ppf(tail, df)))
    lo = min(est, 0.0 if lower is None else lower)  # no solution at the lower end: σ = 0 is inside the interval
    if upper is None:
        return [lo, max(est, prior)], False
    return [lo, max(est, upper)], True


def realized_tokens(log_files, source: str = "push") -> dict[tuple[str, str], list[int]]:
    """(arm, cell) -> realized context tokens of every compile of `source` ("push" or "pull") in the logs."""
    from inspect_ai.log import read_eval_log

    out: dict[tuple[str, str], list[int]] = defaultdict(list)
    for f in log_files:
        log = read_eval_log(str(f))
        args = log.eval.task_args or {}
        arm = args.get("arm") or (log.eval.metadata or {}).get("arm")
        for s in log.samples or []:
            md = s.metadata or {}
            cell = f"{md.get('family', args.get('family'))}-{md.get('level', args.get('level'))}"
            out[(arm, cell)] += [int(r["tokens"]) for r in (s.store or {}).get("compile_log", []) if r.get("source", "push") == source]
    return dict(out)


def median_or_none(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def variance_components(tm: pd.DataFrame, a: str, b: str) -> dict:
    """σ_w and σ_g (logit scale) from task means `tm` (rows cell, world, task; one column per arm; see
    `gate_stats.task_means`) for arms `a` and `b`, by moment matching to the power model (module docstring), with
    80% intervals; also the pooled and per-cell success of each arm and the observed Δ."""
    pair = tm[[a, b]].dropna()
    worlds = pair.groupby(level="cell").apply(lambda g: g.index.get_level_values("world").nunique()).to_dict()
    p_a, p_b = float(pair[a].mean()), float(pair[b].mean())
    cell_success = {str(c): {a: float(g[a].mean()), b: float(g[b].mean())} for c, g in pair.groupby(level="cell")}
    out = {
        "arms": [a, b],
        "success": {a: p_a, b: p_b},
        "cell_success": cell_success,
        "observed_delta_pp": 100 * (p_a - p_b),
        "worlds_per_cell": {c: int(n) for c, n in worlds.items()},
        "method": "moment matching to power_sim's logit model (Gauss–Hermite), χ² pivot for the 80% interval (D-023)",
    }
    cells = []
    for cell, g in pair.groupby(level="cell"):
        per_world = g.groupby(level="world")
        if per_world.ngroups < 2:
            continue
        d_t, s_t = g[a] - g[b], (g[a] + g[b]) / 2
        d_w, s_w = d_t.groupby(level="world").mean(), s_t.groupby(level="world").mean()
        n_t = per_world.size()
        noise_d = float((d_t.groupby(level="world").var(ddof=1).fillna(0) / n_t).mean())
        noise_s = float((s_t.groupby(level="world").var(ddof=1).fillna(0) / n_t).mean())
        raw_d, raw_s = float(d_w.var(ddof=1)), float(s_w.var(ddof=1))
        dd = raw_d - noise_d
        p = min(max((cell_success[str(cell)][a] + cell_success[str(cell)][b]) / 2, P_CLAMP[0]), P_CLAMP[1])
        cells.append({"cell": str(cell), "p": p, "weight": per_world.ngroups - 1, "D": dd, "S": raw_s - noise_s - dd / 4, "raw_d": raw_d, "raw_s": raw_s, "noise_d": noise_d, "noise_s": noise_s})
    if not cells:
        return out | {"estimable": False, "sigma_w": None, "sigma_g": None, "ci80": None, "note": "no cell has 2 or more worlds"}
    sw, sg = _fit(cells, PRIOR_SIGMA_W, PRIOR_SIGMA_G)
    (ci_g, inf_g), (ci_w, inf_w) = _pivot_bounds(cells, "D", sw, sg, CI_LEVEL), _pivot_bounds(cells, "S", sg, sw, CI_LEVEL)
    ci, informative = {"sigma_g": ci_g, "sigma_w": ci_w}, {"sigma_g": inf_g, "sigma_w": inf_w}
    prob = {"sigma_w": math.sqrt(max(0.0, np.average([c["S"] for c in cells], weights=[c["weight"] for c in cells]))), "sigma_g": math.sqrt(max(0.0, np.average([c["D"] for c in cells], weights=[c["weight"] for c in cells]) / 2))}
    return out | {
        "estimable": True,
        "sigma_w": sw,
        "sigma_g": sg,
        "ci80": ci,
        "ci80_upper_informative": informative,
        "probability_scale": prob,
        "df": int(sum(c["weight"] for c in cells)),
        "cells_used": [c["cell"] for c in cells],
    }


@cache
def _power_sim():
    spec = importlib.util.spec_from_file_location("power_sim", POWER_SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ni_power(worlds_per_cell: int, tasks_per_world: int, epochs: int, p_base: float | list[float], sigma_w: float, sigma_g: float, n_sims: int, seed: int = 20260929, alpha: float | None = None, modes: int = 2) -> float:
    """`power/power_sim.py`'s gate power at Δ = 0 (the pre-registered reference point, GATE_PREREG §4): by default
    the probability of an unqualified GO (both delivery modes, Holm at the gate run's α; D-023)."""
    from .gate_stats import STAGE_ALPHA

    p = np.clip(np.asarray(p_base, dtype=float), *P_CLAMP)
    return _power_sim().simulate_ni_power(
        worlds_per_cell, tasks_per_world, epochs, 0.0, p_base=p.tolist() if p.ndim else float(p), sigma_w=sigma_w, sigma_g=sigma_g,
        alpha=STAGE_ALPHA["stage1"] if alpha is None else alpha, modes=modes, n_sims=n_sims, rng=np.random.default_rng(seed),
    )


def _cell_bases(vc: dict) -> float | list[float]:
    """The LightRAG arm's success per gate cell (GATE_CELLS order), or its pooled success when a cell is missing."""
    from .gate_stats import GATE_CELLS

    b = vc["arms"][1]
    per = vc.get("cell_success") or {}
    return [per[c][b] for c in GATE_CELLS] if all(c in per for c in GATE_CELLS) else vc["success"][b]


def power_report(vc: dict, sizes: tuple[int, ...], planned: int, tasks_per_world: int, epochs: int, n_sims: int, target: float = 0.8) -> dict:
    """Power of an unqualified GO at each candidate size under four σ scenarios, and the two-sided recommendation.

    - `pilot`: the pilot's point estimates (the priors where not estimable);
    - `conservative`: the lowest power over the four corners of the pilot's 80% σ_w × σ_g intervals (the priors
      where not estimable). Power falls with σ_g but *rises* with σ_w (a world effect moves both arms alike), so the
      worst corner is usually (σ_w low, σ_g high); taking both upper ends overstated power by 2-6 pp (D-027);
    - `optimistic`: the highest power over the same four corners;
    - `prior`: power_sim's priors, for reference.
    `corners` holds every corner's powers; `conservative` and `optimistic` name the corner that attains them at the
    planned size.

    `decision` at the planned size (D-026, D-027):
    - `suffices`: the conservative power reaches `target` and both upper ends are informative;
    - `insufficient`: even the optimistic power misses `target`;
    - `ambiguous`: otherwise, and whenever σ is not estimable.
    `recommended_worlds_per_cell` is the planned size, except when insufficient: then None, and the analyst adds test
    worlds within budget or rethinks, with the powers at the larger `sizes` reported."""
    from .gate_stats import STAGE_ALPHA

    est_w, est_g, ci = vc.get("sigma_w"), vc.get("sigma_g"), vc.get("ci80") or {}
    informative = vc.get("ci80_upper_informative") or {}
    priors = (PRIOR_SIGMA_W, PRIOR_SIGMA_G)
    sizes = tuple(sorted(set(sizes) | {planned}))
    p_base = _cell_bases(vc)
    memo: dict[tuple[float, float], dict[int, float]] = {}

    def scenario(sw: float, sg: float) -> dict:
        key = (float(sw), float(sg))
        if key not in memo:  # coinciding corners (a degenerate interval) are simulated once
            memo[key] = {n: ni_power(n, tasks_per_world, epochs, p_base, key[0], key[1], n_sims) for n in sizes}
        return {"sigma_w": key[0], "sigma_g": key[1], "power": memo[key]}

    corners = (
        {f"sigma_w {wl}, sigma_g {gl}": scenario(ci["sigma_w"][wi], ci["sigma_g"][gi]) for wi, wl in ((0, "low"), (1, "high")) for gi, gl in ((0, "low"), (1, "high"))}
        if ci
        else {"prior": scenario(*priors)}
    )

    def extreme(pick) -> dict:
        """The corner-wise min (conservative) or max (optimistic) power at every size, named by its corner at `planned`."""
        name = pick(corners, key=lambda k: corners[k]["power"][planned])
        return {"sigma_w": corners[name]["sigma_w"], "sigma_g": corners[name]["sigma_g"], "corner": name, "power": {n: pick(c["power"][n] for c in corners.values()) for n in sizes}}

    powers = {
        "pilot": scenario(PRIOR_SIGMA_W if est_w is None else est_w, PRIOR_SIGMA_G if est_g is None else est_g),
        "conservative": extreme(min),
        "optimistic": extreme(max),
        "prior": scenario(*priors),
    }
    cons, opt, point = (powers[k]["power"] for k in ("conservative", "optimistic", "pilot"))
    upper_ok = bool(ci) and all(informative.get(k, True) for k in ("sigma_w", "sigma_g"))
    larger = [n for n in sizes if n > planned]
    if not ci:
        decision = "ambiguous"
        note = (
            f"ambiguous: σ is not estimable from the pilot ({vc.get('note', 'too few worlds')}); under the priors the planned {planned} worlds "
            f"per cell reach power {point[planned]:.2f}. Proceed with {planned} and rely on the pre-registered extension"
        )
    elif opt[planned] < target:
        decision = "insufficient"
        note = (
            f"insufficient: even at the most favourable corner of the pilot's 80% σ intervals the planned {planned} worlds per cell reach only power "
            f"{opt[planned]:.2f} < {target} (point estimate {point[planned]:.2f}). Add test worlds within budget, or rethink"
            + (": " + "; ".join(f"{n} worlds give {point[n]:.2f} at the point estimate and {cons[n]:.2f} at the least favourable corner" for n in larger) if larger else "")
        )
    elif upper_ok and cons[planned] >= target:
        decision = "suffices"
        note = (
            f"suffices: the planned {planned} worlds per cell reach power {cons[planned]:.2f} >= {target} at the least favourable corner "
            f"of the pilot's 80% σ intervals ({powers['conservative']['corner']})"
        )
        if smaller := [n for n in sizes if n < planned and cons[n] >= target]:
            note += f"; {min(smaller)} would also reach it"
    else:
        decision = "ambiguous"
        flagged = [k for k in ("sigma_w", "sigma_g") if not informative.get(k, True)]
        why = (
            f"the upper end of {' and '.join(flagged)} is not informative (the worlds varied less than the model predicts at σ = 0)"
            if flagged
            else f"power {cons[planned]:.2f} at the least and {opt[planned]:.2f} at the most favourable corner of the 80% σ intervals"
        )
        note = f"ambiguous: {why}; point estimate {point[planned]:.2f}. Proceed with the planned {planned} worlds per cell and rely on the pre-registered extension"
    return {
        "reference": f"P(unqualified GO) at true Δ = 0: margin 5 pp, push and pull with Holm at one-sided α = {STAGE_ALPHA['stage1']} (the gate run, D-023), 4 cells, world-clustered t (power/power_sim.py simulate_ni_power)",
        "tasks_per_world": tasks_per_world,
        "epochs": epochs,
        "p_base": p_base,
        "n_sims": n_sims,
        "target": target,
        "priors": {"sigma_w": PRIOR_SIGMA_W, "sigma_g": PRIOR_SIGMA_G, "sigma_u": SIGMA_U, "sigma_v": SIGMA_V},
        "sigma_ci80": ci or None,
        "sigma_ci80_upper_informative": informative or None,
        "scenarios": powers,
        "corners": corners,
        "planned_worlds_per_cell": planned,
        "decision": decision,
        "recommended_worlds_per_cell": None if decision == "insufficient" else planned,
        "recommendation": note,
    }


def load_task_means(log_files: list[str | Path], require_cost: bool, delivery: str = "push") -> pd.DataFrame:
    """`gate_stats.task_means` of one delivery mode's runs in the logs."""
    from .gate_stats import load_results, task_means

    df = load_results(log_files, require_cost=require_cost)
    return task_means(df[df["delivery"] == delivery])
