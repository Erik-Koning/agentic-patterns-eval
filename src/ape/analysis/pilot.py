"""Pilot measurements the gate freezes (FIX_PLAN FX-6, GATE_PREREG §4 and §6): realized context per arm and
cell, the world variance components of the NI power model, and the re-simulated power.

Realized context is the tiktoken count of each compiled context (`compile_log` in the sample store, one record
per compile), so a per-step arm contributes one value per step. Medians are over compile records.

Variance components (`variance_components`) are method-of-moments estimates on the probability scale,
converted to the power model's logit scale by the delta method (σ_logit ≈ σ_p / (p̄(1 − p̄))):
- σ_g (world × arm): per-world paired differences d_w = m_a,w − m_b,w vary between worlds of a cell by
  2σ_g² plus their sampling variance, estimated within each world as var_t(d_wt) / n_t.
- σ_w (world): per-world means s_w = (m_a,w + m_b,w) / 2 vary by σ_w² + σ_g²/2 plus their sampling variance.
Negative estimates are truncated at 0. Cells with fewer than 2 worlds contribute nothing; when no cell has
2 worlds, the components are not estimable and the caller keeps the priors.
"""

import importlib.util
import math
import statistics
from collections import defaultdict
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT

POWER_SIM = ROOT / "power" / "power_sim.py"
PRIOR_SIGMA_W = 0.5  # power_sim.py defaults (logit SD), used where the pilot cannot estimate
PRIOR_SIGMA_G = 0.3
P_CLAMP = (0.05, 0.95)


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
    `gate_stats.task_means`) for arms `a` and `b`; also the pooled success of each and the observed Δ."""
    pair = tm[[a, b]].dropna()
    worlds = pair.groupby(level="cell").apply(lambda g: g.index.get_level_values("world").nunique()).to_dict()
    p_a, p_b = float(pair[a].mean()), float(pair[b].mean())
    out = {"arms": [a, b], "success": {a: p_a, b: p_b}, "observed_delta_pp": 100 * (p_a - p_b), "worlds_per_cell": {c: int(n) for c, n in worlds.items()}}
    g2, w2, weights = [], [], []
    for _cell, g in pair.groupby(level="cell"):
        per_world = g.groupby(level="world")
        if per_world.ngroups < 2:
            continue
        d_t, s_t = g[a] - g[b], (g[a] + g[b]) / 2
        d_w, s_w = d_t.groupby(level="world").mean(), s_t.groupby(level="world").mean()
        n_t = per_world.size()
        noise_d = float((d_t.groupby(level="world").var(ddof=1).fillna(0) / n_t).mean())
        noise_s = float((s_t.groupby(level="world").var(ddof=1).fillna(0) / n_t).mean())
        sg2 = max(0.0, (float(d_w.var(ddof=1)) - noise_d) / 2)
        sw2 = max(0.0, float(s_w.var(ddof=1)) - noise_s - sg2 / 2)
        g2.append(sg2)
        w2.append(sw2)
        weights.append(per_world.ngroups - 1)
    if not weights:
        return out | {"estimable": False, "sigma_w": None, "sigma_g": None, "note": "no cell has 2 or more worlds"}
    p = min(max((p_a + p_b) / 2, P_CLAMP[0]), P_CLAMP[1])
    scale = 1 / (p * (1 - p))
    sg = math.sqrt(np.average(g2, weights=weights)) * scale
    sw = math.sqrt(np.average(w2, weights=weights)) * scale
    return out | {"estimable": True, "sigma_w": sw, "sigma_g": sg, "probability_scale": {"sigma_w": sw / scale, "sigma_g": sg / scale}, "p_bar": p}


@cache
def _power_sim():
    spec = importlib.util.spec_from_file_location("power_sim", POWER_SIM)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ni_power(worlds_per_cell: int, tasks_per_world: int, epochs: int, p_base: float, sigma_w: float, sigma_g: float, n_sims: int, seed: int = 20260929) -> float:
    """`power/power_sim.py`'s NI gate power at Δ = 0 (the pre-registered reference point, GATE_PREREG §4)."""
    p = min(max(p_base, P_CLAMP[0]), P_CLAMP[1])
    return _power_sim().simulate_ni_power(
        worlds_per_cell, tasks_per_world, epochs, 0.0, p_base=p, sigma_w=sigma_w, sigma_g=sigma_g, n_sims=n_sims, rng=np.random.default_rng(seed)
    )


def power_report(vc: dict, sizes: tuple[int, ...], planned: int, tasks_per_world: int, epochs: int, n_sims: int, target: float = 0.8) -> dict:
    """Power at each candidate size under two σ scenarios, and the recommendation.

    - `pilot`: the pilot estimates (the priors where not estimable);
    - `conservative`: the larger of estimate and prior for each σ.
    The recommendation keeps the planned size when it reaches `target` power in the conservative scenario."""
    est_w, est_g = vc.get("sigma_w"), vc.get("sigma_g")
    scenarios = {
        "pilot": (PRIOR_SIGMA_W if est_w is None else est_w, PRIOR_SIGMA_G if est_g is None else est_g),
        "conservative": (max(PRIOR_SIGMA_W, est_w or 0.0), max(PRIOR_SIGMA_G, est_g or 0.0)),
    }
    p_base = vc["success"][vc["arms"][1]]
    powers = {
        name: {"sigma_w": sw, "sigma_g": sg, "power": {n: ni_power(n, tasks_per_world, epochs, p_base, sw, sg, n_sims) for n in sizes}}
        for name, (sw, sg) in scenarios.items()
    }
    cons = powers["conservative"]["power"]
    enough = [n for n in sizes if cons[n] >= target]
    if cons.get(planned, 0.0) >= target:
        rec, note = planned, f"the planned {planned} worlds per cell reach power {cons[planned]:.2f} >= {target} (conservative σ)"
        if enough and min(enough) < planned:
            note += f"; {min(enough)} would also reach it"
    else:
        rec = min(enough) if enough else None
        note = f"the planned {planned} worlds per cell reach only power {cons.get(planned, float('nan')):.2f} (conservative σ)" + (
            f"; {rec} reaches {target}" if rec else f"; no candidate size {list(sizes)} reaches {target}: the analyst decides"
        )
    return {
        "reference": "true Δ = 0, margin 5 pp, one-sided α = 0.025, 4 cells (power/power_sim.py simulate_ni_power)",
        "tasks_per_world": tasks_per_world,
        "epochs": epochs,
        "p_base": p_base,
        "n_sims": n_sims,
        "priors": {"sigma_w": PRIOR_SIGMA_W, "sigma_g": PRIOR_SIGMA_G, "sigma_u": 1.5, "sigma_v": 0.5},
        "scenarios": powers,
        "planned_worlds_per_cell": planned,
        "recommended_worlds_per_cell": rec,
        "recommendation": note,
    }


def load_task_means(log_files: list[str | Path], require_cost: bool, delivery: str = "push") -> pd.DataFrame:
    """`gate_stats.task_means` of one delivery mode's runs in the logs."""
    from .gate_stats import load_results, task_means

    df = load_results(log_files, require_cost=require_cost)
    return task_means(df[df["delivery"] == delivery])
