"""Session-clustered power and type I error for Study G, through the real analysis path (BUILD_PLAN B10; D-029).

    uv run python -m ape.analysis.g_power --sims 2000 --sims-other 500 --workers 8 [--json out.json]

(about 4 minutes on 8 workers; seeded, so the numbers in g_hypotheses' notes reproduce.)

Each simulated study is a session table in `g_load`'s schema (the outcome columns `g_load.sessions_from_items`
produces from item rows; a test checks the two agree), analysed by the same `g_stats` functions `g_report` calls:
`gap_iut` (G-H2a) and `gh3` (G-H3-pre, G-H3a, G-H3b), the confirmatory rows since D-033, and `gh1` (every
single-agent reference) and `tost` (G-H2b's estimate), which D-033 made descriptive: for those the run reports the
estimate's precision and how often its 95% interval covers 0.

The planned design is D-033's: topology cells at Luna-low and Luna-high with 16 sessions each, Sol-high 8, Astra-high
5 (S1 and M2 only), N = 20, 2 epochs; G-H3 pools the three points with all four topology arms.

**Generative model** (logit scale; per capability point c, world w, arm a, epoch e, item position i):

    base arm (S1 / CM0):   logit p = μ_c + u_w + v_wi + g_cwa + e_cwae;  items from the overflow position on fail
    high arm (M2 / O-state): the same + δ_c, no overflow
    mixture arm (M1, S-CM*, CM-x): each item behaves like the high arm with probability m_cwa, else like the base
                           arm (with its own overflow draw); m_cwa = expit(loc + σ_mix h) with E[m] = the scenario's
                           share, so the isolation share (M1), the recovery (S-CM*) and the headroom recovered R_x are
                           exactly the scenario's value in expectation, which places a null exactly at its margin.

- u_w (world), v_wi (item difficulty) are shared by the arms and epochs of a world, and by every point that runs the
  same world (`shared_worlds`, the default: the plan's cells of one length use the same seeds); g (session × arm) and
  e (session × arm × epoch) are fresh per point.
- The overflow position is floor(N × f) + 1 with f = crossing + world and epoch jitter (D-028: the plan's knobs put
  the median crossing at 0.70 of the session for every long cell); the 10-item control never crosses.
- μ_c is solved so the base arm's pre-crossing item success is the scenario's `base[c]`.
- Cost per session-epoch: the arm's cost per item × N × a world factor × a session-epoch factor (lognormal, CV
  `cost_cv` overall). With `cps_ratio`, S-CM*'s cost per item is set so its expected cost per solved item is exactly
  that share of M2's at every point.
- Measured capability: the scenario's true value plus anchor noise (`capability_se`, the standard error of 96
  anchor tasks), drawn once per simulated study and shared by both blocks.

**Priors** (`Variance`, scenarios): σ_world 0.5, σ_item 1.0, σ_arm 0.3, σ_epoch 0.2, σ_mix 0.5; crossing 0.70
± 0.06 (world) ± 0.03 (epoch); base success 0.65 / 0.75 / 0.82 / 0.88 and measured capability 0.72 / 0.80 / 0.86 /
0.90 at Luna-low / Luna-high / Sol-high / Astra-high. They are assumptions until the micro-pilot (g.pilot.luna)
calibrates them; every function takes them as arguments.

**Outputs** (`power_report`): per confirmatory test, the rejection rate at its null (type I; ≤ nominal) and at the
plausible effects (power), each with its Monte Carlo standard error, for the t-test (read at its calibrated level,
`g_hypotheses.t_level`, as the report decides: S-4) and the sign-flip (at α); for the
descriptive rows, the estimate's mean, SD and interval coverage of 0; the minimum attainable p of an exact
session-level sign-flip per confirmatory test (`flip_floor`), with what would fix a test that cannot reach its level;
alternative designs (the design before D-033, 24 Luna sessions per point, 16 Sol sessions, 8 Astra CM sessions, Astra topology on); and
sensitivities (distinct worlds per point, σ_arm 0.6, no capability noise).
"""

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import g_stats as gs
from .g_hypotheses import ALPHA, COST_RATIO, ISOLATION_SHARE, RECOVERY_SHARE, TIER_ORDER, t_level

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(40)
_GH_W = _GH_W / _GH_W.sum()


@dataclass(frozen=True)
class Cell:
    """One capability point of one block at its planned size."""

    point: str
    sessions: int
    epochs: int
    N: int
    arms: tuple[str, ...]
    worlds: str  # cells with the same `worlds` tag run the same worlds (same seeds) when worlds are shared
    plan_cell: str = ""


CM_ARMS = ("CM0", "O-state", "CM-sum", "CM-todo")
PLANNED_CM = (
    Cell("luna-low", 10, 3, 40, CM_ARMS, "n40", "g.cm.luna-low"),
    Cell("luna-high", 10, 3, 40, CM_ARMS, "n40", "g.cm.luna-high"),
    Cell("sol-high", 8, 1, 40, CM_ARMS, "n40", "g.cm.sol-high"),  # D-052: one epoch, more sessions (was 6 x 2)
    Cell("astra-high", 6, 1, 24, CM_ARMS, "n24", "g.cm.astra-high"),  # D-052 (was 4 x 2)
    Cell("luna-high", 10, 3, 10, ("CM0", "O-state"), "n10", "g.cm.luna-n10"),
)
# D-033: a Luna-low topology cell and 16 Luna sessions per point (was Luna-high 8, Sol 8, Astra 5). D-052: Sol runs 12
# sessions x 1 epoch (was 8 x 2). D-051: the Astra topology cell is off by default (ASTRA_TOPO, if it is switched on).
ASTRA_TOPO = Cell("astra-high", 5, 2, 20, ("S1", "M2"), "n20", "g.topo.astra")
PLANNED_TOPO = (
    Cell("luna-low", 16, 2, 20, gs.TOPO_ARMS, "n20", "g.topo.luna-low"),
    Cell("luna-high", 16, 2, 20, gs.TOPO_ARMS, "n20", "g.topo.luna"),
    Cell("sol-high", 12, 1, 20, gs.TOPO_ARMS, "n20", "g.topo.sol"),
)
PREVIOUS_TOPO = (replace(PLANNED_TOPO[1], sessions=8), replace(PLANNED_TOPO[2], sessions=8, epochs=2), ASTRA_TOPO)  # before D-033, for comparison
LOW = {"topo": "S1", "cm": "CM0"}
HIGH = {"topo": "M2", "cm": "O-state"}

CAPABILITY = {"luna-low": 0.72, "luna-high": 0.80, "sol-high": 0.86, "astra-high": 0.90}
BASE = {"luna-low": 0.65, "luna-high": 0.75, "sol-high": 0.82, "astra-high": 0.88}


@dataclass(frozen=True)
class Variance:
    world: float = 0.5
    item: float = 1.0
    arm: float = 0.3
    epoch: float = 0.2
    mix: float = 0.5
    crossing: float = 0.70
    crossing_sd_world: float = 0.06
    crossing_sd_epoch: float = 0.03
    cost_cv: float = 0.25
    capability_se: float = 0.035


@dataclass(frozen=True)
class Scenario:
    name: str
    block: str  # topo | cm
    base: Mapping[str, float]
    shift: Mapping[str, float]  # δ_c: the high arm's logit shift over the base arm (pre-crossing)
    mix: Mapping[str, Mapping[str, float]]  # arm -> point -> mean share of the high arm's behaviour
    capability: Mapping[str, float] = field(default_factory=lambda: dict(CAPABILITY))
    cost: Mapping[str, float] = field(default_factory=lambda: {"S1": 1.0, "M1": 1.88, "M2": 1.88, "S-CM*": 0.87, "CM0": 1.0, "O-state": 0.43, "CM-sum": 0.82, "CM-todo": 0.87})
    cps_ratio: float | None = None
    low_overflows: bool = True
    note: str = ""


def _const(v: float, points=CAPABILITY) -> dict[str, float]:
    return dict.fromkeys(points, v)


def _mean_expit(loc: float, sd: float) -> float:
    return float(_GH_W @ expit(loc + sd * _GH_X))


def _solve_loc(target: float, sd: float) -> float:
    """loc with E[expit(loc + sd Z)] = target (bisection)."""
    if target <= 0:
        return -np.inf
    if target >= 1:
        return np.inf
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _mean_expit(mid, sd) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _overflow_at(N: int, frac: np.ndarray) -> np.ndarray:
    """Position of the first overflowed item (N + 1: none)."""
    return np.clip(np.floor(N * frac).astype(int) + 1, 1, N + 1)


@cache
def _expected_pass_items(N: int, var: Variance, crosses: bool, draws: int = 400_000, seed: int = 7) -> float:
    """E[items before the overflow] for the base arm."""
    if not crosses:
        return float(N)
    rng = np.random.default_rng(seed)
    f = var.crossing + rng.normal(0, var.crossing_sd_world, draws) + rng.normal(0, var.crossing_sd_epoch, draws)
    return float(np.mean(np.minimum(_overflow_at(N, f) - 1, N)))


def _crosses(cell: Cell) -> bool:
    return cell.N > gs.SHORT_N


def expected_solved(cell: Cell, sc: Scenario, var: Variance) -> dict[str, float]:
    """Expected items solved per session-epoch for each arm of a cell (the truth behind CPS ratios)."""
    sd = math.sqrt(var.world**2 + var.item**2 + var.arm**2 + var.epoch**2)
    mu = _solve_loc(sc.base[cell.point], sd)
    p_hi = _mean_expit(mu + sc.shift[cell.point], sd)
    lo_items = _expected_pass_items(cell.N, var, _crosses(cell) and sc.low_overflows)
    hi, lo = cell.N * p_hi, sc.base[cell.point] * lo_items
    out = {LOW[sc.block]: lo, HIGH[sc.block]: hi}
    for a in cell.arms:
        if a not in out:
            m = sc.mix[a][cell.point]
            out[a] = m * hi + (1 - m) * lo
    return out


def simulate_block(sc: Scenario, design: Sequence[Cell], var: Variance, rng: np.random.Generator, *, shared_worlds: bool = True) -> pd.DataFrame:
    """One simulated block (every cell of `design`) as a session table."""
    sd = math.sqrt(var.world**2 + var.item**2 + var.arm**2 + var.epoch**2)
    low, high = LOW[sc.block], HIGH[sc.block]
    worlds: dict[tuple, dict] = {}

    def world(tag: str, i: int, N: int) -> dict:
        key = (tag, i) if shared_worlds else (tag, i, rng.random())
        if key not in worlds:
            worlds[key] = {"id": f"{tag}-w{i:02d}" if shared_worlds else f"{tag}-w{i:02d}-{len(worlds)}", "u": rng.normal(0, var.world), "v": rng.normal(0, var.item, N), "z": rng.normal()}
        return worlds[key]

    rows = []
    for cell in design:
        mu = _solve_loc(sc.base[cell.point], sd)
        delta = sc.shift[cell.point]
        S, E, N = cell.sessions, cell.epochs, cell.N
        ws = [world(cell.worlds, i, N) for i in range(S)]
        u = np.array([w["u"] for w in ws])[:, None, None]
        v = np.stack([w["v"] for w in ws])[:, None, :]
        zf = np.array([w["z"] for w in ws])
        crosses = _crosses(cell) and sc.low_overflows
        ref_cross = _overflow_at(N, var.crossing + var.crossing_sd_world * zf) if _crosses(cell) else np.full(S, N + 1)
        solved_truth = expected_solved(cell, sc, var) if sc.cps_ratio is not None else None
        cost_world = np.exp(rng.normal(0, var.cost_cv / math.sqrt(2), S) - var.cost_cv**2 / 4)
        pos = np.arange(1, N + 1)[None, None, :]
        for arm in cell.arms:
            g = rng.normal(0, var.arm, (S, 1, 1))
            e = rng.normal(0, var.epoch, (S, E, 1))
            eta = mu + u + v + g + e
            p_hi = expit(eta + delta)

            def low_probs() -> tuple[np.ndarray, np.ndarray]:
                if crosses:
                    f = var.crossing + var.crossing_sd_world * zf[:, None] + rng.normal(0, var.crossing_sd_epoch, (S, E))
                    ov = _overflow_at(N, f)
                else:
                    ov = np.full((S, E), N + 1)
                return expit(eta) * (pos < ov[..., None]), ov

            if arm == high:
                p, ov = p_hi, np.full((S, E), N + 1)
            elif arm == low:
                p, ov = low_probs()
            else:
                m_bar = sc.mix[arm][cell.point]
                loc = _solve_loc(m_bar, var.mix)
                m = expit(loc + var.mix * rng.normal(size=(S, 1, 1))) if np.isfinite(loc) else np.full((S, 1, 1), float(m_bar))
                p_lo, ov = low_probs()
                z = rng.random((S, E, N)) < m
                p = np.where(z, p_hi, p_lo)
                ov = np.where((~z & (pos >= ov[..., None])).any(-1), ov, N + 1)  # the overflow only shows if a low item hit it
            y = (rng.random((S, E, N)) < p).astype(int)
            per_item = sc.cost.get(arm, 1.0)
            if solved_truth is not None and arm == "S-CM*":
                per_item = sc.cps_ratio * sc.cost["M2"] * solved_truth["S-CM*"] / solved_truth["M2"]
            cost = per_item * N * cost_world[:, None] * np.exp(rng.normal(0, var.cost_cv / math.sqrt(2), (S, E)) - var.cost_cv**2 / 4)
            for s in range(S):
                for ep in range(E):
                    o = y[s, ep]
                    rows.append({
                        "plan_cell": cell.plan_cell or f"sim.{sc.block}.{cell.point}",
                        "block": sc.block,
                        "point": cell.point,
                        "arm": arm,
                        "session": ws[s]["id"],
                        "epoch": ep + 1,
                        "N": N,
                        "items_solved": float(o.sum()),
                        "n_items": float(N),
                        "item_success": float(o.mean()),
                        "outcomes": o.tolist(),
                        "overflow": bool(ov[s, ep] <= N),
                        "overflow_at": float(ov[s, ep]) if ov[s, ep] <= N else np.nan,
                        "dependency_success": np.nan,
                        "w_crossing_item": float(ref_cross[s]) if ref_cross[s] <= N else np.nan,
                        "session_success": float(o.all()),
                        "cost_usd": float(cost[s, ep]),
                        "tokens": float(cost[s, ep]) * 1e5,
                        "calls": float(6 * N),
                        "wall_clock": float(cost[s, ep]) * 60,
                        "error": False,
                    })
    return pd.DataFrame(rows)


def to_items(sessions: pd.DataFrame) -> pd.DataFrame:
    """Item rows of a simulated session table (for checking it against `g_load.sessions_from_items`)."""
    rows = []
    for _, r in sessions.iterrows():
        ov = r["overflow_at"]
        for i, o in enumerate(r["outcomes"], start=1):
            rows.append({k: r[k] for k in ("plan_cell", "block", "point", "arm", "session", "epoch", "N")} | {"position": i, "success": float(o), "overflow": bool(np.isfinite(ov) and i >= ov and not o), "dependency": False})
    return pd.DataFrame(rows)


def measured_capability(sc: Scenario, var: Variance, rng: np.random.Generator, noise: bool = True) -> dict[str, float]:
    return {c: float(x + (rng.normal(0, var.capability_se) if noise else 0.0)) for c, x in sc.capability.items()}


# ---------- scenarios ----------


def topo_scenario(name: str, *, base=None, shift=None, iso=None, rec=None, cps=None, note: str = "") -> Scenario:
    return Scenario(
        name,
        "topo",
        base=base or dict(BASE),
        shift=shift or _const(0.3),
        mix={"M1": iso or _const(0.75), "S-CM*": rec or _const(0.8)},
        cps_ratio=cps,
        note=note,
    )


def cm_scenario(name: str, *, base=None, shift=None, r_sum=None, r_todo=None, overflow: bool = True, note: str = "") -> Scenario:
    return Scenario(name, "cm", base=base or dict(BASE), shift=shift or _const(0.3), mix={"CM-sum": r_sum or _const(0.6), "CM-todo": r_todo or _const(0.5)}, low_overflows=overflow, note=note)


def _linear_in_capability(r0: float, theta: float, points: Sequence[str], cap: Mapping[str, float] = CAPABILITY) -> dict[str, float]:
    """R_c = r0 + θ (x_c − x̄) / span: the change along the line over the span is θ (OLS slope × span on true x for
    any session weights)."""
    xs = [cap[c] for c in points]
    span, xbar = max(xs) - min(xs), float(np.mean(xs))
    return {c: r0 + theta * (cap[c] - xbar) / span for c in points}


def scenarios(change: float = 0.2) -> dict[str, Scenario]:
    """The named scenarios. `change` is the fall in R_x over the capability span in `cm-falling` (G-H2b, descriptive)."""
    cm_pts = ("luna-low", "luna-high", "sol-high", "astra-high")
    return {
        # topology block
        "topo-null": topo_scenario("topo-null", base=_const(0.80), shift=_const(0.4), iso=_const(ISOLATION_SHARE), rec=_const(RECOVERY_SHARE), cps=COST_RATIO,
                                   note="every point identical (G-H1 slope 0 for every reference); H3a, H3b recovery and cost exactly at their margins"),
        "topo-converge": topo_scenario("topo-converge", shift={"luna-low": 0.9, "luna-high": 0.6, "sol-high": 0.3, "astra-high": 0.0}, rec={"luna-low": 0.5, "luna-high": 0.6, "sol-high": 0.8, "astra-high": 0.9}, cps=0.5,
                                       note="M2's logit advantage falls 0.9 → 0.6 → 0.3 → 0 (Luna-low → Astra) and S-CM*'s recovery rises 0.5 → 0.6 → 0.8 → 0.9; isolation share 0.75"),
        "topo-constant": topo_scenario("topo-constant", shift=_const(0.3), rec=_const(0.7), cps=0.5,
                                       note="no topology × capability interaction in the mechanism (constant logit advantage, recovery 0.7); base success rises with capability"),
        "topo-h3": topo_scenario("topo-h3", iso=_const(0.8), rec=_const(0.95), cps=0.45, note="isolation share 0.8, recovery 0.95, cost ratio 0.45"),
        "topo-h3-mid": topo_scenario("topo-h3-mid", iso=_const(0.7), rec=_const(0.9), cps=0.5, note="isolation share 0.7, recovery 0.9, cost ratio 0.5"),
        "topo-h3-low": topo_scenario("topo-h3-low", iso=_const(0.65), rec=_const(0.85), cps=0.55, note="isolation share 0.65, recovery 0.85, cost ratio 0.55"),
        # context-management block
        "cm-null-gap": cm_scenario("cm-null-gap", shift=_const(0.0), overflow=False, note="O-state ≡ CM0 (no overflow, no shift): Gap_T = 0 at every point"),
        "cm-equiv": cm_scenario("cm-equiv", note="R_x constant across capability (CM-sum 0.6, CM-todo 0.5); CM0 overflows at 0.70"),
        "cm-falling": cm_scenario("cm-falling", r_sum=_linear_in_capability(0.6, -change, cm_pts), r_todo=_linear_in_capability(0.5, -change, cm_pts),
                                  note=f"R_x falls by {change} over the capability span (G-H2b's estimate, descriptive)"),
    }


# ---------- one simulated study through the analysis ----------


def _covers(ci, value: float = 0.0) -> bool | None:
    return None if not ci or ci[0] is None or ci[1] is None else bool(ci[0] <= value <= ci[1])


def analyse_topo(sessions: pd.DataFrame, cap: Mapping[str, float], *, references: Sequence[str] = ("S-CM*", "S1", "S1-pre"), alpha: float = ALPHA, reps: int = 999, seed: int = 0, flip: bool = True) -> dict:
    """G-H1 (descriptive, D-033: the change over the capability span, its standard error and whether its 95% interval
    covers 0) for every reference, and G-H3's confirmatory tests (rejections by t and sign-flip)."""
    out = {}
    for ref in references:
        r = gs.gh1(sessions, cap, reference=ref, alpha=alpha, reps=reps, seed=seed, flip=False)
        se = None if r.get("se") is None or r.get("span") is None else r["se"] * r["span"]
        out[f"G-H1[{ref}]"] = {"est": r.get("est_span"), "se": se, "covers_0": _covers(r.get("ci_span")), "testable": r.get("testable", False)}
        if ref in ("S-CM*", "S1-pre"):  # D-043: capability as the tier rank
            t = gs.gh1(sessions, TIER_ORDER, reference=ref, alpha=alpha, reps=reps, seed=seed, flip=False)
            tse = None if t.get("se") is None or t.get("span") is None else t["se"] * t["span"]
            out[f"G-H1[{ref}|tier]"] = {"est": t.get("est_span"), "se": tse, "covers_0": _covers(t.get("ci_span"))}
    h = gs.gh3(sessions, alpha=alpha, reps=reps, seed=seed, flip=flip, boot=0, meters=())
    if h.get("testable"):
        pre, a, rec, cost = h["pre"], h["h3a"], h["h3b"]["recovery"], h["h3b"]["cost"]
        # S-4: the t decides at each test's calibrated level (as g_report); the sign-flip is read at α
        lv = lambda x: x.get("level", alpha)  # noqa: E731
        out["G-H3-pre"] = {"t": _rej(pre["p_t"], lv(pre)), "flip": _rej(pre["p_flip"], alpha), "est": pre["est"], "se": pre["se"]}
        out["G-H3a"] = {"t": _rej(a["p_t"], lv(a)), "flip": _rej(a["p_flip"], alpha), "est": h["isolation_share"].get("est"), "se": h["isolation_share"].get("se"), "sequence": a["claim"]}
        out["G-H3b.recovery"] = {"t": _rej(rec["p_t"], lv(rec)), "flip": _rej(rec["p_flip"], alpha), "est": h["recovery_share"].get("est"), "se": h["recovery_share"].get("se")}
        out["G-H3b.cost"] = {"t": _rej(cost.get("p_t"), lv(cost)), "flip": _rej(cost.get("p_flip"), alpha), "est": cost.get("ratio"), "se": cost.get("se")}
        out["G-H3b"] = {"t": _rej(rec["p_t"], lv(rec)) and _rej(cost.get("p_t"), lv(cost)), "flip": _rej(rec["p_flip"], alpha) and _rej(cost.get("p_flip"), alpha), "sequence": h["h3b"]["claim"]}
    return out


def analyse_cm(sessions: pd.DataFrame, cap: Mapping[str, float], *, estimand: str = "R", alpha: float = ALPHA, reps: int = 999, seed: int = 0, flip: bool = True) -> dict:
    """G-H2a's confirmatory tests (per point and the intersection-union decision), and G-H2b (descriptive, D-033): the
    change in R_x over the capability span, its standard error and whether its 95% interval covers 0."""
    out = {}
    g = gs.gap_iut(sessions, alpha=alpha, reps=reps, seed=seed, flip=flip)
    for c, r in g["points"].items():
        out[f"G-H2a[{c}]"] = {"t": bool(r["reject"]), "flip": _rej(r["p_flip"], alpha), "est": r["est"], "se": r["se"]}
    out["G-H2a"] = {"t": g["all_positive"], "flip": g["all_positive_flip"]}
    for s in ("CM-sum", "CM-todo"):
        r = gs.tost(sessions, cap, s, estimand=estimand, alpha=alpha, reps=reps, seed=seed, flip=False)
        out[f"G-H2b[{s}]"] = {"est": r.get("est"), "se": r.get("se"), "covers_0": _covers(r.get("ci"))}
        t = gs.tost(sessions, TIER_ORDER, s, estimand=estimand, alpha=alpha, reps=reps, seed=seed, flip=False)  # D-043
        out[f"G-H2b[{s}|tier]"] = {"est": t.get("est"), "se": t.get("se"), "covers_0": _covers(t.get("ci"))}
    return out


def _rej(p, alpha) -> bool:
    return p is not None and p <= alpha


def run(sc: Scenario, n_sims: int, *, design: Sequence[Cell] | None = None, var: Variance = Variance(), alpha: float = ALPHA, reps: int = 999, seed: int = 0, shared_worlds: bool = True, capability_noise: bool = True, flip: bool = True, **analysis) -> dict:
    """Rejection rates (confirmatory rows), interval coverage of 0 (descriptive rows) and the estimates' mean and SD
    over `n_sims` simulated studies of the scenario's block."""
    design = design or (PLANNED_TOPO if sc.block == "topo" else PLANNED_CM)
    rng = np.random.default_rng(seed)
    acc: dict[str, dict[str, list]] = {}
    t0 = time.time()
    for i in range(n_sims):
        sessions = simulate_block(sc, design, var, rng, shared_worlds=shared_worlds)
        cap = measured_capability(sc, var, rng, capability_noise)
        res = (analyse_topo if sc.block == "topo" else analyse_cm)(sessions, cap, alpha=alpha, reps=reps, seed=seed + i, flip=flip, **analysis)
        for test, r in res.items():
            a = acc.setdefault(test, {})
            for k, v in r.items():
                a.setdefault(k, []).append(v)
    out = {"scenario": sc.name, "note": sc.note, "block": sc.block, "sims": n_sims, "seconds": round(time.time() - t0, 1), "tests": {}}
    for test, a in acc.items():
        row = {}
        for k, vals in a.items():
            if k in ("est", "se"):
                x = np.array([np.nan if v is None else float(v) for v in vals])
                fin = x[np.isfinite(x)]
                row[f"mean_{k}"] = float(fin.mean()) if len(fin) else None
                if k == "est":
                    row["sd_est"] = float(fin.std(ddof=1)) if len(fin) > 1 else None
            elif k == "testable":
                row["testable_rate"] = float(np.mean(vals))
            else:
                known = [bool(v) for v in vals if v is not None]
                r = float(np.mean(known)) if known else None
                row[k] = r
                row[f"{k}_mcse"] = None if r is None else float(math.sqrt(r * (1 - r) / len(known)))
        out["tests"][test] = row
    return out


# ---------- sign-flip floors ----------


def _clusters(design: Sequence[Cell], points: Sequence[str], shared: bool, arms: Sequence[str] = ()) -> int:
    cells = [c for c in design if c.point in points and c.N > gs.SHORT_N and all(a in c.arms for a in arms)]
    if not shared:
        return sum(c.sessions for c in cells)
    by: dict[str, int] = {}
    for c in cells:
        by[c.worlds] = max(by.get(c.worlds, 0), c.sessions)
    return sum(by.values())


def flip_floor(alpha: float = ALPHA, *, cm: Sequence[Cell] = PLANNED_CM, topo: Sequence[Cell] = PLANNED_TOPO, shared: bool = True) -> list[dict]:
    """Per confirmatory test: the clusters its sign-flip enumerates, the smallest p it can give (2^−G), the level it
    must reach, whether it can, and what would fix it. (G-H1 and G-H2b are descriptive since D-033.)"""
    rows = []

    def add(test: str, G: int, level: float, note: str = "") -> None:
        mp = gs.min_flip_p(G) if G else None
        ok = mp is not None and mp <= level
        need = gs.min_sessions_for(level)
        fix = "" if ok else f"≥ {need} independent sessions (2^−{need} = {2.0 ** -need:.4f} ≤ {level:g}), or decide on the t-test"
        rows.append({"test": test, "clusters": G, "min_p": mp, "level": level, "reachable": ok, "fix": fix, "note": note})

    for c in cm:
        if c.N > gs.SHORT_N:
            add(f"G-H2a [{c.point}]", c.sessions, alpha, "each point of the intersection-union test at α")
    h3 = [c.point for c in topo if set(gs.TOPO_ARMS) <= set(c.arms)]
    add("G-H3 (pre, a, b)", _clusters(topo, h3, shared), alpha, f"points {h3}; fixed sequence at α")
    return rows


# ---------- the report ----------


def designs() -> dict[str, tuple[str, tuple[Cell, ...]]]:
    """Alternative designs `power_report` simulates: the design before D-033 for comparison, and what would improve the
    remaining confirmatory rows further."""
    astra8 = tuple(replace(c, sessions=8) if c.point == "astra-high" and c.N > gs.SHORT_N else c for c in PLANNED_CM)
    luna24 = tuple(replace(c, sessions=24) if c.point.startswith("luna") else c for c in PLANNED_TOPO)
    sol16 = tuple(replace(c, sessions=16) if c.point == "sol-high" else c for c in PLANNED_TOPO)
    return {
        "cm-astra8": ("cm", astra8),
        "topo-before-D-033": ("topo", PREVIOUS_TOPO),
        "topo-luna-24": ("topo", luna24),
        "topo-sol-16": ("topo", sol16),
        "topo-with-astra": ("topo", (*PLANNED_TOPO, ASTRA_TOPO)),
    }


def _job(args: tuple) -> tuple[tuple, dict]:
    key, sc, n_sims, kw = args
    return key, run(sc, n_sims, **kw)


def mde(sd: float | None, alpha: float = ALPHA, power: float = 0.8) -> float | None:
    """Normal-approximation minimum detectable effect of a one-sided test at `alpha` with `power`, from the estimate's
    sampling SD (for a descriptive estimate: (z_0.975 + z_0.8) × SD, the effect its 95% interval excludes 0 for 80%
    of the time)."""
    from scipy.stats import norm

    if sd is None:
        return None
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * sd)


def power_report(n_sims: int = 2000, *, n_other: int | None = None, seed: int = 20261003, reps: int = 999, var: Variance = Variance(), quick: bool = False, workers: int = 1) -> dict:
    """Every scenario at the planned sizes (`n_sims` each); unless `quick`, with `n_other` each (default n_sims / 4) the
    alternative designs and the sensitivities. Plus the sign-flip floors and normal-approximation MDEs. `workers` > 1
    runs the simulations in processes."""
    n_other = n_other or max(n_sims // 4, 50)
    sc = scenarios()
    base = {"var": var, "reps": reps}
    jobs: list[tuple] = [(("planned", name), s, n_sims, base | {"seed": seed + 100 * i}) for i, (name, s) in enumerate(sc.items())]
    if not quick:
        for name, (block, design) in designs().items():
            keys = ("cm-null-gap", "cm-equiv") if block == "cm" else ("topo-null", "topo-converge", "topo-h3", "topo-h3-mid", "topo-h3-low")
            jobs += [(("alternatives", name, k), sc[k], n_other, base | {"seed": seed + 11, "design": design}) for k in keys]
        jobs += [(("sensitivity", "distinct-worlds", k), sc[k], n_other, base | {"seed": seed + 17, "shared_worlds": False}) for k in ("topo-null", "topo-h3", "topo-h3-mid", "cm-null-gap")]
        jobs += [(("sensitivity", "sigma_arm-0.6", k), sc[k], n_other, base | {"seed": seed + 19, "var": replace(var, arm=0.6)}) for k in ("topo-null", "topo-h3", "topo-h3-mid", "topo-converge", "cm-null-gap", "cm-equiv")]
        jobs += [(("sensitivity", "no-capability-noise", k), sc[k], n_other, base | {"seed": seed + 13, "capability_noise": False}) for k in ("topo-converge", "cm-falling")]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    out: dict = {"sims": n_sims, "sims_other": n_other, "variance": dict(var.__dict__), "alpha": ALPHA, "scenarios": {k: v.note for k, v in sc.items()}, "planned": {}}
    for key, r in results:
        node = out
        for k in key[:-1]:
            node = node.setdefault(k, {})
        node[key[-1]] = r
    for block in ("planned", "alternatives", "sensitivity"):
        for r in _walk(out.get(block, {})):
            for t in r["tests"].values():
                if t.get("sd_est") is not None:
                    t["mde80"] = mde(t["sd_est"])
    out["flip_floor"] = flip_floor()
    out["flip_floor_distinct_worlds"] = flip_floor(shared=False)
    return out


def _walk(node: dict):
    if "tests" in node:
        yield node
        return
    for v in node.values():
        if isinstance(v, dict):
            yield from _walk(v)


def render(d: dict) -> str:
    L = [f"# Study G power and type I error (one-sided α = {d['alpha']}; {d['sims']} simulated studies per planned row, {d.get('sims_other', '–')} otherwise)", ""]
    L += [
        f"Confirmatory rows (G-H2a, G-H3-pre, G-H3a, G-H3b): P(reject) by the session-clustered t ('t', at the calibrated level "
        f"{t_level(d['alpha']):g}: S-4) and the wild sign-flip ('flip', at α), ± Monte Carlo SE. Descriptive rows (G-H1, G-H2b; D-033): 'covers 0' is the share of 95% intervals that "
        "include 0. sd: the estimate's sampling SD across simulated studies; MDE80: (z_0.975 + z_0.8) × sd.",
        "",
    ]
    L += ["Scenarios:", ""] + [f"- **{k}**: {v}" for k, v in d.get("scenarios", {}).items()] + [""]

    def table(block: dict) -> None:
        L.append("| scenario | row | t | flip | covers 0 | mean estimate | sd | MDE80 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in _walk(block):
            for test, x in r["tests"].items():
                t = "–" if x.get("t") is None else f"{x['t']:.3f} ± {x['t_mcse']:.3f}"
                f = "–" if x.get("flip") is None else f"{x['flip']:.3f}"
                c = "–" if x.get("covers_0") is None else f"{x['covers_0']:.3f}"
                e = "–" if x.get("mean_est") is None else f"{x['mean_est']:+.3f}"
                sd = "–" if x.get("sd_est") is None else f"{x['sd_est']:.3f}"
                m = "–" if x.get("mde80") is None else f"{x['mde80']:.3f}"
                L.append(f"| {r['scenario']} | {test} | {t} | {f} | {c} | {e} | {sd} | {m} |")
        L.append("")

    L += ["## Planned sizes", ""]
    table(d["planned"])
    for key, title in (("alternatives", "Alternative designs"), ("sensitivity", "Sensitivity")):
        for name, block in d.get(key, {}).items():
            L += [f"## {title}: {name}", ""]
            table(block)
    L += ["## Exact sign-flip floors (planned design, shared worlds)", "", "| test | clusters | min p | level | reachable | fix |", "|---|---|---|---|---|---|"]
    L += [f"| {r['test']} | {r['clusters']} | {r['min_p']:.4g} | {r['level']:g} | {'yes' if r['reachable'] else 'NO'} | {r['fix']} |" for r in d["flip_floor"]]
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--sims-other", type=int, default=None, help="per row for alternatives and sensitivities (default sims / 4)")
    ap.add_argument("--reps", type=int, default=999, help="sign-flip resamples above the exact-enumeration limit")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="planned scenarios only")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args(argv)
    d = power_report(args.sims, n_other=args.sims_other, seed=args.seed, reps=args.reps, quick=args.quick, workers=args.workers)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(d, f, indent=1, default=str)
    print(render(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
