"""Power and type I error of every confirmatory main-study test, simulated through the analysis itself (BUILD_PLAN B4).

    uv run python -m ape.analysis.main_power --reps 1000 [--seed 1] [--sigma-w 0.5 --sigma-g 0.3] [--pilot vc.json] [--out power.json]

**Model** (`power/power_sim.py`'s, extended to the main study's designs). For tier r, cell c, world w, task t, arm a:

    logit P(success) = x_{r,c,a} + u_w (σ_w) + g_{r,c,w,a} (σ_g) + v_t (σ_u) + e_{r,t,a} (σ_v)

- u_w is the world (KB) effect, shared by every arm and both tiers; F1/F2 worlds of one seed share one registry, so
  they share u across Study A's cells (the analysis clusters them as one, `main_stats.cluster_of`);
- g is the world × arm effect, drawn independently per tier (conservative for the tier contrast);
- v_t is task difficulty, shared by arms and tiers (the pairing); e is the task × arm interaction;
- x_{r,c,a} is solved so the arm's marginal success is the scenario's target (Gauss–Hermite quadrature);
- epochs are Bernoulli draws; S1 runs its pool (8 runs in Studies A/B, 3 elsewhere), and its task mean is the pool's
  first 3 runs, as in `main_stats.build_tables`;
- each S1 run submits an answer: right; or, when wrong, nothing (prob `abstain`), the task's one attractor wrong answer
  (prob `rho`) or a wrong answer of its own. The S8 frontier is then built from these keys by `frontier` itself;
- each run costs arm multiple × task multiplier × run noise (lognormal CVs) × (1 + κ (1 − success)): with κ > 0 a
  failed run costs more than a successful one (`Settings.kappa`; BUILD_REVIEW S-2: with caps at 8 × B0 a failing run
  that wanders to its cap is the plausible case). S1 runs cost 1 on average when they succeed.

σ defaults are the gate's priors (`pilot.py`: σ_w 0.5, σ_g 0.3, σ_u 1.5, σ_v 0.5); `sigmas_from_pilot` takes a
`pilot.variance_components` result instead. Baselines (S1's success per cell) are assumptions to replace with the
micro-pilot's numbers; cost multiples are the budget priors (`config/budget_assumptions.yaml` arms' `multiplier`,
`cost_multiples`).

**Frontier nulls.** An arm at the S8 frontier's boundary has marginal success p* = S8(r(p*)) + offset, with
r(p) = m_arm (1 + κ (1 − p)) / (1 + κ (1 − p_S1)) its expected cost over S1's expected run cost: a fixed point once
cost depends on success. The population curve S8(k) is computed exactly (`population_s8`: the vote's multinomial over
right, attractor, abstaining and own-wrong answers, integrated over task difficulty by Gauss–Hermite), so a null
boundary carries no Monte Carlo error of its own (BUILD_REVIEW S-8: a 20,000-task Monte Carlo curve put M3's +6 pp
boundary 0.0617 against 0.05).

**The real path.** Each replicate builds `main_stats.Tables` (task means, task-mean costs, S1 pools) and runs
`main_stats.evaluate_family`, the function `main_report` runs: the same contrasts, frontier interpolation at the
arm's realised cost with its matched-cost correction, sign-flip tests, Holm and gatekeeping. `Draw.frame()` emits the
same replicate as the loader's tidy frame, and a test checks that `build_tables` on it equals the direct
`Draw.tables()`.

**Scenarios** per family: its null boundary (type I error; must not exceed the nominal α beyond Monte Carlo error)
and plausible effects (power); M2 and M3 also at κ = 1 and 3. Sizes are the runner's (`planned_sizes`): a cell of n
tasks runs ⌈n / 12⌉ whole worlds of 12 tasks (`run_study.TASKS_PER_WORLD`), so 100 tasks are 9 × 12 = 108 (M1s 60,
Study C 24), 3 epochs, the 8-run S1 pool. `min_p_table` gives the smallest p the exact world-level sign flip can
attain at each test's cluster count and whether it reaches the test's smallest Holm level.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit

from ..config import ROOT
from . import frontier as fr
from .main_hypotheses import HYPOTHESES, NI_MARGIN, PRIMARY_TIER, STUDY_A_CELLS, STUDY_B_CELLS, TIER_CELLS, TOST_MARGIN, Hypothesis, confirmatory
from .main_load import tier_of
from .main_stats import EXACT_MAX_CLUSTERS, S1_EPOCHS, Pool, Tables, evaluate_family

RUN_PLAN = ROOT / "config" / "run_plan.yaml"
MODELS = ROOT / "config" / "models.yaml"
BUDGET_ASSUMPTIONS = ROOT / "config" / "budget_assumptions.yaml"
TEST_WORLDS = 9  # main.build.kg: 11 worlds per KG condition = 9 test + 2 pilot (BUILD_PLAN; run_plan.yaml note)
TASKS_PER_WORLD = 12  # run_study.TASKS_PER_WORLD (a test checks they agree)
WORLD_SEED = 4000  # labels only
# S1's success per Luna cell. F3/F7: the gate's power baselines (D-023); F1/F2/F7-100: assumptions until the
# micro-pilot (F1-32 and F2-10 are the hard endpoints). Sol adds SOL_SHIFT on the logit scale.
BASELINES = {"F1-2": 0.80, "F1-32": 0.40, "F2-2": 0.80, "F2-10": 0.50, "F3-5": 0.75, "F3-60": 0.60, "F7-10": 0.85, "F7-100": 0.65, "F7-1000": 0.45}
SOL_SHIFT = 0.5
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(48)
_GH_W = _GH_W / _GH_W.sum()


def cost_multiples(path: Path = BUDGET_ASSUMPTIONS) -> dict[str, float]:
    """Each arm's mean cost per sample relative to one S1 run: the budget priors' `multiplier` (`like:` resolved; 1 for
    arms without one, e.g. S5 per-step KG ≈ S1 on tokens)."""
    arms = (yaml.safe_load(path.read_text()) or {}).get("arms") or {}

    def mult(name: str, seen: frozenset = frozenset()) -> float:
        a = arms.get(name) or {}
        if "multiplier" in a:
            return float(a["multiplier"])
        like = a.get("like")
        return mult(like, seen | {name}) if like and like not in seen else 1.0

    return {name: mult(name) for name in arms}


COST = cost_multiples()


@dataclass(frozen=True)
class Sigmas:
    w: float = 0.5
    g: float = 0.3
    u: float = 1.5
    v: float = 0.5

    @property
    def total(self) -> float:
        return math.sqrt(self.w**2 + self.g**2 + self.u**2 + self.v**2)


def sigmas_from_pilot(vc: dict | None, base: Sigmas = Sigmas()) -> Sigmas:
    """σ_w and σ_g from a `pilot.variance_components` result (the priors where it could not estimate them)."""
    if not vc or not vc.get("estimable"):
        return base
    return replace(base, w=float(vc.get("sigma_w") if vc.get("sigma_w") is not None else base.w), g=float(vc.get("sigma_g") if vc.get("sigma_g") is not None else base.g))


@dataclass(frozen=True)
class Settings:
    worlds: int = TEST_WORLDS
    rho: float = 0.3  # P(a wrong S1 answer is the task's attractor)
    abstain: float = 0.1  # P(a wrong S1 run submits nothing)
    cost_cv_task: float = 0.3
    cost_cv_run: float = 0.3
    sol_shift: float = SOL_SHIFT
    flip_reps: int = 2000  # Monte Carlo sign flips above EXACT_MAX_CLUSTERS clusters
    exact_max: int = EXACT_MAX_CLUSTERS
    kappa: float = 0.0  # a failed run costs (1 + kappa) x a successful one (every arm)


@dataclass(frozen=True)
class S8Target:
    """An arm at the population S8 frontier (plus `offset`), at its own expected cost: `ratio` is its cost multiple
    (its expected cost per sample over S1's when both succeed; with κ the ratio moves with success, a fixed point)."""

    ratio: float
    offset: float = 0.0


# --------------------------------------------------------------------------- planned sizes


def planned_sizes(path: Path = RUN_PLAN, models: Path = MODELS) -> dict[tuple[str, str, str], dict]:
    """(tier, cell, arm) -> {n_tasks, worlds, epochs, plan_cell} for the main study's test phases (Studies A, B and F),
    as the runner runs them: ⌈n / 12⌉ whole worlds of 12 tasks (100 -> 9 × 12 = 108; 50 -> 60)."""
    plan = yaml.safe_load(path.read_text())
    profiles = (yaml.safe_load(models.read_text()) or {}).get("profiles", {})
    main = plan["studies"]["main"]
    default = main.get("profile", "main_luna")
    out: dict = {}
    for phase in ("study_a", "study_b", "study_f"):
        for cell in main["phases"].get(phase) or []:
            prof = cell.get("profile", default)
            tier = tier_of(((profiles.get(prof) or {}).get("agent") or {}).get("model"))
            worlds = math.ceil(int(cell["n_tasks"]) / TASKS_PER_WORLD)
            for arm in cell["arms"]:
                for c in cell["cells"]:
                    out[(tier, c, arm)] = {"n_tasks": worlds * TASKS_PER_WORLD, "worlds": worlds, "epochs": int(cell["epochs"]), "plan_cell": cell["id"]}
    return out


# --------------------------------------------------------------------------- the generative model


def _marginal(x: float, s: float) -> float:
    return float(_GH_W @ expit(x + s * _GH_X))


def location(target: float, s: float) -> float:
    """x with E[expit(x + s Z)] = target (bisection)."""
    if not 0.0 < target < 1.0:
        raise ValueError(f"target success {target} must lie strictly between 0 and 1")
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _marginal(mid, s) < target else (lo, mid)
    return (lo + hi) / 2


def _answer_codes(rng: np.random.Generator, ok: np.ndarray, rho: float, abstain: float) -> np.ndarray:
    """Per run: 0 right; when wrong: ABSTAIN, 1 (the task's attractor) or a code of its own (2 + run index)."""
    T, K = ok.shape
    r = rng.random((T, K))
    codes = np.where(r < abstain, fr.ABSTAIN, np.where(r < abstain + (1 - abstain) * rho, 1, 2 + np.arange(K)[None, :]))
    return np.where(ok, 0, codes)


def vote_success(k: int, p: np.ndarray, rho: float, abstain: float) -> np.ndarray:
    """E[success of a k-run plurality vote] for tasks whose runs succeed with probability `p` (array), under the
    answer model (`_answer_codes`), exactly: the sum over the multinomial of right (c), attractor (a), abstaining (b)
    and own-wrong (u) answers of the vote's expected success (ties: the share of the tied answers that is right)."""
    p = np.asarray(p, dtype=float)
    q = 1 - p
    probs = (p, q * (1 - abstain) * rho, q * abstain, q * (1 - abstain) * (1 - rho))  # right, attractor, abstain, own
    out = np.zeros_like(p)
    for c in range(k + 1):
        for a in range(k + 1 - c):
            for b in range(k + 1 - c - a):
                u = k - c - a - b
                top = max(c, a, 1 if u else 0)
                if top == 0 or c != top:
                    continue
                tied = 1 + (a == top) + (u if top == 1 else 0)
                coef = math.factorial(k) / (math.factorial(c) * math.factorial(a) * math.factorial(b) * math.factorial(u))
                out += coef * probs[0] ** c * probs[1] ** a * probs[2] ** b * probs[3] ** u / tied
    return out


@cache
def population_s8(x_s1: float, s: float, rho: float, abstain: float, K: int = 8) -> tuple[float, ...]:
    """The population S8(k) success, k = 1..K, under the model, exactly (`vote_success` integrated over the tasks'
    logit-normal success, 96-point Gauss–Hermite)."""
    x, w = np.polynomial.hermite_e.hermegauss(96)
    w = w / w.sum()
    p = expit(x_s1 + s * x)
    return tuple(float(w @ vote_success(k, p, rho, abstain)) for k in range(1, K + 1))


def s8_at(curve: tuple[float, ...], ratio: float) -> float:
    """The population frontier at `ratio` S1-run costs (S8(k) costs k), linear between adjacent k."""
    K = len(curve)
    if ratio <= 1:
        return curve[0]
    if ratio >= K:
        return curve[-1]
    k = int(math.floor(ratio))
    lam = ratio - k
    return (1 - lam) * curve[k - 1] + lam * curve[k]


def frontier_target(curve: tuple[float, ...], target: S8Target, p_s1: float, kappa: float) -> float:
    """The arm's marginal success at the frontier (module docstring): p = S8(r(p)) + offset, by fixed-point iteration."""
    p = s8_at(curve, target.ratio) + target.offset
    for _ in range(200):
        r = target.ratio * (1 + kappa * (1 - p)) / (1 + kappa * (1 - p_s1))
        nxt = s8_at(curve, r) + target.offset
        if abs(nxt - p) < 1e-12:
            break
        p = nxt
    return p


def _world_id(cell: str, seed: int) -> str:
    return f"{cell}-test-s{seed}"


@dataclass
class Draw:
    """One simulated main study (the cells and arms one family needs)."""

    index: dict = field(default_factory=dict)  # (tier, cell) -> MultiIndex (cell, world, task)
    runs: dict = field(default_factory=dict)  # (tier, cell, arm) -> (T, E) success
    cost: dict = field(default_factory=dict)  # (tier, cell, arm) -> (T, E) cost (tokens)
    codes: dict = field(default_factory=dict)  # (tier, cell) -> (T, K) S1 answer codes

    def tables(self, cluster_by: str = "kb", s1_epochs: int = S1_EPOCHS) -> Tables:
        """Tables directly from the arrays (what `build_tables` would compute from `frame()`)."""
        tm: dict = {}
        cost: dict = {}
        pools: dict = {}
        for tier in sorted({t for t, _ in self.index}):
            parts, cparts = [], []
            for (t, cell), idx in self.index.items():
                if t != tier:
                    continue
                arms = sorted(a for (tt, c, a) in self.runs if tt == tier and c == cell)
                cols = {a: (self.runs[(tier, cell, a)][:, :s1_epochs] if a == "S1" else self.runs[(tier, cell, a)]).mean(axis=1) for a in arms}
                ccols = {a: (self.cost[(tier, cell, a)][:, :s1_epochs] if a == "S1" else self.cost[(tier, cell, a)]).mean(axis=1) for a in arms}
                parts.append(pd.DataFrame(cols, index=idx))
                cparts.append(pd.DataFrame(ccols, index=idx))
                if (tier, cell) in self.codes:
                    K = self.runs[(tier, cell, "S1")].shape[1]
                    pools[(tier, cell, None)] = Pool(idx, self.codes[(tier, cell)], self.runs[(tier, cell, "S1")], {"tokens": self.cost[(tier, cell, "S1")]}, K)
                    for k in range(1, K + 1):
                        pools[(tier, cell, k)] = Pool(idx, self.codes[(tier, cell)][:, :k], self.runs[(tier, cell, "S1")][:, :k], {"tokens": self.cost[(tier, cell, "S1")][:, :k]}, k)
            df = pd.concat(parts).sort_index()
            df.columns.name = "arm"
            tm[tier] = df
            cdf = pd.concat(cparts).sort_index()
            cdf.columns.name = "arm"
            cost[tier] = {"tokens": cdf}
        return Tables(tm, cost, {}, {"tokens": "tokens"}, cluster_by, "sum", pools)

    def frame(self) -> pd.DataFrame:
        """The same replicate as the loader's tidy frame: one row per (sample, epoch)."""
        rows = []
        for (tier, cell, arm), y in self.runs.items():
            idx = self.index[(tier, cell)]
            T, E = y.shape
            codes = self.codes.get((tier, cell)) if arm == "S1" else None
            for e in range(E):
                keys = [None if codes is None or codes[i, e] == fr.ABSTAIN else f"{idx[i][2]}:{codes[i, e]}" for i in range(T)] if codes is not None else [None] * T
                rows.append(pd.DataFrame({
                    "plan_cell": f"sim.{tier}", "arm": arm, "cell": cell, "world": idx.get_level_values("world"), "task": idx.get_level_values("task"),
                    "epoch": e + 1, "tier": tier, "success": y[:, e], "answer_key": keys, "tokens": self.cost[(tier, cell, arm)][:, e],
                    "error": False, "cap_hit": False, "delivery": "push",
                }))  # fmt: skip
        return pd.concat(rows, ignore_index=True)


def simulate(spec: dict, n_tasks: dict, epochs: dict, sigmas: Sigmas, settings: Settings, rng: np.random.Generator) -> Draw:
    """One replicate. `spec` {(tier, cell): {arm: target}} (target: marginal success or S8Target); `n_tasks`
    {(tier, cell): n}; `epochs` {(tier, cell, arm): E} (S1: its pool size). Tiers share worlds, tasks, u and v."""
    s = sigmas.total
    draw = Draw()
    seeds = WORLD_SEED + np.arange(settings.worlds)
    u_reg = rng.normal(0, sigmas.w, settings.worlds)  # registry KB per seed (F1/F2)
    shared: dict = {}  # cell -> (u per task, v per task, task multiplier per task)
    for (tier, cell), arms in sorted(spec.items()):
        n = n_tasks[(tier, cell)]
        per_world = np.full(settings.worlds, n // settings.worlds) + (np.arange(settings.worlds) < n % settings.worlds)
        wi = np.repeat(np.arange(settings.worlds), per_world)
        if cell not in shared:
            u = u_reg[wi] if cell.split("-")[0] in ("F1", "F2") else rng.normal(0, sigmas.w, settings.worlds)[wi]
            shared[cell] = (wi, u, rng.normal(0, sigmas.u, n), rng.lognormal(-0.5 * math.log1p(settings.cost_cv_task**2), math.sqrt(math.log1p(settings.cost_cv_task**2)), n))
        wi, u, v, tau = shared[cell]
        worlds = [_world_id(cell, int(seeds[w])) for w in wi]
        draw.index[(tier, cell)] = pd.MultiIndex.from_arrays([[cell] * n, worlds, [f"{w}-t{j:03d}" for j, w in enumerate(worlds)]], names=["cell", "world", "task"])
        s1_target = arms.get("S1")
        x_s1 = location(s1_target, s) if s1_target is not None else None
        for arm, target in arms.items():
            if isinstance(target, S8Target):
                curve = population_s8(round(x_s1, 9), round(s, 9), settings.rho, settings.abstain)
                target = frontier_target(curve, target, s1_target, settings.kappa)
            x = location(float(target), s)
            g = rng.normal(0, sigmas.g, settings.worlds)[wi]
            p = expit(x + u + g + v + rng.normal(0, sigmas.v, n))
            E = epochs[(tier, cell, arm)]
            ok = rng.random((n, E)) < p[:, None]
            draw.runs[(tier, cell, arm)] = ok.astype(float)
            sd = math.sqrt(math.log1p(settings.cost_cv_run**2))
            fail = 1.0 + settings.kappa * (1.0 - ok)  # a failed run costs (1 + kappa) x (BUILD_REVIEW S-2)
            draw.cost[(tier, cell, arm)] = COST.get(arm, 1.0) * tau[:, None] * rng.lognormal(-0.5 * sd**2, sd, (n, E)) * fail
            if arm == "S1":
                draw.codes[(tier, cell)] = _answer_codes(rng, ok, settings.rho, settings.abstain)
    return draw


# --------------------------------------------------------------------------- scenarios per family


def _base(tier: str, cell: str, baselines: dict, settings: Settings) -> float:
    b = baselines[cell]
    return b if tier == PRIMARY_TIER else float(expit(logit(b) + settings.sol_shift))


@dataclass(frozen=True)
class Scenario:
    """One simulated truth for a family: `spec` {(tier, cell): {arm: target}}; `nulls`, the members whose H0 holds
    in it (their rejections are false: the family-wise error rate is P(any of them rejected))."""

    name: str
    spec: dict
    nulls: tuple[str, ...] = ()
    kappa: float | None = None  # overrides Settings.kappa (cost depending on success)

    @property
    def kind(self) -> str:
        return "null" if self.nulls else "power"


def scenarios(hyp_id: str, baselines: dict = BASELINES, settings: Settings = Settings()) -> list[Scenario]:
    """Each confirmatory family's null boundary (type I) and plausible effects (power)."""
    b = lambda c, t=PRIMARY_TIER: _base(t, c, baselines, settings)  # noqa: E731
    L = PRIMARY_TIER
    out = []
    if hyp_id == "M1":
        for d in (0.0, 0.05, 0.10, 0.15):
            out.append(Scenario(f"M1 − S9 = {d:+.2f}", {(L, "F1-32"): {"S9": b("F1-32") + 0.05, "M1": b("F1-32") + 0.05 + d}}, ("M1.F1-32",) if d == 0 else ()))
    elif hyp_id == "M2":
        out.append(Scenario("global null: M1 = S1 (below the frontier)", {(L, "F1-32"): {"S1": b("F1-32"), "M1": b("F1-32")}}, ("M2.gate", "M2.frontier")))
        for d in (0.0, 0.05, 0.10, 0.15):
            out.append(Scenario(f"M1 − S8@M1 = {d:+.2f} (M1 > S1)", {(L, "F1-32"): {"S1": b("F1-32"), "M1": S8Target(COST["M1"], d)}}, ("M2.frontier",) if d == 0 else ()))
        for kappa in (1.0, 3.0):  # cost depending on success (BUILD_REVIEW S-2)
            out.append(Scenario(f"M1 − S8@M1 = +0.00, κ {kappa:g}", {(L, "F1-32"): {"S1": b("F1-32"), "M1": S8Target(COST["M1"], 0.0)}}, ("M2.frontier",), kappa))
            out.append(Scenario(f"M1 − S8@M1 = +0.10, κ {kappa:g}", {(L, "F1-32"): {"S1": b("F1-32"), "M1": S8Target(COST["M1"], 0.10)}}, (), kappa))
    elif hyp_id in ("M3", "M5"):
        m = TOST_MARGIN
        kappas = (None, 1.0, 3.0) if hyp_id == "M3" else (None,)
        for kappa in kappas:
            for d in (m, -m, 0.0, 0.02, 0.03) if kappa is None else (m, -m, 0.0):
                if hyp_id == "M3":
                    spec = {(L, c): {"S1": b(c), "M7": S8Target(COST["M7"], d)} for c in STUDY_A_CELLS}
                    name = f"M7 − S8@M7 = {d:+.2f}" + ("" if kappa is None else f", κ {kappa:g}")
                else:
                    spec = {(L, c): {"M1k": b(c) + 0.03, "M2": b(c) + 0.03 + d} for c in STUDY_B_CELLS}
                    name = f"M2 − M1k = {d:+.2f}"
                out.append(Scenario(name, spec, (f"{hyp_id}.pooled",) if abs(d) >= m else (), kappa))
    elif hyp_id == "K1":
        cells = ("F7-1000", "F3-60")
        for d in (0.0, 0.05, 0.10, 0.15):
            out.append(Scenario(f"S5 − S3s = {d:+.2f} in both cells", {(L, c): {"S3s": b(c) + 0.05, "S5": b(c) + 0.05 + d} for c in cells}, ("K1.F7-1000", "K1.F3-60") if d == 0 else ()))
        out.append(Scenario("S5 − S3s = +0.10 on F7-1000 only", {(L, c): {"S3s": b(c) + 0.05, "S5": b(c) + 0.05 + (0.10 if c == "F7-1000" else 0.0)} for c in cells}, ("K1.F3-60",)))
    elif hyp_id == "K2":
        for d in (0.0, -0.05, -0.08, -0.10, -0.15):
            spec = {(L, c): {"S1": b(c), "M1": b(c) + 0.03, "S5": b(c) + 0.05, "M1k": b(c) + 0.08 + d} for c in STUDY_B_CELLS}
            out.append(Scenario(f"interaction {d:+.2f}", spec, ("K2.interaction",) if d == 0 else ()))
    elif hyp_id == "K2-NI":
        for d in (-NI_MARGIN, -0.02, 0.0, 0.03):
            spec = {(L, c): {"S5": b(c) + 0.05, "M2": b(c) + 0.05 - d} for c in STUDY_B_CELLS}
            out.append(Scenario(f"S5 − M2 = {d:+.2f}", spec, ("K2-NI.pooled",) if d <= -NI_MARGIN else ()))
    elif hyp_id == "T1":  # descriptive since D-033: kept for the two-tier generator check (tests), not in the power table
        def spec6(d: float) -> dict:
            return {(t, c): {"S1": b(c, t), "M1": S8Target(COST["M1"], 0.05 - (d if t != L else 0.0))} for t in (L, "sol") for c in TIER_CELLS}

        for d in (0.0, 0.10):
            out.append(Scenario(f"payoff falls by {d:.2f} Luna → Sol in both cells", spec6(d)))
    else:
        raise KeyError(f"no scenarios for {hyp_id}")
    return out


def _sizes_for(spec: dict, sizes: dict, settings: Settings) -> tuple[dict, dict]:
    n_tasks, epochs = {}, {}
    for (tier, cell), arms in spec.items():
        ns = [sizes[(tier, cell, a)]["n_tasks"] for a in arms if (tier, cell, a) in sizes]
        n_tasks[(tier, cell)] = max(ns) if ns else 100
        for a in arms:
            epochs[(tier, cell, a)] = sizes.get((tier, cell, a), {}).get("epochs", 3)
    return n_tasks, epochs


def scale_sizes(sizes: dict, factor: float) -> dict:
    """The planned sizes with every cell's task count multiplied by `factor` (rounded)."""
    return {k: v | {"n_tasks": int(round(v["n_tasks"] * factor))} for k, v in sizes.items()}


def simulate_family(hyp: Hypothesis, scenario: Scenario, reps: int, seed: int = 0, sigmas: Sigmas = Sigmas(), settings: Settings = Settings(), sizes: dict | None = None) -> dict:
    """Over `reps` replicates of `scenario`: each member's claim rate ("supported": rejected after Holm or gatekeeping,
    with its cost condition), the rate of any and all claims, and the family-wise false-claim rate over the scenario's
    true nulls."""
    sizes = planned_sizes() if sizes is None else sizes
    if scenario.kappa is not None:
        settings = replace(settings, kappa=scenario.kappa)
    n_tasks, epochs = _sizes_for(scenario.spec, sizes, settings)
    rng = np.random.default_rng(seed)
    hits = {m.id: 0 for m in hyp.members}
    any_hit = all_hit = false_hit = 0
    for r in range(reps):
        draw = simulate(scenario.spec, n_tasks, epochs, sigmas, settings, rng)
        res = evaluate_family(draw.tables(), hyp, reps=settings.flip_reps, seed=seed + r, ci=False, exact_max=settings.exact_max)
        rej = {i: m.get("label") == "supported" for i, m in res["members"].items()}
        for i, v in rej.items():
            hits[i] += v
        any_hit += any(rej.values())
        all_hit += all(rej.values())
        false_hit += any(rej[i] for i in scenario.nulls)
    return {"reps": reps, "members": {i: h / reps for i, h in hits.items()}, "any": any_hit / reps, "all": all_hit / reps, "false_claims": false_hit / reps if scenario.nulls else None}


def mc_se(p: float, reps: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / reps)


def run_scenarios(hyp: Hypothesis, scens: list[Scenario], reps: int, seed: int, sigmas: Sigmas, settings: Settings, sizes: dict) -> list[dict]:
    rows = []
    for j, sc in enumerate(scens):
        res = simulate_family(hyp, sc, reps, seed + j, sigmas, settings, sizes)
        row = {"scenario": sc.name, "kind": sc.kind, "nulls": list(sc.nulls)} | res
        if sc.nulls:
            row["exceeds_alpha"] = res["false_claims"] > hyp.alpha + 2 * mc_se(hyp.alpha, reps)
        row["underpowered"] = {m: p < 0.8 for m, p in res["members"].items() if m not in sc.nulls}
        rows.append(row)
    return rows


def power_table(reps: int = 1000, seed: int = 20261003, sigmas: Sigmas = Sigmas(), settings: Settings = Settings(), baselines: dict = BASELINES, families: tuple[str, ...] | None = None, hypotheses=HYPOTHESES, sizes: dict | None = None) -> dict:
    """Every confirmatory family's scenarios through `simulate_family` at the planned sizes. Flags a family-wise
    false-claim rate above α + 2 Monte Carlo SEs (`exceeds_alpha`) and each non-null member's power below 0.8."""
    sizes = planned_sizes() if sizes is None else sizes
    out = {"reps": reps, "seed": seed, "sigmas": asdict(sigmas), "settings": asdict(settings), "baselines": baselines, "families": {}}
    for i, hyp in enumerate(confirmatory(hypotheses)):
        if families and hyp.id not in families:
            continue
        rows = run_scenarios(hyp, scenarios(hyp.id, baselines, settings), reps, seed + 1000 * i, sigmas, settings, sizes)
        out["families"][hyp.id] = {"alpha": hyp.alpha, "procedure": hyp.procedure, "scenarios": rows}
    return out


def _members(hyp: Hypothesis, **changes) -> Hypothesis:
    return replace(hyp, members=tuple(replace(m, **changes) for m in hyp.members))


def alternatives(reps: int = 500, seed: int = 20261004, sigmas: Sigmas = Sigmas(), settings: Settings = Settings(), baselines: dict = BASELINES, only: tuple[str, ...] | None = None) -> list[dict]:
    """Power of the superiority members (MDE ~15 pp at the planned sizes; D-033 keeps them) at +10 pp under design
    changes: more test worlds for the same tasks, more tasks, no Holm partner. The equivalence, NI and T1 alternatives
    that D-031 reported were settled by D-033 (TOST ±6 pp, K2-NI pooled at 5 pp, T1 descriptive).
    `only`: run the cases whose name contains one of these strings."""
    from .main_hypotheses import get

    sizes = planned_sizes()
    sc = {h: {s.name: s for s in scenarios(h, baselines, settings)} for h in ("M1", "M2", "K1")}
    k1_one = replace(get("K1"), members=(get("K1").members[0],))
    m1, m2, k1 = sc["M1"]["M1 − S9 = +0.10"], sc["M2"]["M1 − S8@M1 = +0.10 (M1 > S1)"], sc["K1"]["S5 − S3s = +0.10 in both cells"]
    cases = [
        ("M1 at +0.10, 20 worlds (same 100 tasks)", get("M1"), m1, replace(settings, worlds=20), sizes),
        ("M1 at +0.10, 200 tasks over 9 worlds", get("M1"), m1, settings, scale_sizes(sizes, 2)),
        ("M1 at +0.10, 200 tasks over 20 worlds", get("M1"), m1, replace(settings, worlds=20), scale_sizes(sizes, 2)),
        ("M2 frontier at +0.10, 20 worlds", get("M2"), m2, replace(settings, worlds=20), sizes),
        ("M2 frontier at +0.10, 200 tasks", get("M2"), m2, settings, scale_sizes(sizes, 2)),
        ("K1 F7-1000 alone (no Holm partner) at +0.10", k1_one, k1, settings, sizes),
        ("K1 at +0.10, 20 worlds", get("K1"), k1, replace(settings, worlds=20), sizes),
    ]
    out = []
    for i, (name, hyp, scen, st, sz) in enumerate(cases):
        if only and not any(o in name for o in only):
            continue
        res = simulate_family(hyp, scen, reps, seed + 100 * i, sigmas, st, sz)
        out.append({"case": name, "family": hyp.id, "scenario": scen.name} | res)
    return out


def m3_pool_check(cells: tuple[str, ...] = STUDY_A_CELLS, kappa: float = 0.0, seed: int = 20261040, boundary_reps: int = 4000, power_reps: int = 1000, baselines: dict = BASELINES) -> list[dict]:
    """M3's TOST over `cells` at κ: type I at ±TOST_MARGIN and power at Δ = 0, ±2 pp, scenario i seeded `seed + i`
    (D-048's numbers: the four Study A cells at seeds 20261040 / 20261050 / 20261060 for κ = 0 / 1 / 3; F1-32 and F2-10
    only, the rejected alternative, at 20261010 / 20261020 / 20261030)."""
    from .main_hypotheses import get

    m3 = get("M3")
    hyp = replace(m3, members=(replace(m3.members[0], cells=tuple(cells)),))
    settings = Settings(kappa=kappa)
    sizes = planned_sizes()
    out = []
    for i, (d, reps) in enumerate(((TOST_MARGIN, boundary_reps), (-TOST_MARGIN, boundary_reps), (0.0, power_reps), (0.02, power_reps), (-0.02, power_reps))):
        spec = {(PRIMARY_TIER, c): {"S1": _base(PRIMARY_TIER, c, baselines, settings), "M7": S8Target(COST["M7"], d)} for c in cells}
        sc = Scenario(f"M7 − S8@M7 = {d:+.2f}", spec, ("M3.pooled",) if abs(d) >= TOST_MARGIN else (), kappa)
        res = simulate_family(hyp, sc, reps, seed + i, settings=settings, sizes=sizes)
        out.append({"delta": d, "kind": sc.kind, "reps": reps, "seed": seed + i, "rate": res["members"]["M3.pooled"]})
    return out


def min_p_table(settings: Settings = Settings(), hypotheses=HYPOTHESES, report_flips: int = 10_000) -> list[dict]:
    """Per confirmatory member at the planned sizes: its cluster count, the smallest p its sign flip can attain (exact:
    2^-G; Monte Carlo: 1 / (flips + 1) at the report's `report_flips`), the smallest level Holm may test it at (α over the
    members of its family or gate stage; TOST: α per one-sided test), and whether that level is reachable."""
    out = []
    for hyp in confirmatory(hypotheses):
        for m in hyp.members:
            reg = {c for c in m.cells if c.split("-")[0] in ("F1", "F2")}
            G = (settings.worlds if reg else 0) + settings.worlds * len(set(m.cells) - reg)
            exact = G <= settings.exact_max
            p_min = 2.0**-G if exact else 1.0 / (report_flips + 1)
            stage = [x for x in hyp.members if x.stage == m.stage] if hyp.procedure == "serial" else list(hyp.members)
            level = hyp.alpha / len(stage)
            out.append({"family": hyp.id, "member": m.id, "cells": list(m.cells), "clusters": G, "exact": exact, "min_p": p_min, "smallest_holm_level": level, "reachable": p_min <= level})
    return out


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def render(table: dict, minp: list[dict], alts: list[dict] | None = None) -> str:
    L = [f"# Main-study power and type I error ({table['reps']} replicates per scenario)", "", f"σ: {table['sigmas']}; settings: {table['settings']}", ""]
    L += ["Claim rates per member (rejected after Holm / gatekeeping, with any cost condition). `False claims`: P(any true-null member claimed).", ""]
    L += ["| Family | Scenario | True nulls | Member claim rates | False claims | Flag |", "|---|---|---|---|---|---|"]
    for fam, f in table["families"].items():
        for r in f["scenarios"]:
            flags = (["FALSE CLAIMS > α"] if r.get("exceeds_alpha") else []) + [f"{m} < 0.8" for m, u in r["underpowered"].items() if u]
            L.append(f"| {fam} (α {f['alpha']}) | {r['scenario']} | {', '.join(r['nulls']) or '–'} | " + "; ".join(f"{m} {_fmt(p)}" for m, p in r["members"].items()) + f" | {_fmt(r['false_claims'])} | {', '.join(flags)} |")
    L += ["", "| Member | Clusters | Exact | Min p | Smallest Holm level | Reachable |", "|---|---|---|---|---|---|"]
    L += [f"| {m['member']} | {m['clusters']} | {m['exact']} | {m['min_p']:.5f} | {m['smallest_holm_level']:.4f} | {m['reachable']} |" for m in minp]
    if alts:
        L += ["", "## Design alternatives", "", "| Case | Scenario | Member claim rates |", "|---|---|---|"]
        L += [f"| {a['case']} | {a['scenario']} | " + "; ".join(f"{m} {_fmt(p)}" for m, p in a["members"].items()) + " |" for a in alts]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--sigma-w", type=float, default=None)
    ap.add_argument("--sigma-g", type=float, default=None)
    ap.add_argument("--pilot", type=Path, default=None, help="a pilot.variance_components result (JSON) for σ_w, σ_g")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--flip-reps", type=int, default=Settings().flip_reps)
    ap.add_argument("--alternatives", type=int, default=0, metavar="REPS", help="also simulate the design alternatives at this many replicates")
    ap.add_argument("--out", type=Path, default=None, help="write the JSON here (the markdown always goes to stdout)")
    args = ap.parse_args(argv)
    sig = sigmas_from_pilot(json.loads(args.pilot.read_text())) if args.pilot else Sigmas()
    if args.sigma_w is not None:
        sig = replace(sig, w=args.sigma_w)
    if args.sigma_g is not None:
        sig = replace(sig, g=args.sigma_g)
    settings = Settings(flip_reps=args.flip_reps)
    table = power_table(args.reps, args.seed, sig, settings, families=tuple(args.families) if args.families else None)
    minp = min_p_table(settings=settings)
    if args.out:  # written before the (long) alternatives, so a failure there loses nothing
        args.out.write_text(json.dumps({"power": table, "min_p": minp, "alternatives": None}, indent=1, default=float))
    alts = alternatives(args.alternatives, sigmas=sig, settings=settings) if args.alternatives else None
    if args.out and alts is not None:
        args.out.write_text(json.dumps({"power": table, "min_p": minp, "alternatives": alts}, indent=1, default=float))
    sys.stdout.write(render(table, minp, alts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
