"""Gate statistics (module 8): bootstrap coverage on synthetic data with known Δ, and decision boundaries."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ape.analysis.gate_stats import GATE_CELLS, cluster_bootstrap, decide, invariants, pooled_delta, sign_flip_p

PC_OK = {f"PC{i}": True for i in range(1, 7)}
_spec = importlib.util.spec_from_file_location("power_sim", Path(__file__).parents[1] / "power" / "power_sim.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def synth(true_delta: float, worlds: int = 6, tasks: int = 12, epochs: int = 3, sigma_w: float = 0.6, sigma_g: float = 0.3, seed: int = 0, base: float = 0.6, extra_arms: dict | None = None) -> pd.DataFrame:
    """Task-level epoch means for APG-s, LGR* (and extra arms) with world and world×arm effects."""
    rng = np.random.default_rng(seed)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    arms = {"LGR*": 0.0, "APG-s": None, **(extra_arms or {})}
    rows = []
    for c in GATE_CELLS:
        for w in range(worlds):
            u = rng.normal(0, sigma_w)
            g = {a: rng.normal(0, sigma_g) for a in arms}
            for t in range(tasks):
                v = rng.normal(0, 1.0)
                p_lgr = sig(logit(base) + u + v + g["LGR*"])
                row = {"cell": c, "world": f"{c}-w{w}", "task": f"{c}-w{w}-t{t}", "LGR*": rng.binomial(epochs, p_lgr) / epochs}
                p_apg = np.clip(p_lgr + true_delta, 0.001, 0.999)
                row["APG-s"] = rng.binomial(epochs, sig(logit(p_apg) + g["APG-s"] - g["LGR*"])) / epochs
                for a, shift in (extra_arms or {}).items():
                    row[a] = rng.binomial(epochs, np.clip(p_lgr + shift, 0.001, 0.999)) / epochs
                rows.append(row)
    return pd.DataFrame(rows).set_index(["cell", "world", "task"])


def test_bootstrap_ci_covers_true_delta_about_95_percent():
    covered, trials = 0, 120
    for s in range(trials):
        tm = synth(-0.02, seed=s, sigma_g=0.0)
        boot = cluster_bootstrap(tm, "APG-s", "LGR*", reps=1000, seed=s)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        # Truth is the population Δ; the synthetic effect is approximately -0.02 on the probability scale.
        covered += lo <= -0.02 <= hi
    assert 0.88 <= covered / trials <= 0.995


def test_sign_flip_p_is_small_for_clear_effect_and_large_for_none():
    rng = np.random.default_rng(1)
    assert sign_flip_p(rng.normal(0.3, 1.0, 400)) < 0.01
    assert sign_flip_p(rng.normal(0.0, 1.0, 400)) > 0.05


def test_decision_go_for_equal_systems_with_many_worlds():
    tm = synth(0.0, worlds=40, tasks=20, seed=3, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", PC_OK, cost_ratio=1.0, reps=2000)
    assert d.verdict == "GO", d.reasons
    assert d.ci[0] > -0.05


def test_decision_no_go_when_apg_clearly_worse():
    tm = synth(-0.15, worlds=40, tasks=20, seed=4, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", PC_OK, reps=2000)
    assert d.verdict == "NO_GO" and d.ci[1] < 0


def _decide_with_t(monkeypatch, est, se, df=60.0, cell_f7_1000=0.0):
    """Drive decide() with a fixed world-clustered t statistic to test its branches exactly."""
    import ape.analysis.gate_stats as gs

    tm = synth(0.0, worlds=6, tasks=4, seed=11, extra_arms={"S7": -0.4})
    monkeypatch.setattr(gs, "cluster_t", lambda *a, **k: {"est": est, "se": se, "df": df, "cells": {}})
    monkeypatch.setattr(gs, "cell_deltas", lambda *a, **k: pd.Series({"F7-10": 0.0, "F7-1000": cell_f7_1000, "F3-5": 0.0, "F3-60": 0.0}))
    return gs.decide(tm, "LGR*", PC_OK, reps=500)


def test_decision_boundaries(monkeypatch):
    # lo = est − t(0.975, 60) se = est − 2.0 se.
    assert _decide_with_t(monkeypatch, 0.0, 0.01).verdict == "GO"  # lo −2.0 pp
    assert _decide_with_t(monkeypatch, 0.05, 0.01).superiority
    assert _decide_with_t(monkeypatch, -0.03, 0.02).verdict == "INCONCLUSIVE"  # [−7.0, +1.0] pp spans −5 and 0
    assert _decide_with_t(monkeypatch, -0.08, 0.02).verdict == "NO_GO"  # hi −4.0 pp < 0
    d = _decide_with_t(monkeypatch, 0.0, 0.01, cell_f7_1000=-0.11)
    assert d.verdict == "NO_GO" and "F7-1000" in d.reasons[0]
    # Just inside and outside the margin at the decision level.
    from scipy.stats import t as student_t

    q = float(student_t.ppf(0.975, 60.0))
    assert _decide_with_t(monkeypatch, -0.05 + q * 0.01 + 1e-6, 0.01).verdict == "GO"
    # Just below the margin with a narrow interval (upper bound −1 pp < 0): NO_GO, not INCONCLUSIVE (§8).
    assert _decide_with_t(monkeypatch, -0.05 + q * 0.01 - 1e-6, 0.01).verdict == "NO_GO"
    assert _decide_with_t(monkeypatch, -0.05 + q * 0.03 - 1e-6, 0.03).verdict == "INCONCLUSIVE"  # wide: spans 0


def test_too_few_worlds_is_inconclusive():
    d = decide(synth(0.0, worlds=2, tasks=6, seed=5, extra_arms={"S7": -0.4}), "LGR*", PC_OK, reps=500)
    assert d.verdict == "INCONCLUSIVE" and "too few paired" in d.reasons[0]


def test_min_worlds_counts_paired_apg_lgr_worlds_not_other_arms():
    """RELIABILITY_REVIEW S1: S7 had 6 worlds per cell but APG*/LGR* only 2 paired worlds in F3-60; the old count
    (every arm's worlds) returned GO where §8 requires INCONCLUSIVE."""
    tm = synth(0.0, worlds=6, tasks=12, seed=21, extra_arms={"S7": -0.4})
    unpaired = (tm.index.get_level_values("cell") == "F3-60") & ~tm.index.get_level_values("world").isin(["F3-60-w0", "F3-60-w1"])
    tm.loc[unpaired, "LGR*"] = np.nan
    d = decide(tm, "LGR*", PC_OK, reps=500)
    assert d.verdict == "INCONCLUSIVE" and "'F3-60': 2" in d.reasons[0]
    assert d.details["paired_worlds"]["F3-60"] == 2 and d.details["dropped_tasks"] == {"F3-60": 48}


def test_a_cell_with_both_arms_but_no_shared_task_is_inconclusive_not_a_crash():
    tm = synth(0.0, worlds=6, tasks=12, seed=22, extra_arms={"S7": -0.4})
    f35 = tm.index.get_level_values("cell") == "F3-5"
    even = tm.index.get_level_values("task").str[-1].isin(list("02468"))
    tm.loc[f35 & even, "APG-s"] = np.nan
    tm.loc[f35 & ~even, "LGR*"] = np.nan
    d = decide(tm, "LGR*", PC_OK, reps=500)
    assert d.verdict == "INCONCLUSIVE" and "'F3-5': 0" in d.reasons[0]


def test_cluster_t_matches_a_hand_computation_and_weights_unequal_worlds_by_tasks():
    from ape.analysis.gate_stats import cluster_t

    idx = pd.MultiIndex.from_tuples(
        [("F7-10", "w0", "t0"), ("F7-10", "w0", "t1"), ("F7-10", "w1", "t2"), ("F7-10", "w2", "t3"), ("F7-10", "w2", "t4"), ("F7-10", "w2", "t5")],
        names=["cell", "world", "task"],
    )
    tm = pd.DataFrame({"A": [1.0, 1.0, 0.0, 1.0, 0.0, 1.0], "B": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    st = cluster_t(tm, "A", "B", cells=("F7-10",))
    y, n = np.array([1.0, 0.0, 2.0]), np.array([2.0, 1.0, 3.0])
    r = y.sum() / n.sum()  # the task mean (cell estimand), not the mean of world means
    var = np.sum(((y - r * n) / n.mean()) ** 2) / (3 * 2)
    assert st["est"] == pytest.approx(r) and st["se"] == pytest.approx(np.sqrt(var)) and st["df"] == pytest.approx(2.0)
    # Every world difference identical: a degenerate interval, not a crash.
    flat = pd.DataFrame({"A": [1.0] * 6, "B": [1.0] * 6}, index=idx)
    from ape.analysis.gate_stats import t_bounds

    assert t_bounds(cluster_t(flat, "A", "B", cells=("F7-10",)), 0.025) == (0.0, 0.0)


def test_power_sim_uses_the_analysis_interval():
    """power_sim's vectorized statistic is gate_stats.cluster_t (equal tasks per world)."""
    from ape.analysis.gate_stats import cluster_t

    tm = synth(-0.02, worlds=5, tasks=6, seed=31)
    st = cluster_t(tm, "APG-s", "LGR*")
    wm = (tm["APG-s"] - tm["LGR*"]).groupby(level=["cell", "world"]).mean()
    wd = np.array([wm.xs(c, level="cell").to_numpy() for c in GATE_CELLS])[None]  # (1, cells, worlds)
    est, se, df = ps.cluster_t_stats(wd)
    assert (est[0], se[0], df[0]) == pytest.approx((st["est"], st["se"], st["df"]))


def test_t_interval_type_i_error_is_near_nominal_at_the_margin():
    """RELIABILITY_REVIEW S2 (seeded, modest N): at true Δ = −5 pp the GO rate of the world-clustered t-interval is
    near the nominal one-sided 2.5% (the percentile bootstrap gave 3.3–3.9%). Full runs (20,000 sims per scenario):
    2.61% at 16 worlds, 2.32% at 12, 2.42% with F7-1000 σ_g = 1.0 (D-023)."""
    rate = ps.simulate_ni_power(16, 12, 3, -5.0, p_base=[0.85, 0.45, 0.75, 0.6], alpha=0.025, n_sims=4000, rng=np.random.default_rng(7))
    assert 0.015 <= rate <= 0.035, rate


def test_holm_levels_and_modes():
    from ape.analysis.gate_stats import Decision, holm_levels, holm_modes

    lv, rej = holm_levels({"push": 0.004, "pull": 0.015}, 0.02)
    assert lv == {"push": 0.01, "pull": 0.02} and rej == {"push": True, "pull": True}
    lv, rej = holm_levels({"push": 0.012, "pull": 0.015}, 0.02)
    assert rej == {"push": False, "pull": False} and lv["pull"] == 0.02

    def dec(est, se):
        st = {"est": est, "se": se, "df": 60.0, "cells": {}}
        from ape.analysis.gate_stats import t_p_greater

        inf = {"p_noninferiority": t_p_greater(st, -0.05), "p_superiority": t_p_greater(st, 0.0)}
        sec = {"f7_1000_ok": True, "f7_1000_delta": 0.0, "s7_ok": True, "p_apg_gt_s7": 0.001}
        return Decision("GO", est, (None, None), details={"_stats": st, "inference": inf, "secondary": sec, "cost_ratio": 1.0})

    # Both clearly NI: GO in both; superiority only by gatekeeping, here both far above 0.
    out, info = holm_modes({"push": dec(0.06, 0.01), "pull": dec(0.06, 0.01)}, 0.02)
    assert [d.verdict for d in out.values()] == ["GO", "GO"] and all(d.superiority for d in out.values()) and info["superiority"]["tested"]
    # Pull passes only at its unadjusted level: Holm tests the better mode at 0.01 first, and pull fails there.
    p_only = dec(-0.05 + 2.2 * 0.012, 0.012)  # p(NI) ≈ 0.016: GO at 0.02 alone, not at 0.01
    out, _ = holm_modes({"push": dec(-0.08, 0.02), "pull": p_only}, 0.02)
    assert out["pull"].verdict != "GO" and out["pull"].details["inference"]["holm_level"] == 0.01
    # One mode GO: no superiority test (gatekeeping), whatever its p-value.
    out, info = holm_modes({"push": dec(0.06, 0.01), "pull": dec(-0.08, 0.02)}, 0.02)
    assert out["push"].verdict == "GO" and not out["push"].superiority and not info["superiority"]["tested"]


def test_violation_test_fails_only_on_evidence():
    from ape.analysis.gate_stats import violation_test

    tm = synth(0.0, worlds=9, tasks=12, seed=41, extra_arms={"N": 0.0})
    assert violation_test(tm, "LGR*", "N", tol=0.03)["pass"]  # a tie passes
    worse = synth(0.0, worlds=9, tasks=12, seed=42, extra_arms={"N": 0.2})
    t = violation_test(worse, "LGR*", "N", tol=0.03)
    assert not t["pass"] and t["upper"] < -0.03 and t["p_violation"] <= 0.025


def test_precondition_failure_blocks_the_decision():
    tm = synth(0.0, seed=6, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", {**PC_OK, "PC1": False}, reps=500)
    assert d.verdict == "PRECONDITION_FAIL" and "PC1" in d.reasons[0]


def test_cost_flag_and_placebo_rule():
    tm = synth(0.0, worlds=40, tasks=20, seed=7, extra_arms={"S7": -0.4})
    assert decide(tm, "LGR*", PC_OK, cost_ratio=2.5, reps=2000).verdict == "GO_WITH_COST_FLAG"
    tm_placebo = synth(0.0, worlds=40, tasks=20, seed=8, extra_arms={"S7": 0.0})
    d = decide(tm_placebo, "LGR*", PC_OK, reps=2000)
    assert d.verdict == "NO_GO" and any("placebo" in r for r in d.reasons)


def test_invariants_chain():
    tm = synth(0.0, worlds=10, tasks=10, seed=9, extra_arms={"S6": 0.3, "S5o": 0.1, "S7": -0.4})
    inv = invariants(tm)
    assert inv["pass"], inv
    bad = synth(0.0, worlds=10, tasks=10, seed=9, extra_arms={"S6": -0.3, "S5o": 0.1, "S7": -0.4})
    assert not invariants(bad)["pass"]


def test_invariants_chain_of_ties_passes():
    """RELIABILITY_REVIEW S5: S6 = S5o = APG* near the ceiling failed the point-estimate rule 16% of the time; a
    pair now fails only on evidence of a shortfall beyond 3 pp (simulated: 0 of 1,000 chains of ties fail)."""
    fails = 0
    for s in range(40):
        tm = synth(0.0, worlds=9, tasks=12, seed=100 + s, base=0.85, extra_arms={"S6": 0.0, "S5o": 0.0, "S7": -0.4})
        fails += not invariants(tm)["pass"]
    assert fails <= 2, fails  # seed 103 draws a 4-SD S6 − S5o gap of −6 pp, which the test rightly flags


def test_pooled_delta_weights_cells_equally():
    tm = synth(0.0, seed=10)
    tm.loc[tm.index.get_level_values("cell") == "F7-10", "APG-s"] += 0.4
    assert abs(pooled_delta(tm, "APG-s", "LGR*") - (0.4 / 4 + pooled_delta(synth(0.0, seed=10), "APG-s", "LGR*"))) < 1e-9
