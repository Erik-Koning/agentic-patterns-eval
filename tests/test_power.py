"""Power simulation (module 9): the NI mode agrees with the analytic formula when there is no clustering."""

import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("power_sim", Path(__file__).parents[1] / "power" / "power_sim.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def _empirical_sd_diff(epochs, p_base, sigma_u, sigma_v, rng, n=200_000):
    mu = ps.logit(p_base)
    v = rng.normal(0, sigma_u, n)
    a = rng.binomial(epochs, ps.expit(mu + v + rng.normal(0, sigma_v, n))) / epochs
    b = rng.binomial(epochs, ps.expit(mu + v + rng.normal(0, sigma_v, n))) / epochs
    return float(np.std(a - b))


def test_ni_power_matches_analytic_without_clustering():
    rng = np.random.default_rng(0)
    sd = _empirical_sd_diff(3, 0.6, 1.5, 0.5, rng)
    n = ps.analytic_ni_tasks(sd, true_delta_pp=0.0)
    worlds, cells = 10, 4
    tasks = max(1, round(n / (worlds * cells)))
    power = ps.simulate_ni_power(worlds, tasks, 3, 0.0, sigma_w=0.0, sigma_g=0.0, n_sims=3000, rng=rng)
    assert abs(power - 0.8) < 0.06, (n, tasks, power)


def test_ni_power_falls_with_world_by_arm_variance():
    rng = np.random.default_rng(1)
    low = ps.simulate_ni_power(6, 12, 3, 0.0, sigma_g=0.0, n_sims=2000, rng=rng)
    high = ps.simulate_ni_power(6, 12, 3, 0.0, sigma_g=0.8, n_sims=2000, rng=rng)
    assert high < low


def test_two_mode_holm_power_is_below_one_mode_and_takes_per_cell_baselines():
    """D-023: the gate's unqualified GO needs both delivery modes under Holm, so it is harder than one mode at α."""
    bases = [0.85, 0.45, 0.75, 0.6]
    one = ps.simulate_ni_power(16, 12, 3, 0.0, p_base=bases, alpha=0.02, n_sims=2000, rng=np.random.default_rng(2))
    two = ps.simulate_ni_power(16, 12, 3, 0.0, p_base=bases, alpha=0.02, modes=2, n_sims=2000, rng=np.random.default_rng(2))
    assert 0.75 < two < one, (one, two)  # full runs: 0.896 and 0.837 (20,000 sims)


# --- pilot σ (analysis/pilot.py; RELIABILITY_REVIEW S3) ---

GATE_BASES = {"F7-10": 0.85, "F7-1000": 0.45, "F3-5": 0.75, "F3-60": 0.6}


def _pilot(rng, sigma_w, sigma_g, worlds=4, tasks=12, epochs=1, sigma_u=1.5, sigma_v=0.5):
    """Task means of LGR* and APG* (equal on average) for one pilot under power_sim's logit model."""
    import pandas as pd

    cells = list(GATE_BASES)
    mu = ps.logit(np.array([GATE_BASES[c] for c in cells]))[:, None, None]
    shape = (len(cells), worlds, 1)
    u, ga, gb = rng.normal(0, sigma_w, shape), rng.normal(0, sigma_g, shape), rng.normal(0, sigma_g, shape)
    v = rng.normal(0, sigma_u, (len(cells), worlds, tasks))
    ya = rng.binomial(epochs, ps.expit(mu + u + ga + v + rng.normal(0, sigma_v, v.shape))) / epochs
    yb = rng.binomial(epochs, ps.expit(mu + u + gb + v + rng.normal(0, sigma_v, v.shape))) / epochs
    idx = pd.MultiIndex.from_tuples([(c, f"{c}-w{w}", f"{c}-w{w}-t{t}") for c in cells for w in range(worlds) for t in range(tasks)], names=["cell", "world", "task"])
    return pd.DataFrame({"APG*": ya.ravel(), "LGR*": yb.ravel()}, index=idx)


def test_pilot_sigma_moment_matching_is_roughly_unbiased_where_the_delta_method_was_not():
    """The delta method σ_p / (p̄(1 − p̄)) ignored task heterogeneity and underestimated σ_g = 0.8 at about 0.5
    (300 simulated pilots: 0.50); moment matching to the power model recovers it (300 pilots: 0.77)."""
    from ape.analysis.pilot import variance_components

    rng = np.random.default_rng(5)
    new, old = [], []
    for _ in range(30):
        vc = variance_components(_pilot(rng, 0.5, 0.8, worlds=6), "APG*", "LGR*")
        new.append(vc["sigma_g"])
        p = (vc["success"]["APG*"] + vc["success"]["LGR*"]) / 2
        old.append(vc["probability_scale"]["sigma_g"] / (p * (1 - p)))
    assert 0.62 <= np.mean(new) <= 1.0, np.mean(new)
    assert np.mean(old) < np.mean(new) - 0.15, (np.mean(old), np.mean(new))


def test_pilot_recommendation_uses_the_upper_sigma_bound_and_does_not_overstate_power():
    """At a true σ_g of 0.8 the gate's power at 16 worlds is about 0.38; the old recommendation still said 16 worlds
    reach 0.8 in 70% of pilots. The conservative scenario now uses the upper ends of the 80% σ intervals."""
    from ape.analysis.pilot import power_report, variance_components

    rng = np.random.default_rng(6)
    says_16 = 0
    for _ in range(6):
        vc = variance_components(_pilot(rng, 0.5, 0.8), "APG*", "LGR*")
        assert vc["ci80"]["sigma_g"][0] <= vc["sigma_g"] <= vc["ci80"]["sigma_g"][1]
        r = power_report(vc, (12, 16), 16, 12, 3, 400)
        assert r["scenarios"]["conservative"]["sigma_g"] == vc["ci80"]["sigma_g"][1]
        says_16 += r["recommended_worlds_per_cell"] == 16
    assert says_16 == 0
