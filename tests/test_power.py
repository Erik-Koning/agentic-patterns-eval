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
