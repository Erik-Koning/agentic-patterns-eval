"""Simulation-based power analysis for paired pattern-vs-baseline contrasts.

Generative model (per task type):
  logit P(y_ipr = 1) = mu + u_i + beta_p + v_ip
    u_i  ~ N(0, sigma_u^2)   task difficulty (shared by all patterns -> pairing)
    v_ip ~ N(0, sigma_v^2)   task-specific pattern effect (heterogeneity)
    r i.i.d. Bernoulli trials per (task, pattern)

Test: paired sign-flip permutation test on task-level mean success,
two-sided, alpha = 0.05 / m (Bonferroni over m confirmatory contrasts per
task type; Holm is uniformly more powerful, so these n are conservative).

Calibrate sigma_u, sigma_v and baseline accuracy from the pilot, then rerun.
"""

import argparse
import itertools

import numpy as np
from scipy.special import expit, logit
from scipy.stats import t as student_t


def simulate_power(n_tasks, trials, p_base, delta_pp, sigma_u, sigma_v, alpha, n_sims, rng):
    mu = logit(p_base)
    # Solve for the logit shift that yields the target marginal delta.
    beta = _marginal_shift(mu, sigma_u, sigma_v, p_base + delta_pp / 100.0, rng)
    rejections = 0
    for _ in range(n_sims):
        u = rng.normal(0.0, sigma_u, n_tasks)
        v_base = rng.normal(0.0, sigma_v, n_tasks)
        v_treat = rng.normal(0.0, sigma_v, n_tasks)
        y_base = rng.binomial(trials, expit(mu + u + v_base)) / trials
        y_treat = rng.binomial(trials, expit(mu + u + beta + v_treat)) / trials
        d = y_treat - y_base
        sd = d.std(ddof=1)
        if sd == 0:
            continue
        # Paired t on task-level means; with n >= 30 it matches the sign-flip test closely.
        t_stat = d.mean() / (sd / np.sqrt(n_tasks))
        p = 2 * student_t.sf(abs(t_stat), n_tasks - 1)
        rejections += p < alpha
    return rejections / n_sims


def _marginal_shift(mu, sigma_u, sigma_v, target, rng, draws=200_000):
    z = rng.normal(0.0, np.sqrt(sigma_u**2 + sigma_v**2), draws)
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mu + z + mid).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def min_n(target_power, **kw):
    for n in [30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 650, 800, 1000]:
        if simulate_power(n, **kw) >= target_power:
            return n
    return ">1000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--contrasts", type=int, default=8)
    ap.add_argument("--power", type=float, default=0.8)
    args = ap.parse_args()
    rng = np.random.default_rng(20260929)
    alpha = 0.05 / args.contrasts

    print(f"alpha per test = {alpha:.5f} (0.05 / {args.contrasts}), target power = {args.power}")
    print("p_base  delta_pp  sigma_v  trials  -> min tasks per task type")
    for p_base, delta, sigma_v, trials in itertools.product(
        [0.5], [5, 8, 10, 15], [0.5, 1.0], [1, 3, 5]
    ):
        n = min_n(
            args.power,
            trials=trials,
            p_base=p_base,
            delta_pp=delta,
            sigma_u=1.5,
            sigma_v=sigma_v,
            alpha=alpha,
            n_sims=args.sims,
            rng=rng,
        )
        print(f"{p_base:6.2f}  {delta:8d}  {sigma_v:7.1f}  {trials:6d}  -> {n}")


if __name__ == "__main__":
    main()
