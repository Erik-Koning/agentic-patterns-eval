"""Simulation-based power analysis: paired superiority contrasts (main study) and the
non-inferiority gate (`--mode ni`, world-clustered).

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
    # Solve for the logit shift that moves the marginal success by delta_pp relative to the
    # baseline arm's own marginal (which random effects pull toward 0.5, so it is not p_base).
    beta = _marginal_shift(mu, sigma_u, sigma_v, _marginal(mu, np.hypot(sigma_u, sigma_v), rng) + delta_pp / 100.0, rng)
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


def _marginal(mu, sd, rng, draws=200_000) -> float:
    return float(expit(mu + rng.normal(0.0, sd, draws)).mean())


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


def simulate_ni_power(
    worlds_per_cell: int,
    tasks_per_world: int,
    epochs: int,
    true_delta_pp: float,
    margin_pp: float = 5.0,
    p_base: float = 0.6,
    sigma_u: float = 1.5,
    sigma_v: float = 0.5,
    sigma_w: float = 0.5,
    sigma_g: float = 0.3,
    cells: int = 4,
    alpha: float = 0.025,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Power of the gate's one-sided non-inferiority test (GATE_PREREG.md).

    logit P(y) = mu + u_world(sigma_w) + g_world_arm(sigma_g) + v_task(sigma_u) + e_task_arm(sigma_v) + beta_arm.
    The estimate is the equal-weighted mean over cells of per-cell mean world-level
    paired differences; SE from between-world variance within cells (normal
    approximation to the world-clustered bootstrap used in the real analysis).
    """
    from scipy.stats import norm

    rng = rng or np.random.default_rng(0)
    mu = logit(p_base)
    other = np.hypot(sigma_w, np.hypot(sigma_g, sigma_v))
    beta = _marginal_shift(mu, sigma_u, other, _marginal(mu, np.hypot(sigma_u, other), rng) + true_delta_pp / 100.0, rng)
    z = norm.ppf(1 - alpha)
    shape = (n_sims, cells, worlds_per_cell)
    u = rng.normal(0, sigma_w, shape)[..., None]
    g_a = rng.normal(0, sigma_g, shape)[..., None]
    g_b = rng.normal(0, sigma_g, shape)[..., None]
    tshape = (*shape, tasks_per_world)
    v = rng.normal(0, sigma_u, tshape)
    y_a = rng.binomial(epochs, expit(mu + beta + u + g_a + v + rng.normal(0, sigma_v, tshape))) / epochs
    y_b = rng.binomial(epochs, expit(mu + u + g_b + v + rng.normal(0, sigma_v, tshape))) / epochs
    world_d = (y_a - y_b).mean(axis=-1)  # (sims, cells, worlds)
    est = world_d.mean(axis=-1).mean(axis=-1)
    se = np.sqrt((world_d.var(axis=-1, ddof=1) / worlds_per_cell).sum(axis=-1)) / cells
    return float(np.mean(est - z * se > -margin_pp / 100.0))


def analytic_ni_tasks(sd_diff: float, true_delta_pp: float, margin_pp: float = 5.0, alpha: float = 0.025, power: float = 0.8) -> float:
    """Paired non-inferiority sample size without clustering: n = ((z_{1-a} + z_{1-b}) sd / (delta + margin))^2."""
    from scipy.stats import norm

    return ((norm.ppf(1 - alpha) + norm.ppf(power)) * sd_diff / ((true_delta_pp + margin_pp) / 100.0)) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["superiority", "ni"], default="superiority")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--contrasts", type=int, default=8)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--sigma-w", type=float, default=0.5, help="NI mode: world effect (logit SD); calibrate from the pilot")
    ap.add_argument("--sigma-g", type=float, default=0.3, help="NI mode: world x arm effect (logit SD); calibrate from the pilot")
    args = ap.parse_args()
    rng = np.random.default_rng(20260929)

    if args.mode == "ni":
        print(f"NI gate power: margin 5 pp, one-sided alpha 0.025, 4 cells, sigma_w={args.sigma_w}, sigma_g={args.sigma_g}")
        print("true_delta_pp  worlds/cell  tasks/world  epochs  total_tasks  power")
        for delta, worlds, tasks, epochs in itertools.product([2, 0, -2], [4, 6, 8], [8, 12, 16], [2, 3]):
            pw = simulate_ni_power(worlds, tasks, epochs, delta, sigma_w=args.sigma_w, sigma_g=args.sigma_g, n_sims=args.sims, rng=rng)
            print(f"{delta:13d}  {worlds:11d}  {tasks:11d}  {epochs:6d}  {4 * worlds * tasks:11d}  {pw:5.2f}")
        return

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
