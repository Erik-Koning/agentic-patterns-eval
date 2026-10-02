"""Simulation-based power analysis: paired superiority contrasts (main study) and the
non-inferiority gate (`--mode ni`, world-clustered t-interval; push/pull with Holm, D-023).

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


def cluster_t_stats(world_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """est, se, Satterthwaite df of the gate's world-clustered t statistic (`gate_stats.cluster_t` with equal tasks
    per world), from per-world mean paired differences of shape (sims, cells, worlds)."""
    cells, worlds = world_d.shape[-2], world_d.shape[-1]
    v = world_d.var(axis=-1, ddof=1) / worlds / cells**2  # (sims, cells)
    tot = v.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df = tot**2 / (v**2 / (worlds - 1)).sum(axis=-1)
    return world_d.mean(axis=-1).mean(axis=-1), np.sqrt(tot), np.nan_to_num(df, nan=1e6)


def simulate_ni_power(
    worlds_per_cell: int,
    tasks_per_world: int,
    epochs: int,
    true_delta_pp: float,
    margin_pp: float = 5.0,
    p_base: float | list[float] = 0.6,
    sigma_u: float = 1.5,
    sigma_v: float = 0.5,
    sigma_w: float = 0.5,
    sigma_g: float = 0.3,
    cells: int = 4,
    alpha: float = 0.025,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
    modes: int = 1,
    f7_cells: int = 2,
) -> float:
    """Power of the gate's one-sided non-inferiority decision (GATE_PREREG.md §2, §3, §8; D-023).

    logit P(y) = mu_c + u_world(sigma_w) + g_world_arm(sigma_g) + v_task(sigma_u) + e_task_arm(sigma_v) + beta_arm.
    `p_base` is one baseline for every cell, or one per cell. The estimate is the equal-weighted mean over cells of
    per-cell mean world-level paired differences, tested with the analysis's world-clustered t-interval
    (Satterthwaite df).

    modes=1: P(the lower bound at one-sided `alpha` exceeds −margin).
    modes=2: the co-primary push/pull decision: the first `f7_cells` cells are run in both modes (sharing world,
    task and world × arm effects, as the same KG serves both; task × arm noise and epochs are independent), the rest
    are shared push cells; the two modes' NI hypotheses are tested with Holm at familywise `alpha`. Returns
    P(GO in both modes), the unqualified GO.
    """
    rng = rng or np.random.default_rng(0)
    bases = np.broadcast_to(np.asarray(p_base, dtype=float), (cells,))
    other = np.hypot(sigma_w, np.hypot(sigma_g, sigma_v))
    mu = logit(bases)
    if true_delta_pp == 0:
        beta = np.zeros(cells)  # exact: no shift solves Δ = 0 (the numeric solve would only add Monte Carlo noise)
    else:
        beta = np.array([_marginal_shift(m, sigma_u, other, _marginal(m, np.hypot(sigma_u, other), rng) + true_delta_pp / 100.0, rng) for m in mu])
    mu_c, beta_c = mu[None, :, None, None], beta[None, :, None, None]
    shape = (n_sims, cells, worlds_per_cell)
    u = rng.normal(0, sigma_w, shape)[..., None]
    g_a = rng.normal(0, sigma_g, shape)[..., None]
    g_b = rng.normal(0, sigma_g, shape)[..., None]
    tshape = (*shape, tasks_per_world)
    v = rng.normal(0, sigma_u, tshape)

    def world_diffs() -> np.ndarray:
        y_a = rng.binomial(epochs, expit(mu_c + beta_c + u + g_a + v + rng.normal(0, sigma_v, tshape))) / epochs
        y_b = rng.binomial(epochs, expit(mu_c + u + g_b + v + rng.normal(0, sigma_v, tshape))) / epochs
        return (y_a - y_b).mean(axis=-1)  # (sims, cells, worlds)

    m = -margin_pp / 100.0
    push = world_diffs()
    est, se, df = cluster_t_stats(push)
    if modes == 1:
        return float(np.mean(est - student_t.ppf(1 - alpha, df) * se > m))
    pull = world_diffs()
    pull[:, f7_cells:] = push[:, f7_cells:]  # F3 is push only: the pull verdict pools the same F3 cells
    est2, se2, df2 = cluster_t_stats(pull)
    p1 = student_t.sf((est - m) / se, df)
    p2 = student_t.sf((est2 - m) / se2, df2)
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    return float(np.mean((lo <= alpha / 2) & (hi <= alpha)))


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
    ap.add_argument("--alpha", type=float, default=0.020, help="NI mode: one-sided familywise alpha of the gate run (stage 1, D-023)")
    ap.add_argument("--modes", type=int, choices=[1, 2], default=2, help="NI mode: 2 = push and pull with Holm (P(unqualified GO)); 1 = one mode")
    args = ap.parse_args()
    rng = np.random.default_rng(20260929)

    if args.mode == "ni":
        print(f"NI gate power: margin 5 pp, one-sided alpha {args.alpha} ({args.modes} mode(s), Holm), 4 cells, sigma_w={args.sigma_w}, sigma_g={args.sigma_g}")
        print("true_delta_pp  worlds/cell  tasks/world  epochs  total_tasks  power")
        for delta, worlds, tasks, epochs in itertools.product([2, 0, -2], [8, 12, 16], [12], [3]):
            pw = simulate_ni_power(worlds, tasks, epochs, delta, sigma_w=args.sigma_w, sigma_g=args.sigma_g, alpha=args.alpha, modes=args.modes, n_sims=args.sims, rng=rng)
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
