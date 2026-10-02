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
        says_16 += r["decision"] == "suffices"
    assert says_16 == 0


# --- D-026: the zero upper bound and the two-sided recommendation ---


def _quiet_pilot(rng, worlds=4):
    """A pilot whose worlds vary less than the model predicts even at σ = 0: every world's arms score alike, task by
    task, so the between-world variance of the paired differences is below the task-sampling noise."""
    import pandas as pd

    one = _pilot(rng, 0.0, 0.0, worlds=1)  # one world's task outcomes, copied into every world
    rows = [(c, f"{c}-w{w}", f"{c}-w{w}-t{t.rsplit('-t', 1)[1]}", r["APG*"], r["LGR*"]) for (c, _, t), r in one.iterrows() for w in range(worlds)]
    df = pd.DataFrame(rows, columns=["cell", "world", "task", "APG*", "LGR*"])
    return df.set_index(["cell", "world", "task"]).sort_index()


def test_a_pivot_with_no_upper_solution_is_flagged_and_never_reported_as_zero():
    """RELIABILITY_REVIEW follow-up (pilot-size simulation): in about 4% of 4-world pilots the χ² pivot's upper solve
    returned 0, so the "conservative" σ_g was 0 and the recommendation optimistic. It now falls back to the prior and is
    flagged, and the report cannot say "suffices" on it."""
    from ape.analysis.pilot import PRIOR_SIGMA_G, PRIOR_SIGMA_W, power_report, variance_components

    vc = variance_components(_quiet_pilot(np.random.default_rng(11)), "APG*", "LGR*")
    assert vc["estimable"]
    flags = vc["ci80_upper_informative"]
    assert not all(flags.values()), (vc["ci80"], flags)
    for key, prior in (("sigma_g", PRIOR_SIGMA_G), ("sigma_w", PRIOR_SIGMA_W)):
        lo, hi = vc["ci80"][key]
        assert hi > 0 and lo <= vc[key] <= hi
        if not flags[key]:
            assert hi == max(vc[key], prior)
    r = power_report(vc, (12, 16), 16, 12, 3, 200)
    assert r["decision"] != "suffices" and "not informative" in r["recommendation"]


def _vc(ci_w=(0.4, 0.6), ci_g=(0.2, 0.4), informative=(True, True)):
    """A hand-made variance-components record: power_report needs only these fields."""
    return {
        "arms": ["APG*", "LGR*"],
        "success": {"APG*": 0.66, "LGR*": 0.66},
        "cell_success": {c: {"APG*": b, "LGR*": b} for c, b in GATE_BASES.items()},
        "estimable": True,
        "sigma_w": sum(ci_w) / 2,
        "sigma_g": sum(ci_g) / 2,
        "ci80": {"sigma_w": list(ci_w), "sigma_g": list(ci_g)},
        "ci80_upper_informative": {"sigma_w": informative[0], "sigma_g": informative[1]},
    }


def test_two_sided_recommendation_suffices_insufficient_and_ambiguous():
    from ape.analysis.pilot import power_report

    # σ_g bounded tightly and low: power at the upper ends ≥ 0.8 (16 worlds about 0.83 at σ_g 0.3).
    r = power_report(_vc(ci_g=(0.0, 0.15)), (12, 16, 20, 24), 16, 12, 3, 600)
    assert r["decision"] == "suffices" and r["recommended_worlds_per_cell"] == 16, r["recommendation"]
    # σ_g high even at its lower end (power about 0.4 at 0.8): insufficient, with 20 and 24 reported, no recommendation.
    r = power_report(_vc(ci_g=(1.0, 1.6)), (12, 16, 20, 24), 16, 12, 3, 600)
    assert r["decision"] == "insufficient" and r["recommended_worlds_per_cell"] is None
    assert "20 worlds give" in r["recommendation"] and "24 worlds give" in r["recommendation"]
    # Wide interval straddling the target: ambiguous, proceed with the planned 16 and the extension.
    r = power_report(_vc(ci_g=(0.0, 1.2)), (12, 16), 16, 12, 3, 600)
    assert r["decision"] == "ambiguous" and r["recommended_worlds_per_cell"] == 16 and "extension" in r["recommendation"]
    # An uninformative upper end never supports "suffices", however good the numbers look.
    r = power_report(_vc(ci_g=(0.0, 0.15), informative=(True, False)), (12, 16), 16, 12, 3, 600)
    assert r["decision"] == "ambiguous" and "not informative" in r["recommendation"]
    # Not estimable: ambiguous on the priors.
    r = power_report({"arms": ["APG*", "LGR*"], "success": {"APG*": 0.66, "LGR*": 0.66}, "estimable": False, "note": "no cell has 2 or more worlds"}, (12, 16), 16, 12, 3, 200)
    assert r["decision"] == "ambiguous" and "not estimable" in r["recommendation"]


def test_sigma_estimation_uses_every_world_it_is_given():
    """D-026: the pilot pools the main pilot's 4 worlds and the σ cell's 4 more per cell."""
    from ape.analysis.pilot import variance_components

    tm = _pilot(np.random.default_rng(12), 0.5, 0.5, worlds=8)
    vc = variance_components(tm, "APG*", "LGR*")
    assert vc["worlds_per_cell"] == dict.fromkeys(GATE_BASES, 8) and vc["df"] == 4 * 7
