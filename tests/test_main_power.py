"""The main-study power simulation (analysis/main_power.py): it runs the real analysis path, holds α at each null
boundary, gains power with the effect, and reads the planned sizes from the run plan."""

import numpy as np
import pandas as pd
import pytest

from ape.analysis import main_power as mp
from ape.analysis import main_stats as ms
from ape.analysis.main_hypotheses import get


def test_planned_sizes_are_the_runners_whole_worlds():
    """BUILD_REVIEW S-7: the runner runs ⌈n / 12⌉ whole worlds of 12 tasks, so 100 tasks are 108 and M1s's 50 are 60."""
    from ape import run_study

    assert mp.TASKS_PER_WORLD == run_study.TASKS_PER_WORLD
    s = mp.planned_sizes()
    assert s[("luna", "F1-32", "S1")] == {"n_tasks": 108, "worlds": 9, "epochs": 8, "plan_cell": "main.A.s1-pool"}
    assert s[("luna", "F1-32", "M1s")]["n_tasks"] == 60 and s[("luna", "F7-100", "S1")]["epochs"] == 3
    m2 = s[("sol", "F7-100", "M2")]
    assert (m2["n_tasks"], m2["epochs"]) == (60, 3) and m2["plan_cell"].startswith("main.F.sol")  # D-052: Sol at 60 tasks (5 worlds)
    assert not any(v["plan_cell"].startswith("main.C.") for v in s.values()), "Study C is determinism only"


def test_cost_multiples_are_the_budget_priors():
    """BUILD_REVIEW S-7: the generator's cost multiples come from config/budget_assumptions.yaml (M7 is 5.0)."""
    import yaml

    arms = yaml.safe_load(mp.BUDGET_ASSUMPTIONS.read_text())["arms"]
    assert mp.COST["M7"] == float(arms["M7"]["multiplier"]) and mp.COST["M1s"] == mp.COST["M1"] and mp.COST["S1"] == 1.0
    assert mp.COST["M2"] == mp.COST["M1k"], "M2 is `like: M1k`"


def test_the_population_frontier_is_exact():
    """BUILD_REVIEW S-8: the null boundaries use the vote's exact expectation, which a large Monte Carlo agrees with."""
    from scipy.special import expit

    from ape.analysis import frontier as fr

    s = mp.Sigmas().total
    x = mp.location(0.6, s)
    exact = np.array(mp.population_s8(round(x, 9), round(s, 9), 0.3, 0.1))
    rng = np.random.default_rng(0)
    p = expit(x + s * rng.standard_normal(60_000))
    ok = rng.random((60_000, 8)) < p[:, None]
    ts, _ = fr.s8_success(mp._answer_codes(rng, ok, 0.3, 0.1), ok.astype(float))
    assert np.allclose(exact, ts.mean(axis=0), atol=0.006)
    assert mp.vote_success(1, np.array([0.3]), 0.3, 0.1)[0] == pytest.approx(0.3)
    # 3 runs, no abstention, own wrong answers only: p^3 + 3p^2(1-p) + p(1-p)^2 (a 3-way tie is a third)
    assert mp.vote_success(3, np.array([0.6]), 0.0, 0.0)[0] == pytest.approx(0.6**3 + 3 * 0.36 * 0.4 + 0.6 * 0.16)


def test_frontier_nulls_are_fixed_points_when_cost_depends_on_success():
    curve = (0.4, 0.42, 0.47, 0.52, 0.56, 0.59, 0.61, 0.62)
    t = mp.S8Target(5.0, 0.06)
    assert mp.frontier_target(curve, t, 0.4, 0.0) == pytest.approx(curve[4] + 0.06)
    p = mp.frontier_target(curve, t, 0.4, 3.0)
    r = 5.0 * (1 + 3 * (1 - p)) / (1 + 3 * (1 - 0.4))
    assert p == pytest.approx(mp.s8_at(curve, r) + 0.06, abs=1e-9)
    kappas = {s.kappa for h in ("M2", "M3") for s in mp.scenarios(h)}
    assert {1.0, 3.0} <= kappas, "M2 and M3 are simulated with cost depending on success"


def test_planned_sizes_follow_d034s_study_f_layout(tmp_path):
    """D-034: M2 leaves F1-32 in Study F (main.F.luna runs S5 there; main.F.sol S1, S5, M1; main.F.sol-m2 M2 on F7-100)."""
    import yaml

    plan = yaml.safe_load(mp.RUN_PLAN.read_text())
    plan["studies"]["main"]["phases"]["study_f"] = [
        {"id": "main.F.luna", "arms": ["S5"], "cells": ["F1-32"], "n_tasks": 100, "epochs": 3},
        {"id": "main.F.luna-f7-100", "arms": ["S1", "S5", "M1", "M2"], "cells": ["F7-100"], "n_tasks": 100, "epochs": 3},
        {"id": "main.F.sol", "profile": "main_sol", "arms": ["S1", "S5", "M1"], "cells": ["F1-32", "F7-100"], "n_tasks": 100, "epochs": 3},
        {"id": "main.F.sol-m2", "profile": "main_sol", "arms": ["M2"], "cells": ["F7-100"], "n_tasks": 100, "epochs": 3},
    ]
    path = tmp_path / "run_plan.yaml"
    path.write_text(yaml.safe_dump(plan))
    s = mp.planned_sizes(path)
    assert s[("sol", "F7-100", "M2")]["plan_cell"] == "main.F.sol-m2"
    assert ("sol", "F1-32", "M2") not in s and ("luna", "F1-32", "M2") not in s
    assert s[("sol", "F1-32", "S1")]["epochs"] == 3 and s[("luna", "F1-32", "S5")]["plan_cell"] == "main.F.luna"
    t1 = [sc for sc in mp.scenarios("T1")]
    assert all("M2" not in arms for sc in t1 for arms in sc.spec.values()), "T1 needs no M2"


def test_location_hits_the_target_marginal_and_s8_1_is_the_s1_marginal():
    s = mp.Sigmas().total
    x = mp.location(0.4, s)
    assert mp._marginal(x, s) == pytest.approx(0.4, abs=1e-6)
    curve = mp.population_s8(round(x, 9), round(s, 9), 0.3, 0.1)
    assert curve[0] == pytest.approx(0.4, abs=0.01) and len(curve) == 8
    assert mp.s8_at(curve, 2.5) == pytest.approx((curve[1] + curve[2]) / 2) and mp.s8_at(curve, 0.5) == curve[0]


@pytest.mark.parametrize("hyp_id", ["M3", "T1"])
def test_the_simulation_shortcut_equals_build_tables_on_the_same_replicate(hyp_id):
    """Draw.tables() (what the power runs use) equals build_tables(Draw.frame()) (the loader's path), down to the
    frontier and every member's p-value."""
    sc = mp.scenarios(hyp_id)[0]
    n, ep = mp._sizes_for(sc.spec, mp.planned_sizes(), mp.Settings())
    draw = mp.simulate(sc.spec, n, ep, mp.Sigmas(), mp.Settings(), np.random.default_rng(3))
    direct, via = draw.tables(), ms.build_tables(draw.frame(), meters={"tokens": "tokens"})
    for tier in direct.tm:
        pd.testing.assert_frame_equal(direct.tm[tier].sort_index(), via.tm[tier].sort_index(), check_names=False)
        pd.testing.assert_frame_equal(direct.cost[tier]["tokens"].sort_index(), via.cost[tier]["tokens"].sort_index(), check_names=False)
    tier, cell = next(iter(sc.spec))
    for k in (None, 3):
        a, b = direct.frontier(tier, cell, "tokens", k), via.frontier(tier, cell, "tokens", k)
        assert np.allclose(a.cost, b.cost) and np.allclose(a.success, b.success)
    pa = ms.evaluate_family(direct, get(hyp_id), reps=500, ci=False)
    pb = ms.evaluate_family(via, get(hyp_id), reps=500, ci=False)
    for key in ("est", "p") if get(hyp_id).status == "confirmatory" else ("est",):
        assert {i: m[key] for i, m in pa["members"].items()} == pytest.approx({i: m[key] for i, m in pb["members"].items()})


def _se(p: float, reps: int) -> float:
    return float(np.sqrt(p * (1 - p) / reps))


@pytest.mark.parametrize("hyp_id,scenario,reps", [("M1", 0, 200), ("M2", 1, 150), ("M5", 0, 150)])
def test_false_claims_at_the_null_boundary_stay_within_alpha(hyp_id, scenario, reps):
    """Superiority (M1), the frontier after its gate (M2: M1 exactly at S8's matched cost), and a pooled TOST at +3 pp
    (M5, Monte Carlo flips over 36 clusters). The full run (main_power CLI, 1,000 replicates) is in the B4 report."""
    sc = mp.scenarios(hyp_id)[scenario]
    assert sc.nulls
    res = mp.simulate_family(get(hyp_id), sc, reps, seed=17, settings=mp.Settings(flip_reps=1000))
    alpha = get(hyp_id).alpha
    assert res["false_claims"] <= alpha + 3 * _se(alpha, reps), res


def test_power_grows_with_the_effect():
    hyp = get("M1")
    sc = {s.name: s for s in mp.scenarios("M1")}
    null = mp.simulate_family(hyp, sc["M1 − S9 = +0.00"], 100, seed=5)
    big = mp.simulate_family(hyp, sc["M1 − S9 = +0.15"], 100, seed=5)
    assert big["members"]["M1.F1-32"] > 0.6 > 0.1 > null["members"]["M1.F1-32"]


def test_min_p_table_flags_what_each_cluster_count_can_reach():
    rows = {r["member"]: r for r in mp.min_p_table()}
    assert rows["M1.F1-32"]["clusters"] == 9 and rows["M1.F1-32"]["min_p"] == pytest.approx(1 / 512) and rows["M1.F1-32"]["reachable"]
    assert rows["M3.pooled"]["clusters"] == 9, "Study A's four cells share 9 registry KBs"
    assert rows["M5.pooled"]["clusters"] == 36 and not rows["M5.pooled"]["exact"]
    assert rows["K2-NI.pooled"]["clusters"] == 36 and rows["K2-NI.pooled"]["smallest_holm_level"] == pytest.approx(0.025), "D-033: its own family"
    assert rows["K2.interaction"]["smallest_holm_level"] == pytest.approx(0.025) and not any(m.startswith("T1") for m in rows), "T1 is descriptive"
    tight = {r["member"]: r for r in mp.min_p_table(mp.Settings(worlds=5))}
    assert not tight["K1.F7-1000"]["reachable"], "5 worlds: 2^-5 = 0.031 > 0.0125"


def test_sigmas_from_a_pilot_result_and_the_priors_where_it_could_not_estimate():
    assert mp.sigmas_from_pilot({"estimable": True, "sigma_w": 0.9, "sigma_g": 0.6}) == mp.Sigmas(w=0.9, g=0.6)
    assert mp.sigmas_from_pilot({"estimable": False}) == mp.Sigmas()
    assert mp.sigmas_from_pilot(None) == mp.Sigmas()


def test_render_and_cli_on_a_tiny_run(tmp_path, capsys):
    assert mp.main(["--reps", "3", "--families", "M1", "--out", str(tmp_path / "p.json")]) == 0
    out = capsys.readouterr().out
    assert "M1.F1-32" in out and "Min p" in out and (tmp_path / "p.json").is_file()
