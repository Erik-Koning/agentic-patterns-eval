"""The main-study power simulation (analysis/main_power.py): it runs the real analysis path, holds α at each null
boundary, gains power with the effect, and reads the planned sizes from the run plan."""

import numpy as np
import pandas as pd
import pytest

from ape.analysis import main_power as mp
from ape.analysis import main_stats as ms
from ape.analysis.main_hypotheses import get


def test_planned_sizes_come_from_the_run_plan():
    s = mp.planned_sizes()
    assert s[("luna", "F1-32", "S1")] == {"n_tasks": 100, "epochs": 8, "plan_cell": "main.A.s1-pool"}
    assert s[("luna", "F1-32", "M1s")]["n_tasks"] == 50 and s[("luna", "F7-100", "S1")]["epochs"] == 3
    assert s[("sol", "F7-100", "M2")] == {"n_tasks": 100, "epochs": 3, "plan_cell": "main.F.sol"}
    assert not any(v["plan_cell"].startswith("main.C.") for v in s.values()), "Study C is determinism only"


def test_location_hits_the_target_marginal_and_s8_1_is_the_s1_marginal():
    s = mp.Sigmas().total
    x = mp.location(0.4, s)
    assert mp._marginal(x, s) == pytest.approx(0.4, abs=1e-6)
    curve = mp.population_s8(round(x, 9), round(s, 9), 0.3, 0.1)
    assert curve[0] == pytest.approx(0.4, abs=0.01) and len(curve) == 8
    assert mp.s8_at(curve, 2.5) == pytest.approx((curve[1] + curve[2]) / 2) and mp.s8_at(curve, 0.5) == curve[0]


@pytest.mark.parametrize("hyp_id", ["M3", "H6"])
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
    assert {i: m["p"] for i, m in pa["members"].items()} == pytest.approx({i: m["p"] for i, m in pb["members"].items()})


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
    assert rows["K2.ni-F3"]["clusters"] == 18 and rows["K2.ni-F3"]["smallest_holm_level"] == pytest.approx(0.025 / 3)
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
