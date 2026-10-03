"""Main-study statistics (analysis/main_stats.py): the clustered sign flip, its intervals, margins, Holm and gatekeeping,
the frontier through the tables, and robustness to missing data (reported, never raised)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from ape.analysis import gate_stats
from ape.analysis import main_power as mp
from ape.analysis import main_stats as ms
from ape.analysis.main_hypotheses import NI_MARGIN, TOST_MARGIN, Hypothesis, Member, Term, get

ALPHA = 0.025


def _tm(cells: dict[str, dict[str, list[list[float]]]]) -> pd.DataFrame:
    """Task means from {cell: {arm: [[task values of world 0], [world 1], ...]}}."""
    rows = []
    for cell, arms in cells.items():
        first = next(iter(arms.values()))
        for w, tasks in enumerate(first):
            for t in range(len(tasks)):
                rows.append({"cell": cell, "world": f"{cell}-test-s{4000 + w}", "task": f"{cell}-test-s{4000 + w}-t{t:03d}"} | {a: v[w][t] for a, v in arms.items()})
    return pd.DataFrame(rows).set_index(["cell", "world", "task"]).rename_axis(columns="arm")


def _tables(tm: pd.DataFrame, cost: dict | None = None, tier: str = "luna") -> ms.Tables:
    return ms.Tables({tier: tm}, {tier: cost or {}})


def _const(d: float, worlds: int = 9, tasks: int = 3, base: float = 0.5) -> dict:
    """Arms A and B with A − B = d on every task."""
    return {"A": [[base + d] * tasks for _ in range(worlds)], "B": [[base] * tasks for _ in range(worlds)]}


# ---------------------------------------------------------------- the sign flip


def test_exact_sign_flip_matches_brute_force_enumeration():
    rng = np.random.default_rng(0)
    d = pd.Series(rng.normal(0.05, 0.2, 18), index=_tm({"F7-10": _const(0, worlds=6)}).index)
    c = ms.Contrast.from_series(d)
    s, _ = c.contributions()
    T = s.sum()
    brute = np.mean([np.dot(e, s) >= T - 1e-12 for e in itertools.product([-1, 1], repeat=len(s))])
    assert ms.SignFlip(c).p(0.0, "greater") == pytest.approx(brute)
    brute_less = np.mean([np.dot(e, s) <= T + 1e-12 for e in itertools.product([-1, 1], repeat=len(s))])
    assert ms.SignFlip(c).p(0.0, "less") == pytest.approx(brute_less)


def test_monte_carlo_flips_agree_with_the_exact_test_and_the_floor_is_two_to_the_minus_g():
    rng = np.random.default_rng(1)
    tm = _tm({"F7-10": {"A": rng.random((12, 4)).tolist(), "B": rng.random((12, 4)).tolist()}})
    c = ms.Contrast.from_series(tm["A"] - tm["B"])
    exact = ms.SignFlip(c, exact_max=20).p(0.0, "greater")
    mc = ms.SignFlip(c, reps=20_000, seed=3, exact_max=5).p(0.0, "greater")
    assert abs(exact - mc) < 0.015
    allpos = ms.Contrast.from_series(_tm({"F7-10": _const(0.1)})["A"] - _tm({"F7-10": _const(0.1)})["B"])
    flip = ms.SignFlip(allpos)
    assert flip.p(0.0, "greater") == pytest.approx(2.0**-9) == flip.min_p


def test_shifted_p_is_monotone_in_the_null_and_the_inverted_bounds_sit_at_alpha():
    rng = np.random.default_rng(2)
    tm = _tm({"F3-60": {"A": (0.6 + 0.1 * rng.standard_normal((10, 5))).tolist(), "B": (0.5 + 0.1 * rng.standard_normal((10, 5))).tolist()}})
    c = ms.Contrast.from_series(tm["A"] - tm["B"])
    flip = ms.SignFlip(c)
    ps = [flip.p(x, "greater") for x in np.linspace(-0.3, 0.3, 61)]
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(ps))
    lo, hi = flip.bound(0.025, "lower", c.est), flip.bound(0.025, "upper", c.est)
    assert lo < c.est < hi
    assert flip.p(lo - 1e-4, "greater") <= 0.025 < flip.p(lo + 1e-4, "greater")
    assert flip.p(hi + 1e-4, "less") <= 0.025 < flip.p(hi - 1e-4, "less")


# ---------------------------------------------------------------- margins: exact accept / reject at the boundary


@pytest.mark.parametrize("tost_m,ni_m", [(0.03, 0.03), (TOST_MARGIN, NI_MARGIN)], ids=["3pp", "d033"])
@pytest.mark.parametrize("d", [-1.001, -0.999, -0.5, 0.0, 0.999, 1.001])
def test_tost_and_ni_accept_just_inside_and_reject_just_outside_the_margin(tost_m, ni_m, d):
    """Every world has A − B = Δ: each shifted test has p = 2^-9 when Δ is inside, 1 when outside. Δ = d × margin, at
    the brief's 3 pp and at D-033's margins (TOST ±6 pp, NI 5 pp)."""
    terms = (Term("A"), Term("B", -1))
    for kind, m in (("tost", tost_m), ("ni", ni_m)):
        delta = d * m
        t = _tables(_tm({"F7-10": _const(delta)}))
        res = ms.evaluate_family(t, Hypothesis("T", "test", "", "confirmatory", (Member("T.x", terms, ("F7-10",), kind, m),), alpha=0.05 if kind == "tost" else ALPHA), reps=500)
        member = res["members"]["T.x"]
        inside = abs(delta) < m if kind == "tost" else delta > -m
        assert (member["label"] == "supported") is inside, (kind, m, delta)
        if kind == "tost":
            assert (member["ci"][0] > -m and member["ci"][1] < m) is inside, "the inverted interval agrees with the TOST decision"


def test_cluster_t_reproduces_the_gates_on_nested_clusters():
    rng = np.random.default_rng(4)
    tm = _tm({c: {"A": rng.random((7, 5)).tolist(), "B": rng.random((7, 5)).tolist()} for c in ("F7-10", "F3-60")})
    ours = ms.cluster_t(ms.Contrast.from_series((tm["A"] - tm["B"]).dropna()))
    gate = gate_stats.cluster_t(tm, "A", "B", cells=("F7-10", "F3-60"))
    assert ours["est"] == pytest.approx(gate["est"]) and ours["se"] == pytest.approx(gate["se"]) and ours["df"] == pytest.approx(gate["df"])


def test_registry_worlds_of_one_seed_are_one_cluster_across_cells():
    assert ms.cluster_of("F1-32-test-s4001") == ms.cluster_of("F2-10-test-s4001") == "REG-test-s4001"
    assert ms.cluster_of("F7-1000-test-s4001") == "F7-1000-test-s4001"
    assert ms.cluster_of("F1-32-test-s4001", by="world") == "F1-32-test-s4001"
    rng = np.random.default_rng(5)
    tm = _tm({c: {"A": rng.random((9, 3)).tolist(), "B": rng.random((9, 3)).tolist()} for c in ("F1-2", "F1-32", "F2-2", "F2-10")})
    c = ms.Contrast.from_series(tm["A"] - tm["B"])
    st = ms.cluster_t(c)
    assert c.G == 9 and not c.nested and st["df"] == 8


# ---------------------------------------------------------------- Holm and gatekeeping


def test_holm_levels_and_its_stopping_rule_label_each_member():
    cells = {"F7-10": _const(0.1), "F3-5": _const(0.2), "F3-60": _const(-0.1)}
    tm = pd.concat([_tm({c: v}) for c, v in cells.items()])
    hyp = Hypothesis("H", "test", "", "confirmatory", tuple(Member(f"H.{c}", (Term("A"), Term("B", -1)), (c,), "superiority") for c in cells))
    res = ms.evaluate_family(_tables(tm), hyp, reps=500)["members"]
    assert res["H.F7-10"]["label"] == res["H.F3-5"]["label"] == "supported"
    assert {res["H.F7-10"]["holm_level"], res["H.F3-5"]["holm_level"]} == {ALPHA / 3, ALPHA / 2}
    assert res["H.F3-60"]["label"] == "not supported" and res["H.F3-60"]["holm_level"] == ALPHA
    # The first non-rejection stops Holm: the next member is "not tested".
    tm2 = pd.concat([_tm({"F7-10": _const(-0.1)}), _tm({"F3-5": _const(0.0)})])
    hyp2 = Hypothesis("H", "test", "", "confirmatory", tuple(Member(f"H.{c}", (Term("A"), Term("B", -1)), (c,), "superiority") for c in ("F7-10", "F3-5")))
    labels = sorted(m["label"] for m in ms.evaluate_family(_tables(tm2), hyp2, reps=500)["members"].values())
    assert labels == ["not supported", "not tested"]


@pytest.mark.parametrize("gate_d,expect", [(0.1, ("supported", "supported")), (-0.1, ("not supported", "not tested"))])
def test_serial_gatekeeping_tests_stage_two_only_after_stage_one(gate_d, expect):
    tm = _tm({"F1-32": {"M1": [[0.6] * 3] * 9, "S1": [[0.6 - gate_d] * 3] * 9, "X": [[0.4] * 3] * 9}})
    hyp = Hypothesis("G", "test", "", "confirmatory", (
        Member("G.gate", (Term("M1"), Term("S1", -1)), ("F1-32",), "superiority", stage=1),
        Member("G.next", (Term("M1"), Term("X", -1)), ("F1-32",), "superiority", stage=2),
    ), procedure="serial")  # fmt: skip
    res = ms.evaluate_family(_tables(tm), hyp, reps=500)["members"]
    assert (res["G.gate"]["label"], res["G.next"]["label"]) == expect
    if expect[1] == "not tested":
        assert "gate" in res["G.next"]["why"] and res["G.next"]["p"] is not None, "estimated and reported, never tested"


# ---------------------------------------------------------------- robustness


def test_missing_arms_and_cells_are_not_evaluable_never_raised():
    tm = _tm({"F1-32": {"M1": [[0.6] * 3] * 9, "S1": [[0.5] * 3] * 9}})
    t = _tables(tm)
    res = ms.evaluate_family(t, get("M1"), reps=200)  # S9 missing
    m = res["members"]["M1.F1-32"]
    assert m["label"] == "not evaluable" and "S9" in m["reason"] and m["p"] is None
    res = ms.evaluate_family(t, get("M2"), reps=200)  # no S1 pool rows: the frontier member is not evaluable
    assert res["members"]["M2.frontier"]["label"] in ("not evaluable", "not tested")
    assert not ms.evaluate_family(_tables(tm.iloc[0:0]), get("K2"), reps=200)["supported"]
    tm_nan = tm.copy()
    tm_nan.iloc[:4, 0] = np.nan
    m = ms.evaluate_member(_tables(tm_nan), Member("x", (Term("M1"), Term("S1", -1)), ("F1-32", "F1-2"), "superiority"), ALPHA, reps=200)
    assert m["tasks"] == 23 and m["coverage"]["cells"]["F1-32"]["unpaired_tasks"] == 4 and m["coverage"]["missing_cells"] == ["F1-2"]


def test_build_tables_keeps_s1s_main_epochs_drops_study_c_duplicates_and_nan_rows():
    rows = []
    for e in range(1, 9):
        rows.append({"plan_cell": "main.A.s1-pool", "arm": "S1", "cell": "F1-32", "world": "F1-32-test-s4000", "task": "t0", "epoch": e, "tier": "luna", "success": float(e <= 2), "answer_key": "a" if e <= 2 else f"w{e}", "tokens": 10.0})
    for e in (1, 2, 3):
        rows.append({"plan_cell": "main.A.arms", "arm": "M1", "cell": "F1-32", "world": "F1-32-test-s4000", "task": "t0", "epoch": e, "tier": "luna", "success": 1.0, "answer_key": "a", "tokens": 25.0})
    rows.append(rows[-1] | {"plan_cell": "main.A.duplicate"})
    rows.append(rows[-1] | {"plan_cell": "main.C.a", "epoch": 1, "success": 0.0})
    rows.append(rows[-1] | {"plan_cell": "main.A.arms", "task": "t1", "success": np.nan})
    t = ms.build_tables(pd.DataFrame(rows), meters={"tokens": "tokens"})
    assert t.tm["luna"].loc[("F1-32", "F1-32-test-s4000", "t0"), "S1"] == pytest.approx(2 / 3)
    assert t.tm["luna"].loc[("F1-32", "F1-32-test-s4000", "t0"), "M1"] == 1.0
    assert t.notes["duplicates_dropped"] == 1 and t.notes["nan_success_rows"] == 1 and t.notes["excluded_plan_cells"] == ["main.C.a"]
    pool = t.pool("luna", "F1-32")
    assert pool.K == 8 and pool.success.sum() == 2 and pool.codes.tolist() == [[0, 0, 1, 2, 3, 4, 5, 6]]
    assert t.pool("luna", "F1-32", 3).K == 3


def test_a_pool_split_over_several_logs_keeps_every_run():
    """Eight single-epoch logs (epoch 1 in each): runs are ordered by log file, none is mistaken for a duplicate."""
    rows = [{"plan_cell": "main.A.s1-pool", "log_file": f"s1-{e}.eval", "arm": "S1", "cell": "F1-32", "world": "F1-32-test-s4000", "task": "t0", "epoch": 1, "tier": "luna", "success": float(e < 2), "answer_key": "a" if e < 2 else f"w{e}", "tokens": 1.0} for e in range(8)]
    t = ms.build_tables(pd.DataFrame(rows), meters={"tokens": "tokens"})
    assert t.notes["duplicates_dropped"] == 0 and t.pool("luna", "F1-32").K == 8
    assert t.tm["luna"]["S1"].iloc[0] == pytest.approx(2 / 3), "S1's main epochs are its first 3 runs"
    t2 = ms.build_tables(pd.DataFrame(rows + rows[:1]), meters={"tokens": "tokens"})
    assert t2.notes["duplicates_dropped"] == 1, "the same log loaded twice is a duplicate"


def test_a_pool_keeps_full_tasks_and_counts_the_short_ones():
    rows = [{"cell": "F7-10", "world": "w", "task": f"t{i}", "epoch": e, "success": 1.0, "answer_key": "a", "tokens": 1.0} for i in range(3) for e in range(1, 4)]
    rows = rows[:-1]  # t2 has only 2 runs
    pool = ms.Pool.from_rows(pd.DataFrame(rows), {"tokens": "tokens"})
    assert pool.K == 3 and len(pool.index) == 2 and pool.short_tasks == 1


def test_frontier_member_matches_the_hand_computed_interpolation_and_names_beyond():
    """Pool of 3 S1 runs per task costing 1 each; M1 costs 2.5 runs: S8@M1 = (S8(2) + S8(3)) / 2 per task."""
    rows = []
    keys = [("a", "b", "a"), ("a", "b", "c")]
    for w in range(9):
        for t, ks in enumerate(keys):
            task = f"F1-32-test-s{4000 + w}-t{t}"
            for e, k in enumerate(ks, start=1):
                rows.append({"plan_cell": "p", "arm": "S1", "cell": "F1-32", "world": f"F1-32-test-s{4000 + w}", "task": task, "epoch": e, "tier": "luna", "success": float(k == "a"), "answer_key": k, "tokens": 1.0})
            for e in (1, 2, 3):
                rows.append({"plan_cell": "p", "arm": "M1", "cell": "F1-32", "world": f"F1-32-test-s{4000 + w}", "task": task, "epoch": e, "tier": "luna", "success": 1.0, "answer_key": "a", "tokens": 2.5})
    t = ms.build_tables(pd.DataFrame(rows), meters={"tokens": "tokens"})
    m = ms.evaluate_member(t, get("M2").members[1], ALPHA, reps=200)
    s8 = np.mean([(2 / 3 + 1.0) / 2, (1 / 3 + 1 / 3) / 2])
    assert m["est"] == pytest.approx(1.0 - s8)
    match = m["coverage"]["cells"]["F1-32"]["frontier"]["S8@M1"]["match"]
    assert match["status"] == "inside" and match["k"] == 2 and match["lambda"] == pytest.approx(0.5)
    t.cost["luna"]["tokens"]["M1"] = 3.5  # costlier than S8(3)
    m = ms.evaluate_member(t, get("M2").members[1], ALPHA, reps=200)
    assert not m["evaluable"] and "beyond" in m["reason"]


# ---------------------------------------------------------------- operating characteristics (small simulation)


def test_interval_coverage_and_type_i_error_near_nominal_at_nine_worlds():
    """M1-like single-cell contrast at the planned 100 tasks over 9 worlds (power model, prior σ)."""
    settings = mp.Settings()
    m = Member("x", (Term("M1"), Term("S9", -1)), ("F1-32",), "superiority")
    rng = np.random.default_rng(11)
    reps, cover, rej = 200, 0, 0
    for r in range(reps):
        draw = mp.simulate({("luna", "F1-32"): {"S9": 0.45, "M1": 0.45}}, {("luna", "F1-32"): 100}, {("luna", "F1-32", "S9"): 3, ("luna", "F1-32", "M1"): 3}, mp.Sigmas(), settings, rng)
        res = ms.evaluate_member(draw.tables(), m, ALPHA, reps=500, seed=r)
        cover += res["ci"][0] <= 0.0 <= res["ci"][1]
        rej += res["p"] <= ALPHA
    assert 0.90 <= cover / reps <= 0.995, cover / reps  # 1,000 replicates: 0.962
    assert rej / reps <= ALPHA + 3 * np.sqrt(ALPHA * (1 - ALPHA) / reps), rej / reps  # 1,000 replicates: 0.026
