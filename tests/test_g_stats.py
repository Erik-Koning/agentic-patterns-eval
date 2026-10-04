"""Study G statistics (BUILD_PLAN B10): the session-clustered estimators, tests and decisions on synthetic data with
known effects, their robustness to missing data, and the power simulation's calibration at the planned sizes."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from ape.analysis import g_hypotheses as gh
from ape.analysis import g_power as gp
from ape.analysis import g_report as gr
from ape.analysis import g_stats as gs
from ape.analysis.g_load import sessions_from_items
from ape.analysis.gate_stats import cluster_t


def _values(rng, sizes: dict[str, int], mean=0.0, sd=1.0, shared: bool = False, world_sd: float = 0.0, slope: dict | None = None) -> pd.DataFrame:
    """[point, session, value]: per-point means `mean` (+ slope[c]), session noise `sd`, optional shared worlds."""
    rows = []
    worlds = rng.normal(0, world_sd, max(sizes.values()))
    for c, n in sizes.items():
        for s in range(n):
            w = f"w{s}" if shared else f"{c}-w{s}"
            rows.append({"point": c, "session": w, "value": mean + (slope or {}).get(c, 0.0) + (worlds[s] if shared else rng.normal(0, world_sd)) + rng.normal(0, sd)})
    return pd.DataFrame(rows)


def _session_rows(spec: dict, rng=None, N: int = 20, epochs: int = 2, block: str = "topo") -> pd.DataFrame:
    """Session table rows from {(point, arm): [success rate per session]} with exact item counts (no noise)."""
    rows = []
    for (point, arm), rates in spec.items():
        for s, r in enumerate(rates):
            for e in range(epochs):
                k = int(round(r * N))
                o = [1] * k + [0] * (N - k)
                rows.append({"plan_cell": f"x.{block}.{point}", "block": block, "point": point, "arm": arm, "session": f"w{s}", "epoch": e + 1, "N": N, "items_solved": float(k), "n_items": float(N), "item_success": k / N, "outcomes": o, "overflow": False, "overflow_at": np.nan, "w_crossing_item": float(int(0.7 * N) + 1), "session_success": float(k == N), "cost_usd": 1.0 + 0.1 * s, "error": False})
    return pd.DataFrame(rows)


# ---------- the estimator ----------


def test_lincomb_reproduces_the_gates_cluster_t():
    rng = np.random.default_rng(1)
    v = _values(rng, {"a": 8, "b": 6, "c": 5}, mean=0.1, sd=0.2)
    st = gs.lincomb(v, {c: 1 / 3 for c in "abc"})
    tm = v.rename(columns={"point": "cell", "session": "world"}).assign(task=lambda d: d["world"], A=lambda d: d["value"], B=0.0).set_index(["cell", "world", "task"])[["A", "B"]]
    ct = cluster_t(tm, "A", "B", cells=("a", "b", "c"))
    assert st["est"] == pytest.approx(ct["est"]) and st["se"] == pytest.approx(ct["se"]) and st["df"] == pytest.approx(ct["df"])
    assert not st["shared"]


def test_shared_world_variance_and_effective_clusters():
    """The pairwise variance equals the ψ form for distinct worlds and for full balanced overlap, and stays unbiased
    with partial overlap; the df cap is G − 1 for balanced designs and G_eff − 1 < G − 1 when worlds contribute unequally
    (16 worlds at two points, 8 of them at a third: D-033's topology design)."""
    rng = np.random.default_rng(13)
    w3 = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    for sizes, shared in (({"a": 8, "b": 6, "c": 5}, False), ({"a": 8, "b": 8, "c": 8}, True)):
        v = _values(rng, sizes, sd=0.2, shared=shared, world_sd=0.3)
        _, _, D, a = gs._matrix(v, w3)
        M = ~np.isnan(D)
        _, V, *_, psi = gs._core(D, M, a)
        assert float(V) == pytest.approx(float((psi**2).sum()))
    balanced = gs.lincomb(_values(rng, {"a": 8, "b": 8, "c": 8}, sd=0.2, shared=True), w3)
    assert balanced["effective_clusters"] == pytest.approx(8, rel=0.25) and balanced["df"] <= 7
    # Partial overlap: Var(L) by simulation vs the mean estimate, and a smaller df cap.
    ests, Vs, dfs, geffs = [], [], [], []
    for _ in range(3000):
        v = _values(rng, {"a": 16, "b": 16, "c": 8}, sd=0.2, shared=True, world_sd=0.3)
        st = gs.lincomb(v, w3)
        ests.append(st["est"])
        Vs.append(st["se"] ** 2)
        dfs.append(st["df"])
        geffs.append(st["effective_clusters"])
    assert np.mean(Vs) == pytest.approx(np.var(ests), rel=0.06)
    assert 10 < np.mean(geffs) < 15.5 and np.mean(dfs) < 15


@pytest.mark.parametrize("shared", [False, True])
def test_interval_coverage_is_nominal(shared):
    """95% intervals of a pooled mean and of a slope on capability cover the truth about 95% of the time, with distinct
    worlds per point and with worlds shared across points (a strong world effect)."""
    rng = np.random.default_rng(7 if shared else 8)
    x = {"a": 0.80, "b": 0.86, "c": 0.90}
    true_slope = -2.0
    cover_mean = cover_slope = 0
    reps = 400
    for _ in range(reps):
        v = _values(rng, {"a": 8, "b": 8, "c": 5}, mean=0.3, sd=0.25, shared=shared, world_sd=0.4, slope={c: true_slope * (x[c] - 0.85) for c in x})
        m = gs.contrast(v, {c: 1 / 3 for c in x}, flip=False)
        truth = 0.3 + np.mean([true_slope * (x[c] - 0.85) for c in x])
        cover_mean += m["ci"][0] <= truth <= m["ci"][1]
        s = gs.slope_test(v, x, flip=False)
        cover_slope += s["ci"][0] <= true_slope <= s["ci"][1]
    assert 0.92 <= cover_mean / reps <= 0.985, cover_mean / reps
    assert 0.92 <= cover_slope / reps <= 0.99, cover_slope / reps


def test_exact_sign_flip_floor_and_symmetry():
    v = pd.DataFrame({"point": "a", "session": [f"s{i}" for i in range(4)], "value": [0.3, 0.2, 0.25, 0.4]})
    f = gs.flip_test(v, {"a": 1.0})
    assert f["exact"] and f["flips"] == 16 and f["p"] == pytest.approx(1 / 16) and f["min_p"] == pytest.approx(1 / 16)
    assert gs.min_flip_p(4) == 0.0625 and gs.min_flip_p(6) < 0.025 and gs.min_sessions_for(0.025) == 6 and gs.min_sessions_for(0.0125) == 7
    c = gs.contrast(v, {"a": 1.0})
    assert c["flip_reachable"] is False and c["p_t"] < 0.025  # the t-test can reach α at 4 sessions, the flip cannot
    # Under a symmetric null the exact flip p is uniform: P(p <= 0.25) ≈ 0.25.
    rng = np.random.default_rng(3)
    ps = [gs.flip_test(pd.DataFrame({"point": "a", "session": [f"s{i}" for i in range(8)], "value": rng.normal(size=8)}), {"a": 1.0})["p"] for _ in range(300)]
    assert 0.18 < np.mean(np.array(ps) <= 0.25) < 0.32


def test_decisions_at_margin_boundaries():
    # Exactly at the null with no spread: never rejected; just past it: rejected.
    at = pd.DataFrame({"point": "a", "session": [f"s{i}" for i in range(6)], "value": [-0.2] * 6})
    assert not gs.contrast(at, {"a": 1.0}, null=-0.2, alternative="greater")["reject"]
    past = at.assign(value=[-0.19, -0.18, -0.19, -0.17, -0.18, -0.19])
    assert gs.contrast(past, {"a": 1.0}, null=-0.2, alternative="greater")["reject"]
    # G-H3a: isolation share exactly 0.5 is not supported, 0.6 is; recovery likewise against 0.8.
    def topo(m1: list, scm: list) -> pd.DataFrame:
        spec = {}
        for p in ("luna-high", "sol-high"):
            spec |= {(p, "S1"): [0.40] * 4, (p, "M2"): [0.80] * 4, (p, "M1"): m1, (p, "S-CM*"): scm}
        return _session_rows(spec, N=25)

    # Shares exactly at 0.5 and 0.8 on average (sessions spread symmetrically around them): not supported.
    h = gs.gh3(topo([0.56, 0.64, 0.56, 0.64], [0.68, 0.76, 0.68, 0.76]), boot=0)
    assert h["pre"]["reject"] and not h["h3a"]["reject"] and not h["h3b"]["recovery"]["reject"]
    assert h["isolation_share"]["est"] == pytest.approx(0.5) and h["recovery_share"]["est"] == pytest.approx(0.8)
    # Shares 0.65 and 0.9 with the same spread: supported, in sequence.
    h = gs.gh3(topo([0.64, 0.68, 0.64, 0.68], [0.76, 0.80, 0.76, 0.80]), boot=0)
    assert h["h3a"]["claim"] and h["h3b"]["recovery"]["reject"]
    # TOST: an estimate inside the margin with a tight interval is equivalent; one at the margin is not.
    x = {"p1": 0.7, "p2": 0.8, "p3": 0.9}
    flat = pd.DataFrame([{"point": c, "session": f"{c}{i}", "value": 0.5 + 0.001 * (i % 3)} for c in x for i in range(6)])
    lo, hi = gs.contrast(flat, {c: a * 0.2 for c, a in gs.slope_weights({c: 6 for c in x}, x).items()}, null=-0.1), None
    assert lo["reject"]
    tilted = flat.assign(value=flat["value"] + flat["point"].map({"p1": 0.1, "p2": 0.0, "p3": -0.1}))  # change over span = −0.2
    st = gs.slope_test(tilted, x, null=-0.2, alternative="greater", flip=False)
    assert st["est_span"] == pytest.approx(-0.2, abs=1e-3) and not st["reject"]


def test_ratio_handles_small_and_negative_denominators():
    rng = np.random.default_rng(2)

    def frames(num_mean, den_mean, n=8, sd=0.05):
        s = [f"s{i}" for i in range(n)]
        num = pd.DataFrame({"point": "a", "session": s, "value": rng.normal(num_mean, sd, n)})
        den = pd.DataFrame({"point": "a", "session": s, "value": rng.normal(den_mean, sd, n)})
        return num, den

    r = gs.ratio(*frames(0.12, 0.20), boot=500)
    assert r["testable"] and r["fieller_bounded"] and r["den_significant"] and r["ci_fieller"][0] < 0.6 < r["ci_fieller"][1]
    tiny = gs.ratio(*frames(0.01, 0.01, sd=0.001), boot=0)
    assert tiny["est"] is None and "no headroom" in tiny["reason"]
    neg = gs.ratio(*frames(0.05, -0.10), boot=0)
    assert neg["est"] is None and not neg["testable"]
    weak = gs.ratio(*frames(0.03, 0.04, sd=0.15), boot=500)
    assert weak["est"] is not None and weak["den_significant"] is False and weak["fieller_bounded"] is False and "unbounded" in weak["reason"]
    # Fieller coverage at a healthy denominator.
    cover = 0
    for _ in range(300):
        rr = gs.ratio(*frames(0.12, 0.20, n=8, sd=0.06), boot=0)
        cover += rr["fieller_bounded"] and rr["ci_fieller"][0] <= 0.6 <= rr["ci_fieller"][1]
    assert cover / 300 >= 0.92


def test_headroom_per_point_and_pseudo_values():
    spec = {}
    for p, (lo, hi, x) in {"a": (0.4, 0.8, 0.6), "b": (0.5, 0.9, 0.7)}.items():
        spec |= {(p, "CM0"): [lo] * 5, (p, "O-state"): [hi] * 5, (p, "CM-sum"): [x] * 5}
    s = _session_rows(spec, block="cm", N=40)
    h = gs.headroom(s, "CM-sum", boot=0)
    assert h["points"]["a"]["est"] == pytest.approx(0.5) and h["points"]["b"]["est"] == pytest.approx(0.5)
    wide = gs.session_values(s, ["CM-sum", "O-state", "CM0"])
    v, info = gs.ratio_pseudo(gs.long_values(wide, {"CM-sum": 1, "CM0": -1}), gs.long_values(wide, {"O-state": 1, "CM0": -1}))
    assert v.groupby("point")["value"].mean().to_dict() == pytest.approx({"a": 0.5, "b": 0.5})


def test_gh1_recovers_the_slope_sign_and_the_reference_matters():
    """A gap that shrinks with capability is detected; one that grows is not; the S1 reference's gap carries CM0's
    overflow, so its slope differs from S-CM*'s on the same data."""
    rng = np.random.default_rng(4)
    pts = {"luna-high": 0.70, "sol-high": 0.80, "astra-high": 0.90}

    def study(direction: float):
        spec = {}
        for p, x in pts.items():
            base = 0.55
            m2 = base + 0.25 + direction * (x - 0.8) * 2.0
            spec |= {(p, "S1"): list(np.clip(rng.normal(0.4, 0.03, 8), 0, 1)), (p, "M2"): list(np.clip(rng.normal(m2, 0.03, 8), 0, 1)), (p, "S-CM*"): list(np.clip(rng.normal(base, 0.03, 8), 0, 1))}
        return _session_rows(spec)

    down = gs.gh1(study(-1.0), pts, reference="S-CM*", reps=500)
    assert down["est"] < 0 and down["reject"] and down["testable"]
    up = gs.gh1(study(+1.0), pts, reference="S-CM*", reps=500)
    assert up["est"] > 0 and not up["reject"]
    assert set(down["gaps"]) == set(pts)


def test_s1_pre_uses_only_items_before_s1s_overflow():
    rows = _session_rows({("a", "S1"): [0.5, 0.5], ("a", "M2"): [0.9, 0.9], ("b", "S1"): [0.5, 0.5], ("b", "M2"): [0.9, 0.9]})
    rows.loc[rows["arm"] == "S1", "overflow_at"] = 11.0  # items 1..10 survive
    rows.loc[rows["arm"] == "S1", "outcomes"] = pd.Series([[1] * 10 + [0] * 10] * int((rows["arm"] == "S1").sum()), index=rows.index[rows["arm"] == "S1"])
    wide = gs.session_values(rows, ["M2", "S1"], pre_overflow_of="S1", length=None)
    assert (wide["S1"] == 1.0).all()  # S1 solved all of its first 10
    assert np.allclose(wide["M2"], 1.0)  # M2's first 10 of [1]*18 + [0]*2
    # An S1 that overflowed at item 1 leaves nothing to compare: the session drops out, nothing raises.
    rows.loc[rows["arm"] == "S1", "overflow_at"] = 1.0
    assert gs.session_values(rows, ["M2", "S1"], pre_overflow_of="S1", length=None).empty
    assert not gs.gh1(rows, {"a": 0.8, "b": 0.9}, reference="S1-pre")["testable"]


def test_robust_to_missing_arms_points_and_overflow_only_sessions():
    pts = {"luna-high": 0.8, "sol-high": 0.86}
    no_scm = _session_rows({(p, a): [0.5, 0.6, 0.55] for p in pts for a in ("S1", "M2")})
    r = gs.gh1(no_scm, pts, reference="S-CM*")
    assert not r["testable"] and "slope needs" in r["reason"]
    one_point = _session_rows({("luna-high", a): [0.5, 0.6, 0.55] for a in ("S1", "M2", "S-CM*")})
    assert not gs.gh1(one_point, pts)["testable"]
    assert not gs.gh1(_session_rows({(p, a): [0.5, 0.6] for p in pts for a in ("S1", "M2", "S-CM*")}), {})["testable"]  # no capability
    assert not gs.gh3(no_scm)["testable"]
    # Every CM0 session overflowed at item 1 (all failures): the gap and R are still computed.
    cm = _session_rows({(p, a): ([0.0] * 4 if a == "CM0" else [0.8, 0.7, 0.75, 0.8]) for p in pts for a in ("CM0", "O-state", "CM-sum")}, block="cm", N=40)
    cm.loc[cm["arm"] == "CM0", ["overflow", "overflow_at"]] = [True, 1.0]
    g = gs.gap_iut(cm)
    assert g["all_positive"] and set(g["points"]) == set(pts)
    assert gs.headroom(cm, "CM-sum", boot=0)["points"]["luna-high"]["est"] == pytest.approx(1.0)
    # Nothing at all.
    empty = pd.DataFrame()
    assert not gs.gap_iut(empty)["testable"] and not gs.gh3(empty)["testable"] and not gs.tost(empty, pts, "CM-sum")["testable"]
    assert gs.outcome_table(empty) == [] and gs.probe_table(empty) == [] and gs.taxonomy_table(empty) == [] and gs.degradation(empty) == []


def test_tost_without_headroom_is_not_testable_and_names_the_point():
    pts = {"a": 0.7, "b": 0.8, "c": 0.9}
    spec = {}
    for p in pts:
        hi = 0.41 if p == "c" else 0.8  # no headroom at c
        spec |= {(p, "CM0"): [0.4] * 4, (p, "O-state"): [hi] * 4, (p, "CM-sum"): [0.6] * 4}
    r = gs.tost(_session_rows(spec, block="cm", N=40), pts, "CM-sum")
    assert r["per_point"]["c"]["R"] is None and r["points_used"] == ["a", "b"] and "no headroom" in (r["reason"] or "")


def _items_frame(rng, sessions=6, N=30, slope=-1.0) -> pd.DataFrame:
    rows = []
    for arm in ("CM0", "O-state"):
        for s in range(sessions):
            for e in (1, 2):
                for i in range(1, N + 1):
                    vt = 2000 + 800 * i + rng.normal(0, 200)
                    lv = math.log(vt)
                    p = 1 / (1 + math.exp(-(1.0 + (slope if arm == "CM0" else 0.0) * (lv - 9.5))))
                    rows.append({"block": "cm", "point": "luna-high", "arm": arm, "plan_cell": "g.cm.luna-high", "session": f"w{s}", "epoch": e, "position": i, "rel_position": i / N, "view_tokens": vt, "overflow": False, "success": float(rng.random() < p), "item": f"w{s}#{i:03d}", "dependency": i % 3 == 0})
    return pd.DataFrame(rows)


def test_degradation_slopes_and_glmm():
    rng = np.random.default_rng(5)
    items = _items_frame(rng, slope=-1.5)
    deg = {r["arm"]: r for r in gs.degradation(items, block="cm")}
    assert deg["CM0"]["status"] == "ok" and deg["CM0"]["log_view"]["log_view"]["est"] < 0
    assert deg["CM0"]["log_view"]["log_view"]["ci"][1] < 0 < deg["O-state"]["log_view"]["log_view"]["ci"][1]
    one = items.assign(success=1.0)
    assert gs.degradation(one, block="cm")[0]["status"] == "too few items or no variation"
    two = pd.concat([items, items.assign(point="sol-high", session=lambda d: "v" + d["session"])])
    m = gs.glmm(two, {"luna-high": 0.8, "sol-high": 0.86}, block="cm", reference="CM0")
    assert m["status"] in ("ok", "non-finite") and "fixed" in m
    assert gs.glmm(two, {}, block="cm")["status"] == "skipped"
    assert gs.glmm(two.assign(item=None), {"luna-high": 0.8, "sol-high": 0.86}, block="cm")["status"] in ("ok", "non-finite", "failed")


def _topo_h3() -> pd.DataFrame:
    """Two points, 4 sessions each; isolation share 0.65 and recovery 0.9 on average (as in the margin test)."""
    m1, scm = [0.64, 0.68, 0.64, 0.68], [0.76, 0.80, 0.76, 0.80]
    spec = {}
    for p in ("luna-high", "sol-high"):
        spec |= {(p, "S1"): [0.40] * 4, (p, "M2"): [0.80] * 4, (p, "M1"): m1, (p, "S-CM*"): scm}
    s = _session_rows(spec, N=25)
    s.loc[s["arm"] == "S-CM*", "cost_usd"] *= 0.4  # cost per solved item about 0.4 of M2's
    return s


def test_s4_calibrated_level_and_minimum_sessions():
    """S-4: the confirmatory t-tests decide at the calibrated level (the flip at α), with the matching one-sided bound;
    a planned point below its minimum sessions is missing (left out of the decision, the family INCOMPLETE)."""
    v = pd.DataFrame({"point": "a", "session": [f"s{i}" for i in range(6)], "value": [0.30, -0.05, 0.22, 0.02, 0.15, 0.10]})
    p = gs.contrast(v, {"a": 1.0}, flip=False)["p_t"]
    assert 0 < p < 0.5
    above, below = gs.contrast(v, {"a": 1.0}, flip=False, level=p * 1.01), gs.contrast(v, {"a": 1.0}, flip=False, level=p * 0.99)
    assert above["reject"] and not below["reject"] and below["level"] == pytest.approx(p * 0.99)
    assert above["ci"][0] == pytest.approx(0.0, abs=1e-3)  # the (1 − 2·level) interval's lower end sits at the null when p = level
    assert gh.T_LEVEL == gh.t_level(gh.ALPHA) <= gh.ALPHA
    assert gh.t_level(0.05) == pytest.approx(0.05 * gh.T_LEVEL / gh.ALPHA)
    # The minimum per planned point: from the plan's sessions, never above them.
    assert all(gh.min_sessions(n) <= n for n in (2, 4, 5, 6, 8, 10, 16)) and gh.min_sessions(None) is None
    # The planned sizes' minimums (topology Luna 16, Sol 8; CM Luna 10, Sol 6, Astra 4), validated by simulation.
    assert {n: gh.min_sessions(n) for n in (16, 8, 10, 6, 4)} == {16: 10, 8: 5, 10: 6, 6: 4, 4: 3}
    # G-H2a: a planned point with fewer sessions than its minimum is short: INCOMPLETE, whatever it shows.
    s = _session_rows({("luna-high", "CM0"): [0.4, 0.5, 0.45], ("luna-high", "O-state"): [0.9, 0.95, 0.92], ("sol-high", "CM0"): [0.4, 0.5, 0.45, 0.5], ("sol-high", "O-state"): [0.9, 0.95, 0.92, 0.9]}, block="cm", N=40)
    g = gs.gap_iut(s, planned_points=["luna-high", "sol-high"], min_sessions={"luna-high": 4, "sol-high": 4}, reps=200)
    assert g["incomplete"] and g["short_points"] == {"luna-high": {"sessions": 3, "minimum": 4}} and g["missing_points"] == ["luna-high"]
    assert g["points"]["luna-high"]["reject"] and g["present_positive"] and not g["all_positive"] and g["level"] == gh.T_LEVEL
    assert gs.gap_iut(s, planned_points=["luna-high", "sol-high"], min_sessions={"luna-high": 3, "sol-high": 4}, reps=200)["all_positive"]
    # G-H3: the short point leaves the pool.
    h = gs.gh3(_topo_h3(), planned_points=["luna-high", "sol-high"], min_sessions={"luna-high": 4, "sol-high": 6}, boot=0, reps=200)
    assert h["incomplete"] and h["points"] == ["luna-high"] and h["short_points"] == {"sol-high": {"sessions": 4, "minimum": 6}} and h["level"] == gh.T_LEVEL
    d = gr.g_report(None, _topo_h3(), {}, reps=200, boot=0, glmm=False, planned={"G-H2a": [], "G-H3": ["luna-high", "sol-high"]}, min_sessions={"G-H3": {"sol-high": 6}})
    dec = {r["id"]: r for r in d["decisions"]}
    assert dec["G-H3-pre"]["decision"] == "INCOMPLETE" and dec["G-H3-pre"]["decision_on_present_points"] == "SUPPORTED" and dec["G-H3-pre"]["short_points"] == {"sol-high": {"sessions": 4, "minimum": 6}}
    assert any("minimum sessions per planned point" in c for c in d["caveats"]) and "below the minimum sessions: sol-high (4 of 6)" in gr.render(d)


def test_s9_sections_are_isolated_a_nan_cost_is_not_testable_and_an_unknown_plan_is_incomplete(monkeypatch):
    """S-9: one section's exception is listed and leaves the decisions standing; a cost clause that cannot be computed
    makes G-H3b NOT_TESTABLE (not NOT_SUPPORTED); an unreadable plan makes the confirmatory rows INCOMPLETE."""
    planned = {"G-H2a": [], "G-H3": ["luna-high", "sol-high"]}
    ok = gr.g_report(None, _topo_h3(), {}, reps=200, boot=0, glmm=False, planned=planned)
    assert {r["id"]: r["decision"] for r in ok["decisions"]}["G-H3b"] == "SUPPORTED" and ok["errors"] == []
    # A NaN cost in one session: the cost clause cannot be computed.
    s = _topo_h3()
    s.loc[(s["arm"] == "S-CM*") & (s["point"] == "sol-high") & (s["session"] == "w1"), "cost_usd"] = np.nan
    d = gr.g_report(None, s, {}, reps=200, boot=0, glmm=False, planned=planned)
    dec = {r["id"]: r for r in d["decisions"]}
    assert dec["G-H3a"]["decision"] == "SUPPORTED" and dec["G-H3b"]["decision"] == "NOT_TESTABLE" and "cost_usd missing" in dec["G-H3b"]["reason"]
    assert d["gh3"]["h3b"]["testable"] is False and not d["gh3"]["h3b"]["claim"]

    # Failing sections: listed, the rest (and the decisions) stand, and the report renders.
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(gs, "cost_table", boom)
    monkeypatch.setattr(gs, "gh1", boom)
    d = gr.g_report(None, _topo_h3(), {}, reps=200, boot=0, glmm=False, planned=planned)
    assert {e["section"] for e in d["errors"]} == {"costs", "G-H1"} and all("RuntimeError: boom" in e["error"] for e in d["errors"])
    dec = {r["id"]: r["decision"] for r in d["decisions"]}
    assert dec["G-H3-pre"] == dec["G-H3a"] == dec["G-H3b"] == "SUPPORTED" and dec["G-H1"] == "descriptive"
    md = gr.render(d)
    assert "## Section errors" in md and "## Decisions" in md and "**SUPPORTED**" in md and any("section(s) failed" in c for c in d["caveats"])
    json.dumps(d)
    monkeypatch.setattr(gr, "decisions", boom)
    d = gr.g_report(None, _topo_h3(), {}, reps=200, boot=0, glmm=False, planned=planned)
    assert all(r["decision"] in ("NOT_TESTABLE", "descriptive") for r in d["decisions"]) and "decisions" in {e["section"] for e in d["errors"]}
    monkeypatch.undo()
    # The plan unreadable: INCOMPLETE, with the decision on the points present kept.
    d = gr.g_report(None, _topo_h3(), {}, reps=200, boot=0, glmm=False, planned=gr.PLANNED_UNKNOWN)
    dec = {r["id"]: r for r in d["decisions"]}
    assert d["planned_unknown"] and all(dec[h]["decision"] == "INCOMPLETE" for h in ("G-H2a", "G-H3-pre", "G-H3a", "G-H3b"))
    assert dec["G-H3-pre"]["decision_on_present_points"] == "SUPPORTED" and any("planned points unknown" in p for p in d["header"]["problems"])
    assert "planned points unknown (run plan unreadable)" in gr.render(d)
    (pl, mins), problem = gr.planned_points_and_sessions(plan=None)
    assert problem is None and pl["G-H3"] == ["luna-low", "luna-high", "sol-high"] and set(mins["G-H2a"]) == set(pl["G-H2a"])


# ---------- hypotheses and report ----------


def test_hypothesis_table_is_complete():
    ids = [h.id for h in gh.HYPOTHESES]
    assert len(ids) == len(set(ids))
    conf = gh.confirmatory()
    assert {h.id for h in conf} == {"G-H2a", "G-H3-pre", "G-H3a", "G-H3b"}  # D-033: G-H1 and G-H2b descriptive
    assert gh.by_id("G-H1").role == gh.by_id("G-H2b").role == "descriptive" and gh.DEFAULT_REFERENCE == "S-CM*"
    assert all(h.family and h.test and h.estimand and h.alpha == gh.ALPHA for h in conf)
    assert [h.step for h in conf if h.family == "G-H3"] == [1, 2, 3]
    assert "g.topo.luna-low" in gh.by_id("G-H3a").cells and "g.topo.luna-low" in gh.TOPO_CELLS
    md = gh.markdown()
    assert md.count("\n") == len(gh.HYPOTHESES) + 2 and "G-H2b" in md
    json.dumps(gh.table())


def test_report_on_simulated_data_is_json_and_renders():
    rng = np.random.default_rng(6)
    sc = gp.scenarios()
    var = gp.Variance()
    sessions = pd.concat([gp.simulate_block(sc["topo-h3"], gp.PLANNED_TOPO, var, rng), gp.simulate_block(sc["cm-equiv"], gp.PLANNED_CM, var, rng)], ignore_index=True)
    cap = gp.measured_capability(sc["cm-equiv"], var, rng, noise=False)
    d = gr.g_report(gp.to_items(sessions), sessions, cap, reps=500, boot=200, glmm=False)  # items without view tokens
    json.dumps(d)
    assert d["gh2"]["degradation"][0]["status"].startswith("item table lacks")
    assert gs.glmm(gp.to_items(sessions[sessions["block"] == "topo"]).head(0), cap)["status"] == "no items"
    dec = {r["id"]: r["decision"] for r in d["decisions"]}
    assert dec["G-H2a"] == "SUPPORTED" and dec["G-H3-pre"] == "SUPPORTED"
    assert dec["G-H1"] == dec["G-H2b[CM-sum]"] == dec["G-H2b[CM-todo]"] == "descriptive"
    g1 = next(r for r in d["decisions"] if r["id"] == "G-H1")
    assert g1["reference"] == "S-CM*" and g1["ci_span"][0] < g1["estimate_span"] < g1["ci_span"][1] and g1["sensitivity_S1_pre"]["ci_span"][0] is not None
    assert "p" not in g1 and "p" not in next(r for r in d["decisions"] if r["id"] == "G-H2b[CM-sum]")
    # D-033's design: G-H1 with S-CM* spans the three points where it ran; G-H3 pools them.
    assert d["gh1"]["primary"]["points_used"] == ["luna-low", "luna-high", "sol-high"] and d["gh3"]["points"] == ["luna-low", "luna-high", "sol-high"]
    assert d["planned_points"] == {"G-H2a": ["luna-low", "luna-high", "sol-high", "astra-high"], "G-H3": ["luna-low", "luna-high", "sol-high"]}
    # D-043's sensitivity: the same estimates on the tier rank (equally spaced), carried in the descriptive rows.
    tier = d["gh1"]["tier_order"]["S-CM*"]
    assert tier["points_used"] == ["luna-low", "luna-high", "sol-high"] and tier["span"] == 2.0 and tier["ci_span"][0] < tier["est_span"] < tier["ci_span"][1]
    assert set(g1["tier_order"]) == {"S-CM*", "S1-pre"} and g1["tier_order"]["S1-pre"]["points"] == ["luna-low", "luna-high", "sol-high"]  # Astra topology off (D-051)
    h2b = next(r for r in d["decisions"] if r["id"] == "G-H2b[CM-sum]")
    assert h2b["tier_order"]["ci"][0] < h2b["tier_order"]["estimate"] < h2b["tier_order"]["ci"][1]
    assert d["gh2"]["tost_tier_order"]["CM-sum"]["span"] == 3.0  # Luna-low to Astra: three tier steps
    assert not d["gh2"]["gap"]["incomplete"] and not d["gh3"]["incomplete"]
    assert set(d["gh1"]["references"]) == set(gh.REFERENCES)
    lo, hi = d["gh1"]["primary"]["boot_ci_span"]
    assert lo < hi and d["gh1"]["primary"]["ci_span"][0] < d["gh1"]["primary"]["est_span"] < d["gh1"]["primary"]["ci_span"][1]
    cost = d["gh3"]["costs"]["cost_usd"]
    assert cost["ratio_boot_ci"][0] < cost["ratio"] < cost["ratio_boot_ci"][1] and cost["ratio_ci"][0] < cost["ratio"] < cost["ratio_ci"][1]
    assert not any("astra-high" in c and "sign-flip" in c for c in d["caveats"]), "6 Astra sessions reach the flip level (D-052)"
    md = gr.render(d)
    assert "## Decisions" in md and "G-H3b" in md and "Descriptive estimates (D-033" in md and "G-H1 (M2 − S-CM*)" in md
    assert "G-H1 (M2 − S-CM*), tier order (D-043)" in md and "G-H2b[CM-sum], tier order (D-043)" in md and "CM-todo, tier order (D-043)" in md


def test_a_missing_planned_point_makes_the_confirmatory_rows_incomplete():
    """PREREGISTRATION_G.md restricts each claim to the planned points: a planned point without data labels G-H2a and the
    G-H3 sequence INCOMPLETE (the missing points named), never SUPPORTED, and keeps the estimates."""
    rng = np.random.default_rng(14)
    sc = gp.scenarios()
    var = gp.Variance()
    topo = gp.simulate_block(sc["topo-h3"], gp.PLANNED_TOPO, var, rng)
    cm = gp.simulate_block(sc["cm-equiv"], gp.PLANNED_CM, var, rng)
    sessions = pd.concat([topo[topo["point"] != "sol-high"], cm[~((cm["point"] == "astra-high") & (cm["arm"] == "CM0"))]], ignore_index=True)
    d = gr.g_report(None, sessions, gp.measured_capability(sc["cm-equiv"], var, rng, noise=False), reps=300, boot=0, glmm=False)
    dec = {r["id"]: r for r in d["decisions"]}
    assert dec["G-H2a"]["decision"] == "INCOMPLETE" and dec["G-H2a"]["missing_points"] == ["astra-high"] and dec["G-H2a"]["present_positive"]
    assert set(d["gh2"]["gap"]["points"]) == {"luna-low", "luna-high", "sol-high"}  # the per-point results present are kept
    for hid in ("G-H3-pre", "G-H3a", "G-H3b"):
        assert dec[hid]["decision"] == "INCOMPLETE" and dec[hid]["missing_points"] == ["sol-high"], hid
    assert dec["G-H3-pre"]["decision_on_present_points"] == "SUPPORTED" and dec["G-H3-pre"]["estimate"] > 0 and d["gh3"]["points"] == ["luna-low", "luna-high"]
    assert sum("INCOMPLETE" in c for c in d["caveats"]) == 2
    md = gr.render(d)
    assert "**INCOMPLETE**" in md and "planned point(s) without data: sol-high" in md and "planned point(s) without data: astra-high" in md
    # Explicit planned points also override the plan; without any, the old behaviour (the points present) remains.
    g = gs.gap_iut(sessions, planned_points=["luna-high", "mars"], reps=200)
    assert g["incomplete"] and g["missing_points"] == ["mars"] and not g["all_positive"] and g["unplanned_points"] == ["luna-low", "sol-high"]
    assert gs.gap_iut(sessions, reps=200)["all_positive"] and not gs.gap_iut(sessions, reps=200)["incomplete"]
    assert not gs.gh3(sessions, boot=0, reps=200)["incomplete"]


def test_report_on_partial_data_never_raises():
    # Only the context-management block at one point, no capability, no topology arms.
    s = _session_rows({("luna-high", a): [0.4, 0.5, 0.45] for a in ("CM0", "O-state")}, block="cm", N=40)
    d = gr.g_report(None, s, {}, reps=200, boot=0, glmm=False)
    dec = {r["id"]: r for r in d["decisions"]}
    assert dec["G-H3-pre"]["decision"] == "NOT_TESTABLE" and dec["G-H1"]["decision"] == dec["G-H2b[CM-sum]"]["decision"] == "descriptive"
    assert dec["G-H1"]["estimate_span"] is None and dec["G-H1"]["reason"] and dec["G-H2b[CM-sum]"]["reason"]  # why there is no estimate
    assert d["capability"]["missing"] == ["luna-high"]
    gr.render(d)
    d0 = gr.g_report(pd.DataFrame(), pd.DataFrame(), None, reps=200, boot=0)
    json.dumps(d0)
    df = gr.g_report(None, s, {}, reps=200, boot=0, glmm=False, primary="flip")  # decided by the sign-flip instead
    assert df["gh2"]["gap"]["points"]["luna-high"]["p_flip"] is not None
    assert all(r["decision"] in ("NOT_TESTABLE", "descriptive") for r in d0["decisions"])
    gr.render(d0)


# ---------- the power simulation ----------


def test_simulated_sessions_match_the_item_aggregation():
    """The simulation's session table equals what g_load builds from item rows, so power runs exercise the real path."""
    rng = np.random.default_rng(9)
    sim = gp.simulate_block(gp.scenarios()["topo-converge"], gp.PLANNED_TOPO, gp.Variance(), rng)
    agg = sessions_from_items(gp.to_items(sim))
    m = sim.merge(agg, on=["plan_cell", "arm", "session", "epoch"], suffixes=("", "_agg"))
    assert len(m) == len(sim)
    assert (m["items_solved"] == m["items_solved_agg"]).all() and (m["n_items"] == m["n_items_agg"]).all()
    assert (m["outcomes"] == m["outcomes_agg"]).all()


def test_mixture_arms_hit_their_target_shares():
    """The generative device behind every null-at-margin: E[(mix − low) / (high − low)] equals the scenario's share."""
    rng = np.random.default_rng(10)
    sc = gp.scenarios()["topo-null"]
    big = (gp.Cell("luna-high", 400, 2, 20, gs.TOPO_ARMS, "n20"),)
    s = gp.simulate_block(sc, big, gp.Variance(), rng)
    m = s.groupby("arm")["item_success"].mean()
    assert (m["M1"] - m["S1"]) / (m["M2"] - m["S1"]) == pytest.approx(gh.ISOLATION_SHARE, abs=0.03)
    assert (m["S-CM*"] - m["S1"]) / (m["M2"] - m["S1"]) == pytest.approx(gh.RECOVERY_SHARE, abs=0.03)
    cps = (s[s["arm"] == "S-CM*"]["cost_usd"].sum() / s[s["arm"] == "S-CM*"]["items_solved"].sum()) / (s[s["arm"] == "M2"]["cost_usd"].sum() / s[s["arm"] == "M2"]["items_solved"].sum())
    assert cps == pytest.approx(gh.COST_RATIO, rel=0.04)


def test_type_one_error_at_the_null_through_the_real_path():
    """At the planned (D-033) sizes, every confirmatory topology test at its null rejects at most about α (MC
    tolerance), and the descriptive G-H1 intervals cover the true 0 about 95% of the time."""
    tol = lambda n: 3 * math.sqrt(0.025 * 0.975 / n)  # noqa: E731
    r = gp.run(gp.scenarios()["topo-null"], 150, seed=11, reps=199)
    for test in ("G-H3a", "G-H3b.recovery", "G-H3b.cost"):
        assert r["tests"][test]["t"] <= 0.025 + tol(150), (test, r["tests"][test])
    for ref in ("S-CM*", "S1", "S1-pre"):
        assert r["tests"][f"G-H1[{ref}]"]["covers_0"] >= 0.95 - 3 * math.sqrt(0.05 * 0.95 / 150) and "t" not in r["tests"][f"G-H1[{ref}]"]
    g = gp.run(gp.scenarios()["cm-null-gap"], 100, seed=12, reps=199)
    for c in ("luna-low", "luna-high", "sol-high", "astra-high"):
        assert g["tests"][f"G-H2a[{c}]"]["t"] <= 0.025 + tol(100)
    assert g["tests"]["G-H2a[astra-high]"]["flip"] == 0.0  # 4 sessions: the exact flip cannot reach α


def test_planned_designs_follow_d033_and_the_run_plan():
    topo = {c.point: c for c in gp.PLANNED_TOPO}
    assert {p: (c.sessions, c.epochs, c.N) for p, c in topo.items()} == {"luna-low": (16, 2, 20), "luna-high": (16, 2, 20), "sol-high": (12, 1, 20)}  # D-051, D-052
    assert set(topo["luna-low"].arms) == set(gs.TOPO_ARMS) and gp.ASTRA_TOPO.arms == ("S1", "M2")
    from ape.budget import load_plan

    plan = load_plan()
    if "g.topo.luna-low" not in {c.id for c in plan.cells}:
        pytest.skip("config/run_plan.yaml does not carry D-033's g.topo.luna-low cell yet")
    for cell in (*gp.PLANNED_TOPO, *gp.PLANNED_CM):
        spec = plan.cell(cell.plan_cell).spec
        assert (int(spec["sessions"]), int(spec["epochs"]), int(spec["N"])) == (cell.sessions, cell.epochs, cell.N), cell.plan_cell
        assert set(cell.arms) <= set(spec["arms"]), cell.plan_cell


def test_flip_floor_flags_astra():
    """D-052's 6 Astra sessions bring the exact sign-flip within reach (1/64 < 0.025); the old 4 could not (1/16)."""
    rows = {r["test"]: r for r in gp.flip_floor()}
    assert rows["G-H2a [astra-high]"]["reachable"] and rows["G-H2a [astra-high]"]["min_p"] == 1 / 64
    old = [c if c.point != "astra-high" or c.N <= 10 else type(c)(**{**c.__dict__, "sessions": 4}) for c in gp.PLANNED_CM]
    four = {r["test"]: r for r in gp.flip_floor(cm=old)}["G-H2a [astra-high]"]
    assert four["reachable"] is False and four["min_p"] == 0.0625 and "≥ 6" in four["fix"]
    assert rows["G-H2a [sol-high]"]["reachable"] and not any(r.startswith(("G-H1", "G-H2b")) for r in rows)  # descriptive
    assert rows["G-H3 (pre, a, b)"]["clusters"] == 16 and {r["test"]: r for r in gp.flip_floor(shared=False)}["G-H3 (pre, a, b)"]["clusters"] == 44
