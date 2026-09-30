"""Gate statistics (module 8): bootstrap coverage on synthetic data with known Δ, and decision boundaries."""

import numpy as np
import pandas as pd

from ape.analysis.gate_stats import GATE_CELLS, cluster_bootstrap, decide, invariants, pooled_delta, sign_flip_p

PC_OK = {f"PC{i}": True for i in range(1, 7)}


def synth(true_delta: float, worlds: int = 6, tasks: int = 12, epochs: int = 3, sigma_w: float = 0.6, sigma_g: float = 0.3, seed: int = 0, base: float = 0.6, extra_arms: dict | None = None) -> pd.DataFrame:
    """Task-level epoch means for APG-s, LGR* (and extra arms) with world and world×arm effects."""
    rng = np.random.default_rng(seed)
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731
    arms = {"LGR*": 0.0, "APG-s": None, **(extra_arms or {})}
    rows = []
    for c in GATE_CELLS:
        for w in range(worlds):
            u = rng.normal(0, sigma_w)
            g = {a: rng.normal(0, sigma_g) for a in arms}
            for t in range(tasks):
                v = rng.normal(0, 1.0)
                p_lgr = sig(logit(base) + u + v + g["LGR*"])
                row = {"cell": c, "world": f"{c}-w{w}", "task": f"{c}-w{w}-t{t}", "LGR*": rng.binomial(epochs, p_lgr) / epochs}
                p_apg = np.clip(p_lgr + true_delta, 0.001, 0.999)
                row["APG-s"] = rng.binomial(epochs, sig(logit(p_apg) + g["APG-s"] - g["LGR*"])) / epochs
                for a, shift in (extra_arms or {}).items():
                    row[a] = rng.binomial(epochs, np.clip(p_lgr + shift, 0.001, 0.999)) / epochs
                rows.append(row)
    return pd.DataFrame(rows).set_index(["cell", "world", "task"])


def test_bootstrap_ci_covers_true_delta_about_95_percent():
    covered, trials = 0, 120
    for s in range(trials):
        tm = synth(-0.02, seed=s, sigma_g=0.0)
        boot = cluster_bootstrap(tm, "APG-s", "LGR*", reps=1000, seed=s)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        # Truth is the population Δ; the synthetic effect is approximately -0.02 on the probability scale.
        covered += lo <= -0.02 <= hi
    assert 0.88 <= covered / trials <= 0.995


def test_sign_flip_p_is_small_for_clear_effect_and_large_for_none():
    rng = np.random.default_rng(1)
    assert sign_flip_p(rng.normal(0.3, 1.0, 400)) < 0.01
    assert sign_flip_p(rng.normal(0.0, 1.0, 400)) > 0.05


def test_decision_go_for_equal_systems_with_many_worlds():
    tm = synth(0.0, worlds=40, tasks=20, seed=3, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", PC_OK, cost_ratio=1.0, reps=2000)
    assert d.verdict == "GO", d.reasons
    assert d.ci[0] > -0.05


def test_decision_no_go_when_apg_clearly_worse():
    tm = synth(-0.15, worlds=40, tasks=20, seed=4, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", PC_OK, reps=2000)
    assert d.verdict == "NO_GO" and d.ci[1] < 0


def _decide_with_boot(monkeypatch, boot_values, cell_f7_1000=0.0):
    """Drive decide() with a fixed bootstrap distribution to test its branches exactly."""
    import ape.analysis.gate_stats as gs

    tm = synth(0.0, worlds=6, tasks=4, seed=11, extra_arms={"S7": -0.4})
    monkeypatch.setattr(gs, "cluster_bootstrap", lambda *a, **k: np.asarray(boot_values, dtype=float))
    monkeypatch.setattr(gs, "cell_deltas", lambda *a, **k: pd.Series({"F7-10": 0.0, "F7-1000": cell_f7_1000, "F3-5": 0.0, "F3-60": 0.0}))
    return gs.decide(tm, "LGR*", PC_OK, reps=500)


def test_decision_boundaries(monkeypatch):
    assert _decide_with_boot(monkeypatch, np.linspace(-0.04, 0.02, 1001)).verdict == "GO"
    assert _decide_with_boot(monkeypatch, np.linspace(0.01, 0.05, 1001)).superiority
    assert _decide_with_boot(monkeypatch, np.linspace(-0.08, 0.03, 1001)).verdict == "INCONCLUSIVE"
    assert _decide_with_boot(monkeypatch, np.linspace(-0.12, -0.01, 1001)).verdict == "NO_GO"
    d = _decide_with_boot(monkeypatch, np.linspace(-0.04, 0.02, 1001), cell_f7_1000=-0.11)
    assert d.verdict == "NO_GO" and "F7-1000" in d.reasons[0]


def test_too_few_worlds_is_inconclusive():
    d = decide(synth(0.0, worlds=2, tasks=6, seed=5, extra_arms={"S7": -0.4}), "LGR*", PC_OK, reps=500)
    assert d.verdict == "INCONCLUSIVE" and "too few worlds" in d.reasons[0]


def test_precondition_failure_blocks_the_decision():
    tm = synth(0.0, seed=6, extra_arms={"S7": -0.4})
    d = decide(tm, "LGR*", {**PC_OK, "PC1": False}, reps=500)
    assert d.verdict == "PRECONDITION_FAIL" and "PC1" in d.reasons[0]


def test_cost_flag_and_placebo_rule():
    tm = synth(0.0, worlds=40, tasks=20, seed=7, extra_arms={"S7": -0.4})
    assert decide(tm, "LGR*", PC_OK, cost_ratio=2.5, reps=2000).verdict == "GO_WITH_COST_FLAG"
    tm_placebo = synth(0.0, worlds=40, tasks=20, seed=8, extra_arms={"S7": 0.0})
    d = decide(tm_placebo, "LGR*", PC_OK, reps=2000)
    assert d.verdict == "NO_GO" and any("placebo" in r for r in d.reasons)


def test_invariants_chain():
    tm = synth(0.0, worlds=10, tasks=10, seed=9, extra_arms={"S6": 0.3, "S5o": 0.1, "S7": -0.4})
    inv = invariants(tm)
    assert inv["pass"], inv
    bad = synth(0.0, worlds=10, tasks=10, seed=9, extra_arms={"S6": -0.3, "S5o": 0.1, "S7": -0.4})
    assert not invariants(bad)["pass"]


def test_pooled_delta_weights_cells_equally():
    tm = synth(0.0, seed=10)
    tm.loc[tm.index.get_level_values("cell") == "F7-10", "APG-s"] += 0.4
    assert abs(pooled_delta(tm, "APG-s", "LGR*") - (0.4 / 4 + pooled_delta(synth(0.0, seed=10), "APG-s", "LGR*"))) < 1e-9
