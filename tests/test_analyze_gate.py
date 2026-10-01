"""Gate decision report (FX-7): verdicts per delivery mode and combined, PC2-PC6 on crafted inputs, and
push and pull never pooled. The end-to-end report on an offline run is in tests/test_run_gate.py."""

import json

import pandas as pd
import pytest

from ape import analyze_gate as ag
from ape.analysis.gate_stats import GATE_CELLS, combine_modes, pipeline_miss, task_means

PC_OK = [{"id": f"PC{i}", "pass": True, "reason": None} for i in range(1, 7)]
WORLDS, TASKS, EPOCHS = 8, 8, 2
REPS = 2000
# Per-world Δ (APG* − LGR*) whose pooled CI spans both −5 pp and 0 (8 worlds per cell, 8 tasks per world).
NOISY = (0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.375, 0.125)


def _row(plan_cell, label, delivery, cell, w, t, e, success, group="selected", **extra) -> dict:
    return {
        "plan_cell": plan_cell, "group": group, "label": label, "arm": label, "run_arm": label, "delivery": delivery,
        "cell": cell, "world": f"{cell}-s{3000 + w}", "task": f"{cell}-s{3000 + w}-t{t:03d}", "epoch": e,
        "success": float(success), "error": False, "usd": 0.01, "cost_usd": 0.01, "cap_hit": False,
        "compile_tokens": [100], "budget": 300, "case": None, "pipeline_miss": None,
    } | extra  # fmt: skip


def lgr_success(t: int) -> int:
    return int(t < TASKS // 2)  # p = 0.5, the same in every world


def apg_success(t: int, world_delta: float) -> int:
    """LGR*'s outcome, with round(Δ × TASKS) tasks flipped up (Δ > 0) or down (Δ < 0) in this world."""
    k = round(abs(world_delta) * TASKS)
    if world_delta > 0:
        return 1 if t < TASKS // 2 + k else 0
    return 1 if k <= t < TASKS // 2 else 0


def rows(apg: dict[str, object], usd: dict[str, float] | None = None, s7: float = 0.0, diag: dict[str, float] | None = None) -> pd.DataFrame:
    """Long sample-epoch rows of the primary cells. `apg` maps a mode (push, pull) to APG*'s per-world Δ on the
    F7 cells: a number (every world), a tuple (per world) or "zero" (APG* always fails); F3 cells use push's.
    `diag` adds gate.diag arms with a constant success."""
    usd = usd or {}
    out = []
    for cell in GATE_CELLS:
        modes = ("push", "pull") if cell.startswith("F7") else ("push",)
        plan_cell = ag.F7_CELL if cell.startswith("F7") else ag.F3_CELL
        for w in range(WORLDS):
            for t in range(TASKS):
                for e in range(1, EPOCHS + 1):
                    for mode in modes:
                        spec = apg[mode]
                        d = spec[w] if isinstance(spec, tuple) else spec
                        a = 0 if d == "zero" else apg_success(t, d)
                        out.append(_row(plan_cell, "APG*", mode, cell, w, t, e, a, usd=usd.get("APG*", 0.01)))
                        out.append(_row(plan_cell, "LGR*", mode, cell, w, t, e, lgr_success(t), usd=usd.get("LGR*", 0.01)))
                    out.append(_row(ag.PLACEBO_CELL, "S7", "push", cell, w, t, e, int(t < TASKS * s7)))
                    for label, p in (diag or {}).items():
                        out.append(_row(ag.DIAG_CELL, label, "push", cell, w, t, e, int(t < TASKS * p)))
    return pd.DataFrame(out)


def _verdict(df, pcs=PC_OK):
    return ag.verdict(df, pcs, ("push", "pull"), reps=REPS)


@pytest.mark.parametrize(
    "apg, usd, expected",
    [
        ({"push": 0.0, "pull": 0.0}, None, "GO"),
        ({"push": 0.0, "pull": 0.0}, {"APG*": 0.03, "LGR*": 0.01}, "GO_WITH_COST_FLAG"),
        ({"push": 0.0, "pull": "zero"}, None, "GO_PUSH_ONLY"),
        ({"push": "zero", "pull": 0.0}, None, "NO_GO"),  # F3 is push: a push-zero APG* also sinks the pull pool
        ({"push": NOISY, "pull": NOISY}, None, "INCONCLUSIVE"),
        ({"push": "zero", "pull": "zero"}, None, "NO_GO"),
    ],
)
def test_combined_verdicts(apg, usd, expected):
    v = _verdict(rows(apg, usd))
    assert v["label"] == expected, (v["label"], {m: (x["verdict"], x["delta"], x["ci"], x["reasons"]) for m, x in v["modes"].items()})
    if expected == "INCONCLUSIVE":
        assert any("ONE pre-registered extension" in n for n in v["notes"])
    if expected == "GO_WITH_COST_FLAG":
        assert v["modes"]["push"]["cost_ratio"] == pytest.approx(3.0)


def test_go_pull_only_when_only_pulls_f7_cells_hold():
    # Pull pools F7 pull with the F3 push cells, so a pull-only GO needs F3 to hold and F7 push to fail.
    df = rows({"push": 0.0, "pull": 0.0})
    f7_push_apg = (df["plan_cell"] == ag.F7_CELL) & (df["delivery"] == "push") & (df["label"] == "APG*")
    df.loc[f7_push_apg, "success"] = 0.0
    v = _verdict(df)
    assert v["label"] == "GO_PULL_ONLY" and v["modes"]["push"]["verdict"] == "NO_GO" and "push: NO_GO" in v["reasons"]


def test_precondition_failure_names_the_precondition_and_still_reports_delta():
    pcs = [dict(p) for p in PC_OK]
    pcs[0] |= {"pass": False, "reason": "anchor/pc1.json missing: run the anchor phase"}
    v = _verdict(rows({"push": 0.0, "pull": 0.0}), pcs)
    assert v["label"] == "PRECONDITION_FAIL" and v["reasons"] == ["PC1: anchor/pc1.json missing: run the anchor phase"]
    assert v["modes"]["push"]["delta"] == 0.0 and set(v["modes"]["push"]["cells"]) == set(GATE_CELLS), "§9: Δ is reported whatever the verdict"


def test_missing_primary_data_is_a_precondition_failure_not_a_crash():
    df = rows({"push": 0.0, "pull": 0.0})
    v = _verdict(df[~((df["cell"] == "F3-60") & (df["label"] == "LGR*")) & (df["label"] != "S7")])
    assert v["label"] == "PRECONDITION_FAIL"
    assert any("primary data missing: LGR* on F3-60" in r and "S7 placebo" in r for r in v["reasons"])


def test_pull_pools_f7_pull_with_f3_push_and_never_mixes_modes():
    df = rows({"push": 0.0, "pull": 0.0})
    pull = ag.mode_rows(df, "pull")
    assert set(pull.loc[pull["cell"].str.startswith("F7") & (pull["label"] != "S7"), "delivery"]) == {"pull"}
    assert set(pull.loc[pull["cell"].str.startswith("F3"), "delivery"]) == {"push"}
    assert set(pull["cell"]) == set(GATE_CELLS)
    # The same arm and cell in both modes is refused, never averaged into one task mean.
    with pytest.raises(ValueError, match="mix delivery modes"):
        task_means(df[df["plan_cell"] == ag.F7_CELL])


def test_sensitivity_excludes_errored_samples():
    df = rows({"push": 0.0, "pull": 0.0})
    hit = (df["label"] == "APG*") & (df["delivery"] == "push") & (df["world"].str.endswith("s3000"))
    df.loc[hit, ["error", "success"]] = [True, 0.0]  # errored samples are failures in the primary analysis
    push = _verdict(df)["modes"]["push"]
    assert push["delta"] < 0 and push["sensitivity_excluding_errors"]["delta"] == 0.0
    assert push["sensitivity_excluding_errors"]["excluded_samples"] == int(hit.sum())


def test_combine_modes_table():
    assert combine_modes({"push": "GO", "pull": "GO_WITH_COST_FLAG"})[0] == "GO_WITH_COST_FLAG"
    assert combine_modes({"push": "GO", "pull": "INCONCLUSIVE"}) == ("GO_PUSH_ONLY", ["pull: INCONCLUSIVE"])
    assert combine_modes({"push": "NO_GO", "pull": "INCONCLUSIVE"})[0] == "INCONCLUSIVE"
    assert combine_modes({"push": "PRECONDITION_FAIL", "pull": "GO"})[0] == "PRECONDITION_FAIL"


# --- preconditions ---------------------------------------------------------------------------------


def _f5(lgr: float, naive: float | None) -> pd.DataFrame:
    out = []
    for cell in ag.F5_CELLS:
        for w in range(4):
            for t in range(TASKS):
                out.append(_row(ag.F5_CELL, "LGR*", "push", cell, w, t, 1, int(t < TASKS * lgr)))
                if naive is not None:
                    out.append(_row(ag.F5_CELL, "LGR-naive", "push", cell, w, t, 1, int(t < TASKS * naive), group="lgr-naive"))
    return pd.DataFrame(out)


def test_pc2_lightrag_beats_naive_on_f5():
    assert ag.pc2(_f5(0.75, 0.5), reps=REPS)["pass"]
    # A tie passes: PC3's 3 pp tolerance keeps noise from failing the precondition.
    assert ag.pc2(_f5(0.5, 0.5), reps=REPS)["pass"]
    bad = ag.pc2(_f5(0.25, 0.5), reps=REPS)
    assert not bad["pass"] and "< −3 pp" in bad["reason"]
    missing = ag.pc2(_f5(0.75, None), reps=REPS)
    assert not missing["pass"] and "gate.f5: no results for ['LGR-naive']" in missing["reason"]


def test_pc3_invariant_chain_uses_apg_star():
    good = rows({"push": 0.0, "pull": 0.0}, s7=0.0, diag={"S6": 1.0, "S5o": 0.5})
    pc = ag.pc3(good, reps=REPS)
    assert pc["pass"], pc
    assert set(pc["value"]) == {"S6", "S5o", "APG*", "S7"}
    bad = ag.pc3(rows({"push": 0.0, "pull": 0.0}, s7=0.0, diag={"S6": 0.25, "S5o": 0.5}), reps=REPS)
    assert not bad["pass"] and "S6>=S5o" in bad["reason"]
    missing = ag.pc3(rows({"push": 0.0, "pull": 0.0}, diag={"S5o": 0.5}), reps=REPS)
    assert not missing["pass"] and "['S6']" in missing["reason"]


def _matched(lgr_tokens: int, s3s_tokens: int) -> pd.DataFrame:
    out = []
    for label, toks in (("APG*", 200), ("LGR*", lgr_tokens), ("S3s", s3s_tokens)):
        for cell in GATE_CELLS:
            out.append(_row("gate.sec.matched-300", label, "push", cell, 0, 0, 1, 1, group="matched", compile_tokens=[toks, toks]))
    return pd.DataFrame(out)


def test_pc4_bound_and_matched_budget():
    primary = rows({"push": 0.0, "pull": 0.0})
    ok = ag.pc4(pd.concat([primary, _matched(300, 260)]), "gate.sec.matched-300", 300, True)
    assert ok["pass"], ok["reason"]
    assert ok["details"]["matched"]["APG*"]["gated"] is False
    over = primary.copy()
    over["compile_tokens"] = [[1500] if label == "S7" else ts for label, ts in zip(over["label"], over["compile_tokens"], strict=True)]
    bad = ag.pc4(pd.concat([over, _matched(300, 260)]), "gate.sec.matched-300", 300, True)
    assert not bad["pass"] and "S7 (push): median 1500 tokens = 5.0× its budget 300" in bad["reason"]
    # The ±25% clause is reported with the matched-budget secondary; it never fails PC4 (D-022).
    off = ag.pc4(pd.concat([primary, _matched(400, 260)]), "gate.sec.matched-300", 300, True)
    assert off["pass"] and off["details"]["matched"]["interpretable"] is False
    assert "matched budget LGR*: median 400 tokens outside [225, 375]" in off["details"]["matched"]["notes"]
    gone = ag.pc4(primary, "gate.sec.matched-300", 300, False)
    assert gone["pass"] and gone["details"]["matched"]["status"] == "missing" and "was not run" in gone["details"]["matched"]["notes"][0]


def test_pc5_error_and_cap_hit_rates_in_the_verdicts_cells():
    df = rows({"push": 0.0, "pull": 0.0})
    assert ag.pc5(df)["pass"]
    apg_push = df.index[(df["label"] == "APG*") & (df["delivery"] == "push") & (df["plan_cell"] == ag.F7_CELL)]
    errs = df.copy()
    errs.loc[apg_push[: int(len(apg_push) * 0.05)], "error"] = True
    bad = ag.pc5(errs)
    assert not bad["pass"] and "gate.test.f7 APG* (push): errors 4.7%" in bad["reason"]
    caps = df.copy()
    caps.loc[apg_push[: int(len(apg_push) * 0.2)], "cap_hit"] = True
    capped = ag.pc5(caps)
    assert not capped["pass"] and "cap hits 19.9%" in capped["reason"]
    other = pd.concat([df, pd.DataFrame([_row("gate.sec.messy", "APG*", "push", "F7-10", 0, 0, 1, 0, error=True)])])
    assert ag.pc5(other)["pass"], "only the verdict's cells are gated"


def _tune_dir(tmp_path, records, archived=()):
    d = tmp_path / "tune"
    d.mkdir(parents=True)
    (d / "tuning_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for i, recs in enumerate(archived):
        (d / f"tuning_log.2026010{i}T000000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


GRID = {"budget_per_system": 2, "systems": {"APG": {"candidates": [{"id": "a1"}, {"id": "a2"}]}}}
SELECTED = {"APG*": {"candidate": "a2"}}


def _recs(*ids, selected="a2"):
    return [{"system": "APG", "candidate": {"id": i}, "status": "ok"} for i in ids] + [{"system": "APG", "selected": selected}]


def test_pc6_tuning_log_complete_within_budget_and_matching_the_selection(tmp_path):
    sys_ = (("APG", "APG*"),)
    assert ag.pc6(_tune_dir(tmp_path / "ok", _recs("a1", "a2")), GRID, GRID, SELECTED, sys_)["pass"]
    miss = ag.pc6(_tune_dir(tmp_path / "miss", _recs("a2")), GRID, GRID, SELECTED, sys_)
    assert not miss["pass"] and "APG: no record for ['a1']" in miss["reason"]
    wrong = ag.pc6(_tune_dir(tmp_path / "wrong", _recs("a1", "a2", selected="a1")), GRID, GRID, SELECTED, sys_)
    assert "logged selection a1 != selected.yaml a2" in wrong["reason"]
    # An archived run that tried a third configuration breaks the equal budget (and it was never declared).
    over = ag.pc6(_tune_dir(tmp_path / "over", _recs("a1", "a2"), archived=[_recs("a3", selected="a3")]), GRID, GRID, SELECTED, sys_)
    assert not over["pass"] and "3 configurations tried > budget 2" in over["reason"] and "undeclared configurations ['a3']" in over["reason"]
    assert len(over["details"]["archived_logs"]) == 1
    none = ag.pc6(tmp_path / "absent", GRID, GRID, SELECTED, sys_)
    assert not none["pass"] and "run the tune phase" in none["reason"]


def test_pc1_from_pc1_json(tmp_path):
    missing = ag.pc1(tmp_path / "pc1.json")
    assert not missing["pass"] and "pc1.json missing: run the anchor phase" in missing["reason"]
    (tmp_path / "pc1.json").write_text(json.dumps({"pass": True, "reproduces_published": True, "beats_naive": True, "macro": 64.0, "naive_macro": 61.0, "tolerance_pp": 5.0}))
    assert ag.pc1(tmp_path / "pc1.json")["pass"]


def test_pipeline_miss_labels_the_first_lost_stage():
    def apg(**gold):
        return [{"meta": {"gold": {"nodes": ["n1"], "in_shortlist": True, "in_matches": True, "in_contributors": True} | gold, "truncated": 0}, "fact_ids": []}]

    assert pipeline_miss(apg(nodes=[]), [], {}) == "not_in_graph"
    assert pipeline_miss(apg(in_shortlist=False), [], {}) == "shortlist_miss"
    assert pipeline_miss(apg(in_matches=False), [], {}) == "classify_miss"
    assert pipeline_miss(apg(in_contributors=False), [], {}) == "compose_miss"
    f3 = {"gold": {"calls": [{"tool": "refund"}]}, "gold_fact_ids": ["f1"]}
    assert pipeline_miss(apg(), [{"exposed_tools": ["lookup"]}], f3) == "tool_exposure_miss"
    assert pipeline_miss([{"meta": {}, "fact_ids": ["f2"]}], [], f3) == "evidence_miss"
    assert pipeline_miss([{"meta": {}, "fact_ids": ["f1"]}], [{"exposed_tools": ["refund"]}], f3) == "delivered_but_failed"
