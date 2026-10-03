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
    # Descriptive (D-023, RELIABILITY_REVIEW S7): the 4× bound holds by construction, so an arm over it is flagged as
    # an anomaly and never fails PC4 or the verdict.
    assert bad["pass"] and bad["gated"] is False and "S7 (push): median 1500 tokens = 5.0× its budget 300" in bad["reason"]
    assert bad["details"]["anomalies"] and bad["value"] == {"arms_over_bound": 1}
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
    # S9: the placebo's cap hits are reported, not gated (a random context that keeps the agent searching is the
    # placebo working); its errors still are.
    s7 = df.index[df["label"] == "S7"]
    flail = df.copy()
    flail.loc[s7[: int(len(s7) * 0.5)], "cap_hit"] = True
    p = ag.pc5(flail)
    assert p["pass"] and any(r["arm"] == "S7 (push)" and r["cap_hit_rate"] >= 0.49 and not r["cap_hits_gated"] for r in p["details"]["rates"])
    flail.loc[s7[: int(len(s7) * 0.05)], "error"] = True
    assert not ag.pc5(flail)["pass"]


def test_inconclusive_still_reports_the_secondary_conditions_and_dropped_tasks():
    """RELIABILITY_REVIEW S9: the F7-1000 floor and the S7 test are computed whatever the verdict, and unpaired tasks
    left out of the paired analysis are counted per cell."""
    df = rows({"push": NOISY, "pull": NOISY})
    drop = (df["label"] == "LGR*") & (df["cell"] == "F3-5") & (df["task"].str.endswith("t000"))
    v = _verdict(df[~drop])
    push = v["modes"]["push"]
    assert v["label"] == "INCONCLUSIVE"
    sec = push["details"]["secondary"]
    assert sec["f7_1000_delta"] is not None and sec["p_apg_gt_s7"] is not None
    assert push["details"]["dropped_tasks"] == {"F3-5": WORLDS}


def test_holm_across_modes_in_the_combined_verdict():
    """S4: the modes are tested with Holm at the gate run's α (0.02): the better mode at 0.01, the other at 0.02 only
    if the first is rejected; the levels are in the decision."""
    v = _verdict(rows({"push": 0.0, "pull": 0.0}))
    assert v["label"] == "GO" and v["alpha"] == ag.GATE_ALPHA == 0.02
    assert sorted(v["holm"]["levels"].values()) == [0.01, 0.02] and all(v["holm"]["rejected"].values())
    for m in ("push", "pull"):
        assert v["modes"][m]["details"]["inference"]["holm_rejected"] is True


def _fake_run(tmp_path, name, worlds, label=None):
    """A run dir with build-test/worlds.json and, optionally, an analysed report/decision.json."""
    from types import SimpleNamespace

    d = tmp_path / name
    (d / "build-test").mkdir(parents=True)
    (d / "build-test" / "worlds.json").write_text(json.dumps({"worlds": [{"world_id": w} for w in worlds]}))
    if label:
        (d / "report").mkdir()
        (d / "report" / "decision.json").write_text(json.dumps({"verdict": {"label": label}, "extension": None, "constants": {"alpha_gate_run": 0.02}}))
    return SimpleNamespace(dir=d)


def test_extension_context_needs_an_inconclusive_stage1_and_fresh_worlds(tmp_path):
    """S6: the one extension (§8) reads both runs' test worlds from their manifests, never from seed arithmetic."""
    s1 = _fake_run(tmp_path, "s1", ["F7-10-test-s3000", "F7-10-test-s3001"], "INCONCLUSIVE")
    s2 = _fake_run(tmp_path, "s2", ["F7-10-test-s3100", "F7-10-test-s3101"])
    assert ag.extension_context(s2) is None  # not an extension until marked
    (s2.dir / ag.EXTENSION_FILE).write_text(json.dumps({"stage1_run_id": "s1", "stage1_dir": str(s1.dir)}))
    ok = ag.extension_context(s2)
    assert ok["problems"] == [] and ok["alpha"] == 0.005 and ok["total_alpha"] == 0.025 and ok["stage1_worlds"] == 2
    reused = _fake_run(tmp_path, "s3", ["F7-10-test-s3001", "F7-10-test-s3200"])
    (reused.dir / ag.EXTENSION_FILE).write_text(json.dumps({"stage1_run_id": "s1", "stage1_dir": str(s1.dir)}))
    assert any("reuses 1 stage-1 test world" in p for p in ag.extension_context(reused)["problems"])
    go = _fake_run(tmp_path, "go", ["F7-10-test-s3000"], "GO")
    after_go = _fake_run(tmp_path, "s4", ["F7-10-test-s3300"])
    (after_go.dir / ag.EXTENSION_FILE).write_text(json.dumps({"stage1_run_id": "go", "stage1_dir": str(go.dir)}))
    assert any("not INCONCLUSIVE" in p for p in ag.extension_context(after_go)["problems"])


def test_extension_verdict_uses_the_extension_alpha():
    v = ag.verdict(rows({"push": 0.0, "pull": 0.0}), PC_OK, ("push", "pull"), reps=REPS, alpha=ag.EXTENSION_ALPHA, stage="extension")
    assert v["alpha"] == 0.005 and sorted(v["holm"]["levels"].values()) == [0.0025, 0.005]
    assert not any("ONE pre-registered extension" in n for n in v["notes"])


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
    tol = {"macro": 5.0, "per_type": 10.0}
    ok = {"status": "pass", "pass": True, "macro_within_tolerance": True, "types_within_tolerance": True, "macro": 64.0, "published_macro": 63.92, "naive_macro": 66.0, "beats_naive": False, "tolerance_pp": tol}
    (tmp_path / "pc1.json").write_text(json.dumps(ok))
    passed = ag.pc1(tmp_path / "pc1.json")
    assert passed["pass"] and passed["value"]["status"] == "pass", "hybrid below naive does not fail PC1 (D-025)"
    off = {**ok, "status": "fail", "pass": False, "types_within_tolerance": False, "per_type": {"Complex Reasoning": {"within_tolerance": False}, "Fact Retrieval": {"within_tolerance": True}}}
    (tmp_path / "pc1.json").write_text(json.dumps(off))
    assert "outside ±10.0 pp: Complex Reasoning" in ag.pc1(tmp_path / "pc1.json")["reason"]
    (tmp_path / "pc1.json").write_text(json.dumps({**ok, "status": "not_evaluable", "pass": False, "diagnosis": "gating-scorer parse failures above 5%: Fact Retrieval 12.0%"}))
    assert ag.pc1(tmp_path / "pc1.json")["reason"].startswith("not evaluable: gating-scorer parse failures")


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


def test_a_pc1_failure_accepted_at_the_freeze_does_not_block_the_verdict():
    pcs = [dict(p) for p in PC_OK]
    pcs[0] |= {"pass": False, "reason": "does not reproduce the published accuracy within ±10 pp"}
    assert _verdict(rows({"push": 0.0, "pull": 0.0}), pcs)["label"] == "PRECONDITION_FAIL"
    pcs[0]["accepted"] = {"reason": "judge parse rate 99.8%, no index errors; the Luna judge scores lower than gpt-4o-mini"}
    assert _verdict(rows({"push": 0.0, "pull": 0.0}), pcs)["label"] == "GO"



# --- harness health and matched-budget calibration status ---


def _apg_compile(fallback=False, error=None, repaired=(), unknown=0, **extra) -> dict:
    meta = {"classify_error": error, "classify_fallback": fallback, "classify_repaired": list(repaired), "classify_unknown_ids": unknown}
    return {"step": 0, "tokens": 100, "fact_ids": [], "meta": meta, "source": "push"} | extra


def _lgr_compile(fallback=False, **extra) -> dict:
    meta = {"lightrag": {"mode": "mix", "keyword_fallback": fallback, "keyword_error": "no keywords" if fallback else None}}
    return {"step": 0, "tokens": 100, "fact_ids": [], "meta": meta, "source": "push"} | extra


def test_compile_health_counts_each_counter_and_reads_absent_meta_as_zero():
    from ape.analysis.gate_stats import HEALTH_COLUMNS, compile_health

    log = [
        _apg_compile(),
        _apg_compile(fallback=True, error="unparseable", repaired=("fence", "fence", "id_prefix"), unknown=2),
        _lgr_compile(fallback=True),
        _lgr_compile(source="pull", truncated=True),
        {"step": 1, "tokens": 50, "meta": {}, "source": "pull", "truncated": False},
    ]
    h = compile_health(log, {"search_errors": [{"step": 2, "error": "empty query"}]})
    assert set(h) == set(HEALTH_COLUMNS)
    assert (h["classify_compiles"], h["classify_errors"], h["classify_fallbacks"], h["classify_unknown_ids"]) == (2, 1, 1, 2)
    assert h["classify_repaired"] == {"fence": 1, "id_prefix": 1}  # per compile, not per repair occurrence
    assert (h["keyword_compiles"], h["keyword_fallbacks"], h["keyword_errors"]) == (2, 1, 1)
    assert (h["pull_compiles"], h["pull_truncated"], h["search_errors"]) == (2, 1, 1)
    # Naive mode extracts no keywords, so its always-set flag is not a keyword failure.
    naive = _lgr_compile(fallback=True)
    naive["meta"]["lightrag"]["mode"] = "naive"
    assert compile_health([naive], {})["keyword_compiles"] == 0
    # A log that predates the counters: zero compiles carry them, so the report reads n/a, never 0%.
    old = compile_health([{"step": 0, "tokens": 10, "meta": {"route": {}}}], {})
    assert old["classify_compiles"] == old["keyword_compiles"] == old["search_errors"] == 0


def _health_rows() -> pd.DataFrame:
    zero = {"classify_compiles": 0, "classify_errors": 0, "classify_fallbacks": 0, "classify_repaired": {}, "classify_unknown_ids": 0,
            "keyword_compiles": 0, "keyword_fallbacks": 0, "keyword_errors": 0, "pull_compiles": 0, "pull_truncated": 0, "search_errors": 0}  # fmt: skip
    out = []
    for t in range(20):
        # APG* push F7-10: 10 compiles per sample, a fallback in 2 of 20 samples -> 2 / 200 = 1% (no warning).
        out.append(_row(ag.F7_CELL, "APG*", "push", "F7-10", 0, t, 1, 1, run_arm="APG-s", **zero | {"classify_compiles": 10, "classify_fallbacks": int(t < 2), "classify_repaired": {"fence": 1} if t < 4 else {}}))
        # APG* push F7-1000: 1 compile per sample, a fallback in 3 of 20 -> 15% (warning).
        out.append(_row(ag.F7_CELL, "APG*", "push", "F7-1000", 0, t, 1, 1, run_arm="APG-s", **zero | {"classify_compiles": 1, "classify_fallbacks": int(t < 3)}))
        # LGR* pull F7-10: keyword fallback 1 of 20 (5%, not above the threshold); search errors 2 of 22 searches (9.1%, warning).
        out.append(_row(ag.F7_CELL, "LGR*", "pull", "F7-10", 0, t, 1, 1, run_arm="LGR-s", **zero | {"keyword_compiles": 1, "keyword_fallbacks": int(t < 1), "pull_compiles": 1, "search_errors": int(t < 2)}))
        # S5o: an APG arm whose logs carry no counters -> n/a, no warning.
        out.append(_row(ag.DIAG_CELL, "S5o", "push", "F7-10", 0, t, 1, 1, run_arm="S5o", **zero))
        # S1 has neither counter family: not in the classify or keyword tables.
        out.append(_row(ag.DIAG_CELL, "S1", "push", "F7-10", 0, t, 1, 1, run_arm="S1", **zero))
    return pd.DataFrame(out)


def test_health_rates_per_arm_mode_and_cell_with_warnings_above_five_percent():
    h = ag.health(_health_rows())
    classify = {(e["plan_cell"], e["arm"], e["cell"]): e for e in h["classify"]}
    assert set(classify) == {(ag.F7_CELL, "APG* (push)", "F7-10"), (ag.F7_CELL, "APG* (push)", "F7-1000"), (ag.DIAG_CELL, "S5o (push)", "F7-10")}
    assert classify[(ag.F7_CELL, "APG* (push)", "F7-10")]["fallback_rate"] == pytest.approx(0.01)
    assert classify[(ag.F7_CELL, "APG* (push)", "F7-10")]["repaired_rate"] == {"fence": pytest.approx(0.02)}
    assert classify[(ag.F7_CELL, "APG* (push)", "F7-1000")]["fallback_rate"] == pytest.approx(0.15)
    assert classify[(ag.DIAG_CELL, "S5o (push)", "F7-10")]["fallback_rate"] is None  # n/a
    (kw,) = h["keywords"]
    assert kw["arm"] == "LGR* (pull)" and kw["fallback_rate"] == pytest.approx(0.05)
    (sk,) = h["search_kb"]
    assert sk["searches"] == 22 and sk["error_rate"] == pytest.approx(2 / 22) and sk["truncated_rate"] == 0.0
    assert h["warnings"] == [
        f"APG classify fallback rate 15.0% for APG* (push) F7-1000 in {ag.F7_CELL} (> 5%)",
        f"search_kb error rate 9.1% for LGR* (pull) F7-10 in {ag.F7_CELL} (> 5%)",
    ]
    empty = ag.health(pd.DataFrame())
    assert empty["classify"] == empty["keywords"] == empty["search_kb"] == empty["warnings"] == []


def test_matched_calibration_status_from_the_manifest_then_the_yaml_else_na(tmp_path):
    test = {"cells": {ag.MATCHED_CELL: {"groups": [{"name": "matched", "calibration": {"matched": False, "not_converged": ["LGR*"]}}]}}}
    c = ag.matched_calibration(test, None)
    assert c["matched"] is False and c["label"] == "not matched (calibration did not converge for LGR*)" and c["source"] == "test manifest"
    yml = tmp_path / "budget_calibration.yaml"
    yml.write_text("context: 300\nmatched: true\nnot_converged: []\n")
    assert ag.matched_calibration({}, yml) | {} == {"status": "checked", "matched": True, "not_converged": [], "source": "budget_calibration.yaml", "label": "matched"}
    yml.write_text("context: 300\n")  # a calibration that predates the flag
    assert ag.matched_calibration({}, yml)["status"] == "n/a"
    assert ag.matched_calibration(None, tmp_path / "missing.yaml")["label"] == "calibration status n/a"


def _minimal_decision(health: dict, calibration: dict) -> dict:
    return {
        "header": {"run_id": "r", "offline": True, "generated_at": "t", "git": {}, "freeze": None, "selected": {}, "spend": {}, "test_status": "done", "primary_complete": True, "cells_not_done": {}},
        "verdict": {"label": "GO", "reasons": [], "notes": [], "modes": {}},
        "extension": None,
        "caveats": [],
        "preconditions": [],
        "diagnosis": {"rule": "r", "s5o_vs_lgr": None, "label": None, "reason": "n/a", "pipeline_misses": {}},
        "tables": {"arms": {}, "costs": {}, "cost_notes": [], "build_costs": {"note": "none"}, "latency": {}, "determinism": {}, "exception_applies": {}},
        "health": health,
        "secondaries": {ag.MATCHED_CELL: {"title": "Matched budget", "status": "done", "reason": None, "arms": {}, "comparisons": {}, "calibration": calibration}},
        "secondary_pairings": {"id_only_vs_descriptive": {}, "te_all_vs_retrieved": {}},
        "coverage": [],
        "choices": [],
    }  # fmt: skip


def test_report_renders_health_warnings_tables_and_the_calibration_label():
    h = ag.health(_health_rows())
    cal = ag.matched_calibration({"cells": {ag.MATCHED_CELL: {"groups": [{"name": "matched", "calibration": {"matched": False, "not_converged": ["LGR*", "S3s"]}}]}}}, None)
    md = ag.render(json.loads(json.dumps(ag._clean(_minimal_decision(h, cal)))))
    assert "\n## Harness health\n" in md and "- Harness health: 2 warning(s)" in md
    assert "- ⚠️ APG classify fallback rate 15.0% for APG* (push) F7-1000" in md
    assert "| gate.test.f7 | APG* (push) | F7-1000 | 20 | 15.0% | 0.0% | – | 0.00 |" in md
    assert "| gate.diag | S5o (push) | F7-10 | 0 | n/a | n/a | – | n/a |" in md  # absent meta degrades to n/a
    assert "| gate.test.f7 | LGR* (pull) | F7-10 | 22 | 9.1% | 0.0% |" in md
    assert "**Not matched (calibration did not converge for LGR*, S3s):**" in md
    na = ag.render(json.loads(json.dumps(ag._clean(_minimal_decision(ag.health(pd.DataFrame()), ag.matched_calibration(None, None))))))
    assert "- No warnings." in na and "- n/a (no APG rows)." in na and "Calibration status: n/a" in na
