"""The M5 add-on study's analysis (D-053): its hypothesis table, planned sizes and power simulation (through the main
study's model and tests), its report on simulated frames (concurrent controls only, cost ratios, ledger diagnostics),
the ledger-record reader, and `ape.analyze_m5` on a run directory of real offline logs. Offline: mock models only."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from ape.analysis import m5_hypotheses as mh5
from ape.analysis import m5_power as mp5
from ape.analysis.m5_load import LEDGER_COLUMNS, attach_ledgers, ledger_metrics, read_ledgers, stalled
from ape.analysis.m5_report import m5_report, m5_rows, render_m5
from ape.analysis.main_hypotheses import HYPOTHESES as MAIN_HYPOTHESES
from ape.analysis.main_power import Settings, Sigmas, simulate
from ape.analysis.main_stats import build_tables, evaluate_family
from ape.config import ROOT

MODEL = "mockllm/model"


# --- the hypothesis table ---------------------------------------------------------------------------


def test_the_confirmatory_families_are_l1_on_f2_and_f1_and_l2_over_study_b():
    conf = {h.id: h for h in mh5.confirmatory()}
    assert set(conf) == {"L1", "L2"}
    l1, l2 = conf["L1"].members, conf["L2"].members
    assert [(m.id, m.cells) for m in l1] == [("L1.F2", ("F2-2", "F2-10")), ("L1.F1", ("F1-2", "F1-32"))] and {m.contrast() for m in l1} == {"M5 − M1"}
    assert [m.id for m in l2] == ["L2.pooled"] and l2[0].cells == mh5.STUDY_B_CELLS and l2[0].contrast() == "M5-spec − M2"
    for h in conf.values():
        assert h.alpha == 0.025 and h.procedure == "holm" and "POWER" not in h.note
        assert all(m.test == "superiority" and m.mde and m.power and "POWER" not in m.power for m in h.members)


def test_the_m5_table_adds_nothing_to_the_main_studys_families():
    ids = {h.id for h in mh5.HYPOTHESES} | {m.id for h in mh5.HYPOTHESES for m in h.members}
    main_ids = {h.id for h in MAIN_HYPOTHESES} | {m.id for h in MAIN_HYPOTHESES for m in h.members}
    assert not ids & main_ids, "no ID is shared with the main study's table (its hypothesis M5 is role specialization)"
    assert all(h.id.startswith("L") for h in mh5.HYPOTHESES)


def test_every_member_contrasts_a_ledger_arm_with_its_concurrent_control_in_the_planned_cells():
    sizes = mp5.m5_sizes()
    for h in mh5.HYPOTHESES:
        for m in h.members:
            arms = m.arms()
            assert len(arms) == 2 and mh5.LEDGER_ARMS[arms[0]] == arms[1], m.id
            for c in m.cells:
                plan_cells = {sizes[("luna", c, a)]["plan_cell"] for a in arms}
                assert len(plan_cells) == 1 and next(iter(plan_cells)).startswith(mh5.PLAN_PREFIX), f"{m.id} {c}: both arms in one m5 test cell"


def test_planned_sizes_are_108_tasks_over_9_worlds_3_epochs_with_or_without_the_plan_block(tmp_path):
    sizes = mp5.m5_sizes()
    assert len(sizes) == 16 and all(v == v | {"n_tasks": 108, "worlds": 9, "epochs": 3} for v in sizes.values())
    plan = yaml.safe_load((ROOT / "config" / "run_plan.yaml").read_text())
    plan.get("studies", {}).pop("m5", None)
    bare = tmp_path / "run_plan.yaml"
    bare.write_text(yaml.safe_dump(plan))
    assert mp5.m5_sizes(bare) == sizes, "without studies.m5 the default is D-053's design"


# --- the power simulation -----------------------------------------------------------------------------


def test_scenarios_cap_the_ledger_arm_below_one_and_report_the_realised_delta():
    spec = mp5.spec_for("M5-spec", dict.fromkeys(mh5.STUDY_B_CELLS, 0.10))
    assert all(arms["M5-spec"] <= mp5.CEILING for arms in spec.values())
    assert mp5.realised(spec, mh5.STUDY_B_CELLS) < 0.10, "F7-10's control at 0.88 leaves 0.10 of room only up to the cap"
    assert mp5.realised(mp5.spec_for("M5", {"F2-2": 0.0, "F2-10": 0.2}), mh5.F2_CELLS) == pytest.approx(0.1)


def test_the_simulation_runs_through_the_real_analysis_and_has_power_and_level():
    hyp = mh5.get("L1")
    null = next(sc for sc, kind in mp5.scenarios("L1") if kind == "null")
    big = mp5.Scenario("big", mp5.spec_for("M5", dict.fromkeys(mh5.F2_CELLS, 0.25) | dict.fromkeys(mh5.F1_CELLS, 0.0)))
    sizes = mp5.m5_sizes()
    assert mp5.simulate_family(hyp, null, 150, 7, sizes=sizes)["false_claims"] <= 0.06
    assert mp5.simulate_family(hyp, big, 60, 8, sizes=sizes)["members"]["L1.F2"] >= 0.9


def test_a_simulated_studys_frame_through_m5_report_gives_the_direct_tables_result():
    rng = np.random.default_rng(3)
    spec = mp5._with(mp5.spec_for("M5", dict.fromkeys(mh5.STUDY_A_CELLS, 0.08)), mp5.spec_for("M5-spec", dict.fromkeys(mh5.STUDY_B_CELLS, 0.05)))
    n = {k: 36 for k in spec}
    draw = simulate(spec, n, {(t, c, a): 3 for (t, c), arms in spec.items() for a in arms}, Sigmas(), Settings(), rng)
    frame = _as_m5(draw.frame())
    d = m5_report(frame, reps=500)
    for fid in ("L1", "L2"):
        direct = evaluate_family(draw.tables(), mh5.get(fid), reps=500, seed=101 * [h.id for h in mh5.HYPOTHESES].index(fid))
        for mid, m in direct["members"].items():
            assert d["hypotheses"][fid]["members"][mid]["est"] == pytest.approx(m["est"]) and d["hypotheses"][fid]["members"][mid]["p"] == pytest.approx(m["p"])


def _as_m5(frame: pd.DataFrame) -> pd.DataFrame:
    """A simulated frame as the m5 study's loader would give it: plan cells and a $ meter."""
    f = frame.copy()
    f["plan_cell"] = np.where(f["arm"].isin(["M5", "M1"]), "m5.test.a", "m5.test.b")
    f["usd"] = f["tokens"] * 1e-6
    return f


# --- the report on simulated frames -------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim_frame():
    rng = np.random.default_rng(11)
    spec = mp5._with(mp5.spec_for("M5", dict.fromkeys(mh5.STUDY_A_CELLS, 0.10)), mp5.spec_for("M5-spec", dict.fromkeys(mh5.STUDY_B_CELLS, 0.06)))
    draw = simulate(spec, {k: 36 for k in spec}, {(t, c, a): 3 for (t, c), arms in spec.items() for a in arms}, Sigmas(), Settings(), rng)
    f = _as_m5(draw.frame())
    led = f["arm"].isin(["M5", "M5-spec"])
    control = f[~led].set_index(["cell", "world", "task", "epoch"])["tokens"]
    same = control.reindex(pd.MultiIndex.from_frame(f[["cell", "world", "task", "epoch"]])).to_numpy()
    f["tokens"] = np.where(led, same * 1.3, f["tokens"])  # each ledger run costs 30% more than its control's
    f["usd"] = f["tokens"] * 1e-6
    f["ledger_present"] = led
    f["ledger_rounds"] = np.where(led, 6, None)
    f["ledger_replans"] = np.where(led, (f["task"].str[-1].astype(int) % 3 == 0).astype(int), None)
    f["ledger_stalls"] = np.where(led, 2, None)
    f["ledger_tokens"] = np.where(led, f["tokens"] * 0.25, None)
    f["sample_tokens"] = np.where(led, f["tokens"], None)
    return f


def test_only_this_studys_concurrent_controls_enter_the_contrasts(sim_frame):
    old = sim_frame[sim_frame["arm"] == "M1"].assign(plan_cell="main.A.arms", success=1.0)  # the main study's M1, all successes
    d = m5_report(pd.concat([sim_frame, old], ignore_index=True), reps=300)
    ref = m5_report(sim_frame, reps=300)
    assert d["header"]["scope"]["excluded_plan_cells"] == ["main.A.arms"] and d["header"]["scope"]["excluded_rows"] == len(old)
    assert d["summary"] == ref["summary"], "the main study's rows change nothing"
    assert all(c["concurrent"] for c in d["header"]["concurrency"]) and len(d["header"]["concurrency"]) == 8
    rows, scope = m5_rows(sim_frame.assign(plan_cell=None))
    assert len(rows) == len(sim_frame) and scope["unlabelled_rows"] == len(sim_frame)


def test_the_report_labels_tests_and_prices_the_ledger(sim_frame):
    d = m5_report(sim_frame, reps=500)
    json.dumps(d, allow_nan=False)
    assert [s["member"] for s in d["summary"]] == ["L1.F2", "L1.F1", "L2.pooled"]
    assert all(s["label"] in ("supported", "not supported", "not tested") for s in d["summary"])
    l1 = sorted((s for s in d["summary"] if s["family"] == "L1"), key=lambda s: s["p"])
    assert l1[0]["holm_level"] == 0.0125 and l1[1]["holm_level"] == (0.025 if l1[0]["label"] == "supported" else 0.0125), "Holm of 2 in L1"
    assert next(s for s in d["summary"] if s["family"] == "L2")["holm_level"] == 0.025
    assert {s["member"] for s in d["descriptive"]} >= {"L1-cells.F1-32", "L2-cells.F7-1000"}
    l2 = d["hypotheses"]["L2"]["members"]["L2.pooled"]
    assert l2["clusters"] == 36 and all(d["hypotheses"]["L1"]["members"][m]["clusters"] == 9 for m in ("L1.F2", "L1.F1")), "F1 and F2 cells share the registry clusters"
    assert set(l2["per_cell"]) == set(mh5.STUDY_B_CELLS)
    cost = {(r["arm"], r["cells"], r["meter"]): r for r in d["cost"]}
    for key in (("M5", "Study A", "tokens"), ("M5-spec", "Study B", "usd"), ("M5", "F2-10", "tokens")):
        assert cost[key]["ratio"] == pytest.approx(1.3, rel=1e-6) and cost[key]["ci"][0] is not None
    md = render_m5(d)
    for heading in ("## Confirmatory tests", "## Descriptive contrasts", "## The ledger's price", "## Ledger diagnostics", "## Harness", "## Analysis choices"):
        assert heading in md


def test_ledger_diagnostics_per_arm_and_cell(sim_frame):
    led = m5_report(sim_frame, reps=200)["ledger"]
    assert led["available"] and set(led["arms"]) == {"M5", "M5-spec"}
    pooled = led["arms"]["M5"]["pooled"]
    assert pooled["with_record"] == pooled["samples"] and pooled["rounds_per_sample"] == 6 and pooled["stall_rate"] == pytest.approx(2 / 6)
    assert pooled["ledger_token_share"] == pytest.approx(0.25) and pooled["stalled_share"] == 1.0
    assert 0 < pooled["replanned_share"] < 1 and pooled["replans_per_task"] == pytest.approx(pooled["replanned_share"])
    assert set(led["arms"]["M5"]["cells"]) == set(mh5.STUDY_A_CELLS)
    none = m5_report(sim_frame.drop(columns=list(LEDGER_COLUMNS)), reps=200)["ledger"]
    assert none["available"] is False and none["reason"]


def test_missing_arms_are_not_evaluable_never_raised(sim_frame):
    d = m5_report(sim_frame[sim_frame["arm"] != "M2"], reps=200)
    l2 = next(s for s in d["summary"] if s["member"] == "L2.pooled")
    assert l2["label"] == "not evaluable" and "M2" in l2["reason"] and d["errors"] == []
    empty = m5_report(sim_frame.iloc[0:0], reps=200)
    assert [s["label"] for s in empty["summary"]] == ["not evaluable"] * 3
    render_m5(empty)


# --- the ledger records ---------------------------------------------------------------------------------


def test_ledger_metrics_read_the_record_in_its_spellings():
    acc = {"totals": {"total_tokens": 1000}, "agents": {"orch": {"role": "orchestrator", "models": {"agent": {"total_tokens": 600}, "ledger": {"total_tokens": 150}}}, "w1": {"role": "worker", "all_total_tokens": 250}}}
    counts = ledger_metrics({"rounds": 5, "replans": 1, "stalls": 2}, acc)
    assert counts == {"ledger_present": True, "ledger_rounds": 5, "ledger_replans": 1, "ledger_stalls": 2, "ledger_tokens": 150, "sample_tokens": 1000}
    m1 = {"plans": [{}, {}, {}], "progress": [
        {"is_in_loop": {"answer": False}, "is_progress_being_made": {"answer": True}},
        {"is_in_loop": {"answer": True, "reason": "repeats"}, "is_progress_being_made": {"answer": True}},
        {"is_in_loop": False, "is_progress_being_made": False},
        {"note": "no flags"},
    ], "usage": {"total_tokens": 321}}  # fmt: skip
    got = ledger_metrics(m1, acc)
    assert (got["ledger_rounds"], got["ledger_replans"], got["ledger_stalls"], got["ledger_tokens"]) == (4, 2, 2, 321)
    assert stalled({"stalled": "yes"}) is True and stalled({"x": 1}) is None and stalled("no") is None
    bare = ledger_metrics({"progress": [{"x": 1}]}, None)
    assert bare["ledger_rounds"] == 1 and bare["ledger_stalls"] is None and bare["ledger_replans"] is None and bare["ledger_tokens"] is None
    assert ledger_metrics(None, acc) == dict.fromkeys(LEDGER_COLUMNS) | {"ledger_present": False, "sample_tokens": 1000}
    by_role = ledger_metrics({"rounds": 1}, {"agents": {"l": {"role": "ledger", "all_total_tokens": 40}}})
    assert by_role["ledger_tokens"] == 40


def test_attach_ledgers_joins_on_log_task_and_epoch():
    frame = pd.DataFrame({"log_file": ["a", "a", "b"], "task": ["t1", "t1", "t2"], "epoch": [1, 2, 1], "arm": ["M5", "M5", "M1"]})

    def reader(files):
        assert files == ["a", "b"]
        return pd.DataFrame([{"log_file": "a", "task": "t1", "epoch": 2, "uuid": "u"} | ledger_metrics({"rounds": 3}, None)])

    out = attach_ledgers(frame, reader)
    assert list(out["ledger_present"]) == [False, True, False] and out.loc[1, "ledger_rounds"] == 3 and len(out) == 3


# --- analyze_m5 on a run directory of real offline logs ---------------------------------------------------------------


@pytest.fixture(scope="module")
def m5_run(tmp_path_factory):
    """A run directory as `run_study --study m5` leaves it: a test manifest whose groups hold real offline M1 logs (B2's
    gold mock on F1-2 and F2-2), one group declared as M5 (its log carries a `mas_ledger` record), one as the M1 control;
    a freeze inheriting a main run."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import get_model

    from ape.build import build
    from ape.llm.mock_multi import GoldMulti
    from ape.tasks.main import main_study

    tmp = tmp_path_factory.mktemp("m5-run")
    with pytest.MonkeyPatch.context() as mp:
        for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
            mp.setenv(k, str(tmp / v))
        mp.setenv("APE_EMBEDDINGS", "fake")
        run_dir = tmp / "runs" / "m5" / "r1"
        logs: dict[str, list[str]] = {"M5": [], "M1": []}
        for fam in ("F1", "F2"):
            asyncio.run(build("dev", fam, ["2"], n_worlds=1, n_tasks=2, relational=True, embed=True))
        gold = GoldMulti(tmp / "worlds")
        for label in ("M5", "M1"):
            for fam in ("F1", "F2"):
                log = inspect_eval(main_study(family=fam, level="2", split="dev", arm="M1"), model=get_model(MODEL, custom_outputs=gold, memoize=False), model_roles={"kg": get_model(MODEL, custom_outputs=gold.kg, memoize=False)}, epochs=2, log_dir=str(run_dir / "test" / "logs" / label), display="none")[0]
                assert log.status == "success", log.error
                path = Path(log.location)
                if label == "M5":
                    full = read_eval_log(str(path))
                    for i, s in enumerate(full.samples):
                        s.store["mas_ledger"] = {"rounds": 4, "replans": i % 2, "stalls": 1}
                    write_eval_log(full, str(path))
                logs[label].append(str(path.relative_to(run_dir)))
    groups = [
        {"name": f"m5.test.a/{label}", "kind": "agent", "status": "done", "arms": [{"declared": label, "run": "M1"}], "skipped": [], "cells": ["F1-2", "F2-2"], "profile": "main_luna", "caps": {"F1-2": 4000, "F2-2": 4000}, "log_files": files, "log_dir": f"test/logs/{label}"}
        for label, files in logs.items()
    ]
    manifest = {"phase": "test", "status": "done", "primary_complete": True, "cells": {"m5.test.a": {"plan_phase": "test", "primary": True, "groups": groups}}}
    (run_dir / "test" / "manifest.json").write_text(json.dumps(manifest))
    freeze = {"study": "m5", "frozen_at": "2026-10-04T00:00:00+00:00", "rehearsal": True, "role": {"kind": "primary", "of": None},
              "inherited": {"main_run": "main-a1", "frozen_at": "2026-10-03T00:00:00+00:00", "test_seeds": {"base": 13000, "last": 13008}, "token_caps": {"F1-2": 4000, "F2-2": 4000}, "kg": {"arm": "APG-q", "system": "apg"}}}  # fmt: skip
    (run_dir / "freeze.json").write_text(json.dumps(freeze))
    return run_dir


def _fake_run(run_dir: Path, offline: bool = True) -> SimpleNamespace:
    return SimpleNamespace(study="m5", run_id=run_dir.name, offline=offline, dir=run_dir, phase_dir=lambda p: run_dir / p, freeze_path=run_dir / "freeze.json", models_path=ROOT / "config" / "models.yaml", inheritance_path=run_dir / "config" / "main_inheritance.json")


def test_analyze_writes_the_report_and_decision_from_a_run_directory(m5_run):
    from ape import analyze_m5

    result = analyze_m5.analyze(_fake_run(m5_run), reps=200)
    assert result["status"] == "done" and result["main_run"] == "main-a1" and result["section_errors"] == []
    assert set(result["labels"]) == {"L1.F2", "L1.F1", "L2.pooled"} and result["labels"]["L2.pooled"] == "not evaluable"
    d = json.loads((m5_run / "report" / "decision.json").read_text())
    json.dumps(d, allow_nan=False)
    a = d["analysis"]
    assert {(r["arm"], r["cell"]) for r in a["header"]["design"]} == {(x, c) for x in ("M5", "M1") for c in ("F1-2", "F2-2")}
    assert a["header"]["plan_cells"] == ["m5.test.a"] and all(c["concurrent"] for c in a["header"]["concurrency"])
    led = a["ledger"]["arms"]["M5"]["pooled"]
    assert a["ledger"]["available"] and led["with_record"] == led["samples"] == 8 and led["rounds_per_sample"] == 4 and led["stall_rate"] == 0.25
    assert "M1" not in a["ledger"]["arms"], "controls carry no ledger"
    assert {c["cell"]: c["frozen_cap"] for c in d["caps"]} == {"F1-2": 4000, "F2-2": 4000} and all(c["consistent"] for c in d["caps"])
    md = (m5_run / "report" / "report.md").read_text()
    for text in ("# M5 add-on study report: run `r1` (OFFLINE)", "Inherits main run `main-a1`", "## Coverage", "## Confirmatory tests", "## Ledger diagnostics"):
        assert text in md
    rows = read_ledgers([m5_run / f for f in json.loads((m5_run / "test" / "manifest.json").read_text())["cells"]["m5.test.a"]["groups"][0]["log_files"]])
    assert rows["ledger_present"].all() and set(rows["ledger_replans"]) == {0, 1}


def test_missing_inputs_are_reported_never_raised(tmp_path):
    from ape import analyze_m5

    run_dir = tmp_path / "runs" / "m5" / "empty"
    result = analyze_m5.analyze(_fake_run(run_dir), reps=100)
    assert result["status"] == "partial" and result["samples"] == 0
    joined = " ".join(result["problems"])
    assert "freeze.json missing" in joined and "manifest.json missing" in joined and "no inherited main run" in joined
    assert (run_dir / "report" / "report.md").is_file() and json.loads((run_dir / "report" / "decision.json").read_text())["analysis"]["summary"]


def test_the_inheritance_file_is_read_when_the_freeze_has_none(tmp_path):
    from ape import analyze_m5

    run_dir = tmp_path / "r"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "main_inheritance.json").write_text(json.dumps({"main_run": "m-7", "token_caps": {"F2-10": 9}, "selected": {"M1": {}}}))
    assert analyze_m5.inherited(_fake_run(run_dir), {"frozen_at": "x"}) == {"main_run": "m-7", "token_caps": {"F2-10": 9}}
    assert analyze_m5.inherited(_fake_run(run_dir), {"inherited": {"main_run": "m-8"}}) == {"main_run": "m-8"}
    assert analyze_m5.role_context({"role": {"kind": "extension", "of": "r0"}})["primary_only"] is True
