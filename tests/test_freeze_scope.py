"""Freeze scope (`ape.freeze_scope`): the gate and each study fingerprint and freeze their own slice of the shared
config files, so a frozen gate run survives later edits to the main study's or Study G's cells, profiles, prices and
allocations, and a frozen study survives the others'. Budget totals, allocations and concurrency are never frozen (R-B3).
Offline: config copies only."""

import shutil

import pytest
import yaml

from ape import run_gate, run_study
from ape.config import ROOT
from ape.freeze_scope import ConfigSlice, config_slice, study_profiles
from ape.run_gate import GateRun
from ape.run_study import StudyRun


@pytest.fixture
def config(tmp_path):
    d = tmp_path / "config"
    shutil.copytree(ROOT / "config", d)
    return d


def _edit(path, fn):
    data = yaml.safe_load(path.read_text())
    fn(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _cell(plan: dict, cell_id: str) -> dict:
    return next(c for s in plan["studies"].values() for cells in s["phases"].values() for c in cells if c["id"] == cell_id)


def _record(files: dict) -> dict:
    return {"files": {k: run_gate._file_entry(p) for k, p in files.items()}}


def test_a_frozen_gate_survives_edits_to_the_other_studies_and_refuses_edits_to_its_own(tmp_path, config):
    run = GateRun("scope", offline=True, runs_root=tmp_path / "runs", config_dir=config)
    files = {k: p for k, p in run_gate.frozen_files(run, ROOT / "GATE_PREREG.md").items() if k.startswith("config/") and k[7:] in run_gate.FROZEN_CONFIG}
    assert {k for k, p in files.items() if isinstance(p, ConfigSlice)} == {"config/run_plan.yaml", "config/models.yaml", "config/model_costs.yaml"}
    record = _record(files)
    plan_entry = record["files"]["config/run_plan.yaml"]
    assert plan_entry["slice"] == "gate" and plan_entry["file_sha256"] == run_gate._sha256(config / "run_plan.yaml") != plan_entry["sha256"]
    preflight = run_gate.phase_state(run, "preflight")["fingerprint"]

    # The main study and Study G change: cells, an allocation, the G block, their profiles, a price only G calls.
    _edit(config / "run_plan.yaml", lambda p: (_cell(p, "main.F.luna").update(n_tasks=60), _cell(p, "g.topo.luna").update(sessions=20),
                                              p["budget"]["allocations"].update(main=1200, study_g=3500, gate=350), p["study_g"].update(window=48000)))
    _edit(config / "models.yaml", lambda m: m["profiles"]["main_sol"]["agent"].update(reasoning_effort="medium"))
    _edit(config / "model_costs.yaml", lambda c: c["openai/gpt-6-astra"].update(input=12.0))
    with (config / "run_plan.yaml").open("a") as f:
        f.write("\n# a comment is no input\n")
    assert run_gate.frozen_changes(record) == [], "nothing the gate runs on changed"
    assert run_gate._file_entry(files["config/run_plan.yaml"])["file_sha256"] != plan_entry["file_sha256"], "the whole file did change (recorded, never enforced)"
    assert run_gate.phase_state(run, "preflight")["fingerprint"] == preflight, "the gate's phases do not re-run either"

    # The gate's own cell, profile or prices do break it, by name.
    _edit(config / "run_plan.yaml", lambda p: _cell(p, "gate.test.f7").update(epochs=2))
    _edit(config / "models.yaml", lambda m: m["profiles"]["anchor_luna"]["judge"].update(reasoning_effort="high"))
    _edit(config / "model_costs.yaml", lambda c: c["openai/gpt-6-sol"].update(input=3.0))  # the gate's D-017 fallback builder
    assert [c.split(" (", 1)[0] for c in run_gate.frozen_changes(record)] == ["config/models.yaml", "config/model_costs.yaml", "config/run_plan.yaml"]
    assert all(c.endswith(", gate slice): changed") for c in run_gate.frozen_changes(record))
    assert run_gate._entry_path(record["files"]["config/run_plan.yaml"]).study == "gate"


def test_each_study_freezes_its_own_slice(tmp_path, config):
    main = StudyRun("main", "m", offline=True, runs_root=tmp_path / "runs", config_dir=config)
    g = StudyRun("study_g", "g", offline=True, runs_root=tmp_path / "runs", config_dir=config)
    before = {s: {n: config_slice(config / n, s) for n in ("run_plan.yaml", "models.yaml", "model_costs.yaml")} for s in ("gate", "main", "study_g")}
    assert set(before["main"]["models.yaml"]) == {"main_luna", "main_sol"} and set(before["study_g"]["models.yaml"]) == {"study_g_luna", "study_g_sol", "study_g_astra"}
    assert "openai/gpt-6-astra" in before["study_g"]["model_costs.yaml"] and "openai/gpt-6-astra" not in before["main"]["model_costs.yaml"]
    assert all(set(before[x]["run_plan.yaml"]["budget"]) <= {"sample_cost_limit", "sample_working_limit"} for x in before), "R-B3: no allocation or total"
    assert all("concurrency" not in (p or {}) for x in before for p in before[x]["models.yaml"].values()), "R-B3: concurrency is not design"
    assert "study_g" in before["study_g"]["run_plan.yaml"] and "study_g" not in before["main"]["run_plan.yaml"]
    assert study_profiles(yaml.safe_load((config / "run_plan.yaml").read_text()), "gate") == ["gate", "anchor", "anchor_luna"]
    main_inputs = run_study._cfg_inputs(main, "run_plan.yaml", "models.yaml", main.grid_name)
    assert isinstance(main_inputs["config/run_plan.yaml"], ConfigSlice) and not isinstance(main_inputs[f"config/{main.grid_name}"], ConfigSlice)
    main_record, g_record = _record(main_inputs), _record(run_study._cfg_inputs(g, "run_plan.yaml", "model_costs.yaml"))

    _edit(config / "run_plan.yaml", lambda p: (p["study_g"].update(threshold=18000), _cell(p, "g.cm.luna-high").update(sessions=12)))
    _edit(config / "model_costs.yaml", lambda c: c["openai/gpt-6-astra"].update(output=60.0))
    assert run_gate.frozen_changes(main_record) == [], "Study G's edits leave the main study frozen"
    changed = run_gate.frozen_changes(g_record)
    assert [c.split(" (", 1)[0] for c in changed] == ["config/run_plan.yaml", "config/model_costs.yaml"] and all(c.endswith(", study_g slice): changed") for c in changed)
    # R-B3: the budget guard's inputs (the total, the allocations) and the profiles' concurrency are not design: a
    # budget or rate-limit adjustment after the freeze never blocks the test. The per-sample runaway guard is.
    _edit(config / "run_plan.yaml", lambda p: (p["budget"]["allocations"].update(main=900), p["budget"].update(total_usd=4000)))
    _edit(config / "models.yaml", lambda m: m["profiles"]["main_luna"].update(concurrency={"max_connections": 4, "max_samples": 8, "max_tasks": 1}))
    assert run_gate.frozen_changes(main_record) == [], "allocations, the total and concurrency are guard inputs"
    _edit(config / "run_plan.yaml", lambda p: p["budget"]["sample_cost_limit"].update(multiple=30))
    changed = run_gate.frozen_changes(main_record)
    assert len(changed) == 1 and changed[0].startswith("config/run_plan.yaml (") and changed[0].endswith(", main slice): changed"), "the runaway guard is frozen"


def test_the_m5_add_on_study_freezes_its_own_slice(tmp_path, config):
    """The M5 add-on study's slice is its own section (and the sample rules it inherits); main's and Study G's never see
    it, so enabling or editing the off-by-default m5 cells leaves a frozen main run intact."""
    main_record = _record(run_study._cfg_inputs(StudyRun("main", "m", offline=True, runs_root=tmp_path / "runs", config_dir=config), "run_plan.yaml", "models.yaml"))
    m5_slice = config_slice(config / "run_plan.yaml", "m5")
    assert set(m5_slice["studies"]) == {"m5"} and set(m5_slice["budget"]) == {"sample_cost_limit", "sample_working_limit"}
    assert set(config_slice(config / "models.yaml", "m5")) == {"main_luna"}
    m5_record = _record({"config/run_plan.yaml": run_study.config_input(config / "run_plan.yaml", "m5")})
    _edit(config / "run_plan.yaml", lambda p: [c.update(enabled=True) for cells in p["studies"]["m5"]["phases"].values() for c in cells])
    assert run_gate.frozen_changes(main_record) == [] and run_gate.frozen_changes(m5_record)
