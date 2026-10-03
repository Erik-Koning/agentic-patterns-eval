"""The main study's analyze phase (ape.analyze_main) on a real offline `run_study all --study main` run: the report and
decision files, arms labelled by plan name, tiers from the groups' profiles, coverage and caps; and every missing or
broken input reported, never raised."""

import json
import os
import shutil

import pytest

from ape import analyze_main as am
from ape import run_study
from ape.analysis.main_hypotheses import confirmatory
from ape.config import ROOT
from ape.run_study import StudyRun

CONFIRMATORY = [m.id for h in confirmatory() for m in h.members]


@pytest.fixture(scope="module")
def offline_main(tmp_path_factory):
    """One offline `all` of the main study, whose analyze phase is this module."""
    tmp = tmp_path_factory.mktemp("main-run")
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    with pytest.MonkeyPatch.context() as mp:
        for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
            mp.delenv(k)
        code = run_study.main(["all", "--study", "main", "--run-id", "a1", "--offline", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)])
        yield {"tmp": tmp, "config": config, "runs": tmp / "runs", "code": code}


def _run(root, config, run_id="a1") -> StudyRun:
    return StudyRun("main", run_id, offline=True, runs_root=root, config_dir=config)


def _copy(offline_main, tmp_path, run_id="a1") -> StudyRun:
    """A copy of the offline run to break inputs in (log paths in the manifest still point at the original's logs)."""
    shutil.copytree(offline_main["runs"] / "main" / "a1", tmp_path / "runs" / "main" / run_id)
    return _run(tmp_path / "runs", offline_main["config"], run_id)


def test_the_analyze_phase_writes_the_report_and_the_decision(offline_main):
    assert offline_main["code"] == 0
    run = _run(offline_main["runs"], offline_main["config"])
    manifest = json.loads(run.manifest_path("analyze").read_text())
    assert manifest["status"] == "done" and manifest["analysis"]["status"] == "done" and manifest["analysis"]["problems"] == []
    assert set(manifest["outputs"]) == {"decision.json", "report.md"} and manifest["analysis"]["section_errors"] == []
    d = json.loads((run.dir / "report" / "decision.json").read_text())
    json.dumps(d, allow_nan=False)
    assert [s["member"] for s in d["analysis"]["summary"]] == CONFIRMATORY and set(manifest["analysis"]["labels"]) == set(CONFIRMATORY)
    md = (run.dir / "report" / "report.md").read_text()
    for heading in ("## Run", "## Coverage", "## Token caps and cap hits", "## Tuning", "## Confirmatory tests", "## Analysis choices"):
        assert heading in md
    test = json.loads(run.manifest_path("test").read_text())
    assert {c["plan_cell"] for c in d["coverage"]} == set(test["cells"]) and all(c["samples"] > 0 for c in d["coverage"] if c["status"] == "done")


def test_arms_carry_their_plan_names_and_tiers_their_profiles(offline_main):
    run = _run(offline_main["runs"], offline_main["config"])
    frame, coverage = am.load_test_rows(json.loads(run.manifest_path("test").read_text()), require_cost=False, tiers=am.profile_tiers(run.models_path), run_dir=run.dir)
    assert {"S1", "S5", "S7", "S3s", "M1", "M1k", "M2", "M7", "S9", "M1s", "S8k3"} <= set(frame["arm"])
    assert (frame["arm"] == frame["run_arm"]).all(), "the main study's tuned arms run under their own names; S5 is resolved by the task"
    sol = frame[frame["plan_cell"].str.startswith("main.F.sol")]
    assert len(sol) and (sol["tier"] == "sol").all() and (frame.loc[~frame["plan_cell"].str.startswith("main.F.sol"), "tier"] == "luna").all()
    assert not ((frame["arm"] == "M2") & (frame["cell"] == "F1-32")).any(), "D-034: no M2 on F1-32"
    d = json.loads((run.dir / "report" / "decision.json").read_text())
    freeze = json.loads(run.freeze_path.read_text())
    caps = {c["cell"]: c for c in d["caps"]}
    assert all(caps[tc]["frozen_cap"] == cap and caps[tc]["consistent"] for tc, cap in freeze["token_caps"].items())
    assert d["header"]["kg"]["arm"] == freeze["kg"]["arm"] and d["header"]["role"]["kind"] == "primary"


def test_missing_inputs_are_reported_never_raised(offline_main, tmp_path):
    run = _copy(offline_main, tmp_path)
    test_path = run.manifest_path("test")
    test = json.loads(test_path.read_text())
    victim = next(c for c in test["cells"].values() if c["groups"][0]["log_files"])["groups"][0]
    victim["log_files"] = [str(tmp_path / "gone.eval")]
    clash = test["cells"]["main.F.sol"]["groups"][0]
    clash["arms"] = [{"declared": "S1", "run": "S1"}, {"declared": "S5", "run": "S1"}]
    test_path.write_text(json.dumps(test))
    result = am.analyze(run, reps=200, glmm=False)
    assert result["status"] == "done" and any("log files missing" in p for p in result["problems"]) and any("two declared arms ran as one" in p for p in result["problems"])
    assert (run.dir / "report" / "report.md").is_file()
    run.freeze_path.unlink()
    test_path.unlink()
    result = am.analyze(run, reps=200, glmm=False)
    assert result["status"] == "partial" and result["samples"] == 0 and set(result["labels"].values()) == {"not evaluable"}
    assert any("not frozen" in p for p in result["problems"]) and any("test phase has not run" in p for p in result["problems"])
    json.loads((run.dir / "report" / "decision.json").read_text())


def test_an_extension_without_a_preregistered_alpha_is_a_primary_only_analysis(offline_main, tmp_path):
    run = _copy(offline_main, tmp_path, "ext")
    freeze = json.loads(run.freeze_path.read_text())
    run.freeze_path.write_text(json.dumps(freeze | {"role": {"kind": "extension", "of": "a1"}}))
    result = am.analyze(run, reps=200, glmm=False)
    assert result["role"] == "extension" and result["primary_only"] and any("defines no extension α" in p for p in result["problems"])
    assert "primary-only analysis" in (run.dir / "report" / "report.md").read_text()
    assert am.role_context({"role": {"kind": "primary", "of": None}})["alpha"] is None


def test_the_cli_analyses_a_run(offline_main, tmp_path, capsys):
    run = _copy(offline_main, tmp_path, "cli")
    assert am.main(["--run-id", "cli", "--offline", "--runs-dir", str(tmp_path / "runs"), "--config-dir", str(offline_main["config"]), "--reps", "200"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "done" and (run.dir / "report" / "decision.json").is_file()


def test_tuning_summary_counts_candidates_and_selections(tmp_path):
    log = tmp_path / "tuning_log.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in [
        {"system": "M1", "candidate": {"id": "m1-a"}, "status": "ok", "mean_success": 0.5},
        {"system": "M1", "candidate": {"id": "m1-b"}, "status": "failed", "mean_success": None},
        {"system": "M1", "selected": "m1-a"},
        {"system": "S3s", "candidate": {"id": "s3s-1"}, "status": "ok", "mean_success": 0.4},
    ]) + "\nnot json\n")  # fmt: skip
    t = am.tuning_summary(log)
    assert t["systems"]["M1"] == {"candidates": 2, "failed": 1, "selected": "m1-a", "results": {"m1-a": 0.5, "m1-b": None}}
    assert not t["equal_candidates"] and not am.tuning_summary(tmp_path / "none.jsonl")["available"]
