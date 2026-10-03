"""Study G's analysis phase (`ape.analyze_g.analyze`, D-036's interface): an offline `run_study all --study study_g`
end to end into report/decision.json and report/report.md, and the glue's handling of missing inputs, arm labels and
coverage. Offline: mock models, fake embeddings; every run lives under a temporary directory."""

import json
import os
import shutil

import pandas as pd
import pytest

from ape import analyze_g
from ape.config import ROOT
from ape.run_study import StudyRun, main


@pytest.fixture(scope="module")
def offline_g(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study_g")
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    with pytest.MonkeyPatch.context() as mp:
        for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
            mp.delenv(k)
        code = main(["all", "--study", "study_g", "--run-id", "t1", "--offline", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)])
        yield {"code": code, "run": StudyRun("study_g", "t1", offline=True, runs_root=tmp / "runs", config_dir=config), "tmp": tmp, "config": config}


def _manifest(run: StudyRun, phase: str) -> dict:
    return json.loads(run.manifest_path(phase).read_text())


def test_the_offline_study_g_run_is_analysed_end_to_end(offline_g):
    run = offline_g["run"]
    assert offline_g["code"] == 0
    am = _manifest(run, "analyze")
    assert am["status"] == "done" and am["analysis"]["status"] in ("ok", "partial")
    assert {"decision.json", "report.md"} <= set(am["outputs"])
    d = json.loads((run.dir / "report" / "decision.json").read_text())
    md = (run.dir / "report" / "report.md").read_text()
    assert md.startswith("# Study G report: run `t1` (OFFLINE") and "## Coverage" in md and "## Decisions" in md
    # Every confirmatory row is decided or says why not; the descriptive rows are labelled so.
    dec = {r["id"]: r["decision"] for r in d["decisions"]}
    assert dec["G-H1"] == "descriptive" and dec["G-H2a"] in ("SUPPORTED", "NOT_SUPPORTED", "NOT_TESTABLE")
    assert am["analysis"]["decisions"] == dec
    # Coverage: every arm the runner skipped is reported as skipped with its reason, and every plan arm has a row.
    test = _manifest(run, "test")
    skipped = {(s["cell"], s["declared"]) for s in test["skipped_arms"]}
    cov = {(r["cell"], r["arm"]): r for r in d["coverage"]}
    # Since B9 every planned session arm is built, so an offline run may skip nothing; any skip must be reported.
    assert all(cov[k]["state"] == "skipped" and cov[k]["reason"] for k in skipped)
    assert not {a for _, a in skipped} & {"S1", "M1", "M2"}, "B9 built the topology arms"
    ran = {(c, a["declared"]) for c, x in test["cells"].items() for g in x["groups"] if g["status"] == "done" for a in g["arms"]}
    assert ran and all(cov[k]["state"] == "present" and cov[k]["present"] > 0 for k in ran)
    # Arms carry their plan names; capability comes from the anchor cells, one point per profile and effort.
    plan_arms = {a for r in d["coverage"] for a in [r["arm"]]}
    assert {c["arm"] for c in d["outcomes"]} <= plan_arms
    assert {p["point"] for p in d["capability"]["points"]} == {"luna-low", "luna-high", "sol-high", "astra-high"}
    assert d["run"]["freeze"]["role"]["kind"] == "primary" and d["run"]["test"]["status"] == "done"
    assert d["selections"]["selected"] and d["selections"]["tuning"]


def test_missing_inputs_are_reported_not_raised(tmp_path):
    run = StudyRun("study_g", "empty", offline=True, runs_root=tmp_path / "runs")
    s = analyze_g.analyze(run, reps=200, boot=0, glmm=False)
    assert s["status"] == "no_data" and s["session_epochs"] == 0
    assert any("test manifest" in p for p in s["problems"]) and any("freeze.json" in p for p in s["problems"])
    md = (run.dir / "report" / "report.md").read_text()
    assert "NOT FROZEN" in md and "not run" in md
    d = json.loads((run.dir / "report" / "decision.json").read_text())
    assert all(r["decision"] in ("NOT_TESTABLE", "descriptive") for r in d["decisions"])
    assert d["coverage"] and all(r["state"] == "not run" for r in d["coverage"])


def test_a_failed_group_and_a_lost_log_are_reported(offline_g, tmp_path):
    """A copy of the offline run whose test manifest says one group failed and points one log at a file that is gone."""
    src = offline_g["run"]
    dst_root = tmp_path / "runs"
    shutil.copytree(src.dir, dst_root / "study_g" / "t1")
    run = StudyRun("study_g", "t1", offline=True, runs_root=dst_root, config_dir=offline_g["config"])
    m = _manifest(run, "test")
    g = m["cells"]["g.cm.sol-high"]["groups"][0]
    g |= {"status": "failed", "error": "RuntimeError: boom", "log_files": ["gone/never.eval"]}
    m["cells"]["g.cm.sol-high"]["status"] = "failed"
    m["status"] = "failed"
    run.manifest_path("test").write_text(json.dumps(m))
    s = analyze_g.analyze(run, reps=200, boot=0, glmm=False)
    assert s["status"] == "partial"
    assert any("g.cm.sol-high (selected): group failed: RuntimeError: boom" in p for p in s["problems"])
    assert any("log file not found (gone/never.eval)" in p for p in s["problems"])
    assert any(x.startswith("g.cm.sol-high:") for x in s["missing"])


def test_arms_are_relabelled_to_their_plan_names():
    df = pd.DataFrame({"plan_cell": ["c", "c", "d"], "arm": ["CM-sum", "CM0", "CM-sum"], "x": [1, 2, 3]})
    out = analyze_g.relabel(df, {"c": {"CM-sum": ["S-CM*", "CM-sum"], "CM0": ["CM0"]}})
    assert sorted(out[out["plan_cell"] == "c"]["arm"]) == ["CM-sum", "CM0", "S-CM*"]  # a shared run arm under both names
    assert out.loc[out["arm"] == "S-CM*", "run_arm"].tolist() == ["CM-sum"]
    assert out[out["plan_cell"] == "d"]["arm"].tolist() == ["CM-sum"]  # unmapped: as it ran
    assert analyze_g.relabel(pd.DataFrame(), {}).empty
