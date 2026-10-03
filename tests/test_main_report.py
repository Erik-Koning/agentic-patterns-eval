"""The main-study report (analysis/main_report.py) and its descriptive sections, on a synthetic study (tests/main_synth.py)
and on a mock run's logs: JSON-serialisable, complete, and never crashing on missing arms or cells."""

import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from main_synth import synthetic_frame

from ape.analysis import main_descriptive as desc
from ape.analysis import main_stats as ms
from ape.analysis.main_hypotheses import HYPOTHESES, confirmatory, hypothesis_table, render_hypothesis_table
from ape.analysis.main_report import main_report, render_main, report_from_logs, write_report

CONFIRMATORY = [m.id for h in confirmatory() for m in h.members]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame(seed=1)


@pytest.fixture(scope="module")
def report(frame) -> dict:
    return main_report(frame, reps=300, glmm=False)


def test_the_report_is_complete_and_json_serialisable(report):
    assert report["errors"] == []
    json.dumps(report, allow_nan=False)
    assert [s["member"] for s in report["summary"]] == CONFIRMATORY
    assert all(s["label"] in ("supported", "not supported", "not tested") for s in report["summary"]), "every member evaluable on a full study"
    assert set(report["descriptive"]) == {"K4", "M4", "mechanisms", "tiers"} and report["frontier"] and report["determinism"]["available"]
    tiers = {(t["cell"], t["contrast"]): t for t in report["descriptive"]["tiers"]}
    m1 = tiers[("F1-32", "M1 − S1")]
    assert m1["sol − luna"]["est"] == pytest.approx(m1["sol"]["est"] - m1["luna"]["est"]), "the DiD on the same tasks"
    md = render_main(report)
    for heading in ("## Confirmatory tests", "## S8 frontier", "## Cost-meter rank flips", "## Determinism", "## Invariants", "## Analysis choices"):
        assert heading in md
    assert all(m in md for m in CONFIRMATORY)


def test_strong_effects_are_supported_through_the_whole_path():
    df = synthetic_frame(seed=2, n_tasks=54, effects={"M1": 0.30, "S9": 0.0, "S5": 0.25, "S3s": 0.0})
    d = main_report(df, reps=300, glmm=False)
    labels = {s["member"]: s["label"] for s in d["summary"]}
    assert labels["M1.F1-32"] == "supported" and labels["K1.F7-1000"] == "supported", labels


def test_missing_arms_make_members_not_evaluable_and_the_report_still_stands():
    df = synthetic_frame(seed=3, drop_arms=("M7", "S9", "S3s"))
    d = main_report(df, reps=200, glmm=False)
    assert d["errors"] == []
    labels = {s["member"]: s for s in d["summary"]}
    for mid, arm in (("M1.F1-32", "S9"), ("M3.pooled", "M7"), ("K1.F7-1000", "S3s")):
        assert labels[mid]["label"] == "not evaluable" and arm in labels[mid]["reason"]
    render_main(d)


def test_an_empty_frame_gives_a_report_of_unevaluable_members():
    cols = synthetic_frame(seed=4, n_tasks=9).columns
    d = main_report(pd.DataFrame(columns=cols), reps=100, glmm=True)
    assert d["errors"] == [] and all(s["label"] == "not evaluable" for s in d["summary"])
    assert not d["determinism"]["available"]
    json.dumps(d, allow_nan=False)
    render_main(d)


def test_the_report_from_mock_logs(tmp_path, monkeypatch):
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ape.build import build
    from ape.llm.mock_agent import mock_agent
    from ape.tasks.main import main_study

    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    asyncio.run(build("dev", "F1", ["2"], n_worlds=2, n_tasks=2, relational=True, embed=False))
    logs = {arm: inspect_eval(main_study(family="F1", level="2", split="dev", arm=arm), model=get_model("mockllm/model", custom_outputs=mock_agent), epochs=2, log_dir=str(tmp_path / "logs"), display="none")[0].location for arm in ("S1", "S6")}
    d = report_from_logs({"main.A.s1-pool": [logs["S1"]], "main.A.arms": [logs["S6"]]}, require_cost=False, reps=100, glmm=False)
    assert d["errors"] == [] and d["header"]["rows"] == 16 and [c["samples"] for c in d["header"]["coverage"]] == [8, 8]
    j, m = write_report(d, tmp_path / "report")
    assert json.loads(j.read_text())["summary"] and m.read_text().startswith("# Main-study report")


# ---------------------------------------------------------------- descriptive sections


def test_k4_slopes_are_weighted_contrasts_over_log_kb_size(frame):
    t = ms.build_tables(frame)
    k4 = desc.k4_slopes(t, reps=200)
    s1 = k4["arms"]["S1"]
    means = s1["cell_means"]
    x = np.log10([10, 100, 1000])
    y = np.array([means["F7-10"], means["F7-100"], means["F7-1000"]])
    assert s1["slope"] == pytest.approx(np.polyfit(x, y, 1)[0]) and s1["ci_t"][0] < s1["slope"] < s1["ci_t"][1]
    assert k4["arms"]["S3s"]["cells"] == ["F7-10", "F7-1000"], "S3s has no F7-100 cell"


def test_rank_flips_need_meters_that_disagree(frame):
    t = ms.build_tables(frame)
    flips = desc.rank_flips(t, reps=200)
    assert flips["families_ranked"] == 4 and flips["rule_holds"], flips  # the synthetic meters disagree (main_synth)
    assert set(flips["families"]["F7"]["arms"]) >= {"S3s", "S7", "M1k"}, "F7-100 (four arms only) is not an MVS cell"
    same = frame.assign(usd=frame["tokens"], wall=frame["tokens"])
    t2 = ms.build_tables(same)
    flips2 = desc.rank_flips(t2, reps=200)
    assert not flips2["rule_holds"] and all(f["min_tau"] == pytest.approx(1.0) for f in flips2["families"].values())


def test_pass_hat_k_is_unbiased_and_determinism_uses_five_runs(frame):
    assert desc.pass_hat_k(3, 5, 2) == pytest.approx(3 / 10) and desc.pass_hat_k(5, 5, 5) == 1.0 and desc.pass_hat_k(4, 5, 5) == 0.0
    assert desc.pass_at_k(1, 5, 1) == pytest.approx(0.2)
    det = desc.determinism(frame, reps=200)
    runs = {k.split(" / ")[0]: v["runs_min"] for k, v in det["arms"].items()}
    assert runs["S1"] == runs["M1"] == runs["S5"] == 5 and runs["S8k3"] == 2
    assert {c["contrast"] for c in det["contrasts"]} >= {"S5 − M1", "S5 − M2"}


def test_invariants_check_the_delivery_chain(frame):
    inv = desc.invariants(ms.build_tables(frame), reps=200)
    assert inv["chain"] == ["S5", "S7"] and inv["tests"]["S5>=S7"]["pass"] and inv["s5_gt_s7"]["pass"]


def test_glmm_fits_or_reports_never_raises(monkeypatch):
    small = synthetic_frame(seed=6, n_tasks=9)
    g = desc.glmm(small, families=("F7",))
    assert g["families"]["F7"]["fitted"] and "arm[S5]" in g["families"]["F7"]["fixed"]
    from statsmodels.genmod import bayes_mixed_glm

    monkeypatch.setattr(bayes_mixed_glm.BinomialBayesMixedGLM, "fit_vb", lambda self, *a, **k: (_ for _ in ()).throw(np.linalg.LinAlgError("singular")))
    g = desc.glmm(small, families=("F7", "F9"))
    assert not g["families"]["F7"]["fitted"] and "LinAlgError" in g["families"]["F7"]["reason"]
    assert g["families"]["F9"]["reason"] == "fewer than 2 arms"


# ---------------------------------------------------------------- the hypothesis table


def test_the_hypothesis_table_is_the_single_source_and_follows_d028():
    rows = hypothesis_table()
    ids = [r["member"] for r in rows]
    assert len(ids) == len(set(ids))
    status = {h.id: h.status for h in HYPOTHESES}
    assert status["M4"] == status["K4"] == "descriptive" and "K3" not in status, "D-028: M4 and K4 descriptive, K3 dropped"
    assert all(r["test"] in ("superiority", "less", "ni", "tost") for r in rows if r["status"] == "confirmatory")
    assert all(r["margin"] == 0.03 for r in rows if r["test"] in ("ni", "tost"))
    md = render_hypothesis_table()
    assert all(h.id in md for h in HYPOTHESES)
    assert {line.count("|") for line in md.strip().splitlines()} == {10}, "no cell breaks the markdown table"
    json.dumps(rows, allow_nan=False)
