"""PREREGISTRATION_G.md (BUILD_PLAN B11) against the code it describes.

- Its generated tables equal `g_hypotheses.markdown()` and `design_markdown()` byte for byte.
- Its prose covers every hypothesis row, the confirmatory ones first among them.
- Every marker in its body is well formed, and every `[PILOT: <label>]` is one the runner fills for Study G.
- An offline `run_study --study study_g` freeze rehearsal accepts the file.

Offline: mock models; the run lives under a temporary directory."""

import json
import os
import shutil

import pytest
import yaml

from ape import run_gate
from ape.analysis import g_hypotheses as gh
from ape.config import ROOT
from ape.run_study import StudyRun, main, read_freeze

PREREG = ROOT / "PREREGISTRATION_G.md"
GENERATED = {"g_hypotheses.markdown()": gh.markdown, "g_hypotheses.design_markdown()": gh.design_markdown}
PHASES = ("preflight", "build-dev", "micro-pilot", "tune", "freeze")  # every phase the freeze rests on, then the freeze


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->\n", f"<!-- END GENERATED: {name} -->"


def _block(text: str, name: str) -> str:
    begin, end = _markers(name)
    assert text.count(begin) == 1 and text.count(end) == 1, f"PREREGISTRATION_G.md needs exactly one {name} block"
    start = text.index(begin) + len(begin)
    return text[start : text.index(end, start)]


def _prose(text: str) -> str:
    """The body (from the first `## ` heading) without the generated blocks."""
    body = text[run_gate.prereg_body_start(text) :]
    for name in GENERATED:
        begin, end = _markers(name)
        body = body[: body.index(begin)] + body[body.index(end) + len(end) :]
    return body


@pytest.fixture(scope="module")
def text() -> str:
    return PREREG.read_text()


@pytest.fixture(scope="module")
def offline_g(tmp_path_factory):
    """One offline Study G run through its freeze, on the real PREREGISTRATION_G.md (the CLI's default)."""
    tmp = tmp_path_factory.mktemp("prereg_g")
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    with pytest.MonkeyPatch.context() as mp:
        for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
            mp.delenv(k)
        codes = {p: main([p, "--study", "study_g", "--run-id", "prereg", "--offline", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]) for p in PHASES}
    yield {"codes": codes, "run": StudyRun("study_g", "prereg", offline=True, runs_root=tmp / "runs", config_dir=config)}


def test_the_generated_tables_are_the_code(text):
    body_start = run_gate.prereg_body_start(text)
    for name, fn in GENERATED.items():
        assert _block(text, name) == fn(), f"{name} changed: paste its output into PREREGISTRATION_G.md between its markers"
        assert text.index(_markers(name)[0]) > body_start, f"{name} belongs in the body"


def test_the_prose_covers_every_hypothesis_and_the_harness_window(text):
    prose = _prose(text)
    missing = [h.id for h in gh.confirmatory() if h.id not in prose]
    assert not missing, f"confirmatory rows without prose: {missing}"
    assert not [h.id for h in gh.HYPOTHESES if h.id not in prose], "every row, descriptive ones included, is described in the prose"
    # D-040: the topology contrast is measured against the harness-enforced window, and the document says so.
    assert "harness-enforced window" in prose and "D-040" in prose
    # The fixed sequence and the one-sided level are stated as the code has them.
    assert "G-H3-pre → G-H3a → G-H3b" in prose and f"{gh.ALPHA:g}" in prose


def test_the_plan_the_document_describes_is_the_plan(text):
    plan = yaml.safe_load((ROOT / "config" / "run_plan.yaml").read_text())
    window, threshold = plan["study_g"]["window"], plan["study_g"]["threshold"]
    prose = _prose(text)
    assert f"**{window:,} tokens**" in prose and f"**{threshold:,} tokens**" in prose
    cells = [c for cells in plan["studies"]["study_g"]["phases"].values() for c in cells if c.get("enabled", True)]
    assert cells and not [c["id"] for c in cells if f"`{c['id']}`" not in prose], "every enabled Study G plan cell is in the document"


def test_markers_are_well_formed_and_pilot_labels_are_the_runners(text, offline_g):
    placeholders = run_gate.prereg_placeholders(text)
    assert all(p["label"] for p in placeholders), f"malformed markers: {[p for p in placeholders if not p['label']]}"
    run = offline_g["run"]
    assert offline_g["codes"]["micro-pilot"] == 0
    # Study G has no pilot phase: the runner fills [PILOT: label] items from the micro-pilot summary's prereg_items.
    filled = set((json.loads((run.phase_dir("micro-pilot") / "micro_pilot.json").read_text()) or {}).get("prereg_items") or {})
    pilot = {p["label"] for p in placeholders if p["kind"] == "PILOT"}
    assert pilot <= filled, f"[PILOT: …] labels no Study G phase fills: {sorted(pilot - filled)}"


def test_an_offline_freeze_rehearsal_accepts_the_file(text, offline_g):
    run = offline_g["run"]
    assert offline_g["codes"] == dict.fromkeys(PHASES, 0)
    freeze = read_freeze(run)
    assert freeze is not None
    # The file frozen (a draft: beside its filled copy) is this repository's PREREGISTRATION_G.md.
    assert freeze["files"]["PREREGISTRATION_G.md (draft)" if freeze["rehearsal"] else "PREREGISTRATION_G.md"]["sha256"] == run_gate._sha256(PREREG)
    placeholders = run_gate.prereg_placeholders(text)
    if placeholders:  # a draft: the rehearsal froze a copy with every marker filled
        assert freeze["rehearsal"] is True
        assert [r["marker"] for r in freeze["placeholders_replaced"]] == [p["marker"] for p in placeholders]
        copy = run.work_dir / "PREREGISTRATION_G.md"
        assert copy.is_file() and run_gate.prereg_placeholders(copy.read_text()) == []
    else:  # filled for the real freeze
        assert freeze["rehearsal"] is False
    assert {"config/tuning_grid_study_g.yaml", "config/selected.yaml", "src/ape/analysis/g_hypotheses.py", "src/ape/analyze_g.py", "uv.lock"} <= set(freeze["files"])
    assert freeze["role"] == {"kind": "primary", "of": None} and freeze["test_seeds"]["base"] == 29000
