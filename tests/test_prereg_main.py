"""PREREGISTRATION_MAIN.md (BUILD_PLAN B5) against the code it pre-registers: the hypothesis table is the code's table
verbatim, every hypothesis and mechanism contrast the code tests is in the prose, every `[PILOT: label]` is one the
study runner fills (or one it is known not to fill yet), an offline `run_study` freeze rehearsal accepts the file, and
a live freeze refuses the draft, naming every marker, until each is filled. Each test holds for the draft and for the
filled, frozen file. Offline: mock models, no network; every run lives under a temporary directory."""

import json
import os
import re
import shutil

import pytest
import yaml

from ape import run_gate, run_study
from ape.analysis.main_hypotheses import HYPOTHESES, MECHANISM_CONTRASTS, NI_MARGIN, TOST_MARGIN, confirmatory, render_hypothesis_table
from ape.config import ROOT
from ape.run_study import PhaseError, StudyRun, read_freeze, run_phases

PREREG = ROOT / "PREREGISTRATION_MAIN.md"
BEGIN = "<!-- BEGIN GENERATED: main_hypotheses.render_hypothesis_table() -->\n"
END = "<!-- END GENERATED: main_hypotheses.render_hypothesis_table() -->"
# Labels the pre-registration uses that `run_study`'s pilot does not put in pilot.json's `prereg_items` yet (none: the
# B1 integration writes the D-039 cap multiple and the power re-simulation at the gate pilot's σ, §2.6). A label added
# here is transcribed by the analyst, and an offline rehearsal fills it as "unfilled".
PENDING_RUNNER_LABELS: set[str] = set()
# The decisions PREREGISTRATION_MAIN leaves to the user: the role names (§5; O-3). A new `[USER: …]` item is a new
# pending decision: add it here deliberately.
USER_DECISIONS = {"M-arm prompt author", "skeptic", "analyst"}
REHEARSAL_PHASES = ("preflight", "build-dev", "micro-pilot", "tune", "pilot", "freeze")


def _text() -> str:
    return PREREG.read_text()


def _split(text: str) -> tuple[str, str, str]:
    """(before the generated table, the table, after it)."""
    assert text.count(BEGIN) == 1 and text.count(END) == 1, "exactly one generated hypothesis table"
    head, rest = text.split(BEGIN)
    table, tail = rest.split(END)
    return head, table, tail


def _labels(text: str, kind: str) -> set[str]:
    return {p["label"] for p in run_gate.prereg_placeholders(text) if p["kind"] == kind}


# --- The document against the code ------------------------------------------------------------------


def test_the_hypothesis_table_is_the_codes_table_verbatim():
    text = _text()
    head, table, _ = _split(text)
    assert table == render_hypothesis_table(), "paste render_hypothesis_table()'s output between the BEGIN and END comments"
    assert len(head) > run_gate.prereg_body_start(text) > 0, "the table is in the body (from the first '## ' heading on), where the freeze checks"


def test_every_hypothesis_member_and_mechanism_contrast_is_in_the_prose():
    head, _, tail = _split(_text())
    prose = head + tail

    def named(token: str) -> bool:
        return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w-])", prose) is not None

    missing = [h.id for h in HYPOTHESES if not named(h.id)]
    missing += [m.id for h in confirmatory() for m in h.members if not named(m.id)]
    assert not missing, f"not described outside the generated table: {missing}"
    for margin in (f"±{100 * TOST_MARGIN:g} pp", f"{100 * NI_MARGIN:g} pp"):
        assert margin in prose, f"the prose states the code's margin {margin}"
    for name, a, b, _ in MECHANISM_CONTRASTS:
        assert f"| {name} | {a} − {b} |" in prose, f"mechanism contrast {name} ({a} − {b}) is in §2.5's table"


def test_markers_are_well_formed_and_user_items_are_the_pending_decisions():
    text = _text()
    malformed = [p for p in run_gate.prereg_placeholders(text) if not p["label"]]
    assert not malformed, f"markers without a label (the rehearsal cannot fill them): {malformed}"
    assert _labels(text, "USER") <= USER_DECISIONS, f"undeclared user decisions: {sorted(_labels(text, 'USER') - USER_DECISIONS)}"


# --- The runner: its pilot labels and an offline freeze rehearsal -----------------------------------------


@pytest.fixture(scope="module")
def rehearsal(tmp_path_factory):
    """An offline main run through the freeze (the runner's offline rehearsal of this pre-registration)."""
    tmp = tmp_path_factory.mktemp("prereg-main")
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    with pytest.MonkeyPatch.context() as mp:
        for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
            mp.delenv(k)
        run = StudyRun("main", "prereg", offline=True, runs_root=tmp / "runs", config_dir=config)
        statuses = {phase: run_phases(run, phase)[phase] for phase in REHEARSAL_PHASES}
        yield {"run": run, "statuses": statuses}


def test_every_pilot_label_is_one_the_runner_fills(rehearsal):
    items = json.loads((rehearsal["run"].phase_dir("pilot") / "pilot.json").read_text())["prereg_items"]
    runner, used = set(items), _labels(_text(), "PILOT")
    assert used - runner <= PENDING_RUNNER_LABELS, f"[PILOT: …] labels no runner fills: {sorted(used - runner - PENDING_RUNNER_LABELS)}"
    if used:  # a draft: every value the runner reports has a place in the document
        assert runner <= used, f"the runner fills labels this pre-registration does not use: {sorted(runner - used)}"


def test_an_offline_freeze_rehearsal_accepts_the_file(rehearsal):
    run, text = rehearsal["run"], _text()
    assert set(rehearsal["statuses"].values()) == {"done"}, rehearsal["statuses"]
    freeze = read_freeze(run)
    assert freeze["study"] == "main" and freeze["role"] == {"kind": "primary", "of": None}
    markers = run_gate.prereg_placeholders(text)
    if not markers:  # the filled file freezes as it is
        assert freeze["rehearsal"] is False and freeze["files"]["PREREGISTRATION_MAIN.md"]["sha256"] == run_gate._sha256(PREREG)
        return
    assert freeze["rehearsal"] is True and freeze["files"]["PREREGISTRATION_MAIN.md (draft)"]["sha256"] == run_gate._sha256(PREREG)
    items = json.loads((run.phase_dir("pilot") / "pilot.json").read_text())["prereg_items"]
    replaced = {m["marker"]: m["value"] for m in freeze["placeholders_replaced"]}
    assert len(freeze["placeholders_replaced"]) == len(markers)
    for p in markers:
        assert replaced[p["marker"]] == items.get(p["label"], "unfilled"), p["marker"]
    filled = (run.work_dir / "PREREGISTRATION_MAIN.md").read_text()
    assert run_gate.prereg_placeholders(filled) == [], "the rehearsal copy has no marker left"
    assert _split(filled)[1] == render_hypothesis_table(), "filling the markers leaves the generated table untouched"
    assert "OFFLINE-REHEARSAL" not in _text(), "the rehearsal never writes the real file"


# --- A live freeze refuses the draft ------------------------------------------------------------------


def _fake_gate_run(root, gid: str) -> None:
    """A frozen, analysed GO gate run, as run_gate leaves it (tests/test_run_study.py's pattern)."""
    d = root / gid
    run_gate._write_json(d / "run.json", {"run_id": gid, "offline": False, "smoke": False})
    run_gate._write_json(d / "freeze.json", {"run_id": gid, "offline": False, "frozen_at": "2026-10-01T00:00:00+00:00"})
    run_gate._write_json(d / "report" / "decision.json", {"verdict": {"label": "GO", "reasons": []}})
    (d / "config").mkdir(parents=True)
    selected = {"APG*": {"arm": "APG-q", "env": {}, "candidate": "a"}, "LGR*": {"arm": "LGR-s", "env": {}, "candidate": "l"}, "S3s": {"arm": "S3s", "env": {}, "candidate": "s"}}
    (d / "config" / "selected.yaml").write_text(yaml.safe_dump(selected))


def test_a_live_freeze_refuses_the_draft_naming_every_marker_and_freezes_it_filled(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))  # the spend check reads this ledger, not the real one
    _fake_gate_run(tmp_path / "gates", "g-go")
    prereg, provenance = tmp_path / "PREREGISTRATION_MAIN.md", tmp_path / "PROVENANCE.md"
    shutil.copy(PREREG, prereg)
    shutil.copy(ROOT / "PROVENANCE.md", provenance)
    kw = {"runs_root": tmp_path / "runs", "gate_run_id": "g-go", "gate_runs_root": tmp_path / "gates", "prereg_path": prereg, "provenance_path": provenance}
    run = StudyRun("main", "live", **kw)
    run.out_config_dir.mkdir(parents=True)
    (run.out_config_dir / "selected.yaml").write_text(yaml.safe_dump({"S3s": {"arm": "S3s", "env": {"APE_S3S_BUDGET": "2000"}, "candidate": "c"}}))
    run_study.write_caps(run.token_caps_path, {"F1-2": {"b0": 100.0, "samples": 2}}, "micro-pilot")
    run_study.write_caps(run.pilot_caps_path, {}, "pilot")
    (run.out_config_dir / "s7_targets.json").write_text(json.dumps({"F7-10": 120}))
    run_gate._write_json(run.cap_gate_path, {"multiple": 8, "over": {}, "passed": True, "rates": {}, "rounds": []})  # the pilot's D-039 gate passed
    run_study._check_mode(run)
    run_gate._write_json(run.manifest_path("preflight"), {"phase": "preflight", "status": "done", "fingerprint": "fake", "finished": "2026-10-01T00:00:00+00:00", "outputs": {}})
    for phase in run.spec.rests_on:  # every phase the freeze rests on, done and current
        run_gate._write_json(run.manifest_path(phase), {"phase": phase, "status": "done", "finished": "2026-10-01T00:00:00+00:00", "outputs": {}} | run_study.phase_state(run, phase))
    monkeypatch.setattr(run_gate, "git_tracked_changes", lambda: [])
    monkeypatch.setattr(run_study, "analysis_available", lambda r: True)

    markers = run_gate.prereg_placeholders(prereg.read_text())
    if markers:  # the draft: refused, every marker named, nothing frozen
        with pytest.raises(PhaseError, match=rf"still has {len(markers)} unfilled item\(s\)") as e:
            run_phases(run, "freeze")
        assert all(f"line {p['line']}: {p['marker']}" in str(e.value) for p in markers), "the refusal names every marker"
        assert read_freeze(run) is None
        text = prereg.read_text()
        start = run_gate.prereg_body_start(text)
        prereg.write_text(text[:start] + run_gate.PLACEHOLDER_ITEM.sub("filled", text[start:]))
    assert run_phases(run, "freeze") == {"freeze": "done"}
    freeze = run_study.require_frozen(run)
    assert freeze["rehearsal"] is False and freeze["test_seeds"]["base"] == 13000 and freeze["kg"]["arm"] == "APG-q"
    assert freeze["files"]["PREREGISTRATION_MAIN.md"]["sha256"] == run_gate._sha256(prereg)
    assert {"src/ape/analyze_main.py", "src/ape/analysis/main_hypotheses.py", "uv.lock", "gate/decision.json"} <= set(freeze["files"]), "the freeze guards the analysis code"
