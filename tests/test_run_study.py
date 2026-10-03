"""Study orchestrator (BUILD_PLAN B1, B6): the main study and Study G end to end offline, their seed namespaces and
per-study test lock, unbuilt arms, the KG arm from the gate's verdict, token caps from the micro-pilot, resume, the
budget guard on the study's allocation and the freeze. Offline: mock models, fake embeddings, no network; every run
lives under a temporary directory."""

import json
import math
import os
import shutil
import statistics
from pathlib import Path

import pytest
import yaml

from ape import run_gate, run_study
from ape.budget import BudgetError
from ape.config import ROOT, Config
from ape.run_gate import GateRun
from ape.run_study import STUDIES, PhaseError, StudyRun, main, read_freeze, run_phases
from ape.worlds import generate
from ape.worlds.generate import GATE_SEED_LIMIT, SEED_BLOCK, SPLIT_SEED_BASE, STUDY_SEEDS, TEST_SPLIT_ENV, make_world, require_test_split_unlocked
from ape.worlds.spec import World

SELECTED = {
    "APG*": {"arm": "APG-q", "env": {"APE_APG_SHORTLIST_K": "24"}, "candidate": "apg-q-k24"},
    "LGR*": {"arm": "LGR-s", "env": {"APE_LGR_MODE": "mix"}, "candidate": "lgr-s-mix"},
    "S3s": {"arm": "S3s", "env": {"APE_S3S_BUDGET": "2000"}, "candidate": "s3s-2000"},
}


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No stray APE_* knobs from the shell or other tests; the suite's isolated spend registry stays (conftest)."""
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        monkeypatch.delenv(k)
    return tmp_path


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    """One offline `all` of each study (the acceptance runs), shared by the tests that read them."""
    tmp = tmp_path_factory.mktemp("studies")
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    with pytest.MonkeyPatch.context() as mp:
        for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
            mp.delenv(k)

        def argv(study: str) -> list[str]:
            return ["all", "--study", study, "--run-id", "t1", "--offline", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]

        codes = {study: main(argv(study)) for study in ("main", "study_g")}
        yield {"tmp": tmp, "config": config, "runs": tmp / "runs", "codes": codes, "argv": argv}


def _run(offline, study: str, **kw) -> StudyRun:
    return StudyRun(study, "t1", offline=True, runs_root=offline["runs"], config_dir=offline["config"], **kw)


def _manifest(run: StudyRun, phase: str) -> dict:
    return json.loads(run.manifest_path(phase).read_text())


def _fake_gate_run(root: Path, gid: str, label: str | None, *, offline: bool = False, frozen: bool = True, selected: dict = SELECTED) -> Path:
    """A gate run as run_gate leaves it: run.json, freeze.json, report/decision.json and the run's selected.yaml."""
    d = root / gid
    run_gate._write_json(d / "run.json", {"run_id": gid, "offline": offline, "smoke": False})
    if frozen:
        run_gate._write_json(d / "freeze.json", {"run_id": gid, "offline": offline, "frozen_at": "2026-10-01T00:00:00+00:00"})
    if label is not None:
        run_gate._write_json(d / "report" / "decision.json", {"verdict": {"label": label, "reasons": []}})
    sel = (d / "work" / "config" if offline else d / "config") / "selected.yaml"
    sel.parent.mkdir(parents=True, exist_ok=True)
    sel.write_text(yaml.safe_dump(selected))
    return d


# --- Seeds and the test lock ---------------------------------------------------------------------------


def test_seed_namespaces_never_let_a_main_or_g_world_coincide_with_a_gate_world():
    spans = []
    for study, s in STUDY_SEEDS.items():
        lo, hi = s["test"]
        spans += [(study, "pilot", s["pilot"], s["pilot"] + SEED_BLOCK), (study, "test", lo, hi), (study, "offline", s["offline_test"], s["offline_test"] + SEED_BLOCK)]
        own = [x for x in spans if x[0] == study]
        assert all((end <= GATE_SEED_LIMIT) if study == "gate" else (start >= GATE_SEED_LIMIT) for _, _, start, end in own), study
    for i, a in enumerate(spans):
        for b in spans[i + 1 :]:
            assert a[3] <= b[2] or b[3] <= a[2], f"{a} overlaps {b}"
    # The gate's entry restates its bases; dev is shared on purpose (the gate's dev worlds, and their KG builds).
    gate = STUDY_SEEDS["gate"]
    assert (gate["dev"], gate["pilot"], gate["test"][0]) == tuple(SPLIT_SEED_BASE[s] for s in ("dev", "pilot", "test"))
    assert gate["test"] == (run_gate.FIRST_TEST_SEED_BASE, run_gate.OFFLINE_TEST_SEED_BASE) and gate["offline_test"] == run_gate.OFFLINE_TEST_SEED_BASE
    assert {s["dev"] for s in STUDY_SEEDS.values()} == {1000}
    assert (STUDY_SEEDS["main"]["pilot"], STUDY_SEEDS["main"]["test"], STUDY_SEEDS["main"]["offline_test"]) == (12000, (13000, 19000), 19000)
    assert (STUDY_SEEDS["study_g"]["pilot"], STUDY_SEEDS["study_g"]["test"], STUDY_SEEDS["study_g"]["offline_test"]) == (22000, (23000, 29000), 29000)


def test_a_namespace_edit_that_overlaps_is_refused_at_import(monkeypatch):
    monkeypatch.setitem(generate.STUDY_SEEDS, "main", {"dev": 1000, "pilot": 12000, "test": (3000, 9000), "offline_test": 9000})
    with pytest.raises(ValueError, match="gate seeds lie below"):
        generate._check_namespaces()
    monkeypatch.setitem(generate.STUDY_SEEDS, "main", {"dev": 1000, "pilot": 22000, "test": (13000, 19000), "offline_test": 19000})
    with pytest.raises(ValueError, match="overlap"):
        generate._check_namespaces()


def test_gate_tasks_never_read_another_studys_worlds(clean_env, monkeypatch):
    """World files of every study share worlds/<split>/; as strings s12000 sorts before s2000. A gate task without a
    seed block reads only the gate's namespace, and a study task only its own block."""
    from ape.tasks.gate import gate_samples

    monkeypatch.setenv("APE_WORLDS", str(clean_env / "worlds"))
    cfg = Config()
    for split, bases in (("pilot", (2000, 12000, 22000)), ("test", (3000, 13000, 23000))):
        for base in bases:
            for i in range(2):
                w = make_world("F3", "5", split, i, 2, seed_base=base)
                w.save(cfg.world_path(w.id))

    def seeds(samples):
        return sorted({int(s.metadata["world_id"].rsplit("-s", 1)[1]) for s in samples})

    assert seeds(gate_samples("F3", "5", "pilot", limit_worlds=1)) == [2000], "the gate's first pilot world, not main's s12000"
    assert seeds(gate_samples("F3", "5", "pilot")) == [2000, 2001]
    assert seeds(gate_samples("F3", "5", "test", limit_worlds=2)) == [3000, 3001]
    assert seeds(gate_samples("F3", "5", "pilot", seed_base=12000)) == [12000, 12001]
    assert seeds(gate_samples("F3", "5", "test", seed_base=23000, limit_worlds=1)) == [23000]


def test_the_test_split_lock_is_per_study(clean_env, monkeypatch):
    monkeypatch.setenv("APE_WORLDS", str(clean_env / "worlds"))
    for study in ("gate", "main", "study_g"):
        require_test_split_unlocked("dev", study)  # other splits are never locked
        with pytest.raises(generate.TestSplitLocked, match="run_gate build-test" if study == "gate" else f"run_study build-test --study {study}"):
            require_test_split_unlocked("test", study)
    # The gate's build-test unlock (its run id) opens the gate's test split only.
    monkeypatch.setenv(TEST_SPLIT_ENV, generate.split_lock_value("gate", "gate-run"))
    require_test_split_unlocked("test")
    for study in ("main", "study_g"):
        with pytest.raises(generate.TestSplitLocked, match=f"unlocked for gate .*not for {study}"):
            require_test_split_unlocked("test", study)
    # A main build-test's unlock opens main's only: the gate's world-set builder refuses under it.
    monkeypatch.setenv(TEST_SPLIT_ENV, generate.split_lock_value("main", "m1"))
    assert os.environ[TEST_SPLIT_ENV] == "main/m1"
    require_test_split_unlocked("test", "main")
    with pytest.raises(generate.TestSplitLocked, match="unlocked for main"):
        require_test_split_unlocked("test")
    with pytest.raises(generate.TestSplitLocked, match="unlocked for main .*not for study_g"):
        require_test_split_unlocked("test", "study_g")
    with pytest.raises(generate.TestSplitLocked):
        run_gate.build_world_set(GateRun("x", offline=True, runs_root=clean_env / "runs"), {"outputs": {}}, "build-test", "test", [])
    monkeypatch.delenv(TEST_SPLIT_ENV)
    # A study's build-test refuses before that study's freeze, without writing a manifest or a world.
    run = StudyRun("main", "unfrozen", offline=True, runs_root=clean_env / "runs")
    with pytest.raises(PhaseError, match=r"build-test: main run 'unfrozen' is not frozen"):
        run_phases(run, "build-test")
    assert not run.manifest_path("build-test").exists() and not (run.work_dir / "worlds" / "test").exists()


def test_gate_run_ids_never_take_a_studys_directory(clean_env):
    for rid in ("main", "study_g"):
        with pytest.raises(PhaseError, match="holds that study's runs"):
            GateRun(rid, runs_root=clean_env / "runs")
    run_gate._write_json(clean_env / "runs" / "main" / "run.json", {"run_id": "main"})  # an older gate run named main
    with pytest.raises(PhaseError, match="is a gate run's directory"):
        StudyRun("main", "m1", offline=True, runs_root=clean_env / "runs")


def test_test_seed_blocks_per_study(clean_env):
    provenance = clean_env / "PROVENANCE.md"
    provenance.write_text("# Provenance\n<!-- ape:test-seeds run=g1 base=3000 count=16 -->\n")
    kw = {"runs_root": clean_env / "runs", "provenance_path": provenance, "gate_run_id": "g-go", "gate_runs_root": clean_env / "gates"}
    _fake_gate_run(clean_env / "gates", "g-go", "GO")  # outside runs/, where the gate's own seed-block scan looks
    first = StudyRun("main", "m1", **kw)
    assert run_study.choose_test_seed_base(first) == (13000, []), "the gate's 3000 block is not main's"
    assert run_study.test_seed_count(first) == 9, "100 tasks at 12 per world"
    run_gate._write_json(first.freeze_path, {"run_id": "m1", "offline": False, "test_seeds": {"base": 13000, "count": 9}})
    assert run_study.choose_test_seed_base(StudyRun("main", "m2", **kw))[0] == 13100, "a later main run freezes a fresh block"
    provenance.write_text(provenance.read_text() + "<!-- ape:test-seeds study=main run=m0 base=13100 count=9 -->\n")
    assert run_study.choose_test_seed_base(StudyRun("main", "m2", **kw))[0] == 13200, "PROVENANCE.md's study markers count"
    assert "overlaps" in run_study.choose_test_seed_base(StudyRun("main", "m2", test_seed_base=13005, **kw))[1][0]
    assert "test seeds live in [13000, 19000)" in run_study.choose_test_seed_base(StudyRun("main", "m2", test_seed_base=3000, **kw))[1][0]
    assert run_study.choose_test_seed_base(StudyRun("study_g", "g1", runs_root=clean_env / "runs", provenance_path=provenance))[0] == 23000, "main's blocks are not G's"
    assert run_study.choose_test_seed_base(StudyRun("main", "off", offline=True, runs_root=clean_env / "runs")) == (19000, [])
    gate_blocks = run_gate.used_test_seed_blocks(GateRun("g2", runs_root=clean_env / "runs", provenance_path=provenance))
    assert [u["base"] for u in gate_blocks] == [3000], "a study's marker and freeze are not gate blocks"


# --- The KG arm ---------------------------------------------------------------------------------------


def test_the_kg_arm_follows_the_gate_runs_verdict_and_selection(clean_env, monkeypatch):
    runs = clean_env / "runs"
    for gid, label in (("g-go", "GO"), ("g-pull", "GO_PULL_ONLY"), ("g-nogo", "NO_GO"), ("g-inc", "INCONCLUSIVE"), ("g-pcf", "PRECONDITION_FAIL")):
        _fake_gate_run(runs, gid, label)
    _fake_gate_run(runs, "g-unanalysed", None)
    _fake_gate_run(runs, "g-unfrozen", "GO", frozen=False)
    _fake_gate_run(runs, "g-offline", "GO", offline=True)

    def kg(gid: str | None, offline: bool = False) -> dict:
        return run_study.kg_resolution(StudyRun("main", "m", offline=offline, runs_root=runs, gate_run_id=gid))

    go = kg("g-go")
    assert (go["key"], go["arm"], go["system"], go["env"], go["verdict"]) == ("APG*", "APG-q", "apg", {"APE_APG_SHORTLIST_K": "24"}, "GO")
    assert set(go["files"]) == {"gate/selected.yaml", "gate/decision.json", "gate/freeze.json"}
    assert kg("g-pull")["arm"] == "APG-q" and kg("g-pull")["notes"], "every GO label selects APG*; pull-only is noted"
    nogo = kg("g-nogo")
    assert (nogo["key"], nogo["arm"], nogo["system"], nogo["env"]) == ("LGR*", "LGR-s", "lightrag", {"APE_LGR_MODE": "mix"})
    assert kg("g-nogo", offline=True)["arm"] == "LGRo-s" and kg("g-nogo", offline=True)["declared"] == "LGR-s", "offline: the oracle twin"
    # A live run refuses without a final verdict, and without a gate run at all.
    for gid, why in (("g-inc", "INCONCLUSIVE, not final"), ("g-pcf", "PRECONDITION_FAIL, not final"), ("g-unanalysed", "no decision report"), ("g-unfrozen", "not frozen"), ("g-offline", "offline rehearsal"), ("nope", "run.json missing")):
        with pytest.raises(PhaseError, match=why):
            kg(gid)
    with pytest.raises(PhaseError, match="needs the gate's verdict.*--gate-run-id"):
        kg(None)
    with pytest.raises(PhaseError, match="needs the gate's verdict"):
        run_phases(StudyRun("main", "nokg", runs_root=runs), "preflight")
    # Offline defaults to APG over the fake author's graphs; an unusable gate run falls back with a note.
    assert (kg(None, offline=True)["arm"], kg(None, offline=True)["system"]) == ("APG-s", "apg")
    assert kg("g-inc", offline=True)["arm"] == "APG-s" and any("not final" in n for n in kg("g-inc", offline=True)["notes"])
    # Study G names no KG arm (its M2 is a session arm), so it needs no gate run.
    assert run_study.kg_resolution(StudyRun("study_g", "g", runs_root=runs)) == {"needed": False, "system": None}
    # The run environment sets the KG arm and its knobs; a live shell value that disagrees is refused, not overridden.
    run = StudyRun("main", "m", runs_root=runs, gate_run_id="g-go")
    with run_study.run_environment(run):
        assert os.environ["APE_KG_ARM"] == "APG-q" and os.environ["APE_APG_SHORTLIST_K"] == "24" and os.environ["APE_SPEND_LABEL"] == "main/m"
        assert "APE_KG_ARM" not in run_study.env_knobs(run) and "APE_APG_SHORTLIST_K" not in run_study.env_knobs(run), "recorded as the KG resolution"
    assert "APE_KG_ARM" not in os.environ
    monkeypatch.setenv("APE_APG_SHORTLIST_K", "48")
    with pytest.raises(PhaseError, match=r"APE_APG_SHORTLIST_K=48 \(the gate's verdict sets 24\)"), run_study.run_environment(run):
        pass


def test_a_run_keeps_the_gate_run_its_kg_arm_came_from(clean_env):
    gates = clean_env / "gates"
    _fake_gate_run(gates, "g-go", "GO")
    _fake_gate_run(gates, "g-nogo", "NO_GO")
    first = StudyRun("main", "m", runs_root=clean_env / "runs", gate_run_id="g-go", gate_runs_root=gates)
    run_study._check_mode(first)
    assert run_gate.read_run_info(first)["gate_run_id"] == "g-go"
    later = StudyRun("main", "m", runs_root=clean_env / "runs")  # no --gate-run-id: the recorded one
    run_study.settle_gate_run(later)
    assert later.gate_run_id == "g-go" and run_study.kg_resolution(later)["arm"] == "APG-q"
    with pytest.raises(PhaseError, match="takes its KG arm from gate run 'g-go'.*not 'g-nogo'"):
        run_phases(StudyRun("main", "m", runs_root=clean_env / "runs", gate_run_id="g-nogo", gate_runs_root=gates), "preflight")


def test_a_changed_gate_selection_changes_the_phases_fingerprints(clean_env):
    runs = clean_env / "runs"
    d = _fake_gate_run(runs, "g-go", "GO")
    before = run_study.phase_state(StudyRun("main", "m", runs_root=runs, gate_run_id="g-go"), "build-dev")
    (d / "config" / "selected.yaml").write_text(yaml.safe_dump({**SELECTED, "APG*": {"arm": "APG-s", "env": {"APE_APG_SHORTLIST_K": "12"}}}))
    after = run_study.phase_state(StudyRun("main", "m", runs_root=runs, gate_run_id="g-go"), "build-dev")
    assert before["params"]["kg"]["arm"] == "APG-q" and after["params"]["kg"]["arm"] == "APG-s" and before["fingerprint"] != after["fingerprint"]


# --- Unbuilt arms --------------------------------------------------------------------------------------


def test_unbuilt_arms_are_skipped_offline_recorded_and_refused_live(clean_env, monkeypatch):
    from ape.agent import context_policy, session, solvers

    run = StudyRun("main", "arms", offline=True, runs_root=clean_env / "runs")
    assert run_study.unbuilt_arms(run, "test") == [] and run_study.unbuilt_arms(run, "micro-pilot") == [], "B2 built every main-study arm"
    assert not run_study.arm_built("S99") and not run_study.arm_built("CM-nope", "session") and run_study.arm_built("CM-native", "session")
    # An arm no package has registered (here M1, unregistered for the test) is skipped offline, and recorded...
    monkeypatch.delitem(solvers.MULTI_AGENT_ARMS, "M1")
    missing = run_study.unbuilt_arms(run, "test")
    assert "main.A.arms: M1" in missing and "main.A.arms: S9" not in missing
    (g0,) = run_study.run_groups(run, "micro-pilot")
    assert [a["run"] for a in g0["arms"]] == ["S1", "S5"] and g0["skipped"] == [{"declared": "M1", "run": "M1", "reason": "not built yet"}]
    # ...and a live phase whose cells name it refuses to start, naming it, before any record.
    _fake_gate_run(clean_env / "runs", "g-go", "GO")
    live = StudyRun("main", "live", runs_root=clean_env / "runs", gate_run_id="g-go")
    with pytest.raises(PhaseError, match=r"micro-pilot: its cells name arms that are not built yet: \['main.micro-pilot: M1'\]"):
        run_phases(live, "micro-pilot")
    assert not live.manifest_path("micro-pilot").exists()
    # Study G: a session arm is built when it is a registered context policy (B8's CM arms; B9 adds the topology arms).
    g = StudyRun("study_g", "arms", offline=True, runs_root=clean_env / "runs")
    expected = {f"{c.id}: {a}" for c in run_study.phase_cells(g, "test") for a in rg_arm_names(c) if not run_study.arm_built(a, c.kind)}
    assert set(run_study.unbuilt_arms(g, "test")) == expected and not any(m.startswith("g.cm.") for m in expected)
    monkeypatch.delitem(context_policy.POLICIES, "CM-sum")
    monkeypatch.setattr(session, "SESSION_ARMS", tuple(a for a in session.SESSION_ARMS if a != "CM-sum"))
    with pytest.raises(PhaseError, match=r"not built yet: \['g.pilot.luna: CM-sum'\]"):
        run_phases(StudyRun("study_g", "live", runs_root=clean_env / "runs"), "micro-pilot")


def rg_arm_names(cell) -> list[str]:
    return list(cell.spec["arms"]) if isinstance(cell.spec["arms"], (list, dict)) else [cell.spec["arms"]]


def test_the_offline_runs_record_every_skipped_arm(offline):
    run, g = _run(offline, "main"), _run(offline, "study_g")
    for phase in ("micro-pilot", "pilot", "test"):
        assert _manifest(run, phase)["skipped_arms"] == [], f"every main-study arm runs ({phase})"
    assert not _manifest(run, "preflight")["unbuilt_arms"]
    assert set(x["status"] for x in _manifest(run, "test")["cells"].values()) == {"done"}
    gm = _manifest(g, "test")
    expected = {(c.id, a) for c in run_study.phase_cells(g, "test") for a in rg_arm_names(c) if not run_study.arm_built(run_study._resolved(a, run_study.read_selected(g)), c.kind)}
    assert {(s["cell"], s["declared"]) for s in gm["skipped_arms"]} == expected, "the session arms B9 has not registered yet, and nothing else"
    for cell_id, cell in gm["cells"].items():
        built = [a for c in run_study.phase_cells(g, "test") if c.id == cell_id for a in rg_arm_names(c) if (cell_id, a) not in expected]
        assert cell["status"] == ("done" if built else "skipped"), cell_id


# --- End to end, offline -------------------------------------------------------------------------------


def test_offline_all_runs_each_study_end_to_end(offline):
    assert offline["codes"] == {"main": 0, "study_g": 0}
    for study in ("main", "study_g"):
        run = _run(offline, study)
        manifests = {p: _manifest(run, p) for p in STUDIES[study].phases}
        assert {p: m["status"] for p, m in manifests.items()} == dict.fromkeys(STUDIES[study].phases, "done")
        for p, m in manifests.items():
            assert m["study"] == study and m["offline"] is True and m["errors"] == [] and m["fingerprint"]
            assert m["offline_check"]["ledger_entries"] == 0 and set(m["offline_check"]["models"]) <= {"mockllm/model"}
            assert m["spend"]["spent_usd"] == 0.0 and m["spend"]["study_allocation_usd"] == float(run_study.plan(run).budget["allocations"][study])
            assert p == "analyze" or m["params"]["scale"] == {"offline": True, **run_study.OFFLINE_SCALE}
        assert manifests["test"]["primary_complete"] is True
        # The spend registry is the run's own, every log dir labelled with the study.
        registry = [e for line in (run.work_dir / "spend_registry.jsonl").read_text().splitlines() if (e := json.loads(line))["event"] == "start"]
        assert registry and {e["label"] for e in registry} == {f"{study}-offline/t1"} and {e["study"] for e in registry} == {f"{study}-offline"}
        # Seeds: dev the shared namespace, pilot and test the study's own (offline: the rehearsal block).
        seeds = STUDY_SEEDS[study]
        for phase, lo, hi in (("build-dev", 1000, 1000 + SEED_BLOCK), ("micro-pilot", seeds["pilot"], seeds["pilot"] + SEED_BLOCK), ("build-test", seeds["offline_test"], seeds["offline_test"] + SEED_BLOCK)):
            worlds = json.loads((run.phase_dir(phase) / "worlds.json").read_text())["worlds"]
            assert worlds and all(lo <= int(w["world_id"].rsplit("-s", 1)[1]) < hi for w in worlds), phase
            assert all(Path(w["path"]).resolve().is_relative_to(run.work_dir.resolve()) for w in worlds), "offline worlds live in the run"
        freeze = read_freeze(run)
        assert freeze["rehearsal"] is True and freeze["test_seeds"]["base"] == seeds["offline_test"] and freeze["role"] == {"kind": "primary", "of": None}
        assert (run.work_dir / "PROVENANCE.freeze.md").is_file() and f"study={study} run=t1 base={seeds['offline_test']}" in (run.work_dir / "PROVENANCE.freeze.md").read_text()
        # The analysis hook: the main study's analysis exists (ape.analyze_main, tests/test_analyze_main.py); Study G's
        # stub stays until ape.analyze_g lands.
        if run_study.analysis_available(run):
            assert manifests["analyze"]["analysis"]["status"] == "done" and (run.dir / "report" / "report.md").is_file()
        else:
            report = json.loads((run.dir / "report" / "analysis.json").read_text())
            assert report["status"] == "not_implemented" and "analysis not implemented" in (run.dir / "report" / "report.md").read_text()
            assert manifests["analyze"]["analysis"]["status"] == "not_implemented" and any("analysis not implemented" in w for w in manifests["analyze"]["warnings"])
    assert not (ROOT / "runs" / "main" / "t1").exists() and not (ROOT / "runs" / "study_g" / "t1").exists()


def test_offline_main_runs_its_cells_on_its_worlds_with_the_kg_arm_and_the_selections(offline):
    from inspect_ai.log import read_eval_log

    run = _run(offline, "main")
    selected = yaml.safe_load(run.selected_path.read_text())
    grid, _ = run_study.study_grid(run)
    assert set(selected) == set(grid["systems"]), "one selection per grid system"
    tune = _manifest(run, "tune")
    assert tune["tuning_completeness"]["pass"] is True and set(tune["tuning_completeness"]["systems"]) == set(grid["systems"]), "PC6-style"
    kg = _manifest(run, "test")["kg"]
    assert kg["arm"] == "APG-s" and kg["system"] == "apg" and read_freeze(run)["kg"]["arm"] == "APG-s"
    targets = json.loads(run.s7_targets_path.read_text())
    assert set(targets) == {"F3-5", "F3-60", "F7-10", "F7-1000"} and all(t > 0 for t in targets.values())
    seen, records = set(), {}
    together = run_study.selection_env(selected) or {}
    for f in sorted((run.dir / "test").rglob("*.eval")):
        log = read_eval_log(str(f))
        args, md = log.eval.task_args, log.eval.metadata
        assert log.status == "success" and log.eval.model == "mockllm/model" and not any(s.error for s in log.samples)
        assert args["seed_base"] == STUDY_SEEDS["main"]["offline_test"] and args["split"] == "test" and args["plan_cell"] == md["plan_cell"]
        assert md["knobs"]["APE_KG_ARM"] == "APG-s" and all(md["knobs"].get(k) == v for k, v in together.items() if k.startswith(("APE_S3S_", "APE_APG_", "APE_LGR_")))
        if args["arm"] == "S7":
            assert args["group"] == "s7" and md["knobs"]["APE_S7_PER_STEP"] == "1", "S7 mirrors the per-step KG arm"
        seen.add((args["plan_cell"], args["arm"]))
        records[args["arm"]] = sorted(k for k in log.samples[0].store if k.startswith("mas_"))
    # Every main-study arm runs through micro-pilot, pilot and test offline (the gold multi-role mock plays each role).
    arms = {"S1", "S3s", "S5", "S7", "S9", "M1", "M1s", "M1k", "M2", "M7", "S8k3"}
    assert {a for _, a in seen} == arms and {("main.F.sol-m2", "M2"), ("main.F.luna-f7-100", "S5")} <= seen
    assert all("mas_accounting" in records[a] and "mas_agents" in records[a] for a in arms - {"S1", "S3s", "S5", "S7"}), "B2's per-agent records"
    assert "mas_council" in records["M7"] and "mas_ensemble" in records["S8k3"] and "mas_specialists" in records["M2"]
    ran = {read_eval_log(str(f), header_only=True).eval.task_args["arm"] for phase in ("micro-pilot", "pilot") for f in (run.dir / phase).rglob("*.eval")}
    assert ran == arms - {"M1s", "S8k3"}, "the pilot cells name every arm but M1s and S8k3 (Studies A and C only)"
    sol = [read_eval_log(str(f), header_only=True).eval for f in (run.dir / "test" / "main.F.sol").rglob("*.eval")]
    assert {e.metadata["arm"] for e in sol} == {"S1", "S5", "M1"} and {e.model_generate_config.reasoning_effort for e in sol} == {"high"}
    # B6: the KG builds and their build-quality check, F1 included, in the one KG system built.
    bq = json.loads((run.phase_dir("micro-pilot") / "build_quality.json").read_text())
    assert bq["systems"] == ["apg"] and bq["verdict"] == "builder_passes" and {"F1-2", "F1-32", "F7-10", "F7-1000", "F3-5", "F3-60"} <= set(bq["cells"])
    assert all(r["apg"]["metric"] == "fact_coverage" for r in bq["worlds"] if r["cell"].startswith("F1"))
    test_bq = json.loads((run.phase_dir("build-test") / "build_quality.json").read_text())
    assert {"F7-100", "F1-32"} <= set(test_bq["cells"]) and all(s["lightrag"] is None for s in test_bq["cells"].values())
    worlds = {w["world_id"]: w for w in json.loads((run.phase_dir("build-test") / "worlds.json").read_text())["worlds"]}
    assert worlds["F7-100-rel-desc-test-s19000"]["kinds"] == ["apg"] and worlds["F3-5-test-s19000"]["kinds"] == ["chunks", "apg"], "S5 reads the KG, S3s the chunks"
    assert worlds["F2-10-test-s19000"]["kinds"] == [] and worlds["F2-10-test-s19000"]["artifacts"] == {}, "S1-only cells build no artifacts"


def test_offline_study_g_runs_sessions_and_the_capability_anchor_on_its_own_worlds(offline):
    from inspect_ai.log import read_eval_log

    run = _run(offline, "study_g")
    heads = [read_eval_log(str(f), header_only=True) for f in sorted((run.dir / "test").rglob("*.eval"))]
    sessions = [h for h in heads if h.eval.task.endswith("f8_session")]
    anchor = [h for h in heads if h.eval.task.endswith("main_study")]
    assert {(h.eval.task_args["level"], h.eval.task_args["variant"]) for h in sessions} == {("40", ""), ("10", ""), ("24", "o1750"), ("20", "o2250")}
    # Every built context-management arm runs (B9's session S1/M1/M2 join the topology cells once registered).
    cm = {"CM0", "CM-prune", "CM-sum", "CM-todo", "CM-reset", "CM-native", "O-state", "S-CM*"}
    assert cm <= {h.eval.task_args["arm"] for h in sessions} and all(h.eval.task_args["window"] == 32000 for h in sessions)
    plan = run_study.plan(run).study_g
    assert all((a := h.eval.task_args)["seed_base"] == 29000 and a["threshold"] == plan["threshold"] and a["plan_cell"] and a["group"] for h in sessions)
    sess = read_eval_log(str(next(f for f in sorted((run.dir / "test" / "g.cm.luna-high").rglob("*.eval")) if read_eval_log(str(f), header_only=True).eval.task_args["arm"] == "CM-native")))
    assert sess.samples[0].store["f8_cm_events"] and not sess.samples[0].error, "CM-native took its mock route offline"
    assert _manifest(run, "preflight")["checks"]["cm_native"] == {"g.pilot.luna": "mock", "g.cm.luna-high": "mock"}
    # APE_SESSION_CHECKPOINTS reached the sessions: each sample saved there (a finished session deletes only its file).
    assert {p.parent.name for p in run.session_checkpoints_dir.glob("*/epoch-1")} >= {"F8-40-pilot-s22000", "F8-40-dev-s1000", "F8-40-test-s29000", "F8-20-o2250-test-s29000"}
    tlog = [json.loads(line) for line in (run.phase_dir("tune") / "tuning_log.jsonl").read_text().splitlines()]
    tuned = {c["arm"] for sdef in run_study.study_grid(run)[0]["systems"].values() for c in sdef["candidates"]}
    assert {r["candidate"]["arm"] for r in tlog if "candidate" in r} == tuned and (cm - {"CM0", "O-state", "CM-native"}) <= tuned, "the grid's session arms tune too"
    assert {(h.eval.task_args["family"], h.eval.task_args["level"]) for h in anchor} == {("F7", "10"), ("F3", "5")}
    assert all(h.eval.task_args["seed_base"] == 29000 and h.eval.config.token_limit is None for h in anchor), "G's own worlds; G is not capped"
    efforts = {(h.eval.metadata["plan_cell"], h.eval.model_generate_config.reasoning_effort) for h in anchor}
    assert ("g.cap.luna-low", "low") in efforts and ("g.cap.luna-high", "high") in efforts, "a cell's effort overrides the agent's"
    worlds = {w["world_id"] for w in json.loads((run.phase_dir("build-test") / "worlds.json").read_text())["worlds"]}
    assert {"F8-40-test-s29000", "F8-24-o1750-test-s29000", "F8-20-o2250-test-s29000", "F7-10-rel-desc-test-s29000", "F3-5-test-s29000"} <= worlds
    assert {w["world_id"] for w in json.loads((run.phase_dir("build-dev") / "worlds.json").read_text())["worlds"]} == {"F8-40-dev-s1000"}
    assert set(yaml.safe_load(run.selected_path.read_text())) == set(run_study.study_grid(run)[0]["systems"])
    assert _manifest(run, "micro-pilot")["operational_env"]["APE_SESSION_CHECKPOINTS"] == str(run.session_checkpoints_dir)


def test_a_second_all_reuses_every_completed_phase(offline):
    for study in ("main", "study_g"):
        run = _run(offline, study)
        before = {p: _manifest(run, p) for p in STUDIES[study].phases}
        assert main(offline["argv"](study)) == 0
        after = {p: _manifest(run, p) for p in STUDIES[study].phases}
        assert {p: m["status"] for p, m in after.items()} == dict.fromkeys(STUDIES[study].phases, "skipped"), study
        assert all(after[p]["finished"] == before[p]["finished"] and after[p]["fingerprint"] == before[p]["fingerprint"] for p in after)


def test_the_cli_knows_each_studys_phases(offline, capsys):
    with pytest.raises(SystemExit) as e:
        main(["pilot", "--study", "study_g", "--run-id", "x", "--offline", "--runs-dir", str(offline["tmp"] / "other")])
    assert e.value.code == 2 and "study_g has no pilot phase" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["tune", "--study", "main", "--run-id", "x", "--offline", "--test-seed-base", "13000"])


# --- Token caps ----------------------------------------------------------------------------------------


def test_token_caps_are_eight_times_s1s_median_on_the_micro_pilot_and_bound_every_later_task(offline):
    from inspect_ai.log import read_eval_log

    run = _run(offline, "main")
    caps = json.loads(run.token_caps_path.read_text())
    assert caps["multiple"] == 8 and caps["arm"] == "S1" and set(caps["cells"]) == {"F1-2", "F1-32", "F7-10", "F7-1000"}
    totals: dict[str, list[int]] = {}
    for f in (run.dir / "micro-pilot").rglob("*.eval"):
        log = read_eval_log(str(f))
        assert log.eval.config.token_limit is None, "the micro-pilot measures B0: it runs uncapped"
        if log.eval.task_args["arm"] == "S1":
            for s in log.samples:
                totals.setdefault(f"{log.eval.task_args['family']}-{log.eval.task_args['level']}", []).append(sum(u.total_tokens for u in s.model_usage.values()))
    for cell, v in caps["cells"].items():
        assert v["b0"] == statistics.median(totals[cell]) and v["cap"] == math.ceil(8 * v["b0"])
    # The pilot measures the cells the micro-pilot did not; F7-100 (piloted nowhere) borrows its family's.
    pilot_caps = json.loads(run.pilot_caps_path.read_text())["cells"]
    assert set(pilot_caps) == {"F2-2", "F2-10", "F3-5", "F3-60"}
    all_caps = run_study.token_caps(run)
    assert all_caps["F7-100"]["source"].startswith("borrowed") and all_caps["F7-100"]["cap"] == max(caps["cells"]["F7-10"]["cap"], caps["cells"]["F7-1000"]["cap"])
    assert run_study.uncapped_cells(run) == [] and read_freeze(run)["token_caps"] == {c: v["cap"] for c, v in all_caps.items()}
    # Every later task carries its cell's cap as Inspect's token_limit (tune and pilot: the micro-pilot's caps only).
    for phase, expected in (("tune", run_study.token_caps(run, "tune")), ("pilot", run_study.token_caps(run, "pilot")), ("test", all_caps)):
        for f in (run.dir / phase).rglob("*.eval"):
            head = read_eval_log(str(f), header_only=True)
            cell = f"{head.eval.task_args['family']}-{head.eval.task_args['level']}"
            assert head.eval.config.token_limit == (expected[cell]["cap"] if cell in expected else None), (phase, cell)
    assert "F3-60" not in run_study.token_caps(run, "pilot") and "F3-60" in all_caps


def test_caps_borrow_the_nearest_measured_level_and_leave_unmeasured_families_uncapped(clean_env):
    run = StudyRun("main", "caps", offline=True, runs_root=clean_env / "runs")
    run_study.write_caps(run.token_caps_path, {"F7-10": {"b0": 100.0, "samples": 2}, "F7-1000": {"b0": 50.0, "samples": 2}, "F1-2": {"b0": 10.0, "samples": 2}}, "micro-pilot")
    caps = run_study.token_caps(run)
    assert caps["F7-100"]["cap"] == 800 and "F7-10" in caps["F7-100"]["source"], "equidistant on a log scale: the larger cap"
    assert caps["F1-32"]["cap"] == 80 and caps["F1-32"]["source"] == "borrowed from F1-2 (micro-pilot)"
    assert {"F2-2", "F2-10", "F3-5", "F3-60"} <= set(run_study.uncapped_cells(run)), "no B0 in the family: uncapped"
    run_study.write_caps(run.pilot_caps_path, {"F3-5": {"b0": 7.0, "samples": 1}, "F7-10": {"b0": 1.0, "samples": 1}}, "pilot")
    caps = run_study.token_caps(run)
    assert caps["F3-5"]["cap"] == 56 and caps["F3-60"]["source"] == "borrowed from F3-5 (pilot)" and caps["F7-10"]["cap"] == 800, "the micro-pilot's cap stands"
    assert run_study.token_caps(run, "micro-pilot") == {} and "F3-5" not in run_study.token_caps(run, "pilot")
    assert run_study.token_caps(StudyRun("study_g", "caps", offline=True, runs_root=clean_env / "runs")) == {}, "Study G is not capped"


def test_b0_counts_every_model_of_a_sample_and_skips_errored_ones(monkeypatch):
    from types import SimpleNamespace

    import inspect_ai.log

    usage = lambda n: SimpleNamespace(total_tokens=n)  # noqa: E731
    samples = [
        SimpleNamespace(error=None, metadata={"family": "F1", "level": "2"}, model_usage={"a": usage(100), "b": usage(20)}),
        SimpleNamespace(error=None, metadata={"family": "F1", "level": "2"}, model_usage={"a": usage(300)}),
        SimpleNamespace(error="boom", metadata={"family": "F1", "level": "2"}, model_usage={"a": usage(10**9)}),
    ]
    logs = {"s1.eval": SimpleNamespace(eval=SimpleNamespace(task_args={"arm": "S1", "family": "F1", "level": "2"}, metadata={}), samples=samples)}
    logs["s5.eval"] = SimpleNamespace(eval=SimpleNamespace(task_args={"arm": "S5"}, metadata={}), samples=samples)
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda f, **kw: logs[str(f)])
    assert run_study.b0_from_logs(["s1.eval", "s5.eval"]) == {"F1-2": {"b0": 210.0, "samples": 2}}, "S1 only; both models; no errored sample"


# --- Builds (B6) ---------------------------------------------------------------------------------------


def test_main_builds_cover_the_kg_cells_and_share_the_gates_dev_worlds(clean_env):
    _fake_gate_run(clean_env / "runs", "g-go", "GO")
    _fake_gate_run(clean_env / "runs", "g-nogo", "NO_GO")
    live = StudyRun("main", "b6", runs_root=clean_env / "runs", gate_run_id="g-go")
    assert run_study.kg_worlds(live, "pilot") == {c: 2 for c in ("F1-2", "F1-32", "F7-10", "F7-1000", "F3-5", "F3-60")}
    assert run_study.kg_worlds(live, "test") == {c: 9 for c in ("F3-5", "F3-60", "F7-10", "F7-1000", "F1-32", "F7-100")}
    assert run_study.kg_build_problems(live) == [], "main.build.kg (11 per KG cell) covers pilot + test"
    cell = run_study.kg_build_cell(live, "test")[0]
    assert cell.id == "main.build.kg" and cell.spec["systems"] == ["apg"] and cell.spec["worlds"]["F7-100"] == 9
    assert run_study.kg_build_cell(StudyRun("main", "b6n", runs_root=clean_env / "runs", gate_run_id="g-nogo"), "test")[0].spec["systems"] == ["lightrag"]
    dev = {f"{s['family']}-{s['level']}": s for s in run_study.world_specs(live, "dev")}
    assert set(dev) == {"F1-32", "F2-10", "F3-60", "F7-1000"}
    assert dev["F7-1000"]["shared"] is True and dev["F7-1000"]["n_tasks"] == run_study.plan(live).cell("gate.tune").spec["tasks_per_world"] and dev["F7-1000"]["kinds"] == ["chunks", "apg"]
    assert "shared" not in dev["F1-32"] and dev["F1-32"]["kinds"] == [] and dev["F1-32"]["seed_base"] == 1000
    assert run_study.kg_worlds(live, "dev") == {}, "shared dev worlds are the gate's: not projected again"


def test_a_live_build_dev_refuses_another_builder_than_the_gate_runs(clean_env):
    _fake_gate_run(clean_env / "gates", "g-go", "GO")
    run = StudyRun("main", "b", runs_root=clean_env / "runs", gate_run_id="g-go", gate_runs_root=clean_env / "gates")
    assert run_study._refuse_other_builder(run) is None, "no gate build-dev record: nothing to compare"
    ours = run_study.build_params(run)
    gate_manifest = clean_env / "gates" / "g-go" / "build-dev" / "manifest.json"
    run_gate._write_json(gate_manifest, {"phase": "build-dev", "status": "done", "params": {"builder": ours}})
    assert run_study._refuse_other_builder(run) is None
    run_gate._write_json(gate_manifest, {"phase": "build-dev", "status": "done", "params": {"builder": {"model": "gpt-6-sol", "effort": "medium", "fallback": True}}})
    assert "re-author the gate's artifacts; set APE_BUILD_FALLBACK as the gate run did" in run_study._refuse_other_builder(run)
    assert run_study._refuse_other_builder(StudyRun("main", "b", offline=True, runs_root=clean_env / "runs")) is None, "offline builds are the run's own"


def test_a_shared_dev_world_is_never_overwritten(clean_env, monkeypatch):
    monkeypatch.setenv("APE_WORLDS", str(clean_env / "worlds"))
    gate_world = make_world("F3", "60", "dev", 0, 12)
    gate_world.save(Config().world_path(gate_world.id))
    run = StudyRun("main", "shared", offline=True, runs_root=clean_env / "runs")
    spec = {"group": "worlds", "family": "F3", "level": "60", "count": 1, "n_tasks": 3, "exception_style": "descriptive", "relational": True, "seed_base": 1000, "knobs": None, "kinds": [], "shared": True}
    with pytest.raises(PhaseError, match="another version of F3-60-dev-s1000, a world main shares with the gate's dev split"):
        run_gate.build_world_set(run, {"outputs": {}, "warnings": []}, "build-dev", "dev", [spec], study="main")
    assert World.load(Config().world_path(gate_world.id)).content_hash() == gate_world.content_hash(), "the gate's world is untouched"
    record = {"outputs": {}, "warnings": []}
    run_gate.build_world_set(run, record, "build-dev", "dev", [spec | {"n_tasks": 12}], study="main")
    assert record["worlds"]["count"] == 1, "the gate's own parameters: the same world, kept"


def test_build_quality_measures_registry_worlds_and_single_system_builds():
    from ape import build_quality as bq

    world = make_world("F1", "32", "dev", 0, 2)
    ids = bq.registry_spec_ids(world)
    assert ids and all(i.startswith(("EC-", "SUP-")) for i in ids) and bq.registry_spec_ids(make_world("F7", "10", "dev", 0, 2)) == []
    row = {"world_id": "w", "cell": "F1-32", "exception_style": "descriptive", "decisive": True, "errors": [], "lightrag": None, "apg": {"ids": 10, "covered": 10, "coverage": 1.0, "metric": "fact_coverage"}}
    assert bq.aggregate([row], expected_cells=["F1-32"], systems=("apg",))["verdict"] == "builder_passes"
    assert bq.aggregate([row], expected_cells=["F1-32"])["verdict"] == "fallback_needed", "the gate's default needs both systems"
    low = row | {"apg": row["apg"] | {"covered": 5, "coverage": 0.5}}
    assert "APG fact_coverage 0.500" in bq.aggregate([low], expected_cells=["F1-32"], systems=("apg",))["cells"]["F1-32"]["reasons"][0]


# --- Tuning, sessions, budget, freeze -------------------------------------------------------------------


def test_a_live_tune_refuses_a_placeholder_or_unsigned_grid(clean_env):
    grid = yaml.safe_load((ROOT / "config" / "tuning_grid_main.yaml").read_text())
    assert grid["placeholder"] is True and any("placeholder" in p for p in run_study.tuning_signoff_problems(grid))
    signed = grid | {"placeholder": False, "owners": {"S3s": "Ada"}, "signed_off": {"S3s": True}}
    assert run_study.tuning_signoff_problems(signed) == []
    assert run_study.tuning_signoff_problems(signed | {"owners": {"S3s": "TODO"}}) == ["owners.S3s is not set"]
    assert yaml.safe_load((ROOT / "config" / "tuning_grid_study_g.yaml").read_text())["placeholder"] is True
    run = StudyRun("study_g", "tune", runs_root=clean_env / "runs")
    assert "is not signed off for a live tune" in run_study._refuse_tune(run)
    assert run_study._refuse_tune(StudyRun("study_g", "tune", offline=True, runs_root=clean_env / "runs")) is None


def test_selections_that_set_the_same_knob_run_in_eval_sets_of_their_own(clean_env):
    """Study G's CM arms share APE_CM_* knobs: when two selections set one differently they cannot share an environment
    (knobs are not Inspect task args), so each tuned arm runs alone under its own selection; otherwise every selection's
    knobs apply together, as in the gate."""
    run = StudyRun("study_g", "groups", offline=True, runs_root=clean_env / "runs")
    agree = {"CM-todo": {"arm": "CM-todo", "env": {"APE_CM_TODO_EXTRACT": "false"}}, "S-CM*": {"arm": "S-CM*", "env": {"APE_CM_STACK": "trim+todo"}}}
    assert run_study.selection_env(agree) == {"APE_CM_TODO_EXTRACT": "false", "APE_CM_STACK": "trim+todo"}
    clash = agree | {"CM-reset": {"arm": "CM-reset", "env": {"APE_CM_TODO_EXTRACT": "true", "APE_CM_RESET_EVERY": "3"}}}
    assert run_study.selection_env(clash) is None
    run.out_config_dir.mkdir(parents=True)
    run.selected_path.write_text(yaml.safe_dump(clash))
    groups = {(g["cell"], g["name"]): g for g in run_study.run_groups(run, "test")}
    cm = {name: g for (cell, name), g in groups.items() if cell == "g.cm.luna-high"}
    assert set(cm) == {"selected", "sel-CM-todo", "sel-CM-reset"}
    assert cm["sel-CM-reset"]["env"] == {"APE_CM_TODO_EXTRACT": "true", "APE_CM_RESET_EVERY": "3"} and [a["run"] for a in cm["sel-CM-reset"]["arms"]] == ["CM-reset"]
    assert cm["sel-CM-todo"]["env"] == {"APE_CM_TODO_EXTRACT": "false"} and cm["selected"]["env"] == {}
    assert {a["run"] for a in cm["selected"]["arms"]} == {"CM0", "CM-prune", "CM-sum", "CM-native", "O-state"}
    assert groups[("g.topo.luna", "sel-S-CM_")]["env"] == {"APE_CM_STACK": "trim+todo"}, "a key's odd characters stay out of the dir name"
    run.selected_path.write_text(yaml.safe_dump(agree))
    together = [g for g in run_study.run_groups(run, "test") if g["cell"] == "g.cm.luna-high"]
    assert [g["name"] for g in together] == ["selected"] and together[0]["env"] == run_study.selection_env(agree)


def test_the_budget_guard_counts_the_studys_allocation(clean_env, monkeypatch):
    run = StudyRun("main", "guard", offline=True, runs_root=clean_env / "runs")
    assert run_phases(run, "preflight") == {"preflight": "done"}

    allocation = float(run_study.plan(run).budget["allocations"]["main"])
    spent = allocation - 1.0  # the main study's allocation all but $1 spent; the program has plenty left

    def main_nearly_spent(budget_usd, *_a, **_k):
        return dict.fromkeys(run_gate.SPEND_KEYS) | {"budget_usd": budget_usd, "spent_usd": spent, "remaining_usd": budget_usd - spent, "by_study": {"main-offline": spent}}

    monkeypatch.setattr(run_gate, "program_remaining", main_nearly_spent)
    with pytest.raises(BudgetError, match=r"phase micro-pilot \(main: the smaller of the program's and the study's allocation's remainder\): projected \$[\d.]+ exceeds the remaining \$1.00"):
        run_phases(run, "micro-pilot")
    m = json.loads(run.manifest_path("micro-pilot").read_text())
    assert m["status"] == "failed" and m["spend_at_start"]["study_remaining_usd"] == 1.0 and m["spend_at_start"]["remaining_usd"] == 5000.0 - spent
    assert not (run.work_dir / "worlds" / "pilot").exists(), "nothing was built"


def _fake_current(run: StudyRun) -> None:
    """Done manifests for preflight and every phase the freeze rests on, with their real current fingerprints."""
    run_gate._write_json(run.manifest_path("preflight"), {"phase": "preflight", "status": "done", "fingerprint": "fake", "finished": "2026-10-01T00:00:00+00:00", "outputs": {}})
    for phase in run.spec.rests_on:
        st = run_study.phase_state(run, phase)
        run_gate._write_json(run.manifest_path(phase), {"phase": phase, "status": "done", "finished": "2026-10-01T00:00:00+00:00", "outputs": {}} | st)


def test_a_live_freeze_refuses_placeholders_then_freezes_a_main_block_the_gate_never_counts(clean_env, monkeypatch):
    tmp = clean_env
    monkeypatch.setenv("APE_CACHE", str(tmp / "cache"))  # the spend check reads this ledger, not the real one
    _fake_gate_run(tmp / "gates", "g-go", "GO")  # outside runs/, where the gate's own seed-block scan looks
    prereg, provenance = tmp / "PREREGISTRATION_MAIN.md", tmp / "PROVENANCE.md"
    shutil.copy(ROOT / "PREREGISTRATION_MAIN.md", prereg)
    shutil.copy(ROOT / "PROVENANCE.md", provenance)
    kw = {"runs_root": tmp / "runs", "gate_run_id": "g-go", "gate_runs_root": tmp / "gates", "prereg_path": prereg, "provenance_path": provenance}
    run = StudyRun("main", "live-freeze", **kw)
    out = run.out_config_dir
    out.mkdir(parents=True)
    (out / "selected.yaml").write_text(yaml.safe_dump({"S3s": {"arm": "S3s", "env": {"APE_S3S_BUDGET": "2000"}, "candidate": "c"}}))
    run_study.write_caps(run.token_caps_path, {"F1-2": {"b0": 100.0, "samples": 2}}, "micro-pilot")
    run_study.write_caps(run.pilot_caps_path, {}, "pilot")
    (out / "s7_targets.json").write_text(json.dumps({"F7-10": 120}))
    run_study._check_mode(run)
    _fake_current(run)
    monkeypatch.setattr(run_gate, "git_tracked_changes", lambda: [])

    # 1. The placeholder pre-registration (and no analysis module) blocks the live freeze; nothing is frozen.
    with pytest.raises(PhaseError, match=r"(?s)unfilled item\(s\).*\[USER: main-study pre-registration.*analysis not implemented"):
        run_phases(run, "freeze")
    assert read_freeze(run) is None and json.loads(run.manifest_path("freeze").read_text())["status"] == "failed"

    # 2. Filled, with an analysis module: frozen on main's first block, recorded with the KG arm and a study marker.
    text = prereg.read_text()
    start = run_gate.prereg_body_start(text)
    prereg.write_text(text[:start] + run_gate.PLACEHOLDER_ITEM.sub("filled", text[start:]))
    monkeypatch.setattr(run_study, "analysis_available", lambda r: True)
    real = (ROOT / "PROVENANCE.md").read_bytes()
    assert run_phases(run, "freeze") == {"freeze": "done"}
    freeze = run_study.require_frozen(run)
    assert freeze["rehearsal"] is False and freeze["test_seeds"]["base"] == 13000 and freeze["kg"]["arm"] == "APG-q" and freeze["code_commit"]
    assert {"PREREGISTRATION_MAIN.md", "config/tuning_grid_main.yaml", "config/token_caps.json", "config/s7_targets.json", "gate/selected.yaml", "gate/decision.json", "uv.lock"} <= set(freeze["files"])
    assert "<!-- ape:test-seeds study=main run=live-freeze base=13000 count=9 -->" in provenance.read_text() and (ROOT / "PROVENANCE.md").read_bytes() == real
    assert run_study.choose_test_seed_base(StudyRun("main", "next", **kw))[0] == 13100
    assert run_gate.choose_test_seed_base(GateRun("gate-next", runs_root=tmp / "runs", provenance_path=provenance))[0] == 3000, "the gate never counts main's block"
    # 3. Frozen once; a changed frozen input (the gate run's selection) stops build-test and test, naming it.
    with pytest.raises(PhaseError, match="frozen once"):
        run_phases(StudyRun("main", "live-freeze", force=True, **kw), "freeze")
    for phase in ("micro-pilot", "tune", "pilot"):
        with pytest.raises(PhaseError, match=f"{phase}: run 'live-freeze' is frozen"):
            run_phases(StudyRun("main", "live-freeze", force=True, **kw), phase)
    gate_selected = tmp / "gates" / "g-go" / "config" / "selected.yaml"
    gate_selected.write_text(gate_selected.read_text() + "\n# edited\n")
    with pytest.raises(PhaseError, match=r"gate/selected.yaml \(.*\): changed"):
        run_study.require_frozen(StudyRun("main", "live-freeze", **kw))


def test_live_build_test_and_test_refuse_code_or_knobs_other_than_the_frozen_ones(clean_env, monkeypatch):
    run = StudyRun("study_g", "g2", runs_root=clean_env / "runs")
    monkeypatch.setattr(run_study, "require_frozen", lambda r: {"frozen_at": "t", "code_commit": "abc", "test_seeds": {"base": 23100, "count": 10}})
    monkeypatch.setattr(run_study, "read_freeze", lambda r: {"code_commit": "abc", "test_seeds": {"base": 23100, "count": 10}, "design_env": {}, "kg": {"needed": False}})
    monkeypatch.setattr(run_gate, "code_drift", lambda commit: [])
    assert run_study._refuse_unless_frozen("test")(run) is None, "a later block is fine: f8_session selects the run's own (B7)"
    monkeypatch.setenv("APE_CM_PRUNE_KEEP", "5")
    assert "design knob(s) differ from the frozen ones: APE_CM_PRUNE_KEEP" in run_study._refuse_unless_frozen("test")(run)
    monkeypatch.delenv("APE_CM_PRUNE_KEEP")
    monkeypatch.setattr(run_gate, "code_drift", lambda commit: ["src/ape/agent/session.py"])
    assert "code changed since the freeze commit" in run_study._refuse_unless_frozen("build-test")(run)


def _write_smoke(d: Path, required: list[str]) -> None:
    """A passing live smoke of this code, as readiness/smoke.py records it (report.json, checks.json)."""
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat(timespec="seconds")
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps({"dry": False, "mode": "live", "status": "pass", "models": {"profile": "gate", "overrides": {}}, "checks": {}, "finished_utc": now}))
    entry = {"status": "pass", "finished_utc": now, "git_commit": run_gate.git_state()["commit"], "profile": "gate", "overrides": {}, "snapshots": {}}
    (d / "checks.json").write_text(json.dumps({"required": required, "checks": dict.fromkeys(required, entry)}))


def test_a_live_study_preflight_needs_a_fresh_smoke_and_the_studys_own_checks(clean_env, monkeypatch):
    from dataclasses import replace

    from ape.models import PreflightError

    monkeypatch.setattr(run_gate, "code_drift", lambda commit: [])  # the suite runs on whatever working tree it finds
    monkeypatch.setenv("APE_CACHE", str(clean_env / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")  # nothing here makes a request
    probe = clean_env / "openai_probe.json"
    probe.write_text(json.dumps({"models_available": ["gpt-6-luna", "gpt-6-sol", "gpt-6-astra", "text-embedding-3-small"]}))
    kw = {"runs_root": clean_env / "runs", "probe_path": probe, "env_path": clean_env / "no.env", "smoke_dir": clean_env / "smoke"}
    with pytest.raises(PreflightError, match="no live smoke record") as e:
        run_phases(StudyRun("study_g", "g1", **kw), "preflight")
    assert "g.pilot.luna: CM-native needs confirmed native compaction for openai/gpt-6-luna" in str(e.value)
    _write_smoke(clean_env / "smoke", ["effort", "orchestrator"])
    with pytest.raises(PreflightError, match="g.cm.luna-high: CM-native needs confirmed native compaction") as e:
        run_phases(StudyRun("study_g", "g1", **kw), "preflight")
    assert "no live smoke record" not in str(e.value), "CM-native's support is checked at preflight, before any sample"
    # The B11 probe records the support; then the preflight passes.
    from ape.agent.cm_arms import NATIVE_RECORD_ENV, record_native_support

    monkeypatch.setenv(NATIVE_RECORD_ENV, str(record_native_support("openai/gpt-6-luna", True, evidence="test", path=clean_env / "native.json")))
    assert run_phases(StudyRun("study_g", "g1", **kw), "preflight") == {"preflight": "done"}
    m = json.loads(StudyRun("study_g", "g1", **kw).manifest_path("preflight").read_text())
    assert m["checks"]["smoke"] == "ok" and m["checks"]["probe"]["missing"] == [] and set(m["profiles"]) == {"study_g_luna", "study_g_sol", "study_g_astra"}
    assert m["checks"]["cm_native"] == {"g.pilot.luna": "provider", "g.cm.luna-high": "provider"}
    # A study's own required checks (BUILD_PLAN B12 adds them) are held to the same conditions.
    monkeypatch.setitem(run_study.STUDIES, "study_g", replace(STUDIES["study_g"], smoke_checks=("g-sessions",)))
    with pytest.raises(PreflightError, match="smoke check 'g-sessions' has no live result"):
        run_phases(StudyRun("study_g", "g2", **kw), "preflight")
    _write_smoke(clean_env / "smoke", ["effort", "orchestrator", "g-sessions"])
    assert run_phases(StudyRun("study_g", "g2", **kw), "preflight") == {"preflight": "done"}


def test_a_live_analyze_refuses_without_the_analysis_module(clean_env):
    run = StudyRun("study_g", "an", runs_root=clean_env / "runs")
    with pytest.raises(PhaseError, match="analysis not implemented: ape.analyze_g does not exist yet"):
        run_study._analyze(run, {"outputs": {}, "warnings": []})


def test_the_world_set_builder_takes_a_studys_seed_base_and_f8_knobs(clean_env, monkeypatch):
    """The gate's world-set builder, given a study's spec: its own seed base, the F8 knob variant, no artifacts."""
    monkeypatch.setenv("APE_WORLDS", str(clean_env / "worlds"))
    run = StudyRun("study_g", "w", offline=True, runs_root=clean_env / "runs")
    spec = {"group": "sessions", "family": "F8", "level": "20", "count": 1, "n_tasks": 0, "exception_style": "descriptive", "relational": True, "seed_base": 22000, "knobs": {"output_tokens": 2250}, "kinds": []}
    record = {"outputs": {}, "warnings": []}
    worlds = run_gate.build_world_set(run, record, "micro-pilot", "pilot", [spec], study="study_g")
    assert [w["world_id"] for w in worlds] == ["F8-20-o2250-pilot-s22000"] and worlds[0]["artifacts"] == {} and worlds[0]["kinds"] == []


def test_the_analyze_phase_calls_the_studys_analysis_entry_point(clean_env, monkeypatch):
    """The contract B4 / B10 implement: `ape.analyze_main.analyze(run) -> dict`, writing report/report.md."""
    import sys
    import types

    calls = []

    def analyze(run):
        calls.append(run.run_id)
        (run.dir / "report").mkdir(parents=True, exist_ok=True)
        (run.dir / "report" / "report.md").write_text("# report\n")
        return {"status": "done", "label": "H1 supported"}

    monkeypatch.setitem(sys.modules, "ape.analyze_main", types.SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(run_study, "analysis_available", lambda r: True)
    run = StudyRun("main", "an", runs_root=clean_env / "runs", gate_run_id="g")
    record: dict = {"outputs": {}, "warnings": []}
    run_study._analyze(run, record)
    assert calls == ["an"] and record["analysis"] == {"status": "done", "label": "H1 supported"} and set(record["outputs"]) == {"report.md"}
    monkeypatch.setitem(sys.modules, "ape.analyze_main", types.SimpleNamespace(analyze=lambda r: {}))
    (run.dir / "report" / "report.md").unlink()
    with pytest.raises(PhaseError, match="wrote no"):
        run_study._analyze(run, {"outputs": {}, "warnings": []})


def test_an_extension_names_a_frozen_primary_run_of_the_same_study(clean_env):
    runs = clean_env / "runs"
    run = StudyRun("main", "ext", offline=True, runs_root=runs, extension_of="m1")
    assert "must be a frozen main run" in run_study.role_problems(run, run_study.run_role(run))[0]
    run_gate._write_json(runs / "main" / "m1" / "freeze.json", {"run_id": "m1", "offline": True, "role": {"kind": "primary", "of": None}})
    assert run_study.role_problems(run, run_study.run_role(run)) == []
    run_gate._write_json(runs / "main" / "m1" / "freeze.json", {"run_id": "m1", "offline": False, "role": {"kind": "extension", "of": "m0"}})
    problems = run_study.role_problems(run, run_study.run_role(run))
    assert any("itself an extension" in p for p in problems) and any("is live and this one is not" in p for p in problems)
    assert run_study.role_problems(StudyRun("study_g", "m1", offline=True, runs_root=runs, extension_of="m1"), {"kind": "extension", "of": "m1"}) == ["--extension-of names this run itself"]


def test_a_short_budget_stops_the_test_after_the_primary_cells(offline, monkeypatch):
    """As the gate's: the test phase guards the primary groups at its start and every other group before it runs, so a
    budget stop leaves the primary evidence complete. (Last in this module: it leaves the shared run's test failed.)"""
    run = StudyRun("main", "t1", offline=True, force=True, runs_root=offline["runs"], config_dir=offline["config"])
    real = run_study.group_remaining
    monkeypatch.setattr(run_study, "group_remaining", lambda r, g, p: real(r, g, p) if g["primary"] else 1e9)
    with pytest.raises(BudgetError, match=r"(?s)test main.C.a \(selected\): projected \$1,000,000,000.00 exceeds.*the primary cells are complete"):
        run_phases(run, "test")
    m = _manifest(run, "test")
    assert m["status"] == "failed" and m["primary_complete"] is True
    statuses = {c: x["status"] for c, x in m["cells"].items()}
    primary = ["main.A.s1-pool", "main.A.arms", "main.A.m1s", "main.B.s1-pool", "main.B.arms"]
    assert list(statuses)[:5] == primary, "the primary phases' cells first"
    assert statuses == dict.fromkeys(primary, "done") | dict.fromkeys(["main.C.a", "main.C.b", "main.F.luna", "main.F.luna-f7-100", "main.F.sol", "main.F.sol-m2"], "stopped")
