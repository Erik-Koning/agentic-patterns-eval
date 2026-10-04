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

    def kg(gid: str | None, offline: bool = False) -> dict:  # a run per gate run: a run keeps the resolution it copied (R-A1)
        return run_study.kg_resolution(StudyRun("main", f"m-{gid}", offline=offline, runs_root=runs, gate_run_id=gid))

    go = kg("g-go")
    assert (go["key"], go["arm"], go["system"], go["env"], go["verdict"]) == ("APG*", "APG-q", "apg", {"APE_APG_SHORTLIST_K": "24"}, "GO")
    assert set(go["files"]) == {"config/gate_resolution.json"}, "R-A1: the run's copy of the gate resolution, never the gate's report"
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
        assert os.environ["APE_S3S_BUDGET"] == "2000" and "APE_S3S_BUDGET" not in run_study.env_knobs(run), "S3s's inherited knobs too"
    assert "APE_KG_ARM" not in os.environ and "APE_S3S_BUDGET" not in os.environ
    monkeypatch.setenv("APE_APG_SHORTLIST_K", "48")
    with pytest.raises(PhaseError, match=r"APE_APG_SHORTLIST_K=48 \(the gate's verdict sets 24\)"), run_study.run_environment(run):
        pass


def test_s3s_inherits_the_gate_runs_selection(clean_env, monkeypatch):
    """D-041: the main grid inherits S3s from the gate (`inherited: {S3s: {source: gate, gate_key: S3s}}`): the gate
    run's S3s selection is set with the KG arm for every phase, recorded in the params and the freeze, and a live run
    refuses a gate run without it."""
    runs = clean_env / "runs"
    _fake_gate_run(runs, "g-go", "GO")
    _fake_gate_run(runs, "g-nos3s", "GO", selected={k: v for k, v in SELECTED.items() if k != "S3s"})
    _fake_gate_run(runs, "g-other", "GO", selected=SELECTED | {"S3s": {"arm": "S3", "env": {}, "candidate": "x"}})
    assert run_study.gate_inherited_arms(StudyRun("main", "m", runs_root=runs)) == {"S3s": "S3s"}
    assert run_study.gate_inherited_arms(StudyRun("study_g", "g", runs_root=runs)) == {}
    run = StudyRun("main", "m", runs_root=runs, gate_run_id="g-go")
    kg = run_study.kg_resolution(run)
    assert kg["inherited"] == {"S3s": {"key": "S3s", "arm": "S3s", "env": {"APE_S3S_BUDGET": "2000"}, "candidate": "s3s-2000"}}
    assert run_study.kg_env(run) == {"APE_KG_ARM": "APG-q", "APE_APG_SHORTLIST_K": "24", "APE_S3S_BUDGET": "2000"}
    assert run_study.kg_params(run)["inherited"] == kg["inherited"] and run_study.base_params(run)["kg"]["inherited"] == kg["inherited"], "in every fingerprint"
    for gid, why in (("g-nos3s", "has no S3s selection, which main's S3s inherits"), ("g-other", "the S3s selection runs S3, but main's S3s inherits only its knobs")):
        with pytest.raises(PhaseError, match=why):
            run_study.kg_resolution(StudyRun("main", "m", runs_root=runs, gate_run_id=gid))
        off = run_study.kg_resolution(StudyRun("main", "m", offline=True, runs_root=runs, gate_run_id=gid))
        assert off["inherited"] == {} and any("offline: S3s at its defaults" in n for n in off["notes"])
    assert run_study.kg_resolution(StudyRun("main", "m", offline=True, runs_root=runs))["inherited"] == {}
    # A live shell value for the inherited knob that disagrees is refused, as the KG arm's.
    monkeypatch.setenv("APE_S3S_BUDGET", "4000")
    with pytest.raises(PhaseError, match=r"APE_S3S_BUDGET=4000 \(the gate's verdict sets 2000\)"), run_study.run_environment(run):
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


def test_a_run_keeps_the_gate_resolution_it_copied_and_a_gate_reanalysis_never_touches_it(clean_env):
    """R-A1: the first resolution copies the gate's verdict, KG arm and selection entries into the run
    (config/gate_resolution.json); fingerprints and the freeze hash that copy. A gate re-analysis (a new decision.json,
    same verdict) changes nothing; a changed selection is refused live (a new resolution is a new run id)."""
    runs = clean_env / "runs"
    d = _fake_gate_run(runs, "g-go", "GO")
    run = StudyRun("main", "m", runs_root=runs, gate_run_id="g-go")
    before = run_study.phase_state(run, "build-dev")
    copy = json.loads(run.gate_resolution_path.read_text())
    assert copy["verdict"] == "GO" and copy["arm"] == "APG-q" and copy["gate_selected"]["APG*"]["arm"] == "APG-q" and copy["gate_selected"]["S3s"]["candidate"] == "s3s-2000"
    assert copy["inherited"]["S3s"]["env"] == {"APE_S3S_BUDGET": "2000"} and copy["gate_freeze_sha256"] == run_gate._sha256(d / "freeze.json")
    # The gate re-analyses: decision.json is rewritten (a new generated_at), the verdict stands.
    run_gate._write_json(d / "report" / "decision.json", {"verdict": {"label": "GO", "reasons": []}, "generated_at": "later"})
    assert run_study.phase_state(StudyRun("main", "m", runs_root=runs, gate_run_id="g-go"), "build-dev")["fingerprint"] == before["fingerprint"]
    # The gate's selection changes: refused live; a new run id resolves the new one.
    (d / "config" / "selected.yaml").write_text(yaml.safe_dump({**SELECTED, "APG*": {"arm": "APG-s", "env": {"APE_APG_SHORTLIST_K": "12"}}}))
    with pytest.raises(PhaseError, match=r"resolution changed since this run copied it at .* \(arm, candidate, declared, env, gate_selected\)"):
        run_study.kg_resolution(StudyRun("main", "m", runs_root=runs, gate_run_id="g-go"))
    after = run_study.phase_state(StudyRun("main", "m2", runs_root=runs, gate_run_id="g-go"), "build-dev")
    assert after["params"]["kg"]["arm"] == "APG-s" and after["fingerprint"] != before["fingerprint"]
    # Offline the copy follows the gate run, with a note.
    off = run_study.kg_resolution(StudyRun("main", "m", offline=True, runs_root=runs, gate_run_id="g-go"))
    assert off["arm"] == "APG-s" or off["arm"] == run_gate.OFFLINE_ARMS.get("APG-s", "APG-s")
    # The gate's analyze no longer lists the shared build ledger, which the studies' builds append to.
    gate = GateRun("g-x", offline=True, runs_root=runs)
    assert "ledger" not in run_gate._analyze_inputs(gate) and "ledger" not in run_study._analyze_inputs(run)


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
    # Study G: a session arm is built when it is a registered context policy or session arm (B8's CM arms, B9's
    # topology arms S1, M1 and M2): all of them now.
    g = StudyRun("study_g", "arms", offline=True, runs_root=clean_env / "runs")
    assert all(run_study.unbuilt_arms(g, p) == [] for p in ("micro-pilot", "tune", "test")), "B8 and B9 built every Study G arm"
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
    for phase in ("micro-pilot", "test"):
        assert _manifest(g, phase)["skipped_arms"] == [], f"every Study G arm runs, B9's topology arms included ({phase})"
    assert _manifest(g, "tune")["skipped_systems"] == [] and _manifest(run, "tune")["skipped_systems"] == []
    assert not _manifest(g, "preflight")["unbuilt_arms"]
    assert {c: x["status"] for c, x in _manifest(g, "test")["cells"].items()} == {c.id: "done" for c in run_study.phase_cells(g, "test")}


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
        # The analysis: each study's own module (ape.analyze_main, ape.analyze_g) writes the real report, never the stub.
        assert run_study.analysis_available(run) and not (run.dir / "report" / "analysis.json").exists()
        assert {"decision.json", "report.md"} <= set(manifests["analyze"]["outputs"]) and (run.dir / "report" / "report.md").is_file()
        json.loads((run.dir / "report" / "decision.json").read_text())
        assert manifests["analyze"]["analysis"]["status"] in ({"done"} if study == "main" else {"ok", "partial"})
        # The analysis reads the tuning log and the study's models slice (a change re-runs it); the test manifest
        # records each group's model overrides and effort, and its log files relative to the run directory.
        assert {"tune/tuning_log.jsonl", "config/models.yaml", "config/selected.yaml"} <= set(manifests["analyze"]["inputs"])
        assert manifests["analyze"]["inputs"]["config/models.yaml"]["slice"] == study and manifests["analyze"]["params"]["tune_manifest"]
        test = manifests["test"]
        groups = [g for c in test["cells"].values() for g in c["groups"] if g["status"] == "done"]
        assert test["log_files_relative_to"] == "run_dir" and groups and all("models" in g and "effort" in g for g in groups)
        assert all(not Path(f).is_absolute() and (run.dir / f).is_file() for g in groups for f in g["log_files"])
    assert not (ROOT / "runs" / "main" / "t1").exists() and not (ROOT / "runs" / "study_g" / "t1").exists()


def test_offline_main_runs_its_cells_on_its_worlds_with_the_kg_arm_and_the_selections(offline):
    from inspect_ai.log import read_eval_log

    run = _run(offline, "main")
    selected = yaml.safe_load(run.selected_path.read_text())
    grid = yaml.safe_load((offline["config"] / "tuning_grid_main.yaml").read_text())
    assert set(selected) == set(grid["systems"]) == set(run_study.study_grid(run)[0]["systems"]) and all(sel["arm"] == name for name, sel in selected.items()), "each system runs as its plan arm"
    tune = _manifest(run, "tune")
    assert not any("placeholder" in w for w in tune["warnings"]) and tune["tuning_completeness"]["pass"] is True, "PC6-style"
    assert tune["grid_problems"] == run_study.grid_problems(run), "recorded (a live tune refuses on any; M1k/M2 leave the grid with F-arms, D-047)"
    assert "S3s" not in selected and tune["kg"]["inherited"] == {} and any("S3s: its defaults offline" in w for w in _manifest(run, "preflight")["warnings"]), "S3s is the gate's (D-041)"
    for name, sdef in grid["systems"].items():  # offline: each system's first candidates, one selected
        ran = [c["id"] for c in sdef["candidates"][: run_study.OFFLINE_SCALE["tune_candidates_per_system"]]]
        assert tune["tuning_completeness"]["systems"][name]["logged"] == sorted(ran) and selected[name]["candidate"] in ran
    kg = _manifest(run, "test")["kg"]
    assert kg["arm"] == "APG-s" and kg["system"] == "apg" and read_freeze(run)["kg"]["arm"] == "APG-s"
    targets = json.loads(run.s7_targets_path.read_text())
    assert set(targets) == {"F3-5", "F3-60", "F7-10", "F7-1000"} and all(t > 0 for t in targets.values())
    seen, records = set(), {}
    for f in sorted((run.dir / "test").rglob("*.eval")):
        log = read_eval_log(str(f))
        args, md = log.eval.task_args, log.eval.metadata
        assert log.status == "success" and log.eval.model == "mockllm/model" and not any(s.error for s in log.samples)
        assert args["seed_base"] == STUDY_SEEDS["main"]["offline_test"] and args["split"] == "test" and args["plan_cell"] == md["plan_cell"]
        own = (selected.get(run_study.selection_for(args["arm"], selected)) or {}).get("env") or {}
        assert md["knobs"]["APE_KG_ARM"] == "APG-s" and all(md["knobs"].get(k) == v for k, v in own.items()), "its own selection's knobs"
        others = {k for sel in selected.values() for k in sel["env"]} - set(own)
        assert not others & set(md["knobs"]), f"{args['arm']} runs with no other selection's knobs (D-042)"
        if args["arm"] == "S7":
            assert args["group"] == "s7" and md["knobs"]["APE_S7_PER_STEP"] == "1", "S7 mirrors the per-step KG arm"
        seen.add((args["plan_cell"], args["arm"]))
        records[args["arm"]] = sorted(k for k in log.samples[0].store if k.startswith("mas_"))
    # Every main-study arm runs through micro-pilot, pilot and test offline (the gold multi-role mock plays each role).
    arms = {"S1", "S3s", "S5", "S7", "S9", "M1", "M1s", "M1k", "M2", "M7", "S8k3"}
    assert {a for _, a in seen} == arms and {("main.F.sol-m2", "M2"), ("main.F.luna-f7-100", "S5"), ("main.B.arms", "S5"), ("main.B.arms", "S7"), ("main.F.sol", "S1")} <= seen
    assert all("mas_accounting" in records[a] and "mas_agents" in records[a] for a in arms - {"S1", "S3s", "S5", "S7"}), "B2's per-agent records"
    assert "mas_council" in records["M7"] and "mas_ensemble" in records["S8k3"] and "mas_specialists" in records["M2"]
    ran = {read_eval_log(str(f), header_only=True).eval.task_args["arm"] for phase in ("micro-pilot", "pilot") for f in (run.dir / phase).rglob("*.eval")}
    assert ran == arms - {"M1s", "S8k3"}, "the pilot cells name every arm but M1s and S8k3 (Studies A and C only)"
    sol = [read_eval_log(str(f), header_only=True).eval for f in (run.dir / "test" / "main.F.sol").rglob("*.eval")]
    assert {e.metadata["arm"] for e in sol} == {"S1", "S5", "M1"} and {e.model_generate_config.reasoning_effort for e in sol} == {"high"}, "main.F.sol's arms (M1 is built since B2)"
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
    # Every context-management arm runs, and B9's session topology arms (S1, M1, M2) in the g.topo cells.
    cm = {"CM0", "CM-prune", "CM-sum", "CM-todo", "CM-reset", "CM-native", "O-state", "S-CM*"}
    assert cm | {"S1", "M1", "M2"} == {h.eval.task_args["arm"] for h in sessions} and all(h.eval.task_args["window"] == 32000 for h in sessions)
    topo = {h.eval.task_args["arm"] for h in sessions if h.eval.metadata["plan_cell"].startswith("g.topo.")}
    assert {"S1", "M1", "M2"} <= topo
    for f in sorted((run.dir / "test").rglob("*.eval")):
        head = read_eval_log(str(f), header_only=True)
        if head.eval.task.endswith("f8_session") and head.eval.task_args["arm"] in ("M1", "M2"):
            log = read_eval_log(str(f))
            assert log.status == "success" and not any(x.error for x in log.samples) and all("mas_agents" in x.store for x in log.samples), "B9's team records"
    plan = run_study.plan(run).study_g
    assert all((a := h.eval.task_args)["seed_base"] == 29000 and a["threshold"] == plan["threshold"] and a["plan_cell"] and a["group"] for h in sessions)
    sess = read_eval_log(str(next(f for f in sorted((run.dir / "test" / "g.cm.luna-high").rglob("*.eval")) if read_eval_log(str(f), header_only=True).eval.task_args["arm"] == "CM-native")))
    assert sess.samples[0].store["f8_cm_events"] and not sess.samples[0].error, "CM-native took its mock route offline"
    assert _manifest(run, "preflight")["checks"]["cm_native"] == {"g.pilot.luna": "mock", "g.cm.luna-high": "mock", "g.cm.sol-high": "mock"}, "D-043: Sol too"
    # R-B5 (d, e): the micro-pilot pilots the topology arms (g.pilot.topo) and measures every arm's session health.
    mp = _manifest(run, "micro-pilot")
    assert {g.split("/")[0] for g in mp["log_dirs"]} >= {"study_g/t1/micro-pilot/g.pilot.luna", "study_g/t1/micro-pilot/g.pilot.topo"} or any("g.pilot.topo" in d for d in mp["log_dirs"])
    health = json.loads(run.session_health_path.read_text())["arms"]
    assert {"S1", "M1", "M2", "S-CM*", "CM0", "CM-sum"} <= set(health) and all(h["sessions"] > 0 for h in health.values())
    assert health["CM0"]["overflow_expected"] and health["CM-native"]["overflow_expected"] and not health["CM-sum"]["overflow_expected"] and run_study.read_session_health(run)[1] == []
    assert "config/session_health.json" in read_freeze(run)["files"] and read_freeze(run)["session_health"] == health
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
    assert _manifest(run, "tune")["grid_problems"] == run_study.grid_problems(run), "the grid's problems, recorded offline (a live tune refuses)"
    assert _manifest(run, "micro-pilot")["operational_env"]["APE_SESSION_CHECKPOINTS"] == str(run.session_checkpoints_dir)


def test_offline_runs_give_each_eval_set_its_own_nonce_and_every_task_the_wall_clock_guard(offline):
    """B12: every eval set the runner runs (micro-pilot, tune candidates, pilot, test groups) has its own cache-nonce
    seed, each arm its own nonce within it, carried by every sample and at the head of its system prompt; every task has
    the runaway wall-clock guard (Inspect's `working_limit`)."""
    from inspect_ai.log import read_eval_log

    from ape.agent.cache_nonce import derive
    from ape.budget import sample_working_limit

    for study in ("main", "study_g"):
        run = _run(offline, study)
        created = run_gate.read_run_info(run)["created"]
        nonces: dict[tuple, str] = {}
        for phase in STUDIES[study].phases:
            for f in sorted(run.phase_dir(phase).rglob("*.eval")):
                log = read_eval_log(str(f))
                md, args = log.eval.metadata, log.eval.task_args
                seed, nonce = md["cache_nonce_seed"], md["cache_nonce"]
                assert seed.startswith(f"{study}/t1@{created}/{phase if phase != 'tune' else 'tune/'}"), seed
                parts = (args["arm"], args.get("delivery", "push"), args.get("exposure", "retrieved")) if "family" in args else (args["arm"],)
                assert nonce == derive(seed, *parts)
                assert all(s.metadata["cache_nonce"] == nonce and s.messages[0].text.startswith(f"Run reference: {nonce}\n\n") for s in log.samples)
                nonces[(seed, *parts)] = nonce
                level = args["level"]
                family = args.get("family") or md.get("family")
                assert log.eval.config.working_limit == sample_working_limit(family, level, md.get("max_turns"), run_study.plan(run)) > 0
        assert len(set(nonces.values())) == len(nonces) > 5, "no two eval sets or arms share a nonce"
        assert len({k[0] for k in nonces}) > 3


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


def test_the_pilot_cap_hit_gate_measures_every_arm_and_the_freeze_records_its_multiple(offline):
    """D-039/D-045 on the offline rehearsal: every pilot arm's cap-hit rates per task cell, a projection for every test
    (arm, cell) the pilot does not run, no re-run (the gold mock stays far below 8 x B0), and the freeze records the
    multiple, every rate (projections labelled) and the gate file (frozen like the caps)."""
    from ape.analysis.gate_stats import MAX_CAP_HIT_RATE

    run = _run(offline, "main")
    gate = json.loads(run.cap_gate_path.read_text())
    assert run_study.CAP_HIT_MAX == MAX_CAP_HIT_RATE and gate["max"] == 32 and gate["threshold"] == 0.1
    assert gate["multiple"] == 8 and gate["rule"] == "token_lower_bound" and gate["passed"] and gate["over"] == {} and [r["multiple"] for r in gate["rounds"]] == [8]
    piloted = {(a, tc) for c in run_study.phase_cells(run, "pilot") for a in rg_arm_names(c) for tc in c.spec["cells"]}
    assert {(a, tc) for a, cells in gate["rates"].items() for tc in cells} == piloted, "every pilot arm, per task cell"
    assert all(r["samples"] > 0 and r["token_rate"] == r["turn_rate"] == r["rate"] == 0 for cells in gate["rates"].values() for r in cells.values())
    assert all(r["cap"] for cells in gate["rates"].values() for r in cells.values()), "at the test's caps, F2 and F3 (piloted uncapped) included"
    # Projected: M1s (M1's), S8k3 (S1 triples) in all its cells, F7-100 (nearest level), the Sol cells (Luna's), S5 on F1-32 (the micro-pilot's).
    proj = gate["projected"]
    assert proj["M1s"]["F1-32"]["source"].startswith("M1's") and set(proj["S8k3"]) == {"F1-2", "F1-32", "F2-2", "F2-10", "F3-5", "F3-60", "F7-10", "F7-1000"}
    # Decided per unit on the lower bound (R-B5): each measured arm pooled over its cells, M1s, S8k3, each arm's Sol cells.
    units = gate["units"]
    assert {"M7", "S1", "M1s", "S8k3", "S1 (main_sol)", "M1 (main_sol)", "M2 (main_sol)"} <= set(units) and units["S1"]["measured"] and not units["S8k3"]["measured"]
    assert units["S1"]["token"]["n"] == sum(r["samples"] for r in gate["rates"]["S1"].values()) and all(u["token"]["lower_bound"] == 0 for u in units.values())
    assert units["S5"]["cells"] == sorted([*gate["rates"]["S5"], "F1-32"]), "S5's micro-pilot F1 samples join its unit; F7-100 (nearest level) does not"
    assert {"F7-100", "F7-100 (main_sol)"} <= set(proj["S1"]) and "F1-32 (main_sol)" in proj["M1"] and "F7-100 (main_sol)" in proj["M2"]
    assert proj["S5"]["F1-32"]["source"].startswith("the micro-pilot's") and gate["unprojectable"] == []
    assert all(r["projected"] is True for cells in proj.values() for r in cells.values()) and gate["turn_over"] == {} and gate["over_projected"] == {}
    pilot = _manifest(run, "pilot")
    assert pilot["cap_gate"]["multiple"] == 8 and pilot["params"]["cap_gate"] == {"base": 8, "doublings": 2, "threshold": 0.1, "rule": "token_rate", "turn_caps": "refuse"}
    assert gate["other_over"] == {} and gate["warnings"] == []
    assert json.loads((run.phase_dir("pilot") / "pilot.json").read_text())["cap_gate"]["rates"] == gate["rates"]
    freeze = read_freeze(run)
    assert freeze["cap_multiple"] == 8 and freeze["cap_gate"]["rates"] == gate["rates"] and freeze["cap_gate"]["projected"] == proj and freeze["cap_gate"]["passed"] is True
    assert freeze["cap_gate"]["units"] == units and freeze["cap_gate"]["warnings"] == []
    assert "config/cap_gate.json" in freeze["files"] and run_study.read_cap_gate(run) == (gate, [])
    # The pre-registration's pilot items, the cap multiple and the power re-simulation (no gate run offline: priors).
    items = json.loads((run.phase_dir("pilot") / "pilot.json").read_text())["prereg_items"]
    assert {"token caps", "cap multiple", "S7 targets per cell", "KG arm", "pilot σ and power"} == set(items)
    assert items["cap multiple"].startswith("8 (D-039, D-047; no unit shows token-cap-hit evidence over 10% on the pilot, measured or projected")
    assert items["pilot σ and power"].startswith("σ_w 0.500, σ_g 0.300 from the gate's priors (no gate run)") and "Full table:" in items["pilot σ and power"]
    power = json.loads((run.phase_dir("pilot") / "power.json").read_text())
    assert power["key"]["reps"] == run_study.OFFLINE_POWER["reps"] and set(power["power"]["families"]) and pilot["outputs"]["power"]


def test_the_power_resimulation_takes_the_gate_pilots_sigma_and_is_reused(clean_env, monkeypatch):
    """PREREGISTRATION_MAIN §2.6: σ_w and σ_g from the gate run's pilot (`pilot/power.json`, `variance_components`),
    the priors otherwise, saying which; the re-simulation is reused while its inputs stand."""
    from ape.analysis import main_power

    runs = clean_env / "runs"
    gdir = _fake_gate_run(runs, "g-go", "GO")
    run = StudyRun("main", "pw", runs_root=runs, gate_run_id="g-go")
    vc, source, path = run_study.gate_pilot_sigma(run)
    assert vc is None and path is None and "missing" in source and run_study._gate_power_input(run) == {}
    run_gate._write_json(gdir / "pilot" / "power.json", {"variance_components": {"estimable": False, "reason": "too few worlds"}})
    assert "could not estimate σ_w and σ_g: too few worlds" in run_study.gate_pilot_sigma(run)[1]
    good = {"estimable": True, "sigma_w": 0.42, "sigma_g": 0.21, "worlds_per_cell": {"F7-10": 8, "F3-5": 8}}
    run_gate._write_json(gdir / "pilot" / "power.json", {"variance_components": good})
    vc, source, path = run_study.gate_pilot_sigma(run)
    assert vc == good and source.startswith("gate run g-go's pilot (") and source.endswith("; 8 worlds per cell)") and path == gdir / "pilot" / "power.json"
    assert run_study._gate_power_input(run) == {"gate/pilot/power.json": path}, "a pilot input: a changed σ re-runs the pilot"
    calls = []

    def fake_table(reps, seed, sigmas, settings, sizes=None):
        calls.append((reps, sigmas.w, sigmas.g, settings.flip_reps))
        return {"families": {"K1": {"scenarios": [
            {"scenario": "null", "nulls": ["K1.F7-1000"], "members": {"K1.F7-1000": 0.02}, "false_claims": 0.021},
            {"scenario": "+0.10", "nulls": [], "members": {"K1.F7-1000": 0.71}, "false_claims": None},
            {"scenario": "+0.15", "nulls": [], "members": {"K1.F7-1000": 0.93}, "false_claims": None},
        ]}}}  # fmt: skip

    monkeypatch.setattr(main_power, "power_table", fake_table)
    record: dict = {"outputs": {}, "warnings": []}
    out = run_study._pilot_power(run, record)
    assert calls == [(run_study.POWER_REPS, 0.42, 0.21, main_power.Settings().flip_reps)] and out["sigmas"]["w"] == 0.42
    assert out["summary"].startswith("σ_w 0.420, σ_g 0.210 from gate run g-go's pilot") and "K1.F7-1000 0.71-0.93" in out["summary"] and "K1 0.021" in out["summary"]
    run_study._pilot_power(run, record)
    assert len(calls) == 1, "reused while σ, replicates, seed and sizes stand"
    run_gate._write_json(gdir / "pilot" / "power.json", {"variance_components": good | {"sigma_w": 0.6}})
    run_study._pilot_power(run, record)
    assert len(calls) == 2 and calls[-1][1] == 0.6


def _fake_samples(name: str, token: int = 0, turn: int = 0, other: int = 0, tokens: int = 100, n: int = 20):
    """The rows `pilot_samples` gives for one fake log `x<multiple>-<arm>.eval` on F1-2 (capped at multiple x 1,000)."""
    m, arm = name.removesuffix(".eval").split("-")
    rows = []
    for i in range(n):
        tok, tr, ot = i < token, token <= i < token + turn, token + turn <= i < token + turn + other
        rows.append({"arm": arm, "cell": "F1-2", "log_file": name, "error": False, "cap_hit": tok or tr or ot, "token_hit": tok, "turn_hit": tr, "other_hit": ot, "total_tokens": tokens, "token_limit": 1000 * int(m[1:])})
    return rows


def _cap_gate_pilot(clean_env, monkeypatch, *, offline: bool = False, hits, remaining: float = 1e6):
    """A pilot of S1 and M7 on F1-2 (B0 1,000) whose logs are fakes: `hits(arm, multiple)` -> `_fake_samples` keywords
    for its 20 samples. Returns the run, its groups and every re-run's groups."""
    import pandas as pd

    run = StudyRun("main", "cg", offline=offline, runs_root=clean_env / "runs")
    run_study.write_caps(run.token_caps_path, {"F1-2": {"b0": 1000.0, "samples": 20}}, "micro-pilot")
    run_study.write_caps(run.pilot_caps_path, {}, "pilot")
    groups = [
        {"cell": "main.pilot.a", "plan_phase": "pilot", "name": "selected", "kind": "agent", "primary": False, "arms": [{"declared": "S1", "run": "S1"}, {"declared": "M7", "run": "M7"}],
         "skipped": [], "env": {}, "cells": ["F1-2"], "deliveries": ["push"], "caps": {"F1-2": 8000}, "log_files": ["x8-S1.eval", "x8-M7.eval"], "dir": "main.pilot.a/selected-0"},
    ]  # fmt: skip
    reruns: list[list[dict]] = []

    def fake_run(run_, record, phase, gs):
        reruns.append(gs)
        for g in gs:
            g["log_files"] = [f"x{g['cap_multiple']}-{a['run']}.eval" for a in g["arms"]]
        return [f for g in gs for f in g["log_files"]]

    def fake_samples(logs):
        rows = [r for f in logs for r in _fake_samples(f, **hits(f.removesuffix(".eval").split("-")[1], int(f.split("-")[0][1:])))]
        return pd.DataFrame(rows, columns=list(run_study.SAMPLE_COLUMNS))

    monkeypatch.setattr(run_study, "run_phase_groups", fake_run)
    monkeypatch.setattr(run_study, "pilot_samples", fake_samples)
    monkeypatch.setattr(run_study, "group_projected", lambda r, g: 1.0)
    monkeypatch.setattr(run_study, "guard_remaining", lambda r: remaining)
    return run, groups, reruns


def _gate(run, groups):
    record: dict = {"outputs": {}, "warnings": [], "log_dirs": []}
    gate, latest = run_study._cap_gate(run, record, groups)
    return gate, latest, record


def test_the_cap_hit_gate_doubles_the_multiple_for_every_arm_and_reruns_only_the_arms_over(clean_env, monkeypatch):
    """D-039/D-047: M7's token-cap hits are 8 of its 20 F1-2 samples at 8 x B0 (lower bound 19%), 6 at 16 (12%) and 2
    at 32: the multiple doubles twice, only M7 re-runs, and the bound decides (6 of 20 is over, 5 of 20 is not)."""
    m7 = {8: 8, 16: 6, 32: 2}
    run, groups, reruns = _cap_gate_pilot(clean_env, monkeypatch, hits=lambda arm, m: {"token": m7[m]} if arm == "M7" else {})
    gate, latest, record = _gate(run, groups)
    assert gate["rule"] == "token_lower_bound" and gate["multiple"] == 32 and gate["passed"] and gate["over"] == {} and gate["turn_over"] == {}
    assert [(r["multiple"], r["rerun"], sorted(r["over"])) for r in gate["rounds"]] == [(8, [], ["M7"]), (16, ["M7"], ["M7"]), (32, ["M7"], [])]
    assert gate["rounds"][1]["units"]["M7"]["cells"] == ["F1-2"] and gate["rounds"][1]["units"]["M7"]["token"]["k"] == 6, "its F1-32, projected from F1-2, adds no samples"
    # Each re-run: M7 alone, the pilot's cells, under the test's caps at the new multiple, in a log dir of its own.
    assert [[(a["run"], g["caps"], g["cap_multiple"]) for g in gs for a in g["arms"]] for gs in reruns] == [[("M7", {"F1-2": 16000}, 16)], [("M7", {"F1-2": 32000}, 32)]]
    assert len({g["dir"] for gs in reruns for g in gs} | {groups[0]["dir"]}) == 3 and "-x16-" in reruns[0][0]["dir"]
    assert sorted(latest) == ["x32-M7.eval", "x8-S1.eval"], "the calibration reads every arm's latest logs"
    assert gate["rates"]["M7"]["F1-2"]["token_rate"] == 0.1 and gate["rates"]["S1"]["F1-2"]["token_rate"] == 0.0 and gate["rates"]["M7"]["F1-2"]["cap"] == 32000
    assert not any(w.startswith("M7 F1-2: token-hit") for w in gate["warnings"]), "10% is not over"
    # One cap for every arm: the test applies 32 x B0; the phases before it ran at 8.
    assert run_study.cap_multiple(run) == 32 and run_study.cap_multiple(run, "pilot") == run_study.cap_multiple(run, "tune") == 8
    assert run_study.token_caps(run)["F1-2"]["cap"] == 32000 and run_study.token_caps(run, "pilot")["F1-2"]["cap"] == 8000
    assert run_study.read_cap_gate(run)[1] == [] and any("raised from 8" in w for w in record["warnings"])
    # The pilot's projection includes both re-runs of every arm (pilot-sized; live sizes, offline too).
    proj = StudyRun("main", "proj", offline=True, runs_root=clean_env / "runs")
    once = run_study.project(proj, [run_study.group_cell(proj, g) for g in run_study.run_groups(proj, "pilot", offline=False) if g["arms"]])
    assert run_study._pilot_projected(proj) == pytest.approx(3 * once) and once > 0


def test_session_health_refuses_the_freeze_on_evidence_of_errors_limits_or_unexpected_overflows():
    """R-B5 (d): Study G's freeze applies the cap gate's rule to the micro-pilot's sessions: a lower bound over 10%
    refuses; an arm whose overflowing W is its measured outcome (CM0, S1, M1, M2, CM-prune, CM-native) never refuses on
    it, though its errors and limits still do."""
    assert run_study.OVERFLOW_EXPECTED == ("CM0", "S1", "M1", "M2", "CM-prune", "CM-native")
    b = run_study._rate_bound
    ok = {"sessions": 20, "error": b(0, 20), "limit": b(0, 20), "overflow": b(0, 20)}
    health = {
        "CM0": ok | {"overflow": b(20, 20), "overflow_expected": True},
        "CM-sum": ok | {"overflow": b(8, 20), "overflow_expected": False},
        "M1": ok | {"error": b(8, 20), "overflow_expected": True},
        "O-state": ok | {"limit": b(3, 20), "overflow_expected": False},  # 15%: no evidence over 10%
        # CM-native: the provider's compacted block not keeping the view under W is a finding about the arm, not a fault.
        "CM-native": ok | {"overflow": b(12, 20), "limit": b(8, 20), "overflow_expected": True},
    }
    problems = run_study.session_health_problems(health)
    assert problems == [
        "CM-sum: overflows of the window W in 8 of 20 micro-pilot sessions (lower bound 19.1% > 10%)",
        "M1: harness errors in 8 of 20 micro-pilot sessions (lower bound 19.1% > 10%)",
        "CM-native: Inspect limits (tokens, working time, cost guard) in 8 of 20 micro-pilot sessions (lower bound 19.1% > 10%)",
    ]


def test_a_point_estimate_over_10_percent_without_evidence_is_a_warning_not_an_escalation(clean_env, monkeypatch):
    """S-5: 3 of 20 (15%) is over 10% as a point estimate, but its lower bound (3.2%) is not: the reviewer's point rule
    escalated by chance at true rates of 2-5%. Recorded as a warning, in the freeze record too."""
    run, groups, reruns = _cap_gate_pilot(clean_env, monkeypatch, hits=lambda arm, m: {"token": 3, "turn": 3, "other": 3} if arm == "M7" else {})
    gate, _, record = _gate(run, groups)
    assert gate["multiple"] == 8 and reruns == [] and gate["passed"] and gate["over"] == gate["turn_over"] == gate["other_over"] == {}
    assert gate["units"]["M7"]["token"]["lower_bound"] < 0.1 < gate["units"]["M7"]["token"]["rate"]
    assert any(w.startswith("M7 F1-2: token-hit rate 15.0% over 10% (point estimate") for w in gate["warnings"]) and run_study.read_cap_gate(run)[1] == []
    assert any("cap gate (warning only)" in w for w in record["warnings"])


def test_turn_and_other_limit_evidence_never_raise_the_multiple_and_the_freeze_refuses_them(clean_env, monkeypatch):
    """D-045, R-B5 (b): a bigger token cap cannot change a turn-cap hit or a working-time / cost-guard stop: reported per
    unit, never re-run, and the freeze refuses while either's lower bound is over 10% (offline and live alike)."""
    for offline in (False, True):
        run, groups, reruns = _cap_gate_pilot(clean_env / str(offline), monkeypatch, offline=offline, hits=lambda arm, m: {"turn": 7, "other": 7} if arm == "M7" else {})
        gate, _, record = _gate(run, groups)
        assert gate["multiple"] == 8 and reruns == [] and gate["over"] == {} and not gate["passed"]
        assert set(gate["turn_over"]) == set(gate["other_over"]) == {"M7"} and gate["rates"]["M7"]["F1-2"]["turn_rate"] == 0.35
        problems = run_study.read_cap_gate(run)[1]
        assert problems[0].startswith("D-045: {'M7': ") and problems[0].endswith("the turn caps need a decision, not a bigger token cap")
        assert problems[1].startswith("R-B5: {'M7': ") and "fix the harness or the guards" in problems[1]
        assert any("turn caps are decided" in w for w in record["warnings"]) and any("other-limit evidence" in w for w in record["warnings"])


def test_the_cap_hit_gate_stops_at_32_and_the_freeze_refuses_an_arm_still_over(clean_env, monkeypatch):
    run, groups, reruns = _cap_gate_pilot(clean_env, monkeypatch, hits=lambda arm, m: {"token": 8} if arm == "M7" else {})
    gate, _, record = _gate(run, groups)
    assert gate["multiple"] == 32 and not gate["passed"] and len(reruns) == 2
    assert gate["over"] == {"M7": ["F1-2"]} and gate["over_projected"] == {}
    gate_, problems = run_study.read_cap_gate(run)
    assert gate_ == gate and problems == ["D-039: {'M7': ['F1-2']} still show token-cap-hit evidence over 10% at 32 x B0 (the largest multiple): the caps cannot be frozen"]
    assert any("the freeze refuses" in w for w in record["warnings"])
    run.cap_gate_path.unlink()
    assert "has not run" in run_study.read_cap_gate(run)[1][0]
    assert run_study.read_cap_gate(StudyRun("study_g", "cg", runs_root=clean_env / "runs")) == (None, []), "Study G has no caps"
    # A re-run the budget cannot afford stops before it runs.
    run2, groups2, reruns2 = _cap_gate_pilot(clean_env / "b", monkeypatch, hits=lambda arm, m: {"token": 8} if arm == "M7" else {}, remaining=0.5)
    with pytest.raises(BudgetError, match="pilot cap re-run at 16 x B0"):
        _gate(run2, groups2)
    assert reruns2 == []


def test_a_projected_unit_raises_the_multiple_without_a_rerun(clean_env, monkeypatch):
    """S8k3 is not in the pilot. Its projection is the share of resampled triples of S1's samples whose total exceeds
    the cap (R-B5 (c)): S1 at 3,000 tokens, three of them 9,000 > 8,000, so the multiple doubles; at 16 they fit."""
    run, groups, reruns = _cap_gate_pilot(clean_env, monkeypatch, hits=lambda arm, m: {"tokens": 3000} if arm == "S1" else {})
    gate, _, _ = _gate(run, groups)
    assert reruns == [] and gate["multiple"] == 16 and gate["passed"]
    assert [(r["multiple"], r["rerun"], sorted(r["over"])) for r in gate["rounds"]] == [(8, [], ["S8k3"]), (16, [], [])]
    assert gate["projected"]["S8k3"]["F1-2"]["token_rate"] == 0.0 and gate["projected"]["S8k3"]["F1-2"]["source"] == "resampled triples of S1's pilot samples against the cap"
    assert gate["rounds"][0]["units"]["S8k3"]["token"]["k"] == 20 and gate["rounds"][1]["raised_by"] == {"S8k3": ["F1-2"]}


def test_s8k3_is_projected_from_resampled_triples_of_s1_samples():
    """R-B5 (c): P(X1 + X2 + X3 > cap) over the S1 samples, not the share over cap / 3 (which overstates it)."""
    import pandas as pd

    def s1(tokens, hit=()):
        return pd.DataFrame([{"error": False, "token_hit": i in hit, "total_tokens": t} for i, t in enumerate(tokens)])

    n, p = run_study.s8k3_triples(s1([1000, 1000, 5000]), 8000)
    # triples (ordered, with replacement): a sum over 8,000 needs at least two 5,000s: 3 positions x 2 choices for the third (7) / 27
    assert n == 3 and p == pytest.approx(7 / 27)
    assert run_study.s8k3_triples(s1([100, 100], hit=(1,)), 10**9)[1] == pytest.approx(1 - 1 / 8), "a sample its own cap stopped is unbounded"
    assert run_study.s8k3_triples(s1([100]), None) == (1, 0.0) and run_study.s8k3_triples(pd.DataFrame(columns=["error", "token_hit", "total_tokens"]), 10) == (0, 0.0)


def test_cap_hit_rates_split_token_turn_and_other_hits_and_project_uncapped_samples():
    import pandas as pd

    rows = [
        {"arm": "M7", "cell": "F1-2", "error": False, "cap_hit": True, "token_hit": True, "turn_hit": False, "other_hit": False, "total_tokens": 8000, "token_limit": 8000},
        {"arm": "M7", "cell": "F1-2", "error": False, "cap_hit": True, "token_hit": False, "turn_hit": True, "other_hit": False, "total_tokens": 500, "token_limit": 8000},
        {"arm": "M7", "cell": "F1-2", "error": False, "cap_hit": True, "token_hit": False, "turn_hit": False, "other_hit": True, "total_tokens": 500, "token_limit": 8000},
        {"arm": "M7", "cell": "F1-2", "error": False, "cap_hit": False, "token_hit": False, "turn_hit": False, "other_hit": False, "total_tokens": 500, "token_limit": 8000},
        # F3-5 ran uncapped (the pilot runs a family the micro-pilot did not measure): over the test's cap is a token hit.
        {"arm": "S1", "cell": "F3-5", "error": False, "cap_hit": False, "token_hit": False, "turn_hit": False, "other_hit": False, "total_tokens": 9000, "token_limit": None},
        {"arm": "S1", "cell": "F3-5", "error": True, "cap_hit": False, "token_hit": False, "turn_hit": False, "other_hit": False, "total_tokens": 9000, "token_limit": None},
        {"arm": "S1", "cell": "F3-5", "error": False, "cap_hit": False, "token_hit": False, "turn_hit": False, "other_hit": False, "total_tokens": 10, "token_limit": None},
    ]
    df = pd.DataFrame([r | {"log_file": "x.eval"} for r in rows], columns=list(run_study.SAMPLE_COLUMNS))
    rates = run_study.cap_hit_rates(df, {"F1-2": 8000, "F3-5": 5000})
    assert rates["M7"]["F1-2"] == {"samples": 4, "cap": 8000, "token_hits": 1, "token_hits_projected": 0, "turn_hits": 1, "other_hits": 1, "token_rate": 0.25, "turn_rate": 0.25, "other_rate": 0.25, "rate": 0.75}
    assert rates["S1"]["F3-5"]["token_hits_projected"] == 1 and rates["S1"]["F3-5"]["token_rate"] == round(1 / 3, 4), "an errored sample is never a cap hit"
    assert run_study.arms_over(rates) == {"M7": ["F1-2"], "S1": ["F3-5"]} and run_study.arms_over({"S1": {"F1-2": {"token_rate": 0.1}}}) == {}, "over is > 10%"
    assert run_study.arms_over(rates, "turn_rate") == {"M7": ["F1-2"]}
    assert run_study.pilot_samples([]).empty
    # The decision is the pooled unit's lower bound (R-B5): 1 of 4 is no evidence of a rate over 10%.
    units = run_study.cap_units(rates, {})
    assert units["M7"]["token"]["k"] == 1 and units["M7"]["token"]["n"] == 4 and units["M7"]["token"]["lower_bound"] < 0.1 and run_study.units_over(units) == {}


def test_projections_for_m1s_unpiloted_levels_and_sol_cells(clean_env):
    import pandas as pd

    run = StudyRun("main", "pj", offline=True, runs_root=clean_env / "runs")
    r = lambda rate, cap=1000: {"samples": 20, "cap": cap, "token_hits": round(rate * 20), "token_rate": rate, "turn_rate": 0.0}  # noqa: E731
    rates = {"M1": {"F1-32": r(0.2)}, "S1": {"F7-10": r(0.0, 100), "F7-1000": r(0.15, 900)}}
    proj, missing = run_study.projected_rates(run, pd.DataFrame(columns=list(run_study.SAMPLE_COLUMNS)), rates, {})
    assert proj["M1s"]["F1-32"]["token_rate"] == 0.2 and proj["M1s"]["F1-32"]["source"].startswith("M1's (measured)")
    assert proj["S1"]["F7-100"]["token_rate"] == 0.15 and "F7-1000" in proj["S1"]["F7-100"]["source"], "equidistant levels: the higher rate"
    assert proj["M1"]["F1-32 (main_sol)"]["token_rate"] == 0.2 and proj["S1"]["F7-100 (main_sol)"]["source"].startswith("main_luna's S1 F7-100")
    assert "S8k3 F1-2" in missing and "S5 F1-32" in missing and all(x["projected"] for c in proj.values() for x in c.values())
    # Units: a Sol unit pools its cells' Luna counts; a nearest-level projection in the study's profile adds no samples.
    assert proj["S1"]["F7-100"]["pool"] is False and proj["S1"]["F7-100 (main_sol)"]["pool"] is True and proj["M1s"]["F1-32"]["unit"] == "M1s"
    units = run_study.cap_units(rates, proj)
    assert units["S1"]["token"]["n"] == 40 and units["S1 (main_sol)"]["cells"] == ["F7-100 (main_sol)"] and "S1 F1-32 (main_sol)" in missing


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
    assert dev["F7-1000"]["shared"] is True and dev["F7-1000"]["n_tasks"] == run_study.plan(live).cell("gate.tune").spec["tasks_per_world"] and dev["F7-1000"]["kinds"] == [], "only M1 tunes on F7-1000 (D-041, D-047): no KG or chunks to build"
    assert "shared" not in dev["F1-32"] and dev["F1-32"]["kinds"] == [] and dev["F1-32"]["seed_base"] == 1000
    assert run_study.kg_worlds(live, "dev") == {}, "shared dev worlds are the gate's: not projected again"


def test_a_live_build_dev_refuses_another_builder_than_the_gate_runs(clean_env, monkeypatch):
    _fake_gate_run(clean_env / "gates", "g-go", "GO")
    # Since D-047 no main tuning cell reads an artifact on the shared dev worlds; the guard still stands for one that does.
    real = run_study.world_specs
    monkeypatch.setattr(run_study, "world_specs", lambda r, split: [s | ({"kinds": ["apg"]} if s.get("shared") else {}) for s in real(r, split)])
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
    assert "placeholder" not in grid and "owners.M1 is not set" in run_study.tuning_signoff_problems(grid), "B3's grid, not signed off yet"
    assert any("placeholder" in p for p in run_study.tuning_signoff_problems(grid | {"placeholder": True}))
    signed = grid | {"owners": dict.fromkeys(grid["owners"], "Ada") | {"S1": "Sam"}, "signed_off": dict.fromkeys(grid["signed_off"], True)}
    assert run_study.tuning_signoff_problems(signed) == []
    assert run_study.tuning_signoff_problems(signed | {"owners": signed["owners"] | {"M1": "TODO"}}) == ["owners.M1 is not set"]
    # Every owners key needs its owner and sign-off, S1 (the skeptic's sign-off on the baseline) included, though no
    # system tunes it; and the skeptic never owns a multi-agent arm's prompts (brief §7.3).
    assert run_study.tuning_signoff_problems(signed | {"signed_off": signed["signed_off"] | {"S1": False}}) == ["signed_off.S1 is not true"]
    skeptic_owns = run_study.tuning_signoff_problems(signed | {"owners": signed["owners"] | {"M7": " sam "}})
    assert skeptic_owns == ["owners.M7 is the skeptic (owners.S1: Sam): a multi-agent arm's prompts need an independent author (brief §7.3)"]
    g_grid = yaml.safe_load((ROOT / "config" / "tuning_grid_study_g.yaml").read_text())
    assert "placeholder" not in g_grid and "owners.S-CM* is not set" in run_study.tuning_signoff_problems(g_grid), "B11's grid, not signed off yet"
    run = StudyRun("study_g", "tune", runs_root=clean_env / "runs")
    assert "is not signed off for a live tune" in run_study._refuse_tune(run)
    assert run_study._refuse_tune(StudyRun("study_g", "tune", offline=True, runs_root=clean_env / "runs")) is None


def test_a_live_tune_refuses_a_grid_that_breaks_the_tuning_rules(clean_env, monkeypatch):
    """`tuning.study_grid_problems` before any dev run (B3): a live tune refuses a signed-off grid that breaks a rule;
    offline the tune records the problems and runs."""
    grid = yaml.safe_load((ROOT / "config" / "tuning_grid_main.yaml").read_text())
    grid |= {"owners": dict.fromkeys(grid["owners"], "Ada") | {"S1": "Sam"}, "signed_off": dict.fromkeys(grid["signed_off"], True)}
    # The plan's tuning cells price M1 and M7 (D-047): the grid's other systems (S9 until F-arms moves it to `inherited`)
    # are left out here, so the check meets a grid that agrees with the plan.
    grid["systems"] = {k: v for k, v in grid["systems"].items() if k in ("M1", "M7")}
    grid["owners"] = {k: v for k, v in grid["owners"].items() if k in ("M1", "M7", "S1")}
    grid["signed_off"] = {k: v for k, v in grid["signed_off"].items() if k in ("M1", "M7", "S1")}
    monkeypatch.setattr(run_study, "_load_grid", lambda r: grid)
    monkeypatch.setattr(run_study, "_refuse_frozen", lambda phase: lambda r: None)
    monkeypatch.setattr(run_study, "_refuse_unbuilt", lambda phase: lambda r: None)
    run = StudyRun("main", "tune", runs_root=clean_env / "runs")
    assert run_study.grid_problems(run) == [] and run_study._refuse_tune(run) is None, "B3's grid passes once signed off"
    grid["systems"]["M7"]["candidates"] = grid["systems"]["M7"]["candidates"][:3]
    reason = run_study._refuse_tune(run)
    assert reason.startswith("tune:") and "breaks the tuning rules" in reason and "M7: 3 candidates; equal budgets give every system budget_per_system (4)" in reason
    assert run_study._refuse_tune(StudyRun("main", "tune", offline=True, runs_root=clean_env / "runs")) is None


def test_each_tuned_arm_runs_with_exactly_its_own_selections_knobs(clean_env):
    """D-042: Study G's CM knobs share one namespace (APE_CM_PRUNE_KEEP is CM-prune's and S-CM*'s; CM-todo and CM-reset
    both read todo_extract), so each tuned arm runs in an eval set of its own under its own selection's knobs alone,
    and the untuned arms under none; two selections setting one knob name never refuse."""
    run = StudyRun("study_g", "groups", offline=True, runs_root=clean_env / "runs")
    selected = {
        "CM-prune": {"arm": "CM-prune", "env": {"APE_CM_PRUNE_KEEP": "5"}},
        "CM-sum": {"arm": "CM-sum", "env": {"APE_CM_SUM_RATIO": "0.3"}},
        "CM-todo": {"arm": "CM-todo", "env": {"APE_CM_TODO_EXTRACT": "false"}},
        "CM-reset": {"arm": "CM-reset", "env": {"APE_CM_TODO_EXTRACT": "true", "APE_CM_RESET_EVERY": "3"}},
        "S-CM*": {"arm": "S-CM*", "env": {"APE_CM_STACK": "trim+todo"}},
    }
    run.out_config_dir.mkdir(parents=True)
    run.selected_path.write_text(yaml.safe_dump(selected))
    groups = {(g["cell"], g["name"]): g for g in run_study.run_groups(run, "test")}
    cm = {name: g for (cell, name), g in groups.items() if cell == "g.cm.luna-high"}
    assert set(cm) == {"selected", "sel-CM-prune", "sel-CM-sum", "sel-CM-todo", "sel-CM-reset"}
    for name in ("CM-prune", "CM-sum", "CM-todo", "CM-reset"):
        assert cm[f"sel-{name}"]["env"] == selected[name]["env"] and [a["run"] for a in cm[f"sel-{name}"]["arms"]] == [name]
    assert cm["selected"]["env"] == {} and {a["run"] for a in cm["selected"]["arms"]} == {"CM0", "CM-native", "O-state"}, "untuned: no selection's knobs"
    scm = groups[("g.topo.luna", "sel-S-CM_")]
    assert scm["env"] == {"APE_CM_STACK": "trim+todo"} and "APE_CM_PRUNE_KEEP" not in scm["env"], "S-CM* sees none of CM-prune's knobs"
    assert groups[("g.topo.luna", "selected")]["env"] == {} and {a["run"] for a in groups[("g.topo.luna", "selected")]["arms"]} == {"S1", "M1", "M2"}
    # The main study: M1s runs under M1's selection (it reads M1's knobs, D-041), with M1, never another arm's.
    main_sel = {"M1": {"arm": "M1", "env": {"APE_MAS_M1_PROMPT": "concise"}}, "M7": {"arm": "M7", "env": {"APE_MAS_M7_PROMPT": "verify"}}, "S9": {"arm": "S9", "env": {}}}
    assert run_study.env_group("M1s", "M1s", main_sel) == ("sel-M1", {"APE_MAS_M1_PROMPT": "concise"})
    assert run_study.env_group("M1k", "M1k", main_sel) == run_study.env_group("M2", "M2", main_sel) == ("sel-M1", {"APE_MAS_M1_PROMPT": "concise"}), "D-047: the M1 chain runs M1's selection"
    assert run_study.env_group("S9", "S9", {k: v for k, v in main_sel.items() if k != "S9"}) == ("sel-M1", {"APE_MAS_M1_PROMPT": "concise"}), "D-047: S9 too"
    planned = __import__("ape.tuning", fromlist=["planned_tuning"]).planned_tuning((c.id, c.spec) for c in run_study.phase_cells(StudyRun("main", "p", offline=True, runs_root=clean_env / "runs"), "tune"))
    assert planned["M1"] == {"candidates": {"main.tune.a": 4, "main.tune.b": 4}, "cells": ["F1-32", "F2-10", "F3-60", "F7-1000"]} and not {"S9", "M1k", "M2"} & set(planned)
    assert run_study.env_group("M7", "M7", main_sel) == ("sel-M7", {"APE_MAS_M7_PROMPT": "verify"})
    assert run_study.env_group("S9", "S9", main_sel) == ("selected", {}) and run_study.env_group("S1", "S1", main_sel) == ("selected", {})
    assert run_study.env_group("S7", "S7", main_sel) == ("s7", {}), "S7's group carries only its schedule knob"


def test_study_g_grid_checks_allow_the_shared_cm_namespace_and_hold_each_arm_to_its_own_knobs(clean_env, monkeypatch):
    """D-042 / B11: Study G's candidates write out their arm's whole APE_CM_* configuration, so systems share knob names;
    the grid check holds each candidate to the knobs its own arm reads, with values the policy takes, and the plan's
    session tuning cell prices its F8 cell. T_abs is no knob: the runner passes the plan's to every session task."""
    from ape import tuning

    assert tuning.plan_task_cells({"N": 40, "sessions": 3}) == ["F8-40"] and tuning.plan_task_cells({"cells": ["F1-2"]}) == ["F1-2"]
    run = StudyRun("study_g", "gg", offline=True, runs_root=clean_env / "runs")
    planned = tuning.planned_tuning((c.id, c.spec) for c in run_study.phase_cells(run, "tune"))
    assert planned["CM-sum"] == {"candidates": {"g.tune.luna": 2}, "cells": ["F8-40"]} and planned["S-CM*"]["candidates"] == {"g.tune.luna": 3}
    grid = {
        "budget_per_system": 3, "equal_budgets": False, "dev_cells": ["F8-40"],
        "systems": {
            "CM-prune": {"candidates": [{"id": f"p{k}", "arm": "CM-prune", "env": {"APE_CM_PRUNE_KEEP": str(k)}} for k in (3, 6, 12)]},
            "CM-sum": {"candidates": [{"id": f"s-{v}", "arm": "CM-sum", "env": {"APE_CM_SUM_PROMPT": v}} for v in ("structured", "plain")]},
            "CM-todo": {"candidates": [{"id": f"t-{v}", "arm": "CM-todo", "env": {"APE_CM_TODO_EXTRACT": v}} for v in ("true", "false")]},
            "CM-reset": {"candidates": [{"id": f"r{e}", "arm": "CM-reset", "env": {"APE_CM_TODO_EXTRACT": "true", "APE_CM_RESET_EVERY": e}} for e in ("5", "0", "3")]},
            "S-CM*": {"candidates": [{"id": f"x{i}", "arm": "S-CM*", "env": {"APE_CM_STACK": st, "APE_CM_PRUNE_KEEP": "3", "APE_CM_TODO_EXTRACT": "true"}} for i, st in enumerate(("prune+todo+reset", "prune+todo+sum", "prune+todo"))]},
        },
    }  # fmt: skip
    monkeypatch.setattr(run_study, "_load_grid", lambda r: grid)
    assert run_study.grid_problems(run) == [], "APE_CM_PRUNE_KEEP and APE_CM_TODO_EXTRACT are set by several systems, by design"
    grid["systems"]["CM-prune"]["candidates"][0]["env"] |= {"APE_CM_SUM_PROMPT": "plain"}
    grid["systems"]["S-CM*"]["candidates"][2]["env"]["APE_CM_STACK"] = "prune+trim"
    grid["systems"]["CM-todo"]["candidates"][1]["env"] = {"APE_CM_TODO_EXTRACT": "maybe", "APE_CM_THRESHOLD": "30000"}
    assert run_study.grid_problems(run) == [
        "CM-prune: candidate p3: CM-prune reads none of ['APE_CM_SUM_PROMPT'] (its knobs: ['prune_keep'])",
        "CM-todo: candidate t-false: CM-todo reads none of ['APE_CM_THRESHOLD'] (its knobs: ['todo_extract'])",
        "CM-todo: candidate t-false: CM-todo rejects {'APE_CM_TODO_EXTRACT': 'maybe', 'APE_CM_THRESHOLD': '30000'}: not a boolean: 'maybe'",
        "S-CM*: candidate x2: S-CM* rejects {'APE_CM_STACK': 'prune+trim', 'APE_CM_PRUNE_KEEP': '3', 'APE_CM_TODO_EXTRACT': 'true'}: stack 'prune+trim': at most one of ['prune', 'trim'] and one of ['sum', 'reset']",
    ]
    assert run_study.cm_candidate_problems("CM-nope", {})[0].startswith("session arm 'CM-nope' is not built yet")
    # Topology is a primary test plan phase (G-H3 is confirmatory, D-033 / D-042): its groups run with the primary ones.
    assert run_study.STUDIES["study_g"].primary == ("capability_anchor", "context_management", "topology")
    assert all(g["primary"] for g in run_study.run_groups(run, "test") if g["cell"].startswith(("g.cap.", "g.cm.", "g.topo.")))


def test_each_tuning_system_runs_under_its_own_per_sample_cost_limit(clean_env):
    """R-C4: a system's candidates get the cost model's $ per sample of that system's arms in the tuning cells, not
    the plan's flat default: M7 (a council) costs more per sample than M1 (orchestrator and workers)."""
    run = StudyRun("main", "tc", offline=True, runs_root=clean_env / "runs")
    cells = run_study.phase_cells(run, "tune")
    m7, m1 = run_gate.tune_sample_usd(run, cells, {"M7": 4}), run_gate.tune_sample_usd(run, cells, {"M1": 4})
    assert m7 and m1 and m7 != m1 and run_gate.tune_sample_usd(run, cells, {"nope": 1}) is None
    gate = GateRun("tc", offline=True, runs_root=clean_env / "gates")
    assert run_gate.tune_sample_usd(gate, [run_gate.plan(gate).cell("gate.tune")], {"APG-s": 4, "APG-q": 2}) > 0


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

    # 1. The draft pre-registration (its [USER: ...] items) and the missing cap-hit gate block the live freeze;
    #    nothing is frozen.
    with pytest.raises(PhaseError, match=r"(?s)unfilled item\(s\).*\[USER: .*cap-hit gate \(D-039\) has not run"):
        run_phases(run, "freeze")
    assert read_freeze(run) is None and json.loads(run.manifest_path("freeze").read_text())["status"] == "failed"

    # 2. Filled, with an analysis module: frozen on main's first block, recorded with the KG arm and a study marker.
    text = prereg.read_text()
    start = run_gate.prereg_body_start(text)
    prereg.write_text(text[:start] + run_gate.PLACEHOLDER_ITEM.sub("filled", text[start:]))
    monkeypatch.setattr(run_study, "analysis_available", lambda r: True)
    run_gate._write_json(run.cap_gate_path, {"multiple": 16, "base": 8, "max": 32, "threshold": 0.1, "rule": "rate", "passed": True, "over": {}, "rates": {"M7": {"F1-2": {"rate": 0.05}}}, "rounds": []})
    real = (ROOT / "PROVENANCE.md").read_bytes()
    assert run_phases(run, "freeze") == {"freeze": "done"}
    freeze = run_study.require_frozen(run)
    assert freeze["rehearsal"] is False and freeze["test_seeds"]["base"] == 13000 and freeze["kg"]["arm"] == "APG-q" and freeze["code_commit"]
    assert {"PREREGISTRATION_MAIN.md", "config/tuning_grid_main.yaml", "config/token_caps.json", "config/cap_gate.json", "config/s7_targets.json", "config/gate_resolution.json", "uv.lock"} <= set(freeze["files"])
    assert not any(k.startswith("gate/") for k in freeze["files"]), "R-A1: no gate file is frozen, only the run's copy of its resolution"
    assert freeze["test_affordability"]["projected_usd"] <= freeze["test_affordability"]["remaining_usd"] and freeze["test_cost_limits"], "R-B3, R-C6"
    assert freeze["cap_multiple"] == 16 and freeze["token_caps"]["F1-2"] == 1600 and freeze["cap_gate"]["rates"] == {"M7": {"F1-2": {"rate": 0.05}}}, "D-039: the test's multiple"
    assert "<!-- ape:test-seeds study=main run=live-freeze base=13000 count=9 -->" in provenance.read_text() and (ROOT / "PROVENANCE.md").read_bytes() == real
    assert run_study.choose_test_seed_base(StudyRun("main", "next", **kw))[0] == 13100
    assert run_gate.choose_test_seed_base(GateRun("gate-next", runs_root=tmp / "runs", provenance_path=provenance))[0] == 3000, "the gate never counts main's block"
    # 3. Frozen once; a changed frozen input (the gate run's selection) stops build-test and test, naming it.
    with pytest.raises(PhaseError, match="frozen once"):
        run_phases(StudyRun("main", "live-freeze", force=True, **kw), "freeze")
    for phase in ("micro-pilot", "tune", "pilot"):
        with pytest.raises(PhaseError, match=f"{phase}: run 'live-freeze' is frozen"):
            run_phases(StudyRun("main", "live-freeze", force=True, **kw), phase)
    # R-A1: a gate re-analysis (decision.json rewritten) leaves the frozen run intact; an allocation change too (R-B3).
    run_gate._write_json(tmp / "gates" / "g-go" / "report" / "decision.json", {"verdict": {"label": "GO", "reasons": []}, "generated_at": "later"})
    assert run_study.require_frozen(StudyRun("main", "live-freeze", **kw))["frozen_at"] == freeze["frozen_at"]
    gate_selected = tmp / "gates" / "g-go" / "config" / "selected.yaml"
    gate_selected.write_text(yaml.safe_dump({**SELECTED, "S3s": {"arm": "S3s", "env": {"APE_S3S_BUDGET": "4000"}, "candidate": "s3s-4000"}}))
    with pytest.raises(PhaseError, match=r"resolution changed since this run copied it"):
        run_study.kg_resolution(StudyRun("main", "live-freeze", **kw))


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
    with pytest.raises(PreflightError, match="smoke check 'g_cm_sum' has no live result"):
        run_phases(StudyRun("study_g", "g0", **kw), "preflight")  # B12: Study G's own checks are required
    _write_smoke(clean_env / "smoke", ["effort", "orchestrator", *STUDIES["study_g"].smoke_checks])
    with pytest.raises(PreflightError, match="g.cm.luna-high: CM-native needs confirmed native compaction") as e:
        run_phases(StudyRun("study_g", "g1", **kw), "preflight")
    assert "no live smoke record" not in str(e.value), "CM-native's support is checked at preflight, before any sample"
    # The B11 probe records the support; then the preflight passes.
    from ape.agent.cm_arms import NATIVE_RECORD_ENV, record_native_support

    monkeypatch.setenv(NATIVE_RECORD_ENV, str(record_native_support("openai/gpt-6-luna", True, evidence="test", path=clean_env / "native.json")))
    with pytest.raises(PreflightError, match="g.cm.sol-high: CM-native needs confirmed native compaction for openai/gpt-6-sol") as e:
        run_phases(StudyRun("study_g", "g1", **kw), "preflight")
    assert "g.cm.luna-high" not in str(e.value), "D-043: Sol's CM-native cell needs Sol's own support record"
    record_native_support("openai/gpt-6-sol", True, evidence="test", path=clean_env / "native.json")
    assert run_phases(StudyRun("study_g", "g1", **kw), "preflight") == {"preflight": "done"}
    m = json.loads(StudyRun("study_g", "g1", **kw).manifest_path("preflight").read_text())
    assert m["checks"]["smoke"] == "ok" and m["checks"]["probe"]["missing"] == [] and set(m["profiles"]) == {"study_g_luna", "study_g_sol", "study_g_astra"}
    assert m["checks"]["cm_native"] == {"g.pilot.luna": "provider", "g.cm.luna-high": "provider", "g.cm.sol-high": "provider"}
    # A study's own required checks (BUILD_PLAN B12 adds them) are held to the same conditions.
    monkeypatch.setitem(run_study.STUDIES, "study_g", replace(STUDIES["study_g"], smoke_checks=("g-sessions",)))
    with pytest.raises(PreflightError, match="smoke check 'g-sessions' has no live result"):
        run_phases(StudyRun("study_g", "g2", **kw), "preflight")
    _write_smoke(clean_env / "smoke", ["effort", "orchestrator", "g-sessions"])
    assert run_phases(StudyRun("study_g", "g2", **kw), "preflight") == {"preflight": "done"}


def test_without_its_analysis_module_a_live_analyze_refuses_and_an_offline_one_writes_a_stub(clean_env, monkeypatch):
    """Both modules exist now (ape.analyze_main, ape.analyze_g): the missing-module path, as a study without one meets it."""
    monkeypatch.setattr(run_study, "analysis_available", lambda r: False)
    for study in ("main", "study_g"):
        run = StudyRun(study, "an", runs_root=clean_env / "runs")
        with pytest.raises(PhaseError, match=f"analysis not implemented: {run.spec.analysis} does not exist yet"):
            run_study._analyze(run, {"outputs": {}, "warnings": []})
        offline = StudyRun(study, "an-off", offline=True, runs_root=clean_env / "runs")
        record: dict = {"outputs": {}, "warnings": []}
        run_study._analyze(offline, record)
        assert record["analysis"] == {"status": "not_implemented", "module": offline.spec.analysis} and any("analysis not implemented" in w for w in record["warnings"])
        assert json.loads((offline.dir / "report" / "analysis.json").read_text())["status"] == "not_implemented"


def test_log_paths_are_recorded_relative_to_the_run_and_old_absolute_ones_still_resolve(clean_env, tmp_path):
    run = StudyRun("main", "rel", offline=True, runs_root=clean_env / "runs")
    log = run.dir / "test" / "main.A.arms" / "selected-x" / "a.eval"
    log.parent.mkdir(parents=True)
    log.write_text("x")
    assert run_study.run_relative(run, log) == "test/main.A.arms/selected-x/a.eval"
    assert run_study.resolve_log(run.dir, "test/main.A.arms/selected-x/a.eval") == run.dir / "test/main.A.arms/selected-x/a.eval"
    assert run_study.resolve_log(run.dir, str(log)) == log, "an absolute entry (older manifests) is read as it is"
    # A run moved, or restored from a backup elsewhere, still finds its logs.
    moved = tmp_path / "elsewhere" / "rel"
    shutil.copytree(run.dir, moved)
    assert run_study.resolve_log(moved, run_study.run_relative(run, log)).read_text() == "x"
    assert run_study.resolve_log(moved, "src/ape/run_study.py") == ROOT / "src/ape/run_study.py", "a repo-relative entry (`_show`)"


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
    with pytest.raises(PhaseError, match=r"analyze: the report would rest on what the freeze did not fix: run 'an' is not frozen"):
        run_study._analyze(run, {"outputs": {}, "warnings": []})
    assert calls == []
    run = StudyRun("main", "an", runs_root=clean_env / "runs", gate_run_id="g", deviation="a test of the entry point, no freeze")
    record: dict = {"outputs": {}, "warnings": []}
    run_study._analyze(run, record)
    assert calls == ["an"] and record["analysis"] == {"status": "done", "label": "H1 supported"} and set(record["outputs"]) == {"report.md"}
    assert record["deviation"]["reason"] == "a test of the entry point, no freeze" and "Deviation from the freeze" in (run.dir / "report" / "report.md").read_text()
    monkeypatch.setitem(sys.modules, "ape.analyze_main", types.SimpleNamespace(analyze=lambda r: {}))
    (run.dir / "report" / "report.md").unlink()
    with pytest.raises(PhaseError, match="wrote no"):
        run_study._analyze(run, {"outputs": {}, "warnings": []})


def test_a_live_analyze_refuses_changed_frozen_files_unless_a_deviation_is_recorded(clean_env, monkeypatch):
    """R-A2: live, the analysis refuses while a frozen file (the config slices, the analysis code, the run's frozen
    outputs) changed since the freeze; `--deviation` records why and is stamped into decision.json and report.md.
    The gate's analyze applies the same check. Offline the changes are warnings."""
    import sys
    import types

    config = clean_env / "config"
    shutil.copytree(ROOT / "config", config)
    run = StudyRun("main", "dv", offline=True, runs_root=clean_env / "runs", config_dir=config)
    frozen = {"config/run_plan.yaml": run_study.config_input(config / "run_plan.yaml", "main")}
    run_gate._write_json(run.freeze_path, {"frozen_at": "then", "files": {k: run_gate._file_entry(p) for k, p in frozen.items()}})
    plan_path = config / "run_plan.yaml"
    raw = yaml.safe_load(plan_path.read_text())
    next(c for c in raw["studies"]["main"]["phases"]["study_a"] if c["id"] == "main.A.arms")["epochs"] = 4
    plan_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    def analyze(r):
        out = r.dir / "report"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text("# report\nbody\n")
        (out / "decision.json").write_text(json.dumps({"status": "done"}))
        return {"status": "done"}

    monkeypatch.setitem(sys.modules, "ape.analyze_main", types.SimpleNamespace(analyze=analyze))
    monkeypatch.setattr(run_study, "analysis_available", lambda r: True)
    record: dict = {"outputs": {}, "warnings": []}
    run_study._analyze(run, record)  # offline: a warning
    assert record["freeze_problems"] and any("does not rest on the freeze" in w for w in record["warnings"]) and "deviation" not in json.loads((run.dir / "report" / "decision.json").read_text())
    live = StudyRun("main", "dv", runs_root=clean_env / "runs2", config_dir=ROOT / "config")
    live_freeze = {"frozen_at": "then", "files": {"config/run_plan.yaml": run_gate._file_entry(run_study.config_input(config / "run_plan.yaml", "main")) | {"sha256": "0" * 64}}}
    run_gate._write_json(live.freeze_path, live_freeze)
    with pytest.raises(PhaseError, match=r"frozen file changed since the freeze at then: config/run_plan.yaml .*: changed.*--deviation"):
        run_study._analyze(live, {"outputs": {}, "warnings": []})
    live = StudyRun("main", "dv", runs_root=clean_env / "runs2", config_dir=ROOT / "config", deviation="epochs raised after the freeze (logged)")
    record = {"outputs": {}, "warnings": []}
    run_study._analyze(live, record)
    d = json.loads((live.dir / "report" / "decision.json").read_text())
    assert d["deviation"]["reason"] == "epochs raised after the freeze (logged)" and d["deviation"]["changes"] == record["freeze_problems"]
    md = (live.dir / "report" / "report.md").read_text().splitlines()
    assert md[0] == "# report" and md[2].startswith("> **Deviation from the freeze**") and md[-1] == "body"
    # The gate: the same check (a live gate run without a freeze refuses; with a deviation it records it).
    gate = GateRun("gd", runs_root=clean_env / "gates")
    with pytest.raises(PhaseError, match=r"run 'gd' is not frozen"):
        run_gate.analyze_freeze_check(gate, run_gate.read_freeze(gate), run_gate.frozen_changes)
    assert run_gate.analyze_freeze_check(GateRun("gd", runs_root=clean_env / "gates", deviation="why"), None, run_gate.frozen_changes) == ["run 'gd' is not frozen (" + run_gate._show(gate.freeze_path) + " missing)"]


def test_paid_phases_recheck_the_smoke_and_an_override_never_outlives_its_code(clean_env, monkeypatch):
    """R-B1: every live paid phase re-checks that the smoke the run settled still covers the code; an override counts
    only at the code it was recorded at, and never for build-test or test."""
    run = StudyRun("main", "sm", runs_root=clean_env / "runs", gate_run_id="g")
    assert run_gate.smoke_currency_problems(run, "micro-pilot") == ["no live smoke is settled for this run (run.json smoke_check)"]
    run_gate.update_run_info(run, smoke_check={"commits": ["abc123"]})
    monkeypatch.setattr(run_gate, "code_drift", lambda commit: ["src/ape/run_study.py"])
    assert run_gate.smoke_currency_problems(run, "pilot")[0].startswith("src, power, uv.lock changed since the smoke this run accepted (at abc123)")
    monkeypatch.setattr(run_gate, "code_drift", lambda commit: [])
    assert run_gate.smoke_currency_problems(run, "test") == []
    run_gate.update_run_info(run, smoke_check={"override": "no key yet", "at": "then", "code": run_gate.code_identity()})
    assert run_gate.smoke_currency_problems(run, "tune") == []
    assert "covers only the phases before the freeze" in run_gate.smoke_currency_problems(run, "test")[0]
    monkeypatch.setattr(run_gate, "code_identity", lambda: "other")
    assert "an override never outlives the code it was given for" in run_gate.smoke_currency_problems(run, "tune")[0]
    # The phase refuses before it records anything (the refusal names the fix).
    monkeypatch.setattr(run_study, "_refuse_unbuilt", lambda phase: lambda r: None)
    monkeypatch.setattr(run_study, "_refuse_frozen", lambda phase: lambda r: None)
    monkeypatch.setattr(run_study, "kg_resolution", lambda r: {"needed": False, "system": None})
    monkeypatch.setattr(run_gate, "is_complete", lambda r, p: True)
    monkeypatch.setattr(run_study, "stray_environment", lambda r: [])
    with pytest.raises(PhaseError, match=r"micro-pilot: the live smoke this run accepted does not cover the code that would run: .*Re-run preflight"):
        run_study.run_phase(run, "micro-pilot")
    assert not run.manifest_path("micro-pilot").exists()


def test_pre_freeze_phases_carry_the_code_identity_in_params_and_log_dirs(clean_env, monkeypatch):
    """R-B2: a code change makes the micro-pilot, tune and pilot stale, and their groups write new log dirs instead of
    resuming the old code's logs."""
    run = StudyRun("main", "code", offline=True, runs_root=clean_env / "runs")
    monkeypatch.setattr(run_gate, "code_identity", lambda: "aaaa")
    before = {p: run_study.phase_state(run, p)["fingerprint"] for p in ("build-dev", "micro-pilot")}
    dirs = [g["dir"] for g in run_study.run_groups(run, "micro-pilot")]
    assert run_study._tune_params(run)["code_identity"] == "aaaa"
    monkeypatch.setattr(run_gate, "code_identity", lambda: "bbbb")
    assert all(run_study.phase_state(run, p)["fingerprint"] != f for p, f in before.items())
    assert not set(dirs) & {g["dir"] for g in run_study.run_groups(run, "micro-pilot")}
    # The working limit is in the group digest too (R-C2): Inspect's task identity holds it.
    cfg = clean_env / "config"
    shutil.copytree(ROOT / "config", cfg)
    r1 = StudyRun("main", "wl", offline=True, runs_root=clean_env / "runs", config_dir=cfg)
    d1 = {g["dir"] for g in run_study.run_groups(r1, "micro-pilot")}
    raw = yaml.safe_load((cfg / "run_plan.yaml").read_text())
    raw["budget"]["sample_working_limit"]["agent_s"] = 3600
    (cfg / "run_plan.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    assert not d1 & {g["dir"] for g in run_study.run_groups(StudyRun("main", "wl", offline=True, runs_root=clean_env / "runs", config_dir=cfg), "micro-pilot")}


def test_the_runner_index_reads_only_the_latest_runs_identities():
    """R-C2: a log dir's index keeps every identity it ever held; the final logs are the latest run's."""
    from ape.runner import latest_tasks

    index = {"tasks": {"old": {"log": "a.eval", "status": "success"}, "new": {"log": "b.eval", "status": "success"}}, "runs": [{"tasks": ["old"]}, {"tasks": ["new"]}, {"tasks": []}]}
    assert latest_tasks(index) == {"new": {"log": "b.eval", "status": "success"}} and latest_tasks({"tasks": {}, "runs": []}) == {}


def test_a_live_freeze_records_the_cost_limits_checks_affordability_and_freezes_the_native_record(clean_env, monkeypatch):
    """R-B3: the allocation is a guard input, so a live freeze checks the test fits what is left; R-C6: the freeze
    records each test group's per-sample cost limit (the test applies it) and, live with CM-native cells, freezes the
    support record."""
    from ape.agent.cm_arms import NATIVE_RECORD_ENV, record_native_support

    run = StudyRun("main", "aff", offline=True, runs_root=clean_env / "runs")
    run.out_config_dir.mkdir(parents=True)
    run.selected_path.write_text(yaml.safe_dump({}))
    need, left = run_study.test_affordability(run)
    assert need == pytest.approx(run_study._test_projected(run)) and left > 0
    limits = run_study.test_cost_limits(run)
    groups = {g["dir"]: g for g in run_study.run_groups(run, "test") if g["arms"]}
    assert set(limits) == set(groups) and all(v["cost_limit_usd"] >= 0.5 and v["sample_usd"] > 0 for v in limits.values())
    g = next(iter(groups.values()))
    assert limits[g["dir"]]["sample_usd"] == pytest.approx(run_study.group_projected(run, g) / run_study.group_live_samples(run, g), rel=1e-4)
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(record_native_support("openai/gpt-6-luna", True, evidence="test", path=clean_env / "native.json")))
    live_g = StudyRun("study_g", "nat", runs_root=clean_env / "runs")
    monkeypatch.setattr(run_study, "native_routes", lambda r: ({"g.cm.luna-high": "provider"}, []))
    assert run_study.frozen_files(live_g, live_g.prereg_path)["native_compaction.json"] == clean_env / "native.json"
    assert "native_compaction.json" not in run_study.frozen_files(StudyRun("study_g", "nat", offline=True, runs_root=clean_env / "runs"), live_g.prereg_path)


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


# --- The M5 add-on study ------------------------------------------------------------------------------------------


def _m5_config(tmp: Path, enabled: bool = True) -> Path:
    """A copy of config/ with the m5 cells enabled (off by default in run_plan.yaml)."""
    cfg = tmp / "config-m5"
    shutil.copytree(ROOT / "config", cfg)
    raw = yaml.safe_load((cfg / "run_plan.yaml").read_text())
    for cells in raw["studies"]["m5"]["phases"].values():
        for c in cells:
            c["enabled"] = enabled
    (cfg / "run_plan.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return cfg


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(d.rglob("*")) if p.is_file()}


def test_m5_is_off_by_default_and_priced_when_enabled(tmp_path):
    """The M5 add-on study's cells are all disabled in the plan, so the program's totals are unchanged; enabled they
    cost about $110 conservative (M5 = M1 x 1.2, M5-spec = M2 x 1.2, plus fresh M1 and M2 controls)."""
    from ape.budget import estimate, load_plan, projected_cost

    plan = load_plan()
    m5 = [c for c in plan.cells if c.study == "m5"]
    assert {c.id for c in m5} == {"m5.pilot.a", "m5.pilot.b", "m5.test.a", "m5.test.b"} and not any(c.enabled for c in m5)
    assert projected_cost(study="m5") == 0 and plan.budget["allocations"]["m5"] == 150
    est = estimate(load_plan(_m5_config(tmp_path) / "run_plan.yaml"), measured=[])
    assert 100 < est.total(study="m5") < 115 and est.total(study="m5", arm="M5") > est.total(study="m5", arm="M1"), "M5 is 1.2 x M1"
    assert est.total(study="m5") < plan.budget["allocations"]["m5"]
    assert STUDIES["m5"].depends_on == "main" and STUDIES["m5"].phases == ("preflight", "pilot", "freeze", "test", "analyze") and "mas_ledger" in STUDIES["m5"].smoke_checks


def test_the_m5_offline_rehearsal_inherits_the_frozen_main_run_and_never_writes_into_it(offline, tmp_path):
    """`run_study all --study m5 --offline --main-run-id t1` on the offline main rehearsal: the m5 cells run on the main
    run's pilot and test worlds (its frozen test block), at its frozen caps, under M1's selection, with the KG arm it
    froze; m5 builds no worlds, freezes the inherited design, and the main run's directory is left untouched. Until the
    M5 arms are registered they are skipped as unbuilt, and the M1 and M2 controls run."""
    from inspect_ai.log import read_eval_log

    cfg = _m5_config(tmp_path)
    main_run_ = _run(offline, "main")
    before = _snapshot(main_run_.dir)
    argv = ["all", "--study", "m5", "--run-id", "m5a", "--offline", "--runs-dir", str(offline["runs"]), "--config-dir", str(cfg), "--main-run-id", "t1"]
    assert main(argv) == 0
    assert _snapshot(main_run_.dir) == before, "nothing is written under the main run's directory"
    run = StudyRun("m5", "m5a", offline=True, runs_root=offline["runs"], config_dir=cfg)
    mfreeze, freeze = read_freeze(main_run_), read_freeze(run)
    assert {p: _manifest(run, p)["status"] for p in STUDIES["m5"].phases} == dict.fromkeys(STUDIES["m5"].phases, "done")
    inh = freeze["inherited"]
    assert freeze["test_seeds"]["base"] == mfreeze["test_seeds"]["base"] and inh["main_run"] == "t1" and inh["token_caps"] == mfreeze["token_caps"] and inh["cap_multiple"] == mfreeze["cap_multiple"]
    assert inh["selected"] == yaml.safe_load(main_run_.selected_path.read_text()) == yaml.safe_load(run.selected_path.read_text())
    assert freeze["kg"]["arm"] == mfreeze["kg"]["arm"] and freeze["kg"]["source"].startswith("main run t1")
    assert {"main/freeze.json", "config/main_inheritance.json", "config/selected.yaml", "config/pilot_health.json"} <= set(freeze["files"])
    assert not (run.work_dir / "worlds").exists(), "m5 builds no worlds"
    test = _manifest(run, "test")
    groups = [g for c in test["cells"].values() for g in c["groups"]]
    built = run_study.arm_built("M5")
    for g in groups:
        assert g["seed_base"] == mfreeze["test_seeds"]["base"] and g["caps"] == {tc: mfreeze["token_caps"][tc] for tc in g["cells"]}
        assert g["name"] == "sel-M1" and g["env"] == {k: str(v) for k, v in (inh["selected"]["M1"].get("env") or {}).items()}, "M5, M5-spec and the controls run under M1's selection"
        assert {a["run"] for a in g["arms"]} & {"M1", "M2"} and (built or {s["run"] for s in g["skipped"]} <= {"M5", "M5-spec"})
    for f in sorted((run.dir / "test").rglob("*.eval")):
        args = read_eval_log(str(f), header_only=True).eval.task_args
        assert args["seed_base"] == mfreeze["test_seeds"]["base"] and args["split"] == "test" and args["plan_cell"].startswith("m5.test.")
    health = json.loads(run.pilot_health_path.read_text())
    assert health["multiple"] == mfreeze["cap_multiple"] and health["passed"] and run_study.read_pilot_health(run)[1] == []
    # A second invocation needs no --main-run-id (run.json), and one naming another main run is refused.
    assert main(argv[:-2]) == 0
    with pytest.raises(PhaseError, match="inherits from main run 't1' .*not 'other'"):
        run_phases(StudyRun("m5", "m5a", offline=True, runs_root=offline["runs"], config_dir=cfg, main_run_id="other"), "pilot")


def _fake_main(root: Path, rid: str = "mm", *, offline: bool = False, frozen: bool = True, test_started: bool = True, caps: dict | None = None) -> StudyRun:
    m = StudyRun("main", rid, offline=offline, runs_root=root)
    if frozen:
        run_gate._write_json(m.freeze_path, {
            "frozen_at": "then", "offline": offline, "files": {}, "code_commit": "abc", "test_seeds": {"base": 13000, "count": 9},
            "token_caps": caps or {"F1-2": 1000}, "cap_multiple": 8, "kg": {"needed": True, "arm": "APG-q", "env": {}, "system": "apg", "files": {}}, "design_env": {},
        })  # fmt: skip
    if test_started:
        run_gate._write_json(m.manifest_path("test"), {"phase": "test", "status": "running"})
    m.out_config_dir.mkdir(parents=True, exist_ok=True)
    m.selected_path.write_text(yaml.safe_dump({"M1": {"arm": "M1", "env": {"APE_MAS_M1_PROMPT": "concise"}}}))
    return m


def test_a_live_m5_run_refuses_without_a_frozen_main_run_whose_test_has_started(clean_env):
    """Live, the add-on study refuses without --main-run-id, on an unfrozen main run, one of the other mode, or one whose
    test has not started; it copies the inherited design once and refuses a main run whose design changed since."""
    root = clean_env / "runs"
    with pytest.raises(PhaseError, match="runs on a frozen main run: pass --main-run-id"):
        run_study.inheritance(StudyRun("m5", "x", runs_root=root))
    _fake_main(root, "nf", frozen=False)
    with pytest.raises(PhaseError, match="main run 'nf' is not frozen"):
        run_study.inheritance(StudyRun("m5", "x", runs_root=root, main_run_id="nf"))
    _fake_main(root, "off", offline=True)
    with pytest.raises(PhaseError, match="main run 'off' is an offline run; a live m5 run inherits from a run of its own mode"):
        run_study.inheritance(StudyRun("m5", "x", runs_root=root, main_run_id="off"))
    _fake_main(root, "nt", test_started=False)
    with pytest.raises(PhaseError, match="has not started its test"):
        run_study.inheritance(StudyRun("m5", "x", runs_root=root, main_run_id="nt"))
    m = _fake_main(root, "mm")
    run = StudyRun("m5", "x", runs_root=root, main_run_id="mm")
    inh = run_study.inheritance(run)
    assert inh["test_seeds"] == {"base": 13000, "count": 9} and inh["token_caps"] == {"F1-2": 1000} and run_study._caps_params(run, "pilot") == {"F1-2": 1000}
    assert json.loads(run.inheritance_path.read_text())["main_run"] == "mm" and yaml.safe_load(run.selected_path.read_text())["M1"]["env"] == {"APE_MAS_M1_PROMPT": "concise"}
    assert run_study.choose_test_seed_base(run) == (13000, []) and run_study.test_seed_count(run) == 9 and run_study.split_seed_base(run, "pilot") == STUDY_SEEDS["main"]["pilot"]
    assert run_study.env_group("M5", "M5", yaml.safe_load(run.selected_path.read_text())) == ("sel-M1", {"APE_MAS_M1_PROMPT": "concise"})
    # The main run's design changes after this run copied it: refused live.
    _fake_main(root, "mm", caps={"F1-2": 2000})
    with pytest.raises(PhaseError, match=r"main run 'mm''s design changed since this run copied it at .* \(freeze_sha256, token_caps\)"):
        run_study.inheritance(StudyRun("m5", "x", runs_root=root, main_run_id="mm"))
    assert m.dir.is_dir() and not (m.dir / "m5").exists()


def test_m5_reads_only_its_main_runs_worlds_and_never_builds_or_unlocks_the_test_split(clean_env, tmp_path):
    """m5 has no build phase and never sets the test split's lock, which guards building only: it reads the one block
    its main run froze, from that run's worlds (offline: its work/), and refuses when the worlds its cells read are not
    there or the limit rules differ from the main run's frozen ones."""
    from ape.worlds.generate import TestSplitLocked, require_test_split_unlocked

    assert not {"build-dev", "micro-pilot", "build-test"} & set(STUDIES["m5"].phases)
    cfg = _m5_config(tmp_path)
    root = clean_env / "runs"
    m = _fake_main(root, "mo", offline=True)
    run = StudyRun("m5", "w", offline=True, runs_root=root, config_dir=cfg, main_run_id="mo")
    problems = run_study.inheritance_problems(run, m, read_freeze(m))
    assert any("m5.test.a reads 1 test world(s) of F1-2; main run 'mo' built 0" in p for p in problems) and any("m5.pilot.a reads 1 pilot world(s) of F2-10" in p for p in problems)
    with run_study.run_environment(run):
        assert os.environ["APE_WORLDS"] == str(m.work_dir / "worlds") and os.environ["APE_INDICES"] == str(m.work_dir / "indices") and os.environ["APE_CACHE"] == str(run.work_dir / "cache")
        assert TEST_SPLIT_ENV not in os.environ
        with pytest.raises(TestSplitLocked):
            require_test_split_unlocked("test", "m5")
    # The limit rules: this run's plan must say what the main run's frozen plan says.
    frozen_plan = clean_env / "main-plan.yaml"
    raw = yaml.safe_load((cfg / "run_plan.yaml").read_text())
    raw["budget"]["sample_working_limit"]["agent_s"] = 99
    frozen_plan.write_text(yaml.safe_dump(raw))
    fz = read_freeze(m) | {"files": {"config/run_plan.yaml": {"path": str(frozen_plan), "sha256": run_gate._sha256(frozen_plan)}}}
    run_gate._write_json(m.freeze_path, fz)
    inh = run_study.inheritance(StudyRun("m5", "w2", offline=True, runs_root=root, config_dir=cfg, main_run_id="mo"))
    assert any("budget.sample_working_limit differ from main run 'mo''s frozen run_plan.yaml" in n for n in inh["notes"])


def test_m5_pilot_health_refuses_the_freeze_on_evidence_and_never_escalates():
    """The add-on pilot's gate (D-045/D-050 lower bounds, per arm): evidence of token-cap, turn-cap or other-limit hits
    refuses the freeze; the caps are the main run's and frozen, so there is no escalation."""
    run = StudyRun("m5", "h", offline=True)
    b = run_study._rate_bound
    health = {"multiple": 8, "over": {"M5": ["F2-10"]}, "turn_over": {"M5-spec": ["F3-60"]}, "other_over": {}, "units": {"M5": {"token": b(8, 20)}}}
    problems = run_study.pilot_health_problems(run, health)
    assert problems[0].startswith("['M5'] show token-cap-hit evidence over 10% at the main run's frozen caps (8 x B0): the caps are frozen, so the m5 study cannot escalate them")
    assert problems[1].startswith("['M5-spec'] show turn-cap-hit evidence over 10%") and run_study.pilot_health_problems(run, {"over": {}, "turn_over": {}, "other_over": {}}) == []


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
    assert statuses == dict.fromkeys(primary, "done") | dict.fromkeys(["main.C.a", "main.C.b", "main.C.s8", "main.F.luna", "main.F.luna-f7-100", "main.F.sol", "main.F.sol-m2"], "stopped")
