"""Gate orchestrator (FX-6): offline end to end, idempotence, the budget guard, live preflight, per-arm context
budgets (FX-6 §0), the pilot and freeze (part 1), and the test split's build and runs (part 2). Offline: mock
models, fake embeddings, oracle indices, no network."""

import asyncio
import json
import os
import shutil

import pytest
import yaml
from inspect_ai.model import get_model
from inspect_ai.model._model import init_model_roles

from ape import run_gate
from ape.agent import arms
from ape.analysis.gate_stats import GATE_CELLS
from ape.budget import BudgetError
from ape.build import build
from ape.config import ROOT, Config
from ape.lgr.adapter import query_params
from ape.lgr.build import build_index
from ape.llm.mock_agent import mock_kg
from ape.models import PreflightError
from ape.run_gate import PHASES, GateRun, PhaseError, main, read_manifest, read_selected, run_phases
from ape.tuning import env, load_grid
from ape.worlds.spec import World

REAL_SELECTED = ROOT / "config" / "selected.yaml"


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No stray APE_* knobs from the shell or other tests; the orchestrator sets what it needs."""
    for k in [k for k in os.environ if k.startswith("APE_")]:
        monkeypatch.delenv(k)
    return tmp_path


PART_A = ("preflight", "build-dev", "tune", "anchor")  # the phases before the pilot; a freeze would block re-tuning
THROUGH_FREEZE = PHASES[: PHASES.index("freeze") + 1]


def _manifests(run_dir, phases=PART_A) -> dict[str, dict]:
    return {p: json.loads((run_dir / p / "manifest.json").read_text()) for p in phases}


def _statuses(run_dir, phases=PART_A) -> dict[str, str]:
    return {p: m["status"] for p, m in _manifests(run_dir, phases).items()}


def _part_a(argv) -> int:
    return max(main([p, *argv]) for p in PART_A)


def test_offline_part_a_runs_every_phase_then_skips_forces_and_reruns_on_changed_inputs(clean_env):
    tmp = clean_env
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    real_selected = REAL_SELECTED.read_bytes() if REAL_SELECTED.exists() else None
    argv = ["--offline", "--run-id", "t1", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]
    run_dir = tmp / "runs" / "t1"

    # 1. Every part-A phase completes, with its manifest.
    assert _part_a(argv) == 0
    manifests = _manifests(run_dir)
    assert {p: m["status"] for p, m in manifests.items()} == dict.fromkeys(PART_A, "done")
    for phase, m in manifests.items():
        assert m["phase"] == phase and m["run_id"] == "t1" and m["offline"] is True and m["errors"] == []
        assert m["started"] <= m["finished"] and m["git"]["commit"] and isinstance(m["git"]["dirty"], bool)
        assert m["profile"]["roles"]["agent"]["model"] and "mockllm" in m["profile"]["model_swap"], "the real roles, swapped for mocks"
        if phase != "anchor":
            assert m["profile"]["name"] == "gate" and m["profile"]["roles"]["agent"]["reasoning_effort"] == "high"
        assert m["inputs"] and all(v["sha256"] for v in m["inputs"].values()), "config inputs are hashed"
        assert m["params"]["scale"] == {"offline": True, **run_gate.OFFLINE_SCALE}, "the offline scale is recorded"
        assert m["spend"]["spent_usd"] == 0.0 and m["spend"]["budget_usd"] == 5000.0
        assert m["offline_check"]["models"] in ([], ["mockllm/model"]) and m["offline_check"]["ledger_entries"] == 0
        assert m["projected_usd"] is not None
    assert manifests["preflight"]["checks"]["apg_core"]["installed_commit"] == run_gate.APG_PIN
    assert manifests["build-dev"]["projected_usd"] > 0 and manifests["tune"]["projected_usd"] > 0 and manifests["anchor"]["projected_usd"] > 0
    assert manifests["tune"]["upstream"] == {"build-dev": manifests["build-dev"]["fingerprint"]}
    assert manifests["anchor"]["profile"]["name"] == "anchor" and manifests["anchor"]["profile"]["roles"]["judge"]["seed"] == 42

    worlds = json.loads((run_dir / "build-dev" / "worlds.json").read_text())["worlds"]
    assert {w["group"] for w in worlds} == {"gate", "id_only", "messy", "f5"}
    assert {(w["family"], w["level"]) for w in worlds if w["group"] == "gate"} == {("F7", "10"), ("F7", "1000"), ("F3", "5"), ("F3", "60")}
    assert all(w["artifacts"] == {"chunks": "built", "apg": "built", "lightrag": "built"} for w in worlds)
    assert all(w["path"].startswith(str((run_dir / "work" / "worlds").resolve())) for w in worlds), "offline worlds live in the run"

    # The selection, in the form part B reads, in the run and in the run's offline config dir only.
    selected = read_selected(run_dir / "tune" / "selected.yaml")
    assert (run_dir / "work" / "config" / "selected.yaml").read_text() == (run_dir / "tune" / "selected.yaml").read_text()
    assert selected["APG*"]["arm"] in ("APG-q", "APG-s") and "APE_APG_SHORTLIST_K" in selected["APG*"]["env"]
    assert selected["LGR*"]["arm"] in ("LGRo-q", "LGRo-s") and selected["LGR*"]["declared_arm"] in ("LGR-q", "LGR-s")
    assert selected["S3s"]["arm"] == "S3s" and set(selected["S3s"]["env"]) == {"APE_S3S_BUDGET"}
    assert (REAL_SELECTED.read_bytes() if REAL_SELECTED.exists() else None) == real_selected, "an offline run never writes config/"
    tlog = [json.loads(line) for line in (run_dir / "tune" / "tuning_log.jsonl").read_text().splitlines()]
    assert sum("candidate" in r for r in tlog) == 3 * run_gate.OFFLINE_SCALE["tune_candidates_per_system"]
    assert [r["selected"] for r in tlog if "selected" in r] == [selected[k]["candidate"] for k in ("APG*", "LGR*", "S3s")]

    pc1 = json.loads((run_dir / "anchor" / "pc1.json").read_text())
    assert set(pc1["per_type"]) == {"Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"}
    assert pc1["mode"] == "hybrid" and isinstance(pc1["pass"], bool) and pc1["n_per_type"] == 2
    assert all(os.path.isfile(p) for p in pc1["logs"].values())
    assert "APE_WORLDS" not in os.environ and os.environ.get("OPENAI_API_KEY") != run_gate.OFFLINE_KEY, "the run's environment is restored"

    # 2. Re-running skips every complete phase.
    assert _part_a(argv) == 0
    again = _manifests(run_dir)
    assert {p: m["status"] for p, m in again.items()} == dict.fromkeys(PART_A, "skipped")
    assert all(again[p]["finished"] == manifests[p]["finished"] and again[p]["outputs"] == manifests[p]["outputs"] for p in PART_A)

    # 3. --force re-runs one phase (its candidate logs are reused) and leaves the others alone.
    assert main(["tune", "--force", *argv]) == 0
    forced = _manifests(run_dir)
    assert _statuses(run_dir) == {"preflight": "skipped", "build-dev": "skipped", "tune": "done", "anchor": "skipped"}
    assert forced["tune"]["history"][-1]["action"] == "run" and forced["tune"]["history"][-1]["forced"] is True
    assert read_selected(run_dir / "tune" / "selected.yaml") == selected
    assert len(list((run_dir / "tune").glob("tuning_log.*.jsonl"))) == 1, "the previous tuning log is archived"

    # 4. A changed input config re-runs exactly the phases that read it.
    grid = config / "tuning_grid.yaml"
    grid.write_text(grid.read_text() + "\n# edited after the first run\n")
    assert _part_a(argv) == 0
    assert _statuses(run_dir) == {"preflight": "skipped", "build-dev": "skipped", "tune": "done", "anchor": "skipped"}
    assert read_manifest(GateRun("t1", offline=True, runs_root=tmp / "runs", config_dir=config), "tune")["inputs"]["config/tuning_grid.yaml"]["sha256"] != forced["tune"]["inputs"]["config/tuning_grid.yaml"]["sha256"]

    # 5. A complete phase whose recorded output is gone re-runs.
    (run_dir / "work" / "config" / "selected.yaml").unlink()
    assert _part_a(argv) == 0
    assert _statuses(run_dir)["tune"] == "done" and (run_dir / "work" / "config" / "selected.yaml").is_file()
    assert any("missing" in w for w in read_manifest(GateRun("t1", offline=True, runs_root=tmp / "runs", config_dir=config), "tune")["warnings"])


def test_budget_guard_refuses_a_phase_projected_over_what_is_left(clean_env, monkeypatch):
    run = GateRun("guard", offline=True, runs_root=clean_env / "runs")
    assert run_phases(run, "preflight") == {"preflight": "done"}

    def nearly_spent(budget_usd, *_args, **_kwargs):
        return {"budget_usd": budget_usd, "inspect_usd": 4999.5, "ledger_usd": 0.0, "spent_usd": 4999.5, "remaining_usd": budget_usd - 4999.5}

    monkeypatch.setattr(run_gate, "remaining", nearly_spent)
    with pytest.raises(BudgetError, match=r"phase build-dev: projected \$[\d.]+ exceeds the remaining \$0.50"):
        run_phases(run, "build-dev")
    m = read_manifest(run, "build-dev")
    assert m["status"] == "failed" and "BudgetError" in m["errors"][0] and m["projected_usd"] > 0.5
    assert not (run.work_dir / "worlds").exists(), "nothing was built"
    assert main(["build-dev", "--offline", "--run-id", "guard", "--runs-dir", str(clean_env / "runs")]) == 1


def test_phases_need_their_prerequisites_and_a_run_keeps_its_mode(clean_env):
    runs = clean_env / "runs"
    with pytest.raises(PhaseError, match="tune needs preflight first"):
        run_phases(GateRun("order", offline=True, runs_root=runs), "tune")
    assert read_manifest(GateRun("order", offline=True, runs_root=runs), "tune") is None, "a refused start writes no manifest"
    with pytest.raises(PhaseError, match="is an offline run"):
        run_phases(GateRun("order", runs_root=runs), "preflight")
    with pytest.raises(PhaseError, match="offline runs and tests"):
        GateRun("live", config_dir=clean_env)
    with pytest.raises(PhaseError, match="run id"):
        GateRun("../escape")


def test_live_preflight_needs_the_key_and_the_probe_and_picks_the_anchor_profile(clean_env, monkeypatch):
    tmp = clean_env
    monkeypatch.setenv("APE_CACHE", str(tmp / "cache"))  # the spend check reads this ledger, not the real one
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probe = tmp / "openai_probe.json"
    run = GateRun("live", runs_root=tmp / "runs", env_path=tmp / "no.env", probe_path=probe)

    with pytest.raises(PreflightError) as e:
        run_phases(run, "preflight")
    assert "OPENAI_API_KEY" in str(e.value) and "readiness/probe_openai.py" in str(e.value)
    assert read_manifest(run, "preflight")["status"] == "failed"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")  # nothing here makes a request
    with pytest.raises(PreflightError, match="openai_probe.json missing"):
        run_phases(run, "preflight")

    probe.write_text(json.dumps({"models_available": ["gpt-6-luna", "gpt-6-sol"]}))
    with pytest.raises(PreflightError, match="text-embedding-3-small"):
        run_phases(run, "preflight")

    probe.write_text(json.dumps({"models_available": ["gpt-6-luna", "gpt-6-sol", "text-embedding-3-small"]}))
    assert run_phases(run, "preflight") == {"preflight": "done"}
    m = read_manifest(run, "preflight")
    assert m["checks"]["anchor_profile"]["name"] == "anchor_luna", "gpt-4o-mini retired: the Luna anchor fallback"
    assert m["offline"] is False and "model_swap" not in m["profile"]

    probe.write_text(json.dumps({"models_available": ["gpt-6-luna", "gpt-6-sol", "text-embedding-3-small", "gpt-4o-mini"]}))
    assert run_phases(run, "preflight") == {"preflight": "done"}, "a new probe re-runs preflight"
    assert read_manifest(run, "preflight")["checks"]["anchor_profile"]["name"] == "anchor"
    assert run_phases(run, "preflight") == {"preflight": "skipped"}


# --- FX-6 §0: per-arm context budgets ------------------------------------------------------------


CASES = {
    "default": {},
    "s3s": {"APE_S3S_BUDGET": "500"},
    "apg": {"APE_APG_BUDGET": "300"},
    "lgr": {"APE_LGR_BUDGET": "900"},
    "shared": {"APE_CONTEXT_BUDGET": "1000"},
    "shared+s3s": {"APE_CONTEXT_BUDGET": "1000", "APE_S3S_BUDGET": "500"},
}


def test_one_arms_budget_knob_never_moves_another_arms_realized_budget(clean_env, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(clean_env / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    (wid,) = asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    world = World.load(Config().world_path(wid))
    asyncio.run(build_index(world, "oracle", Config()))

    async def realized() -> dict[str, dict]:
        """The budget each built arm actually uses, under each case's environment."""
        init_model_roles({"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)})
        out = {}
        for name, knobs in CASES.items():
            with env(knobs):
                cfg = Config()
                s3s, apg, lgr = [await arms._build(a, world, cfg) for a in ("S3s", "S5o", "LGRo-s")]
                await lgr.rag.finalize_storages()
            out[name] = {"S3s": s3s.budget, "APG": apg.budget, "LGR": lgr.params}
        return out

    got = asyncio.run(realized())
    base = got["default"]
    assert base == {"S3s": 2000, "APG": 2000, "LGR": query_params(2000)}
    assert got["s3s"] == {**base, "S3s": 500}, "S3s's knob moves S3s only"
    assert got["apg"] == {**base, "APG": 300}, "APG's knob moves APG only"
    assert got["lgr"] == {**base, "LGR": query_params(900)} and got["lgr"]["LGR"]["max_total_tokens"] == 3600
    assert got["shared"] == {"S3s": 1000, "APG": 1000, "LGR": query_params(1000)}, "APE_CONTEXT_BUDGET stays the shared default"
    assert got["shared+s3s"] == {**got["shared"], "S3s": 500}
    assert not any(k.endswith("_BUDGET") for k in os.environ if k.startswith("APE_"))


def test_tuning_grid_tunes_each_system_through_its_own_knobs():
    grid = load_grid()
    prefixes = {"APG": "APE_APG_", "LightRAG": "APE_LGR_", "S3s": "APE_S3S_"}
    for system, prefix in prefixes.items():
        for cand in grid["systems"][system]["candidates"]:
            assert all(k.startswith(prefix) for k in cand.get("env", {})), (system, cand["id"])
    assert yaml.safe_load((ROOT / "config" / "tuning_grid.yaml").read_text())["systems"]["S3s"]["candidates"][0]["env"] == {"APE_S3S_BUDGET": "1000"}


def test_the_plan_is_the_one_source_of_dev_worlds_and_anchor_modes(clean_env):
    run = GateRun("plan-check", runs_root=clean_env / "runs")
    p = run_gate.plan(run)
    specs = run_gate.dev_world_specs(run)
    # Every dev world built is priced from a plan cell, with that cell's exception style.
    for group, cell_id in run_gate.DEV_BUILD_CELLS.items():
        cell = p.cell(cell_id).spec
        built = {f"{s['family']}-{s['level']}": s for s in specs if s["group"] == group}
        assert {c: s["count"] for c, s in built.items()} == cell["worlds"], group
        assert {s["exception_style"] for s in built.values()} == {cell.get("exception_style", "descriptive")}, group
    # id_only worlds pair the gate's F7 worlds one for one.
    gate = p.cell("gate.build.dev").spec["worlds"]
    assert p.cell("gate.build.dev-id-only").spec["worlds"] == {c: n for c, n in gate.items() if c.startswith("F7-")}
    # PC1 compares hybrid with naive: the plan prices exactly the modes the anchor runs.
    assert tuple(p.cell("gate.anchor.runs").spec["modes"]) == run_gate.ANCHOR_MODES


# --- FX-6b part 1: pilot and freeze ----------------------------------------------------------------


def _body_labels(text: str) -> set[str]:
    return {p["label"] for p in run_gate.prereg_placeholders(text) if p["kind"] == "PILOT"}


def test_offline_through_freeze_pilots_calibrates_rehearses_the_freeze_and_then_refuses_changes(clean_env, monkeypatch):
    tmp = clean_env
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    argv = ["--offline", "--run-id", "t2", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]
    run = GateRun("t2", offline=True, runs_root=tmp / "runs", config_dir=config)
    run_dir, out = run.dir, run.out_config_dir
    real_provenance = (ROOT / "PROVENANCE.md").read_bytes()

    # 1. A calibration that cannot land fails the pilot clearly, after recording every iteration; nothing frozen.
    real_calibrate = run_gate.calibrate_caps

    def never_lands(target, base, start, measure, **kw):
        return real_calibrate(target, base, start, lambda it, knobs: {a: {"median": 5000.0, "per_cell": {}, "log_dir": None} for a in knobs}, **kw)

    monkeypatch.setattr(run_gate, "calibrate_caps", never_lands)
    with pytest.raises(PhaseError, match=r"budget calibration: \['LGR\*', 'S3s'\] did not land within ±25% of 300 tokens in 3 iterations"):
        run_phases(run, "all")
    assert read_manifest(run, "pilot")["status"] == "failed" and read_manifest(run, "freeze") is None
    cal = json.loads((run_dir / "pilot" / "budget-cal" / "calibration.json").read_text())
    assert all(len(cal["arms"][a]["iterations"]) == run_gate.CAL_MAX_ITERATIONS for a in ("LGR*", "S3s"))
    assert not (out / "budget_calibration.yaml").exists() and not run.freeze_path.exists()
    monkeypatch.setattr(run_gate, "calibrate_caps", real_calibrate)

    # 2. The re-run resumes the pilot and completes every phase through a rehearsal freeze.
    assert max(main([p, *argv]) for p in THROUGH_FREEZE) == 0
    m = _manifests(run_dir, THROUGH_FREEZE)
    assert {p: x["status"] for p, x in m.items()} == {**dict.fromkeys(PART_A, "skipped"), "pilot": "done", "freeze": "done"}
    assert m["pilot"]["upstream"] == {"tune": m["tune"]["fingerprint"]} and m["pilot"]["projected_usd"] > 0
    assert m["pilot"]["offline_check"]["ledger_entries"] == 0 and m["freeze"]["offline_check"]["ledger_entries"] == 0
    worlds = json.loads((run_dir / "pilot" / "worlds.json").read_text())["worlds"]
    assert {(w["family"], w["level"]) for w in worlds} == {("F7", "10"), ("F7", "1000"), ("F3", "5"), ("F3", "60")}
    assert all("-pilot-" in w["world_id"] for w in worlds)

    # S7 targets: APG*'s median realized context per gate cell, read by S7 through APE_S7_TARGETS.
    targets = json.loads((out / "s7_targets.json").read_text())
    assert set(targets) == set(GATE_CELLS) and all(isinstance(t, int) and t > 0 for t in targets.values())
    s7_index = json.loads(next((run_dir / "pilot" / "logs").glob("s7-*/runner_index.json")).read_text())
    assert {e["task_args"]["arm"] for e in s7_index["tasks"].values()} == {"S7"} and len(s7_index["tasks"]) == 4

    # Matched-budget calibration: both capped arms land in the window; APG fills.
    calibration = yaml.safe_load((out / "budget_calibration.yaml").read_text())
    assert calibration["context"] == 300 and calibration["arms"]["APG*"]["env"]["APE_APG_FILL"] == "1"
    for arm in ("LGR*", "S3s"):
        c = calibration["arms"][arm]
        assert c["converged"] and 225 <= c["median"] <= 375 and 1 <= len(c["iterations"]) <= run_gate.CAL_MAX_ITERATIONS
        assert set(c["env"]) <= set(run_gate.CAL_KNOBS[arm])

    # Power re-simulation and the analyst's transcription sheet, keyed by every [PILOT: ...] label in the prereg.
    power = json.loads((run_dir / "pilot" / "power.json").read_text())
    assert set(power["scenarios"]) == {"pilot", "conservative"} and set(power["scenarios"]["pilot"]["power"]) == {"12", "16"}
    assert power["variance_components"]["estimable"] is False, "one offline world per cell: the priors are kept"
    pilot = json.loads((run_dir / "pilot" / "pilot.json").read_text())
    assert _body_labels((ROOT / "GATE_PREREG.md").read_text()) <= set(pilot["prereg_items"])
    assert (out / "budget_calibration_measured.yaml").is_file() and pilot["cost_model_entries"] > 0

    # D-017: build-dev measured the build quality on every gate cell, marked offline (coverage 1.0 by construction),
    # and the pilot turned it into the [USER: builder] item and the worlds-per-cell note.
    bq = json.loads((run_dir / "build-dev" / "build_quality.json").read_text())
    assert bq["offline"] is True and "not a quality measurement" in bq["note"] and bq["verdict"] == "builder_passes"
    assert set(bq["cells"]) == set(GATE_CELLS) and all(c["pass"] and c["apg"]["coverage"] == 1.0 for c in bq["cells"].values())
    assert bq["fallback_extra_usd"] > 0 and bq["builder"].startswith("scripted perfect_author (offline stand-in for gpt-6-luna")
    assert m["build-dev"]["build_quality"]["verdict"] == "builder_passes" and m["build-dev"]["outputs"]["build_quality"]
    assert pilot["prereg_items"]["builder"] == bq["recommendation"] and bq["recommendation"].startswith("OFFLINE, not a quality measurement: ")
    assert "D-017 build check: the builder passes, so 16 per D-017" in pilot["prereg_items"]["test worlds per cell"]
    assert pilot["build_quality"]["verdict"] == "builder_passes"

    # The rehearsal freeze: every body placeholder filled in a copy; the real prereg and PROVENANCE.md untouched.
    freeze = json.loads(run.freeze_path.read_text())
    n_markers = len(run_gate.prereg_placeholders((ROOT / "GATE_PREREG.md").read_text()))
    assert freeze["rehearsal"] is True and len(freeze["placeholders_replaced"]) == n_markers > 0
    assert {"marker": "[USER: builder]", "value": bq["recommendation"]} in freeze["placeholders_replaced"]
    assert run_gate.prereg_placeholders((run.work_dir / "GATE_PREREG.md").read_text()) == []
    expected = {"GATE_PREREG.md", "GATE_PREREG.md (draft)", *(f"config/{n}" for n in run_gate.FROZEN_CONFIG + run_gate.FROZEN_OUTPUTS), *run_gate.FROZEN_CODE}
    assert set(freeze["files"]) == expected and all(f["sha256"] for f in freeze["files"].values())
    assert freeze["apg_core"]["installed_commit"] == run_gate.APG_PIN and freeze["analysis_commit"]
    assert "OFFLINE REHEARSAL" in (run.work_dir / "PROVENANCE.freeze.md").read_text()
    assert (ROOT / "PROVENANCE.md").read_bytes() == real_provenance
    assert run_gate.require_frozen(run)["run_id"] == "t2"

    # 3. Re-running skips everything, freeze included.
    assert max(main([p, *argv]) for p in THROUGH_FREEZE) == 0
    assert _statuses(run_dir, THROUGH_FREEZE) == dict.fromkeys(THROUGH_FREEZE, "skipped")

    # 4. A frozen run never re-tunes, re-pilots or re-freezes, even with --force; the manifests are left alone.
    forced = GateRun("t2", offline=True, force=True, runs_root=tmp / "runs", config_dir=config)
    for phase in ("tune", "pilot"):
        with pytest.raises(PhaseError, match=rf"{phase}: run 't2' is frozen .* logged deviation"):
            run_phases(forced, phase)
        assert read_manifest(run, phase)["status"] == "skipped"
    with pytest.raises(PhaseError, match="a run is frozen once"):
        run_phases(forced, "freeze")
    assert main(["tune", "--force", *argv]) == 1

    # 5. Editing a frozen file trips the guard (naming the file) and blocks the phases that read it.
    (out / "selected.yaml").write_text((out / "selected.yaml").read_text() + "\n# edited after the freeze\n")
    with pytest.raises(PhaseError, match=r"config/selected.yaml \(.*selected.yaml\): changed"):
        run_gate.require_frozen(run)
    with pytest.raises(PhaseError, match="pilot: run 't2' is frozen"):
        run_phases(run, "all")


def test_s7_reads_its_targets_from_APE_S7_TARGETS(clean_env, monkeypatch):
    from ape.worlds.generate import make_world

    world = make_world("F7", "10", "dev", 0, 2)
    targets = clean_env / "s7_targets.json"
    targets.write_text(json.dumps({"F7-10": 123}))
    monkeypatch.setenv("APE_S7_TARGETS", str(targets))
    assert arms.s7_target(Config(), world) == 123
    monkeypatch.setenv("APE_S7_TARGET", "77")
    assert arms.s7_target(Config(), make_world("F3", "5", "dev", 0, 2)) == 77, "a cell the file lacks falls back"
    monkeypatch.setenv("APE_S7_TARGETS", str(clean_env / "missing.json"))
    assert arms.s7_target(Config(), world) == 77
    monkeypatch.delenv("APE_S7_TARGET")
    monkeypatch.setenv("APE_CONTEXT_BUDGET", "900")
    assert arms.s7_target(Config(), world) == 900


def test_budget_calibration_search_interpolates_across_lightrags_threshold_and_reports_failure():
    def runner(fn):
        calls = []

        def measure(it, knobs):
            calls.append(knobs)
            return {a: {"median": fn(a, int(next(iter(k.values())))), "per_cell": {}, "log_dir": f"iter{it}"} for a, k in knobs.items()}

        return measure, calls

    # Linear: the first guess (target / start median) lands at once; explicit per-part caps scale with the budget.
    measure, calls = runner(lambda a, b: 0.75 * b)
    got = run_gate.calibrate_caps(300, {"S3s": {"APE_S3S_BUDGET": 1000}, "LGR*": {"APE_LGR_BUDGET": 2000, "APE_LGR_TOTAL_TOKENS": 6000}}, {"S3s": 750, "LGR*": 1500}, measure)
    assert all(g["converged"] for g in got.values()) and len(calls) == 1
    assert got["LGR*"]["env"] == {"APE_LGR_BUDGET": "400", "APE_LGR_TOTAL_TOKENS": "1200"}

    # A threshold like LightRAG's (below some cap its context collapses): the ratio step undershoots to the
    # collapsed side, and interpolation between the bracketing points lands.
    measure, calls = runner(lambda a, b: 16.0 if b < 300 else 1.2 * b - 100)
    got = run_gate.calibrate_caps(300, {"LGR*": {"APE_LGR_BUDGET": 2000}}, {"LGR*": 2300}, measure)["LGR*"]
    assert [i["median"] for i in got["iterations"]][1] > 300, "the second step interpolates past the threshold"
    assert got["converged"] and 225 <= got["median"] <= 375 and len(got["iterations"]) == 3

    # Out of reach: not converged after the cap on iterations; the closest iteration is reported.
    measure, calls = runner(lambda a, b: 900.0 if b > 10 else 5.0)
    got = run_gate.calibrate_caps(300, {"S3s": {"APE_S3S_BUDGET": 1000}}, {"S3s": 900}, measure)["S3s"]
    assert not got["converged"] and len(got["iterations"]) == run_gate.CAL_MAX_ITERATIONS and got["median"] in (900.0, 5.0)


def _fake_done(run: GateRun, phase: str, **extra) -> None:
    run_gate._write_json(run.manifest_path(phase), {"phase": phase, "status": "done", "fingerprint": f"fake-{phase}", "finished": "2026-10-01T00:00:00+00:00", "outputs": {}} | extra)


def test_live_freeze_refuses_placeholders_and_uncommitted_changes_and_its_guard_names_changed_files(clean_env, monkeypatch):
    tmp = clean_env
    monkeypatch.setenv("APE_CACHE", str(tmp / "cache"))  # the spend check reads this ledger, not the real one
    out = tmp / "out"
    monkeypatch.setattr(GateRun, "out_config_dir", property(lambda self: out))  # live outputs, never config/
    out.mkdir()
    (out / "selected.yaml").write_text(yaml.safe_dump({k: {"arm": a, "env": {}, "candidate": "c"} for k, a in (("APG*", "APG-s"), ("LGR*", "LGR-s"), ("S3s", "S3s"))}))
    (out / "s7_targets.json").write_text(json.dumps({c: 120 for c in GATE_CELLS}))
    (out / "budget_calibration.yaml").write_text("context: 300\n")
    prereg, provenance = tmp / "GATE_PREREG.md", tmp / "PROVENANCE.md"
    shutil.copy(ROOT / "GATE_PREREG.md", prereg)
    shutil.copy(ROOT / "PROVENANCE.md", provenance)
    paths = {"runs_root": tmp / "runs", "prereg_path": prereg, "provenance_path": provenance}
    run = GateRun("live-freeze", **paths)
    run_gate._check_mode(run)
    for phase in ("preflight", "tune", "pilot"):
        _fake_done(run, phase)
    _fake_done(run, "anchor", pc1_pass=True)
    changes: list[str] = []
    monkeypatch.setattr(run_gate, "git_tracked_changes", lambda: list(changes))

    # 1. The draft: every unfilled body item is listed with its line; nothing is frozen.
    with pytest.raises(PhaseError, match=r"unfilled item\(s\)(.|\n)*\[PILOT: test worlds per cell\](.|\n)*\[USER: skeptic\]"):
        run_phases(run, "freeze")
    assert not run.freeze_path.exists() and read_manifest(run, "freeze")["status"] == "failed"

    # 2. Filled (the header's own mentions of the markers do not count), but with uncommitted changes.
    text = prereg.read_text()
    start = run_gate.prereg_body_start(text)
    prereg.write_text(text[:start] + run_gate.PLACEHOLDER_ITEM.sub("filled", text[start:]))
    assert "[PILOT: …]" in prereg.read_text()[:start] and run_gate.prereg_placeholders(prereg.read_text()) == []
    changes[:] = ["src/ape/analysis/gate_stats.py"]
    with pytest.raises(PhaseError, match=r"uncommitted changes \['src/ape/analysis/gate_stats.py'\]"):
        run_phases(run, "freeze")

    # 3. Committed: frozen, recorded in PROVENANCE.md; the real one is untouched.
    changes.clear()
    real = (ROOT / "PROVENANCE.md").read_bytes()
    assert run_phases(run, "freeze") == {"freeze": "done"}
    freeze = run_gate.require_frozen(run)
    assert freeze["rehearsal"] is False and freeze["files"]["GATE_PREREG.md"]["sha256"] == run_gate._sha256(prereg)
    assert "Gate freeze: run `live-freeze`" in provenance.read_text() and (ROOT / "PROVENANCE.md").read_bytes() == real

    # 4. The guard names a changed frozen file; tune, pilot and freeze refuse to run, --force or not.
    (out / "s7_targets.json").write_text(json.dumps({c: 999 for c in GATE_CELLS}))
    with pytest.raises(PhaseError, match=r"config/s7_targets.json \(.*\): changed"):
        run_gate.require_frozen(run)
    forced = GateRun("live-freeze", force=True, **paths)
    for phase in ("tune", "pilot"):
        with pytest.raises(PhaseError, match=f"{phase}: run 'live-freeze' is frozen"):
            run_phases(forced, phase)
    with pytest.raises(PhaseError, match=r"frozen once, and frozen file\(s\) changed since: config/s7_targets.json"):
        run_phases(run, "freeze")
    (out / "s7_targets.json").unlink()
    with pytest.raises(PhaseError, match=r"config/s7_targets.json \(.*\): missing"):
        run_gate.require_frozen(run)


def test_the_real_prereg_marks_every_open_item_with_a_label():
    text = (ROOT / "GATE_PREREG.md").read_text()
    items = run_gate.prereg_placeholders(text)
    assert items and all(p["label"] for p in items), "every body marker is a labelled [PILOT: ...] or [USER: ...]"
    assert {p["label"] for p in items if p["kind"] == "USER"} == {"builder", "APG owner", "skeptic", "analyst"}
    assert run_gate.PLACEHOLDER.search(text[: run_gate.prereg_body_start(text)]), "the header describes the markers"


# --- FX-6b part 2: build-test and test -------------------------------------------------------------


def test_offline_all_builds_the_test_split_after_the_freeze_runs_the_primary_first_and_resumes(clean_env, monkeypatch):
    from inspect_ai.log import read_eval_log

    tmp = clean_env
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    argv = ["--offline", "--run-id", "t3", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]
    run = GateRun("t3", offline=True, runs_root=tmp / "runs", config_dir=config)
    run_dir, out = run.dir, run.out_config_dir
    plan_cells = [c.id for c in run_gate._test_cells(run_gate.plan(run))]
    primary = list(run_gate.PRIMARY_CELLS)

    # 1. A budget that only covers the primary: the test phase runs the GO-rule cells first, then stops.
    real_projected = run_gate._group_projected
    monkeypatch.setattr(run_gate, "_group_projected", lambda r, g: real_projected(r, g) if g["cell"] in run_gate.PRIMARY_CELLS else 1e9)
    with pytest.raises(BudgetError, match=r"test gate.sec.id-only \(selected\): projected \$1,000,000,000.00 exceeds(.|\n)*the primary cells are complete"):
        run_phases(run, "all")
    m = read_manifest(run, "test")
    assert m["status"] == "failed" and m["primary_complete"] is True
    assert list(m["cells"])[: len(primary)] == primary and set(m["cells"]) == set(plan_cells)
    assert {c: x["status"] for c, x in m["cells"].items()} == {c: "done" if c in primary else "stopped" for c in plan_cells}
    monkeypatch.setattr(run_gate, "_group_projected", real_projected)
    # `all` still writes the report after the budget stop, on what exists: the stopped cells are missing, by name.
    a = read_manifest(run, "analyze")
    assert a["status"] == "done" and a["verdict"] == "PRECONDITION_FAIL"
    d = json.loads((run_dir / "report" / "decision.json").read_text())
    assert d["header"]["primary_complete"] is True and d["header"]["cells_not_done"] == {c: "stopped" for c in plan_cells if c not in primary}
    assert d["secondaries"]["gate.sec.messy"]["status"] == "missing" and d["secondaries"]["gate.sec.messy"]["reason"] == "selected: group stopped"
    pcs = {p["id"]: p for p in d["preconditions"]}
    # The verdict's cells all ran (PC2 and PC3 have their data); the stopped matched-budget secondary is only noted.
    assert "no results" not in (pcs["PC2"]["reason"] or "") and "no results" not in (pcs["PC3"]["reason"] or "")
    assert "matched-budget secondary was not run" in pcs["PC4"]["details"]["matched"]["notes"][0]
    assert {m: x["verdict"] for m, x in d["verdict"]["modes"].items()} == {"push": "PRECONDITION_FAIL", "pull": "PRECONDITION_FAIL"}

    # 2. A failing group is recorded; the other cells still run; the phase fails.
    real_tasks = run_gate.run_gate_tasks

    def te_all_breaks(r, gp, models, tasks, log_dir, epochs, what):
        if "gate.sec.te-all" in what:
            raise PhaseError("injected failure")
        return real_tasks(r, gp, models, tasks, log_dir, epochs, what)

    monkeypatch.setattr(run_gate, "run_gate_tasks", te_all_breaks)
    with pytest.raises(PhaseError, match=r"1 test group\(s\) failed .*gate.sec.te-all \(selected\): PhaseError: injected failure"):
        run_phases(run, "test")
    m = read_manifest(run, "test")
    assert {c: x["status"] for c, x in m["cells"].items()} == {c: "failed" if c == "gate.sec.te-all" else "done" for c in plan_cells}
    monkeypatch.setattr(run_gate, "run_gate_tasks", real_tasks)

    # 3. The re-run resumes (finished logs are reused) and completes every cell.
    assert main(["all", *argv]) == 0
    m = _manifests(run_dir, PHASES)
    assert m["test"]["status"] == "done" and m["build-test"]["status"] == "skipped" and m["test"]["primary_complete"] is True
    assert m["test"]["upstream"] == {"freeze": m["freeze"]["fingerprint"], "build-test": m["build-test"]["fingerprint"]}
    assert set(m["test"]["inputs"]) == set(json.loads(run.freeze_path.read_text())["files"]), "the frozen files are the test's inputs"
    assert m["test"]["offline_check"]["models"] == ["mockllm/model"] and m["test"]["projected_usd"] > 100, "the live projection, whole worlds"

    # The decision report: every section, every cell covered; the mock anchor fails PC1, so PRECONDITION_FAIL.
    from ape.analyze_gate import SECTIONS

    assert m["analyze"]["status"] == "done" and m["analyze"]["outputs"].keys() == {"decision", "report"}
    d = json.loads((run_dir / "report" / "decision.json").read_text())
    report = (run_dir / "report" / "report.md").read_text()
    assert all(f"\n## {s}\n" in report for s in SECTIONS), [s for s in SECTIONS if f"\n## {s}\n" not in report]
    assert d["verdict"]["label"] == "PRECONDITION_FAIL" and d["verdict"]["reasons"][0].startswith("PC1: ")
    assert set(d["verdict"]["modes"]) == {"push", "pull"} and all(c["status"] == "done" for c in d["coverage"])
    assert {c: s["status"] for c, s in d["secondaries"].items()} == dict.fromkeys(d["secondaries"], "done")
    assert d["header"]["freeze"]["rehearsal"] is True and set(d["tables"]["arms"]) >= {"APG* (push)", "APG* (pull)", "LGR* (push)", "S7 (push)", "S5o (push)"}

    # The test worlds: split "test", built after the freeze; id_only renderings pair the first test seeds.
    worlds = json.loads((run_dir / "build-test" / "worlds.json").read_text())["worlds"]
    assert {w["group"] for w in worlds} == {"test", "id_only", "f5"} and all("-test-" in w["world_id"] for w in worlds)
    seeds = {g: {w["world_id"].rsplit("-", 1)[1] for w in worlds if w["group"] == g} for g in ("test", "id_only")}
    assert seeds["id_only"] <= seeds["test"]

    # Every plan cell's groups: their logs, and what each log says it ran.
    selected = read_selected(out / "selected.yaml")
    calibration = yaml.safe_load((out / "budget_calibration.yaml").read_text())
    matched = {k: str(v) for a in calibration["arms"].values() for k, v in a["env"].items()}
    for cell_id, cell in m["test"]["cells"].items():
        assert cell["status"] == "done" and cell["primary"] == (cell_id in primary)
        for g in cell["groups"]:
            assert g["status"] == "done" and g["log_files"] and all(os.path.isfile(f) for f in g["log_files"])
            assert g["log_dir"].endswith(f"test/{cell_id}/{g['name']}-" + g["log_dir"].rsplit("-", 1)[1])
            for f in g["log_files"]:
                head = read_eval_log(f, header_only=True)
                md = head.eval.metadata
                assert head.eval.model == "mockllm/model" and head.status == "success"
                assert (md["plan_cell"], md["group"], md["split"], md["exposure"]) == (cell_id, g["name"], g["split"], g["exposure"])
                assert md["delivery"] in g["deliveries"] and md["arm"] in {a["run"] for a in g["arms"]}
                assert md["knobs"] == {k: v for k, v in g["env"].items()} | {"APE_S7_TARGETS": str(out / "s7_targets.json")}
    groups = {(c, g["name"]): g for c, x in m["test"]["cells"].items() for g in x["groups"]}
    assert groups[("gate.sec.messy", "selected")]["split"] == "dev" and groups[("gate.sec.messy", "selected")]["exception_style"] == "messy"
    assert groups[("gate.sec.id-only", "selected")]["exception_style"] == "id_only" and groups[("gate.sec.te-all", "selected")]["exposure"] == "all"
    assert groups[("gate.test.f7", "selected")]["deliveries"] == ["push", "pull"]
    # The matched-budget secondary: the selected knobs, then the calibration's on top.
    m300 = groups[("gate.sec.matched-300", "matched")]
    assert m300["env"] == run_gate.selected_env(selected) | matched and m300["env"]["APE_APG_FILL"] == "1"
    # LightRAG naive (PC2): LGR*'s arm in naive mode, in its own group; every compile ran naive.
    naive = groups[("gate.f5", "lgr-naive")]
    assert [a["run"] for a in naive["arms"]] == [selected["LGR*"]["arm"]] and naive["env"]["APE_LGR_MODE"] == "naive"
    modes = {r["meta"]["lightrag"]["mode"] for f in naive["log_files"] for s in read_eval_log(f).samples for r in s.store["compile_log"]}
    assert modes == {"naive"}
    assert groups[("gate.f5", "selected")]["env"].get("APE_LGR_MODE") != "naive"

    # 4. Re-running skips every phase.
    assert main(["all", *argv]) == 0
    assert _statuses(run_dir, PHASES) == dict.fromkeys(PHASES, "skipped")

    from ape.analyze_gate import main as analyze_main

    assert analyze_main(["--run-id", "t3", "--runs-dir", str(tmp / "runs")]) == 0, "the analyze CLI finds the run's mode and config"
    assert read_manifest(run, "analyze")["status"] == "skipped"

    # 5. A frozen file edited after the freeze: build-test and test refuse, naming it; their manifests stay as they were.
    (out / "selected.yaml").write_text((out / "selected.yaml").read_text() + "\n# edited after the freeze\n")
    for phase in ("build-test", "test"):
        with pytest.raises(PhaseError, match=rf"{phase}: frozen file\(s\) changed since the freeze at .*config/selected.yaml \(.*\): changed"):
            run_phases(run, phase)
        assert read_manifest(run, phase)["status"] == "skipped"

    # 6. A missing input fails its precondition by name; the report is still written.
    (run_dir / "anchor" / "pc1.json").rename(run_dir / "anchor" / "pc1.json.moved")
    assert run_phases(run, "analyze") == {"analyze": "done"}
    d = json.loads((run_dir / "report" / "decision.json").read_text())
    assert d["verdict"]["label"] == "PRECONDITION_FAIL" and "pc1.json missing: run the anchor phase" in d["verdict"]["reasons"][0]


def test_the_test_split_is_built_only_by_the_orchestrators_build_test(clean_env, monkeypatch):
    import sys

    from ape import build as build_cli
    from ape.worlds.generate import TEST_SPLIT_ENV, TestSplitLocked

    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache"}.items():
        monkeypatch.setenv(k, str(clean_env / v))
    with pytest.raises(TestSplitLocked, match="run_gate build-test"):
        asyncio.run(build("test", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=False))
    monkeypatch.setattr(sys, "argv", ["ape.build", "--split", "test", "--family", "F7", "--levels", "10", "--no-embed"])
    with pytest.raises(SystemExit) as e:
        build_cli.main()
    assert e.value.code == 2
    with pytest.raises(TestSplitLocked):
        run_gate.build_world_set(GateRun("x", offline=True, runs_root=clean_env / "runs"), {"outputs": {}}, "build-test", "test", [])
    assert not (clean_env / "worlds" / "test").exists()
    monkeypatch.setenv(TEST_SPLIT_ENV, "a-frozen-run")  # what build-test sets, after its freeze guard
    assert asyncio.run(build("test", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=False))
    assert asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=False)), "other splits are never locked"


def test_test_groups_at_live_sizes_follow_the_plan():
    run = GateRun("plan-check")
    p = run_gate.plan(run)
    selected = {"APG*": {"arm": "APG-s", "env": {"APE_APG_SHORTLIST_K": "24"}}, "LGR*": {"arm": "LGR-s", "env": {"APE_LGR_MODE": "mix"}}, "S3s": {"arm": "S3s", "env": {"APE_S3S_BUDGET": "2000"}}}
    calibration = {"context": 300, "arms": {"APG*": {"env": {"APE_APG_FILL": "1", "APE_APG_BUDGET": "300", "APE_APG_SHORTLIST_K": "48"}}, "LGR*": {"converged": True, "env": {"APE_LGR_BUDGET": "350"}}, "S3s": {"converged": True, "env": {"APE_S3S_BUDGET": "310"}}}}
    groups = run_gate.test_groups(run, selected, calibration, offline=False)
    order = list(dict.fromkeys(g["cell"] for g in groups))
    assert order[: len(run_gate.PRIMARY_CELLS)] == list(run_gate.PRIMARY_CELLS) and set(order) == {c.id for c in p.cells if c.study == "gate" and c.phase in run_gate.TEST_RUN_PHASES}
    by = {(g["cell"], g["name"]): g for g in groups}
    tests = p.cell("gate.build.test").spec["worlds"]
    # worlds-cells run the plan's worlds; n_tasks-cells the first whole worlds covering n_tasks (12 tasks each).
    assert by[("gate.test.f7", "selected")]["n_worlds"] == {c: tests[c] for c in ("F7-10", "F7-1000")} and by[("gate.test.f7", "selected")]["epochs"] == 3
    assert by[("gate.diag", "selected")]["n_worlds"] == dict.fromkeys(("F7-10", "F7-1000", "F3-5", "F3-60"), 9) and by[("gate.diag", "selected")]["n_tasks"]["F3-5"] == 108
    assert by[("gate.sec.id-only", "selected")]["n_worlds"] == {"F7-10": 9, "F7-1000": 9} and by[("gate.sec.id-only", "selected")]["exception_style"] == "id_only"
    assert by[("gate.sec.messy", "selected")]["n_worlds"] == {"F7-10": 1, "F7-1000": 1} and by[("gate.sec.messy", "selected")]["split"] == "dev"
    assert by[("gate.f5", "selected")]["n_worlds"] == {"F5-1hop": 9, "F5-2hop": 9}
    # Env groups: the matched budget on top of the selections; LightRAG naive is LGR* in naive mode.
    assert by[("gate.sec.matched-300", "matched")]["env"] == {"APE_APG_SHORTLIST_K": "48", "APE_LGR_MODE": "mix", "APE_S3S_BUDGET": "310", "APE_APG_FILL": "1", "APE_APG_BUDGET": "300", "APE_LGR_BUDGET": "350"}
    assert by[("gate.f5", "lgr-naive")]["arms"] == [{"declared": "LGR-naive", "run": "LGR-s"}] and by[("gate.f5", "lgr-naive")]["env"]["APE_LGR_MODE"] == "naive"
    assert [a["declared"] for a in by[("gate.f5", "selected")]["arms"]] == ["APG*", "LGR*"]
    assert by[("gate.diag.s7", "selected")]["env"] == {"APE_APG_SHORTLIST_K": "24", "APE_LGR_MODE": "mix", "APE_S3S_BUDGET": "2000"}
    # A calibration for another budget, or one that did not converge, is refused.
    with pytest.raises(PhaseError, match="calibrates 2000 tokens"):
        run_gate.test_groups(run, selected, calibration | {"context": 2000}, offline=False)
    with pytest.raises(PhaseError, match="S3s has no converged"):
        run_gate.test_groups(run, selected, {**calibration, "arms": {**calibration["arms"], "S3s": {"converged": False, "env": {"APE_S3S_BUDGET": "9"}}}}, offline=False)
    # The projection prices what runs: n_tasks-cells as whole worlds (108 tasks for 100).
    assert run_gate._test_projected(run) > run_gate.project(run, [c for c in p.cells if c.study == "gate" and c.phase in run_gate.TEST_RUN_PHASES])


def test_a_budget_override_only_lowers_the_plans_budget(clean_env):
    low = GateRun("cap-low", offline=True, runs_root=clean_env / "runs", budget_usd=3.0)
    high = GateRun("cap-high", offline=True, runs_root=clean_env / "runs", budget_usd=1e6)
    plan_total = float(run_gate.plan(low).budget["total_usd"])
    assert run_gate.spend(low)["budget_usd"] == 3.0
    assert run_gate.spend(high)["budget_usd"] == plan_total
