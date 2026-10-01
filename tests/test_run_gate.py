"""Gate orchestrator (FX-6 part A): offline end to end, idempotence, the budget guard, live preflight, and
per-arm context budgets (FX-6 §0). Offline: mock models, fake embeddings, oracle indices, no network."""

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


def _manifests(run_dir) -> dict[str, dict]:
    return {p: json.loads((run_dir / p / "manifest.json").read_text()) for p in PHASES}


def _statuses(run_dir) -> dict[str, str]:
    return {p: m["status"] for p, m in _manifests(run_dir).items()}


def test_offline_all_runs_every_phase_then_skips_forces_and_reruns_on_changed_inputs(clean_env):
    tmp = clean_env
    config = tmp / "config"
    shutil.copytree(ROOT / "config", config)
    real_selected = REAL_SELECTED.read_bytes() if REAL_SELECTED.exists() else None
    argv = ["--offline", "--run-id", "t1", "--runs-dir", str(tmp / "runs"), "--config-dir", str(config)]
    run_dir = tmp / "runs" / "t1"

    # 1. Every part-A phase completes, with its manifest.
    assert main(["all", *argv]) == 0
    manifests = _manifests(run_dir)
    assert {p: m["status"] for p, m in manifests.items()} == dict.fromkeys(PHASES, "done")
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
    assert main(["all", *argv]) == 0
    again = _manifests(run_dir)
    assert {p: m["status"] for p, m in again.items()} == dict.fromkeys(PHASES, "skipped")
    assert all(again[p]["finished"] == manifests[p]["finished"] and again[p]["outputs"] == manifests[p]["outputs"] for p in PHASES)

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
    assert main(["all", *argv]) == 0
    assert _statuses(run_dir) == {"preflight": "skipped", "build-dev": "skipped", "tune": "done", "anchor": "skipped"}
    assert read_manifest(GateRun("t1", offline=True, runs_root=tmp / "runs", config_dir=config), "tune")["inputs"]["config/tuning_grid.yaml"]["sha256"] != forced["tune"]["inputs"]["config/tuning_grid.yaml"]["sha256"]

    # 5. A complete phase whose recorded output is gone re-runs.
    (run_dir / "work" / "config" / "selected.yaml").unlink()
    assert main(["all", *argv]) == 0
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
