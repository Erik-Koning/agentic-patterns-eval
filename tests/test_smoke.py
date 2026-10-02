"""Smoke coverage (FIX_PLAN FX-8): each check's pass/fail logic on crafted inputs, the spend guard, the smoke-only
fault injection through the FX-3 runner, and run_gate's SMOKE_SCALE. Offline: mock models, fake embeddings."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ape import run_gate as rg
from ape.config import ROOT

sys.path.insert(0, str(ROOT / "readiness"))
import smoke_checks as sc  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"readiness_{name}", ROOT / "readiness" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load("smoke")


@pytest.fixture
def clean_env(tmp_path):
    """No APE_* knobs going in; and the environment restored afterwards, since smoke.main and run_gate set
    variables directly."""
    import os

    saved = dict(os.environ)
    for k in [k for k in os.environ if k.startswith("APE_")]:
        del os.environ[k]
    yield tmp_path
    os.environ.clear()
    os.environ.update(saved)


def _usage(reasoning: int | None) -> dict:
    return {"completion_tokens_details": {"reasoning_tokens": reasoning}} if reasoning is not None else {"completion_tokens": 10}


# ---------- effort ----------


def test_effort_passes_only_when_high_reasons_more_than_low():
    ok = sc.effort_verdict({"accepted": True, "usage": _usage(800)}, {"accepted": True, "usage": _usage(100)})
    assert ok["status"] == sc.PASS and ok["measured"] == {"reasoning_tokens_high": 800, "reasoning_tokens_low": 100}
    same = sc.effort_verdict({"accepted": True, "usage": _usage(64)}, {"accepted": True, "usage": _usage(64)})
    assert same["status"] == sc.FAIL and "does not change reasoning" in same["reason"]
    rejected = sc.effort_verdict({"accepted": False, "error": "BadRequestError: unsupported"}, {"accepted": True, "usage": _usage(1)})
    assert rejected["status"] == sc.FAIL and "high rejected" in rejected["reason"] and "unsupported" in rejected["reason"]
    no_field = sc.effort_verdict({"accepted": True, "usage": _usage(None)}, {"accepted": True, "usage": _usage(None)})
    assert no_field["status"] == sc.FAIL and "reasoning_tokens" in no_field["reason"]


def test_effort_from_probe_needs_every_probed_role_and_the_new_effort_block():
    good = {"verdict": sc.result(sc.PASS, {})}
    assert sc.effort_from_probe({"agent": {"model": "m", "effort": good}, "kg": {"model": "m", "effort": good}})["status"] == sc.PASS
    old_probe = sc.effort_from_probe({"agent": {"model": "m", "plain": {}}})
    assert old_probe["status"] == sc.FAIL and "re-run readiness/probe_openai.py" in old_probe["reason"]
    bad = sc.effort_from_probe({"agent": {"model": "m", "effort": good}, "build": {"model": "m", "effort": {"verdict": sc.result(sc.FAIL, {})}}})
    assert bad["status"] == sc.FAIL and "['build']" in bad["reason"]
    assert sc.effort_from_probe({"models_available": []})["status"] == sc.FAIL


# ---------- pull, recovery ----------


def test_pull_needs_search_kb_in_three_quarters_of_each_arms_samples_and_no_errors():
    rows = lambda arm, calls, err=None: [{"arm": arm, "search_kb_calls": c, "error": err} for c in calls]  # noqa: E731
    assert sc.pull_verdict(rows("S3s", [1, 2, 1, 0]) + rows("APG-s", [1, 1, 1, 1]))["status"] == sc.PASS
    low = sc.pull_verdict(rows("S3s", [1, 0, 1, 0]) + rows("APG-s", [1, 1, 1, 1]))
    assert low["status"] == sc.FAIL and "S3s: search_kb in 50%" in low["reason"]
    err = sc.pull_verdict(rows("S3s", [1, 1, 1, 1]) + rows("APG-s", [1], err="boom"))
    assert err["status"] == sc.FAIL and "APG-s: 1 errored" in err["reason"]
    assert sc.pull_verdict([])["status"] == sc.FAIL


def test_recovery_needs_a_recorded_retry_and_a_clean_finish():
    ok = sc.recovery_verdict("success", [{"id": "t0", "error": None, "retries": 1, "retry_errors": ["RuntimeError(...)"]}])
    assert ok["status"] == sc.PASS
    assert "no retry recorded" in sc.recovery_verdict("success", [{"id": "t0", "error": None, "retries": 0}])["reason"]
    assert "still errored" in sc.recovery_verdict("success", [{"id": "t0", "error": "x", "retries": 2}])["reason"]
    assert "log status error" in sc.recovery_verdict("error", [{"id": "t0", "error": None, "retries": 1}])["reason"]


def test_the_smoke_fault_is_retried_by_the_runner_and_recorded(clean_env):
    """The smoke-only wrapper fails each sample's first attempt; the FX-3 runner's sample retry recovers it."""
    import os

    os.environ |= {"APE_WORLDS": str(clean_env / "worlds"), "APE_CACHE": str(clean_env / "cache"), "APE_INDICES": str(clean_env / "indices"), "APE_EMBEDDINGS": "fake"}
    from inspect_ai.log import read_eval_log

    from ape.build import build
    from ape.llm.mock_agent import mock_agent, mock_kg
    from ape.models import agent_model, load_profile, role_models
    from ape.runner import run_evals

    fault = _load("smoke_fault")
    asyncio.run(build("dev", "F3", ["5"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    p = load_profile("gate")
    agent = agent_model(p, model="mockllm/model", custom_outputs=mock_agent)
    roles = role_models(p, ("kg",), model="mockllm/model", custom_outputs=mock_kg)
    _, headers = run_evals(fault.faulted_gate(), clean_env / "logs", profile=p, model=agent, model_roles=roles, display="none", limit=1, retry_wait=1)
    log = read_eval_log(headers[0].location)
    samples = [{"id": s.id, "error": s.error, "retries": len(s.error_retries or []), "retry_errors": [str(e.message) for e in s.error_retries or []]} for s in log.samples]
    res = sc.recovery_verdict(log.status, samples)
    assert res["status"] == sc.PASS, res
    assert fault.FAULT in res["measured"]["retry_errors"][0]


# ---------- burst ----------


def test_rate_limit_signals_count_429s_and_rate_limit_messages():
    errors = ["RateLimitError: Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}", None, "APITimeoutError"]
    logs = ["-> openai/gpt-6-luna retry 1 (rate limit)", "HTTP Request: POST ... 200 OK", "Rate-limited; waiting"]
    assert sc.rate_limit_signals(errors, logs) == 3
    assert sc.tpm_limit({"x-ratelimit-limit-tokens": "2,000,000"}) == 2_000_000
    assert sc.tpm_limit({"x-ratelimit-limit-tokens": "lots"}) is None and sc.tpm_limit(None) is None


def test_concurrency_recommendation_halves_on_retries_and_respects_the_tpm_limit():
    assert sc.recommend_concurrency(16, 100, 0, 0, None, None, None)[0] == 16
    rec, why = sc.recommend_concurrency(16, 100, 10, 0, None, None, None)
    assert rec == 8 and "halve" in why
    assert sc.recommend_concurrency(6, 100, 0, 1, None, None, None)[0] == sc.MIN_CONCURRENCY
    # 10k tokens per call at 5 s: 120k tokens/min per connection; 80% of a 600k TPM limit allows 4.
    rec, why = sc.recommend_concurrency(16, 100, 0, 0, 10_000, 5.0, 600_000)
    assert rec == 4 and "caps it at 4" in why
    assert sc.recommend_concurrency(16, 100, 0, 0, 1_000, 5.0, 10_000_000)[0] == 16  # never more than measured


def test_burst_fails_on_failed_samples_warns_on_retries_and_reports_latency():
    samples = [{"error": None, "total_time": t, "working_time": t} for t in (1.0, 2.0, 3.0, 4.0)]
    events = [{"retries": 0, "error": None, "working_time": 0.5, "tokens": 1000} for _ in range(8)]
    ok = sc.burst_verdict(16, samples, events)
    assert ok["status"] == sc.PASS and ok["measured"]["sample_total_time_s"] == {"n": 4, "p50": 3.0, "p90": 4.0, "max": 4.0}
    assert ok["measured"]["recommended_max_connections"] == 16
    retried = sc.burst_verdict(16, samples, [*events[:-1], {"retries": 2, "error": None, "working_time": 9.0, "tokens": 1000}], ["429 Too Many Requests"])
    assert retried["status"] == sc.WARN and retried["measured"]["rate_limit_signals"] == 1 and retried["measured"]["recommended_max_connections"] == 8
    failed = sc.burst_verdict(16, [*samples, {"error": "boom"}], events)
    assert failed["status"] == sc.FAIL and failed["measured"]["recommended_max_connections"] == 8


# ---------- L5, retrieval ----------


def test_l5_fails_over_the_cap_and_warns_outside_the_band():
    assert sc.l5_verdict([2000, 2400, 3000], 8000)["status"] == sc.PASS
    over = sc.l5_verdict([2000, 8001], 8000)
    assert over["status"] == sc.FAIL and "over max_total_tokens 8000" in over["reason"]
    tiny = sc.l5_verdict([16, 20, 30], 8000)
    assert tiny["status"] == sc.WARN and "outside the expected band" in tiny["reason"]
    assert sc.l5_verdict([], 8000)["status"] == sc.FAIL


def test_retrieval_warns_below_point_eight_live_and_only_reports_dry():
    assert sc.retrieval_verdict([1.0, 0.9], [True, True], graph="authored", dry=False)["status"] == sc.PASS
    low = sc.retrieval_verdict([0.5, 0.6], [True, False, True, True], graph="oracle", dry=False)
    assert low["status"] == sc.WARN and "S3s recall 0.55" in low["reason"] and "APG shortlist gold rate 0.75" in low["reason"]
    dry = sc.retrieval_verdict([0.1], [False], graph="oracle", dry=True)
    assert dry["status"] == sc.PASS and "reported only" in dry["reason"]
    assert sc.retrieval_verdict([], [], graph="oracle", dry=False)["status"] == sc.FAIL


# ---------- orchestrator ----------


def test_orchestrator_verdict_needs_the_phases_outputs_and_a_smoke_freeze_refusal():
    done = dict.fromkeys(sc.SMOKE_PHASES, "done")
    files = {"tune/tuning_log.jsonl": True, "selected.yaml": True, "anchor/pc1.json": True}
    refusal = "freeze refused:\n  - run 'smoke-live' is a smoke run ...\n  - GATE_PREREG.md still has 13 unfilled item(s)"
    kw = {"placeholders": 13, "provenance_unchanged": True, "config_unchanged": True}
    assert sc.orchestrator_verdict(done, files, refusal, **kw)["status"] == sc.PASS
    assert "pilot: failed" in sc.orchestrator_verdict(done | {"pilot": "failed"}, files, refusal, **kw)["reason"]
    assert "selected.yaml not written" in sc.orchestrator_verdict(done, files | {"selected.yaml": False}, refusal, **kw)["reason"]
    assert "did not refuse" in sc.orchestrator_verdict(done, files, None, **kw)["reason"]
    assert "open item" in sc.orchestrator_verdict(done, files, "freeze refused: run 'x' is a smoke run", **kw)["reason"]
    assert "PROVENANCE.md changed" in sc.orchestrator_verdict(done, files, refusal, **kw | {"provenance_unchanged": False})["reason"]


# ---------- spend guard and selection ----------


def test_the_spend_guard_stops_before_a_step_that_would_pass_the_cap():
    sc.require_step_fits("H4", 0.5, 1.0, 3.0)
    with pytest.raises(SystemExit, match="stopping before burst: spent \\$2.9000 \\+ projected \\$0.2000 exceeds --max-usd 3.0"):
        sc.require_step_fits("burst", 0.2, 2.9, 3.0)
    assert sc.plan_fits({"a": 1.0, "b": 1.5}, 3.0) == (True, 2.5)
    assert sc.plan_fits({"a": 2.0, "b": 1.5}, 3.0)[0] is False


def test_step_selection_adds_what_a_check_needs_and_refuses_skipping_it():
    assert smoke.select_steps(["APG"], None) == ["L2", "D017", "APG"]
    assert smoke.select_steps(None, ["orchestrator", "burst"]) == [s for s in smoke.STEPS if s not in ("orchestrator", "burst")]
    with pytest.raises(SystemExit, match="--skip L2, but a selected check needs it"):
        smoke.select_steps(["L4_L5"], ["L2"])
    with pytest.raises(SystemExit, match="unknown check"):
        smoke.select_steps(["nope"], None)


def test_projections_price_every_model_check_and_refuse_a_cap_they_exceed(clean_env, monkeypatch):
    proj = smoke.projections(["effort", "H4", "burst", "retrieval"], "gate", dry=True)
    assert proj["effort"] == proj["retrieval"] == 0.0 and proj["H4"] > 0 and proj["burst"] > proj["H4"]
    monkeypatch.setattr(smoke, "OUT", clean_env / "smoke")
    with pytest.raises(SystemExit, match="exceeds --max-usd 0.001"):
        smoke.main(["--dry", "--only", "H4", "--max-usd", "0.001"])


def test_a_run_stops_before_the_check_the_cap_cannot_cover_and_still_writes_its_report(clean_env, monkeypatch):
    monkeypatch.setattr(smoke, "OUT", clean_env / "smoke")
    monkeypatch.setattr(smoke.Spend, "total", lambda self: 2.99)
    code = smoke.main(["--dry", "--only", "H4", "--max-usd", "3"])
    report = json.loads((clean_env / "smoke" / "report.json").read_text())
    assert code == 2 and report["status"] == "stopped" and "stopping before H4" in report["stopped"] and report["checks"] == {}
    assert (clean_env / "smoke" / "report.md").is_file()


# ---------- run_gate SMOKE_SCALE ----------


def test_smoke_scale_restricts_cells_worlds_tasks_and_candidates(clean_env):
    run = rg.GateRun("s", offline=True, smoke=True, runs_root=clean_env / "runs")
    keep = set(rg.SMOKE_SCALE["cells"])
    specs = rg.dev_world_specs(run)
    assert specs and {f"{s['family']}-{s['level']}" for s in specs} <= keep
    assert {(s["count"], s["n_tasks"]) for s in specs} == {(1, 2)}
    assert {s["group"] for s in specs} == {"gate", "id_only", "messy"}  # F5 has no smoke cell
    assert {f"{s['family']}-{s['level']}" for s in rg.pilot_world_specs(run)} == keep
    for spec in rg._pilot_cells(run).values():
        assert set(spec["cells"]) <= keep and spec["worlds"] == 1 and spec["epochs"] == 1
    grid = rg.tuning_grid(run)
    assert grid["dev_cells"] == ["F7-10", "F3-5"] and all(len(s["candidates"]) == 2 for s in grid["systems"].values())
    assert rg.scale(run)["cells"] == ["F7-10", "F3-5"] and rg.scale(run)["smoke"] is True
    # Live smoke keeps the real arms (no oracle swap) and is priced at smoke sizes, far below the full phases.
    live = rg.GateRun("s2", smoke=True, runs_root=clean_env / "runs")
    assert {c["arm"] for s in rg.tuning_grid(live)["systems"].values() for c in s["candidates"]} >= {"LGR-s"}
    full = rg.GateRun("f", offline=True, runs_root=clean_env / "runs")
    for phase in ("build-dev", "tune", "pilot"):
        assert 0 < rg.PHASE_DEFS[phase].projected(live) < rg.PHASE_DEFS[phase].projected(full) / 10, phase


def test_smoke_runs_are_isolated_and_never_freeze(clean_env, monkeypatch):
    with pytest.raises(rg.PhaseError, match="smoke runs live outside runs/"):
        rg.GateRun("s", smoke=True)
    run = rg.GateRun("s", smoke=True, runs_root=clean_env / "runs")
    assert run.out_config_dir == run.work_dir / "config" and run.anchor_data_dir == ROOT / "cache" / "graphragbench"
    with rg.run_environment(run):  # live smoke: real models, but its own worlds, indices and cache
        import os

        assert os.environ["APE_WORLDS"] == str(run.work_dir / "worlds") and os.environ["APE_CACHE"] == str(run.work_dir / "cache")
    # The freeze refuses before writing anything, listing the pre-registration's open items too.
    provenance = (ROOT / "PROVENANCE.md").read_bytes()
    record = {"warnings": [], "upstream": {}, "outputs": {}}
    with pytest.raises(rg.PhaseError) as e:
        rg._freeze(run, record)
    assert "is a smoke run" in str(e.value) and "unfilled item" in str(e.value)
    assert not run.freeze_path.exists() and not (run.work_dir / "GATE_PREREG.md").exists()
    assert (ROOT / "PROVENANCE.md").read_bytes() == provenance
    assert "smoke run" in rg.PHASE_DEFS["build-test"].refuse(run) and "smoke run" in rg.PHASE_DEFS["test"].refuse(run)
    # A run is smoke or not for its whole life.
    rg._check_mode(run)
    with pytest.raises(rg.PhaseError, match="is a smoke run"):
        rg._check_mode(rg.GateRun("s", runs_root=clean_env / "runs"))


def test_the_d017_check_has_no_coverage_code_of_its_own():
    """One definition of D-017's coverage (`ape.build_quality`) for the smoke check and run_gate's build-dev."""
    from ape import build_quality

    assert smoke.SPEC_THRESHOLD == build_quality.THRESHOLD
    assert not any(hasattr(smoke, name) for name in ("_coverage", "_entity_names", "_entity_file", "ID_PATTERN"))


def test_the_dry_smoke_passes_every_check_end_to_end(clean_env, monkeypatch):
    """`readiness/smoke.py --dry` from scratch (~20 s): every check, the orchestrator chain at SMOKE_SCALE included,
    passes offline; nothing outside its scratch area changes."""
    monkeypatch.setattr(smoke, "OUT", clean_env / "smoke")
    provenance, config = (ROOT / "PROVENANCE.md").read_bytes(), smoke._tree_hash(ROOT / "config")
    assert smoke.main(["--dry"]) == 0
    report = json.loads((clean_env / "smoke" / "report.json").read_text())
    assert report["status"] == sc.PASS and list(report["checks"]) == list(smoke.STEPS)
    assert {n: r["status"] for n, r in report["checks"].items()} == dict.fromkeys(smoke.STEPS, sc.PASS)
    assert report["spend_usd"] == 0.0 and 0 < report["projection_usd"]["total"] <= smoke.DEFAULT_MAX_USD
    orch = report["checks"]["orchestrator"]["measured"]
    assert orch["phases"] == dict.fromkeys(sc.SMOKE_PHASES, "done") and "is a smoke run" in orch["freeze_refusal"]
    assert report["checks"]["recovery"]["measured"]["retries"] == 1
    # D017 measures through `ape.build_quality`, the code run_gate's build-dev check uses: one decisive F7-100 world.
    d017 = report["checks"]["D017"]["measured"]
    assert d017["verdict"] == "builder_passes" and d017["offline"] is True and d017["world"]["decisive"]
    assert d017["world"]["cell"] == "F7-100" and d017["lightrag_id_coverage"] == d017["world"]["lightrag"]["coverage"] == 1.0
    assert (ROOT / "PROVENANCE.md").read_bytes() == provenance and smoke._tree_hash(ROOT / "config") == config
