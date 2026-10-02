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
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:  # the suite's isolated registry stays
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
    assert no_field["status"] == sc.FAIL and "no reasoning tokens" in no_field["reason"]
    # Inspect's ModelUsage (the Responses API path) reports them at the top level.
    inspect_usage = sc.effort_verdict({"accepted": True, "usage": {"reasoning_tokens": 700}}, {"accepted": True, "usage": {"reasoning_tokens": 90}})
    assert inspect_usage["status"] == sc.PASS


def _path(effort_status: str = sc.PASS, accepted: bool = True, snapshot: str = "s1") -> dict:
    call = {"accepted": accepted, "served_model": snapshot} | ({} if accepted else {"error": "BadRequestError: nope"})
    return {"api": "responses", "as_configured": call, "structured_output": call, "effort": {"verdict": sc.result(effort_status, {})}, "snapshot": snapshot, "snapshots_seen": [snapshot]}


def test_effort_from_probe_checks_every_role_and_path_of_the_call_path_probe():
    roles = {r: {"model": "openai/gpt-6-luna", "paths": {"inspect": _path()}} for r in ("agent", "kg")} | {"build": {"model": "gpt-6-luna", "paths": {"build": _path()}}}
    probe = {"version": sc.PROBE_VERSION, "roles": roles, "snapshots": {"gpt-6-luna": {"snapshot": "s1", "snapshots_seen": ["s1"]}}}
    assert sc.effort_from_probe(probe)["status"] == sc.PASS
    old_probe = sc.effort_from_probe({"agent": {"model": "m", "plain": {}}})
    assert old_probe["status"] == sc.FAIL and "re-run readiness/probe_openai.py" in old_probe["reason"]
    skipped = probe | {"roles": roles | {"judge": {"model": "openai/gpt-4o-mini", "paths": {"inspect": _path(sc.SKIP)}}}}
    assert sc.effort_from_probe(skipped)["status"] == sc.PASS, "a role with no effort setting is skipped, not failed"
    bad = sc.effort_from_probe(probe | {"roles": roles | {"build": {"model": "gpt-6-luna", "paths": {"build": _path(sc.FAIL)}}}})
    assert bad["status"] == sc.FAIL and "build/build: effort not honoured" in bad["reason"]
    rejected = sc.effort_from_probe(probe | {"roles": roles | {"kg": {"model": "m", "paths": {"inspect": _path(accepted=False)}}}})
    assert "kg/inspect: as configured call rejected (BadRequestError: nope)" in rejected["reason"]
    missing = sc.effort_from_probe(probe | {"roles": {k: v for k, v in roles.items() if k != "kg"}})
    assert missing["status"] == sc.FAIL and "did not cover ['kg']" in missing["reason"]
    moved = sc.effort_from_probe(probe | {"snapshots": {"gpt-6-luna": {"snapshot": None, "snapshots_seen": ["s1", "s2"]}}})
    assert moved["status"] == sc.FAIL and "more than one snapshot" in moved["reason"]
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
    # R5: one F8 session per arm with a forked probe; the per-step structure on every call after a tool step; one
    # F7-1000 world through the builder; retrieval credited by the delivered-text rule, LightRAG measured off L2.
    f8 = report["checks"]["f8_session"]["measured"]["arms"]
    assert set(f8) == {"CM0", "O-state"} and all(a["item_records"] == a["cases"] == smoke.F8["N"] and a["probes"] >= 1 for a in f8.values())
    ps = report["checks"]["perstep_reasoning"]["measured"]
    assert ps["calls_after_tool_step"] > 0 and ps["structure_ok"] == ps["calls_after_tool_step"]
    ex = report["checks"]["extract_f7_1000"]["measured"]
    assert ex["verdict"] == "builder_passes" and ex["offline"] is True and ex["chunks"] > 500 and ex["kinds"] == dict.fromkeys(("chunks", "apg", "lightrag"), "built")
    rt = report["checks"]["retrieval"]["measured"]
    assert "delivered-text" in rt["provenance"] and rt["lightrag_recall_query_keywords"] is not None
    assert (ROOT / "PROVENANCE.md").read_bytes() == provenance and smoke._tree_hash(ROOT / "config") == config


# ---------- R5: F8 session, F7-1000 extraction, per-step reasoning, retrieval provenance ----------

CATS = ("completed", "pending", "memos_in_force", "open_followups", "escalated")


def _session(**over) -> dict:
    good = {
        "log_status": "success",
        "sample_error": None,
        "n_cases": 2,
        "items": [{"position": 1, "view_tokens_first": 900}, {"position": 2, "view_tokens_first": 1400}],
        "views": [{"item": 1, "view_tokens": 900}, {"item": 2, "view_tokens": 1400}],
        "probes": [{"k": 2, "answer": dict.fromkeys(CATS, []) | {"completed": ["C-1"]}, "error": None}],
        "report_submitted": True,
        "report_nudged": False,
        "probe_in_history": False,
        "reasoning_only_turns": 0,
        "reasoning_only_unanswered": False,
    }
    return good | over


def test_f8_session_needs_records_report_valid_probes_kept_out_of_history():
    assert sc.f8_session_verdict({"CM0": _session(), "O-state": _session()}, CATS)["status"] == sc.PASS
    # A report nudge counts as the report phase working, even without a report.
    assert sc.f8_session_verdict({"CM0": _session(report_submitted=False, report_nudged=True)}, CATS)["status"] == sc.PASS
    cases = {
        "sample error": _session(sample_error="TypeError: boom"),
        "1 item record(s) for 2": _session(items=[{"position": 1, "view_tokens_first": 900}]),
        "without view tokens": _session(items=[{"position": 1, "view_tokens_first": 900}, {"position": 2, "view_tokens_first": None}]),
        "no report submitted": _session(report_submitted=False),
        "no forked probe": _session(probes=[]),
        "not schema-valid": _session(probes=[{"k": 2, "answer": {"completed": [1]} | dict.fromkeys(CATS[1:], []), "error": None}]),
        "entered the session history": _session(probe_in_history=True),
        "reasoning-only turn ended": _session(reasoning_only_turns=1, reasoning_only_unanswered=True),
        "log status error": _session(log_status="error"),
    }
    for needle, rec in cases.items():
        res = sc.f8_session_verdict({"CM0": _session(), "O-state": rec}, CATS)
        assert res["status"] == sc.FAIL and needle in res["reason"] and res["reason"].startswith("O-state"), (needle, res["reason"])
    # A probe missing a category is not schema-valid either.
    missing = _session(probes=[{"k": 2, "answer": {"completed": []}, "error": None}])
    assert sc.f8_session_verdict({"CM0": missing}, CATS)["status"] == sc.FAIL
    assert sc.f8_session_verdict({}, CATS)["status"] == sc.FAIL


def test_extraction_fails_only_on_a_broken_build_and_warns_on_low_coverage_or_lost_chunks():
    good = {"apg_id_coverage": 0.98, "lightrag_id_coverage": 0.97, "verdict": "builder_passes"}
    assert sc.extraction_verdict({}, good, [], 0.95)["status"] == sc.PASS
    broken = sc.extraction_verdict({"apg": "BuildError: 30 lost chunks"}, good, [], 0.95)
    assert broken["status"] == sc.FAIL and "apg: BuildError" in broken["reason"]
    low = sc.extraction_verdict({}, good | {"apg_id_coverage": 0.80}, [], 0.95)
    assert low["status"] == sc.WARN and "APG ID coverage 0.800" in low["reason"] and "Sol fallback" in low["reason"]
    lost = sc.extraction_verdict({}, good, ["F7-1000-x: APG: 3 lost chunk(s)"], 0.95)
    assert lost["status"] == sc.WARN and "lost chunk" in lost["reason"]
    unmeasured = sc.extraction_verdict({}, good | {"lightrag_id_coverage": None}, [], 0.95)
    assert unmeasured["status"] == sc.WARN and "not measured: LightRAG ID coverage" in unmeasured["reason"]


def test_perstep_verdict_judges_the_structure_and_only_reports_reasoning():
    ok = {"after_tool_step": True, "kb_in_last_tool": True, "user_after_assistant": False, "reasoning_after_last_user": 2, "reasoning_tokens": 40}
    first = {"after_tool_step": False, "kb_in_last_tool": False, "user_after_assistant": False}
    res = sc.perstep_verdict([first, ok, ok | {"reasoning_after_last_user": 0}])
    assert res["status"] == sc.PASS and res["measured"]["reasoning_items_carried_per_call"] == [2, 0]
    old = ok | {"kb_in_last_tool": False, "user_after_assistant": True, "turn": 1, "sample": "s"}
    assert sc.perstep_verdict([first, old])["status"] == sc.FAIL
    assert "not exercised" in sc.perstep_verdict([first])["reason"]
    assert sc.perstep_verdict([first, ok | {"error": "400 bad request"}])["status"] == sc.FAIL


def test_perstep_calls_tell_the_fixed_placement_from_the_one_that_drops_reasoning():
    """The classifier on hand-built inputs: the knowledge on the last tool result keeps the model's reasoning after
    the last user message (OpenAI carries it); a knowledge-only user message after the tool result does not."""
    from types import SimpleNamespace

    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentReasoning
    from inspect_ai.tool import ToolCall

    from ape.agent.kb_react import STEP_KB_HEADER

    kb = f"{STEP_KB_HEADER}\nPolicy P-1: refunds need a receipt."
    turn = ChatMessageAssistant(content=[ContentReasoning(reasoning="gAAAA-ENCRYPTED", redacted=True)], tool_calls=[ToolCall(id="c1", function="lookup_customer", arguments={"customer_id": "CU-1"})])
    result = ChatMessageTool(content='{"tier": "gold"}', tool_call_id="c1", function="lookup_customer")
    base = [ChatMessageSystem(content="sys"), ChatMessageUser(content="Case: refund for CU-1")]
    fixed = [*base, turn, result.model_copy(update={"content": f"{result.content}\n\n{kb}"})]
    old = [*base, turn, result, ChatMessageUser(content=kb)]

    def sample(inputs):
        def ev(msgs):
            return SimpleNamespace(event="model", role=None, input=msgs, output=SimpleNamespace(usage=SimpleNamespace(reasoning_tokens=12)), error=None)

        return SimpleNamespace(id="s1", events=[ev([*base, ChatMessageUser(content=kb)]), ev(inputs)])

    first, new = smoke._perstep_calls(sample(fixed), STEP_KB_HEADER)
    assert not first["after_tool_step"]
    assert new["after_tool_step"] and new["kb_in_last_tool"] and not new["user_after_assistant"]
    assert new["reasoning_items"] == 1 and new["reasoning_after_last_user"] == 1 and new["reasoning_tokens"] == 12
    _, dropped = smoke._perstep_calls(sample(old), STEP_KB_HEADER)
    assert dropped["after_tool_step"] and not dropped["kb_in_last_tool"] and dropped["user_after_assistant"]
    assert dropped["reasoning_after_last_user"] == 0
    assert sc.perstep_verdict([new])["status"] == sc.PASS and sc.perstep_verdict([dropped])["status"] == sc.FAIL


def test_retrieval_credits_apg_nodes_by_their_delivered_text_not_their_source_chunks():
    from ape.kb.provenance import fact_matcher
    from ape.worlds.generate import make_world

    world = make_world("F7", "10", "dev", 0, 2)
    matcher = fact_matcher(world)
    fact = next(f for f in world.facts.values() if f.kind == "policy")

    def node(text, **props):
        return {"prompt": {"slots": {"knowledge": text}}, "props": props}

    assert fact.id in smoke.node_facts(matcher, node(fact.text))
    # The old mapping credited every fact of a node's source chunks; the delivered-text rule credits none of them
    # when the knowledge does not carry them.
    assert smoke.node_facts(matcher, node("General guidance about customer service.", sourceChunkIds=["c-0", "c-1"])) == set()
    # An oracle node without text keeps its spec fact IDs (as `ApgArm._facts_of`).
    assert smoke.node_facts(matcher, node("", factIds=[fact.id])) == {fact.id}


def test_lightrag_retrieval_recall_is_reported_never_judged():
    res = sc.retrieval_verdict([0.9, 1.0], [True, True], graph="authored", dry=False, lightrag_recall=[0.1, 0.2], lightrag_note="extract index")
    assert res["status"] == sc.PASS and res["measured"]["lightrag_recall_query_keywords"] == pytest.approx(0.15)


def test_the_new_checks_are_priced_and_build_their_own_worlds(clean_env):
    proj = smoke.projections(["perstep_reasoning", "f8_session", "extract_f7_1000"], "gate", dry=True)
    assert all(v > 0 for v in proj.values()), proj
    assert proj["extract_f7_1000"] > proj["f8_session"] > proj["perstep_reasoning"]  # one F7-1000 world's builds dominate
    assert not {"perstep_reasoning", "f8_session", "extract_f7_1000"} & smoke.SHARED_WORLD_STEPS
    assert smoke.select_steps(["f8_session"], None) == ["f8_session"]
