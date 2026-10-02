"""Error tolerance and resumable runs (FX-3): `ape.runner.run_evals` over Inspect's `eval_set`.
Offline: mock models, fake embeddings, tiny worlds."""

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
from inspect_ai.model._model_info import clear_model_info_cache
from inspect_ai.solver import generate

from ape.agent import arms
from ape.build import build
from ape.llm.mock_agent import mock_agent
from ape.models import COSTS_PATH, Concurrency, PreflightError, load_profile
from ape.runner import INDEX_NAME, log_path, run_evals
from ape.tasks.gate import gate, gate_samples

MOCK = "mockllm/model"
MODEL_ENV = ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_FALLBACK", "APE_EMBEDDING_MODEL")
FAST = {"display": "none", "retry_wait": 0.01}  # keep the task-retry back-off out of test time


@pytest.fixture(autouse=True)
def offline_env(tmp_path, monkeypatch):
    for k in MODEL_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    clear_model_info_cache()  # eval_cost_kwargs registers prices process-wide
    yield tmp_path
    clear_model_info_cache()


def _world(n_tasks: int = 4) -> None:
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=n_tasks, relational=True, embed=True))


def _gate(arm: str = "S1") -> Task:
    return gate(family="F7", level="10", split="dev", arm=arm)


def _user_text(messages) -> str:
    return next((m.text for m in messages if m.role == "user"), "")


def _index(log_dir: Path) -> dict:
    return json.loads((log_dir / INDEX_NAME).read_text())


@task
def echo(n: int = 100) -> Task:
    """A plain task with n samples ("q0".."q{n-1}"), for exact error-budget counts."""
    return Task(dataset=[Sample(id=f"q{i}", input=f"q{i}", target="x") for i in range(n)], solver=generate())


# ---------- sample retries ----------


def test_transient_sample_errors_are_retried_and_the_retries_are_logged(offline_env):
    _world(4)
    flaky = {s.input for s in gate_samples("F7", "10", "dev")[:2]}
    calls: Counter = Counter()

    def first_call_fails(messages, tools, tool_choice, config):
        key = _user_text(messages)
        calls[key] += 1
        if calls[key] == 1 and any(f in key for f in flaky):
            raise RuntimeError("transient API error")
        return mock_agent(messages, tools, tool_choice, config)

    log_dir = offline_env / "logs"
    success, logs = run_evals(_gate(), log_dir, model_override=MOCK, model_args={"custom_outputs": first_call_fails}, **FAST)
    assert success and [h.status for h in logs] == ["success"]

    # Inspect keeps each retried error on the sample (error_retries); the summary counts them.
    full = read_eval_log(logs[0].location)
    retried = [s for s in full.samples if s.error_retries]
    assert len(retried) == len(flaky) == 2
    assert all(len(s.error_retries) == 1 and "transient API error" in s.error_retries[0].message for s in retried)
    assert all(s.error is None and s.scores for s in full.samples)
    assert sorted(s.retries or 0 for s in read_eval_log_sample_summaries(logs[0].location)) == [0, 0, 1, 1]

    (entry,) = _index(log_dir)["tasks"].values()
    assert entry["status"] == "success" and entry["samples"] == {"planned": 4, "logged": 4, "completed": 4, "errored": 0, "retries": 2}


# ---------- error budget (PC5: harness errors < 2%) ----------


@pytest.mark.parametrize("n_bad,ok", [(2, True), (3, False)])
def test_errored_samples_at_the_tasks_threshold_fail_it(offline_env, n_bad, ok):
    """A 100-sample task is under 150, so its circuit breaker is a count of max(3, ceil(2)) = 3: 2 errored are
    tolerated (status success, samples unscored); 3 fail the task. Retries cannot help: those samples raise on
    every attempt."""
    bad = {f"q{i}" for i in range(n_bad)}

    def always_fails(messages, tools, tool_choice, config):
        if _user_text(messages) in bad:
            raise RuntimeError("permanent error")
        return mock_agent(messages, tools, tool_choice, config)

    log_dir = offline_env / "logs"
    success, (header,) = run_evals(echo(), log_dir, model_override=MOCK, model_args={"custom_outputs": always_fails}, **FAST)
    assert success is ok and header.status == ("success" if ok else "error")
    (entry,) = _index(log_dir)["tasks"].values()
    assert entry["status"] == header.status and entry["samples"]["errored"] == n_bad and entry["samples"]["planned"] == 100
    assert entry["samples"]["retries"] >= 2 * n_bad, "each bad sample was retried (retry_on_error=2) before counting"
    if ok:
        assert entry["samples"]["completed"] == 100 - n_bad and entry["error"] is None
    else:
        assert "permanent error" in entry["error"]


# ---------- resume ----------


def test_rerun_resumes_only_the_failed_task(offline_env, monkeypatch):
    _world(3)
    log_dir = offline_env / "logs"
    run = {"model_override": MOCK, "model_args": {"custom_outputs": mock_agent}, **FAST}

    def broken(world):
        raise RuntimeError("S6 arm is broken")

    with monkeypatch.context() as m:
        m.setattr(arms, "OracleContext", broken)  # every S6 sample fails; S1 is untouched
        success, logs = run_evals([_gate("S1"), _gate("S6")], log_dir, **run)
    assert not success
    status = {h.eval.task_args["arm"]: h.status for h in logs}
    assert status == {"S1": "success", "S6": "error"}
    first = {e["task_args"]["arm"]: e for e in _index(log_dir)["tasks"].values()}
    assert first["S6"]["status"] == "error" and first["S6"]["samples"]["errored"] == 3
    s1_log = Path(first["S1"]["log"])
    s1_mtime = s1_log.stat().st_mtime

    # Cause fixed; same log dir, freshly created tasks.
    success, logs = run_evals([_gate("S1"), _gate("S6")], log_dir, **run)
    assert success and all(h.status == "success" for h in logs)
    second = {e["task_args"]["arm"]: e for e in _index(log_dir)["tasks"].values()}
    assert second["S1"]["eval_id"] == first["S1"]["eval_id"], "the finished task was reused, not re-run"
    assert Path(second["S1"]["log"]) == s1_log and s1_log.stat().st_mtime == s1_mtime
    assert second["S6"]["status"] == "success" and second["S6"]["samples"]["errored"] == 0
    # retry_cleanup=False: the failed S6 attempts' logs stay (their spend stays countable); the index and eval_set
    # point at the newest one.
    s6_logs = [p for p in log_dir.glob("*.eval") if read_eval_log(str(p), header_only=True).eval.task_args.get("arm") == "S6"]
    assert len(s6_logs) >= 2 and Path(second["S6"]["log"]) in s6_logs
    assert [r["success"] for r in _index(log_dir)["runs"]] == [False, True]
    assert [r["status"] for r in _index(log_dir)["runs"]] == ["done", "done"]


def test_tasks_differing_only_in_arm_are_distinct_and_indexed(offline_env):
    _world(2)
    log_dir = offline_env / "logs"
    success, logs = run_evals([_gate("S1"), _gate("S6")], log_dir, model_override=MOCK, model_args={"custom_outputs": mock_agent}, **FAST)
    assert success and len(logs) == 2
    index = _index(log_dir)
    assert len(index["tasks"]) == 2, "two task identifiers"
    entries = sorted(index["tasks"].values(), key=lambda e: e["task_args"]["arm"])
    assert [e["task_args"] for e in entries] == [{"family": "F7", "level": "10", "split": "dev", "arm": a} for a in ("S1", "S6")]
    assert all(e["status"] == "success" and e["samples"]["completed"] == 2 and Path(e["log"]).is_file() for e in entries)
    assert {e["log"] for e in entries} == {log_path(h) for h in logs}


# ---------- models, prices, concurrency, preflight ----------


def test_profile_models_efforts_concurrency_and_prices_reach_the_log(offline_env):
    _world(2)
    costs = offline_env / "costs.yaml"
    table = yaml.safe_load(COSTS_PATH.read_text())
    table[MOCK] = {"input": 1.0, "output": 2.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
    costs.write_text(yaml.safe_dump(table))

    success, (header,) = run_evals(
        _gate("S1"), offline_env / "logs", profile="gate", model_override=MOCK, model_args={"custom_outputs": mock_agent}, costs_path=costs, **FAST
    )
    assert success
    log = read_eval_log(header.location)
    assert log.eval.model_generate_config.reasoning_effort == "high"  # FX-1: agent effort from the profile
    assert log.eval.model_roles["kg"].config.reasoning_effort == "low"
    assert log.eval.model_generate_config.max_connections == log.eval.model_roles["kg"].config.max_connections == 16
    cfg = log.eval.config
    # A 2-sample task: the circuit breaker is a count (3); every model is priced, so the per-sample cost guard is
    # set, at its default with no projection passed (run_plan.yaml budget.sample_cost_limit.default_usd).
    assert (cfg.fail_on_error, cfg.retry_on_error, cfg.max_samples, cfg.max_tasks, cfg.cost_limit) == (3.0, 2, 32, 2, 2.0)
    usage = [u for s in log.samples for u in s.model_usage.values()]
    assert usage and all(u.total_cost is not None and u.total_cost > 0 for u in usage)  # FX-2


def test_preflight_runs_before_any_model_or_log(offline_env):
    profiles = offline_env / "models.yaml"
    roles = {"agent": {"model": "openai/gpt-unpriced"}, "kg": {"model": "openai/gpt-6-luna"}, "build": {"model": "gpt-6-luna"}}
    profiles.write_text(yaml.safe_dump({"profiles": {"p": {**roles, "embeddings": {"model": "text-embedding-3-small"}}}}))
    with pytest.raises(PreflightError, match="gpt-unpriced"):
        run_evals(echo(), offline_env / "logs", profile=load_profile("p", profiles), live=False, **FAST)
    assert not (offline_env / "logs").exists()


def test_profile_concurrency_defaults_and_validation(offline_env):
    path = offline_env / "models.yaml"
    roles = {r: {"model": m} for r, m in [("agent", MOCK), ("kg", MOCK), ("build", "gpt-6-luna"), ("embeddings", "text-embedding-3-small")]}
    path.write_text(yaml.safe_dump({"profiles": {"plain": roles, "tuned": {**roles, "concurrency": {"max_connections": None, "max_tasks": 4}}}}))
    assert load_profile("plain", path).concurrency == Concurrency(16, 32, 2)
    assert load_profile("tuned", path).concurrency == Concurrency(None, 32, 4)  # null: Inspect's adaptive connections
    assert load_profile("gate").concurrency == Concurrency(16, 32, 2)
    for bad in ({"max_conections": 4}, {"max_samples": 0}, {"max_tasks": True}, [4]):
        path.write_text(yaml.safe_dump({"profiles": {"bad": {**roles, "concurrency": bad}}}))
        with pytest.raises(ValueError, match="concurrency"):
            load_profile("bad", path)
