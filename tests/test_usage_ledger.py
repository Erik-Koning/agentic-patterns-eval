"""The per-call usage ledger (`ape.usage_ledger`, D-030): Inspect logs a sample's last attempt only, so the usage of an
errored, retried attempt is in no log. Every run_evals run records each model call; spend counts the logs plus what
only the ledger holds, never twice. Offline: priced mock models."""

import json
from pathlib import Path

import pytest
import yaml
from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.model._model_info import clear_model_info_cache
from inspect_ai.solver import solver
from test_session_resume import Flaky, ckpt_dir, worlds  # noqa: F401  (fixtures)

from ape import run_gate, spend, usage_ledger
from ape.budget import logs_spend, program_spend, unlogged_spend, usage_ledgers
from ape.llm.mock_session import mock_session_agent
from ape.models import COSTS_PATH
from ape.runner import read_index, run_evals
from ape.tasks.study_g import f8_session

MOCK = "mockllm/model"
CALL_USD = 2.0  # 1M input tokens at $1/M + 100K output tokens at $10/M


@pytest.fixture
def priced(tmp_path, monkeypatch):
    monkeypatch.setenv(spend.REGISTRY_ENV, str(tmp_path / "registry.jsonl"))
    monkeypatch.setenv(spend.LABEL_ENV, "main/ledger-test")
    table = yaml.safe_load(COSTS_PATH.read_text())
    table[MOCK] = {"input": 1.0, "output": 10.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
    costs = tmp_path / "costs.yaml"
    costs.write_text(yaml.safe_dump(table))
    clear_model_info_cache()
    yield costs
    clear_model_info_cache()


def outputs(messages, tools, tool_choice, config) -> ModelOutput:
    out = ModelOutput.from_content(MOCK, "x")
    out.usage = ModelUsage(input_tokens=1_000_000, output_tokens=100_000, total_tokens=1_100_000)
    return out


FAILED_ONCE: set[str] = set()


@solver
def fails_once(fail_id: str, calls: int = 1):
    """`calls` generates per sample; sample `fail_id` then raises on its first attempt only (a transient error)."""

    async def solve(state, generate):
        for _ in range(calls):
            state = await generate(state)
        if state.sample_id == fail_id and fail_id not in FAILED_ONCE:
            FAILED_ONCE.add(fail_id)
            raise RuntimeError("transient error (test): the sample is retried")
        return state

    return solve


@task
def retried(fail_id: str = "q1", calls: int = 1) -> Task:
    return Task(dataset=[Sample(id=f"q{i}", input=f"ledger q{i}", target="x") for i in range(3)], solver=fails_once(fail_id, calls))


def test_a_retried_samples_first_attempt_is_counted_once(tmp_path, priced):
    FAILED_ONCE.clear()
    log_dir = tmp_path / "logs"
    ok, (header,) = run_evals(retried(calls=2), log_dir, model_override=MOCK, model_args={"custom_outputs": outputs}, costs_path=priced, display="none", retry_wait=0.01, sample_cost_usd=100.0)
    assert ok
    (q1,) = [s for s in read_eval_log(header.location).samples if s.id == "q1"]
    assert len(q1.error_retries) == 1, "q1 errored once and Inspect retried it"
    ledger = log_dir / usage_ledger.LEDGER_NAME
    assert read_index(log_dir)["runs"][-1]["usage_ledger"] == str(ledger)
    entries = usage_ledger.read_entries([ledger])
    assert len(entries) == 4 * 2 and all(e["cost_usd"] == pytest.approx(CALL_USD) and e["model"] == MOCK for e in entries)
    assert sorted({e["attempt"] for e in entries if e["sample_id"] == "q1"}) == [1, 2] and {e["attempt"] for e in entries if e["sample_id"] != "q1"} == {1}

    # The logs hold 3 samples x 2 calls; the ledger adds q1's first attempt (2 calls), which no log holds.
    logs = sorted(log_dir.glob("*.eval"))
    assert logs_spend(logs)["inspect_usd"] == pytest.approx(3 * 2 * CALL_USD)
    u = unlogged_spend(logs, usage_ledgers(log_dir))
    assert u["unlogged_usd"] == pytest.approx(2 * CALL_USD) and u["retried_samples"] == 1 and u["samples_in_no_log"] == 0
    assert u["ledger_usd"] == pytest.approx(4 * 2 * CALL_USD) and u["final_attempt_diff_usd"] == pytest.approx(0.0)
    s = program_spend(tmp_path / "registry.jsonl", priced)
    assert (s["inspect_usd"], s["unlogged_usd"], s["spent_usd"]) == pytest.approx((12.0, 4.0, 16.0)), "the measured difference: one retried attempt, $4"
    assert s["by_study"] == pytest.approx({"main": 16.0}) and s["usage_ledger_usd"] == pytest.approx(s["inspect_usd"] + s["unlogged_usd"])
    assert run_gate.group_spent(log_dir) == pytest.approx(16.0), "the study runner's per-group resume credit counts it too"


def test_a_killed_runs_in_flight_samples_count_from_the_ledger_and_limits_from_the_logs():
    """A sample in no log (in flight when the run was killed) counts all its calls; a final attempt the log holds counts
    from the log (which also has a call that crossed a sample limit: it raised before the hook ran); unpriced calls of
    a real model are an error, never $0."""
    entries = [
        {"sample_uuid": "a", "attempt": 1, "cost_usd": 1.0, "model": "openai/x", "ts": 10},
        {"sample_uuid": "a", "attempt": 2, "cost_usd": 1.5, "model": "openai/x", "ts": 20},
        {"sample_uuid": "killed", "attempt": 1, "cost_usd": 0.7, "model": "openai/x", "ts": 30},
        {"sample_uuid": None, "attempt": None, "cost_usd": 0.1, "model": "openai/x", "ts": 40},
    ]
    u = usage_ledger.unlogged_spend(entries, {"a": 2.0})
    assert u["unlogged_usd"] == pytest.approx(1.0 + 0.7 + 0.1) and u["samples_in_no_log"] == 1
    assert u["final_attempt_diff_usd"] == pytest.approx(1.5 - 2.0), "the log's limit-crossing call is not in the ledger"
    assert usage_ledger.unlogged_spend(entries, {"a": 2.0}, since=25)["unlogged_usd"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="unpriced"):
        usage_ledger.unlogged_spend([{"sample_uuid": "b", "cost_usd": None, "model": "openai/x", "input_tokens": 5}], {})
    assert usage_ledger.unlogged_spend([{"sample_uuid": "b", "cost_usd": None, "model": MOCK, "input_tokens": 5}], {})["unlogged_usd"] == 0.0


def test_the_ledger_records_only_inside_run_evals(tmp_path):
    assert usage_ledger.active_ledger() is None and not usage_ledger.UsageLedgerHook().enabled()
    inspect_eval(retried(fail_id="none"), model=get_model(MOCK, custom_outputs=outputs), log_dir=str(tmp_path / "direct"), display="none")
    assert not list((tmp_path / "direct").rglob(usage_ledger.LEDGER_NAME)), "a plain inspect eval writes no ledger"
    with usage_ledger.recording(tmp_path / "x" / usage_ledger.LEDGER_NAME):
        assert usage_ledger.UsageLedgerHook().enabled()
    assert usage_ledger.active_ledger() is None


def test_a_resumed_sessions_unlogged_usage_is_the_ledgers_and_never_counted_twice(worlds, ckpt_dir, tmp_path, monkeypatch):  # noqa: F811
    """B7 records a resumed session's earlier-attempt usage in f8_resume.unlogged (for analysis); the ledger holds the
    same calls, and spend counts them once, from the ledger."""
    for k in ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_FALLBACK", "APE_EMBEDDING_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(spend.REGISTRY_ENV, str(tmp_path / "registry.jsonl"))
    costs = tmp_path / "costs.yaml"
    table = yaml.safe_load(COSTS_PATH.read_text())
    table[MOCK] = {"input": 1.0, "output": 2.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
    costs.write_text(yaml.safe_dump(table))
    clear_model_info_cache()
    try:
        log_dir = tmp_path / "logs"
        ok, (header,) = run_evals(f8_session(level="12", split="dev", arm="O-state", checkpoints="5,10"), log_dir, model_override=MOCK,
                                  model_args={"custom_outputs": Flaky(mock_session_agent)}, costs_path=costs, roles=(), display="none", retry_wait=0.01)  # fmt: skip
        assert ok
        (s,) = read_eval_log(header.location).samples
        session_unlogged = s.store["f8_resume"]["unlogged"]["by_model"][MOCK]["total_cost"]
        u = unlogged_spend([Path(header.location.removeprefix("file://"))], usage_ledgers(log_dir))
        assert session_unlogged > 0 and u["unlogged_usd"] == pytest.approx(session_unlogged), "the same calls, by two records"
        p = program_spend(tmp_path / "registry.jsonl", costs)
        whole = s.store["f8_usage"]["by_model"][MOCK]["total_cost"]
        assert p["spent_usd"] == pytest.approx(whole), "the session's whole cost, once: the log's attempt plus the ledger's earlier one"
    finally:
        clear_model_info_cache()
    assert json.loads((log_dir / usage_ledger.LEDGER_NAME).read_text().splitlines()[0])["sample_uuid"] == s.uuid
