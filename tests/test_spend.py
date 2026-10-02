"""Program-wide spend and failure safety (READINESS_AUDIT §3, item B): the spend registry, program spend across run ids
and smoke runs (retries and killed runs included), the crash-visible runner index, the per-sample cost guard, and
the per-task error circuit breaker. Offline: priced mock models, fake embeddings."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.model._model_info import clear_model_info_cache
from inspect_ai.solver import solver

from ape import budget, runner, spend
from ape.analysis.gate_stats import load_results
from ape.budget import BudgetError, program_spend, sample_cost_limit
from ape.build import build
from ape.config import ROOT
from ape.llm.ledger import Ledger, LedgerEntry
from ape.llm.mock_agent import mock_agent
from ape.models import COSTS_PATH
from ape.run_gate import GateRun, read_manifest, run_phases
from ape.runner import INDEX_NAME, run_evals, task_fail_on_error
from ape.tasks.gate import gate

MOCK = "mockllm/model"
CALL_USD = 2.0  # each mock call: 1M input tokens at $1/M + 100K output tokens at $10/M
FAST = {"display": "none", "retry_wait": 0.01}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fresh registry and price table with a priced mock model; worlds, cache and indices under tmp."""
    registry = tmp_path / "registry.jsonl"
    monkeypatch.setenv(spend.REGISTRY_ENV, str(registry))
    monkeypatch.delenv(spend.LABEL_ENV, raising=False)
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    table = yaml.safe_load(COSTS_PATH.read_text())
    table[MOCK] = {"input": 1.0, "output": 10.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
    costs = tmp_path / "costs.yaml"
    costs.write_text(yaml.safe_dump(table))
    clear_model_info_cache()
    yield {"tmp": tmp_path, "registry": registry, "costs": costs}
    clear_model_info_cache()


def priced_outputs(messages, tools, tool_choice, config) -> ModelOutput:
    out = ModelOutput.from_content(MOCK, "x")
    out.usage = ModelUsage(input_tokens=1_000_000, output_tokens=100_000, total_tokens=1_100_000)
    return out


ATTEMPTS = {"n": 0}


@solver
def flaky(fail_id: str | None = None, calls: int = 1):
    """`calls` generates per sample; sample `fail_id` raises on the first eval attempt only."""

    async def solve(state, generate):
        for _ in range(calls):
            state = await generate(state)
        if fail_id is not None and state.sample_id == fail_id and ATTEMPTS["n"] == 0:
            raise RuntimeError("first attempt fails")
        return state

    return solve


@task
def plain(n: int = 3, fail_id: str | None = None, calls: int = 1, tag: str = "a") -> Task:
    return Task(dataset=[Sample(id=f"q{i}", input=f"{tag} q{i}", target="x") for i in range(n)], solver=flaky(fail_id, calls))


def _run(env, log_dir: Path, tasks, **kw):
    """A priced mock run; the per-sample cost guard is far above a sample's $2 unless a test sets it."""
    kw = {**FAST, "sample_cost_usd": 100.0, **kw}
    return run_evals(tasks, log_dir, model_override=MOCK, model_args={"custom_outputs": priced_outputs}, costs_path=env["costs"], **kw)


def _sample_usd(log_dir: Path) -> float:
    """Ground truth: every distinct sample's cost over every log in the dir."""
    seen: dict[str, float] = {}
    for p in log_dir.rglob("*.eval"):
        for s in read_eval_log_sample_summaries(str(p)):
            seen[s.uuid] = sum(u.total_cost or 0.0 for u in s.model_usage.values())
    return sum(seen.values())


# ---------- the per-task circuit breaker ----------


def test_fail_on_error_is_a_proportion_for_large_tasks_and_a_count_for_small_ones():
    assert task_fail_on_error(48) == 3.0  # the gate's pilot cells: 2% would abort on the first persistent error
    assert task_fail_on_error(149) == 3.0
    assert task_fail_on_error(150) == 0.02  # 2% of 150 = 3: the two rules meet
    assert task_fail_on_error(576) == 0.02  # a gate test task: 16 worlds x 12 tasks x 3 epochs
    assert task_fail_on_error(10) == 3.0  # a Study G session task


def test_run_evals_sets_each_tasks_threshold_and_a_caller_value_wins(env, monkeypatch):
    seen = {}

    def fake_eval_set(tasks, **kwargs):
        seen["tasks"] = {t.name + str(len(t.dataset)): t.fail_on_error for t in tasks}
        seen["eval_level"] = kwargs.get("fail_on_error")
        return True, []

    monkeypatch.setattr(runner, "eval_set", fake_eval_set)
    small, large = plain(n=16), plain(n=192)
    _run(env, env["tmp"] / "a", [small, large], epochs=3)  # 48 and 576 planned samples
    assert seen["tasks"] == {"plain16": 3.0, "plain192": 0.02} and seen["eval_level"] is None
    _run(env, env["tmp"] / "b", [plain(n=16)], fail_on_error=0.5)
    assert seen["eval_level"] == 0.5 and seen["tasks"] == {"plain16": None}


# ---------- the registry, written before eval_set; the index while running and after a crash ----------


def test_registry_and_index_are_written_before_eval_set_and_completed_after_a_crash(env, monkeypatch):
    log_dir = env["tmp"] / "logs"
    monkeypatch.setenv(spend.LABEL_ENV, "gate/run-x")
    during = {}

    def killed_mid_run(tasks, **kwargs):
        during["registry"] = spend.read_registry(env["registry"])
        during["index"] = json.loads((log_dir / INDEX_NAME).read_text())
        raise KeyboardInterrupt("killed")

    monkeypatch.setattr(runner, "eval_set", killed_mid_run)
    with pytest.raises(KeyboardInterrupt):
        _run(env, log_dir, [plain(n=2)])
    (start,) = during["registry"]
    assert start["event"] == "start" and start["log_dir"] == str(log_dir.resolve()) and start["label"] == "gate/run-x" and start["study"] == "gate"
    (run,) = during["index"]["runs"]
    assert run["status"] == "running" and run["pid"] == os.getpid() and run["tasks_planned"][0]["planned"] == 2
    # After the exception: the same run entry is completed as failed; the registry records the finish.
    (run,) = json.loads((log_dir / INDEX_NAME).read_text())["runs"]
    assert run["status"] == "failed" and "killed" in run["error"]
    assert [e["event"] for e in spend.read_registry(env["registry"])] == ["start", "finish"]


def test_registry_isolation():
    saved = os.environ.pop(spend.REGISTRY_ENV)
    try:
        assert spend.registry_path(live=False) is None, "offline runs are not recorded unless a registry is named"
        with pytest.raises(spend.SpendRegistryError, match="test reached the real spend registry"):
            spend.registry_path(live=True)
    finally:
        os.environ[spend.REGISTRY_ENV] = saved
    assert not spend.DEFAULT_REGISTRY.exists() or all(str(ROOT / "tests") not in line for line in spend.DEFAULT_REGISTRY.read_text().splitlines())


# ---------- program spend: across run ids and smoke, retries counted once ----------


def test_program_spend_sums_run_ids_and_smoke_and_counts_retried_samples_once(env, monkeypatch, capsys):
    dirs = {}
    for label, n in (("gate/run-1", 2), ("gate/run-2", 3), ("smoke", 1)):
        monkeypatch.setenv(spend.LABEL_ENV, label)
        dirs[label] = env["tmp"] / label.replace("/", "-")
        ok, _ = _run(env, dirs[label], [plain(n=n, tag=label)])
        assert ok
    # A retried task: sample q1 fails its first attempt (fail_on_error=True forces the task retry). The retry's
    # log copies the completed samples (same uuids) and re-runs q1; the failed attempt's log is kept.
    monkeypatch.setenv(spend.LABEL_ENV, "gate/run-3")
    retried = env["tmp"] / "retried"
    ATTEMPTS["n"] = 0
    ok, _ = _run(env, retried, [plain(n=3, fail_id="q1", tag="retry")], fail_on_error=True, retry_attempts=1)
    assert not ok
    ATTEMPTS["n"] = 1  # the cause is fixed; the same call resumes
    ok, _ = _run(env, retried, [plain(n=3, fail_id="q1", tag="retry")], fail_on_error=True, retry_attempts=1)
    assert ok
    logs = list(retried.glob("*.eval"))
    assert len(logs) >= 2, "the failed attempt's log is kept (retry_cleanup=False)"
    uuids = {s.uuid for p in logs for s in read_eval_log_sample_summaries(str(p))}
    naive = sum(sum(u.total_cost or 0 for s in read_eval_log_sample_summaries(str(p)) for u in s.model_usage.values()) for p in logs)
    truth = _sample_usd(retried)
    assert truth == pytest.approx(len(uuids) * CALL_USD) and len(uuids) >= 4, "q0, q2 once; q1 once per attempt"
    assert naive > truth, "summing the logs would count the reused samples twice"

    s = program_spend(env["registry"], env["costs"])
    assert s["inspect_usd"] == pytest.approx((2 + 3 + 1) * CALL_USD + truth)
    assert s["by_study"] == pytest.approx({"gate": (2 + 3) * CALL_USD + truth, "smoke": CALL_USD})
    assert {r["label"] for r in s["dirs"]} == {"gate/run-1", "gate/run-2", "smoke", "gate/run-3"} and not s["unfinished_dirs"]
    # A registered ledger counts too (registered on its first append).
    monkeypatch.setenv(spend.LABEL_ENV, "gate/run-1")
    Ledger(env["tmp"] / "cache" / "ledger.jsonl").append(LedgerEntry(role="build", model="gpt-6-luna", kind="chat", input_tokens=1_000_000))
    s2 = program_spend(env["registry"], env["costs"])
    assert s2["ledger_usd"] == pytest.approx(0.10) and s2["spent_usd"] == pytest.approx(s["spent_usd"] + 0.10)
    # The CLI prints it by study and dir.
    assert budget.main(["spend", "--registry", str(env["registry"]), "--costs", str(env["costs"])]) == 0
    out = capsys.readouterr().out
    assert "gate/run-2" in out and "smoke" in out and f"${s2['spent_usd']:,.2f}" in out


CHILD = r"""
import asyncio, sys
from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelCost, ModelOutput, ModelUsage, get_model
from inspect_ai.model._model_info import ModelInfo, set_model_info
from inspect_ai.solver import solver
cost = ModelCost(input=1.0, output=10.0, input_cache_write=1.0, input_cache_read=1.0)
set_model_info("mockllm/model", ModelInfo(cost=cost))
def outputs(messages, tools, tool_choice, config):
    out = ModelOutput.from_content("mockllm/model", "x")
    out.usage = ModelUsage(input_tokens=1_000_000, output_tokens=100_000, total_tokens=1_100_000)
    return out
@solver
def slow():
    async def solve(state, generate):
        state = await generate(state)
        await asyncio.sleep(0.3)
        return state
    return solve
@task
def t():
    return Task(dataset=[Sample(input=f"q{i}", target="x", id=i) for i in range(60)], solver=slow())
eval_set(t(), log_dir=sys.argv[1], model=get_model("mockllm/model", custom_outputs=outputs), max_samples=2, log_buffer=2,
         display="none", model_cost_config={"mockllm/model": cost})
"""


def test_a_killed_runs_flushed_spend_is_counted(env):
    """A real started log: the child process is SIGKILLed mid-run. Its header has no usage (header-based
    accounting reads $0); its flushed sample summaries carry the spend, and program_spend counts them."""
    log_dir = env["tmp"] / "killed"
    log_dir.mkdir()
    spend.register_logs(log_dir, live=True, tasks=["t"], label_="gate/killed")  # as run_evals does before eval_set
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(log_dir)], env={**os.environ, "PYTHONWARNINGS": "ignore"})
    try:
        deadline, flushed = time.time() + 90, 0
        while time.time() < deadline and flushed < 4:
            time.sleep(0.3)
            logs = list(log_dir.glob("*.eval"))
            try:
                flushed = len(read_eval_log_sample_summaries(str(logs[0]))) if logs else 0
            except Exception:  # noqa: BLE001  (the zip may be mid-write)
                flushed = 0
    finally:
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
    assert flushed >= 4, "the child flushed samples before it was killed"
    (log,) = log_dir.glob("*.eval")
    header = read_eval_log(str(log), header_only=True)
    assert header.status == "started" and not header.stats.model_usage
    s = program_spend(env["registry"], env["costs"])
    n = len(read_eval_log_sample_summaries(str(log)))
    assert s["inspect_usd"] == pytest.approx(n * CALL_USD) and n >= 4
    assert s["partial_logs"] == 1 and s["unfinished_dirs"] == [str(log_dir.resolve())]


# ---------- the guard counts other run ids ----------


def test_the_gate_guard_refuses_when_another_run_ids_spend_leaves_too_little(env, monkeypatch):
    # Another run's priced logs, registered under its own label.
    monkeypatch.setenv(spend.LABEL_ENV, "gate/other-run")
    other = env["tmp"] / "other-run-logs"
    _run(env, other, [plain(n=3)])
    other_usd = 3 * CALL_USD
    monkeypatch.delenv(spend.LABEL_ENV)
    # This run is offline (its own registry under work/); that registry also lists the other run's dir, as the
    # program registry would for a live run.
    run = GateRun("guarded", offline=True, runs_root=env["tmp"] / "runs", budget_usd=other_usd + 0.40)
    assert run_phases(run, "preflight") == {"preflight": "done"}
    reg = run.work_dir / "spend_registry.jsonl"
    reg.parent.mkdir(parents=True, exist_ok=True)
    with reg.open("a") as f:
        for e in spend.read_registry(env["registry"]):
            f.write(json.dumps(e) + "\n")
    # (The other run's sample costs are in its logs, priced when it ran; the run's own table has no mock price.)
    with pytest.raises(BudgetError, match=r"phase build-dev: projected \$[\d.,]+ exceeds the remaining \$0.40"):
        run_phases(run, "build-dev")
    m = read_manifest(run, "build-dev")
    assert m["spend_at_start"]["spent_usd"] == pytest.approx(other_usd) and m["spend_at_start"]["run_spent_usd"] == 0
    assert m["spend_at_start"]["program"]["by_study"] == pytest.approx({"gate": other_usd})


# ---------- the per-sample runaway guard ----------


def test_cost_limit_rule():
    plan = budget.load_plan()
    assert sample_cost_limit(None, plan) == 2.0  # no projection: the default
    assert sample_cost_limit(0.0056, plan) == 0.5  # a gate test sample (~$0.0056): 20x is under the floor
    assert sample_cost_limit(0.2, plan) == pytest.approx(4.0)


def test_a_sample_over_its_cost_limit_ends_and_is_flagged_as_a_cap_hit(env):
    """Priced mock, gate task: the first call already exceeds the limit (floor $0.50), so every sample ends with
    limit "cost"; load_results flags it and PC5 counts it as a cap hit. The limit is pinned in the index."""
    from ape import analyze_gate as ag

    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log_dir = env["tmp"] / "limited"

    def expensive_agent(messages, tools, tool_choice, config):
        out = mock_agent(messages, tools, tool_choice, config)
        out.usage = ModelUsage(input_tokens=1_000_000, output_tokens=100_000, total_tokens=1_100_000)
        return out

    ok, (header,) = run_evals(
        gate(family="F7", level="10", split="dev", arm="S1"), log_dir, model_override=MOCK, model_args={"custom_outputs": expensive_agent},
        costs_path=env["costs"], sample_cost_usd=0.001, **FAST,
    )
    assert ok
    log = read_eval_log(header.location)
    assert log.eval.config.cost_limit == 0.5
    assert json.loads((log_dir / INDEX_NAME).read_text())["cost_limit_usd"] == 0.5
    df = load_results([header.location], require_cost=True)
    assert list(df["limit_hit"]) == ["cost", "cost"] and df["cap_hit"].all()
    pc5 = ag.pc5(df.assign(plan_cell="gate.test.f7", label="S1"))
    assert not pc5["pass"] and pc5["value"]["max_cap_hit_rate"] == 1.0
    # A resume with a different projection keeps the pinned limit (it is part of Inspect's task identity).
    ok, (again,) = run_evals(
        gate(family="F7", level="10", split="dev", arm="S1"), log_dir, model_override=MOCK, model_args={"custom_outputs": expensive_agent},
        costs_path=env["costs"], sample_cost_usd=10.0, **FAST,
    )
    assert again.eval.eval_id == header.eval.eval_id, "the finished task was reused, not re-run"


def test_an_unpriced_run_gets_no_cost_limit(env, monkeypatch):
    """Inspect refuses a cost_limit when a model has no price (offline mockllm), so the runner sets none."""
    seen = {}
    monkeypatch.setattr(runner, "eval_set", lambda tasks, **kw: (seen.update({t.name: t.cost_limit for t in tasks}), (True, []))[1])
    run_evals([plain(n=2)], env["tmp"] / "u", model_override=MOCK, model_args={"custom_outputs": priced_outputs}, **FAST)
    assert seen == {"plain": None}
