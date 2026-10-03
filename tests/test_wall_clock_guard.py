"""The runaway wall-clock guard of the main study and Study G (BUILD_PLAN B12, brief §4.5): Inspect's per-sample
`working_limit` from run_plan.yaml `budget.sample_working_limit`, scaled with a task's turn cap or a session's cases,
applied by the study runner and never by the gate (Inspect's task identity includes it). A sample that hits it ends with
limit "working", a limit hit like the cost guard's. Offline: mock models, fake embeddings."""

import asyncio
import os

import anyio
import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape import run_gate, run_study
from ape.budget import WORKING_LIMIT_DEFAULTS, Plan, load_plan, sample_working_limit
from ape.build import build
from ape.config import ROOT
from ape.freeze_scope import config_slice
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.runner import read_index, run_evals, working_guard
from ape.tasks.gate import gate
from ape.tasks.main import main_study
from ape.tasks.study_g import f8_session

M = "mockllm/model"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("guard")
    mp = pytest.MonkeyPatch()
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        mp.delenv(k)
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for fam, level in (("F7", "10"), ("F1", "32"), ("F8", "6")):
        asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    yield tmp
    mp.undo()


def test_the_guard_is_generous_and_scales_with_the_turn_cap_and_the_sessions_cases():
    plan = load_plan()
    cfg = plan.budget["sample_working_limit"]
    assert cfg == WORKING_LIMIT_DEFAULTS, "run_plan.yaml states the defaults the code falls back to"
    assert sample_working_limit("F7", "10", 12, plan) == 1800 and sample_working_limit("F3", "60", None, plan) == 1800
    assert sample_working_limit("F1", "32", 44, plan) == 6600 and sample_working_limit("F2", "10", 28, plan) == 4200
    assert sample_working_limit("F8", "40", None, plan) == 1800 + 40 * 600 and sample_working_limit("F8", "10", None, plan) == 7800
    other = Plan((), {"sample_working_limit": {"agent_s": 60, "session_per_case_s": 10}}, {}, ())
    assert sample_working_limit("F7", "10", 12, other) == 60 and sample_working_limit("F8", "6", None, other) == 1800 + 60


def test_study_tasks_get_it_the_gates_never_do(env, tmp_path):
    run = run_study.StudyRun("main", "w1", offline=True, runs_root=tmp_path / "runs")
    t = run_study.agent_task(run, family="F1", level="32", split="dev", arm="S1", limit_worlds=1, seed_base=None, cap=5000)
    assert t.working_limit == 6600 and t.token_limit == 5000, "F1-32's 44-turn cap"
    g = run_study.StudyRun("study_g", "w1", offline=True, runs_root=tmp_path / "runs")
    s = run_study.session_task(g, level="6", split="dev", arm="CM0", limit_worlds=1, variant="", seed_base=None)
    assert s.working_limit == 1800 + 6 * 600
    own = main_study(family="F7", level="10", split="dev", arm="S1")
    own.working_limit = 99
    assert working_guard(own).working_limit == 99, "a task's own limit stands"
    # The gate's tasks, and the runner on its own, never set one: the gate's task identities are unchanged.
    assert gate(family="F7", level="10", split="dev", arm="S1").working_limit is None
    log_dir = tmp_path / "gate"
    gp = run_gate.load_profile("gate")
    agent = get_model(M, custom_outputs=mock_agent, memoize=False)
    paths = run_gate.run_gate_tasks(run_gate.GateRun("g", offline=True, runs_root=tmp_path / "gruns"), gp, (agent, {"kg": get_model(M, custom_outputs=mock_kg, memoize=False)}), [gate(family="F7", level="10", split="dev", arm="S1", limit_worlds=1)], log_dir, 1, "gate")
    from inspect_ai.log import read_eval_log

    assert read_eval_log(paths[0], header_only=True).eval.config.working_limit is None
    assert [t["working_limit"] for r in read_index(log_dir)["runs"] for t in r["tasks_planned"]] == [None]


class SlowAgent:
    """A runaway in miniature: 1.2 s per call, and it never answers (a lookup every turn, up to the 12-turn cap)."""

    async def __call__(self, messages, tools, tool_choice, config):
        from inspect_ai.model import ModelOutput

        await anyio.sleep(1.2)
        return ModelOutput.for_tool_call(M, "lookup_customer", {"customer_id": "CU-0"})


def test_a_sample_that_runs_past_it_ends_with_a_working_limit_hit_counted_as_a_cap_hit(env, tmp_path):
    from inspect_ai.log import read_eval_log

    from ape.analysis.main_load import load_main

    task = main_study(family="F7", level="10", split="dev", arm="S1", limit_worlds=1)
    task.working_limit = 2  # the guard in miniature: two slow calls pass it
    _, logs = run_evals(task, tmp_path / "logs", profile="gate", model=get_model(M, custom_outputs=SlowAgent(), memoize=False),
                        model_roles={"kg": get_model(M, custom_outputs=mock_kg, memoize=False)}, display="none", limit=1)  # fmt: skip
    log = read_eval_log(logs[0].location)
    s = log.samples[0]
    assert s.error is None and s.limit is not None and s.limit.type == "working"
    assert s.store.get("turns_used"), "the loop's records are kept"
    row = load_main([logs[0].location], require_cost=False).iloc[0]
    assert row["limit_hit"] == "working" and bool(row["cap_hit"]) is True, "a limit hit like the cost guard's"
    assert [t["working_limit"] for r in read_index(tmp_path / "logs")["runs"] for t in r["tasks_planned"]] == [2]


def test_a_hung_model_call_is_cut_at_the_limit_too(tmp_path):
    """Inspect 0.3.273's working-time monitor interrupts a call still in flight (a hung request), not only loops."""
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import generate

    async def hung(messages, tools, tool_choice, config):
        await anyio.sleep(30)
        return ModelOutput.from_content(M, "late")

    log = inspect_eval(Task(dataset=[Sample(input="hi", id="a")], solver=generate(), working_limit=2), model=get_model(M, custom_outputs=hung, memoize=False),
                       display="none", log_dir=str(tmp_path))[0]  # fmt: skip
    assert log.samples[0].limit is not None and log.samples[0].limit.type == "working" and log.samples[0].error is None


def test_only_the_studies_freeze_the_guards_setting():
    path = ROOT / "config" / "run_plan.yaml"
    raw = yaml.safe_load(path.read_text())
    assert "sample_working_limit" not in config_slice(path, "gate")["budget"], "a guard change never touches the gate's slice"
    for study in ("main", "study_g"):
        assert config_slice(path, study)["budget"]["sample_working_limit"] == raw["budget"]["sample_working_limit"]


def test_a_session_task_gets_a_guard_for_its_length(env):
    t = working_guard(f8_session(level="6", split="dev", arm="CM0", limit_worlds=1))
    assert t.working_limit == WORKING_LIMIT_DEFAULTS["session_base_s"] + 6 * WORKING_LIMIT_DEFAULTS["session_per_case_s"]
