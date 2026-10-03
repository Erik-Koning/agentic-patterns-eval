"""The per-run cache nonce (BUILD_PLAN B12; brief §6.2, §9; `ape.agent.cache_nonce`): without a seed every prompt is
byte-identical to a build without it (the gate never sets one); with one, every agent, management and probe call of a
main-study or Study G task opens with the task's nonce line, every sample records it, and the study runner gives each
eval set its own seed. Offline: mock models, fake embeddings."""

import asyncio
import os

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape import run_gate, run_study
from ape.agent import cache_nonce
from ape.agent.cache_nonce import NONCE_ENV
from ape.agent.kb_react import system_prompt
from ape.apg.author import author_world
from ape.build import build
from ape.config import Config
from ape.llm.fake import perfect_author
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.llm.mock_multi import GoldMulti
from ape.llm.mock_session import gold_session_agent
from ape.tasks.gate import gate
from ape.tasks.main import main_study
from ape.tasks.study_g import f8_session
from ape.worlds import gen_f8
from ape.worlds.spec import World

M = "mockllm/model"
SEED = "main/r1@2026-10-03T00:00:00+00:00/test/main.A.f7/selected-abc"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nonce")
    mp = pytest.MonkeyPatch()
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        mp.delenv(k)
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    ids = {}
    for fam, level in (("F7", "10"), ("F8", "6")):
        ids[fam] = asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))[0]
    asyncio.run(author_world(World.load(Config().world_path(ids["F7"])), Config(), perfect_author))
    yield {"tmp": tmp, "ids": ids}
    mp.undo()


def _built(seed: str | None, make):
    """A task created under APE_CACHE_NONCE=`seed` (unset for None); the environment is restored before it runs, so the
    task must carry its nonce itself."""
    old = os.environ.pop(NONCE_ENV, None)
    if seed is not None:
        os.environ[NONCE_ENV] = seed
    try:
        return make()
    finally:
        os.environ.pop(NONCE_ENV, None)
        if old is not None:
            os.environ[NONCE_ENV] = old


def _eval(env, task, agent, kg=None, **kw):
    roles = {"kg": get_model(M, custom_outputs=kg, memoize=False)} if kg is not None else None
    from inspect_ai.log import read_eval_log

    log = inspect_eval(task, model=get_model(M, custom_outputs=agent, memoize=False), model_roles=roles, log_dir=str(env["tmp"] / "logs"), display="none", **kw)[0]
    assert log.status == "success", log.error
    log = read_eval_log(log.location, resolve_attachments=True)  # long prompts are stored as attachments
    assert not any(s.error for s in log.samples), [s.error for s in log.samples]
    return log


def _calls(sample, roles=(None, "cm", "probe")):
    return [e for e in sample.events if e.event == "model" and e.role in roles]


# --- the helpers -------------------------------------------------------------------------------------------------------


def test_without_a_seed_nothing_changes_and_with_one_each_arm_gets_its_own_short_nonce(monkeypatch):
    monkeypatch.delenv(NONCE_ENV, raising=False)
    text = "You are an operations assistant."
    assert cache_nonce.with_nonce(text, None) == text and cache_nonce.task_nonce("S1") is None and cache_nonce.task_metadata(None) == {}
    monkeypatch.setenv(NONCE_ENV, "  ")  # blank is unset
    assert cache_nonce.task_nonce("S1") is None
    monkeypatch.setenv(NONCE_ENV, SEED)
    s1, m1 = cache_nonce.task_nonce("S1", "push", "retrieved"), cache_nonce.task_nonce("M1", "push", "retrieved")
    assert s1 == cache_nonce.derive(SEED, "S1", "push", "retrieved") and len(s1) == cache_nonce.LENGTH and int(s1, 16) >= 0
    assert s1 != m1 and s1 != cache_nonce.task_nonce("S1", "pull", "retrieved") and s1 != cache_nonce.derive(SEED + "x", "S1", "push", "retrieved")
    assert cache_nonce.task_nonce("S1", "push", "retrieved") == s1, "one arm's tasks share its nonce (caching within the arm is kept)"
    assert cache_nonce.with_nonce(text, s1) == f"Run reference: {s1}\n\n{text}" and cache_nonce.starts_with_nonce(cache_nonce.with_nonce(text, s1), s1)
    assert cache_nonce.task_metadata(s1) == {"cache_nonce": s1, "cache_nonce_seed": SEED}


# --- the gate and unset seeds: byte-identical ----------------------------------------------------------------------------


def test_the_gate_task_never_carries_a_nonce_even_with_a_seed_set(env):
    """The gate's task builder never reads the seed, so the gate's prompts and records are unchanged whatever the
    environment holds."""
    task = _built(SEED, lambda: gate(family="F7", level="10", split="dev", arm="S1"))
    assert "cache_nonce" not in (task.metadata or {}) and not any("cache_nonce" in (s.metadata or {}) for s in task.dataset)
    log = _eval(env, task, mock_agent, mock_kg, limit=1)
    s = log.samples[0]
    assert "cache_nonce" not in s.metadata
    system = s.messages[0].text
    assert system.startswith("You are an operations assistant") and system == system_prompt(True, False, False, system.split("## Knowledge base\n", 1)[1])


def test_without_a_seed_study_tasks_are_unchanged(env):
    main = _built(None, lambda: main_study(family="F7", level="10", split="dev", arm="M1"))
    assert "cache_nonce" not in main.metadata
    log = _eval(env, main, GoldMulti(Config().worlds_dir), GoldMulti(Config().worlds_dir).kg, limit=1)
    assert all(not (e.input[0].text or "").startswith("Run reference") for e in _calls(log.samples[0]))
    world = World.load(Config().world_path(env["ids"]["F8"]))
    sess = _built(None, lambda: f8_session(level="6", split="dev", arm="CM0", limit_worlds=1, checkpoints="6"))
    log = _eval(env, sess, gold_session_agent(world))
    assert log.samples[0].messages[0].text == gen_f8.system_prompt(world) and "cache_nonce" not in log.samples[0].metadata


# --- with a seed: every agent prompt -------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["S1", "M1", "M1k", "M7", "S8k3"])
def test_every_agent_prompt_of_a_main_study_sample_opens_with_its_nonce(env, arm):
    task = _built(SEED, lambda: main_study(family="F7", level="10", split="dev", arm=arm))
    nonce = cache_nonce.derive(SEED, arm, "push", "retrieved")
    assert task.metadata["cache_nonce"] == nonce and task.metadata["cache_nonce_seed"] == SEED
    gold = GoldMulti(Config().worlds_dir)
    log = _eval(env, task, gold, gold.kg, limit=2)
    for s in log.samples:
        assert s.metadata["cache_nonce"] == nonce
        agent_calls = _calls(s)
        assert agent_calls and all(e.input[0].role == "system" and e.input[0].text.startswith(f"Run reference: {nonce}\n\n") for e in agent_calls)
        assert all(not (e.input[0].text or "").startswith("Run reference") for e in _calls(s, ("kg",))), "kg calls are not prefixed"
        assert s.messages[0].text.startswith(f"Run reference: {nonce}\n\nYou are an operations assistant")
    if arm == "M1k":
        assert any(_calls(s, ("kg",)) for s in log.samples)


@pytest.mark.parametrize("arm", ["CM-sum", "CM-todo", "M1"])
def test_every_agent_management_and_probe_call_of_a_session_opens_with_its_nonce(env, arm, tmp_path, monkeypatch):
    monkeypatch.setenv("APE_SESSION_CHECKPOINTS", str(tmp_path / "ckpt"))
    world = World.load(Config().world_path(env["ids"]["F8"]))
    task = _built(SEED, lambda: f8_session(level="6", split="dev", arm=arm, limit_worlds=1, window=16000, threshold=4000, checkpoints="6"))
    nonce = cache_nonce.derive(SEED, arm)
    log = _eval(env, task, gold_session_agent(world))
    s = log.samples[0]
    assert s.metadata["cache_nonce"] == nonce and s.messages[0].text.startswith(f"Run reference: {nonce}\n\n{gen_f8.system_prompt(world)}")
    calls = _calls(s)
    roles = {e.role for e in calls}
    assert {None, "probe"} <= roles and all(e.input[0].role == "system" and e.input[0].text.startswith(f"Run reference: {nonce}") for e in calls)
    if arm in ("CM-sum", "CM-todo"):
        cm = _calls(s, ("cm",))
        assert cm and all(e.input[0].text == f"Run reference: {nonce}" for e in cm), "a management call gets the line as its own system message"
        assert all(v["view_tokens"] >= 4 for v in s.store["f8_views"] if v["kind"] == "cm")
    if arm == "M1":
        workers = [e for e in calls if "report" in {t.name for t in e.tools}]
        assert workers and all(e.input[0].text == f"Run reference: {nonce}\n\n{gen_f8.system_prompt(world)}" for e in workers)


# --- the study runner: one seed per eval set -----------------------------------------------------------------------------


def test_the_study_runner_gives_each_eval_set_its_own_seed_and_clears_the_shells(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        monkeypatch.delenv(k)
    run = run_study.StudyRun("main", "n1", offline=True, runs_root=tmp_path / "runs")
    run_study._check_mode(run)  # run.json, with the run's creation time
    created = run_gate.read_run_info(run)["created"]
    groups = [g for g in run_study.run_groups(run, "test") if g["arms"]][:2]
    assert len(groups) == 2
    seen: list[str | None] = []
    monkeypatch.setattr(run_study, "group_tasks", lambda r, g: seen.append(os.environ.get(NONCE_ENV)) or [])
    monkeypatch.setattr(run_study, "group_models", lambda r, profile, kind: (None, {}))
    monkeypatch.setattr(run_gate, "run_gate_tasks", lambda *a, **k: [])
    monkeypatch.setenv(NONCE_ENV, "from-the-shell")
    assert NONCE_ENV not in run_study.env_knobs(run), "the runner manages it: never a knob in a fingerprint"
    with run_study.run_environment(run):
        assert NONCE_ENV not in os.environ, "a shell value never applies"
        for g in groups:
            run_study.run_group(run, g, run.phase_dir("test"), "test")
        assert NONCE_ENV not in os.environ
    assert os.environ[NONCE_ENV] == "from-the-shell"
    assert seen == [f"main/n1@{created}/test/{g['dir']}" for g in groups] and len(set(seen)) == 2
    other = run_study.StudyRun("main", "n2", offline=True, runs_root=tmp_path / "runs")
    assert run_study.cache_nonce_seed(other, "test", groups[0]["dir"]) != seen[0], "another run, another seed"
