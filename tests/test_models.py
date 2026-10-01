"""Model profiles (FX-1, D-015): every role's reasoning effort reaches its calls and the logs, offline."""

import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from lightrag.prompt import PROMPTS

from ape.apg.author import build_llm_author
from ape.build import build
from ape.config import Config
from ape.lgr.build import build_index
from ape.llm.build_client import BuildLlm
from ape.llm.ledger import Ledger
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.models import agent_model, build_settings, embedding_model, load_profile, role_models
from ape.tasks.gate import gate
from ape.worlds.spec import World

MOCK = "mockllm/model"
BUILD_ENV = ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_EFFORT", "APE_BUILD_FALLBACK")


@pytest.fixture(autouse=True)
def clean_model_env(monkeypatch):
    for k in BUILD_ENV:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def recording(fn, seen: list, label: str):
    """mockllm `custom_outputs` that records the resolved config each call actually received."""

    def out(messages, tools, tool_choice, config):
        schema = config.response_schema.name if config.response_schema is not None else None
        seen.append((label, config.reasoning_effort, schema))
        return fn(messages, tools, tool_choice, config)

    return out


class FakeOpenAI:
    """Mimics `AsyncOpenAI().chat.completions.create` and records its kwargs."""

    def __init__(self, content: str):
        self.calls: list[dict] = []
        self.content = content
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


# ---- Inspect roles: agent high, kg low, through the strict-schema calls ----


@pytest.mark.parametrize("arm,schema", [("S5o", "classify"), ("LGRo-s", "keywords")])
def test_agent_and_kg_calls_carry_their_profile_efforts(offline_env, arm, schema):
    (wid,) = asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    if arm.startswith("LGR"):
        asyncio.run(build_index(World.load(Config().world_path(wid)), "oracle", Config()))
    profile = load_profile("gate")
    seen: list = []
    log = inspect_eval(
        gate(family="F7", level="10", split="dev", arm=arm),
        model=agent_model(profile, model=MOCK, custom_outputs=recording(mock_agent, seen, "agent")),
        model_roles=role_models(profile, ("kg",), model=MOCK, custom_outputs=recording(mock_kg, seen, "kg")),
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error

    agent = [c for c in seen if c[0] == "agent"]
    kg = [c for c in seen if c[0] == "kg"]
    assert agent and {effort for _, effort, _ in agent} == {"high"}
    # Every kg call is a strict-schema call (APG classify / LightRAG keywords) and still runs at low.
    assert kg and set(kg) == {("kg", "low", schema)}

    # Recorded natively in the log: the agent's config, each role's config, and each call's config.
    saved = read_eval_log(log.location)
    assert saved.eval.model_generate_config.reasoning_effort == "high"
    assert saved.eval.model_roles["kg"].config.reasoning_effort == "low"
    events = [e for s in saved.samples for e in s.events if e.event == "model"]
    assert {e.config.reasoning_effort for e in events if e.role == "kg"} == {"low"}
    assert {e.config.reasoning_effort for e in events if e.role != "kg"} == {"high"}


def test_judge_role_is_built_with_its_effort():
    (judge,) = role_models(load_profile("gate"), ("judge",), model=MOCK).values()
    assert judge.config.reasoning_effort == "low"


# ---- build role (OpenAI SDK, outside Inspect) ----


def test_build_llm_sends_reasoning_effort_and_records_it(tmp_path):
    client = FakeOpenAI('{"units": []}')
    ledger = Ledger(tmp_path / "l.jsonl")
    llm = BuildLlm("gpt-6-luna", ledger, {"world": "w"}, client=client, reasoning_effort="high")
    asyncio.run(llm.json("sys", "user", "units", {"type": "object"}))
    asyncio.run(llm.lightrag_func()("prompt", system_prompt="sys"))
    assert [c["reasoning_effort"] for c in client.calls] == ["high", "high"]
    assert "response_format" in client.calls[0]
    assert {e.context["reasoning_effort"] for e in ledger.read()} == {"high"}


def test_build_llm_omits_reasoning_effort_when_unset(tmp_path):
    client = FakeOpenAI("{}")
    ledger = Ledger(tmp_path / "l.jsonl")
    asyncio.run(BuildLlm("gpt-4o-mini", ledger, {"world": "w"}, client=client).chat([{"role": "user", "content": "x"}]))
    assert "reasoning_effort" not in client.calls[0]
    assert "reasoning_effort" not in ledger.read()[0].context


def test_apg_authoring_uses_the_profile_build_settings(tmp_path, monkeypatch):
    client = FakeOpenAI('{"units": []}')
    monkeypatch.setattr(BuildLlm, "_client_", lambda self: client)
    model, effort = build_settings(load_profile("gate"))
    author = build_llm_author(model, Ledger(tmp_path / "l.jsonl"), "w", effort)
    asyncio.run(author("sys", "user"))
    assert (client.calls[0]["model"], client.calls[0]["reasoning_effort"]) == ("gpt-6-luna", "high")


def test_lightrag_extraction_uses_the_profile_build_settings(offline_env, monkeypatch):
    client = FakeOpenAI(PROMPTS["DEFAULT_COMPLETION_DELIMITER"])
    monkeypatch.setattr(BuildLlm, "_client_", lambda self: client)
    (wid,) = asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=1, relational=True, embed=True))
    manifest = asyncio.run(build_index(World.load(Config().world_path(wid)), "extract", Config()))
    assert client.calls and {(c["model"], c["reasoning_effort"]) for c in client.calls} == {("gpt-6-luna", "high")}
    assert (manifest["build_model"], manifest["build_effort"]) == ("gpt-6-luna", "high")


def test_build_settings_precedence(monkeypatch):
    p = load_profile("gate")
    assert build_settings(p) == ("gpt-6-luna", "high")
    assert build_settings(p, fallback=True) == ("gpt-6-sol", "medium")
    monkeypatch.setenv("APE_BUILD_FALLBACK", "1")
    assert build_settings(p) == ("gpt-6-sol", "medium")
    monkeypatch.delenv("APE_BUILD_FALLBACK")
    # APE_BUILD_MODEL keeps working; a model outside the profile gets no effort unless asked.
    monkeypatch.setenv("APE_BUILD_MODEL", "gpt-4o-mini")
    assert build_settings(p) == ("gpt-4o-mini", None)
    monkeypatch.setenv("APE_BUILD_EFFORT", "low")
    assert build_settings(p) == ("gpt-4o-mini", "low")
    monkeypatch.delenv("APE_BUILD_EFFORT")
    monkeypatch.setenv("APE_BUILD_MODEL", "gpt-6-sol")
    assert build_settings(p) == ("gpt-6-sol", "medium")


# ---- profile loading ----


def test_profiles_load_and_env_selects_one(monkeypatch):
    gate_p = load_profile()
    assert gate_p.name == "gate" and embedding_model(gate_p) == "text-embedding-3-small"
    assert {r: s.reasoning_effort for r, s in gate_p.roles.items() if r in ("agent", "kg", "judge", "build")} == {
        "agent": "high",
        "kg": "low",
        "judge": "low",
        "build": "high",
    }
    monkeypatch.setenv("APE_MODEL_PROFILE", "study_g_astra")
    astra = load_profile()
    assert astra.name == "study_g_astra" and astra.role("agent").model == "openai/gpt-6-astra"
    assert astra.role("kg") == gate_p.role("kg")


def test_unknown_profile_or_role_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="unknown model profile 'nope'.*known: .*gate"):
        load_profile("nope")
    p = load_profile("gate")
    with pytest.raises(ValueError, match="has no role 'summarizer'"):
        p.role("summarizer")
    with pytest.raises(ValueError, match="not an Inspect role"):
        role_models(p, ("build",))

    def write(body: str):
        path = tmp_path / "models.yaml"
        path.write_text(body)
        return path

    base = "agent: {model: m}\n    kg: {model: m}\n    build: {model: m}\n    embeddings: {model: e}\n"
    with pytest.raises(ValueError, match=r"unknown role\(s\) \['planner'\]"):
        load_profile("x", write(f"profiles:\n  x:\n    {base}    planner: {{model: m}}\n"))
    with pytest.raises(ValueError, match=r"missing role\(s\) \['embeddings'\]"):
        load_profile("x", write("profiles:\n  x:\n    agent: {model: m}\n    kg: {model: m}\n    build: {model: m}\n"))
    with pytest.raises(ValueError, match="reasoning_effort 'extreme'"):
        load_profile("x", write(f"profiles:\n  x:\n    {base}    judge: {{model: m, reasoning_effort: extreme}}\n"))
    with pytest.raises(ValueError, match="needs a `model`"):
        load_profile("x", write(f"profiles:\n  x:\n    {base}    judge: {{reasoning_effort: low}}\n"))
    p = load_profile("x", write(f"profiles:\n  x:\n    {base}    judge: {{model: {MOCK}, reasoning_effort: low, max_tokens: 64}}\n"))
    assert role_models(p, ("judge",))["judge"].config.max_tokens == 64


def test_sampling_parameters_only_where_the_model_accepts_them():
    from ape.models import load_profile

    for name in ("gate", "study_g_luna", "study_g_sol", "study_g_astra", "anchor_luna"):
        for role, spec in load_profile(name).roles.items():
            assert (spec.temperature, spec.top_p, spec.seed) == (None, None, None), f"{name}.{role} sends sampling params"
    judge = load_profile("anchor").role("judge").generate_config()
    assert (judge.temperature, judge.top_p, judge.seed) == (0.0, 1.0, 42), "official GraphRAG-Bench judge settings"
    assert load_profile("anchor").role("agent").generate_config().temperature == 0.7
