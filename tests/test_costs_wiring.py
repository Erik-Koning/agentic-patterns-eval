"""Price table everywhere, plus preflight (FX-2): Inspect logs carry $, the smoke cap sees them, and
an unpriced model or a missing key stops a run before it spends. Offline: mock models, fake embeddings."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelUsage, get_model, get_model_info
from inspect_ai.model._model_info import clear_model_info_cache
from inspect_ai.solver import generate

from ape.analysis.gate_stats import load_results
from ape.build import build
from ape.config import ROOT, Config
from ape.llm.ledger import Ledger, LedgerEntry
from ape.llm.mock_agent import mock_agent, mock_classifier
from ape.models import MODELS_PATH, PreflightError, eval_cost_kwargs, load_profile, preflight, require_preflight
from ape.tasks.gate import gate
from ape.tuning import _priced_cost

MOCK = "mockllm/model"
MOCK_PRICE = {"input": 1.0, "output": 2.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
MODEL_ENV = ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_FALLBACK", "APE_EMBEDDING_MODEL")


def _load_smoke():
    spec = importlib.util.spec_from_file_location("readiness_smoke", ROOT / "readiness" / "smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load_smoke()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Inspect keeps registered prices process-wide; start and end each test without them."""
    for k in MODEL_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    clear_model_info_cache()
    yield
    clear_model_info_cache()


@pytest.fixture
def mock_costs(tmp_path) -> Path:
    path = tmp_path / "costs.yaml"
    path.write_text(yaml.safe_dump({MOCK: MOCK_PRICE}))
    return path


def _tiny_log(tmp_path, **cost_kwargs):
    task = Task(dataset=[Sample(input="hi", target="x")], solver=generate())
    log = inspect_eval(task, model=get_model(MOCK, custom_outputs=mock_agent), log_dir=str(tmp_path / "logs"), display="none", **cost_kwargs)[0]
    assert log.status == "success", log.error
    return log


# ---------- Inspect pricing ----------


def test_mock_gate_eval_is_priced_for_agent_and_kg_role(tmp_path, mock_costs):
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log = inspect_eval(
        gate(family="F7", level="10", split="dev", arm="APGo-q"),
        model=get_model(MOCK, custom_outputs=mock_agent),
        model_roles={"kg": get_model(MOCK, custom_outputs=mock_classifier, memoize=False)},
        log_dir=str(tmp_path / "logs"),
        display="none",
        **eval_cost_kwargs(mock_costs),
    )[0]
    assert log.status == "success", log.error
    for s in log.samples:
        assert s.model_usage and all(u.total_cost is not None and u.total_cost > 0 for u in s.model_usage.values())
    kg = [s.role_usage["kg"] for s in log.samples if "kg" in (s.role_usage or {})]
    assert kg, "the kg role made no calls; the role-pricing check would be vacuous"
    assert all(u.total_cost is not None and u.total_cost > 0 for u in kg)
    df = load_results([log.location])  # require_cost=True
    assert (df["cost_usd"] > 0).all()


def test_without_a_price_the_log_has_no_cost_and_load_results_refuses_it(tmp_path):
    log = _tiny_log(tmp_path)
    assert all(u.total_cost is None for s in log.samples for u in s.model_usage.values())
    with pytest.raises(ValueError, match="no cost for"):
        load_results([log.location])


def test_the_real_price_file_applies_to_inspect():
    """Inspect's cost config raises on models outside its database (here the embedding model);
    eval_cost_kwargs registers them, so the whole table applies and every priced model is set."""
    kwargs = eval_cost_kwargs()
    assert kwargs == {"model_cost_config": str(ROOT / "config" / "model_costs.yaml")}
    task = Task(dataset=[Sample(input="hi", target="x")], solver=generate())
    log = inspect_eval(task, model=get_model(MOCK, custom_outputs=mock_agent), display="none", log_dir=str(Config().cache_dir / "logs"), **kwargs)[0]
    assert log.status == "success", log.error
    for name, price in yaml.safe_load((ROOT / "config" / "model_costs.yaml").read_text()).items():
        assert get_model_info(name).cost.input == price["input"], name


# ---------- preflight ----------


def _profile_file(tmp_path, agent="openai/gpt-6-luna", build_model="gpt-6-luna") -> Path:
    path = tmp_path / "models.yaml"
    roles = {
        "agent": {"model": agent},
        "kg": {"model": "openai/gpt-6-luna"},
        "build": {"model": build_model},
        "embeddings": {"model": "text-embedding-3-small"},
    }
    path.write_text(yaml.safe_dump({"profiles": {"p": roles}}))
    return path


def _costs_file(tmp_path, **extra) -> Path:
    path = tmp_path / "prices.yaml"
    price = {"input": 0.1, "output": 0.5, "input_cache_write": 0.1, "input_cache_read": 0.1}
    path.write_text(yaml.safe_dump({"openai/gpt-6-luna": price, "openai/text-embedding-3-small": price, **extra}))
    return path


def test_preflight_passes_a_fully_priced_profile(tmp_path):
    assert preflight(load_profile("p", _profile_file(tmp_path)), costs_path=_costs_file(tmp_path)) == []


def test_preflight_reports_an_unpriced_role_model(tmp_path):
    profile = load_profile("p", _profile_file(tmp_path, agent="openai/gpt-unpriced", build_model="gpt-unpriced-build"))
    problems = preflight(profile, costs_path=_costs_file(tmp_path))
    assert len(problems) == 2
    assert "role agent" in problems[0] and "'openai/gpt-unpriced'" in problems[0]
    assert "role build" in problems[1] and "'gpt-unpriced-build'" in problems[1]
    with pytest.raises(PreflightError, match="gpt-unpriced") as err:
        require_preflight(profile, costs_path=_costs_file(tmp_path))
    assert "gpt-unpriced-build" in str(err.value), "every problem is listed"


def test_preflight_keys_inspect_roles_by_provider_and_ledger_roles_by_bare_name(tmp_path):
    # Priced only under the bare name: fine for the build role, not for an Inspect role.
    price = {"input": 1, "output": 1, "input_cache_write": 1, "input_cache_read": 1}
    costs = _costs_file(tmp_path, **{"gpt-6-sol": price})
    assert preflight(load_profile("p", _profile_file(tmp_path, build_model="gpt-6-sol")), costs_path=costs) == []
    problems = preflight(load_profile("p", _profile_file(tmp_path, agent="openai/gpt-6-sol")), costs_path=costs)
    assert len(problems) == 1 and "'openai/gpt-6-sol'" in problems[0] and "provider/model" in problems[0]


def test_preflight_checks_overrides(tmp_path, monkeypatch):
    profile, costs = load_profile("p", _profile_file(tmp_path)), _costs_file(tmp_path)
    assert "openai/gpt-cli" in preflight(profile, overrides={"agent": "openai/gpt-cli", "kg": None}, costs_path=costs)[0]
    monkeypatch.setenv("APE_BUILD_MODEL", "gpt-env-build")
    assert "APE_BUILD_MODEL" in preflight(profile, costs_path=costs)[0]


def test_preflight_exempts_unpriced_mockllm(tmp_path):
    assert preflight(load_profile("p", _profile_file(tmp_path, agent=MOCK)), costs_path=_costs_file(tmp_path)) == []


def test_preflight_reports_a_bad_or_unparsable_price_file(tmp_path):
    profile = load_profile("p", _profile_file(tmp_path))
    bad = _costs_file(tmp_path, **{"openai/gpt-6-sol": {"input": 1.0, "output": "cheap"}})
    problems = preflight(profile, costs_path=bad)
    assert len(problems) == 1 and "'openai/gpt-6-sol'" in problems[0] and "output" in problems[0] and "input_cache_read" in problems[0]
    broken = tmp_path / "broken.yaml"
    broken.write_text("openai/gpt-6-luna: [unclosed\n")
    assert "does not parse" in preflight(profile, costs_path=broken)[0]
    assert "does not parse" in preflight(profile, costs_path=tmp_path / "missing.yaml")[0]


def test_preflight_live_requires_an_api_key(tmp_path, monkeypatch):
    profile, costs = load_profile("p", _profile_file(tmp_path)), _costs_file(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    assert preflight(profile, live=False, costs_path=costs, env_path=env_file) == []
    problems = preflight(profile, live=True, costs_path=costs, env_path=env_file)
    assert problems == [f"OPENAI_API_KEY is not set in the environment or in {env_file}"]
    env_file.write_text("OPENAI_API_KEY=sk-test-secret\n")
    assert preflight(profile, live=True, costs_path=costs, env_path=env_file) == []
    env_file.write_text("OTHER=1\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    assert preflight(profile, live=True, costs_path=costs, env_path=env_file) == []


REPO_PROFILES = sorted((yaml.safe_load(MODELS_PATH.read_text()) or {})["profiles"])


def test_repo_profiles_include_the_planned_ones():
    assert {"gate", "study_g_luna", "study_g_sol", "study_g_astra", "anchor", "anchor_luna"} <= set(REPO_PROFILES)


@pytest.mark.parametrize("name", REPO_PROFILES)
def test_every_repo_profile_is_priced(name):
    assert preflight(name, live=False) == []


# ---------- spend guards ----------


def test_smoke_spend_counts_inspect_cost_and_the_guard_stops_over_the_cap(tmp_path, mock_costs):
    log = _tiny_log(tmp_path, **eval_cost_kwargs(mock_costs))
    spend = smoke._spend([log])
    assert spend > 0
    report: dict = {}
    smoke._guard(report, [log], max_usd=1.0)
    assert report["spend_usd"] == spend
    with pytest.raises(SystemExit, match="exceeds --max-usd"):
        smoke._guard(report, [log], max_usd=spend / 2)


def test_smoke_spend_adds_the_ledger(tmp_path, mock_costs):
    log = _tiny_log(tmp_path, **eval_cost_kwargs(mock_costs))
    inspect_only = smoke._spend([log])
    Ledger(Config().ledger_path).append(LedgerEntry(role="build", model="gpt-6-luna", kind="chat", input_tokens=1_000_000))
    assert smoke._spend([log]) == pytest.approx(inspect_only + 0.10)


def test_smoke_spend_refuses_unpriced_calls_unless_dry(tmp_path, capsys):
    log = _tiny_log(tmp_path)  # no cost config: total_cost is None
    with pytest.raises(SystemExit, match="refusing to count it as \\$0"):
        smoke._spend([log])
    assert smoke._spend([log], dry=True) == 0.0
    assert "--dry" in capsys.readouterr().out


def test_smoke_spend_refuses_an_unpriced_ledger_entry():
    Ledger(Config().ledger_path).append(LedgerEntry(role="build", model="gpt-unpriced", kind="chat", input_tokens=10))
    with pytest.raises(SystemExit, match="gpt-unpriced"):
        smoke._spend([])


def test_tuning_cost_refuses_unpriced_non_mock_calls():
    def log(model, cost):
        return SimpleNamespace(samples=[SimpleNamespace(model_usage={model: ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2, total_cost=cost)})])

    assert _priced_cost(log("openai/gpt-6-luna", 0.25), "c") == 0.25
    assert _priced_cost(log(MOCK, None), "c") == 0.0
    with pytest.raises(RuntimeError, match="no Inspect cost"):
        _priced_cost(log("openai/gpt-6-luna", None), "c")
