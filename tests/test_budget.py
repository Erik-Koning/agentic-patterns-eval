"""Cost model and right-sized run plan (FIX_PLAN FX-5). Offline: mock models, fake embeddings, temp price files."""

import asyncio
from pathlib import Path

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model._model_info import clear_model_info_cache

from ape.budget import (
    ASSUMPTIONS_PATH,
    PLAN_PATH,
    BudgetError,
    Prices,
    apply_cut,
    calibrate,
    estimate,
    load_assumptions,
    load_measured,
    load_plan,
    main,
    remaining,
    require_affordable,
    right_size,
)
from ape.build import build
from ape.config import ROOT, Config
from ape.llm.ledger import Ledger, LedgerEntry
from ape.llm.mock_agent import mock_agent, mock_classifier
from ape.models import agent_model, eval_cost_kwargs, load_profile, role_models
from ape.tasks.gate import gate

MOCK = "mockllm/model"
MOCK_PRICE = {"input": 1.0, "output": 2.0, "input_cache_write": 1.0, "input_cache_read": 1.0}

# The right-sized plan's conservative $ per study, as documented in BUDGET.md and DECISIONS.md D-021
# (priors only, no measured calibration). Update the docs with these when the plan or the priors change.
DOCUMENTED = {"gate": 214, "main": 1025, "study_g": 3486, "total": 4725, "expected": 3168}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    monkeypatch.delenv("APE_MODEL_PROFILE", raising=False)
    clear_model_info_cache()
    yield
    clear_model_info_cache()


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


# ---------- hand-computed cells ----------


def _hand_assumptions() -> dict:
    a = load_assumptions()
    a["output_tokens_per_call"] = {"high": 1000, "default": 500}
    a["calls_per_sample"] = {"F7": {"push": 2, "pull": 3}}
    a["base_input_per_call"] = 500
    a["history_per_prior_call"] = 100
    a["pull_context_factor"] = 2.0
    a["arms"] = {"APG-q": {"context": 200, "compile": "per_query", "kg": "apg_classify", "embed": True, "cache": "per_query"}}
    a["kg_calls"] = {"apg_classify": {"input": 300, "output": 100}}
    a["query_embedding_tokens"] = 50
    a["study_g"] = {
        "calls_per_item": 2,
        "view": {"full": 0.6, "managed": 0.5, "oracle_tokens": 4000},
        "probe_checkpoints": [5, 15],
        "arms": {"CM-sum": {"view": "managed", "overhead": 0.1, "cache": "session_managed"}},
    }
    a["cache_scenarios"] = {
        "conservative": {"cached_price": "input"},
        "expected": {"cached_price": "ratio", "cached_price_ratio": 0.1, "cached_fraction": {"per_query": 0.5, "session_managed": 0.25}},
    }
    return a


def test_two_small_cells_reproduce_hand_computed_costs(tmp_path):
    costs = _write(tmp_path / "costs.yaml", {
        "openai/agent-x": {"input": 2.0, "output": 10.0, "input_cache_write": 2.0, "input_cache_read": 2.0},
        "openai/kg-x": {"input": 1.0, "output": 4.0, "input_cache_write": 1.0, "input_cache_read": 1.0},
        "emb-x": {"input": 0.5, "output": 0.0, "input_cache_write": 0.5, "input_cache_read": 0.5},
    })  # fmt: skip
    models = {"agent": "openai/agent-x", "kg": "openai/kg-x", "embeddings": "emb-x"}
    plan = _write(tmp_path / "plan.yaml", {
        "budget": {"total_usd": 10},
        "study_g": {"window": 10000},
        "studies": {"s": {"profile": "gate", "phases": {"p": [
            {"id": "agent-cell", "models": models, "effort": "high", "arms": ["APG-q"], "cells": ["F7-10"],
             "deliveries": ["push", "pull"], "n_tasks": 5, "epochs": 2},
            {"id": "session-cell", "kind": "session", "models": models, "effort": "high", "arms": ["CM-sum"],
             "N": 10, "sessions": 2, "epochs": 3},
        ]}}},
    })  # fmt: skip
    est = estimate(load_plan(plan), _hand_assumptions(), Prices.load(costs), measured=[])

    # Agent cell, push: 2 calls x (500 base + 200 context + 100 x (2-1)/2 history = 750 in, 1,000 out) at $2/$10 per M;
    # per-query compile once: 1 classify (300 in, 100 out at $1/$4) and 1 query embedding (50 tokens at $0.50).
    push = 2 * (750 * 2 + 1000 * 10) / 1e6 + (300 * 1 + 100 * 4) / 1e6 + 50 * 0.5 / 1e6
    # Pull: 3 calls, context 200 x 2 = 400, history 100 x 1 = 100 -> 1,000 in; a compile per call (3 classify, 3 embeds).
    pull = 3 * (1000 * 2 + 1000 * 10) / 1e6 + 3 * (300 * 1 + 100 * 4) / 1e6 + 3 * 50 * 0.5 / 1e6
    assert push == pytest.approx(0.023725) and pull == pytest.approx(0.038175)
    assert est.total(cell="agent-cell") == pytest.approx(10 * (push + pull))  # 5 tasks x 2 epochs per mode
    # Expected: half the agent input cached at 0.1 x input -> input rate 2 x (0.5 + 0.05) = 1.1; kg and embeddings uncached.
    push_exp = 2 * (750 * 1.1 + 1000 * 10) / 1e6 + (300 + 400) / 1e6 + 25 / 1e6
    pull_exp = 3 * (1000 * 1.1 + 1000 * 10) / 1e6 + 3 * (300 + 400) / 1e6 + 75 / 1e6
    assert est.total("expected", cell="agent-cell") == pytest.approx(10 * (push_exp + pull_exp))

    # Session cell: N=10 items x 2 calls = 20 agent calls on a managed view of 0.5 x W = 5,000 tokens, 1,000 out;
    # summaries add 10% (2 calls); one probe (checkpoint 5 <= N). Every call costs (5,000 x 2 + 1,000 x 10) / 1e6.
    per_call = (5000 * 2 + 1000 * 10) / 1e6
    session = (20 + 2 + 1) * per_call
    assert session == pytest.approx(0.46)
    assert est.total(cell="session-cell") == pytest.approx(2 * 3 * session)  # 2 sessions x 3 epochs
    per_call_exp = (5000 * 2 * (0.75 + 0.25 * 0.1) + 1000 * 10) / 1e6
    assert est.total("expected", cell="session-cell") == pytest.approx(6 * 23 * per_call_exp)
    assert est.total() == pytest.approx(10 * (push + pull) + 6 * session)


# ---------- the real plan ----------


def test_real_plan_fits_the_budget_with_the_documented_allocation():
    plan = load_plan()
    est = estimate(plan, measured=[])
    assert est.total() <= plan.budget["total_usd"] == 5000
    assert est.total("expected") < est.total()
    got = {s: est.total(study=s) for s in ("gate", "main", "study_g")}
    for study, usd in got.items():
        assert usd == pytest.approx(DOCUMENTED[study], abs=1.0), study
    assert est.total() == pytest.approx(DOCUMENTED["total"], abs=1.0)
    assert est.total("expected") == pytest.approx(DOCUMENTED["expected"], abs=1.0)
    assert got["gate"] <= plan.budget["allocations"]["gate"]
    budget_md = (ROOT / "BUDGET.md").read_text()
    for key in ("gate", "main", "study_g", "total", "expected"):
        assert f"{DOCUMENTED[key]:,}" in budget_md, f"BUDGET.md does not show the {key} figure ${DOCUMENTED[key]:,}"


def test_cuts_are_the_in_order_minimum_and_protected_cells_are_untouched():
    plan = load_plan()
    before = load_plan(uncut_plan=True)
    assert estimate(before, measured=[]).total() > plan.budget["total_usd"], "the cuts must have been needed"
    resized, applied = right_size(before, measured=[])
    assert applied == plan.applied_cuts == ["C1", "C2", "C3"]
    assert [c.spec for c in resized.cells] == [c.spec for c in plan.cells]
    # Never cut: gate primary sizes and epochs, the S7 placebo's epochs.
    for cell_id in ("gate.test.f7", "gate.test.f3"):
        spec = plan.cell(cell_id).spec
        assert (spec["worlds"], spec["tasks_per_world"], spec["epochs"]) == (16, 12, 3)
    assert plan.cell("gate.test.f7").spec["deliveries"] == ["push", "pull"]
    assert plan.cell("gate.diag.s7").spec["epochs"] == 3
    edited = {e["cell"] for c in plan.cuts for e in c["set"]}
    assert not edited & {"gate.test.f7", "gate.test.f3", "gate.diag.s7"}


def test_a_cut_inconsistent_with_the_cells_is_rejected(tmp_path):
    raw = yaml.safe_load(PLAN_PATH.read_text())
    for cell in raw["studies"]["study_g"]["phases"]["context_management"]:
        if cell["id"] == "g.cm.sol-high":
            cell["sessions"] = 7  # C2 says applied: 8 -> 6
    with pytest.raises(BudgetError, match="C2 is applied"):
        load_plan(_write(tmp_path / "plan.yaml", raw))
    with pytest.raises(BudgetError, match="no cut"):
        apply_cut(load_plan(), "C99")


# ---------- calibration from logs ----------


def _mock_costs(tmp_path) -> Path:
    """The repo's price table plus a mockllm price, so the gate profile still preflights."""
    table = yaml.safe_load((ROOT / "config" / "model_costs.yaml").read_text())
    return _write(tmp_path / "costs.yaml", {**table, MOCK: MOCK_PRICE})


def _mock_gate_log(tmp_path, costs: Path):
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=3, relational=True, embed=True))
    profile = load_profile("gate")
    log = inspect_eval(
        gate(family="F7", level="10", split="dev", arm="APGo-q"),
        model=agent_model(profile, model=MOCK, custom_outputs=mock_agent),
        model_roles=role_models(profile, ("kg",), model=MOCK, custom_outputs=mock_classifier),
        log_dir=str(tmp_path / "logs"),
        display="none",
        epochs=2,
        **eval_cost_kwargs(costs),
    )[0]
    assert log.status == "success", log.error
    return log


def test_calibrate_writes_measured_tokens_and_the_model_prefers_them(tmp_path):
    costs = _mock_costs(tmp_path)
    log = _mock_gate_log(tmp_path, costs)
    out = tmp_path / "measured.yaml"
    calibrate([log.location], out)

    # Independent count from the log: agent calls are model events without a role.
    calls = {"agent": 0, "kg": 0}
    tokens = {"agent": [0, 0], "kg": [0, 0]}
    for s in log.samples:
        for e in s.events:
            if e.event == "model":
                role = e.role or "agent"
                calls[role] += 1
                tokens[role][0] += e.output.usage.input_tokens
                tokens[role][1] += e.output.usage.output_tokens
    (entry,) = load_measured(out)
    assert (entry["arm"], entry["model"], entry["effort"], entry["cell"], entry["delivery"]) == ("APGo-q", MOCK, "high", "F7-10", "push")
    assert entry["samples"] == len(log.samples) == 6
    for role in ("agent", "kg"):
        r = entry["roles"][role]
        assert calls[role] > 0
        assert r["calls_per_sample"] == pytest.approx(calls[role] / 6, abs=1e-4)
        assert r["input_per_call"] == pytest.approx(tokens[role][0] / calls[role], abs=0.01)
        assert r["output_per_call"] == pytest.approx(tokens[role][1] / calls[role], abs=0.01)

    plan = load_plan(_write(tmp_path / "plan.yaml", {
        "budget": {"total_usd": 100},
        "studies": {"s": {"profile": "gate", "phases": {"p": [
            {"id": "c", "models": {"agent": MOCK, "kg": MOCK}, "arms": ["APGo-q"], "cells": ["F7-10"], "n_tasks": 10, "epochs": 1},
        ]}}},
    }))  # fmt: skip
    prices = Prices.load(costs)
    with_measured = estimate(plan, prices=prices, measured=load_measured(out))
    priors_only = estimate(plan, prices=prices, measured=[])
    assert [r["measured"] for r in with_measured.rows] == [True]
    assert [r["measured"] for r in priors_only.rows] == [False]
    a, k = entry["roles"]["agent"], entry["roles"]["kg"]
    embed = 60 * 0.02 / 1e6  # prior: one query embedding per per-query compile, text-embedding-3-small
    per_sample = a["calls_per_sample"] * (a["input_per_call"] * 1.0 + a["output_per_call"] * 2.0) / 1e6
    per_sample += k["calls_per_sample"] * (k["input_per_call"] * 1.0 + k["output_per_call"] * 2.0) / 1e6
    assert with_measured.total() == pytest.approx(10 * (per_sample + embed))
    assert with_measured.total() != pytest.approx(priors_only.total())

    # Re-calibrating merges: an entry for another key survives.
    other = {"arm": "S1", "model": MOCK, "effort": "high", "cell": "F3-5", "delivery": "push", "samples": 1, "roles": {}}
    out.write_text(yaml.safe_dump({"entries": [other, entry]}))
    calibrate([log.location], out)
    assert [e["arm"] for e in load_measured(out)] == ["S1", "APGo-q"]


def test_remaining_counts_logs_and_ledger_and_refuses_an_unaffordable_phase(tmp_path):
    costs = _mock_costs(tmp_path)
    log = _mock_gate_log(tmp_path, costs)
    ledger = Config().ledger_path
    Ledger(ledger).append(LedgerEntry(role="build", model="gpt-6-luna", kind="chat", input_tokens=1_000_000))  # $0.10
    left = remaining(5.0, [log.location], ledger, costs)
    inspect_usd = sum(u.total_cost for s in log.samples for u in s.model_usage.values())
    assert left["inspect_usd"] == pytest.approx(inspect_usd) and inspect_usd > 0
    assert left["ledger_usd"] == pytest.approx(0.10)
    assert left["remaining_usd"] == pytest.approx(5.0 - inspect_usd - 0.10)
    require_affordable(1.0, left["remaining_usd"])
    with pytest.raises(BudgetError, match="exceeds the remaining"):
        require_affordable(left["remaining_usd"] + 0.01, left["remaining_usd"], "test phase")


# ---------- CLI ----------


def test_cli_passes_on_the_real_plan_and_fails_on_an_inflated_one(tmp_path, capsys):
    empty = tmp_path / "none.yaml"  # no measured calibration
    assert main(["--measured", str(empty)]) == 0
    assert "FITS" in capsys.readouterr().out
    raw = yaml.safe_load(PLAN_PATH.read_text())
    for cell in raw["studies"]["gate"]["phases"]["test"]:
        cell["epochs"] = 300  # inflate the gate's primary cells far beyond the budget
    inflated = _write(tmp_path / "plan.yaml", raw)
    assert main(["--plan", str(inflated), "--measured", str(empty), "--study", "gate"]) == 1
    assert "EXCEEDS BUDGET" in capsys.readouterr().out
    assert main(["--plan", str(inflated), "--measured", str(empty), "--scenario", "expected"]) == 1


def test_the_assumptions_file_names_both_cache_scenarios_and_every_planned_arm():
    a = yaml.safe_load(ASSUMPTIONS_PATH.read_text())
    assert set(a["cache_scenarios"]) == {"conservative", "expected"}
    assert a["cache_scenarios"]["conservative"]["cached_price"] == "input"
    # Every arm and task cell in the plan resolves (estimate raises on an unknown one).
    assert len(estimate(load_plan(), measured=[]).rows) > 300
