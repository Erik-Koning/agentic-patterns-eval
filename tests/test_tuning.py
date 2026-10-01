"""Tuning runner: equal budget enforced, every candidate logged, selection rule applied, env restored."""

import asyncio
import json
import os

import pytest
from inspect_ai.model import get_model

from ape.build import build
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.tuning import candidates, env, select, tune


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def test_budget_is_enforced():
    grid = {"budget_per_system": 2, "systems": {"X": {"candidates": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}}}
    with pytest.raises(ValueError, match="exceed the equal budget"):
        candidates(grid, "X")


def test_selection_rule_prefers_cheaper_within_tie():
    recs = [
        {"candidate": {"id": "a"}, "mean_success": 0.80, "cost_usd": 2.0},
        {"candidate": {"id": "b"}, "mean_success": 0.795, "cost_usd": 1.0},
        {"candidate": {"id": "c"}, "mean_success": 0.70, "cost_usd": 0.1},
    ]
    assert select(recs, tie_pp=1.0)["candidate"]["id"] == "b"
    assert select(recs, tie_pp=0.1)["candidate"]["id"] == "a"


def test_env_overrides_are_restored(monkeypatch):
    monkeypatch.delenv("APE_TEST_KNOB", raising=False)
    with env({"APE_TEST_KNOB": "1"}):
        assert os.environ["APE_TEST_KNOB"] == "1"
    assert "APE_TEST_KNOB" not in os.environ


def test_tune_logs_every_candidate_and_the_selection(offline_env):
    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    grid = {
        "budget_per_system": 2,
        "tie_pp": 1.0,
        "dev_cells": ["F7-10"],
        "systems": {"S3s": {"candidates": [{"id": "small", "arm": "S3s", "env": {"APE_CONTEXT_BUDGET": "300"}}, {"id": "big", "arm": "S3s", "env": {"APE_CONTEXT_BUDGET": "2000"}}]}},
    }
    log_path = offline_env / "tuning.jsonl"
    chosen = tune(
        "S3s",
        get_model("mockllm/model", custom_outputs=mock_agent),
        {"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)},
        grid=grid,
        log_path=log_path,
    )
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [line["candidate"]["id"] for line in lines[:2]] == ["small", "big"]
    # Candidates differ only in env knobs, which are not part of Inspect's task identity: each needs its own log dir.
    assert lines[0]["log_files"] != lines[1]["log_files"]
    assert lines[2]["selected"] == chosen["candidate"]["id"]
    assert "APE_CONTEXT_BUDGET" not in os.environ


def test_failed_candidate_is_logged_and_unselectable():
    recs = [
        {"candidate": {"id": "broken"}, "status": "failed", "mean_success": None, "cost_usd": None},
        {"candidate": {"id": "ok"}, "status": "ok", "mean_success": 0.5, "cost_usd": 1.0},
    ]
    assert select(recs, tie_pp=1.0)["candidate"]["id"] == "ok"
    with pytest.raises(RuntimeError, match="no tuning candidate completed"):
        select(recs[:1], tie_pp=1.0)
