"""End-to-end dry run (readiness H2 for the baseline arms): mock agent, fake embeddings, no network."""

import asyncio

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.build import build
from ape.llm.mock_agent import mock_agent
from ape.tasks.gate import gate

REQUIRED_STORE = ["compile_log", "turns_used", "arm"]


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APE_WORLDS", str(tmp_path / "worlds"))
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5"), ("F5", "2hop")])
@pytest.mark.parametrize("arm", ["S1", "S3s", "S6", "S7"])
def test_baseline_arms_run_end_to_end(offline_env, family, level, arm):
    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    logs = inspect_eval(
        gate(family=family, level=level, split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        log_dir=str(offline_env / "logs"),
        display="none",
    )
    log = logs[0]
    assert log.status == "success", log.error
    assert len(log.samples) == 2
    for s in log.samples:
        for key in REQUIRED_STORE:
            assert key in s.store, key
        rec = s.store["compile_log"][0]
        assert {"tokens", "prompt_hash", "unit_ids", "fact_ids"} <= set(rec)
        assert set(s.scores) == {"task_success", "delivered_evidence"}
    if arm == "S6":
        assert all(s.scores["delivered_evidence"].value["evidence_recall"] == 1.0 for s in log.samples)
    per_step = arm == "S3s"
    turns = log.samples[0].store["turns_used"]
    assert len(log.samples[0].store["compile_log"]) == (turns if per_step else 1)
    history = [m.text for m in log.samples[0].messages]
    assert not any("refreshed for this step" in t for t in history), "per-step context must never enter the history"
    assert ("## Knowledge base" in history[0]) == (not per_step), "per-query context lives in the system prompt"
