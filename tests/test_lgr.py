"""LightRAG adapter (module 6): offline oracle index, provenance mapping, loop affinity, isolation."""

import asyncio

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.build import build
from ape.config import Config
from ape.lgr.build import build_index
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.tasks.gate import gate
from ape.worlds.spec import World


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    monkeypatch.setenv("APE_WORLDS", str(tmp_path / "worlds"))
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("APE_INDICES", str(tmp_path / "indices"))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def _build(family, level, n_worlds=1, n_tasks=2):
    ids = asyncio.run(build("dev", family, [level], n_worlds=n_worlds, n_tasks=n_tasks, relational=True, embed=True))
    cfg = Config()
    for wid in ids:
        asyncio.run(build_index(World.load(cfg.world_path(wid)), "oracle", cfg))
    return ids


def _run(family, level, arm, offline_env):
    return inspect_eval(
        gate(family=family, level=level, split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        model_roles={"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)},
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5"), ("F5", "2hop")])
@pytest.mark.parametrize("arm", ["LGRo-q", "LGRo-s"])
def test_lightrag_oracle_arms_run_end_to_end(offline_env, family, level, arm):
    _build(family, level)
    log = _run(family, level, arm, offline_env)
    assert log.status == "success", log.error
    for s in log.samples:
        recs = s.store["compile_log"]
        assert recs and all(r["meta"]["index_hash"] for r in recs)
        assert all(r["fact_ids"] for r in recs), "provenance must map LightRAG sources to world facts"
        kg_calls = [e for e in s.events if e.event == "model" and e.role == "kg"]
        assert len(kg_calls) == len(recs), "exactly one metered keyword call per compile"


def test_two_worlds_stay_isolated_and_concurrent_queries_share_a_loop(offline_env):
    ids = _build("F7", "10", n_worlds=2, n_tasks=10)
    log = _run("F7", "10", "LGRo-s", offline_env)
    assert log.status == "success", log.error
    assert len(log.samples) == 20
    by_world = {}
    for s in log.samples:
        facts = {f for r in s.store["compile_log"] for f in r["fact_ids"]}
        by_world.setdefault(s.metadata["world_id"], set()).update(facts)
    a, b = (by_world[i] for i in ids)
    worlds = [World.load(Config().world_path(i)) for i in ids]
    assert a <= set(worlds[0].facts) and b <= set(worlds[1].facts), "a workspace leaked another world's facts"


def test_stale_index_is_refused(offline_env):
    from ape.lgr.adapter import build_lgr_arm
    from ape.lgr.common import index_dir, read_manifest, write_manifest

    (wid,) = _build("F7", "10")
    cfg = Config()
    wd = index_dir(cfg, wid, "oracle")
    m = read_manifest(wd)
    write_manifest(wd, {**m, "world_hash": "different"})
    with pytest.raises(RuntimeError, match="different world version"):
        asyncio.run(build_lgr_arm("LGRo-q", World.load(cfg.world_path(wid)), cfg))
