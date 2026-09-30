"""APG authoring (module 7) with a scripted "perfect" author: graph shape, links, allowlists, end to end."""

import asyncio
import json
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.apg.arm import graph_path
from ape.apg.author import author_world
from ape.build import build
from ape.config import Config
from ape.llm.mock_agent import mock_agent, mock_kg
from ape.tasks.gate import gate
from ape.worlds.spec import World

ID = re.compile(r"(?:Policy|Exception|Procedure|Tool) ([A-Za-z0-9_-]+?)(?=[\s.(,:]|$)")


async def perfect_author(system: str, user: str) -> dict:
    excerpt = user.split("Excerpt:\n", 1)[1].split("\n", 1)[1]  # drop the "[Document title]" header line
    units = []
    for para in excerpt.split("\n\n"):
        if not para.strip():
            continue
        ids = ID.findall(para)
        tools = re.findall(r"call (\w+)", para)
        units.append(
            {
                "declares": ids[:1],
                "title": " ".join(para.split()[:14]),
                "description": para[:120],
                "knowledge": para,
                "references": ids[1:],
                "tools": tools,
            }
        )
    units.append({"declares": [], "title": "empty", "description": "", "knowledge": "  ", "references": [], "tools": []})
    return {"units": units}


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def _author(family, level, n_tasks=2):
    (wid,) = asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=n_tasks, relational=True, embed=True))
    cfg = Config()
    world = World.load(cfg.world_path(wid))
    report = asyncio.run(author_world(world, cfg, perfect_author))
    doc = json.loads(graph_path(cfg, wid, "authored").read_text())
    return world, doc, report


def test_f7_exceptions_and_policies_bring_each_other(offline_env):
    world, doc, report = _author("F7", "100")
    assert report["id_coverage"] == 1.0 and report["unresolved_references"] == 0
    leaves = {n["props"]["declares"][0]: n for n in doc["nodes"] if n.get("props", {}).get("declares")}
    for x in world.exceptions:
        assert leaves[x.policy_id]["id"] in leaves[x.id]["bring"]
        assert leaves[x.id]["id"] in leaves[x.policy_id]["bring"]
    assert all(n.get("prompt", {}).get("slots", {}).get("knowledge", "").strip() for n in doc["nodes"] if n.get("routable"))


def test_f3_allowlists_on_procedure_leaves_only(offline_env):
    world, doc, _ = _author("F3", "20")
    assert doc["profile"] == "L3"
    by_decl = {n["props"]["declares"][0]: n for n in doc["nodes"] if n.get("props", {}).get("declares")}
    for proc in world.procedures:
        assert by_decl[proc.id]["toolAllowlist"] == [s["tool"] for s in proc.steps]
    for t in world.tools:
        leaf = by_decl[t.name]
        assert "toolAllowlist" not in leaf
        assert not any(b in {by_decl[p.id]["id"] for p in world.procedures} for b in leaf.get("bring", [])), "tool specs must not bring procedures"
    assert all("toolAllowlist" not in n for n in doc["nodes"] if not n.get("routable"))


@pytest.mark.parametrize("family,level", [("F7", "10"), ("F3", "5")])
def test_authored_graph_runs_end_to_end(offline_env, family, level):
    _author(family, level)
    log = inspect_eval(
        gate(family=family, level=level, split="dev", arm="APG-s"),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        model_roles={"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)},
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    recs = [r for s in log.samples for r in s.store["compile_log"]]
    assert recs and all(r["fact_ids"] for r in recs), "authored leaves map back to facts via sourceChunkIds"
