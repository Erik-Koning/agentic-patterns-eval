"""Build throughput (FX-4): parallel per-world artifact builds, idempotent reruns, multi-process safety.

Workers are spawned processes; they inherit the test's environment (temp worlds, cache, indices,
fake embeddings), so everything stays offline.
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import pytest
from lightrag import LightRAG

from ape.apg.arm import graph_path
from ape.apg.author import FAKE_AUTHOR_ID, author_prompt_version, author_world, authored_graph_current
from ape.artifacts import build_artifacts, main, world_paths
from ape.build import build
from ape.config import BuildConcurrency, Config
from ape.lgr.common import index_dir, open_rag, read_manifest
from ape.llm.build_client import MAX_RETRIES, BuildLlm
from ape.llm.embeddings import EmbeddingCache
from ape.llm.fake import FakeEmbeddingsClient, perfect_author
from ape.llm.ledger import Ledger, LedgerEntry
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world, chunks_hash
from ape.worlds.spec import World

ALL = ("chunks", "apg", "lightrag")
KNOBS = ("APE_BUILD_LLM_CONCURRENCY", "APE_BUILD_PARALLEL_INSERT", "APE_BUILD_EMBED_CONCURRENCY", "APE_BUILD_WORKERS")


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    for k in (*KNOBS, "APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_EFFORT", "APE_BUILD_FALLBACK"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _worlds(embed: bool = False) -> list[Path]:
    """Three small worlds: two F7-10 and one F3-5."""
    ids = asyncio.run(build("dev", "F7", ["10"], n_worlds=2, n_tasks=2, relational=True, embed=embed))
    ids += asyncio.run(build("dev", "F3", ["5"], n_worlds=1, n_tasks=2, relational=True, embed=embed))
    cfg = Config()
    return [cfg.world_path(i) for i in ids]


def _build(paths, workers, **kw) -> list:
    kw = {"kinds": ALL, "lightrag_kind": "oracle", "fake_author": True, **kw}
    return build_artifacts(paths, workers=workers, **kw)


def _lightrag_content(wd: Path) -> dict:
    """The stable content of a LightRAG index (its stores also hold timestamps, so no file hashes)."""

    def store(name: str):
        (path,) = wd.rglob(name)
        return json.loads(path.read_text())

    return {
        "entities": sorted((d["entity_name"], d["content"], d["source_id"]) for d in store("vdb_entities.json")["data"]),
        "relations": sorted((d["src_id"], d["tgt_id"], d["content"]) for d in store("vdb_relationships.json")["data"]),
        "chunks": sorted((k, v["content"], v["file_path"]) for k, v in store("kv_store_text_chunks.json").items()),
    }


def _artifacts(cfg: Config, world: World) -> dict:
    wd = index_dir(cfg, world.id, "oracle")
    manifest = read_manifest(wd)
    manifest.pop("index_hash")  # includes LightRAG's write timestamps
    path = graph_path(cfg, world.id, "authored")
    return {"lightrag": _lightrag_content(wd), "manifest": manifest, "apg": path.read_bytes(), "report": path.with_suffix(".report.json").read_bytes()}


def test_parallel_builds_match_serial_builds(offline_env, monkeypatch):
    paths = _worlds()
    worlds = [World.load(p) for p in paths]
    runs, seconds = {}, {}
    for name, workers in (("serial", 1), ("parallel", 3)):
        monkeypatch.setenv("APE_INDICES", str(offline_env / f"indices-{name}"))
        monkeypatch.setenv("APE_CACHE", str(offline_env / f"cache-{name}"))
        t0 = time.monotonic()
        results = _build(paths, workers)
        seconds[name] = time.monotonic() - t0
        assert [r.status for r in results] == ["built"] * 3, [r.summary() for r in results]
        assert all(r.kinds == {k: "built" for k in ALL} for r in results)
        cfg = Config()
        runs[name] = {w.id: _artifacts(cfg, w) for w in worlds}
        for w in worlds:
            assert runs[name][w.id]["manifest"]["world_hash"] == w.artifact_hash()
            assert authored_graph_current(w, cfg, FAKE_AUTHOR_ID)
    for w in worlds:
        s, p = runs["serial"][w.id], runs["parallel"][w.id]
        assert s["lightrag"]["entities"] and s["lightrag"] == p["lightrag"]
        assert s["manifest"] == p["manifest"]
        assert s["apg"] == p["apg"] and s["report"] == p["report"], "authored APG graphs must match byte for byte"
    print(f"\nwall clock, 3 worlds (chunks+apg+oracle lightrag): serial {seconds['serial']:.1f}s, 3 workers {seconds['parallel']:.1f}s")


def test_reruns_skip_current_worlds_and_rebuild_only_what_changed(offline_env):
    paths = _worlds()
    first = _build(paths, 3)
    assert [r.status for r in first] == ["built"] * 3

    second = _build(paths, 3)
    assert all(r.kinds == {k: "skipped" for k in ALL} for r in second), [r.summary() for r in second]

    cfg = Config()
    a, b, c = (World.load(p) for p in paths)
    (index_dir(cfg, a.id, "oracle") / "ape_manifest.json").unlink()  # as if its LightRAG build had crashed
    graph_path(cfg, b.id, "authored").with_suffix(".report.json").unlink()  # crashed between graph and report
    third = _build(paths, 3)
    assert [r.kinds for r in third] == [
        {"chunks": "skipped", "apg": "skipped", "lightrag": "built"},
        {"chunks": "skipped", "apg": "built", "lightrag": "skipped"},
        {"chunks": "skipped", "apg": "skipped", "lightrag": "skipped"},
    ]

    # Same world ID with other tasks: graphs and indices are built from the world without its tasks, so nothing rebuilds.
    regenerated = make_world("F3", "5", "dev", 0, n_tasks=5)
    assert regenerated.id == c.id and regenerated.content_hash() != c.content_hash() and regenerated.artifact_hash() == c.artifact_hash()
    regenerated.save(paths[2])
    fourth = _build(paths, 3)
    assert [r.status for r in fourth] == ["skipped", "skipped", "skipped"], [r.summary() for r in fourth]

    # A changed document is a new world version: its graph and index rebuild.
    edited = World.load(paths[2])
    para = next(p for d in edited.documents for p in d.paragraphs if p.fact_ids)
    para.text += " (revised)"
    edited.save(paths[2])
    fifth = _build(paths, 3)
    assert [r.status for r in fifth] == ["skipped", "skipped", "built"]
    assert fifth[2].kinds["apg"] == "built" and fifth[2].kinds["lightrag"] == "built"
    assert read_manifest(index_dir(cfg, c.id, "oracle"))["world_hash"] == edited.artifact_hash()
    assert json.loads(graph_path(cfg, c.id, "authored").read_text())["meta"]["worldHash"] == edited.artifact_hash()


def test_a_failing_world_is_reported_while_the_others_build(offline_env, capsys):
    paths = _worlds()
    paths[1].write_text("{ not a world")
    results = _build(paths, 3)
    assert [r.status for r in results] == ["built", "failed", "built"]
    assert "world" in results[1].errors and set(results[1].kinds.values()) == {"failed"}
    with pytest.raises(SystemExit) as exit_:
        main(["--split", "dev", "--family", "F7", "--kinds", "chunks,apg,lightrag", "--lightrag-kind", "oracle", "--fake-author", "--workers", "2"])
    assert exit_.value.code == 1
    out = capsys.readouterr().out
    assert "1 failed" in out and "1 skipped" in out


def test_cli_selects_levels_and_validates_kinds(offline_env):
    _worlds()
    cfg = Config()
    assert len(world_paths(cfg, "dev", "F7")) == 2 and len(world_paths(cfg, "dev", "F7", ["10"])) == 2
    assert world_paths(cfg, "dev", "F7", ["100"]) == [] and len(world_paths(cfg, "dev", "F3", ["5"])) == 1
    with pytest.raises(ValueError, match="unknown artifact kind"):
        build_artifacts([], ["graphs"])
    with pytest.raises(ValueError, match="LightRAG kind"):
        build_artifacts([], ["lightrag"])
    with pytest.raises(SystemExit):
        main(["--split", "dev", "--family", "F7", "--kinds", "lightrag"])  # no --lightrag-kind


def test_authored_graph_records_world_author_and_embedding_model(offline_env):
    (path,) = _worlds()[2:]
    world, cfg = World.load(path), Config()
    asyncio.run(author_world(world, cfg, perfect_author, author_id=FAKE_AUTHOR_ID))
    meta = json.loads(graph_path(cfg, world.id, "authored").read_text())["meta"]
    assert meta == {
        "worldHash": world.artifact_hash(),
        "chunksHash": chunks_hash(chunk_world(world)),
        "author": FAKE_AUTHOR_ID,
        "authorVersion": author_prompt_version(),
        "embeddingModel": "fake-bow",
        "embeddingDim": 256,
    }
    assert authored_graph_current(world, cfg, FAKE_AUTHOR_ID)
    assert not authored_graph_current(world, cfg, "gpt-6-luna@high"), "another author means a rebuild"


# ---- concurrency knobs ----


def test_build_time_lightrag_gets_the_concurrency_knobs_and_query_time_keeps_defaults(tmp_path, monkeypatch, fake_embeddings):
    monkeypatch.setenv("APE_BUILD_LLM_CONCURRENCY", "5")
    monkeypatch.setenv("APE_BUILD_PARALLEL_INSERT", "2")
    monkeypatch.setenv("APE_BUILD_EMBED_CONCURRENCY", "7")

    async def llm(*_a, **_k):
        return ""

    async def knobs(query_time: bool, name: str) -> tuple:
        rag = await open_rag(tmp_path / name, f"knobs-{name}", llm, fake_embeddings, query_time=query_time)
        try:
            return rag.llm_model_max_async, rag.max_parallel_insert, rag.embedding_func_max_async
        finally:
            await rag.finalize_storages()

    assert asyncio.run(knobs(False, "build")) == (5, 2, 7)
    defaults = LightRAG.__dataclass_fields__
    assert asyncio.run(knobs(True, "query")) == tuple(defaults[f].default for f in ("llm_model_max_async", "max_parallel_insert", "embedding_func_max_async"))


def test_concurrency_defaults_and_validation(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    assert BuildConcurrency.from_env() == BuildConcurrency(llm=16, parallel_insert=8, embed=16, workers=4)
    monkeypatch.setenv("APE_BUILD_WORKERS", "0")
    with pytest.raises(ValueError, match="APE_BUILD_WORKERS"):
        BuildConcurrency.from_env()
    monkeypatch.setenv("APE_BUILD_WORKERS", "many")
    with pytest.raises(ValueError, match="APE_BUILD_WORKERS"):
        BuildConcurrency.from_env()


def test_apg_authoring_concurrency_comes_from_the_environment(offline_env, monkeypatch):
    (path,) = _worlds()[:1]
    monkeypatch.setenv("APE_BUILD_LLM_CONCURRENCY", "3")
    state = {"now": 0, "peak": 0}

    async def author(system: str, user: str) -> dict:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return await perfect_author(system, user)

    asyncio.run(author_world(World.load(path), Config(), author))
    assert state["peak"] == 3


def test_build_llm_client_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")  # constructing the client makes no request
    client = BuildLlm("gpt-6-luna", Ledger(tmp_path / "l.jsonl"), {})._client_()
    assert client.max_retries == MAX_RETRIES == 6


# ---- LightRAG extraction: incomplete builds are never marked complete, rebuilds reuse the LLM cache ----

EXTRACT_SCRIPT = textwrap.dedent(
    """
    import asyncio, json, re, sys
    from types import SimpleNamespace
    from ape.config import Config
    from ape.lgr.build import build_index
    from ape.llm.build_client import BuildLlm
    from ape.worlds.spec import World

    fail_after = int(sys.argv[2])
    calls = 0

    async def create(**kw):
        global calls
        calls += 1
        if fail_after >= 0 and calls > fail_after:
            raise RuntimeError("simulated outage")
        ids = sorted(set(re.findall(r"(?:Policy|Exception) ([A-Za-z0-9_-]+?)(?=[\\s.(,:]|$)", kw["messages"][-1]["content"])))
        out = "\\n".join([f"entity<|#|>{i}<|#|>policy<|#|>{i} is a rule." for i in ids] + ["<|COMPLETE|>"])
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5), choices=[SimpleNamespace(message=SimpleNamespace(content=out))])

    BuildLlm._client_ = lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    try:
        asyncio.run(build_index(World.load(sys.argv[1]), "extract", Config()))
        error = None
    except Exception as e:
        error = str(e)
    print("RESULT " + json.dumps({"calls": calls, "error": error}))
    """
)


def _extract(script: Path, world: Path, fail_after: int, **env: str) -> dict:
    """One extract build of `world` in a fresh process; its first `fail_after` LLM calls succeed (-1: all)."""
    proc = subprocess.run([sys.executable, str(script), str(world), str(fail_after)], capture_output=True, text=True, env={**os.environ, **env}, timeout=300, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_failed_extraction_writes_no_manifest_and_the_rebuild_reuses_the_llm_cache(offline_env):
    """Each build runs in a fresh process, as under `ape.artifacts` (LightRAG keeps a workspace's stores in memory)."""
    (path,) = _worlds(embed=True)[:1]
    script = offline_env / "extract.py"
    script.write_text(EXTRACT_SCRIPT)
    wd = index_dir(Config(), World.load(path).id, "extract")
    clean = _extract(script, path, fail_after=-1, APE_INDICES=str(offline_env / "indices-clean"))
    assert clean["error"] is None and clean["calls"] > 3

    failed = _extract(script, path, fail_after=3)
    assert failed["error"] and "incomplete" in failed["error"]
    assert not (wd / "ape_manifest.json").exists(), "an index with FAILED documents must not look complete"

    rebuilt = _extract(script, path, fail_after=-1)
    assert rebuilt["error"] is None and read_manifest(wd)["world_hash"] == World.load(path).artifact_hash()
    assert rebuilt["calls"] == clean["calls"] - 3, "the rebuild pays only for extractions the failed build did not cache"
    assert _lightrag_content(wd) == _lightrag_content(offline_env / "indices-clean" / wd.relative_to(Config().indices_dir))
    again = _extract(script, path, fail_after=-1)  # a forced rebuild with the same build model: all cached
    assert again["error"] is None and again["calls"] == 0


# ---- multi-process safety of the shared embedding cache and ledger ----


def _embed_worker(base: str, worker: int, start_at: float, n_fresh: int, rounds: int) -> int:
    time.sleep(max(0.0, start_at - time.time()))  # all workers start together, so they really contend

    async def run() -> int:
        for i in range(n_fresh):  # every worker opens each brand-new cache at once: creation and WAL switch race
            fresh = EmbeddingCache(f"{base}-fresh{i}.sqlite", "fake", client=FakeEmbeddingsClient())
            await fresh.embed([f"fresh {i}", f"worker {worker} fresh {i}"])
        cache = EmbeddingCache(f"{base}.sqlite", "fake", client=FakeEmbeddingsClient())
        for i in range(rounds):
            # Distinct texts (each a write) plus a shared one every process writes too.
            await cache.embed([f"worker {worker} text {i}", f"worker {worker} other {i}", f"shared text {i}"])
            cache.lookup([f"shared text {max(0, i - 1)}"])
        return rounds

    return asyncio.run(run())


def test_embedding_cache_takes_concurrent_writers_from_several_processes(tmp_path):
    base = str(tmp_path / "shared")
    start_at = time.time() + 3.0  # after the spawned workers have imported
    with ProcessPoolExecutor(max_workers=3, mp_context=get_context("spawn")) as pool:
        done = list(pool.map(_embed_worker, [base] * 3, range(3), [start_at] * 3, [20] * 3, [150] * 3))  # raises on "database is locked"
    assert done == [150] * 3
    cache = EmbeddingCache(f"{base}.sqlite", "fake", client=FakeEmbeddingsClient())
    texts = [f"worker {w} {kind} {i}" for w in range(3) for kind in ("text", "other") for i in range(150)] + [f"shared text {i}" for i in range(150)]
    assert len(cache.lookup(texts)) == len(texts)
    assert cache._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    for i in range(20):
        fresh = EmbeddingCache(f"{base}-fresh{i}.sqlite", "fake", client=FakeEmbeddingsClient())
        assert len(fresh.lookup([f"fresh {i}"] + [f"worker {w} fresh {i}" for w in range(3)])) == 4


def test_embedding_cache_waits_to_switch_a_busy_rollback_journal_file_to_wal(tmp_path):
    """The one case SQLite's busy timeout does not cover: switching to WAL while another connection
    holds a write on a file still in rollback mode (a new cache, or one created before WAL)."""
    path = tmp_path / "old.sqlite"
    other = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    other.execute("CREATE TABLE emb (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    other.execute("BEGIN IMMEDIATE")
    other.execute("INSERT INTO emb VALUES ('k', '[]')")
    threading.Timer(0.5, other.execute, ("COMMIT",)).start()
    cache = EmbeddingCache(path, "fake", client=FakeEmbeddingsClient())  # waits for the commit, does not raise
    assert cache._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert len(asyncio.run(cache.embed(["after the switch"]))) == 1


def _ledger_worker(path: str, worker: int, n: int) -> int:
    ledger = Ledger(path)
    pad = "x" * 2500  # entries near the 4 KB bound make interleaving likelier if writes were split
    for i in range(n):
        ledger.append(LedgerEntry(role="build", model="m", kind="chat", input_tokens=i, context={"worker": worker, "i": i, "pad": pad}))
    return n


def test_ledger_lines_stay_whole_under_concurrent_appends(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        assert sum(pool.map(_ledger_worker, [path] * 4, range(4), [300] * 4)) == 1200
    lines = Path(path).read_text().splitlines()
    assert len(lines) == 1200 and all(len(ln.encode()) < 4096 for ln in lines)
    entries = [json.loads(ln) for ln in lines]  # every line is one whole entry
    seen = {(e["context"]["worker"], e["context"]["i"]) for e in entries}
    assert seen == {(w, i) for w in range(4) for i in range(300)}
    assert len(Ledger(path).read()) == 1200
