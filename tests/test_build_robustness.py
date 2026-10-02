"""Build robustness (RELIABILITY_REVIEW K1, K2, K7): bad replies are retried or repaired, a few lost chunks are
tolerated and recorded, more fail the world without paying twice, degraded extractions never pass silently, and
calls that may be billed are metered. Scripted OpenAI stand-ins only; no network."""

import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from lightrag.prompt import PROMPTS
from lightrag.utils import TruncatedResponse
from scripted_openai import ScriptedChat, completion

from ape.apg.arm import graph_path
from ape.apg.author import UNIT_SCHEMA, AuthoringError, author_world, build_llm_author
from ape.build_quality import builder_roles, world_health
from ape.config import Config
from ape.lgr.common import index_dir, read_manifest
from ape.llm.build_client import REPAIR_PROMPT, BuildLlm, BuildResponseError, max_failed_chunks
from ape.llm.fake import FakeEmbeddingsClient, perfect_author
from ape.llm.ledger import Ledger
from ape.models import load_profile
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world

TUPLE, DONE = PROMPTS["DEFAULT_TUPLE_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"]


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    monkeypatch.delenv("APE_BUILD_MAX_FAILED_SHARE", raising=False)
    return tmp_path


def _world(family: str, level: str):
    w = make_world(family, level, "dev", 0, 2)
    w.save(Config().world_path(w.id))
    return w


def _perfect(messages) -> str:
    """perfect_author's reply for this prompt (it awaits nothing, so it runs synchronously)."""
    coro = perfect_author("", messages[-1]["content"])
    try:
        coro.send(None)
    except StopIteration as e:
        return json.dumps(e.value)
    raise AssertionError("perfect_author awaited")


def _llm_author(client, ledger_path: Path, world_id: str):
    llm = BuildLlm("gpt-6-luna", Ledger(ledger_path), {"world": world_id, "system": "apg-author"}, client=client, reasoning_effort="high")

    async def author(system, user):
        return await llm.json(system, user, "units", UNIT_SCHEMA)

    return author


# ---- the build client: retries, repair, LightRAG's no-cache marker, metering ----------------------------------


def test_json_retries_refusals_and_truncation_and_repairs_invalid_json(tmp_path):
    replies = [completion(None, refusal="I can't help with that."), completion('{"units": [', finish_reason="length"), completion('{"units": [}'), completion('{"units": []}')]
    client = ScriptedChat(lambda i, m, k: replies[i])
    llm = BuildLlm("gpt-6-luna", Ledger(tmp_path / "l.jsonl"), {}, client=client)
    out = asyncio.run(llm.json("sys", "user", "units", UNIT_SCHEMA, attempts=4))
    assert out == {"units": []} and client.calls == 4
    # A refusal or a truncation is asked again as is; invalid JSON is repaired with the bad reply and the error.
    assert [len(m) for m in client.messages] == [2, 2, 2, 4]
    assert client.messages[3][2] == {"role": "assistant", "content": '{"units": [}'}
    assert client.messages[3][3]["content"].startswith(REPAIR_PROMPT.split("(")[0])
    assert len(Ledger(tmp_path / "l.jsonl").read()) == 4, "every attempt is metered"

    client = ScriptedChat(lambda i, m, k: completion(""))
    with pytest.raises(BuildResponseError, match="no usable reply after 3 attempt"):
        asyncio.run(BuildLlm("m", Ledger(tmp_path / "l2.jsonl"), {}, client=client).json("s", "u", "units", UNIT_SCHEMA))


def test_lightrag_replies_are_checked_and_truncation_is_marked_uncacheable(tmp_path):
    system = f"Use {TUPLE} between fields and end with {DONE}."  # an entity-extraction system prompt
    good = f"entity{TUPLE}P-1{TUPLE}policy{TUPLE}A rule.\n{DONE}"
    for replies, expect in [
        ([completion(None, refusal="no"), completion(good)], good),  # refusal: asked again
        ([completion("The excerpt describes refund policies."), completion(good)], good),  # prose instead of records
    ]:
        client = ScriptedChat(lambda i, m, k, r=replies: r[i])
        llm = BuildLlm("m", Ledger(tmp_path / "l.jsonl"), {}, client=client).lightrag_func()
        assert asyncio.run(llm("extract this", system_prompt=system)) == expect and client.calls == 2
    client = ScriptedChat(lambda i, m, k: completion("prose only"))
    with pytest.raises(BuildResponseError, match="no extraction records"):
        asyncio.run(BuildLlm("m", Ledger(tmp_path / "l.jsonl"), {}, client=client).lightrag_func()("x", system_prompt=system))
    assert client.calls == 3
    # Truncated with content: LightRAG parses it, and never caches it (its TruncatedResponse marker).
    client = ScriptedChat(lambda i, m, k: completion(f"entity{TUPLE}P-1", finish_reason="length"))
    out = asyncio.run(BuildLlm("m", Ledger(tmp_path / "l.jsonl"), {}, client=client).lightrag_func()("x", system_prompt=system))
    assert isinstance(out, TruncatedResponse) and out == f"entity{TUPLE}P-1"
    # Gleaning (history) and summaries (no record format in the system prompt) are not held to the record check.
    client = ScriptedChat(lambda i, m, k: completion("A merged summary."))
    llm = BuildLlm("m", Ledger(tmp_path / "l.jsonl"), {}, client=client).lightrag_func()
    assert asyncio.run(llm("summarise", system_prompt="Summarise.")) == "A merged summary."
    assert asyncio.run(llm("continue", system_prompt=system, history_messages=[{"role": "user", "content": "x"}])) == "A merged summary."


def test_a_cancelled_call_is_metered_with_its_estimated_input(tmp_path):
    client = ScriptedChat(lambda i, m, k: completion('{"units": []}'), delay=5.0)
    llm = BuildLlm("gpt-6-luna", Ledger(tmp_path / "l.jsonl"), {"world": "w"}, client=client)

    async def run():
        task = asyncio.create_task(llm.json("system prompt", "a user prompt of some length", "units", UNIT_SCHEMA))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    (entry,) = Ledger(tmp_path / "l.jsonl").read()
    assert entry.context == {"world": "w", "status": "cancelled"} and entry.input_tokens > 0 and entry.output_tokens == 0


def test_max_failed_chunks_is_a_rounded_down_share(monkeypatch):
    monkeypatch.delenv("APE_BUILD_MAX_FAILED_SHARE", raising=False)
    assert [max_failed_chunks(n) for n in (6, 49, 50, 547)] == [0, 0, 1, 10]
    monkeypatch.setenv("APE_BUILD_MAX_FAILED_SHARE", "0.5")
    assert max_failed_chunks(6) == 3
    monkeypatch.setenv("APE_BUILD_MAX_FAILED_SHARE", "1")
    with pytest.raises(ValueError):
        max_failed_chunks(6)


# ---- APG authoring (K1, K2) ---------------------------------------------------------------------------------


def test_a_few_lost_chunks_are_tolerated_and_recorded(offline_env):
    w = _world("F7", "100")  # 58 chunks: one may be lost
    bad = chunk_world(w)[3]
    client = ScriptedChat(lambda i, m, k: completion(None, refusal="I can't help with that.") if bad.text in m[-1]["content"] else completion(_perfect(m)))
    report = asyncio.run(author_world(w, Config(), _llm_author(client, offline_env / "l.jsonl", w.id), concurrency=8, author_id="gpt-6-luna@high"))
    assert list(report["lost_chunks"]) == [bad.id] and "refusal" in report["lost_chunks"][bad.id]
    assert client.calls == 57 + 3, "the refusing chunk is asked three times, the others once"
    assert report["id_coverage"] < 1.0 and 0.9 < report["fact_coverage"] < 1.0
    assert graph_path(Config(), w.id, "authored").is_file()


def test_one_lost_chunk_too_many_fails_the_world_cancels_the_rest_and_the_rerun_pays_only_for_the_rest(offline_env):
    w = _world("F7", "10")  # 6 chunks: none may be lost
    chunks = chunk_world(w)
    bad = chunks[-1]
    client = ScriptedChat(lambda i, m, k: completion("") if bad.text in m[-1]["content"] else completion(_perfect(m)), delay=0.05)
    author = _llm_author(client, offline_env / "l.jsonl", w.id)

    async def first():
        with pytest.raises(AuthoringError, match=rf"1 of 6 chunk\(s\) lost(.|\n)*{re.escape(bad.id)}"):
            await author_world(w, Config(), author, concurrency=2, author_id="gpt-6-luna@high")
        at_failure = client.calls
        await asyncio.sleep(0.3)
        return at_failure

    at_failure = asyncio.run(first())
    assert client.calls == at_failure, "the calls still running were cancelled, none were started after the failure"
    assert not graph_path(Config(), w.id, "authored").exists()
    good_done = sum(1 for m in client.messages[: client.completed] if bad.text not in m[-1]["content"])
    cancelled = [e for e in Ledger(offline_env / "l.jsonl").read() if e.context.get("status") == "cancelled"]
    assert client.calls - client.completed == len(cancelled), "every call cut short is metered"

    rerun = ScriptedChat(lambda i, m, k: completion(_perfect(m)))
    report = asyncio.run(author_world(w, Config(), _llm_author(rerun, offline_env / "l2.jsonl", w.id), author_id="gpt-6-luna@high"))
    assert 0 < rerun.calls == len(chunks) - good_done, "chunks authored before the failure come from the unit cache"
    assert report["lost_chunks"] == {} and report["id_coverage"] == 1.0
    again = ScriptedChat(lambda i, m, k: completion(_perfect(m)))
    asyncio.run(author_world(w, Config(), _llm_author(again, offline_env / "l3.jsonl", w.id), author_id="gpt-6-luna@high"))
    assert again.calls == 0, "a rebuild by the same author pays for nothing"


def test_a_fatal_error_stops_authoring_at_once(offline_env):
    class AuthenticationError(Exception):
        pass

    w = _world("F7", "10")

    def script(i, m, k):
        raise AuthenticationError("invalid api key")

    client = ScriptedChat(script, delay=0.05)

    async def run():
        with pytest.raises(AuthenticationError):
            await author_world(w, Config(), _llm_author(client, offline_env / "l.jsonl", w.id), concurrency=2, author_id="a")
        at_failure = client.calls
        await asyncio.sleep(0.3)
        return at_failure

    at_failure = asyncio.run(run())
    assert client.calls == at_failure < len(chunk_world(w)), "the rest were cancelled: nothing starts after the fatal error"


def test_an_empty_authoring_writes_no_graph_and_ids_are_not_vacuously_covered(offline_env, monkeypatch):
    async def empty(system, user):
        return {"units": []}

    w = _world("F5", "2hop")
    with pytest.raises(AuthoringError, match="lost"):
        asyncio.run(author_world(w, Config(), empty))
    monkeypatch.setenv("APE_BUILD_MAX_FAILED_SHARE", "0.99")
    with pytest.raises(AuthoringError, match="no knowledge units"):
        asyncio.run(author_world(w, Config(), empty))
    assert not graph_path(Config(), w.id, "authored").exists()
    monkeypatch.delenv("APE_BUILD_MAX_FAILED_SHARE")
    report = asyncio.run(author_world(w, Config(), perfect_author))
    assert report["id_coverage"] is None and report["fact_coverage"] == 1.0, "F5 has no spec IDs: coverage is measured on facts"


def test_half_an_authoring_is_measured_as_half(offline_env, monkeypatch):
    w = _world("F7", "100")
    chunks = chunk_world(w)
    keep = {c.text for c in chunks[::2]}

    async def half(system, user):
        out = await perfect_author(system, user)
        return out if any(t in user for t in keep) else {"units": []}

    monkeypatch.setenv("APE_BUILD_MAX_FAILED_SHARE", "0.6")
    report = asyncio.run(author_world(w, Config(), half))
    assert len(report["lost_chunks"]) == len(chunks) // 2 and 0.35 < report["fact_coverage"] < 0.65 and report["id_coverage"] < 0.65


# ---- LightRAG extraction (K2): degraded builds fail, and the rebuild asks only the lost chunks ---------------

EXTRACT_SCRIPT = textwrap.dedent(
    """
    import asyncio, json, re, sys
    from types import SimpleNamespace
    from ape.config import Config
    from ape.lgr.build import build_index
    from ape.llm.build_client import BuildLlm
    from ape.worlds.render import chunk_world
    from ape.worlds.spec import World

    world = World.load(sys.argv[1])
    mode = sys.argv[2]  # clean | degraded
    chunks = chunk_world(world)
    D = "<|#|>"
    calls = {"n": 0, "by_chunk": {}}

    def reply(text):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=None), finish_reason="stop")])

    async def create(**kw):
        calls["n"] += 1
        msgs = kw["messages"]
        user = msgs[-1]["content"]
        if len(msgs) > 2:  # gleaning: history + continue
            return reply("<|COMPLETE|>")
        idx = next((k for k, c in enumerate(chunks) if c.text in user), None)
        if idx is None:
            return reply("A merged summary.")
        calls["by_chunk"][idx] = calls["by_chunk"].get(idx, 0) + 1
        ids = list(dict.fromkeys(re.findall(r"(?:P|X)-[0-9]+", chunks[idx].text)))
        if mode == "degraded" and idx == 1:
            return reply("The excerpt lists several policies; no structured output is needed.")
        if mode == "degraded" and idx == 2:  # an old record format: LightRAG parses nothing, and caches it
            return reply("\\n".join(f'("entity"<|>"{x}"<|>"policy")' for x in ids) + "\\n<|COMPLETE|>")
        return reply("\\n".join(f"entity{D}{x}{D}policy{D}{x} is a company rule." for x in ids) + "\\n<|COMPLETE|>")

    BuildLlm._client_ = lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    try:
        manifest = asyncio.run(build_index(world, "extract", Config()))
        error = None
    except Exception as e:
        manifest, error = None, str(e)
    print("RESULT " + json.dumps({"calls": calls["n"], "by_chunk": calls["by_chunk"], "error": error, "extraction": (manifest or {}).get("extraction")}))
    """
)


def _extract(script: Path, world: Path, mode: str, **env: str) -> dict:
    proc = subprocess.run([sys.executable, str(script), str(world), mode], capture_output=True, text=True, env={**os.environ, **env}, timeout=300, check=False)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_a_degraded_extraction_fails_and_the_rebuild_asks_only_the_lost_chunks(offline_env):
    w = _world("F7", "10")
    path = Config().world_path(w.id)
    script = offline_env / "extract.py"
    script.write_text(EXTRACT_SCRIPT)
    chunks = chunk_world(w)
    wd = index_dir(Config(), w.id, "extract")

    degraded = _extract(script, path, "degraded")
    assert "incomplete" in degraded["error"] and chunks[1].id in degraded["error"] and chunks[2].id in degraded["error"]
    assert degraded["by_chunk"]["1"] == 3, "the prose reply was asked three times"
    assert not (wd / "ape_manifest.json").exists()

    rebuilt = _extract(script, path, "clean")
    assert rebuilt["error"] is None and rebuilt["extraction"]["lost"] == 0
    assert sorted(rebuilt["by_chunk"]) == ["1", "2"], "only the lost chunks are asked again; the others come from the cache"
    assert read_manifest(wd)["extraction"]["failed_docs"] == []

    tolerated = _extract(script, path, "degraded", APE_INDICES=str(offline_env / "indices-tolerant"), APE_BUILD_MAX_FAILED_SHARE="0.5")
    assert tolerated["error"] is None
    assert tolerated["extraction"]["failed_docs"] == [chunks[1].id] and tolerated["extraction"]["empty_docs"] == [chunks[2].id]


# ---- metering and labels (K7) and build health (K2) ----------------------------------------------------------


def test_embedding_client_retries_like_the_build_client_and_build_embeddings_are_attributed(tmp_path, monkeypatch):
    import openai

    from ape.lgr.common import embedding_func
    from ape.llm.embeddings import EmbeddingCache

    seen = {}

    class Client(FakeEmbeddingsClient):
        def __init__(self, **kw):
            seen.update(kw)
            super().__init__()

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    asyncio.run(EmbeddingCache(tmp_path / "e.sqlite", "text-embedding-3-small").embed(["x"]))
    assert seen == {"max_retries": 6}

    ledger = Ledger(tmp_path / "l.jsonl")
    emb = EmbeddingCache(tmp_path / "f.sqlite", "fake", client=FakeEmbeddingsClient(), ledger=ledger)
    asyncio.run(embedding_func(emb, 256, {"world": "w-1", "system": "lightrag"}).func(["an entity description"]))
    assert ledger.read()[-1].context == {"world": "w-1", "system": "lightrag"}


def test_the_d017_builder_is_the_fallback_under_ape_build_fallback(monkeypatch):
    p = load_profile("gate")
    monkeypatch.delenv("APE_BUILD_FALLBACK", raising=False)
    build, fallback, using = builder_roles(p)
    assert (build.model, using) == ("gpt-6-luna", False) and fallback.model == "gpt-6-sol"
    monkeypatch.setenv("APE_BUILD_FALLBACK", "1")
    build, _, using = builder_roles(p)
    assert (build.model, build.reasoning_effort, using) == ("gpt-6-sol", "medium", True)


def test_build_health_flags_lost_chunks_and_low_coverage(offline_env):
    from ape.lgr.build import build_index

    w = _world("F7", "10")
    cfg = Config()
    asyncio.run(author_world(w, cfg, perfect_author))
    asyncio.run(build_index(w, "oracle", cfg))
    assert world_health(w, cfg, "oracle")["warnings"] == []
    report_path = graph_path(cfg, w.id, "authored").with_suffix(".report.json")
    report = json.loads(report_path.read_text())
    report_path.write_text(json.dumps({**report, "lost_chunks": {"c#1": "refusal"}, "fact_coverage": 0.5}))
    warnings = world_health(w, cfg, "oracle")["warnings"]
    assert any("1 lost chunk" in x for x in warnings) and any("fact_coverage 0.500" in x for x in warnings)
