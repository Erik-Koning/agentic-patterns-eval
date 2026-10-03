"""RELIABILITY_REVIEW 2, KG and PC1 (the review of aa6f09c): regression tests built from the reviewer's repros.

- the bge embedder in LightRAG (LightRAG deep-copies its config; the model holds locks);
- the anchor index: a failed or empty extraction never gets a manifest, and an unhealthy or damaged index is refused;
- bge internals: no cache-eviction race, a dedicated executor thread;
- provenance: kind-level template words, folded inflections and number words, no pooling across `<SEP>` fragments or
  distant sentences;
- freshness: index files verified against their hash, the authoring prompt in authored graphs' meta;
- the anchor scorers: a truncated judge reply is a NaN sample, and one scorer's failure cancels the other.

The real bge model is used only with APE_TEST_BGE=1 (a ~1.3 GB download, normally already in the HF cache).
"""

import asyncio
import copy
import json
import os
import random
import re
import threading
from collections import defaultdict

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from ape.anchor import bge, ragas
from ape.anchor import graphragbench as gb
from ape.anchor.bge import BgeOnnx, anchor_embedder, fake_embedder
from ape.anchor.fixture import CORPUS, fake_extractor
from ape.config import Config
from ape.kb.provenance import FactMatcher
from ape.llm.build_client import BuildResponseError
from ape.worlds.generate import make_world


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


class _RealEmbeddings:
    embeddings_backend = "openai"  # anchor_embedder -> the (lazily loaded) bge model


# ---- finding 1: bge inside LightRAG ----------------------------------------------------------------------------


def test_the_bge_embedder_survives_lightrags_config_deepcopy(offline_env):
    """p8: LightRAG.__post_init__ runs dataclasses.asdict(self), which deep-copied the bound method and, through it,
    the model's threading.Lock: TypeError before any LLM call. Nothing here loads the model."""
    emb = anchor_embedder(_RealEmbeddings())
    assert copy.deepcopy(emb) is emb and copy.deepcopy(bge.bge()) is bge.bge()
    copy.deepcopy(EmbeddingFunc(embedding_dim=emb.dim, max_token_size=8192, func=emb.lightrag_func()))

    async def no_llm(*_a, **_k):
        raise AssertionError("no LLM call expected")

    async def go():
        rag = await gb.open_anchor_rag(offline_env / "anchor-wd", no_llm, emb, query_time=True)
        try:
            assert rag.default_embedding_timeout >= 600 and rag.embedding_func_max_async <= 2
        finally:
            await rag.finalize_storages()

    asyncio.run(go())
    assert bge.bge()._session is None, "constructing LightRAG must not load the model"


# ---- finding 2: the anchor index's health ----------------------------------------------------------------------


def _bench(name: str, corpus: str = CORPUS) -> gb.Bench:
    return gb.Bench(questions=[], documents=[{"corpus_name": name, "context": corpus}])


def test_an_anchor_build_with_a_failed_chunk_raises_and_writes_no_manifest(offline_env):
    """p6: one chunk's extraction kept failing, LightRAG marked the one corpus document FAILED with 0 entities, and
    the build still wrote a manifest that index_manifest accepted. A unique document name keeps this build apart
    from other tests' in-process LightRAG state."""
    paras = [f"Section {i}. Drug D{i} treats condition C{i} in adult patients. " * 60 for i in range(6)]
    bench = _bench("MedicalFailing", "\n\n".join(paras))
    extract = fake_extractor()

    async def llm(prompt, system_prompt=None, history_messages=None, **kw):
        if "Drug D3" in prompt and "---Input Text---" in prompt:
            raise BuildResponseError("fake LightRAG extraction", ["empty reply"] * 3)
        return await extract(prompt, system_prompt, history_messages, **kw)

    cfg = Config()
    with pytest.raises(gb.AnchorBuildError, match=r"MedicalFailing: status failed, 0 entities"):
        asyncio.run(gb.build_index(bench, cfg, llm, "gpt-4o-mini"))
    assert not (gb.working_dir(cfg) / "ape_manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        gb.index_manifest(cfg, bench)


def test_an_unhealthy_or_damaged_anchor_index_is_refused(offline_env):
    # Its own text: LightRAG skips a document whose content another build in this process already inserted.
    bench, cfg = _bench("MedicalHealthy", CORPUS.replace("guideline", "handbook")), Config()
    manifest = asyncio.run(gb.build_index(bench, cfg, fake_extractor(), None))
    assert manifest["extraction"]["healthy"] and manifest["extraction"]["documents"]["MedicalHealthy"]["entities"] > 0
    assert gb.index_manifest(cfg, bench)["index_hash"] == manifest["index_hash"]
    path = gb.working_dir(cfg) / "ape_manifest.json"
    path.write_text(json.dumps({k: v for k, v in manifest.items() if k != "extraction"}))  # built before the check
    with pytest.raises(RuntimeError, match="no healthy extraction"):
        gb.index_manifest(cfg, bench)
    path.write_text(json.dumps(manifest))
    store = next(p for p in gb.working_dir(cfg).rglob("kv_store_*.json"))
    store.write_text(store.read_text() + " ")
    with pytest.raises(RuntimeError, match="index_hash"):
        gb.index_manifest(cfg, bench)


# ---- bge internals ---------------------------------------------------------------------------------------------


def test_pooled_vectors_survive_cache_eviction_under_concurrency(monkeypatch):
    """A concurrent insert could evict an entry between this call's insert and its read: KeyError."""
    monkeypatch.setattr(bge, "CACHE_ITEMS", 1)
    m = BgeOnnx()

    def hidden(texts):
        return [np.full((2, bge.DIM), float(len(t)), dtype=np.float32) for t in texts]

    monkeypatch.setattr(m, "hidden_states", hidden)
    assert m.documents(["a", "bb", "ccc"]).shape == (3, bge.DIM)
    errors: list[BaseException] = []

    def worker(i):
        try:
            for j in range(50):
                m.mean_pooled([f"t{i}-{j}-{k}" for k in range(4)])
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_anchor_embeddings_run_on_their_own_thread():
    names = []
    emb = fake_embedder()
    orig = emb._mean

    def mean(texts):
        names.append(threading.current_thread().name)
        return orig(texts)

    emb._mean = mean
    asyncio.run(emb.retrieval(["Basal cell carcinoma"]))
    assert names and names[0].startswith("ape-anchor-embed"), names


# ---- findings 3 and 4: provenance -------------------------------------------------------------------------------


def _rate(m: FactMatcher, items) -> float:
    return sum(f in m.delivered(t) for f, t in items) / len(items)


def test_restatements_that_keep_every_value_are_credited():
    """p3: exception boilerplate (a minority kind's template) and inflected verbs or spelled-out numbers were
    treated as distinguishing, so faithful restatements got no credit (0/300, 0/1000, 13/907, 0/60)."""
    w = make_world("F7", "100", "dev", 1, 4)
    m = FactMatcher(w)
    exc = [(x.fact_id, w.facts[x.fact_id].text.replace("(amends Policy", "(modifies Policy").replace("does not apply as written: the agent must instead", "is overridden: the agent must")) for x in w.exceptions]
    assert _rate(m, exc) == 1.0
    inflected = [(p.fact_id, w.facts[p.fact_id].text.replace("must approve the request", "must have the request approved").replace("must deny the request", "must have the request denied").replace("must escalate the request", "must have the request escalated")) for p in w.policies]
    assert _rate(m, inflected) == 1.0
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    spelled = [(p.fact_id, re.sub(r"within (\d) business", lambda g: f"within {words[int(g.group(1))]} business", w.facts[p.fact_id].text)) for p in w.policies if p.outcome.get("deadline_days", 0) < 10]
    assert _rate(m, spelled) == 1.0
    w3 = make_world("F3", "60", "dev", 1, 4)
    m3 = FactMatcher(w3)
    assert _rate(m3, [(t.fact_id, w3.facts[t.fact_id].text.replace("Parameters:", "Arguments:")) for t in w3.tools]) == 1.0
    # Still strict on content: a restatement missing a value is no evidence, and siblings are not credited.
    p = w.policies[0]
    assert p.fact_id not in m.delivered(w.facts[p.fact_id].text.replace(f"${p.lo:,}", "$X"))


def test_merged_lightrag_descriptions_never_jointly_credit_a_foreign_fact():
    """p4: LightRAG joins up to 7 descriptions with <SEP> in one JSON value; read as one segment, their pooled tokens
    credited facts none of them states (0.10-0.12 per 7-fragment record on F7-1000)."""
    w = make_world("F7", "100", "dev", 1, 4)
    m = FactMatcher(w)
    by_doc = defaultdict(list)
    for d in w.documents:
        for p in d.paragraphs:
            by_doc[d.id].extend(p.fact_ids)
    rng = random.Random(0)
    foreign = credited = 0
    for _ in range(300):
        group = rng.sample(by_doc[rng.choice([d for d in by_doc if len(by_doc[d]) >= 7])], 7)
        desc = "<SEP>".join(w.facts[f].text.replace(". ", ". Note: ", 1) for f in group)  # not verbatim
        got = set(m.delivered(json.dumps({"entity": "X", "type": "policy", "description": desc})))
        foreign += len(got - set(group))
        credited += len(got & set(group))
    assert foreign == 0 and credited == 300 * 7


def test_a_restatement_must_fit_in_one_window_of_sentences():
    w = make_world("F7", "10", "dev", 1, 4)
    m = FactMatcher(w)
    p = w.policies[0]
    first, second = w.facts[p.fact_id].text.split(". ", 1)
    assert p.fact_id in m.delivered(f"{first}. Unrelated remark. {second}"), "within the window"
    filler = " ".join(f"Unrelated remark number {i}." for i in range(m.window + 1))
    assert p.fact_id not in m.delivered(f"{first}. {filler} {second}"), "tokens from distant sentences do not pool"


# ---- finding 5: freshness ---------------------------------------------------------------------------------------


def test_index_current_verifies_the_index_files(offline_env):
    from ape.lgr.build import build_index, index_current
    from ape.lgr.common import index_dir

    w = make_world("F7", "10", "dev", 0, 2)
    cfg = Config()
    asyncio.run(build_index(w, "oracle", cfg))
    assert index_current(w, "oracle", cfg)
    assert index_current(make_world("F7", "10", "dev", 0, 6), "oracle", cfg), "more tasks: same index"
    store = next(index_dir(cfg, w.id, "oracle").rglob("kv_store_*.json"))
    store.write_text(store.read_text() + " ")
    assert not index_current(w, "oracle", cfg)


def test_a_new_authoring_prompt_marks_authored_graphs_stale(offline_env, monkeypatch):
    from ape.apg import author
    from ape.apg.arm import graph_path, graph_problems
    from ape.config import embedding_cache
    from ape.llm.fake import perfect_author

    w = make_world("F3", "5", "dev", 0, 2)
    cfg = Config()
    asyncio.run(author.author_world(w, cfg, perfect_author, author_id=author.FAKE_AUTHOR_ID))
    assert author.authored_graph_current(w, cfg, author.FAKE_AUTHOR_ID)
    monkeypatch.setattr(author, "AUTHOR_SYSTEM", author.AUTHOR_SYSTEM + " Be brief.")
    assert not author.authored_graph_current(w, cfg, author.FAKE_AUTHOR_ID)
    doc = json.loads(graph_path(cfg, w.id, "authored").read_text())
    assert any("authoring prompt" in p for p in graph_problems(doc, w, embedding_cache(cfg), "authored"))


# ---- findings 7 and 9: the anchor scorers -----------------------------------------------------------------------


def test_a_truncated_judge_reply_is_a_nan_sample_for_both_scorers():
    from ape.tasks.anchor_graphragbench import score_both

    async def truncated(prompt):
        raise ragas.Truncated("judge reply truncated (stop reason max_tokens)")

    for qtype in ("Fact Retrieval", "Creative Generation"):
        matched, current = asyncio.run(score_both("Q?", "An answer.", "The reference.", qtype, truncated, fake_embedder()))
        assert matched["accuracy"] is None and matched["parse_failed"], qtype
        assert current["accuracy"] is None and current["excluded"] and current["classification_parse_failed"], qtype


def test_one_scorers_failure_cancels_the_others_judge_calls():
    from ape.tasks.anchor_graphragbench import score_both

    started, finished = [], []

    async def generate(prompt):
        started.append(prompt)
        if "Generated Statements:" in prompt:  # only the official scorer's statement prompt ends this way
            raise RuntimeError("judge exploded")
        await asyncio.sleep(0.5)
        finished.append(prompt)
        return "{}"

    with pytest.raises(RuntimeError, match="judge exploded"):
        asyncio.run(score_both("Q?", "An answer.", "The reference.", "Fact Retrieval", generate, fake_embedder()))
    assert any("Generated Statements:" not in p for p in started), "the vendored-RAGAS scorer had a call in flight"
    assert finished == [], "and it was cancelled, not left running"


# ---- the real model (opt-in) -----------------------------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("APE_TEST_BGE"), reason="real bge-large-en-v1.5 (~1.3 GB): set APE_TEST_BGE=1")
def test_a_tiny_anchor_index_builds_and_answers_with_real_bge(offline_env, monkeypatch):
    monkeypatch.setenv("APE_EMBEDDINGS", "openai")  # the anchor's embeddings: real bge, loaded from the HF cache
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = Config()
    bench = _bench("MedicalBge", " ".join(CORPUS.split(". ")[:30]))
    manifest = asyncio.run(gb.build_index(bench, cfg, fake_extractor(), None))
    assert manifest["extraction"]["healthy"] and manifest["embedding_model"].startswith(bge.REPO)

    async def query():
        from lightrag import QueryParam

        async def kw(*_a, **_k):
            return '{"high_level_keywords": ["skin cancer"], "low_level_keywords": ["melanoma"]}'

        rag = await gb.open_anchor_rag(gb.working_dir(cfg), kw, anchor_embedder(cfg), query_time=True)
        try:
            return await rag.aquery_data("How is melanoma treated?", QueryParam(mode="hybrid", top_k=5))
        finally:
            await rag.finalize_storages()

    assert asyncio.run(query()).get("status") == "success"
