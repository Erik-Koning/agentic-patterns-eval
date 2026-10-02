"""Provenance, IDs, classify and freshness (RELIABILITY_REVIEW K3-K6).

K3: one delivered-text rule for every arm. It reproduces the chunk mapping exactly for the chunk-based arms and the
oracle mapping for oracle graphs, and it stops LightRAG entities from claiming every fact of every source chunk.
K4: identifiers are normalised before linking and measuring. K5: classify output is repaired or counted, never
silently empty. K6: a stale graph or index is rebuilt or refused, never used."""

import asyncio
import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from apg_core import load_graph
from inspect_ai.model import ModelOutput, get_model
from lightrag import QueryParam

from ape.apg.adapter import compile_context
from ape.apg.arm import ensure_graph, graph_path, tuned
from ape.apg.author import assemble, author_world
from ape.apg.classify import parse_classification
from ape.apg.oracle import build_oracle
from ape.build_quality import lightrag_entity_ids, lightrag_id_coverage
from ape.config import Config, embedding_cache
from ape.kb.baselines import Monolith, OracleContext
from ape.kb.provenance import ChunkIndex, FactMatcher, fact_matcher, segments
from ape.lgr.adapter import LgrArm, _paths, build_lgr_arm, query_params
from ape.lgr.build import build_index
from ape.lgr.common import index_dir, open_rag, read_manifest, write_manifest
from ape.llm.build_client import BuildLlm
from ape.llm.embeddings import EmbeddingCache
from ape.llm.fake import FakeEmbeddingsClient, perfect_author
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world

WORLDS = [
    ("F7", "100", {}),
    ("F7", "100", {"exception_style": "id_only"}),
    ("F7", "100", {"exception_style": "messy"}),
    ("F7", "100", {"relational": False}),
    ("F3", "60", {}),
    ("F5", "2hop", {}),
    ("F1", "32", {}),
    ("F2", "10", {}),
    ("F8", "12", {}),
]


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    for k in ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_EFFORT", "APE_BUILD_FALLBACK"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def _saved(family, level, **kw):
    w = make_world(family, level, "dev", 0, 4, **kw)
    w.save(Config().world_path(w.id))
    return w


# ---- K3: the delivered-text rule ----------------------------------------------------------------------------


@pytest.mark.parametrize("family,level,kw", WORLDS, ids=[f"{f}-{lv}-{'-'.join(map(str, k.values()))}" for f, lv, k in WORLDS])
def test_the_rule_equals_the_chunk_mapping_for_verbatim_arms(family, level, kw):
    w = make_world(family, level, "dev", 0, 4, **kw)
    m = FactMatcher(w)
    for c in chunk_world(w):  # S1, S3s and S7 deliver whole chunks
        assert m.delivered(c.text) == [f for f in w.facts if f in c.fact_ids], c.id
    for t in w.tasks:  # S6 delivers exactly the gold facts' text
        ctx = asyncio.run(OracleContext(w).compile(t.prompt, t))
        assert set(m.delivered(ctx.text)) == set(t.gold_fact_ids)
    mono = asyncio.run(Monolith(w, chunk_world(w)).compile("q", w.tasks[0])) if w.tasks else None
    if mono is not None:
        assert set(m.delivered(mono.text)) == set(m.kb_facts)


def test_the_rule_reproduces_the_oracle_graph_mapping():
    for family, level in (("F7", "100"), ("F3", "60"), ("F5", "2hop")):
        w = make_world(family, level, "dev", 0, 2)
        m = fact_matcher(w)
        for node in build_oracle(w, 2000)["nodes"]:
            text = ((node.get("prompt") or {}).get("slots") or {}).get("knowledge")
            if text:
                assert sorted(m.delivered(text)) == sorted(node["props"]["factIds"]), node["id"]


def test_restatements_count_only_with_every_distinguishing_token():
    w = make_world("F7", "100", "dev", 0, 2)
    m = FactMatcher(w)
    p = w.policies[0]
    o = p.outcome
    entity = {"entity": p.id, "type": "policy",
              "description": f"{p.id} covers {p.domain.replace('_', ' ')} requests in {p.region} from ${p.lo:,} up to ${p.hi:,}: "
                             f"{w.facts[p.fact_id].text.split('the agent must ', 1)[1]}"}
    line = json.dumps(entity)
    assert m.delivered(line) == [p.fact_id], "a restatement with every value counts"
    dropped = json.dumps({**entity, "description": entity["description"].replace(f"within {o['deadline_days']} ", "within ")})
    assert p.fact_id not in m.delivered(dropped), "without its deadline it does not"
    sibling = next(q for q in w.policies if q.domain == p.domain and q.lo == p.lo and q.region != p.region)
    assert sibling.fact_id not in m.delivered(line), "another region's policy of the same band is not credited"
    # A record's single-paragraph values are one segment, a multi-paragraph value one per paragraph (JSON escapes
    # undone), and separate records never combine.
    assert segments(line + "\n" + json.dumps({"entity": "x", "description": "a\n\nb"}))[1:] == ["x", "a", "b"]


def test_a_generic_lightrag_entity_no_longer_claims_every_fact(offline_env, monkeypatch):
    """RELIABILITY_REVIEW r8: an entity extracted from every chunk ("Customer Tier") named all of a world's facts."""
    w = _saved("F7", "100")
    chunks = chunk_world(w)
    d = "<|#|>"

    async def create(**kw):
        user = kw["messages"][-1]["content"]
        if "---Input Text---" not in user:
            text = "<|COMPLETE|>" if len(kw["messages"]) > 2 else "Customer Tier is the loyalty status that policies and exceptions condition on."
        else:
            chunk = next(c for c in chunks if c.text in user)
            ids = list(dict.fromkeys(re.findall(r"\b(?:P|X)-\d+", chunk.text)))
            rows = [f"entity{d}{x}{d}Concept{d}{x} as stated in the excerpt." for x in ids]
            rows += [f"entity{d}Customer Tier{d}Concept{d}Customer loyalty tier referenced by the rules."]
            rows += [f"relation{d}{x}{d}Customer Tier{d}conditions on{d}{x} depends on the customer tier." for x in ids]
            text = "\n".join(rows) + "\n<|COMPLETE|>"
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5), choices=[SimpleNamespace(message=SimpleNamespace(content=text, refusal=None), finish_reason="stop")])

    monkeypatch.setattr(BuildLlm, "_client_", lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    cfg = Config()
    asyncio.run(build_index(w, "extract", cfg))

    async def query():
        async def kg(prompt, system_prompt=None, history_messages=None, **kw):
            return '{"high_level_keywords": ["customer tier"], "low_level_keywords": ["Customer Tier"]}'

        wd = index_dir(cfg, w.id, "extract")
        rag = await open_rag(wd, w.id, kg, embedding_cache(cfg), query_time=True)
        try:
            arm = LgrArm("LGR-q", False, rag, fact_matcher(w), query_params(cfg.lgr_budget_tokens), read_manifest(wd))
            ctx = await arm.compile(w.tasks[0].prompt, w.tasks[0])
            data = (await rag.aquery_data(w.tasks[0].prompt, QueryParam(**arm.params, hl_keywords=["customer tier"], ll_keywords=["Customer Tier"]))).get("data", {})
        finally:
            await rag.finalize_storages()
        return ctx, data

    ctx, data = asyncio.run(query())
    index = ChunkIndex(chunks)
    old = index.facts(list(dict.fromkeys(p for rec in (*data["entities"], *data["relationships"], *data["chunks"]) for p in _paths(rec) if p in index.by_id)))
    verbatim = [f for f in w.facts if fact_matcher(w).delivered(w.facts[f].text) and w.facts[f].text[:60] in ctx.text]
    assert len(old) > 100, "the old mapping credited (nearly) every fact of the world"
    assert set(ctx.fact_ids) <= set(old) and set(verbatim) <= set(ctx.fact_ids) and len(ctx.fact_ids) < len(old) / 3
    assert ctx.meta["lightrag"]["source_chunks"] >= len(chunks) // 2


def test_lightrag_keyword_fallback_is_recorded_and_its_units_match_the_text(offline_env):
    w = _saved("F7", "10")
    cfg = Config()
    asyncio.run(build_index(w, "oracle", cfg))

    async def run(reply):
        async def kg(prompt, system_prompt=None, history_messages=None, **kw):
            return reply

        wd = index_dir(cfg, w.id, "oracle")
        rag = await open_rag(wd, w.id, kg, embedding_cache(cfg), query_time=True)
        try:
            return await LgrArm("LGRo-q", False, rag, fact_matcher(w), query_params(cfg.lgr_budget_tokens), read_manifest(wd)).compile(w.tasks[0].prompt, w.tasks[0])
        finally:
            await rag.finalize_storages()

    for reply in ("", "I'm sorry, I can't help with that.", '{"high_level_keywords": [], "low_level_keywords": []}'):
        ctx = asyncio.run(run(reply))
        lg = ctx.meta["lightrag"]
        assert lg["keyword_fallback"] and lg["keyword_error"] and lg["keywords"]["low"] == [w.tasks[0].prompt]
        assert sum(lg["counts"]) > 0 and ctx.fact_ids, "the fallback context's units and facts are reported, not zero"
    ok = asyncio.run(run('{"high_level_keywords": ["refund policy"], "low_level_keywords": ["EU"]}'))
    assert ok.meta["lightrag"]["keyword_fallback"] is False and ok.meta["lightrag"]["keyword_error"] is None


# ---- K4: identifiers ----------------------------------------------------------------------------------------


def test_prefixed_and_cased_ids_link_and_cover_like_bare_ones():
    w = make_world("F7", "100", "dev", 0, 2)
    chunks = chunk_world(w)
    titles = {d.id: d.title for d in w.documents}
    base = [asyncio.run(perfect_author("", f"Document: {titles[c.doc_id]}\n\nExcerpt:\n{c.text}"))["units"] for c in chunks]

    def prefixed(x):
        return ("Policy " if x.startswith("P-") else "Exception ") + x

    def links(doc):
        return sum(len(n.get("bring", [])) for n in doc["nodes"])

    doc0, rep0 = assemble(w, chunks, base, 2000)
    variants = {
        "prefixed": [[{**u, "declares": [prefixed(x) for x in u["declares"]], "references": [prefixed(x) for x in u["references"]]} for u in us] for us in base],
        "mixed": [[{**u, "declares": [prefixed(x) for x in u["declares"]]} for u in us] for us in base],
        "lower": [[{**u, "declares": [x.lower() for x in u["declares"]]} for u in us] for us in base],
    }
    for name, units in variants.items():
        doc, rep = assemble(w, chunks, units, 2000)
        assert (rep["id_coverage"], rep["unresolved_references"], links(doc)) == (1.0, 0, links(doc0)), name
    names = {"x": {"entity_names": ["Policy " + p.id for p in w.policies] + [x.id.lower() for x in w.exceptions]}}
    assert lightrag_id_coverage(w, lightrag_entity_ids(names)) == 1.0


# ---- K5: classify -------------------------------------------------------------------------------------------


def test_classify_output_is_repaired_or_counted():
    ok = json.dumps({"matches": [{"nodeId": "u00060", "confidence": 0.9, "reason": "r"}]})
    assert parse_classification(ok) == ([{"nodeId": "u00060", "confidence": 0.9, "reason": "r"}], None, [])
    m, err, rep = parse_classification("```json\n" + json.dumps({"matches": [{"nodeId": "u00060: Refund policy (EU)", "confidence": 90, "reason": "r"}]}) + "\n```")
    assert (m[0]["nodeId"], m[0]["confidence"], err, rep) == ("u00060", 0.9, None, ["fence", "id_prefix", "percent"])
    assert parse_classification("")[1] is not None


async def test_classify_failures_are_counted_in_the_compile(fake_embeddings):
    w = make_world("F7", "100", "dev", 0, 2)
    from ape.apg.arm import embed_graph

    doc = await embed_graph(build_oracle(w, 2000), fake_embeddings, {})
    g = load_graph(tuned(doc))
    task = w.tasks[0]

    def kg(text):
        return get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", text)], memoize=False)

    probe = await compile_context(g, task.prompt, kg('{"matches": []}'), fake_embeddings, 2000)
    leaf = probe.route["shortlist"][0]
    fenced = await compile_context(g, task.prompt, kg("```json\n" + json.dumps({"matches": [{"nodeId": leaf, "confidence": 0.9, "reason": "r"}]}) + "\n```"), fake_embeddings, 2000)
    assert (fenced.classify_error, fenced.classify_fallback, fenced.classify_repaired) == (None, False, ["fence"]) and fenced.contributors
    empty = await compile_context(g, task.prompt, kg(""), fake_embeddings, 2000)
    assert empty.classify_error and empty.classify_fallback
    unknown = await compile_context(g, task.prompt, kg(json.dumps({"matches": [{"nodeId": "no-such-node", "confidence": 0.9, "reason": "r"}]})), fake_embeddings, 2000)
    assert (unknown.classify_unknown_ids, unknown.classify_fallback, unknown.classify_error) == (1, True, None)


# ---- K6: freshness ------------------------------------------------------------------------------------------


class _Wide:
    """A different embedding model: 1536-d vectors (scripted, no network)."""

    def __init__(self):
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input, encoding_format=None):
        return SimpleNamespace(data=[SimpleNamespace(embedding=np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(1536).astype(np.float32).tolist()) for t in input], usage=None)


def test_stale_or_unlabelled_oracle_graphs_are_rebuilt(offline_env):
    w = _saved("F7", "10")
    cfg = Config()
    path = graph_path(cfg, w.id, "oracle")
    fake = embedding_cache(cfg)
    doc = asyncio.run(ensure_graph(w, "oracle", cfg, fake))
    assert doc["meta"] == {"worldHash": w.content_hash(), "author": "oracle", "embeddingModel": "fake-bow", "embeddingDim": 256}
    legacy = {k: v for k, v in doc.items() if k != "meta"}  # as written before freshness checks
    path.write_text(json.dumps(legacy))
    assert asyncio.run(ensure_graph(w, "oracle", cfg, fake))["meta"]["worldHash"] == w.content_hash()
    wide = EmbeddingCache(cfg.cache_dir / "wide.sqlite", "text-embedding-3-small", client=_Wide())
    rebuilt = asyncio.run(ensure_graph(w, "oracle", cfg, wide))
    assert rebuilt["meta"]["embeddingModel"] == "text-embedding-3-small" and rebuilt["meta"]["embeddingDim"] == 1536


def test_stale_authored_graphs_are_refused_and_vector_mismatches_fail_loudly(offline_env):
    w = _saved("F7", "10")
    cfg = Config()
    asyncio.run(author_world(w, cfg, perfect_author, author_id="gpt-6-luna@high"))
    path = graph_path(cfg, w.id, "authored")
    doc = json.loads(path.read_text())
    assert asyncio.run(ensure_graph(w, "authored", cfg, embedding_cache(cfg)))["meta"]["author"] == "gpt-6-luna@high"
    path.write_text(json.dumps({**doc, "meta": {**doc["meta"], "author": "gpt-6-sol@medium"}}))
    with pytest.raises(RuntimeError, match="authored by 'gpt-6-sol@medium'"):
        asyncio.run(ensure_graph(w, "authored", cfg, embedding_cache(cfg)))
    path.write_text(json.dumps({**doc, "meta": {**doc["meta"], "worldHash": "old"}}))
    with pytest.raises(RuntimeError, match="another version of the world"):
        asyncio.run(ensure_graph(w, "authored", cfg, embedding_cache(cfg)))
    wide = EmbeddingCache(cfg.cache_dir / "wide.sqlite", "text-embedding-3-small", client=_Wide())
    kg = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", '{"matches": []}')], memoize=False)
    with pytest.raises(ValueError, match="1536-d but the graph's node vectors are 256-d"):
        asyncio.run(compile_context(load_graph(doc), w.tasks[0].prompt, kg, wide, 2000))


def test_a_lightrag_index_built_with_other_settings_is_refused(offline_env):
    w = _saved("F7", "10")
    cfg = Config()
    asyncio.run(build_index(w, "oracle", cfg))
    wd = index_dir(cfg, w.id, "oracle")
    write_manifest(wd, {**read_manifest(wd), "embedding_model": "text-embedding-3-small"})
    with pytest.raises(RuntimeError, match="embedding_model is 'text-embedding-3-small', expected 'fake-bow'"):
        asyncio.run(build_lgr_arm("LGRo-q", w, cfg))


def test_fake_embeddings_of_a_graph_and_a_query_agree(fake_embeddings):
    """Regression guard for the dimension check: the fake backend's graph and query vectors share a dimension."""
    w = make_world("F3", "5", "dev", 0, 1)
    from ape.apg.arm import embed_graph

    doc = asyncio.run(embed_graph(build_oracle(w, 2000), fake_embeddings, {}))
    kg = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", '{"matches": []}')], memoize=False)
    asyncio.run(compile_context(load_graph(doc), w.tasks[0].prompt, kg, fake_embeddings, 2000))
    assert isinstance(FakeEmbeddingsClient(), FakeEmbeddingsClient)
