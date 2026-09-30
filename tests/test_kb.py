import asyncio

from ape.kb.baselines import FlatHybrid, Monolith, OracleContext, RandomUnits
from ape.kb.provenance import ChunkIndex, evidence_pr
from ape.worlds.generate import make_world
from ape.worlds.render import chunk_world


def _world():
    w = make_world("F7", "100", "dev", 0, 10)
    return w, chunk_world(w)


def test_evidence_pr():
    assert evidence_pr(["a", "b"], ["b", "c"]) == (0.5, 0.5)
    assert evidence_pr([], ["a"]) == (0.0, 0.0)
    assert evidence_pr(["a"], []) == (0.0, 1.0)


def test_chunk_index_maps_to_facts():
    w, chunks = _world()
    idx = ChunkIndex(chunks)
    assert set(idx.facts([c.id for c in chunks])) == set(w.facts)


def test_monolith_delivers_everything():
    w, chunks = _world()
    ctx = asyncio.run(Monolith(w, chunks).compile("anything", w.tasks[0]))
    assert set(ctx.fact_ids) == set(w.facts)
    assert ctx.tools is None


def test_flat_hybrid_is_deterministic_budgeted_and_usually_finds_the_policy(fake_embeddings):
    w = make_world("F7", "100", "dev", 0, 20)
    chunks = chunk_world(w)
    asyncio.run(fake_embeddings.embed([c.text for c in chunks]))
    arm = FlatHybrid(chunks, fake_embeddings, budget_tokens=2000)
    hits = 0
    for t in w.tasks:
        cust = t.setup["customers"][t.tags["customer_id"]]
        query = f"{t.prompt} region {cust['region']} tier {cust['tier']}"
        a = asyncio.run(arm.compile(query, t))
        assert a.unit_ids == asyncio.run(arm.compile(query, t)).unit_ids  # deterministic
        assert a.tokens <= 2000 + max(len(c.text) for c in chunks)  # budget respected up to one chunk
        hits += t.gold_fact_ids[0] in a.fact_ids
    # Lexical fake embeddings only; the real embedder is measured on dev, not here.
    assert hits / len(w.tasks) >= 0.7, hits


def test_oracle_context_is_exactly_gold():
    w, _ = _world()
    t = w.tasks[0]
    ctx = asyncio.run(OracleContext(w).compile(t.prompt, t))
    assert ctx.fact_ids == t.gold_fact_ids


def test_random_units_seeded_per_task_and_token_matched():
    w, chunks = _world()
    arm = RandomUnits(chunks, target_tokens=500, seed=7)
    a = asyncio.run(arm.compile("", w.tasks[0]))
    assert a.unit_ids == asyncio.run(arm.compile("", w.tasks[0])).unit_ids
    assert a.unit_ids != asyncio.run(arm.compile("", w.tasks[1])).unit_ids
    assert 0 < a.tokens <= 500 + 300
