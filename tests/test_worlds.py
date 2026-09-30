from collections import Counter

import pytest

from ape.llm.tokens import count_tokens
from ape.worlds.generate import make_world, solve
from ape.worlds.render import chunk_world, corpus_text
from ape.worlds.spec import World

CASES = [("F7", "10"), ("F7", "1000"), ("F3", "5"), ("F3", "60"), ("F5", "1hop"), ("F5", "2hop")]


@pytest.mark.parametrize("family,level", CASES)
def test_generation_is_deterministic_per_seed(family, level):
    a = make_world(family, level, "dev", 0, 20)
    b = make_world(family, level, "dev", 0, 20)
    c = make_world(family, level, "dev", 1, 20)
    assert a.content_hash() == b.content_hash()
    assert a.content_hash() != c.content_hash()


def test_splits_never_share_seeds():
    assert make_world("F7", "10", "dev", 0, 5).seed != make_world("F7", "10", "test", 0, 5).seed


@pytest.mark.parametrize("family,level", CASES)
def test_reference_solver_matches_gold(family, level):
    w = make_world(family, level, "dev", 0, 60)
    for t in w.tasks:
        assert solve(w, t) == t.gold, t.id


@pytest.mark.parametrize("family,level", CASES)
def test_every_fact_rendered_once_and_chunks_keep_paragraphs_whole(family, level):
    w = make_world(family, level, "dev", 0, 10)
    rendered = Counter(f for d in w.documents for p in d.paragraphs for f in p.fact_ids)
    assert set(rendered) == set(w.facts)
    assert set(rendered.values()) == {1}
    chunks = chunk_world(w)
    in_chunks = Counter(f for c in chunks for f in c.fact_ids)
    assert in_chunks == rendered
    assert len({c.id for c in chunks}) == len(chunks)
    for t in w.tasks:
        assert set(t.gold_fact_ids) <= set(w.facts)


def test_kb_sizes_match_levels():
    assert len(make_world("F7", "10", "dev", 0, 1).policies) == 10
    assert len(make_world("F7", "100", "dev", 0, 1).policies) == 100
    assert len(make_world("F7", "1000", "dev", 0, 1).policies) == 1000
    for level, n in [("5", 5), ("20", 20), ("60", 60)]:
        assert len(make_world("F3", level, "dev", 0, 1).tools) == n


def test_f7_relational_cases_and_independent_variant():
    w = make_world("F7", "100", "dev", 0, 200)
    cases = Counter(t.tags["case"] for t in w.tasks)
    assert set(cases) == {"exception_applies", "exception_not_applicable", "no_exception"}
    ind = make_world("F7", "100", "dev", 0, 50, relational=False)
    assert not ind.exceptions and {t.tags["case"] for t in ind.tasks} == {"no_exception"}


def test_f7_largest_monolith_fits_nominal_window():
    w = make_world("F7", "1000", "dev", 0, 1)
    assert count_tokens(corpus_text(w)) <= 0.8 * 128_000


def test_world_round_trips(tmp_path):
    w = make_world("F3", "20", "dev", 0, 5)
    w.save(tmp_path / "w.json")
    assert World.load(tmp_path / "w.json").content_hash() == w.content_hash()
