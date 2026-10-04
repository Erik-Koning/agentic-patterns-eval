"""Registry families (worlds/gen_registry.py): F1 breadth aggregation and F2 dependency chains."""

import asyncio
import json
import re
from collections import Counter

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model
from inspect_ai.util._store import Store, init_subtask_store

from ape.build import build
from ape.kb.baselines import OracleContext
from ape.llm.mock_agent import mock_agent
from ape.llm.tokens import count_tokens
from ape.scorers.success import is_success
from ape.scorers.taxonomy import f1_error, f1_items, f2_error, f2_prefix
from ape.tasks.main import main_study
from ape.worlds.env_tools import ANSWER, FAULTS_FIRED, always_on, build_tools
from ape.worlds.gen_registry import LEVELS, RATINGS, render_record, turn_cap
from ape.worlds.generate import SPLIT_SEED_BASE, make_world, solve
from ape.worlds.render import chunk_world, corpus_text
from ape.worlds.spec import TaskItem, World

CASES = [(f, lv) for f, levels in LEVELS.items() for lv in levels]


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def _call(tool, **kwargs):
    return asyncio.run(tool.tool(**kwargs))


# ---------- generation ----------


@pytest.mark.parametrize("family,level", CASES)
def test_worlds_are_byte_identical_per_seed(family, level, tmp_path):
    a, b, c = (make_world(family, level, "dev", i, 6) for i in (0, 0, 1))
    a.save(tmp_path / "a.json")
    b.save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert a.content_hash() != c.content_hash()
    assert World.load(tmp_path / "a.json").content_hash() == a.content_hash()


def test_splits_are_seed_disjoint_and_levels_share_one_registry():
    dev, test = make_world("F1", "2", "dev", 0, 2), make_world("F1", "2", "test", 0, 2)
    assert (dev.seed, test.seed) == (SPLIT_SEED_BASE["dev"], SPLIT_SEED_BASE["test"])
    assert dev.entities["suppliers"].keys().isdisjoint(test.entities["suppliers"].keys())
    # Same split and seed: every family and level sees the same KB and records (the knob is the only difference).
    worlds = [make_world(f, lv, "dev", 0, 2) for f, lv in CASES]
    assert len({corpus_text(w) for w in worlds}) == 1
    assert len({json.dumps(w.entities["suppliers"], sort_keys=True) for w in worlds}) == 1


@pytest.mark.parametrize("family,level", CASES)
def test_reference_solver_scores_every_task(family, level):
    w = make_world(family, level, "dev", 0, 30)
    for t in w.tasks:
        assert solve(w, t) == t.gold, t.id
        assert is_success(t, t.gold, []), t.id


@pytest.mark.parametrize("family,level", [("F1", "32"), ("F2", "10")])
def test_oracle_context_holds_every_gold_fact(family, level):
    w = make_world(family, level, "dev", 0, 10)
    oracle = OracleContext(w)
    for t in w.tasks:
        ctx = asyncio.run(oracle.compile(t.prompt, t))
        assert set(t.gold_fact_ids) <= set(ctx.fact_ids) and set(t.gold_fact_ids) <= set(w.facts)
        for f in t.gold_fact_ids:
            assert w.facts[f].text in ctx.text


@pytest.mark.parametrize("family,level", [("F1", "8"), ("F2", "5")])
def test_paraphrases_are_equivalent_and_carry_the_same_parameters(family, level):
    w = make_world(family, level, "dev", 0, 10)
    for t in w.tasks:
        paras = t.tags["paraphrases"]
        assert len(paras) == 2 and t.prompt not in paras and len(set(paras)) == 2
        for text in [t.prompt, *paras]:
            if family == "F1":
                assert all(sid in text for sid in t.tags["suppliers"]) and t.tags["quarter"] in text and str(t.tags["n"]) in text
            else:
                assert t.tags["start"] in text and f" {t.tags['k']} " in text and t.tags["dispute"] in text


def test_each_fact_appears_r_times_and_chunks_keep_paragraphs_whole():
    w = make_world("F2", "10", "dev", 0, 4)
    rendered = Counter(f for d in w.documents for p in d.paragraphs for f in p.fact_ids)
    assert dict(rendered) == w.entities["redundancy"]
    assert set(rendered.values()) == {1, 2}
    assert Counter(f for c in chunk_world(w) for f in c.fact_ids) == rendered
    assert any(not p.fact_ids for d in w.documents for p in d.paragraphs), "distractor paragraphs exist"


def test_corpus_fits_the_nominal_window_at_every_level():
    for family, level in CASES:
        assert count_tokens(corpus_text(make_world(family, level, "dev", 0, 1))) <= 0.8 * 128_000


def test_records_are_bulky_and_lead_with_the_fields_a_step_needs():
    w = make_world("F1", "32", "dev", 0, 1)
    for rec in w.entities["suppliers"].values():
        text = render_record(rec)
        assert 300 <= count_tokens(text) <= 600
        head = text[: text.index('"contacts"')]
        assert '"segment"' in head and '"escalation_code"' in head
    # Records never name another supplier, and the KB never holds a record.
    corpus = corpus_text(w)
    for sid, rec in w.entities["suppliers"].items():
        others = set(re.findall(r"SUP-\d+", render_record(rec))) - {sid}
        assert not others
        assert json.dumps(rec["scorecard"]) not in corpus


# ---------- F1 ----------


def test_f1_subtasks_compose_to_the_gold_and_stay_out_of_the_prompt():
    w = make_world("F1", "32", "dev", 0, 12)
    labels = Counter()
    for t in w.tasks:
        subs = t.tags["subtasks"]
        assert [s["id"] for s in subs] == t.tags["suppliers"] and len(subs) == 32 == len(set(t.tags["suppliers"]))
        assert {s["id"]: s["gold"] for s in subs} == t.gold["ratings"]
        assert t.gold_fact_ids == list(dict.fromkeys(f for s in subs for f in s["gold_fact_ids"]))
        for s in subs:
            seg = w.entities["suppliers"][s["id"]]["segment"]
            assert s["gold_fact_ids"] == [f"f-R-{seg}"]
            assert s["question"] not in t.prompt
        assert not any(r in t.prompt for r in RATINGS)
        labels.update(t.gold["ratings"].values())
    assert set(labels) == set(RATINGS) and min(labels.values()) > 0.2 * sum(labels.values())


def test_f1_scoring_exact_and_partial():
    w = make_world("F1", "8", "dev", 0, 1)
    t = w.tasks[0]
    gold = t.gold["ratings"]
    assert is_success(t, {"ratings": {k.lower(): v.upper() for k, v in gold.items()}}, [])  # case-insensitive
    one_wrong = dict(gold)
    k0 = next(iter(one_wrong))
    one_wrong[k0] = next(r for r in RATINGS if r != gold[k0])
    assert not is_success(t, {"ratings": one_wrong}, [])
    assert f1_items({"ratings": one_wrong}, t.gold)["item_f1"] == pytest.approx(7 / 8)
    assert f1_error(t, {"ratings": one_wrong}) == "wrong_ratings"
    missing = {k: v for k, v in gold.items() if k != k0}
    assert not is_success(t, {"ratings": missing}, [])
    assert f1_items({"ratings": missing}, t.gold) | {} == pytest.approx({"item_precision": 1.0, "item_recall": 7 / 8, "item_f1": 2 * 7 / 8 / (1 + 7 / 8), "items_correct": 7, "items": 8})
    assert f1_error(t, {"ratings": missing}) == "missing_items"
    assert not is_success(t, {"ratings": {**gold, "SUP-1": "approved"}}, []) and f1_error(t, {"ratings": {**gold, "SUP-1": "approved"}}) == "extra_items"
    assert f1_error(t, None) == "no_answer" and f1_error(t, {"ratings": "x"}) == "malformed_answer"
    assert f1_error(t, t.gold) == "correct"


# ---------- F2 ----------


def test_f2_chains_follow_the_registry_and_each_key_appears_only_after_the_previous_hop():
    w = make_world("F2", "10", "dev", 0, 12)
    sup, routes = w.entities["suppliers"], w.entities["routes"]
    records = {sid: render_record(r) for sid, r in sup.items()}
    kb_paragraphs = [p for d in w.documents for p in d.paragraphs]
    for t in w.tasks:
        chain, codes, start = t.gold["chain"], t.tags["codes"], t.tags["start"]
        assert len(chain) == 10 == len(set(chain)) and start not in chain and t.gold["final"] == chain[-1]
        prev = start
        for code, nxt in zip(codes, chain, strict=True):
            assert sup[prev]["escalation_code"] == code and routes[code] == nxt
            # Hop j's key (its code) is in exactly one record, the previous hop's.
            assert [sid for sid, text in records.items() if code in text] == [prev]
            # Hop j's answer is in the KB only in that code's directory entries.
            assert {tuple(p.fact_ids) for p in kb_paragraphs if nxt in p.text} == {(f"f-{code}",)}
            prev = nxt
        assert start in t.prompt and not any(x in t.prompt for x in [*chain, *codes])
        assert t.gold_fact_ids == ["f-ESC-protocol", *(f"f-{c}" for c in codes)]


def test_f2_scoring_final_exact_and_correct_prefix():
    w = make_world("F2", "5", "dev", 0, 1)
    t = w.tasks[0]
    chain = t.gold["chain"]
    assert is_success(t, {"final": chain[-1].lower(), "chain": chain}, []) and f2_error(t, t.gold) == "correct"
    broken = chain[:2] + ["SUP-1"] + chain[3:]
    assert f2_prefix({"chain": broken}, t.gold) == 2 and f2_error(t, {"final": "SUP-1", "chain": broken}) == "wrong_hop"
    assert f2_error(t, {"final": chain[2], "chain": chain[:3]}) == "stopped_early"
    assert f2_error(t, {"final": "SUP-1", "chain": [*chain, "SUP-1"]}) == "overshot"
    assert f2_error(t, {"final": chain[-1], "chain": broken}) == "correct_final_wrong_path"
    assert not is_success(t, {"final": chain[-2], "chain": chain}, [])
    assert f2_error(t, None) == "no_answer" and f2_error(t, {"final": "x", "chain": "x"}) == "malformed_answer"


def test_fault_hook_corrupts_exactly_the_first_surfacing_and_only_when_enabled():
    w = make_world("F2", "5", "dev", 0, 4)
    for t in w.tasks:
        fault = t.tags["fault"]
        key, true_code = fault["key"], fault["true_value"]
        assert fault["value"] != true_code and w.entities["routes"][fault["value"]] not in [t.tags["start"], *t.gold["chain"]]
        # Disabled (the default): the record is the truth, every time.
        init_subtask_store(Store())
        tools = build_tools(w, t)
        for _ in range(2):
            assert json.loads(_call(tools["lookup_supplier"], supplier_id=key))["escalation_code"] == true_code
        from inspect_ai.util import store

        assert not store().get(FAULTS_FIRED)
        # Enabled: the first lookup of the key reports the wrong code; later lookups and other keys are untouched.
        init_subtask_store(Store())
        enabled = TaskItem(**{**t.__dict__, "setup": {"faults_enabled": True}})
        tools = build_tools(w, enabled)
        other = t.tags["start"] if key != t.tags["start"] else t.gold["chain"][0]
        assert json.loads(_call(tools["lookup_supplier"], supplier_id=other))["escalation_code"] == w.entities["suppliers"][other]["escalation_code"]
        assert json.loads(_call(tools["lookup_supplier"], supplier_id=key.lower()))["escalation_code"] == fault["value"]
        assert json.loads(_call(tools["lookup_supplier"], supplier_id=key))["escalation_code"] == true_code
        assert len(store().get(FAULTS_FIRED)) == 1


def test_answer_tools_record_answers():
    w1, w2 = make_world("F1", "2", "dev", 0, 1), make_world("F2", "2", "dev", 0, 1)
    from inspect_ai.util import store

    init_subtask_store(Store())
    t = w1.tasks[0]
    assert always_on(w1) == ["lookup_supplier", "submit_ratings"]
    _call(build_tools(w1, t)["submit_ratings"], ratings=json.dumps(t.gold["ratings"]))  # a JSON string is accepted too
    assert store().get(ANSWER) == t.gold and is_success(t, store().get(ANSWER), [])
    init_subtask_store(Store())  # a new sample; within one sample the first answer wins (env_tools._answer)
    t = w2.tasks[0]
    _call(build_tools(w2, t)["submit_chain"], final_supplier=t.gold["final"], chain=t.gold["chain"])
    assert store().get(ANSWER) == t.gold


# ---------- task wiring ----------


def test_turn_caps_scale_with_the_knob():
    assert turn_cap("F1", "32") == 44 and turn_cap("F2", "10") == 28 and turn_cap("F1", "2") == 14


@pytest.mark.parametrize("family,level", [("F1", "2"), ("F2", "2")])
@pytest.mark.parametrize("arm", ["S1", "S3s", "S6"])
def test_baseline_arms_run_end_to_end(offline_env, family, level, arm):
    asyncio.run(build("dev", family, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log = inspect_eval(
        main_study(family=family, level=level, split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert len(log.samples) == 2 and log.eval.metadata["max_turns"] == max(12, turn_cap(family, level))
    for s in log.samples:
        assert set(s.scores) == {"task_success", "delivered_evidence", "error_analysis"}
        assert s.scores["error_analysis"].metadata["error"]
        assert s.store["env_answer"] is not None, "the mock reaches the answer tool"
        assert s.store.get("env_lookups"), "the mock looked a supplier up"
    if arm == "S6":
        assert all(s.scores["delivered_evidence"].value["evidence_recall"] == 1.0 for s in log.samples)


def test_faults_are_enabled_per_task_only_on_request(offline_env):
    asyncio.run(build("dev", "F2", ["2"], n_worlds=1, n_tasks=2, relational=True, embed=False))
    plain = main_study(family="F2", level="2", arm="S6")
    faulty = main_study(family="F2", level="2", arm="S6", faults=True)
    assert not any(s.metadata["task"]["setup"].get("faults_enabled") for s in plain.dataset)
    assert all(s.metadata["task"]["setup"]["faults_enabled"] for s in faulty.dataset)
    with pytest.raises(ValueError, match="F2 only"):
        main_study(family="F1", level="2", faults=True)
    with pytest.raises(ValueError, match="main_study covers"):
        main_study(family="F5", level="1hop")
    with pytest.raises(ValueError, match="main_study covers"):
        main_study(family="F7", level="5")


def test_main_study_covers_the_gate_families_and_resolves_s5_to_the_kg_arm(offline_env, monkeypatch):
    """F3 and F7 run through the same task; S5 is the gate-selected KG arm (APE_KG_ARM); an unknown arm id is refused
    (every planned multi-agent arm is registered since B2, so the old "refused until registered" branch is gone)."""
    from ape.agent import arms
    from ape.agent.solvers import MULTI_AGENT_ARMS, PLANNED_MULTI_AGENT_ARMS

    asyncio.run(build("dev", "F7", ["10"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    log = inspect_eval(
        main_study(family="F7", level="10", split="dev", arm="S1"),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.eval.metadata["max_turns"] == 12 and log.eval.metadata["multi_agent"] is False
    assert arms.kg_arm_name() == "APG-s"
    monkeypatch.setenv("APE_KG_ARM", "LGR-s")
    assert arms.kg_arm_name() == "LGR-s"
    assert main_study(family="F7", level="10", split="dev", arm="S5").metadata["knobs"]["APE_KG_ARM"] == "LGR-s"
    monkeypatch.setenv("APE_KG_ARM", "S5")
    with pytest.raises(ValueError, match="APG or LightRAG"):
        arms.kg_arm_name()
    assert set(PLANNED_MULTI_AGENT_ARMS) <= set(MULTI_AGENT_ARMS)
    monkeypatch.delenv("APE_KG_ARM")
    bad = inspect_eval(
        main_study(family="F7", level="10", split="dev", arm="S99"),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert bad.status == "error" and "unknown arm S99" in bad.error.message, "an unknown arm id never runs as some default arm"


@pytest.mark.parametrize("arm", ["S5o", "LGRo-s"])
def test_kg_oracle_arms_run_on_registry_worlds(offline_env, arm):
    from ape.config import Config
    from ape.lgr.build import build_index
    from ape.llm.mock_agent import mock_kg

    ids = asyncio.run(build("dev", "F2", ["2"], n_worlds=1, n_tasks=2, relational=True, embed=True))
    if arm.startswith("LGR"):
        cfg = Config()
        asyncio.run(build_index(World.load(cfg.world_path(ids[0])), "oracle", cfg))
    log = inspect_eval(
        main_study(family="F2", level="2", split="dev", arm=arm),
        model=get_model("mockllm/model", custom_outputs=mock_agent),
        model_roles={"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)},
        log_dir=str(offline_env / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    for s in log.samples:
        recs = s.store["compile_log"]
        assert recs and all(r["fact_ids"] for r in recs), "provenance maps the KG back to world facts"


def test_budget_prices_a_per_cell_history_override(tmp_path):
    """`history_per_prior_call_by_cell` (for bulky tool results, e.g. F1/F2 records) replaces the global prior per cell."""
    import yaml

    from ape.budget import Prices, estimate, load_assumptions, load_plan

    costs = tmp_path / "costs.yaml"
    costs.write_text(yaml.safe_dump({"openai/agent-x": {"input": 1.0, "output": 0.0, "input_cache_write": 1.0, "input_cache_read": 1.0}}))
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({
        "budget": {"total_usd": 10},
        "studies": {"s": {"profile": "gate", "phases": {"p": [
            {"id": "c", "models": {"agent": "openai/agent-x"}, "arms": ["S6"], "cells": ["F1-32", "F2-10"], "n_tasks": 1, "tasks_per_world": 1, "epochs": 1},
        ]}}},
    }))  # fmt: skip
    a = load_assumptions()
    a["output_tokens_per_call"] = {"high": 0, "medium": 0, "low": 0, "default": 0}
    a["calls_per_sample"] = {"F1-32": {"push": 3}, "F2-10": {"push": 3}}
    a["base_input_per_call"], a["history_per_prior_call"] = 0, 100
    a["arms"] = {"S6": {"context": 0, "compile": "per_query", "kg": None, "embed": False, "cache": "per_query"}}
    a["history_per_prior_call_by_cell"] = {"F1-32": 1000}
    est = estimate(load_plan(plan), a, Prices.load(costs), measured=[])
    # 3 calls x history x (3 - 1) / 2 tokens at $1/M: F1-32 uses its override, F2-10 the global prior.
    assert est.total(cell="c") == pytest.approx((3 * 1000 + 3 * 100) / 1e6)
