"""The agent loops under a misbehaving model (RELIABILITY_REVIEW §2, L1-L11).

The offline mocks always make well-formed calls; these tests drive the real `kb_agent` and F8 session loops with
the review's scripted misbehaviour (mockllm `custom_outputs`) and assert that no sample errors and the scores
come out right.
"""

import asyncio
import json
import re
import uuid

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText, ModelOutput, get_model
from inspect_ai.model._openai_responses import openai_responses_inputs
from inspect_ai.tool import ToolCall

from ape.agent import kb_react
from ape.agent.arms import load_world
from ape.agent.kb_react import STEP_KB_HEADER, kb_agent, search_kb_max_output, step_view, system_prompt
from ape.agent.step import exposed_tools
from ape.build import build
from ape.kb.context import ContextResult
from ape.llm import fake
from ape.llm.mock_session import _current
from ape.scorers.session import probe_score, report_score, score_session, taxonomy
from ape.scorers.success import is_success, task_success
from ape.scorers.taxonomy import error_analysis, f2_error, f2_prefix, f3_error, f5_error
from ape.tasks.gate import gate, gate_samples
from ape.tasks.study_g import f8_session
from ape.worlds import gen_f8
from ape.worlds.env_tools import ALREADY_ANSWERED
from ape.worlds.spec import TaskItem

M = "mockllm/model"


def tc(name, args):
    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", function=name, arguments=args)


def out(*calls, text=""):
    msg = ChatMessageAssistant(content=text, tool_calls=list(calls) or None, source="generate", model=M)
    return ModelOutput(model=M, choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls" if calls else "stop")])


def reasoning_out(*calls):
    """What an OpenAI reasoning model returns with store=False: an encrypted reasoning item, then function calls."""
    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning="gAAAA-ENCRYPTED", redacted=True)], tool_calls=list(calls), source="generate", model=M)
    return ModelOutput(model=M, choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls")])


class Script:
    """Turn i returns steps[i]; records every input the model was given."""

    def __init__(self, steps):
        self.steps, self.i, self.inputs = list(steps), 0, []

    def __call__(self, messages, tools, tool_choice, config):
        self.inputs.append(list(messages))
        step = self.steps[min(self.i, len(self.steps) - 1)] if self.i < len(self.steps) else out(text="(script exhausted)")
        self.i += 1
        return step


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("robust")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for fam, level in (("F7", "10"), ("F3", "5"), ("F5", "2hop")):
        asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=2, relational=True, embed=True))
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


def _run(env, task, model, **kw):
    log = inspect_eval(task, model=get_model(M, custom_outputs=model), log_dir=str(env / "logs"), display="none", limit=1, **kw)[0]
    s = log.samples[0]
    assert s.error is None, s.error
    return log, s


def _f7_task():
    return TaskItem(**gate_samples("F7", "10", "dev")[0].metadata["task"])


# --- L2: per-step context never ends the model's reasoning turn -------------------------------------------------


def test_per_step_context_rides_on_the_last_tool_result_not_a_new_user_turn(env):
    t = _f7_task()
    cid = t.tags["customer_id"]
    script = Script([
        reasoning_out(tc("lookup_customer", {"customer_id": cid})),
        reasoning_out(tc("lookup_customer", {"customer_id": cid})),
        reasoning_out(tc("submit_decision", dict(t.gold))),
    ])
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S3s"), script)
    assert s.scores["task_success"].value == "C" and len(script.inputs) == 3
    # Turn 0: the task, then the knowledge in a user message of its own (no reasoning yet).
    first = script.inputs[0]
    assert [m.role for m in first] == ["system", "user", "user"] and first[-1].text.startswith(STEP_KB_HEADER)
    for msgs in script.inputs[1:]:
        items = asyncio.run(openai_responses_inputs(msgs))
        kinds = [it.get("type") + (":" + it["role"] if it.get("type") == "message" else "") for it in items]
        first_reasoning = kinds.index("reasoning")
        # No user message after the model's reasoning: OpenAI keeps every reasoning item of the turn.
        assert "message:user" not in kinds[first_reasoning:], kinds
        assert kinds[-1] == "function_call_output" and STEP_KB_HEADER in items[-1]["output"]
    # The logged history stays clean: the knowledge is in the view only, never accumulated (D-005).
    assert not any(STEP_KB_HEADER in (m.text or "") for m in s.messages)


def test_step_view_appends_to_the_last_tool_result_in_a_copy():
    tool = ChatMessageTool(content="lookup result", tool_call_id="c1", function="lookup_customer")
    history = [ChatMessageUser(content="task"), tool]
    view = step_view(history, "KB text")
    assert view[-1].text == f"lookup result\n\n{STEP_KB_HEADER}\nKB text" and history[-1].text == "lookup result"
    listed = ChatMessageTool(content=[ContentText(text="part")], tool_call_id="c1", function="x")
    assert step_view([listed], "KB")[-1].text.endswith(f"{STEP_KB_HEADER}\nKB")
    # After a nudge (last message is the user's) the knowledge gets its own user message, as at turn 0.
    v = step_view([ChatMessageUser(content="nudge")], "KB")
    assert [m.role for m in v] == ["user", "user"] and v[-1].text.startswith(STEP_KB_HEADER)


# --- L3 / L4: search_kb is never cut silently and never kills the sample -----------------------------------------


class StubArm:
    name, per_step = "stub", False

    def __init__(self, text: str = "Policy text.", exc: Exception | None = None):
        self.text, self.exc, self.queries = text, exc, []

    async def compile(self, query, task):
        self.queries.append(query)
        if self.exc:
            raise self.exc
        return ContextResult(text=self.text, unit_ids=["u1"], fact_ids=["f-1"])


def _stub_task(arm: StubArm) -> Task:
    async def provider(world):
        return arm

    return Task(dataset=MemoryDataset(gate_samples("F7", "10", "dev")[:1]), solver=kb_agent(provider, load_world, delivery="pull"), scorer=[task_success()])


def test_search_kb_output_limit_covers_the_largest_budget(env, monkeypatch):
    monkeypatch.delenv("APE_LGR_TOTAL_TOKENS", raising=False)
    for k in ("APE_S3S_BUDGET", "APE_APG_BUDGET", "APE_LGR_BUDGET", "APE_CONTEXT_BUDGET"):
        monkeypatch.delenv(k, raising=False)
    assert search_kb_max_output() == 60_000  # LightRAG's default total cap, 4 x 2,000 tokens, x 6 bytes x 1.25
    monkeypatch.setenv("APE_S3S_BUDGET", "16000")
    assert search_kb_max_output() == 120_000
    t = _f7_task()
    text = "Refund policy paragraph with IDs P-1234 and X-5678. " * 600  # ~31 KB: over Inspect's 16 KiB default
    log, s = _run(env, _stub_task(StubArm(text)), Script([out(tc("search_kb", {"query": "refunds"})), out(tc("submit_decision", dict(t.gold)))]))
    seen = next(m for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "search_kb")
    assert seen.text == text and "too long" not in seen.text
    rec = next(r for r in s.store["compile_log"] if r["source"] == "pull")
    assert rec["raw_bytes"] == rec["seen_bytes"] == len(text.encode()) and rec["truncated"] is False
    event = next(e for e in s.events if e.event == "tool" and e.function == "search_kb")
    assert event.truncated is None


def test_search_kb_truncation_beyond_the_limit_is_visible_and_logged(env, monkeypatch):
    monkeypatch.setattr(kb_react, "search_kb_max_output", lambda cfg=None: 20_000)
    t = _f7_task()
    text = "x" * 50_000
    log, s = _run(env, _stub_task(StubArm(text)), Script([out(tc("search_kb", {"query": "refunds"})), out(tc("submit_decision", dict(t.gold)))]))
    seen = next(m for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "search_kb")
    rec = next(r for r in s.store["compile_log"] if r["source"] == "pull")
    assert rec["truncated"] is True and rec["raw_bytes"] == 50_000 and rec["seen_bytes"] <= 20_000
    assert "knowledge base result truncated" in seen.text and len(seen.text.encode()) == rec["seen_bytes"]


def test_search_kb_empty_huge_and_failing_queries_are_tool_errors(env, monkeypatch):
    """The review's r5: an embeddings client enforcing OpenAI's input rules (non-empty, <= 8,192 tokens)."""
    import httpx
    import openai

    from ape.llm.tokens import count_tokens

    orig = fake.FakeEmbeddingsClient._create

    async def strict(self, model, input, encoding_format=None):
        for text in input:
            if text == "" or count_tokens(text) > 8192:
                req = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
                raise openai.BadRequestError("Error code: 400 - '$.input' is invalid", response=httpx.Response(400, request=req), body=None)
        return await orig(self, model, input, encoding_format)

    monkeypatch.setattr(fake.FakeEmbeddingsClient, "_create", strict)
    t = _f7_task()
    steps = [out(tc("search_kb", {"query": "   "})), out(tc("search_kb", {"query": "refund policy " * 4600})), out(tc("submit_decision", dict(t.gold)))]
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S3s", delivery="pull"), Script(steps))
    assert log.status == "success" and s.scores["task_success"].value == "C"
    results = [m for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "search_kb"]
    assert results[0].error is not None and "Empty query" in results[0].error.message
    assert results[1].error is None  # the huge query was capped, not rejected
    assert s.store["search_errors"] == [{"step": 0, "error": "empty query"}]
    assert all(r["source"] in ("push", "pull") and "error" not in r for r in s.store["compile_log"])

    # Any compile failure comes back to the model as a tool error; the sample goes on and is scored.
    arm = StubArm(exc=RuntimeError("lightrag exploded"))
    log, s = _run(env, _stub_task(arm), Script([out(tc("search_kb", {"query": "refunds"})), out(tc("submit_decision", dict(t.gold)))]))
    err = next(m for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "search_kb").error
    assert err is not None and "search_kb failed (RuntimeError)" in err.message
    assert s.scores["task_success"].value == "C" and "lightrag exploded" in s.store["search_errors"][0]["error"]


def test_a_sample_limit_hit_inside_search_kb_ends_the_sample_at_once(env):
    """B2's finding: the kg call inside a pull compile can hit the sample's token limit. That is no retrieval failure
    for the model to route around: the limit ends the sample there, with no further model call."""
    from inspect_ai.util import LimitExceededError

    t = _f7_task()
    arm = StubArm(exc=LimitExceededError("token", value=1200, limit=1000, message="token limit (1,000) exceeded"))
    script = Script([out(tc("search_kb", {"query": "refunds"})), out(tc("submit_decision", dict(t.gold)))])
    log, s = _run(env, _stub_task(arm), script)
    assert script.i == 1, "one model call: the limit ended the sample before the model saw a tool error"
    assert s.limit is not None and s.limit.type == "token"
    assert not s.store.get("search_errors") and not any(isinstance(m, ChatMessageTool) and m.function == "search_kb" for m in s.messages)
    assert s.store["compile_log"] == [] and s.store["turns_used"] == 1, "the logs are still written"


# --- L5 / L9: labels and normalization -----------------------------------------------------------------------


def test_correct_f3_calls_are_labelled_correct_whatever_the_key_order(env):
    w = load_world(gate_samples("F3", "5", "dev")[0].metadata["world_id"])
    t = TaskItem(**gate_samples("F3", "5", "dev")[0].metadata["task"])
    oid = t.tags["order_id"]
    natural = [{"tool": c["tool"], "args": {"order_id": oid, **{k: v for k, v in c["args"].items() if k != "order_id"}}} for c in t.gold["calls"]]
    assert f3_error(t, natural) == "correct" and f3_error(t, list(reversed(natural))) == "correct"
    steps = [out(tc("order_lookup", {"order_id": oid})), out(*[tc(c["tool"], c["args"]) for c in reversed(natural)]), out(tc("finish", {}))]
    log, s = _run(env, gate(family="F3", level="5", split="dev", arm="S1", exposure="all"), Script(steps))
    assert s.scores["task_success"].value == "C" and s.scores["error_analysis"].metadata["error"] == "correct"
    assert w.family == "F3"


def test_f2_chain_with_the_start_supplier_gets_full_credit():
    gold = {"final": "SUP-3", "chain": ["SUP-2", "SUP-3"]}
    t = TaskItem(id="x", world_id="w", family="F2", level="2", prompt="", gold=gold, gold_fact_ids=[], answer_tool="submit_chain", tags={"start": "SUP-1"})
    with_start = {"final": "SUP-3", "chain": ["sup-1", "SUP-2", "SUP-3"]}
    assert f2_prefix(with_start, gold, "SUP-1") == 2 and f2_error(t, with_start) == "correct"
    assert f2_error(t, {"final": "SUP-3", "chain": ["SUP-1", "SUP-9", "SUP-3"]}) == "correct_final_wrong_path"
    assert f2_prefix({"chain": ["SUP-2", "SUP-3"]}, gold) == 2


@pytest.mark.parametrize("answer, ok", [("Krakow", True), ("krakow.", True), ("Kraków", True), ("Krakow office", True), ("the Krakow office", True), ("Krakow, Poland", True), ("Lisbon", False), ("Lisbon office", False), ("Krakowiak", False)])
def test_f5_accepts_harmless_variants_and_never_another_city(answer, ok):
    t = TaskItem(id="x", world_id="w", family="F5", level="2hop", prompt="", answer_tool="submit_answer", gold={"answer": "Krakow"}, gold_fact_ids=[], setup={}, tags={})
    assert is_success(t, {"answer": answer}, []) is ok


def test_f5_label_uses_the_same_normalization(env):
    w = load_world(gate_samples("F5", "2hop", "dev")[0].metadata["world_id"])
    t = w.tasks[0]
    assert f5_error(w, t, {"answer": f"the {t.gold['answer']} office"}) == "correct"


# --- L6 / L8 / L10 / L11: nudges, limits, duplicate answers, exposure, prompts ---------------------------------


def test_a_second_text_reply_after_tool_calls_gets_its_own_nudge(env):
    """The review's r14: text, lookup, text, answer. One nudge per text-only streak, not per sample."""
    t = _f7_task()
    steps = [out(text="I'll look up the customer first."), out(tc("lookup_customer", {"customer_id": t.tags["customer_id"]})), out(text="Per policy it must be escalated."), out(tc("submit_decision", dict(t.gold)))]
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S1"), Script(steps))
    assert s.scores["task_success"].value == "C" and s.store["nudges"] == 2 and s.store["turns_used"] == 4


def test_a_model_that_only_chats_stops_after_one_nudge_and_nudges_are_capped(env):
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S1"), Script([out(text="hello")] * 10))
    assert s.store["turns_used"] == 2 and s.store["nudges"] == 1 and s.scores["task_success"].value == "I"
    cid = _f7_task().tags["customer_id"]
    chatty = [out(text="thinking"), out(tc("lookup_customer", {"customer_id": cid}))] * 6
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S1"), Script(chatty))
    assert s.store["nudges"] == kb_react.MAX_NUDGES and s.store["turns_used"] < 12


def test_a_sample_limit_keeps_the_logs(env):
    """The review's r8: a message limit cuts the loop; the compile log, turns and messages survive."""
    cid = _f7_task().tags["customer_id"]
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S3s"), Script([out(tc("lookup_customer", {"customer_id": cid}))] * 10), message_limit=7)
    assert s.limit is not None
    assert s.store["compile_log"] and s.store["turns_used"] >= 2 and len(s.messages) > 2


@pytest.mark.parametrize("first_is_gold", [True, False])
def test_the_first_answer_wins_in_the_gate(env, first_is_gold):
    t = _f7_task()
    wrong = {**t.gold, "action": "deny" if t.gold["action"] != "deny" else "approve"}
    pair = [dict(t.gold), wrong] if first_is_gold else [wrong, dict(t.gold)]
    log, s = _run(env, gate(family="F7", level="10", split="dev", arm="S1"), Script([out(*(tc("submit_decision", a) for a in pair))]))
    assert s.scores["task_success"].value == ("C" if first_is_gold else "I")
    second = [m for m in s.messages if isinstance(m, ChatMessageTool)][-1]
    assert second.text == ALREADY_ANSWERED


def test_tool_exposure_matches_tool_names_in_any_case():
    tools = {n: object() for n in ("order_lookup", "finish", "wrong_item_credit", "reroute_parcel")}
    ctx = ContextResult(text="Entity: Wrong_Item_Credit (tool) applies a goodwill credit.", unit_ids=[], fact_ids=[])
    assert exposed_tools(tools, ctx, "retrieved", ["order_lookup", "finish"]) == [tools["order_lookup"], tools["finish"], tools["wrong_item_credit"]]


def test_the_system_prompt_says_where_knowledge_comes_from_per_delivery():
    push = system_prompt(push=True, pull=False, per_step=False, kb_text="KB!")
    assert "provided below" in push and push.endswith("## Knowledge base\nKB!")
    step = system_prompt(push=True, pull=False, per_step=True, kb_text="KB!")
    assert "refreshed every step" in step and "provided below" not in step and "KB!" not in step
    pull = system_prompt(push=False, pull=True, per_step=False)
    assert "search_kb" in pull and "provided below" not in pull and "## Knowledge base" not in pull
    both = system_prompt(push=True, pull=True, per_step=False, kb_text="KB!")
    assert "provided below" in both and "search_kb" in both


# --- F8 sessions (L1, L7, L8, L10) ---------------------------------------------------------------------------------

CASE = re.compile(r"(?:Case|Ticket) (C-\d{6})")


def f8_agent(world, variant: str = "oracle"):
    """Gold answers, gold report, true-state probes, plus one realistic deviation per variant."""
    by_case = {t.tags["case_id"]: t for t in world.tasks}
    gold_report = world.entities["session"]["report_gold"]
    answered: list[str] = []
    turns: dict[str, int] = {}
    report_tries = [0]

    def agent(messages, tools, tool_choice, config):
        if config.response_schema is not None:
            return ModelOutput.from_content(M, json.dumps(gen_f8.state_at(world, len(answered))))
        i, text = _current(messages)
        if "End of shift" in text:
            report_tries[0] += 1
            if variant == "report_objects" and report_tries[0] == 1:
                rich = {**gold_report, "pending_recheck": [{"case_id": c, "status": "escalated"} for c in gold_report["pending_recheck"]]}
                return out(tc("submit_shift_report", {"report": json.dumps(rich)}))
            if variant == "report_text" and report_tries[0] == 1:
                return out(text="Shift report:\n" + json.dumps(gold_report))
            return out(tc("submit_shift_report", {"report": json.dumps(gold_report)}))
        t = by_case[CASE.findall(text)[-1]]
        cid, pos = t.tags["case_id"], t.tags["position"]
        n = turns[cid] = turns.get(cid, 0) + 1
        if variant == "report_early" and pos == 3 and n == 1:
            return out(tc("submit_shift_report", {"report": json.dumps({"dispositions": {}, "pending_recheck": [], "open_followups": []})}))
        if variant == "unknown_id" and pos == 3 and n == 1:
            return out(tc("submit_decision" if t.tags["kind"] != "ticket" else "finish", {"case_id": "C-000000", **({} if t.tags["kind"] == "ticket" else t.gold)}))
        say = cid.lower() + " " if variant == "lower_id" and pos == 2 else cid
        if t.tags["kind"] == "ticket":
            done = sum(isinstance(m, ChatMessageTool) and m.function != "finish" and not m.error for m in messages[i + 1 :])
            calls = t.gold["calls"]
            if done < len(calls):
                return out(tc(calls[done]["tool"], calls[done]["args"]))
            answered.append(cid)
            return out(tc("finish", {"case_id": say}))
        answered.append(cid)
        return out(tc("submit_decision", {"case_id": say, **t.gold}))

    return agent


def _f8(env, variant, **kw):
    world = load_world("F8-12-dev-s1000")
    log, s = _run(env, f8_session(level="12", split="dev", arm="CM0", checkpoints="5,10", **kw), f8_agent(world, variant))
    return world, s, s.scores["f8_session_score"]


@pytest.mark.parametrize("variant", ["oracle", "report_objects", "report_text", "report_early", "lower_id", "unknown_id"])
def test_f8_session_survives_realistic_deviations(env, variant):
    world, s, sc = _f8(env, variant)
    assert sc.value["item_success"] == 1.0 and sc.value["report_exact"] == 1.0 and sc.value["session_success"] == 1.0, (variant, sc.metadata["taxonomy"])
    if variant in ("report_objects", "report_early", "unknown_id"):
        errors = [m.error.message for m in s.messages if isinstance(m, ChatMessageTool) and m.error]
        assert errors, "the deviation came back to the model as a tool error"
    if variant == "unknown_id":
        assert any(e.get("rejected") and e["args"]["case_id"] == "C-000000" for e in s.store["f8_events"])
    if variant == "lower_id":
        assert all(e["args"]["case_id"] == e["args"]["case_id"].strip().upper() for e in s.store["f8_events"] if "case_id" in e["args"])


def test_f8_scorer_never_raises_on_odd_report_and_probe_shapes(env):
    world = load_world("F8-12-dev-s1000")
    gold = world.entities["session"]["report_gold"]
    for report in ({**gold, "pending_recheck": [{"case_id": "C-1"}], "open_followups": 5}, {"dispositions": ["x"], "pending_recheck": None}, {"dispositions": {"c-1": {"a": 1}}}):
        r = report_score(report, gold)
        assert r["submitted"] and not r["exact"]
        taxonomy(world, [], report, probes=[{"answer": {"completed": [{"id": 1}], "pending": 7}}])
        score_session(world, [{"item": 1, "tool": "wrong_item_credit", "args": {"order_id": {"nested": 1}}}], report)
    assert probe_score({"completed": [{"case_id": "C-1"}, 3, None]}, gen_f8.state_at(world, 0))["completed"] in (0.0, 1.0)


def test_f8_missed_checkpoints_count_as_zero(env):
    world = load_world("F8-12-dev-s1000")
    _, s, _ = _f8(env, "oracle", window=10**6)
    by_item = {}
    for v in s.store["f8_views"]:
        by_item[v["item"]] = max(by_item.get(v["item"], 0), v["view_tokens"])
    window = max(t for i, t in by_item.items() if i <= 7) + 1
    _, s, sc = _f8(env, "oracle", window=window)
    at = s.store["f8_overflow_at"]
    assert at is not None and 5 < at <= 10
    by_k = sc.metadata["probes_by_checkpoint"]
    assert by_k["5"]["taken"] and not by_k["10"]["taken"] and by_k["10"]["mean_f1"] == 0.0
    assert sc.value["probe_coverage"] == 0.5 and sc.value["probe_f1"] == pytest.approx(by_k["5"]["mean_f1"] / 2)


def test_f8_session_cut_by_a_limit_keeps_its_records(env):
    world = load_world("F8-12-dev-s1000")
    log, s = _run(env, f8_session(level="12", split="dev", arm="CM0", checkpoints="5,10"), f8_agent(world), message_limit=20)
    assert s.limit is not None and len(s.messages) > 1
    assert s.store["f8_events"] and s.store["f8_items"]
    sc = s.scores["f8_session_score"]
    assert 0 < sc.value["item_success"] < 1
