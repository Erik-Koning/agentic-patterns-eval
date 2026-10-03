"""Mid-session checkpoint/resume (BUILD_PLAN B7): a killed F8 session resumes after its last completed item.

A mock that raises at the first call of item K on its first attempt stands in for a transient API error. With
Inspect's sample retries (directly and through `ape.runner.run_evals`) the session completes; items 1..K-1 are
not generated again; its records are complete, contiguous and equal to an uninterrupted session's; the restored
calls' usage is recorded; a finished session leaves no checkpoint; a checkpoint is never reused under another
configuration. Policy state (a test policy, and Inspect's own compaction handler) resumes too. Offline: mockllm."""

import asyncio
import json
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model import CompactionEdit, GenerateConfig, compaction, get_model
from inspect_ai.model._model_info import clear_model_info_cache
from test_session_policy import Ledger, ledger_agent

from ape.agent import context_policy
from ape.agent.context_policy import ContextPolicy, TrackedState, load_messages
from ape.agent.session_checkpoint import ENV, SessionCheckpoints, model_identity, resume_summary
from ape.build import build
from ape.llm.mock_session import _current, mock_session_agent
from ape.models import COSTS_PATH
from ape.runner import run_evals
from ape.tasks.study_g import f8_session

M = "mockllm/model"
K = 7  # the item whose first call fails on the first attempt
RECORD_KEYS = ("f8_items", "f8_views", "f8_probes", "f8_events", "f8_report", "f8_overflow_at", "f8_cm_events", "f8_usage")


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("f8-resume")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    d = tmp_path / "checkpoints"
    monkeypatch.setenv(ENV, str(d))
    return d


def case_position(messages) -> int | None:
    """The queue position of the case the view is working on (None for the report or a management call)."""
    from ape.agent.arms import load_world

    _, text = _current(messages)
    world = load_world("F8-12-dev-s1000")
    for t in world.tasks:
        if t.tags["case_id"] in text[:40]:
            return t.tags["position"]
    return None


class Flaky:
    """Wraps a mock agent: the first call for item `k` raises (once per instance); every agent call is logged with
    the item it works on."""

    def __init__(self, agent, k: int = K, times: int = 1):
        self.agent, self.k, self.left = agent, k, times
        self.calls: list[int | None] = []

    def __call__(self, messages, tools, tool_choice, config):
        if tools:  # an agent call (probes and management calls have no tools)
            pos = case_position(messages)
            if pos == self.k and self.left:
                self.left -= 1
                raise RuntimeError("transient API error (test)")
            self.calls.append(pos)
        return self.agent(messages, tools, tool_choice, config)


def _eval(worlds, agent, retries=2, arm="CM0", config=None, **kw):
    log = inspect_eval(
        f8_session(level="12", split="dev", arm=arm, checkpoints="5,10", **kw),
        model=get_model(M, custom_outputs=agent, config=config or GenerateConfig()),
        log_dir=str(worlds / "logs"),
        display="none",
        retry_on_error=retries,
        fail_on_error=False,
    )[0]
    assert log.status == "success", log.error
    return log.samples[0]


def _records(sample) -> dict:
    return {k: sample.store.get(k) for k in RECORD_KEYS}


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*.json")) if d.exists() else []


def test_a_killed_session_resumes_mid_way(worlds, ckpt_dir):
    clean = _eval(worlds, mock_session_agent)
    flaky = Flaky(mock_session_agent)
    s = _eval(worlds, flaky)
    assert s.error is None and len(s.error_retries) == 1
    assert "transient API error" in s.error_retries[0].message
    st = s.store
    # Items 1..K-1 ran once, in the first attempt; the resumed attempt started at item K.
    first = flaky.calls[: flaky.calls.index(K)] if K in flaky.calls else flaky.calls
    assert sorted({p for p in first if p is not None}) == list(range(1, K))
    assert all(p is None or p >= K for p in flaky.calls[len(first) :])
    gens = {i["position"]: i["generations"] for i in st["f8_items"]}
    assert all(flaky.calls.count(p) == gens[p] for p in range(1, 13))
    # Records: complete, contiguous, and exactly an uninterrupted session's.
    assert [i["position"] for i in st["f8_items"]] == list(range(1, 13))
    items = [v["item"] for v in st["f8_views"]]
    assert items == sorted(items) and set(items) == set(range(1, 14))
    assert [p["k"] for p in st["f8_probes"]] == [5, 10]
    assert _records(s) == _records(clean)
    assert s.scores["f8_session_score"].value == clean.scores["f8_session_score"].value
    assert [m.text for m in s.messages] == [m.text for m in clean.messages]
    # The resume is recorded, with the restored calls' usage.
    r = st["f8_resume"]
    (failure,) = r["failures"]
    assert r["count"] == 1 and r["after_items"] == [K - 1] and failure["after_item"] == K - 1
    assert "transient API error" in failure["error"] and failure["usage"]["calls"] == 0  # the failing call made no usage
    restored = [v for v in st["f8_views"] if v["item"] < K] + [p for p in st["f8_probes"] if p["k"] < K]
    (entry,) = r["resumes"]
    assert entry["restored"]["calls"] == len(restored) and entry["sample_uuid"] == s.uuid
    assert entry["restored"]["by_kind"]["agent"]["input_tokens"] == sum(v["usage"]["input_tokens"] for v in restored if v["kind"] == "agent")
    # A sample-level retry keeps the sample UUID and Inspect drops the first attempt's usage: those calls are
    # `unlogged`, and Inspect's own usage covers exactly the rest.
    assert r["unlogged"]["calls"] == len(restored) and r["logged_elsewhere"] is None
    total = st["f8_usage"]["by_model"][M]["input_tokens"]
    assert s.model_usage[M].input_tokens == total - r["unlogged"]["by_model"][M]["input_tokens"]
    assert clean.model_usage[M].input_tokens == total
    meta = s.scores["f8_session_score"].metadata["resume"]
    assert meta["count"] == 1 and meta["after_items"] == [K - 1] and meta["unlogged"][M]["input_tokens"] > 0
    assert clean.scores["f8_session_score"].metadata["resume"] is None and clean.store["f8_resume"] is None
    # A finished session leaves no checkpoint behind.
    assert _files(ckpt_dir) == []


@pytest.mark.parametrize("k,times,after", [(1, 1, [0]), (None, 1, [12]), (K, 2, [K - 1, K - 1])], ids=["first-item", "report", "twice"])
def test_a_session_resumes_from_any_point(worlds, ckpt_dir, k, times, after):
    """A failure at the first call (resumed from the start checkpoint), in the report phase (all items restored),
    or twice in a row (two resumes from the same checkpoint)."""
    clean = _eval(worlds, mock_session_agent)
    flaky = Flaky(mock_session_agent, k=k, times=times)
    s = _eval(worlds, flaky)
    assert s.error is None and len(s.error_retries) == times
    r = s.store["f8_resume"]
    assert r["after_items"] == after and len(r["failures"]) == times and r["count"] == times
    assert _records(s) == _records(clean)
    restored = r["resumes"][-1]["restored"]["calls"]
    assert restored == sum(1 for c in s.store["f8_views"] + s.store["f8_probes"] if c["item"] <= after[-1])
    assert (r["unlogged"] or {}).get("calls", 0) == restored  # every earlier attempt shares this sample's UUID
    assert _files(ckpt_dir) == []


def test_overflow_and_limits_end_a_session_for_good(worlds, ckpt_dir):
    """An overflow or a sample limit completes the sample: its checkpoint is deleted, never resumed."""
    s = _eval(worlds, mock_session_agent, window=8000)
    assert s.store["f8_overflow_at"] is not None and _files(ckpt_dir) == []
    log = inspect_eval(f8_session(level="12", split="dev", arm="CM0", checkpoints="5,10"), model=get_model(M, custom_outputs=mock_session_agent), log_dir=str(worlds / "logs"), display="none", message_limit=20)[0]
    assert log.samples[0].limit is not None and log.samples[0].store["f8_items"] and _files(ckpt_dir) == []


def test_resume_through_the_runner_with_priced_usage(worlds, ckpt_dir, tmp_path, monkeypatch):
    """The study runner's path (`run_evals`: eval_set, retry_on_error=2) with a priced mock: the restored calls'
    cost is recorded and Inspect's sample cost plus the unlogged cost is the session's whole cost."""
    for k in ("APE_MODEL_PROFILE", "APE_BUILD_MODEL", "APE_BUILD_FALLBACK", "APE_EMBEDDING_MODEL"):
        monkeypatch.delenv(k, raising=False)
    costs = tmp_path / "costs.yaml"
    table = yaml.safe_load(COSTS_PATH.read_text())
    table[M] = {"input": 1.0, "output": 2.0, "input_cache_write": 1.0, "input_cache_read": 1.0}
    costs.write_text(yaml.safe_dump(table))
    clear_model_info_cache()
    try:
        task = f8_session(level="12", split="dev", arm="O-state", checkpoints="5,10")
        success, (header,) = run_evals(task, tmp_path / "logs", model_override=M, model_args={"custom_outputs": Flaky(mock_session_agent)}, costs_path=costs, roles=(), display="none", retry_wait=0.01)
    finally:
        clear_model_info_cache()
    assert success
    from inspect_ai.log import read_eval_log

    (s,) = read_eval_log(header.location).samples
    assert len(s.error_retries) == 1 and s.store["f8_resume"]["after_items"] == [K - 1]
    calls = s.store["f8_views"] + s.store["f8_probes"]
    assert all(c["usage"]["total_cost"] > 0 for c in calls)
    unlogged = s.store["f8_resume"]["unlogged"]["by_model"][M]["total_cost"]
    whole = s.store["f8_usage"]["by_model"][M]["total_cost"]
    assert unlogged > 0 and s.model_usage[M].total_cost == pytest.approx(whole - unlogged)
    assert s.scores["f8_session_score"].metadata["resume"]["unlogged"][M]["total_cost"] == pytest.approx(unlogged)
    assert _files(ckpt_dir) == []


def test_a_checkpoint_is_reused_only_under_its_configuration(worlds, ckpt_dir, monkeypatch):
    # Attempt 1 fails without retries: the checkpoint after item K-1 stays.
    s = _eval(worlds, Flaky(mock_session_agent), retries=0)
    assert s.error is not None  # an errored sample (the task does not fail on it)
    (kept,) = _files(ckpt_dir)
    payload = json.loads(kept.read_text())
    assert payload["done"] == K - 1 and len(payload["failures"]) == 1 and payload["failures"][0]["after_item"] == K - 1
    assert kept.parent.name == "epoch-1" and kept.parent.parent.name == "F8-12-dev-s1000"
    # Another configuration never sees it: a fresh session, whose own checkpoint is gone when it finishes.
    for kw, env in (({"max_turns_per_item": 7}, {}), ({"window": 31_999}, {}), ({"threshold": 19_000}, {}), ({}, {"APE_CM_SOME_KNOB": "1"})):
        with monkeypatch.context() as m:
            for k, v in env.items():
                m.setenv(k, v)
            s = _eval(worlds, Flaky(mock_session_agent, times=0), retries=0, **kw)
        assert s.error is None and s.store["f8_resume"] is None, kw
        assert _files(ckpt_dir) == [kept]
    # Another model configuration (here the agent's temperature) is another key too.
    s = _eval(worlds, mock_session_agent, retries=0, config=GenerateConfig(temperature=0.5))
    assert s.store["f8_resume"] is None and _files(ckpt_dir) == [kept]
    # The same configuration resumes from it (a new eval: a new sample UUID, so the first attempt's usage is in the
    # earlier log's errored sample, not unlogged), and the finished session deletes it.
    s = _eval(worlds, Flaky(mock_session_agent, times=0), retries=0)
    r = s.store["f8_resume"]
    assert s.error is None and r["after_items"] == [K - 1] and len(r["failures"]) == 1
    assert r["unlogged"] is None and r["logged_elsewhere"]["calls"] == r["resumes"][0]["restored"]["calls"]
    assert _files(ckpt_dir) == []
    clean = _eval(worlds, mock_session_agent)
    assert _records(s) == _records(clean)


def test_a_damaged_checkpoint_starts_the_session_afresh(worlds, ckpt_dir):
    s = _eval(worlds, Flaky(mock_session_agent), retries=0)
    (kept,) = _files(ckpt_dir)
    payload = json.loads(kept.read_text())
    kept.write_text(json.dumps({**payload, "history": [{"role": "nobody", "content": 3}]}))
    s = _eval(worlds, mock_session_agent, retries=0)
    assert s.error is None and s.store["f8_resume"] is None and len(s.store["f8_items"]) == 12
    assert _files(ckpt_dir) == []


def test_the_key_covers_the_configuration(tmp_path):
    base = {"arm": "CM0", "knobs": {}, "window": 32000, "threshold": 20000, "max_turns_per_item": 8, "checkpoints": [5], "world": {"id": "w", "hash": "h"}, "code": "c", "models": {"agent": model_identity(get_model(M))}}
    key = SessionCheckpoints(tmp_path, "w", 1, base).key
    variants = [
        ("epoch", SessionCheckpoints(tmp_path, "w", 2, base)),
        ("sample", SessionCheckpoints(tmp_path, "v", 1, base)),
        *[(f, SessionCheckpoints(tmp_path, "w", 1, {**base, f: v})) for f, v in (("arm", "O-state"), ("knobs", {"keep": 2}), ("window", 1), ("threshold", 1), ("max_turns_per_item", 1), ("checkpoints", [6]), ("world", {"id": "w", "hash": "h2"}), ("code", "c2"))],
        ("effort", SessionCheckpoints(tmp_path, "w", 1, {**base, "models": {"agent": model_identity(get_model(M, config=GenerateConfig(reasoning_effort="low")))}})),
    ]
    for what, v in variants:
        assert v.key != key, what
    # Transport settings are not part of a model's identity.
    assert model_identity(get_model(M, config=GenerateConfig(max_connections=3, timeout=9))) == model_identity(get_model(M))
    # A file under another key, or unreadable, is never loaded.
    c = SessionCheckpoints(tmp_path, "w", 1, base)
    c.save({"done": 3})
    assert c.load()["done"] == 3
    other = SessionCheckpoints(tmp_path, "w", 1, {**base, "code": "c2"})
    other.path.parent.mkdir(parents=True, exist_ok=True)
    other.path.write_text(json.dumps({"format": 1, "key": "nope"}))
    assert other.load() is None
    other.path.write_text("{not json")
    assert other.load() is None
    c.complete()
    assert not c.path.exists() and c.load() is None


def test_resume_summary_splits_usage_by_where_inspect_logged_it():
    u = lambda n: {"input_tokens": n, "output_tokens": 1}  # noqa: E731
    views = [{"item": i, "kind": "agent", "model": M, "usage": u(10)} for i in range(1, 7)]
    # Attempts 1 and 2 share a UUID (a sample-level retry), attempt 3 is a task-level retry (new UUID) and so is 4.
    segments = [
        {"attempt": 1, "sample_uuid": "A", "views_from": 0, "probes_from": 0},
        {"attempt": 2, "sample_uuid": "A", "views_from": 2, "probes_from": 0},
        {"attempt": 3, "sample_uuid": "B", "views_from": 4, "probes_from": 0},
        {"attempt": 4, "sample_uuid": "B", "views_from": 5, "probes_from": 0},
    ]
    failures = [{"attempt": 2, "usage": {"calls": 1, "by_kind": {"agent": u(7)}, "by_model": {M: u(7)}}}]
    r = resume_summary(views, [], [{"after_item": 2}, {"after_item": 4}, {"after_item": 5}], segments, failures)
    # Attempt 1 (A, not A's last) and attempt 3 (B, not B's last, as attempt 4 is this one) are in no log; attempt 2
    # (A's last: the errored sample of the earlier log) holds its records and the calls it made after its last save.
    assert r["unlogged"]["by_model"][M]["input_tokens"] == 20 + 10
    assert r["logged_elsewhere"]["by_model"][M]["input_tokens"] == 20 + 7 and r["logged_elsewhere"]["calls"] == 3
    assert r["count"] == 3 and r["after_items"] == [2, 4, 5]
    assert resume_summary(views, [], [], segments[:1], []) is None


# --- Policy state resumes ------------------------------------------------------------------------------------------


def test_a_managed_policys_state_resumes(worlds, ckpt_dir, monkeypatch):
    monkeypatch.setitem(context_policy.POLICIES, "CM-ledger", Ledger)
    clean = _eval(worlds, ledger_agent, arm="CM-ledger", threshold=3000)
    flaky = Flaky(ledger_agent)
    s = _eval(worlds, flaky, arm="CM-ledger", threshold=3000)
    assert s.error is None and s.store["f8_resume"]["after_items"] == [K - 1]
    # Summaries, notes, compactions and every view continue exactly where the first attempt left them.
    assert _records(s) == _records(clean)
    assert [m.text for m in s.messages] == [m.text for m in clean.messages]
    assert any(v["kind"] == "cm" and v["item"] < K for v in s.store["f8_views"])
    assert _files(ckpt_dir) == []


class InspectPrune(ContextPolicy):
    """CM-prune's shape on Inspect's own primitive: `CompactionEdit` run by `compaction()`. The harness's T_abs
    triggers it (`force=True` from on_threshold; Inspect's own threshold is out of reach, since it counts tokens its
    own way), and its closure state is checkpointed through `TrackedState`."""

    name = "inspect-prune"
    KNOBS: ClassVar[dict] = {"keep_tool_uses": 2}

    def _build(self, history, restored=None):
        self.tracked = TrackedState(restored)
        strategy = CompactionEdit(threshold=10**9, keep_tool_uses=self.knobs["keep_tool_uses"], memory=False)
        self.handler = compaction(strategy, prefix=list(history[:2]), tools=self.session.tools, model=self.session.agent_model, checkpointer=self.tracked)

    async def start(self, history):
        self._build(history)

    def load_state(self, state, history):
        self._build(history, state["inspect"])

    async def view(self, history, item_start, done):
        view, _ = await self.handler.compact_input(history)
        return view

    async def on_threshold(self, history, view, tokens):
        view, _ = await self.handler.compact_input(history, force=True)
        self.session.log("prune", view_tokens=tokens)
        return view

    async def after_generate(self, history, view, output, appended):
        await self.handler.record_output(view, output)

    async def probe_view(self, history, done):
        st = self.tracked.state_dict()["compaction"]
        processed = set(st["processed_message_ids"])
        return [*load_messages(st["compacted_input"]), *[m for m in history if m.id not in processed]]

    def state_dict(self):
        return {"inspect": self.tracked.state_dict()}


def test_an_inspect_compaction_policy_prunes_and_resumes(worlds, ckpt_dir, monkeypatch):
    monkeypatch.setitem(context_policy.POLICIES, "CM-iprune", InspectPrune)
    clean = _eval(worlds, mock_session_agent, arm="CM-iprune", threshold=6000)
    full = _eval(worlds, mock_session_agent, arm="CM0")
    pruned = [e for e in clean.store["f8_cm_events"] if e["event"] == "threshold"]
    assert pruned and all(e["view_tokens_after"] < e["view_tokens"] for e in pruned)
    assert max(v["view_tokens"] for v in clean.store["f8_views"]) < max(v["view_tokens"] for v in full.store["f8_views"])
    assert not any(m.text == "(Tool result removed)" for m in clean.messages)  # the history is never edited
    # Killed at item K and resumed: Inspect's compaction state continues exactly.
    s = _eval(worlds, Flaky(mock_session_agent), arm="CM-iprune", threshold=6000)
    assert s.error is None and s.store["f8_resume"]["after_items"] == [K - 1]
    assert _records(s) == _records(clean)
    assert _files(ckpt_dir) == []
