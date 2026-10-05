"""The context-length sweep (CONTEXT_SWEEP.md, D-054): worlds, history, solver and scorer, budget, runner."""

import json
from dataclasses import replace

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, ChatMessageUser, ModelOutput, get_model

from ape import run_sweep
from ape.agent.context_policy import view_tokens
from ape.budget import estimate, load_assumptions, load_plan, long_context_rate, sweep_sample_usd
from ape.config import Config
from ape.llm.mock_sweep import MODEL, gold_sweep_agent, naive_sweep_agent
from ape.scorers.sweep import failure
from ape.worlds import gen_f8, gen_sweep

SIZES = [8_000, 64_000]


@pytest.fixture(scope="module")
def worlds():
    return {kind: gen_sweep.generate_set(kind, SIZES) for kind in gen_sweep.KINDS}


# ---------- worlds ----------


@pytest.mark.parametrize("kind", gen_sweep.KINDS)
def test_each_size_fits_under_its_target_and_shares_case_1_the_probe_and_a_middle_prefix(worlds, kind):
    small, large = worlds[kind]
    for w, size in zip((small, large), SIZES, strict=True):
        d = w.entities["sweep"]
        assert d["target_tokens"] == size and size - 2_000 < d["context_tokens"] <= size
        assert d["context_tokens"] == gen_sweep.context_tokens(w)
        assert w.id == gen_sweep.world_id(kind, size, gen_sweep.kind_seed(kind))
    ids = lambda w: [t.tags["case_id"] for t in w.tasks]
    assert ids(small)[0] == ids(large)[0] and ids(small)[-1] == ids(large)[-1]
    assert ids(small)[1:-1] == ids(large)[1 : len(small.tasks) - 1]  # the smaller middle is a prefix
    assert small.tasks[-1].gold == large.tasks[-1].gold
    assert small.tasks[-1].prompt == large.tasks[-1].prompt


@pytest.mark.parametrize("kind", gen_sweep.KINDS)
def test_only_the_probe_depends_on_session_state(worlds, kind):
    w = worlds[kind][1]
    probe, middle = w.tasks[-1], w.tasks[:-1]
    for t in middle:  # no memo, quota or follow-up reaches case 1 or the middle
        assert not t.tags["dependency_kinds"] and not t.tags["counterfactuals"]["without_memo"]
    memos = gen_f8.session(w)["memos"]
    cf = probe.tags["counterfactuals"]
    if kind == "fresh":
        assert memos == [] and probe.gold == cf["stateless"] and probe.tags["kind"] == "policy"
    elif kind == "memo":
        assert [m["position"] for m in memos] == [1] and memos[0]["withdrawn_at"] is None
        assert "M-1" in cf["without_memo"] and not gen_f8.decisions_equal(probe.gold, cf["stateless"])
        assert memos[0]["text"] in w.tasks[0].prompt and memos[0]["text"] not in gen_f8.system_prompt(w)
    else:
        assert probe.tags["kind"] == "followup" and probe.tags["original"] == w.tasks[0].tags["case_id"]
        assert w.tasks[0].tags["awaiting"] and w.tasks[0].tags["followed_up_at"] == len(w.tasks)
        assert [m["position"] for m in memos] == [len(w.tasks)] and memos[0]["text"] in probe.prompt
        assert not gen_f8.decisions_equal(probe.gold, cf["original_gold"])  # copying case 1's decision fails


@pytest.mark.parametrize("kind", gen_sweep.KINDS)
def test_history_replays_the_reference_trajectory_and_ends_at_the_probe(worlds, kind):
    w = worlds[kind][0]
    msgs = gen_sweep.history(w)
    assert view_tokens(msgs) + gen_sweep.tool_tokens(w) == w.entities["sweep"]["context_tokens"]
    assert msgs[1].text == gen_sweep.START
    probe = w.tasks[-1]
    decisions = [m for m in msgs if isinstance(m, ChatMessageTool) and m.function == "submit_decision"]
    assert len(decisions) == sum(t.tags["kind"] != "ticket" for t in w.tasks[:-1])  # every case before the probe decided
    if kind == "followup":
        assert isinstance(msgs[-1], ChatMessageUser) and msgs[-1].text == probe.prompt
    else:  # the probe's lookup is replayed; the decision is the model's
        assert isinstance(msgs[-1], ChatMessageTool) and msgs[-1].function == "lookup_customer"
        assert msgs[-3].text == probe.prompt
    assert gen_sweep.history(w, "abc123def456")[0].text.startswith("Run reference: abc123def456")


def test_the_smallest_size_must_hold_the_fixed_parts():
    with pytest.raises(ValueError, match="more than the smallest size"):
        gen_sweep.generate_set("fresh", [3_000])


# ---------- solver and scorer ----------


def _run(monkeypatch, tmp_path, worlds, agent, kinds="fresh,memo,followup"):
    monkeypatch.setenv("APE_WORLDS", str(tmp_path / "worlds"))
    for ws in worlds.values():
        for w in ws:
            w.save(Config().world_path(w.id))
    from ape.tasks.sweep import context_sweep

    log = inspect_eval(context_sweep(kinds=kinds, sizes="8000,64000"), model=get_model(MODEL, custom_outputs=agent), log_dir=str(tmp_path / "logs"), display="none")[0]
    assert log.status == "success"
    return {s.id: s.scores["sweep_score"] for s in log.samples}


def test_the_gold_agent_gets_every_probe_right_in_one_generation(monkeypatch, tmp_path, worlds):
    scores = _run(monkeypatch, tmp_path, worlds, gold_sweep_agent([w for ws in worlds.values() for w in ws]))
    assert len(scores) == 6
    for sid, sc in scores.items():
        assert sc.value == {"success": 1.0, "answered": 1.0, "measured": 1.0}, sid
        md = sc.metadata
        assert md["failure"] is None and md["generations"] == 1
        assert md["calls"][0]["view_tokens"] == md["context_tokens"]  # the solver meters what the world recorded


def test_failure_labels(monkeypatch, tmp_path, worlds):
    memo, follow = worlds["memo"][0].tasks[-1], worlds["followup"][0].tasks[-1]
    cid = lambda t: t.tags["case_id"]
    stale = {cid(memo): memo.tags["counterfactuals"]["stateless"], cid(follow): follow.tags["counterfactuals"]["original_gold"]}

    def agent(messages, tools, tool_choice, config):
        case = next(c for c in stale if any(c in m.text for m in messages[-3:]))
        return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": case, **stale[case]})

    scores = _run(monkeypatch, tmp_path, worlds, agent, kinds="memo,followup")
    labels = {sid.split("-")[1]: sc.metadata["failure"] for sid, sc in scores.items()}
    assert labels == {"memo": "ignored_memo", "followup": "copied_original"}
    assert failure(memo, None) == "unanswered"
    naive = _run(monkeypatch, tmp_path / "naive", worlds, naive_sweep_agent, kinds="fresh")
    assert all(sc.value["answered"] == 1.0 for sc in naive.values())


# ---------- budget ----------


def test_sweep_cells_are_off_by_default_and_priced_at_the_long_context_rate_when_on():
    plan = load_plan()
    cells = [c for c in plan.cells if c.study == "context_sweep"]
    assert {c.id for c in cells} == {"sweep.luna", "sweep.sol", "sweep.astra"} and not any(c.enabled for c in cells)
    assert estimate(plan, measured=[]).total(study="context_sweep") == 0
    on = replace(plan, cells=tuple(replace(c, spec={**c.spec, "enabled": True}) if c.study == "context_sweep" else c for c in plan.cells))
    est = estimate(on, measured=[])
    A = load_assumptions()
    assert long_context_rate(A, "openai/gpt-6-astra", 272_001) and not long_context_rate(A, "openai/gpt-6-astra", 272_000)
    assert not long_context_rate(A, "openai/gpt-6-sol", 960_000)
    # Astra at 960K: 2 calls x (0.96M x $20 + 4,000 x $75 / 1M), every row 3 samples (one per kind).
    assert sweep_sample_usd("openai/gpt-6-astra", 960_000) == pytest.approx(2 * (0.96 * 20 + 0.004 * 75))
    assert est.total(cell="sweep.astra", task_cell="F8S-960k") == pytest.approx(3 * 39.0)
    assert est.total(cell="sweep.astra", task_cell="F8S-256k") == pytest.approx(3 * 2 * (0.256 * 10 + 0.004 * 50))
    assert est.total(study="context_sweep") == pytest.approx(238.2, abs=1.0)
    assert est.total(study="context_sweep") <= plan.budget["allocations"]["context_sweep"]


# ---------- runner ----------


def test_offline_runner_builds_runs_reruns_and_analyzes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_sweep, "RUNS", tmp_path / "runs")
    args = ["--run-id", "t", "--offline", "--sizes", "8000", "--kinds", "fresh,memo"]
    assert run_sweep.main(["all", *args]) == 0
    assert run_sweep.main(["run", "--run-id", "t", "--offline", "--rerun", "--models", "luna", "--mock", "naive"]) == 0
    assert run_sweep.main(["analyze", "--run-id", "t", "--offline"]) == 0
    out = tmp_path / "runs" / "t" / "report"
    rows = json.loads((out / "results.json").read_text())["rows"]
    assert len(rows) == 2 * 3 + 2  # replicate 1: 2 kinds x 3 tiers; replicate 2: Luna only
    assert {(r["replicate"], r["tier"]) for r in rows if not r["success"]} == {(2, "luna")}
    md = (out / "report.md").read_text()
    assert "| 8K | ✓✗ | ✓✗ | 2/4 |" in md and "Replicates: 1, 2" in md
    assert "| held | none (first miss 8K) | 8K | 8K |" in md  # Luna's replicate 2 (naive) missed at 8K
    assert "| 8K | 2/4 (50%, 15–85) | 2/2 (100%, 34–100) | 2/2 (100%, 34–100) |" in md
    assert "| memo | 1/2 (50%, 9–91) |" in md
    # The design is fixed at the first step; a live run refuses while the plan cells are off.
    assert run_sweep.main(["run", "--run-id", "t", "--offline", "--sizes", "64000"]) == 2
    assert run_sweep.main(["run", "--run-id", "t"]) == 2
    assert run_sweep.main(["run", "--run-id", "live", "--models", "luna"]) == 2
    err = capsys.readouterr().err
    assert "fixed its design" in err and "is offline" in err and "are off" in err


def test_more_tasks_per_kind_get_their_own_seeds_and_pool_in_the_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(run_sweep, "RUNS", tmp_path / "runs")
    assert gen_sweep.task_seed("memo", 2, 40_000) == 40_000 + 3 * 2 + 1
    args = ["--run-id", "many", "--offline", "--sizes", "8000", "--kinds", "memo", "--tasks-per-kind", "2", "--models", "luna"]
    assert run_sweep.main(["all", *args]) == 0
    rows = json.loads((tmp_path / "runs" / "many" / "report" / "results.json").read_text())["rows"]
    assert sorted(r["seed"] for r in rows) == [40_001, 40_004] and all(r["success"] for r in rows)
    md = (tmp_path / "runs" / "many" / "report" / "report.md").read_text()
    assert "Tasks: 2 (2 per kind)" in md and "| 8K | 2/2 (100%, 34–100) |" in md


# ---------- requests the provider refuses ----------


def test_the_output_cap_keeps_input_and_output_inside_the_window():
    from ape.agent.sweep import MAX_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, output_cap

    assert output_cap(512_000, 2_000) == MAX_OUTPUT_TOKENS
    cap = output_cap(960_000, 4_000)  # the 960K world: ~4,000 messages
    assert 40_000 < cap < MAX_OUTPUT_TOKENS and 960_000 * 1.03 + 4 * 4_000 + cap <= 1_050_000
    assert output_cap(1_040_000, 4_000) == MIN_OUTPUT_TOKENS


def _refuser(limit_messages: int, how: str):
    """Gold-like answers for short histories; a refusal for histories longer than `limit_messages`."""
    import httpx
    import openai

    def agent(messages, tools, tool_choice, config):
        case = next(m.text for m in reversed(messages) if m.role == "user").split(":")[0].split()[-1]
        if len(messages) > limit_messages:
            if how == "context":
                return ModelOutput.from_content(MODEL, "maximum context length exceeded", stop_reason="model_length")
            text = "Invalid 'input': too many input items" if how == "items" else "Invalid value for 'tool_choice'"
            raise openai.BadRequestError(text, response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/responses")), body=None)
        return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": case, "action": "approve", "approver": "none", "deadline_days": 5, "document": "none"})

    return agent


@pytest.mark.parametrize(("how", "label"), [("context", "over_limit"), ("items", "over_limit"), ("other", "rejected")])
def test_a_refused_request_is_not_measured_and_does_not_error_the_sample(monkeypatch, tmp_path, worlds, how, label):
    scores = _run(monkeypatch, tmp_path, worlds, _refuser(40, how), kinds="fresh")
    small, large = scores["F8S-fresh-8k-sweep-s40000"], scores["F8S-fresh-64k-sweep-s40000"]
    assert small.value["measured"] == 1.0 and small.metadata["calls"][0]["max_output_tokens"] == 64_000
    assert large.value == {"success": 0.0, "answered": 0.0, "measured": 0.0}
    assert large.metadata["failure"] == label and large.metadata["calls"][0]["error"]


def test_the_runner_reports_a_refused_size_as_not_measured_with_the_recovery(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_sweep, "RUNS", tmp_path / "runs")
    args = ["all", "--run-id", "big", "--offline", "--sizes", "8000,64000", "--kinds", "fresh,memo", "--models", "luna", "--mock", "refuse-largest"]
    assert run_sweep.main(args) == 0
    out = capsys.readouterr().out
    assert "refused luna's requests at 64K" in out and "--output-tokens 2000" in out
    run = json.loads((tmp_path / "runs" / "big" / "run.json").read_text())
    assert run["replicates"]["1"]["tiers"]["luna"]["over_limit_sizes"] == [64_000]
    md = (tmp_path / "runs" / "big" / "report" / "report.md").read_text()
    assert "| held | 8K (64K not measured) |" in md
    assert "| 64K | – · 2 not measured |" in md and "| 8K | 2/2 (100%, 34–100) |" in md
    assert "## Not measured" in md and "| luna | 64K | over_limit | 2 | This model's maximum context length" in md and "| 64K | ⊘ | ⊘ | 0/0 |" in md


def test_the_recovery_rendering_fits_every_size_and_a_bulkier_one_is_refused(monkeypatch, tmp_path, capsys):
    """`--output-tokens 2000` (the advised recovery) still fits 8K with fewer messages; 3000 does not, and says why."""
    small = gen_sweep.generate_set("memo", [8_000, 64_000], output_tokens=2_000)
    assert all(w.entities["sweep"]["context_tokens"] <= w.entities["sweep"]["target_tokens"] for w in small)
    default = gen_sweep.generate_set("memo", [64_000])[0]
    assert len(gen_sweep.history(small[1])) < 0.7 * len(gen_sweep.history(default))
    monkeypatch.setattr(run_sweep, "RUNS", tmp_path / "runs")
    assert run_sweep.main(["build", "--run-id", "bulky", "--offline", "--sizes", "8000", "--kinds", "memo", "--output-tokens", "3000"]) == 2
    assert "use --output-tokens 2000 or less" in capsys.readouterr().err


def test_wilson_interval():
    from ape.analyze_sweep import wilson

    lo, hi = wilson(3, 3)
    assert (round(lo, 3), hi) == (0.438, 1.0)
    lo, hi = wilson(0, 3)
    assert lo == 0.0 and round(hi, 3) == 0.562
