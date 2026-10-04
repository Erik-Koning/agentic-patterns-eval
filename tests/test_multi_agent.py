"""Multi-agent and ensemble arms (BUILD_PLAN B2): the gold mock scores 100% through the real loop for every arm where it
applies, the per-agent accounting sums exactly to the sample's usage, M1 and M1s give the model identical inputs,
S8k3's attempts never share environment state, the switch vectors are single-switch contrasts, and the shared turn
loop gives the model exactly what `kb_agent` gives it."""

import asyncio
import json

import anyio
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, get_model
from inspect_ai.solver import Generate, TaskState, solver

from ape.agent.arms import arm_provider, load_world
from ape.agent.multi import prompts as P
from ape.agent.multi.core import AGENT_MODEL, TOKEN_FIELDS, Delivery, Team, react_loop
from ape.agent.multi.primitives import answer_key, plurality
from ape.agent.multi.solvers import REGISTRY, S1_SWITCHES, switch_vector
from ape.agent.multi.specialists import specialists
from ape.agent.solvers import MULTI_AGENT_ARMS, PLANNED_MULTI_AGENT_ARMS
from ape.apg.author import author_world
from ape.build import build
from ape.config import Config
from ape.llm.fake import perfect_author
from ape.llm.mock_multi import GoldMulti, out
from ape.scorers.success import task_success
from ape.tasks.gate import gate, gate_samples
from ape.tasks.main import main_study
from ape.worlds.env_tools import ANSWER, CALLS, always_on, build_tools
from ape.worlds.generate import make_world
from ape.worlds.spec import TaskItem, World

M = "mockllm/model"
CELLS = {"F1": "8", "F2": "5", "F3": "5", "F7": "10"}
KG_CELLS = [("F3", "5"), ("F3", "20"), ("F7", "10"), ("F1", "8")]
N_TASKS = 3


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("multi")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv("APE_KG_ARM", raising=False)
    built = set()
    for fam, level in [*CELLS.items(), ("F3", "20")]:
        ids = asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=N_TASKS, relational=True, embed=True))
        built.update(ids)
    cfg = Config()
    for wid in sorted(built):
        if not wid.startswith("F2"):
            asyncio.run(author_world(World.load(cfg.world_path(wid)), cfg, perfect_author))
    yield tmp
    mp.undo()


def _eval(env, family, level, arm, model, **kw):
    gold = GoldMulti(env / "worlds")
    roles = {"kg": get_model(M, custom_outputs=gold.kg, memoize=False)}
    log = inspect_eval(main_study(family=family, level=level, split="dev", arm=arm), model=get_model(M, custom_outputs=model, memoize=False),
                       model_roles=roles, log_dir=str(env / "logs"), display="none", **kw)[0]
    assert log.status == "success", log.error
    return log


def _usage_sums_exactly(s) -> None:
    acc = s.store["mas_accounting"]
    assert acc["unattributed"] == {}, "every model call belongs to an agent"
    for f in ("input_tokens", "output_tokens", "total_tokens"):
        per_agent = sum(b[f] for a in acc["agents"].values() for b in a["models"].values())
        assert per_agent == sum(getattr(u, f) for u in s.model_usage.values()) == acc["totals"][f], f
    calls = [e for e in s.events if e.event == "model"]
    assert sum(b["calls"] for a in acc["agents"].values() for b in a["models"].values()) == len(calls)
    # Per-model-role usage agrees with Inspect's own per-role accounting (the kg role's classify calls).
    for role, usage in s.role_usage.items():
        assert sum(a["models"].get(role, {}).get("total_tokens", 0) for a in acc["agents"].values()) == usage.total_tokens, role


APPLIES = [
    *[("S9", f) for f in ("F1", "F2")],
    *[(a, f) for a in ("M1", "M1s", "S8k3") for f in CELLS],
    *[("M7", f) for f in ("F1", "F2")],
    *[(a, f, lv) for a in ("M1k", "M2") for f, lv in KG_CELLS if (a, f) != ("M1k", "F1")],
]


@pytest.mark.parametrize("case", APPLIES, ids=lambda c: "-".join(c))
def test_gold_mock_scores_100_percent_through_the_real_loop(env, case):
    arm, family = case[:2]
    level = case[2] if len(case) == 3 else CELLS[family]
    log = _eval(env, family, level, arm, GoldMulti(env / "worlds"))
    assert all(s.error is None and s.limit is None for s in log.samples), [s.error for s in log.samples]
    assert [s.scores["task_success"].value for s in log.samples] == ["C"] * N_TASKS
    for s in log.samples:
        sw = s.store["mas_switches"]
        assert {k: sw[k] for k in S1_SWITCHES} == {**S1_SWITCHES, **REGISTRY[arm]} | ({"DEL": "KGs"} if arm in ("M1k", "M2") else {})
        assert s.store["mas_agents"] and s.store["compile_log"] and s.store["arm"]["multi_agent"] is True
        assert all(r["agent"] in {a["id"] for a in s.store["mas_agents"]} for r in s.store["compile_log"])
        assert s.scores["delivered_evidence"].value["compiles"] == len(s.store["compile_log"])
        _usage_sums_exactly(s)
        roles = {a["role"] for a in s.store["mas_agents"]}
        if arm == "S9":
            assert roles == {"planner"} and s.store["mas_plan"][0]["turn"] == 0
        elif arm in ("M1", "M1s", "M1k", "M2"):
            assert "orchestrator" in roles and s.store["mas_rounds"] and s.store["mas_plan"]
            assert all(1 <= len(r["subtasks"]) <= 3 for r in s.store["mas_rounds"])
            workers = roles - {"orchestrator"}
            assert (workers == {"worker"}) if arm != "M2" else all(r.startswith("specialist_") for r in workers)
            # Only the orchestrator answers; workers never get the answer tool.
            workers = [a for a in s.store["mas_agents"] if a["role"] != "orchestrator"]
            assert not any(s.metadata["task"]["answer_tool"] in a["tool_calls"] for a in workers)
        elif arm == "M7":
            assert roles == {"member", "chair"}
            assert len(s.store["mas_council"]["phases"]) == 3 and all(len(p) == 3 for p in s.store["mas_council"]["phases"])
        elif arm == "S8k3":
            ens = s.store["mas_ensemble"]
            assert roles == {"attempt"} and ens["aggregation"] == "majority" and ens["winner"] == 1


def test_m2_routes_by_specialty_and_specialists_hold_only_their_tools(env):
    log = _eval(env, "F3", "20", "M2", GoldMulti(env / "worlds"))
    world = load_world(log.samples[0].metadata["world_id"])
    specs = {s.name: s for s in specialists(world)}
    for s in log.samples:
        for a in s.store["mas_agents"]:
            if a["role"].startswith("specialist_"):
                own = set(specs[a["role"]].tools) | {"report"}
                assert set(a["tool_calls"]) <= own
        exposed = [st for st in s.store["step_log"] if st["agent"] != "orchestrator"]
        assert exposed and all(set(st["exposed_tools"]) <= set(specs[next(a["role"] for a in s.store["mas_agents"] if a["id"] == st["agent"])].tools) | {"report"} for st in exposed)
        assert s.store["mas_specialists"] == [{"name": n, "covers": list(sp.covers), "tools": list(sp.tools)} for n, sp in specs.items()]


def test_specialists_partition_the_world_and_cover_every_domain():
    w60 = make_world("F3", "60", "dev", 0, 2)
    specs = specialists(w60)
    assert [s.name for s in specs] == ["specialist_1", "specialist_2", "specialist_3"]
    covered = [d for s in specs for d in s.covers]
    assert sorted(covered) == sorted({t.domain for t in w60.tools}) and len(covered) == len(set(covered))
    for s in specs:
        assert set(s.tools) == {"order_lookup", *(t.name for t in w60.tools if t.domain in s.covers)}
        assert s.query_prefix.startswith("Specialty: ")
    w5 = make_world("F3", "5", "dev", 0, 2)  # two domains, three specialists: dealt cyclically, none empty
    assert [s.covers for s in specialists(w5)] == [(d,) for d in [*sorted({t.domain for t in w5.tools})] * 2][:3]
    f7 = make_world("F7", "1000", "dev", 0, 2)
    assert sorted(d for s in specialists(f7) for d in s.covers) == sorted({p.domain for p in f7.policies})
    assert all(s.tools is None for s in [*specialists(f7), *specialists(make_world("F1", "8", "dev", 0, 2))])
    with pytest.raises(ValueError, match="F2"):
        specialists(make_world("F2", "5", "dev", 0, 2))


# --- M1 vs M1s: identical inputs, different latency -------------------------------------------------------------------


def _canon(messages) -> str:
    def one(m):
        rec = {"role": m.role, "text": m.text}
        if isinstance(m, ChatMessageAssistant):
            rec["calls"] = [(c.id, c.function, c.arguments) for c in m.tool_calls or []]
        if isinstance(m, ChatMessageTool):
            rec |= {"id": m.tool_call_id, "fn": m.function, "error": m.error.message if m.error else None}
        return rec

    return json.dumps([one(m) for m in messages], sort_keys=True)


class Recorder:
    """The gold mock, slowed down (so concurrency shows in the wall-clock), recording every input it is given."""

    def __init__(self, gold, delay: float):
        self.gold, self.delay, self.inputs = gold, delay, []

    async def __call__(self, messages, tools, tool_choice, config):
        self.inputs.append(_canon(messages) + "|" + ",".join(sorted(t.name for t in tools)) + f"|{tool_choice}")
        await anyio.sleep(self.delay)
        return self.gold(messages, tools, tool_choice, config)


@pytest.mark.parametrize("family", ["F1", "F3"])
def test_m1_and_m1s_give_the_model_identical_inputs_and_differ_only_in_latency(env, family):
    level = CELLS[family]
    runs = {}
    for arm in ("M1", "M1s"):
        rec = Recorder(GoldMulti(env / "worlds"), delay=0.05 if family == "F1" else 0.0)
        log = _eval(env, family, level, arm, rec, max_samples=1)
        runs[arm] = (rec, log)
    (r1, l1), (rs, ls) = runs["M1"], runs["M1s"]
    assert sorted(r1.inputs) == sorted(rs.inputs) and len(r1.inputs) > 0
    for a, b in zip(l1.samples, ls.samples, strict=True):
        assert a.scores["task_success"].value == b.scores["task_success"].value == "C"
        for key in (ANSWER, CALLS, "mas_plan", "mas_rounds", "env_lookups"):
            assert a.store.get(key) == b.store.get(key), key
        strip = lambda agents: [{k: v for k, v in x.items() if k != "wall_s"} for x in agents]  # noqa: E731
        assert strip(a.store["mas_agents"]) == strip(b.store["mas_agents"])
        assert a.store["mas_params"]["workers_scheduled"] == "concurrent" and b.store["mas_params"]["workers_scheduled"] == "serial"
    if family == "F1":  # three rounds of 3, 3 and 2 workers, two calls each
        wall = {arm: sum(s.store["mas_accounting"]["sample_wall_s"] for s in log.samples) for arm, (_, log) in runs.items()}
        assert wall["M1"] < 0.8 * wall["M1s"], wall
        assert all(s.store["mas_accounting"]["realized_parallelism"] > 1.2 for s in l1.samples)


# --- S8k3: isolation, vote, adoption ----------------------------------------------------------------------------------


class Divergent(GoldMulti):
    """S8k3 attempts that behave differently: the i-th attempt to start plays `plays[i]` (gold or a wrong procedure).
    An attempt is recognised later by its first tool call's ID, which encodes its play."""

    def __init__(self, worlds, plays):
        super().__init__(worlds)
        self.plays, self.started = plays, 0

    def __call__(self, messages, tools, tool_choice, config):
        names = {t.name for t in tools}
        if "select_answer" in names:
            return super().__call__(messages, tools, tool_choice, config)
        world, task = self.task_of(messages)
        first = next((m for m in messages if isinstance(m, ChatMessageAssistant)), None)
        if first is None:
            play = self.plays[self.started]
            self.started += 1
            o = out(messages, ("order_lookup", {"order_id": task.tags["order_id"]}))
            o.message.tool_calls[0].id = f"call_{play}_{self.started}"
            return o
        play = first.tool_calls[0].id.split("_")[1]
        made = [(c.function, c.arguments) for m in messages if isinstance(m, ChatMessageAssistant) for c in m.tool_calls or []]
        want = task.gold["calls"] if play == "gold" else self.wrong(world, task, play)
        pending = [c for c in want if (c["tool"], c["args"]) not in made]
        return out(messages, (pending[0]["tool"], pending[0]["args"]) if pending else ("finish", {}))

    @staticmethod
    def wrong(world, task, play):
        other = [p for p in world.procedures if p.situation != f"{task.tags['domain']}/{task.setup['orders'][task.tags['order_id']]['region']}"]
        proc = other[0 if play == "wrongA" else 1]
        return [{"tool": s["tool"], "args": {k: (task.tags["order_id"] if v == "{order_id}" else v) for k, v in s["args"].items()}} for s in proc.steps]


@pytest.mark.parametrize("plays,verdict,how", [
    (["gold", "wrongA", "gold"], "C", "majority"),
    (["wrongA", "gold", "wrongB"], "C", "aggregator"),
    (["wrongA", "gold", "wrongA"], "I", "majority"),
])
def test_s8k3_attempts_never_share_the_environment_and_the_winner_is_scored(env, plays, verdict, how):
    log = _eval(env, "F3", "5", "S8k3", Divergent(env / "worlds", plays), limit=1)
    s = log.samples[0]
    task = TaskItem(**s.metadata["task"])
    ens = s.store["mas_ensemble"]
    assert s.scores["task_success"].value == verdict and ens["aggregation"] == how
    world = load_world(s.metadata["world_id"])
    # Each attempt saw only its own environment: its answer (finish) lists only its own calls.
    by_attempt = {a["id"]: a for a in s.store["mas_agents"]}
    assert set(by_attempt) >= {"attempt_1", "attempt_2", "attempt_3"}
    for answer in ens["answers"]:
        assert len(answer["calls"]) == 2
    # The sample's end state is exactly the winner's, never a union.
    winner = ens["answers"][ens["winner"] - 1]
    assert s.store[CALLS] == winner["calls"] and s.store[ANSWER] == winner
    wanted = task.gold["calls"] if verdict == "C" else Divergent.wrong(world, task, "wrongA")
    canon = lambda calls: sorted(json.dumps(c, sort_keys=True) for c in calls)  # noqa: E731
    assert canon(s.store[CALLS]) == canon(wanted)
    if how == "aggregator":
        assert ens["tied"] == [1, 2, 3] and "aggregator" in by_attempt
    _usage_sums_exactly(s)


def test_vote_keys_follow_the_scorer():
    f2 = TaskItem(id="t", world_id="w", family="F2", level="2", prompt="", gold={"final": "SUP-2", "chain": ["SUP-1", "SUP-2"]}, gold_fact_ids=[], answer_tool="submit_chain")
    # Same final, different path: one vote (the scorer grades the final supplier).
    assert answer_key(f2, {"final": "sup-2 ", "chain": ["x"]}, []) == answer_key(f2, {"final": "SUP-2", "chain": []}, [])
    f7 = TaskItem(id="t", world_id="w", family="F7", level="10", prompt="", gold={}, gold_fact_ids=[], answer_tool="submit_decision")
    assert answer_key(f7, {"action": "deny", "approver": "none", "deadline_days": "3", "document": "none"}, []) == answer_key(f7, {"action": "deny", "approver": "none", "deadline_days": 3, "document": "none"}, [])
    f3 = TaskItem(id="t", world_id="w", family="F3", level="5", prompt="", gold={}, gold_fact_ids=[], answer_tool="finish")
    a, b = {"tool": "x", "args": {"o": 1}}, {"tool": "y", "args": {"o": 1}}
    assert answer_key(f3, None, [a, b]) == answer_key(f3, None, [b, a]) and answer_key(f3, None, []) is None
    assert plurality(["a", None, "a"]) == ([0], True)
    assert plurality(["a", "b", "c"]) == ([0, 1, 2], False)
    assert plurality(["b", "a", "a", "b"]) == ([0, 1], False)
    assert plurality([None, None]) == ([], False)


# --- switches -------------------------------------------------------------------------------------------------------


class _Arm:
    def __init__(self, name, per_step):
        self.name, self.per_step = name, per_step


@pytest.mark.parametrize("a,b,switch", [
    ("S9", "S1", "DEC"), ("M1s", "S9", "ISO"), ("M1", "M1s", "CONC"), ("M1k", "M1", "DEL"), ("M2", "M1k", "SPEC"),
    ("M7", "S8k3", "COMM"), ("S8k3", "S1", "ENS"), ("M5", "M1", "STATE"), ("M5-spec", "M2", "STATE"),
])
def test_each_mechanism_contrast_differs_in_exactly_one_switch(a, b, switch):
    def vec(arm):
        if arm == "S1":
            return dict(S1_SWITCHES)
        kg = arm in ("M1k", "M2", "M5-spec")
        sw = switch_vector(arm, _Arm("APG-s" if kg else "S1", kg))
        return {k: v for k, v in sw.items() if k in S1_SWITCHES}

    va, vb = vec(a), vec(b)
    assert [k for k in S1_SWITCHES if va[k] != vb[k]] == [switch]


def test_every_planned_arm_is_registered():
    assert set(PLANNED_MULTI_AGENT_ARMS) == set(MULTI_AGENT_ARMS) == set(REGISTRY)
    with pytest.raises(ValueError, match="push delivery only"):
        MULTI_AGENT_ARMS["M1"]("M1", exposure="retrieved", max_turns=12, delivery="pull")


def test_arms_refuse_families_they_are_not_defined_on(env):
    for arm, family in (("M7", "F3"), ("M2", "F2")):
        log = inspect_eval(main_study(family=family, level=CELLS[family], split="dev", arm=arm), model=get_model(M, custom_outputs=GoldMulti(env / "worlds")),
                           log_dir=str(env / "logs"), display="none", limit=1, fail_on_error=False)[0]
        s = log.samples[0]
        assert s.error is not None and "not defined on" in s.error.message
        assert s.store["mas_switches"] == {} or "mas_agents" in s.store


# --- the shared turn loop is kb_agent's -------------------------------------------------------------------------------


@solver
def _loop_as_single_agent(arm: str = "S1"):
    """`core.react_loop` configured as a single agent of `arm`, as kb_agent runs it."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        team = Team(state, "loop", {}, {})
        dlv = Delivery(await arm_provider(arm)(team.world), team)
        agent = team.agent("a", "single")
        task = team.task
        async with team.running(agent):
            ctx = await dlv.compile(agent, task.prompt, 0)
            state.messages = [dlv.system(ctx), ChatMessageUser(content=task.prompt)]
            await react_loop(team, agent, state.messages, delivery=dlv, ctx=ctx, query=task.prompt, tools=build_tools(team.world, task),
                             always=always_on(team.world), exposure="retrieved", max_turns=Config().max_turns, done=team.answered,
                             nudge=P.NUDGE_TASK.format(answer_tool=task.answer_tool))
        team.write(dlv, "retrieved")
        return state

    return solve


class Script:
    def __init__(self, steps):
        self.steps, self.i, self.inputs = steps, 0, []

    def __call__(self, messages, tools, tool_choice, config):
        self.inputs.append(_canon(messages) + "|" + ",".join(t.name for t in tools))
        step = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return step(messages) if callable(step) else step


@pytest.mark.parametrize("arm", ["S1", "S3s", "APG-s"])
def test_the_shared_loop_gives_the_model_exactly_what_kb_agent_gives_it(env, arm):
    sample = gate_samples("F7", "10", "dev")[:1]
    t = TaskItem(**sample[0].metadata["task"])
    steps = [
        lambda m: out(m, text="Let me think."),
        lambda m: out(m, ("lookup_customer", {"customer_id": t.tags["customer_id"]})),
        lambda m: out(m, text="Policy says so."),
        lambda m: out(m, ("lookup_customer", {"customer_id": t.tags["customer_id"]})),
        lambda m: out(m, ("submit_decision", dict(t.gold))),
    ]
    gold = GoldMulti(env / "worlds")
    roles = {"kg": get_model(M, custom_outputs=gold.kg, memoize=False)}
    inputs, stores = [], []
    for task in (gate(family="F7", level="10", split="dev", arm=arm), Task(dataset=MemoryDataset(sample), solver=_loop_as_single_agent(arm), scorer=[task_success()])):
        script = Script(steps)
        log = inspect_eval(task, model=get_model(M, custom_outputs=script, memoize=False), model_roles=roles, log_dir=str(env / "logs"), display="none", limit=1)[0]
        assert log.status == "success" and log.samples[0].scores["task_success"].value == "C"
        inputs.append(script.inputs)
        stores.append(log.samples[0].store)
    assert inputs[0] == inputs[1]
    ref, loop = stores
    assert [(r["fact_ids"], r["prompt_hash"]) for r in ref["compile_log"]] == [(r["fact_ids"], r["prompt_hash"]) for r in loop["compile_log"]]
    assert [s["exposed_tools"] for s in ref["step_log"]] == [s["exposed_tools"] for s in loop["step_log"]]
    assert ref["turns_used"] == loop["turns_used"] and ref["nudges"] == loop["nudges"] == 2


def test_accounting_attributes_kg_calls_to_the_agent_that_compiled(env):
    log = _eval(env, "F7", "10", "M1k", GoldMulti(env / "worlds"), limit=1)
    s = log.samples[0]
    agents = s.store["mas_accounting"]["agents"]
    kg = {aid: a["models"].get("kg", {}).get("calls", 0) for aid, a in agents.items()}
    compiles = {aid: sum(1 for r in s.store["compile_log"] if r["agent"] == aid and not r["meta"]["route"]["bypass"]) for aid in agents}
    assert kg == compiles and sum(kg.values()) > 0
    for a in agents.values():
        assert a["calls"] == a["models"][AGENT_MODEL]["calls"] and set(TOKEN_FIELDS) <= set(a)
