"""The main study's tuning grid and the multi-agent arms' tuning knobs (BUILD_PLAN B3): the default prompt variant is
B2's notes byte for byte; every variant keeps the protocol rules; the knobs are per arm, validated when a solver is
built and recorded in the task metadata (S9, M1s, M1k and M2 read M1's, so the chain runs one protocol variant:
BUILD_REVIEW A-1, D-047); every candidate of `config/tuning_grid_main.yaml` builds its
solver and runs its own text through the real loop; the grid's static checks (equal budgets, own knobs, priced by the
plan); and a live tune refuses an unsigned grid. The offline tune over this grid runs in `tests/test_run_study.py`'s
offline `all`."""

import asyncio
import hashlib
import json
import os
import re
import string

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageTool, get_model

from ape import run_study, tuning
from ape.agent.multi import knobs as K
from ape.agent.multi import prompts as P
from ape.agent.solvers import arm_solver
from ape.apg.author import author_world
from ape.budget import load_plan
from ape.build import build
from ape.config import ROOT, Config
from ape.llm.fake import perfect_author
from ape.llm.mock_multi import GoldMulti
from ape.run_study import StudyRun
from ape.tasks.main import main_study
from ape.worlds.spec import World

M = "mockllm/model"
GRID = ROOT / "config" / "tuning_grid_main.yaml"
B3_SYSTEMS = {"S1", "S5", "S9", "M1", "M7", "M1k", "M2"}  # BUILD_PLAN B3: equal budgets for these
ROLES = ("s9", "orchestrator", "worker", "member", "critique", "chair")
# sha256 of B2's six role notes (git show f2161bd:src/ape/agent/multi/prompts.py), in ROLES order.
B2_NOTES_SHA = {
    "s9": "c4e09c16aeb4c9c9",
    "orchestrator": "5fac556585e9f3e2",
    "worker": "4abe7837e14f1e80",
    "member": "9397953435e2a090",
    "critique": "97bcdcaa30aa45fa",
    "chair": "648cf2710441c635",
}
# One cell per system where its arm applies, small worlds (the KG arms on authored worlds).
CELL = {"S9": ("F1", "8"), "M1": ("F7", "10"), "M7": ("F2", "5"), "M1k": ("F7", "10"), "M2": ("F3", "5")}


def _fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _literals(template: str) -> list[str]:
    """The template's fixed text between its fields (each long enough to identify the variant)."""
    return [lit.strip() for lit, *_ in string.Formatter().parse(template) if len(lit.strip()) >= 12]


def _grid() -> dict:
    return yaml.safe_load(GRID.read_text())


def _planned(plan=None) -> dict:
    plan = plan or load_plan()
    return tuning.planned_tuning((c.id, c.spec) for c in plan.cells if c.study == "main" and c.phase == "tuning" and c.enabled)


@pytest.fixture
def no_mas_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith(K.PREFIX)]:
        monkeypatch.delenv(k)
    return monkeypatch


# --- the variants -----------------------------------------------------------------------------------------------------


def test_the_default_variant_is_b2s_notes_byte_for_byte():
    d = P.VARIANTS[P.DEFAULT_VARIANT]
    assert (d.s9, d.orchestrator, d.worker, d.member, d.critique, d.chair) == (
        P.S9_NOTE, P.ORCHESTRATOR_NOTE, P.WORKER_NOTE, P.COUNCIL_MEMBER_NOTE, P.CRITIQUE_ROUND, P.CHAIR_NOTE,
    )  # fmt: skip
    assert {r: hashlib.sha256(getattr(d, r).encode()).hexdigest()[:16] for r in ROLES} == B2_NOTES_SHA, "B2's notes changed"
    assert K.resolve("M1", {}) == K.Knobs() and K.Knobs().notes is d and K.Knobs().clip == 2000


@pytest.mark.parametrize("name", sorted(P.VARIANTS))
def test_every_variant_keeps_the_protocol_rules(name):
    v, d = P.VARIANTS[name], P.VARIANTS[P.DEFAULT_VARIANT]
    for role in ROLES:
        text = getattr(v, role)
        assert _fields(text) == _fields(getattr(d, role)), (name, role, "a variant keeps each note's fields")
        assert "Subtask:" not in text and "proposed answer:" not in text, "the data formats the readers parse stay in their templates"
        assert not any(w in text.lower() for w in ("expert", "better", "best ", "superior", "smarter")), (name, role, "neutral wording")
    for role in ("orchestrator", "worker"):  # M1 and M1s must give the model identical inputs
        text = getattr(v, role).lower()
        assert not any(w in text for w in ("parallel", "concurrent", "simultaneous", "same time", "one after another", "in turn", "serial", "sequen")), (name, role)
    if name != P.DEFAULT_VARIANT:
        assert v != d and len({v.sha(), d.sha()}) == 2


# --- the knobs --------------------------------------------------------------------------------------------------------


def test_knobs_are_per_tuned_arm_and_the_orchestrator_chain_reads_m1s():
    env = {"APE_MAS_M1_PROMPT": "verify", "APE_MAS_M1_CLIP": "500", "APE_MAS_M7_PROMPT": "structured", "APE_MAS_S9_PROMPT": ""}
    chain = ("S9", "M1s", "M1", "M1k", "M2")
    assert {K.resolve(a, env) for a in chain} == {K.Knobs("verify", 500)}, "an empty APE_MAS_S9_PROMPT counts as unset, not as a knob"
    assert K.resolve("M7", env) == K.Knobs("structured")
    assert K.resolve("S8k3", env) == K.Knobs() and K.arm_knobs("S8k3") == []
    assert all(K.arm_knobs(a) == ["APE_MAS_M1_PROMPT", "APE_MAS_M1_CLIP"] for a in chain)
    assert set(K.KNOBS) == {f"APE_MAS_{a}_PROMPT" for a in ("M1", "M7")} | {f"APE_MAS_{a}_CLIP" for a in ("M1", "M7")}
    no_knobs = ("APE_MAS_S9_PROMPT", "APE_MAS_S9_CLIP", "APE_MAS_M1K_PROMPT", "APE_MAS_M2_PROMPT", "APE_MAS_M1K_CLIP", "APE_MAS_M2_CLIP")
    for stale in no_knobs:  # the chain has no knobs of its own
        with pytest.raises(ValueError, match=f"{stale} is not a multi-agent knob"):
            K.resolve("M1k", {stale: "concise"})


@pytest.mark.parametrize("env,match", [
    ({"APE_MAS_M1_PROMT": "verify"}, "APE_MAS_M1_PROMT is not a multi-agent knob"),
    ({"APE_MAS_S9_CLIP": "500"}, "APE_MAS_S9_CLIP is not a multi-agent knob"),
    ({"APE_MAS_M7_PROMPT": "fancy"}, "is not a prompt variant"),
    ({"APE_MAS_M1_CLIP": "50"}, "from 100 to 8000"),
    ({"APE_MAS_M7_CLIP": "lots"}, "from 100 to 8000"),
    ({"APE_MAS_M1K_PROMPT": "concise"}, "APE_MAS_M1K_PROMPT is not a multi-agent knob"),
])  # fmt: skip
def test_a_bad_knob_is_refused_when_any_multi_agent_solver_is_built(no_mas_env, env, match):
    with pytest.raises(ValueError, match=match):
        K.resolve("S9", env)
    for k, v in env.items():
        no_mas_env.setenv(k, v)
    for arm in ("S9", "M1", "M7", "S8k3"):  # whichever arm's knob it is: a group runs every selection's knobs at once
        with pytest.raises(ValueError, match=match):
            arm_solver(arm, max_turns=12)


def test_candidate_env_checks():
    assert K.candidate_problems("M1", {"APE_MAS_M1_PROMPT": "concise", "APE_MAS_M1_CLIP": 1000}) == []
    assert K.candidate_problems("M1k", {"APE_MAS_M1_PROMPT": "concise"}) == []  # M1k reads M1's knobs (it has no system)
    assert K.candidate_problems("M7", {"APE_MAS_M1_PROMPT": "concise"}) == ["M7 candidates set only M7's knobs (APE_MAS_M7_PROMPT, APE_MAS_M7_CLIP), not APE_MAS_M1_PROMPT"]
    assert K.candidate_problems("M7", {"APE_S3S_BUDGET": "1000"}) == ["M7 candidates set only M7's knobs (APE_MAS_M7_PROMPT, APE_MAS_M7_CLIP), not APE_S3S_BUDGET"]
    assert K.candidate_problems("S3s", {"APE_S3S_BUDGET": "1000"}) == []
    assert K.candidate_problems("S3s", {"APE_MAS_M7_PROMPT": "verify"}) == ["S3s is not a multi-agent arm, so it sets no APE_MAS_* knob (APE_MAS_M7_PROMPT)"]
    assert K.candidate_problems("M7", {"APE_MAS_M7_PROMPT": "nope"}) == ["APE_MAS_M7_PROMPT='nope' is not a prompt variant (default, concise, structured, verify)"]


# --- every candidate through the real loop ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mas-tuning")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_INDICES": tmp / "indices", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    for k in [k for k in os.environ if k.startswith(K.PREFIX) or k == "APE_KG_ARM"]:
        mp.delenv(k)
    cfg = Config()
    for fam, level in {*CELL.values()}:
        for wid in asyncio.run(build("dev", fam, [level], n_worlds=1, n_tasks=2, relational=True, embed=True)):
            if fam in ("F3", "F7"):
                asyncio.run(author_world(World.load(cfg.world_path(wid)), cfg, perfect_author))
    yield tmp
    mp.undo()


class Recorder(GoldMulti):
    """The gold mock, keeping the text of every message it is shown."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.seen: list[str] = []

    def __call__(self, messages, tools, tool_choice, config):
        self.seen += [m.text for m in messages if m.role in ("user", "tool")]
        return super().__call__(messages, tools, tool_choice, config)


def _run(env, arm, model, family=None, level=None):
    family, level = (family, level) if family else CELL[arm]
    gold = GoldMulti(env / "worlds")
    task = main_study(family=family, level=level, split="dev", arm=arm)
    log = inspect_eval(task, model=get_model(M, custom_outputs=model, memoize=False), model_roles={"kg": get_model(M, custom_outputs=gold.kg, memoize=False)},
                       log_dir=str(env / "logs"), display="none", limit=1)[0]  # fmt: skip
    assert log.status == "success", log.error
    assert log.samples[0].error is None, log.samples[0].error
    return task, log.samples[0]


ROLE_TEXTS = {"S9": ("s9",), "M1": ("orchestrator", "worker"), "M1k": ("orchestrator", "worker"), "M2": ("orchestrator", "worker"), "M7": ("member", "critique", "chair")}
CANDIDATES = [(name, c) for name, s in yaml.safe_load(GRID.read_text())["systems"].items() for c in s["candidates"]]


@pytest.mark.parametrize("system,cand", CANDIDATES, ids=[c["id"] for _, c in CANDIDATES])
def test_each_grid_candidate_builds_its_solver_records_its_knobs_and_runs_its_own_text(env, monkeypatch, system, cand):
    cand_env = {k: str(v) for k, v in cand["env"].items()}
    for k, v in cand_env.items():
        monkeypatch.setenv(k, v)
    rec = Recorder(env / "worlds")
    task, s = _run(env, cand["arm"], rec)
    assert task.metadata["arm"] == system and {k: task.metadata["knobs"][k] for k in cand_env} == cand_env, "recorded in the task metadata"
    assert s.scores["task_success"].value == "C", "the gold mock scores the variant through the real loop"
    knobs = K.resolve(cand["arm"])
    params = s.store["mas_params"]
    assert params["prompt_variant"] == knobs.prompt == cand_env.get(f"APE_MAS_{K.KNOB_ARM[system]}_PROMPT", "default")
    assert params["prompt_sha"] == knobs.notes.sha() and params.get("result_clip_tokens", 2000) == 2000
    shown = "\n".join(rec.seen)
    for role in ROLE_TEXTS[system]:
        assert all(lit in shown for lit in _literals(getattr(knobs.notes, role))), (role, "the model saw this variant's text")
    for other, notes in P.VARIANTS.items():
        if notes.orchestrator != knobs.notes.orchestrator and system in ("M1", "M1k", "M2"):
            assert not all(lit in shown for lit in _literals(notes.orchestrator)), f"{other}'s orchestrator note was shown too"


def test_m1s_runs_m1s_selection_with_identical_inputs(env, monkeypatch):
    monkeypatch.setenv("APE_MAS_M1_PROMPT", "structured")
    monkeypatch.setenv("APE_MAS_M7_PROMPT", "concise")  # another arm's knob moves neither
    inputs, params = {}, {}
    for arm in ("M1", "M1s"):
        rec = Recorder(env / "worlds")
        _, s = _run(env, arm, rec, "F1", "8")
        inputs[arm], params[arm] = sorted(rec.seen), s.store["mas_params"]
    assert inputs["M1"] == inputs["M1s"] and any(_literals(P.VARIANTS["structured"].orchestrator)[-1] in t for t in inputs["M1"])
    assert params["M1"]["prompt_variant"] == params["M1s"]["prompt_variant"] == "structured" and params["M1"]["prompt_sha"] == params["M1s"]["prompt_sha"]


class LongWinded(GoldMulti):
    """Workers pad their reports and members their rationales, so the clip shows."""

    PAD = "Notes on the lookup. " * 120

    def worker(self, messages, names):
        o = super().worker(messages, names)
        for c in o.message.tool_calls or []:
            if c.function == "report":
                c.arguments = {"result": f"{c.arguments['result']}\n{self.PAD}"}
        return o

    def __call__(self, messages, tools, tool_choice, config):
        o = super().__call__(messages, tools, tool_choice, config)
        for c in o.message.tool_calls or []:
            if "rationale" in c.arguments:
                c.arguments = {**c.arguments, "rationale": self.PAD}
        return o


@pytest.mark.parametrize("arm", ["M1", "M7"])
def test_the_clip_knob_cuts_what_one_agent_hands_another(env, monkeypatch, arm):
    cell = ("F1", "8") if arm == "M1" else CELL[arm]  # F1: three workers' results per round
    monkeypatch.setenv(f"APE_MAS_{arm}_CLIP", "150")
    _, s = _run(env, arm, LongWinded(env / "worlds"), *cell)
    assert s.store["mas_params"]["result_clip_tokens"] == 150 and s.scores["task_success"].value == "C"
    assert _handed(arm, s) and all("[... cut to 150 tokens]" in t for t in _handed(arm, s))
    monkeypatch.delenv(f"APE_MAS_{arm}_CLIP")
    _, s = _run(env, arm, LongWinded(env / "worlds"), *cell)
    assert s.store["mas_params"]["result_clip_tokens"] == 2000 and _handed(arm, s) and not any("[... cut to" in t for t in _handed(arm, s)), "the padding fits the default clip"


def _handed(arm, s) -> list[str]:
    """What the orchestrator read of its workers (M1), or the members' recorded rationales (M7)."""
    if arm == "M1":
        return [m.text for m in s.messages if isinstance(m, ChatMessageTool) and m.function == "delegate"]
    return [p["rationale"] for phase in s.store["mas_council"]["phases"] for p in phase if p["rationale"]]


@pytest.mark.parametrize("variant", sorted(P.VARIANTS))
def test_the_orchestrator_chain_runs_one_variant_byte_for_byte(env, no_mas_env, variant):
    """BUILD_REVIEW A-1: under any M1 variant, M1 -> M1k changes only the delivery and M1k -> M2 only the specialization.
    The orchestrator's and the workers' notes are byte-identical along the chain, but for M2's roster and specialty."""
    no_mas_env.setenv("APE_MAS_M1_PROMPT", variant)
    seen, params = {}, {}
    for arm in ("M1", "M1k", "M2"):
        rec = ChainRecorder(env / "worlds")
        _, s = _run(env, arm, rec, "F7", "10")
        seen[arm], params[arm] = rec.first, s.store["mas_params"]
    assert {p["prompt_variant"] for p in params.values()} == {variant} and len({p["prompt_sha"] for p in params.values()}) == 1
    # M1 -> M1k: every role note identical; only the system prompts differ (the knowledge's delivery).
    assert seen["M1"]["orchestrator"] == seen["M1k"]["orchestrator"] and seen["M1"]["worker"] == seen["M1k"]["worker"]
    assert seen["M1"]["system"] != seen["M1k"]["system"]
    # M1k -> M2: the orchestrator's note differs in the team text only, the worker's in the specialty sentence only.
    a, b = seen["M1k"]["orchestrator"].splitlines(), seen["M2"]["orchestrator"].splitlines()
    only_a, only_b = [x for x in a if x not in b], [y for y in b if y not in a]
    assert len(only_a) == 1 and only_a[0].startswith(P.TEAM_IDENTICAL.split("{")[0])
    assert only_b[0] == P.TEAM_SPECIALISTS.split("\n")[0] and all(line.startswith("- specialist_") for line in only_b[1:])
    assert [x for x in a if x in b] == [y for y in b if y in a]
    assert re.sub(r" You are the team's specialist for: [^\n]*\.", "", seen["M2"]["worker"], count=1) == seen["M1k"]["worker"] != seen["M2"]["worker"]
    assert seen["M1k"]["system"] == seen["M2"]["system"]


@pytest.mark.parametrize("variant", sorted(P.VARIANTS))
def test_s9_runs_m1s_variant(env, no_mas_env, variant):
    """D-047: S9 joins the chain (M1's contrast is M1 - S9, the ISO step M1s - S9). Under each M1 variant S9 and M1s record
    the same variant and text hash, and S9's planning note is that variant's (`RoleNotes.s9`), no other's."""
    no_mas_env.setenv("APE_MAS_M1_PROMPT", variant)
    params, shown = {}, {}
    for arm in ("S9", "M1s"):
        rec = Recorder(env / "worlds")
        _, s = _run(env, arm, rec, "F1", "8")
        params[arm], shown[arm] = s.store["mas_params"], "\n".join(rec.seen)
    assert params["S9"]["prompt_variant"] == params["M1s"]["prompt_variant"] == variant
    assert params["S9"]["prompt_sha"] == params["M1s"]["prompt_sha"] == P.VARIANTS[variant].sha()
    assert "result_clip_tokens" not in params["S9"] and params["M1s"]["result_clip_tokens"] == 2000
    mine = P.VARIANTS[variant]
    assert all(lit in shown["S9"] for lit in _literals(mine.s9)), "S9 ran the variant's planning note"
    assert all(lit in shown["M1s"] for lit in _literals(mine.orchestrator)), "M1s ran the variant's orchestrator note"
    for other, notes in P.VARIANTS.items():
        if notes.s9 != mine.s9:
            assert not all(lit in shown["S9"] for lit in _literals(notes.s9)), f"{other}'s planning note was shown"


class ChainRecorder(GoldMulti):
    """The gold mock, keeping the system prompt and the first user message the orchestrator and a worker are shown."""

    def __init__(self, worlds):
        super().__init__(worlds)
        self.first: dict[str, str] = {}

    def __call__(self, messages, tools, tool_choice, config):
        names = {t.name for t in tools}
        role = "worker" if "report" in names else "orchestrator" if "delegate" in names else None
        if role and role not in self.first:
            self.first[role] = messages[1].text
            if role == "orchestrator":
                self.first["system"] = messages[0].text
        return super().__call__(messages, tools, tool_choice, config)


# --- the grid -----------------------------------------------------------------------------------------------------------


def test_the_main_grid_gives_every_b3_system_an_equal_budget_or_a_reason():
    grid = _grid()
    systems, inherited = grid["systems"], grid["inherited"]
    assert set(systems) | set(inherited) >= B3_SYSTEMS and not set(systems) & set(inherited)
    assert set(systems) == {"M1", "M7"}
    assert {a: inherited[a]["source"] for a in inherited} == {"S1": "fixed", "S5": "gate", "S3s": "gate", "S9": "M1", "M1k": "M1", "M2": "M1"}  # A-1, D-047
    assert systems["M1"]["dev_cells"] == ["F1-32", "F2-10", "F3-60", "F7-1000"], "M1's selection is the whole chain's: it tunes on Study A's and B's cells"
    assert grid["equal_budgets"] is True and grid["budget_per_system"] == 4 and "placeholder" not in grid
    for name, s in systems.items():
        cands = s["candidates"]
        assert len(cands) == 4 and {c["arm"] for c in cands} == {name}
        assert [c["env"][f"APE_MAS_{K.KNOB_ARM[name]}_PROMPT"] for c in cands] == list(P.VARIANTS), "every variant, B2's first (the offline tune runs the first two)"
    assert set(grid["owners"]) == set(grid["signed_off"]) == {*systems, "S1"} and set(grid["owners"].values()) == {"TODO"} and set(grid["signed_off"].values()) == {False}
    assert run_study.OFFLINE_SCALE["tune_candidates_per_system"] == 2


def test_the_main_grid_passes_its_static_checks_against_the_plan():
    grid, planned = _grid(), _planned()
    problems = tuning.study_grid_problems(grid, planned, K.candidate_problems)
    # Until run_plan.yaml's tuning cells drop them, they may still price arms this grid inherits (S3s from the gate; S9,
    # M1k and M2 from M1, BUILD_REVIEW A-1, D-047): each is a redundant spend; once dropped, none.
    source = {"S3s": "gate", "S9": "M1", "M1k": "M1", "M2": "M1"}
    redundant = [f"{a} is inherited ({source[a]}), yet {', '.join(planned[a]['candidates'])} prices tuning it: a redundant spend; drop it from the plan"
                 for a in planned if a in source]
    assert problems == redundant
    for arm in source:
        planned.pop(arm, None)
    assert tuning.study_grid_problems(grid, planned, K.candidate_problems) == []
    for name, s in grid["systems"].items():
        assert s["dev_cells"] == planned[name]["cells"]


def _edit(grid: dict, system: str, i: int, **cand) -> dict:
    g = json.loads(json.dumps(grid))
    g["systems"][system]["candidates"][i] |= cand
    return g


def test_the_static_checks_catch_what_the_tune_or_the_pilot_would_find_too_late():
    grid, planned = _grid(), _planned()
    for arm in ("S3s", "S9", "M1k", "M2"):  # inherited (see above)
        planned.pop(arm, None)

    def problems(g, p=planned):
        return tuning.study_grid_problems(g, p, K.candidate_problems)

    assert problems(_edit(grid, "M7", 1, env={"APE_MAS_M1_PROMPT": "concise"})) == [
        "M7: candidate m7-concise: M7 candidates set only M7's knobs (APE_MAS_M7_PROMPT, APE_MAS_M7_CLIP), not APE_MAS_M1_PROMPT",
    ]  # each arm runs under exactly its own selection's knobs (D-042): the arm's own-knob check is the rule, not a shared name
    assert problems(_edit(grid, "M7", 2, env={"APE_MAS_M7_PROMPT": "fancy"})) == ["M7: candidate m7-structured: APE_MAS_M7_PROMPT='fancy' is not a prompt variant (default, concise, structured, verify)"]
    assert problems(_edit(grid, "M1", 0, arm="M1k")) == ["M1: candidate m1-default runs M1k, not M1 (a system's candidates run as the plan arm it is named for)"]
    assert problems(_edit(grid, "M1", 3, id="m1-default")) == ["M1: candidate ids ['m1-default'] are declared more than once"]
    short = json.loads(json.dumps(grid))
    short["systems"]["M7"]["candidates"].pop()
    assert problems(short) == ["M7: 3 candidates; equal budgets give every system budget_per_system (4)", "M7: 3 candidates, but run_plan.yaml prices {'main.tune.a': 4}"]
    assert problems({**short, "equal_budgets": False}) == ["M7: 3 candidates, but run_plan.yaml prices {'main.tune.a': 4}"]
    moved = json.loads(json.dumps(grid))
    moved["systems"]["M1"]["dev_cells"] = ["F1-32", "F2-10"]
    assert problems(moved) == ["M1: dev cells ['F1-32', 'F2-10'], but run_plan.yaml prices it on ['F1-32', 'F2-10', 'F3-60', 'F7-1000']"]
    dropped = json.loads(json.dumps(grid))
    del dropped["systems"]["M7"]
    assert problems(dropped) == ["main.tune.a prices tuning M7, which the grid does not declare"]
    extra = json.loads(json.dumps(grid))
    extra["systems"]["S1"] = {"dev_cells": ["F1-32"], "candidates": [{"id": "s1", "arm": "S1", "env": {}}]}
    assert problems(extra) == [
        "S1: 1 candidates; equal budgets give every system budget_per_system (4)",
        "S1 is both tuned here and inherited ({'source': 'fixed', 'owner': 'skeptic'})",
        "S1: no run_plan.yaml tuning cell prices it",
    ]


# --- sign-off -----------------------------------------------------------------------------------------------------------


def test_a_live_main_tune_refuses_without_owners_and_sign_off(tmp_path):
    grid = _grid()
    problems = run_study.tuning_signoff_problems(grid)
    assert {f"owners.{s} is not set" for s in grid["systems"]} | {f"signed_off.{s} is not true" for s in grid["systems"]} <= set(problems)
    live = StudyRun("main", "tune", runs_root=tmp_path / "runs")
    reason = run_study._refuse_tune(live)
    assert reason and "is not signed off for a live tune" in reason and "owners.M1 is not set" in reason
    assert run_study._refuse_tune(StudyRun("main", "tune", offline=True, runs_root=tmp_path / "runs")) is None, "offline tunes need no sign-off"
    signed = grid | {"owners": dict.fromkeys(grid["owners"], "Ada") | {"S1": "Sam"}, "signed_off": dict.fromkeys(grid["signed_off"], True)}  # the skeptic (S1) is not the M-arm author
    assert run_study.tuning_signoff_problems(signed) == []
