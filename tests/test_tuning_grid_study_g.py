"""Study G's tuning grid (BUILD_PLAN B11; `config/tuning_grid_study_g.yaml`).

- The grid tunes exactly the systems run_plan.yaml's g.tune.luna prices, on its dev cell, within its budget, each
  system as its own plan arm, unsigned until the owners sign off (a live tune refuses it), and names every other
  session arm of the plan as untuned with a reason.
- Every candidate is valid for its arm by the arm's own parser and validator (`context_policy.resolve_knobs`,
  `validate`), sets only APE_CM_* knobs its arm reads (a typo or another arm's knob would silently tune nothing), and
  is the arm's whole configuration: exactly the knobs its components read. The candidates of a system differ, the
  arm's default (D-037) comes first, and CM-sum's and CM-todo's two candidates cover their one knob.
- So every tuned arm runs in the test exactly as it ran on dev, for every combination of selections, under the
  study runner's rule for selections that share a knob (agree: together; disagree: each tuned arm apart), and the
  untuned arms keep their defaults.
- Every candidate runs a perfect gold session through the real loop with its knobs recorded and its mechanism
  visibly configured (extraction on or off, the reset schedule, the handoff, the summary prompt).
"""

import asyncio
import itertools
import os

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape import run_study, tuning
from ape.agent import cm_prompts
from ape.agent.arms import load_world
from ape.agent.cm_arms import ALL_KNOBS, DEFAULT_STACK, ManagedContext, parse_stack
from ape.agent.context_policy import KNOB_ENV_PREFIX, policy_class, resolve_knobs
from ape.agent.session_checkpoint import ENV as CHECKPOINTS_ENV
from ape.budget import load_plan
from ape.build import build
from ape.config import ROOT
from ape.llm.mock_session import gold_session_agent
from ape.run_study import StudyRun
from ape.tasks.study_g import f8_session

GRID_PATH = ROOT / "config" / "tuning_grid_study_g.yaml"
GRID = yaml.safe_load(GRID_PATH.read_text())
TUNE_CELL = "g.tune.luna"
# The knobs each component of a managed arm reads (ape.agent.cm_arms: prune_keep in the prune compactor, trim_preserve
# in trim, sum_prompt in the summary mover, todo_extract at shift start and on memo cases, reset_every and
# reset_summary in the reset mover). S-CM* also reads `stack`.
COMPONENT_KNOBS = {"prune": ("prune_keep",), "trim": ("trim_preserve",), "sum": ("sum_prompt",), "todo": ("todo_extract",), "reset": ("reset_every", "reset_summary")}
COUNTS = {"CM-prune": 3, "CM-sum": 2, "CM-todo": 2, "CM-reset": 3, "S-CM*": 3}  # CM-sum and CM-todo: their whole design space
LOW_T = 6000  # on 12 cases every mechanism fires (the system prompt alone is ~2.5K tokens; tests/test_cm_arms.py)


def _candidates() -> list[tuple[str, dict]]:
    return [(name, c) for name, sdef in GRID["systems"].items() for c in sdef["candidates"]]


def _env_knob(var: str) -> str:
    return var.removeprefix(KNOB_ENV_PREFIX).lower()


def _reads(arm: str, knobs: dict) -> set[str]:
    """The knobs `arm` reads under `knobs` (resolved): its components' and, for S-CM*, its stack."""
    cls = policy_class(arm)
    if not issubclass(cls, ManagedContext):
        return set(cls.KNOBS)
    out = {k for c in cls.components(knobs) for k in COMPONENT_KNOBS[c]}
    return out | ({"stack"} if "stack" in cls.KNOBS else set())


def effective(arm: str, env: dict[str, str]) -> dict:
    """What `arm` runs with under `env`: the knobs it reads, as it resolves them."""
    knobs = resolve_knobs(policy_class(arm), env)
    return {k: knobs[k] for k in sorted(_reads(arm, knobs))}


def candidate_problems(arm: str, env: dict[str, str]) -> list[str]:
    """Why a candidate env is not a valid, whole configuration of `arm` (empty when it is)."""
    cls = policy_class(arm)
    problems = [f"{k} is not an APE_CM_* knob" for k in env if not k.startswith(KNOB_ENV_PREFIX)]
    problems += [f"{k}: {v!r} is not a string (the environment holds strings)" for k, v in env.items() if not isinstance(v, str)]
    if unknown := sorted(k for k in env if k.startswith(KNOB_ENV_PREFIX) and _env_knob(k) not in cls.KNOBS):
        return [*problems, f"{arm} reads none of {unknown} (its knobs: {sorted(cls.KNOBS)})"]
    try:
        knobs = resolve_knobs(cls, {k: str(v) for k, v in env.items()})
        cls.validate(knobs)
    except ValueError as e:
        return [*problems, f"{arm} rejects {env}: {e}"]
    reads, sets = _reads(arm, knobs), {_env_knob(k) for k in env}
    if missing := sorted(reads - sets):
        problems.append(f"{arm} also reads {missing}: write them out, or another system's selection can change them in the test")
    if unread := sorted(sets - reads):
        problems.append(f"{arm} does not read {unread} in this configuration")
    return problems


def _test_envs(selected: dict[str, dict]) -> dict[str, dict[str, str]]:
    """The env each tuned arm runs under in the test phase, by ape.run_study's rule for selections that share APE_CM_*
    knobs (BUILD_PLAN B1, `selection_env` and `env_group`): every selection's knobs together when they agree on each
    shared knob, else each tuned arm under its own selection. The runner's own functions once it has them."""
    if hasattr(run_study, "env_group"):
        together = run_study.selection_env(selected)
        return {name: run_study.env_group(name, s["arm"], selected, together)[1] for name, s in selected.items()} | {"*untuned*": dict(together or {})}
    together: dict[str, str] = {}
    for s in selected.values():
        if any(together.get(k, v) != v for k, v in s["env"].items()):
            return {name: dict(s["env"]) for name, s in selected.items()} | {"*untuned*": {}}
        together |= s["env"]
    return dict.fromkeys(selected, together) | {"*untuned*": together}


# --- The grid against the plan ---------------------------------------------------------------------------------------


def test_the_grid_tunes_the_plans_systems_on_its_dev_cell_within_budget_and_unsigned(tmp_path):
    plan = load_plan()
    tune = plan.cell(TUNE_CELL)
    assert "placeholder" not in GRID, "B11's grid, not B1's placeholder"
    assert list(GRID["systems"]) == list(tune.spec["arms"]) == list(COUNTS), "exactly g.tune.luna's systems, components before S-CM*"
    assert GRID["budget_per_system"] == max(tune.spec["arms"].values()) == 3 and GRID["tie_pp"] == 1.0
    assert {n: len(s["candidates"]) for n, s in GRID["systems"].items()} == COUNTS
    assert all(len(tuning.candidates(GRID, n)) <= tune.spec["arms"][n] for n in GRID["systems"]), "within the plan's priced count"
    ids = [c["id"] for _, c in _candidates()]
    assert len(ids) == len(set(ids)), "candidate ids are unique (each names its log dir)"
    # Each candidate runs as its system's plan arm, and is only {id, arm, env}: the runner reads nothing else (no
    # `threshold`: run_study passes the plan's T_abs to every session task).
    assert all(c["arm"] == name and set(c) == {"id", "arm", "env"} for name, c in _candidates())
    # The plan's tune cell: F8 sessions of N 40 at the default knobs.
    assert tune.kind == "session" and not tune.spec.get("knobs") and GRID["dev_cells"] == [f"F8-{tune.spec['N']}"]
    assert all(run_study._system_cells(GRID, n) == ["F8-40"] and run_study._system_kind(GRID, n) == "session" for n in GRID["systems"])
    # Owners and sign-off: every system named, none signed; a live tune refuses, an offline one runs.
    assert set(GRID["owners"]) == set(GRID["signed_off"]) == set(GRID["systems"])
    assert set(GRID["owners"].values()) == {"TODO"} and set(GRID["signed_off"].values()) == {False}
    problems = run_study.tuning_signoff_problems(GRID)
    assert problems and all(p.startswith(("owners.", "signed_off.")) for p in problems)
    signed = GRID | {"owners": dict.fromkeys(GRID["owners"], "Ada"), "signed_off": dict.fromkeys(GRID["signed_off"], True)}
    assert run_study.tuning_signoff_problems(signed) == []
    assert "is not signed off for a live tune" in run_study._refuse_tune(StudyRun("study_g", "tune", runs_root=tmp_path / "runs"))
    offline = StudyRun("study_g", "tune", offline=True, runs_root=tmp_path / "runs")
    assert run_study._refuse_tune(offline) is None and run_study.unbuilt_arms(offline, "tune") == []
    grid, skipped = run_study.study_grid(offline)
    assert skipped == [] and {n: [c["id"] for c in s["candidates"]] for n, s in grid["systems"].items()} == {n: [c["id"] for c in s["candidates"][:2]] for n, s in GRID["systems"].items()}


def test_every_session_arm_of_the_plan_is_tuned_here_or_untuned_with_a_reason():
    session_arms = {a for c in load_plan().cells if c.study == "study_g" and c.kind == "session" and c.enabled for a in c.spec["arms"]}
    untuned = GRID["inherited"]
    assert session_arms <= set(GRID["systems"]) | set(untuned) and not set(untuned) & set(GRID["systems"])
    assert all(v["source"] == "fixed" and v["why"] for v in untuned.values())
    assert {"CM0", "O-state", "CM-trim", "CM-native"} <= set(untuned), "the references, the naive control and the provider arm are not tuned"
    # CM-native is never a grid candidate (its support gate is the probe's record, not a tune).
    assert all(c["arm"] != "CM-native" for _, c in _candidates())


# --- Each candidate: valid, whole, distinct --------------------------------------------------------------------------


@pytest.mark.parametrize("system,cand", _candidates(), ids=[c["id"] for _, c in _candidates()])
def test_every_candidate_is_a_valid_whole_configuration_of_its_arm(system, cand):
    assert candidate_problems(cand["arm"], cand["env"]) == []
    if cand["arm"] == "S-CM*":
        stack = parse_stack(cand["env"]["APE_CM_STACK"])
        assert "prune" in stack and "trim" not in stack, "every S-CM* candidate keeps prune; trim (the naive control) is left out"


def test_the_checker_catches_typos_foreign_knobs_bad_values_and_partial_configurations():
    assert candidate_problems("CM-prune", {"APE_CM_PRUNE_KEPE": "6"}) == ["CM-prune reads none of ['APE_CM_PRUNE_KEPE'] (its knobs: ['prune_keep'])"]
    assert "CM-sum reads none of ['APE_CM_PRUNE_KEEP']" in candidate_problems("CM-sum", {"APE_CM_SUM_PROMPT": "plain", "APE_CM_PRUNE_KEEP": "3"})[0]
    assert "rejects" in candidate_problems("CM-sum", {"APE_CM_SUM_PROMPT": "fancy"})[0]
    assert "rejects" in candidate_problems("S-CM*", {"APE_CM_STACK": "prune+trim"})[0]
    assert "not a string" in candidate_problems("CM-prune", {"APE_CM_PRUNE_KEEP": 6})[0]
    assert candidate_problems("CM-reset", {"APE_CM_RESET_EVERY": "10"}) == ["CM-reset also reads ['reset_summary', 'todo_extract']: write them out, or another system's selection can change them in the test"]
    assert candidate_problems("S-CM*", {"APE_CM_STACK": "prune+sum", "APE_CM_PRUNE_KEEP": "3", "APE_CM_SUM_PROMPT": "plain", "APE_CM_TODO_EXTRACT": "true"}) == ["S-CM* does not read ['todo_extract'] in this configuration"]


def test_candidates_differ_the_default_comes_first_and_binary_knobs_are_covered():
    for name, sdef in GRID["systems"].items():
        configs = [effective(c["arm"], c["env"]) for c in sdef["candidates"]]
        assert all(a != b for a, b in itertools.combinations(configs, 2)), f"{name}: two candidates run the same configuration"
        assert configs[0] == effective(name, {}), f"{name}: the arm's default (D-037) comes first, so the offline rehearsal runs it"
    assert effective("S-CM*", {})["stack"] == DEFAULT_STACK
    assert {c["env"]["APE_CM_SUM_PROMPT"] for c in GRID["systems"]["CM-sum"]["candidates"]} == set(cm_prompts.SUMMARY_PROMPTS)
    assert {c["env"]["APE_CM_TODO_EXTRACT"] for c in GRID["systems"]["CM-todo"]["candidates"]} == {"true", "false"}
    keeps = sorted(int(c["env"]["APE_CM_PRUNE_KEEP"]) for c in GRID["systems"]["CM-prune"]["candidates"])
    assert keeps == [ALL_KNOBS["prune_keep"], 6, 12]


def test_main_studys_static_check_passes_except_for_the_deliberately_shared_knobs():
    """`tuning.study_grid_problems` (B3) holds for the main study's per-arm APE_MAS_* knobs. Here the arms share APE_CM_*
    knobs by design (D-037) and each candidate writes out its whole configuration, so its no-shared-variable rule
    fires, and only it: the study runner runs disagreeing selections apart (BUILD_PLAN B1)."""
    problems = tuning.study_grid_problems(GRID, None, lambda system, env: candidate_problems(system, env))
    assert problems and all(" is set by the candidates of " in p for p in problems), problems
    assert {p.split(" ")[0] for p in problems} == {"APE_CM_PRUNE_KEEP", "APE_CM_SUM_PROMPT", "APE_CM_TODO_EXTRACT", "APE_CM_RESET_EVERY", "APE_CM_RESET_SUMMARY"}


def test_every_tuned_arm_runs_in_the_test_as_on_dev_whichever_selections_win():
    systems = GRID["systems"]
    together = apart = 0
    for combo in itertools.product(*(sdef["candidates"] for sdef in systems.values())):
        selected = {name: {"arm": c["arm"], "env": dict(c["env"]), "candidate": c["id"]} for name, c in zip(systems, combo, strict=True)}
        envs = _test_envs(selected)
        for name, c in zip(systems, combo, strict=True):
            assert effective(c["arm"], envs[name]) == effective(c["arm"], c["env"]), (name, c["id"], envs[name])
        for arm in ("CM-trim", "CM-native"):  # untuned arms keep their defaults
            assert effective(arm, envs["*untuned*"]) == effective(arm, {})
        if all(envs[name] == envs["*untuned*"] for name in systems) and envs["*untuned*"]:
            together += 1
        else:
            apart += 1
    assert together and apart, "both of the runner's cases occur among the 108 combinations"


# --- Each candidate through the real session loop --------------------------------------------------------------------


@pytest.fixture(scope="module")
def world12(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("g-grid")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv(CHECKPOINTS_ENV, raising=False)
    asyncio.run(build("dev", "F8", ["12"], n_worlds=1, n_tasks=0, relational=True, embed=False))
    yield tmp
    mp.undo()


def _recording(agent, requests: list[str]):
    """The mock, keeping the text of every management request (a call without tools: summary, handoff, extraction)."""

    def outputs(messages, tools, tool_choice, config):
        if not tools and messages and cm_prompts.purpose_of(messages[-1].text or ""):
            requests.append(messages[-1].text)
        return agent(messages, tools, tool_choice, config)

    return outputs


@pytest.mark.parametrize("system,cand", _candidates(), ids=[c["id"] for _, c in _candidates()])
def test_every_candidate_runs_a_perfect_gold_session_with_its_knobs(world12, monkeypatch, system, cand):
    for k in [k for k in os.environ if k.startswith(KNOB_ENV_PREFIX)]:
        monkeypatch.delenv(k)
    for k, v in cand["env"].items():
        monkeypatch.setenv(k, v)
    requests: list[str] = []
    log = inspect_eval(
        f8_session(level="12", split="dev", arm=cand["arm"], threshold=LOW_T, checkpoints="5,10"),
        model=get_model("mockllm/model", custom_outputs=_recording(gold_session_agent(load_world("F8-12-dev-s1000")), requests)),
        log_dir=str(world12 / "logs" / cand["id"]),
        display="none",
    )[0]
    assert log.status == "success", log.error
    s = log.samples[0]
    assert s.error is None and s.scores["f8_session_score"].value["item_success"] == 1.0
    knobs = s.store["arm"]["knobs"]
    assert {k: knobs[k] for k in effective(cand["arm"], cand["env"])} == effective(cand["arm"], cand["env"]), "the session ran the candidate's knobs"
    assert log.eval.metadata["policy_env"] == cand["env"], "the task records the candidate's env"
    events = s.store["f8_cm_events"]
    names = {e["event"] for e in events}
    stack = parse_stack(knobs["stack"]) if "stack" in knobs else policy_class(cand["arm"]).STACK
    if "prune" in stack:
        assert "prune" in names
    if "todo" in stack:
        assert ("todo_extract" in names) == knobs["todo_extract"], "extraction calls exactly when it is on"
    if "reset" in stack:
        every = [e for e in events if e["event"] == "reset_drop" and str(e.get("trigger", "")).startswith("every")]
        assert bool(every) == (knobs["reset_every"] > 0) and all(e["trigger"] == f"every {knobs['reset_every']} items" for e in every)
        assert ("handoff" in names) == knobs["reset_summary"] and "reset_drop" in names
    summaries = [t for t in requests if cm_prompts.purpose_of(t) == "summary"]
    assert all(cm_prompts.SUMMARY_PROMPTS[knobs["sum_prompt"]] in t for t in summaries), "every summary asked with the candidate's prompt"
    if stack == ("sum",):
        assert "summary" in names and summaries
    assert ("todo_extract" in {cm_prompts.purpose_of(t) for t in requests}) == ("todo" in stack and knobs["todo_extract"])
