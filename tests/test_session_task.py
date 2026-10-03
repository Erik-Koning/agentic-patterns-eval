"""The F8 session task (BUILD_PLAN B7): a run_plan cell's knobs reach the generator and select the worlds, a study's
seed block and skipped worlds select like the gate's, and the study runner's arguments are task args only when
passed. Offline: mockllm, fake embeddings."""

import asyncio
import shutil

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from ape.agent.session import plan_threshold
from ape.build import build
from ape.llm.mock_session import mock_session_agent
from ape.tasks.study_g import f8_session, session_samples
from ape.worlds import gen_f8


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """F8-10 dev worlds: three at the default knobs from seed 1000, two more from 1100 (another seed block), and two
    with 2,250-token tool files (the D-028 knob)."""
    tmp = tmp_path_factory.mktemp("f8-task")
    mp = pytest.MonkeyPatch()
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        mp.setenv(k, str(v))
    mp.delenv("APE_SESSION_CHECKPOINTS", raising=False)
    asyncio.run(build("dev", "F8", ["10"], n_worlds=3, n_tasks=0, relational=True, embed=False))
    asyncio.run(build("dev", "F8", ["10"], n_worlds=2, n_tasks=0, relational=True, embed=False, seed_base=1100))
    asyncio.run(build("dev", "F8", ["10"], n_worlds=2, n_tasks=0, relational=True, embed=False, knobs={"output_tokens": 2250}))
    yield tmp
    mp.undo()


def _ids(samples) -> list[str]:
    return [s.id for s in samples]


def test_the_task_selects_the_variant_seed_block_and_skipped_worlds(worlds):
    assert _ids(session_samples("10", "dev")) == ["F8-10-dev-s1000", "F8-10-dev-s1001", "F8-10-dev-s1002", "F8-10-dev-s1100", "F8-10-dev-s1101"]
    assert _ids(session_samples("10", "dev", seed_base=1000)) == ["F8-10-dev-s1000", "F8-10-dev-s1001", "F8-10-dev-s1002"]
    assert _ids(session_samples("10", "dev", seed_base=1100)) == ["F8-10-dev-s1100", "F8-10-dev-s1101"]
    assert _ids(session_samples("10", "dev", seed_base=1000, skip_worlds=1, limit_worlds=1)) == ["F8-10-dev-s1001"]
    variant = gen_f8.variant_tag({"output_tokens": 2250})
    big = session_samples("10", "dev", variant=variant)
    assert _ids(big) == ["F8-10-o2250-dev-s1000", "F8-10-o2250-dev-s1001"] and all(s.metadata["knobs"]["output_tokens"] == 2250 for s in big)
    with pytest.raises(FileNotFoundError, match=r"seeds 1200\.\.1299"):
        session_samples("10", "dev", seed_base=1200)


def test_a_world_whose_knobs_are_not_the_variant_is_refused(worlds):
    """The world ID's tag selects; the knobs inside must agree (a hand-copied or renamed file never slips in)."""
    d = worlds / "worlds" / "dev"
    shutil.copy(d / "F8-10-o2250-dev-s1000.json", d / "F8-10-dev-s1900.json")
    try:
        with pytest.raises(ValueError, match="are not variant ''"):
            session_samples("10", "dev", seed_base=1900)
    finally:
        (d / "F8-10-dev-s1900.json").unlink()


def test_the_task_records_knobs_and_takes_plan_args_only_when_given(worlds, monkeypatch):
    monkeypatch.setenv("APE_CM_KEEP_ITEMS", "3")
    t = f8_session(level="10", split="dev", arm="CM0", limit_worlds=1, variant="o2250", plan_cell="g.topo.luna", group="g1", seed_base=1000, skip_worlds=1)
    md = t.metadata
    assert md["variant"] == "o2250" and md["knobs"]["output_tokens"] == 2250 and md["policy_env"] == {"APE_CM_KEEP_ITEMS": "3"}
    assert (md["plan_cell"], md["group"], md["seed_base"], md["skip_worlds"]) == ("g.topo.luna", "g1", 1000, 1)
    assert _ids(t.dataset) == ["F8-10-o2250-dev-s1001"] and md["threshold"] == plan_threshold()
    plain = f8_session(level="10", split="dev", arm="CM0")
    assert not {"plan_cell", "group", "seed_base", "skip_worlds"} & set(plain.metadata) and plain.metadata["variant"] == ""
    assert plain.metadata["knobs"] == gen_f8.DEFAULT_KNOBS
    # None is "not given" (a runner may pass every argument).
    nones = f8_session(level="10", split="dev", arm="CM0", plan_cell=None, group=None, seed_base=None, skip_worlds=None)
    assert _ids(nones.dataset) == _ids(plain.dataset) and not {"plan_cell", "group", "seed_base", "skip_worlds"} & set(nones.metadata)
    assert f8_session(level="10", split="dev", arm="CM0", threshold=1234).metadata["threshold"] == 1234


def test_a_knob_variant_runs_end_to_end(worlds):
    """The study runner's call shape: the cell's knobs reach the generator (world IDs carry the tag) and the session
    runs on them."""
    knobs = {"output_tokens": 2250}
    t = f8_session(level="10", split="dev", arm="O-state", limit_worlds=1, variant=gen_f8.variant_tag(knobs), plan_cell="g.topo.luna", group="default", seed_base=1000, skip_worlds=0, checkpoints="5")
    log = inspect_eval(t, model=get_model("mockllm/model", custom_outputs=mock_session_agent), log_dir=str(worlds / "logs"), display="none")[0]
    assert log.status == "success" and log.samples[0].id == "F8-10-o2250-dev-s1000"
    assert log.eval.task_args_passed["variant"] == "o2250" and log.eval.metadata["knobs"]["output_tokens"] == 2250
    # The solver's passed params (part of Inspect's task identity) are what they were before B7 unless a threshold
    # is given.
    (step,) = log.plan.steps
    assert set(step.params_passed) == {"arm", "window", "max_turns_per_item", "checkpoints"}
    t = f8_session(level="10", split="dev", arm="O-state", limit_worlds=1, variant="o2250", checkpoints="5", threshold=1234)
    log = inspect_eval(t, model=get_model("mockllm/model", custom_outputs=mock_session_agent), log_dir=str(worlds / "logs"), display="none")[0]
    assert log.plan.steps[0].params_passed["threshold"] == 1234 and log.samples[0].store["arm"]["threshold"] == 1234
