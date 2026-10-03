"""Study orchestrator (BUILD_PLAN B1, B6): the main study and Study G as named phases, each idempotent, resumable and
recorded, with the gate orchestrator's guarantees.

    uv run --locked python -m ape.run_study <phase|all> --study main|study_g --run-id <id> [--offline] [--gate-run-id <id>]

`ape.run_gate` is the shared machinery. Every phase here goes through the same manifests and fingerprints, budget
guard and spend registry, offline check, live-smoke settlement, freeze record and backups, read from `ape.run_gate`
when called (so a test that patches one there patches it here too). This module holds what differs per study: its
cells, seeds, test lock, KG arm, caps, tuning grid, pre-registration and analysis.

Phases (`all` runs a study's in order and stops at the first failure; a failed or budget-stopped test is still analysed):

    main      preflight  build-dev  micro-pilot  tune  pilot  freeze  build-test  test  analyze
    study_g   preflight  build-dev  micro-pilot  tune         freeze  build-test  test  analyze

    preflight    FX-2 preflight for every profile the study's cells use (prices; live: the key, the readiness probe lists
                 every model, the APG pin); the arms each phase names that are not built yet; live: a passing live smoke
                 of this code (`run_gate._settle_smoke`, plus the study's own required checks, `Study.smoke_checks`) and
                 storage warnings. A dirty tree is a warning.
    build-dev    the dev worlds the study tunes on (`world_specs`) and the artifacts its arms read (`arm_kinds`). Live, it
                 refuses a builder other than the gate run's when it would touch the gate's shared dev worlds.
    micro-pilot  the run_plan `micro_pilot` cells on the pilot split, after building every pilot world. Main: B0, the
                 median realized total tokens of S1 per task cell (`b0_from_logs`), and the token caps, CAP_MULTIPLE x
                 B0 (ORCHESTRATOR_BRIEF_v2 §4.5) -> the run's config/token_caps.json. The cost model is recalibrated
                 from the logs (the run's config/budget_calibration_measured.yaml), as the gate's pilot does.
    tune         the study's grid (config/tuning_grid_<study>.yaml, `ape.tuning`) on dev -> tune/selected.yaml and the
                 run's config/selected.yaml. Live, it refuses a grid marked `placeholder`, or one in which a system has no
                 owner or no sign-off (`tuning_signoff_problems`), as the gate does.
    pilot        main only: the run_plan `pilot` cells with the selections and caps; S7 last, sized per cell by the KG
                 arm's median realized context (the run's config/s7_targets.json, as the gate sizes S7 by APG*); B0 of
                 the cells the micro-pilot did not measure -> the run's config/token_caps_pilot.json; pilot/pilot.json.
                 Then the cap-hit gate (D-039, `_cap_gate`): each arm's cap-hit rate per task cell; while an arm is over
                 10% (PC5), the multiple doubles for every arm (8 -> 16 -> 32) and that arm's pilot cells re-run under
                 the new caps (its projection includes both re-runs for every arm) -> config/cap_gate.json, whose final
                 multiple the test applies. Offline, only token-cap hits count (the mocks' turn-cap runs are reported).
    freeze       the study's pre-registration (`Study.prereg`) has no `[PILOT` / `[USER` marker left (offline: a
                 rehearsal on a filled copy), the tree is committed, the analysis entry point exists and every phase the
                 freeze rests on is current. It hashes the pre-registration, the config inputs (the shared run_plan,
                 models and model_costs files as the study's own slice, `ape.freeze_scope`, so another study's later
                 edits never break this freeze; its tuning grid whole), the run's outputs
                 (`frozen_outputs`), uv.lock, the study's analysis code and the gate run the KG arm came from, and
                 records the commit, the design knobs, the KG resolution, the role (primary, or the extension of an
                 earlier run), the token caps with their multiple and every pilot cap-hit rate (main; it refuses while
                 an arm is over 10% at 32 x B0, D-039) and a fresh test-seed block -> freeze.json, then PROVENANCE.md
                 (offline: work/PROVENANCE.freeze.md).
    build-test   the test worlds of the run's frozen seed block. Only this phase unlocks the study's test split
                 (TEST_SPLIT_ENV = `<study>/<run id>`, `worlds.generate.split_lock_value`): the gate's unlock never
                 opens it, nor this one the gate's.
    test         every enabled cell of the study's test plan phases (`Study.cells["test"]`), one eval set per cell and
                 environment group (`run_groups`), the primary phases' cells first. The budget is re-checked before each
                 group; a failing group is recorded and the others still run (the phase then fails; a re-run resumes).
    analyze      the study's analysis entry point (`Study.analysis`): `analyze(run) -> dict` in `ape.analyze_main` or
                 `ape.analyze_g`, which writes report/report.md (and whatever else) under the run directory. Until that
                 module exists, an offline run writes a stub report noting "analysis not implemented"; a live run
                 refuses to complete the phase.

Seeds (`worlds.generate` STUDY_SEEDS). dev is the gate's (1000+; dev only tunes, and a gate dev cell's worlds are the
gate's own worlds, built with the gate's exact parameters and never overwritten, so their paid KG artifacts are
reused); pilot main 12000+, Study G 22000+; test blocks main [13000, 19000), Study G [23000, 29000), each live run
freezing the next unused block of SEED_BLOCK (`choose_test_seed_base`: runs/<study>/*/freeze.json and PROVENANCE.md's
`<!-- ape:test-seeds study=<study> run=... -->` markers); offline rehearsals 19000 and 29000. Every task names its
block (`seed_base`), so it reads only its run's worlds, though world files of every study share worlds/<split>/.

Cells -> tasks. `agent` cells run `tasks.main.main_study` (F1, F2, F3, F7; Study G's capability anchor too, on its own
worlds) with `plan_cell`, `group` and `seed_base`, and, in the main study, the cell's token cap as Inspect's
`token_limit` (it bounds the whole sample, every agent). `session` cells run `tasks.study_g.f8_session` (level N,
`variant` = `gen_f8.variant_tag(knobs)`, `limit_worlds` = sessions, the plan's window and compaction threshold),
with `plan_cell`, `group` and `seed_base`, and APE_SESSION_CHECKPOINTS points at the run's session_checkpoints/
directory for B7's mid-session resume. A cell's `profile`, `effort` (the agent's) and `models` choose its models (the
test manifest records all three per group). A tuned arm (a selection key) runs as its selection, under its own
selection's knobs alone, in an eval set of its own (`env_group`, D-042; M1s under M1's, whose knobs it reads), never
another selection's; S7 runs in its own group (`s7`) under APE_S7_PER_STEP, mirroring the KG arm's schedule (D-024).
The test manifest records each group's final log files relative to the run directory (`run_relative`,
`resolve_log`).

Arms not built yet. Offline, a cell's arms without a solver (a multi-agent arm not in `agent.solvers.MULTI_AGENT_ARMS`,
a session arm neither a context policy nor in `agent.session.SESSION_ARMS`) are skipped and every manifest lists them
(`skipped_arms`); grid systems likewise. Live, a phase whose cells name an unbuilt arm refuses to start. When a package
registers an arm, it runs here unchanged. Since B2, B8 and B9 every arm of both studies is built.

The KG arm (S5, and the KG workers of M1k and M2; `agent.arms.kg_arm_name`). `--gate-run-id` names a frozen, analysed
gate run: GO (any GO label) -> its APG*, NO_GO -> its LGR*, with that selection's knobs (the gate run's selected.yaml),
set as APE_KG_ARM and the knobs for every phase (`kg_resolution`). A live run refuses without a live gate run whose
verdict is final; offline defaults to APG-s with the fake author's graphs (an offline gate run's LightRAG arm is its
oracle twin, `run_gate.OFFLINE_ARMS`). run.json records the gate run at the run's first phase, so a later invocation
may omit the flag, and one naming another gate run is refused (`settle_gate_run`). The gate run's selection also hands
down the other arms the study's grid lists as `inherited: {source: gate}` (D-041: main's S3s, the gate's `S3s` entry):
their knobs are set alongside the KG arm's for every phase, and a live run refuses when the gate run has no such entry
(`inherited_selections`). The resolution is in every manifest's params and in the freeze, which hashes the gate run's
selected.yaml, decision.json and freeze.json. The KG arm's system (APG or LightRAG) is the one the builds make for the
KG cells (BUILD_PLAN B6: `main.build.kg`, with the build-quality check per build phase, F1 included).

Budget. Before a phase (and before each test group, the primary groups at the start), its conservative projection
(`ape.budget` over the plan cells it runs, at live sizes) must fit in both what is left of the program's budget
(budget.total_usd, or a lower --budget-usd) and of the study's allocation (budget.allocations), from the spend
registry, where every log dir this run writes is labelled `<study>/<run id>` (offline `<study>-offline/<run id>`, in
the run's own registry). Every eval set gets the runner's per-sample `cost_limit`.

Run directory: `runs/<study>/<id>/`, laid out as a gate run's (`ape.run_gate`): run.json, <phase>/manifest.json,
config/ (live outputs: selected.yaml, token_caps.json, token_caps_pilot.json, cap_gate.json, s7_targets.json,
budget_calibration_measured.yaml), freeze.json, report/, session_checkpoints/, and work/ for offline runs (worlds/,
indices/, cache/, config/, the rehearsal pre-registration and PROVENANCE.freeze.md). Gate run ids `main` and
`study_g` are refused, so the two layouts never meet.
"""

import argparse
import contextlib
import functools
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import run_gate as rg
from .agent.arms import KG_ARM_ENV
from .budget import BudgetError, Plan, PlanCell, require_affordable
from .config import ROOT, Config
from .freeze_scope import config_input
from .models import PreflightError, Profile, RoleSpec, load_profile, preflight, storage_warnings
from .spend import LABEL_ENV, REGISTRY_ENV
from .worlds.generate import SEED_BLOCK, STUDY_SEEDS, TEST_SEED_BASE_ENV, TEST_SPLIT_ENV, split_lock_value

PhaseError = rg.PhaseError
COMPLETE = rg.COMPLETE
SPLITS = {"build-dev": "dev", "micro-pilot": "pilot", "tune": "dev", "pilot": "pilot", "build-test": "test", "test": "test"}
TASKS_PER_WORLD = 12  # the gate's convention: 100 tasks -> 9 worlds, 20 -> 2 (main.build.kg: "9 test + 2 pilot worlds")
OFFLINE_SCALE = {"worlds_per_cell": 1, "tasks_per_world": 2, "sessions": 1, "epochs": 1, "tune_candidates_per_system": 2}
CAP_MULTIPLE = 8  # ORCHESTRATOR_BRIEF_v2 §4.5: C_max = 8 x B0 total tokens, one cap for every arm
CAP_DOUBLINGS = 2  # D-039: the pilot's cap-hit gate doubles the multiple at most twice (8 -> 16 -> 32)
CAP_HIT_MAX = 0.10  # D-039: PC5's cap-hit threshold (`analysis.gate_stats.MAX_CAP_HIT_RATE`), per arm and task cell
CAP_ARM = "S1"  # B0 is S1's realized total tokens
KG_ARMS = ("S5", "M1k", "M2")  # agent arms that read the KG (the gate-selected arm, APE_KG_ARM); session arms read none yet
ARM_KINDS = {"S3s": ("chunks",)}  # other artifacts an arm reads (S3s embeds the shared chunks); the KG arms read the KG
GO_LABELS = ("GO", "GO_PUSH_ONLY", "GO_PULL_ONLY", "GO_WITH_COST_FLAG")  # analyze_gate's GO family: the KG arm is APG*
NO_GO = "NO_GO"  # the KG arm is LGR*
OFFLINE_KG_ARM = "APG-s"  # offline without a gate run: APG over the fake author's graphs
ORACLE_KG_ARMS = ("APGo-q", "S5o", "LGRo-q", "LGRo-s")  # read the world spec: never a live run's KG arm
SMOKE_PROFILE = "gate"  # the profile readiness/smoke.py runs (`check_live_smoke` compares the smoke's profile with it)
SESSION_CHECKPOINTS_ENV = "APE_SESSION_CHECKPOINTS"  # BUILD_PLAN B7: where F8 sessions checkpoint, for mid-session resume
# Set by this orchestrator for every phase, so never "knobs from the shell" in a fingerprint: the KG arm (its knobs are
# part of the KG resolution, recorded in params) and the session checkpoint directory (operational).
STUDY_MANAGED_ENV = (KG_ARM_ENV, SESSION_CHECKPOINTS_ENV)
SEED_MARKER = re.compile(r"<!-- ape:test-seeds study=(\S+) run=(\S+) base=(\d+) count=(\d+) -->")
FROZEN_CONFIG = ("models.yaml", "model_costs.yaml", "run_plan.yaml")
REPORT_DIR = "report"
ANALYSIS_NOT_IMPLEMENTED = "analysis not implemented"


@dataclass(frozen=True)
class Study:
    """A study's shape. `cells`: runner phase -> the run_plan.yaml phases whose cells it runs; `primary`: the test plan
    phases whose cells run first (the confirmatory evidence, so a budget stop loses only the rest); `analysis`: the
    module whose `analyze(run)` the analyze phase calls, `analysis_code` the code the freeze hashes with it; `kg_build`:
    the plan's KG build cell (projection and B6's world-count check); `caps`: token caps from the micro-pilot;
    `smoke_checks`: live smoke check ids its live runs need on top of smoke.py's (BUILD_PLAN B12 adds them)."""

    name: str
    title: str
    phases: tuple[str, ...]
    cells: dict[str, tuple[str, ...]]
    primary: tuple[str, ...]
    prereg: str
    analysis: str
    analysis_code: tuple[str, ...]
    kg_build: str | None = None
    caps: bool = False
    smoke_checks: tuple[str, ...] = ()

    @property
    def rests_on(self) -> tuple[str, ...]:
        """The phases whose outputs the freeze fixes: every phase between preflight and the freeze."""
        return self.phases[1 : self.phases.index("freeze")]


STUDIES = {
    "main": Study(
        name="main",
        title="Main study",
        phases=("preflight", "build-dev", "micro-pilot", "tune", "pilot", "freeze", "build-test", "test", "analyze"),
        cells={"micro-pilot": ("micro_pilot",), "tune": ("tuning",), "pilot": ("pilot",), "test": ("study_a", "study_b", "study_c", "study_f")},
        primary=("study_a", "study_b"),  # the MVS mechanism and delivery studies (A, B); C and F follow
        prereg="PREREGISTRATION_MAIN.md",
        analysis="ape.analyze_main",
        analysis_code=("src/ape/analyze_main.py", "src/ape/analysis"),
        kg_build="main.build.kg",
        caps=True,
    ),
    "study_g": Study(
        name="study_g",
        title="Study G",
        phases=("preflight", "build-dev", "micro-pilot", "tune", "freeze", "build-test", "test", "analyze"),
        cells={"micro-pilot": ("micro_pilot",), "tune": ("tuning",), "test": ("capability_anchor", "context_management", "topology")},
        primary=("capability_anchor", "context_management", "topology"),  # G-H3 is confirmatory (D-033, D-042)
        prereg="PREREGISTRATION_G.md",
        analysis="ape.analyze_g",
        analysis_code=("src/ape/analyze_g.py", "src/ape/analysis"),
    ),
}
ALL_PHASES = tuple(dict.fromkeys(p for s in STUDIES.values() for p in s.phases))


# --- The run ---------------------------------------------------------------------------------------


@dataclass
class StudyRun:
    """One run of a study: `runs/<study>/<id>/`, laid out as a gate run's, and the attributes `ape.run_gate`'s shared
    functions read (`dir`, `phase_dir`, `out_config_dir`, `config()`, the smoke fields, ...). As in the gate, config
    inputs are read from `config_dir` (tests point it at a copy) and the outputs the phases write go to the run's own
    config/ (offline: work/config/). `gate_run_id` names the gate run the KG arm comes from (`gate_runs_root`, default
    the same runs root)."""

    study: str
    run_id: str
    offline: bool = False
    force: bool = False
    runs_root: Path = ROOT / "runs"
    config_dir: Path = ROOT / "config"
    env_path: Path = ROOT / ".env"
    probe_path: Path = ROOT / "cache" / "openai_probe.json"
    provenance_path: Path = ROOT / "PROVENANCE.md"
    prereg_path: Path | None = None  # default ROOT / the study's pre-registration
    budget_usd: float | None = None
    test_seed_base: int | None = None  # freeze only
    extension_of: str | None = None  # freeze only: this run extends an earlier frozen run of the study (a fresh block)
    gate_run_id: str | None = None
    gate_runs_root: Path | None = None
    smoke_dir: Path = rg.SMOKE_ROOT / "live"
    smoke_max_age_days: float = rg.SMOKE_MAX_AGE_DAYS
    skip_smoke_check: str | None = None
    _kg: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.study not in STUDIES:
            raise PhaseError(f"unknown study {self.study!r}; studies are {sorted(STUDIES)} (the gate runs with `python -m ape.run_gate`)")
        if not rg._RUN_ID.match(self.run_id):
            raise PhaseError(f"run id {self.run_id!r}: use letters, digits, '.', '_' and '-' only")
        self.runs_root, self.config_dir = Path(self.runs_root), Path(self.config_dir)
        self.gate_runs_root = Path(self.gate_runs_root) if self.gate_runs_root is not None else self.runs_root
        self.prereg_path = Path(self.prereg_path) if self.prereg_path is not None else ROOT / self.spec.prereg
        if not self.offline and self.config_dir.resolve() != (ROOT / "config").resolve():
            raise PhaseError("--config-dir is for offline runs and tests; live runs read config/ (build workers load config/models.yaml)")
        if (self.runs_root / self.study / "run.json").is_file():
            raise PhaseError(f"{rg._show(self.runs_root / self.study)} is a gate run's directory; {self.study} runs live in runs/{self.study}/<id>/")

    smoke = False  # studies have no smoke-scale mode (readiness/smoke.py smokes the wiring); run_gate's helpers read it

    @property
    def spec(self) -> Study:
        return STUDIES[self.study]

    @property
    def tiny(self) -> bool:
        return self.offline

    @property
    def isolated(self) -> bool:
        return self.offline

    @property
    def dir(self) -> Path:
        return self.runs_root / self.study / self.run_id

    def phase_dir(self, phase: str) -> Path:
        return self.dir / phase

    def manifest_path(self, phase: str) -> Path:
        return self.phase_dir(phase) / "manifest.json"

    @property
    def work_dir(self) -> Path:
        return self.dir / "work"

    @property
    def out_config_dir(self) -> Path:
        return self.work_dir / "config" if self.offline else self.dir / "config"

    @property
    def selected_path(self) -> Path:
        return self.out_config_dir / "selected.yaml"

    @property
    def s7_targets_path(self) -> Path:
        return self.out_config_dir / "s7_targets.json"

    @property
    def token_caps_path(self) -> Path:
        return self.out_config_dir / "token_caps.json"

    @property
    def pilot_caps_path(self) -> Path:
        return self.out_config_dir / "token_caps_pilot.json"

    @property
    def cap_gate_path(self) -> Path:
        return self.out_config_dir / "cap_gate.json"

    @property
    def measured_out_path(self) -> Path:
        return self.out_config_dir / "budget_calibration_measured.yaml"

    @property
    def freeze_path(self) -> Path:
        return self.dir / "freeze.json"

    @property
    def session_checkpoints_dir(self) -> Path:
        return (self.work_dir if self.offline else self.dir) / "session_checkpoints"

    def config(self, name: str) -> Path:
        return self.config_dir / name

    @property
    def grid_name(self) -> str:
        return f"tuning_grid_{self.study}.yaml"

    @property
    def costs_path(self) -> Path:
        return self.config("model_costs.yaml")

    @property
    def models_path(self) -> Path:
        return self.config("models.yaml")


def _show(path: Path) -> str:
    return rg._show(path)


def run_relative(run: StudyRun, path: str | Path) -> str:
    """A path inside the run directory as the test manifest records it (`log_files`): relative to the run directory, so
    a moved run, or one restored from a backup elsewhere, still finds its logs; a path outside it as `_show` records it."""
    p = Path(path).resolve()
    root = run.dir.resolve()
    return str(p.relative_to(root)) if p.is_relative_to(root) else _show(p)


def resolve_log(run_dir: Path, recorded: str) -> Path:
    """A recorded log path back as a path: absolute as is (manifests written before run-relative paths); relative, under
    the run directory when it is there, else under the repo (`_show`'s convention)."""
    p = Path(recorded)
    if p.is_absolute():
        return p
    return run_dir / p if (run_dir / p).exists() else rg._resolve(recorded)


def _check_mode(run: StudyRun) -> None:
    """A run is offline or live, of one study, and takes its KG arm from one gate run, for its whole life (run.json)."""
    path = run.dir / "run.json"
    if path.is_file():
        info = json.loads(path.read_text())
        if bool(info.get("offline")) != run.offline:
            raise PhaseError(f"run {run.run_id!r} is {'an offline' if info.get('offline') else 'a live'} run ({_show(path)}); use another --run-id")
        if run.gate_run_id and not info.get("gate_run_id"):
            rg.update_run_info(run, gate_run_id=run.gate_run_id, gate_runs_dir=_show(run.gate_runs_root))
        return
    gate = {"gate_run_id": run.gate_run_id, "gate_runs_dir": _show(run.gate_runs_root)} if run.gate_run_id else {}
    rg._write_json(path, {"run_id": run.run_id, "study": run.study, "offline": run.offline, "created": rg._now(), "config_dir": _show(run.config_dir), "git": rg.git_state()} | gate)


def settle_gate_run(run: StudyRun) -> None:
    """The gate run a study run took its KG arm from, recorded in run.json at its first phase: a later invocation
    without --gate-run-id uses it, and one naming another gate run is refused (a new KG arm is a new run)."""
    info = rg.read_run_info(run)
    recorded = info.get("gate_run_id")
    if not recorded:
        return
    if run.gate_run_id and run.gate_run_id != recorded:
        raise PhaseError(f"run {run.run_id!r} takes its KG arm from gate run {recorded!r} (run.json), not {run.gate_run_id!r}; a new KG arm is a new --run-id")
    if not run.gate_run_id:
        run.gate_run_id, run.gate_runs_root, run._kg = recorded, rg._resolve(info.get("gate_runs_dir") or _show(run.gate_runs_root)), None


# --- Plan, profiles, scale ---------------------------------------------------------------------------


def plan(run: StudyRun) -> Plan:
    return rg.plan(run)


def default_profile(run: StudyRun) -> str:
    raw = yaml.safe_load(run.config("run_plan.yaml").read_text()) or {}
    return str(((raw.get("studies") or {}).get(run.study) or {}).get("profile") or "gate")


def study_profile(run: StudyRun) -> Profile:
    """The study's default profile: its builder builds, its embedding model embeds."""
    return load_profile(default_profile(run), run.models_path)


def cell_profile(run: StudyRun, spec: dict) -> Profile:
    """A cell's models: its profile, then its `models` overrides (role -> model), then its `effort` (the agent's)."""
    p = load_profile(spec["profile"], run.models_path)
    roles = dict(p.roles)
    for role, model in (spec.get("models") or {}).items():
        roles[role] = replace(roles[role], model=str(model)) if role in roles else RoleSpec(str(model))
    if spec.get("effort"):
        roles["agent"] = replace(roles["agent"], reasoning_effort=spec["effort"])
    return replace(p, roles=roles) if roles != p.roles else p


def scale(run: StudyRun) -> dict:
    if run.offline:
        return {"offline": True, **OFFLINE_SCALE}
    return {"offline": False, "tasks_per_world": TASKS_PER_WORLD} | ({"cap_multiple": CAP_MULTIPLE} if run.spec.caps else {})


def phase_cells(run: StudyRun, phase: str) -> list[PlanCell]:
    """The enabled agent and session cells a runner phase runs, in plan order (test: the primary phases' first)."""
    plan_phases = run.spec.cells.get(phase, ())
    cells = [c for c in plan(run).cells if c.study == run.study and c.phase in plan_phases and c.kind in ("agent", "session") and c.enabled]
    if phase == "test":
        primary = run.spec.primary
        cells.sort(key=lambda c: (0 if c.phase in primary else 1, plan_phases.index(c.phase)))
    return cells


def _world_count(spec: dict) -> int:
    """Worlds an agent cell reads per task cell at live size: `worlds` (+ `world_offset`), else whole worlds for n_tasks."""
    if "worlds" in spec:
        return int(spec["worlds"]) + int(spec.get("world_offset", 0))
    return math.ceil(int(spec["n_tasks"]) / TASKS_PER_WORLD)


def _f8_cell(spec: dict) -> str:
    from .worlds import gen_f8

    tag = gen_f8.variant_tag(spec.get("knobs"))
    return f"F8-{int(spec['N'])}{'-' + tag if tag else ''}"


def window(run: StudyRun) -> int:
    from .worlds import gen_f8

    return int(plan(run).study_g.get("window", gen_f8.WINDOW))


# --- Arms --------------------------------------------------------------------------------------------


@functools.cache
def single_agent_arms() -> frozenset[str]:
    """The single-agent delivery arms `agent.arms` builds: the baselines, S5 (the KG arm), the APG and LightRAG arms."""
    from .agent.arms import BASELINE_ARMS
    from .apg.arm import ARMS as APG_ARMS
    from .lgr.adapter import ARMS as LGR_ARMS

    return frozenset({*BASELINE_ARMS, "S5", *APG_ARMS, *LGR_ARMS})


def arm_built(arm: str, kind: str = "agent") -> bool:
    """Whether `arm` has a solver now: a session arm registered as a context policy (`agent.context_policy.POLICIES`,
    which `agent.session` fills with CM0, O-state and B8's arms) or listed in `agent.session.SESSION_ARMS`; a
    multi-agent or ensemble arm registered in `agent.solvers.MULTI_AGENT_ARMS`; a single-agent delivery arm
    (`single_agent_arms`). Anything else, a typo included, is not built. Read at call time, so the arms other packages
    register flow through."""
    if kind == "session":
        from .agent.context_policy import POLICIES
        from .agent.session import SESSION_ARMS

        return arm in POLICIES or arm in SESSION_ARMS
    from .agent.solvers import MULTI_AGENT_ARMS

    return arm in MULTI_AGENT_ARMS or arm in single_agent_arms()


def arm_kinds(run: StudyRun, arms: Sequence[str], kind: str) -> tuple[str, ...]:
    """The artifacts (`ape.artifacts` kinds) these arms read on a world: the KG system for the KG arms, chunk
    embeddings for S3s, none for sessions and the other arms (S1, S6 and S7 read the rendered corpus)."""
    from .artifacts import KINDS

    if kind == "session":
        return ()
    need: set[str] = set()
    for a in arms:
        if a in KG_ARMS:
            need.add(kg_resolution(run)["system"])
        need.update(ARM_KINDS.get(a, ()))
    return tuple(k for k in KINDS if k in need)


def _resolved(arm: str, selected: dict) -> str:
    return str(selected[arm]["arm"]) if arm in selected else arm


def unbuilt_arms(run: StudyRun, phase: str) -> list[str]:
    """The arms a runner phase would run that have no solver yet, as `cell: arm`; for tune, the grid systems'."""
    if phase == "tune":
        grid = _load_grid(run)
        return [f"{name} (grid): {c['arm']}" for name, sdef in grid["systems"].items() for c in sdef["candidates"] if not arm_built(c["arm"], _system_kind(grid, name))]
    selected = read_selected(run) if run.selected_path.is_file() and phase not in ("micro-pilot",) else {}
    out = []
    for cell in phase_cells(run, phase):
        for a in rg._arm_names(cell.spec["arms"]):
            if not arm_built(_resolved(a, selected), cell.kind):
                out.append(f"{cell.id}: {a}" + (f" (as {_resolved(a, selected)})" if a in selected else ""))
    return out


def _refuse_unbuilt(phase: str) -> Callable[[StudyRun], str | None]:
    def refuse(run: StudyRun) -> str | None:
        if run.offline or not (missing := unbuilt_arms(run, phase)):
            return None
        return (
            f"{phase}: its cells name arms that are not built yet: {missing}. A live run never skips an arm; build them "
            "(BUILD_PLAN B2, B8, B9), or disable the cells in config/run_plan.yaml (a recorded deviation)"
        )

    return refuse


# --- The KG arm (S5, M1k, M2) ------------------------------------------------------------------------


def study_needs_kg(run: StudyRun) -> bool:
    """Whether any enabled cell of the study (or its tuning grid) names a KG arm."""
    p = plan(run)
    cells = [c for c in p.cells if c.study == run.study and c.enabled and c.kind == "agent"]
    if any(a in KG_ARMS for c in cells for a in rg._arm_names(c.spec["arms"])):
        return True
    grid = run.config(run.grid_name)
    return grid.is_file() and any(c["arm"] in KG_ARMS for s in (_load_grid(run)["systems"] or {}).values() for c in s["candidates"])


def kg_resolution(run: StudyRun) -> dict:
    """The KG arm and its knobs, from the gate's verdict (module docstring), cached on the run. Raises PhaseError when a
    live run has no gate run with a final verdict (GO or NO_GO)."""
    if run._kg is not None:
        return run._kg
    if not study_needs_kg(run):
        run._kg = {"needed": False, "system": None}
        return run._kg
    notes: list[str] = []
    if run.gate_run_id is None:
        if not run.offline:
            raise PhaseError(
                f"a live {run.study} run needs the gate's verdict for its KG arm (S5, M1k, M2): pass --gate-run-id <the frozen, analysed gate run>"
            )
        notes += [f"{a}: its defaults offline (no gate run to inherit its selection from)" for a in gate_inherited_arms(run)]
        run._kg = {"needed": True, "source": "offline default (no --gate-run-id)", "gate_run": None, "verdict": None, "key": None, "declared": OFFLINE_KG_ARM, "arm": OFFLINE_KG_ARM, "env": {}, "system": "apg", "files": {}, "inherited": {}, "notes": notes}
        return run._kg
    gdir = run.gate_runs_root / run.gate_run_id
    info_path, freeze_path, decision_path = gdir / "run.json", gdir / "freeze.json", gdir / REPORT_DIR / "decision.json"  # analyze_gate's REPORT_DIR
    problems = []
    info = json.loads(info_path.read_text()) if info_path.is_file() else None
    if info is None:
        raise PhaseError(f"--gate-run-id {run.gate_run_id!r}: {_show(info_path)} missing; name a gate run in {_show(run.gate_runs_root)} (--gate-runs-dir)")
    gate_offline = bool(info.get("offline"))
    if gate_offline and not run.offline:
        problems.append(f"gate run {run.gate_run_id!r} is an offline rehearsal; a live {run.study} run takes its KG arm from a live gate run")
    if not freeze_path.is_file():
        problems.append(f"gate run {run.gate_run_id!r} is not frozen ({_show(freeze_path)} missing)")
    label = json.loads(decision_path.read_text())["verdict"]["label"] if decision_path.is_file() else None
    if label is None:
        problems.append(f"gate run {run.gate_run_id!r} has no decision report ({_show(decision_path)}); analyse it first")
    elif label not in (*GO_LABELS, NO_GO):
        problems.append(f"gate run {run.gate_run_id!r}'s verdict is {label}, not final (GO or NO_GO): an extension or fix cycle decides first; name that run")
    gate = rg.GateRun(run.gate_run_id, offline=gate_offline, runs_root=run.gate_runs_root)
    selected_path = gate.selected_path
    if not selected_path.is_file():
        problems.append(f"gate run {run.gate_run_id!r} has no selection ({_show(selected_path)})")
    if problems:
        if not run.offline:
            raise PhaseError("the KG arm cannot be resolved: " + "; ".join(problems))
        notes += [*problems, f"offline: {OFFLINE_KG_ARM} instead", *(f"{a}: its defaults offline" for a in gate_inherited_arms(run))]
        run._kg = {"needed": True, "source": f"offline default (gate run {run.gate_run_id!r} unusable)", "gate_run": run.gate_run_id, "verdict": label, "key": None, "declared": OFFLINE_KG_ARM, "arm": OFFLINE_KG_ARM, "env": {}, "system": "apg", "files": {}, "inherited": {}, "notes": notes}
        return run._kg
    key = "APG*" if label in GO_LABELS else "LGR*"
    gate_selected = yaml.safe_load(selected_path.read_text()) or {}
    if key not in gate_selected:
        raise PhaseError(f"gate run {run.gate_run_id!r}: {_show(selected_path)} has no {key} selection; run its tune phase")
    entry = gate_selected[key]
    inherited = inherited_selections(run, gate_selected, selected_path, notes)
    arm = str(entry["arm"])
    if run.offline:
        arm = rg.OFFLINE_ARMS.get(arm, arm)
    elif arm in ORACLE_KG_ARMS:
        raise PhaseError(f"gate run {run.gate_run_id!r} selected the oracle arm {arm} for {key}: a live run's KG arm is never an oracle")
    if label == "GO_PULL_ONLY":
        notes.append("GO in pull mode only: the KG arm runs as each cell's delivery says (push by default)")
    run._kg = {
        "needed": True,
        "source": f"gate run {run.gate_run_id}",
        "gate_run": run.gate_run_id,
        "gate_offline": gate_offline,
        "verdict": label,
        "key": key,
        "declared": str(entry.get("declared_arm", entry["arm"])),
        "arm": arm,
        "env": {str(k): str(v) for k, v in (entry.get("env") or {}).items()},
        "system": "apg" if arm.startswith(("APG", "S5o")) else "lightrag",
        "candidate": entry.get("candidate"),
        "files": {"gate/selected.yaml": _show(selected_path), "gate/decision.json": _show(decision_path), "gate/freeze.json": _show(freeze_path)},
        "inherited": inherited,
        "notes": notes,
    }
    if clash := sorted(set(run._kg["env"]) & {k for x in inherited.values() for k in x["env"]}):
        raise PhaseError(f"gate run {run.gate_run_id!r}: the KG arm's and the inherited selections' knobs overlap ({clash}); each sets only its own")
    return run._kg


def gate_inherited_arms(run: StudyRun) -> dict[str, str]:
    """The arms the study's grid inherits from the gate's selection other than the KG arm (D-041: main's S3s), that the
    study's cells run: {arm: the gate's selected.yaml key} (the grid's `inherited.<arm>: {source: gate, gate_key}`)."""
    path = run.config(run.grid_name)
    if not path.is_file():
        return {}
    named = {a for c in plan(run).cells if c.study == run.study and c.enabled for a in rg._arm_names(c.spec.get("arms") or [])}
    out = {}
    for arm, spec in ((_load_grid(run).get("inherited") or {})).items():
        if isinstance(spec, dict) and spec.get("source") == "gate" and arm not in KG_ARMS and arm in named:
            out[str(arm)] = str(spec.get("gate_key") or arm)
    return out


def inherited_selections(run: StudyRun, gate_selected: dict, selected_path: Path, notes: list[str]) -> dict[str, dict]:
    """{arm: {key, arm, env, candidate}}: each `gate_inherited_arms` arm's selection in the gate run's selected.yaml,
    whose knobs every phase sets alongside the KG arm's (`kg_env`), recorded in the KG resolution (so in every
    fingerprint and the freeze). A live run refuses when the gate run has no such entry, or one that runs another arm."""
    out: dict[str, dict] = {}
    for arm, key in gate_inherited_arms(run).items():
        entry = gate_selected.get(key)
        problem = None
        if not isinstance(entry, dict):
            problem = f"gate run {run.gate_run_id!r}: {_show(selected_path)} has no {key} selection, which {run.study}'s {arm} inherits (D-041)"
        elif str(entry.get("arm", arm)) != arm:
            problem = f"gate run {run.gate_run_id!r}: the {key} selection runs {entry.get('arm')}, but {run.study}'s {arm} inherits only its knobs"
        if problem is not None:
            if not run.offline:
                raise PhaseError(problem)
            notes.append(f"{problem}; offline: {arm} at its defaults")
            continue
        out[arm] = {"key": key, "arm": arm, "env": {str(k): str(v) for k, v in (entry.get("env") or {}).items()}, "candidate": entry.get("candidate")}
    return out


def kg_env(run: StudyRun) -> dict[str, str]:
    """What the gate run decides for every phase's environment: the KG arm, its knobs and the inherited selections'."""
    kg = kg_resolution(run)
    if not kg.get("needed"):
        return {}
    return {KG_ARM_ENV: kg["arm"], **kg["env"], **{k: v for x in (kg.get("inherited") or {}).values() for k, v in x["env"].items()}}


def kg_params_of(kg: dict | None) -> dict:
    """A KG resolution as it enters fingerprints and the freeze's check: what it resolved (the inherited selections
    too), and its source files by hash."""
    if not kg or not kg.get("needed"):
        return {"needed": False}
    return {k: kg.get(k) for k in ("verdict", "key", "declared", "arm", "env", "system", "gate_run", "inherited")} | {"files": {k: rg._sha256(rg._resolve(p)) for k, p in (kg.get("files") or {}).items()}}


def kg_params(run: StudyRun) -> dict:
    return kg_params_of(kg_resolution(run))


def s7_env(run: StudyRun) -> dict[str, str]:
    """S7 follows the KG arm's delivery schedule (D-024): per step when the KG arm is per-step."""
    arm = kg_resolution(run).get("arm") or OFFLINE_KG_ARM
    if arm.startswith("LGR"):
        from .lgr.adapter import ARMS
    else:
        from .apg.arm import ARMS
    if arm not in ARMS:
        raise PhaseError(f"the KG arm {arm!r} has no delivery schedule (apg.arm / lgr.adapter ARMS)")
    return {"APE_S7_PER_STEP": "1" if ARMS[arm][1] else "0"}


# --- Seeds -------------------------------------------------------------------------------------------


def seeds(run: StudyRun) -> dict:
    return STUDY_SEEDS[run.study]


def split_seed_base(run: StudyRun, split: str) -> int:
    return run_test_seed_base(run) if split == "test" else int(seeds(run)[split])


def test_seed_count(run: StudyRun) -> int:
    """Seeds a run's test worlds use: the most worlds any test cell reads (world i has seed base+i)."""
    count = max([int(n["count"]) for n in _world_needs(run, "test", offline=False).values()] or [1])
    if count > SEED_BLOCK:
        raise PhaseError(f"{run.study}'s test cells read {count} worlds per cell, more than a seed block ({SEED_BLOCK})")
    return count


def used_test_seed_blocks(run: StudyRun) -> list[dict]:
    """Blocks other live runs of this study froze: runs/<study>/*/freeze.json and PROVENANCE.md's markers."""
    used: dict[tuple[str, int], dict] = {}
    for path in sorted((run.runs_root / run.study).glob("*/freeze.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rid = rec.get("run_id") or path.parent.name
        if rid == run.run_id or rec.get("offline") or not rec.get("test_seeds"):
            continue
        used[(rid, int(rec["test_seeds"]["base"]))] = {"run": rid, "base": int(rec["test_seeds"]["base"]), "count": int(rec["test_seeds"]["count"]), "source": _show(path)}
    text = run.provenance_path.read_text() if run.provenance_path.is_file() else ""
    for m in SEED_MARKER.finditer(text):
        study, rid, base, count = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if study == run.study and rid != run.run_id:
            used.setdefault((rid, base), {"run": rid, "base": base, "count": count, "source": _show(run.provenance_path)})
    return sorted(used.values(), key=lambda u: u["base"])


def choose_test_seed_base(run: StudyRun) -> tuple[int, list[str]]:
    """(base, problems): offline the study's rehearsal base; live `--test-seed-base` if given (inside the study's test
    range, overlapping no other frozen run's block), else the first block no other frozen run of the study used."""
    first, end = seeds(run)["test"]
    if run.offline:
        return (run.test_seed_base if run.test_seed_base is not None else int(seeds(run)["offline_test"])), []
    count = test_seed_count(run)
    used = used_test_seed_blocks(run)
    if run.test_seed_base is not None:
        base, problems = int(run.test_seed_base), []
        if not first <= base <= end - count:
            problems.append(f"--test-seed-base {base}: {run.study}'s test seeds live in [{first}, {end})")
        if clash := rg._overlaps(base, count, used):
            problems.append(f"--test-seed-base {base}: overlaps " + "; ".join(f"run {u['run']!r}'s {u['base']}-{u['base'] + u['count'] - 1} ({u['source']})" for u in clash))
        return base, problems
    base = first
    while rg._overlaps(base, count, used):
        base += SEED_BLOCK
    if base + count > end:
        return base, [f"{run.study}'s test range [{first}, {end}) has no free block left"]
    return base, []


def run_test_seed_base(run: StudyRun) -> int:
    freeze = read_freeze(run)
    if freeze is not None:
        return int(freeze["test_seeds"]["base"])
    return choose_test_seed_base(run)[0]


# --- Worlds ------------------------------------------------------------------------------------------


def _gate_dev_cells(run: StudyRun) -> dict[str, int]:
    """The gate's dev world cells (descriptive) and their tasks per world: a study's dev world of such a cell is the
    gate's own world, so it is built with exactly these parameters (shared)."""
    p = plan(run)
    tasks = int(p.cell("gate.tune").spec["tasks_per_world"])
    return dict.fromkeys(p.cell(rg.DEV_BUILD_CELLS["gate"]).spec["worlds"], tasks)


def _grid_needs(run: StudyRun) -> list[tuple[str, list[str], str]]:
    """(dev cell, its candidate arms, kind) of every grid system."""
    path = run.config(run.grid_name)
    if not path.is_file():
        return []
    grid = _load_grid(run)
    return [(c, [x["arm"] for x in s["candidates"]], _system_kind(grid, name)) for name, s in grid["systems"].items() for c in _system_cells(grid, name)]


def _world_needs(run: StudyRun, split: str, offline: bool) -> dict[str, dict]:
    """Per task cell (an F8 session cell: per N and knob variant) of a split: the most worlds any of its cells reads and
    the artifacts every arm there reads (declared arms, built or not, so the worlds stay the same when one lands).
    dev: the tune cells and the grid's dev cells; pilot: the micro-pilot's and the pilot's cells; test: the test's."""
    phases = {"dev": ("tune",), "pilot": ("micro-pilot", "pilot"), "test": ("test",)}[split]
    needs: dict[str, dict] = {}
    for phase in phases:
        for cell in phase_cells(run, phase):
            s, arms = cell.spec, rg._arm_names(cell.spec["arms"])
            if cell.kind == "session":
                n = needs.setdefault(_f8_cell(s), {"family": "F8", "level": str(int(s["N"])), "count": 0, "knobs": dict(s.get("knobs") or {}) or None, "kinds": set(), "session": True})
                n["count"] = max(n["count"], OFFLINE_SCALE["sessions"] if offline else int(s["sessions"]))
                continue
            for task_cell in s["cells"]:
                family, level = task_cell.split("-", 1)
                n = needs.setdefault(task_cell, {"family": family, "level": level, "count": 0, "knobs": None, "kinds": set(), "session": False})
                n["count"] = max(n["count"], OFFLINE_SCALE["worlds_per_cell"] if offline else _world_count(s))
                n["kinds"].update(arm_kinds(run, arms, "agent"))
    if split == "dev":
        for task_cell, arms, kind in _grid_needs(run):
            if task_cell in needs:
                needs[task_cell]["kinds"].update(arm_kinds(run, arms, kind))
            else:
                raise PhaseError(f"{run.grid_name}: dev cell {task_cell} is in no tuning cell of config/run_plan.yaml, so no dev worlds are planned for it")
    return needs


def world_specs(run: StudyRun, split: str, offline: bool | None = None) -> list[dict]:
    """Every world group a split needs (`run_gate.build_world_set` specs, `_world_needs`) at the split's seed base
    (test: the run's frozen block). A dev world of a gate dev cell is the gate's own world: built with the gate's
    tasks per world and marked `shared`, so another version on disk is refused. `offline` overrides the run's sizes
    (the live projection of an offline run)."""
    offline = run.offline if offline is None else offline
    needs = _world_needs(run, split, offline)
    shared = {} if offline or split != "dev" else _gate_dev_cells(run)
    base = split_seed_base(run, split)
    tasks = OFFLINE_SCALE["tasks_per_world"] if offline else TASKS_PER_WORLD
    specs = []
    for key, n in needs.items():
        spec = {
            "group": "sessions" if n["session"] else "worlds",
            "family": n["family"],
            "level": n["level"],
            "count": int(n["count"]),
            "n_tasks": 0 if n["session"] else tasks,
            "exception_style": "descriptive",
            "relational": True,
            "seed_base": base,
            "knobs": n["knobs"],
            "kinds": sorted(n["kinds"], key=("chunks", "apg", "lightrag").index),
        }
        if key in shared:
            spec |= {"n_tasks": shared[key], "shared": True}
        specs.append(spec)
    return specs


def build_world_set(run: StudyRun, record: dict, phase: str, split: str) -> list[dict]:
    """The split's worlds and artifacts through `run_gate.build_world_set` (the gate's path), then the build-quality
    check of its KG worlds (`record_kg_quality`) and B6's check of the plan's KG build cell."""
    specs = world_specs(run, split)
    Config().indices_dir.mkdir(parents=True, exist_ok=True)  # a recorded output even when no world needs an index (sessions)
    worlds = rg.build_world_set(run, record, phase, split, specs, study=run.study, profile=study_profile(run))
    record_kg_quality(run, record, phase, worlds)
    if run.spec.kg_build:
        if problems := kg_build_problems(run):
            record["warnings"] += [f"{run.spec.kg_build}: {p}" for p in problems]
    return worlds


def kg_worlds(run: StudyRun, split: str) -> dict[str, int]:
    """KG task cell -> worlds a split's build makes with the KG system, at live sizes (a shared gate dev world is the
    gate's, already built)."""
    system = kg_resolution(run).get("system")
    shared = _gate_dev_cells(run) if split == "dev" else {}
    return {c: int(n["count"]) for c, n in _world_needs(run, split, offline=False).items() if system and system in n["kinds"] and c not in shared}


def kg_build_problems(run: StudyRun) -> list[str]:
    """B6: the plan's KG build cell (`Study.kg_build`, priced by `ape.budget`) covers every KG world the pilot and test
    splits build: each KG cell listed, with at least the pilot's plus the test's world count."""
    planned = plan(run).cell(run.spec.kg_build).spec["worlds"]
    need: dict[str, int] = {}
    for split in ("pilot", "test"):
        for c, n in kg_worlds(run, split).items():
            need[c] = need.get(c, 0) + n
    return [f"{c} builds {n} KG worlds (pilot + test) but the cell plans {planned.get(c, 0)}" for c, n in need.items() if n > int(planned.get(c, 0))]


def kg_build_cell(run: StudyRun, split: str) -> list[PlanCell]:
    """The plan's KG build cell as a split's build makes it: its KG worlds there, in the resolved KG system."""
    if not run.spec.kg_build or not (worlds := kg_worlds(run, split)):
        return []
    base = plan(run).cell(run.spec.kg_build)
    return [PlanCell(base.id, base.study, base.phase, {**base.spec, "worlds": worlds, "systems": [kg_resolution(run)["system"]]})]


def record_kg_quality(run: StudyRun, record: dict, phase: str, worlds: list[dict]) -> dict | None:
    """The build-quality check (`ape.build_quality.assess`, D-017's measure) on the KG worlds a build phase just made,
    per KG cell and in the one KG system built (F1 included): `<phase>/build_quality.json` and a summary; a cell under
    the threshold is a warning, never a failure."""
    from .build_quality import assess
    from .worlds.spec import World

    system = kg_resolution(run).get("system")
    kg = [w for w in worlds if system and system in (w.get("kinds") or []) and not w.get("errors")]
    if not kg:
        return None
    cfg = Config()
    builder = rg._builder_label(study_profile(run).role("build"))
    report = assess(
        (World.load(cfg.world_path(w["world_id"])) for w in kg),
        cfg,
        "oracle" if run.offline else "extract",
        offline=run.offline,
        builder=f"scripted perfect_author (offline stand-in for {builder})" if run.offline else builder,
        fallback_builder="the build_fallback role",
        expected_cells=list(dict.fromkeys(f"{w['family']}-{w['level']}" for w in kg)),
        systems=(system,),
    )
    path = run.phase_dir(phase) / "build_quality.json"
    rg._write_json(path, report)
    record["outputs"] |= {"build_quality": _show(path)}
    record["build_quality"] = {"verdict": report["verdict"], "offline": report["offline"], "system": system, "cells": {c: {system: (s[system] or {}).get("coverage"), "pass": s["pass"]} for c, s in report["cells"].items()}}
    if report["verdict"] != "builder_passes":
        record["warnings"].append(f"build quality ({system}): {report['recommendation']}")
    return report


# --- Selections, caps --------------------------------------------------------------------------------


def read_selected(run: StudyRun) -> dict:
    """The run's selected.yaml (tune): {system: {arm, env, candidate, ...}}; plan arms named by a key run as it."""
    path = run.selected_path
    if not path.is_file():
        raise PhaseError(f"{_show(path)} missing: run the tune phase")
    return yaml.safe_load(path.read_text()) or {}


def selection_for(declared: str, selected: dict) -> str | None:
    """The selection whose knobs a declared plan arm runs under: its own (a tuned arm, a key of selected.yaml), else
    the selection of the arm whose knobs it reads (`agent.multi.knobs.KNOB_ARM`: M1s reads M1's, D-041), else none."""
    if declared in selected:
        return declared
    from .agent.multi.knobs import KNOB_ARM

    part = KNOB_ARM.get(declared)
    return next((name for name in selected if part is not None and KNOB_ARM.get(name) == part), None)


def env_group(declared: str, run_arm: str, selected: dict) -> tuple[str, dict[str, str]]:
    """(group name, env) an arm runs in (D-042: each tuned arm with exactly its own selection's knobs). S7 apart (`s7`;
    its schedule knob is added by the caller); an arm with a selection that sets knobs, in a group of that selection's
    alone (`sel-<selection>`, M1s with M1); every other arm (untuned, or a selection without knobs) in `selected`, with
    no selection's knobs. The study-wide knobs (the KG arm's and the inherited S3s's, `kg_env`) are every phase's."""
    if run_arm == "S7":
        return "s7", {}
    system = selection_for(declared, selected)
    own = {k: str(v) for k, v in ((selected.get(system) or {}).get("env") or {}).items()} if system else {}
    return (f"sel-{re.sub(r'[^A-Za-z0-9._-]', '_', system)}", own) if own else ("selected", {})


def b0_from_logs(log_files: Sequence[str], arm: str = CAP_ARM) -> dict[str, dict]:
    """task cell -> {b0, samples}: the median realized total tokens per sample (every model of the sample, as Inspect's
    `token_limit` meters them) of `arm`'s samples in the logs. Errored samples are left out."""
    import statistics

    from inspect_ai.log import read_eval_log

    per: dict[str, list[int]] = {}
    for f in log_files:
        log = read_eval_log(str(f))
        args = log.eval.task_args or {}
        if (args.get("arm") or (log.eval.metadata or {}).get("arm")) != arm:
            continue
        for s in log.samples or []:
            if s.error is not None:
                continue
            md = s.metadata or {}
            total = sum(int(u.total_tokens or 0) for u in (s.model_usage or {}).values())
            per.setdefault(f"{md.get('family', args.get('family'))}-{md.get('level', args.get('level'))}", []).append(total)
    return {c: {"b0": float(statistics.median(v)), "samples": len(v)} for c, v in per.items() if v and statistics.median(v) > 0}


def write_caps(path: Path, b0: dict[str, dict], source: str) -> dict:
    caps = {"multiple": CAP_MULTIPLE, "arm": CAP_ARM, "source": source, "cells": {c: {**v, "cap": int(math.ceil(CAP_MULTIPLE * v["b0"]))} for c, v in sorted(b0.items())}}
    rg._write_json(path, caps)
    return caps


def _level_distance(a: str, b: str) -> float:
    """Two levels' distance on a log scale, rounded so that equidistant levels (10 and 1000 from 100) tie exactly."""
    try:
        return round(abs(math.log(float(a)) - math.log(float(b))), 9)
    except ValueError:
        return math.inf


CAP_SOURCES = {"micro-pilot": 0, "tune": 1, "pilot": 1, "test": 2}  # how many cap files (micro-pilot's, pilot's) a phase reads
BASE_MULTIPLE_PHASES = ("micro-pilot", "tune", "pilot")  # they run at CAP_MULTIPLE; the test at the pilot gate's (D-039)


def cap_multiple(run: StudyRun, phase: str = "test") -> int:
    """The token caps' multiple of B0 `phase` applies: CAP_MULTIPLE up to the pilot (which may re-run arms at a larger
    one, `cap_gate`); after it, the pilot cap-hit gate's final multiple (config/cap_gate.json, D-039)."""
    if phase in BASE_MULTIPLE_PHASES or not run.cap_gate_path.is_file():
        return CAP_MULTIPLE
    return int(json.loads(run.cap_gate_path.read_text()).get("multiple") or CAP_MULTIPLE)


def token_caps(run: StudyRun, phase: str = "test", multiple: int | None = None) -> dict[str, dict]:
    """task cell -> {cap, b0, source} as `phase` applies them: `multiple` (default `cap_multiple(run, phase)`) x B0, from
    the micro-pilot's B0 (token_caps.json), then the pilot's for cells the micro-pilot did not run
    (token_caps_pilot.json). The micro-pilot itself runs uncapped (it measures B0); tune and the pilot read the
    micro-pilot's B0 only (never their own outputs); the test both. A cell still unmeasured borrows the cap of its
    family's nearest measured level (log scale; a tie takes the larger cap, so an extrapolation never cuts an arm
    short); a family never measured runs uncapped (`uncapped_cells`)."""
    if not run.spec.caps:
        return {}
    m = cap_multiple(run, phase) if multiple is None else multiple
    measured: dict[str, dict] = {}
    for path, source in ((run.token_caps_path, "micro-pilot"), (run.pilot_caps_path, "pilot"))[: CAP_SOURCES.get(phase, 2)]:
        if path.is_file():
            for c, v in (json.loads(path.read_text()).get("cells") or {}).items():
                measured.setdefault(c, {"cap": int(math.ceil(m * float(v["b0"]))), "b0": v["b0"], "source": source})
    out = dict(measured)
    for c in agent_task_cells(run):
        if c in out:
            continue
        family, level = c.split("-", 1)
        near = sorted(((_level_distance(level, m.split("-", 1)[1]), -v["cap"], m) for m, v in measured.items() if m.split("-", 1)[0] == family))
        if near and math.isfinite(near[0][0]):
            m = near[0][2]
            out[c] = {"cap": measured[m]["cap"], "b0": measured[m]["b0"], "source": f"borrowed from {m} ({measured[m]['source']})"}
    return out


def agent_task_cells(run: StudyRun) -> list[str]:
    return list(dict.fromkeys(tc for phase in ("micro-pilot", "tune", "pilot", "test") for c in phase_cells(run, phase) if c.kind == "agent" for tc in c.spec["cells"]))


def uncapped_cells(run: StudyRun) -> list[str]:
    caps = token_caps(run)
    return [c for c in agent_task_cells(run) if c not in caps] if run.spec.caps else []


def _caps_params(run: StudyRun, phase: str = "test") -> dict:
    return {c: v["cap"] for c, v in token_caps(run, phase).items()}


# --- Groups: one eval set per cell and environment ---------------------------------------------------


def run_groups(run: StudyRun, phase: str, offline: bool | None = None) -> list[dict]:
    """Every eval set a run phase runs, in order: per plan cell, the `selected` group (its arms with no selection's
    knobs), the `s7` group (S7 under the KG arm's schedule) and one `sel-<system>` group per selection with knobs, its
    arm alone under exactly those knobs (`env_group`, D-042). Each group carries its arms (declared and as run), the arms skipped
    because they are not built, its profile, split, seed block, worlds, epochs and, for agent cells of a capped study,
    each task cell's token cap. Knobs are not part of Inspect's task identity, so each group has its own log dir, keyed
    by everything that shapes it."""
    offline = run.offline if offline is None else offline
    selected = read_selected(run) if phase != "micro-pilot" and run.selected_path.is_file() else {}
    caps = _caps_params(run, phase)
    split = SPLITS[phase]
    seed_base = split_seed_base(run, split)
    targets = json.loads(run.s7_targets_path.read_text()) if phase == "test" and run.s7_targets_path.is_file() else None
    groups = []
    for cell in phase_cells(run, phase):
        s = cell.spec
        by_name: dict[str, list[dict]] = {}
        envs: dict[str, dict[str, str]] = {}
        for a in rg._arm_names(s["arms"]):
            run_arm = _resolved(a, selected)
            name, envs_ = env_group(a, run_arm, selected)
            envs[name] = envs_
            by_name.setdefault(name, []).append({"declared": a, "run": run_arm, "built": arm_built(run_arm, cell.kind)})
        for name in sorted(by_name, key=lambda n: (("selected", "s7").index(n) if n in ("selected", "s7") else 2, n)):
            members = by_name[name]
            g: dict[str, Any] = {
                "cell": cell.id,
                "plan_phase": cell.phase,
                "name": name,
                "kind": cell.kind,
                "primary": cell.phase in run.spec.primary,
                "arms": [{"declared": a["declared"], "run": a["run"]} for a in members if a["built"]],
                "skipped": [{"declared": a["declared"], "run": a["run"], "reason": "not built yet"} for a in members if not a["built"]],
                "env": envs[name] | (s7_env(run) if name == "s7" else {}),
                "profile": s["profile"],
                "effort": s.get("effort"),
                "models": dict(s.get("models") or {}),
                "split": split,
                "seed_base": seed_base,
                "epochs": OFFLINE_SCALE["epochs"] if offline else int(s.get("epochs", 1)),
            }
            if cell.kind == "agent":
                g |= {
                    "cells": list(s["cells"]),
                    "deliveries": list(s.get("deliveries", ["push"])),
                    "exposure": s.get("exposure", "retrieved"),
                    "n_worlds": {tc: (OFFLINE_SCALE["worlds_per_cell"] if offline else _world_count(s) - int(s.get("world_offset", 0))) for tc in s["cells"]},
                    "skip_worlds": int(s.get("world_offset", 0)) if not offline else 0,
                    "caps": {tc: caps[tc] for tc in s["cells"] if tc in caps},
                }
            else:
                from .worlds import gen_f8

                g |= {"level": str(int(s["N"])), "knobs": dict(s.get("knobs") or {}), "variant": gen_f8.variant_tag(s.get("knobs")), "n_worlds": OFFLINE_SCALE["sessions"] if offline else int(s["sessions"]), "window": window(run)}
            if name == "s7" and targets is not None:
                g["s7_targets"] = targets
            g["dir"] = f"{cell.id}/{name}-{rg._digest({k: v for k, v in g.items() if k not in ('primary', 'skipped', 'plan_phase')})}"
            groups.append(g)
    return groups


def group_params(groups: list[dict]) -> list[dict]:
    return [{k: v for k, v in g.items() if k != "primary"} for g in groups]


def skipped_arms(groups: list[dict]) -> list[dict]:
    return [{"cell": g["cell"], **a} for g in groups for a in g["skipped"]]


def group_profile(run: StudyRun, g: dict) -> Profile:
    return cell_profile(run, {"profile": g["profile"], "effort": g.get("effort"), "models": g.get("models")})


_GOLD: dict[tuple, Any] = {}


def offline_gold(run: StudyRun) -> Any:
    """The offline agent of agent cells: `llm.mock_multi.GoldMulti` over the run's worlds. It plays every role of every
    main-study arm from the worlds' gold (single agent, planner, orchestrator, worker, specialist, council member, chair,
    aggregator; `GoldMulti.kg` the APG classify), so the multi-agent arms run their real plumbing; the naive
    `mock_agent` would answer at once and never plan or delegate. Offline success is therefore about 100% by
    construction: a plumbing check, never a result. One instance per world set (the worlds' files and mtimes)."""
    from .llm.mock_multi import GoldMulti

    root = Config().worlds_dir
    key = (str(root), tuple(sorted((str(p), p.stat().st_mtime_ns) for p in root.rglob("*.json"))))
    if key not in _GOLD:
        _GOLD.clear()
        _GOLD[key] = GoldMulti(root)
    return _GOLD[key]


def group_models(run: StudyRun, profile: Profile, kind: str) -> tuple[Any, dict[str, Any]]:
    """Offline: the mocks under the profile's settings: `offline_gold` (agent cells; its `kg` for the kg role) or the
    naive `mock_session_agent` (sessions: it also answers management calls, todo extractions, CM-native's mock
    compaction and probes). Live: the profile's models, after the FX-2 preflight (sessions: the `probe` role when the
    profile sets one; management calls use the agent's model)."""
    if run.offline:
        if kind == "session":
            from .llm.mock_session import mock_session_agent

            return rg._mock_models(profile, {"agent": mock_session_agent})
        gold = offline_gold(run)
        return rg._mock_models(profile, {"agent": gold, "kg": gold.kg})
    from .models import agent_model, require_preflight, role_models

    require_preflight(profile, live=True, costs_path=run.costs_path, env_path=run.env_path)
    roles = ("probe",) if kind == "session" and "probe" in profile.roles else ("kg",) if kind == "agent" else ()
    return agent_model(profile), role_models(profile, roles)


def threshold(run: StudyRun) -> int:
    """T_abs, the sessions' compaction threshold (run_plan.yaml `study_g.threshold`), passed to every session task."""
    from .agent.session import plan_threshold

    return int(plan(run).study_g.get("threshold") or plan_threshold())


def agent_task(run: StudyRun, *, family: str, level: str, split: str, arm: str, delivery: str = "push", exposure: str = "retrieved", limit_worlds: int | None, seed_base: int, cap: int | None, **labels: Any):
    """One main-study task (`tasks.main.main_study`) with the cell's token cap as Inspect's `token_limit`."""
    from inspect_ai import task_with

    from .tasks.main import main_study

    t = main_study(family=family, level=level, split=split, arm=arm, exposure=exposure, delivery=delivery, limit_worlds=limit_worlds, seed_base=seed_base, **{k: v for k, v in labels.items() if v})
    return task_with(t, token_limit=int(cap)) if cap is not None else t


def session_task(run: StudyRun, *, level: str, split: str, arm: str, limit_worlds: int, variant: str, seed_base: int, **labels: Any):
    """One F8 session task (`tasks.study_g.f8_session`): the run's seed block, the plan's window and threshold."""
    from .tasks.study_g import f8_session

    return f8_session(level=level, split=split, arm=arm, limit_worlds=limit_worlds, window=window(run), variant=variant, threshold=threshold(run), seed_base=seed_base, **labels)


def group_tasks(run: StudyRun, g: dict) -> list:
    """The tasks of one group; call under the group's env (each task records its knobs)."""
    labels = {"plan_cell": g["cell"], "group": g["name"]}
    if g["kind"] == "session":
        return [session_task(run, level=g["level"], split=g["split"], arm=a["run"], limit_worlds=g["n_worlds"], variant=g["variant"], seed_base=g["seed_base"], **labels) for a in g["arms"]]
    skip = {"skip_worlds": g["skip_worlds"]} if g.get("skip_worlds") else {}
    return [
        agent_task(
            run, family=tc.split("-", 1)[0], level=tc.split("-", 1)[1], split=g["split"], arm=a["run"], delivery=delivery, exposure=g["exposure"],
            limit_worlds=g["n_worlds"][tc], seed_base=g["seed_base"], cap=g["caps"].get(tc), **labels, **skip,
        )  # fmt: skip
        for a in g["arms"]
        for delivery in g["deliveries"]
        for tc in g["cells"]
    ]


def group_cell(run: StudyRun, g: dict) -> PlanCell:
    """A group as a plan cell for the projection: its built arms at live sizes (whole worlds for an n_tasks cell)."""
    cell = plan(run).cell(g["cell"])
    spec = dict(cell.spec) | {"arms": [a["declared"] for a in g["arms"]]}
    if cell.kind == "agent" and "n_tasks" in spec:
        spec["n_tasks"] = _world_count(cell.spec) * TASKS_PER_WORLD
    return PlanCell(cell.id, cell.study, cell.phase, spec)


def group_projected(run: StudyRun, g: dict) -> float:
    return rg.project(run, [group_cell(run, g)]) if g["arms"] else 0.0


def run_group(run: StudyRun, g: dict, base_dir: Path, what: str) -> list[str]:
    """One group's eval set in `base_dir`/<its dir> (`run_gate.run_gate_tasks`: resumable, with the per-sample cost
    limit); its final logs."""
    profile = group_profile(run, g)
    models = group_models(run, profile, g["kind"])
    with rg._environ(g["env"]):
        tasks = group_tasks(run, g)
        per = rg.sample_usd(group_projected(run, g), [(tasks, g["epochs"])])
        arms = ", ".join(a["run"] for a in g["arms"])
        where = g["cells"] if g["kind"] == "agent" else f"F8-{g['level']}{'-' + g['variant'] if g['variant'] else ''}"
        print(f"[{what.split(' ', 1)[0]}] {g['cell']} / {g['name']}: {arms} x {where} ({g['split']}, {g['epochs']} epoch(s), profile {g['profile']}{', effort ' + g['effort'] if g.get('effort') else ''})", flush=True)
        return rg.run_gate_tasks(run, profile, models, tasks, base_dir / g["dir"], g["epochs"], what, per)


def run_phase_groups(run: StudyRun, record: dict, phase: str, groups: list[dict]) -> list[str]:
    """Run every group with an arm to run (a failure raises: a re-run resumes); records each group's log dir and keeps
    its final logs in the group (`log_files`)."""
    logs: list[str] = []
    for g in groups:
        if not g["arms"]:
            continue
        record["log_dirs"].append(_show(run.phase_dir(phase) / g["dir"]))
        g["log_files"] = run_group(run, g, run.phase_dir(phase), f"{phase} {g['cell']} ({g['name']})")
        logs += g["log_files"]
    return logs


def agent_logs(groups: list[dict]) -> list[str]:
    return [f for g in groups if g["kind"] == "agent" for f in g.get("log_files", [])]


# --- Phases ------------------------------------------------------------------------------------------


def _cfg_inputs(run: StudyRun, *names: str) -> dict[str, Path]:
    return {f"config/{n}": config_input(run.config(n), run.study) for n in names}


def env_knobs(run: StudyRun) -> dict[str, str]:
    """APE_* knobs that can change a phase's results (`run_gate._env_knobs`), less what this orchestrator sets itself."""
    managed = set(STUDY_MANAGED_ENV) | set(kg_env(run))
    return {k: v for k, v in rg._env_knobs().items() if k not in managed}


def base_params(run: StudyRun) -> dict:
    return {"scale": scale(run), "kg": kg_params(run), "env_knobs": env_knobs(run)}


def build_params(run: StudyRun) -> dict:
    from .models import build_settings

    model, effort = build_settings(study_profile(run))
    return {"model": model, "effort": effort, "fallback": os.environ.get("APE_BUILD_FALLBACK") == "1"}


def _build_params(run: StudyRun, split: str) -> dict:
    return base_params(run) | {"worlds": world_specs(run, split), "lightrag_kind": "oracle" if run.offline else "extract", "fake_author": run.offline, "builder": build_params(run)}


def project(run: StudyRun, cells: Sequence[PlanCell]) -> float:
    return rg.project(run, cells)


# preflight -------------------------------------------------------------------------------------------


def profiles_used(run: StudyRun) -> list[Profile]:
    names = dict.fromkeys([default_profile(run), *(c.spec["profile"] for c in plan(run).cells if c.study == run.study and c.enabled)])
    return [load_profile(n, run.models_path) for n in names]


def _preflight_params(run: StudyRun) -> dict:
    env: dict[str, Any] = {"code": rg.code_state(), "apg_core_commit": rg._apg_installed_commit(), "provenance": rg._provenance_facts(run)}
    if not run.offline:
        from .models import _api_key_present

        env["api_key_present"] = _api_key_present(run.env_path)
        env["probe_sha256"] = rg._sha256(run.probe_path)
    return base_params(run) | {"environment": env, "smoke_checks": list(run.spec.smoke_checks)}


NATIVE_ARM = "CM-native"


def native_routes(run: StudyRun) -> tuple[dict[str, str], list[str]]:
    """({where: route}, problems) for every session cell, and tuning-grid candidate, that runs CM-native: its agent
    model's route (`agent.cm_arms.native_route`: "provider" when the support record confirms native compaction,
    "mock" for the offline mock). A live model without a confirmed record is a problem."""
    from .agent.cm_arms import NativeCompactionUnsupported, native_route

    uses: dict[str, str] = {}
    for phase in ("micro-pilot", "test"):
        for cell in phase_cells(run, phase):
            if cell.kind == "session" and NATIVE_ARM in rg._arm_names(cell.spec["arms"]):
                uses[cell.id] = cell_profile(run, cell.spec).role("agent").model
    if run.config(run.grid_name).is_file():
        grid = _load_grid(run)
        tune = phase_cells(run, "tune")
        profile = cell_profile(run, tune[0].spec) if tune else study_profile(run)
        for name, sdef in grid["systems"].items():
            if any(c["arm"] == NATIVE_ARM for c in sdef["candidates"]):
                uses[f"{run.grid_name}: {name}"] = profile.role("agent").model
    routes, problems = {}, []
    for where, model in uses.items():
        try:
            routes[where] = native_route(rg.MOCK if run.offline else model)
        except NativeCompactionUnsupported as e:
            routes[where] = "unsupported"
            problems.append(f"{where}: {e}")
    return routes, list(dict.fromkeys(problems))


def _preflight(run: StudyRun, record: dict) -> None:
    problems: list[str] = []
    checks: dict[str, Any] = {}
    profiles = profiles_used(run)
    record["profiles"] = {p.name: p.summary() for p in profiles}
    # 1. FX-2 for every profile the study's cells use; live, the snapshot check against the probe and PROVENANCE.md.
    fx2 = [x for p in profiles for x in preflight(p, live=not run.offline, costs_path=run.costs_path, env_path=run.env_path, probe_path=run.probe_path, provenance_path=run.provenance_path)]
    checks["fx2_preflight"] = "ok" if not fx2 else "failed (see errors)"
    problems += fx2
    # 2. Live: the readiness probe lists every model the study calls.
    if run.offline:
        checks["probe"] = "skipped (offline)"
    elif not run.probe_path.is_file():
        problems.append(f"{_show(run.probe_path)} missing: run readiness/probe_openai.py --list first")
    else:
        try:
            available = set(json.loads(run.probe_path.read_text()).get("models_available") or [])
        except ValueError as e:
            problems.append(f"{_show(run.probe_path)} does not parse ({e}): re-run readiness/probe_openai.py --list")
            available = set()
        missing = sorted({f"{p.name}.{r}: {s.model}" for p in profiles for r, s in p.roles.items() if r != "build_fallback" and s.model.split("/", 1)[-1] not in available})
        if missing:
            problems.append(f"{_show(run.probe_path)} does not list {missing}: re-run readiness/probe_openai.py, or fix config/models.yaml")
        checks["probe"] = {"path": _show(run.probe_path), "sha256": rg._sha256(run.probe_path), "missing": missing}
    # 3. The installed APG is the pinned commit, and PROVENANCE.md records the pin.
    installed = rg._apg_installed_commit()
    checks["apg_core"] = {"installed_commit": installed, "pinned_commit": rg.APG_PIN}
    if installed != rg.APG_PIN:
        problems.append(f"installed apg-core is at {installed!r}, not the pinned {rg.APG_PIN} (PROVENANCE.md); run `uv sync`")
    if rg.APG_PIN not in (run.provenance_path.read_text() if run.provenance_path.is_file() else ""):
        problems.append(f"{_show(run.provenance_path)} does not record the APG pin {rg.APG_PIN}")
    if record["git"].get("dirty"):
        record["warnings"].append("the git tree is dirty: commit before a live run so results trace to a commit")
    # 4. The arms each phase runs that are not built yet: skipped offline, refused live (each phase checks its own).
    arms = {p: unbuilt_arms(run, p) for p in ("micro-pilot", "tune", "pilot", "test") if p in run.spec.phases}
    record["unbuilt_arms"] = {p: a for p, a in arms.items() if a}
    if record["unbuilt_arms"]:
        what = "skipped (offline)" if run.offline else "a live run refuses those phases until they are built"
        record["warnings"].append(f"arms not built yet, {what}: " + "; ".join(f"{p}: {a}" for p, a in record["unbuilt_arms"].items()))
    record["warnings"] += [f"KG arm: {n}" for n in kg_resolution(run).get("notes") or []]
    # 4b. CM-native runs only on a model whose native compaction is confirmed (`agent.cm_arms.native_route`): checked
    #     here for every model a CM-native cell or grid candidate uses, so a live run refuses now, not per sample.
    checks["cm_native"], native = native_routes(run)
    problems += native
    # 5. Live: a passing live smoke of this code (and of the study's own required checks), settled once per run.
    settlement = None
    if not run.offline:
        smoke_problems, settlement = rg._settle_smoke(run, load_profile(SMOKE_PROFILE, run.models_path), record, checks, extra_required=run.spec.smoke_checks)
        problems += smoke_problems
        record["warnings"] += storage_warnings()
    record["checks"] = checks
    if problems:
        raise PreflightError("preflight failed:\n" + "\n".join(f"  - {x}" for x in dict.fromkeys(problems)))
    if settlement is not None:
        rg.update_run_info(run, smoke_check=settlement)


# build-dev -------------------------------------------------------------------------------------------


def _build_dev(run: StudyRun, record: dict) -> None:
    build_world_set(run, record, "build-dev", "dev")


def _refuse_other_builder(run: StudyRun) -> str | None:
    """Live: the shared gate dev worlds' KG artifacts were authored by the gate run's builder (D-017: the profile's, or
    the fallback under APE_BUILD_FALLBACK=1). A study building them with another builder would re-author them in place,
    rewriting the gate's artifacts, so build-dev refuses until the builders agree."""
    if run.offline or not run.gate_run_id or not any(s.get("shared") and s["kinds"] for s in world_specs(run, "dev")):
        return None
    gate = rg.read_manifest(rg.GateRun(run.gate_run_id, runs_root=run.gate_runs_root), "build-dev") or {}
    theirs, ours = (gate.get("params") or {}).get("builder") or {}, build_params(run)
    if theirs and (theirs.get("model"), theirs.get("effort")) != (ours["model"], ours["effort"]):
        return (
            f"build-dev: gate run {run.gate_run_id!r} built the shared dev worlds with {theirs.get('model')} ({theirs.get('effort')}), this run would "
            f"build with {ours['model']} ({ours['effort']}) and re-author the gate's artifacts; set APE_BUILD_FALLBACK as the gate run did"
        )
    return None


def _build_dev_projected(run: StudyRun) -> float:
    return project(run, kg_build_cell(run, "dev"))  # shared gate dev worlds are the gate's (already built)


# micro-pilot -----------------------------------------------------------------------------------------


def _refuse_frozen(name: str) -> Callable[[StudyRun], str | None]:
    def refuse(run: StudyRun) -> str | None:
        if run.freeze_path.is_file():
            return f"{name}: run {run.run_id!r} is frozen ({_show(run.freeze_path)}) and {name} would rewrite frozen inputs; any change after the freeze is a logged deviation and a new --run-id"
        return None

    return refuse


def _refuse_all(*checks: Callable[[StudyRun], str | None]) -> Callable[[StudyRun], str | None]:
    def refuse(run: StudyRun) -> str | None:
        for check in checks:
            if reason := check(run):
                return reason
        return None

    return refuse


def _harness(run: StudyRun, record: dict, groups: list[dict], what: str) -> list[dict]:
    """PC5 early warning on the agent cells' logs (`gate_stats.harness_rates`; a token-cap or cost-limit hit is a cap
    hit): a warning per arm over a limit, never a failure."""
    from .analysis.gate_stats import MAX_CAP_HIT_RATE, MAX_ERROR_RATE, harness_rates, load_results

    if not (logs := agent_logs(groups)):
        return []
    try:
        rates = harness_rates(load_results(logs, require_cost=not run.offline), by=("arm", "delivery"), cap_exempt=("S7",))
    except Exception as e:  # noqa: BLE001  (a health report never fails a phase)
        record["warnings"].append(f"harness rates: could not be measured ({type(e).__name__}: {e})")
        return []
    record["warnings"] += [
        f"PC5 early warning ({what}): {h['arm']} ({h['delivery']}) errors {h['error_rate']:.1%}, cap hits {h['cap_hit_rate']:.1%} over {h['samples']} samples "
        f"(limits {MAX_ERROR_RATE:.0%} and {MAX_CAP_HIT_RATE:.0%}): fix the harness before the test phase"
        for h in rates
        if h["error_over"] or h["cap_hit_over"]
    ]
    return rates


def _summary(logs: list[str]) -> dict:
    """Per (arm, task cell): samples and mean success (agent cells: task_success; sessions: item_success)."""
    from inspect_ai.log import read_eval_log

    out: dict[str, dict[str, dict]] = {}
    for f in logs:
        log = read_eval_log(str(f))
        args, md = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or md.get("arm")
        cell = f"{md.get('family') or args.get('family')}-{args.get('level')}"
        vals = []
        for s in log.samples or []:
            sc = s.scores or {}
            if "task_success" in sc:
                vals.append(1.0 if sc["task_success"].value == "C" else 0.0)
            elif "f8_session_score" in sc:
                vals.append(float((sc["f8_session_score"].value or {}).get("item_success", 0.0)))
            else:
                vals.append(0.0)
        e = out.setdefault(str(arm), {}).setdefault(cell, {"samples": 0, "success": 0.0})
        e["success"] = (e["success"] * e["samples"] + sum(vals)) / max(1, e["samples"] + len(vals))
        e["samples"] += len(vals)
    return out


def _micro_pilot(run: StudyRun, record: dict) -> None:
    from .budget import calibrate

    build_world_set(run, record, "micro-pilot", "pilot")
    groups = run_groups(run, "micro-pilot")
    record["skipped_arms"] = skipped_arms(groups)
    logs = run_phase_groups(run, record, "micro-pilot", groups)
    summary: dict[str, Any] = {"run_id": run.run_id, "offline": run.offline, "study": run.study, "skipped_arms": record["skipped_arms"], "success": _summary(logs)}
    if run.spec.caps:
        b0 = b0_from_logs(agent_logs(groups))
        caps = write_caps(run.token_caps_path, b0, "micro-pilot")
        record["outputs"] |= {"token_caps": _show(run.token_caps_path)}
        record["token_caps"] = {c: v["cap"] for c, v in caps["cells"].items()}
        cells = {tc for g in groups if g["kind"] == "agent" for tc in g["cells"]}
        if missing := sorted(cells - set(b0)):
            record["warnings"].append(f"no B0 for {missing}: no {CAP_ARM} samples there; the pilot measures them, or they borrow a cap (`token_caps`)")
        summary["token_caps"] = caps
    if logs:
        run.out_config_dir.mkdir(parents=True, exist_ok=True)
        calibrate(logs, out_path=run.measured_out_path)  # the run's later projections use it (`run_gate._cost_kwargs`)
        record["outputs"] |= {"measured": _show(run.measured_out_path)}
    summary["harness"] = _harness(run, record, groups, "micro-pilot")
    out = run.phase_dir("micro-pilot") / "micro_pilot.json"
    rg._write_json(out, summary)
    record["outputs"] |= {"summary": _show(out)}


def _micro_pilot_params(run: StudyRun) -> dict:
    return _build_params(run, "pilot") | {"groups": group_params(run_groups(run, "micro-pilot")), "cap_multiple": CAP_MULTIPLE if run.spec.caps else None}


def _micro_pilot_projected(run: StudyRun) -> float:
    return project(run, kg_build_cell(run, "pilot") + [group_cell(run, g) for g in run_groups(run, "micro-pilot", offline=False) if g["arms"]])


# tune ------------------------------------------------------------------------------------------------


def _load_grid(run: StudyRun) -> dict:
    from .tuning import load_grid

    path = run.config(run.grid_name)
    if not path.is_file():
        raise PhaseError(f"{_show(path)} missing: the study's tuning grid (BUILD_PLAN B3/B11 write the real one)")
    grid = load_grid(path) or {}
    grid.setdefault("systems", {})
    return grid


def _system_cells(grid: dict, name: str) -> list[str]:
    return list(grid["systems"][name].get("dev_cells") or grid.get("dev_cells") or [])


def _system_kind(grid: dict, name: str) -> str:
    return "session" if any(c.startswith("F8-") for c in _system_cells(grid, name)) else "agent"


def study_grid(run: StudyRun) -> tuple[dict, list[dict]]:
    """The study's grid as this run tunes it, and the systems it skips: offline, the first N candidates per system,
    LightRAG arms as their oracle twins, and no system whose arms are not built (each recorded)."""
    grid = _load_grid(run)
    systems, skipped = {}, []
    for name, sdef in grid["systems"].items():
        kind = _system_kind(grid, name)
        cands = sdef["candidates"][: OFFLINE_SCALE["tune_candidates_per_system"]] if run.offline else list(sdef["candidates"])
        if run.offline:
            cands = [({**c, "declared_arm": c["arm"], "arm": rg.OFFLINE_ARMS[c["arm"]]} if c["arm"] in rg.OFFLINE_ARMS else dict(c)) for c in cands]
        if missing := sorted({c["arm"] for c in cands if not arm_built(c["arm"], kind)}):
            skipped.append({"system": name, "arms": missing, "reason": "not built yet"})
            continue
        systems[name] = {**sdef, "candidates": cands}
    return {**grid, "systems": systems}, skipped


SKEPTIC_KEY = "S1"  # the owners key of the skeptic, who signs off the baseline S1 runs as engineered (D-041)


def tuning_signoff_problems(grid: dict) -> list[str]:
    """Why a study grid is not ready for a live tune: a placeholder; a system, or any other key of `owners` (main's S1,
    the skeptic's sign-off that S1 runs as engineered), without a named owner and sign-off; or a multi-agent arm's
    system owned by the skeptic (`owners.S1`): the M arms' prompts need an independent author (brief §7.3)."""
    from .agent.solvers import MULTI_AGENT_ARMS

    owners, signed, systems = grid.get("owners") or {}, grid.get("signed_off") or {}, grid.get("systems") or {}
    problems = ["the grid is a placeholder (`placeholder: true`): BUILD_PLAN B3 / B11 declare the real one"] if grid.get("placeholder") else []
    for key in dict.fromkeys([*systems, *owners]):
        if rg._unset(owners.get(key)):
            problems.append(f"owners.{key} is not set")
        if signed.get(key) is not True:
            problems.append(f"signed_off.{key} is not true")
    skeptic = owners.get(SKEPTIC_KEY)
    if not rg._unset(skeptic):
        for name, sdef in systems.items():
            arms = {name, *(c.get("arm") for c in sdef.get("candidates") or [])}
            owner = owners.get(name)
            if arms & set(MULTI_AGENT_ARMS) and not rg._unset(owner) and str(owner).strip().casefold() == str(skeptic).strip().casefold():
                problems.append(f"owners.{name} is the skeptic (owners.{SKEPTIC_KEY}: {skeptic}): a multi-agent arm's prompts need an independent author (brief §7.3)")
    return problems


def cm_candidate_problems(system: str, env: dict[str, str]) -> list[str]:
    """A Study G candidate's env, for the session arm (system) it tunes: it sets only APE_CM_* knobs that arm's policy
    reads (`KNOBS`), with values the policy takes (`resolve_knobs`, `validate`: e.g. S-CM*'s stack). Other systems may
    set the same knob names: each tuned arm runs under exactly its own selection's knobs (D-042)."""
    from .agent import session  # noqa: F401  (registers the context policies)
    from .agent.context_policy import KNOB_ENV_PREFIX, policy_class, resolve_knobs

    try:
        cls = policy_class(system)
    except ValueError as e:
        return [str(e)]
    problems = [f"{k} is not an {KNOB_ENV_PREFIX}* knob" for k in env if not k.startswith(KNOB_ENV_PREFIX)]
    if foreign := sorted(k for k in env if k.startswith(KNOB_ENV_PREFIX) and k.removeprefix(KNOB_ENV_PREFIX).lower() not in cls.KNOBS):
        problems.append(f"{system} reads none of {foreign} (its knobs: {sorted(cls.KNOBS)})")
    try:
        cls.validate(resolve_knobs(cls, {k: str(v) for k, v in env.items()}))
    except ValueError as e:
        problems.append(f"{system} rejects {dict(env)}: {e}")
    return problems


def grid_problems(run: StudyRun) -> list[str]:
    """`tuning.study_grid_problems` on the study's whole grid (as declared, not the offline cut), against what its
    run_plan tuning cells price, with each study's own-knob check (main: `agent.multi.knobs.candidate_problems`;
    Study G: `cm_candidate_problems`)."""
    from .agent.multi.knobs import candidate_problems
    from .tuning import planned_tuning, study_grid_problems

    check = cm_candidate_problems if run.study == "study_g" else candidate_problems
    return study_grid_problems(_load_grid(run), planned_tuning((c.id, c.spec) for c in phase_cells(run, "tune")), check)


def _refuse_tune(run: StudyRun) -> str | None:
    if reason := _refuse_frozen("tune")(run):
        return reason
    if run.offline:
        return None
    path = run.config(run.grid_name)
    if problems := tuning_signoff_problems(_load_grid(run)):
        return f"tune: {_show(path)} is not signed off for a live tune: {'; '.join(problems)}. Each system's owner declares its candidates, then sets signed_off.<system>: true"
    if problems := grid_problems(run):
        return f"tune: {_show(path)} breaks the tuning rules (`tuning.study_grid_problems`), found before any dev run: {'; '.join(problems)}"
    return _refuse_unbuilt("tune")(run)


def _dev_limit(run: StudyRun, grid: dict, name: str) -> int:
    """Worlds per dev cell a system's candidates run on: the dev specs' count (sessions: per F8 variant)."""
    counts = {(f"{s['family']}-{s['level']}" if not s["knobs"] else _f8_cell({"N": s["level"], "knobs": s["knobs"]})): s["count"] for s in world_specs(run, "dev")}
    return max([int(counts.get(c, 1)) for c in _system_cells(grid, name)] or [1])


def _tune_params(run: StudyRun) -> dict:
    grid, skipped = study_grid(run)
    return base_params(run) | {
        "systems": sorted(grid["systems"]),
        "skipped_systems": skipped,
        "limit_worlds": {name: _dev_limit(run, grid, name) for name in grid["systems"]},
        "epochs": OFFLINE_SCALE["epochs"] if run.offline else max([int(c.spec.get("epochs", 1)) for c in phase_cells(run, "tune")] or [1]),
        "offline_arms": rg.OFFLINE_ARMS if run.offline else None,
        "caps": _caps_params(run, "tune"),
        "dev_seed_base": split_seed_base(run, "dev"),
    }


def _session_success(sample: Any) -> float:
    score = (sample.scores or {}).get("f8_session_score")
    return float((score.value or {}).get("item_success", 0.0)) if score is not None and isinstance(score.value, dict) else 0.0


def _tune(run: StudyRun, record: dict) -> None:
    from .tuning import tune

    params = _tune_params(run)
    grid, skipped = study_grid(run)
    record["skipped_systems"] = skipped
    record["grid_problems"] = grid_problems(run)  # a live tune refuses on any (`_refuse_tune`); offline, recorded
    if record["grid_problems"]:
        record["warnings"].append(f"the grid breaks the tuning rules (a live tune refuses it): {record['grid_problems']}")
    tdir = run.phase_dir("tune")
    log_dir, tlog = tdir / "logs", tdir / "tuning_log.jsonl"
    if tlog.exists():
        tlog.rename(tdir / f"tuning_log.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl")
    record["outputs"] = {"tuning_log": _show(tlog)}
    record["log_dirs"] = [_show(log_dir)]
    caps = _caps_params(run, "tune")
    dev_base = params["dev_seed_base"]
    tune_cells = phase_cells(run, "tune")
    profile = cell_profile(run, tune_cells[0].spec) if tune_cells else study_profile(run)
    selected: dict[str, dict] = {}
    for name in grid["systems"]:
        kind = _system_kind(grid, name)

        def make_task(cell: str, cand: dict, limit_worlds: int | None, kind: str = kind):
            if kind == "session":
                n, _, tag = cell.removeprefix("F8-").partition("-")
                return session_task(run, level=n, split="dev", arm=cand["arm"], limit_worlds=limit_worlds or 1, variant=tag, seed_base=dev_base)
            family, level = cell.split("-", 1)
            return agent_task(run, family=family, level=level, split="dev", arm=cand["arm"], delivery=cand.get("delivery", "push"), limit_worlds=limit_worlds, seed_base=dev_base, cap=caps.get(cell))

        agent, roles = group_models(run, profile, kind)
        print(f"[tune] {name}: {len(grid['systems'][name]['candidates'])} candidate(s) x {_system_cells(grid, name)}", flush=True)
        chosen = tune(
            name, agent, dict(roles), params["limit_worlds"][name], params["epochs"], grid, tlog, profile, log_dir=log_dir, costs_path=run.costs_path,
            make_task=make_task, sample_success=_session_success if kind == "session" else None,
        )  # fmt: skip
        cand = chosen["candidate"]
        entry = {"arm": cand["arm"], "env": {k: str(v) for k, v in (cand.get("env") or {}).items()}}
        entry |= {k: cand[k] for k in ("delivery", "declared_arm") if k in cand}
        selected[name] = entry | {"candidate": cand["id"], "mean_success": chosen["mean_success"], "cost_usd": chosen["cost_usd"]}
    header = (
        f"# Dev-tuned selections ({run.spec.title}), written by `python -m ape.run_study tune --study {run.study} --run-id {run.run_id}`"
        f"{' (OFFLINE: mock models)' if run.offline else ''}{' from a PLACEHOLDER grid' if grid.get('placeholder') else ''}.\n"
        "# A plan arm named by a key runs as its selection, under exactly its own selection's knobs (D-042).\n"
    )
    text = header + yaml.safe_dump(selected, sort_keys=False)
    for path in (tdir / "selected.yaml", run.selected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    record["outputs"] |= {"selected": _show(tdir / "selected.yaml"), "selected_config": _show(run.selected_path)}
    record["selected"] = selected
    # PC6-style completeness (the gate's check, on the study's systems): a record for every declared candidate, no
    # more configurations than budget_per_system (archived logs included), each selection logged as selected.yaml.
    from .analyze_gate import pc6

    completeness = pc6(tdir, grid, _load_grid(run), selected, [(name, name) for name in grid["systems"]])
    record["tuning_completeness"] = {k: completeness[k] for k in ("pass", "value", "threshold", "reason")} | {"systems": completeness["details"].get("systems")}
    if not completeness["pass"]:
        record["warnings"].append(f"tuning completeness (PC6-style) fails: {completeness['reason']}")
    if grid.get("placeholder"):
        record["warnings"].append(f"{run.grid_name} is a placeholder grid (BUILD_PLAN B3/B11 write the real one); a live tune refuses it")
    if skipped:
        record["warnings"].append(f"grid systems skipped, their arms not built yet: {skipped}")


def _tune_projected(run: StudyRun) -> float:
    return project(run, [c for c in plan(run).cells if c.study == run.study and c.phase in run.spec.cells["tune"] and c.enabled])


# pilot (main) ----------------------------------------------------------------------------------------


def _pilot(run: StudyRun, record: dict) -> None:
    from .analysis.pilot import median_or_none, realized_tokens
    from .budget import calibrate

    build_world_set(run, record, "pilot", "pilot")
    groups = run_groups(run, "pilot")
    record["skipped_arms"] = skipped_arms(groups)
    logs = run_phase_groups(run, record, "pilot", [g for g in groups if g["name"] != "s7"])
    # S7 targets: the KG arm's (S5's) median realized context per cell, then S7 (D-012 and D-024, as the gate's pilot).
    s7 = [g for g in groups if g["name"] == "s7" and g["arms"]]
    targets: dict[str, int] = {}
    if s7:
        tokens = realized_tokens(agent_logs(groups), "push")
        for c in dict.fromkeys(tc for g in s7 for tc in g["cells"]):
            m = median_or_none(tokens.get(("S5", c), []))
            if m is None:
                raise PhaseError(f"no realized context for S5 in {c} (no compiles in the pilot logs): cannot size S7")
            targets[c] = max(1, round(m))
        rg._write_json(run.s7_targets_path, targets)
        record["outputs"] |= {"s7_targets": _show(run.s7_targets_path)}
        for g in s7:  # knobs are not Inspect's task identity, nor are the targets: the log dir is keyed by them
            g["s7_targets"] = targets
            g["dir"] = f"{g['cell']}/s7-{rg._digest({k: v for k, v in g.items() if k not in ('primary', 'skipped', 'plan_phase', 'dir')})}"
        logs += run_phase_groups(run, record, "pilot", s7)
    summary: dict[str, Any] = {"run_id": run.run_id, "offline": run.offline, "study": run.study, "skipped_arms": record["skipped_arms"], "s7_targets": targets, "success": _summary(logs)}
    # D-039: the cap-hit gate (may re-run arms under a larger multiple); the calibration reads every arm's latest logs.
    if run.spec.caps:
        summary["cap_gate"], latest = _cap_gate(run, record, groups)
        logs = latest + [f for f in logs if f not in set(agent_logs(groups))]
    # B0 for the cells the micro-pilot did not measure (its caps stand where it did).
    measured = set(json.loads(run.token_caps_path.read_text()).get("cells") or {}) if run.token_caps_path.is_file() else set()
    b0 = {c: v for c, v in b0_from_logs(agent_logs(groups)).items() if c not in measured}
    caps = write_caps(run.pilot_caps_path, b0, "pilot")
    record["outputs"] |= {"token_caps_pilot": _show(run.pilot_caps_path)}
    record["token_caps"] = {c: v["cap"] for c, v in token_caps(run).items()}
    if uncapped := uncapped_cells(run):
        record["warnings"].append(f"no token cap for {uncapped}: no {CAP_ARM} B0 in the family; they run uncapped")
    summary["token_caps"] = {"pilot": caps, "all": token_caps(run), "uncapped": uncapped}
    if logs:
        run.out_config_dir.mkdir(parents=True, exist_ok=True)
        calibrate(logs, out_path=run.measured_out_path)  # the run's later projections use it (`run_gate._cost_kwargs`)
        record["outputs"] |= {"measured": _show(run.measured_out_path)}
    summary["harness"] = _harness(run, record, groups, "pilot")
    multiple = cap_multiple(run)
    summary["prereg_items"] = {
        "token caps": f"{multiple} x S1's B0{f' (D-039: raised from {CAP_MULTIPLE})' if multiple != CAP_MULTIPLE else ''}: " + (", ".join(f"{c} {v['cap']}" for c, v in token_caps(run).items()) or "none"),
        "S7 targets per cell": ", ".join(f"{c} {t}" for c, t in targets.items()) or "none",
        "KG arm": f"{kg_resolution(run).get('arm')} ({kg_resolution(run).get('source')})",
    }
    out = run.phase_dir("pilot") / "pilot.json"
    rg._write_json(out, summary)
    record["outputs"] |= {"pilot": _show(out)}
    record["s7_targets"] = targets


def cap_hit_rates(log_files: Sequence[str]) -> dict[str, dict[str, dict]]:
    """arm (as run) -> task cell -> {samples, cap_hits, token_hits, rate, token_rate} over the logs (D-039). A cap hit
    is B4's (`analysis.main_load.cap_hit_of`): an Inspect sample limit; or, unanswered, the single agent at its turn cap
    or any agent of a multi-agent arm stopped at its turn cap or on a limit. A token hit is a cap hit the token cap
    decided: Inspect's `token` limit, or an agent stopped on a limit while the sample records no other limit."""
    from .analysis.main_load import load_main

    if not log_files:
        return {}
    df = load_main(list(log_files), require_cost=False)
    out: dict[str, dict[str, dict]] = {}
    for (arm, cell), sub in df.groupby(["arm", "cell"], sort=True):
        cap = sub["cap_hit"].fillna(False).astype(bool)
        agent_limit = sub["agent_stops"].map(lambda d: isinstance(d, dict) and int(d.get("limit", 0)) > 0).astype(bool)
        token = cap & (sub["limit_hit"].eq("token") | (sub["limit_hit"].isna() & agent_limit))
        n = int(len(sub))
        out.setdefault(str(arm), {})[str(cell)] = {
            "samples": n, "cap_hits": int(cap.sum()), "token_hits": int(token.sum()), "rate": round(float(cap.sum()) / n, 4), "token_rate": round(float(token.sum()) / n, 4),
        }  # fmt: skip
    return out


def cap_gate_rule(run: StudyRun) -> str:
    """Which rate the gate tests (D-039). Live: the cap-hit rate (B4's rule, every cap). Offline: the token-hit rate
    only. The mocks' cap hits that are not the token cap's (a mock that never answers runs to the turn cap) are mock
    artefacts a larger token cap cannot change, so the rehearsal reports them and loops only on real token-cap hits."""
    return "token_rate" if run.offline else "rate"


def arms_over_cap(rates: dict[str, dict[str, dict]], rule: str) -> dict[str, list[str]]:
    """arm -> the task cells where its `rule` rate exceeds CAP_HIT_MAX (only arms with one)."""
    over = {arm: sorted(c for c, r in cells.items() if r[rule] > CAP_HIT_MAX) for arm, cells in sorted(rates.items())}
    return {arm: cells for arm, cells in over.items() if cells}


def cap_rerun_groups(run: StudyRun, groups: list[dict], arms: Sequence[str], multiple: int) -> list[dict]:
    """The pilot's agent groups again for `arms` only (by the arm as run), under `multiple` x B0 caps (the pilot's B0
    sources, `token_caps(run, "pilot", multiple)`), each in a log dir of its own (`<name>-x<multiple>-<digest>`)."""
    caps = {c: v["cap"] for c, v in token_caps(run, "pilot", multiple).items()}
    out = []
    for g in groups:
        mine = [a for a in g["arms"] if a["run"] in arms]
        if g["kind"] != "agent" or not mine:
            continue
        r = {k: v for k, v in g.items() if k not in ("log_files", "dir")} | {"arms": mine, "skipped": [], "caps": {tc: caps[tc] for tc in g["cells"] if tc in caps}, "cap_multiple": multiple}
        r["dir"] = f"{g['cell']}/{g['name']}-x{multiple}-{rg._digest({k: v for k, v in r.items() if k not in ('primary', 'skipped', 'plan_phase')})}"
        out.append(r)
    return out


def _log_arm(path: str) -> str | None:
    from inspect_ai.log import read_eval_log

    head = read_eval_log(str(path), header_only=True)
    return (head.eval.task_args or {}).get("arm") or (head.eval.metadata or {}).get("arm")


def _cap_gate(run: StudyRun, record: dict, groups: list[dict]) -> tuple[dict, list[str]]:
    """D-039, the pilot cap-hit gate: each arm's cap-hit rate per task cell over the pilot's logs (`cap_hit_rates`);
    while an arm exceeds CAP_HIT_MAX (`cap_gate_rule`), the multiple doubles for every arm (8 -> 16 -> 32, at most
    CAP_DOUBLINGS times) and the arms over it re-run their pilot cells under the new caps (pilot-sized; the budget is
    checked before each round), and the check repeats on their new logs. Writes config/cap_gate.json: the final
    multiple (the test's, `cap_multiple`), every round's rates, the arms still over it (the freeze refuses while any
    is) and the arms over the B4 rate that the offline rule only reports. Returns it and the latest logs of every arm."""
    rule = cap_gate_rule(run)
    logs = agent_logs(groups)
    latest_logs: dict[str, list[str]] = {}
    for f in logs:
        latest_logs.setdefault(str(_log_arm(f)), []).append(f)
    rates = cap_hit_rates(logs)
    latest = {arm: dict(cells) for arm, cells in rates.items()}
    multiple, top = CAP_MULTIPLE, CAP_MULTIPLE * 2**CAP_DOUBLINGS
    over = arms_over_cap(latest, rule)
    rounds: list[dict] = [{"multiple": multiple, "rerun": [], "rates": rates, "over": over}]
    while over and multiple < top:
        multiple *= 2
        rerun = cap_rerun_groups(run, groups, list(over), multiple)
        what = f"pilot cap re-run at {multiple} x B0 (D-039: {', '.join(f'{a} {cells}' for a, cells in over.items())} over {CAP_HIT_MAX:.0%})"
        print(f"[pilot] {what}", flush=True)
        require_affordable(sum(group_projected(run, g) for g in rerun), guard_remaining(run), what)
        new_logs = run_phase_groups(run, record, "pilot", rerun)
        new = cap_hit_rates(new_logs)
        for arm in over:
            latest[arm] = dict(new.get(arm) or {})
            latest_logs[arm] = [f for f in new_logs if _log_arm(f) == arm]
        rounds.append({"multiple": multiple, "rerun": sorted(over), "dirs": [_show(run.phase_dir("pilot") / g["dir"]) for g in rerun], "rates": new, "over": (over := arms_over_cap(latest, rule))})
    informational = {a: c for a, c in arms_over_cap(latest, "rate").items() if a not in over} if rule != "rate" else {}
    gate = {
        "decision": "D-039",
        "multiple": multiple,
        "base": CAP_MULTIPLE,
        "max": top,
        "threshold": CAP_HIT_MAX,
        "rule": rule,
        "rule_note": "cap-hit rate (B4's cap_hit_of)" if rule == "rate" else "offline: token-cap hits only; other cap hits are reported (`reported`), not looped on",
        "passed": not over,
        "over": over,
        "reported": informational,
        "rates": latest,
        "rounds": rounds,
    }
    rg._write_json(run.cap_gate_path, gate)
    record["outputs"] |= {"cap_gate": _show(run.cap_gate_path)}
    record["cap_gate"] = {k: gate[k] for k in ("multiple", "rule", "passed", "over", "reported")} | {"rounds": [{k: r[k] for k in ("multiple", "rerun", "over")} for r in rounds]}
    if multiple != CAP_MULTIPLE:
        record["warnings"].append(f"D-039: the token caps' multiple is {multiple} x B0 for every arm (raised from {CAP_MULTIPLE}); the test applies it")
    if over:
        record["warnings"].append(f"D-039: {over} still over {CAP_HIT_MAX:.0%} cap hits at {multiple} x B0: the freeze refuses this run")
    if informational:
        record["warnings"].append(f"cap hits over {CAP_HIT_MAX:.0%} that are not the token cap's (offline mock artefacts, reported only): {informational}")
    return gate, [f for fs in latest_logs.values() for f in fs]


def _pilot_params(run: StudyRun) -> dict:
    gate = {"base": CAP_MULTIPLE, "doublings": CAP_DOUBLINGS, "threshold": CAP_HIT_MAX, "rule": cap_gate_rule(run)} if run.spec.caps else None
    return _build_params(run, "pilot") | {"groups": group_params(run_groups(run, "pilot")), "caps": _caps_params(run, "pilot"), "s7_env": s7_env(run) if kg_resolution(run).get("needed") else None, "cap_gate": gate}


def _pilot_projected(run: StudyRun) -> float:
    """The pilot at live sizes and, for a capped study, D-039's worst case on top: every pilot arm re-run once per
    doubling (pilot-sized: the pilot's own tasks and epochs per cell)."""
    once = project(run, [group_cell(run, g) for g in run_groups(run, "pilot", offline=False) if g["arms"]])
    return once * (1 + (CAP_DOUBLINGS if run.spec.caps else 0))


# freeze ----------------------------------------------------------------------------------------------


def read_cap_gate(run: StudyRun) -> tuple[dict | None, list[str]]:
    """(the pilot's cap-hit gate record, the freeze's problems with it): a capped study with a pilot freezes only with
    the gate run and no arm still over CAP_HIT_MAX at the largest multiple (D-039)."""
    if not (run.spec.caps and "pilot" in run.spec.phases):
        return None, []
    if not run.cap_gate_path.is_file():
        return None, [f"{_show(run.cap_gate_path)} missing: the pilot's cap-hit gate (D-039) has not run; re-run the pilot"]
    gate = json.loads(run.cap_gate_path.read_text())
    if gate.get("over"):
        return gate, [f"D-039: {gate['over']} still over {CAP_HIT_MAX:.0%} cap hits at {gate['multiple']} x B0 (the largest multiple): the caps cannot be frozen"]
    return gate, []


def frozen_outputs(run: StudyRun) -> list[str]:
    """The run's outputs the freeze hashes: selected.yaml; main also its token caps (B0 files and the pilot's cap-hit
    gate, whose multiple the test applies) and, when the pilot runs S7, the S7 targets."""
    out = ["selected.yaml"]
    if run.spec.caps:
        out += ["token_caps.json", "token_caps_pilot.json"] + (["cap_gate.json"] if "pilot" in run.spec.phases else [])
    if "pilot" in run.spec.phases and any(a == "S7" for c in phase_cells(run, "pilot") for a in rg._arm_names(c.spec["arms"])):
        out.append("s7_targets.json")
    return out


def frozen_code(run: StudyRun) -> list[str]:
    """uv.lock and the study's analysis code (each existing file; a directory: its .py files)."""
    files = ["uv.lock"]
    for rel in run.spec.analysis_code:
        p = ROOT / rel
        files += sorted(str(f.relative_to(ROOT)) for f in p.rglob("*.py")) if p.is_dir() else ([rel] if p.is_file() else [])
    return list(dict.fromkeys(files))


def frozen_files(run: StudyRun, prereg: Path) -> dict[str, Path]:
    kg = kg_resolution(run)
    return (
        {run.spec.prereg: prereg}
        | {f"config/{n}": config_input(run.config(n), run.study) for n in (*FROZEN_CONFIG, run.grid_name)}
        | {f"config/{n}": run.out_config_dir / n for n in frozen_outputs(run)}
        | {f: ROOT / f for f in frozen_code(run)}
        | {k: rg._resolve(p) for k, p in (kg.get("files") or {}).items()}
    )


def read_freeze(run: StudyRun) -> dict | None:
    return json.loads(run.freeze_path.read_text()) if run.freeze_path.is_file() else None


def require_frozen(run: StudyRun) -> dict:
    record = read_freeze(run)
    if record is None:
        raise PhaseError(f"{run.study} run {run.run_id!r} is not frozen: python -m ape.run_study freeze --study {run.study} --run-id {run.run_id}{' --offline' if run.offline else ''}")
    if changed := rg.frozen_changes(record):
        raise PhaseError(f"frozen file(s) changed since the freeze at {record['frozen_at']}: " + "; ".join(changed) + ". Restore them; any change after the freeze is a logged deviation and a new --run-id")
    return record


def stale_upstream(run: StudyRun) -> list[str]:
    out = []
    for name in run.spec.rests_on:
        m = rg.read_manifest(run, name)
        if m is None or m.get("status") not in COMPLETE:
            out.append(f"{name} has not completed")
            continue
        try:
            state = phase_state(run, name)
        except (PhaseError, OSError, ValueError, BudgetError) as e:
            out.append(f"{name}: cannot verify it is current ({type(e).__name__}: {e})")
            continue
        if state["fingerprint"] != m.get("fingerprint"):
            changed = sorted(k for k, v in state["inputs"].items() if (m.get("inputs") or {}).get(k, {}).get("sha256") != v["sha256"])
            changed += [f"upstream {u}" for u, f in state["upstream"].items() if (m.get("upstream") or {}).get(u) != f]
            if state["params"] != m.get("params"):
                changed.append("params")
            out.append(f"{name} ran with other inputs than now ({', '.join(changed) or 'fingerprint'}): re-run {name} and the phases after it")
    return out


def run_role(run: StudyRun) -> dict:
    return {"kind": "extension", "of": run.extension_of} if run.extension_of else {"kind": "primary", "of": None}


def role_problems(run: StudyRun, role: dict) -> list[str]:
    """An extension names another frozen run of this study, in the same mode, that is not itself an extension."""
    if role["kind"] == "primary":
        return []
    of = role["of"]
    if of == run.run_id:
        return ["--extension-of names this run itself"]
    ref = run.runs_root / run.study / of / "freeze.json"
    if not ref.is_file():
        return [f"extension of {of!r}: {_show(ref)} missing; it must be a frozen {run.study} run"]
    rec = json.loads(ref.read_text())
    problems = []
    if (rec.get("role") or {}).get("kind") == "extension":
        problems.append(f"extension of {of!r}: that run is itself an extension; extend the primary run")
    if bool(rec.get("offline")) != run.offline:
        problems.append(f"extension of {of!r}: that run is {'offline' if rec.get('offline') else 'live'} and this one is not")
    return problems


def _rehearsal_prereg(run: StudyRun, text: str, items: dict[str, str]) -> tuple[Path, list[dict]]:
    start = rg.prereg_body_start(text)
    replaced: list[dict] = []

    def fill(m: re.Match) -> str:
        value = items.get((m.group(2) or "").strip(), "unfilled")
        replaced.append({"marker": m.group(0), "value": value})
        return f"OFFLINE-REHEARSAL({value})"

    copy = run.work_dir / run.spec.prereg
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text[:start] + rg.PLACEHOLDER_ITEM.sub(fill, text[start:]))
    return copy, replaced


def _prereg_items(run: StudyRun) -> dict[str, str]:
    for phase, name in (("pilot", "pilot.json"), ("micro-pilot", "micro_pilot.json")):
        path = run.phase_dir(phase) / name
        if path.is_file():
            return (json.loads(path.read_text()) or {}).get("prereg_items") or {}
    return {}


def _provenance_path(run: StudyRun) -> Path:
    return run.work_dir / "PROVENANCE.freeze.md" if run.offline else run.provenance_path


def _freeze_heading(run: StudyRun, freeze: dict) -> str:
    return f"## {run.spec.title} freeze: run `{run.run_id}` ({freeze['frozen_at']})"


def _write_freeze_provenance(run: StudyRun, freeze: dict) -> Path:
    path = _provenance_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _freeze_heading(run, freeze) in (path.read_text() if path.is_file() else ""):
        return path
    seeds_ = freeze["test_seeds"]
    kg = freeze.get("kg") or {}
    lines = [
        f"\n{_freeze_heading(run, freeze)}{' (OFFLINE REHEARSAL)' if run.offline else ''}\n",
        f"- **Commit:** `{freeze['git']['commit']}`; apg-core `{freeze['apg_core']['installed_commit']}` (pin `{rg.APG_PIN}`).",
        "- **Frozen files** (sha256):",
        *[f"  - `{f['path']}`{'' if f['path'] == k else f' ({k})'}{f' (the {run.study} slice)' if f.get('slice') else ''}: `{f['sha256']}`" for k, f in freeze["files"].items()],
        f"- **Design knobs** (APE_* set when the frozen phases ran; build-test and test refuse others): {rg._fmt_env(freeze.get('design_env'))}.",
        f"- **KG arm:** {kg.get('arm')} ({kg.get('source')}{', verdict ' + kg['verdict'] if kg.get('verdict') else ''}); knobs {rg._fmt_env(kg.get('env'))}." if kg.get("needed") else "- **KG arm:** none (no cell reads the KG).",
        *[f"- **{a} (inherited from the gate run's `{x['key']}`):** candidate {x.get('candidate')}; knobs {rg._fmt_env(x['env'])}." for a, x in (kg.get("inherited") or {}).items()],
        *([f"- **Role:** the extension of run `{freeze['role']['of']}`."] if freeze["role"]["kind"] == "extension" else []),
        f"- **Test seeds:** {seeds_['base']}-{seeds_['last']} ({run.study}'s block base {seeds_['base']}). A later {run.study} run freezes a fresh block. "
        f"<!-- ape:test-seeds study={run.study} run={run.run_id} base={seeds_['base']} count={seeds_['count']} -->",
        f"- **Code:** build-test and test run only at `{freeze['git']['commit']}` ({', '.join(rg.CODE_PATHS)} unchanged).",
        f"- **Record:** `{_show(run.freeze_path)}`.",
        "",
    ]
    with path.open("a") as f:
        f.write("\n".join(lines))
    return path


def _record_freeze(run: StudyRun, record: dict, freeze: dict, provenance: Path) -> None:
    record["outputs"] |= {"freeze": _show(run.freeze_path), "provenance": _show(provenance)}
    record["frozen"] = {k: freeze.get(k) for k in ("files", "rehearsal", "code_commit", "test_seeds", "design_env", "kg", "role")}
    if not run.offline:
        record["warnings"].append(f"commit {_show(run.provenance_path)} (the freeze record) before build-test")


def _refuse_refreeze(run: StudyRun) -> str | None:
    record = read_freeze(run)
    if record is None:
        return None
    changed = rg.frozen_changes(record)
    if not changed and (rg.read_manifest(run, "freeze") or {}).get("status") not in COMPLETE:
        return None  # interrupted between freeze.json and PROVENANCE.md: `_freeze` completes it
    what = ("frozen file(s) changed since: " + "; ".join(changed)) if changed else "its inputs are unchanged (--force does not re-freeze)"
    return f"freeze: run {run.run_id!r} was frozen at {record['frozen_at']}; a run is frozen once, and {what}. Any change after the freeze is a logged deviation and a new --run-id"


def _freeze(run: StudyRun, record: dict) -> None:
    if (existing := read_freeze(run)) is not None:
        provenance = _write_freeze_provenance(run, existing)
        record["warnings"].append(f"completed a freeze interrupted after freeze.json was written ({existing['frozen_at']})")
        _record_freeze(run, record, existing, provenance)
        return
    problems: list[str] = []
    text = run.prereg_path.read_text() if run.prereg_path.is_file() else ""
    if not text:
        problems.append(f"{_show(run.prereg_path)} is missing or empty (BUILD_PLAN B5 / B11 write it)")
    placeholders = rg.prereg_placeholders(text)
    record["placeholders"] = placeholders
    prereg, rehearsal = run.prereg_path, None
    if placeholders:
        if run.offline:
            prereg, rehearsal = _rehearsal_prereg(run, text, _prereg_items(run))
            if left := rg.prereg_placeholders(prereg.read_text()):
                problems.append("malformed placeholder(s) the rehearsal cannot fill: " + "; ".join(f"line {p['line']}: {p['marker']}" for p in left))
            record["warnings"].append(f"offline rehearsal: {len(rehearsal)} placeholder(s) in {_show(run.prereg_path)} filled in the copy {_show(prereg)}")
        else:
            listing = "\n".join(f"    line {p['line']}: {p['marker']}" for p in placeholders)
            problems.append(f"{_show(run.prereg_path)} still has {len(placeholders)} unfilled item(s):\n{listing}")
    analysis_ready = analysis_available(run)
    if not analysis_ready:
        msg = f"{run.spec.analysis} does not exist ({ANALYSIS_NOT_IMPLEMENTED}): the analysis code is frozen with the design (BUILD_PLAN B4 / B10)"
        (record["warnings"] if run.offline else problems).append(msg)
    files = frozen_files(run, prereg)
    if missing := [f"{k} ({_show(p)})" for k, p in files.items() if not p.is_file()]:
        problems.append(f"missing frozen input(s): {missing}")
    changes = rg.git_tracked_changes()
    if changes is None:
        (record["warnings"] if run.offline else problems).append("git is unavailable: the freeze must record a clean commit")
    elif changes:
        msg = f"the git tree has uncommitted changes {changes[:10]}{' ...' if len(changes) > 10 else ''}: commit them so the frozen design traces to a commit"
        (record["warnings"] if run.offline else problems).append(msg)
    if stale := stale_upstream(run):
        (record["warnings"] if run.offline else problems).append("not current, so the frozen design would not be what ran: " + "; ".join(stale))
    if uncapped := uncapped_cells(run):
        record["warnings"].append(f"cells without a token cap (no B0 in their family): {uncapped}")
    gate, gate_problems = read_cap_gate(run)
    problems += gate_problems
    seed_base, seed_problems = choose_test_seed_base(run)
    problems += seed_problems
    role = run_role(run)
    problems += role_problems(run, role)
    if problems:
        raise PhaseError("freeze refused:\n" + "\n".join(f"  - {x}" for x in problems))
    git = rg.git_state()
    count = test_seed_count(run)
    code = [c for c in frozen_code(run) if c != "uv.lock"]
    freeze = {
        "run_id": run.run_id,
        "study": run.study,
        "frozen_at": rg._now(),
        "offline": run.offline,
        "rehearsal": rehearsal is not None,
        "placeholders_replaced": rehearsal or [],
        "files": {k: rg._file_entry(p) for k, p in files.items()},
        "git": git,
        "code_commit": git["commit"],
        "analysis": {"module": run.spec.analysis, "implemented": analysis_ready, "code": code, "commit": rg._git("log", "-1", "--format=%H", "--", *code) if code else None},
        "apg_core": {"installed_commit": rg._apg_installed_commit(), "pinned_commit": rg.APG_PIN},
        "upstream": record["upstream"],
        "phases": {n: (rg.read_manifest(run, n) or {}).get("fingerprint") for n in run.spec.rests_on},
        "test_seeds": {"base": seed_base, "count": count, "block": SEED_BLOCK, "last": seed_base + count - 1, "range": list(seeds(run)["test"])},
        "role": role,
        "kg": kg_resolution(run),
        "token_caps": _caps_params(run),
        "cap_multiple": cap_multiple(run) if run.spec.caps else None,
        "cap_gate": {k: gate.get(k) for k in ("multiple", "base", "max", "threshold", "rule", "passed", "rates", "rounds")} if gate else None,
        "design_env": env_knobs(run),
    }
    if rehearsal is not None:
        freeze["files"][f"{run.spec.prereg} (draft)"] = {"path": _show(run.prereg_path), "sha256": rg._sha256(run.prereg_path)}
    rg._write_json(run.freeze_path, freeze)
    provenance = _write_freeze_provenance(run, freeze)
    if rehearsal is not None:
        record["outputs"]["prereg_rehearsal"] = _show(prereg)
    _record_freeze(run, record, freeze, provenance)


def _freeze_inputs(run: StudyRun) -> dict[str, Path]:
    return frozen_files(run, run.prereg_path)


# build-test, test ------------------------------------------------------------------------------------


def _frozen_inputs(run: StudyRun) -> dict[str, Path]:
    return {k: rg._entry_path(f) for k, f in ((read_freeze(run) or {}).get("files") or {}).items()}


def design_env_changes(run: StudyRun) -> list[str]:
    frozen = (read_freeze(run) or {}).get("design_env")
    if frozen is None:
        return []
    now = env_knobs(run)
    return [f"{k}: frozen {frozen.get(k)!r}, now {now.get(k)!r}" for k in sorted(set(frozen) | set(now)) if frozen.get(k) != now.get(k)]


def _refuse_unless_frozen(name: str) -> Callable[[StudyRun], str | None]:
    """build-test and test: frozen, every frozen file unchanged; live also the freeze commit's code, the frozen design
    knobs and the frozen KG resolution."""

    def refuse(run: StudyRun) -> str | None:
        try:
            freeze = require_frozen(run)
        except PhaseError as e:
            return f"{name}: {e}"
        if run.offline:
            return None
        drift = rg.code_drift(freeze.get("code_commit"))
        if drift is None:
            return f"{name}: cannot verify that {', '.join(rg.CODE_PATHS)} are the freeze commit's (git unavailable, or freeze.json has no commit)"
        if drift:
            return f"{name}: code changed since the freeze commit {freeze.get('code_commit')}: {drift[:10]}{' ...' if len(drift) > 10 else ''}. Check out that commit to run it"
        if knobs := design_env_changes(run):
            return f"{name}: design knob(s) differ from the frozen ones: {'; '.join(knobs)}"
        if kg_params(run) != kg_params_of(freeze.get("kg")):
            return f"{name}: the KG resolution differs from the frozen one ({(freeze.get('kg') or {}).get('arm')} from {(freeze.get('kg') or {}).get('source')}): pass the frozen --gate-run-id"
        return None

    return refuse


def _warn_drift(run: StudyRun, record: dict) -> None:
    if not run.offline:
        return
    freeze = read_freeze(run) or {}
    if drift := rg.code_drift(freeze.get("code_commit")):
        record["warnings"].append(f"code differs from the rehearsal freeze's commit: {drift[:10]}{' ...' if len(drift) > 10 else ''} (a live run would refuse)")
    if knobs := design_env_changes(run):
        record["warnings"].append(f"design knob(s) differ from the rehearsal freeze's: {'; '.join(knobs)} (a live run would refuse)")


def _build_test(run: StudyRun, record: dict) -> None:
    freeze = require_frozen(run)
    record["frozen_at"] = freeze["frozen_at"]
    _warn_drift(run, record)
    base, count = run_test_seed_base(run), int(freeze["test_seeds"]["count"])
    if not run.offline and (clash := rg._overlaps(base, count, used_test_seed_blocks(run))):
        raise PhaseError(f"build-test: the frozen test seeds {base}+ overlap another frozen {run.study} run's block: {clash}")
    record["test_seeds"] = {"base": base, "count": count}
    with rg._environ({TEST_SPLIT_ENV: split_lock_value(run.study, run.run_id)}):
        build_world_set(run, record, "build-test", "test")


def _build_test_params(run: StudyRun) -> dict:
    if read_freeze(run) is None:
        return base_params(run)  # not frozen: the freeze check refuses the phase
    return _build_params(run, "test") | {"test_seed_base": run_test_seed_base(run)}


def _test_params(run: StudyRun) -> dict:
    if read_freeze(run) is None or not run.selected_path.is_file():
        return base_params(run)
    return base_params(run) | {"groups": group_params(run_groups(run, "test")), "lightrag_kind": "oracle" if run.offline else "extract"}


def _test_projected(run: StudyRun) -> float:
    if not run.selected_path.is_file():
        return project(run, phase_cells(run, "test"))
    return sum(group_projected(run, g) for g in run_groups(run, "test", offline=False))


def _test_remaining(run: StudyRun) -> float:
    """What the test phase's start must afford (as the gate's): the primary groups still to do, group by group
    (finished groups count 0, partly run ones their projection less their logs' spend); every other group is guarded
    before it runs, so a short budget stops the rest and leaves the primary evidence complete."""
    if read_freeze(run) is None or not run.selected_path.is_file():
        return _test_projected(run)
    return sum(group_remaining(run, g, group_projected(run, g)) for g in run_groups(run, "test") if g["primary"] and g["arms"])


def group_done(log_dir: Path, g: dict) -> bool:
    """Whether a group's eval set finished: its runner index's last run is `done` and every task has a successful log."""
    from .runner import read_index

    index = read_index(log_dir)
    runs = index.get("runs") or []
    ok = [t for t in (index.get("tasks") or {}).values() if t.get("status") == "success"]
    expected = len(g["arms"]) * (len(g["deliveries"]) * len(g["cells"]) if g["kind"] == "agent" else 1)
    return bool(runs) and runs[-1].get("status") == "done" and len(ok) >= expected


def group_remaining(run: StudyRun, g: dict, projected: float) -> float:
    """What a group still costs: 0 when it finished, else its (live) projection less what its logs already hold."""
    log_dir = run.phase_dir("test") / g["dir"]
    return 0.0 if group_done(log_dir, g) else max(0.0, projected - rg.group_spent(log_dir))


def guard_remaining(run: StudyRun) -> float:
    """What is left for this study: the smaller of the program's remainder and the study's allocation's."""
    s = spend(run)
    return min(s["remaining_usd"], s["study_remaining_usd"])


def _test(run: StudyRun, record: dict) -> None:
    freeze = require_frozen(run)
    record["frozen_at"] = freeze["frozen_at"]
    _warn_drift(run, record)
    groups = run_groups(run, "test")
    tdir = run.phase_dir("test")
    record["skipped_arms"] = skipped_arms(groups)
    cells: dict[str, dict] = {}
    entries = []
    for g in groups:
        entry = {
            "name": g["name"], "kind": g["kind"], "arms": g["arms"], "skipped": g["skipped"], "profile": g["profile"], "effort": g.get("effort"), "models": g.get("models") or {},
            "split": g["split"], "seed_base": g["seed_base"], "epochs": g["epochs"], "env": g["env"], "log_dir": _show(tdir / g["dir"]), "log_files": [],
            "status": "pending" if g["arms"] else "skipped",
        } | ({"cells": g["cells"], "deliveries": g["deliveries"], "n_worlds": g["n_worlds"], "caps": g["caps"]} if g["kind"] == "agent" else {"level": g["level"], "variant": g["variant"], "n_worlds": g["n_worlds"]})  # fmt: skip
        cells.setdefault(g["cell"], {"status": "pending", "primary": g["primary"], "plan_phase": g["plan_phase"], "groups": []})["groups"].append(entry)
        entries.append((g, entry))
    record["cells"], record["primary_complete"] = cells, False
    record["log_files_relative_to"] = "run_dir"  # `log_files` entries: relative to the run directory (`resolve_log`)
    record["log_dirs"] = [e["log_dir"] for g, e in entries if g["arms"]]
    stopped: BudgetError | None = None

    def statuses() -> None:
        for c in cells.values():
            st = {x["status"] for x in c["groups"]}
            c["status"] = next((s for s in ("failed", "stopped", "pending") if s in st), "done" if "done" in st else "skipped")
        record["primary_complete"] = all(c["status"] in ("done", "skipped") for c in cells.values() if c["primary"])

    for g, entry in entries:
        if not g["arms"]:
            continue
        what = f"test {g['cell']} ({g['name']})"
        if stopped is None:
            try:
                entry["projected_usd"] = round(group_projected(run, g), 4)  # live sizes (`group_cell`), offline too
                entry["projected_remaining_usd"] = round(group_remaining(run, g, entry["projected_usd"]), 4)
                require_affordable(entry["projected_remaining_usd"], guard_remaining(run), what)
            except BudgetError as e:
                stopped = e
        if stopped is not None:
            entry["status"] = "stopped"
            continue
        try:
            entry["log_files"] = [run_relative(run, f) for f in run_group(run, g, tdir, what)]
            entry["status"] = "done"
        except Exception as e:  # noqa: BLE001  (recorded; the other cells still run, and the phase then fails)
            entry |= {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        statuses()
        rg._write_json(run.manifest_path("test"), record)
    statuses()
    record["outputs"] = {f"logs:{g['dir']}": e["log_dir"] for g, e in entries if e["status"] == "done"}
    failed = [f"{g['cell']} ({g['name']}): {e['error']}" for g, e in entries if e["status"] == "failed"]
    if stopped is not None:
        not_run = [f"{g['cell']} ({g['name']})" for g, e in entries if e["status"] == "stopped"]
        raise BudgetError(f"{stopped}. Stopped before {not_run}; the primary cells are {'complete' if record['primary_complete'] else 'NOT complete'}" + (f"; failed: {failed}" if failed else ""))
    if failed:
        raise PhaseError(f"{len(failed)} test group(s) failed (re-run to resume; finished logs are reused): " + "; ".join(failed))


# analyze ---------------------------------------------------------------------------------------------


def analysis_available(run: StudyRun) -> bool:
    return importlib.util.find_spec(run.spec.analysis) is not None


def _analyze_inputs(run: StudyRun) -> dict[str, Path]:
    """Every file the analysis modules read (`ape.analyze_main`, `ape.analyze_g`), so a changed one re-runs the phase:
    the freeze, the test worlds, the selections and the tuning log, the study's slices of run_plan, models (each
    group's tier / capability point) and model_costs, and the ledger. The test manifest's cells and the tune manifest
    are in the params (`_analyze_params`), without the fields a skip rewrites."""
    return {
        "freeze.json": run.freeze_path,
        "build-test/worlds.json": run.phase_dir("build-test") / "worlds.json",
        "config/selected.yaml": run.selected_path,
        "tune/tuning_log.jsonl": run.phase_dir("tune") / "tuning_log.jsonl",
        "config/run_plan.yaml": config_input(run.config("run_plan.yaml"), run.study),
        "config/models.yaml": config_input(run.models_path, run.study),
        "config/model_costs.yaml": config_input(run.costs_path, run.study),
        "ledger": Config().ledger_path,
    }


SKIP_REWRITES = ("status", "skipped_at", "skip_reason", "history")  # what `run_phase` rewrites when it skips a phase


def manifest_digest(run: StudyRun, phase: str) -> str | None:
    """A phase manifest's digest without the fields a skip rewrites (a re-invoked `all` skips the phase and rewrites
    them, which must not make the analysis stale)."""
    m = rg.read_manifest(run, phase)
    return None if m is None else rg._digest({k: v for k, v in m.items() if k not in SKIP_REWRITES})


def _analyze_params(run: StudyRun) -> dict:
    test = rg.read_manifest(run, "test") or {}
    return {
        "test_cells": rg._digest(test.get("cells")),
        "primary_complete": test.get("primary_complete"),
        "tune_manifest": manifest_digest(run, "tune"),
        "module": run.spec.analysis,
        "implemented": analysis_available(run),
        "code": {f: rg._sha256(ROOT / f) for f in frozen_code(run) if f != "uv.lock"},
    }


def _analyze(run: StudyRun, record: dict) -> None:
    out = run.dir / REPORT_DIR
    if not analysis_available(run):
        if not run.offline:
            raise PhaseError(f"{ANALYSIS_NOT_IMPLEMENTED}: {run.spec.analysis} does not exist yet (BUILD_PLAN B4 / B10); a live analyze phase never completes without it")
        test = rg.read_manifest(run, "test") or {}
        stub = {
            "status": "not_implemented",
            "note": f"{run.spec.analysis}.analyze(run) does not exist yet (BUILD_PLAN B4 / B10); this offline stub lists what ran",
            "study": run.study,
            "run_id": run.run_id,
            "offline": True,
            "test": {"status": test.get("status"), "primary_complete": test.get("primary_complete"), "cells": {c: x.get("status") for c, x in (test.get("cells") or {}).items()}},
            "skipped_arms": test.get("skipped_arms") or [],
        }
        rg._write_json(out / "analysis.json", stub)
        lines = [f"# {run.spec.title} report: run `{run.run_id}` (OFFLINE STUB)", "", f"**{ANALYSIS_NOT_IMPLEMENTED}.** {stub['note']}.", "", "| Cell | Status |", "|---|---|"]
        lines += [f"| {c} | {s} |" for c, s in stub["test"]["cells"].items()]
        (out / "report.md").write_text("\n".join(lines) + "\n")
        record["outputs"] = {"analysis": _show(out / "analysis.json"), "report": _show(out / "report.md")}
        record["analysis"] = {"status": "not_implemented", "module": run.spec.analysis}
        record["warnings"].append(f"{ANALYSIS_NOT_IMPLEMENTED} ({run.spec.analysis}): an offline stub report was written; a live run refuses this phase")
        return
    result = importlib.import_module(run.spec.analysis).analyze(run)
    record["analysis"] = result if isinstance(result, dict) else {"result": str(result)}
    record["outputs"] = {f.name: _show(f) for f in sorted(out.glob("*")) if f.is_file()} if out.is_dir() else {}
    if "report.md" not in record["outputs"]:
        raise PhaseError(f"{run.spec.analysis}.analyze wrote no {_show(out / 'report.md')}")


# --- Phase registry ----------------------------------------------------------------------------------


def phase_defs(study: str) -> dict[str, rg.Phase]:
    s = STUDIES[study]
    defs = {
        "preflight": rg.Phase("preflight", _preflight, inputs=lambda r: _cfg_inputs(r, "models.yaml", "model_costs.yaml", "run_plan.yaml"), params=_preflight_params, projected=lambda r: 0.0, profile=study_profile),
        "build-dev": rg.Phase(
            "build-dev", _build_dev, inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml", r.grid_name), params=lambda r: _build_params(r, "dev"),
            projected=_build_dev_projected, profile=study_profile, requires=("preflight",), refuse=_refuse_other_builder,
        ),
        "micro-pilot": rg.Phase(
            "micro-pilot", _micro_pilot, inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml", "model_costs.yaml"), params=_micro_pilot_params,
            projected=_micro_pilot_projected, profile=study_profile, requires=("preflight",), refuse=_refuse_all(_refuse_frozen("micro-pilot"), _refuse_unbuilt("micro-pilot")),
        ),
        "tune": rg.Phase(
            "tune", _tune, inputs=lambda r: _cfg_inputs(r, r.grid_name, "run_plan.yaml", "models.yaml", "model_costs.yaml") | ({"config/token_caps.json": r.token_caps_path} if r.spec.caps else {}),
            params=_tune_params, projected=_tune_projected, profile=study_profile, requires=("preflight", "build-dev", *(("micro-pilot",) if s.caps else ())),
            upstream=("build-dev", *(("micro-pilot",) if s.caps else ())), refuse=_refuse_tune,
        ),
        "pilot": rg.Phase(
            "pilot", _pilot, inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml", "model_costs.yaml") | {"config/selected.yaml": r.selected_path, "config/token_caps.json": r.token_caps_path},
            params=_pilot_params, projected=_pilot_projected, profile=study_profile, requires=("preflight", "micro-pilot", "tune"), upstream=("micro-pilot", "tune"),
            refuse=_refuse_all(_refuse_frozen("pilot"), _refuse_unbuilt("pilot")),
        ),
        "freeze": rg.Phase(
            "freeze", _freeze, inputs=_freeze_inputs, params=lambda r: base_params(r) | {"frozen": list(frozen_files(r, r.prereg_path)), "rehearsal": r.offline},
            projected=lambda r: 0.0, profile=study_profile, requires=("preflight", *s.rests_on), upstream=tuple(p for p in s.rests_on if p != "build-dev"), refuse=_refuse_refreeze,
        ),
        "build-test": rg.Phase(
            "build-test", _build_test, inputs=_frozen_inputs, params=_build_test_params, projected=lambda r: project(r, kg_build_cell(r, "test")),
            profile=study_profile, requires=("freeze",), upstream=("freeze",), refuse=_refuse_unless_frozen("build-test"),
        ),
        "test": rg.Phase(
            "test", _test, inputs=_frozen_inputs, params=_test_params, projected=_test_projected, profile=study_profile, requires=("freeze", "build-test"),
            upstream=("freeze", "build-test"), refuse=_refuse_all(_refuse_unless_frozen("test"), _refuse_unbuilt("test")), remaining=_test_remaining,
        ),
        "analyze": rg.Phase("analyze", _analyze, inputs=_analyze_inputs, params=_analyze_params, projected=lambda r: 0.0, profile=study_profile),
    }  # fmt: skip
    return {p: defs[p] for p in s.phases}


PHASE_DEFS = {study: phase_defs(study) for study in STUDIES}
assert all(tuple(PHASE_DEFS[s]) == STUDIES[s].phases for s in STUDIES)


# --- Running phases ----------------------------------------------------------------------------------


def spend(run: StudyRun) -> dict:
    """`run_gate.spend` (the program's spend against its budget, and this run's own), plus the study's allocation
    (run_plan.yaml budget.allocations) less what the registry attributes to the study's label."""
    out = rg.spend(run)
    allocation = float((plan(run).budget.get("allocations") or {}).get(run.study, out["budget_usd"]))
    label = run.study + ("-offline" if run.offline else "")
    spent = float((out["program"].get("by_study") or {}).get(label, 0.0))
    return out | {"study": run.study, "study_allocation_usd": allocation, "study_spent_usd": spent, "study_remaining_usd": allocation - spent}


def stray_environment(run: StudyRun) -> list[str]:
    """Live runs: settings that would silently move a paid phase off the design or its directories (as the gate's)."""
    if run.offline:
        return []
    out = []
    for name, default in (("APE_WORLDS", ROOT / "worlds"), ("APE_INDICES", ROOT / "indices"), ("APE_CACHE", ROOT / "cache")):
        raw = os.environ.get(name, "").strip()
        if raw and Path(raw).resolve() != default.resolve():
            out.append(f"{name}={raw} (studies use {_show(default)})")
    if (raw := os.environ.get("APE_EMBEDDINGS", "").strip()) and raw != "openai":
        out.append(f"APE_EMBEDDINGS={raw} (live runs use the profile's embedding model)")
    sp = study_profile(run)
    for name, want in (("APE_EMBEDDING_MODEL", sp.role("embeddings").model), ("APE_MODEL_PROFILE", sp.name)):
        raw = os.environ.get(name, "").strip()
        if raw and raw != want:
            out.append(f"{name}={raw} (the {run.study} profile's is {want})")
    out += [f"{name}={os.environ[name]} (the builder comes from config/models.yaml)" for name in ("APE_BUILD_MODEL", "APE_BUILD_EFFORT") if os.environ.get(name, "").strip()]
    return out


@contextlib.contextmanager
def run_environment(run: StudyRun) -> Iterator[None]:
    """As `run_gate.run_environment`, for a study: offline every path under runs/<study>/<id>/work, fake embeddings and
    an OpenAI key that reaches nothing; the run's S7 targets; the test split locked (only build-test unlocks it, for this
    study); the spend label `<study>/<id>`; the session checkpoint directory; and the KG arm with its knobs and the
    inherited selections' (`kg_env`). A live shell value for one of those knobs that differs from the resolution is
    refused, never silently overridden."""
    kg = kg_env(run)
    if not run.offline and (clash := [f"{k}={os.environ[k]} (the gate's verdict sets {v})" for k, v in kg.items() if os.environ.get(k) not in (None, v)]):
        raise PhaseError("the environment sets the KG arm's knobs to other values than the gate run's selection; unset: " + "; ".join(clash))
    if run.offline:
        w = run.work_dir
        updates: dict[str, str | Path | None] = {
            "APE_WORLDS": w / "worlds", "APE_INDICES": w / "indices", "APE_CACHE": w / "cache", "APE_EMBEDDINGS": "fake", "APE_MODEL_PROFILE": None,
            "APE_BUILD_MODEL": None, "APE_BUILD_EFFORT": None, "APE_BUILD_FALLBACK": None, "OPENAI_API_KEY": rg.OFFLINE_KEY, "OPENAI_BASE_URL": rg.OFFLINE_BASE_URL,
        }  # fmt: skip
    else:
        updates = {} if os.environ.get("APE_EMBEDDING_MODEL") else {"APE_EMBEDDING_MODEL": study_profile(run).role("embeddings").model}
    updates |= {"APE_S7_TARGETS": run.s7_targets_path, TEST_SPLIT_ENV: None, TEST_SEED_BASE_ENV: None, SESSION_CHECKPOINTS_ENV: run.session_checkpoints_dir}
    updates[LABEL_ENV] = f"{run.study}{'-offline' if run.offline else ''}/{run.run_id}"
    if run.offline:
        updates[REGISTRY_ENV] = run.work_dir / "spend_registry.jsonl"
    updates |= kg
    with rg._environ(updates):
        yield


def _fingerprint(inputs: dict, params: dict, upstream: dict) -> str:
    return rg._fingerprint(inputs, params, upstream)


def phase_state(run: StudyRun, name: str) -> dict:
    phase = PHASE_DEFS[run.study][name]
    inputs = {k: rg._file_entry(p) for k, p in phase.inputs(run).items()}
    params = json.loads(json.dumps(phase.params(run), default=str))
    upstream = {u: (rg.read_manifest(run, u) or {}).get("fingerprint") for u in phase.upstream}
    return {"inputs": inputs, "params": params, "upstream": upstream, "fingerprint": _fingerprint(inputs, params, upstream)}


def run_phase(run: StudyRun, name: str) -> str:
    """Run one phase (inside `run_environment`); "done" or "skipped", raising when it fails or is refused. The same
    steps as `run_gate.run_phase`: skip a complete phase whose fingerprint and outputs stand, the phase's refusal, its
    prerequisites, the stray-environment check, the record, the budget guard (program and study allocation), the
    body, the offline check, the backup."""
    phase = PHASE_DEFS[run.study][name]
    _check_mode(run)
    state = phase_state(run, name)
    inputs, params, upstream, fingerprint = state["inputs"], state["params"], state["upstream"], state["fingerprint"]
    old = rg.read_manifest(run, name)
    history = list((old or {}).get("history") or [])
    missing = rg._missing_outputs(old) if old else []
    if old and old.get("status") in COMPLETE and old.get("fingerprint") == fingerprint and not missing and not run.force:
        old |= {"status": "skipped", "skipped_at": rg._now(), "skip_reason": f"done at {old.get('finished')} with the same inputs (fingerprint {fingerprint[:12]})"}
        old["history"] = [*history, {"action": "skip", "at": old["skipped_at"]}]
        rg._write_json(run.manifest_path(name), old)
        print(f"[{name}] skipped: done at {old.get('finished')} with the same inputs", flush=True)
        return "skipped"
    if phase.refuse and (reason := phase.refuse(run)):
        raise PhaseError(reason)
    for req in phase.requires:
        if not rg.is_complete(run, req):
            raise PhaseError(f"{name} needs {req} first: python -m ape.run_study {req} --study {run.study} --run-id {run.run_id}{' --offline' if run.offline else ''}")
    if name not in ("preflight", "freeze", "analyze") and (stray := stray_environment(run)):
        raise PhaseError(f"{name}: the environment would move this live run off the design or its directories; unset: " + "; ".join(stray))
    profile = phase.profile(run)
    record: dict[str, Any] = {
        "phase": name,
        "study": run.study,
        "run_id": run.run_id,
        "status": "running",
        "offline": run.offline,
        "started": rg._now(),
        "finished": None,
        "git": rg.git_state(),
        "profile": {"name": profile.name, "roles": profile.summary(), "concurrency": asdict(profile.concurrency)}
        | ({"model_swap": f"{rg.MOCK} for every Inspect role (efforts and sampling settings kept)"} if run.offline else {}),
        "inputs": inputs,
        "budget_inputs": rg.budget_inputs(run),
        "params": params,
        "upstream": upstream,
        "fingerprint": fingerprint,
        "operational_env": rg._operational_env() | ({SESSION_CHECKPOINTS_ENV: os.environ[SESSION_CHECKPOINTS_ENV]} if os.environ.get(SESSION_CHECKPOINTS_ENV) else {}),
        "kg": {k: v for k, v in kg_resolution(run).items() if k != "notes"},
        "projected_usd": None,
        "spend_at_start": None,
        "spend": None,
        "outputs": {},
        "log_dirs": [],
        "warnings": [],
        "errors": [],
        "history": [*history, {"action": "run", "at": rg._now(), "forced": bool(run.force and old and old.get("status") in COMPLETE), "fingerprint": fingerprint}],
    }
    if missing and old.get("status") in COMPLETE:
        record["warnings"].append(f"re-run because recorded output(s) are missing: {missing}")
    if knobs := [k for k in env_knobs(run) if not k.startswith("APE_BUILD_")]:
        record["warnings"].append(f"APE_* knob(s) set in the environment apply to every arm of this phase: {knobs}")
    rg._write_json(run.manifest_path(name), record)
    print(f"[{name}] running ({run.study}{', offline' if run.offline else ''})", flush=True)
    try:
        record["projected_usd"] = round(phase.projected(run), 4)
        record["projection_basis"] = "run_plan.yaml cells, conservative" + ("; offline runs spend $0, priced at live sizes" if run.offline else "")
        if run.budget_usd is not None:
            record["budget_basis"] = f"run budget ${run.budget_usd:,.2f} (not the plan's budget.total_usd)"
        record["spend_at_start"] = spend(run)
        if phase.remaining is not None:
            record["projected_remaining_usd"] = round(phase.remaining(run), 4)
        else:
            record["resume_credit_usd"] = round(rg.prior_attempt_spend(run, name, old, fingerprint), 4)
            record["projected_remaining_usd"] = round(max(0.0, record["projected_usd"] - record["resume_credit_usd"]), 4)
        rg._write_json(run.manifest_path(name), record)
        s = record["spend_at_start"]
        require_affordable(record["projected_remaining_usd"], min(s["remaining_usd"], s["study_remaining_usd"]), f"phase {name} ({run.study}: the smaller of the program's and the study's allocation's remainder)")
        phase.body(run, record)
        if run.offline:
            record["offline_check"] = rg.verify_offline(run)
    except BaseException as e:
        record |= {"status": "failed", "finished": rg._now()}
        record["errors"].append(f"{type(e).__name__}: {e}")
        with contextlib.suppress(Exception):
            record["spend"] = spend(run)
        rg._write_json(run.manifest_path(name), record)
        rg.backup_after_phase(run, name, record)
        raise
    record |= {"status": "done", "finished": rg._now(), "spend": spend(run)}
    rg._write_json(run.manifest_path(name), record)
    rg.backup_after_phase(run, name, record)
    for w in record["warnings"]:
        print(f"[{name}] WARNING: {w}", flush=True)
    print(f"[{name}] done (spent so far ${record['spend']['spent_usd']:,.2f} of ${record['spend']['budget_usd']:,.0f}; {run.study} ${record['spend']['study_spent_usd']:,.2f} of ${record['spend']['study_allocation_usd']:,.0f})", flush=True)
    return "done"


def run_phases(run: StudyRun, phase: str) -> dict[str, str]:
    """`phase` (or every phase of the study for "all", in order, stopping at the first failure)."""
    names = run.spec.phases if phase == "all" else (phase,)
    if unknown := [n for n in names if n not in PHASE_DEFS[run.study]]:
        raise PhaseError(f"unknown phase(s) {unknown} for {run.study}; its phases are {list(run.spec.phases)} or all")
    settle_gate_run(run)
    statuses: dict[str, str] = {}
    with run_environment(run):
        for name in names:
            try:
                statuses[name] = run_phase(run, name)
            except Exception:
                if phase == "all" and name == "test" and rg.read_manifest(run, "test") is not None:
                    try:
                        run_phase(run, "analyze")
                    except Exception as e:  # noqa: BLE001  (the test's failure is the one to raise)
                        print(f"[analyze] after the failed test: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                raise
    return statuses


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ape.run_study", description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("phase", choices=[*ALL_PHASES, "all"])
    ap.add_argument("--study", required=True, choices=sorted(STUDIES))
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--offline", action="store_true", help="zero-spend end-to-end run: mock models, fake embeddings, oracle indices, tiny sizes")
    ap.add_argument("--force", action="store_true", help="re-run complete phases even when their inputs are unchanged")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="where run directories live (a run is <runs-dir>/<study>/<id>/)")
    ap.add_argument("--config-dir", type=Path, default=ROOT / "config", help="read config inputs from here (offline runs and tests only)")
    ap.add_argument("--budget-usd", type=float, help="lower the guard's program budget for this run (never above the plan's)")
    ap.add_argument("--gate-run-id", metavar="RUN_ID", help="the frozen, analysed gate run whose verdict and selection fix the KG arm (S5, M1k, M2)")
    ap.add_argument("--gate-runs-dir", type=Path, help="where the gate run lives (default: --runs-dir)")
    ap.add_argument("--test-seed-base", type=int, metavar="SEED", help=f"freeze only: the first test seed (a block of {SEED_BLOCK}) inside the study's test range")
    ap.add_argument("--extension-of", metavar="RUN_ID", help="freeze only: this run extends an earlier frozen run of the study (fresh test worlds)")
    ap.add_argument("--skip-smoke-check", metavar="REASON", help="live runs: run without a fresh passing live smoke, recording why")
    ap.add_argument("--smoke-max-age-days", type=float, default=rg.SMOKE_MAX_AGE_DAYS, metavar="DAYS")
    a = ap.parse_args(argv)
    if a.phase != "all" and a.phase not in STUDIES[a.study].phases:
        ap.error(f"{a.study} has no {a.phase} phase; its phases are {', '.join(STUDIES[a.study].phases)}")
    if a.skip_smoke_check is not None and a.offline:
        ap.error("--skip-smoke-check applies to live runs (offline runs never check the live smoke)")
    if (a.test_seed_base is not None or a.extension_of) and a.phase not in ("freeze", "all"):
        ap.error("--test-seed-base / --extension-of apply to the freeze phase")
    try:
        run = StudyRun(
            a.study, a.run_id, offline=a.offline, force=a.force, runs_root=a.runs_dir, config_dir=a.config_dir, budget_usd=a.budget_usd,
            test_seed_base=a.test_seed_base, extension_of=a.extension_of, gate_run_id=a.gate_run_id, gate_runs_root=a.gate_runs_dir,
            skip_smoke_check=a.skip_smoke_check, smoke_max_age_days=a.smoke_max_age_days,
        )  # fmt: skip
        statuses = run_phases(run, a.phase)
    except (PhaseError, PreflightError, BudgetError) as e:
        print(f"run_study: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001  (recorded in the phase manifest; the CLI reports and exits non-zero)
        traceback.print_exc()
        print(f"run_study: phase failed; see {a.runs_dir / a.study / a.run_id}/<phase>/manifest.json", file=sys.stderr)
        return 1
    print(json.dumps({"study": run.study, "run_id": run.run_id, "offline": run.offline, "phases": statuses, "dir": _show(run.dir)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
