"""Gate orchestrator (FIX_PLAN FX-6): the gate as named phases, each idempotent, resumable and recorded.

    uv run python -m ape.run_gate <phase> --run-id <id> [--offline] [--force]

Phases, in order (`all` runs them in this order and stops at the first failure):

    preflight   FX-2 preflight (prices, and live: OPENAI_API_KEY) for the gate and anchor profiles; live:
                cache/openai_probe.json lists every model the profiles call; the installed apg-core is the
                pinned commit recorded in PROVENANCE.md; a dirty git tree is a warning.
    build-dev   dev worlds (see `dev_world_specs`) and their artifacts through `ape.artifacts`: chunk
                embeddings, the authored APG graph, the LightRAG index (extract; offline: oracle).
    tune        `ape.tuning.tune` for APG, LightRAG and S3s on the grid's dev cells -> tune/selected.yaml
                and <config>/selected.yaml.
    anchor      PC1: the GraphRAG-Bench index, hybrid and naive runs, `pc1_from_logs` -> anchor/pc1.json.

Part B (pilot, freeze, build-test, test, analyze) appends phases to `PHASES`.

Run directory (`runs/<id>/`, git-ignored):

    run.json                 run id, mode (offline or live), created; a run never switches mode
    <phase>/manifest.json    one per phase (fields below)
    build-dev/worlds.json    every dev world: id, path, group, family, level, style, tasks, artifact status
    tune/selected.yaml       the selection (format below); also written to <config>/selected.yaml
    tune/tuning_log.jsonl    every candidate tried and each system's selection (PC6); a re-run archives the old one
    tune/logs/<system>/<candidate>-<hash>/   Inspect logs + runner_index.json, one dir per candidate
    anchor/logs/             Inspect logs of the hybrid and naive runs + runner_index.json
    anchor/pc1.json          `pc1_from_logs` plus the logs, profile and sizes it came from
    work/                    offline only: worlds/, indices/, cache/ and config/ (the outputs live mode
                             writes to config/), so an offline run never touches live artifacts

Manifest fields: phase, run_id, status (running | done | failed | skipped), offline, started, finished,
git {commit, dirty}, profile {name, roles (`Profile.summary()`), concurrency, model_swap (offline)},
inputs {key: {path, sha256}} (the config files that determine the phase's outputs), budget_inputs (the
files the budget guard read), params (scale and other non-file inputs), upstream {phase: fingerprint},
fingerprint, projected_usd, spend_at_start and spend (`ape.budget.remaining`: Inspect cost over every
log in the run's runner indexes + the ledger, against the plan's budget.total_usd), outputs, log_dirs,
warnings, errors, history, plus phase-specific results (checks, selected, pc1_pass, offline_check, ...).

Idempotence. A phase is complete when its status is `done` or `skipped`. A complete phase whose
fingerprint (sha256 over its input file hashes, params and upstream fingerprints) is unchanged and
whose recorded outputs still exist is skipped: its manifest is marked `skipped` (outputs and `finished` kept, `skipped_at` added) unless
`--force`. A `failed` or `running` manifest (a crash) re-runs; the work underneath resumes, since
`ape.runner` reuses finished eval logs and `ape.artifacts` skips current artifacts.

Budget guard. Before a phase runs, its projected cost (`ape.budget.projected_cost`, conservative, over
the plan cells the phase runs, adjusted to what it actually runs) must fit in budget.total_usd ($5,000)
minus the spend so far (`require_affordable`); otherwise the phase is refused and its manifest is
`failed`. Offline runs apply the same guard with the live projection (they spend $0).

Offline mode (`--offline`, zero spend, for end-to-end tests): `mockllm` agent (`mock_agent`), kg
(`mock_kg`), anchor answerer and judge (`ape.anchor.fixture`), all built from the profile so efforts
and sampling settings are kept; fake embeddings; `perfect_author` APG graphs; oracle LightRAG indices,
so the LightRAG candidates run as their oracle twins (LGR-s -> LGRo-s, recorded as `declared_arm`);
the synthetic anchor dataset of `ape.anchor.fixture`; and `OFFLINE_SCALE`. OPENAI_API_KEY is replaced
by a sentinel and OPENAI_BASE_URL points at a closed local port for the whole run, and after each phase
every eval log's models must be mockllm and the ledger empty (`offline_check`).

selected.yaml (part B reads it; `read_selected`):

    APG*: {arm: APG-s, env: {APE_APG_SHORTLIST_K: "24"}, candidate: apg-s-k24, mean_success: 0.71, cost_usd: 1.9}
    LGR*: {arm: LGR-s, env: {APE_LGR_MODE: mix}, candidate: lgr-s-mix, mean_success: 0.69, cost_usd: 2.3}
    S3s:  {arm: S3s, env: {APE_S3S_BUDGET: "2000"}, candidate: s3s-2000, mean_success: 0.6, cost_usd: 1.1}

`env` holds only per-arm knobs (APE_APG_*, APE_LGR_*, APE_S3S_BUDGET), so the three can be applied together.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .budget import BudgetError, Plan, PlanCell, Prices, load_assumptions, load_measured, load_plan, projected_cost, remaining, require_affordable
from .config import ROOT, Config, embedding_cache
from .models import PreflightError, Profile, load_profile, preflight
from .runner import INDEX_NAME

STUDY = "gate"
PHASES = ("preflight", "build-dev", "tune", "anchor")  # part B appends: pilot, freeze, build-test, test, analyze
COMPLETE = ("done", "skipped")
APG_PIN = "d47f7f3749dac38e9f917429810136e4ec6ebb37"  # PROVENANCE.md: tag apg-eval-baseline
MOCK = "mockllm/model"
OFFLINE_KEY = "sk-ape-offline-run-no-network"
OFFLINE_BASE_URL = "http://127.0.0.1:9/v1"  # the discard port: a stray OpenAI call fails at once, locally
PAPER_MODEL = "gpt-4o-mini"
ANCHOR_MODES = ("hybrid", "naive")  # PC1: hybrid against naive (GATE_PREREG §7)
DEFAULT_N_PER_TYPE = 200
TUNED_SYSTEMS = (("APG", "APG*"), ("LightRAG", "LGR*"), ("S3s", "S3s"))  # tuning_grid system -> selected.yaml key
OFFLINE_ARMS = {"LGR-q": "LGRo-q", "LGR-s": "LGRo-s"}  # offline has no LLM extraction: oracle indices instead
# Tiny and explicit; recorded in every offline manifest's params.
OFFLINE_SCALE = {
    "worlds_per_cell": 1,  # every dev world group (gate cells, F5, id_only, messy)
    "tasks_per_world": 2,
    "epochs": 1,
    "tune_candidates_per_system": 2,  # the first N declared candidates of each system
    "anchor_n_per_type": 2,  # of the fixture's 5 per type
}
# Dev world groups and their run_plan.yaml build cells. The plan is the one source of counts, so the
# projection and the build cover the same worlds. World i of every group has seed 1000+i, so the id_only
# and messy worlds are paired renderings of gate worlds.
DEV_BUILD_CELLS = {"gate": "gate.build.dev", "id_only": "gate.build.dev-id-only", "messy": "gate.build.dev-messy", "f5": "gate.build.dev-f5"}
# Environment variables the orchestrator manages itself; every other APE_* knob is recorded in params.
MANAGED_ENV = ("APE_WORLDS", "APE_INDICES", "APE_CACHE", "APE_EMBEDDINGS", "APE_EMBEDDING_MODEL", "APE_MODEL_PROFILE")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class PhaseError(RuntimeError):
    """A phase cannot start (missing prerequisite, wrong mode) or its work failed; the message says what to do."""


# --- The run ---------------------------------------------------------------------------------------


@dataclass
class GateRun:
    """One gate run: its id, mode and where it reads and writes.

    `config_dir` is where config inputs are read (tests point it at a copy). Outputs that live runs write
    to config/ (selected.yaml; part B: s7_targets.json, budget_calibration.yaml) go to `out_config_dir`:
    config_dir when live, `runs/<id>/work/config/` offline.
    """

    run_id: str
    offline: bool = False
    force: bool = False
    runs_root: Path = ROOT / "runs"
    config_dir: Path = ROOT / "config"
    env_path: Path = ROOT / ".env"
    probe_path: Path = ROOT / "cache" / "openai_probe.json"
    provenance_path: Path = ROOT / "PROVENANCE.md"

    def __post_init__(self) -> None:
        if not _RUN_ID.match(self.run_id):
            raise PhaseError(f"run id {self.run_id!r}: use letters, digits, '.', '_' and '-' only")
        self.runs_root, self.config_dir = Path(self.runs_root), Path(self.config_dir)
        if not self.offline and self.config_dir.resolve() != (ROOT / "config").resolve():
            raise PhaseError("--config-dir is for offline runs and tests; live runs read config/ (build workers load config/models.yaml)")

    @property
    def dir(self) -> Path:
        return self.runs_root / self.run_id

    def phase_dir(self, phase: str) -> Path:
        return self.dir / phase

    def manifest_path(self, phase: str) -> Path:
        return self.phase_dir(phase) / "manifest.json"

    @property
    def work_dir(self) -> Path:
        return self.dir / "work"

    @property
    def out_config_dir(self) -> Path:
        return self.work_dir / "config" if self.offline else self.config_dir

    @property
    def selected_path(self) -> Path:
        return self.out_config_dir / "selected.yaml"

    def config(self, name: str) -> Path:
        return self.config_dir / name

    @property
    def costs_path(self) -> Path:
        return self.config("model_costs.yaml")

    @property
    def models_path(self) -> Path:
        return self.config("models.yaml")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _show(path: Path) -> str:
    """A path as recorded in manifests: relative to the repo when inside it."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=False, default=str))
    os.replace(tmp, path)


def read_manifest(run: GateRun, phase: str) -> dict | None:
    path = run.manifest_path(phase)
    return json.loads(path.read_text()) if path.is_file() else None


def is_complete(run: GateRun, phase: str) -> bool:
    m = read_manifest(run, phase)
    return m is not None and m.get("status") in COMPLETE


def read_selected(path: Path) -> dict:
    """selected.yaml as written by the tune phase: {APG*: {arm, env, ...}, LGR*: {...}, S3s: {...}}."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if missing := [k for _, k in TUNED_SYSTEMS if k not in data]:
        raise PhaseError(f"{path}: no selection for {missing}; run the tune phase")
    return data


def git_state() -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


@contextlib.contextmanager
def _environ(updates: dict[str, str | Path | None]) -> Iterator[None]:
    """Set (or, for None, remove) environment variables for the duration; restore them after."""
    old = {k: os.environ.get(k) for k in updates}
    for k, v in updates.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def run_environment(run: GateRun) -> Iterator[None]:
    """Offline: every path under runs/<id>/work, fake embeddings, and an OpenAI key that cannot reach anything.
    Live: the embedding model from the gate profile unless APE_EMBEDDING_MODEL is set (as the CLIs do)."""
    if run.offline:
        w = run.work_dir
        updates: dict[str, str | Path | None] = {
            "APE_WORLDS": w / "worlds",
            "APE_INDICES": w / "indices",
            "APE_CACHE": w / "cache",
            "APE_EMBEDDINGS": "fake",
            "APE_MODEL_PROFILE": None,
            "APE_BUILD_MODEL": None,
            "APE_BUILD_EFFORT": None,
            "APE_BUILD_FALLBACK": None,
            "OPENAI_API_KEY": OFFLINE_KEY,
            "OPENAI_BASE_URL": OFFLINE_BASE_URL,
        }
    else:
        updates = {} if os.environ.get("APE_EMBEDDING_MODEL") else {"APE_EMBEDDING_MODEL": gate_profile(run).role("embeddings").model}
    with _environ(updates):
        yield


def _check_mode(run: GateRun) -> None:
    """A run is offline or live for its whole life: an offline `done` must never satisfy a live phase."""
    path = run.dir / "run.json"
    if path.is_file():
        info = json.loads(path.read_text())
        if bool(info.get("offline")) != run.offline:
            mode = "an offline" if info.get("offline") else "a live"
            raise PhaseError(f"run {run.run_id!r} is {mode} run ({path}); use another --run-id")
        return
    _write_json(path, {"run_id": run.run_id, "offline": run.offline, "created": _now(), "config_dir": _show(run.config_dir), "git": git_state()})


# --- Plan, profiles, scale, budget ----------------------------------------------------------------


def plan(run: GateRun) -> Plan:
    return load_plan(run.config("run_plan.yaml"))


def gate_profile(run: GateRun) -> Profile:
    return load_profile(plan(run).cell("gate.tune").spec["profile"], run.models_path)


def read_probe(run: GateRun) -> dict:
    if not run.probe_path.is_file():
        raise PhaseError(f"{_show(run.probe_path)} missing: run readiness E2-E4 first ({probe_command(run)})")
    try:
        return json.loads(run.probe_path.read_text())
    except ValueError as e:
        raise PhaseError(f"{_show(run.probe_path)} does not parse ({e}): re-run {probe_command(run)}") from None


def probe_command(run: GateRun) -> str:
    try:
        p = gate_profile(run)
        bare = {r: p.role(r).model.split("/", 1)[-1] for r in ("agent", "kg", "build", "embeddings")}
        return f"uv run python readiness/probe_openai.py --list --agent {bare['agent']} --kg {bare['kg']} --build {bare['build']} --embed {bare['embeddings']}"
    except (OSError, ValueError, BudgetError):
        return "uv run python readiness/probe_openai.py --list --agent <id> --kg <id> --build <id> --embed <id>"


def anchor_profile_name(run: GateRun) -> tuple[str, str]:
    """(profile, reason): the paper-faithful `anchor` (gpt-4o-mini) unless the probe shows it is not
    available, then `anchor_luna` (D-009, E3). Offline always uses `anchor` (its settings on mock models)."""
    if run.offline:
        return "anchor", "offline: the paper's settings on mock models"
    available = set(read_probe(run).get("models_available") or [])
    if PAPER_MODEL in available:
        return "anchor", f"{PAPER_MODEL} is available (probe)"
    return "anchor_luna", f"{PAPER_MODEL} is not in the probe's models_available (E3): GPT-6 Luna fallback, ±10 pp"


def scale(run: GateRun) -> dict:
    """The sizes this run uses: OFFLINE_SCALE offline, else run_plan.yaml's gate cells."""
    if run.offline:
        return {"offline": True, **OFFLINE_SCALE}
    p = plan(run)
    tune = p.cell("gate.tune").spec
    runs = p.cell("gate.anchor.runs").spec
    return {
        "offline": False,
        "tasks_per_world": int(tune["tasks_per_world"]),
        "tune_worlds": int(tune["worlds"]),
        "epochs": int(tune.get("epochs", 1)),
        "anchor_n_per_type": int(runs.get("n_per_type", DEFAULT_N_PER_TYPE)),
    }


def dev_world_specs(run: GateRun) -> list[dict]:
    """Every dev world group the gate needs, from its run_plan.yaml build cell (DEV_BUILD_CELLS): world
    counts and exception style from the cell; tasks per world from `gate.tune` tasks_per_world, except
    messy worlds, which split `gate.sec.messy` n_tasks per cell over the cell's worlds."""
    p = plan(run)
    tasks = int(p.cell("gate.tune").spec["tasks_per_world"])
    messy_n = int(p.cell("gate.sec.messy").spec["n_tasks"])
    specs = []
    for group, cell_id in DEV_BUILD_CELLS.items():
        cell = p.cell(cell_id).spec
        for world_cell, count in cell["worlds"].items():
            family, level = world_cell.split("-", 1)
            n_tasks = math.ceil(messy_n / count) if group == "messy" else tasks
            if run.offline:
                count, n_tasks = OFFLINE_SCALE["worlds_per_cell"], OFFLINE_SCALE["tasks_per_world"]
            style = cell.get("exception_style", "descriptive")
            specs.append({"group": group, "family": family, "level": level, "count": int(count), "n_tasks": int(n_tasks), "exception_style": style, "relational": True})
    return specs


def _cost_kwargs(run: GateRun) -> dict:
    return {
        "assumptions": load_assumptions(run.config("budget_assumptions.yaml")),
        "prices": Prices.load(run.costs_path),
        "measured": load_measured(run.config("budget_calibration_measured.yaml")),
        "models_path": run.models_path,
    }


def project(run: GateRun, cells: Sequence[PlanCell]) -> float:
    """Conservative $ for these plan cells (plan cells, or ad-hoc ones for what the plan does not list)."""
    return projected_cost(plan=Plan(tuple(cells), {}, {}, ()), **_cost_kwargs(run))


def budget_inputs(run: GateRun) -> dict:
    names = ("run_plan.yaml", "budget_assumptions.yaml", "model_costs.yaml", "models.yaml", "budget_calibration_measured.yaml")
    return {f"config/{n}": {"path": _show(run.config(n)), "sha256": _sha256(run.config(n))} for n in names if run.config(n).exists()}


def run_log_files(run: GateRun) -> list[str]:
    """Every final eval log of this run, from the runner indexes under its directory."""
    files = set()
    for index in run.dir.rglob(INDEX_NAME):
        for entry in json.loads(index.read_text()).get("tasks", {}).values():
            if entry.get("log") and Path(entry["log"]).is_file():
                files.add(entry["log"])
    return sorted(files)


def spend(run: GateRun) -> dict:
    """What is spent and left of the plan's budget: this run's Inspect logs plus the build/embedding ledger."""
    return remaining(float(plan(run).budget["total_usd"]), run_log_files(run), Config().ledger_path, run.costs_path)


def verify_offline(run: GateRun) -> dict:
    """Offline runs must not have called any real model: every log's models mockllm, the ledger empty."""
    from .llm.ledger import Ledger

    models: set[str] = set()
    for index in run.dir.rglob(INDEX_NAME):
        for entry in json.loads(index.read_text()).get("tasks", {}).values():
            models.add(str(entry.get("model")))
            models.update(str(m) for m in (entry.get("model_roles") or {}).values())
    ledger = Config().ledger_path
    entries = Ledger(ledger).read() if ledger.is_file() else []
    problems = [f"eval logs used non-mock model(s) {sorted(m for m in models if not m.startswith('mockllm/'))}"] if any(not m.startswith("mockllm/") for m in models) else []
    if entries:
        problems.append(f"the ledger {ledger} has {len(entries)} paid build/embedding call(s)")
    if os.environ.get("OPENAI_API_KEY") != OFFLINE_KEY or os.environ.get("OPENAI_BASE_URL") != OFFLINE_BASE_URL:
        problems.append("the offline OpenAI sentinel was replaced during the phase")
    if problems:
        raise PhaseError("offline run made real calls: " + "; ".join(problems))
    return {"models": sorted(models), "ledger_entries": 0, "openai": "sentinel key, closed local base URL"}


def _env_knobs() -> dict[str, str]:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("APE_") and k not in MANAGED_ENV}


def _result_knobs() -> list[str]:
    """APE_* knobs in the environment that change what arms do (build concurrency knobs do not)."""
    return [k for k in _env_knobs() if not k.startswith("APE_BUILD_")]


def _missing_outputs(manifest: dict) -> list[str]:
    """The recorded outputs that no longer exist (a complete phase whose outputs are gone re-runs)."""
    missing = []
    for key, shown in (manifest.get("outputs") or {}).items():
        path = Path(shown) if Path(shown).is_absolute() else ROOT / shown
        if not path.exists():
            missing.append(key)
    return missing


def _mock_models(profile: Profile, roles: dict[str, Callable]) -> tuple[Any, dict[str, Any]]:
    """The profile's agent and roles as mockllm models with the given scripts; efforts and settings are kept."""
    from .models import agent_model, role_models

    agent = agent_model(profile, model=MOCK, custom_outputs=roles.pop("agent"))
    return agent, {r: role_models(profile, (r,), model=MOCK, custom_outputs=fn)[r] for r, fn in roles.items()}


# --- Phases ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    name: str
    body: Callable[[GateRun, dict], None]  # does the work; adds outputs, log_dirs, warnings, results to the record
    inputs: Callable[[GateRun], dict[str, Path]]  # key -> config file that determines the outputs
    params: Callable[[GateRun], dict]  # non-file inputs (scale, profile, options); part of the fingerprint
    projected: Callable[[GateRun], float]
    profile: Callable[[GateRun], Profile]
    requires: tuple[str, ...] = ()
    upstream: tuple[str, ...] = ()  # phases whose fingerprint is part of this one's


def _cfg_inputs(run: GateRun, *names: str) -> dict[str, Path]:
    return {f"config/{n}": run.config(n) for n in names}


# preflight -----------------------------------------------------------------------------------------


def _apg_installed_commit() -> str | None:
    try:
        raw = importlib.metadata.distribution("apg-core").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return ((json.loads(raw) if raw else {}).get("vcs_info") or {}).get("commit_id")


def _preflight_params(run: GateRun) -> dict:
    # The environment facts preflight checks, so a new commit, another apg-core or a new key re-runs it.
    env = {"git": git_state(), "apg_core_commit": _apg_installed_commit()}
    if not run.offline:
        from .models import _api_key_present

        env["api_key_present"] = _api_key_present(run.env_path)
        env["probe_sha256"] = _sha256(run.probe_path)
    return {"scale": scale(run), "environment": env, "env_knobs": _env_knobs()}


def _preflight(run: GateRun, record: dict) -> None:
    problems: list[str] = []
    checks: dict[str, Any] = {}
    gp = gate_profile(run)
    try:
        anchor_name, why = anchor_profile_name(run)
    except PhaseError as e:
        problems.append(str(e))
        anchor_name, why = "anchor", "probe unreadable; checked the paper profile"
    checks["anchor_profile"] = {"name": anchor_name, "reason": why}
    ap = load_profile(anchor_name, run.models_path)
    record["anchor_profile"] = {"name": ap.name, "roles": ap.summary()}
    # 1. FX-2: prices for every model both profiles call; live: an API key.
    fx2 = [x for p in (gp, ap) for x in preflight(p, live=not run.offline, costs_path=run.costs_path, env_path=run.env_path)]
    checks["fx2_preflight"] = "ok" if not fx2 else "failed (see errors)"
    problems += fx2
    # 2. Live: the readiness probe lists every model the run calls.
    if run.offline:
        checks["probe"] = "skipped (offline)"
    elif run.probe_path.is_file():
        try:
            available = set(read_probe(run).get("models_available") or [])
        except PhaseError as e:
            problems.append(str(e))
            available = None
        if available is not None:
            missing = sorted(
                {f"{p.name}.{r}: {s.model}" for p in (gp, ap) for r, s in p.roles.items() if r != "build_fallback" and s.model.split("/", 1)[-1] not in available}
            )
            if missing:
                problems.append(f"{_show(run.probe_path)} does not list {missing}: re-run {probe_command(run)}, or fix config/models.yaml")
            fallback = gp.roles.get("build_fallback")
            if fallback and fallback.model.split("/", 1)[-1] not in available:
                record["warnings"].append(f"the D-017 fallback builder {fallback.model} is not in the probe's models_available")
            checks["probe"] = {"path": _show(run.probe_path), "sha256": _sha256(run.probe_path), "missing": missing}
    # 3. The installed APG is the pinned commit, and PROVENANCE.md records that pin.
    installed = _apg_installed_commit()
    checks["apg_core"] = {"installed_commit": installed, "pinned_commit": APG_PIN}
    if installed != APG_PIN:
        problems.append(f"installed apg-core is at {installed!r}, not the pinned {APG_PIN} (PROVENANCE.md); run `uv sync`")
    provenance = run.provenance_path.read_text() if run.provenance_path.is_file() else ""
    if APG_PIN not in provenance:
        problems.append(f"{_show(run.provenance_path)} does not record the APG pin {APG_PIN}")
    # 4. A dirty tree is a warning: results must trace to a commit.
    if record["git"].get("dirty"):
        record["warnings"].append("the git tree is dirty: commit before a live run so results trace to a commit")
    record["checks"] = checks
    if problems:
        raise PreflightError("preflight failed:\n" + "\n".join(f"  - {x}" for x in dict.fromkeys(problems)))


# build-dev -----------------------------------------------------------------------------------------


def _build_dev_projected(run: GateRun) -> float:
    p = plan(run)
    return project(run, [p.cell(cell_id) for cell_id in DEV_BUILD_CELLS.values()])


def _build_dev(run: GateRun, record: dict) -> None:
    from .artifacts import KINDS, build_artifacts
    from .worlds.generate import make_world
    from .worlds.spec import World

    cfg = Config()
    specs = dev_world_specs(run)
    worlds: list[dict] = []
    for s in specs:
        for i in range(s["count"]):
            w = make_world(s["family"], s["level"], "dev", i, s["n_tasks"], s["relational"], s["exception_style"])
            path = cfg.world_path(w.id)
            if not path.is_file() or World.load(path).content_hash() != w.content_hash():
                w.save(path)  # unchanged worlds keep their file, so their artifacts stay current
            worlds.append({"world_id": w.id, "path": _show(path), "group": s["group"], "family": s["family"], "level": s["level"], "exception_style": s["exception_style"], "n_tasks": s["n_tasks"]})
    lightrag_kind = "oracle" if run.offline else "extract"
    if run.offline:
        updates: dict[str, str | None] = {}
    else:
        from .models import require_preflight

        gp = gate_profile(run)
        require_preflight(gp, live=True, costs_path=run.costs_path, env_path=run.env_path)
        updates = {"APE_MODEL_PROFILE": gp.name}  # build workers read the build model from this profile
    print(f"[build-dev] {len(worlds)} dev world(s): building {','.join(KINDS)} (lightrag {lightrag_kind}{', fake author' if run.offline else ''})", flush=True)
    with _environ(updates):
        results = build_artifacts([cfg.world_path(w["world_id"]) for w in worlds], KINDS, lightrag_kind, run.offline, on_result=lambda r: print(r.summary(), flush=True))
    for w, r in zip(worlds, results, strict=True):
        w["artifacts"], w["errors"] = r.kinds, r.errors
    out = run.phase_dir("build-dev") / "worlds.json"
    _write_json(out, {"worlds_dir": _show(cfg.worlds_dir), "indices_dir": _show(cfg.indices_dir), "lightrag_kind": lightrag_kind, "fake_author": run.offline, "worlds": worlds})
    record["outputs"] = {"worlds": _show(out), "worlds_dir": _show(cfg.worlds_dir), "indices_dir": _show(cfg.indices_dir)}
    record["worlds"] = {"count": len(worlds), "by_group": {g: sum(w["group"] == g for w in worlds) for g in dict.fromkeys(s["group"] for s in specs)}}
    if failed := [r for r in results if r.status == "failed"]:
        raise PhaseError(f"{len(failed)} dev world(s) failed to build (re-run to resume):\n" + "\n".join(r.summary() for r in failed))


# tune ----------------------------------------------------------------------------------------------


def tuning_grid(run: GateRun) -> dict:
    """config/tuning_grid.yaml; offline: the first N candidates per system, LightRAG arms as their oracle twins."""
    from .tuning import load_grid

    grid = load_grid(run.config("tuning_grid.yaml"))
    if not run.offline:
        return grid
    systems = {}
    for name, sdef in grid["systems"].items():
        cands = []
        for c in sdef["candidates"][: OFFLINE_SCALE["tune_candidates_per_system"]]:
            c = dict(c)
            if c.get("arm") in OFFLINE_ARMS:
                c["declared_arm"], c["arm"] = c["arm"], OFFLINE_ARMS[c["arm"]]
            cands.append(c)
        systems[name] = {**sdef, "candidates": cands}
    return {**grid, "systems": systems}


def _tune_params(run: GateRun) -> dict:
    sc = scale(run)
    return {
        "scale": sc,
        "systems": [s for s, _ in TUNED_SYSTEMS],
        "limit_worlds": sc["worlds_per_cell"] if run.offline else sc["tune_worlds"],
        "epochs": sc["epochs"],
        "offline_arms": OFFLINE_ARMS if run.offline else None,
        "env_knobs": _env_knobs(),
    }


def _tune(run: GateRun, record: dict) -> None:
    from .tuning import tune

    params = _tune_params(run)
    grid = tuning_grid(run)
    gp = gate_profile(run)
    if run.offline:
        from .llm.mock_agent import mock_agent, mock_kg

        agent, roles = _mock_models(gp, {"agent": mock_agent, "kg": mock_kg})
    else:
        from .models import agent_model, require_preflight, role_models

        require_preflight(gp, live=True, costs_path=run.costs_path, env_path=run.env_path)
        agent, roles = agent_model(gp), role_models(gp, ("kg",))
    tdir = run.phase_dir("tune")
    log_dir, tlog = tdir / "logs", tdir / "tuning_log.jsonl"
    if tlog.exists():  # a re-run rewrites the log from the (reused) candidate runs; the old one is kept
        tlog.rename(tdir / f"tuning_log.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.jsonl")
    record["outputs"] = {"tuning_log": _show(tlog)}
    record["log_dirs"] = [_show(log_dir)]
    selected: dict[str, dict] = {}
    for system, key in TUNED_SYSTEMS:
        print(f"[tune] {system}: {len(grid['systems'][system]['candidates'])} candidate(s) x {grid['dev_cells']}", flush=True)
        chosen = tune(system, agent, dict(roles), params["limit_worlds"], params["epochs"], grid, tlog, gp, log_dir=log_dir, costs_path=run.costs_path)
        cand = chosen["candidate"]
        entry = {"arm": cand["arm"], "env": {k: str(v) for k, v in (cand.get("env") or {}).items()}}
        entry |= {k: cand[k] for k in ("delivery", "declared_arm") if k in cand}
        entry |= {"candidate": cand["id"], "mean_success": chosen["mean_success"], "cost_usd": chosen["cost_usd"]}
        selected[key] = entry
    header = (
        f"# Dev-tuned selections (GATE_PREREG §5), written by `python -m ape.run_gate tune --run-id {run.run_id}`"
        f"{' (OFFLINE: mock models, oracle LightRAG)' if run.offline else ''}.\n"
        f"# Tuning log: {_show(tlog)}. Each `env` holds only that arm's knobs; part B applies them together.\n"
    )
    text = header + yaml.safe_dump(selected, sort_keys=False)
    for path in (tdir / "selected.yaml", run.selected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    record["outputs"] |= {"selected": _show(tdir / "selected.yaml"), "selected_config": _show(run.selected_path)}
    record["selected"] = selected


# anchor --------------------------------------------------------------------------------------------


def _anchor_cells(run: GateRun, profile: str) -> list[PlanCell]:
    """The plan's anchor cells as this run executes them: its profile and ANCHOR_MODES."""
    p = plan(run)
    index, runs = p.cell("gate.anchor.index"), p.cell("gate.anchor.runs")
    return [
        PlanCell(index.id, index.study, index.phase, {**index.spec, "profile": profile}),
        PlanCell(runs.id, runs.study, runs.phase, {**runs.spec, "profile": profile, "modes": list(ANCHOR_MODES)}),
    ]


def _anchor_params(run: GateRun) -> dict:
    name, _ = anchor_profile_name(run)
    return {"scale": scale(run), "profile": name, "modes": list(ANCHOR_MODES), "data": "fixture" if run.offline else "cache/graphragbench", "env_knobs": _env_knobs()}


def _anchor(run: GateRun, record: dict) -> None:
    import asyncio

    from inspect_ai.log import read_eval_log

    from .anchor import graphragbench as gb
    from .lgr.common import MANIFEST, read_manifest as read_index_manifest
    from .runner import log_path, run_evals
    from .tasks.anchor_graphragbench import graphragbench_anchor

    cfg = Config()
    name, why = anchor_profile_name(run)
    profile = load_profile(name, run.models_path)
    n_per_type = scale(run)["anchor_n_per_type"]
    adir = run.phase_dir("anchor")
    if run.offline:
        from .anchor import fixture

        data = fixture.write_dataset(adir / "data")
    else:
        from .models import require_preflight

        require_preflight(profile, live=True, costs_path=run.costs_path, env_path=run.env_path)
        data = gb.default_data_dir(cfg)
    try:
        bench = gb.load_medical(data, 0, 0)
    except FileNotFoundError as e:
        raise PhaseError(f"anchor data: {e}. Fetch GraphRAG-Bench @ fdbab59 (Datasets/) into {_show(data)}") from None
    # The index: built once per corpus and build model.
    wd = gb.working_dir(cfg)
    if run.offline:
        build_model, build_effort, llm = None, None, fixture.fake_extractor()
    else:
        from .llm.build_client import BuildLlm
        from .llm.ledger import Ledger
        from .models import build_settings

        build_model, build_effort = build_settings(profile)
        llm = BuildLlm(build_model, Ledger(cfg.ledger_path), {"anchor": gb.ANCHOR_ID, "system": "lightrag"}, reasoning_effort=build_effort).lightrag_func()
    emb_model = embedding_cache(cfg).model
    if (wd / MANIFEST).is_file():
        m = read_index_manifest(wd)
        have = (m.get("corpus_hash"), m.get("build_model"), m.get("build_effort"), m.get("embedding_model"))
        if have != (bench.corpus_hash(), build_model, build_effort, emb_model):
            raise PhaseError(f"{_show(wd)} holds an anchor index of another corpus or build (build {have[1]}@{have[2]}, embeddings {have[3]}); move it aside to rebuild")
        record["index"] = {"path": _show(wd), "status": "current", "index_hash": m.get("index_hash")}
    else:
        print(f"[anchor] building the index ({build_model or 'fake extractor'})", flush=True)
        m = asyncio.run(gb.build_index(bench, cfg, llm, build_model, build_effort))
        record["index"] = {"path": _show(wd), "status": "built", "index_hash": m.get("index_hash")}
    # The runs: one eval set, hybrid and naive.
    tasks = [graphragbench_anchor(mode=mode, n_per_type=n_per_type, data=str(data)) for mode in ANCHOR_MODES]
    log_dir = adir / "logs"
    record["log_dirs"] = [_show(log_dir)]
    if run.offline:
        from .anchor.fixture import mock_answerer, mock_judge

        agent, roles = _mock_models(profile, {"agent": mock_answerer, "judge": mock_judge})
        success, logs = run_evals(tasks, log_dir, profile=profile, model=agent, model_roles=roles, live=False, costs_path=run.costs_path, display="none", log_dir_allow_dirty=True)
    else:
        success, logs = run_evals(tasks, log_dir, profile=profile, roles=("judge",), live=True, costs_path=run.costs_path, log_dir_allow_dirty=True)
    by_mode = {h.eval.task_args.get("mode"): h for h in logs}
    if not success or any(by_mode[m].status != "success" for m in ANCHOR_MODES):
        errors = [f"{m}: {by_mode[m].error.message if by_mode[m].error else by_mode[m].status}" for m in ANCHOR_MODES if m in by_mode and by_mode[m].status != "success"]
        raise PhaseError("anchor runs did not finish (re-run to resume): " + "; ".join(errors or ["eval set failed"]))
    graph, naive = (read_eval_log(log_path(by_mode[m]), header_only=True) for m in ANCHOR_MODES)
    pc1 = gb.pc1_from_logs(graph, naive)
    pc1 |= {
        "logs": {m: log_path(by_mode[m]) for m in ANCHOR_MODES},
        "profile": name,
        "profile_reason": why,
        "n_per_type": n_per_type,
        "samples": {m: by_mode[m].results.total_samples if by_mode[m].results else None for m in ANCHOR_MODES},
        "offline": run.offline,
    }
    out = adir / "pc1.json"
    _write_json(out, pc1)
    record["outputs"] = {"pc1": _show(out), "index": _show(wd)}
    record["pc1_pass"] = pc1["pass"]
    if not pc1["pass"]:
        record["warnings"].append("PC1 does not pass (see pc1.json): fix and re-pilot before the test phase (GATE_PREREG §7)")


PHASE_DEFS: dict[str, Phase] = {
    "preflight": Phase(
        "preflight",
        _preflight,
        inputs=lambda r: {**_cfg_inputs(r, "models.yaml", "model_costs.yaml", "run_plan.yaml"), "PROVENANCE.md": r.provenance_path},
        params=_preflight_params,
        projected=lambda r: 0.0,
        profile=gate_profile,
    ),
    "build-dev": Phase(
        "build-dev",
        _build_dev,
        inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml"),
        params=lambda r: {"scale": scale(r), "worlds": dev_world_specs(r), "lightrag_kind": "oracle" if r.offline else "extract", "fake_author": r.offline},
        projected=_build_dev_projected,
        profile=gate_profile,
        requires=("preflight",),
    ),
    "tune": Phase(
        "tune",
        _tune,
        inputs=lambda r: _cfg_inputs(r, "tuning_grid.yaml", "run_plan.yaml", "models.yaml", "model_costs.yaml"),
        params=_tune_params,
        projected=lambda r: project(r, [plan(r).cell("gate.tune")]),
        profile=gate_profile,
        requires=("preflight", "build-dev"),
        upstream=("build-dev",),
    ),
    "anchor": Phase(
        "anchor",
        _anchor,
        inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml"),
        params=_anchor_params,
        projected=lambda r: project(r, _anchor_cells(r, anchor_profile_name(r)[0])),
        profile=lambda r: load_profile(anchor_profile_name(r)[0], r.models_path),
        requires=("preflight",),
    ),
}
assert tuple(PHASE_DEFS) == PHASES


# --- Running phases --------------------------------------------------------------------------------


def _fingerprint(inputs: dict, params: dict, upstream: dict) -> str:
    key = {"inputs": {k: v["sha256"] for k, v in inputs.items()}, "params": params, "upstream": upstream}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def run_phase(run: GateRun, name: str) -> str:
    """Run one phase (inside `run_environment`); returns "done" or "skipped", raises when it fails or is refused."""
    phase = PHASE_DEFS[name]
    _check_mode(run)
    inputs = {k: {"path": _show(p), "sha256": _sha256(p)} for k, p in phase.inputs(run).items()}
    params = phase.params(run)
    upstream = {u: (read_manifest(run, u) or {}).get("fingerprint") for u in phase.upstream}
    fingerprint = _fingerprint(inputs, params, upstream)
    old = read_manifest(run, name)
    history = list((old or {}).get("history") or [])
    missing = _missing_outputs(old) if old else []
    if old and old.get("status") in COMPLETE and old.get("fingerprint") == fingerprint and not missing and not run.force:
        old |= {"status": "skipped", "skipped_at": _now(), "skip_reason": f"done at {old.get('finished')} with the same inputs (fingerprint {fingerprint[:12]})"}
        old["history"] = [*history, {"action": "skip", "at": old["skipped_at"]}]
        _write_json(run.manifest_path(name), old)
        print(f"[{name}] skipped: done at {old.get('finished')} with the same inputs", flush=True)
        return "skipped"
    for req in phase.requires:
        if not is_complete(run, req):
            raise PhaseError(f"{name} needs {req} first: python -m ape.run_gate {req} --run-id {run.run_id}{' --offline' if run.offline else ''}")
    profile = phase.profile(run)
    record: dict[str, Any] = {
        "phase": name,
        "run_id": run.run_id,
        "status": "running",
        "offline": run.offline,
        "started": _now(),
        "finished": None,
        "git": git_state(),
        "profile": {"name": profile.name, "roles": profile.summary(), "concurrency": asdict(profile.concurrency)}
        | ({"model_swap": f"{MOCK} for every Inspect role (efforts and sampling settings kept)"} if run.offline else {}),
        "inputs": inputs,
        "budget_inputs": budget_inputs(run),
        "params": params,
        "upstream": upstream,
        "fingerprint": fingerprint,
        "projected_usd": None,
        "spend_at_start": None,
        "spend": None,
        "outputs": {},
        "log_dirs": [],
        "warnings": [],
        "errors": [],
        "history": [*history, {"action": "run", "at": _now(), "forced": bool(run.force and old and old.get("status") in COMPLETE)}],
    }
    if missing and old.get("status") in COMPLETE:
        record["warnings"].append(f"re-run because recorded output(s) are missing: {missing}")
    if knobs := _result_knobs():
        record["warnings"].append(f"APE_* knob(s) set in the environment apply to every arm of this phase: {knobs}")
    _write_json(run.manifest_path(name), record)
    print(f"[{name}] running{' (offline)' if run.offline else ''}", flush=True)
    try:
        record["projected_usd"] = round(phase.projected(run), 4)
        record["projection_basis"] = "run_plan.yaml cells, conservative" + ("; live sizes (offline runs spend $0)" if run.offline else "")
        record["spend_at_start"] = spend(run)
        _write_json(run.manifest_path(name), record)
        require_affordable(record["projected_usd"], record["spend_at_start"]["remaining_usd"], f"phase {name}")
        phase.body(run, record)
        if run.offline:
            record["offline_check"] = verify_offline(run)
    except BaseException as e:
        record |= {"status": "failed", "finished": _now()}
        record["errors"].append(f"{type(e).__name__}: {e}")
        with contextlib.suppress(Exception):
            record["spend"] = spend(run)
        _write_json(run.manifest_path(name), record)
        raise
    record |= {"status": "done", "finished": _now(), "spend": spend(run)}
    _write_json(run.manifest_path(name), record)
    for w in record["warnings"]:
        print(f"[{name}] WARNING: {w}", flush=True)
    print(f"[{name}] done (spent so far ${record['spend']['spent_usd']:,.2f} of ${record['spend']['budget_usd']:,.0f})", flush=True)
    return "done"


def run_phases(run: GateRun, phase: str) -> dict[str, str]:
    """`phase` (or every phase for "all", in order, stopping at the first failure); phase -> "done" | "skipped"."""
    names = PHASES if phase == "all" else (phase,)
    if unknown := [n for n in names if n not in PHASE_DEFS]:
        raise PhaseError(f"unknown phase(s) {unknown}; phases are {list(PHASES)} or all")
    statuses: dict[str, str] = {}
    with run_environment(run):
        for name in names:
            statuses[name] = run_phase(run, name)
    return statuses


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ape.run_gate", description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("phase", choices=[*PHASES, "all"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--offline", action="store_true", help="zero-spend end-to-end run: mock models, fake embeddings, oracle indices, tiny sizes")
    ap.add_argument("--force", action="store_true", help="re-run complete phases even when their inputs are unchanged")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="where run directories live (default: runs/)")
    ap.add_argument("--config-dir", type=Path, default=ROOT / "config", help="read config inputs from here (offline runs and tests only)")
    a = ap.parse_args(argv)
    try:
        run = GateRun(a.run_id, offline=a.offline, force=a.force, runs_root=a.runs_dir, config_dir=a.config_dir)
        statuses = run_phases(run, a.phase)
    except (PhaseError, PreflightError, BudgetError) as e:
        print(f"run_gate: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001  (recorded in the phase manifest; the CLI reports and exits non-zero)
        traceback.print_exc()
        print(f"run_gate: phase failed; see {a.runs_dir / a.run_id}/<phase>/manifest.json", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": run.run_id, "offline": run.offline, "phases": statuses, "dir": _show(run.dir)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
