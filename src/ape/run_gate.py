"""Gate orchestrator (FIX_PLAN FX-6): the gate as named phases, each idempotent, resumable and recorded.

    uv run --locked python -m ape.run_gate <phase> --run-id <id> [--offline] [--smoke] [--force] [--budget-usd N]

`--locked`: uv never rewrites uv.lock under a run (the freeze hashes it).

Phases, in order (`all` runs them in this order and stops at the first failure):

    preflight   FX-2 preflight (prices, and live: OPENAI_API_KEY) for the gate and anchor profiles; live:
                cache/openai_probe.json lists every model the profiles call; the installed apg-core is the
                pinned commit recorded in PROVENANCE.md; a dirty git tree is a warning. A live run (not offline,
                not smoke) also needs a passing live smoke of this code (`check_live_smoke`: readiness/smoke.py's
                record in cache/smoke/live/), settled once per run in run.json (`_settle_smoke`: a later preflight
                reuses it while the code is the smoked code, without re-checking its age), unless
                `--skip-smoke-check "<reason>"`; and storage warnings (`ape.models.storage_warnings`: APE_BACKUP_DIR,
                free disk). Its fingerprint follows the code (`code_state`), not the commit, so a run's documented
                steps (the pre-registration's commit, the freeze's PROVENANCE.md record) do not re-run it.
    build-dev  dev worlds (see `dev_world_specs`) and their artifacts through `ape.artifacts`: chunk
                embeddings, the authored APG graph, the LightRAG index (extract; offline: oracle); then
                D-017's build-quality check (`ape.build_quality`) -> build-dev/build_quality.json. Its params name
                the builder (`build_params`, as pilot's and build-test's do), so switching to D-017's fallback
                (APE_BUILD_FALLBACK=1) re-runs the builds and every phase after them.
    tune      `ape.tuning.tune` for APG, LightRAG and S3s on the grid's dev cells -> tune/selected.yaml
                and the run's config/selected.yaml. Live, it refuses until config/tuning_grid.yaml names every
                system's owner and each owner has signed off its candidates (`tuning_signoff_problems`).
    anchor      PC1: the GraphRAG-Bench index (always the paper's builder, never D-017's fallback), hybrid and
                naive runs, `pc1_from_logs` -> anchor/pc1.json. The runs' logs are keyed by the anchor's code
                (`anchor_code_hash`), so after a harness fix the anchor scores afresh.
    pilot       the pilot worlds (`gate.build.pilot`, split "pilot") and their artifacts; `gate.pilot` arms but
                S7 and `gate.pilot.pull`, with every selection's knobs; S7 targets (APG*'s median realized
                context per cell) -> the run's config/s7_targets.json, then S7; `gate.pilot.sigma` (APG*, LGR* on the
                pilot worlds after the main pilot's, push; D-026); the matched-budget calibration of
                LGR* and S3s (`calibrate_caps`) -> the run's config/budget_calibration.yaml; the NI power
                re-simulation with σ_w, σ_g from the main and σ cells' worlds and its two-sided recommendation
                (suffices / insufficient / ambiguous) -> pilot/power.json; the cost model recalibrated
                (`ape.budget.calibrate`) -> the run's config/budget_calibration_measured.yaml (the run's later
                projections use it; promoting it to config/ is a deliberate copy); pilot/pilot.json.
    freeze      GATE_PREREG.md's body has no `[PILOT` / `[USER` marker left, the tree is committed, PC1
                passes (or its failure is accepted with a diagnosis) and every phase it rests on is current
                (`stale_upstream`: a re-run of build-dev, tune, anchor and pilot would each be a skip); offline: a
                rehearsal on a filled copy, with warnings. Then the sha256 of the pre-registration, the
                FROZEN_CONFIG files, the run's FROZEN_OUTPUTS and the FROZEN_CODE files, the commit (`code_commit`),
                the APG pin, the design knobs tune and pilot ran with (`design_env`) and the run's test-seed block
                (`choose_test_seed_base`) -> freeze.json, then PROVENANCE.md (a freeze interrupted between the two
                is completed by the next `freeze`: `_complete_freeze`). `require_frozen` is the guard build-test
                and test call.
    build-test  the test worlds (split "test", the run's frozen seed block; `test_world_specs`) and their
                artifacts, from the frozen plan's TEST_BUILD_CELLS. Only this phase may generate the test split:
                it sets TEST_SPLIT_ENV (and TEST_SEED_BASE_ENV) after its freeze guard, and every other generator
                refuses (`worlds.generate`).
    test        every enabled run cell of the frozen plan's TEST_RUN_PHASES (`test_groups`), one eval set per
                cell and environment, PRIMARY_CELLS first; the budget is re-checked before each group, a
                failing group is recorded and the others still run (the phase then fails; a re-run resumes).
    analyze     the decision report (FX-7, `ape.analyze_gate`): PC1-PC6, the verdict per delivery mode and
                combined, the NO-GO diagnosis, tables and secondaries -> report/decision.json and report.md.
                It needs no other phase complete and reports what exists (a missing input fails its
                precondition with the reason), so it also runs after a failed or budget-stopped test: `all`
                runs it then too, before reporting the test's failure.

`all` runs every phase through analyze.

Freeze. A run is frozen once: after freeze.json exists, `tune`, `pilot` and `anchor` refuse to run (even with
--force; they would rewrite frozen inputs or pc1.json), and `freeze` re-runs only as a skip. `build-test` and
`test` take the frozen files as inputs and refuse (naming the file) unless every one still matches its hash,
and, live, unless CODE_PATHS (src/, power/, uv.lock) are exactly the freeze commit's (`code_drift`; offline
rehearsals only warn), and unless the design knobs are the frozen `design_env` (`design_env_changes`). Any change
after the freeze is a logged deviation (GATE_PREREG.md, Deviations log) and a new --run-id.

Test seeds (RELIABILITY_REVIEW S6). Each frozen run owns one block of SEED_BLOCK test seeds: the first run 3000,
a later one (an INCONCLUSIVE extension, a NO-GO fix cycle) the next block no other frozen run used, or
`--test-seed-base` (refused if it overlaps one). Used blocks come from runs/*/freeze.json and PROVENANCE.md's
`<!-- ape:test-seeds ... -->` markers. The test tasks of a run past the first block select only that block
(`tasks.gate.gate_samples`), since every run's test worlds share worlds/test/. Offline rehearsals use
OFFLINE_TEST_SEED_BASE, so they never generate the real test worlds.

Environment. Operational settings (OPERATIONAL_ENV: APE_BACKUP_DIR, build concurrency) are recorded in each
manifest but never enter a fingerprint. A live, full-size run refuses a paid phase (PAID_PHASES) in a stray
environment (`stray_environment`: moved worlds/cache/indices, fake embeddings, another embedding model or
profile, a builder override) instead of silently overriding a value someone set; the test split's lock and seed
base are always cleared (`run_environment`), since only build-test may set them.

Run directory (`runs/<id>/`, git-ignored):

    run.json                 run id, mode (offline or live), created; a run never switches mode. `smoke_check`: the
                             live smoke the run settled at its first preflight (commits, checks), or its override
    config/                  live: the run's outputs (selected.yaml, s7_targets.json, budget_calibration.yaml,
                             budget_calibration_measured.yaml); the freeze hashes the first three. The repo's
                             config/ holds templates and inputs only, so runs never overwrite each other's outputs
    <phase>/manifest.json    one per phase (fields below)
    build-dev/worlds.json    every dev world: id, path, group, family, level, style, tasks, artifact status
    build-dev/build_quality.json  D-017: APG and LightRAG ID coverage per world and gate cell, the verdict and the
                             builder recommendation (offline: marked offline, coverage 1.0 by construction)
    tune/selected.yaml       the selection (format below); also written to the run's config/selected.yaml
    tune/tuning_log.jsonl    every candidate tried and each system's selection (PC6); a re-run archives the old one
    tune/logs/<system>/<candidate>-<hash>/   Inspect logs + runner_index.json, one dir per candidate
    anchor/logs-<code>/      Inspect logs of the hybrid and naive runs + runner_index.json; <code> `anchor_code_hash`
    anchor/pc1.json          `pc1_from_logs` plus the logs, profile and sizes it came from
    pilot/worlds.json        every pilot world, as build-dev/worlds.json
    pilot/logs/main-<hash>/  the selected arms and diagnostics (push, pull); <hash> of the selections' knobs
    pilot/logs/s7-<hash>/    S7, sized by the targets in <hash>
    pilot/logs/sigma-<hash>/ the σ cell: APG*, LGR* on pilot worlds 5-8 (offline: the second world), push
    pilot/budget-cal/        one log dir per calibration iteration, and calibration.json (every iteration); an arm
                             that never lands in the window keeps its closest caps and the file says `matched:
                             false` (`not_converged` lists the arms): the matched-budget secondary still runs and is
                             reported as not matched, never failing the pilot
    pilot/power.json         variance components and NI power at POWER_SIZES (`ape.analysis.pilot`)
    pilot/pilot.json         the summary the analyst transcribes; `prereg_items` is keyed by placeholder label
    freeze.json              frozen files {key: {path, sha256}}, git, code_commit, analysis commit, APG pin,
                             rehearsal, phases (the fingerprints it rests on), test_seeds {base, count, block, last}
    build-test/worlds.json   every test world, as build-dev/worlds.json (groups test, id_only, f5)
    test/<cell>/<group>-<hash>/   one eval set: Inspect logs + runner_index.json; <cell> the plan cell id,
                             <group> its env group, <hash> of the group's env and arms (knobs are not part of
                             Inspect's task identity, so a new env never reuses another env's logs)
    report/decision.json     the decision report (`ape.analyze_gate`), and report/report.md
    work/                    offline and smoke only: worlds/, indices/, cache/ and config/ (the outputs a live
                             run writes to its config/), GATE_PREREG.md (the rehearsal copy) and
                             PROVENANCE.freeze.md, so an offline run never touches live artifacts

Manifest fields: phase, run_id, status (running | done | failed | skipped), offline, started, finished,
git {commit, dirty}, profile {name, roles (`Profile.summary()`), concurrency, model_swap (offline)},
inputs {key: {path, sha256}} (the config files that determine the phase's outputs), budget_inputs (the
files the budget guard read), params (scale and other non-file inputs), upstream {phase: fingerprint},
fingerprint, projected_usd, spend_at_start and spend (`spend()`: the whole program's spend from the spend
registry, `ape.budget.program_remaining`, against the plan's budget.total_usd or a lower --budget-usd, plus
this run's own `run_spent_usd`: every eval log under runs/<id>/, each sample once, + the ledger), outputs, log_dirs,
warnings, errors, history, plus phase-specific results (checks, build_quality, selected, pc1_pass, budget_calibration,
s7_targets, power, placeholders, frozen, offline_check, ...).

Idempotence. A phase is complete when its status is `done` or `skipped`. A complete phase whose
fingerprint (sha256 over its input file hashes, params and upstream fingerprints) is unchanged and
whose recorded outputs still exist is skipped: its manifest is marked `skipped` (outputs and `finished` kept, `skipped_at` added) unless
`--force`. A `failed` or `running` manifest (a crash) re-runs; the work underneath resumes, since
`ape.runner` reuses finished eval logs and `ape.artifacts` skips current artifacts. Before a phase runs
(after the skip check), its `refuse` check may stop it without touching its manifest (the freeze rules).

Budget guard. Before a phase runs, the projected cost of its remaining work must fit in budget.total_usd
($5,000) minus the whole program's spend so far (`require_affordable`); otherwise the phase is refused and its
manifest is `failed`. `projected_usd` is the phase's whole conservative projection (`ape.budget.projected_cost`
over the plan cells it runs, adjusted to what it actually runs); `projected_remaining_usd`, which the guard
checks, subtracts what earlier attempts with the same fingerprint already spent (`resume_credit_usd`,
`prior_attempt_spend`: a resume reuses their finished logs and artifacts, whose spend is already counted). The
test phase guards its PRIMARY_CELLS groups' remaining work at the start and every other group before it runs
(finished groups count 0), so a short budget stops secondaries and leaves the verdict's evidence complete.
- **Spend so far** comes from the program spend registry (`ape.spend`). It covers every run id, smoke run and
  later study, plus killed runs' flushed samples.
- **Offline runs** apply the same guard with the live projection (they spend $0) against their own registry
  under work/.
- **Per-sample guard.** Every eval set also gets a per-sample `cost_limit` (`ape.runner`). Each run_gate
  call passes its cell's conservative projection per sample (`sample_cost_usd`), so the limit is 20x that, at
  least $0.50.

Offline mode (`--offline`, zero spend, for end-to-end tests): `mockllm` agent (`mock_agent`), kg
(`mock_kg`), anchor answerer and judge (`ape.anchor.fixture`), all built from the profile so efforts
and sampling settings are kept; fake embeddings; `perfect_author` APG graphs; oracle LightRAG indices,
so the LightRAG candidates run as their oracle twins (LGR-s -> LGRo-s, recorded as `declared_arm`);
the synthetic anchor dataset of `ape.anchor.fixture`; `OFFLINE_SCALE`; and a rehearsal freeze. OPENAI_API_KEY is replaced
by a sentinel and OPENAI_BASE_URL points at a closed local port for the whole run, and after each phase
every eval log's models must be mockllm and the ledger empty (`offline_check`).

Smoke mode (`--smoke`, FIX_PLAN FX-8; `readiness/smoke.py` drives it): the live wiring at `SMOKE_SCALE`, i.e.
OFFLINE_SCALE's sizes restricted to the cheapest gate cells (F7-10, F3-5): live models (or, with
`--offline`, the offline mocks at the same scale), its own worlds/, indices/, cache/ and config outputs
under runs/<id>/work/ (default runs dir: cache/smoke/runs/), the GraphRAG-Bench data read from
cache/graphragbench, and projections priced at smoke sizes. `budget_usd` (smoke.py passes what is left
of its --max-usd) replaces the plan's budget in the guard. A smoke run never freezes (the freeze refuses,
listing the pre-registration's open items too), so it never reaches build-test or test, and it never
writes config/ or PROVENANCE.md. A run is smoke or not for its whole life (run.json).

selected.yaml (pilot and later phases read it; `read_selected`):

    APG*: {arm: APG-s, env: {APE_APG_SHORTLIST_K: "24"}, candidate: apg-s-k24, mean_success: 0.71, cost_usd: 1.9}
    LGR*: {arm: LGR-s, env: {APE_LGR_MODE: mix}, candidate: lgr-s-mix, mean_success: 0.69, cost_usd: 2.3}
    S3s:  {arm: S3s, env: {APE_S3S_BUDGET: "2000"}, candidate: s3s-2000, mean_success: 0.6, cost_usd: 1.1}

`env` holds only per-arm knobs (APE_APG_*, APE_LGR_*, APE_S3S_BUDGET), so the three can be applied together.

Test runs (the contract FX-7's analysis reads). Groups (`test_groups`): `selected` (the cell's arms under every
selection's knobs together), `matched` (a cell with `context`: the selected knobs, then budget_calibration.yaml's
on top), and arm variants (ARM_VARIANTS: `lgr-naive` is LGR* with APE_LGR_MODE=naive). World selection: a
cell's `split` (default test) and `exception_style` (default descriptive) name its worlds; a `worlds:` cell
uses that many per task cell, an `n_tasks:` cell the first ceil(n_tasks / tasks per world) (whole worlds, the
same seeds as the primary's; offline: OFFLINE_SCALE). test/manifest.json adds:

    primary_complete: bool                   every PRIMARY_CELLS cell is done
    cells: {<plan cell id>: {                in run order
        status: done | failed | stopped (budget) | pending,
        primary: bool,
        groups: [{name, arms: [{declared, run}], split, cells, deliveries, exposure, exception_style,
                  epochs, n_worlds: {task cell: n}, env, log_dir, log_files, status, projected_usd,
                  projected_remaining_usd, calibration? ({matched, not_converged}: the `matched` group only),
                  seed_base? (a run past the first test-seed block), error?}]}}

A group that runs S7 carries APE_S7_PER_STEP in its env: S7 follows APG*'s delivery schedule (D-024, `s7_env`).

Every test-phase task also records, in its eval metadata: arm (as run), delivery, exposure, split,
exception_style, plan_cell, group and knobs (the APE_* arm knobs it ran under, `tasks.gate`).
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

from .budget import BudgetError, Plan, PlanCell, Prices, load_assumptions, load_measured, load_plan, program_remaining, projected_cost, remaining, require_affordable
from .build_quality import worlds_per_cell_note
from .config import ROOT, Config, embedding_cache
from .models import PreflightError, Profile, load_profile, preflight, storage_warnings
from .runner import INDEX_NAME
from .spend import LABEL_ENV, REGISTRY_ENV
from .worlds.generate import SEED_BLOCK, SPLIT_SEED_BASE, TEST_SEED_BASE_ENV, TEST_SPLIT_ENV, require_test_split_unlocked

STUDY = "gate"
PHASES = ("preflight", "build-dev", "tune", "anchor", "pilot", "freeze", "build-test", "test", "analyze")
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
# Live smoke scale (FIX_PLAN FX-8): OFFLINE_SCALE's sizes on the cheapest gate cells only. Plan cells of other
# world cells are dropped (an anchor index keeps its corpus); see `_smoke_cell` for the projection.
SMOKE_SCALE = {**OFFLINE_SCALE, "cells": ("F7-10", "F3-5")}
# readiness/smoke.py keeps dry and live state apart: cache/smoke/dry/ and cache/smoke/live/ (worlds, indices, cache,
# logs, the orchestrator's runs, report.json; live also checks.json, the latest result of every check).
SMOKE_ROOT = ROOT / "cache" / "smoke"
SMOKE_RUNS = SMOKE_ROOT / "live" / "runs"
SMOKE_DRY_RUNS = SMOKE_ROOT / "dry" / "runs"
# A live gate run needs a passing live smoke no older than this (`check_live_smoke`; --smoke-max-age-days).
SMOKE_MAX_AGE_DAYS = 7.0
SMOKE_OK = ("pass", "warn")  # smoke statuses that count as passing (a warn-level check is a finding, not a breakage)
# Dev world groups and their run_plan.yaml build cells. The plan is the one source of counts, so the
# projection and the build cover the same worlds. World i of every group has seed 1000+i, so the id_only
# and messy worlds are paired renderings of gate worlds.
DEV_BUILD_CELLS = {"gate": "gate.build.dev", "id_only": "gate.build.dev-id-only", "messy": "gate.build.dev-messy", "f5": "gate.build.dev-f5"}
# The pilot (GATE_PREREG §4): its world build cell and its run cells.
PILOT_BUILD_CELL = "gate.build.pilot"
PILOT_RUN_CELLS = ("gate.pilot", "gate.pilot.pull", "gate.pilot.budget-cal")
# D-026: APG* and LGR* on the pilot worlds after the main pilot's (world_offset), push only, so σ_w and σ_g are
# estimated from twice as many worlds per cell. Optional: a plan without it estimates σ from gate.pilot alone.
PILOT_SIGMA_CELL = "gate.pilot.sigma"
# Matched-budget calibration (GATE_PREREG §6.2, PC4): each capped arm's knobs are scaled together by one factor,
# searched until the arm's median realized context is within CAL_TOLERANCE of the cell's `context`.
CAL_KNOBS = {"LGR*": ("APE_LGR_BUDGET", "APE_LGR_ENTITY_TOKENS", "APE_LGR_RELATION_TOKENS", "APE_LGR_TOTAL_TOKENS"), "S3s": ("APE_S3S_BUDGET",)}
CAL_MAX_ITERATIONS = 3
CAL_TOLERANCE = 0.25
CAL_STEP = (0.1, 10.0)  # bounds on one step's scale change
APG_MATCHED_SHORTLIST_K = 48  # APG fills instead of being capped: the largest shortlistK in the tuning grid
POWER_SIZES = (12, 16, 20, 24)  # test worlds per cell the pilot power re-simulation compares (D-017; 20/24 for "insufficient", D-026)
POWER_SIMS = {"live": 2000, "offline": 200}
# Frozen at `freeze` (FIX_PLAN FX-6): config inputs read from config_dir, and the outputs tune and pilot write.
FROZEN_CONFIG = ("models.yaml", "model_costs.yaml", "run_plan.yaml", "tuning_grid.yaml")
FROZEN_OUTPUTS = ("selected.yaml", "s7_targets.json", "budget_calibration.yaml")
# Code the verdict depends on, frozen with the design (GATE_PREREG §2: "decide at the frozen commit"): the analysis
# code and the dependency lockfile. A change after the freeze is a logged deviation, like a config change.
FROZEN_CODE = ("uv.lock", "src/ape/analyze_gate.py", "src/ape/analysis/gate_stats.py", "src/ape/analysis/cost.py", "src/ape/analysis/pilot.py")
# Everything the results depend on that git tracks. After the freeze, build-test and test refuse to run unless these
# are exactly the frozen commit's (`code_drift`): scorers, generators, the agent loop, the adapters, S7, the lockfile.
CODE_PATHS = ("src", "power", "uv.lock")
PLACEHOLDER = re.compile(r"\[(PILOT|USER)\b")  # any marker in GATE_PREREG.md's body blocks the freeze
PLACEHOLDER_ITEM = re.compile(r"\[(PILOT|USER)(?::\s*([^\]\n]*))?\]")
PC1_REASON_MIN_CHARS = 40  # an acceptance states the diagnosis, not just "ok"
DEVIATION = "any change after the freeze is a logged deviation (GATE_PREREG.md, Deviations log): record it there and start a new --run-id"
# The test split (GATE_PREREG §4), built and run only on a frozen run: world groups and their plan build cells.
TEST_BUILD_CELLS = {"test": "gate.build.test", "id_only": "gate.build.test-id-only", "f5": "gate.build.test-f5"}
# The test phase runs every enabled run cell of these plan phases, the verdict's evidence first: the GO rule's cells
# (GATE_PREREG §2, §8) and the cells PC3 (gate.diag) and PC2 (gate.f5) need, so a budget stop only loses secondaries.
TEST_RUN_PHASES = ("test", "diagnostics", "f5", "secondaries")
PRIMARY_CELLS = ("gate.test.f7", "gate.test.f3", "gate.diag.s7", "gate.diag", "gate.f5")
# Plan arms that run as another arm under extra knobs: LightRAG naive (PC2) is LGR* in naive mode.
ARM_VARIANTS = {"LGR-naive": ("LGR*", {"APE_LGR_MODE": "naive"})}
# Test seeds (S6). Each gate run's test worlds are one block of SEED_BLOCK seeds from its frozen base: the first run
# 3000 (SPLIT_SEED_BASE), a later run (an INCONCLUSIVE extension, a NO-GO fix cycle) the next unused block, so
# already-analysed worlds are never reused. Offline rehearsals use their own block and never touch a real one.
FIRST_TEST_SEED_BASE = SPLIT_SEED_BASE["test"]
OFFLINE_TEST_SEED_BASE = 9000
TEST_SEED_MARKER = re.compile(r"<!-- ape:test-seeds run=(\S+) base=(\d+) count=(\d+) -->")
# Environment variables the orchestrator manages itself; every other APE_* knob is recorded in params.
MANAGED_ENV = (
    "APE_WORLDS", "APE_INDICES", "APE_CACHE", "APE_EMBEDDINGS", "APE_EMBEDDING_MODEL", "APE_MODEL_PROFILE", "APE_S7_TARGETS",
    TEST_SPLIT_ENV, TEST_SEED_BASE_ENV, REGISTRY_ENV, LABEL_ENV,
)  # fmt: skip
# Operational settings change how a run executes (backups, build concurrency), never what it measures. They stay out
# of phase fingerprints (toggling a backup must not re-run, or after a freeze block, a phase) and the knob warning.
OPERATIONAL_ENV = ("APE_BACKUP_DIR", "APE_BUILD_LLM_CONCURRENCY", "APE_BUILD_PARALLEL_INSERT", "APE_BUILD_EMBED_CONCURRENCY", "APE_BUILD_WORKERS")
# Phases that call models or build artifacts: a live run refuses them in a stray environment (`stray_environment`).
PAID_PHASES = ("build-dev", "tune", "anchor", "pilot", "build-test", "test")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


class PhaseError(RuntimeError):
    """A phase cannot start (missing prerequisite, wrong mode) or its work failed; the message says what to do."""


# --- The run ---------------------------------------------------------------------------------------


@dataclass
class GateRun:
    """One gate run: its id, mode and where it reads and writes.

    `config_dir` is where config inputs are read (tests point it at a copy); config/ holds templates and inputs
    only. The outputs tune and pilot write (selected.yaml, s7_targets.json, budget_calibration.yaml,
    budget_calibration_measured.yaml) go to `out_config_dir`, the run's own: `runs/<id>/config/` live,
    `runs/<id>/work/config/` offline and smoke. So a second live run (an extension, a fix cycle) never rewrites
    a first run's frozen outputs, and a re-analysis reads its own run's selection. `prereg_path` is the
    pre-registration the freeze checks and hashes (tests point it at a copy).
    """

    run_id: str
    offline: bool = False
    force: bool = False
    runs_root: Path = ROOT / "runs"
    config_dir: Path = ROOT / "config"
    env_path: Path = ROOT / ".env"
    probe_path: Path = ROOT / "cache" / "openai_probe.json"
    provenance_path: Path = ROOT / "PROVENANCE.md"
    prereg_path: Path = ROOT / "GATE_PREREG.md"
    smoke: bool = False
    budget_usd: float | None = None  # lowers the guard's budget below the plan's budget.total_usd (smoke: what --max-usd leaves)
    anchor_data_dir: Path | None = None  # GraphRAG-Bench data; default <cache>/graphragbench (smoke: the repo's cache/)
    accept_pc1_failure: str | None = None  # freeze only: the analyst's recorded reason for freezing despite a failed PC1 (GATE_PREREG §7)
    test_seed_base: int | None = None  # freeze only: the test-seed block to freeze (default 3000, else the next unused block)
    # freeze only: what this gate run is (GATE_PREREG §8, D-027): the primary gate run, the one extension of an
    # INCONCLUSIVE run, or the one fix cycle of a NO-GO run. Recorded in freeze.json, from which analyze_gate derives
    # the α and stage; never part of a fingerprint (like --accept-pc1-failure).
    extension_of: str | None = None
    fix_cycle_of: str | None = None
    smoke_dir: Path = SMOKE_ROOT / "live"  # live preflight: the live smoke's report.json and checks.json (`check_live_smoke`)
    smoke_max_age_days: float = SMOKE_MAX_AGE_DAYS
    skip_smoke_check: str | None = None  # live preflight: the recorded reason for running without a fresh live smoke

    def __post_init__(self) -> None:
        if not _RUN_ID.match(self.run_id):
            raise PhaseError(f"run id {self.run_id!r}: use letters, digits, '.', '_' and '-' only")
        self.runs_root, self.config_dir = Path(self.runs_root), Path(self.config_dir)
        if not self.offline and self.config_dir.resolve() != (ROOT / "config").resolve():
            raise PhaseError("--config-dir is for offline runs and tests; live runs read config/ (build workers load config/models.yaml)")
        if self.smoke and self.runs_root.resolve() == (ROOT / "runs").resolve():
            raise PhaseError(f"smoke runs live outside runs/ (default {_show(SMOKE_RUNS)}): pass another --runs-dir")
        if self.smoke and self.anchor_data_dir is None:
            self.anchor_data_dir = ROOT / "cache" / "graphragbench"

    @property
    def tiny(self) -> bool:
        """Offline or smoke: the small sizes (OFFLINE_SCALE, or SMOKE_SCALE on its cells)."""
        return self.offline or self.smoke

    @property
    def isolated(self) -> bool:
        """Offline or smoke: worlds, indices, cache and config outputs under runs/<id>/work/."""
        return self.offline or self.smoke

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
        return self.work_dir / "config" if self.isolated else self.dir / "config"

    @property
    def selected_path(self) -> Path:
        return self.out_config_dir / "selected.yaml"

    @property
    def s7_targets_path(self) -> Path:
        return self.out_config_dir / "s7_targets.json"

    @property
    def budget_calibration_path(self) -> Path:
        return self.out_config_dir / "budget_calibration.yaml"

    @property
    def measured_out_path(self) -> Path:
        return self.out_config_dir / "budget_calibration_measured.yaml"

    @property
    def freeze_path(self) -> Path:
        return self.dir / "freeze.json"

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


def _resolve(shown: str) -> Path:
    """The path a manifest records (`_show`), back as a path."""
    return Path(shown) if Path(shown).is_absolute() else ROOT / shown


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:10]


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


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    status = _git("status", "--porcelain")
    return {"commit": _git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


CODE_DIRS = ("src", "power")  # untracked files here are code a result could depend on


def git_tracked_changes() -> list[str] | None:
    """Uncommitted changes the freeze refuses: modified or staged tracked files, and untracked files under
    CODE_DIRS (other untracked files are not counted); None when git is unavailable."""
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("status", "--porcelain", "--untracked-files=all", "--", *CODE_DIRS)
    if tracked is None or untracked is None:
        return None
    lines = tracked.splitlines() + [line for line in untracked.splitlines() if line.lstrip().startswith("??")]
    return list(dict.fromkeys(line.split(maxsplit=1)[1] for line in lines if line.strip()))


def code_drift(commit: str | None) -> list[str] | None:
    """The files under CODE_PATHS that differ from `commit` (the freeze commit): changed in a later commit, modified
    or staged in the working tree, or new and untracked. None when git cannot tell (or there is no commit)."""
    if not commit:
        return None
    diff = _git("diff", "--name-only", commit, "--", *CODE_PATHS)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS)
    if diff is None or untracked is None:
        return None
    return sorted({f for f in (*diff.splitlines(), *untracked.splitlines()) if f.strip()})


def code_state() -> dict | None:
    """What CODE_PATHS hold now, for the preflight fingerprint: HEAD's object id of each (a commit that touches only
    other files, e.g. GATE_PREREG.md, PROVENANCE.md or config/, leaves them unchanged), plus a digest of their
    uncommitted changes and of their untracked files. None when git cannot tell."""
    head = {}
    for p in CODE_PATHS:
        head[p] = _git("rev-parse", f"HEAD:{p}")
    diff = _git("diff", "HEAD", "--", *CODE_PATHS)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *CODE_PATHS)
    if diff is None or untracked is None or all(v is None for v in head.values()):
        return None
    h = hashlib.sha256()
    for name in sorted(f for f in untracked.splitlines() if f.strip()):
        h.update(name.encode() + b"\0" + (_sha256(ROOT / name) or "").encode() + b"\0")
    return {"head": head, "uncommitted": _digest(diff) if diff else None, "untracked": h.hexdigest()[:16] if untracked.strip() else None}


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
    Live: the embedding model from the gate profile unless APE_EMBEDDING_MODEL is set (as the CLIs do; a value other
    than the profile's is refused by `stray_environment`); smoke also puts worlds, indices and cache under
    runs/<id>/work.
    Both: APE_S7_TARGETS is the run's S7 targets file, and the test split's lock and seed base are cleared: only
    build-test sets them, after its freeze guard, so a stray value from the shell never unlocks or re-seeds it."""
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
        if run.smoke:  # live models, but nothing shared with real runs
            w = run.work_dir
            updates |= {"APE_WORLDS": w / "worlds", "APE_INDICES": w / "indices", "APE_CACHE": w / "cache"}
    updates["APE_S7_TARGETS"] = run.s7_targets_path  # S7 reads the targets this run's pilot writes
    updates[TEST_SPLIT_ENV] = None
    updates[TEST_SEED_BASE_ENV] = None
    # Spend registry (ape.spend): every log dir and ledger this run writes is labelled with the run; an offline run
    # has its own registry, so its guard exercises the program-wide count without touching the real one.
    updates[LABEL_ENV] = f"gate{'-offline' if run.offline else '-smoke' if run.smoke else ''}/{run.run_id}"
    if run.offline:
        updates[REGISTRY_ENV] = run.work_dir / "spend_registry.jsonl"
    with _environ(updates):
        yield


def _check_mode(run: GateRun) -> None:
    """A run is offline or live, and smoke or not, for its whole life: an offline `done` must never satisfy a live
    phase, nor a smoke-size one a full-size phase."""
    path = run.dir / "run.json"
    if path.is_file():
        info = json.loads(path.read_text())
        if bool(info.get("offline")) != run.offline:
            mode = "an offline" if info.get("offline") else "a live"
            raise PhaseError(f"run {run.run_id!r} is {mode} run ({path}); use another --run-id")
        if bool(info.get("smoke")) != run.smoke:
            raise PhaseError(f"run {run.run_id!r} is {'a smoke' if info.get('smoke') else 'not a smoke'} run ({path}); use another --run-id")
        return
    _write_json(path, {"run_id": run.run_id, "offline": run.offline, "smoke": run.smoke, "created": _now(), "config_dir": _show(run.config_dir), "git": git_state()})


def read_run_info(run: GateRun) -> dict:
    """runs/<id>/run.json: the run's mode, and facts settled once for the run (e.g. `smoke_check`)."""
    path = run.dir / "run.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def update_run_info(run: GateRun, **fields: Any) -> None:
    _write_json(run.dir / "run.json", read_run_info(run) | fields)


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
    """The sizes this run uses: SMOKE_SCALE for smoke runs, OFFLINE_SCALE offline, else run_plan.yaml's gate cells."""
    if run.smoke:
        return {"offline": run.offline, "smoke": True, **SMOKE_SCALE, "cells": list(SMOKE_SCALE["cells"])}
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


def smoke_keeps(run: GateRun, world_cell: str) -> bool:
    """Whether this run uses `world_cell`: every cell, except a smoke run's cells outside SMOKE_SCALE."""
    return not run.smoke or world_cell in SMOKE_SCALE["cells"]


def _smoke_cell(run: GateRun, cell: PlanCell) -> PlanCell:
    """A plan cell as a smoke run executes it, for the projection: builds of SMOKE_SCALE's cells, one world each
    (the anchor index keeps its corpus); agent cells on those cells at one world x tasks_per_world, one epoch,
    and gate.tune's arms counted from the smoke grid; the anchor at anchor_n_per_type per question type."""
    from .anchor.graphragbench import QUESTION_TYPES

    s = dict(cell.spec)
    kind = s.get("kind", "agent")
    if kind == "build" and cell.id != "gate.anchor.index":
        s["worlds"] = {c: SMOKE_SCALE["worlds_per_cell"] for c in s["worlds"] if c in SMOKE_SCALE["cells"]}
    elif kind == "anchor":
        s["n_tasks"] = SMOKE_SCALE["anchor_n_per_type"] * len(QUESTION_TYPES)
        s["n_per_type"] = SMOKE_SCALE["anchor_n_per_type"]
    elif kind == "agent":
        s.pop("n_tasks", None)
        s |= {"cells": [c for c in s["cells"] if c in SMOKE_SCALE["cells"]], "worlds": SMOKE_SCALE["worlds_per_cell"], "tasks_per_world": SMOKE_SCALE["tasks_per_world"], "epochs": SMOKE_SCALE["epochs"]}
        if isinstance(s.get("arms"), dict):  # gate.tune: candidates per arm
            counts: dict[str, int] = {}
            for sdef in tuning_grid(run)["systems"].values():
                for c in sdef["candidates"]:
                    counts[c.get("declared_arm", c["arm"])] = counts.get(c.get("declared_arm", c["arm"]), 0) + 1
            s["arms"] = counts
    return PlanCell(cell.id, cell.study, cell.phase, s)


def dev_world_specs(run: GateRun, offline: bool | None = None) -> list[dict]:
    """Every dev world group the gate needs, from its run_plan.yaml build cell (DEV_BUILD_CELLS): world
    counts and exception style from the cell; tasks per world from `gate.tune` tasks_per_world, except
    messy worlds, which split `gate.sec.messy` n_tasks per cell over the cell's worlds. `offline` overrides
    the run's small sizes (OFFLINE_SCALE), for the live projection of an offline run; a smoke run keeps only
    SMOKE_SCALE's cells."""
    offline = run.tiny if offline is None else offline
    p = plan(run)
    tasks = int(p.cell("gate.tune").spec["tasks_per_world"])
    messy_n = int(p.cell("gate.sec.messy").spec["n_tasks"])
    specs = []
    for group, cell_id in DEV_BUILD_CELLS.items():
        cell = p.cell(cell_id).spec
        for world_cell, count in cell["worlds"].items():
            if not smoke_keeps(run, world_cell):
                continue
            family, level = world_cell.split("-", 1)
            n_tasks = math.ceil(messy_n / count) if group == "messy" else tasks
            if offline:
                count, n_tasks = OFFLINE_SCALE["worlds_per_cell"], OFFLINE_SCALE["tasks_per_world"]
            style = cell.get("exception_style", "descriptive")
            specs.append({"group": group, "family": family, "level": level, "count": int(count), "n_tasks": int(n_tasks), "exception_style": style, "relational": True})
    return specs


def _cost_kwargs(run: GateRun) -> dict:
    """The cost model's inputs. Measured entries: config/'s (promoted program-wide by hand), then this run's pilot
    recalibration (`run.measured_out_path`), which wins where both measure the same (arm, model, effort, cell,
    delivery). The pilot never writes config/: promoting its measurements to the program is a deliberate copy."""
    measured = load_measured(run.config("budget_calibration_measured.yaml")) + load_measured(run.measured_out_path)
    return {
        "assumptions": load_assumptions(run.config("budget_assumptions.yaml")),
        "prices": Prices.load(run.costs_path),
        "measured": measured,
        "models_path": run.models_path,
    }


def project(run: GateRun, cells: Sequence[PlanCell]) -> float:
    """Conservative $ for these plan cells (plan cells, or ad-hoc ones for what the plan does not list); a smoke
    run prices them at its sizes (`_smoke_cell`)."""
    if run.smoke:
        cells = [_smoke_cell(run, c) for c in cells]
    return projected_cost(plan=Plan(tuple(cells), {}, {}, ()), **_cost_kwargs(run))


def budget_inputs(run: GateRun) -> dict:
    names = ("run_plan.yaml", "budget_assumptions.yaml", "model_costs.yaml", "models.yaml", "budget_calibration_measured.yaml")
    out = {f"config/{n}": {"path": _show(run.config(n)), "sha256": _sha256(run.config(n))} for n in names if run.config(n).exists()}
    if run.measured_out_path.is_file():  # the run's own pilot recalibration (`_cost_kwargs`)
        out["run/budget_calibration_measured.yaml"] = {"path": _show(run.measured_out_path), "sha256": _sha256(run.measured_out_path)}
    return out


def run_log_files(run: GateRun) -> list[str]:
    """Every eval log under this run's directory: final logs, failed attempts the runner keeps
    (`retry_cleanup=False`) and killed runs' started logs. Spend counts each sample once (by uuid)."""
    return sorted(str(p) for p in run.dir.rglob("*.eval"))


SPEND_KEYS = ("registry", "inspect_usd", "ledger_usd", "spent_usd", "by_study", "partial_logs", "unfinished_dirs", "missing", "unreadable")


def spend(run: GateRun) -> dict:
    """The guard's view: the whole program's spend (`ape.budget.program_remaining` over the spend registry; offline
    runs have their own registry under work/) against the plan's budget, or a lower `run.budget_usd`. Also this
    run's own spend (its logs plus the ledger it writes to): `run_spent_usd`."""
    budget = float(plan(run).budget["total_usd"])
    if run.budget_usd is not None:
        budget = min(budget, run.budget_usd)  # an override may only lower the plan's budget, never raise it
    program = program_remaining(budget, None, run.costs_path)
    mine = remaining(budget, run_log_files(run), Config().ledger_path, run.costs_path)
    return {
        "budget_usd": program["budget_usd"],
        "spent_usd": program["spent_usd"],
        "remaining_usd": program["remaining_usd"],
        "run_spent_usd": mine["spent_usd"],
        "run_inspect_usd": mine["inspect_usd"],
        "run_ledger_usd": mine["ledger_usd"],
        "program": {k: program[k] for k in SPEND_KEYS},
    }


def prior_attempt_spend(run: GateRun, name: str, old: dict | None, fingerprint: str) -> float:
    """$ that earlier attempts of this phase with the same fingerprint already spent on its current work (a resume
    after a crash, a failure or a budget stop, or a --force re-run): `ape.runner` reuses their finished eval logs and
    `ape.artifacts` their current artifacts, so the guard should project only the rest. It is the Inspect spend of
    the eval logs under the phase's directory written since the first such attempt, plus the build/embedding
    ledger's entries since then. 0 for a first run or changed inputs (new work: new log dirs, new artifacts).
    An approximation: a retried sample's first, wasted attempt is credited too, and a concurrent run's ledger
    entries in that window would be."""
    from .budget import ledger_spend, logs_spend

    if not old or old.get("fingerprint") != fingerprint:
        return 0.0
    same = [h["at"] for h in old.get("history") or [] if h.get("action") == "run" and h.get("fingerprint") == fingerprint]
    since = min(same) if same else old.get("started")
    if not since:
        return 0.0
    t0 = datetime.fromisoformat(since).timestamp()
    logs = [p for p in run.phase_dir(name).rglob("*.eval") if p.stat().st_mtime >= t0]
    return logs_spend(logs)["inspect_usd"] + ledger_spend(Config().ledger_path, run.costs_path, since=t0)


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
    """APE_* knobs in the environment that can change a phase's results: part of its params (fingerprint)."""
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("APE_") and k not in MANAGED_ENV and k not in OPERATIONAL_ENV}


def _operational_env() -> dict[str, str]:
    """OPERATIONAL_ENV as set: recorded in each manifest, never part of a fingerprint."""
    return {k: os.environ[k] for k in OPERATIONAL_ENV if os.environ.get(k)}


def stray_environment(run: GateRun) -> list[str]:
    """Live, full-size runs: environment settings that would silently move a paid phase off the gate's design or
    its directories. They are refused, not cleared: a value someone set on purpose must not be overridden quietly.
    - APE_WORLDS, APE_INDICES, APE_CACHE: the gate reads and writes worlds/, indices/ and cache/ under the repo
      (put them on another disk with a symlink); a moved cache also moves the ledger.
    - APE_EMBEDDINGS: any value but the default, `openai` (the fake embedder is for offline runs).
    - APE_EMBEDDING_MODEL, APE_MODEL_PROFILE: only the gate profile's values.
    - APE_BUILD_MODEL, APE_BUILD_EFFORT: the builder comes from config/models.yaml (APE_BUILD_FALLBACK=1 selects
      D-017's fallback builder, recorded in the phase params)."""
    if run.isolated:
        return []
    out = []
    for name, default in (("APE_WORLDS", ROOT / "worlds"), ("APE_INDICES", ROOT / "indices"), ("APE_CACHE", ROOT / "cache")):
        raw = os.environ.get(name, "").strip()
        if raw and Path(raw).resolve() != default.resolve():
            out.append(f"{name}={raw} (the gate uses {_show(default)})")
    if (raw := os.environ.get("APE_EMBEDDINGS", "").strip()) and raw != "openai":  # `openai` is the default backend
        out.append(f"APE_EMBEDDINGS={raw} (live runs use the profile's embedding model)")
    gp = gate_profile(run)
    for name, want in (("APE_EMBEDDING_MODEL", gp.role("embeddings").model), ("APE_MODEL_PROFILE", gp.name)):
        raw = os.environ.get(name, "").strip()
        if raw and raw != want:
            out.append(f"{name}={raw} (the gate profile's is {want})")
    out += [f"{name}={os.environ[name]} (the builder comes from config/models.yaml)" for name in ("APE_BUILD_MODEL", "APE_BUILD_EFFORT") if os.environ.get(name, "").strip()]
    return out


def _result_knobs() -> list[str]:
    """APE_* knobs in the environment that change what arms do (the builder's knobs act on builds only)."""
    return [k for k in _env_knobs() if not k.startswith("APE_BUILD_")]


def _missing_outputs(manifest: dict) -> list[str]:
    """The recorded outputs that no longer exist (a complete phase whose outputs are gone re-runs)."""
    missing = []
    for key, shown in (manifest.get("outputs") or {}).items():
        if not _resolve(shown).exists():
            missing.append(key)
    return missing


def _mock_models(profile: Profile, roles: dict[str, Callable]) -> tuple[Any, dict[str, Any]]:
    """The profile's agent and roles as mockllm models with the given scripts; efforts and settings are kept."""
    from .models import agent_model, role_models

    agent = agent_model(profile, model=MOCK, custom_outputs=roles.pop("agent"))
    return agent, {r: role_models(profile, (r,), model=MOCK, custom_outputs=fn)[r] for r, fn in roles.items()}


def gate_models(run: GateRun, gp: Profile) -> tuple[Any, dict[str, Any]]:
    """The gate's agent and kg models: offline the mock scripts (`mock_agent`, `mock_kg`) under the profile's
    settings; live the profile's models, after the FX-2 preflight."""
    if run.offline:
        from .llm.mock_agent import mock_agent, mock_kg

        return _mock_models(gp, {"agent": mock_agent, "kg": mock_kg})
    from .models import agent_model, require_preflight, role_models

    require_preflight(gp, live=True, costs_path=run.costs_path, env_path=run.env_path)
    return agent_model(gp), role_models(gp, ("kg",))


def sample_usd(projected_usd: float | None, groups: Sequence[tuple[list, int]]) -> float | None:
    """The cost model's conservative $ per sample over `groups` of (tasks, epochs): the base of the runner's
    per-sample `cost_limit`. It is a mean over the group's arms, so an arm that costs ~20x its group's mean
    would hit the limit; PC5 counts such hits."""
    from .runner import planned_samples

    n = sum(planned_samples(t, epochs) for tasks, epochs in groups for t in tasks)
    return projected_usd / n if projected_usd and n else None


def run_gate_tasks(run: GateRun, gp: Profile, models: tuple[Any, dict], tasks: list, log_dir: Path, epochs: int, what: str, sample_cost_usd: float | None = None) -> list[str]:
    """One resumable eval set (`ape.runner.run_evals`) of gate tasks; the final log files. Raises PhaseError
    when the set did not finish (a re-run resumes it). `sample_cost_usd` sets the per-sample `cost_limit`."""
    from .runner import log_path, run_evals

    agent, roles = models
    extra = {"display": "none"} if run.tiny else {}
    success, logs = run_evals(
        tasks, log_dir, profile=gp, model=agent, model_roles=dict(roles), epochs=epochs, costs_path=run.costs_path, sample_cost_usd=sample_cost_usd, log_dir_allow_dirty=True, **extra
    )
    if not success:
        failed = [f"{h.eval.task_args.get('arm')} {h.eval.task_args.get('family')}-{h.eval.task_args.get('level')}: {h.error.message if h.error else h.status}" for h in logs if h.status != "success"]
        raise PhaseError(f"{what}: eval set did not finish (re-run to resume): " + "; ".join(failed or ["eval set failed"]))
    return [log_path(h) for h in logs]


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
    refuse: Callable[[GateRun], str | None] | None = None  # a reason the phase must not run now (checked before any record)
    # The projected $ of the work still to do, for the budget guard; default: `projected` less what earlier attempts
    # with the same fingerprint already spent (`prior_attempt_spend`).
    remaining: Callable[[GateRun], float] | None = None


def _cfg_inputs(run: GateRun, *names: str) -> dict[str, Path]:
    return {f"config/{n}": run.config(n) for n in names}


# preflight -----------------------------------------------------------------------------------------


def _apg_installed_commit() -> str | None:
    try:
        raw = importlib.metadata.distribution("apg-core").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return ((json.loads(raw) if raw else {}).get("vcs_info") or {}).get("commit_id")


def _provenance_facts(run: GateRun) -> dict:
    """What preflight reads from PROVENANCE.md: whether it records the APG pin, and the snapshot pins. Not the whole
    file, so a freeze's appended record does not re-run preflight."""
    from .snapshots import read_pins

    path = run.provenance_path
    text = path.read_text() if path.is_file() else ""
    return {"apg_pin_recorded": APG_PIN in text, "snapshot_pins": read_pins(path) if path.is_file() else {}}


def _preflight_params(run: GateRun) -> dict:
    # The facts preflight checks, so a code change, another apg-core, a new key, probe or pin re-runs it. Not the
    # commit or the tree's other files: a run's documented steps (outputs under runs/<id>/, the pre-registration's
    # commit, the freeze's PROVENANCE.md record) must not re-run it. The live smoke is settled once per run
    # (run.json `smoke_check`, `_settle_smoke`), so a newer smoke record does not re-run it either.
    env = {"code": code_state(), "apg_core_commit": _apg_installed_commit(), "provenance": _provenance_facts(run)}
    if not run.offline:
        from .models import _api_key_present

        env["api_key_present"] = _api_key_present(run.env_path)
        env["probe_sha256"] = _sha256(run.probe_path)
    return {"scale": scale(run), "environment": env, "env_knobs": _env_knobs()}


def _smoke_time(entry: dict) -> datetime | None:
    """When a smoke report or check finished: `finished_utc` (ISO, with zone), else the older local `finished`."""
    for key, local in (("finished_utc", False), ("finished", True)):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            t = datetime.fromisoformat(str(raw))
        except ValueError:
            continue
        return (t.astimezone() if local and t.tzinfo is None else t).astimezone(UTC)
    return None


def check_live_smoke(run: GateRun, gp: Profile, now: datetime | None = None) -> tuple[list[str], dict]:
    """Problems that stop a live gate run for want of a fresh passing live smoke (readiness/smoke.py), and what was
    checked. Two files in `run.smoke_dir` (cache/smoke/live/):

    - report.json, the latest live invocation: live (never a dry run), status pass or warn (not fail, not
      stopped by its spend cap), and run on this gate profile.
    - checks.json, every check's latest live result across invocations (a check re-run alone with --only
      replaces its own entry): each check smoke.py lists as required has a pass or warn result, run on this
      profile without model overrides, on committed code (no uncommitted change under CODE_PATHS when it ran) whose
      CODE_PATHS are exactly what would run now (`code_drift` of its commit is empty: ancestry alone would accept a
      smoke of older code), at most `run.smoke_max_age_days` old, and, where PROVENANCE.md pins snapshots, served by
      the pinned snapshot of every alias this profile calls.
    """
    from . import snapshots as snaps

    now = now or datetime.now(UTC)
    report_path, record_path = run.smoke_dir / "report.json", run.smoke_dir / "checks.json"
    info: dict[str, Any] = {"report": _show(report_path), "record": _show(record_path), "max_age_days": run.smoke_max_age_days}
    how = f"run `uv run --locked python readiness/smoke.py --max-usd 4`, or pass --skip-smoke-check \"<reason>\""
    if not report_path.is_file() or not record_path.is_file():
        missing = report_path if not report_path.is_file() else record_path
        return [f"no live smoke record ({_show(missing)} missing): a live gate run needs a passing live smoke first; {how}"], info
    try:
        report, record = json.loads(report_path.read_text()), json.loads(record_path.read_text())
    except (OSError, ValueError) as e:
        return [f"the live smoke record in {_show(run.smoke_dir)} is unreadable ({e}); {how}"], info
    problems: list[str] = []
    mode = report.get("mode") or ("dry" if report.get("dry") else "live")
    if report.get("dry") or mode != "live":
        problems.append(f"{_show(report_path)} is a dry run's report (mode {mode!r}), not a live smoke; {how}")
    status = report.get("status")
    info["latest_status"] = status
    if status not in SMOKE_OK:
        detail = f": {report['stopped']}" if report.get("stopped") else (f" (failed: {', '.join(n for n, r in (report.get('checks') or {}).items() if r.get('status') == 'fail')})" if status == "fail" else "")
        problems.append(f"the latest live smoke's status is {status!r}{detail}; it must pass (warn-level checks allowed): re-run it ({how})")
    profile = (report.get("models") or {}).get("profile")
    if profile != gp.name:
        problems.append(f"the latest live smoke ran profile {profile!r}, not the gate's {gp.name!r}; {how}")
    pins = snaps.read_pins(run.provenance_path) if run.provenance_path.is_file() else {}
    fallback = os.environ.get("APE_BUILD_FALLBACK") == "1"
    called = {snaps.alias(s.model) for r, s in gp.roles.items() if not s.model.startswith("mockllm/") and (r != "build_fallback" or fallback)}
    pinned = {a: pins[a] for a in sorted(called & set(pins))}
    info["pinned"] = pinned
    checks = record.get("checks") or {}
    required = list(record.get("required") or [])
    info["required"] = required
    drift: dict[str, list[str] | None] = {}
    rows = {}
    for name in required:
        entry = checks.get(name)
        if entry is None:
            problems.append(f"smoke check {name!r} has no live result; run it (`readiness/smoke.py --only {name}`)")
            continue
        when = _smoke_time(entry)
        age = (now - when).total_seconds() / 86400 if when else None
        commit = entry.get("git_commit")
        if commit and commit not in drift:
            drift[commit] = code_drift(commit)
        rows[name] = {"status": entry.get("status"), "age_days": None if age is None else round(age, 2), "git_commit": commit}
        if entry.get("status") not in SMOKE_OK:
            problems.append(f"smoke check {name!r}: latest live result is {entry.get('status')!r}; it must pass (warn allowed)")
        if entry.get("profile") != gp.name:
            problems.append(f"smoke check {name!r} ran profile {entry.get('profile')!r}, not {gp.name!r}")
        if entry.get("overrides"):
            problems.append(f"smoke check {name!r} ran with model overrides {entry['overrides']}: re-run it on the profile's own models")
        if age is None:
            problems.append(f"smoke check {name!r} has no finish time")
        elif age > run.smoke_max_age_days:
            problems.append(f"smoke check {name!r} is {age:.1f} days old (limit {run.smoke_max_age_days:g}; --smoke-max-age-days); re-run it")
        if not commit:
            problems.append(f"smoke check {name!r} records no commit")
        elif drift[commit] is None:
            problems.append(f"smoke check {name!r}: cannot tell whether the code it ran at {commit[:12]} is the code that would run (git unavailable or unknown commit)")
        elif d := drift[commit]:
            problems.append(
                f"smoke check {name!r} ran at {commit[:12]}, but {', '.join(CODE_PATHS)} differ from that commit now: {d[:10]}{' ...' if len(d) > 10 else ''}; "
                "the code it smoked is not the code that would run: re-run the smoke on this code"
            )
        dirty = entry.get("code_dirty")
        if dirty is None and entry.get("git_dirty"):  # a record from before `code_dirty`: any uncommitted change counts
            dirty = ["(uncommitted changes; the record predates code_dirty)"]
        if dirty:
            problems.append(f"smoke check {name!r} ran with uncommitted code changes {list(dirty)[:10]}: what it smoked is no commit; commit, then re-run it")
        seen = entry.get("snapshots") or {}
        for a, snap in pinned.items():
            if seen.get(a) != snap:
                problems.append(f"smoke check {name!r} ran on snapshot {seen.get(a)!r} of {a}, but PROVENANCE.md pins {snap!r}: re-run the smoke on the pinned models")
    info["checks"] = rows
    info["commits"] = sorted(drift)
    return list(dict.fromkeys(problems)), info


def _smoke_code_changed(settled: dict) -> list[str]:
    """Problems when CODE_PATHS no longer match the commits of the smoke a run accepted (`code_drift`)."""
    out = []
    for commit in settled.get("commits") or []:
        d = code_drift(commit)
        if d is None:
            out.append(f"cannot tell whether the code is still what the smoke this run accepted ran at {commit[:12]} (git unavailable or unknown commit)")
        elif d:
            out.append(f"{', '.join(CODE_PATHS)} changed since the smoke this run accepted (at {commit[:12]}): {d[:10]}{' ...' if len(d) > 10 else ''}")
    return out


def _settle_smoke(run: GateRun, gp: Profile, record: dict, checks: dict) -> tuple[list[str], dict | None]:
    """Live runs: the live smoke, settled once per run in run.json (`smoke_check`), so a later preflight (after the
    pre-registration's commit, a freeze, a resume) neither re-checks the smoke's age nor fails on it. Returns
    (problems, the settlement to record once preflight passes, or None to keep run.json as is).

    - `--skip-smoke-check "<reason>"` on this invocation: recorded as the run's override.
    - Settled by an earlier preflight on a smoke: reused while CODE_PATHS still match the smoke's commits. Once the
      code changes, only a smoke of this code will do: `check_live_smoke` again, as at a first preflight.
    - Settled by an override: a passing smoke now replaces it; otherwise the override stands (a warning).
    - Not settled: `check_live_smoke`."""
    settled = read_run_info(run).get("smoke_check") or {}
    if run.skip_smoke_check is not None:
        reason = run.skip_smoke_check.strip()
        checks["smoke"] = {"skipped": reason}
        record["smoke_check"] = {"skipped": reason}
        record["warnings"].append(f"the live smoke check was skipped: {reason!r}")
        if not reason:
            return ["--skip-smoke-check needs a reason: why this live run may go without a fresh passing live smoke"], None
        return [], {"override": reason, "at": _now()}
    changed: list[str] = []
    if settled.get("commits"):
        if not (changed := _smoke_code_changed(settled)):
            checks["smoke"] = f"ok (settled at this run's first preflight, {settled.get('at')}; its age is not re-checked)"
            record["smoke_check"] = settled | {"reused": True}
            return [], None
    smoke_problems, info = check_live_smoke(run, gp)
    if smoke_problems and settled.get("override"):
        checks["smoke"] = {"skipped": settled["override"], "reused": True}
        record["smoke_check"] = settled | {"reused": True, "smoke_now": info | {"problems": smoke_problems}}
        record["warnings"].append(f"the live smoke check was skipped at this run's first preflight: {settled['override']!r}")
        return [], None
    checks["smoke"] = "ok" if not smoke_problems else "failed (see errors)"
    record["smoke_check"] = info | {"problems": changed + smoke_problems}
    if smoke_problems:
        return changed + smoke_problems, None
    return [], {"at": _now(), "commits": info["commits"], "report": info["report"], "record": info["record"], "max_age_days": run.smoke_max_age_days, "checks": info["checks"]}


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
    # Live, this includes the model-snapshot check against this run's probe and PROVENANCE.md (ape.snapshots).
    fx2 = [x for p in (gp, ap) for x in preflight(p, live=not run.offline, costs_path=run.costs_path, env_path=run.env_path, probe_path=run.probe_path, provenance_path=run.provenance_path)]
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
    # 5. Live runs (not smoke runs): a passing live smoke of this code on these models, settled once per run
    #    (`_settle_smoke`), unless skipped with a recorded reason; and storage warnings (backup destination, free disk).
    settlement = None
    if not run.offline and not run.smoke:
        smoke_problems, settlement = _settle_smoke(run, gp, record, checks)
        problems += smoke_problems
        record["warnings"] += storage_warnings()
    record["checks"] = checks
    if problems:
        raise PreflightError("preflight failed:\n" + "\n".join(f"  - {x}" for x in dict.fromkeys(problems)))
    if settlement is not None:
        update_run_info(run, smoke_check=settlement)


# build-dev -----------------------------------------------------------------------------------------


def build_params(run: GateRun) -> dict:
    """The builder a build phase uses (D-017): the gate profile's build role, or its fallback under
    APE_BUILD_FALLBACK=1 (`ape.models.build_settings`). Part of build-dev's, pilot's and build-test's params, so a
    switch of builder re-runs the builds and, through upstream fingerprints, every phase that rests on them."""
    from .models import build_settings

    model, effort = build_settings(gate_profile(run))
    return {"model": model, "effort": effort, "fallback": os.environ.get("APE_BUILD_FALLBACK") == "1"}


def _build_dev_projected(run: GateRun) -> float:
    p = plan(run)
    return project(run, [p.cell(cell_id) for cell_id in DEV_BUILD_CELLS.values()])


def build_world_set(run: GateRun, record: dict, phase: str, split: str, specs: list[dict]) -> list[dict]:
    """Generate the worlds of `specs` in `split` (world i of a spec has seed base+i) and build their artifacts
    through `ape.artifacts` (offline: oracle LightRAG indices, the fake author); writes `<phase>/worlds.json`
    and the record's outputs and world counts. Raises (after recording) when any world failed to build."""
    from .artifacts import KINDS, build_artifacts
    from .worlds.generate import make_world
    from .worlds.spec import World

    require_test_split_unlocked(split)  # the test split: only build-test, after the freeze guard
    cfg = Config()
    worlds: list[dict] = []
    for s in specs:
        for i in range(s["count"]):
            w = make_world(s["family"], s["level"], split, i, s["n_tasks"], s["relational"], s["exception_style"])
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
    print(f"[{phase}] {len(worlds)} {split} world(s): building {','.join(KINDS)} (lightrag {lightrag_kind}{', fake author' if run.offline else ''})", flush=True)
    with _environ(updates):
        results = build_artifacts([cfg.world_path(w["world_id"]) for w in worlds], KINDS, lightrag_kind, run.offline, on_result=lambda r: print(r.summary(), flush=True))
    for w, r in zip(worlds, results, strict=True):
        w["artifacts"], w["errors"] = r.kinds, r.errors
    record_build_health(run, record, phase, [w for w in worlds if not w["errors"]], lightrag_kind)
    out = run.phase_dir(phase) / "worlds.json"
    _write_json(out, {"worlds_dir": _show(cfg.worlds_dir), "indices_dir": _show(cfg.indices_dir), "lightrag_kind": lightrag_kind, "fake_author": run.offline, "worlds": worlds})
    record["outputs"] |= {"worlds": _show(out), "worlds_dir": _show(cfg.worlds_dir), "indices_dir": _show(cfg.indices_dir)}
    record["worlds"] = {"count": len(worlds), "by_group": {g: sum(w["group"] == g for w in worlds) for g in dict.fromkeys(s["group"] for s in specs)}}
    if failed := [r for r in results if r.status == "failed"]:
        raise PhaseError(f"{len(failed)} {split} world(s) failed to build (re-run to resume):\n" + "\n".join(r.summary() for r in failed))
    return worlds


def record_build_health(run: GateRun, record: dict, phase: str, worlds: list[dict], lightrag_kind: str) -> None:
    """Every build phase (dev, pilot, test): degradation signals of the worlds just built (`ape.build_quality.build_health`:
    lost chunks, coverage under D-017's threshold) in `<phase>/build_health.json`, a summary in the manifest and a
    warning per flagged world. Never fails the phase; D-017's builder verdict stays build-dev's (`record_build_quality`)."""
    from .build_quality import build_health
    from .worlds.spec import World

    cfg = Config()
    try:
        health = build_health((World.load(cfg.world_path(w["world_id"])) for w in worlds), cfg, lightrag_kind, offline=run.offline)
    except Exception as e:  # noqa: BLE001  (a health report must never fail a build)
        record["warnings"].append(f"build health: could not be measured ({type(e).__name__}: {e})")
        return
    path = run.phase_dir(phase) / "build_health.json"
    _write_json(path, health)
    record["outputs"] |= {"build_health": _show(path)}
    record["build_health"] = {"worlds_checked": health["worlds_checked"], "flagged_worlds": health["flagged_worlds"]}
    shown = health["warnings"][:20]
    record["warnings"] += [f"build health: {w}" for w in shown]
    if len(health["warnings"]) > len(shown):
        record["warnings"].append(f"build health: {len(health['warnings']) - len(shown)} more in {_show(path)}")


def _build_dev(run: GateRun, record: dict) -> None:
    worlds = build_world_set(run, record, "build-dev", "dev", dev_world_specs(run))
    record_build_quality(run, record, worlds)


def build_quality_path(run: GateRun) -> Path:
    return run.phase_dir("build-dev") / "build_quality.json"


def read_build_quality(run: GateRun) -> dict | None:
    path = build_quality_path(run)
    return json.loads(path.read_text()) if path.is_file() else None


def _builder_label(spec: Any) -> str:
    return f"{spec.model} ({spec.reasoning_effort} effort)" if spec.reasoning_effort else spec.model


def record_build_quality(run: GateRun, record: dict, worlds: list[dict]) -> dict:
    """D-017's build-quality check on the dev worlds just built (`ape.build_quality`): writes
    build-dev/build_quality.json and a summary in the manifest. A failing or incomplete check is a warning, not a
    failure: the outcome is a builder decision for the user (GATE_PREREG's `[USER: builder]`), not a broken build.
    The decisive cells are the gate's dev cells (`gate.build.dev`) even in a smoke run, so a smoke run, which
    builds only some of them, reports `incomplete` rather than a pass."""
    from .budget import fallback_build_delta
    from .build_quality import assess, builder_roles
    from .worlds.spec import World

    cfg = Config()
    gp = gate_profile(run)
    build, fallback, using_fallback = builder_roles(gp)  # the fallback builds under APE_BUILD_FALLBACK=1
    label = _builder_label(build) + (" (the D-017 fallback, APE_BUILD_FALLBACK=1)" if using_fallback else "")
    builder = f"scripted perfect_author (offline stand-in for {label})" if run.offline else label
    try:
        fallback_usd: float | None = fallback_build_delta(plan(run), **_cost_kwargs(run))
    except Exception as e:  # noqa: BLE001  (the check stands without the price; say why it is missing)
        fallback_usd = None
        record["warnings"].append(f"build quality: could not price the D-017 fallback builder ({type(e).__name__}: {e})")
    report = assess(
        (World.load(cfg.world_path(w["world_id"])) for w in worlds),
        cfg,
        "oracle" if run.offline else "extract",
        offline=run.offline,
        builder=builder,
        fallback_builder=_builder_label(fallback) if fallback else "the build_fallback role (not in this profile)",
        fallback_usd=fallback_usd,
        expected_cells=list(plan(run).cell(DEV_BUILD_CELLS["gate"]).spec["worlds"]),
    )
    path = build_quality_path(run)
    _write_json(path, report)
    record["outputs"] |= {"build_quality": _show(path)}
    record["build_quality"] = {
        "verdict": report["verdict"],
        "offline": report["offline"],
        "recommendation": report["recommendation"],
        "cells": {
            c: {"apg": s["apg"]["coverage"] if s["apg"] else None, "lightrag": s["lightrag"]["coverage"] if s["lightrag"] else None, "pass": s["pass"]}
            for c, s in report["cells"].items()
        },
    }
    if report["verdict"] != "builder_passes":
        record["warnings"].append(f"D-017 build check: {report['recommendation']}")
    print(f"[build-dev] D-017 build check: {report['recommendation']}", flush=True)
    return report


# tune ----------------------------------------------------------------------------------------------


def tuning_grid(run: GateRun) -> dict:
    """config/tuning_grid.yaml; offline and smoke: the first N candidates per system (offline: LightRAG arms as
    their oracle twins; smoke: the dev cells it keeps)."""
    from .tuning import load_grid

    grid = load_grid(run.config("tuning_grid.yaml"))
    if not run.tiny:
        return grid
    systems = {}
    for name, sdef in grid["systems"].items():
        cands = []
        for c in sdef["candidates"][: OFFLINE_SCALE["tune_candidates_per_system"]]:
            c = dict(c)
            if run.offline and c.get("arm") in OFFLINE_ARMS:
                c["declared_arm"], c["arm"] = c["arm"], OFFLINE_ARMS[c["arm"]]
            cands.append(c)
        systems[name] = {**sdef, "candidates": cands}
    return {**grid, "systems": systems, "dev_cells": [c for c in grid["dev_cells"] if smoke_keeps(run, c)]}


SKEPTIC_SYSTEMS = ("LightRAG", "S3s")  # owned by the skeptic, who is not on the APG side (GATE_PREREG §5, O-3)


def _unset(owner: Any) -> bool:
    return owner is None or not str(owner).strip() or str(owner).strip().upper() in ("TODO", "TBD", "NONE")


def tuning_signoff_problems(grid: dict) -> list[str]:
    """Why config/tuning_grid.yaml is not ready for a live tune: every tuned system needs a named owner in `owners`
    and `signed_off: true`, and the skeptic's systems (LightRAG, S3s) must not be owned by the APG owner."""
    owners, signed = grid.get("owners") or {}, grid.get("signed_off") or {}
    problems = []
    for system, _ in TUNED_SYSTEMS:
        if _unset(owners.get(system)):
            problems.append(f"owners.{system} is not set")
        if signed.get(system) is not True:
            problems.append(f"signed_off.{system} is not true")
    apg = owners.get("APG")
    if not _unset(apg):
        for system in SKEPTIC_SYSTEMS:
            if not _unset(owners.get(system)) and str(owners[system]).strip().casefold() == str(apg).strip().casefold():
                problems.append(f"owners.{system} is the APG owner ({apg}): the skeptic owns {' and '.join(SKEPTIC_SYSTEMS)} and is not on the APG side")
    return problems


def _refuse_tune(run: GateRun) -> str | None:
    """Frozen runs never re-tune; a live tune also needs the grid's owners and their sign-off (offline and smoke runs
    tune their tiny grids without it)."""
    if reason := _refuse_if_frozen("tune")(run):
        return reason
    if run.tiny:
        return None
    from .tuning import load_grid

    path = run.config("tuning_grid.yaml")
    grid = load_grid(path)
    if problems := tuning_signoff_problems(grid):
        return (
            f"tune: {_show(path)} is not signed off for a live tune: {'; '.join(problems)}. Each system's owner declares "
            f"its candidates, then sets signed_off.<system>: true (GATE_PREREG §5; O-3: the skeptic owns "
            f"{' and '.join(SKEPTIC_SYSTEMS)} and is not on the APG side). Sign off before tuning: PC6 counts every "
            f"configuration ever tried for a system, archived tuning logs included, against budget_per_system "
            f"({grid.get('budget_per_system')}), so a candidate replaced after a live tune still counts against that budget"
        )
    return None


def _tune_params(run: GateRun) -> dict:
    sc = scale(run)
    return {
        "scale": sc,
        "systems": [s for s, _ in TUNED_SYSTEMS],
        "limit_worlds": sc["worlds_per_cell"] if run.tiny else sc["tune_worlds"],
        "epochs": sc["epochs"],
        "offline_arms": OFFLINE_ARMS if run.offline else None,
        "env_knobs": _env_knobs(),
    }


def _tune(run: GateRun, record: dict) -> None:
    from .tuning import tune

    params = _tune_params(run)
    grid = tuning_grid(run)
    gp = gate_profile(run)
    agent, roles = gate_models(run, gp)
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
        f"# Tuning log: {_show(tlog)}. Each `env` holds only that arm's knobs; the pilot and test apply them together.\n"
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


ANCHOR_CODE = ("src/ape/anchor", "src/ape/tasks/anchor_graphragbench.py")  # the PC1 answerer, scorers and index code


def anchor_code_hash() -> str:
    """sha256 (12 hex) over the anchor's code (ANCHOR_CODE): its answerer, both scorers, bge and the index build.
    Inspect's task identity leaves the scorer out, so the anchor's eval logs live in a directory keyed by this
    hash: after a harness fix the anchor runs and scores afresh, instead of reusing the old logs' scores."""
    files: list[Path] = []
    for rel in ANCHOR_CODE:
        p = ROOT / rel
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(ROOT)).encode() + b"\0" + (f.read_bytes() if f.is_file() else b"") + b"\0")
    return h.hexdigest()[:12]


def _anchor_params(run: GateRun) -> dict:
    name, _ = anchor_profile_name(run)
    data = "fixture" if run.offline else _show(run.anchor_data_dir) if run.anchor_data_dir else "cache/graphragbench"
    # The anchor always builds with the paper's builder (D-009), never D-017's fallback: APE_BUILD_FALLBACK is not its knob.
    knobs = {k: v for k, v in _env_knobs().items() if k != "APE_BUILD_FALLBACK"}
    return {"scale": scale(run), "profile": name, "modes": list(ANCHOR_MODES), "data": data, "code": anchor_code_hash(), "env_knobs": knobs}


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
        data = run.anchor_data_dir or gb.default_data_dir(cfg)
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

        build_model, build_effort = build_settings(profile, fallback=False)  # the paper's builder, whatever D-017 decides for the gate
        llm = BuildLlm(build_model, Ledger(cfg.ledger_path), {"anchor": gb.ANCHOR_ID, "system": "lightrag"}, reasoning_effort=build_effort).lightrag_func()
    from .anchor.bge import anchor_embedder

    emb_model = anchor_embedder(cfg).identity  # the paper's bge-large-en-v1.5 (pinned), or the fake one offline
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
    log_dir = adir / f"logs-{anchor_code_hash()}"  # a fixed scorer scores afresh: Inspect's task identity omits the scorer
    record["log_dirs"] = [_show(log_dir)]
    record["anchor_code"] = anchor_code_hash()
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
    record["pc1_status"] = pc1["status"]
    if pc1["status"] == "not_evaluable":
        record["warnings"].append(f"PC1 is not evaluable: {pc1['diagnosis']} (see pc1.json; GATE_PREREG §7)")
    elif not pc1["pass"]:
        record["warnings"].append("PC1 does not pass (see pc1.json): fix and re-pilot before the test phase (GATE_PREREG §7)")


# pilot ---------------------------------------------------------------------------------------------


def pilot_world_specs(run: GateRun) -> list[dict]:
    """The pilot worlds (split "pilot", seeds 2000+): counts from `gate.build.pilot`, tasks per world from
    `gate.pilot` (offline: OFFLINE_SCALE, doubled when the σ cell runs on worlds of its own)."""
    p = plan(run)
    tasks = int(p.cell("gate.pilot").spec["tasks_per_world"])
    tiny_count = OFFLINE_SCALE["worlds_per_cell"] * (2 if _pilot_sigma_cell(run) else 1)
    specs = []
    for world_cell, count in p.cell(PILOT_BUILD_CELL).spec["worlds"].items():
        if not smoke_keeps(run, world_cell):
            continue
        family, level = world_cell.split("-", 1)
        n_tasks = tasks
        if run.tiny:
            count, n_tasks = tiny_count, OFFLINE_SCALE["tasks_per_world"]
        specs.append({"group": "pilot", "family": family, "level": level, "count": int(count), "n_tasks": int(n_tasks), "exception_style": "descriptive", "relational": True})
    return specs


def _pilot_cells(run: GateRun) -> dict[str, dict]:
    """The pilot run cells as this run executes them: world limits and epochs (offline and smoke: OFFLINE_SCALE;
    smoke: on SMOKE_SCALE's cells)."""
    out = {}
    for cell_id in PILOT_RUN_CELLS:
        spec = dict(plan(run).cell(cell_id).spec)
        if run.tiny:
            spec |= {"worlds": OFFLINE_SCALE["worlds_per_cell"], "epochs": OFFLINE_SCALE["epochs"]}
        if run.smoke:
            spec["cells"] = [c for c in spec["cells"] if smoke_keeps(run, c)]
        spec.setdefault("deliveries", ["push"])
        spec.setdefault("epochs", 1)
        out[cell_id] = spec
    return out


def _pilot_sigma_cell(run: GateRun) -> dict | None:
    """The σ cell as this run executes it (D-026), or None when the plan has none (or it is disabled). Offline and
    smoke: OFFLINE_SCALE's worlds, after the main pilot's."""
    try:
        cell = plan(run).cell(PILOT_SIGMA_CELL)
    except BudgetError:
        return None
    if not cell.enabled:
        return None
    spec = dict(cell.spec)
    if run.tiny:
        spec |= {"worlds": OFFLINE_SCALE["worlds_per_cell"], "world_offset": OFFLINE_SCALE["worlds_per_cell"], "epochs": OFFLINE_SCALE["epochs"]}
    if run.smoke:
        spec["cells"] = [c for c in spec["cells"] if smoke_keeps(run, c)]
    spec["deliveries"] = ["push"]
    spec.setdefault("epochs", 1)
    return spec


def selected_arms(selected: dict) -> dict[str, str]:
    """Plan arm names -> the arm each runs as: APG*, LGR* and S3s as selected (offline LightRAG: its oracle twin)."""
    return {key: selected[key]["arm"] for _, key in TUNED_SYSTEMS}


def selected_env(selected: dict) -> dict[str, str]:
    """The selected configurations' knobs, together. Each holds only its own arm's knobs, so none may collide."""
    env: dict[str, str] = {}
    for _, key in TUNED_SYSTEMS:
        knobs = {k: str(v) for k, v in (selected[key].get("env") or {}).items()}
        if clash := sorted(set(knobs) & set(env)):
            raise PhaseError(f"selected.yaml: {key} sets {clash}, which another selection also sets; each `env` must hold only its own arm's knobs")
        env |= knobs
    return env


def s7_env(selected: dict) -> dict[str, str]:
    """S7's delivery schedule mirrors APG*'s (D-024): per step when the selected APG* is per-step (APG-s), else once
    per task. Applied wherever S7 runs; tasks record it with their knobs."""
    from .apg.arm import ARMS as APG_ARMS

    arm = selected["APG*"]["arm"]
    if arm not in APG_ARMS:
        raise PhaseError(f"selected.yaml: APG* is {arm!r}, not an APG arm ({sorted(APG_ARMS)})")
    return {"APE_S7_PER_STEP": "1" if APG_ARMS[arm][1] else "0"}


def _resolve_arm(name: str, arms: dict[str, str]) -> str:
    return arms.get(name, name)


def _gate_tasks(spec: dict, arms: dict[str, str], split: str, only: Sequence[str] | None = None) -> list:
    from .tasks.gate import gate

    skip = {"skip_worlds": int(spec["world_offset"])} if spec.get("world_offset") else {}  # passed only when set: task identities stay
    return [
        gate(family=family, level=level, split=split, arm=_resolve_arm(a, arms), delivery=delivery, limit_worlds=int(spec["worlds"]), **skip)
        for a in spec["arms"]
        if only is None or a in only
        for delivery in spec["deliveries"]
        for family, level in (cell.split("-", 1) for cell in spec["cells"])
    ]


def _medians(tokens: dict[tuple[str, str], list[int]], arm: str, cells: Sequence[str]) -> tuple[float | None, dict[str, float | None]]:
    """An arm's median realized context pooled over `cells`, and per cell."""
    from .analysis.pilot import median_or_none

    per_cell = {c: median_or_none(tokens.get((arm, c), [])) for c in cells}
    return median_or_none([t for c in cells for t in tokens.get((arm, c), [])]), per_cell


def _cal_base(key: str, env: dict[str, str]) -> dict[str, int]:
    """A capped arm's knobs at its selected configuration: its budget knob (as `Config` resolves it under the
    selected env) and any explicit per-part caps the selection sets."""
    knobs = CAL_KNOBS[key]
    with _environ(env):
        cfg = Config()
        budget = cfg.lgr_budget_tokens if key == "LGR*" else cfg.s3s_budget_tokens
    return {knobs[0]: int(budget)} | {k: int(env[k]) for k in knobs[1:] if k in env}


def _scaled(base: dict[str, int], scale: float) -> dict[str, str]:
    return {k: str(max(1, round(v * scale))) for k, v in base.items()}


def _next_scale(points: list[tuple[float, float]], target: float) -> float:
    """The next scale factor from (scale, median) points. Once two points bracket the target: linear
    interpolation between the closest bracketing pair (regula falsi), which copes with LightRAG's threshold
    (below some total cap its context collapses to a few tokens, so a ratio or log step overshoots). Before
    that: the last scale times target/median, within CAL_STEP."""
    below = [(s, m) for s, m in points if m < target]
    above = [(s, m) for s, m in points if m > target]
    if below and above:
        (s0, m0), (s1, m1) = max(below, key=lambda p: p[1]), min(above, key=lambda p: p[1])
        if m0 != m1:
            return s0 + (target - m0) * (s1 - s0) / (m1 - m0)
    s, m = points[-1]
    step = target / m if m > 0 else CAL_STEP[1]
    return s * min(max(step, CAL_STEP[0]), CAL_STEP[1])


def calibrate_caps(
    target: float,
    base: dict[str, dict[str, int]],
    start: dict[str, float | None],
    measure: Callable[[int, dict[str, dict[str, str]]], dict[str, dict]],
    max_iterations: int = CAL_MAX_ITERATIONS,
    tolerance: float = CAL_TOLERANCE,
) -> dict[str, dict]:
    """Search each capped arm's knob scale until its median realized context is within `tolerance` of `target`.

    `base`: arm -> knobs at the selected configuration (scale 1); `start`: arm -> its median there (from the
    pilot runs), which gives the first guess, scale target/median. `measure(iteration, {arm: knobs})` runs the
    pending arms once at those knobs and returns {arm: {"median", "per_cell", "log_dir"}}. Every arm still out of
    tolerance is re-measured at the next scale (`_next_scale`), at most `max_iterations` times. Returns, per
    arm: converged, the knobs (the first in-tolerance iteration's; else the closest), the median and every
    iteration."""
    lo, hi = target * (1 - tolerance), target * (1 + tolerance)
    points = {a: ([(1.0, float(start[a]))] if start.get(a) else []) for a in base}
    scale = {a: _next_scale(points[a], target) if points[a] else 1.0 for a in base}
    out = {a: {"converged": False, "base": base[a], "start_median": start.get(a), "iterations": []} for a in base}
    pending = list(base)
    for it in range(1, max_iterations + 1):
        if not pending:
            break
        knobs = {a: _scaled(base[a], scale[a]) for a in pending}
        got = measure(it, knobs)
        for a in list(pending):
            m = got[a]["median"]
            out[a]["iterations"].append({"iteration": it, "scale": round(scale[a], 4), "env": knobs[a], "median": m, "per_cell": got[a].get("per_cell"), "log_dir": got[a].get("log_dir")})
            if m is not None and lo <= m <= hi:
                out[a] |= {"converged": True, "env": knobs[a], "median": m, "per_cell": got[a].get("per_cell")}
                pending.remove(a)
                continue
            points[a].append((scale[a], float(m or 0.0)))
            scale[a] = _next_scale(points[a], target)
    for a in pending:
        best = min((i for i in out[a]["iterations"] if i["median"] is not None), key=lambda i: abs(i["median"] - target), default=None)
        out[a] |= {"env": best["env"] if best else None, "median": best["median"] if best else None, "per_cell": best["per_cell"] if best else None}
    for a in out:
        out[a] |= {"target": target, "window": [lo, hi]}
    return out


def _matched_apg_env(selected: dict, context: int) -> dict[str, str]:
    """APG* in the matched-budget secondary fills to the budget (GATE_PREREG §6.2): fill on, compose budget =
    the matched budget, and a shortlist at least APG_MATCHED_SHORTLIST_K so there are nodes to fill with."""
    k = int((selected["APG*"].get("env") or {}).get("APE_APG_SHORTLIST_K") or 12)
    return {"APE_APG_FILL": "1", "APE_APG_BUDGET": str(int(context)), "APE_APG_SHORTLIST_K": str(max(APG_MATCHED_SHORTLIST_K, k))}


def _pilot_params(run: GateRun) -> dict:
    return {
        "scale": scale(run),
        "worlds": pilot_world_specs(run),
        "cells": _pilot_cells(run),
        "sigma_cell": _pilot_sigma_cell(run),
        "calibration": {"knobs": CAL_KNOBS, "max_iterations": CAL_MAX_ITERATIONS, "tolerance": CAL_TOLERANCE, "step": CAL_STEP, "apg_shortlist_k": APG_MATCHED_SHORTLIST_K},
        "power": {"sizes": POWER_SIZES, "sims": POWER_SIMS["offline" if run.tiny else "live"]},
        "lightrag_kind": "oracle" if run.offline else "extract",
        "fake_author": run.offline,
        "builder": build_params(run),
        "env_knobs": _env_knobs(),
    }


def _pilot_projected(run: GateRun) -> float:
    """The pilot's cells, with the budget calibration at its worst case (every iteration runs)."""
    p = plan(run)
    main, pull, cal = (p.cell(c) for c in PILOT_RUN_CELLS)
    sigma = [p.cell(PILOT_SIGMA_CELL)] if _pilot_sigma_cell(run) else []
    return project(run, [p.cell(PILOT_BUILD_CELL), main, pull, *sigma]) + CAL_MAX_ITERATIONS * project(run, [cal])


def _fmt_env(env: dict | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (env or {}).items()) or "defaults"


def _cap_item(caps: dict[str, dict], arm: str) -> str:
    """pilot.json's line for a capped arm's matched-budget setting; a miss says so."""
    if arm not in caps:
        return "n/a"
    c = caps[arm]
    line = f"{_fmt_env(c['env'])} (median {c['median']:.0f} tokens)"
    return line if c["converged"] else line + f"; NOT MATCHED: outside ±{CAL_TOLERANCE:.0%} of {c['target']:.0f} after {len(c['iterations'])} iterations"


def _pilot(run: GateRun, record: dict) -> None:
    from .analysis.pilot import load_task_means, power_report, realized_tokens, variance_components
    from .budget import calibrate

    selected = read_selected(run.selected_path)
    arms, env = selected_arms(selected), selected_env(selected)
    cells = _pilot_cells(run)
    main, pull, cal = (cells[c] for c in PILOT_RUN_CELLS)
    gp = gate_profile(run)
    pdir = run.phase_dir("pilot")
    # 1. The pilot worlds.
    build_world_set(run, record, "pilot", "pilot", pilot_world_specs(run))
    models = gate_models(run, gp)
    # 2. The selected arms and the diagnostics except S7, push and pull, with every selection's knobs. Knobs
    #    are not part of Inspect's task identity, so the log dir is keyed by them.
    main_dir = pdir / "logs" / f"main-{_digest({'env': env, 'arms': arms})}"
    record["log_dirs"] = [_show(main_dir)]
    logs: list[str] = []
    with _environ(env):  # tasks are created under the knobs they run with (they record them)
        by_epochs: dict[int, list] = {}
        for spec, only in ((main, [a for a in main["arms"] if a != "S7"]), (pull, None)):
            by_epochs.setdefault(int(spec["epochs"]), []).extend(_gate_tasks(spec, arms, "pilot", only))
        print(f"[pilot] {sum(map(len, by_epochs.values()))} task(s): {', '.join(main['arms'])} x {main['cells']} (push) + {', '.join(pull['arms'])} x {pull['cells']} (pull)", flush=True)
        per_sample = sample_usd(project(run, [plan(run).cell(c) for c in PILOT_RUN_CELLS[:2]]), [(t, e) for e, t in by_epochs.items()])
        for epochs, tasks in by_epochs.items():
            logs += run_gate_tasks(run, gp, models, tasks, main_dir / f"epochs-{epochs}", epochs, "pilot runs", per_sample)
    tokens = realized_tokens(logs, "push")
    # 3. S7 targets: APG*'s median realized context per cell; then S7.
    apg = arms["APG*"]
    targets = {}
    for c in main["cells"]:
        m = _medians(tokens, apg, [c])[0]
        if m is None:
            raise PhaseError(f"no realized context for {apg} in {c} (no compiles in its pilot logs): cannot size S7")
        targets[c] = max(1, round(m))
    _write_json(run.s7_targets_path, targets)
    record["outputs"] |= {"s7_targets": _show(run.s7_targets_path)}
    if "S7" in main["arms"]:
        s7_knobs = env | s7_env(selected)
        s7_dir = pdir / "logs" / f"s7-{_digest({'env': s7_knobs, 'targets': targets})}"
        record["log_dirs"].append(_show(s7_dir))
        with _environ(s7_knobs):
            s7_logs = run_gate_tasks(run, gp, models, _gate_tasks(main, arms, "pilot", ["S7"]), s7_dir, int(main["epochs"]), "pilot S7 runs", per_sample)
        logs += s7_logs
        tokens |= realized_tokens(s7_logs, "push")
    # 3b. The σ cell (D-026): APG* and LGR* on the pilot worlds after the main pilot's, push, for σ only. S7 targets
    #     and the calibration starts above come from the main pilot's worlds, unchanged.
    sigma_logs: list[str] = []
    if sigma := _pilot_sigma_cell(run):
        sigma_dir = pdir / "logs" / f"sigma-{_digest({'env': env, 'arms': arms})}"
        record["log_dirs"].append(_show(sigma_dir))
        with _environ(env):
            tasks = _gate_tasks(sigma, arms, "pilot")
            per = sample_usd(project(run, [plan(run).cell(PILOT_SIGMA_CELL)]), [(tasks, int(sigma["epochs"]))])
            print(f"[pilot] σ cell: {', '.join(sigma['arms'])} x {sigma['cells']} on pilot worlds {int(sigma['world_offset']) + 1}-{int(sigma['world_offset']) + int(sigma['worlds'])} (push)", flush=True)
            sigma_logs = run_gate_tasks(run, gp, models, tasks, sigma_dir, int(sigma["epochs"]), "pilot σ runs", per)
        logs += sigma_logs
    # 4. Matched-budget calibration of the capped arms (LGR*, S3s) on the budget-cal cell's worlds.
    context = int(cal["context"])
    capped = [a for a in cal["arms"] if a in CAL_KNOBS]
    base = {a: _cal_base(a, env) for a in capped}
    start = {a: _medians(tokens, arms[a], cal["cells"])[0] for a in capped}
    cal_dir = pdir / "budget-cal"
    record["log_dirs"].append(_show(cal_dir))

    def measure(iteration: int, knobs: dict[str, dict[str, str]]) -> dict[str, dict]:
        d = cal_dir / f"iter{iteration}-{_digest({'env': env, 'knobs': knobs})}"
        print(f"[pilot] budget calibration {iteration}: " + "; ".join(f"{a} {_fmt_env(k)}" for a, k in knobs.items()), flush=True)
        with _environ(env | {k: v for kn in knobs.values() for k, v in kn.items()}):
            tasks = _gate_tasks(cal, arms, "pilot", list(knobs))
            per = sample_usd(project(run, [plan(run).cell(PILOT_RUN_CELLS[2])]), [(tasks, int(cal["epochs"]))])
            got = realized_tokens(run_gate_tasks(run, gp, models, tasks, d, int(cal["epochs"]), f"budget calibration {iteration}", per), "push")
        return {a: dict(zip(("median", "per_cell"), _medians(got, arms[a], cal["cells"]), strict=True)) | {"log_dir": _show(d)} for a in knobs}

    caps = calibrate_caps(context, base, start, measure)
    if unmeasured := [a for a in capped if caps[a]["env"] is None]:
        raise PhaseError(f"budget calibration: {unmeasured} delivered no context in any iteration (no compiles in their logs, see {_show(cal_dir)}): the arm is broken, not uncalibrated")
    failed = [a for a in capped if not caps[a]["converged"]]
    calibration = {
        "context": context,
        "tolerance": CAL_TOLERANCE,
        "cells": list(cal["cells"]),
        # `matched` is false when a capped arm's closest caps miss the window: the matched-budget secondary still runs
        # with them, and is reported as not matched (GATE_PREREG §6.2, D-022); it never blocks the pilot or the verdict.
        "matched": not failed,
        "not_converged": failed,
        "arms": {"APG*": {"arm": arms["APG*"], "env": _matched_apg_env(selected, context), "method": "fill (not capped); PC4 checks its realized median at test time"}}
        | {a: {"arm": arms[a], "method": "capped: knobs scaled together until the median realized context is in the window"} | caps[a] for a in capped},
    }
    _write_json(cal_dir / "calibration.json", calibration)
    record["budget_calibration"] = {a: {"converged": c["converged"], "env": c["env"], "median": c["median"], "iterations": len(c["iterations"])} for a, c in caps.items()}
    record["budget_calibration_matched"] = not failed
    if failed:
        detail = "; ".join(f"{a}: " + ", ".join(f"[{_fmt_env(i['env'])}] -> {i['median']}" for i in caps[a]["iterations"]) for a in failed)
        record["warnings"].append(
            f"budget calibration{' (smoke sizes)' if run.smoke else ''}: {failed} did not land within ±{CAL_TOLERANCE:.0%} of {context} tokens in "
            f"{CAL_MAX_ITERATIONS} iterations ({detail}). The closest caps are recorded and the matched-budget secondary runs with them, marked not "
            f"matched (budget_calibration.yaml `matched: false`). The calibration knobs apply to that secondary only; the selections (selected.yaml) "
            f"define LGR* and S3s and are never changed to make it converge."
        )
    for a in capped:
        if not caps[a]["converged"]:
            continue
        lo, hi = caps[a]["window"]
        if off := [c for c, m in (caps[a]["per_cell"] or {}).items() if m is not None and not lo <= m <= hi]:
            record["warnings"].append(f"budget calibration: {a}'s pooled median is in the window, but its median in {off} is not")
    header = (
        f"# Matched-budget settings (GATE_PREREG §6.2, PC4), written by `python -m ape.run_gate pilot --run-id {run.run_id}`"
        f"{' (OFFLINE: mock models, oracle LightRAG)' if run.offline else ''}.\n"
        "# The matched-budget secondary applies the selected envs (selected.yaml), then each arm's `env` here on top.\n"
    )
    run.budget_calibration_path.parent.mkdir(parents=True, exist_ok=True)
    run.budget_calibration_path.write_text(header + yaml.safe_dump(json.loads(json.dumps(calibration)), sort_keys=False))  # plain copies: no YAML aliases
    record["outputs"] |= {"budget_calibration": _show(run.budget_calibration_path)}
    # 5. Power re-simulation with the pilot's world variance components (APG* against LGR*, push): the main pilot's
    #    worlds plus the σ cell's (D-026), so 8 worlds per cell at full scale.
    test = plan(run).cell("gate.test.f7").spec
    main_logs = [f for f in logs if Path(f).resolve().is_relative_to(main_dir.resolve())]
    tm = load_task_means(main_logs + sigma_logs, require_cost=not run.offline, delivery="push")
    vc = variance_components(tm, apg, arms["LGR*"])
    power = {"variance_components": vc} | power_report(
        vc, POWER_SIZES, int(test["worlds"]), int(test["tasks_per_world"]), int(test["epochs"]), POWER_SIMS["offline" if run.tiny else "live"]
    )
    _write_json(pdir / "power.json", power)
    if not vc["estimable"]:
        record["warnings"].append(f"power: σ_w and σ_g are not estimable from the pilot ({vc['note']}); the priors are used")
    # 6. The cost model, recalibrated from the pilot's arm runs (not the calibration iterations, which run off-plan budgets).
    measured = calibrate(logs, out_path=run.measured_out_path)
    record["outputs"] |= {"power": _show(pdir / "power.json"), "measured": _show(run.measured_out_path)}
    # 6b. PC5 early warning (D-027): the pilot's arm runs' harness-error and cap-hit rates, so a broken harness shows
    #     before the test phase (where PC5 fails a run only on evidence). Warnings only; the pilot never fails on them.
    from .analysis.gate_stats import MAX_CAP_HIT_RATE, MAX_ERROR_RATE, harness_rates, load_results

    harness = harness_rates(load_results(logs, require_cost=not run.offline), by=("arm", "delivery"), cap_exempt=("S7",))
    harness_warnings = [
        f"PC5 early warning: {h['arm']} ({h['delivery']}) errors {h['error_rate']:.1%}, cap hits {h['cap_hit_rate']:.1%} over {h['samples']} pilot samples "
        f"(limits {MAX_ERROR_RATE:.0%} and {MAX_CAP_HIT_RATE:.0%}): fix the harness before the test phase"
        for h in harness
        if h["error_over"] or h["cap_hit_over"]
    ]
    record["warnings"] += harness_warnings
    # 7. pilot.json: everything the analyst transcribes into GATE_PREREG.md, keyed by its placeholder labels.
    success = {a: {c: float(v) for c, v in tm[a].dropna().groupby(level="cell").mean().items()} for a in tm.columns}
    realized = {a: _medians(tokens, a, main["cells"])[1] for a in sorted({a for a, _ in tokens})}
    pw = power["scenarios"]["pilot"]
    rec = power["recommended_worlds_per_cell"]
    bq = read_build_quality(run)  # D-017, measured on the dev builds
    ci80, informative = vc.get("ci80") or {}, vc.get("ci80_upper_informative") or {}

    def _sigma(key: str) -> str:
        """A σ point estimate with its 80% interval and, when flagged, that its upper end is not informative."""
        if key not in ci80:
            return f"{pw[key]:.2f}"
        flag = "" if informative.get(key, True) else "; upper end not informative"
        return f"{pw[key]:.2f} (80% {ci80[key][0]:.2f}–{ci80[key][1]:.2f}{flag})"

    cons = power["scenarios"]["conservative"]
    items = {
        "test worlds per cell": f"{rec if rec is not None else 'analyst decides'} ({power['recommendation']}; {worlds_per_cell_note(bq)})",
        "builder": bq["recommendation"] if bq else "D-017 build check not recorded (build-dev/build_quality.json missing): re-run build-dev",
        "pilot σ_w, σ_g and power": f"σ_w {_sigma('sigma_w')}, σ_g {_sigma('sigma_g')} ({'pilot estimates' if vc['estimable'] else 'priors: not estimable'}); power at Δ = 0: "
        + ", ".join(f"{n} worlds {p:.2f}" for n, p in pw["power"].items())
        + f"; at the least favourable corner of the intervals ({cons.get('corner', 'priors')}): "
        + ", ".join(f"{n} worlds {p:.2f}" for n, p in cons["power"].items()),
        "S7 targets per cell": ", ".join(f"{c} {t}" for c, t in targets.items()) + " tokens",
        "APG* matched-budget settings": _fmt_env(calibration["arms"]["APG*"]["env"]),
        "LGR* matched-budget caps": _cap_item(caps, "LGR*"),
        "S3s matched-budget cap": _cap_item(caps, "S3s"),
    } | {f"{key} selection": f"{selected[key]['arm']} ({selected[key]['candidate']}: {_fmt_env(selected[key].get('env'))})" for _, key in TUNED_SYSTEMS}
    summary = {
        "run_id": run.run_id,
        "offline": run.offline,
        "arms": arms,
        "selected_env": env,
        "success": success,
        "realized_median_tokens": realized,
        "s7_targets": targets,
        "budget_calibration": record["budget_calibration"],
        "budget_calibration_matched": calibration["matched"],
        "s7_delivery": "per step (APG* is per-step)" if s7_env(selected)["APE_S7_PER_STEP"] == "1" else "once per task (APG* is per-query)",
        "power": {k: power[k] for k in ("decision", "recommended_worlds_per_cell", "recommendation")}
        | {"pilot": pw, "conservative": power["scenarios"]["conservative"], "optimistic": power["scenarios"]["optimistic"], "sigma_worlds_per_cell": vc.get("worlds_per_cell")},
        "harness": {"rates": harness, "warnings": harness_warnings},
        "cost_model_entries": len(measured["entries"]),
        "build_quality": {"verdict": bq["verdict"], "offline": bq["offline"], "path": _show(build_quality_path(run))} if bq else None,
        "prereg_items": items,
        "files": {k: record["outputs"][k] for k in ("s7_targets", "budget_calibration", "power", "measured")},
    }
    _write_json(pdir / "pilot.json", summary)
    record["outputs"] |= {"pilot": _show(pdir / "pilot.json")}
    record["s7_targets"], record["power"] = targets, summary["power"]


# test seeds ----------------------------------------------------------------------------------------


def test_seed_count(run: GateRun) -> int:
    """Seeds a run's test worlds use: the largest world count of any test build cell (world i has seed base+i)."""
    p = plan(run)
    count = max(int(n) for cell_id in TEST_BUILD_CELLS.values() for n in p.cell(cell_id).spec["worlds"].values())
    if count > SEED_BLOCK:
        raise PhaseError(f"run_plan.yaml builds {count} test worlds per cell, more than a seed block ({SEED_BLOCK}); raise SEED_BLOCK in worlds/generate.py")
    return count


def used_test_seed_blocks(run: GateRun) -> list[dict]:
    """Test-seed blocks other live gate runs froze: their runs_root/<id>/freeze.json (`test_seeds`; a live freeze
    from before seed bases existed used 3000) and PROVENANCE.md's machine-readable markers, which survive a lost
    runs/ directory. Offline rehearsals use OFFLINE_TEST_SEED_BASE and never count."""
    used: dict[tuple[str, int], dict] = {}
    for path in sorted(run.runs_root.glob("*/freeze.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        rid = rec.get("run_id") or path.parent.name
        if rid == run.run_id or rec.get("offline"):
            continue
        seeds = rec.get("test_seeds") or {"base": FIRST_TEST_SEED_BASE, "count": SEED_BLOCK}
        used[(rid, int(seeds["base"]))] = {"run": rid, "base": int(seeds["base"]), "count": int(seeds["count"]), "source": _show(path)}
    text = run.provenance_path.read_text() if run.provenance_path.is_file() else ""
    for m in TEST_SEED_MARKER.finditer(text):
        rid, base, count = m.group(1), int(m.group(2)), int(m.group(3))
        if rid != run.run_id:
            used.setdefault((rid, base), {"run": rid, "base": base, "count": count, "source": _show(run.provenance_path)})
    return sorted(used.values(), key=lambda u: u["base"])


def _overlaps(base: int, count: int, used: list[dict]) -> list[dict]:
    return [u for u in used if base < u["base"] + u["count"] and u["base"] < base + count]


def choose_test_seed_base(run: GateRun) -> tuple[int, list[str]]:
    """(base, problems): the test-seed block this run would freeze. Offline: OFFLINE_TEST_SEED_BASE (a rehearsal never
    materialises a real block). Live: `run.test_seed_base` if given, else 3000 if no other frozen run used it, else the
    next free block of SEED_BLOCK seeds. A given base that overlaps another run's block is a problem, never adjusted."""
    count = test_seed_count(run)
    if run.offline:
        return (run.test_seed_base if run.test_seed_base is not None else OFFLINE_TEST_SEED_BASE), []
    used = used_test_seed_blocks(run)
    if run.test_seed_base is not None:
        base, problems = int(run.test_seed_base), []
        if not FIRST_TEST_SEED_BASE <= base <= OFFLINE_TEST_SEED_BASE - count:
            problems.append(f"--test-seed-base {base}: test seeds live in [{FIRST_TEST_SEED_BASE}, {OFFLINE_TEST_SEED_BASE}) ({OFFLINE_TEST_SEED_BASE}+ is the offline rehearsals')")
        if clash := _overlaps(base, count, used):
            problems.append(f"--test-seed-base {base}: seeds {base}-{base + count - 1} overlap " + "; ".join(f"run {u['run']!r}'s {u['base']}-{u['base'] + u['count'] - 1} ({u['source']})" for u in clash) + ": those worlds were already generated (and possibly analysed)")
        return base, problems
    base = FIRST_TEST_SEED_BASE
    while _overlaps(base, count, used):
        base += SEED_BLOCK
    return base, []


def run_test_seed_base(run: GateRun) -> int:
    """The run's test-seed base: the frozen one (freeze.json; a freeze from before seed bases existed used 3000), or
    before the freeze the one the freeze would choose."""
    freeze = read_freeze(run)
    if freeze is not None:
        return int((freeze.get("test_seeds") or {}).get("base", FIRST_TEST_SEED_BASE))
    return choose_test_seed_base(run)[0]


# freeze --------------------------------------------------------------------------------------------


def prereg_body_start(text: str) -> int:
    """Offset of the body: the first `## ` heading. The header before it is instructions, never checked."""
    m = re.search(r"^## ", text, flags=re.M)
    return m.start() if m else 0


def prereg_placeholders(text: str) -> list[dict]:
    """Every unfilled item in the body: {line, marker, kind, label} (label None for a malformed marker)."""
    start = prereg_body_start(text)
    out = []
    for m in PLACEHOLDER.finditer(text, start):
        item = PLACEHOLDER_ITEM.match(text, m.start())
        line = text.count("\n", 0, m.start()) + 1
        if item:
            marker, label = item.group(0), (item.group(2) or "").strip() or None
        else:
            marker, label = text[m.start() : text.find("\n", m.start())].strip(), None
        out.append({"line": line, "marker": marker, "kind": m.group(1), "label": label})
    return out


def frozen_files(run: GateRun, prereg: Path) -> dict[str, Path]:
    """What the freeze hashes: the pre-registration as frozen, the config inputs and the tune/pilot outputs."""
    return (
        {"GATE_PREREG.md": prereg}
        | {f"config/{n}": run.config(n) for n in FROZEN_CONFIG}
        | {f"config/{n}": run.out_config_dir / n for n in FROZEN_OUTPUTS}
        | {n: ROOT / n for n in FROZEN_CODE}
    )


def frozen_changes(record: dict) -> list[str]:
    """The frozen files that are missing or no longer match their hash, as `key (path): what`."""
    out = []
    for key, f in record["files"].items():
        now = _sha256(_resolve(f["path"]))
        if now != f["sha256"]:
            out.append(f"{key} ({f['path']}): {'missing' if now is None else 'changed'}")
    return out


def read_freeze(run: GateRun) -> dict | None:
    return json.loads(run.freeze_path.read_text()) if run.freeze_path.is_file() else None


def require_frozen(run: GateRun) -> dict:
    """The freeze record, if the run is frozen and every frozen file still matches its hash; else PhaseError.
    build-test and test call this before anything that depends on the frozen design."""
    record = read_freeze(run)
    if record is None:
        raise PhaseError(f"run {run.run_id!r} is not frozen: python -m ape.run_gate freeze --run-id {run.run_id}{' --offline' if run.offline else ''}")
    if changed := frozen_changes(record):
        raise PhaseError(f"frozen file(s) changed since the freeze at {record['frozen_at']}: " + "; ".join(changed) + f". Restore them; {DEVIATION}")
    return record


def _refuse_if_frozen(name: str) -> Callable[[GateRun], str | None]:
    def refuse(run: GateRun) -> str | None:
        if run.freeze_path.is_file():
            return f"{name}: run {run.run_id!r} is frozen ({_show(run.freeze_path)}) and {name} would rewrite frozen inputs; {DEVIATION}"
        return None

    return refuse


def _smoke_freeze_refusal(run: GateRun) -> str:
    """A smoke run's freeze refusal: it never freezes; the draft pre-registration's open items are listed too."""
    text = run.prereg_path.read_text() if run.prereg_path.is_file() else ""
    items = prereg_placeholders(text)
    msg = f"freeze refused:\n  - run {run.run_id!r} is a smoke run (SMOKE_SCALE, FIX_PLAN FX-8): smoke runs never freeze, so they never reach the test split"
    if items:
        listing = "\n".join(f"    line {p['line']}: {p['marker']}" for p in items)
        msg += f"\n  - {_show(run.prereg_path)} still has {len(items)} unfilled item(s):\n{listing}"
    return msg


def _refuse_refreeze(run: GateRun) -> str | None:
    """Before any record (and before the budget guard): a smoke run never freezes; a frozen run is frozen once. A
    freeze interrupted after writing freeze.json (its manifest not complete, the frozen files unchanged) is not
    refused: `_freeze` completes it (`_complete_freeze`)."""
    if run.smoke:
        return _smoke_freeze_refusal(run)
    record = read_freeze(run)
    if record is None:
        return None
    changed = frozen_changes(record)
    if not changed and (read_manifest(run, "freeze") or {}).get("status") not in COMPLETE:
        return None
    what = ("frozen file(s) changed since: " + "; ".join(changed)) if changed else "its inputs are unchanged (--force does not re-freeze)"
    return f"freeze: run {run.run_id!r} was frozen at {record['frozen_at']} ({_show(run.freeze_path)}); a run is frozen once, and {what}. {DEVIATION[0].upper() + DEVIATION[1:]}"


def _rehearsal_prereg(run: GateRun, text: str, items: dict[str, str]) -> tuple[Path, list[dict]]:
    """Offline: a copy of the pre-registration with every body placeholder replaced by an OFFLINE-REHEARSAL
    value (the pilot's value for its label where pilot.json has one)."""
    start = prereg_body_start(text)
    replaced: list[dict] = []

    def fill(m: re.Match) -> str:
        label = (m.group(2) or "").strip()
        value = items.get(label, "unfilled")
        replaced.append({"marker": m.group(0), "value": value})
        return f"OFFLINE-REHEARSAL({value})"

    copy = run.work_dir / "GATE_PREREG.md"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_text(text[:start] + PLACEHOLDER_ITEM.sub(fill, text[start:]))
    return copy, replaced


FREEZE_RESTS_ON = ("build-dev", "tune", "anchor", "pilot")  # the phases whose outputs the freeze fixes (O2)


def stale_upstream(run: GateRun) -> list[str]:
    """The phases the freeze rests on that are not current: missing, not complete, or whose fingerprint (recomputed
    from their inputs, params and upstream now, as `run_phase` would) differs from the one they ran with; a
    pilot that ran on an older tune shows here, since tune's fingerprint is part of the pilot's."""
    out = []
    for name in FREEZE_RESTS_ON:
        m = read_manifest(run, name)
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


ROLE_KINDS = ("primary", "extension", "fix_cycle")
ROLE_STAGE1_LABEL = {"extension": "INCONCLUSIVE", "fix_cycle": "NO_GO"}  # what the referenced run's verdict must be (§8)


def run_role(run: GateRun) -> dict:
    """The role this run asks to freeze with: {kind: primary | extension | fix_cycle, of: <run id> | None}."""
    if run.extension_of and run.fix_cycle_of:
        raise PhaseError("a gate run is an extension or a fix cycle, not both: pass one of --extension-of and --fix-cycle-of")
    if run.extension_of:
        return {"kind": "extension", "of": run.extension_of}
    if run.fix_cycle_of:
        return {"kind": "fix_cycle", "of": run.fix_cycle_of}
    return {"kind": "primary", "of": None}


def frozen_role(record: dict | None) -> dict:
    """A freeze record's role; a freeze from before roles were recorded was a primary gate run."""
    role = (record or {}).get("role") or {}
    return {"kind": role.get("kind", "primary"), "of": role.get("of")}


def role_problems(run: GateRun, role: dict) -> list[str]:
    """Why `run` may not freeze with `role` (§8; D-027): an extension or fix cycle names a frozen, analysed run of the
    right verdict that is itself a primary run, and is the only one of its kind for that run (two extensions would
    spend the extension α twice). Other runs are found by scanning runs_root/*/freeze.json."""
    if role["kind"] == "primary":
        return []
    of, kind, problems = role["of"], role["kind"], []
    if of == run.run_id:
        return [f"--{kind.replace('_', '-')}-of names this run itself"]
    ref = run.runs_root / of
    ref_freeze = json.loads((ref / "freeze.json").read_text()) if (ref / "freeze.json").is_file() else None
    if ref_freeze is None:
        problems.append(f"{kind} of {of!r}: {_show(ref / 'freeze.json')} missing; the run it extends or fixes must be a frozen gate run in {_show(run.runs_root)}")
    else:
        if (ref_kind := frozen_role(ref_freeze)["kind"]) != "primary":
            article = "an" if ref_kind == "extension" else "a"
            problems.append(f"{kind} of {of!r}: that run is itself {article} {ref_kind.replace('_', ' ')}; §8 allows one extension or fix cycle of the primary gate run")
        if bool(ref_freeze.get("offline")) != run.offline:
            problems.append(f"{kind} of {of!r}: that run is {'offline' if ref_freeze.get('offline') else 'live'} and this one is not")
    decision = ref / "report" / "decision.json"
    label = json.loads(decision.read_text())["verdict"]["label"] if decision.is_file() else None
    if label is None:
        problems.append(f"{kind} of {of!r}: it has no decision report ({_show(decision)}); analyse it first")
    elif label != ROLE_STAGE1_LABEL[kind]:
        problems.append(f"{kind} of {of!r}: its verdict is {label}; §8 allows a{'n' if kind == 'extension' else ''} {kind.replace('_', ' ')} only after {ROLE_STAGE1_LABEL[kind]}")
    for path in sorted(run.runs_root.glob("*/freeze.json")):
        if path.parent.name == run.run_id:
            continue
        try:
            other = frozen_role(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
        if other == role:
            problems.append(f"{path.parent.name!r} is already the {kind.replace('_', ' ')} of {of!r} ({_show(path)}); §8 allows one")
    return problems


def _freeze_inputs(run: GateRun) -> dict[str, Path]:
    return {"GATE_PREREG.md": run.prereg_path} | {k: p for k, p in frozen_files(run, run.prereg_path).items() if k != "GATE_PREREG.md"}


def _freeze_provenance_lines(run: GateRun, freeze: dict) -> list[str]:
    """The freeze's record in PROVENANCE.md (its heading names the run and `frozen_at`, so a completion can find it)."""
    git, seed_base, pc1_accepted = freeze["git"], freeze["test_seeds"]["base"], freeze.get("pc1_accepted")
    return [
        f"\n## Gate freeze: run `{run.run_id}` ({freeze['frozen_at']}){' (OFFLINE REHEARSAL)' if run.offline else ''}\n",
        f"- **Commit:** `{git['commit']}`; analysis code (`src/ape/analysis/`, `src/ape/analyze_gate.py`) at `{freeze['analysis_commit']}`; apg-core `{freeze['apg_core']['installed_commit']}` (pin `{APG_PIN}`).",
        "- **Frozen files** (sha256):",
        *[f"  - `{f['path']}`{'' if f['path'] == k else f' ({k})'}: `{f['sha256']}`" for k, f in freeze["files"].items()],
        f"- **Design knobs** (APE_* set when tune and pilot ran; build-test and test refuse others): {_fmt_env(freeze.get('design_env'))}.",
        *([f"- **PC1 failed and was accepted at the freeze:** {pc1_accepted['reason']}"] if pc1_accepted else []),
        *(
            [f"- **Role:** the {r['kind'].replace('_', ' ')} of run `{r['of']}` (GATE_PREREG §8); analysed at that stage's α."]
            if (r := frozen_role(freeze))["kind"] != "primary"
            else []
        ),
        f"- **Test seeds:** {seed_base}-{freeze['test_seeds']['last']} (block base {seed_base}). A later gate run (an extension or a fix cycle) freezes a fresh block. "
        f"<!-- ape:test-seeds run={run.run_id} base={seed_base} count={freeze['test_seeds']['count']} -->",
        f"- **Code:** build-test and test run only at `{git['commit']}` ({', '.join(CODE_PATHS)} unchanged).",
        f"- **Record:** `{_show(run.freeze_path)}`. {DEVIATION[0].upper() + DEVIATION[1:]}.",
        "",
    ]


def _freeze_provenance_path(run: GateRun) -> Path:
    return run.work_dir / "PROVENANCE.freeze.md" if run.offline else run.provenance_path


def _write_freeze_provenance(run: GateRun, freeze: dict) -> Path:
    """Append the freeze's record to PROVENANCE.md (offline: work/PROVENANCE.freeze.md) unless it is already there."""
    path = _freeze_provenance_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    if f"## Gate freeze: run `{run.run_id}` ({freeze['frozen_at']})" not in (path.read_text() if path.is_file() else ""):
        with path.open("a") as f:
            f.write("\n".join(_freeze_provenance_lines(run, freeze)))
    return path


def _record_freeze(run: GateRun, record: dict, freeze: dict, provenance: Path) -> None:
    record["outputs"] |= {"freeze": _show(run.freeze_path), "provenance": _show(provenance)}
    record["frozen"] = {k: freeze.get(k) for k in ("files", "rehearsal", "analysis_commit", "code_commit", "test_seeds", "design_env")} | {"role": frozen_role(freeze)}
    if not run.offline:
        record["warnings"].append(f"commit {_show(run.provenance_path)} (the freeze record) before build-test")


def _complete_freeze(run: GateRun, record: dict, freeze: dict) -> None:
    """A freeze interrupted after writing freeze.json (`_refuse_refreeze` verified its frozen files are unchanged): add
    PROVENANCE.md's record if it is missing and finish the phase, so the run is not stuck between the two files."""
    provenance = _write_freeze_provenance(run, freeze)
    record["warnings"].append(f"completed a freeze interrupted after freeze.json was written ({freeze['frozen_at']}); PROVENANCE.md checked")
    if freeze.get("rehearsal") and (copy := run.work_dir / "GATE_PREREG.md").is_file():
        record["outputs"]["prereg_rehearsal"] = _show(copy)
    _record_freeze(run, record, freeze, provenance)


def _freeze(run: GateRun, record: dict) -> None:
    if not run.smoke and (existing := read_freeze(run)) is not None:
        _complete_freeze(run, record, existing)
        return
    problems: list[str] = []
    text = run.prereg_path.read_text() if run.prereg_path.is_file() else ""
    if not text:
        problems.append(f"{_show(run.prereg_path)} is missing or empty")
    placeholders = prereg_placeholders(text)
    record["placeholders"] = placeholders
    prereg, rehearsal = run.prereg_path, None
    if run.smoke:  # checked first: a smoke run writes nothing here, not even a rehearsal copy
        problems.append(f"run {run.run_id!r} is a smoke run (SMOKE_SCALE, FIX_PLAN FX-8): smoke runs never freeze, so they never reach the test split")
    if placeholders:
        if run.offline and not run.smoke:
            items = (json.loads((run.phase_dir("pilot") / "pilot.json").read_text()) or {}).get("prereg_items", {})
            prereg, replaced = _rehearsal_prereg(run, text, items)
            if left := prereg_placeholders(prereg.read_text()):
                problems.append("malformed placeholder(s) the rehearsal cannot fill: " + "; ".join(f"line {p['line']}: {p['marker']}" for p in left))
            rehearsal = replaced
            record["warnings"].append(f"offline rehearsal: {len(replaced)} placeholder(s) in {_show(run.prereg_path)} filled in the copy {_show(prereg)}")
        else:
            listing = "\n".join(f"    line {p['line']}: {p['marker']}" for p in placeholders)
            problems.append(f"{_show(run.prereg_path)} still has {len(placeholders)} unfilled item(s); fill each ({_show(run.phase_dir('pilot') / 'pilot.json')} lists the pilot values):\n{listing}")
    files = frozen_files(run, prereg)
    if missing := [f"{k} ({_show(p)})" for k, p in files.items() if not p.is_file()]:
        problems.append(f"missing frozen input(s): {missing}")
    changes = git_tracked_changes()
    if changes is None:
        (record["warnings"] if run.offline else problems).append("git is unavailable: the freeze must record a clean commit")
    elif changes:
        msg = f"the git tree has uncommitted changes {changes[:10]}{' ...' if len(changes) > 10 else ''}: commit them so the frozen design traces to a commit"
        (record["warnings"] if run.offline else problems).append(msg)
    # O2: the phases the freeze rests on must be current, i.e. a re-run of each would be a skip. A tune re-run
    # after the pilot would otherwise freeze a new APG*/LGR* with S7 targets and caps sized for the old one.
    if stale := stale_upstream(run):
        (record["warnings"] if run.offline else problems).append("not current, so the frozen design would not be what ran: " + "; ".join(stale))
    seed_base, seed_problems = choose_test_seed_base(run)
    problems += seed_problems
    role = run_role(run)
    problems += role_problems(run, role)
    anchor = read_manifest(run, "anchor") or {}
    pc1 = anchor.get("pc1_pass")
    pc1_accepted = None
    if pc1 is not True and anchor.get("pc1_status") == "not_evaluable" and not run.offline:
        # D-025: unreadable judge replies are a harness defect to fix, never a result to accept.
        problems.append("PC1 is not evaluable (anchor/pc1.json `diagnosis`): the judge's replies could not be read; fix the harness and re-run the anchor. --accept-pc1-failure does not apply")
    elif pc1 is not True:
        reason = (run.accept_pc1_failure or "").strip()
        if reason:
            if len(reason) < PC1_REASON_MIN_CHARS:
                problems.append(f"--accept-pc1-failure needs the diagnosis in words (at least {PC1_REASON_MIN_CHARS} characters): what the anchor logs showed and why no harness defect explains the miss")
            else:
                pc1_accepted = {"reason": reason, "pc1_pass": pc1, "pc1": _show(run.phase_dir("anchor") / "pc1.json"), "accepted_at": _now()}
                record["warnings"].append(f"PC1 failed and is accepted at the freeze: {reason!r}; the report carries this caveat")
        else:
            msg = (
                f"PC1 does not pass (anchor/pc1.json, pc1_pass={pc1}). Diagnose the anchor logs first: a harness defect is fixed and "
                "the anchor re-run (a logged deviation). With no defect found, the analyst may accept the failure explicitly: "
                "--accept-pc1-failure '<diagnosis>' (GATE_PREREG §7)"
            )
            (record["warnings"] if run.offline else problems).append(msg)
    if problems:
        raise PhaseError("freeze refused:\n" + "\n".join(f"  - {x}" for x in problems))
    git = git_state()
    freeze = {
        "run_id": run.run_id,
        "frozen_at": _now(),
        "offline": run.offline,
        "rehearsal": rehearsal is not None,
        "placeholders_replaced": rehearsal or [],
        "files": {k: {"path": _show(p), "sha256": _sha256(p)} for k, p in files.items()},
        "git": git,
        "analysis_commit": _git("log", "-1", "--format=%H", "--", "src/ape/analysis/", "src/ape/analyze_gate.py"),
        "apg_core": {"installed_commit": _apg_installed_commit(), "pinned_commit": APG_PIN},
        "pilot": {"path": _show(run.phase_dir("pilot") / "pilot.json"), "sha256": _sha256(run.phase_dir("pilot") / "pilot.json")},
        "upstream": record["upstream"],
        # Every phase this freeze rests on, by fingerprint (O2 checked they are current).
        "phases": {n: (read_manifest(run, n) or {}).get("fingerprint") for n in FREEZE_RESTS_ON},
        # build-test and test run only at this commit's CODE_PATHS (O3, `code_drift`).
        "code_commit": git["commit"],
        "test_seeds": {"base": seed_base, "count": test_seed_count(run), "block": SEED_BLOCK, "last": seed_base + test_seed_count(run) - 1},
        "pc1_accepted": pc1_accepted,
        # What this run is (§8, D-027): analyze_gate derives the stage and α from this record, not from a CLI flag.
        "role": role,
        # The design knobs tune and pilot ran with (stale_upstream checked they still apply): build-test and test refuse
        # any other value (`_refuse_unless_frozen`), since freeze.json would not otherwise say what design ran.
        "design_env": _env_knobs(),
    }
    if rehearsal is not None:
        freeze["files"]["GATE_PREREG.md (draft)"] = {"path": _show(run.prereg_path), "sha256": _sha256(run.prereg_path)}
    # freeze.json first, then PROVENANCE.md: a crash between the two leaves freeze.json with an incomplete freeze
    # manifest, which `_refuse_refreeze` lets through and `_complete_freeze` finishes (never a stuck run).
    _write_json(run.freeze_path, freeze)
    provenance = _write_freeze_provenance(run, freeze)
    if rehearsal is not None:
        record["outputs"]["prereg_rehearsal"] = _show(prereg)
    _record_freeze(run, record, freeze, provenance)
    if run.accept_pc1_failure:  # recorded, but not part of the freeze's fingerprint (O4): a later `all` without it still skips
        record["accept_pc1_failure"] = run.accept_pc1_failure


# build-test ----------------------------------------------------------------------------------------


def _frozen_inputs(run: GateRun) -> dict[str, Path]:
    """The files freeze.json froze, as a phase's inputs: an edit changes the fingerprint, so the phase is not
    skipped and its freeze check (`_refuse_unless_frozen`) names the file."""
    record = read_freeze(run) or {}
    return {k: _resolve(f["path"]) for k, f in (record.get("files") or {}).items()}


def frozen_code_drift(run: GateRun) -> list[str] | None:
    """CODE_PATHS files that differ from the freeze commit (`code_drift`); None when git cannot tell."""
    freeze = read_freeze(run) or {}
    return code_drift(freeze.get("code_commit") or (freeze.get("git") or {}).get("commit"))


def design_env_changes(run: GateRun) -> list[str]:
    """The design knobs (`_env_knobs`) set now that differ from those the freeze recorded (`design_env`), as
    `KNOB: frozen 'a', now 'b'`. Empty for a freeze from before design knobs were recorded."""
    frozen = (read_freeze(run) or {}).get("design_env")
    if frozen is None:
        return []
    now = _env_knobs()
    return [f"{k}: frozen {frozen.get(k)!r}, now {now.get(k)!r}" for k in sorted(set(frozen) | set(now)) if frozen.get(k) != now.get(k)]


def _refuse_unless_frozen(name: str) -> Callable[[GateRun], str | None]:
    """build-test and test: the run is frozen, every frozen file matches its hash, and (live) the code is exactly the
    freeze commit's (O3) and the design knobs are the frozen ones (`design_env_changes`). Offline rehearsals run on a
    working tree, so drift and knobs are only warnings there (`_warn_drift`)."""

    def refuse(run: GateRun) -> str | None:
        if run.smoke:
            return f"{name}: run {run.run_id!r} is a smoke run; smoke runs never freeze, so they never reach the test split"
        try:
            require_frozen(run)
        except PhaseError as e:
            return f"{name}: {e}"
        if run.offline:
            return None
        drift = frozen_code_drift(run)
        if drift is None:
            return f"{name}: cannot verify that {', '.join(CODE_PATHS)} are the freeze commit's (git unavailable, or freeze.json has no commit)"
        if drift:
            return f"{name}: code changed since the freeze commit {(read_freeze(run) or {}).get('code_commit')}: {drift[:10]}{' ...' if len(drift) > 10 else ''}. Check out that commit to run it; {DEVIATION}"
        if knobs := design_env_changes(run):
            return f"{name}: design knob(s) differ from the ones the freeze recorded (tune and pilot ran with those): {'; '.join(knobs)}. Restore them; {DEVIATION}"
        return None

    return refuse


def _warn_drift(run: GateRun, record: dict) -> None:
    """Offline rehearsals: code drift and changed design knobs since the rehearsal freeze are warnings (a live run refuses)."""
    if not run.offline:
        return
    if drift := frozen_code_drift(run):
        record["warnings"].append(f"code differs from the rehearsal freeze's commit: {drift[:10]}{' ...' if len(drift) > 10 else ''} (a live run would refuse)")
    if knobs := design_env_changes(run):
        record["warnings"].append(f"design knob(s) differ from the rehearsal freeze's: {'; '.join(knobs)} (a live run would refuse)")


def _test_tasks_per_world(p: Plan) -> int:
    """Tasks per test world: the `tasks_per_world` of the plan's test-phase cells, which must agree."""
    values = {int(c.spec["tasks_per_world"]) for c in _test_cells(p) if "tasks_per_world" in c.spec}
    if len(values) != 1:
        raise PhaseError(f"run_plan.yaml: the gate's test cells must share one tasks_per_world (got {sorted(values)})")
    return values.pop()


def test_world_specs(run: GateRun, offline: bool | None = None) -> list[dict]:
    """The test worlds (split "test", the run's frozen seed block; `run_test_seed_base`) from the frozen plan's build
    cells (TEST_BUILD_CELLS): counts and exception style from the cell, tasks per world from the test cells
    (offline: OFFLINE_SCALE; `offline` overrides the run's mode, for the live projection). World i of every group
    has seed base+i, so the id_only worlds are paired renderings of the first gate test worlds."""
    offline = run.offline if offline is None else offline
    p = plan(run)
    tasks = _test_tasks_per_world(p)
    specs = []
    for group, cell_id in TEST_BUILD_CELLS.items():
        cell = p.cell(cell_id).spec
        for world_cell, count in cell["worlds"].items():
            family, level = world_cell.split("-", 1)
            n_tasks = tasks
            if offline:
                count, n_tasks = OFFLINE_SCALE["worlds_per_cell"], OFFLINE_SCALE["tasks_per_world"]
            style = cell.get("exception_style", "descriptive")
            specs.append({"group": group, "family": family, "level": level, "count": int(count), "n_tasks": int(n_tasks), "exception_style": style, "relational": True})
    return specs


def _build_test(run: GateRun, record: dict) -> None:
    freeze = require_frozen(run)
    record["frozen_at"] = freeze["frozen_at"]
    _warn_drift(run, record)
    base = run_test_seed_base(run)
    if not run.offline and (clash := _overlaps(base, test_seed_count(run), used_test_seed_blocks(run))):
        raise PhaseError(f"build-test: the frozen test seeds {base}+ overlap another frozen run's block: {clash}; {DEVIATION}")
    record["test_seeds"] = {"base": base, "count": test_seed_count(run)}
    with _environ({TEST_SPLIT_ENV: run.run_id, TEST_SEED_BASE_ENV: str(base)}):
        build_world_set(run, record, "build-test", "test", test_world_specs(run))


# test ----------------------------------------------------------------------------------------------


def _test_cells(p: Plan) -> list[PlanCell]:
    """The plan's gate run cells the test phase runs, in run order: PRIMARY_CELLS (the verdict's evidence)
    first, so a budget stop leaves them complete; then the others in plan order."""
    cells = [c for c in p.cells if c.study == STUDY and c.phase in TEST_RUN_PHASES and c.kind == "agent" and c.enabled]
    ids = [c.id for c in cells]
    if missing := [c for c in PRIMARY_CELLS if c not in ids]:
        raise PhaseError(f"run_plan.yaml has no enabled primary cell(s) {missing}")
    return sorted(cells, key=lambda c: (PRIMARY_CELLS.index(c.id) if c.id in PRIMARY_CELLS else len(PRIMARY_CELLS), ids.index(c.id)))


def _world_sets(run: GateRun, offline: bool) -> dict[tuple[str, str, str], dict]:
    """(split, cell, exception style) -> {count, n_tasks}: the worlds build-dev and build-test make."""
    out = {}
    for split, specs in (("dev", dev_world_specs(run, offline)), ("test", test_world_specs(run, offline))):
        for s in specs:
            out[(split, f"{s['family']}-{s['level']}", s["exception_style"])] = {"count": s["count"], "n_tasks": s["n_tasks"]}
    return out


def _cell_worlds(run: GateRun, cell: PlanCell, offline: bool) -> dict:
    """Which worlds a run cell uses: its split and exception style (plan keys, default test and descriptive), and
    per task cell how many worlds (the first, by seed): `worlds` for worlds-cells; for n_tasks-cells the
    first ceil(n_tasks / tasks per world), drawn from the same worlds as the primary (offline: OFFLINE_SCALE)."""
    s = cell.spec
    split, style = s.get("split", "test"), s.get("exception_style", "descriptive")
    sets = _world_sets(run, offline)
    n_worlds, n_tasks = {}, {}
    for task_cell in s["cells"]:
        have = sets.get((split, task_cell, style if task_cell.startswith("F7-") else "descriptive"))
        if have is None:
            raise PhaseError(f"{cell.id}: no {split} worlds for {task_cell} ({style}) in the plan's build cells")
        if offline:
            n = OFFLINE_SCALE["worlds_per_cell"]
        elif "worlds" in s:
            n = int(s["worlds"])
        else:
            n = math.ceil(int(s["n_tasks"]) / have["n_tasks"])
        if n > have["count"]:
            raise PhaseError(f"{cell.id}: needs {n} {split} worlds for {task_cell}, but the plan builds {have['count']}")
        n_worlds[task_cell], n_tasks[task_cell] = n, n * have["n_tasks"]
    return {"split": split, "exception_style": style, "n_worlds": n_worlds, "n_tasks": n_tasks}


def read_calibration(run: GateRun) -> dict:
    path = run.budget_calibration_path
    if not path.is_file():
        raise PhaseError(f"{_show(path)} missing: run the pilot phase")
    return yaml.safe_load(path.read_text()) or {}


def matched_env(calibration: dict, context: int) -> dict[str, str]:
    """The matched-budget knobs of every arm (budget_calibration.yaml), together; applied on top of the selected env.
    An arm whose calibration did not converge runs with its closest caps (the secondary is then `matched: false`)."""
    if int(calibration.get("context", -1)) != int(context):
        raise PhaseError(f"budget_calibration.yaml calibrates {calibration.get('context')} tokens, but the matched-budget cell runs at {context}: re-pilot")
    env: dict[str, str] = {}
    for key, a in (calibration.get("arms") or {}).items():
        if not a.get("env"):
            raise PhaseError(f"budget_calibration.yaml: {key} has no matched-budget setting: re-pilot")
        knobs = {k: str(v) for k, v in a["env"].items()}
        if clash := sorted(set(knobs) & set(env)):
            raise PhaseError(f"budget_calibration.yaml: {key} sets {clash}, which another arm also sets")
        env |= knobs
    return env


def test_groups(run: GateRun, selected: dict, calibration: dict, offline: bool | None = None) -> list[dict]:
    """Every eval set the test phase runs, in order: per plan cell, one group per environment.

    - `selected`: the cell's arms under every selection's knobs together (`selected_env`).
    - `matched` (a cell with `context`): every arm under the selected knobs, then the matched-budget knobs on top.
    - a variant (ARM_VARIANTS, e.g. `lgr-naive`): the base arm under the selected knobs plus the variant's.
    Knobs are not part of Inspect's task identity, so each group has its own log dir, named by its env."""
    offline = run.offline if offline is None else offline
    arms, base = selected_arms(selected), selected_env(selected)
    seed_base = run_test_seed_base(run)
    p = plan(run)
    groups = []
    for cell in _test_cells(p):
        s = cell.spec
        worlds = _cell_worlds(run, cell, offline)
        by_name: dict[str, dict] = {}
        for a in _arm_names(s["arms"]):
            if "context" in s:
                name, env, run_arm = "matched", base | matched_env(calibration, int(s["context"])), _resolve_arm(a, arms)
            elif a in ARM_VARIANTS:
                key, extra = ARM_VARIANTS[a]
                name, env, run_arm = a.lower(), base | extra, arms[key]
            else:
                name, env, run_arm = "selected", base, _resolve_arm(a, arms)
            g = by_name.setdefault(name, {"cell": cell.id, "name": name, "env": env, "arms": []})
            g["arms"].append({"declared": a, "run": run_arm})
        for g in by_name.values():
            if any(a["run"] == "S7" for a in g["arms"]):
                g["env"] = g["env"] | s7_env(selected)  # S7 mirrors APG*'s delivery schedule (D-024)
            if g["name"] == "matched":
                g["calibration"] = {"matched": bool(calibration.get("matched", True)), "not_converged": list(calibration.get("not_converged") or [])}
            # The run's frozen test-seed block; passed to the tasks only when it is not the first run's (identities kept).
            g["seed_base"] = seed_base if worlds["split"] == "test" and seed_base != FIRST_TEST_SEED_BASE else None
            g |= {
                "primary": cell.id in PRIMARY_CELLS,
                "split": worlds["split"],
                "cells": list(s["cells"]),
                "deliveries": list(s.get("deliveries", ["push"])),
                "exposure": s.get("exposure", "retrieved"),
                "exception_style": worlds["exception_style"],
                "epochs": OFFLINE_SCALE["epochs"] if offline else int(s.get("epochs", 1)),
                "n_worlds": worlds["n_worlds"],
                "n_tasks": worlds["n_tasks"],
                "dir": f"{cell.id}/{g['name']}-{_digest({'env': g['env'], 'arms': g['arms']})}",
            }
            groups.append(g)
    return groups


def _arm_names(arms: Any) -> list[str]:
    return list(arms) if isinstance(arms, (list, tuple, dict)) else [arms]


def _group_cell(run: GateRun, g: dict) -> PlanCell:
    """A group as a plan cell for the projection: its declared arms, at the live sizes it runs (an n_tasks-cell
    runs whole worlds, so its n_tasks is worlds x tasks per world)."""
    cell = plan(run).cell(g["cell"])
    live = _cell_worlds(run, cell, offline=False)
    spec = dict(cell.spec) | {"arms": [a["declared"] for a in g["arms"]]}
    if "n_tasks" in spec:
        spec["n_tasks"] = max(live["n_tasks"].values())
    return PlanCell(cell.id, cell.study, cell.phase, spec)


def _group_projected(run: GateRun, g: dict) -> float:
    return project(run, [_group_cell(run, g)])


def _group_expected_tasks(g: dict) -> int:
    return len(g["arms"]) * len(g["deliveries"]) * len(g["cells"])


def group_done(log_dir: Path, g: dict) -> bool:
    """Whether a test group's eval set already finished: its runner index's last run is `done` and every one of its
    tasks has a successful log (a resume reuses them all, so nothing is left to pay for)."""
    from .runner import read_index

    index = read_index(log_dir)
    runs = index.get("runs") or []
    ok = [t for t in (index.get("tasks") or {}).values() if t.get("status") == "success"]
    return bool(runs) and runs[-1].get("status") == "done" and len(ok) >= _group_expected_tasks(g)


def group_spent(log_dir: Path) -> float:
    """Inspect $ the group's eval logs already hold (each sample once)."""
    from .budget import logs_spend

    return logs_spend(sorted(log_dir.rglob("*.eval")))["inspect_usd"] if log_dir.is_dir() else 0.0


def group_remaining(run: GateRun, g: dict, projected: float) -> float:
    """What a group still costs: 0 when it finished, else its projection less what its logs already hold."""
    log_dir = run.phase_dir("test") / g["dir"]
    return 0.0 if group_done(log_dir, g) else max(0.0, projected - group_spent(log_dir))


def _test_remaining(run: GateRun) -> float:
    """What the test phase's start must afford (O4): the PRIMARY_CELLS groups still to do, group by group (finished
    groups count 0, partly run ones their projection less their logs' spend). Every other group is guarded before
    it runs, so a short budget stops the secondaries and leaves the verdict's evidence complete. Before the freeze,
    the whole projection (the freeze check refuses the phase anyway)."""
    if read_freeze(run) is None or not run.selected_path.is_file() or not run.budget_calibration_path.is_file():
        return _test_projected(run)
    groups = test_groups(run, read_selected(run.selected_path), read_calibration(run))
    return sum(group_remaining(run, g, _group_projected(run, g)) for g in groups if g["primary"])


def _test_projected(run: GateRun) -> float:
    """Every test-phase cell at the live sizes it runs (n_tasks-cells as whole worlds)."""
    p = plan(run)
    cells = []
    for cell in _test_cells(p):
        spec = dict(cell.spec)
        if "n_tasks" in spec:
            spec["n_tasks"] = max(_cell_worlds(run, cell, offline=False)["n_tasks"].values())
        cells.append(PlanCell(cell.id, cell.study, cell.phase, spec))
    return project(run, cells)


def _test_params(run: GateRun) -> dict:
    params: dict[str, Any] = {"scale": scale(run), "lightrag_kind": "oracle" if run.offline else "extract", "env_knobs": _env_knobs()}
    if read_freeze(run) is None or not run.selected_path.is_file() or not run.budget_calibration_path.is_file():
        return params  # not frozen: the freeze check refuses the phase
    groups = test_groups(run, read_selected(run.selected_path), read_calibration(run))
    return params | {"groups": [{k: g[k] for k in ("cell", "name", "dir", "arms", "env", "deliveries", "exposure", "split", "exception_style", "epochs", "n_worlds")} for g in groups]}


def _test_group_tasks(g: dict) -> list:
    """The gate tasks of one group; call under the group's env (each task records its knobs)."""
    from .tasks.gate import gate

    return [
        gate(
            family=family,
            level=level,
            split=g["split"],
            arm=a["run"],
            exposure=g["exposure"],
            exception_style=g["exception_style"],
            delivery=delivery,
            limit_worlds=g["n_worlds"][task_cell],
            plan_cell=g["cell"],
            group=g["name"],
            **({"seed_base": g["seed_base"]} if g.get("seed_base") is not None else {}),
        )
        for a in g["arms"]
        for delivery in g["deliveries"]
        for task_cell in g["cells"]
        for family, level in [task_cell.split("-", 1)]
    ]


def _cell_status(groups: list[dict]) -> str:
    statuses = {g["status"] for g in groups}
    for s in ("failed", "stopped", "pending"):
        if s in statuses:
            return s
    return "done"


def _test(run: GateRun, record: dict) -> None:
    freeze = require_frozen(run)
    record["frozen_at"] = freeze["frozen_at"]
    _warn_drift(run, record)
    selected = read_selected(run.selected_path)
    groups = test_groups(run, selected, read_calibration(run))
    gp = gate_profile(run)
    models = gate_models(run, gp)
    tdir = run.phase_dir("test")
    cells: dict[str, dict] = {}
    entries = []
    for g in groups:
        entry = {
            "name": g["name"],
            "arms": g["arms"],
            "split": g["split"],
            "cells": g["cells"],
            "deliveries": g["deliveries"],
            "exposure": g["exposure"],
            "exception_style": g["exception_style"],
            "epochs": g["epochs"],
            "n_worlds": g["n_worlds"],
            "env": g["env"],
            "log_dir": _show(tdir / g["dir"]),
            "log_files": [],
            "status": "pending",
        } | ({"calibration": g["calibration"]} if "calibration" in g else {}) | ({"seed_base": g["seed_base"]} if g.get("seed_base") is not None else {})
        cells.setdefault(g["cell"], {"status": "pending", "primary": g["primary"], "groups": []})["groups"].append(entry)
        entries.append((g, entry))
    record["cells"], record["primary_complete"] = cells, False
    record["log_dirs"] = [e["log_dir"] for _, e in entries]
    stopped: BudgetError | None = None
    for g, entry in entries:
        what = f"test {g['cell']} ({g['name']})"
        if stopped is None:
            try:
                entry["projected_usd"] = round(_group_projected(run, g), 4)
                entry["projected_remaining_usd"] = round(group_remaining(run, g, entry["projected_usd"]), 4)  # a resume pays only the rest
                require_affordable(entry["projected_remaining_usd"], spend(run)["remaining_usd"], what)
            except BudgetError as e:
                stopped = e
        if stopped is not None:
            entry["status"] = "stopped"
            continue
        print(f"[test] {g['cell']} / {g['name']}: {', '.join(a['run'] for a in g['arms'])} x {g['cells']} x {g['deliveries']} ({g['split']}, {g['epochs']} epoch(s))", flush=True)
        try:
            with _environ(g["env"]):
                tasks = _test_group_tasks(g)
                entry["log_files"] = run_gate_tasks(run, gp, models, tasks, tdir / g["dir"], g["epochs"], what, sample_usd(entry["projected_usd"], [(tasks, g["epochs"])]))
            entry["status"] = "done"
        except Exception as e:  # noqa: BLE001  (recorded; the other cells still run, and the phase then fails)
            entry |= {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        for c in cells.values():
            c["status"] = _cell_status(c["groups"])
        record["primary_complete"] = all(cells[c]["status"] == "done" for c in PRIMARY_CELLS)
        _write_json(run.manifest_path("test"), record)  # progress: the manifest shows each finished group
    for c in cells.values():
        c["status"] = _cell_status(c["groups"])
    record["primary_complete"] = all(cells[c]["status"] == "done" for c in PRIMARY_CELLS)
    record["outputs"] = {f"logs:{g['dir']}": e["log_dir"] for g, e in entries if e["status"] == "done"}
    failed = [f"{g['cell']} ({g['name']}): {e['error']}" for g, e in entries if e["status"] == "failed"]
    if stopped is not None:
        not_run = [f"{g['cell']} ({g['name']})" for g, e in entries if e["status"] == "stopped"]
        raise BudgetError(
            f"{stopped}. Stopped before {not_run}; the primary cells are {'complete' if record['primary_complete'] else 'NOT complete'}"
            + (f"; failed: {failed}" if failed else "")
        )
    if failed:
        raise PhaseError(f"{len(failed)} test group(s) failed (re-run to resume; finished logs are reused): " + "; ".join(failed))


# analyze -------------------------------------------------------------------------------------------

ANALYSIS_CODE = ("src/ape/analyze_gate.py", "src/ape/analysis/gate_stats.py", "src/ape/analysis/cost.py")


def _analyze_inputs(run: GateRun) -> dict[str, Path]:
    """The files the report reads that a phase writes only when it runs (manifests are rewritten on every skip,
    so the test results enter through `_analyze_params` instead)."""
    return {
        "anchor/pc1.json": run.phase_dir("anchor") / "pc1.json",
        "tune/tuning_log.jsonl": run.phase_dir("tune") / "tuning_log.jsonl",
        "freeze.json": run.freeze_path,
        "build-test/worlds.json": run.phase_dir("build-test") / "worlds.json",
        "config/selected.yaml": run.selected_path,
        "config/tuning_grid.yaml": run.config("tuning_grid.yaml"),
        "config/run_plan.yaml": run.config("run_plan.yaml"),
        "config/model_costs.yaml": run.costs_path,
        "ledger": Config().ledger_path,
    }


def _analyze_params(run: GateRun) -> dict:
    test = read_manifest(run, "test") or {}
    return {
        "test_cells": _digest(test.get("cells")),
        "primary_complete": test.get("primary_complete"),
        "archived_tuning_logs": sorted(p.name for p in run.phase_dir("tune").glob("tuning_log.*.jsonl")),
        "code": {f: _sha256(ROOT / f) for f in ANALYSIS_CODE},
    }


def _analyze(run: GateRun, record: dict) -> None:
    from .analyze_gate import REPORT_DIR, analyze

    decision = analyze(run)
    out = run.dir / REPORT_DIR
    record["outputs"] = {"decision": _show(out / "decision.json"), "report": _show(out / "report.md")}
    record["verdict"] = decision["verdict"]["label"]
    record["verdict_reasons"] = decision["verdict"]["reasons"]
    record["modes"] = {m: v["verdict"] for m, v in decision["verdict"]["modes"].items()}
    if decision["verdict"]["label"] == "PRECONDITION_FAIL":
        record["warnings"].append("verdict PRECONDITION_FAIL: " + "; ".join(decision["verdict"]["reasons"]))


PHASE_DEFS: dict[str, Phase] = {
    "preflight": Phase(
        "preflight",
        _preflight,
        inputs=lambda r: _cfg_inputs(r, "models.yaml", "model_costs.yaml", "run_plan.yaml"),  # PROVENANCE.md: `_provenance_facts`
        params=_preflight_params,
        projected=lambda r: 0.0,
        profile=gate_profile,
    ),
    "build-dev": Phase(
        "build-dev",
        _build_dev,
        inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml"),
        params=lambda r: {"scale": scale(r), "worlds": dev_world_specs(r), "lightrag_kind": "oracle" if r.offline else "extract", "fake_author": r.offline, "builder": build_params(r)},
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
        refuse=_refuse_tune,
    ),
    "anchor": Phase(
        "anchor",
        _anchor,
        inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml"),
        params=_anchor_params,
        projected=lambda r: project(r, _anchor_cells(r, anchor_profile_name(r)[0])),
        profile=lambda r: load_profile(anchor_profile_name(r)[0], r.models_path),
        requires=("preflight",),
        refuse=_refuse_if_frozen("anchor"),  # pc1.json is part of what the freeze decided on (O3)
    ),
    "pilot": Phase(
        "pilot",
        _pilot,
        inputs=lambda r: _cfg_inputs(r, "run_plan.yaml", "models.yaml", "model_costs.yaml") | {"config/selected.yaml": r.selected_path},
        params=_pilot_params,
        projected=_pilot_projected,
        profile=gate_profile,
        requires=("preflight", "tune"),
        upstream=("tune",),
        refuse=_refuse_if_frozen("pilot"),
    ),
    "freeze": Phase(
        "freeze",
        _freeze,
        inputs=_freeze_inputs,
        # --accept-pc1-failure and the role (--extension-of / --fix-cycle-of) are recorded by the freeze, not part of its
        # fingerprint (O4, D-027): a later `all` without them still skips.
        params=lambda r: {"scale": scale(r), "frozen": list(frozen_files(r, r.prereg_path)), "rehearsal": r.offline},
        projected=lambda r: 0.0,
        profile=gate_profile,
        requires=("preflight", "tune", "anchor", "pilot"),
        upstream=("tune", "pilot"),
        refuse=_refuse_refreeze,
    ),
    "build-test": Phase(
        "build-test",
        _build_test,
        inputs=_frozen_inputs,
        params=lambda r: {
            "scale": scale(r), "worlds": test_world_specs(r), "test_seed_base": run_test_seed_base(r), "lightrag_kind": "oracle" if r.offline else "extract",
            "fake_author": r.offline, "builder": build_params(r),
        },  # fmt: skip
        projected=lambda r: project(r, [plan(r).cell(c) for c in TEST_BUILD_CELLS.values()]),
        profile=gate_profile,
        requires=("freeze",),
        upstream=("freeze",),
        refuse=_refuse_unless_frozen("build-test"),
    ),
    "test": Phase(
        "test",
        _test,
        inputs=_frozen_inputs,
        params=_test_params,
        projected=_test_projected,
        profile=gate_profile,
        requires=("freeze", "build-test"),
        upstream=("freeze", "build-test"),
        refuse=_refuse_unless_frozen("test"),
        remaining=_test_remaining,
    ),
    "analyze": Phase(
        "analyze",
        _analyze,
        inputs=_analyze_inputs,
        params=_analyze_params,
        projected=lambda r: 0.0,
        profile=gate_profile,
    ),
}
assert tuple(PHASE_DEFS) == PHASES


# --- Running phases --------------------------------------------------------------------------------


def _fingerprint(inputs: dict, params: dict, upstream: dict) -> str:
    key = {"inputs": {k: v["sha256"] for k, v in inputs.items()}, "params": params, "upstream": upstream}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()


def phase_state(run: GateRun, name: str) -> dict:
    """A phase's inputs (hashed), params, upstream fingerprints and fingerprint, as they are now. `run_phase` skips a
    complete phase whose recorded fingerprint equals this one; the freeze requires that of what it rests on."""
    phase = PHASE_DEFS[name]
    inputs = {k: {"path": _show(p), "sha256": _sha256(p)} for k, p in phase.inputs(run).items()}
    params = json.loads(json.dumps(phase.params(run), default=str))  # as a manifest stores them
    upstream = {u: (read_manifest(run, u) or {}).get("fingerprint") for u in phase.upstream}
    return {"inputs": inputs, "params": params, "upstream": upstream, "fingerprint": _fingerprint(inputs, params, upstream)}


def run_phase(run: GateRun, name: str) -> str:
    """Run one phase (inside `run_environment`); returns "done" or "skipped", raises when it fails or is refused."""
    phase = PHASE_DEFS[name]
    _check_mode(run)
    state = phase_state(run, name)
    inputs, params, upstream, fingerprint = state["inputs"], state["params"], state["upstream"], state["fingerprint"]
    old = read_manifest(run, name)
    history = list((old or {}).get("history") or [])
    missing = _missing_outputs(old) if old else []
    if old and old.get("status") in COMPLETE and old.get("fingerprint") == fingerprint and not missing and not run.force:
        old |= {"status": "skipped", "skipped_at": _now(), "skip_reason": f"done at {old.get('finished')} with the same inputs (fingerprint {fingerprint[:12]})"}
        old["history"] = [*history, {"action": "skip", "at": old["skipped_at"]}]
        _write_json(run.manifest_path(name), old)
        print(f"[{name}] skipped: done at {old.get('finished')} with the same inputs", flush=True)
        return "skipped"
    if phase.refuse and (reason := phase.refuse(run)):
        raise PhaseError(reason)
    for req in phase.requires:
        if not is_complete(run, req):
            raise PhaseError(f"{name} needs {req} first: python -m ape.run_gate {req} --run-id {run.run_id}{' --offline' if run.offline else ''}{' --smoke' if run.smoke else ''}")
    if name in PAID_PHASES and (stray := stray_environment(run)):
        raise PhaseError(f"{name}: the environment would move this live run off the gate's design or directories; unset: " + "; ".join(stray))
    profile = phase.profile(run)
    record: dict[str, Any] = {
        "phase": name,
        "run_id": run.run_id,
        "status": "running",
        "offline": run.offline,
        "smoke": run.smoke,
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
        "operational_env": _operational_env(),
        "projected_usd": None,
        "spend_at_start": None,
        "spend": None,
        "outputs": {},
        "log_dirs": [],
        "warnings": [],
        "errors": [],
        "history": [*history, {"action": "run", "at": _now(), "forced": bool(run.force and old and old.get("status") in COMPLETE), "fingerprint": fingerprint}],
    }
    if missing and old.get("status") in COMPLETE:
        record["warnings"].append(f"re-run because recorded output(s) are missing: {missing}")
    if knobs := _result_knobs():
        record["warnings"].append(f"APE_* knob(s) set in the environment apply to every arm of this phase: {knobs}")
    _write_json(run.manifest_path(name), record)
    print(f"[{name}] running{' (offline)' if run.offline else ''}{' (smoke)' if run.smoke else ''}", flush=True)
    try:
        record["projected_usd"] = round(phase.projected(run), 4)
        basis = "run_plan.yaml cells, conservative" + ("; smoke sizes (SMOKE_SCALE)" if run.smoke else "")
        record["projection_basis"] = basis + ("" if not run.offline else "; offline runs spend $0" + ("" if run.smoke else ", priced at live sizes"))
        if run.budget_usd is not None:
            record["budget_basis"] = f"run budget ${run.budget_usd:,.2f} (not the plan's budget.total_usd)"
        record["spend_at_start"] = spend(run)
        # The guard checks the work that is left (O4): a resume reuses finished eval logs and current artifacts, and
        # their spend is already in spend_at_start, so counting it again in the projection would count it twice.
        if phase.remaining is not None:
            record["projected_remaining_usd"] = round(phase.remaining(run), 4)
        else:
            record["resume_credit_usd"] = round(prior_attempt_spend(run, name, old, fingerprint), 4)
            record["projected_remaining_usd"] = round(max(0.0, record["projected_usd"] - record["resume_credit_usd"]), 4)
        _write_json(run.manifest_path(name), record)
        require_affordable(record["projected_remaining_usd"], record["spend_at_start"]["remaining_usd"], f"phase {name}")
        phase.body(run, record)
        if run.offline:
            record["offline_check"] = verify_offline(run)
    except BaseException as e:
        record |= {"status": "failed", "finished": _now()}
        record["errors"].append(f"{type(e).__name__}: {e}")
        with contextlib.suppress(Exception):
            record["spend"] = spend(run)
        _write_json(run.manifest_path(name), record)
        backup_after_phase(run, name, record)
        raise
    record |= {"status": "done", "finished": _now(), "spend": spend(run)}
    _write_json(run.manifest_path(name), record)
    backup_after_phase(run, name, record)
    for w in record["warnings"]:
        print(f"[{name}] WARNING: {w}", flush=True)
    print(f"[{name}] done (spent so far ${record['spend']['spent_usd']:,.2f} of ${record['spend']['budget_usd']:,.0f})", flush=True)
    return "done"


def backup_after_phase(run: GateRun, name: str, record: dict) -> None:
    """Live runs with APE_BACKUP_DIR set: copy runs/, cache/ and indices/ there after every finished or failed
    phase (`ape.backup`, incremental). Best effort: a failed backup is a warning in the manifest, never a failed phase."""
    from .backup import backup, backup_dir

    dest = backup_dir()
    if dest is None or run.offline:
        return
    try:
        r = backup(dest)
        record["backup"] = {"dest": r.dest, "copied": r.copied, "skipped": r.skipped, "bytes_copied": r.bytes_copied}
    except Exception as e:  # noqa: BLE001  (never fail a phase over its backup)
        record["warnings"].append(f"backup to {dest} failed: {type(e).__name__}: {e}")
        print(f"[{name}] WARNING: backup to {dest} failed: {e}", file=sys.stderr, flush=True)
    _write_json(run.manifest_path(name), record)


def run_phases(run: GateRun, phase: str) -> dict[str, str]:
    """`phase` (or every phase for "all", in order, stopping at the first failure); phase -> "done" | "skipped"."""
    names = PHASES if phase == "all" else (phase,)
    if unknown := [n for n in names if n not in PHASE_DEFS]:
        raise PhaseError(f"unknown phase(s) {unknown}; phases are {list(PHASES)} or all")
    statuses: dict[str, str] = {}
    with run_environment(run):
        for name in names:
            try:
                statuses[name] = run_phase(run, name)
            except Exception:
                # A failed or budget-stopped test still gets its report (`analyze` reports what exists).
                if phase == "all" and name == "test" and read_manifest(run, "test") is not None:
                    try:
                        run_phase(run, "analyze")
                    except Exception as e:  # noqa: BLE001  (the test's failure is the one to raise)
                        print(f"[analyze] after the failed test: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                raise
    return statuses


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ape.run_gate", description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("phase", choices=[*PHASES, "all"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--offline", action="store_true", help="zero-spend end-to-end run: mock models, fake embeddings, oracle indices, tiny sizes")
    ap.add_argument("--smoke", action="store_true", help="SMOKE_SCALE: tiny sizes on F7-10 and F3-5, isolated under the runs dir; never freezes (readiness/smoke.py)")
    ap.add_argument("--budget-usd", type=float, help="lower the guard's budget for this run below the plan's budget.total_usd (never above it)")
    ap.add_argument("--force", action="store_true", help="re-run complete phases even when their inputs are unchanged")
    ap.add_argument("--runs-dir", type=Path, help=f"where run directories live (default: runs/; smoke: {_show(SMOKE_RUNS)})")
    ap.add_argument("--config-dir", type=Path, default=ROOT / "config", help="read config inputs from here (offline runs and tests only)")
    ap.add_argument("--accept-pc1-failure", metavar="DIAGNOSIS", help="freeze only: freeze despite a failed PC1, recording the analyst's diagnosis (GATE_PREREG §7)")
    ap.add_argument(
        "--test-seed-base", type=int, metavar="SEED",
        help=f"freeze only: the first test seed (a block of {SEED_BLOCK}); default {FIRST_TEST_SEED_BASE}, or the next block no other frozen run used",
    )  # fmt: skip
    role = ap.add_mutually_exclusive_group()
    role.add_argument("--extension-of", metavar="RUN_ID", help="freeze only: this run is the one pre-registered extension of an INCONCLUSIVE gate run (GATE_PREREG §8)")
    role.add_argument("--fix-cycle-of", metavar="RUN_ID", help="freeze only: this run is the one fix cycle after a NO-GO gate run (GATE_PREREG §8)")
    ap.add_argument(
        "--skip-smoke-check", metavar="REASON",
        help="live runs: run without a fresh passing live smoke (readiness/smoke.py), recording why in the preflight manifest",
    )  # fmt: skip
    ap.add_argument("--smoke-max-age-days", type=float, default=SMOKE_MAX_AGE_DAYS, metavar="DAYS", help=f"live preflight: the oldest live smoke result accepted (default {SMOKE_MAX_AGE_DAYS:g})")
    a = ap.parse_args(argv)
    if a.skip_smoke_check is not None and (a.offline or a.smoke):
        ap.error("--skip-smoke-check applies to live gate runs (offline and smoke runs never check the live smoke)")
    if a.accept_pc1_failure and a.phase not in ("freeze", "all"):
        ap.error("--accept-pc1-failure applies to the freeze phase")
    if a.test_seed_base is not None and a.phase not in ("freeze", "all"):
        ap.error("--test-seed-base applies to the freeze phase (build-test and test use the frozen base)")
    if (a.extension_of or a.fix_cycle_of) and a.phase not in ("freeze", "all"):
        ap.error("--extension-of / --fix-cycle-of apply to the freeze phase (the role is recorded in freeze.json)")
    a.runs_dir = a.runs_dir or ((SMOKE_DRY_RUNS if a.offline else SMOKE_RUNS) if a.smoke else ROOT / "runs")
    try:
        run = GateRun(
            a.run_id, offline=a.offline, force=a.force, runs_root=a.runs_dir, config_dir=a.config_dir, smoke=a.smoke, budget_usd=a.budget_usd,
            accept_pc1_failure=a.accept_pc1_failure, test_seed_base=a.test_seed_base, extension_of=a.extension_of, fix_cycle_of=a.fix_cycle_of,
            skip_smoke_check=a.skip_smoke_check, smoke_max_age_days=a.smoke_max_age_days,
        )  # fmt: skip
        statuses = run_phases(run, a.phase)
    except (PhaseError, PreflightError, BudgetError) as e:
        print(f"run_gate: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001  (recorded in the phase manifest; the CLI reports and exits non-zero)
        traceback.print_exc()
        print(f"run_gate: phase failed; see {a.runs_dir / a.run_id}/<phase>/manifest.json", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": run.run_id, "offline": run.offline, "smoke": run.smoke, "phases": statuses, "dir": _show(run.dir)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
