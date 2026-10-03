"""Pre-run mitigations: a live gate run needs a fresh passing live smoke (M2), a live tune needs the grid's owners and
their sign-off (M4), and live preflight warns about storage (M5). Offline: nothing here calls a model."""

import json
import os
from datetime import UTC, datetime, timedelta

import pytest

from ape import run_gate as rg
from ape.config import ROOT
from ape.models import PreflightError, load_profile, storage_warnings
from ape.run_gate import GateRun, PhaseError, read_manifest, run_phases
from ape.snapshots import BEGIN, END

GATE = load_profile("gate")
REQUIRED = ["effort", "L2", "orchestrator"]


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        monkeypatch.delenv(k)
    return tmp_path


@pytest.fixture(autouse=True)
def code_unchanged(monkeypatch):
    """The suite runs on whatever working tree it finds (uncommitted edits included): the smoke's code is the code
    that would run unless a test says otherwise (`code_drift` is tested on its own in test_run_gate)."""
    monkeypatch.setattr(rg, "code_drift", lambda commit: [])


def _head() -> str:
    return rg.git_state()["commit"]


def _write_smoke(d, *, status="pass", dry=False, profile="gate", checks=None, required=REQUIRED, commit=None, age_days=0.0, snapshots=None, overrides=None):
    """A live smoke's report.json and checks.json as readiness/smoke.py writes them."""
    d.mkdir(parents=True, exist_ok=True)
    when = (datetime.now(UTC) - timedelta(days=age_days)).isoformat(timespec="seconds")
    commit = commit or _head()
    report = {"dry": dry, "mode": "dry" if dry else "live", "status": status, "models": {"profile": profile, "overrides": {}}, "checks": {}, "finished_utc": when}
    (d / "report.json").write_text(json.dumps(report))
    entry = {"status": "pass", "finished_utc": when, "git_commit": commit, "profile": profile, "overrides": overrides or {}, "snapshots": snapshots or {}}
    record = {"required": list(required), "checks": {n: dict(entry) for n in required} | (checks or {})}
    (d / "checks.json").write_text(json.dumps(record))
    return record


def _run(tmp, **kw) -> GateRun:
    return GateRun("live", runs_root=tmp / "runs", smoke_dir=tmp / "smoke" / "live", provenance_path=tmp / "PROVENANCE.md", **kw)


def _pins(path, pins: dict[str, str]) -> None:
    rows = "\n".join(f"| `{a}` | `{s}` |" for a, s in pins.items())
    path.write_text(f"# Provenance\n\n{BEGIN}\n| Alias | Snapshot |\n|---|---|\n{rows}\n{END}\n")


# --- M2: the live smoke record ---------------------------------------------------------------------


def test_a_fresh_passing_live_smoke_on_this_code_passes(clean_env):
    run = _run(clean_env)
    _write_smoke(run.smoke_dir)
    problems, info = rg.check_live_smoke(run, GATE)
    assert problems == [] and set(info["checks"]) == set(REQUIRED) and info["latest_status"] == "pass"
    # warn-level checks are findings, not breakages
    _write_smoke(run.smoke_dir, status="warn", checks={"L2": {"status": "warn", "finished_utc": datetime.now(UTC).isoformat(), "git_commit": _head(), "profile": "gate", "snapshots": {}}})
    assert rg.check_live_smoke(run, GATE)[0] == []


def test_no_record_and_a_dry_report_are_refused(clean_env):
    run = _run(clean_env)
    problems, _ = rg.check_live_smoke(run, GATE)
    assert len(problems) == 1 and problems[0].startswith("no live smoke record (") and "readiness/smoke.py --max-usd 4" in problems[0] and "--skip-smoke-check" in problems[0]
    _write_smoke(run.smoke_dir, dry=True)
    assert any("is a dry run's report (mode 'dry'), not a live smoke" in p for p in rg.check_live_smoke(run, GATE)[0])


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"status": "fail"}, "the latest live smoke's status is 'fail'"),
        ({"status": "stopped"}, "the latest live smoke's status is 'stopped'"),
        ({"profile": "study_g_sol"}, "the latest live smoke ran profile 'study_g_sol', not the gate's 'gate'"),
        ({"age_days": 8.0}, "days old (limit 7; --smoke-max-age-days); re-run it"),
        ({"overrides": {"agent": "openai/gpt-6-sol"}}, "ran with model overrides"),
    ],
)
def test_a_failed_stopped_foreign_stale_or_overridden_smoke_is_refused(clean_env, kw, message):
    run = _run(clean_env)
    _write_smoke(run.smoke_dir, **kw)
    problems = rg.check_live_smoke(run, GATE)[0]
    assert any(message in p for p in problems), problems


def test_every_required_check_needs_a_passing_live_result(clean_env):
    run = _run(clean_env)
    record = _write_smoke(run.smoke_dir)
    del record["checks"]["L2"]
    record["checks"]["orchestrator"]["status"] = "fail"
    (run.smoke_dir / "checks.json").write_text(json.dumps(record))
    problems = rg.check_live_smoke(run, GATE)[0]
    assert "smoke check 'L2' has no live result; run it (`readiness/smoke.py --only L2`)" in problems
    assert any("smoke check 'orchestrator': latest live result is 'fail'" in p for p in problems)
    # the age limit is configurable
    _write_smoke(run.smoke_dir, age_days=10)
    assert rg.check_live_smoke(_run(clean_env, smoke_max_age_days=14), GATE)[0] == []


def test_a_smoke_on_other_code_or_other_snapshots_is_refused(clean_env, monkeypatch):
    run = _run(clean_env)
    _write_smoke(run.smoke_dir, commit="f" * 40)
    # Not ancestry but the code itself: an ancestor commit whose src/ differs from what would run is refused.
    monkeypatch.setattr(rg, "code_drift", lambda commit: ["src/ape/run_gate.py"])
    problems = rg.check_live_smoke(run, GATE)[0]
    assert any("ran at ffffffffffff, but src, power, uv.lock differ from that commit now: ['src/ape/run_gate.py']" in p for p in problems), problems
    monkeypatch.setattr(rg, "code_drift", lambda commit: None)
    assert any("cannot tell whether the code it ran at ffffffffffff is the code that would run" in p for p in rg.check_live_smoke(run, GATE)[0])
    monkeypatch.setattr(rg, "code_drift", lambda commit: [])
    assert rg.check_live_smoke(run, GATE)[0] == [], "another commit with the same code is the same code"
    # A smoke of uncommitted code smoked no commit; a record from before `code_dirty` falls back to `git_dirty`.
    record = _write_smoke(run.smoke_dir)
    record["checks"]["L2"]["code_dirty"] = ["src/ape/lgr/build.py"]
    record["checks"]["effort"]["git_dirty"] = True
    record["checks"]["orchestrator"] |= {"git_dirty": True, "code_dirty": []}  # dirty elsewhere only: fine
    (run.smoke_dir / "checks.json").write_text(json.dumps(record))
    problems = rg.check_live_smoke(run, GATE)[0]
    assert any("smoke check 'L2' ran with uncommitted code changes ['src/ape/lgr/build.py']" in p for p in problems)
    assert any("smoke check 'effort' ran with uncommitted code changes" in p for p in problems)
    assert not any("'orchestrator'" in p for p in problems)
    # PROVENANCE.md pins: the smoke must have run on the pinned snapshot of every alias the gate profile calls;
    # a pinned alias the profile does not call (Sol) does not matter.
    _pins(run.provenance_path, {"gpt-6-luna": "gpt-6-luna-2026-09-01", "gpt-6-sol": "gpt-6-sol-2026-09-01"})
    _write_smoke(run.smoke_dir, snapshots={"gpt-6-luna": "gpt-6-luna-2026-10-01"})
    problems = rg.check_live_smoke(run, GATE)[0]
    assert any("ran on snapshot 'gpt-6-luna-2026-10-01' of gpt-6-luna, but PROVENANCE.md pins 'gpt-6-luna-2026-09-01'" in p for p in problems)
    assert not any("gpt-6-sol" in p for p in problems)
    _write_smoke(run.smoke_dir, snapshots={"gpt-6-luna": "gpt-6-luna-2026-09-01"})
    assert rg.check_live_smoke(run, GATE)[0] == []


def _live_ready(tmp, monkeypatch):
    """A live run whose preflight passes everything but the smoke check (key, probe, APG pin)."""
    monkeypatch.setenv("APE_CACHE", str(tmp / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")  # nothing here makes a request
    probe = tmp / "openai_probe.json"
    probe.write_text(json.dumps({"models_available": ["gpt-6-luna", "gpt-6-sol", "text-embedding-3-small", "gpt-4o-mini"]}))
    (tmp / "PROVENANCE.md").write_text((ROOT / "PROVENANCE.md").read_text())
    return {"env_path": tmp / "no.env", "probe_path": probe}


def test_live_preflight_refuses_without_the_smoke_and_records_an_override(clean_env, monkeypatch):
    paths = _live_ready(clean_env, monkeypatch)
    run = _run(clean_env, **paths)
    with pytest.raises(PreflightError, match=r"no live smoke record"):
        run_phases(run, "preflight")
    m = read_manifest(run, "preflight")
    assert m["status"] == "failed" and m["smoke_check"]["problems"]
    # An empty reason is not a reason.
    with pytest.raises(PreflightError, match="--skip-smoke-check needs a reason"):
        run_phases(_run(clean_env, skip_smoke_check="  ", **paths), "preflight")
    # The override runs, and the manifest records why.
    assert run_phases(_run(clean_env, skip_smoke_check="probe only: smoke deferred by the analyst", **paths), "preflight") == {"preflight": "done"}
    m = read_manifest(run, "preflight")
    assert m["smoke_check"] == {"skipped": "probe only: smoke deferred by the analyst"} and m["checks"]["smoke"] == {"skipped": "probe only: smoke deferred by the analyst"}
    assert any("the live smoke check was skipped" in w for w in m["warnings"])
    assert rg.read_run_info(run)["smoke_check"]["override"] == "probe only: smoke deferred by the analyst"
    # The smoke is settled once per run: a passing smoke written afterwards does not re-run preflight (it is no input)...
    _write_smoke(run.smoke_dir)
    assert run_phases(run, "preflight") == {"preflight": "skipped"}
    # ...but a preflight that runs again (here forced) takes it, replacing the run's override.
    assert run_phases(_run(clean_env, force=True, **paths), "preflight") == {"preflight": "done"}
    assert read_manifest(run, "preflight")["checks"]["smoke"] == "ok"
    assert rg.read_run_info(run)["smoke_check"]["commits"] == [_head()] and "override" not in rg.read_run_info(run)["smoke_check"]


def test_the_smoke_is_settled_once_per_run_and_the_documented_steps_never_reopen_it(clean_env, monkeypatch):
    """The documented flow (READINESS "Running the gate"): anchor, then all to the freeze; the pre-registration filled
    and committed; freeze (PROVENANCE.md's record); then all again, perhaps a week later. Only the run's first preflight
    checks the smoke's age; the later steps neither re-run preflight nor reopen the smoke while the code is unchanged."""
    paths = _live_ready(clean_env, monkeypatch)
    run = _run(clean_env, **paths)
    _write_smoke(run.smoke_dir)
    assert run_phases(run, "preflight") == {"preflight": "done"}
    settled = rg.read_run_info(run)["smoke_check"]
    assert settled["commits"] == [_head()] and set(settled["checks"]) == set(REQUIRED)
    fingerprint = read_manifest(run, "preflight")["fingerprint"]
    # A week later: the smoke is stale; the pre-registration's commit moved HEAD (no code changed); the freeze appended
    # its record to PROVENANCE.md; the run's outputs are under runs/<id>/config/.
    _write_smoke(run.smoke_dir, age_days=8)
    real_git = rg._git
    monkeypatch.setattr(rg, "_git", lambda *a: "e" * 40 if a == ("rev-parse", "HEAD") else real_git(*a))
    with run.provenance_path.open("a") as f:
        f.write("\n## Gate freeze: run `live` (2026-10-09T00:00:00+00:00)\n- **Record:** `runs/live/freeze.json`.\n")
    run.out_config_dir.mkdir(parents=True, exist_ok=True)
    (run.out_config_dir / "selected.yaml").write_text("APG*: {}\n")
    assert rg.phase_state(run, "preflight")["fingerprint"] == fingerprint
    assert run_phases(run, "preflight") == {"preflight": "skipped"}
    # A preflight that does run (forced here; or after a new probe) reuses the settlement: the age is not re-checked.
    assert run_phases(_run(clean_env, force=True, **paths), "preflight") == {"preflight": "done"}
    m = read_manifest(run, "preflight")
    assert m["smoke_check"]["reused"] is True and m["checks"]["smoke"].startswith("ok (settled at this run's first preflight")
    # Another run checks afresh, and refuses the week-old smoke.
    with pytest.raises(PreflightError, match="days old"):
        run_phases(GateRun("live-2", runs_root=clean_env / "runs", smoke_dir=run.smoke_dir, provenance_path=run.provenance_path, **paths), "preflight")
    # Changed code reopens it: only a smoke of this code will do.
    monkeypatch.setattr(rg, "code_drift", lambda commit: ["src/ape/agent/kb_react.py"])
    with pytest.raises(PreflightError, match=r"changed since the smoke this run accepted(.|\n)*days old"):
        run_phases(_run(clean_env, force=True, **paths), "preflight")


def test_the_preflight_fingerprint_follows_the_code_not_the_commit(clean_env, monkeypatch):
    """`code_state`: HEAD's object ids of src/, power/ and uv.lock plus their uncommitted changes, so a commit of other
    files leaves it alone, and a code change (committed or not) does not."""
    real_git = rg._git
    base = rg.code_state()
    assert base is not None and set(base["head"]) == set(rg.CODE_PATHS)
    monkeypatch.setattr(rg, "_git", lambda *a: "e" * 40 if a == ("rev-parse", "HEAD") else real_git(*a))
    assert rg.code_state() == base, "a new commit that leaves CODE_PATHS alone"
    monkeypatch.setattr(rg, "_git", lambda *a: "f" * 40 if a == ("rev-parse", "HEAD:src") else real_git(*a))
    assert rg.code_state() != base, "a commit that changes src/"
    monkeypatch.setattr(rg, "_git", lambda *a: "diff --git a/src/x.py b/src/x.py" if a[0] == "diff" else real_git(*a))
    assert rg.code_state()["uncommitted"] not in (None, base["uncommitted"]), "an uncommitted change to the code"
    monkeypatch.setattr(rg, "_git", lambda *a: None)
    assert rg.code_state() is None


def test_offline_and_smoke_runs_never_check_the_live_smoke(clean_env):
    for run in (GateRun("o", offline=True, runs_root=clean_env / "runs"), GateRun("s", offline=True, smoke=True, runs_root=clean_env / "sruns")):
        record = {"git": {}, "warnings": [], "checks": {}}
        rg._preflight(run, record)
        assert "smoke_check" not in record and "smoke" not in record["checks"]


def test_the_cli_keeps_the_smoke_override_for_live_runs_and_smoke_runs_under_their_mode_dir(clean_env, capsys):
    with pytest.raises(SystemExit):
        rg.main(["preflight", "--run-id", "x", "--offline", "--skip-smoke-check", "nope"])
    assert "--skip-smoke-check applies to live gate runs" in capsys.readouterr().err
    assert rg.SMOKE_RUNS == rg.SMOKE_ROOT / "live" / "runs" and rg.SMOKE_DRY_RUNS == rg.SMOKE_ROOT / "dry" / "runs"


# --- M4: tuning sign-off ---------------------------------------------------------------------------------


def _signed(grid: dict, owners=None, signed=True) -> dict:
    return grid | {"owners": owners or {"APG": "A. Owner", "LightRAG": "S. Keptic", "S3s": "S. Keptic"}, "signed_off": dict.fromkeys(("APG", "LightRAG", "S3s"), signed)}


def test_the_grid_ships_unsigned_and_its_new_keys_are_not_candidates():
    from ape.tuning import candidates, load_grid

    grid = load_grid()
    assert grid["owners"] == dict.fromkeys(("APG", "LightRAG", "S3s"), "TODO") and grid["signed_off"] == dict.fromkeys(("APG", "LightRAG", "S3s"), False)
    assert set(grid["systems"]) == {"APG", "LightRAG", "S3s"} and all(candidates(grid, s) for s in grid["systems"])
    problems = rg.tuning_signoff_problems(grid)
    assert "owners.APG is not set" in problems and "signed_off.LightRAG is not true" in problems and len(problems) == 6


def test_a_live_tune_refuses_until_every_owner_signs_off(clean_env, monkeypatch):
    from ape import tuning

    live = GateRun("t", runs_root=clean_env / "runs")
    reason = rg.PHASE_DEFS["tune"].refuse(live)
    assert reason.startswith("tune: ") and "is not signed off for a live tune" in reason
    assert "PC6 counts every configuration ever tried for a system, archived tuning logs included, against budget_per_system (8)" in reason
    base = tuning.load_grid()
    monkeypatch.setattr(tuning, "load_grid", lambda path=None: _signed(base))
    assert rg.PHASE_DEFS["tune"].refuse(live) is None
    # The skeptic is not on the APG side.
    monkeypatch.setattr(tuning, "load_grid", lambda path=None: _signed(base, owners={"APG": "Ana", "LightRAG": "ana ", "S3s": "Sam"}))
    assert "owners.LightRAG is the APG owner (Ana)" in rg.PHASE_DEFS["tune"].refuse(live)
    monkeypatch.setattr(tuning, "load_grid", lambda path=None: _signed(base, signed=False))
    assert "signed_off.APG is not true" in rg.PHASE_DEFS["tune"].refuse(live)
    # Offline and smoke runs tune their tiny grids without it; a frozen run still refuses first.
    monkeypatch.setattr(tuning, "load_grid", lambda path=None: base)
    assert rg.PHASE_DEFS["tune"].refuse(GateRun("o", offline=True, runs_root=clean_env / "runs")) is None
    assert rg.PHASE_DEFS["tune"].refuse(GateRun("s", smoke=True, runs_root=clean_env / "sruns")) is None


# --- M5: storage warnings ------------------------------------------------------------------------------


class _Usage:
    def __init__(self, free_gb: float):
        self.free = free_gb * 1e9


def test_storage_warnings_cover_an_unset_misplaced_or_unwritable_backup_and_low_disk(clean_env, monkeypatch):
    plenty = lambda path: _Usage(500)  # noqa: E731
    monkeypatch.delenv("APE_BACKUP_DIR", raising=False)
    w = storage_warnings(disk_usage=plenty)
    assert len(w) == 1 and w[0].startswith("APE_BACKUP_DIR is not set")
    monkeypatch.setenv("APE_BACKUP_DIR", str(ROOT / "cache" / "backup"))
    assert any("is inside the repo" in x for x in storage_warnings(disk_usage=plenty))
    locked = clean_env / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        monkeypatch.setenv("APE_BACKUP_DIR", str(locked / "ape-backup"))  # not created yet: its parent is checked
        if os.access(locked, os.W_OK):
            pytest.skip("running as a user who can write anywhere")
        assert any("is not writable" in x for x in storage_warnings(disk_usage=plenty))
    finally:
        locked.chmod(0o700)
    monkeypatch.setenv("APE_BACKUP_DIR", str(clean_env / "backup"))
    assert storage_warnings(disk_usage=plenty) == []
    low = storage_warnings(disk_usage=lambda path: _Usage(5))
    assert len(low) == 2 and all("5.0 GB free (< 20 GB)" in x for x in low)


def test_live_preflight_carries_the_storage_warnings(clean_env, monkeypatch):
    paths = _live_ready(clean_env, monkeypatch)
    monkeypatch.delenv("APE_BACKUP_DIR", raising=False)
    run = _run(clean_env, skip_smoke_check="test: storage warnings only", **paths)
    assert run_phases(run, "preflight") == {"preflight": "done"}
    assert any(w.startswith("APE_BACKUP_DIR is not set") for w in read_manifest(run, "preflight")["warnings"])


def test_a_live_tune_is_refused_before_any_record(clean_env):
    run = GateRun("t", runs_root=clean_env / "runs")
    with pytest.raises(PhaseError, match="is not signed off for a live tune"):
        run_phases(run, "tune")
    assert read_manifest(run, "tune") is None
