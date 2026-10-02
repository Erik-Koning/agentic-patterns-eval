"""Program-wide spend registry (READINESS_AUDIT §3): every eval log dir and build/embedding ledger the program
writes, so one guard can count spend across run ids, studies and smoke runs, killed runs included.

    cache/spend/registry.jsonl        the registry (APE_SPEND_REGISTRY overrides the path)

It is append-only JSON lines. Each entry goes to the kernel as one write() on an O_APPEND descriptor, as the
ledger does, so lines from several processes never interleave:

    {"kind": "logs", "event": "start", "log_dir": "/abs/dir", "label": "gate/gate-1", "study": "gate",
     "live": true, "pid": 4242, "tasks": ["gate", ...], "ts": "2026-10-02T12:00:00+00:00"}
    {"kind": "logs", "event": "finish", "log_dir": "/abs/dir", "success": true, "ts": "..."}
    {"kind": "ledger", "event": "register", "path": "/abs/cache/ledger.jsonl", "label": "...", "study": "...", "ts": "..."}

- **Who writes.** `ape.runner.run_evals` registers its log dir *before* `eval_set` starts and records the
  finish after, so a killed run's dir is already known. `ape.llm.ledger.Ledger` registers its file on its
  first append in a process. APE_CACHE can differ per run, so ledgers live in different places.
- **Labels.** Each entry's label comes from APE_SPEND_LABEL: `run_gate` sets `gate/<run id>`, a smoke-scale run
  `gate-smoke/<run id>`, an offline one `gate-offline/<run id>`, and `readiness/smoke.py` sets `smoke`.
  Anything else is `adhoc`. The study is the label up to the first "/".
- **Isolation.**
  - Live runs use the real registry, or the one APE_SPEND_REGISTRY names.
  - Offline runs (mock agents) register only where APE_SPEND_REGISTRY points: `run_gate` points an offline
    run at `runs/<id>/work/spend_registry.jsonl`, and the test suite at a temporary file (tests/conftest.py).
  - Under pytest a live registration with no APE_SPEND_REGISTRY raises, so a test can never write the real
    registry.

`ape.budget.program_spend` sums it: each sample once (by its uuid) over every `.eval` log in every registered
dir, finished or not, plus every registered ledger.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ROOT

REGISTRY_ENV = "APE_SPEND_REGISTRY"
LABEL_ENV = "APE_SPEND_LABEL"
DEFAULT_REGISTRY = ROOT / "cache" / "spend" / "registry.jsonl"
DEFAULT_LABEL = "adhoc"

_registered_ledgers: set[tuple[str, str]] = set()  # (registry, ledger) pairs this process already recorded


class SpendRegistryError(RuntimeError):
    """The registry cannot be used as asked (e.g. a test that did not isolate it)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def registry_path(live: bool = True) -> Path | None:
    """The registry in effect: APE_SPEND_REGISTRY if set; else the program registry for live runs, and none
    for offline ones (they are not recorded unless a registry is named)."""
    if override := os.environ.get(REGISTRY_ENV, "").strip():
        return Path(override)
    if not live:
        return None
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise SpendRegistryError(f"a test reached the real spend registry; set {REGISTRY_ENV} (tests/conftest.py does) and keep it set")
    return DEFAULT_REGISTRY


def label() -> str:
    return os.environ.get(LABEL_ENV, "").strip() or DEFAULT_LABEL


def study_of(label_: str) -> str:
    return label_.split("/", 1)[0] or DEFAULT_LABEL


def _append(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = os.write(fd, data)
    finally:
        os.close(fd)
    if written != len(data):
        raise OSError(f"short write to spend registry {path}: {written} of {len(data)} bytes")


def register_logs(log_dir: str | Path, *, live: bool, tasks: list[str] | None = None, label_: str | None = None) -> Path | None:
    """Record that `log_dir` is about to receive eval logs; returns the registry used (None: not recorded)."""
    path = registry_path(live)
    if path is None:
        return None
    lab = label_ or label()
    _append(path, {"kind": "logs", "event": "start", "log_dir": str(Path(log_dir).resolve()), "label": lab, "study": study_of(lab), "live": live, "pid": os.getpid(), "tasks": tasks or [], "ts": _now()})
    return path


def finish_logs(registry: Path | None, log_dir: str | Path, success: bool, error: str | None = None) -> None:
    if registry is not None:
        _append(registry, {"kind": "logs", "event": "finish", "log_dir": str(Path(log_dir).resolve()), "success": success, "error": error, "ts": _now()})


def register_ledger(ledger_path: str | Path) -> None:
    """Record a ledger file once per process (the ledger calls this on its first append)."""
    registry = registry_path(live=True)
    if registry is None:
        return
    key = (str(registry.resolve()), str(Path(ledger_path).resolve()))
    if key in _registered_ledgers:
        return
    lab = label()
    _append(registry, {"kind": "ledger", "event": "register", "path": key[1], "label": lab, "study": study_of(lab), "ts": _now()})
    _registered_ledgers.add(key)


def read_registry(path: Path | None = None) -> list[dict]:
    path = path if path is not None else registry_path(live=True)
    if path is None or not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def registered(path: Path | None = None) -> tuple[dict[str, dict], dict[str, dict]]:
    """({log dir: {label, study, live, starts, finishes, last_success, first_ts}}, {ledger path: {label, study}}),
    in registration order. A dir's label is its first registration's."""
    dirs: dict[str, dict] = {}
    ledgers: dict[str, dict] = {}
    for e in read_registry(path):
        if e.get("kind") == "logs":
            d = dirs.setdefault(e["log_dir"], {"label": e.get("label", DEFAULT_LABEL), "study": e.get("study", DEFAULT_LABEL), "live": False, "starts": 0, "finishes": 0, "last_success": None, "first_ts": e.get("ts")})
            if e.get("event") == "start":
                d["starts"] += 1
                d["live"] = d["live"] or bool(e.get("live"))
            elif e.get("event") == "finish":
                d["finishes"] += 1
                d["last_success"] = e.get("success")
        elif e.get("kind") == "ledger":
            ledgers.setdefault(e["path"], {"label": e.get("label", DEFAULT_LABEL), "study": e.get("study", DEFAULT_LABEL)})
    return dirs, ledgers
