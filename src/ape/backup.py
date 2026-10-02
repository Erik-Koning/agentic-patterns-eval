"""Incremental backup of everything a live run paid for (READINESS_AUDIT §3).

    uv run python -m ape.backup --dest /Volumes/External/ape-backup     # or set APE_BACKUP_DIR

`runs/`, `cache/` and `indices/` are git-ignored and live only on this disk. They hold eval logs,
manifests, the build/embedding ledger, the embedding cache and the LightRAG/APG indices: all of it cost
money to produce, and `PROVENANCE.md`'s freeze records point into it. This copies them to `dest`
under the same relative paths:

- A file is copied when it is missing at the destination or its source signature (size and mtime, plus
  the WAL sidecar's for a database) differs from the one recorded at its last copy in
  `backup_state.json` at the destination. Nothing at the destination is ever deleted.
- SQLite databases (`*.sqlite`, `*.db`) are copied with SQLite's online backup API, which gives a
  consistent snapshot even while another process writes (WAL mode). Their `-wal`/`-shm` sidecars are
  skipped: the snapshot already contains their content.
- A `backup_manifest.json` at the destination records the time, source root, git commit and the
  counts of copied and skipped files.

`ape.run_gate` calls `backup()` after every completed phase when APE_BACKUP_DIR is set. A failed backup
there is a warning, never a failed phase.
"""

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import ROOT

BACKUP_ENV = "APE_BACKUP_DIR"
DEFAULT_SOURCES = ("runs", "cache", "indices")
SQLITE_SUFFIXES = (".sqlite", ".db")
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")


@dataclass
class BackupResult:
    dest: str
    copied: int = 0
    skipped: int = 0
    bytes_copied: int = 0
    sqlite_snapshots: int = 0
    missing_sources: list[str] = field(default_factory=list)


def backup_dir() -> Path | None:
    raw = os.environ.get(BACKUP_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def _is_sidecar(path: Path) -> bool:
    return any(path.name.endswith(s) for suffix in SQLITE_SUFFIXES for s in (suffix + x for x in SQLITE_SIDECARS))


def _signature(src: Path) -> list[int]:
    """Size and mtime of the file, and of its WAL sidecar for a database (recent writes live there)."""
    sig = [src.stat().st_size, src.stat().st_mtime_ns]
    if src.suffix in SQLITE_SUFFIXES:
        wal = src.with_name(src.name + "-wal")
        sig += [wal.stat().st_size, wal.stat().st_mtime_ns] if wal.exists() else [0, 0]
    return sig


def _snapshot_sqlite(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as source, sqlite3.connect(tmp) as target:
        source.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")  # a standalone file: opening the copy creates no -wal sidecar
    tmp.replace(dst)
    shutil.copystat(src, dst)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def backup(dest: Path, root: Path = ROOT, sources: tuple[str, ...] = DEFAULT_SOURCES) -> BackupResult:
    """Copy `sources` (relative to `root`) into `dest`, incrementally; never deletes at `dest`."""
    dest = Path(dest).expanduser().resolve()
    root = Path(root).resolve()
    if dest == root or root in dest.parents:
        raise ValueError(f"backup destination {dest} is inside the source tree {root}")
    result = BackupResult(dest=str(dest))
    state_path = dest / "backup_state.json"
    state: dict[str, list[int]] = json.loads(state_path.read_text()) if state_path.is_file() else {}
    for name in sources:
        top = root / name
        if not top.exists():
            result.missing_sources.append(name)
            continue
        for src in sorted(p for p in top.rglob("*") if p.is_file() and not p.is_symlink()):
            if _is_sidecar(src):
                continue
            rel = str(src.relative_to(root))
            dst, sig = dest / rel, _signature(src)
            if dst.exists() and state.get(rel) == sig:
                result.skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix in SQLITE_SUFFIXES:
                _snapshot_sqlite(src, dst)
                result.sqlite_snapshots += 1
            else:
                shutil.copy2(src, dst)
            state[rel] = sig
            result.copied += 1
            result.bytes_copied += src.stat().st_size
    manifest = {**asdict(result), "time": datetime.now(UTC).isoformat(timespec="seconds"), "source_root": str(root), "sources": list(sources), "git_commit": _git_commit(root)}
    dest.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, sort_keys=True))
    (dest / "backup_manifest.json").write_text(json.dumps(manifest, indent=1))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", help=f"backup destination (default: ${BACKUP_ENV})")
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES), help="directories under the repo root")
    args = ap.parse_args(argv)
    dest = Path(args.dest).expanduser() if args.dest else backup_dir()
    if dest is None:
        ap.error(f"no destination: pass --dest or set {BACKUP_ENV}")
    r = backup(dest, sources=tuple(args.sources))
    print(f"backup -> {r.dest}: {r.copied} copied ({r.bytes_copied / 1e6:,.1f} MB, {r.sqlite_snapshots} SQLite snapshots), {r.skipped} unchanged"
          + (f"; not present: {r.missing_sources}" if r.missing_sources else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
