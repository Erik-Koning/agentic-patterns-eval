"""Incremental backup: copies what changed, snapshots SQLite consistently, never deletes, refuses in-tree dests."""

import json
import sqlite3

import pytest

from ape.backup import backup, main


def _tree(root):
    (root / "runs" / "r1" / "test").mkdir(parents=True)
    (root / "runs" / "r1" / "test" / "manifest.json").write_text('{"status": "done"}')
    (root / "cache").mkdir()
    db = sqlite3.connect(root / "cache" / "embeddings.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE emb (k TEXT PRIMARY KEY, v BLOB)")
    db.execute("INSERT INTO emb VALUES ('a', x'00')")
    db.commit()
    return db


def test_backup_copies_then_skips_unchanged_and_snapshots_sqlite(tmp_path):
    root, dest = tmp_path / "repo", tmp_path / "backup"
    root.mkdir()
    db = _tree(root)
    r1 = backup(dest, root=root)
    assert r1.copied == 2 and r1.sqlite_snapshots == 1 and r1.missing_sources == ["indices"]
    # The snapshot is a complete database even though the writer still holds it open in WAL mode.
    assert sqlite3.connect(dest / "cache" / "embeddings.sqlite").execute("SELECT count(*) FROM emb").fetchone() == (1,)
    assert not list(dest.rglob("*-wal")) and not list(dest.rglob("*-shm"))
    r2 = backup(dest, root=root)
    assert r2.copied == 0 and r2.skipped == 2
    db.execute("INSERT INTO emb VALUES ('b', x'01')")
    db.commit()  # lands in the WAL sidecar: the signature includes it
    r3 = backup(dest, root=root)
    assert r3.copied == 1
    assert sqlite3.connect(dest / "cache" / "embeddings.sqlite").execute("SELECT count(*) FROM emb").fetchone() == (2,)
    manifest = json.loads((dest / "backup_manifest.json").read_text())
    assert manifest["copied"] == 1 and manifest["sources"] == ["runs", "cache", "indices"]


def test_backup_never_deletes_at_the_destination(tmp_path):
    root, dest = tmp_path / "repo", tmp_path / "backup"
    root.mkdir()
    _tree(root)
    backup(dest, root=root)
    (root / "runs" / "r1" / "test" / "manifest.json").unlink()
    backup(dest, root=root)
    assert (dest / "runs" / "r1" / "test" / "manifest.json").is_file()


def test_backup_refuses_a_destination_inside_the_source_tree(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _tree(root)
    with pytest.raises(ValueError, match="inside the source tree"):
        backup(root / "runs" / "backup", root=root)


def test_cli_needs_a_destination(monkeypatch):
    monkeypatch.delenv("APE_BACKUP_DIR", raising=False)
    with pytest.raises(SystemExit):
        main([])
