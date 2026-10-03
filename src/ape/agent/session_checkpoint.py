"""Mid-session checkpoints for F8 sessions: a retried session resumes after its last completed item.

**Why.** `ape.runner` runs every task with `retry_on_error=2`, and Inspect re-runs an errored sample from scratch.
For a 40-item session an error at item 35 would repeat 34 items' calls. Inspect 0.3.273 has its own sample
checkpointing (`inspect_ai.util.checkpointer`, `Task(checkpoint=...)`), but it does not cover this case: resume is
resolved once per sample run, before its `retry_on_error` loop, so it serves only task-level retries and requeues;
its host store needs a restic binary; and enabling it changes the eval config. Sessions therefore keep this small
file store, switched on by `APE_SESSION_CHECKPOINTS=<dir>` (the study runner sets one directory per run).

**What a checkpoint holds** (one JSON file per sample, epoch and configuration, written atomically): the full
history, the policy's state, the tool recorder (events, report), every record (items, calls with usage, probes,
management events, resumes), the attempts that produced them (`segments`) and the failed attempts (`failures`: the
error, and the usage of the calls made after the last save, which a resume makes again). It is written when a
session starts and after every completed item (the item's boundary hook and probe included), so a resume continues
with the next item.

**Key.** A checkpoint is reused only under the same configuration: the SHA-256 of the sample ID and epoch, the arm,
the policy class and its resolved knobs, every APE_CM_* variable, the agent, `cm` and `probe` models with their
generate configs (effort included; transport settings such as max_connections excluded), the window, threshold,
turn cap and probe checkpoints, the world ID and content hash (so its knob variant and generator output), and the
source of the modules that shape a session (this one, the loop, the policies, the F8 generator, tools, renderer,
token meter and item scorer, plus the policy's own module). Any change gives a new key and so a fresh session; the
stale file is left alone, never read.

**Layout.** `<dir>/<sample id>/epoch-<n>/<key[:24]>.json` (temporary files `*.tmp-<pid>` next to it).

**Lifecycle.** A session that ends normally (report phase done, or overflow) or by an Inspect limit deletes its
checkpoint: the sample is complete, and a later run of the same task never resumes stale state. An error keeps it,
with the failure recorded (its error and the usage of the calls made since the last save), so the retry resumes. A
process kill keeps it too, without the failure record.

**A resumed session is not bit-identical to an uninterrupted one.** The item that was interrupted runs again from
its first call (its partial calls and tool events are dropped; F8 tools have no side effects outside the
recorder), with a cold provider cache and fresh sampling. Inspect's transcript and `model_usage` for the sample
cover only the attempt that finished; Inspect's token, cost and message limits count per attempt (the restored
history counts toward the message limit). What shows a session was resumed: the store key `f8_resume` (one entry
per resume: after which item, which checkpoint, the restored calls and their usage) and the score metadata
`resume`. The restored calls keep their per-call records and usage in `f8_views` / `f8_probes`.

**Cost accounting.** Inspect logs an attempt's usage only if it is the last attempt of its sample record: a
sample-level retry keeps the sample UUID and drops the earlier attempts' usage, while a task-level retry starts a
new sample whose predecessor (errored, in the kept failed log) carries its last attempt's usage. `resume_summary`
splits the earlier attempts' usage accordingly: `unlogged` is in no log (add it to spend), `logged_elsewhere` is in
an earlier log's errored sample (already counted when that log is).
"""

import hashlib
import importlib
import json
import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.model import Model

from .context_policy import add_usage, usage_by

log = logging.getLogger(__name__)

ENV = "APE_SESSION_CHECKPOINTS"
FORMAT = 1
# Modules whose source shapes a session's history and records (the policy's own module is added per policy).
CODE_MODULES = (
    "ape.agent.session",
    "ape.agent.context_policy",
    "ape.agent.session_checkpoint",
    "ape.worlds.env_f8",
    "ape.worlds.gen_f8",
    "ape.worlds.render",
    "ape.llm.tokens",
    "ape.scorers.session",
)
# Generate-config fields that change how a call is transported, not what it returns.
TRANSPORT_FIELDS = frozenset({"max_retries", "timeout", "attempt_timeout", "max_connections", "batch", "cache"})


def checkpoint_root() -> Path | None:
    raw = os.environ.get(ENV, "").strip()
    return Path(raw) if raw else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def code_version(modules: Sequence[str] = ()) -> str:
    """SHA-256 over the source of CODE_MODULES and `modules`."""
    h = hashlib.sha256()
    for name in sorted({*CODE_MODULES, *modules}):
        mod = sys.modules.get(name) or importlib.import_module(name)
        path = getattr(mod, "__file__", None)
        h.update(name.encode())
        h.update(Path(path).read_bytes() if path else b"<no source>")
    return h.hexdigest()


def model_identity(model: Model) -> dict:
    """A model as it shapes outputs: name, role and generate config (transport fields excluded)."""
    config = {k: v for k, v in model.config.model_dump(mode="json", exclude_none=True).items() if k not in TRANSPORT_FIELDS}
    return {"model": str(model), "role": model.role, "config": config}


def checkpoint_key(components: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(components, sort_keys=True, default=str).encode()).hexdigest()


def _safe(part: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(part)) or "_"


class SessionCheckpoints:
    """The checkpoint file of one sample, epoch and configuration (module docstring)."""

    def __init__(self, root: Path, sample_id: int | str, epoch: int, components: Mapping[str, Any]):
        self.components = {"format": FORMAT, "sample_id": sample_id, "epoch": epoch, **components}
        self.key = checkpoint_key(self.components)
        self.path = Path(root) / _safe(sample_id) / f"epoch-{epoch}" / f"{self.key[:24]}.json"
        self.last: dict | None = None  # the payload last saved or loaded

    def load(self) -> dict | None:
        """The saved payload, or None (no file, another key, an unreadable file: the session then starts fresh and
        its first save replaces the file)."""
        if not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, ValueError) as e:
            log.warning("ignoring unreadable session checkpoint %s: %s", self.path, e)
            return None
        if payload.get("format") != FORMAT or payload.get("key") != self.key:
            log.warning("ignoring session checkpoint %s: format or key mismatch", self.path)
            return None
        self.last = payload
        return payload

    def save(self, payload: Mapping[str, Any]) -> None:
        """Write `payload` atomically (temporary file, fsync, rename)."""
        full = {"format": FORMAT, "key": self.key, "components": self.components, "saved_at": _now(), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(full, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.last = full

    def complete(self) -> None:
        """The session ended (normally or by a limit): delete its checkpoint, so it is never resumed again."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError as e:  # never fail a finished session over its cleanup; a stale file is keyed, and logged here
            log.warning("could not delete the finished session's checkpoint %s: %s", self.path, e)
        self.last = None


# --- Attempts, restored usage and failures ------------------------------------------------------------------


def segment_ranges(segments: Sequence[Mapping], n_views: int, n_probes: int) -> list[tuple[Mapping, range, range]]:
    """Each attempt's own records: views and probes from its start up to the next attempt's start (a resume starts
    from the previous attempt's last save) or to the end."""
    out = []
    for i, s in enumerate(segments):
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        out.append((s, range(s["views_from"], nxt["views_from"] if nxt else n_views), range(s["probes_from"], nxt["probes_from"] if nxt else n_probes)))
    return out


def usage_summary(records: Sequence[Mapping]) -> dict:
    """Calls and their usage by kind and by model."""
    return {"calls": len(records), "by_kind": usage_by(records, "kind"), "by_model": usage_by(records, "model")}


def _merge(a: dict, b: Mapping) -> dict:
    out = {"calls": a.get("calls", 0) + b.get("calls", 0), "by_kind": dict(a.get("by_kind", {})), "by_model": dict(a.get("by_model", {}))}
    for part in ("by_kind", "by_model"):
        for k, u in b.get(part, {}).items():
            out[part][k] = add_usage(out[part].get(k), u)
    return out


def resume_summary(views: Sequence[Mapping], probes: Sequence[Mapping], resumes: Sequence[Mapping], segments: Sequence[Mapping], failures: Sequence[Mapping]) -> dict | None:
    """The store's `f8_resume`: None for a session that ran in one attempt. Otherwise its resumes (after which item,
    the restored calls and their usage), the failed attempts (error, and the usage of the calls each made after its
    last checkpoint), and the earlier attempts' usage split by where Inspect logged it (module docstring): an
    attempt's usage is logged iff it is the last attempt of its sample UUID; the last segment is the attempt that
    finished, logged in this sample."""
    if len(segments) <= 1 and not failures:
        return None
    last_of_uuid = {s["sample_uuid"]: i for i, s in enumerate(segments)}
    unlogged: dict = {}
    elsewhere: dict = {}
    for i, (s, vr, pr) in enumerate(segment_ranges(segments, len(views), len(probes))):
        if i == len(segments) - 1:
            continue  # this attempt: in this sample's own usage
        mine = usage_summary([views[j] for j in vr] + [probes[j] for j in pr])
        for f in failures:
            if f["attempt"] == s["attempt"]:
                mine = _merge(mine, f["usage"])
        if last_of_uuid[s["sample_uuid"]] == i:
            elsewhere = _merge(elsewhere, mine)
        else:
            unlogged = _merge(unlogged, mine)
    return {
        "resumes": list(resumes),
        "count": len(resumes),
        "after_items": [r["after_item"] for r in resumes],
        "failures": list(failures),
        "unlogged": unlogged or None,
        "logged_elsewhere": elsewhere or None,
    }
