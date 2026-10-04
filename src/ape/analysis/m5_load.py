"""The ledger arms' per-sample records for the M5 add-on study's diagnostics (D-053; descriptive, `m5_report`).

The tidy frame itself is `main_load.load_main`'s (the m5 study reuses the main study's loader, tables and tests); this
module adds, per sample-epoch, what the ledger did, read from the sample's store:
- `mas_ledger` (the ledger orchestrator's record): rounds, replans and stalled rounds, and the ledger's own tokens;
- `mas_accounting` (every multi-agent arm's): the sample's total tokens, and per agent its role and model buckets.

The record is read tolerantly, so the diagnostics do not depend on one spelling of it:
- rounds: `rounds` (a count or a list), else the length of the progress-ledger list (`progress`, `progress_ledger`,
  `progress_ledgers`);
- replans: `replans` (a count or a list), else the task-ledger list (`plans`, `task_ledgers`) less the first plan;
- stalled rounds: `stalls` / `stall_rounds` (a count or a list), else the progress entries that are stalled: `stalled`,
  or looping (`is_in_loop`, `looping`) or not progressing (`is_progress_being_made`, `progressing`); a value may be a
  bool or Magentic-One's {"answer": bool, "reason": ...};
- the ledger's tokens: the record's `usage.total_tokens` / `total_tokens` / `tokens`, else `mas_accounting`'s agents
  whose role names the ledger, plus any agent's model bucket named for it;
- the sample's tokens: `mas_accounting.totals.total_tokens`.
A field the record does not carry is None (reported as not available), never 0.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LEDGER_KEY = "mas_ledger"
ACCOUNTING_KEY = "mas_accounting"
LEDGER_COLUMNS = ("ledger_present", "ledger_rounds", "ledger_replans", "ledger_stalls", "ledger_tokens", "sample_tokens")
_PROGRESS = ("progress", "progress_ledger", "progress_ledgers")
_PLANS = ("plans", "task_ledgers")


def _count(x: Any) -> int | None:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x) if math.isfinite(x) else None
    if isinstance(x, (list, tuple)):
        return len(x)
    return None


def _flag(v: Any) -> bool | None:
    if isinstance(v, dict):
        v = v.get("answer")
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
        return v.strip().lower() in ("true", "yes")
    return None


def _first(d: dict, keys: Sequence[str]) -> Any:
    return next((d[k] for k in keys if k in d and d[k] is not None), None)


def stalled(entry: Any) -> bool | None:
    """Whether one progress-ledger entry is a stalled round (None: it does not say)."""
    if not isinstance(entry, dict):
        return None
    s = _flag(_first(entry, ("stalled", "stall")))
    if s is not None:
        return s
    loop = _flag(_first(entry, ("is_in_loop", "looping", "in_loop")))
    prog = _flag(_first(entry, ("is_progress_being_made", "progressing", "making_progress")))
    if loop is None and prog is None:
        return None
    return bool(loop) or prog is False


def _ledger_tokens(ledger: dict, accounting: dict | None) -> int | None:
    usage = ledger.get("usage")
    own = (usage.get("total_tokens") if isinstance(usage, dict) else None) or _first(ledger, ("total_tokens", "tokens"))
    if isinstance(own, (int, float)) and not isinstance(own, bool):
        return int(own)
    if not isinstance(accounting, dict) or not isinstance(accounting.get("agents"), dict):
        return None
    found, total = False, 0
    for aid, rec in accounting["agents"].items():
        if not isinstance(rec, dict):
            continue
        if "ledger" in str(rec.get("role") or "").lower() or "ledger" in str(aid).lower():
            found = True
            total += int(rec.get("all_total_tokens") or rec.get("total_tokens") or 0)
            continue
        for name, b in (rec.get("models") or {}).items():
            if "ledger" in str(name).lower() and isinstance(b, dict):
                found = True
                total += int(b.get("total_tokens") or 0)
    return total if found else None


def ledger_metrics(ledger: Any, accounting: Any = None) -> dict:
    """One sample's ledger columns (LEDGER_COLUMNS) from its `mas_ledger` and `mas_accounting` records."""
    acc = accounting if isinstance(accounting, dict) else None
    totals = (acc or {}).get("totals")
    sample_tokens = totals.get("total_tokens") if isinstance(totals, dict) else None
    out = dict.fromkeys(LEDGER_COLUMNS)
    out |= {"ledger_present": isinstance(ledger, dict), "sample_tokens": sample_tokens}
    if not isinstance(ledger, dict):
        return out
    progress = _first(ledger, _PROGRESS)
    progress = progress if isinstance(progress, list) else None
    rounds = _count(ledger.get("rounds"))
    out["ledger_rounds"] = rounds if rounds is not None else (len(progress) if progress is not None else None)
    replans = _count(ledger.get("replans"))
    if replans is None and isinstance(plans := _first(ledger, _PLANS), list):
        replans = max(0, len(plans) - 1)
    out["ledger_replans"] = replans
    stalls = _count(_first(ledger, ("stalls", "stall_rounds")))
    if stalls is None and progress is not None:
        flags = [stalled(e) for e in progress]
        stalls = sum(1 for f in flags if f) if any(f is not None for f in flags) else None
    out["ledger_stalls"] = stalls
    out["ledger_tokens"] = _ledger_tokens(ledger, acc)
    return out


def read_ledgers(log_files: Sequence[str | Path]) -> pd.DataFrame:
    """One row per (log file, task, epoch) with its sample uuid and ledger columns."""
    from inspect_ai.log import read_eval_log

    rows = []
    for f in log_files:
        log = read_eval_log(str(f))
        for s in log.samples or []:
            store = s.store or {}
            rows.append({"log_file": str(f), "task": s.id, "epoch": int(s.epoch), "uuid": getattr(s, "uuid", None)} | ledger_metrics(store.get(LEDGER_KEY), store.get(ACCOUNTING_KEY)))
    return pd.DataFrame(rows, columns=["log_file", "task", "epoch", "uuid", *LEDGER_COLUMNS])


def attach_ledgers(frame: pd.DataFrame, reader=read_ledgers) -> pd.DataFrame:
    """The frame with LEDGER_COLUMNS joined on (log file, task, epoch); rows without a log or a record get None."""
    out = frame.copy()
    if out.empty or "log_file" not in out.columns:
        for c in LEDGER_COLUMNS:
            out[c] = None
        return out
    led = reader(sorted({str(f) for f in out["log_file"].dropna()}))
    out = out.drop(columns=[c for c in LEDGER_COLUMNS if c in out.columns])
    merged = out.merge(led.drop(columns=["uuid"]), on=["log_file", "task", "epoch"], how="left")
    merged["ledger_present"] = merged["ledger_present"].fillna(False).astype(bool)
    return merged
