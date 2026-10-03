"""Every model call's usage and cost, as it happens: a per-log-dir ledger that also holds what no Inspect log does (D-030).

**Why.** Inspect logs a sample's last attempt only. Under `retry_on_error` (`ape.runner` sets 2) an errored attempt is
re-run from scratch under the same sample UUID, and the earlier attempts' usage is in no log, so spend read from the
logs (`ape.budget.logs_spend`) undercounts every retried sample. A killed run loses its in-flight samples' usage too
(the logs flush in batches).

**What.** `UsageLedgerHook`, an Inspect hook (`@hooks`), appends one JSON line per successful model generate
(`on_model_usage`: every role, every agent, errored attempts included) to `<log dir>/usage_ledger.jsonl` while a
`recording()` context is active; `ape.runner.run_evals` opens one around every eval set, so every run of the program
writes one, and the hook is off otherwise. Each line: time, model, usage, `cost_usd` (Inspect's own price for the
call, `ModelUsage.total_cost`, from the same config/model_costs.yaml the logs are priced from; None for an unpriced
model), and the sample it belongs to (UUID, id, epoch, the attempt number from `on_sample_attempt_start`), the eval
and its log file. One `write()` per line on an O_APPEND descriptor, as the spend registry does, so a killed process
loses nothing it wrote and concurrent writers never interleave.

**Counting** (`unlogged_spend`): the logs stay the record of every sample's final attempt (they also hold a call that
crossed a sample limit, which raises before the hook fires). The ledger adds what they lack: for a sample a log
holds, the calls of its earlier attempts; for a sample no log holds, all its calls. Nothing is counted twice: the
F8 sessions' `f8_resume.unlogged` (B7) describes the same earlier-attempt usage for analysis and never enters spend.
`ape.budget.program_spend` and the orchestrators' guards count logs + this; the ledger's own total and the
difference on final attempts are reported as a cross-check.
"""

import contextlib
import json
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from inspect_ai.hooks import Hooks, ModelUsageData, SampleAttemptStart, hooks

LEDGER_NAME = "usage_ledger.jsonl"
_active: dict[str, Path | None] = {"path": None}


@contextlib.contextmanager
def recording(path: Path) -> Iterator[Path]:
    """Record every model call to `path` for the duration (`ape.runner.run_evals` wraps each eval set in this)."""
    previous, _active["path"] = _active["path"], Path(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        yield Path(path)
    finally:
        _active["path"] = previous


def active_ledger() -> Path | None:
    return _active["path"]


def _append(path: Path, entry: dict[str, Any]) -> None:
    data = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


@hooks(name="ape_usage_ledger", description="Appends every model call's usage and cost to the run's usage ledger (ape.usage_ledger)")
class UsageLedgerHook(Hooks):
    def __init__(self) -> None:
        self.attempts: dict[str, int] = {}  # sample UUID -> its current attempt

    def enabled(self) -> bool:
        return _active["path"] is not None

    async def on_sample_attempt_start(self, data: SampleAttemptStart) -> None:
        self.attempts[data.sample_id] = data.attempt

    async def on_model_usage(self, data: ModelUsageData) -> None:
        path = _active["path"]
        if path is None:
            return
        from inspect_ai.log._samples import sample_active

        active = sample_active()
        u = data.usage
        uuid = active.sample_uuid if active is not None else None
        _append(
            path,
            {
                "ts": time.time(),
                "model": data.model_name,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens,
                "input_tokens_cache_read": u.input_tokens_cache_read,
                "input_tokens_cache_write": u.input_tokens_cache_write,
                "reasoning_tokens": u.reasoning_tokens,
                "cost_usd": u.total_cost,
                "sample_uuid": uuid,
                "sample_id": str(active.sample.id) if active is not None and active.sample.id is not None else None,
                "epoch": active.epoch if active is not None else None,
                "attempt": self.attempts.get(uuid) if uuid else None,
                "eval_id": data.eval_id,
                "task": data.task_name,
                "log": active.log_location if active is not None else None,
                "call_seconds": data.call_duration,
            },
        )


def read_entries(paths: Iterable[str | Path]) -> list[dict]:
    """Every entry of the ledgers (a torn last line, from a kill mid-write, is skipped)."""
    out = []
    for p in paths:
        try:
            lines = Path(p).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def unlogged_spend(entries: Iterable[dict], logged: dict[str, float], since: float | None = None) -> dict:
    """What the ledger adds to the logs (module docstring), from its `entries` and `logged` (sample UUID -> the $ the
    logs hold for it): `unlogged_usd` (earlier attempts of logged samples, and every call of samples no log holds),
    `ledger_usd` (every call, a cross-check against the logs plus `unlogged_usd`), `final_attempt_diff_usd` (the ledger's
    final attempts less the logs, for the samples a log holds: about 0, less any call that crossed a sample limit),
    and counts. `since` (a Unix time) keeps only the entries written from then on. Unpriced non-mock calls raise
    ValueError (never $0)."""
    by_uuid: dict[str, dict[int, float]] = {}
    total = standalone = 0.0
    calls, unpriced = 0, set()
    for e in entries:
        if since is not None and float(e.get("ts") or 0) < since:
            continue
        cost = e.get("cost_usd")
        if cost is None:
            if not str(e.get("model", "")).startswith("mockllm/") and (e.get("input_tokens") or e.get("output_tokens")):
                unpriced.add(e.get("model"))
            cost = 0.0
        calls += 1
        total += float(cost)
        if e.get("sample_uuid"):
            attempts = by_uuid.setdefault(e["sample_uuid"], {})
            a = int(e.get("attempt") or 0)
            attempts[a] = attempts.get(a, 0.0) + float(cost)
        else:
            standalone += float(cost)
    if unpriced:
        raise ValueError(f"usage ledger: unpriced calls for {sorted(unpriced)}; run with ape.models.eval_cost_kwargs()")
    unlogged, final_ledger, final_logged, retried, orphans = standalone, 0.0, 0.0, 0, 0
    for uuid, attempts in by_uuid.items():
        last = max(attempts)
        if uuid in logged:
            unlogged += sum(v for a, v in attempts.items() if a != last)
            final_ledger += attempts[last]
            final_logged += float(logged[uuid])
            retried += len(attempts) > 1
        else:
            unlogged += sum(attempts.values())
            orphans += 1
    return {
        "ledger_usd": total,
        "unlogged_usd": unlogged,
        "final_attempt_diff_usd": final_ledger - final_logged,
        "calls": calls,
        "samples": len(by_uuid),
        "retried_samples": retried,
        "samples_in_no_log": orphans,
    }
