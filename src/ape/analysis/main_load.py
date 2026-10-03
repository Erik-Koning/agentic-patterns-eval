"""Main-study rows from Inspect logs: one row per (sample, epoch), the tidy frame every main-study statistic reads.

`gate_stats.load_results` is the gate's loader and stays as it is. The main study needs more from each sample, so
this loader reads each log once and reuses the gate's per-sample pieces (`pipeline_miss`, the error-label rule):

- **The turn cap from the eval's metadata** (`max_turns`, which `tasks.main` scales with the registry knob: 44 turns
  at F1-32). The gate's loader takes it from the APE_MAX_TURNS knob (default 12), which would flag every F1-32
  sample that used more than 12 turns without answering as a cap hit.
- **The four cost meters** (brief §6.2), per sample, over every Inspect-metered call (agent and roles, subagents
  included): `tokens` (input incl. cache reads and writes, plus output incl. reasoning), `calls` (completed model
  events), `usd` (cache-adjusted: Inspect's own cost from the model-cost config, `config/model_costs.yaml`) and
  `wall` (Inspect working time, which leaves out rate-limit and shared-resource waits; `total_time` is kept too).
  `usd_list` prices every input token at the list input price (no cache discount), from the same table.
- **A canonical answer key** (`answer_key`): the part of the submitted answer the scorer reads, normalised exactly
  as `scorers.success.is_success` normalises it, as a short digest. Equal keys therefore always score alike, which
  the post-hoc S8 vote needs (`frontier`). F1: the normalised ratings; F2: the final supplier; F7: the whole answer
  with `deadline_days` as an int; F3: the end state (the sorted mutating calls, possibly none); F5: the folded
  answer. None when the run submitted nothing or errored (an abstention in the vote).
- **The tier** from the eval's model and effort (`tier` = "luna", "sol", ...; `effort`; `profile` = "luna-high").
- `multi_agent` from the eval metadata, and per-agent accounting from the sample store when an arm records it
  (`PER_AGENT_KEYS`; surfaced as `per_agent`, never required).

Errored samples are failures (success 0, `error` True, `error_label` "harness_error"), as in the gate.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..scorers.success import norm_f5, norm_id, norm_ratings
from ..worlds.env_tools import ANSWER, CALLS
from .gate_stats import DEFAULT_MAX_TURNS, pipeline_miss

METERS = {"tokens": "tokens", "calls": "calls", "usd": "usd", "wall": "wall"}  # meter -> tidy-frame column
# Sample-store keys a multi-agent arm may use for per-agent accounting (BUILD_PLAN B2); the first present is surfaced.
PER_AGENT_KEYS = ("agent_usage", "agent_accounting", "per_agent", "agents")
COLUMNS = (
    "plan_cell", "arm", "family", "level", "cell", "world", "task", "epoch", "tier", "model", "effort", "profile",
    "success", "error", "error_label", "answered", "answer_key", "turns_used", "max_turns", "cap_hit", "limit_hit",
    "tokens", "tokens_input", "tokens_cache_read", "tokens_cache_write", "tokens_output", "tokens_reasoning", "calls",
    "usd", "usd_list", "cost_usd", "wall", "total_time", "working_time", "multi_agent", "delivery", "split",
    "partial_credit", "evidence_recall", "pipeline_miss", "per_agent", "log_file",
)  # fmt: skip


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def canonical_answer(family: str, answer: dict | None, calls: list | None):
    """What `is_success` compares, normalised as it normalises; None for no answer (F3: the end state always exists)."""
    if family == "F3":
        return sorted(json.dumps(c, sort_keys=True) for c in (calls or []))
    if not isinstance(answer, dict):
        return None
    if family == "F1":
        return norm_ratings(answer.get("ratings"))
    if family == "F2":
        return norm_id(answer.get("final", ""))
    if family == "F7":
        out = dict(answer)
        try:
            out["deadline_days"] = int(answer["deadline_days"])
        except (KeyError, TypeError, ValueError):
            pass
        return out
    return norm_f5(answer.get("answer", ""))


def answer_key(family: str, answer: dict | None, calls: list | None) -> str | None:
    """The canonical answer as a short digest (module docstring), or None (abstention)."""
    canon = canonical_answer(family, answer, calls)
    return None if canon is None else _digest([family, canon])


def tier_of(model: str | None) -> str:
    """'openai/gpt-6-luna' -> 'luna'; other models keep their bare name."""
    bare = str(model or "unknown").split("/", 1)[-1]
    return bare.removeprefix("gpt-6-")


def _usage(model_usage: dict) -> dict:
    tin = sum(int(u.input_tokens or 0) for u in model_usage.values())
    cr = sum(int(u.input_tokens_cache_read or 0) for u in model_usage.values())
    cw = sum(int(u.input_tokens_cache_write or 0) for u in model_usage.values())
    out = sum(int(u.output_tokens or 0) for u in model_usage.values())
    rs = sum(int(u.reasoning_tokens or 0) for u in model_usage.values())
    return {"tokens": tin + cr + cw + out, "tokens_input": tin, "tokens_cache_read": cr, "tokens_cache_write": cw, "tokens_output": out, "tokens_reasoning": rs}


def _list_usd(model_usage: dict, prices: dict) -> float:
    """Every input token (cached or not) at the list input price, plus output; NaN when a model has no price."""
    total = 0.0
    for model, u in model_usage.items():
        p = prices.get(model.split("/", 1)[-1])
        if p is None:
            if model.startswith("mockllm/"):
                continue
            return float("nan")
        tin = int(u.input_tokens or 0) + int(u.input_tokens_cache_read or 0) + int(u.input_tokens_cache_write or 0)
        total += (tin * p["input"] + int(u.output_tokens or 0) * p.get("output", 0.0)) / 1e6
    return total


def _model_calls(sample) -> float:
    """Completed model generations in the sample (every role and subagent), or NaN when the log kept no events."""
    events = sample.events or []
    if not events:
        return float("nan")
    return float(sum(1 for e in events if e.event == "model" and not getattr(e, "pending", False) and e.error is None and e.output is not None))


def load_main(log_files: Sequence[str | Path], plan_cell: str | None = None, require_cost: bool = True, store_keys: Sequence[str] = PER_AGENT_KEYS) -> pd.DataFrame:
    """One row per (sample, epoch) of main-study Inspect logs (module docstring for the columns).

    `plan_cell` labels every row (else the eval metadata's `plan_cell`, if any). A missing price would read as $0
    and corrupt the cost meters, so it is an error unless `require_cost=False` (offline dry runs, mock models)."""
    from inspect_ai.log import read_eval_log

    from .cost import load_prices

    prices = load_prices()
    rows = []
    for f in log_files:
        log = read_eval_log(str(f))
        args, emd = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or emd.get("arm")
        knobs = emd.get("knobs") or {}
        max_turns = int(emd.get("max_turns") or knobs.get("APE_MAX_TURNS") or DEFAULT_MAX_TURNS)
        model = log.eval.model
        gen = log.eval.model_generate_config
        effort = getattr(gen, "reasoning_effort", None) if gen is not None else None
        tier = tier_of(model)
        for s in log.samples or []:
            md = s.metadata or {}
            usage = s.model_usage or {}
            missing = [m for m, u in usage.items() if u.total_cost is None and not m.startswith("mockllm/")]
            if missing and require_cost:
                raise ValueError(f"{f}: no cost for {missing}; run eval() with **ape.models.eval_cost_kwargs() (config/model_costs.yaml)")
            family = md.get("family", emd.get("family"))
            level = md.get("level", emd.get("level"))
            scores = s.scores or {}
            ts = scores.get("task_success")
            ts_md = (ts.metadata or {}) if ts is not None else {}
            store = s.store or {}
            errored = s.error is not None
            answer, calls = store.get(ANSWER), store.get(CALLS)
            if answer is None and ts is not None and ts.answer and family != "F3":
                try:
                    answer = json.loads(ts.answer)
                except (TypeError, ValueError):
                    answer = None
            success = 1.0 if ts is not None and ts.value == "C" else 0.0
            answered = answer is not None if family != "F3" else not errored
            turns = ts_md.get("turns_used", store.get("turns_used"))
            ea = scores.get("error_analysis")
            ev = scores.get("delivered_evidence")
            meters = _usage(usage)
            rows.append(
                {
                    "plan_cell": plan_cell or emd.get("plan_cell"),
                    "arm": arm,
                    "family": family,
                    "level": str(level),
                    "cell": f"{family}-{level}",
                    "world": md.get("world_id"),
                    "task": s.id,
                    "epoch": int(s.epoch),
                    "tier": tier,
                    "model": model,
                    "effort": effort,
                    "profile": f"{tier}-{effort}" if effort else tier,
                    "success": success,
                    "error": errored,
                    "error_label": ea.metadata.get("error") if ea is not None and ea.metadata else ("harness_error" if errored else None),
                    "answered": bool(answered),
                    "answer_key": None if errored else answer_key(family, answer, calls),
                    "turns_used": turns,
                    "max_turns": max_turns,
                    "cap_hit": (not errored) and ((turns is not None and turns >= max_turns and not ts_md.get("answered", answered)) or s.limit is not None),
                    "limit_hit": s.limit.type if s.limit is not None else None,
                    **meters,
                    "calls": _model_calls(s),
                    "usd": sum((u.total_cost or 0.0) for u in usage.values()),
                    "usd_list": _list_usd(usage, prices),
                    "wall": float(s.working_time) if s.working_time is not None else np.nan,
                    "total_time": s.total_time,
                    "working_time": s.working_time,
                    "multi_agent": bool(emd.get("multi_agent", False)),
                    "delivery": args.get("delivery", emd.get("delivery", "push")),
                    "split": md.get("split", args.get("split")),
                    "partial_credit": ea.value.get("partial_credit", np.nan) if ea is not None and isinstance(ea.value, dict) else np.nan,
                    "evidence_recall": ev.value.get("evidence_recall", np.nan) if ev is not None and isinstance(ev.value, dict) else np.nan,
                    "pipeline_miss": None if success or errored else pipeline_miss(store.get("compile_log", []), store.get("step_log", []), md.get("task") or {}),
                    "per_agent": next((store[k] for k in store_keys if k in store), None),
                    "log_file": str(f),
                }
            )
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    df["cost_usd"] = df["usd"]
    return df


def load_plan_cells(cells: Mapping[str, Sequence[str | Path]], require_cost: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """{plan_cell: [log paths]} -> (the tidy frame, coverage per plan cell: files, samples, and the reason when a
    cell's logs are missing or unreadable). A bad cell is reported, never fatal."""
    frames, coverage = [], []
    for plan_cell, files in cells.items():
        entry = {"plan_cell": plan_cell, "files": len(files or []), "samples": 0, "reason": None}
        missing = [str(f) for f in files or [] if not Path(f).is_file()]
        if not files:
            entry["reason"] = "no log files"
        elif missing:
            entry["reason"] = f"log files missing: {missing}"
        else:
            try:
                df = load_main(files, plan_cell=plan_cell, require_cost=require_cost)
            except (ValueError, OSError, KeyError) as e:
                entry["reason"] = f"logs unreadable: {type(e).__name__}: {e}"
            else:
                frames.append(df)
                entry["samples"] = int(len(df))
        coverage.append(entry)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(COLUMNS))
    return frame, coverage
