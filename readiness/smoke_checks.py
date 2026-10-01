"""Pass/fail logic of the smoke checks (FIX_PLAN FX-8), free of I/O so tests drive it with crafted inputs.

Every check returns `result(status, measured, reason=...)`: status is pass | warn | fail | skip. warn means
the wiring works but a number deserves a look; only fail stops the readiness row from closing.
`readiness/smoke.py` runs the checks; `readiness/probe_openai.py` uses `effort_verdict`.
"""

import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

# Effort honoured: a short problem that rewards thinking, so high effort has room to reason more than low.
EFFORT_PROMPT = (
    "How many integers n with 1 <= n <= 2000 are divisible by 6 or 10 but not by 15? "
    "Work it out, then reply with the number only."
)
EFFORT_MAX_TOKENS = 4096  # room for high effort's reasoning; output is billed, so the cap also bounds cost
EFFORT_ROLES = ("agent", "kg", "build")
PULL_MIN_RATE = 0.75  # pull mode: share of samples that call search_kb at least once
RETRY_WARN_RATE = 0.05  # burst: retried calls above this share halve the recommended concurrency
TPM_HEADROOM = 0.8  # burst: use at most this share of the account's tokens-per-minute limit
MIN_CONCURRENCY = 4
L5_BAND = (0.1, 1.0)  # L5: median realized context as a share of LightRAG's max_total_tokens
RECALL_WARN = 0.8  # real-embedding retrieval: S3s evidence recall and APG shortlist gold rate
RATE_LIMIT = re.compile(r"\b429\b|rate[ _-]?limit", re.IGNORECASE)


def result(status: str, measured: Mapping[str, Any] | None = None, *, reason: str | None = None, **extra: Any) -> dict:
    return {"status": status, "reason": reason, "measured": dict(measured or {})} | extra


def _pct(values: Sequence[float], q: float) -> float | None:
    """The q-quantile (0..1) by nearest rank; None for no values."""
    if not values:
        return None
    s = sorted(values)
    return float(s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))])


def latency(values: Iterable[float | None]) -> dict:
    v = [float(x) for x in values if x is not None]
    r = lambda x: round(x, 3) if x is not None else None  # noqa: E731
    return {"n": len(v), "p50": r(_pct(v, 0.5)), "p90": r(_pct(v, 0.9)), "max": r(max(v)) if v else None}


# --- Effort honoured (probe_openai.py) -------------------------------------------------------------


def reasoning_tokens(usage: Mapping | None) -> int | None:
    details = (usage or {}).get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    return int(value) if value is not None else None


def effort_verdict(high: Mapping, low: Mapping) -> dict:
    """`high` and `low` are probe results ({"accepted", "usage" | "error"}) for the same prompt at reasoning_effort
    high and low. Pass: both accepted and high used more reasoning tokens than low."""
    rejected = [name for name, r in (("high", high), ("low", low)) if not r.get("accepted")]
    if rejected:
        return result(FAIL, {"rejected": rejected}, reason=f"reasoning_effort {'/'.join(rejected)} rejected: " + "; ".join(str((high if n == "high" else low).get("error")) for n in rejected))
    rh, rl = reasoning_tokens(high.get("usage")), reasoning_tokens(low.get("usage"))
    measured = {"reasoning_tokens_high": rh, "reasoning_tokens_low": rl}
    if rh is None or rl is None:
        return result(FAIL, measured, reason="usage has no completion_tokens_details.reasoning_tokens: effort cannot be compared")
    if rh <= rl:
        return result(FAIL, measured, reason=f"high used {rh} reasoning tokens, low {rl}: the effort setting does not change reasoning")
    return result(PASS, measured)


def effort_from_probe(probe: Mapping, roles: Sequence[str] = EFFORT_ROLES) -> dict:
    """The effort check from cache/openai_probe.json: every probed role's `effort` verdict must pass."""
    per_role, missing = {}, []
    for role in roles:
        entry = probe.get(role)
        if not isinstance(entry, Mapping):
            continue
        effort = entry.get("effort")
        if not isinstance(effort, Mapping) or "verdict" not in effort:
            missing.append(role)
            continue
        per_role[role] = {"model": entry.get("model")} | effort["verdict"]
    if missing:
        return result(FAIL, {"roles": per_role}, reason=f"the probe has no effort check for {missing}: re-run readiness/probe_openai.py (FX-8)")
    if not per_role:
        return result(FAIL, {}, reason="the probe probed no role models: run readiness/probe_openai.py with --agent/--kg/--build")
    failed = [r for r, v in per_role.items() if v["status"] != PASS]
    return result(FAIL if failed else PASS, {"roles": per_role}, reason=f"effort not honoured for {failed}" if failed else None)


# --- Pull mode live --------------------------------------------------------------------------------


def pull_verdict(samples: Sequence[Mapping], min_rate: float = PULL_MIN_RATE) -> dict:
    """`samples`: {"arm", "search_kb_calls", "error"}. Pass: every arm calls search_kb in >= min_rate of its
    samples, and no sample errored."""
    if not samples:
        return result(FAIL, {}, reason="no samples")
    arms: dict[str, dict] = {}
    for s in samples:
        a = arms.setdefault(str(s.get("arm")), {"samples": 0, "with_search": 0, "errors": 0})
        a["samples"] += 1
        a["with_search"] += int((s.get("search_kb_calls") or 0) > 0)
        a["errors"] += int(bool(s.get("error")))
    for a in arms.values():
        a["rate"] = a["with_search"] / a["samples"]
    reasons = [f"{arm}: search_kb in {a['rate']:.0%} of samples (< {min_rate:.0%})" for arm, a in arms.items() if a["rate"] < min_rate]
    reasons += [f"{arm}: {a['errors']} errored sample(s)" for arm, a in arms.items() if a["errors"]]
    return result(FAIL if reasons else PASS, {"arms": arms}, reason="; ".join(reasons) or None)


# --- Error recovery --------------------------------------------------------------------------------


def recovery_verdict(log_status: str, samples: Sequence[Mapping]) -> dict:
    """`samples`: {"id", "error", "retries", "retry_errors"}. Pass: the log succeeded, every sample finished
    without an error, and the injected first-attempt fault shows up as a recorded retry."""
    retries = sum(int(s.get("retries") or 0) for s in samples)
    errors = [s.get("id") for s in samples if s.get("error")]
    measured = {"log_status": log_status, "samples": len(samples), "retries": retries, "errored": errors, "retry_errors": [e for s in samples for e in (s.get("retry_errors") or [])][:4]}
    reasons = []
    if log_status != "success":
        reasons.append(f"log status {log_status}")
    if not samples:
        reasons.append("no samples")
    if errors:
        reasons.append(f"sample(s) still errored after retries: {errors}")
    if retries < 1:
        reasons.append("no retry recorded: the injected fault did not go through Inspect's sample retry")
    return result(FAIL if reasons else PASS, measured, reason="; ".join(reasons) or None)


# --- Concurrency burst -----------------------------------------------------------------------------


def rate_limit_signals(event_errors: Iterable[str | None], logger_messages: Iterable[str | None]) -> int:
    """Transcript signs of rate limiting: model-event errors and logger messages that mention 429 or a rate limit."""
    return sum(1 for text in [*event_errors, *logger_messages] if text and RATE_LIMIT.search(text))


def tpm_limit(ratelimit_headers: Mapping[str, str] | None) -> int | None:
    """x-ratelimit-limit-tokens from the probe's response headers (tokens per minute), if recorded."""
    raw = (ratelimit_headers or {}).get("x-ratelimit-limit-tokens")
    try:
        return int(str(raw).replace(",", "")) if raw is not None else None
    except ValueError:
        return None


def recommend_concurrency(
    current: int, calls: int, retries: int, failed: int, tokens_per_call: float | None, call_latency_s: float | None, tpm: int | None
) -> tuple[int, str]:
    """The max_connections to run the gate at, from one burst at `current`.

    Failed samples, or retries on more than RETRY_WARN_RATE of calls, halve it (not below MIN_CONCURRENCY).
    Otherwise it stays, capped by the account's tokens-per-minute limit when the probe recorded one: one
    connection moves about tokens_per_call x 60 / latency tokens a minute, and the gate may use TPM_HEADROOM of
    the limit. A 24-sample burst cannot show headroom above `current`, so it never recommends more."""
    rate = retries / calls if calls else 0.0
    if failed or rate > RETRY_WARN_RATE:
        rec = max(MIN_CONCURRENCY, current // 2)
        why = f"{failed} failed sample(s), {retries} retried call(s) of {calls} ({rate:.0%}): halve to {rec}"
    else:
        rec, why = current, f"{retries} retried call(s) of {calls} ({rate:.0%}), no failed samples: keep {current}"
    if tpm and tokens_per_call and call_latency_s:
        per_connection = tokens_per_call * 60.0 / call_latency_s
        ceiling = max(1, int(TPM_HEADROOM * tpm / per_connection))
        if ceiling < rec:
            rec, why = ceiling, why + f"; the TPM limit {tpm:,} at ~{per_connection:,.0f} tokens/min per connection caps it at {ceiling}"
        else:
            why += f"; the TPM limit {tpm:,} allows up to {ceiling}"
    return rec, why


def burst_verdict(
    max_connections: int,
    samples: Sequence[Mapping],
    model_events: Sequence[Mapping],
    logger_messages: Sequence[str | None] = (),
    ratelimit_headers: Mapping[str, str] | None = None,
) -> dict:
    """`samples`: {"error", "total_time", "working_time"}; `model_events`: {"retries", "error", "working_time",
    "tokens"}. Fail on any failed sample; warn when calls were retried or rate limited; the report always carries
    the latency distribution and a recommended concurrency."""
    failed = sum(1 for s in samples if s.get("error"))
    calls = len(model_events)
    retries = sum(int(e.get("retries") or 0) for e in model_events)
    signals = rate_limit_signals((e.get("error") for e in model_events), logger_messages)
    times = [float(e["working_time"]) for e in model_events if e.get("working_time")]
    toks = [float(e["tokens"]) for e in model_events if e.get("tokens")]
    rec, why = recommend_concurrency(max_connections, calls, retries, failed, statistics.fmean(toks) if toks else None, statistics.fmean(times) if times else None, tpm_limit(ratelimit_headers))
    measured = {
        "max_connections": max_connections,
        "samples": len(samples),
        "failed_samples": failed,
        "model_calls": calls,
        "retried_calls": sum(1 for e in model_events if e.get("retries")),
        "retries": retries,
        "rate_limit_signals": signals,
        "sample_total_time_s": latency(s.get("total_time") for s in samples),
        "sample_working_time_s": latency(s.get("working_time") for s in samples),
        "model_call_time_s": latency(times),
        "recommended_max_connections": rec,
        "recommendation": why,
    }
    if not samples:
        return result(FAIL, measured, reason="no samples")
    if failed:
        return result(FAIL, measured, reason=f"{failed} failed sample(s) at max_connections {max_connections}")
    if retries or signals:
        return result(WARN, measured, reason=f"{retries} retries and {signals} rate-limit signal(s): see the recommendation")
    return result(PASS, measured)


# --- L5 --------------------------------------------------------------------------------------------


def l5_verdict(ctx_tokens: Sequence[int], cap: int, band: tuple[float, float] = L5_BAND) -> dict:
    """LightRAG's realized context per compile against its max_total_tokens. Fail when there is no compile or
    any compile exceeds the cap; warn when the median is outside band x cap (near-empty contexts, or a cap so
    loose the realized size is set by something else)."""
    v = [int(t) for t in ctx_tokens]
    med = statistics.median(v) if v else None
    lo, hi = band[0] * cap, band[1] * cap
    measured = {"cap": cap, "compiles": len(v), "median": med, "max": max(v) if v else None, "band": [lo, hi], "median_share_of_cap": (med / cap) if med is not None and cap else None}
    if not v:
        return result(FAIL, measured, reason="no LightRAG compiles recorded")
    if (over := [t for t in v if t > cap]):
        return result(FAIL, measured, reason=f"{len(over)} compile(s) over max_total_tokens {cap} (max {max(over)})")
    if not lo <= med <= hi:
        return result(WARN, measured, reason=f"median {med:.0f} outside the expected band [{lo:.0f}, {hi:.0f}]")
    return result(PASS, measured)


# --- Real-embedding retrieval ----------------------------------------------------------------------


def retrieval_verdict(s3s_recall: Sequence[float], apg_in_shortlist: Sequence[bool], *, graph: str, dry: bool, threshold: float = RECALL_WARN) -> dict:
    """S3s evidence recall at its budget and the share of tasks whose gold APG node is in the embedding shortlist,
    with no LLM. Live: warn below `threshold`. Dry (fake embeddings): reported, not judged."""
    recall = statistics.fmean(s3s_recall) if s3s_recall else None
    gold = sum(apg_in_shortlist) / len(apg_in_shortlist) if apg_in_shortlist else None
    measured = {"tasks": len(s3s_recall), "s3s_recall": recall, "apg_shortlist_gold_rate": gold, "apg_graph": graph}
    if recall is None or gold is None:
        return result(FAIL, measured, reason="no tasks to retrieve for")
    if dry:
        return result(PASS, measured, reason="dry: fake embeddings, reported only")
    low = [f"{name} {v:.2f}" for name, v in (("S3s recall", recall), ("APG shortlist gold rate", gold)) if v < threshold]
    return result(WARN if low else PASS, measured, reason=f"below {threshold}: " + ", ".join(low) if low else None)


# --- Orchestrator wiring (run_gate at SMOKE_SCALE) -------------------------------------------------

SMOKE_PHASES = ("preflight", "build-dev", "tune", "anchor", "pilot")


def orchestrator_verdict(statuses: Mapping[str, str | None], files: Mapping[str, bool], freeze_error: str | None, *, placeholders: int, provenance_unchanged: bool, config_unchanged: bool) -> dict:
    """`statuses`: phase -> manifest status; `files`: output -> exists; `freeze_error`: the freeze's refusal (None
    if it did not refuse). Pass: preflight..pilot done or skipped, the tuning log, selection and pc1.json written,
    the freeze refused (as a smoke run, and for the pre-registration's open items while it has any), and neither
    PROVENANCE.md nor config/ changed."""
    reasons = [f"{p}: {statuses.get(p) or 'not run'}" for p in SMOKE_PHASES if statuses.get(p) not in ("done", "skipped")]
    reasons += [f"{name} not written" for name, ok in files.items() if not ok]
    if freeze_error is None:
        reasons.append("the freeze did not refuse a smoke run")
    else:
        if "smoke run" not in freeze_error:
            reasons.append("the freeze refused, but not as a smoke run")
        if placeholders and "unfilled item" not in freeze_error:
            reasons.append(f"the freeze did not list the pre-registration's {placeholders} open item(s)")
    if not provenance_unchanged:
        reasons.append("PROVENANCE.md changed")
    if not config_unchanged:
        reasons.append("config/ changed")
    measured = {"phases": dict(statuses), "files": dict(files), "freeze_refusal": (freeze_error or "")[:600], "placeholders": placeholders}
    return result(FAIL if reasons else PASS, measured, reason="; ".join(reasons) or None)


# --- Spend ------------------------------------------------------------------------------------------


class SpendCapExceeded(SystemExit):
    """A step would take spend past --max-usd: the run stops before it."""


def require_step_fits(step: str, projected: float, spent: float, max_usd: float) -> None:
    if spent + projected > max_usd:
        raise SpendCapExceeded(f"stopping before {step}: spent ${spent:.4f} + projected ${projected:.4f} exceeds --max-usd {max_usd}")


def plan_fits(projections: Mapping[str, float], max_usd: float) -> tuple[bool, float]:
    total = float(sum(projections.values()))
    return total <= max_usd, total
