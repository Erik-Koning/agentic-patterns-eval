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
EFFORT_ROLES = ("agent", "kg", "build")  # every probe must cover these; other probed roles (judge, probe) count too
PROBE_VERSION = 2  # cache/openai_probe.json layout: roles -> call paths (readiness/probe_openai.py)
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
    """Reasoning tokens from either call path's usage: Inspect's `ModelUsage` (`reasoning_tokens`, from the
    Responses API's output_tokens_details) or a chat completion's (`completion_tokens_details.reasoning_tokens`)."""
    usage = usage or {}
    value = usage.get("reasoning_tokens")
    if value is None:
        value = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
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
        return result(FAIL, measured, reason="usage reports no reasoning tokens: effort cannot be compared")
    if rh <= rl:
        return result(FAIL, measured, reason=f"high used {rh} reasoning tokens, low {rl}: the effort setting does not change reasoning")
    return result(PASS, measured)


def effort_from_probe(probe: Mapping, roles: Sequence[str] = EFFORT_ROLES) -> dict:
    """The effort check from cache/openai_probe.json (version 2: roles -> call paths).

    Every probed role must pass on each call path it uses: the Inspect path (`inspect`, the Responses API for GPT-6)
    for agent, kg, judge and probe; the build client (`build`) for build. On each, the role's own configuration
    must be accepted, as must a strict structured-output call, and reasoning effort must be honoured (high uses
    more reasoning tokens than low; a role that sets no effort is skipped). `roles` must all be probed."""
    if probe.get("version") != PROBE_VERSION or not isinstance(probe.get("roles"), Mapping):
        return result(FAIL, {}, reason="cache/openai_probe.json predates the call-path probe: re-run readiness/probe_openai.py")
    per_path: dict[str, dict] = {}
    problems: list[str] = []
    for role, entry in probe["roles"].items():
        for path, rec in (entry.get("paths") or {}).items():
            key = f"{role}/{path}"
            verdict = ((rec.get("effort") or {}).get("verdict")) or result(FAIL, {}, reason="no effort verdict")
            per_path[key] = {"model": entry.get("model"), "api": rec.get("api"), "snapshot": rec.get("snapshot")} | verdict
            if verdict["status"] == FAIL:
                problems.append(f"{key}: effort not honoured ({verdict.get('reason')})")
            for call in ("as_configured", "structured_output"):
                if not (rec.get(call) or {}).get("accepted"):
                    problems.append(f"{key}: {call.replace('_', ' ')} call rejected ({(rec.get(call) or {}).get('error')})")
    if missing := [r for r in roles if r not in probe["roles"]]:
        problems.append(f"the probe did not cover {missing}: re-run readiness/probe_openai.py --profile <profile>")
    snapshots = probe.get("snapshots") or {}
    measured = {"paths": per_path, "snapshots": {a: (v or {}).get("snapshot") for a, v in snapshots.items()}}
    if inconsistent := sorted(a for a, v in snapshots.items() if len(set((v or {}).get("snapshots_seen") or [])) > 1):
        problems.append(f"served by more than one snapshot within the probe: {inconsistent}")
    return result(FAIL if problems else PASS, measured, reason="; ".join(problems) or None)


def ratelimit_headers(probe: Mapping | None, model: str) -> dict | None:
    """The probe's x-ratelimit-* headers for `model` (bare or "provider/model"), for the burst check's TPM cap."""
    entry = ((probe or {}).get("ratelimits") or {}).get(model.split("/", 1)[-1]) or {}
    return entry.get("headers") or None


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


def retrieval_verdict(
    s3s_recall: Sequence[float],
    apg_in_shortlist: Sequence[bool],
    *,
    graph: str,
    dry: bool,
    threshold: float = RECALL_WARN,
    lightrag_recall: Sequence[float] | None = None,
    lightrag_note: str | None = None,
) -> dict:
    """S3s evidence recall at its budget and the share of tasks whose gold APG node is in the embedding shortlist,
    with no LLM; facts are credited by the delivered-text rule (`ape.kb.provenance`), as the arms do. Live: warn
    below `threshold`. Dry (fake embeddings): reported, not judged. `lightrag_recall` (LightRAG's context with the
    query as its keywords, no kg call) is reported, never judged: the live arm extracts keywords first."""
    recall = statistics.fmean(s3s_recall) if s3s_recall else None
    gold = sum(apg_in_shortlist) / len(apg_in_shortlist) if apg_in_shortlist else None
    measured = {"tasks": len(s3s_recall), "s3s_recall": recall, "apg_shortlist_gold_rate": gold, "apg_graph": graph}
    measured["lightrag_recall_query_keywords"] = statistics.fmean(lightrag_recall) if lightrag_recall else None
    if lightrag_note:
        measured["lightrag_note"] = lightrag_note
    if recall is None or gold is None:
        return result(FAIL, measured, reason="no tasks to retrieve for")
    if dry:
        return result(PASS, measured, reason="dry: fake embeddings, reported only")
    low = [f"{name} {v:.2f}" for name, v in (("S3s recall", recall), ("APG shortlist gold rate", gold)) if v < threshold]
    return result(WARN if low else PASS, measured, reason=f"below {threshold}: " + ", ".join(low) if low else None)


# --- F8 session live (RELIABILITY_REVIEW L12) ------------------------------------------------------


def f8_session_verdict(arms: Mapping[str, Mapping], categories: Sequence[str]) -> dict:
    """One short F8 session per session arm (CM0, O-state) through the real Study G task. Each arm's record:
    {"log_status", "sample_error", "n_cases", "items", "views", "probes", "report_submitted", "report_nudged",
    "probe_in_history", "reasoning_only_turns", "reasoning_only_unanswered"}.

    Pass, per arm: the log succeeded with no sample error; every case has a per-item record carrying its view
    tokens; the report was submitted or its nudge fired; at least one forked probe ran and returned schema-valid
    JSON (every category a list of strings) without entering the history; every recorded view has tokens; and a
    reasoning-only assistant turn (no text, no tool call), if the model produced one, was followed by a further
    call rather than ending the session."""
    if not arms:
        return result(FAIL, {}, reason="no session arms ran")
    reasons: list[str] = []
    measured: dict[str, dict] = {}
    for arm, r in arms.items():
        items, probes, views = list(r.get("items") or []), list(r.get("probes") or []), list(r.get("views") or [])
        bad_probes = [
            p.get("k")
            for p in probes
            if p.get("error") or not isinstance(p.get("answer"), Mapping) or any(not isinstance(p["answer"].get(c), list) or not all(isinstance(x, str) for x in p["answer"][c]) for c in categories)
        ]
        no_view = [i.get("position") for i in items if not i.get("view_tokens_first")]
        measured[arm] = {
            "log_status": r.get("log_status"),
            "cases": r.get("n_cases"),
            "item_records": len(items),
            "calls": len(views),
            "view_tokens_max": max((v.get("view_tokens") or 0 for v in views), default=None),
            "probes": len(probes),
            "report_submitted": bool(r.get("report_submitted")),
            "report_nudged": bool(r.get("report_nudged")),
            "reasoning_only_turns": int(r.get("reasoning_only_turns") or 0),
            "scores": dict(r.get("scores") or {}),
        }
        if r.get("log_status") != "success":
            reasons.append(f"{arm}: log status {r.get('log_status')}")
        if r.get("sample_error"):
            reasons.append(f"{arm}: sample error {str(r['sample_error'])[:160]}")
        if len(items) != int(r.get("n_cases") or 0):
            reasons.append(f"{arm}: {len(items)} item record(s) for {r.get('n_cases')} case(s)")
        if no_view:
            reasons.append(f"{arm}: item(s) {no_view} without view tokens")
        if not (r.get("report_submitted") or r.get("report_nudged")):
            reasons.append(f"{arm}: no report submitted and no report nudge")
        if not probes:
            reasons.append(f"{arm}: no forked probe ran")
        if bad_probes:
            reasons.append(f"{arm}: probe(s) at k={bad_probes} not schema-valid JSON")
        if r.get("probe_in_history"):
            reasons.append(f"{arm}: the probe question entered the session history")
        if not views or any(not v.get("view_tokens") for v in views):
            reasons.append(f"{arm}: calls without view tokens")
        if r.get("reasoning_only_unanswered"):
            reasons.append(f"{arm}: a reasoning-only turn ended the session (no further call)")
    return result(FAIL if reasons else PASS, {"arms": measured}, reason="; ".join(reasons) or None)


# --- The studies' new arm families (BUILD_PLAN B12) -------------------------------------------------
# One small task set or one short session per family, run as `ape.run_study` runs them (the cache nonce set, the
# wall-clock guard on). Each check combines its parts (`combine`): the arm's records, the effort sent on every call, the
# cache nonce on every agent prompt, and the realised cost against the check's projection.

SANE_STOPS = ("done", "text", "turn_cap")  # how a main-study agent may end its loop in a healthy sample
SANE_WORKER_STOPS = ("reported", "text", "turn_cap")  # a session team's worker
MAS_RECORDS = ("mas_switches", "mas_params", "mas_agents", "mas_accounting", "compile_log", "step_log", "turns_used", "arm")
MAS_ARM_RECORDS = {"M1": ("mas_plan", "mas_rounds"), "M1k": ("mas_plan", "mas_rounds"), "M7": ("mas_council",), "S8k3": ("mas_ensemble",)}
MAS_ARM_ROLES = {"M1": {"orchestrator": 1, "worker": 1}, "M1k": {"orchestrator": 1, "worker": 1}, "M7": {"member": 3, "chair": 1}, "S8k3": {"attempt": 3}}
MAS_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "input_tokens_cache_read", "input_tokens_cache_write")
SESSION_MANAGEMENT = {"CM-sum": ("summary", "sum_drop"), "CM-todo": ("todo_extract", "todo_drop")}  # cm purpose, T_abs drop event
COST_BAND = (0.05, 2.0)  # live: realised / projected (conservative) cost of a check; outside it, a cost-model finding


def combine(parts: Mapping[str, Mapping]) -> dict:
    """One check from named parts: the worst status (fail > warn > pass), every part's reason prefixed with its name,
    each part's result under `measured`."""
    statuses = [p["status"] for p in parts.values()]
    status = FAIL if FAIL in statuses else WARN if WARN in statuses else PASS
    reasons = [f"{name}: {p['reason']}" for name, p in parts.items() if p.get("reason") and p["status"] != PASS]
    return result(status, {name: dict(p) for name, p in parts.items()}, reason="; ".join(reasons) or None)


def effort_sent_verdict(calls: Sequence[Mapping], expected: Mapping[str, str | None]) -> dict:
    """The configured effort reached every model call of the check (the probe's `effort` check shows the provider
    honours it). `calls`: {"role" (None = the agent model), "effort", "reasoning_tokens"}; `expected`: role -> effort
    (the profile's; calls on the agent model, management and probe calls included, expect the agent's)."""
    seen: dict[str, dict] = {}
    wrong = []
    for c in calls:
        role = c.get("role") or "agent"
        want = expected.get(role, expected.get("agent"))
        s = seen.setdefault(role, {"calls": 0, "effort": want, "reasoning_tokens": 0})
        s["calls"] += 1
        s["reasoning_tokens"] += int(c.get("reasoning_tokens") or 0)
        if want is not None and c.get("effort") != want:
            wrong.append(f"{role} call sent effort {c.get('effort')!r}, the profile's is {want!r}")
    if not calls:
        return result(FAIL, {"roles": seen}, reason="no model calls")
    return result(FAIL if wrong else PASS, {"roles": seen}, reason="; ".join(sorted(set(wrong))) or None)


def nonce_verdict(expected: str | None, metadata: Sequence[str | None], calls: Sequence[Mapping]) -> dict:
    """The per-run cache nonce (`ape.agent.cache_nonce`): every sample records the task's nonce, and every call on the
    agent model (agents, management calls, probes) opens with its line. `calls`: {"kind", "prefixed"}."""
    measured = {"nonce": expected, "samples": len(metadata), "agent_model_calls": len(calls), "prefixed": sum(bool(c.get("prefixed")) for c in calls)}
    if not expected:
        return result(FAIL, measured, reason="the task carries no cache nonce (APE_CACHE_NONCE was not set when it was built)")
    reasons = []
    if bad := [m for m in metadata if m != expected]:
        reasons.append(f"{len(bad)} sample(s) record cache_nonce {sorted(set(map(str, bad)))}, not {expected}")
    if missing := [c.get("kind") for c in calls if not c.get("prefixed")]:
        reasons.append(f"{len(missing)} agent-model call(s) do not open with the nonce line ({sorted(set(map(str, missing)))})")
    if not calls:
        reasons.append("no agent-model calls")
    return result(FAIL if reasons else PASS, measured, reason="; ".join(reasons) or None)


def cost_band_verdict(realised: float | None, projected: float, dry: bool, band: tuple[float, float] = COST_BAND) -> dict:
    """The check's realised cost against its conservative projection. Dry: not measured (mock models are unpriced).
    Live: $0 means the calls were not metered (fail); outside `band` x the projection is a finding about the cost
    model's priors for this arm (warn), as the cost-model check's."""
    measured = {"realised_usd": None if realised is None else round(realised, 6), "projected_usd": round(projected, 6), "band": list(band)}
    if dry:
        return result(PASS, measured, reason="dry: mock models are unpriced, cost not measured")
    if realised is None or realised <= 0:
        return result(FAIL, measured, reason="no metered cost: the check's calls were not priced")
    ratio = realised / projected if projected > 0 else float("inf")
    measured["ratio"] = round(ratio, 3)
    if not band[0] <= ratio <= band[1]:
        return result(WARN, measured, reason=f"realised ${realised:.4f} is {ratio:.2f}x the projection ${projected:.4f} (band {band[0]}-{band[1]}x): recalibrate this arm's priors")
    return result(PASS, measured)


def _usage_mismatch(acc: Mapping | None, usage: Mapping | None, fields: Sequence[str]) -> list[str]:
    return [f"{f} {int((acc or {}).get(f) or 0)} vs {int((usage or {}).get(f) or 0)}" for f in fields if int((acc or {}).get(f) or 0) != int((usage or {}).get(f) or 0)]


def mas_verdict(arm: str, samples: Sequence[Mapping]) -> dict:
    """A main-study multi-agent or ensemble arm's samples (`readiness/smoke.py` `_mas_record`). Each: {"label",
    "log_status", "sample_error", "limit", "store_keys", "accounting" {"error", "unattributed", "totals"}, "usage"
    (the sample's Inspect usage summed over models), "agents" [{"id", "role", "stop", "error"}], "answered",
    "kg_calls"}.

    Fail, per sample: the log or sample errored or a sample limit fired; a record is missing (MAS_RECORDS and the arm's
    own); `mas_accounting` failed, left calls unattributed or does not sum to the sample's usage on every token field;
    an agent recorded an error or stopped otherwise than done, text or turn cap; the arm's roles did not all run; M1k's
    workers made no kg call. Warn: a sample never answered (a model finding, not the harness)."""
    if not samples:
        return result(FAIL, {}, reason=f"{arm}: no samples")
    reasons: list[str] = []
    warnings: list[str] = []
    rows = []
    for s in samples:
        tag = f"{arm}{' ' + s['label'] if s.get('label') else ''} {s.get('id', '')}".strip()
        agents = list(s.get("agents") or [])
        roles: dict[str, int] = {}
        for a in agents:
            roles[str(a.get("role"))] = roles.get(str(a.get("role")), 0) + 1
        acc = s.get("accounting") or {}
        rows.append({"sample": tag, "agents": len(agents), "roles": roles, "stops": sorted({str(a.get("stop")) for a in agents}), "kg_calls": s.get("kg_calls"),
                     "total_tokens": (s.get("usage") or {}).get("total_tokens"), "answered": bool(s.get("answered"))})  # fmt: skip
        if s.get("log_status") != "success":
            reasons.append(f"{tag}: log status {s.get('log_status')}")
        if s.get("sample_error"):
            reasons.append(f"{tag}: sample error {str(s['sample_error'])[:160]}")
        if s.get("limit"):
            reasons.append(f"{tag}: a sample limit fired ({s['limit']})")
        if missing := [k for k in (*MAS_RECORDS, *MAS_ARM_RECORDS.get(arm, ())) if k not in set(s.get("store_keys") or [])]:
            reasons.append(f"{tag}: records missing {missing}")
        if acc.get("error"):
            reasons.append(f"{tag}: mas_accounting failed ({acc['error']})")
        if acc.get("unattributed"):
            reasons.append(f"{tag}: calls outside every agent span {sorted(acc['unattributed'])}")
        if off := _usage_mismatch(acc.get("totals"), s.get("usage"), MAS_TOKEN_FIELDS):
            reasons.append(f"{tag}: mas_accounting does not sum to the sample's usage ({'; '.join(off)})")
        if errs := [f"{a.get('id')}: {str(a['error'])[:120]}" for a in agents if a.get("error")]:
            reasons.append(f"{tag}: agent error(s) {errs}")
        if odd := [f"{a.get('id')}={a.get('stop')}" for a in agents if a.get("stop") not in SANE_STOPS]:
            reasons.append(f"{tag}: agent stop(s) {odd} (expected one of {list(SANE_STOPS)})")
        if short := [f"{role} ({roles.get(role, 0)} of {n})" for role, n in MAS_ARM_ROLES.get(arm, {}).items() if roles.get(role, 0) < n]:
            reasons.append(f"{tag}: roles that did not run: {short}")
        if arm == "M1k" and not s.get("kg_calls"):
            reasons.append(f"{tag}: the KG workers made no kg call")
        if not s.get("answered"):
            warnings.append(f"{tag}: never answered")
    status = FAIL if reasons else WARN if warnings else PASS
    return result(status, {"arm": arm, "samples": rows}, reason="; ".join(reasons + warnings) or None)


def session_arm_verdict(arm: str, rec: Mapping, categories: Sequence[str], *, window: int, threshold: int, dry: bool) -> dict:
    """One short F8 session of a Study G arm (`readiness/smoke.py` `_session_record`, plus "views" with each call's
    kind, purpose and usage, "cm_events", "policy_tools", "overflow", "usage_by_kind", "inspect_usage", and for the
    team "mas_agents" and "mas_accounting").

    The session itself as `f8_session_verdict` judges it (records per case, the report, schema-valid forked probes
    kept out of the history), and: W enforced (no recorded call over `window`, no overflow); every call's kind is
    agent or cm, probes apart; the Inspect usage equals the per-call records' by kind. CM-sum and CM-todo: their
    management call fired (`summary`, `todo_extract`), is metered as `cm` (live: with usage), and the view crossed T_abs
    and the arm dropped the completed cases (`sum_drop`, `todo_drop`); CM-todo warns when the agent never wrote the
    todo list (todo_write). M1: workers ran, none stopped on a limit, an overflow or an interruption, and
    `mas_accounting` sums to the session's agent usage."""
    base = f8_session_verdict({arm: rec}, categories)
    reasons = [base["reason"]] if base["status"] == FAIL and base.get("reason") else []
    warnings: list[str] = []
    views = list(rec.get("views") or [])
    probes = list(rec.get("probes") or [])
    cm = [v for v in views if v.get("kind") == "cm"]
    by_kind = rec.get("usage_by_kind") or {}
    measured = dict((base.get("measured") or {}).get("arms", {}).get(arm, {})) | {
        "window": window,
        "threshold": threshold,
        "max_call_tokens": max((int(v.get("view_tokens") or 0) for v in [*views, *probes]), default=0),
        "overflow": rec.get("overflow"),
        "kinds": sorted({str(v.get("kind")) for v in views}),
        "cm_calls": {p: sum(v.get("purpose") == p for v in cm) for p in sorted({str(v.get("purpose")) for v in cm})},
        "cm_events": sorted({str(e.get("event")) for e in rec.get("cm_events") or []}),
        "policy_tool_calls": len(rec.get("policy_tools") or []),
    }
    if over := [v.get("view_tokens") for v in [*views, *probes] if int(v.get("view_tokens") or 0) > window]:
        reasons.append(f"{len(over)} call(s) over W={window} were sent (max {max(over)})")
    if rec.get("overflow") is not None:
        reasons.append(f"the session overflowed W={window} at item {rec['overflow']}: the smoke's window is too small for this model")
    if odd := sorted({str(v.get("kind")) for v in views} - {"agent", "cm"}):
        reasons.append(f"calls of unexpected kind {odd}")
    usage_total = sum(int((u or {}).get("total_tokens") or 0) for u in by_kind.values())
    if int(rec.get("inspect_usage") or 0) != usage_total:
        reasons.append(f"Inspect's usage ({rec.get('inspect_usage')} tokens) is not the per-call records' ({usage_total}, by kind)")
    if arm in SESSION_MANAGEMENT:
        purpose, drop = SESSION_MANAGEMENT[arm]
        mine = [v for v in cm if v.get("purpose") == purpose]
        if not mine:
            reasons.append(f"no {purpose} management call (kind cm) was made")
        elif not dry and (bad := [v for v in mine if not int(((v.get("usage") or {}).get("total_tokens")) or 0)]):
            reasons.append(f"{len(bad)} {purpose} call(s) recorded without usage: management is not metered")
        if drop not in measured["cm_events"]:
            reasons.append(f"the view never crossed T_abs={threshold} with completed cases to drop (no {drop} event): management never fired")
        if arm == "CM-todo" and not rec.get("policy_tools"):
            warnings.append("the agent never called todo_write (the list came from extraction only)")
    if arm in ("M1", "M2"):
        workers = [a for a in rec.get("mas_agents") or [] if a.get("role") != "orchestrator"]
        measured["workers"] = len(workers)
        measured["worker_stops"] = sorted({str(a.get("stop")) for a in workers})
        if not workers:
            reasons.append("no worker ran: the orchestrator never delegated, so the team was not exercised")
        if odd := [f"{a.get('id')}={a.get('stop')}" for a in workers if a.get("stop") not in SANE_WORKER_STOPS]:
            reasons.append(f"worker stop(s) {odd} (expected one of {list(SANE_WORKER_STOPS)})")
        totals = (rec.get("mas_accounting") or {}).get("totals") or {}
        agent_usage = by_kind.get("agent") or {}
        if int(totals.get("calls") or 0) != sum(1 for v in views if v.get("kind") == "agent"):
            reasons.append(f"mas_accounting counts {totals.get('calls')} agent calls, the session recorded {sum(1 for v in views if v.get('kind') == 'agent')}")
        if off := _usage_mismatch(totals, agent_usage, ("input_tokens", "output_tokens", "total_tokens")):
            reasons.append(f"mas_accounting does not sum to the session's agent usage ({'; '.join(off)})")
    status = FAIL if reasons else WARN if warnings else PASS
    return result(status, measured, reason="; ".join(reasons + warnings) or None)


# --- Real extraction on one F7-1000 world ----------------------------------------------------------


def extraction_verdict(build_errors: Mapping[str, str], quality: Mapping, health_warnings: Sequence[str], threshold: float) -> dict:
    """The real builder (APG authoring and LightRAG extraction) on one F7-1000 dev world. Fail when a build kind
    failed outright (the world could not be built); warn when coverage is under D-017's threshold or the build
    lost chunks (`ape.build_quality`): like D017 a finding (the Sol fallback needs approval), not a broken harness.
    `quality`: {"apg_id_coverage", "lightrag_id_coverage", "verdict", ...}."""
    measured = dict(quality) | {"threshold": threshold, "health_warnings": list(health_warnings)}
    if build_errors:
        return result(FAIL, measured, reason="build failed: " + "; ".join(f"{k}: {v}" for k, v in build_errors.items()))
    low = [f"{name} {v:.3f}" for name, v in (("APG ID coverage", quality.get("apg_id_coverage")), ("LightRAG ID coverage", quality.get("lightrag_id_coverage"))) if v is not None and v < threshold]
    missing = [name for name, key in (("APG ID coverage", "apg_id_coverage"), ("LightRAG ID coverage", "lightrag_id_coverage")) if quality.get(key) is None]
    reasons = ([f"below {threshold}: " + ", ".join(low)] if low else []) + ([f"not measured: {', '.join(missing)}"] if missing else []) + list(health_warnings)
    if reasons:
        return result(WARN, measured, reason="; ".join(reasons) + (" (D-017: the Sol fallback builder needs approval)" if low else ""))
    return result(PASS, measured)


# --- Per-step reasoning carry-over (live confirmation of RELIABILITY_REVIEW L2) ---------------------


def perstep_verdict(calls: Sequence[Mapping]) -> dict:
    """The model inputs of a per-step arm's calls. Each call: {"sample", "turn", "after_tool_step",
    "kb_in_last_tool", "user_after_assistant", "reasoning_items", "reasoning_after_last_user", "reasoning_tokens",
    "error"}.

    Pass (structural, the part that can be checked): at least one call followed a tool step, and every such call's
    input ends with the tool result carrying the step's knowledge block, with no user message after the model's
    last turn (so OpenAI keeps the earlier reasoning); no call errored. The carried reasoning items and reasoning
    tokens are reported, not judged: whether the model uses them is not observable from the outside."""
    after = [c for c in calls if c.get("after_tool_step")]
    errors = [f"sample {c.get('sample')} turn {c.get('turn')}: {str(c['error'])[:120]}" for c in calls if c.get("error")]
    broken = [f"sample {c.get('sample')} turn {c.get('turn')}" for c in after if not c.get("kb_in_last_tool") or c.get("user_after_assistant")]
    carried = [c.get("reasoning_after_last_user") for c in after if c.get("reasoning_after_last_user") is not None]
    measured = {
        "calls": len(calls),
        "calls_after_tool_step": len(after),
        "structure_ok": len(after) - len(broken),
        "reasoning_items_carried_per_call": carried,
        "reasoning_tokens_per_call": [c.get("reasoning_tokens") for c in calls],
        "note": "reasoning carry-over is reported only: the structure is what the harness controls",
    }
    reasons = errors + ([f"knowledge not in the last tool result, or a user message after the model's turn: {broken}"] if broken else [])
    if not after:
        reasons.append("no per-step call followed a tool step: the check was not exercised")
    return result(FAIL if reasons else PASS, measured, reason="; ".join(reasons) or None)


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


def anchor_parse_rates(pc1: Mapping[str, Any] | None) -> dict:
    """Judge parse-failure rates from the smoke anchor's pc1.json, both scorers (D-025).

    With 2 questions per type this is a wiring check: does the judge's reply format parse? Any gating-scorer
    failure is a WARN, because above 5% at full size PC1 is not evaluable. The strict official scorer's rate is
    reported: a high rate there is the GraphRAG-Benchmark PR #56 parse collapse."""
    if not pc1:
        return {"status": None, "warn": None}
    gating = {t: x.get("parse_failure_rate") for t, x in (pc1.get("per_type") or {}).items()}
    current = dict(pc1.get("current_scorer_parse_failure_rate") or {})
    fix = dict(pc1.get("fix_format_rate") or {})
    warn = [f"gating {t} {r:.0%}" for t, r in gating.items() if r] + [f"gating {t} unknown" for t, r in gating.items() if r is None]
    return {"status": pc1.get("status"), "gating": gating, "current_strict": current, "ragas_fix_format": fix, "warn": "; ".join(warn) or None}


# --- Spend ------------------------------------------------------------------------------------------


class SpendCapExceeded(SystemExit):
    """A step would take spend past --max-usd: the run stops before it."""


def require_step_fits(step: str, projected: float, spent: float, max_usd: float) -> None:
    if spent + projected > max_usd:
        raise SpendCapExceeded(f"stopping before {step}: spent ${spent:.4f} + projected ${projected:.4f} exceeds --max-usd {max_usd}")


def plan_fits(projections: Mapping[str, float], max_usd: float) -> tuple[bool, float]:
    total = float(sum(projections.values()))
    return total <= max_usd, total


# --- Cost-model check (a live smoke's own logs and ledger against the budget priors) -------------------------
# The smoke's calls are few, so these numbers are indicative; they flag a prior that is badly off before the
# gate spends on it. Inspect's output tokens include reasoning tokens (`ape.budget.calibrate`).

AGENT_LIKE_ROLES = ("agent", "probe", "cm")  # priced with the effort table (budget_assumptions output_tokens_per_call)
BUILD_SYSTEMS = {"apg-author": "apg_author", "lightrag": "lightrag_extract"}  # ledger context system -> build_calls key


def _role_effort(role: str, entry_effort: str | None, role_efforts: Mapping[str, str | None]) -> str | None:
    if role == "agent" or role not in role_efforts:
        return entry_effort
    return role_efforts[role]


def measured_per_call(entries: Sequence[Mapping], role_efforts: Mapping[str, str | None]) -> dict[str, dict]:
    """Per role and effort, over `ape.budget.calibrate` entries weighted by calls: calls, and input and output
    tokens per call. The agent's effort is the entry's; another role's is its profile effort when it has one."""
    acc: dict[str, dict] = {}
    for e in entries:
        for role, r in (e.get("roles") or {}).items():
            calls = float(r["calls_per_sample"]) * float(e["samples"])
            if calls <= 0:
                continue
            effort = _role_effort(role, e.get("effort"), role_efforts)
            a = acc.setdefault(f"{role}@{effort or 'default'}", {"role": role, "effort": effort, "model": r.get("model"), "calls": 0.0, "input": 0.0, "output": 0.0})
            a["calls"] += calls
            a["input"] += calls * float(r["input_per_call"])
            a["output"] += calls * float(r["output_per_call"])
    return {
        k: {"role": a["role"], "effort": a["effort"], "model": a["model"], "calls": round(a["calls"], 2), "input_per_call": round(a["input"] / a["calls"], 1), "output_per_call": round(a["output"] / a["calls"], 1)}
        for k, a in sorted(acc.items())
    }


def prior_output_per_call(role: str, effort: str | None, assumptions: Mapping) -> float | None:
    """The budget's prior output tokens per call (reasoning included) for a role; None where it has none."""
    if role in AGENT_LIKE_ROLES:
        table = assumptions["output_tokens_per_call"]
        return float(table.get(effort or "default", table["default"]))
    if role == "kg":
        outs = [float(v["output"]) for v in (assumptions.get("kg_calls") or {}).values()]
        return statistics.fmean(outs) if outs else None
    return None


def output_vs_prior(measured: Mapping[str, Mapping], assumptions: Mapping, reasoning: Mapping[str, tuple[float, float]] | None = None) -> list[dict]:
    """Measured against prior output tokens per call, per role and effort; `reasoning` (role -> (reasoning tokens,
    output tokens)) adds each role's reasoning share of its output."""
    rows = []
    for m in measured.values():
        prior = prior_output_per_call(m["role"], m["effort"], assumptions)
        r = (reasoning or {}).get(m["role"])
        rows.append(
            {
                "role": m["role"], "effort": m["effort"], "model": m["model"], "calls": m["calls"],
                "input_per_call": m["input_per_call"], "output_per_call": m["output_per_call"], "prior_output_per_call": prior,
                "ratio": round(m["output_per_call"] / prior, 3) if prior else None,
                "reasoning_share": round(r[0] / r[1], 3) if r and r[1] else None,
            }
        )  # fmt: skip
    return rows


def prior_calls_per_sample(cell: str, delivery: str, assumptions: Mapping) -> float | None:
    """The budget's prior agent generations per sample for a measured cell (F8 sessions: N x calls per item)."""
    family, _, level = cell.partition("-")
    if family == "F8":
        try:
            return float(level) * float(assumptions["study_g"]["calls_per_item"])
        except (KeyError, ValueError):
            return None
    table = assumptions.get("calls_per_sample") or {}
    for key in (cell, family):
        if delivery in (table.get(key) or {}):
            return float(table[key][delivery])
    return None


def calls_vs_prior(entries: Sequence[Mapping], assumptions: Mapping) -> list[dict]:
    """Per measured (arm, cell, delivery): the agent's calls per sample and input per call against the priors."""
    rows = []
    for e in entries:
        agent = (e.get("roles") or {}).get("agent")
        if not agent:
            continue
        prior = prior_calls_per_sample(e["cell"], e.get("delivery", "push"), assumptions)
        rows.append(
            {
                "arm": e["arm"], "cell": e["cell"], "delivery": e.get("delivery", "push"), "samples": e["samples"],
                "calls_per_sample": agent["calls_per_sample"], "prior_calls_per_sample": prior,
                "ratio": round(agent["calls_per_sample"] / prior, 3) if prior else None, "input_per_call": agent["input_per_call"],
            }
        )  # fmt: skip
    return rows


def _world_cell(world_id: str) -> str:
    """`F7-1000-rel-desc-dev-s1000` -> `F7-1000`."""
    return "-".join(world_id.split("-")[:2])


def build_per_call(rows: Sequence[Mapping], assumptions: Mapping) -> dict[str, dict]:
    """Finished build calls in the ledger (role build; unfinished ones carry a `status`) of the gate's worlds, per
    system against the build_calls priors: calls, input, output and reasoning tokens per call, and calls per chunk
    where every world's chunk count is known (budget_assumptions chunks_per_world). Rows with no world (the PC1
    anchor's index, built by another model on another corpus) are not the gate's builds and would skew both."""
    chunks = assumptions.get("chunks_per_world") or {}
    acc: dict[str, dict] = {}
    for r in rows:
        ctx = r.get("context") or {}
        key = BUILD_SYSTEMS.get(ctx.get("system"))
        if r.get("role") != "build" or key is None or ctx.get("status") or not ctx.get("world"):
            continue
        a = acc.setdefault(key, {"calls": 0, "input": 0.0, "output": 0.0, "reasoning": 0.0, "worlds": set()})
        a["calls"] += 1
        a["input"] += float(r.get("input_tokens") or 0)
        a["output"] += float(r.get("output_tokens") or 0)
        a["reasoning"] += float(r.get("reasoning_tokens") or 0)
        if ctx.get("world"):
            a["worlds"].add(ctx["world"])
    out = {}
    for key, a in sorted(acc.items()):
        prior = (assumptions.get("build_calls") or {}).get(key) or {}
        cells = [_world_cell(w) for w in sorted(a["worlds"])]
        known = bool(cells) and all(c in chunks for c in cells)
        per_chunk = round(a["calls"] / sum(float(chunks[c]) for c in cells), 3) if known else None
        out[key] = {
            "calls": a["calls"], "worlds": sorted(a["worlds"]),
            "input_per_call": round(a["input"] / a["calls"], 1), "output_per_call": round(a["output"] / a["calls"], 1),
            "reasoning_per_call": round(a["reasoning"] / a["calls"], 1), "calls_per_chunk": per_chunk,
            "prior": {k: prior.get(k) for k in ("input", "output", "calls_per_chunk")},
        }  # fmt: skip
    return out


def adjusted_assumptions(assumptions: Mapping, outputs: Sequence[Mapping], builds: Mapping[str, Mapping]) -> dict:
    """The priors with the smoke's measurements in place, for a whole-program re-projection: the agent's output
    tokens per call at each measured effort, the kg calls' output, and each measured build system's input,
    output and calls per chunk. Everything not measured keeps its prior."""
    import copy

    A = copy.deepcopy(dict(assumptions))
    for row in outputs:
        if row["role"] == "agent" and row["effort"]:
            A["output_tokens_per_call"][row["effort"]] = row["output_per_call"]
        elif row["role"] == "kg":
            for spec in (A.get("kg_calls") or {}).values():
                spec["output"] = row["output_per_call"]
    for key, b in builds.items():
        spec = A["build_calls"][key]
        spec["input"], spec["output"] = b["input_per_call"], b["output_per_call"]
        if b.get("calls_per_chunk") is not None:
            spec["calls_per_chunk"] = b["calls_per_chunk"]
    return A


def cost_verdict(projections: Mapping[str, Mapping[str, float]], total_usd: float, gate_usd: float) -> dict:
    """WARN when a measured re-projection of the program exceeds the program budget, or the gate its allocation.
    `projections`: name -> {"total": $, "gate": $} (conservative); "baseline" is the current priors' projection."""
    reasons = []
    for name, p in projections.items():
        if name == "baseline":
            continue
        if p["total"] > total_usd:
            reasons.append(f"{name}: program ${p['total']:,.0f} > ${total_usd:,.0f}")
        if p["gate"] > gate_usd:
            reasons.append(f"{name}: gate ${p['gate']:,.0f} > its ${gate_usd:,.0f} allocation")
    return {"status": WARN if reasons else PASS, "reasons": reasons}


# --- The live record (cache/smoke/live/checks.json): every check's latest live result ------------------------


def update_live_record(record: Mapping | None, report: Mapping, required: Sequence[str]) -> dict:
    """`record` with this live invocation's checks replacing their earlier entries. Each entry carries what
    run_gate's preflight checks (`check_live_smoke`): status, finish time, commit and its uncommitted code changes
    (`code_dirty`), profile, model overrides and the probe's snapshots."""
    out = dict(record or {})
    checks = dict(out.get("checks") or {})
    vcs = report.get("git") or {}
    models = report.get("models") or {}
    for name, r in (report.get("checks") or {}).items():
        checks[name] = {
            "status": r["status"], "reason": r.get("reason"), "finished_utc": r.get("finished_utc") or report.get("finished_utc"),
            "git_commit": vcs.get("commit"), "git_dirty": vcs.get("dirty"), "code_dirty": vcs.get("code_dirty"), "profile": models.get("profile"),
            "overrides": {k: v for k, v in (models.get("overrides") or {}).items() if v},
            "snapshots": dict(report.get("snapshots") or {}), "report_started_utc": report.get("started_utc"),
        }  # fmt: skip
    out |= {"required": list(required), "checks": checks, "updated_utc": report.get("finished_utc")}
    return out
