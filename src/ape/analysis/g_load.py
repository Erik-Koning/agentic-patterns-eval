"""Study G loaders: Inspect session logs and capability-anchor logs into the tidy tables `g_stats` works on.

    data = load_g_cells({"g.cm.luna-high": [...], "g.topo.sol": [...], "g.cap.sol-high": [...], ...})
    data.items, data.sessions, data.capability

**Cells.** A plan cell (config/run_plan.yaml `study_g`) fixes the block (cm, topo, cap, pilot, tune), the capability
point and N. The point is the agent's model and effort, `<tier>-<effort>` (e.g. "luna-low", "sol-high"): the cell's
profile (config/models.yaml) gives the agent model and its effort unless the cell overrides `effort`. The arm comes
from the log (task args or eval metadata), the session from the sample's `world_id`, so one log may hold any arm.

**Item table** (one row per session-epoch × position, every position 1..N; with plan cell, block, point, tier, effort,
arm, session, epoch, N and knob variant): success (overflowed and never-reached items fail, as the scorer counts
them), answered, overflow and the session's overflow position, reached, kind, dependency flags, generations, view tokens
at the first and decision calls (`view_tokens` = decision, else first), relative position, the taxonomy labels, and
per-item agent / cm / probe tokens where per-call records carry them.

**Session table** (one row per plan cell × arm × session × epoch): the item outcomes by position (`outcomes`) and
their counts (from the item table, `sessions_from_items`, which `g_power` reproduces), dependency success, report
score, the binary session success, overflow position, the reference W crossing, probe F1 per checkpoint (`probes`:
{k: {f1, taken}}; a checkpoint not taken scores 0), taxonomy counts (`tax_*`), and the cost meters with probes
excluded: `tokens` (agent + management input and output tokens), `calls` (agent + management generations),
`cost_usd` (cache-adjusted $: Inspect's priced usage, or the price table) and `wall_clock` (seconds; probe time is
not separable unless per-call records carry it), and each meter per solved item (`cps_<meter>`, NaN when nothing was
solved; per arm and point `g_stats.cost_table` takes the ratio of sums instead). Probe and management tokens are
kept apart (`tokens_probe`, `tokens_cm`, `cost_usd_probe`, `cost_usd_cm`). Also the profile, model, effort and the
sample's knobs.

**Usage and resumes.** Two record formats are read:
- **B7 sessions** (`ape.agent.session` with the ContextPolicy layer): the store's `f8_usage` {by_kind, by_model}
  (or the score metadata's `usage`) is the usage of every call the session consists of, calls restored after a
  mid-session resume included; Inspect's own sample usage covers only the attempt that finished, so `f8_usage` is
  authoritative. Calls are counted from the per-call records (`f8_views`: {item, view_tokens, kind agent | cm,
  model, usage}; probes in `f8_probes`), and an item's tokens by kind come from its record's `usage`. A resumed
  session's `f8_resume` (or the score's `resume`) gives `resumes` and the `unlogged` usage of earlier attempts that
  no Inspect log holds (`tokens_unlogged`, `cost_usd_unlogged`: spend accounting, not an arm's cost; a resumed
  item's discarded partial calls are not an arm's cost either).
- **Earlier sessions:** Inspect's sample usage, with probe and management usage from `role_usage` ("probe", "cm")
  or from per-call records with `kind` and `usage`, else from the probes' own usage records.
Unpriced usage is priced from config/model_costs.yaml (cache reads at their price). Every field may be missing;
nothing raises on a missing score (an errored sample's unscored items fail, as in the gate), a missing store, or
unknown extra fields. Duplicate session-epochs (a retried sample in a second log) keep the last one without an error.

**Capability** (audit §2.1): S1's success on F7-10 and F3-5 from the `g.cap.*` agent logs (`gate_stats.load_results`),
the equal-weight mean of the two cells' task means, with a world-clustered standard error.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..worlds import env_f8
from ..worlds.env_f8 import ITEMS, OVERFLOW, PROBES, VIEWS

# B7's store keys (ape.worlds.env_f8 after B7): usage by kind and model, and resumes.
USAGE, RESUME = getattr(env_f8, "USAGE", "f8_usage"), getattr(env_f8, "RESUME", "f8_resume")

SCORER = "f8_session_score"
CALL_LOG_KEYS = (VIEWS,)  # per-call records (agent and cm calls; probes are in f8_probes): the first key present
CALL_KINDS = ("agent", "cm", "probe")
PHASE_BLOCKS = {"context_management": "cm", "topology": "topo", "capability_anchor": "cap", "micro_pilot": "pilot", "tuning": "tune"}
PREFIX_BLOCKS = {"g.cm.": "cm", "g.topo.": "topo", "g.cap.": "cap", "g.pilot.": "pilot", "g.tune.": "tune"}
CAP_CELLS = ("F7-10", "F3-5")
COST_METERS = ("cost_usd", "tokens", "calls", "wall_clock")

ITEM_COLUMNS = (
    "plan_cell", "block", "point", "tier", "effort", "arm", "session", "epoch", "N", "variant", "position", "rel_position", "item", "case_id", "kind",
    "dependency", "dependency_kinds", "success", "answered", "overflow", "reached", "generations", "view_tokens_first",
    "view_tokens_decision", "view_tokens", "w_crossing_item", "overflow_at", "labels", "tokens_agent", "tokens_cm", "tokens_probe", "error",
)  # fmt: skip
OUTCOME_COLUMNS = ("items_solved", "n_items", "item_success", "outcomes", "overflow", "overflow_at", "dependency_success")


@dataclass
class GData:
    items: pd.DataFrame
    sessions: pd.DataFrame
    capability: pd.DataFrame
    cells: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


# ---------- plan cells ----------


def point_label(model: str, effort: str | None) -> str:
    """"openai/gpt-6-luna", "low" -> "luna-low" (ape.budget's tier label)."""
    short = str(model).split("/", 1)[-1].removeprefix("gpt-6-")
    return f"{short}-{effort}" if effort else short


def cell_info(plan_cell: str, plan=None) -> dict:
    """Block, capability point, N, knobs and design of a Study G plan cell. A cell the plan does not know gets its
    block from the id prefix and no point (pass `points=` to `load_g_cells`)."""
    from ..budget import BudgetError, load_plan

    try:
        cell = (plan or load_plan()).cell(plan_cell)
    except (BudgetError, FileNotFoundError):
        block = next((b for p, b in PREFIX_BLOCKS.items() if plan_cell.startswith(p)), "other")
        return {"plan_cell": plan_cell, "block": block, "point": None, "known": False}
    s = cell.spec
    point = model = effort = None
    try:
        from ..models import load_profile

        agent = load_profile(s.get("profile")).role("agent")
        model, effort = agent.model, s.get("effort") or agent.reasoning_effort
        point = point_label(model, effort)
    except (ValueError, FileNotFoundError):
        pass
    block = PHASE_BLOCKS.get(cell.phase) or next((b for p, b in PREFIX_BLOCKS.items() if plan_cell.startswith(p)), "other")
    return {
        "plan_cell": plan_cell,
        "block": block,
        "point": point,
        "model": model,
        "effort": effort,
        "profile": s.get("profile"),
        "N": s.get("N"),
        "knobs": s.get("knobs") or {},
        "arms": list(s.get("arms") or []),
        "sessions": s.get("sessions"),
        "epochs": s.get("epochs"),
        "enabled": cell.enabled,
        "known": True,
    }


# ---------- usage ----------


def _u(x: Any) -> dict:
    """{input, output, cache_read, cost} from a ModelUsage or a usage dict (missing fields 0 / None)."""
    if x is None:
        return {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cost": None}
    get = (lambda k: x.get(k)) if isinstance(x, Mapping) else (lambda k: getattr(x, k, None))
    cost = get("total_cost")
    cost = get("cost_usd") if cost is None else cost
    return {
        "input": float(get("input_tokens") or 0),
        "output": float(get("output_tokens") or 0),
        "cache_read": float(get("input_tokens_cache_read") or 0),
        "cost": None if cost is None else float(cost),
    }


def _is_usage(x: Any) -> bool:
    return isinstance(x, Mapping) and any(k in x for k in ("input_tokens", "output_tokens", "total_tokens"))


def _price(model: str, u: dict, prices: Mapping | None) -> float | None:
    if u["cost"] is not None:
        return u["cost"]
    p = (prices or {}).get(str(model).split("/", 1)[-1]) or (prices or {}).get(str(model))
    if p is None:
        return None
    uncached = max(u["input"] - u["cache_read"], 0.0)
    return (uncached * p["input"] + u["cache_read"] * p.get("input_cache_read", p["input"]) + u["output"] * p.get("output", 0.0)) / 1e6


# ---------- one sample ----------


def _score(sample) -> Any:
    scores = sample.scores or {}
    if SCORER in scores:
        return scores[SCORER]
    return next((s for s in scores.values() if isinstance(getattr(s, "metadata", None), Mapping) and "items" in s.metadata), None)


def _int_keys(d: Mapping | None) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _item_tokens(record: Mapping, pos: int, per_call: Mapping, kinds: set) -> dict:
    """An item's tokens by call kind: the B7 item record's usage by kind when it has one, else the per-call records;
    NaN for a kind no record carries."""
    if isinstance(record.get("usage"), Mapping):
        by = {k: _u(v) for k, v in record["usage"].items() if isinstance(v, Mapping)}
        return {f"tokens_{k}": (by[k]["input"] + by[k]["output"]) if k in by else 0.0 for k in CALL_KINDS}
    return {f"tokens_{k}": (per_call.get((pos, k), 0.0) if k in kinds else np.nan) for k in CALL_KINDS}


def sample_rows(sample, ctx: Mapping, prices: Mapping | None = None) -> tuple[list[dict], dict]:
    """Item rows and the session row (without the outcome columns `sessions_from_items` adds) of one sample."""
    md = sample.metadata or {}
    store = sample.store or {}
    sc = _score(sample)
    smd = (sc.metadata or {}) if sc is not None else {}
    val = (sc.value if sc is not None and isinstance(sc.value, Mapping) else {}) or {}
    session = md.get("world_id") or str(sample.id)
    store_items = {int(i["position"]): i for i in store.get(ITEMS) or [] if isinstance(i, Mapping) and "position" in i}
    score_items = {int(i["position"]): i for i in smd.get("items") or [] if isinstance(i, Mapping) and "position" in i}
    N = md.get("N") or ctx.get("N") or (max(score_items) if score_items else max(store_items) if store_items else 0)
    N = int(N)
    overflow_at = store.get(OVERFLOW, smd.get("overflow_at"))
    overflow_at = int(overflow_at) if overflow_at is not None else None
    labels = _int_keys((smd.get("taxonomy") or {}).get("per_item"))
    call_key = next((k for k in CALL_LOG_KEYS if store.get(k)), None)
    calls: list[dict] = [c for c in (store.get(call_key) or []) if isinstance(c, Mapping)] if call_key else []
    per_item_tok: dict[tuple[int, str], float] = {}
    for c in calls:
        if "usage" in c and c.get("item") is not None:
            u = _u(c["usage"])
            k = (int(c["item"]), c.get("kind") or "agent")
            per_item_tok[k] = per_item_tok.get(k, 0.0) + u["input"] + u["output"]
    probes_rec = [p for p in store.get(PROBES) or [] if isinstance(p, Mapping)]
    probe_calls = any(c.get("kind") == "probe" for c in calls)
    for p in probes_rec if not probe_calls else []:  # per-call probe records, when present, already carry these
        if p.get("usage") and p.get("k") is not None:
            u = _u(p["usage"])
            per_item_tok[(int(p["k"]), "probe")] = per_item_tok.get((int(p["k"]), "probe"), 0.0) + u["input"] + u["output"]
    has_kind = {k for (_, k) in per_item_tok}
    errored = sample.error is not None
    wc = md.get("w_crossing_item")
    items = []
    for pos in range(1, N + 1):
        si, st = score_items.get(pos), store_items.get(pos, {})
        lost = overflow_at is not None and pos >= overflow_at
        if si is not None:
            success, answered, ovf = bool(si.get("success")), bool(si.get("answered")), bool(si.get("overflow", lost))
        else:
            success, answered, ovf = bool(st.get("success", False)) and not lost, bool(st.get("answered", False)) and not lost, lost
        base = si or st
        vf, vd = st.get("view_tokens_first"), st.get("view_tokens_decision")
        items.append({
            **{k: ctx.get(k) for k in ("plan_cell", "block", "point", "tier", "effort", "arm", "variant")},
            "session": session,
            "epoch": int(sample.epoch),
            "N": N,
            "position": pos,
            "rel_position": pos / N if N else np.nan,
            "item": f"{session}#{pos:03d}",
            "case_id": base.get("case_id"),
            "kind": base.get("kind"),
            "dependency": bool(base.get("dependency", False)),
            "dependency_kinds": ",".join(base.get("dependency_kinds") or []),
            "success": float(success),
            "answered": answered,
            "overflow": ovf,
            "reached": pos in store_items,
            "generations": st.get("generations"),
            "view_tokens_first": vf,
            "view_tokens_decision": vd,
            "view_tokens": vd if vd is not None else vf,
            "w_crossing_item": wc,
            "overflow_at": overflow_at,
            "labels": ",".join(labels.get(pos) or []),
            **_item_tokens(st, pos, per_item_tok, has_kind),
            "error": errored,
        })


    # session-level usage and cost meters
    tok = lambda u: u["input"] + u["output"]  # noqa: E731
    model_usage = {m: _u(u) for m, u in (sample.model_usage or {}).items()}
    agent_model = next((c.get("model") for c in calls if (c.get("kind") or "agent") == "agent" and c.get("model")), None) or next(iter(model_usage), "")
    usage = store.get(USAGE) if isinstance(store.get(USAGE), Mapping) else {}
    by_kind = usage.get("by_kind") or (smd.get("usage") if isinstance(smd.get("usage"), Mapping) else None)
    by_model = usage.get("by_model") or {}
    if by_kind:
        # B7: usage of every call the session consists of, restored calls included, by kind. Inspect's own sample usage
        # covers only the attempt that finished, so this is authoritative.
        kinds = {k: _u(v) for k, v in by_kind.items() if isinstance(v, Mapping)}
        price_model = next(iter(by_model)) if len(by_model) == 1 else agent_model
        t = {k: tok(u) for k, u in kinds.items()}
        usd = {k: _price(price_model, u, prices) for k, u in kinds.items()}
        tokens_total = sum(t.values())
        t_probe, t_cm = t.get("probe", 0.0), t.get("cm", 0.0)
        cost_total = None if any(v is None for v in usd.values()) else sum(usd.values())
        usd_probe, usd_cm = usd.get("probe", 0.0), usd.get("cm", 0.0)
        src = {"usage": "f8_usage" if usage.get("by_kind") else "score", "probe": "f8_usage", "cm": "f8_usage"}
    else:
        # Pre-B7 logs: Inspect's sample usage, with probe and management usage from role usage or the call records.
        role_usage = {r: _u(u) for r, u in (sample.role_usage or {}).items()}
        tokens_total = sum(tok(u) for u in model_usage.values())
        costs = [_price(m, u, prices) for m, u in model_usage.items()]
        cost_total = None if any(c is None for c in costs) else sum(costs)

        def kind_usage(kind: str) -> tuple[float, float | None, str]:
            """(tokens, $, source) of the probe or cm calls."""
            if kind in role_usage:
                u = role_usage[kind]
                return tok(u), _price(agent_model, u, prices), "role_usage"
            recs = [c for c in calls if (c.get("kind") or "agent") == kind and c.get("usage")]
            if kind == "probe" and not recs:
                recs = [{"usage": p["usage"]} for p in probes_rec if p.get("usage")]
            if not recs:
                return 0.0, 0.0, "none"
            t_, usd_ = 0.0, 0.0
            for c in recs:
                u = _u(c["usage"])
                t_ += tok(u)
                cu = c.get("cost_usd", u["cost"])
                pu = cu if cu is not None else _price(agent_model, u, prices)
                usd_ = None if usd_ is None or pu is None else usd_ + pu
            return t_, usd_, "calls" if kind != "probe" or probe_calls else "probes"

        t_probe, usd_probe, src_probe = kind_usage("probe")
        t_cm, usd_cm, src_cm = kind_usage("cm")
        if usd_probe is None and cost_total is not None and tokens_total:
            usd_probe = cost_total * t_probe / tokens_total  # proportional fallback (no price for the model)
        if usd_cm is None and cost_total is not None and tokens_total:
            usd_cm = cost_total * t_cm / tokens_total
        src = {"usage": "model_usage", "probe": src_probe, "cm": src_cm}
    t_agent = max(tokens_total - t_probe - t_cm, 0.0)
    # Resumes (B7): the store's f8_resume or the score's compact `resume`. `unlogged` usage is spend no Inspect log
    # holds (earlier attempts of a retried sample): reported for spend accounting, not an arm's cost.
    resume = store.get(RESUME) if isinstance(store.get(RESUME), Mapping) else (smd.get("resume") if isinstance(smd.get("resume"), Mapping) else None)
    unlogged = (resume or {}).get("unlogged") or {}
    unlogged = unlogged.get("by_model", unlogged) if isinstance(unlogged, Mapping) else {}
    t_unlogged = sum(tok(_u(v)) for v in unlogged.values() if isinstance(v, Mapping))
    usd_unlogged = [_price(m, _u(v), prices) for m, v in unlogged.items() if isinstance(v, Mapping)]
    n_cm = sum(1 for c in calls if c.get("kind") == "cm")
    n_agent = sum(1 for c in calls if (c.get("kind") or "agent") == "agent")
    by_k = _int_keys(smd.get("probes_by_checkpoint"))
    if by_k:
        probes = {k: {"f1": float(p.get("mean_f1") or 0.0), "taken": bool(p.get("taken"))} for k, p in by_k.items()}
    else:
        cps = (store.get("arm") or {}).get("checkpoints") or [p.get("k") for p in probes_rec]
        taken = {int(p["k"]): p for p in probes_rec if p.get("k") is not None}
        probes = {int(k): {"f1": float(((taken.get(int(k)) or {}).get("scores") or {}).get("mean_f1") or 0.0), "taken": int(k) in taken} for k in cps if k is not None and int(k) <= N}
    rep = smd.get("report") or {}
    counts = (smd.get("taxonomy") or {}).get("counts") or {}
    session_row = {
        **{k: ctx.get(k) for k in ("plan_cell", "block", "point", "tier", "effort", "profile", "model", "arm", "variant")},
        "knobs": md.get("knobs") or {},
        "session": session,
        "epoch": int(sample.epoch),
        "N": N,
        "w_crossing_item": wc,
        "report_exact": float(val["report_exact"]) if "report_exact" in val else (float(bool(rep.get("exact"))) if rep else 0.0),
        "report_disp_acc": rep.get("dispositions_accuracy"),
        "report_submitted": rep.get("submitted"),
        "session_success": float(val.get("session_success", 0.0) or 0.0),
        "probe_f1": float(np.mean([p["f1"] for p in probes.values()])) if probes else np.nan,
        "probe_coverage": float(np.mean([p["taken"] for p in probes.values()])) if probes else np.nan,
        "probes": probes,
        **{f"tax_{k}": counts.get(k) for k in ("forgot_constraint", "stale_state", "resurrected_done_item", "dropped_item", "hallucinated_state", "overflow")},
        "tokens_total": tokens_total,
        "tokens_probe": t_probe,
        "tokens_cm": t_cm,
        "tokens_agent": t_agent,
        "tokens": t_agent + t_cm,
        "calls_agent": n_agent,
        "calls_cm": n_cm,
        "calls_probe": len(probes_rec),
        "calls": n_agent + n_cm,
        "cost_usd_total": cost_total if cost_total is not None else np.nan,
        "cost_usd_probe": usd_probe if usd_probe is not None else np.nan,
        "cost_usd_cm": usd_cm if usd_cm is not None else np.nan,
        "cost_usd": (cost_total - (usd_probe or 0.0)) if cost_total is not None else np.nan,
        "usage_sources": src,
        "resumes": int((resume or {}).get("count") or 0),
        "tokens_unlogged": t_unlogged,
        "cost_usd_unlogged": (sum(usd_unlogged) if all(v is not None for v in usd_unlogged) else np.nan) if usd_unlogged else 0.0,
        "wall_clock": float(sample.total_time) if sample.total_time is not None else np.nan,
        "working_time": float(sample.working_time) if sample.working_time is not None else np.nan,
        "error": errored,
        "limit_hit": sample.limit.type if getattr(sample, "limit", None) is not None else None,
        "scored": sc is not None,
    }
    return items, session_row


# ---------- tables ----------


def sessions_from_items(items: pd.DataFrame) -> pd.DataFrame:
    """The outcome columns per session-epoch from item rows: items solved, items, rate, the outcomes by position,
    overflow and its position, dependency success. `g_power`'s simulated session table has exactly these."""
    keys = ["plan_cell", "block", "point", "arm", "session", "epoch", "N"]
    if items is None or not len(items):
        return pd.DataFrame(columns=[*keys, *OUTCOME_COLUMNS])
    rows = []
    for key, g in items.sort_values("position").groupby(keys, dropna=False, sort=False):
        o = g["success"].to_numpy(dtype=float)
        ov = g.loc[g["overflow"].astype(bool), "position"]
        dep = g[g["dependency"].astype(bool)]
        rows.append({
            **dict(zip(keys, key, strict=True)),
            "items_solved": float(o.sum()),
            "n_items": float(len(o)),
            "item_success": float(o.mean()) if len(o) else np.nan,
            "outcomes": o.astype(int).tolist(),
            "overflow": bool(len(ov)),
            "overflow_at": float(ov.min()) if len(ov) else np.nan,
            "dependency_success": float(dep["success"].mean()) if len(dep) else np.nan,
        })
    return pd.DataFrame(rows)


def load_session_logs(log_files: Sequence[str | Path], ctx: Mapping | None = None, prices: Mapping | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Item rows, session rows and problems from F8 session logs; `ctx` gives plan_cell, block, point (the arm is read
    from each log). `prices` defaults to config/model_costs.yaml (used only where Inspect left a cost unset)."""
    from inspect_ai.log import read_eval_log

    from . import cost as costs

    if prices is None:
        try:
            prices = costs.load_prices()
        except FileNotFoundError:
            prices = {}
    ctx = dict(ctx or {})
    items: list[dict] = []
    sess: dict[tuple, tuple[dict, list[dict]]] = {}
    problems: list[str] = []
    for f in log_files:
        try:
            log = read_eval_log(str(f))
        except Exception as e:  # noqa: BLE001 - one unreadable log is a reported problem, not a crash
            problems.append(f"{f}: unreadable ({type(e).__name__}: {e})")
            continue
        args, emd = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or emd.get("arm")
        if not log.samples:
            problems.append(f"{f}: no samples (status {log.status})")
            continue
        variant = args.get("variant") or ""
        model = ctx.get("model") or log.eval.model
        c = {**ctx, "arm": arm, "variant": variant, "model": model, "tier": ctx.get("tier") or point_label(model, None)}
        for s in log.samples:
            it, row = sample_rows(s, c, prices)
            key = (c.get("plan_cell"), arm, row["session"], row["epoch"])
            old = sess.get(key)
            if old is not None:
                problems.append(f"duplicate session-epoch {key}: kept {'the one without an error' if old[0]['error'] and not row['error'] else 'the later one'}")
                if not old[0]["error"] and row["error"]:
                    continue
            sess[key] = (row, it)
    for row, it in sess.values():
        items.extend(it)
    item_df = pd.DataFrame(items, columns=list(ITEM_COLUMNS)) if items else pd.DataFrame(columns=list(ITEM_COLUMNS))
    extra = pd.DataFrame([r for r, _ in sess.values()])
    if not len(extra):
        return item_df, extra, problems
    out = extra.merge(sessions_from_items(item_df).drop(columns=["block", "point", "N"]), on=["plan_cell", "arm", "session", "epoch"], how="left")
    for m in COST_METERS:
        out[f"cps_{m}"] = out[m] / out["items_solved"].where(out["items_solved"] > 0)  # NaN when nothing was solved
    return item_df, out, problems


def capability_table(rows: pd.DataFrame, cells: Sequence[str] = CAP_CELLS) -> pd.DataFrame:
    """Measured capability per point from anchor rows (`point`, `cell`, `world`, `task`, `success`): the equal-weight
    mean over `cells` of each cell's task-mean success, with a world-clustered standard error (cells independent)."""
    cols = ["point", "capability", "se", "n_tasks", "n_worlds", "by_cell", "missing_cells"]
    if rows is None or not len(rows):
        return pd.DataFrame(columns=cols)
    out = []
    for point, g in rows.groupby("point"):
        tm = g.groupby(["cell", "world", "task"])["success"].mean()
        by_cell, var, ok = {}, 0.0, True
        for c in cells:
            if c not in tm.index.get_level_values("cell"):
                continue
            tc = tm.xs(c, level="cell")
            wsum = tc.groupby(level="world").sum()
            wn = tc.groupby(level="world").count()
            m = float(wsum.sum() / wn.sum())
            k = len(wsum)
            v = float(np.sum(((wsum - m * wn) / wn.mean()) ** 2) / (k * (k - 1))) if k > 1 else np.nan
            by_cell[c] = {"success": m, "tasks": int(wn.sum()), "worlds": k, "se": math.sqrt(v) if np.isfinite(v) else None}
            var += v
            ok = ok and np.isfinite(v)
        present = [c for c in cells if c in by_cell]
        missing = [c for c in cells if c not in by_cell]
        cap = float(np.mean([by_cell[c]["success"] for c in present])) if present else np.nan
        se = math.sqrt(var) / len(present) if present and ok else np.nan
        out.append({"point": point, "capability": cap, "se": se, "n_tasks": int(g["task"].nunique()), "n_worlds": int(g["world"].nunique()), "by_cell": by_cell, "missing_cells": missing})
    return pd.DataFrame(out, columns=cols)


def load_capability(logs_by_point: Mapping[str, Sequence[str | Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(capability table, anchor rows) from the g.cap.* agent logs, keyed by point."""
    from .gate_stats import load_results

    frames = []
    for point, files in logs_by_point.items():
        if files:
            df = load_results(list(files), require_cost=False)
            if len(df):
                frames.append(df.assign(point=point))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return capability_table(rows), rows


def load_g_cells(cells: Mapping[str, Sequence[str | Path]], *, plan=None, points: Mapping[str, str] | None = None, prices: Mapping | None = None, capability: Mapping[str, float] | None = None) -> GData:
    """Everything `g_report` needs from {plan_cell: [log paths]}: session cells (g.cm.*, g.topo.*, pilot, tuning)
    become item and session rows, g.cap.* cells measured capability per point. `points` overrides a cell's point;
    `capability` ({point: value}) overrides or fills measured capability (e.g. before the anchor has run)."""
    points = dict(points or {})
    info: dict[str, dict] = {}
    item_frames, sess_frames, problems = [], [], []
    cap_logs: dict[str, list] = {}
    for pc, files in cells.items():
        ci = cell_info(pc, plan)
        if pc in points:
            ci["point"] = points[pc]
        if ci["point"] is None:
            problems.append(f"{pc}: unknown capability point (not in the plan; pass points=)")
        info[pc] = ci | {"logs": len(files or [])}
        if ci["block"] == "cap":
            cap_logs.setdefault(ci["point"], []).extend(files or [])
            continue
        ctx = {"plan_cell": pc, "block": ci["block"], "point": ci["point"], "model": ci.get("model"), "effort": ci.get("effort"), "profile": ci.get("profile")}
        it, ss, pr = load_session_logs(files or [], ctx, prices)
        problems += [f"{pc}: {p}" for p in pr]
        info[pc] |= {"session_epochs": len(ss), "sessions": int(ss["session"].nunique()) if len(ss) else 0, "errors": int(ss["error"].sum()) if len(ss) else 0}
        if ci.get("known") and ci.get("sessions") and len(ss) and ss["session"].nunique() < int(ci["sessions"]):
            problems.append(f"{pc}: {ss['session'].nunique()} of {ci['sessions']} planned sessions present")
        item_frames.append(it)
        sess_frames.append(ss)
    cap, _ = load_capability({p: f for p, f in cap_logs.items() if p is not None})
    if capability:
        over = pd.DataFrame([{"point": p, "capability": float(v), "se": np.nan, "n_tasks": 0, "n_worlds": 0, "by_cell": {}, "missing_cells": [], "source": "override"} for p, v in capability.items()])
        cap = pd.concat([cap[~cap["point"].isin(list(capability))].assign(source="anchor"), over], ignore_index=True)
    elif len(cap):
        cap = cap.assign(source="anchor")
    items = pd.concat([f for f in item_frames if len(f)], ignore_index=True) if any(len(f) for f in item_frames) else pd.DataFrame(columns=list(ITEM_COLUMNS))
    sessions = pd.concat([f for f in sess_frames if len(f)], ignore_index=True) if any(len(f) for f in sess_frames) else pd.DataFrame()
    return GData(items, sessions, cap, info, problems)
