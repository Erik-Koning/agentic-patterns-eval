"""Gate statistics and the GO / NO-GO decision (GATE_PREREG.md is the normative spec).

Estimand: Δ = success(APG*) − success(LGR*), the mean over gate cells (equal weights)
of the mean paired task-level difference, where a task's success is its mean over
epochs. Worlds are the independent units (KG build quality varies per world).

Inference (D-023): a world-clustered t-interval. Each cell's Δ is a ratio estimator over its worlds (sum of
paired task differences / number of paired tasks) with the linearized cluster variance
Σ_w e_w² / (m (m − 1)), e_w = (y_w − Δ_c n_w) / n̄; the pooled Δ averages the cells, its variance is
Σ_c var_c / K², and the t quantile uses Satterthwaite's df over cells. With ≤ 16 worlds per cell the
pre-registered percentile bootstrap was anti-conservative (type I error 3.3–3.9% at a nominal 2.5%, RELIABILITY_REVIEW
S2); the t-interval holds the nominal level in the same simulations. The bootstrap is still reported.

Delivery modes (§3): each mode's NI hypothesis is tested with Holm across the modes (`holm_modes`), so the
chance of any false GO-type label stays at α; superiority is tested only after GO in every mode (serial
gatekeeping), again with Holm.
"""

import json
import math
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

GATE_CELLS = ("F7-10", "F7-1000", "F3-5", "F3-60")
MARGIN = 0.05  # δ, absolute
CELL_FLOOR = -0.10  # F7-1000 point estimate must exceed this
INVARIANT_TOL = 0.03
COST_FLAG_RATIO = 2.0
MIN_WORLDS_PER_CELL = 4  # paired APG*/LGR* worlds per cell; below this clustered inference is not trusted
S7_P = 0.05  # APG* > S7: world-level sign-flip, one-sided
# One-sided α for the gate (D-023): the whole procedure, including the one pre-registered extension on fresh worlds,
# spends 0.025. Stage 1 (the gate run) spends 0.020 and the extension 0.005, each with Holm across delivery modes;
# the stages use disjoint worlds, so the familywise error is at most their sum (Bonferroni).
ALPHA_TOTAL = 0.025
STAGE_ALPHA = {"stage1": 0.020, "extension": 0.005}
DEFAULT_MAX_TURNS = 12  # Config.max_turns without APE_MAX_TURNS
DEFAULT_BUDGET = 2000  # config.DEFAULT_CONTEXT_BUDGET


# ---------- loading ----------

def load_results(log_files: list[str | Path], require_cost: bool = True) -> pd.DataFrame:
    """One row per (sample, epoch) from Inspect eval logs.

    Cost comes from Inspect's model-cost config. A missing price would otherwise read as $0
    and silently disable the cost flag, so it is an error unless `require_cost=False`
    (offline dry runs only).

    Besides the outcome columns, each row carries what the decision report (FX-7) needs: the turn cap and
    whether it was hit, plus any Inspect sample limit that fired (`limit_hit`, e.g. "cost" from the runner's
    per-sample guard), both counted by PC5's `cap_hit`; the arm's configured context budget from the task's recorded knobs and the
    realized tokens and latency of every compile (PC4, latency), and `pipeline_miss`, where a failed
    sample lost the gold knowledge (§8 NO-GO diagnosis; `pipeline_miss`). The `HEALTH_COLUMNS` count APG classify,
    LightRAG keyword and search_kb failures per sample (`compile_health`).
    """
    from inspect_ai.log import read_eval_log

    rows = []
    for f in log_files:
        log = read_eval_log(str(f))
        args, emd = log.eval.task_args or {}, log.eval.metadata or {}
        arm = args.get("arm") or emd.get("arm")
        knobs = emd.get("knobs") or {}
        max_turns = int(knobs.get("APE_MAX_TURNS", DEFAULT_MAX_TURNS))
        for s in log.samples or []:
            md = s.metadata
            scores = s.scores or {}
            ev = scores.get("delivered_evidence")
            ts = scores.get("task_success")
            ts_md = (ts.metadata or {}) if ts is not None else {}
            usage = {role: u for role, u in (s.role_usage or {}).items()}
            missing = [m for m, u in (s.model_usage or {}).items() if u.total_cost is None]
            if missing and require_cost:
                raise ValueError(f"{f}: no cost for {missing}; run eval() with **ape.models.eval_cost_kwargs() (config/model_costs.yaml)")
            cell = f"{md['family']}-{md['level']}"
            store = s.store or {}
            compiles = store.get("compile_log", [])
            success = 1.0 if ts is not None and ts.value == "C" else 0.0
            turns, answered = ts_md.get("turns_used"), ts_md.get("answered")
            rows.append(
                {
                    "arm": arm,
                    "cell": cell,
                    "world": md["world_id"],
                    "task": s.id,
                    "epoch": s.epoch,
                    # An errored sample (tolerated under fail_on_error, PC5) has no scores. Primary analysis counts
                    # it as a failure (GATE_PREREG §2); the `error` column lets the sensitivity analysis exclude it.
                    "success": success,
                    "evidence_recall": ev.value["evidence_recall"] if ev else np.nan,
                    "evidence_recall_first": ev.value.get("evidence_recall_first", np.nan) if ev else np.nan,
                    "evidence_recall_step_mean": ev.value.get("evidence_recall_step_mean", np.nan) if ev else np.nan,
                    "partial_credit": scores["error_analysis"].value["partial_credit"] if "error_analysis" in scores else np.nan,
                    "error_label": scores["error_analysis"].metadata.get("error") if "error_analysis" in scores else ("harness_error" if s.error else None),
                    "case": md.get("case"),
                    "delivery": args.get("delivery", "push"),
                    "exposure": args.get("exposure", emd.get("exposure", "retrieved")),
                    "split": md.get("split", args.get("split")),
                    "exception_style": md.get("exception_style"),
                    "ctx_tokens": ev.value["ctx_tokens_mean"] if ev else np.nan,
                    "compile_tokens": [int(r["tokens"]) for r in compiles],
                    "compile_ms": [float(r["compile_ms"]) for r in compiles if r.get("compile_ms") is not None],
                    "budget": configured_budget(arm, knobs, cell),
                    "turns_used": turns,
                    "max_turns": max_turns,
                    # An Inspect sample limit fired (cost: the runner's per-sample runaway guard; or token, time,
                    # message): the sample ended early and is scored as it stood.
                    "limit_hit": s.limit.type if s.limit is not None else None,
                    # PC5: the agent ran out of turns without answering, or a sample limit cut it short (GATE_PREREG §7).
                    "cap_hit": s.error is None and ((turns is not None and turns >= max_turns and not answered) or s.limit is not None),
                    "pipeline_miss": None if success or s.error else pipeline_miss(compiles, store.get("step_log", []), md.get("task") or {}),
                    "cost_usd": sum((u.total_cost or 0.0) for u in (s.model_usage or {}).values()),
                    "kg_input_tokens": sum(u.input_tokens for r, u in usage.items() if r == "kg"),
                    "total_time": s.total_time,
                    "working_time": s.working_time,
                    "error": s.error is not None,
                    **compile_health(compiles, store),
                }
            )
    return pd.DataFrame(rows)


# Per-sample harness-health counters the decision report aggregates (`analyze_gate.health`).
HEALTH_COLUMNS = (
    "classify_compiles", "classify_errors", "classify_fallbacks", "classify_repaired", "classify_unknown_ids",
    "keyword_compiles", "keyword_fallbacks", "keyword_errors", "empty_retrievals", "pull_compiles", "pull_truncated",
    "search_errors",
)  # fmt: skip


def compile_health(compiles: list[dict], store: dict) -> dict:
    """Harness-health counts for one sample from its compile log and store.

    - APG classify (`apg/arm.py` compile meta): compiles carrying the counters, and how many had a parse error,
      fell back to the root for lack of a usable match, needed each repair kind, or named unknown node IDs.
    - LightRAG keywords (`lgr/adapter.py` meta `lightrag`): compiles carrying the flags, keyword fallbacks and errors.
      Naive-mode compiles are left out: naive mode extracts no keywords. Empty retrievals (keywords in hand, nothing
      found; any mode) are counted separately, as `empty_retrievals`.
    - search_kb (pull, `agent/kb_react.py`): pull compiles, those whose result was cut to `max_output`, and failed
      searches (`search_errors`, never in the compile log).

    A compile counts only if its meta carries the key, so logs that predate a counter read as 0 compiles ("n/a")."""
    metas = [r.get("meta") or {} for r in compiles]
    apg = [m for m in metas if "classify_fallback" in m]
    lgr = [m["lightrag"] for m in metas if isinstance(m.get("lightrag"), dict) and "keyword_fallback" in m["lightrag"] and m["lightrag"].get("mode") != "naive"]
    pulls = [r for r in compiles if r.get("source") == "pull"]
    repaired: dict[str, int] = {}
    for m in apg:
        for kind in set(m.get("classify_repaired") or []):
            repaired[kind] = repaired.get(kind, 0) + 1
    return {
        "classify_compiles": len(apg),
        "classify_errors": sum(1 for m in apg if m.get("classify_error")),
        "classify_fallbacks": sum(1 for m in apg if m.get("classify_fallback")),
        "classify_repaired": repaired,
        "classify_unknown_ids": sum(int(m.get("classify_unknown_ids") or 0) for m in apg),
        "keyword_compiles": len(lgr),
        "keyword_fallbacks": sum(1 for m in lgr if m.get("keyword_fallback")),
        "keyword_errors": sum(1 for m in lgr if m.get("keyword_error")),
        "empty_retrievals": sum(1 for m in metas if isinstance(m.get("lightrag"), dict) and m["lightrag"].get("empty_retrieval")),
        "pull_compiles": len(pulls),
        "pull_truncated": sum(1 for r in pulls if r.get("truncated")),
        "search_errors": len((store or {}).get("search_errors") or []),
    }


def _s7_targets(path: str) -> dict:
    p = Path(path)
    return _read_targets(str(p), p.stat().st_mtime_ns) if p.is_file() else {}


@cache
def _read_targets(path: str, _mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text())


def _knob(knobs: dict, *names: str, default: int = DEFAULT_BUDGET) -> int:
    for name in names:
        raw = str(knobs.get(name, "")).strip()
        if raw:
            return int(raw)
    return default


def configured_budget(arm: str | None, knobs: dict, cell: str) -> int | None:
    """The context budget an arm was configured with, from the knobs its task recorded (`tasks.gate`), with the
    precedence of `ape.config` (per-arm knob, then APE_CONTEXT_BUDGET, then 2000). S7's budget is its frozen
    per-cell target (`agent.arms.s7_target`). S1 (whole corpus) and S6 (gold facts) have none."""
    arm = arm or ""
    if arm == "S3s":
        return _knob(knobs, "APE_S3S_BUDGET", "APE_CONTEXT_BUDGET")
    if arm.startswith("APG") or arm == "S5o":
        return _knob(knobs, "APE_APG_BUDGET", "APE_CONTEXT_BUDGET")
    if arm.startswith("LGR"):
        return _knob(knobs, "APE_LGR_BUDGET", "APE_CONTEXT_BUDGET")
    if arm == "S7":
        targets = _s7_targets(knobs["APE_S7_TARGETS"]) if knobs.get("APE_S7_TARGETS") else {}
        return int(targets[cell]) if cell in targets else _knob(knobs, "APE_S7_TARGET", "APE_CONTEXT_BUDGET")
    return None


# Where a failed sample lost the gold knowledge, in pipeline order (GATE_PREREG §8 NO-GO diagnosis). APG's
# compiles record the gold nodes' fate (`apg.arm`: in_shortlist, in_matches, in_contributors); other arms only
# whether every gold fact was delivered.
MISS_LABELS = (
    "not_in_graph",  # APG: no node of the graph carries a gold fact (extraction)
    "shortlist_miss",  # APG: no gold node in the embedding shortlist
    "classify_miss",  # APG: shortlisted, but the classify call did not match it
    "compose_miss_truncation",  # APG: matched, but cut by the token budget
    "compose_miss",  # APG: matched, but compose left it out (descendant / slot)
    "evidence_miss",  # other arms: some gold fact never delivered
    "tool_exposure_miss",  # F3: a tool the procedure needs was never exposed
    "delivered_but_failed",  # the knowledge and tools were there; the agent failed
)


def pipeline_miss(compiles: list[dict], steps: list[dict], task: dict) -> str:
    """The first stage at which a failed sample lost what it needed (MISS_LABELS), from its compile and step logs."""
    golds = [r["meta"]["gold"] for r in compiles if isinstance((r.get("meta") or {}).get("gold"), dict)]
    if golds:
        if not any(g.get("nodes") for g in golds):
            return "not_in_graph"
        if not any(g.get("in_shortlist") for g in golds):
            return "shortlist_miss"
        if not any(g.get("in_matches") for g in golds):
            return "classify_miss"
        if not any(g.get("in_contributors") for g in golds):
            return "compose_miss_truncation" if any((r.get("meta") or {}).get("truncated") for r in compiles) else "compose_miss"
    else:
        gold_facts = set(task.get("gold_fact_ids") or [])
        delivered = {f for r in compiles for f in r.get("fact_ids") or []}
        if gold_facts and not gold_facts <= delivered:
            return "evidence_miss"
    needed = {c["tool"] for c in (task.get("gold") or {}).get("calls", []) if isinstance(c, dict) and "tool" in c}
    exposed = {t for st in steps for t in st.get("exposed_tools") or []}
    if needed and steps and not needed <= exposed:
        return "tool_exposure_miss"
    return "delivered_but_failed"


def task_means(df: pd.DataFrame, value: str = "success") -> pd.DataFrame:
    """Rows (cell, world, task); one column per arm; epoch-averaged.

    An arm's runs in one cell must share one delivery mode: push and pull are co-primary and analysed separately
    (GATE_PREREG §3), so averaging them into one task mean is refused. Select a mode, or label arms by mode."""
    if "delivery" in df.columns and len(df):
        modes = df.groupby(["arm", "cell"])["delivery"].nunique()
        if (mixed := modes[modes > 1]).size:
            raise ValueError(f"task_means would mix delivery modes for {sorted(map(tuple, mixed.index))}: select one mode per arm and cell")
    return df.groupby(["cell", "world", "task", "arm"])[value].mean().unstack("arm")


# ---------- estimation ----------

def paired(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> pd.Series:
    """Task-level paired differences a − b in `cells`, on the tasks both arms have (empty when either is absent)."""
    if a not in tm or b not in tm:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["cell", "world", "task"]))
    d = (tm[a] - tm[b]).dropna()
    return d[d.index.get_level_values("cell").isin(list(cells))]


def paired_worlds(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> dict[str, int]:
    """Per cell: worlds with at least one task both arms ran (the independent units of the paired analysis)."""
    d = paired(tm, a, b, cells)
    got = d.index.to_frame(index=False).groupby("cell")["world"].nunique().to_dict() if len(d) else {}
    return {c: int(got.get(c, 0)) for c in cells}


def dropped_tasks(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> dict[str, int]:
    """Per cell: tasks one arm ran and the other did not (left out of every paired statistic)."""
    if a not in tm or b not in tm:
        return {}
    one = tm[a].notna() ^ tm[b].notna()
    sub = one[one.index.get_level_values("cell").isin(list(cells))]
    counts = sub.groupby(level="cell").sum().to_dict() if len(sub) else {}
    return {c: int(counts.get(c, 0)) for c in cells if counts.get(c, 0)}


def cell_deltas(tm: pd.DataFrame, a: str, b: str) -> pd.Series:
    d = (tm[a] - tm[b]).dropna()
    return d.groupby(level="cell").mean()


def pooled_delta(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> float:
    return float(cell_deltas(tm, a, b).reindex(cells).mean())


def cluster_t(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS) -> dict:
    """The world-clustered t statistic of Δ = a − b pooled over `cells` (equal weights), on paired tasks.

    Returns est, se, df and per-cell {est, var, worlds}. Cells need ≥ 2 paired worlds for a variance; otherwise
    `se` is None and `reason` names them. A zero standard error (every world difference equal) gives se 0, df None.
    """
    d = paired(tm, a, b, cells)
    out: dict = {"cells": {}, "est": None, "se": None, "df": None}
    for c in cells:
        dc = d[d.index.get_level_values("cell") == c]
        if not len(dc):
            out["cells"][c] = {"est": None, "var": None, "worlds": 0, "tasks": 0}
            continue
        g = dc.groupby(level="world")
        y, n = g.sum().to_numpy(dtype=float), g.count().to_numpy(dtype=float)
        m, r = len(y), float(y.sum() / n.sum())
        var = float(np.sum(((y - r * n) / n.mean()) ** 2) / (m * (m - 1))) if m >= 2 else None
        out["cells"][c] = {"est": r, "var": var, "worlds": int(m), "tasks": int(n.sum())}
    if any(out["cells"][c]["est"] is None for c in cells):
        return out | {"reason": f"no paired tasks in {[c for c in cells if out['cells'][c]['est'] is None]}"}
    k = len(cells)
    out["est"] = float(np.mean([out["cells"][c]["est"] for c in cells]))
    if short := [c for c in cells if out["cells"][c]["var"] is None]:
        return out | {"reason": f"fewer than 2 paired worlds in {short}"}
    v = {c: out["cells"][c]["var"] / k**2 for c in cells}
    total = sum(v.values())
    out["se"] = math.sqrt(total)
    if total > 0:
        out["df"] = total**2 / sum(v[c] ** 2 / (out["cells"][c]["worlds"] - 1) for c in cells if v[c] > 0)
    return out


def t_bounds(st: dict, level: float) -> tuple[float | None, float | None]:
    """The two-sided (1 − 2·level) interval of a `cluster_t` result: its lower bound is the one-sided level-`level`
    bound. None when no variance could be estimated."""
    if st.get("se") is None:
        return None, None
    if st["se"] == 0 or st["df"] is None:
        return st["est"], st["est"]
    q = float(student_t.ppf(1 - level, st["df"]))
    return st["est"] - q * st["se"], st["est"] + q * st["se"]


def t_p_greater(st: dict, null: float) -> float | None:
    """One-sided p-value for H0: Δ ≤ `null` against Δ > `null` (NI: null = −margin; superiority: null = 0)."""
    if st.get("se") is None:
        return None
    if st["se"] == 0 or st["df"] is None:
        return 0.0 if st["est"] > null else 1.0
    return float(student_t.sf((st["est"] - null) / st["se"], st["df"]))


def t_p_less(st: dict, null: float) -> float | None:
    """One-sided p-value for H0: Δ ≥ `null` against Δ < `null` (preconditions: evidence of a violation)."""
    p = t_p_greater(st, null)
    if p is None:
        return None
    if st["se"] == 0 or st["df"] is None:
        return 0.0 if st["est"] < null else 1.0
    return 1.0 - p


def cluster_bootstrap(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS, reps: int = 10_000, seed: int = 0) -> np.ndarray:
    """Bootstrap distribution of the pooled Δ, resampling worlds within each cell (vectorized)."""
    rng = np.random.default_rng(seed)
    d = (tm[a] - tm[b]).dropna()
    per_cell = []
    for c in cells:
        dc = d.xs(c, level="cell")
        g = dc.groupby(level="world")
        sums, counts = g.sum().to_numpy(), g.count().to_numpy()
        w = rng.multinomial(len(sums), np.full(len(sums), 1 / len(sums)), size=reps)
        per_cell.append((w @ sums) / (w @ counts))
    return np.mean(per_cell, axis=0)


def sign_flip_p(d: np.ndarray, reps: int = 10_000, seed: int = 0) -> float:
    """One-sided p-value for mean(d) > 0 by random sign flips of independent units."""
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(reps, len(d)))
    null = (flips * d).mean(axis=1)
    return float((1 + np.sum(null >= d.mean())) / (reps + 1))


def world_diffs(tm: pd.DataFrame, a: str, b: str) -> np.ndarray:
    """Per-world mean paired difference: the independent units for clustered sign-flip tests."""
    return (tm[a] - tm[b]).dropna().groupby(level=["cell", "world"]).mean().to_numpy()


def pooled_success(tm: pd.DataFrame, arm: str, cells=GATE_CELLS) -> float:
    return float(tm[arm].dropna().groupby(level="cell").mean().reindex(cells).mean())


# ---------- decision ----------

GO_VERDICTS = ("GO", "GO_WITH_COST_FLAG")


@dataclass
class Decision:
    verdict: str  # GO | GO_WITH_COST_FLAG | INCONCLUSIVE | NO_GO | PRECONDITION_FAIL
    delta: float | None
    ci: tuple[float | None, float | None]  # the two-sided (1 − 2·level) interval at the level the verdict used
    reasons: list[str] = field(default_factory=list)
    superiority: bool = False
    details: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items() if k != "details"} | {"details": {k: v for k, v in self.details.items() if k != "_stats"}}, indent=1, default=float)


def secondary_conditions(tm: pd.DataFrame, apg_arm: str, lgr_arm: str, reps: int = 10_000, seed: int = 0) -> dict:
    """§2's secondary conditions, computed whatever the verdict (§9): the F7-1000 point Δ against CELL_FLOOR, and
    APG* > S7 by a world-level one-sided sign-flip test (p < S7_P)."""
    cell = cell_deltas(tm, apg_arm, lgr_arm) if apg_arm in tm and lgr_arm in tm else pd.Series(dtype=float)
    f7 = cell.get("F7-1000")
    s7 = world_diffs(tm, apg_arm, "S7") if "S7" in tm and apg_arm in tm else np.array([])
    p_s7 = sign_flip_p(s7, reps, seed) if len(s7) else None
    return {
        "f7_1000_delta": None if f7 is None else float(f7),
        "f7_1000_ok": f7 is not None and float(f7) > CELL_FLOOR,
        "p_apg_gt_s7": p_s7,
        "s7_worlds": int(len(s7)),
        "s7_ok": p_s7 is not None and p_s7 < S7_P,
    }


def classify(st: dict, level: float, secondary: dict, cost_ratio: float | None, apg_arm: str) -> tuple[str, list[str], tuple]:
    """§8 at one-sided `level`, from a `cluster_t` result: GO when the level-`level` lower bound exceeds −MARGIN
    and the secondary conditions hold; INCONCLUSIVE when the (1 − 2·level) interval spans both −MARGIN and 0;
    NO_GO otherwise. Returns (verdict, reasons, interval)."""
    lo, hi = t_bounds(st, level)
    reasons = []
    if lo <= -MARGIN:
        if hi >= 0:
            return "INCONCLUSIVE", [f"the {100 * (1 - 2 * level):.2f}% CI [{lo:+.3f}, {hi:+.3f}] spans both -{MARGIN} and 0"], (lo, hi)
        reasons.append(f"non-inferiority not shown: lower bound {lo:+.3f} <= -{MARGIN} (one-sided level {level:g})")
    if not secondary["f7_1000_ok"]:
        f7 = secondary["f7_1000_delta"]
        reasons.append(f"F7-1000 cell Δ {'missing' if f7 is None else f'{f7:+.3f}'} <= {CELL_FLOOR}")
    if not secondary["s7_ok"]:
        p = secondary["p_apg_gt_s7"]
        reasons.append(f"{apg_arm} not better than random-node placebo S7 (p={'n/a' if p is None else f'{p:.3f}'})")
    if reasons:
        return "NO_GO", reasons, (lo, hi)
    return ("GO_WITH_COST_FLAG" if cost_ratio is not None and cost_ratio > COST_FLAG_RATIO else "GO"), [], (lo, hi)


def decide(
    tm: pd.DataFrame,
    lgr_arm: str,
    preconditions: dict[str, bool],
    cost_ratio: float | None = None,
    alpha: float = 0.025,
    reps: int = 10_000,
    seed: int = 0,
    apg_arm: str = "APG-s",
) -> Decision:
    """One delivery mode's decision at one-sided level `alpha`, before any adjustment across modes (`holm_modes`).
    `apg_arm` / `lgr_arm` are the dev-selected configurations (APG*, LGR*; GATE_PREREG §5).

    Order (§8): preconditions; then at least MIN_WORLDS_PER_CELL *paired* APG*/LGR* worlds in every cell (counted on
    the tasks both arms ran, never on other arms' worlds); then the world-clustered t-interval. Δ, its intervals,
    the per-cell Δ, the secondary conditions and the dropped unpaired tasks are reported whatever the verdict."""
    failed = [k for k, ok in preconditions.items() if not ok]
    worlds = paired_worlds(tm, apg_arm, lgr_arm)
    st = cluster_t(tm, apg_arm, lgr_arm)
    sec = secondary_conditions(tm, apg_arm, lgr_arm, reps, seed)
    details = {
        "cell_deltas": cell_deltas(tm, apg_arm, lgr_arm).to_dict() if apg_arm in tm and lgr_arm in tm else {},
        "p_apg_gt_s7": sec["p_apg_gt_s7"],
        "secondary": sec,
        "cost_ratio": cost_ratio,
        "paired_worlds": worlds,
        "dropped_tasks": dropped_tasks(tm, apg_arm, lgr_arm),
        "inference": {
            "method": "world-clustered t (Satterthwaite df)",
            "level": alpha,
            "se": st["se"],
            "df": st["df"],
            "p_noninferiority": t_p_greater(st, -MARGIN),
            "p_superiority": t_p_greater(st, 0.0),
            "ci95": list(t_bounds(st, 0.025)),
        },
    }
    if st["se"] is not None:
        boot = cluster_bootstrap(tm, apg_arm, lgr_arm, reps=reps, seed=seed)
        details["bootstrap_ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    ci = t_bounds(st, alpha)
    if failed:
        return Decision("PRECONDITION_FAIL", st["est"], ci, [f"precondition failed: {k}" for k in failed], details=details)
    if short := {c: n for c, n in worlds.items() if n < MIN_WORLDS_PER_CELL}:
        reason = f"too few paired {apg_arm}/{lgr_arm} worlds for clustered inference: {short} (< {MIN_WORLDS_PER_CELL})"
        return Decision("INCONCLUSIVE", st["est"], ci, [reason], details=details)
    verdict, reasons, ci = classify(st, alpha, sec, cost_ratio, apg_arm)
    p_sup = details["inference"]["p_superiority"]
    superior = verdict in GO_VERDICTS and p_sup is not None and p_sup <= alpha
    return Decision(verdict, st["est"], ci, reasons, superiority=superior, details=details | {"_stats": st})


def holm_levels(pvalues: dict[str, float | None], alpha: float) -> tuple[dict[str, float], dict[str, bool]]:
    """Holm across hypotheses at familywise one-sided `alpha`: each hypothesis's level and whether it is rejected.

    Sorted by p-value, the i-th (0-based) of k is tested at alpha / (k − i) while every earlier one was rejected;
    after the first non-rejection the rest keep the level they would have been tested at and are not rejected.
    A None p-value (not testable) sorts last and is never rejected."""
    order = sorted(pvalues, key=lambda m: (pvalues[m] is None, pvalues[m] if pvalues[m] is not None else 1.0))
    k, levels, rejected, going = len(order), {}, {}, True
    for i, m in enumerate(order):
        levels[m] = alpha / (k - i)
        rejected[m] = going and pvalues[m] is not None and pvalues[m] <= levels[m]
        going = rejected[m]
    return levels, rejected


def holm_modes(decisions: dict[str, Decision], alpha: float) -> tuple[dict[str, Decision], dict]:
    """The delivery modes' NI hypotheses tested with Holm at familywise `alpha` (§3, §8; D-023), then superiority
    by serial gatekeeping (tested only after GO in every mode, again with Holm at `alpha`).

    Each testable mode (a `decide` result that reached the t-interval) is re-classified at its Holm level: GO needs
    Holm's rejection plus the secondary conditions; a mode Holm does not reject is INCONCLUSIVE or NO_GO by its
    interval at that level. PRECONDITION_FAIL and few-worlds INCONCLUSIVE modes are kept as they are."""
    testable = {m: d for m, d in decisions.items() if "_stats" in d.details}
    p_ni = {m: d.details["inference"]["p_noninferiority"] for m, d in testable.items()}
    levels, rejected = holm_levels(p_ni, alpha) if testable else ({}, {})
    out: dict[str, Decision] = {}
    for m, d in decisions.items():
        if m not in testable:
            out[m] = d
            continue
        lvl = levels[m]
        verdict, reasons, ci = classify(d.details["_stats"], lvl, d.details["secondary"], d.details["cost_ratio"], "APG*")
        if verdict in GO_VERDICTS and not rejected[m]:  # GO at lvl means p <= lvl, so Holm rejects; kept as a guard
            verdict, reasons = "INCONCLUSIVE", [f"not rejected by Holm at {lvl:g}"]
        info = d.details["inference"] | {"holm_level": lvl, "holm_rejected": bool(rejected[m])}
        out[m] = replace(d, verdict=verdict, reasons=reasons, ci=ci, superiority=False, details={**d.details, "inference": info})
    all_go = bool(out) and all(x.verdict in GO_VERDICTS for x in out.values())
    sup: dict = {"rule": f"serial gatekeeping: tested only after GO in every mode, Holm at {alpha:g}", "tested": all_go}
    if all_go:
        s_levels, s_rejected = holm_levels({m: out[m].details["inference"]["p_superiority"] for m in out}, alpha)
        for m in out:
            out[m] = replace(out[m], superiority=bool(s_rejected[m]))
        sup |= {"levels": s_levels, "rejected": s_rejected}
    return out, {"alpha": alpha, "levels": levels, "rejected": rejected, "superiority": sup}


def violation_test(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS, tol: float = INVARIANT_TOL, alpha: float = 0.025) -> dict:
    """A precondition of the form success(a) ≥ success(b) − tol (PC2, PC3; D-023), failed only on evidence of a
    violation: H0: Δ ≥ −tol is rejected when the world-clustered t test's one-sided p-value for Δ < −tol is at most
    `alpha`, i.e. when the level-`alpha` upper bound of Δ is below −tol. A true tie (or anything ≥ −tol) then fails
    at most `alpha` of the time; the earlier point-estimate rule failed 16–18% of ties (RELIABILITY_REVIEW S5).
    Cells with fewer than 2 paired worlds are left out; with no such cell the point estimate decides, labelled."""
    usable = tuple(c for c in cells if paired_worlds(tm, a, b, (c,))[c] >= 2)
    st = cluster_t(tm, a, b, usable) if usable else {"se": None, "est": None}
    p = t_p_less(st, -tol)
    if p is None:
        present = tuple(c for c in cells if paired_worlds(tm, a, b, (c,))[c] > 0)
        point = pooled_delta(tm, a, b, present) if present else None
        ok = point is not None and point >= -tol
        return {"pass": bool(ok), "delta": point, "upper": None, "p_violation": None, "level": alpha, "cells": list(present), "method": "point estimate (no clustered variance)"}
    return {"pass": bool(p > alpha), "delta": st["est"], "upper": t_bounds(st, alpha)[1], "p_violation": p, "level": alpha, "cells": list(usable), "method": "world-clustered t (Satterthwaite df)"}


def invariants(tm: pd.DataFrame, chain=("S6", "S5o", "APG-s", "S7"), tol: float = INVARIANT_TOL, alpha: float = 0.025) -> dict:
    """PC3: each arm in the chain may fall below its predecessor by at most `tol` (`violation_test`, at
    alpha / (number of pairs), so a chain of ties fails at most `alpha` overall); and S6 > S7 by the world-level
    sign-flip test (p < 0.05)."""
    present = [a for a in chain if a in tm]
    succ = {a: pooled_success(tm, a) for a in present}
    level = alpha / max(1, len(present) - 1)
    tests = {f"{hi}>={lo}": violation_test(tm, hi, lo, tol=tol, alpha=level) for hi, lo in zip(present, present[1:], strict=False)}
    pairs = {k: t["pass"] for k, t in tests.items()}
    ok_s6 = True
    if {"S6", "S7"} <= set(present):
        ok_s6 = sign_flip_p(world_diffs(tm, "S6", "S7")) < 0.05
    return {"success": succ, "pairs": pairs, "tests": tests, "level": level, "s6_gt_s7_significant": ok_s6, "pass": all(pairs.values()) and ok_s6}


# ---------- report helpers (FX-7) ----------


def delta_ci(tm: pd.DataFrame, a: str, b: str, cells=GATE_CELLS, alpha: float = 0.025, reps: int = 10_000, seed: int = 0) -> dict:
    """Δ = a − b pooled over `cells` (equal weights) on the tasks both arms have, with the world-clustered
    t-interval (two-sided 1 − 2α; `lo` is the one-sided level-α bound) and, for reference, the percentile
    bootstrap. Cells without paired data are left out and named; with none, `delta` is None with a `reason`.
    With fewer than 2 paired worlds in a cell no variance exists: `lo`/`hi` are None and `few_worlds` is set."""
    out: dict = {"a": a, "b": b, "cells": list(cells)}
    if a not in tm or b not in tm:
        return out | {"delta": None, "lo": None, "hi": None, "reason": f"no data for {[x for x in (a, b) if x not in tm]}"}
    d = paired(tm, a, b, cells)
    worlds = paired_worlds(tm, a, b, cells)
    present = [c for c in cells if worlds[c] > 0]
    out |= {"missing_cells": [c for c in cells if c not in present], "worlds_per_cell": worlds, "tasks": int(len(d)), "dropped_tasks": dropped_tasks(tm, a, b, cells)}
    if not present:
        return out | {"delta": None, "lo": None, "hi": None, "reason": f"no paired tasks of {a} and {b} in {list(cells)}"}
    st = cluster_t(tm, a, b, present)
    lo, hi = t_bounds(st, alpha)
    out |= {"delta": st["est"], "lo": lo, "hi": hi, "se": st["se"], "df": st["df"], "method": "world-clustered t (Satterthwaite df)"}
    out["few_worlds"] = min(worlds[c] for c in present) < MIN_WORLDS_PER_CELL
    if st["se"] is not None:
        boot = cluster_bootstrap(tm, a, b, cells=present, reps=reps, seed=seed)
        out["bootstrap"] = [float(x) for x in np.quantile(boot, [alpha, 1 - alpha])]
    else:
        out["reason"] = st.get("reason")
    return out


def shared_tasks(tm: pd.DataFrame, arms) -> pd.DataFrame:
    """The task means of `arms` on the tasks every one of them has (so their successes compare like with like)."""
    cols = [a for a in arms if a in tm]
    return tm[cols].dropna() if cols else tm.iloc[0:0]


def epoch_agreement(df: pd.DataFrame) -> float | None:
    """Determinism: the share of (task, arm) pairs with two or more epochs whose epochs all agree on success."""
    multi = df.groupby(["arm", "cell", "world", "task"])["success"].agg(["count", "nunique"])
    multi = multi[multi["count"] > 1]
    return float((multi["nunique"] == 1).mean()) if len(multi) else None


def combine_modes(verdicts: dict[str, str]) -> tuple[str, list[str]]:
    """The gate's label from per-delivery-mode verdicts (GATE_PREREG §3, §8), {"push": ..., "pull": ...}:

    - PRECONDITION_FAIL in any mode → PRECONDITION_FAIL;
    - GO (or GO_WITH_COST_FLAG) in both → GO, or GO_WITH_COST_FLAG if either mode is cost-flagged;
    - GO in one mode only → GO_PUSH_ONLY / GO_PULL_ONLY (the user decides; the other mode's verdict is a reason);
    - otherwise INCONCLUSIVE if any mode is INCONCLUSIVE (the one pre-registered extension applies), else NO_GO.
    """
    if any(v == "PRECONDITION_FAIL" for v in verdicts.values()):
        return "PRECONDITION_FAIL", [f"{m}: {v}" for m, v in verdicts.items() if v == "PRECONDITION_FAIL"]
    go = [m for m, v in verdicts.items() if v in GO_VERDICTS]
    if len(go) == len(verdicts):
        flagged = [m for m, v in verdicts.items() if v == "GO_WITH_COST_FLAG"]
        return ("GO_WITH_COST_FLAG", [f"cost flag in {flagged}"]) if flagged else ("GO", [])
    if go:
        other = {m: v for m, v in verdicts.items() if m not in go}
        label = f"GO_{go[0].upper()}_ONLY" if len(go) == 1 else "GO"
        reasons = [f"{m}: {v}" for m, v in other.items()] + [f"{m}: cost flag" for m in go if verdicts[m] == "GO_WITH_COST_FLAG"]
        return label, reasons
    if any(v == "INCONCLUSIVE" for v in verdicts.values()):
        return "INCONCLUSIVE", [f"{m}: {v}" for m, v in verdicts.items()]
    return "NO_GO", [f"{m}: {v}" for m, v in verdicts.items()]
