"""Analysis of a context-length sweep run (CONTEXT_SWEEP.md): `analyze(run_dir) -> dict`.

Reads every replicate's eval logs (`<run>/rep-<n>/<tier>/`, the runner index's latest tasks; sample summaries only,
so the 960K-token transcripts are never loaded) and writes, under `<run>/report/`:

- `results.csv` / `results.json`: one row per sample (replicate, tier, model, kind, task seed, target and metered
  context tokens, the provider's input tokens, success, answered, failure label, generations, output and reasoning
  tokens, cost at Inspect's flat price and at the long-context rate, error, limit);
- `report.md`:
  - **Metrics**: per tier, the largest size at which every sample was right (`held`) and the first size with a miss;
    the success rate by size, by kind, and short (the smaller half of the sizes) against long (the larger half), each
    with its 95% Wilson interval; median output tokens and mean generations by size;
  - per tier, a grid of marks (one per task and replicate, in seed then replicate order) by size and kind;
  - failure labels, the metered against the provider's input tokens, and spend.

Descriptive: the intervals are per cell over its samples (tasks x replicates), and replicates of one task are not
independent evidence about other tasks. Spend comes from the usage ledgers (every call, errored attempts included)
priced with `ape.budget`'s long-context rates; Inspect's own cost uses the flat table and is shown beside it.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from inspect_ai.log import read_eval_log_sample_summaries

from . import usage_ledger
from .budget import Prices, load_assumptions, long_context_rate
from .runner import INDEX_NAME, latest_tasks, read_index
from .worlds import gen_sweep

FIELDS = (
    "replicate", "tier", "model", "kind", "seed", "target_tokens", "context_tokens", "provider_input_tokens", "success", "answered",
    "failure", "generations", "output_tokens", "reasoning_tokens", "cost_flat_usd", "cost_usd", "error", "limit", "sample_id",
)  # fmt: skip


def call_cost(model: str, input_tokens: float, output_tokens: float, cache_read: float = 0.0, assumptions: dict | None = None, prices: Prices | None = None) -> float:
    """USD for one call: the long-context rate (every token) when its input exceeds the model's threshold, else the
    price table's (cached input at its cached price)."""
    A = assumptions or load_assumptions()
    if rate := long_context_rate(A, model, input_tokens):
        return (input_tokens * float(rate["input"]) + output_tokens * float(rate["output"])) / 1e6
    p = (prices or Prices.load()).entry(model)
    return ((input_tokens - cache_read) * p["input"] + cache_read * p["input_cache_read"] + output_tokens * p["output"]) / 1e6


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson score interval of k successes in n (NaN for n = 0)."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(rows: list[dict]) -> str:
    """"k/n (p%, lo–hi)" for the rows' successes, or "–" for none."""
    n = len(rows)
    if not n:
        return "–"
    k = sum(r["success"] for r in rows)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}–{100 * hi:.0f})"


def replicate_dirs(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for d in sorted(Path(run_dir).glob("rep-*")):
        try:
            out.append((int(d.name.split("-", 1)[1]), d))
        except ValueError:
            continue
    return sorted(out)


def ledger_spend(log_dir: Path, assumptions: dict | None = None, prices: Prices | None = None) -> dict:
    """Every recorded call of one eval set (its usage ledger): calls, Inspect's flat cost and the long-context cost."""
    A, P = assumptions or load_assumptions(), prices or Prices.load()
    flat = tiered = 0.0
    entries = usage_ledger.read_entries([Path(log_dir) / usage_ledger.LEDGER_NAME])
    for e in entries:
        flat += float(e.get("cost_usd") or 0.0)
        if not str(e.get("model", "")).startswith("mockllm/"):
            tiered += call_cost(str(e["model"]), float(e.get("input_tokens") or 0), float(e.get("output_tokens") or 0), float(e.get("input_tokens_cache_read") or 0), A, P)
    return {"calls": len(entries), "flat_usd": flat, "usd": tiered}


def _seed(sample_id: str) -> int | None:
    try:
        return int(sample_id.rsplit("-s", 1)[1])
    except (IndexError, ValueError):
        return None


def _rows(rep: int, tier: str, log_dir: Path, A: dict, P: Prices) -> list[dict]:
    index = read_index(log_dir)
    rows = []
    for entry in latest_tasks(index).values():
        log = entry.get("log")
        if not log or not Path(log).exists():
            continue
        model = entry.get("model") or ""
        for s in read_eval_log_sample_summaries(log):
            sc = (s.scores or {}).get("sweep_score")
            md = dict(sc.metadata or {}) if sc is not None else {}
            value = sc.value if sc is not None and isinstance(sc.value, dict) else {}
            calls = md.get("calls") or []
            usage = [c.get("usage") or {} for c in calls]
            first = usage[0] if usage else {}
            flat = sum(float(u.get("total_cost") or 0.0) for u in usage)
            cost = sum(call_cost(model, float(u.get("input_tokens") or 0), float(u.get("output_tokens") or 0), float(u.get("input_tokens_cache_read") or 0), A, P) for u in usage) if not model.startswith("mockllm/") else 0.0
            wid = str(s.id)
            smd = s.metadata or {}
            rows.append({
                "replicate": rep, "tier": tier, "model": model, "kind": md.get("kind") or smd.get("kind") or wid.split("-")[1],
                "seed": smd.get("seed") or _seed(wid),
                "target_tokens": md.get("target_tokens") or smd.get("target_tokens"),
                "context_tokens": md.get("context_tokens") or smd.get("context_tokens"),
                "provider_input_tokens": first.get("input_tokens"),
                "success": int(value.get("success", 0) or 0), "answered": int(value.get("answered", 0) or 0),
                "failure": md.get("failure") if sc is not None else "error",
                "generations": md.get("generations", 0),
                "output_tokens": sum(int(u.get("output_tokens") or 0) for u in usage),
                "reasoning_tokens": sum(int(u.get("reasoning_tokens") or 0) for u in usage),
                "cost_flat_usd": round(flat, 6), "cost_usd": round(cost, 6),
                "error": (s.error or "")[:200] if getattr(s, "error", None) else "",
                "limit": getattr(s, "limit", None) or "",
                "sample_id": wid,
            })  # fmt: skip
    return rows


def collect(run_dir: Path) -> tuple[list[dict], dict]:
    """(rows, spend) over every replicate and tier of the run."""
    A, P = load_assumptions(), Prices.load()
    rows, spend = [], {}
    for rep, rdir in replicate_dirs(run_dir):
        for tdir in sorted(p for p in rdir.iterdir() if p.is_dir() and (p / INDEX_NAME).exists()):
            rows += _rows(rep, tdir.name, tdir, A, P)
            spend[f"rep-{rep}/{tdir.name}"] = ledger_spend(tdir, A, P)
    order = {k: i for i, k in enumerate(gen_sweep.KINDS)}
    rows.sort(key=lambda r: (r["tier"], r["replicate"], order.get(r["kind"], 9), r["seed"] or 0, int(r["target_tokens"] or 0)))
    return rows, spend


def _k(n: int) -> str:
    return f"{n // 1000}K"


def held_size(by: dict, tier: str, kinds: list[str], sizes: list[int]) -> tuple[int | None, int | None]:
    """(the largest size up to which every sample of the tier was right, the first size with a miss); a size with no
    rows ends the run of sizes."""
    held = None
    for s in sizes:
        rs = [r for k in kinds for r in by.get((tier, k, s), [])]
        if not rs or not all(r["success"] for r in rs):
            return held, s
        held = s
    return held, None


def held_line(by: dict, tier: str, kinds: list[str], sizes: list[int]) -> str:
    held, miss = held_size(by, tier, kinds, sizes)
    if held is None:
        return f"A task missed already at {_k(miss)}." if miss else "No results."
    return f"Every task right at every size through {_k(held)}." if miss is None else f"Every task right through {_k(held)}; the first miss at {_k(miss)}."


def _mark(r: dict) -> str:
    if r["failure"] == "error" or r["error"]:
        return "!"
    return "✓" if r["success"] else ("–" if not r["answered"] else "✗")


def metrics_md(rows: list[dict], tiers: list[str], kinds: list[str], sizes: list[int]) -> list[str]:
    by_size = defaultdict(list)
    by_kind = defaultdict(list)
    by = defaultdict(list)
    for r in rows:
        s = int(r["target_tokens"] or 0)
        by_size[(r["tier"], s)].append(r)
        by_kind[(r["tier"], r["kind"])].append(r)
        by[(r["tier"], r["kind"], s)].append(r)
    head = "| | " + " | ".join(tiers) + " |"
    rule = "|---|" + "---|" * len(tiers)
    short, long_ = sizes[: (len(sizes) + 1) // 2], sizes[(len(sizes) + 1) // 2 :]
    lines = ["## Metrics", "", "Rates are right / samples (%, 95% Wilson interval), pooling tasks and replicates.", ""]
    lines += ["**Usable context** (every sample right at every size up to it):", "", head, rule]
    cells = []
    for t in tiers:
        held, miss = held_size(by, t, kinds, sizes)
        cells.append(f"{_k(held) if held else 'none'}{'' if miss is None else f' (first miss {_k(miss)})'}")
    lines += ["| held | " + " | ".join(cells) + " |", ""]
    lines += ["**Success by size** (all kinds):", "", head.replace("| |", "| size |"), rule]
    lines += [f"| {_k(s)} | " + " | ".join(rate(by_size[(t, s)]) for t in tiers) + " |" for s in sizes]
    if short and long_:
        lines += [f"| short ({_k(short[0])}–{_k(short[-1])}) | " + " | ".join(rate([r for s in short for r in by_size[(t, s)]]) for t in tiers) + " |"]
        lines += [f"| long ({_k(long_[0])}–{_k(long_[-1])}) | " + " | ".join(rate([r for s in long_ for r in by_size[(t, s)]]) for t in tiers) + " |"]
    lines += ["", "**Success by kind** (all sizes):", "", head.replace("| |", "| kind |"), rule]
    lines += [f"| {k} | " + " | ".join(rate(by_kind[(t, k)]) for t in tiers) + " |" for k in kinds]
    lines += ["", "**Effort by size** (median output tokens per sample, reasoning included · mean generations):", "", head.replace("| |", "| size |"), rule]

    def effort(rs: list[dict]) -> str:
        return f"{median(r['output_tokens'] for r in rs):,.0f} · {mean(r['generations'] for r in rs):.1f}" if rs else "–"

    lines += [f"| {_k(s)} | " + " | ".join(effort(by_size[(t, s)]) for t in tiers) + " |" for s in sizes]
    return [*lines, ""]


def report_md(rows: list[dict], spend: dict, run: dict) -> str:
    design = run.get("design") or {}
    order = {t: i for i, t in enumerate(design.get("tiers") or [])}
    tiers = sorted(dict.fromkeys(r["tier"] for r in rows), key=lambda t: (order.get(t, 99), t))
    kinds = [k for k in gen_sweep.KINDS if any(r["kind"] == k for r in rows)]
    sizes = sorted({int(r["target_tokens"]) for r in rows if r["target_tokens"]})
    reps = sorted({r["replicate"] for r in rows})
    tasks = len({(r["kind"], r["seed"]) for r in rows})
    lines = [
        f"# Context-length sweep: {run.get('run_id', '?')}",
        "",
        f"{'Offline dry run (mock agent): plumbing only, not results.' if run.get('offline') else 'Live run.'} "
        f"Tiers: {', '.join(tiers) or 'none'}. Kinds: {', '.join(kinds)}. Sizes: {', '.join(_k(s) for s in sizes)}. "
        f"Tasks: {tasks} ({design.get('tasks_per_kind', 1)} per kind). Replicates: {', '.join(map(str, reps)) or 'none'}. "
        f"World seeds from {design.get('seed_base')}; tool files ~{design.get('output_tokens')} tokens.",
        "",
    ]
    if not rows:
        return "\n".join([*lines, "No samples logged yet."])
    lines += metrics_md(rows, tiers, kinds, sizes)
    by = defaultdict(list)
    for r in rows:
        by[(r["tier"], r["kind"], int(r["target_tokens"] or 0))].append(r)
    lines += ["## Per sample", "", "One mark per task and replicate (seed order, then replicate): ✓ right, ✗ wrong, – no decision, ! error.", ""]
    for tier in tiers:
        lines += [f"### {tier}", "", "| size | " + " | ".join(kinds) + " | right |", "|---|" + "---|" * (len(kinds) + 1)]
        for size in sizes:
            cells, right, total = [], 0, 0
            for kind in kinds:
                rs = sorted(by.get((tier, kind, size), []), key=lambda r: (r["seed"] or 0, r["replicate"]))
                cells.append("".join(_mark(r) for r in rs) or " ")
                right += sum(r["success"] for r in rs)
                total += len(rs)
            lines.append(f"| {_k(size)} | " + " | ".join(cells) + f" | {right}/{total} |")
        lines += ["", held_line(by, tier, kinds, sizes), ""]
    fails = defaultdict(int)
    for r in rows:
        if not r["success"]:
            fails[(r["tier"], r["kind"], r["failure"])] += 1
    if fails:
        lines += ["## Failure labels", "", "| tier | kind | label | n |", "|---|---|---|---|"]
        lines += [f"| {t} | {k} | {f} | {n} |" for (t, k, f), n in sorted(fails.items())]
        lines += ["", "`ignored_memo` (memo): the decision without the memo announced at the start; `copied_original` (followup): case 1's "
                  "earlier decision repeated, i.e. the original found but the probe's memo not applied; `wrong`: anything else; "
                  "`unanswered`: no decision within 2 generations.", ""]  # fmt: skip
    ratios = [r["provider_input_tokens"] / r["context_tokens"] for r in rows if r["provider_input_tokens"] and r["context_tokens"]]
    if ratios:
        lines += [f"Provider input tokens / metered context: median {median(ratios):.3f} (range {min(ratios):.3f}–{max(ratios):.3f}).", ""]
    if spend:
        lines += ["## Spend", "", "| eval set | calls | at Inspect's flat price | with long-context rates |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['calls']} | ${v['flat_usd']:.2f} | ${v['usd']:.2f} |" for k, v in spend.items()]
        lines += [f"| total | {sum(v['calls'] for v in spend.values())} | ${sum(v['flat_usd'] for v in spend.values()):.2f} | ${sum(v['usd'] for v in spend.values()):.2f} |", ""]
    return "\n".join(lines)


def analyze(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    rows, spend = collect(run_dir)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in FIELDS} for r in rows)
    (out / "results.json").write_text(json.dumps({"run": run, "rows": rows, "spend": spend}, indent=1, default=str))
    (out / "report.md").write_text(report_md(rows, spend, run))
    return {"rows": len(rows), "spend_usd": sum(v["usd"] for v in spend.values()), "report": str(out / "report.md")}
