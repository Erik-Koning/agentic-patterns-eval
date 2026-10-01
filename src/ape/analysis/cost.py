"""Cost report: per-query cost per arm, build cost per world, and the gate's cost ratio.

Two sources:
- Inspect eval logs (`load_results`): agent and kg-role calls made during samples,
  priced by Inspect from `config/model_costs.yaml`.
- The JSONL ledger: calls made outside samples, priced here from the same table.
  These are build calls (APG authoring, LightRAG extraction) and embeddings, including
  query-time embeddings attributed to an arm and sample via `EMBED_CONTEXT`.

    uv run python -m ape.analysis.cost logs/*.eval
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ..config import ROOT, Config
from ..llm.ledger import Ledger, LedgerEntry

PRICES_PATH = ROOT / "config" / "model_costs.yaml"


def load_prices(path: Path = PRICES_PATH) -> dict[str, dict[str, float]]:
    """{model: {input, output, input_cache_read, input_cache_write}} in USD per 1M tokens, keyed without provider prefix."""
    raw = yaml.safe_load(path.read_text()) or {}
    return {k.split("/", 1)[-1]: v for k, v in raw.items()}


def price_entry(e: LedgerEntry, prices: dict[str, dict[str, float]]) -> float:
    p = prices.get(e.model)
    if p is None:
        raise KeyError(f"no price for {e.model!r} in {PRICES_PATH}")
    uncached = max(e.input_tokens - e.cached_input_tokens, 0)
    return (uncached * p["input"] + e.cached_input_tokens * p.get("input_cache_read", p["input"]) + e.output_tokens * p.get("output", 0.0)) / 1e6


def ledger_frame(ledger: Ledger, prices: dict) -> pd.DataFrame:
    rows = [{**e.context, "role": e.role, "model": e.model, "kind": e.kind, "usd": price_entry(e, prices)} for e in ledger.read()]
    return pd.DataFrame(rows)


def per_query_cost(results: pd.DataFrame, ledger_df: pd.DataFrame) -> pd.DataFrame:
    """Mean $ per sample-epoch by arm: Inspect-metered calls plus attributed query-time embeddings."""
    by_arm = results.groupby("arm").agg(samples=("task", "size"), inspect_usd=("cost_usd", "sum"))
    if not ledger_df.empty and {"arm", "sample"} <= set(ledger_df.columns):
        emb = ledger_df[(ledger_df["kind"] == "embed") & ledger_df["arm"].notna()].groupby("arm")["usd"].sum()
        by_arm["embedding_usd"] = emb.reindex(by_arm.index).fillna(0.0)
    else:
        by_arm["embedding_usd"] = 0.0
    by_arm["usd_per_query"] = (by_arm["inspect_usd"] + by_arm["embedding_usd"]) / by_arm["samples"]
    return by_arm


def build_cost(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """$ per world and system for one-time builds (authoring, extraction, chunk and graph embeddings)."""
    if ledger_df.empty or "world" not in ledger_df.columns:
        return pd.DataFrame(columns=["world", "system", "usd"])
    build = ledger_df[ledger_df.get("arm").isna()] if "arm" in ledger_df.columns else ledger_df
    return build.groupby(["world", "system"], dropna=False)["usd"].sum().reset_index()


def cost_ratio(per_query: pd.DataFrame, apg_arm: str, lgr_arm: str) -> float:
    return float(per_query.loc[apg_arm, "usd_per_query"] / per_query.loc[lgr_arm, "usd_per_query"])


def main() -> None:
    from .gate_stats import load_results

    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--apg", default="APG-s")
    ap.add_argument("--lgr", default="LGR-s")
    args = ap.parse_args()
    prices = load_prices()
    led = ledger_frame(Ledger(Config().ledger_path), prices)
    pq = per_query_cost(load_results(args.logs), led)
    print(pq.to_string())
    print(build_cost(led).to_string(index=False))
    if {args.apg, args.lgr} <= set(pq.index):
        print(f"cost ratio {args.apg}/{args.lgr}: {cost_ratio(pq, args.apg, args.lgr):.2f}")


if __name__ == "__main__":
    main()
