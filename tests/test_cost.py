"""Cost report: ledger pricing, attribution of query-time embeddings, build cost, ratio."""

import pandas as pd
import pytest

from ape.analysis.cost import build_cost, cost_ratio, ledger_frame, per_query_cost, price_entry
from ape.llm.ledger import Ledger, LedgerEntry

PRICES = {
    "gpt-6-luna": {"input": 0.10, "output": 0.50, "input_cache_read": 0.01},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}


def test_price_entry_handles_cached_tokens():
    e = LedgerEntry(role="build", model="gpt-6-luna", kind="chat", input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=100_000)
    assert price_entry(e, PRICES) == pytest.approx((600_000 * 0.10 + 400_000 * 0.01 + 100_000 * 0.50) / 1e6)
    with pytest.raises(KeyError):
        price_entry(LedgerEntry(role="build", model="unknown", kind="chat"), PRICES)


def test_per_query_and_build_costs(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    led.append(LedgerEntry(role="build", model="gpt-6-luna", kind="chat", input_tokens=2_000_000, output_tokens=200_000, context={"world": "w1", "system": "apg-author"}))
    led.append(LedgerEntry(role="embeddings", model="text-embedding-3-small", kind="embed", input_tokens=1_000_000, context={"world": "w1", "system": "chunks"}))
    led.append(LedgerEntry(role="embeddings", model="text-embedding-3-small", kind="embed", input_tokens=500_000, context={"arm": "APG-s", "world": "w1", "sample": "t1"}))
    df = ledger_frame(led, PRICES)
    results = pd.DataFrame({"arm": ["APG-s", "APG-s", "LGR-s", "LGR-s"], "task": ["t1", "t2", "t1", "t2"], "cost_usd": [0.02, 0.02, 0.03, 0.05]})
    pq = per_query_cost(results, df)
    assert pq.loc["APG-s", "embedding_usd"] == pytest.approx(0.01)
    assert pq.loc["APG-s", "usd_per_query"] == pytest.approx((0.04 + 0.01) / 2)
    assert cost_ratio(pq, "APG-s", "LGR-s") == pytest.approx(0.025 / 0.04)
    bc = build_cost(df).set_index("system")["usd"]
    assert bc["apg-author"] == pytest.approx(0.2 + 0.1) and bc["chunks"] == pytest.approx(0.02)
