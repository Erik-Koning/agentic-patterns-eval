"""Management overhead of the context-management arms, measured offline against config/budget_assumptions.yaml.

    uv run python -m ape.agent.cm_overhead --level 40 --seeds 5

Builds F8 dev sessions in a temporary directory and runs every CM arm at its default knobs, the plan's T_abs and
W, through the real loop with two mocks: `gold_session_agent` (the gen_f8 reference trajectory, perfect, plus
one todo_write per case where the arm offers it) and `mock_session_agent` (naive). Per arm it reports agent and
management (kind cm) calls per session and their ratio, the ratio of their input tokens (the harness meter), and
the overhead the measured management calls imply at the budget's `calls_per_item` agent calls per item, next to
the budget's `overhead`. The mocks make the fewest calls a case needs (about 2.3 per item, against the budget's
6) and add no reasoning or text, so their views grow more slowly than a real agent's: management that fires on
T_abs fires more often for real agents. Offline (mockllm); nothing is logged to the spend registry.
"""

import argparse
import asyncio
import os
import statistics
import tempfile
from pathlib import Path

import yaml

from ..config import ROOT


def measure(level: str = "40", seeds: int = 5, threshold: int | None = None) -> list[dict]:
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ..build import build
    from ..llm.mock_session import gold_session_agent, mock_session_agent
    from ..tasks.study_g import f8_session
    from ..worlds.spec import World
    from .cm_arms import CM_ARMS

    tmp = Path(tempfile.mkdtemp(prefix="cm-overhead-"))
    for k, v in {"APE_WORLDS": tmp / "worlds", "APE_CACHE": tmp / "cache", "APE_EMBEDDINGS": "fake"}.items():
        os.environ[k] = str(v)
    os.environ.pop("APE_SESSION_CHECKPOINTS", None)
    asyncio.run(build("dev", "F8", [level], n_worlds=seeds, n_tasks=0, relational=True, embed=False))
    budget = yaml.safe_load((ROOT / "config" / "budget_assumptions.yaml").read_text())["study_g"]
    rows = []
    for arm in CM_ARMS:
        for mock in ("gold", "naive"):
            per = []
            for path in sorted((tmp / "worlds" / "dev").glob(f"F8-{level}-dev-s*.json")):
                world = World.load(path)
                agent = gold_session_agent(world) if mock == "gold" else mock_session_agent
                kw = {"threshold": threshold} if threshold is not None else {}
                task = f8_session(level=level, split="dev", arm=arm, limit_worlds=1, skip_worlds=seeds_index(path, tmp, level), **kw)
                log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=agent), log_dir=str(tmp / "logs"), display="none")[0]
                s = log.samples[0]
                views = s.store["f8_views"]
                agent_calls = [v for v in views if v["kind"] == "agent"]
                cm_calls = [v for v in views if v["kind"] == "cm"]
                per.append({
                    "agent_calls": len(agent_calls),
                    "cm_calls": len(cm_calls),
                    "agent_tokens": sum(v["view_tokens"] for v in agent_calls),
                    "cm_tokens": sum(v["view_tokens"] for v in cm_calls),
                    "session_success": s.scores["f8_session_score"].value["session_success"],
                })
            n_items = int(level)

            def mean(key: str, per: list[dict] = per) -> float:
                return statistics.mean(p[key] for p in per)

            rows.append({
                "arm": arm,
                "mock": mock,
                "sessions": len(per),
                "agent_calls": mean("agent_calls"),
                "cm_calls": mean("cm_calls"),
                "cm_per_agent_call": mean("cm_calls") / mean("agent_calls"),
                "cm_per_agent_token": mean("cm_tokens") / mean("agent_tokens"),
                "implied_overhead": mean("cm_calls") / (n_items * budget["calls_per_item"]),
                "budget_overhead": budget["arms"].get(arm, {}).get("overhead"),
                "session_success": mean("session_success"),
            })
    return rows


def seeds_index(path: Path, tmp: Path, level: str) -> int:
    return sorted((tmp / "worlds" / "dev").glob(f"F8-{level}-dev-s*.json")).index(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--level", default="40")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threshold", type=int, default=None)
    args = ap.parse_args()
    rows = measure(args.level, args.seeds, args.threshold)
    print(f"{'arm':10s} {'mock':6s} {'agent':>6s} {'cm':>5s} {'cm/agent':>9s} {'cm tok/agent tok':>17s} {'implied@6/item':>15s} {'budget':>7s} {'success':>8s}")
    for r in rows:
        print(f"{r['arm']:10s} {r['mock']:6s} {r['agent_calls']:6.1f} {r['cm_calls']:5.1f} {r['cm_per_agent_call']:9.3f} {r['cm_per_agent_token']:17.3f} {r['implied_overhead']:15.3f} {r['budget_overhead']!s:>7s} {r['session_success']:8.2f}")


if __name__ == "__main__":
    main()
