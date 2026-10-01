"""Live smoke tests and the Luna build-quality check (readiness E2-E5, L2-L5, H4; DECISIONS D-017).

    uv run python readiness/smoke.py --agent openai/gpt-6-luna --kg openai/gpt-6-luna --build gpt-6-luna \
        --embed text-embedding-3-small --max-usd 2
    uv run python readiness/smoke.py --dry      # offline wiring check (mock models, fake embeddings)

Everything runs in an isolated scratch area (`cache/smoke/`) on tiny worlds. The script stops
before the next step if ledger-plus-Inspect spend exceeds `--max-usd`. It writes
`cache/smoke/report.json`.
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cache" / "smoke"
ID_PATTERN = re.compile(r"\b(?:P-\d+|X-\d+|SOP-[\w-]+)")
SPEC_THRESHOLD = 0.95  # D-017


def _isolate(dry: bool) -> None:
    os.environ["APE_WORLDS"] = str(OUT / "worlds")
    os.environ["APE_INDICES"] = str(OUT / "indices")
    os.environ["APE_CACHE"] = str(OUT / "cache")
    if dry:
        os.environ["APE_EMBEDDINGS"] = "fake"


def _spend(logs) -> float:
    from ape.analysis.cost import ledger_frame, load_prices
    from ape.config import Config
    from ape.llm.ledger import Ledger

    inspect_usd = sum((u.total_cost or 0.0) for log in logs for s in (log.samples or []) for u in (s.model_usage or {}).values())
    led = Ledger(Config().ledger_path)
    try:
        ledger_usd = float(ledger_frame(led, load_prices())["usd"].sum()) if led.read() else 0.0
    except KeyError:
        ledger_usd = float("nan")
    return inspect_usd + ledger_usd


def _guard(report: dict, logs: list, max_usd: float) -> None:
    report["spend_usd"] = _spend(logs)
    if report["spend_usd"] > max_usd:
        raise SystemExit(f"spend {report['spend_usd']:.4f} exceeds --max-usd {max_usd}; stopping")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="openai/gpt-6-luna")
    ap.add_argument("--kg", default="openai/gpt-6-luna")
    ap.add_argument("--build", default="gpt-6-luna")
    ap.add_argument("--embed", default="text-embedding-3-small")
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    _isolate(args.dry)
    os.environ.setdefault("APE_EMBEDDING_MODEL", args.embed)
    os.environ["APE_BUILD_MODEL"] = args.build

    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model

    from ape.apg.author import author_world, build_llm_author
    from ape.build import build
    from ape.config import Config
    from ape.lgr.build import build_index
    from ape.llm.ledger import Ledger
    from ape.tasks.gate import gate
    from ape.worlds.render import chunk_world
    from ape.worlds.spec import World

    report: dict = {"dry": args.dry, "models": vars(args)}
    logs: list = []
    cfg = Config()
    if args.dry:
        from ape.llm.mock_agent import mock_agent, mock_kg

        agent = get_model("mockllm/model", custom_outputs=mock_agent)
        roles = {"kg": get_model("mockllm/model", custom_outputs=mock_kg, memoize=False)}
    else:
        agent, roles = args.agent, {"kg": args.kg}

    # Tiny worlds: F7-100 descriptive (relational) and F3-5.
    f7 = asyncio.run(build("dev", "F7", ["100"], n_worlds=1, n_tasks=2, relational=True, embed=True))[0]
    f3 = asyncio.run(build("dev", "F3", ["5"], n_worlds=1, n_tasks=2, relational=True, embed=True))[0]
    world = World.load(cfg.world_path(f7))
    chunk_ids = {c.id for c in chunk_world(world)}

    # L2: LightRAG build (extraction live; oracle custom KG when dry) and source-to-chunk mapping.
    kind = "oracle" if args.dry else "extract"
    manifest = asyncio.run(build_index(world, kind, cfg))
    ents = json.loads((cfg.indices_dir / "lightrag" / f"{world.id}.{kind}").joinpath(*_entity_file(cfg, world, kind)).read_text())
    names = _entity_names(ents)
    report["L2"] = {
        "entities": len(names),
        "policy_ids_as_entities": _coverage(world, names),
        "index_hash": manifest["index_hash"][:12],
        "chunks": len(chunk_ids),
    }
    _guard(report, logs, args.max_usd)

    # D-017: authoring quality with the build model (perfect-author fake when dry).
    if args.dry:
        from ape.llm.fake import perfect_author as author
    else:
        author = build_llm_author(args.build, Ledger(cfg.ledger_path), world.id)
    report["D017_authoring"] = asyncio.run(author_world(world, cfg, author))
    report["D017_pass"] = report["D017_authoring"]["id_coverage"] >= SPEC_THRESHOLD and report["L2"]["policy_ids_as_entities"] >= SPEC_THRESHOLD
    _guard(report, logs, args.max_usd)

    # H4 (changing tool sets) + L4 (one keyword call per compile) + L5 (realized context) on live models.
    runs = {"H4_S3s_F3": dict(family="F3", level="5", arm="S3s"), "L4_L5_LGR_F7": dict(family="F7", level="100", arm="LGR-s" if not args.dry else "LGRo-s"), "APG_F7": dict(family="F7", level="100", arm="APG-s")}
    for name, kw in runs.items():
        log = inspect_eval(gate(split="dev", **kw), model=agent, model_roles=roles, log_dir=str(OUT / "logs"), display="none", message_limit=40)[0]
        logs.append(log)
        recs = [r for s in log.samples for r in s.store.get("compile_log", [])]
        kg_calls = sum(1 for s in log.samples for e in s.events if e.event == "model" and getattr(e, "role", None) == "kg")
        report[name] = {
            "status": log.status,
            "error": str(log.error)[:300] if log.error else None,
            "success": [s.scores["task_success"].value for s in log.samples] if log.status == "success" else None,
            "compiles": len(recs),
            "kg_calls": kg_calls,
            "ctx_tokens": [r["tokens"] for r in recs],
            "exposed_tools_per_step": [st["exposed_tools"] for s in log.samples for st in s.store.get("step_log", [])][:6],
        }
        _guard(report, logs, args.max_usd)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "models"}, indent=1, default=str)[:4000])


def _entity_file(cfg, world, kind) -> tuple[str, ...]:
    """LightRAG keeps entities in a per-workspace JSON KV file; find it."""
    from ape.lgr.common import workspace_name

    base = cfg.indices_dir / "lightrag" / f"{world.id}.{kind}"
    hits = sorted(base.rglob("*full_entities*.json")) or sorted(base.rglob("*vdb_entities*.json"))
    if not hits:
        raise FileNotFoundError(f"no entity store under {base} (workspace {workspace_name(world.id)})")
    return hits[0].relative_to(base).parts


def _entity_names(store: dict) -> set[str]:
    text = json.dumps(store)
    return set(ID_PATTERN.findall(text))


def _coverage(world, names: set[str]) -> float:
    ids = [p.id for p in world.policies] + [x.id for x in world.exceptions] + [p.id for p in world.procedures]
    return sum(i in names for i in ids) / len(ids) if ids else 1.0


if __name__ == "__main__":
    main()
