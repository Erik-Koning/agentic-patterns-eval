"""Live smoke tests and the Luna build-quality check (readiness E2-E5, L2-L5, H4; DECISIONS D-017).

    uv run python readiness/smoke.py --max-usd 2            # models and efforts from config/models.yaml (gate)
    uv run python readiness/smoke.py --profile gate --agent openai/gpt-6-sol   # a flag swaps a role's model, keeps its effort
    uv run python readiness/smoke.py --dry      # offline wiring check (mock models, fake embeddings)

Everything runs in an isolated scratch area (`cache/smoke/`) on tiny worlds. A preflight
(`ape.models.require_preflight`) first checks that every model is priced and, for live runs, that
OPENAI_API_KEY is set. The script stops before the next step if ledger-plus-Inspect spend exceeds
`--max-usd`; a live call without a price stops it too, rather than counting as $0. The three evals
run through `ape.runner.run_evals` (FX-3 retries and error budget), each in a fresh log dir under
`cache/smoke/logs/<timestamp>/`, so a smoke run never reuses an earlier run's logs. It writes
`cache/smoke/report.json`.
"""

import argparse
import asyncio
import json
import os
import re
import time
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


def _inspect_spend(logs, dry: bool) -> float:
    """Inspect-metered $ (agent and role calls), priced via `eval_cost_kwargs()`. An unpriced call is an
    error on a live run, since counting it as $0 would blind the --max-usd cap. Dry runs use mockllm,
    which has no price: those count as $0, with a note."""
    unpriced = sorted({m for log in logs for s in (log.samples or []) for m, u in (s.model_usage or {}).items() if u.total_cost is None})
    if unpriced and not dry:
        raise SystemExit(f"no Inspect cost for {unpriced}; refusing to count it as $0 (is the model in config/model_costs.yaml?)")
    if unpriced:
        print(f"note: --dry: no price for {unpriced}; counted as $0")
    return sum(u.total_cost or 0.0 for log in logs for s in (log.samples or []) for u in (s.model_usage or {}).values())


def _ledger_spend() -> float:
    """Ledger-metered $ (build and embedding calls outside Inspect) in this scratch area."""
    from ape.analysis.cost import ledger_frame, load_prices
    from ape.config import Config
    from ape.llm.ledger import Ledger

    led = Ledger(Config().ledger_path)
    if not led.read():
        return 0.0
    try:
        return float(ledger_frame(led, load_prices())["usd"].sum())
    except KeyError as e:  # an unpriced build/embedding model: never let it read as $0 (or NaN, which passes the cap)
        raise SystemExit(f"ledger spend cannot be priced: {e}") from e


def _spend(logs, dry: bool = False) -> float:
    return _inspect_spend(logs, dry) + _ledger_spend()


def _guard(report: dict, logs: list, max_usd: float, dry: bool = False) -> None:
    report["spend_usd"] = _spend(logs, dry)
    if report["spend_usd"] > max_usd:
        raise SystemExit(f"spend {report['spend_usd']:.4f} exceeds --max-usd {max_usd}; stopping")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="config/models.yaml profile (default: $APE_MODEL_PROFILE or gate)")
    ap.add_argument("--agent", help="override the profile's agent model (keeps its effort)")
    ap.add_argument("--kg", help="override the profile's kg model (keeps its effort)")
    ap.add_argument("--build", help="override the build model (sets APE_BUILD_MODEL)")
    ap.add_argument("--embed", help="override the embedding model")
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    _isolate(args.dry)
    if args.profile:
        os.environ["APE_MODEL_PROFILE"] = args.profile
    if args.build:
        os.environ["APE_BUILD_MODEL"] = args.build

    from inspect_ai.log import read_eval_log

    from ape.apg.author import author_world, build_llm_author
    from ape.build import build
    from ape.config import Config
    from ape.lgr.build import build_index
    from ape.llm.ledger import Ledger
    from ape.models import agent_model, build_settings, embedding_model, load_profile, require_preflight, role_models
    from ape.runner import run_evals
    from ape.tasks.gate import gate
    from ape.worlds.render import chunk_world
    from ape.worlds.spec import World

    profile = load_profile()
    if args.embed:
        os.environ["APE_EMBEDDING_MODEL"] = args.embed
    os.environ.setdefault("APE_EMBEDDING_MODEL", embedding_model(profile))
    # Dry runs swap in mockllm agent/kg models, so the overrides only count live.
    require_preflight(profile, live=not args.dry, overrides={} if args.dry else {"agent": args.agent, "kg": args.kg})
    build_model, build_effort = build_settings(profile)
    report: dict = {"dry": args.dry, "models": {"profile": profile.name, "roles": profile.summary(), "overrides": vars(args), "build": [build_model, build_effort]}}
    logs: list = []
    cfg = Config()
    if args.dry:  # mock models, still built from the profile so the effort wiring is exercised
        from ape.llm.mock_agent import mock_agent, mock_kg

        agent = agent_model(profile, model="mockllm/model", custom_outputs=mock_agent)
        roles = role_models(profile, ("kg",), model="mockllm/model", custom_outputs=mock_kg)
    else:
        agent = agent_model(profile, model=args.agent)
        roles = role_models(profile, ("kg",), model=args.kg)

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
    _guard(report, logs, args.max_usd, args.dry)

    # D-017: authoring quality with the build model (perfect-author fake when dry).
    if args.dry:
        from ape.llm.fake import perfect_author as author
    else:
        author = build_llm_author(build_model, Ledger(cfg.ledger_path), world.id, build_effort)
    report["D017_authoring"] = asyncio.run(author_world(world, cfg, author))
    report["D017_pass"] = report["D017_authoring"]["id_coverage"] >= SPEC_THRESHOLD and report["L2"]["policy_ids_as_entities"] >= SPEC_THRESHOLD
    _guard(report, logs, args.max_usd, args.dry)

    # H4 (changing tool sets) + L4 (one keyword call per compile) + L5 (realized context) on live models.
    runs = {"H4_S3s_F3": dict(family="F3", level="5", arm="S3s"), "L4_L5_LGR_F7": dict(family="F7", level="100", arm="LGR-s" if not args.dry else "LGRo-s"), "APG_F7": dict(family="F7", level="100", arm="APG-s")}
    log_root = OUT / "logs" / time.strftime("%Y%m%dT%H%M%S")
    for name, kw in runs.items():
        _, (header,) = run_evals(gate(split="dev", **kw), log_root / name, profile=profile, model=agent, model_roles=roles, display="none", message_limit=40)
        log = read_eval_log(header.location)
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
        _guard(report, logs, args.max_usd, args.dry)

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
