"""Readiness E2-E4: verify the OpenAI key, list models, and probe each role's model on the call path the run uses.

    uv run python readiness/probe_openai.py --list --profile gate   # probe; writes cache/openai_probe.json
    uv run python readiness/probe_openai.py --pin                   # after review: pin the snapshots in PROVENANCE.md
    uv run python readiness/probe_openai.py --profile gate --agent gpt-6-sol    # a flag swaps a role's model, keeps its effort

**Why two paths.** Inspect sends GPT-6 calls through the OpenAI **Responses API** (gpt-5+ families), but other
models such as the anchor's gpt-4o-mini through chat completions. Build calls (APG authoring, LightRAG
extraction) go through `ape.llm.build_client.BuildLlm` (chat completions). So each role is probed where its
calls go:
- **inspect:** agent, kg, judge and probe (when the profile sets it). These go through `inspect_ai`'s own model
  (`get_model(model, config)`), so the request is exactly what an eval sends. Inspect's choice of API is recorded.
- **build:** build, and build_fallback with `--with-fallback`. These go through `BuildLlm`.

**Calls per path.**
- **as_configured:** the role's own config, as the run sets it.
- **structured_output:** a strict JSON schema, as APG classify, LightRAG keywords, the judge, F8 probes and APG
  authoring use.
- **effort:** one problem (`smoke_checks.EFFORT_PROMPT`) at reasoning effort high and low, up to 4,096 output
  tokens each. It passes when both are accepted and high uses more reasoning tokens than low
  (`smoke_checks.effort_verdict`). A role that sets no effort (gpt-4o-mini) is skipped.
- Effort calls are made once per (path, model) and shared by roles of the same model.

**Also recorded.**
- Per alias: the served snapshot (the response's `model` field), x-ratelimit-* headers (E4), `models.list()`
  (E2), the embedding model's dimension, and the price table's checked date.
- Build-path calls are metered in the build ledger (context `source: probe`).

**Cost:** about 8,700 output tokens per model per path. That is ≈ $0.01 for the gate profile (Luna only),
≈ $0.09 per Sol path and ≈ $0.45 per Astra path. It needs approval (checkpoint 3).

The key is read from the environment or `.env`; it is never printed. The probe writes `cache/openai_probe.json`
(format version 2: `roles -> paths`), which `readiness/smoke.py` reads (`effort` and `burst`) and
`ape.run_gate`'s preflight reads (`models_available`).

**Pinning.** `--pin` makes no API calls. It writes the probe's snapshots into PROVENANCE.md (`ape.snapshots`).
It refuses an alias with no served snapshot, or one served by several snapshots in the same probe. Run it only
after reviewing the probe. Live preflight then refuses any run whose latest probe sees a different snapshot.
"""

import argparse
import asyncio
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, Model, ResponseSchema, get_model
from inspect_ai.util import JSONSchema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_checks import EFFORT_MAX_TOKENS, EFFORT_PROMPT, PROBE_VERSION, SKIP, effort_verdict, result  # noqa: E402

OUT = ROOT / "cache" / "openai_probe.json"
INSPECT_PATH_ROLES = ("agent", "kg", "judge", "probe")  # = ape.models.INSPECT_ROLES
CONFIGURED_PROMPT = "Reply with the single word: ok"
STRUCTURED_PROMPT = 'Reply with the JSON object {"ok": true}.'
STRICT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
# A rejected parameter is a 400, which Inspect never retries; keep transient retries and hangs short.
PROBE_CONFIG = GenerateConfig(max_retries=2, timeout=180)
RATELIMIT_MAX_TOKENS = 16  # the Responses API's minimum max_output_tokens

ModelFactory = Callable[[str, GenerateConfig], Model]


def inspect_factory(model: str, config: GenerateConfig) -> Model:
    return get_model(model, config=config, memoize=False)


def price_table_date(costs_path: Path) -> str | None:
    """The date config/model_costs.yaml says its prices were checked ("checked YYYY-MM-DD")."""
    m = re.search(r"checked (\d{4}-\d{2}-\d{2})", costs_path.read_text()) if costs_path.is_file() else None
    return m.group(1) if m else None


def _alias(model: str) -> str:
    return model.split("/", 1)[-1]


def _error(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e)[:300]}"


def inspect_api(model: Model) -> str:
    """Which OpenAI API Inspect uses for this model: "responses" or "chat.completions" (other providers: their name)."""
    flag = getattr(model.api, "responses_api", None)
    return "responses" if flag is True else "chat.completions" if flag is False else type(model.api).__name__


async def _inspect_call(model: Model, prompt: str, config: GenerateConfig) -> dict:
    try:
        out = await model.generate(prompt, config=PROBE_CONFIG.merge(config))
    except Exception as e:  # the point is to record what the API rejects
        return {"accepted": False, "error": _error(e)}
    rec = {
        "accepted": out.error is None,
        "served_model": out.model or None,
        "usage": out.usage.model_dump(exclude_none=True) if out.usage else {},
        "stop_reason": out.stop_reason,
        "text": out.completion[:200],
    }
    if out.error:
        rec["error"] = str(out.error)[:300]
    return rec


def _structured_ok(rec: dict) -> dict:
    """A structured-output call counts as accepted only when its text parses to {"ok": <bool>}."""
    if rec.get("accepted"):
        try:
            ok = isinstance(json.loads(rec.get("text") or "").get("ok"), bool)
        except (ValueError, AttributeError):
            ok = False
        if not ok:
            rec = rec | {"accepted": False, "error": f"the reply does not match the strict schema: {rec.get('text')!r}"}
    return rec


def _effort_block(high: dict, low: dict) -> dict:
    return {"prompt": EFFORT_PROMPT, "max_tokens": EFFORT_MAX_TOKENS, "high": high, "low": low, "verdict": effort_verdict(high, low)}


def _skipped_effort() -> dict:
    return {"verdict": result(SKIP, {}, reason="the role sets no reasoning_effort")}


def _with_snapshots(path: dict) -> dict:
    calls = [path.get("as_configured"), path.get("structured_output"), *((path.get("effort") or {}).get(e) for e in ("high", "low"))]
    seen = [c["served_model"] for c in calls if isinstance(c, Mapping) and c.get("served_model")]
    path["snapshots_seen"] = sorted(set(seen))
    path["snapshot"] = seen[0] if len(set(seen)) == 1 else None
    return path


async def probe_inspect_role(spec: Any, factory: ModelFactory, effort_cache: dict) -> dict:
    """One Inspect-path role: `spec` is an `ape.models.RoleSpec`; `factory(model, config)` builds the Model."""
    model = factory(spec.model, spec.generate_config())
    path: dict = {"api": inspect_api(model)}
    path["as_configured"] = await _inspect_call(model, CONFIGURED_PROMPT, GenerateConfig())
    schema = ResponseSchema(name="probe", json_schema=JSONSchema.model_validate(STRICT_SCHEMA), strict=True)
    path["structured_output"] = _structured_ok(await _inspect_call(model, STRUCTURED_PROMPT, GenerateConfig(response_schema=schema)))
    if spec.reasoning_effort is None:
        path["effort"] = _skipped_effort()
    else:
        key = ("inspect", spec.model)
        if key not in effort_cache:
            high, low = [await _inspect_call(model, EFFORT_PROMPT, GenerateConfig(reasoning_effort=e, max_tokens=EFFORT_MAX_TOKENS)) for e in ("high", "low")]
            effort_cache[key] = _effort_block(high, low)
        path["effort"] = effort_cache[key]
    return _with_snapshots(path)


async def _build_call(llm: Any, call: Callable) -> dict:
    try:
        reply = await call(llm)
    except Exception as e:
        return {"accepted": False, "error": _error(e)}
    text = json.dumps(reply) if isinstance(reply, dict) else str(reply)
    return {"accepted": True, "served_model": llm.last_model, "usage": llm.last_usage or {}, "text": text[:200]}


async def probe_build_role(model: str, effort: str | None, ledger: Any, client: Any, effort_cache: dict, role: str) -> dict:
    """One build-path role through `BuildLlm` (chat completions), metered in `ledger` with context source=probe."""
    from ape.llm.build_client import BuildLlm

    def llm(e: str | None) -> Any:
        return BuildLlm(model, ledger, {"source": "probe", "role": role}, client=client, reasoning_effort=e)

    path: dict = {"api": "chat.completions"}
    path["as_configured"] = await _build_call(llm(effort), lambda x: x.chat([{"role": "user", "content": CONFIGURED_PROMPT}]))
    path["structured_output"] = _structured_ok(await _build_call(llm(effort), lambda x: x.json("You are a readiness probe.", STRUCTURED_PROMPT, "probe", STRICT_SCHEMA)))
    if effort is None:
        path["effort"] = _skipped_effort()
    else:
        key = ("build", model)
        if key not in effort_cache:
            msgs = [{"role": "user", "content": EFFORT_PROMPT}]
            high, low = [await _build_call(llm(e), lambda x: x.chat(msgs, max_completion_tokens=EFFORT_MAX_TOKENS)) for e in ("high", "low")]
            effort_cache[key] = _effort_block(high, low)
        path["effort"] = effort_cache[key]
    return _with_snapshots(path)


def ratelimit(client: Any, model: str, api: str) -> dict:
    """x-ratelimit-* headers for `model` from one tiny raw call on `api` (E4)."""
    try:
        if api == "responses":
            raw = client.responses.with_raw_response.create(model=model, input="ok", max_output_tokens=RATELIMIT_MAX_TOKENS)
        else:
            raw = client.chat.completions.with_raw_response.create(model=model, messages=[{"role": "user", "content": "ok"}], max_completion_tokens=RATELIMIT_MAX_TOKENS)
        return {"api": api, "headers": {k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")}}
    except Exception as e:
        return {"api": api, "error": _error(e)}


def snapshot_summary(roles: Mapping[str, dict]) -> dict:
    """Per alias: the snapshots that served it across every role and path, and which roles use it."""
    out: dict[str, dict] = {}
    for role, entry in roles.items():
        for path_name, path in entry["paths"].items():
            a = out.setdefault(_alias(entry["model"]), {"snapshots_seen": set(), "roles": []})
            a["snapshots_seen"] |= set(path.get("snapshots_seen") or [])
            a["roles"].append(f"{role} ({path_name})")
    for a in out.values():
        seen = sorted(a.pop("snapshots_seen"))
        a["snapshots_seen"] = seen
        a["snapshot"] = seen[0] if len(seen) == 1 else None
    return out


async def run_probe(
    profile: Any,
    *,
    factory: ModelFactory = inspect_factory,
    client: Any = None,
    build_client: Any = None,
    ledger: Any = None,
    with_fallback: bool = False,
    costs_path: Path = ROOT / "config" / "model_costs.yaml",
    embed: bool = True,
) -> dict:
    """Probe every role of `profile` (an `ape.models.Profile`) on its call path; returns the report (version 2).

    `client` is a sync OpenAI client for models.list, rate-limit headers and embeddings (None skips those);
    `build_client` an AsyncOpenAI-compatible client for `BuildLlm` (None: BuildLlm's own); `ledger` meters the
    build-path calls (None: the build ledger, `Config().ledger_path`)."""
    from ape.config import Config
    from ape.llm.ledger import Ledger

    ledger = ledger or Ledger(Config().ledger_path)
    report: dict = {
        "version": PROBE_VERSION,
        "probed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "profile": profile.name,
        "price_table": {"path": str(costs_path.relative_to(ROOT)) if costs_path.is_relative_to(ROOT) else str(costs_path), "checked": price_table_date(costs_path)},
    }
    if client is not None:
        try:
            report["models_available"] = sorted(m.id for m in client.models.list())
        except Exception as e:
            report["models_available_error"] = _error(e)
            report["models_available"] = []
    roles: dict[str, dict] = {}
    cache: dict = {}
    for role in INSPECT_PATH_ROLES:
        if role in profile.roles:
            spec = profile.roles[role]
            roles[role] = {"model": spec.model, "configured_effort": spec.reasoning_effort, "paths": {"inspect": await probe_inspect_role(spec, factory, cache)}}
    for role in ("build", "build_fallback") if with_fallback else ("build",):
        if role in profile.roles:
            spec = profile.roles[role]
            path = await probe_build_role(spec.model, spec.reasoning_effort, ledger, build_client, cache, role)
            roles[role] = {"model": spec.model, "configured_effort": spec.reasoning_effort, "paths": {"build": path}}
    report["roles"] = roles
    report["snapshots"] = snapshot_summary(roles)
    if client is not None:
        apis: dict[str, str] = {}
        for entry in roles.values():
            for path in entry["paths"].values():
                apis.setdefault(_alias(entry["model"]), path["api"] if path["api"] in ("responses", "chat.completions") else "chat.completions")
        report["ratelimits"] = {a: ratelimit(client, a, api) for a, api in sorted(apis.items())}
        if embed and "embeddings" in profile.roles:
            m = profile.roles["embeddings"].model
            try:
                e = client.embeddings.create(model=m, input=["dimension probe"])
                report["embeddings"] = {"model": m, "dim": len(e.data[0].embedding), "served_model": getattr(e, "model", None), "usage": e.usage.model_dump()}
            except Exception as ex:
                report["embeddings"] = {"model": m, "error": _error(ex)}
    return report


# --- Offline stand-ins (readiness/smoke.py --dry): the probe's own code on mock models, no network ------------

OFFLINE_SNAPSHOT = "offline-mock-snapshot"
OFFLINE_REASONING = {"high": 900, "medium": 400, "low": 120}


def _offline_reply(prompt: str, effort: str | None) -> tuple[str, int]:
    text = '{"ok": true}' if '"ok"' in prompt else "466"
    return text, OFFLINE_REASONING.get(effort or "", 0)


def offline_factory(model: str, config: GenerateConfig) -> Model:
    """A mockllm model that reports reasoning tokens by effort, as the dry run's Inspect path."""
    from inspect_ai.model import ModelOutput, ModelUsage

    def outputs(messages, tools, tool_choice, cfg) -> ModelOutput:
        text, rt = _offline_reply(messages[-1].text, cfg.reasoning_effort)
        out = ModelOutput.from_content(OFFLINE_SNAPSHOT, text)
        out.usage = ModelUsage(input_tokens=30, output_tokens=rt + 5, total_tokens=rt + 35, reasoning_tokens=rt)
        return out

    return get_model("mockllm/model", config=config, custom_outputs=outputs, memoize=False)


class OfflineBuildClient:
    """An AsyncOpenAI stand-in for `BuildLlm` in the dry run: chat completions with reasoning tokens by effort."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, model: str, messages: list[dict], reasoning_effort: str | None = None, **_: Any) -> Any:
        from openai.types.chat import ChatCompletion

        text, rt = _offline_reply(messages[-1]["content"], reasoning_effort)
        return ChatCompletion.model_validate({
            "id": "offline", "object": "chat.completion", "created": 0, "model": OFFLINE_SNAPSHOT,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": rt + 5, "total_tokens": rt + 35, "completion_tokens_details": {"reasoning_tokens": rt}},
        })


def apply_overrides(profile: Any, overrides: Mapping[str, str | None]) -> Any:
    """The profile with each given role's model swapped (its effort and other settings kept)."""
    roles = dict(profile.roles)
    for role, model in overrides.items():
        if model and role in roles:
            roles[role] = replace(roles[role], model=model)
    return replace(profile, roles=roles)


def summary(report: Mapping) -> dict:
    from smoke_checks import effort_from_probe

    check = effort_from_probe(report)
    return {
        "check": check["status"],
        "reason": check.get("reason"),
        "paths": {k: {"api": v["api"], "effort": v["status"], "snapshot": v["snapshot"]} for k, v in check["measured"].get("paths", {}).items()},
        "snapshots": {a: v.get("snapshot") for a, v in (report.get("snapshots") or {}).items()},
        "embeddings": report.get("embeddings"),
    }


def pin(probe_path: Path, provenance_path: Path) -> dict:
    """`--pin`: the probe's snapshots into PROVENANCE.md. No API calls."""
    from ape.snapshots import write_pins

    probe = json.loads(probe_path.read_text())
    if probe.get("version") != PROBE_VERSION:
        raise SystemExit(f"{probe_path} predates the call-path probe: re-run readiness/probe_openai.py before pinning")
    try:
        return write_pins(provenance_path, probe, probe_path=probe_path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print models.list()")
    ap.add_argument("--profile", help="config/models.yaml profile (default: $APE_MODEL_PROFILE or gate)")
    for role in ("agent", "kg", "judge", "build"):
        ap.add_argument(f"--{role}", help=f"override the profile's {role} model (keeps its effort)")
    ap.add_argument("--embed", help="override the embedding model")
    ap.add_argument("--with-fallback", action="store_true", help="also probe the D-017 fallback builder (build path)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--pin", action="store_true", help="pin the probe's snapshots in PROVENANCE.md (no API calls)")
    ap.add_argument("--provenance", type=Path, default=ROOT / "PROVENANCE.md")
    args = ap.parse_args(argv)

    if args.pin:
        res = pin(args.out, args.provenance)
        print(json.dumps({"provenance": str(args.provenance)} | res, indent=1))
        return

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (E2). Put it in .env; see DECISIONS O-1.")
    from openai import OpenAI

    from ape.models import load_profile

    profile = load_profile(args.profile)
    overrides = {r: (f"openai/{getattr(args, r)}" if getattr(args, r) and r != "build" and "/" not in getattr(args, r) else getattr(args, r)) for r in ("agent", "kg", "judge", "build")}
    overrides["embeddings"] = args.embed
    profile = apply_overrides(profile, overrides)
    client = OpenAI()
    report = asyncio.run(run_probe(profile, client=client, with_fallback=args.with_fallback))
    if args.list:
        print("\n".join(report.get("models_available") or []))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"written": str(args.out)} | summary(report), indent=1))
    print("Next: review the probe, then pin the snapshots: uv run python readiness/probe_openai.py --pin", file=sys.stderr)


if __name__ == "__main__":
    main()
