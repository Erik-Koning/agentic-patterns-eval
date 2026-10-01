"""Readiness E2-E4: verify the OpenAI key, list models, and probe which parameters each role's model honours.

    uv run python readiness/probe_openai.py --list
    uv run python readiness/probe_openai.py --agent <id> --kg <id> --build <id> --embed <id>
    uv run python readiness/probe_openai.py --profile gate      # the profile's models for any role not given

Per role model: plain, temperature 0, seed, reasoning_effort low and strict JSON schema, each a call of at most
64 output tokens; and (FX-8) whether reasoning effort is honoured: one problem (`smoke_checks.EFFORT_PROMPT`)
at reasoning_effort high and low, up to 4,096 output tokens each, passing when both are accepted and high uses
more reasoning tokens than low (`smoke_checks.effort_verdict`). Cost: at most ~8,600 output tokens per role
model, so ≈ $0.005 per GPT-6 Luna role, ≈ $0.09 per Sol and ≈ $0.43 per Astra; ≈ $0.02 for the gate profile.
Requires approval (checkpoint 3). The key is read from the environment or `.env`; it is never printed.
Writes `cache/openai_probe.json`, which PROVENANCE.md cites and `readiness/smoke.py` reads (its effort check).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_checks import EFFORT_MAX_TOKENS, EFFORT_PROMPT, effort_verdict  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "cache" / "openai_probe.json"
STRICT = {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}}}


def _try(client, model: str, prompt: str = 'Reply with {"ok": true}.', max_tokens: int = 64, **params) -> dict:
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=max_tokens, **params
        )
        resp = raw.parse()
        headers = {k: v for k, v in raw.headers.items() if k.startswith("x-ratelimit")}
        return {"accepted": True, "usage": resp.usage.model_dump(), "ratelimit": headers}
    except Exception as e:  # the point is to record what the API rejects
        return {"accepted": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> None:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (E2). Put it in .env; see DECISIONS O-1.")
    from openai import OpenAI

    client = OpenAI()
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--agent")
    ap.add_argument("--kg")
    ap.add_argument("--build")
    ap.add_argument("--embed")
    ap.add_argument("--profile", help="config/models.yaml profile whose models fill any role not given")
    args = ap.parse_args()
    if args.profile:
        from ape.models import load_profile

        p = load_profile(args.profile)
        for role, name in (("agent", "agent"), ("kg", "kg"), ("build", "build"), ("embed", "embeddings")):
            if not getattr(args, role) and name in p.roles:
                setattr(args, role, p.role(name).model.split("/", 1)[-1])

    models = sorted(m.id for m in client.models.list())
    report: dict = {"models_available": models}
    if args.list:
        print("\n".join(models))
    for role in ("agent", "kg", "build"):
        model = getattr(args, role)
        if not model:
            continue
        report[role] = {
            "model": model,
            "plain": _try(client, model),
            "temperature_0": _try(client, model, temperature=0),
            "seed": _try(client, model, seed=7),
            "reasoning_effort_low": _try(client, model, reasoning_effort="low"),
            "strict_json_schema": _try(client, model, response_format=STRICT),
        }
        effort = {e: _try(client, model, prompt=EFFORT_PROMPT, max_tokens=EFFORT_MAX_TOKENS, reasoning_effort=e) for e in ("high", "low")}
        report[role]["effort"] = {"prompt": EFFORT_PROMPT, "max_completion_tokens": EFFORT_MAX_TOKENS, **effort, "verdict": effort_verdict(effort["high"], effort["low"])}
    if args.embed:
        try:
            e = client.embeddings.create(model=args.embed, input=["dimension probe"])
            report["embeddings"] = {"model": args.embed, "dim": len(e.data[0].embedding), "usage": e.usage.model_dump()}
        except Exception as e:
            report["embeddings"] = {"model": args.embed, "error": str(e)[:200]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    summary = {r: {k: v["accepted"] for k, v in report[r].items() if isinstance(v, dict) and "accepted" in v} for r in ("agent", "kg", "build") if r in report}
    effort = {r: {k: report[r]["effort"]["verdict"][k] for k in ("status", "reason", "measured")} for r in ("agent", "kg", "build") if r in report}
    print(json.dumps({"written": str(OUT), "honoured": summary, "effort": effort, "embeddings": report.get("embeddings")}, indent=1))


if __name__ == "__main__":
    main()
