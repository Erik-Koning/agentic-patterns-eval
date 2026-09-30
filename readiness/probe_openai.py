"""Readiness E2-E4: verify the OpenAI key, list models, and probe which parameters each role's model honours.

    uv run python readiness/probe_openai.py --list
    uv run python readiness/probe_openai.py --agent <id> --kg <id> --build <id> --embed <id>

Costs a handful of 16-token calls (well under $0.01). Requires approval (checkpoint 3).
The key is read from the environment or `.env`; it is never printed.
Writes `cache/openai_probe.json`, which PROVENANCE.md cites.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

OUT = Path(__file__).resolve().parents[1] / "cache" / "openai_probe.json"
STRICT = {"type": "json_schema", "json_schema": {"name": "probe", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}}}


def _try(client, model: str, **params) -> dict:
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": 'Reply with {"ok": true}.'}], max_completion_tokens=64, **params
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
    args = ap.parse_args()

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
    if args.embed:
        try:
            e = client.embeddings.create(model=args.embed, input=["dimension probe"])
            report["embeddings"] = {"model": args.embed, "dim": len(e.data[0].embedding), "usage": e.usage.model_dump()}
        except Exception as e:
            report["embeddings"] = {"model": args.embed, "error": str(e)[:200]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    summary = {r: {k: v["accepted"] for k, v in report[r].items() if isinstance(v, dict) and "accepted" in v} for r in ("agent", "kg", "build") if r in report}
    print(json.dumps({"written": str(OUT), "honoured": summary, "embeddings": report.get("embeddings")}, indent=1))


if __name__ == "__main__":
    main()
