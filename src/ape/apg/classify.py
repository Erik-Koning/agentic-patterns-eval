"""APG's routing classifier, ported from the TypeScript Anthropic driver and run through
Inspect's model API (role "kg"), so its tokens are metered like any other call.

`CLASSIFY_SYSTEM` must stay string-equal to `typescript/packages/connectors/src/anthropic.ts`
(readiness A5, checked by tests). The TS driver forces a tool call; OpenAI's
equivalent is strict structured output, which requires every property to be
required, so `reason` becomes required here.
"""

import json

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, Model, ResponseSchema
from inspect_ai.util import JSONSchema

CLASSIFY_SYSTEM = (
    "You are a routing classifier for an Adaptive Prompt Graph. Given a user query and a category outline "
    "(one node per line: `id: descriptor`), report every node the query plausibly belongs to with a confidence "
    "in [0,1]. Prefer the most specific (deepest) applicable nodes. Report nothing for irrelevant nodes."
)

STRICT_CLASSIFY_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": ["matches"],
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["nodeId", "confidence", "reason"],
                    "properties": {
                        "nodeId": {"type": "string"},
                        "confidence": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                },
            }
        },
    }
)


def classify_messages(query: str, outline: str, multi: bool) -> list:
    system = CLASSIFY_SYSTEM + ("" if multi else " Report at most one node.")
    return [ChatMessageSystem(content=system), ChatMessageUser(content=f"Query: {query}\n\nCategory outline:\n{outline}")]


def parse_matches(text: str) -> tuple[list[dict], str | None]:
    """Returns (matches, error). Malformed output yields no matches, so routing falls back."""
    try:
        raw = json.loads(text)
        matches = raw["matches"]
        out = [
            {"nodeId": str(m["nodeId"]), "confidence": float(m["confidence"]), "reason": str(m.get("reason", ""))}
            for m in matches
        ]
        return out, None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return [], f"{type(e).__name__}: {e}"


async def classify(model: Model, query: str, outline: str, multi: bool) -> tuple[list[dict], str | None]:
    out = await model.generate(
        classify_messages(query, outline, multi),
        config=GenerateConfig(response_schema=ResponseSchema(name="classify", json_schema=STRICT_CLASSIFY_SCHEMA, strict=True)),
    )
    return parse_matches(out.completion)
