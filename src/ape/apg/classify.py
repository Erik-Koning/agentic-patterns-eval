"""APG's routing classifier, ported from the TypeScript Anthropic driver and run through
Inspect's model API (role "kg"), so its tokens are metered like any other call.

`CLASSIFY_SYSTEM` must stay string-equal to `typescript/packages/connectors/src/anthropic.ts`
(readiness A5, checked by tests). The TS driver forces a tool call; OpenAI's
equivalent is strict structured output, which requires every property to be
required, so `reason` becomes required here.
"""

import json
import re

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, Model, ResponseSchema
from inspect_ai.util import JSONSchema

from ..llm.build_client import strip_fences

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


_NODE_ID = re.compile(r"^\s*([^\s:]+?)\s*(?::|\s-\s|\s\(|\s|$)")


def _node_id(raw: str) -> str:
    """The node ID in `u00060`, `u00060: Refund policy` or `u00060 - Refund policy` (the outline's `id: descriptor`
    form echoed back)."""
    m = _NODE_ID.match(raw)
    return m.group(1) if m else raw.strip()


def parse_classification(text: str) -> tuple[list[dict], str | None, list[str]]:
    """(matches, error, repairs). Malformed output yields no matches, so routing falls back. Harmless deviations are
    repaired and named: a code fence around the JSON ("fence"), a node ID echoed with its descriptor ("id_prefix"),
    a confidence given as a percentage ("percent")."""
    repairs: list[str] = []
    cleaned = strip_fences(text)
    if cleaned != text:
        repairs.append("fence")
    try:
        raw = json.loads(cleaned)
        out = []
        for m in raw["matches"]:
            node_id = _node_id(str(m["nodeId"]))
            if node_id != str(m["nodeId"]):
                repairs.append("id_prefix")
            confidence = float(m["confidence"])
            if 1 < confidence <= 100:
                confidence /= 100
                repairs.append("percent")
            out.append({"nodeId": node_id, "confidence": confidence, "reason": str(m.get("reason", ""))})
        return out, None, list(dict.fromkeys(repairs))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return [], f"{type(e).__name__}: {e}", list(dict.fromkeys(repairs))


def parse_matches(text: str) -> tuple[list[dict], str | None]:
    """Returns (matches, error); see `parse_classification`."""
    matches, error, _ = parse_classification(text)
    return matches, error


async def classify(model: Model, query: str, outline: str, multi: bool) -> tuple[list[dict], str | None, list[str]]:
    out = await model.generate(
        classify_messages(query, outline, multi),
        config=GenerateConfig(response_schema=ResponseSchema(name="classify", json_schema=STRICT_CLASSIFY_SCHEMA, strict=True)),
    )
    return parse_classification(out.completion)
