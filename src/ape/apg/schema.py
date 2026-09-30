"""JSON-Schema validation for APG documents (gap G7: the Python kernel only runs semantic checks).

`apg.schema.json` is vendored from EvolvingWisdomAgents/schema (sha256 in PROVENANCE.md).
"""

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("apg.schema.json")


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.validators.validator_for(schema)(schema)


def schema_errors(doc: dict) -> list[str]:
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _validator().iter_errors(doc)]
