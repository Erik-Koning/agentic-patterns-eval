"""The contract every delivery arm implements.

An arm turns a query into the knowledge text the agent sees, plus provenance
(which units and facts were delivered) and, optionally, the tools it allows.
`per_step` arms are recompiled every agent step; the others compile once.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from ..llm.tokens import count_tokens
from ..worlds.spec import TaskItem


@dataclass
class ContextResult:
    text: str
    unit_ids: list[str]  # chunk IDs, APG node IDs or LightRAG entity/relation/chunk IDs
    fact_ids: list[str]  # world-spec facts contained in the delivered text
    tools: list[str] | None = None  # tool names the arm allows; None = arm expresses no tool scope
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return count_tokens(self.text)

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def log_record(self, step: int) -> dict:
        return {
            "step": step,
            "tokens": self.tokens,
            "prompt_hash": self.prompt_hash,
            "unit_ids": self.unit_ids,
            "fact_ids": self.fact_ids,
            "tools": self.tools,
            "meta": self.meta,
        }


class DeliveryArm(Protocol):
    name: str
    per_step: bool

    async def compile(self, query: str, task: TaskItem) -> ContextResult: ...
