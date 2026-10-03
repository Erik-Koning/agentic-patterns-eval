"""World spec: the single structured source that every KB artifact and task derives from.

A world holds atomic facts (with IDs), the structured objects those facts describe
(policies, tools, procedures, events), the rendered documents, and generated tasks.
Documents carry per-paragraph fact IDs so any chunk or KG node can be mapped back
to the facts it contains (gold-evidence scoring at fact granularity).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Fact:
    id: str
    text: str
    kind: str  # "policy" | "exception" | "tool" | "procedure" | "event"


@dataclass
class Policy:
    id: str
    domain: str
    region: str
    lo: int
    hi: int
    outcome: dict  # {"action", "approver", "deadline_days", "document"}
    fact_id: str


@dataclass
class Exception_:
    id: str
    policy_id: str  # the base policy it overrides
    tier: str  # applies when the customer holds this tier
    outcome: dict
    fact_id: str


@dataclass
class Tool:
    name: str
    domain: str
    description: str
    params: dict  # {param_name: description}; all params are strings
    mutating: bool
    fact_id: str


@dataclass
class Procedure:
    id: str
    domain: str
    situation: str  # canonical situation key, e.g. "lost_parcel/EU"
    steps: list[dict]  # [{"tool": name, "args": {k: constant or "{order_id}"}}]
    fact_id: str


@dataclass
class Event:
    id: str
    date: str  # ISO date the change takes effect
    subject: str  # team name
    relation: str  # "manager" | "office"
    value: str  # person name or city
    fact_id: str


@dataclass
class Paragraph:
    text: str
    fact_ids: list[str]


@dataclass
class Document:
    id: str
    title: str
    paragraphs: list[Paragraph]

    @property
    def text(self) -> str:
        return f"# {self.title}\n\n" + "\n\n".join(p.text for p in self.paragraphs)


@dataclass
class TaskItem:
    id: str
    world_id: str
    family: str
    level: str
    prompt: str
    gold: dict
    gold_fact_ids: list[str]
    answer_tool: str
    setup: dict = field(default_factory=dict)  # per-task environment state (customer records, orders)
    tags: dict = field(default_factory=dict)  # e.g. {"case": "exception_applies"}


@dataclass
class World:
    id: str
    family: str  # "F7" | "F3" | "F5"
    level: str
    seed: int
    split: str  # "dev" | "pilot" | "test"
    facts: dict[str, Fact] = field(default_factory=dict)
    policies: list[Policy] = field(default_factory=list)
    exceptions: list[Exception_] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    documents: list[Document] = field(default_factory=list)
    tasks: list[TaskItem] = field(default_factory=list)

    def add_fact(self, fact_id: str, text: str, kind: str) -> str:
        assert fact_id not in self.facts, f"duplicate fact id {fact_id}"
        self.facts[fact_id] = Fact(fact_id, text, kind)
        return fact_id

    def to_dict(self) -> dict:
        return asdict(self)

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def artifact_hash(self) -> str:
        """Hash of what knowledge artifacts (APG graphs, LightRAG indices, fact matchers) are built from: the world
        without its task list. The same world generated with 20 or 40 tasks shares its graphs and indices."""
        d = self.to_dict()
        d.pop("tasks", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=1))

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        d = dict(d)
        d["facts"] = {k: Fact(**v) for k, v in d["facts"].items()}
        d["policies"] = [Policy(**p) for p in d["policies"]]
        d["exceptions"] = [Exception_(**x) for x in d["exceptions"]]
        d["tools"] = [Tool(**t) for t in d["tools"]]
        d["procedures"] = [Procedure(**p) for p in d["procedures"]]
        d["events"] = [Event(**e) for e in d["events"]]
        d["documents"] = [
            Document(doc["id"], doc["title"], [Paragraph(**p) for p in doc["paragraphs"]]) for doc in d["documents"]
        ]
        d["tasks"] = [TaskItem(**t) for t in d["tasks"]]
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "World":
        return cls.from_dict(json.loads(Path(path).read_text()))
