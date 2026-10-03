"""Worker specialization for M2 (SPEC=1), defined from the world, since the worlds carry no persona content (D-028).

A world's knowledge falls into domains: F3's service domains (each owns its tools and its standard operating
procedures), F7's policy domains (each owns a handbook; exceptions amend policies of a domain) and F1's supplier
segments (each has one rating rule). The fixed three workers become three specialists: the world's domains, sorted,
are dealt round-robin (specialist i gets domains i, i+3, ...). A world with fewer domains than workers (F3-5 and F7-10
have two) deals its domains cyclically, so a domain can have more than one specialist and none is left empty.

What a specialist gets, and an M1k worker does not (M2 differs from M1k in this only):
- **Role context:** its first message names the domains it covers (`prompts.WORKER_SPECIALTY`), and every knowledge
  compile it asks for is prefixed with them (`query_prefix`), so the KG arm delivers its specialty's procedures and
  policies. Same KG arm, same budget and schedule as M1k.
- **A tool subset:** on F3 a specialist binds only its domains' tools (plus `order_lookup`); exposure then works as for
  any worker, inside that subset. F1's and F7's only non-terminal tools are their lookups, which every specialist keeps.
- **Routing:** the orchestrator sees the roster (domains and tools per specialist) and names a specialist per subtask.

F2 has no domain structure (an escalation hop is the same work for any supplier), so M2 is not defined there; the
brief's registry runs M2 on F3 and F7, and the run plan's Study F cells add F1-32.
"""

from dataclasses import dataclass

from ...worlds.gen_registry import SEGMENTS
from ...worlds.spec import World

N_WORKERS = 3  # fixed a priori (brief §4.2): 3 workers, 2 critique rounds, council k = 3
SPECIALIZED_FAMILIES = ("F1", "F3", "F7")


@dataclass(frozen=True)
class Specialist:
    name: str
    covers: tuple[str, ...]  # domain keys (F3 tool domains, F7 policy domains, F1 segments)
    labels: tuple[str, ...]  # the same, as the prompts word them
    tools: tuple[str, ...] | None  # the non-terminal tools it binds; None = every non-terminal tool

    @property
    def coverage(self) -> str:
        return ", ".join(self.labels)

    @property
    def query_prefix(self) -> str:
        return f"Specialty: {self.coverage}."


def domains(world: World) -> list[str]:
    if world.family == "F3":
        return sorted({t.domain for t in world.tools})
    if world.family == "F7":
        return sorted({p.domain for p in world.policies})
    if world.family == "F1":
        return sorted(r["segment"] for r in world.entities.get("rules", [{"segment": s} for s in SEGMENTS]))
    raise ValueError(f"M2 has no specialization on {world.family} (defined for {SPECIALIZED_FAMILIES}; F2 has no domains)")


def _label(world: World, key: str) -> str:
    return f"{key} suppliers" if world.family == "F1" else key.replace("_", " ")


def specialists(world: World, n: int = N_WORKERS) -> list[Specialist]:
    keys = domains(world)
    groups = [keys[i::n] for i in range(n)] if len(keys) >= n else [[keys[i % len(keys)]] for i in range(n)]
    out = []
    for i, covers in enumerate(groups, 1):
        tools = None
        if world.family == "F3":
            tools = ("order_lookup", *sorted(t.name for t in world.tools if t.domain in covers))
        out.append(Specialist(f"specialist_{i}", tuple(covers), tuple(_label(world, k) for k in covers), tools))
    return out
