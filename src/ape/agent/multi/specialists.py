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
brief's registry runs M2 on F3 and F7 (D-034 dropped it from F1-32; F1's definition stays for completeness).

**F8 sessions (Study G, B9).** An F8 world merges an F7 policy world and an F3 procedure world, so its domains are
both kinds: the policy domains (sorted), then the service domains (sorted), dealt the same way. A specialist covering a
policy domain gets `lookup_customer`; one covering a service domain gets `order_lookup` and that domain's procedure
tools. Sessions have no KG (the corpus is in every agent's system prompt), so `query_prefix` is unused there: an F8
specialist's role context is its specialty sentence, its tool subset and the orchestrator's routing.
"""

from dataclasses import dataclass

from ...worlds import gen_f7
from ...worlds.gen_registry import SEGMENTS
from ...worlds.spec import World

N_WORKERS = 3  # fixed a priori (brief §4.2): 3 workers, 2 critique rounds, council k = 3
SPECIALIZED_FAMILIES = ("F1", "F3", "F7", "F8")


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
    if world.family == "F8":
        return [*sorted({p.domain for p in world.policies}), *sorted({t.domain for t in world.tools})]
    raise ValueError(f"M2 has no specialization on {world.family} (defined for {SPECIALIZED_FAMILIES}; F2 has no domains)")


def _label(world: World, key: str) -> str:
    if world.family == "F1":
        return f"{key} suppliers"
    if world.family == "F8":
        policy = dict(gen_f7.DOMAINS)
        return f"{policy[key]} requests" if key in {p.domain for p in world.policies} else f"{key.replace('_', ' ')} tickets"
    return key.replace("_", " ")


def _f8_tools(world: World, covers: list[str]) -> tuple[str, ...]:
    policy, service = {p.domain for p in world.policies}, {t.domain for t in world.tools}
    tools = ["lookup_customer"] if set(covers) & policy else []
    if set(covers) & service:
        tools += ["order_lookup", *sorted(t.name for t in world.tools if t.domain in covers)]
    return tuple(tools)


def specialists(world: World, n: int = N_WORKERS) -> list[Specialist]:
    keys = domains(world)
    groups = [keys[i::n] for i in range(n)] if len(keys) >= n else [[keys[i % len(keys)]] for i in range(n)]
    out = []
    for i, covers in enumerate(groups, 1):
        tools = None
        if world.family == "F3":
            tools = ("order_lookup", *sorted(t.name for t in world.tools if t.domain in covers))
        elif world.family == "F8":
            tools = _f8_tools(world, covers)
        out.append(Specialist(f"specialist_{i}", tuple(covers), tuple(_label(world, k) for k in covers), tools))
    return out
