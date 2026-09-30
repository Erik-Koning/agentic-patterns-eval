"""F5 relational/temporal QA worlds (LightRAG's home turf; reported, never pooled into the gate).

Teams change managers and offices over 2021-2025; changes are announced in
quarterly newsletters. Level "1hop" asks who managed a team on a date; "2hop"
asks where the team managed by a person was based on a date (manager lookup, then
office lookup, both time-scoped).
"""

import datetime as dt
import random

from .spec import Document, Event, Paragraph, TaskItem, World

TEAMS = ["Atlas", "Borealis", "Cobalt", "Delta", "Ember", "Fjord", "Granite", "Harbor", "Iris", "Juniper", "Kestrel", "Lumen"]
PEOPLE = [
    "Priya Nair", "Omar Li", "Dana Reyes", "Tomas Berg", "Aiko Tanaka", "Lena Fischer", "Kwame Mensah", "Sofia Rossi",
    "Mateo Silva", "Hana Kim", "Ravi Patel", "Elena Petrova", "Jonas Weber", "Amara Okafor", "Lucas Martin",
    "Yara Haddad", "Noah Cohen", "Mei Chen", "Ivan Horvat", "Zara Ali", "Felix Dubois", "Nina Larsen", "Diego Ramos",
    "Chloe Moreau", "Samir Aziz", "Ines Costa", "Oskar Nowak", "Leila Karimi", "Ben Carter", "Tara Singh",
    "Hugo Laurent", "Maya Levi", "Arjun Rao", "Freya Olsen", "Kenji Sato", "Lucia Romero", "Emil Stroem", "Nadia Farah",
]
CITIES = ["Lisbon", "Toronto", "Osaka", "Nairobi", "Denver", "Krakow", "Santiago", "Melbourne"]
START, END = dt.date(2021, 1, 1), dt.date(2025, 12, 31)
LEVELS = ("1hop", "2hop")


def _dates(rng: random.Random, k: int) -> list[dt.date]:
    span = (END - START).days
    return sorted(START + dt.timedelta(days=d) for d in rng.sample(range(30, span - 30), k))


def generate(level: str, split: str, seed: int, n_tasks: int) -> World:
    assert level in LEVELS
    rng = random.Random(f"F5|{level}|{split}|{seed}")
    world = World(id=f"F5-{level}-{split}-s{seed}", family="F5", level=level, seed=seed, split=split)
    people = rng.sample(PEOPLE, len(PEOPLE))  # each person manages at most one team, ever
    timelines: dict[str, dict[str, list[tuple[dt.date, str]]]] = {}
    n_event = 0

    def add_event(date: dt.date, team: str, relation: str, value: str) -> None:
        nonlocal n_event
        eid = f"E-{n_event:04d}"
        n_event += 1
        if relation == "manager":
            text = f"Effective {date.isoformat()}, {value} became manager of the {team} team."
        else:
            text = f"Effective {date.isoformat()}, the {team} team is based in the {value} office."
        world.add_fact(f"f-{eid}", text, "event")
        world.events.append(Event(eid, date.isoformat(), team, relation, value, f"f-{eid}"))
        timelines.setdefault(team, {}).setdefault(relation, []).append((date, value))

    for team in TEAMS:
        add_event(START, team, "manager", people.pop())
        add_event(START, team, "office", rng.choice(CITIES))
        for d in _dates(rng, 2):
            add_event(d, team, "manager", people.pop())
        for d in _dates(rng, 2):
            prev = timelines[team]["office"][-1][1]
            add_event(d, team, "office", rng.choice([c for c in CITIES if c != prev]))

    by_quarter: dict[str, list[Event]] = {}
    for e in sorted(world.events, key=lambda e: (e.date, e.id)):
        d = dt.date.fromisoformat(e.date)
        by_quarter.setdefault(f"{d.year}-Q{(d.month - 1) // 3 + 1}", []).append(e)
    for q, evs in sorted(by_quarter.items()):
        world.documents.append(
            Document(f"newsletter-{q}", f"Company Newsletter {q}", [Paragraph(world.facts[e.fact_id].text, [e.fact_id]) for e in evs])
        )

    def value_at(team: str, relation: str, date: dt.date) -> tuple[str, str]:
        """(value, fact_id) in force on `date`."""
        best = None
        for e in world.events:
            if e.subject == team and e.relation == relation and dt.date.fromisoformat(e.date) <= date:
                if best is None or e.date > best.date:
                    best = e
        return best.value, best.fact_id

    for t in range(n_tasks):
        team = rng.choice(TEAMS)
        date = START + dt.timedelta(days=rng.randint(10, (END - START).days - 10))
        manager, f_mgr = value_at(team, "manager", date)
        if level == "1hop":
            prompt = f"Who was the manager of the {team} team on {date.isoformat()}? Submit the full name with submit_answer."
            gold, facts = manager, [f_mgr]
        else:
            city, f_off = value_at(team, "office", date)
            prompt = (
                f"On {date.isoformat()}, in which city was the team managed by {manager} based? "
                f"Submit the city with submit_answer."
            )
            gold, facts = city, [f_mgr, f_off]
        world.tasks.append(
            TaskItem(
                id=f"{world.id}-t{t:03d}",
                world_id=world.id,
                family="F5",
                level=level,
                prompt=prompt,
                gold={"answer": gold},
                gold_fact_ids=facts,
                answer_tool="submit_answer",
                tags={"team": team, "date": date.isoformat()},
            )
        )
    return world
