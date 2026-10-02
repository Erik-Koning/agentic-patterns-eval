"""F1 breadth-aggregation and F2 dependency-chain worlds: one shared supplier registry (main study).

The world (ORCHESTRATOR_BRIEF_v2 §5.1) depends only on (split, seed), never on the family or level, so an F1-2,
F1-32, F2-2 and F2-10 world with the same seed share one KB and one record set: the knob is the only difference.

- **Records** (tool data, never in the KB): POOL suppliers, each a bulky JSON record (~400 tokens: profile,
  contacts, four quarters of seven scorecard metrics, certifications, notes) served by `lookup_supplier`.
  `render_record` fixes the field order, so the fields a step needs (segment, escalation code) come first and
  survive the shared step query's truncation.
- **KB** (the rendered corpus every arm draws on, ~3.5K tokens at every level, so S1 fits far below 0.8 W with the
  main study's nominal W = 128K; brief §0):
  - "Supplier Scorecard Rules": metric definitions and one rating rule per segment (facts), program text.
  - "Supplier Review FAQ": paraphrased second renderings of about half the rules, plus distractor Q&As.
  - "Escalation Directory": the escalation protocol and one entry per escalation code (code -> supplier; facts),
    with retired codes as distractors.
  - "Regional Escalation Contacts": paraphrased second renderings of about half the directory entries.
  - "Supplier Program Overview": distractor text only.
  Every fact appears in r ∈ {1, 2} documents (`entities["redundancy"]`); paragraphs without facts are distractors.

**F1 breadth aggregation** (levels 2, 8, 32 = N): "rate these N suppliers for the Qx review". Every supplier needs
its own lookup (a bulky record) and its segment's rule, and the N lookups are independent: the work decomposes
into N sub-questions (`tags["subtasks"]`, each with its gold and gold facts, for orchestrator arms; never in the
prompt). At N = 32 the records alone are ~13K tokens of near-identical JSON in one context; isolated workers each
see a few. Answer: `submit_ratings({supplier_id: rating})`. Primary success: all N exact; secondary: item F1.

**F2 dependency chains** (levels 2, 5, 10 = k hops): "dispute opened with supplier A0, escalated k times: who holds
it now?" Hop j: look up A(j-1)'s record for its escalation code (tool), then resolve the code in the Escalation
Directory (KB) to A(j). Code c(j) appears only in A(j-1)'s record, and A(j) only in c(j)'s directory entry (and its
own record), so the chain cannot be parallelized or skipped. The escalation graph is one random cycle over the
pool, so chains up to POOL-1 hops never repeat. Answer: `submit_chain(final_supplier, chain)`. Gold: final supplier
plus the chain. Primary success: final exact; secondary: correct-prefix length of the chain.

**Environment fault (Study D data side, brief §5.3).** Every F2 task carries `tags["fault"]`: on hop j the first
lookup of A(j-1) reports a wrong but valid escalation code (one whose target is off the true chain); later lookups
return the truth. `env_tools` applies it only when the task's `setup["faults_enabled"]` is set (`tasks.main`'s
`faults=True`), so ordinary runs never see it.

The turn cap scales with the knob (`turn_cap`): sequential lookups at N = 32, and a lookup plus a pull per hop at
k = 10, must not hit the gate's default of 12 turns.
"""

import json
import random

from .spec import Document, Paragraph, TaskItem, World

POOL = 48
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
RATINGS = ("preferred", "approved", "probation")
LEVELS = {"F1": ("2", "8", "32"), "F2": ("2", "5", "10")}
REDUNDANCY_RATE = 0.5  # share of rule and directory facts rendered twice (r = 2)

# metric: (label for rules, higher_is_better, domain lo, domain hi, step, thresholds (good, bad) choices)
METRICS = {
    "on_time_pct": ("on-time delivery rate", True, 70.0, 100.0, 0.1, [(p, p - d) for p in (94.0, 95.0, 96.0, 97.0) for d in (8.0, 10.0, 12.0)]),
    "fill_pct": ("fill rate", True, 80.0, 100.0, 0.1, [(p, p - d) for p in (95.0, 96.0, 97.0, 98.0) for d in (6.0, 8.0, 10.0)]),
    "defect_ppm": ("defect rate", False, 20, 3000, 1, [(p, p * m) for p in (200, 300, 400, 500) for m in (3, 4)]),
    "lead_time_days": ("average lead time", False, 2, 40, 1, [(p, p + d) for p in (7, 10, 14) for d in (7, 10)]),
    "disputes": ("number of disputed invoices", False, 0, 15, 1, [(p, p + d) for p in (1, 2) for d in (3, 4, 5)]),
    "audit_score": ("audit score", True, 50, 100, 1, [(p, p - d) for p in (85, 88, 90) for d in (15, 20)]),
    "return_pct": ("return rate", False, 0.2, 12.0, 0.1, [(p, p + d) for p in (2.0, 3.0, 4.0) for d in (3.0, 4.0)]),
}
UNITS = {"on_time_pct": "%", "fill_pct": "%", "defect_ppm": " ppm", "lead_time_days": " days", "disputes": "", "audit_score": " points", "return_pct": "%"}
SEGMENTS = ["packaging", "electronics", "apparel", "furniture", "cosmetics", "groceries", "toys", "hardware"]
REGIONS = ["NA", "EU", "UK", "APAC", "LATAM"]
CURRENCY = {"NA": "USD", "EU": "EUR", "UK": "GBP", "APAC": "SGD", "LATAM": "BRL"}
NAME_A = ["Northwind", "Bluepeak", "Cedar", "Harbor", "Ironleaf", "Juniper", "Kestrel", "Lumen", "Meridian", "Oakridge",
          "Pinecrest", "Quarry", "Redstone", "Silverline", "Tidewater", "Upland", "Vantage", "Westbrook", "Yarrow", "Zenith"]
NAME_B = ["Fasteners", "Packaging", "Textiles", "Components", "Supplies", "Goods", "Logistics", "Labs", "Works", "Trading",
          "Industries", "Mills", "Foods", "Plastics", "Devices", "Crafts"]
NAME_C = ["Ltd", "Inc", "GmbH", "Co", "S.A.", "Pty", "LLC", "plc"]
PEOPLE = ["Priya Nair", "Omar Li", "Dana Reyes", "Tomas Berg", "Aiko Tanaka", "Lena Fischer", "Kwame Mensah", "Sofia Rossi",
          "Mateo Silva", "Hana Kim", "Ravi Patel", "Elena Petrova", "Jonas Weber", "Amara Okafor", "Lucas Martin", "Yara Haddad"]
ROLES = ["quality lead", "account director", "logistics coordinator", "finance contact", "sales manager"]
CERTS = ["ISO 9001", "ISO 14001", "SA8000", "FSC", "BRCGS", "OEKO-TEX", "RoHS"]
TERMS = ["net 30", "net 45", "net 60", "2/10 net 30"]
NOTES = [
    "Prefers consolidated shipments; urgent orders go through the supplier portal.",
    "Seasonal capacity constraints in the fourth quarter; book slots early.",
    "Invoices must quote the purchase order number on every line.",
    "Has a second warehouse used for overflow during peak periods.",
    "Samples are shipped free of charge for new product lines.",
    "Requests sixty days' notice before any volume change.",
    "Uses its own carrier for domestic deliveries.",
    "Packaging specifications were updated at the last contract renewal.",
]
# Fields in record order: identity and the fields a step needs first, bulk after.
RECORD_FIELDS = ("supplier_id", "legal_name", "segment", "region", "escalation_code", "status", "account_manager", "onboarded",
                 "payment_terms", "currency", "contacts", "scorecard", "certifications", "open_tickets", "last_audit", "notes")


# --- rules and ratings ------------------------------------------------------------------------------


def _fmt(metric: str, x: float) -> str:
    step = METRICS[metric][4]
    s = f"{x:.1f}" if step < 1 else f"{int(x):,}"
    return f"{s}{UNITS[metric]}"


def rate(rule: dict, metrics: dict) -> str:
    """The rating a segment rule gives one quarter's metrics (the reference solver's core)."""
    good = bad = 0
    for m, (p, q) in ((rule["m1"], rule["t1"]), (rule["m2"], rule["t2"])):
        x, higher = metrics[m], METRICS[m][1]
        good += (x >= p) if higher else (x <= p)
        bad += (x < q) if higher else (x > q)
    if bad:
        return "probation"
    return "preferred" if good == 2 else "approved"


def _rule_text(rule: dict) -> str:
    (m1, (p1, q1)), (m2, (p2, q2)) = (rule["m1"], rule["t1"]), (rule["m2"], rule["t2"])

    def good(m, p):
        return f"its {METRICS[m][0]} ({m}) is {'at least' if METRICS[m][1] else 'at most'} {_fmt(m, p)}"

    def bad(m, q):
        return f"its {METRICS[m][0]} is {'below' if METRICS[m][1] else 'above'} {_fmt(m, q)}"

    return (
        f"Rating rule for {rule['segment']} suppliers. Use the scorecard of the quarter under review only. A {rule['segment']} "
        f"supplier is rated preferred when {good(m1, p1)} and {good(m2, p2)}. It is rated probation when {bad(m1, q1)} or "
        f"{bad(m2, q2)}. In every other case it is rated approved."
    )


def _rule_faq(rule: dict) -> str:
    (m1, (p1, q1)), (m2, (p2, q2)) = (rule["m1"], rule["t1"]), (rule["m2"], rule["t2"])
    ge = lambda m: "≥" if METRICS[m][1] else "≤"  # noqa: E731
    lt = lambda m: "<" if METRICS[m][1] else ">"  # noqa: E731
    return (
        f"Q: How do we rate a supplier in the {rule['segment']} segment? A: Look only at the reviewed quarter. Preferred needs "
        f"{m1} {ge(m1)} {_fmt(m1, p1)} together with {m2} {ge(m2)} {_fmt(m2, p2)}; probation applies as soon as {m1} "
        f"{lt(m1)} {_fmt(m1, q1)} or {m2} {lt(m2)} {_fmt(m2, q2)}; anything else is approved."
    )


def _grid(metric: str, lo: float, hi: float) -> list[float]:
    """Values on the metric's display grid in [lo, hi] (inclusive), so a rendered value is exactly the value rated."""
    step = METRICS[metric][4]
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 1) if step < 1 else int(lo + i * step) for i in range(n + 1)] if n >= 0 else []


def _regions(metric: str, p: float, q: float) -> dict[str, list[float]]:
    """Good, middle and bad values, each one display step clear of the thresholds (no boundary cases)."""
    _, higher, lo, hi, step, _ = METRICS[metric]
    if higher:  # good: x >= p; bad: x < q
        return {"good": _grid(metric, p + step, hi), "middle": _grid(metric, q + step, p - step), "bad": _grid(metric, lo, q - step)}
    return {"good": _grid(metric, lo, p - step), "middle": _grid(metric, p + step, q - step), "bad": _grid(metric, q + step, hi)}


def _sample_metrics(rng: random.Random, rule: dict, target: str) -> dict:
    """One quarter's seven metrics such that `rate(rule, ·) == target`; the other five metrics are free."""
    r1, r2 = _regions(rule["m1"], *rule["t1"]), _regions(rule["m2"], *rule["t2"])
    if target == "preferred":
        a, b = "good", "good"
    elif target == "probation":
        a, b = rng.choice([("bad", rng.choice(["good", "middle"])), (rng.choice(["good", "middle"]), "bad"), ("bad", "bad")])
    else:
        a, b = rng.choice([("middle", rng.choice(["good", "middle"])), ("good", "middle")])
    out = {}
    for m, (_, _, lo, hi, _, _) in METRICS.items():
        out[m] = rng.choice(_grid(m, lo, hi))
    out[rule["m1"]] = rng.choice(r1[a])
    out[rule["m2"]] = rng.choice(r2[b])
    assert rate(rule, out) == target, (rule, out, target)
    return out


# --- world ------------------------------------------------------------------------------------------


def _unique_ids(rng: random.Random, prefix: str, n: int, lo: int, hi: int) -> list[str]:
    return [f"{prefix}-{i}" for i in rng.sample(range(lo, hi), n)]


def _registry(world: World, rng: random.Random) -> None:
    """The KB and record set shared by F1 and F2 (a function of split and seed only)."""
    # Rules: one per segment, two distinct metrics each.
    rules = []
    for seg in SEGMENTS:
        m1, m2 = rng.sample(sorted(METRICS), 2)
        rule = {"segment": seg, "m1": m1, "t1": rng.choice(METRICS[m1][5]), "m2": m2, "t2": rng.choice(METRICS[m2][5]), "fact_id": f"f-R-{seg}"}
        rules.append(rule)
        world.add_fact(rule["fact_id"], _rule_text(rule), "rule")
    world.add_fact("f-DEF-metrics", (
        "Scorecard metric definitions. on_time_pct is the share of purchase orders delivered by the promised date; fill_pct "
        "is the share of ordered units shipped; defect_ppm counts defective units per million received; lead_time_days is "
        "the average days from order to delivery; disputes is the number of invoices disputed in the quarter; audit_score "
        "is the latest site audit result out of 100; return_pct is the share of units returned by customers."
    ), "definition")
    world.add_fact("f-ESC-protocol", (
        "Escalation protocol. When a supplier dispute is escalated, it moves to the supplier that the current supplier's "
        "escalation code routes to in the Escalation Directory. Each escalation is one such move; the escalation code is "
        "listed on the current supplier's record."
    ), "procedure")

    # Suppliers and their records; the escalation graph is one random cycle over the pool.
    sids = _unique_ids(rng, "SUP", POOL, 10000, 99999)
    codes = _unique_ids(rng, "EC", POOL, 1000, 9999)
    names = set()
    suppliers: dict[str, dict] = {}
    for sid, code in zip(sids, codes, strict=True):
        name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}"
        while name in names:
            name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)} {rng.choice(NAME_C)}"
        names.add(name)
        seg, region = rng.choice(SEGMENTS), rng.choice(REGIONS)
        rule = next(r for r in rules if r["segment"] == seg)
        targets = {q: rng.choice(RATINGS) for q in QUARTERS}
        slug = name.split()[0].lower()
        suppliers[sid] = {
            "supplier_id": sid,
            "legal_name": name,
            "segment": seg,
            "region": region,
            "escalation_code": code,
            "status": "active",
            "account_manager": rng.choice(PEOPLE),
            "onboarded": f"{rng.randint(2012, 2023)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "payment_terms": rng.choice(TERMS),
            "currency": CURRENCY[region],
            "contacts": [
                {"name": p, "role": rng.choice(ROLES), "email": f"{p.split()[0].lower()}@{slug}.example", "phone": f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"}
                for p in rng.sample(PEOPLE, 2)
            ],
            "scorecard": {q: _sample_metrics(rng, rule, targets[q]) for q in QUARTERS},
            "certifications": [{"name": c, "expires": f"{rng.randint(2026, 2029)}-{rng.randint(1, 12):02d}-28"} for c in rng.sample(CERTS, 2)],
            "open_tickets": rng.randint(0, 9),
            "last_audit": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "notes": rng.choice(NOTES),
        }
    order = rng.sample(sids, len(sids))
    routes = {suppliers[order[i]]["escalation_code"]: order[(i + 1) % len(order)] for i in range(len(order))}
    for code, target in routes.items():
        world.add_fact(f"f-{code}", f"Escalation code {code} routes to supplier {target} ({suppliers[target]['legal_name']}).", "directory")

    # Redundancy: about half of the rules and directory entries get a paraphrased second rendering.
    twice = {r["fact_id"] for r in rng.sample(rules, round(REDUNDANCY_RATE * len(rules)))}
    twice |= {f"f-{c}" for c in rng.sample(sorted(routes), round(REDUNDANCY_RATE * len(routes)))}
    world.entities.update({
        "suppliers": suppliers,
        "rules": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rules],
        "routes": routes,
        "redundancy": {f: (2 if f in twice else 1) for f in world.facts},
    })

    retired = _unique_ids(rng, "EC", 12, 1000, 9999)
    retired = [c for c in retired if c not in routes]
    world.documents += [
        Document("scorecard-rules", "Supplier Scorecard Rules", [
            Paragraph("This handbook sets out how supplier performance is rated at each quarterly review. Ratings decide "
                      "sourcing priority for the following quarter and are shared with the supplier's account manager.", []),
            Paragraph(world.facts["f-DEF-metrics"].text, ["f-DEF-metrics"]),
            *[Paragraph(world.facts[r["fact_id"]].text, [r["fact_id"]]) for r in rules],
            Paragraph("Ratings are published to suppliers within ten business days of the review. A supplier may appeal a "
                      "rating once per quarter through its account manager; appeals do not change the rating rules.", []),
        ]),
        Document("review-faq", "Supplier Review FAQ", [
            Paragraph("Q: Who runs the quarterly review? A: The sourcing team, with input from quality and finance.", []),
            *[Paragraph(_rule_faq(r), [r["fact_id"]]) for r in rules if r["fact_id"] in twice],
            Paragraph("Q: Do ratings carry over between quarters? A: No. Every review starts from that quarter's scorecard.", []),
            Paragraph("Q: Can a supplier be rated before its first full quarter? A: New suppliers are reviewed after one "
                      "complete quarter of deliveries.", []),
        ]),
        Document("escalation-directory", "Escalation Directory", [
            Paragraph(world.facts["f-ESC-protocol"].text, ["f-ESC-protocol"]),
            *_interleave(rng, [Paragraph(world.facts[f"f-{c}"].text, [f"f-{c}"]) for c in sorted(routes)],
                         [Paragraph(f"Escalation code {c} was retired in 2023 and is no longer assigned; disputes that "
                                    f"still carry it go to the sourcing review desk.", []) for c in retired]),
        ]),
        Document("escalation-contacts", "Regional Escalation Contacts", [
            Paragraph("Escalated disputes are worked by the receiving supplier's account team during local business hours.", []),
            *[Paragraph(f"Disputes carrying code {c} are handled by {suppliers[t]['legal_name']} ({t}).", [f"f-{c}"])
              for c, t in sorted(routes.items()) if f"f-{c}" in twice],
        ]),
        Document("program-overview", "Supplier Program Overview", [
            Paragraph("The supplier program covers every vendor that ships goods to our distribution centers. Each "
                      "supplier has one account manager, a segment and a home region.", []),
            Paragraph("Supplier records are kept in the supplier registry and can be looked up by supplier ID. Records list "
                      "contacts, payment terms, certifications and quarterly scorecards.", []),
            Paragraph("Sustainability reporting is collected annually and does not affect quarterly ratings.", []),
        ]),
    ]


def _interleave(rng: random.Random, main: list[Paragraph], extra: list[Paragraph]) -> list[Paragraph]:
    out = list(main)
    for p in extra:
        out.insert(rng.randint(0, len(out)), p)
    return out


def render_record(record: dict) -> str:
    """The record as `lookup_supplier` returns it: compact JSON in RECORD_FIELDS order."""
    return json.dumps({k: record[k] for k in RECORD_FIELDS}, separators=(",", ":"))


def turn_cap(family: str, level: str) -> int:
    """Turns the agent loop allows: one lookup per supplier (F1) or a lookup and a pull per hop (F2), plus slack."""
    n = int(level)
    return n + 12 if family == "F1" else 2 * n + 8


# --- tasks ------------------------------------------------------------------------------------------


F1_PROMPTS = (
    "Quarterly supplier review ({q}): rate each of these {n} suppliers under the supplier scorecard rules: {ids}. "
    "Look up each supplier's record, then submit every rating at once with submit_ratings.",
    "For the {q} scorecard review, determine the rating company rules assign to each of the following {n} suppliers: "
    "{ids}. Submit all {n} ratings together with submit_ratings.",
    "Please rate suppliers {ids} for the {q} review, following the scorecard rules, and record all {n} ratings with "
    "submit_ratings.",
)
F2_PROMPTS = (
    "Dispute {d} was opened with supplier {start} and has since been escalated {k} times under the escalation protocol. "
    "Which supplier holds the dispute now? Submit the final supplier ID and the suppliers it passed through, in order, "
    "with submit_chain.",
    "Supplier {start} received dispute {d}, which was then escalated {k} times following the escalation protocol. "
    "Identify the supplier currently responsible and the full escalation path, and submit both with submit_chain.",
    "Trace dispute {d}: it started with supplier {start} and moved {k} escalation steps under the protocol. Report "
    "where it ended up, plus every supplier along the way, using submit_chain.",
)


def _rule_of(world: World, segment: str) -> dict:
    return next(r for r in world.entities["rules"] if r["segment"] == segment)


def _f1_tasks(world: World, rng: random.Random, n: int, n_tasks: int) -> None:
    suppliers = world.entities["suppliers"]
    for t in range(n_tasks):
        q = rng.choice(QUARTERS)
        ids = sorted(rng.sample(sorted(suppliers), n))
        ids = rng.sample(ids, len(ids))  # listed in random order
        subtasks = []
        for sid in ids:
            rec = suppliers[sid]
            rule = _rule_of(world, rec["segment"])
            subtasks.append({
                "id": sid,
                "question": f"Under the scorecard rules, what rating does supplier {sid} get for the {q} review?",
                "gold": rate(rule, rec["scorecard"][q]),
                "gold_fact_ids": [rule["fact_id"]],
            })
        prompts = [p.format(q=q, n=n, ids=", ".join(ids)) for p in F1_PROMPTS]
        i = rng.randrange(len(prompts))
        world.tasks.append(TaskItem(
            id=f"{world.id}-t{t:03d}",
            world_id=world.id,
            family="F1",
            level=world.level,
            prompt=prompts[i],
            gold={"ratings": {s["id"]: s["gold"] for s in subtasks}},
            gold_fact_ids=list(dict.fromkeys(f for s in subtasks for f in s["gold_fact_ids"])),
            answer_tool="submit_ratings",
            tags={"quarter": q, "n": n, "suppliers": ids, "subtasks": subtasks, "paraphrases": prompts[:i] + prompts[i + 1:]},
        ))


def _f2_tasks(world: World, rng: random.Random, k: int, n_tasks: int) -> None:
    suppliers, routes = world.entities["suppliers"], world.entities["routes"]
    for t in range(n_tasks):
        start = rng.choice(sorted(suppliers))
        chain, codes, cur = [], [], start
        for _ in range(k):
            code = suppliers[cur]["escalation_code"]
            cur = routes[code]
            codes.append(code)
            chain.append(cur)
        d = f"D-{rng.randint(100000, 999999)}"
        prompts = [p.format(d=d, start=start, k=k) for p in F2_PROMPTS]
        i = rng.randrange(len(prompts))
        # Fault (applied only when enabled): on hop j, the first lookup of the hop's source reports a valid code
        # whose target lies off the true chain.
        j = rng.randint(1, k)
        source = start if j == 1 else chain[j - 2]
        on_chain = {start, *chain}
        wrong = rng.choice(sorted(c for c, tgt in routes.items() if tgt not in on_chain and c != codes[j - 1]))
        world.tasks.append(TaskItem(
            id=f"{world.id}-t{t:03d}",
            world_id=world.id,
            family="F2",
            level=world.level,
            prompt=prompts[i],
            gold={"final": chain[-1], "chain": chain},
            gold_fact_ids=["f-ESC-protocol", *(f"f-{c}" for c in codes)],
            answer_tool="submit_chain",
            tags={
                "start": start, "k": k, "codes": codes, "dispute": d, "paraphrases": prompts[:i] + prompts[i + 1:],
                "fault": {"kind": "tool_result", "tool": "lookup_supplier", "key": source, "field": "escalation_code",
                          "value": wrong, "true_value": codes[j - 1], "hop": j, "fact_id": f"f-{codes[j - 1]}", "occurrence": 1},
            },
        ))


def generate(family: str, level: str, split: str, seed: int, n_tasks: int) -> World:
    assert family in LEVELS and level in LEVELS[family], (family, level)
    world = World(id=f"{family}-{level}-{split}-s{seed}", family=family, level=level, seed=seed, split=split)
    _registry(world, random.Random(f"REG|{split}|{seed}"))  # family- and level-independent: paired worlds
    rng = random.Random(f"{family}|{level}|{split}|{seed}")
    (_f1_tasks if family == "F1" else _f2_tasks)(world, rng, int(level), n_tasks)
    return world


def solve(world: World, task: TaskItem) -> dict:
    """Reference solver from the structured spec: rules and records (F1), records and routes (F2)."""
    suppliers = world.entities["suppliers"]
    if task.family == "F1":
        q = task.tags["quarter"]
        return {"ratings": {sid: rate(_rule_of(world, suppliers[sid]["segment"]), suppliers[sid]["scorecard"][q]) for sid in task.tags["suppliers"]}}
    chain, cur = [], task.tags["start"]
    for _ in range(task.tags["k"]):
        cur = world.entities["routes"][suppliers[cur]["escalation_code"]]
        chain.append(cur)
    return {"final": chain[-1], "chain": chain}
