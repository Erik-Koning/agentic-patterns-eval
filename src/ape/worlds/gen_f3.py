"""F3 tool-load worlds.

Twenty service domains each own three mutating tools. A world binds 5, 20 or 60
of them (the level). Standard operating procedures say which tools to call, with
which region-specific constants, for each (domain, region) situation. A task names
a situation and an order; the order's region is only available via `order_lookup`.
Success means the mutating calls made equal the procedure's steps exactly.
"""

import random

from .spec import Document, Paragraph, Procedure, TaskItem, Tool, World

DOMAINS = [
    ("lost_parcel", "the customer reports the parcel was lost in transit"),
    ("damaged_item", "the item arrived damaged"),
    ("late_delivery", "the delivery is more than five days late"),
    ("wrong_item", "the customer received the wrong item"),
    ("missing_item", "an item is missing from the package"),
    ("address_error", "the shipping address on the order is wrong"),
    ("customs_hold", "the parcel is held at customs"),
    ("return_pickup", "the customer wants a return pickup scheduled"),
    ("refund_delay", "the customer's refund has not arrived"),
    ("duplicate_charge", "the customer was charged twice"),
    ("warranty_repair", "the product needs a warranty repair"),
    ("exchange_request", "the customer wants to exchange the item for another size"),
    ("backorder", "an item on the order is backordered"),
    ("preorder_change", "the customer wants to change a pre-order"),
    ("gift_wrap_issue", "the gift wrapping was missing or wrong"),
    ("installation_booking", "the customer needs an installation appointment"),
    ("recall_notice", "the product is subject to a safety recall"),
    ("subscription_skip", "the customer wants to skip the next subscription delivery"),
    ("loyalty_adjustment", "loyalty points were not credited for the order"),
    ("account_merge", "the order was placed under a duplicate account"),
]
VERBS = [
    ("reissue", "carrier", "Re-ship the items of an order through the given carrier service."),
    ("hold", "hold_reason", "Place an order on hold with a hold-reason code."),
    ("escalate", "queue", "Escalate the order case to a specialist queue."),
    ("credit", "credit_code", "Apply a goodwill credit to the customer's account for the order."),
    ("notify", "template", "Send the customer a notification using a message template."),
    ("reroute", "depot_code", "Reroute an in-transit order through a depot."),
]
CODE_PREFIX = {"carrier": "CAR", "hold_reason": "HR", "queue": "Q", "credit_code": "CR", "template": "T", "depot_code": "DEP"}
REGIONS = ["NA", "EU", "UK", "APAC", "LATAM"]
LEVELS = {"5": 5, "20": 20, "60": 60}
ALWAYS_ON = ["order_lookup", "finish"]


def _domain_tools(d_index: int, domain: str) -> list[tuple[str, str, str]]:
    verbs = [VERBS[(d_index + k * 2) % len(VERBS)] for k in range(3)]
    return [(f"{domain}_{v}", p, desc) for v, p, desc in verbs]


def generate(level: str, split: str, seed: int, n_tasks: int) -> World:
    rng = random.Random(f"F3|{level}|{split}|{seed}")
    n_tools = LEVELS[level]
    world = World(id=f"F3-{level}-{split}-s{seed}", family="F3", level=level, seed=seed, split=split)

    all_domains = list(enumerate(DOMAINS))
    rng.shuffle(all_domains)
    task_domain_count = 1 if n_tools <= 5 else 3
    chosen: list[tuple[int, tuple[str, str]]] = []
    tools: list[tuple[str, str, str, str]] = []  # (name, param, desc, domain)
    for d_index, (dom, _) in all_domains:
        if len(tools) >= n_tools:
            break
        dt = [(n, p, desc, dom) for n, p, desc in _domain_tools(d_index, dom)]
        chosen.append((d_index, (dom, _)))
        tools.extend(dt[: n_tools - len(tools)])
    task_domains = [dom for _, (dom, _) in chosen[:task_domain_count]]

    for name, param, desc, dom in sorted(tools):
        text = (
            f"Tool {name} (domain: {dom.replace('_', ' ')}). {desc} Parameters: order_id (the order ID) and "
            f"{param} (a {param.replace('_', ' ')} code)."
        )
        world.add_fact(f"f-tool-{name}", text, "tool")
        world.tools.append(Tool(name, dom, desc, {"order_id": "The order ID.", param: f"The {param.replace('_', ' ')} code."}, True, f"f-tool-{name}"))

    by_domain: dict[str, list[Tool]] = {}
    for t in world.tools:
        by_domain.setdefault(t.domain, []).append(t)
    for dom, dom_tools in sorted(by_domain.items()):
        if len(dom_tools) < 2:
            continue
        for region in REGIONS:
            step_tools = sorted(rng.sample(dom_tools, 2), key=lambda t: t.name)
            steps = []
            for t in step_tools:
                param = next(k for k in t.params if k != "order_id")
                steps.append({"tool": t.name, "args": {"order_id": "{order_id}", param: f"{CODE_PREFIX[param]}-{rng.randint(100, 999)}"}})
            pid = f"SOP-{dom}-{region}"
            described = "; then ".join(
                f"call {s['tool']} with " + ", ".join(f"{k}={v}" for k, v in s["args"].items() if k != "order_id") for s in steps
            )
            text = (
                f"Procedure {pid}. When {dict(DOMAINS)[dom]} and the order ships to the {region} region: {described}. "
                f"Pass the order ID as order_id. Do not call any other mutating tool."
            )
            world.add_fact(f"f-{pid}", text, "procedure")
            world.procedures.append(Procedure(pid, dom, f"{dom}/{region}", steps, f"f-{pid}"))

    world.documents.append(
        Document("tool-catalog", "Operations Tool Catalog", [Paragraph(world.facts[t.fact_id].text, [t.fact_id]) for t in world.tools])
    )
    for dom in sorted({p.domain for p in world.procedures}):
        paras = [Paragraph(world.facts[p.fact_id].text, [p.fact_id]) for p in world.procedures if p.domain == dom]
        world.documents.append(Document(f"sop-{dom}", f"Standard Operating Procedures: {dom.replace('_', ' ').title()}", paras))

    procs = [p for p in world.procedures if p.domain in task_domains]
    used: set[str] = set()
    for t in range(n_tasks):
        proc = rng.choice(procs)
        region = proc.situation.split("/")[1]
        oid = f"O-{rng.randint(100000, 999999)}"
        while oid in used:
            oid = f"O-{rng.randint(100000, 999999)}"
        used.add(oid)
        expected = [{"tool": s["tool"], "args": {k: (oid if v == "{order_id}" else v) for k, v in s["args"].items()}} for s in proc.steps]
        world.tasks.append(
            TaskItem(
                id=f"{world.id}-t{t:03d}",
                world_id=world.id,
                family="F3",
                level=level,
                prompt=(
                    f"Ticket K-{rng.randint(10000, 99999)}: {dict(DOMAINS)[proc.domain]} (order {oid}). Resolve it "
                    f"exactly as the standard operating procedure requires, then call finish."
                ),
                gold={"calls": expected},
                gold_fact_ids=[proc.fact_id] + [f"f-tool-{s['tool']}" for s in proc.steps],
                answer_tool="finish",
                setup={"orders": {oid: {"region": region, "status": "open"}}},
                tags={"procedure": proc.id, "domain": proc.domain, "order_id": oid},
            )
        )
    return world


def solve(world: World, task: TaskItem) -> dict:
    oid = task.tags["order_id"]
    region = task.setup["orders"][oid]["region"]
    proc = next(p for p in world.procedures if p.situation == f"{task.tags['domain']}/{region}")
    return {"calls": [{"tool": s["tool"], "args": {k: (oid if v == "{order_id}" else v) for k, v in s["args"].items()}} for s in proc.steps]}
