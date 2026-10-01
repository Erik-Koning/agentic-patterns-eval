"""F7 policy-compliance worlds.

Each base policy covers one (domain, region, amount band) cell and fixes an outcome
(action, approver, deadline, required document). In the relational variant some
policies have an exception, recorded in a separate register, that overrides the
outcome for one customer tier, so answering correctly means finding both the
policy and any exception that amends it. KB size is the level: 10, 100 or 1000
base policies. Customer region and tier are only available via `lookup_customer`,
which makes the task multi-step.

`exception_style` controls how an exception refers to its policy (EXPERIMENT_AUDIT B3):
"descriptive" restates the policy's domain, region and band, so any retriever can find it;
"id_only" names only the policy ID, so finding it requires following the reference.
"messy" drops all IDs and varies the wording (synonyms, templates), and exceptions point at
their policy only by description: a realistic, harder test of graph authoring and
extraction. All styles share one random stream for content (identical policies, tasks and
gold answers); messy wording draws from its own stream. So the effect of the rendering is a
paired comparison.
"""

import random

from .spec import Document, Exception_, Paragraph, Policy, TaskItem, World

DOMAINS = [
    ("refunds", "refund"),
    ("returns", "return"),
    ("shipping_delays", "shipping-delay compensation"),
    ("lost_parcels", "lost-parcel"),
    ("account_lockouts", "account-unlock"),
    ("data_deletion", "data-deletion"),
    ("billing_disputes", "billing-dispute"),
    ("chargebacks", "chargeback"),
    ("warranty_claims", "warranty"),
    ("price_adjustments", "price-adjustment"),
    ("subscription_cancellations", "subscription-cancellation"),
    ("loyalty_points", "loyalty-points"),
    ("gift_cards", "gift-card"),
    ("damaged_goods", "damaged-goods"),
    ("order_modifications", "order-modification"),
    ("address_changes", "address-change"),
    ("fraud_reviews", "fraud-review"),
    ("tax_exemptions", "tax-exemption"),
    ("bulk_orders", "bulk-order"),
    ("service_credits", "service-credit"),
]
REGIONS = ["NA", "EU", "UK", "APAC", "LATAM"]
BANDS = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 1000), (1000, 2500), (2500, 5000), (5000, 10000), (10000, 25000), (25000, 100000)]
TIERS = ["basic", "silver", "gold", "platinum"]
ACTIONS = ["approve", "deny", "escalate"]
APPROVERS = ["team_lead", "tier2_supervisor", "compliance_officer", "regional_manager"]
DOCUMENTS = ["none", "receipt", "photo_id", "police_report", "bank_statement", "delivery_photo"]
LEVELS = {"10": (2, 1, 5), "100": (5, 2, 10), "1000": (20, 5, 10)}  # (domains, regions, bands)
EXCEPTION_RATE = 0.3
EXCEPTION_STYLES = {"descriptive": "desc", "id_only": "idonly", "messy": "messy"}
SYNONYMS = {
    "refund": ["money-back", "reimbursement"],
    "return": ["send-back", "return-of-goods"],
    "shipping-delay compensation": ["late-delivery compensation", "delay credit"],
    "lost-parcel": ["missing-package", "undelivered-parcel"],
    "account-unlock": ["account-access restoration", "login-lockout"],
    "data-deletion": ["personal-data erasure", "account-data removal"],
    "billing-dispute": ["invoice-dispute", "charge-query"],
    "chargeback": ["card-reversal", "payment-reversal"],
    "warranty": ["guarantee", "product-warranty"],
    "price-adjustment": ["price-match", "price-correction"],
    "subscription-cancellation": ["plan-cancellation", "membership-termination"],
    "loyalty-points": ["rewards-points", "points-balance"],
    "gift-card": ["gift-voucher", "store-card"],
    "damaged-goods": ["broken-item", "damaged-delivery"],
    "order-modification": ["order-change", "order-amendment"],
    "address-change": ["delivery-address update", "shipping-address change"],
    "fraud-review": ["suspected-fraud check", "fraud investigation"],
    "tax-exemption": ["tax-relief", "VAT-exemption"],
    "bulk-order": ["wholesale-order", "large-quantity order"],
    "service-credit": ["service-outage credit", "downtime credit"],
}

_APPROVER_TEXT = {
    "team_lead": "team lead",
    "tier2_supervisor": "Tier-2 supervisor",
    "compliance_officer": "compliance officer",
    "regional_manager": "regional manager",
}
_DOC_TEXT = {
    "receipt": "a receipt",
    "photo_id": "a government photo ID",
    "police_report": "a police report",
    "bank_statement": "a bank statement",
    "delivery_photo": "a photo of the delivered package",
}


def _outcome(rng: random.Random) -> dict:
    action = rng.choice(ACTIONS)
    return {
        "action": action,
        "approver": rng.choice(APPROVERS) if action == "escalate" else "none",
        "deadline_days": rng.randint(1, 10),
        "document": rng.choice(DOCUMENTS),
    }


def outcome_text(o: dict) -> str:
    if o["action"] == "escalate":
        act = f"escalate the request to the {_APPROVER_TEXT[o['approver']]}"
    else:
        act = f"{o['action']} the request"
    doc = "no supporting document is required" if o["document"] == "none" else f"the customer must provide {_DOC_TEXT[o['document']]}"
    days = f"{o['deadline_days']} business day{'s' if o['deadline_days'] != 1 else ''}"
    return f"{act}, respond within {days}, and {doc}"


def _money(x: int) -> str:
    return f"${x:,}"


def _messy_policy(mrng: random.Random, phrase: str, region: str, lo: int, hi: int, outcome: str) -> str:
    name = mrng.choice([phrase, *SYNONYMS[phrase]])
    return mrng.choice(
        [
            f"For {name} requests from {region} customers between {_money(lo)} and {_money(hi)} (upper bound excluded), {outcome}.",
            f"In {region}, when a customer asks for {name} and the amount is {_money(lo)} up to but not including {_money(hi)}, staff must {outcome}.",
            f"{name.capitalize()} cases ({region}; {_money(lo)} or more, below {_money(hi)}): {outcome}.",
        ]
    )


def _messy_exception(mrng: random.Random, phrase: str, region: str, lo: int, hi: int, tier: str, outcome: str) -> str:
    name = mrng.choice([phrase, *SYNONYMS[phrase]])
    return mrng.choice(
        [
            f"Note for {tier}-status customers: the {name} rule for {region} covering {_money(lo)} to under {_money(hi)} is replaced; instead, {outcome}.",
            f"Amendment: where a {tier} customer in {region} raises a {name} request of at least {_money(lo)} and under {_money(hi)}, the usual handling does not apply; {outcome}.",
        ]
    )


def generate(level: str, relational: bool, split: str, seed: int, n_tasks: int, exception_style: str = "descriptive") -> World:
    rng = random.Random(f"F7|{level}|{relational}|{split}|{seed}")  # style-independent: paired worlds
    mrng = random.Random(f"F7-messy|{level}|{relational}|{split}|{seed}")  # wording only
    n_dom, n_reg, n_band = LEVELS[level]
    domains = sorted(rng.sample(DOMAINS, n_dom))
    regions = sorted(rng.sample(REGIONS, n_reg))
    bands = sorted(rng.sample(BANDS, n_band))
    world = World(
        id=f"F7-{level}-{'rel-' + EXCEPTION_STYLES[exception_style] if relational else 'ind'}-{split}-s{seed}",
        family="F7",
        level=level,
        seed=seed,
        split=split,
        entities={"relational": relational, "exception_style": exception_style if relational else None},
    )

    # Base policies, one per cell; adjacent bands never share an outcome.
    policy_ids = rng.sample(range(1000, 10000), len(domains) * len(regions) * len(bands))
    i = 0
    for dom, dom_phrase in domains:
        for region in regions:
            prev = None
            for lo, hi in bands:
                o = _outcome(rng)
                while o == prev:
                    o = _outcome(rng)
                prev = o
                pid = f"P-{policy_ids[i]}"
                i += 1
                if exception_style == "messy":
                    text = _messy_policy(mrng, dom_phrase, region, lo, hi, outcome_text(o))
                else:
                    text = (
                        f"Policy {pid} ({dom.replace('_', ' ').title()}, {region} region). For a {dom_phrase} request "
                        f"from a customer in the {region} region with an amount of at least {_money(lo)} and under "
                        f"{_money(hi)}, the agent must {outcome_text(o)}."
                    )
                world.add_fact(f"f-{pid}", text, "policy")
                world.policies.append(Policy(pid, dom, region, lo, hi, o, f"f-{pid}"))

    if relational:
        n_exc = max(1, round(EXCEPTION_RATE * len(world.policies)))
        exc_ids = rng.sample(range(100, 1000), n_exc)
        for j, p in enumerate(rng.sample(world.policies, n_exc)):
            tier = rng.choice(["gold", "platinum"])
            o = _outcome(rng)
            while o == p.outcome:
                o = _outcome(rng)
            xid = f"X-{exc_ids[j]}"
            scope = (
                f": {dict(DOMAINS)[p.domain]} requests from customers in the {p.region} region with an amount of at "
                f"least {_money(p.lo)} and under {_money(p.hi)}"
                if exception_style == "descriptive"
                else ""
            )
            if exception_style == "messy":
                text = _messy_exception(mrng, dict(DOMAINS)[p.domain], p.region, p.lo, p.hi, tier, outcome_text(o))
            else:
                text = (
                    f"Exception {xid} (amends Policy {p.id}{scope}). When the customer holds {tier} status, Policy {p.id} "
                    f"does not apply as written: the agent must instead {outcome_text(o)}."
                )
            world.add_fact(f"f-{xid}", text, "exception")
            world.exceptions.append(Exception_(xid, p.id, tier, o, f"f-{xid}"))

    # Documents: one handbook per domain; exceptions live in a separate register.
    for dom, _ in domains:
        paras = [Paragraph(world.facts[p.fact_id].text, [p.fact_id]) for p in world.policies if p.domain == dom]
        world.documents.append(
            Document(f"handbook-{dom}", f"Customer Policy Handbook: {dom.replace('_', ' ').title()}", paras)
        )
    if world.exceptions:
        paras = [Paragraph(world.facts[x.fact_id].text, [x.fact_id]) for x in sorted(world.exceptions, key=lambda x: x.id)]
        world.documents.append(Document("exceptions-register", "Policy Exceptions Register", paras))

    _make_tasks(world, rng, domains, n_tasks, relational)
    return world


def _make_tasks(world: World, rng: random.Random, domains, n_tasks: int, relational: bool) -> None:
    phrase = dict(domains)
    exc_by_policy = {x.policy_id: x for x in world.exceptions}
    with_exc = [p for p in world.policies if p.id in exc_by_policy]
    without_exc = [p for p in world.policies if p.id not in exc_by_policy]
    used_ids: set[str] = set()
    for t in range(n_tasks):
        if relational and with_exc:
            case = rng.choices(["exception_applies", "exception_not_applicable", "no_exception"], [0.5, 0.25, 0.25])[0]
        else:
            case = "no_exception"
        pool = with_exc if case != "no_exception" else (without_exc or world.policies)
        p = rng.choice(pool)
        exc = exc_by_policy.get(p.id)
        if case == "exception_applies":
            tier = exc.tier
        elif case == "exception_not_applicable":
            tier = rng.choice([t for t in TIERS if t != exc.tier])
        else:
            tier = rng.choice(TIERS)
        amount = rng.randint(p.lo, p.hi - 1)
        cid = f"CU-{rng.randint(10000, 99999)}"
        while cid in used_ids:
            cid = f"CU-{rng.randint(10000, 99999)}"
        used_ids.add(cid)
        gold = exc.outcome if case == "exception_applies" else p.outcome
        world.tasks.append(
            TaskItem(
                id=f"{world.id}-t{t:03d}",
                world_id=world.id,
                family="F7",
                level=world.level,
                prompt=(
                    f"Case C-{rng.randint(100000, 999999)}: customer {cid} has submitted a {phrase[p.domain]} request "
                    f"for {_money(amount)}. Look up the customer, determine exactly how company policy requires this "
                    f"request to be handled, and submit your decision with submit_decision."
                ),
                gold=dict(gold),
                gold_fact_ids=[p.fact_id] + ([exc.fact_id] if exc else []),
                answer_tool="submit_decision",
                setup={"customers": {cid: {"region": p.region, "tier": tier}}},
                tags={"case": case, "domain": p.domain, "amount": amount, "customer_id": cid, "policy": p.id},
            )
        )


def solve(world: World, task: TaskItem) -> dict:
    """Reference solver from the structured spec; used to validate generated gold answers."""
    cust = task.setup["customers"][task.tags["customer_id"]]
    amount = task.tags["amount"]
    matches = [
        p for p in world.policies
        if p.domain == task.tags["domain"] and p.region == cust["region"] and p.lo <= amount < p.hi
    ]
    assert len(matches) == 1, f"{task.id}: {len(matches)} applicable policies"
    p = matches[0]
    for x in world.exceptions:
        if x.policy_id == p.id and x.tier == cust["tier"]:
            return x.outcome
    return p.outcome
