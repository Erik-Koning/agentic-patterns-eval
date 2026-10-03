"""F8 "shift" sessions: one long-horizon session of N cases for Study G (CONTEXT_MANAGEMENT_AUDIT §4.1).

One world is one session. Its knowledge base is an F7 policy world plus an F3 procedure world of the same
seed (default sizes F7-10 and F3-5, the capability anchor's cells), merged, so the corpus stays small enough
for the system prompt and context pressure comes from the session itself:

- **Context pressure.** `lookup_customer` and `order_lookup` return realistic, bulky files (profile, order
  history, shipment scans, support notes, distractor fields such as past tiers and the billing country) of
  about `output_tokens` each (0.6x-1.4x, deterministic per ID). The reference trajectory records the item at
  which the full history first exceeds the nominal window W (`w_crossing_item`).
- **Mid-session memos.** Announced in a case's message, in force from that case until a withdrawal message.
  They override the handbook, exceptions and earlier memos (escalate above an amount, set a deadline, require
  a document, swap a procedure code).
- **Cross-item dependencies.**
  - Quota: at most `QUOTA` approvals per customer per shift; a further approval goes to the team lead.
  - Follow-ups: a follow-up names only the earlier case ID, so it needs the original request from memory.
  - Deferred work: escalations and cases awaiting a follow-up are listed in the end-of-shift report.
- **End-of-shift report.** Every case's disposition, the escalations pending recheck, the open follow-ups.

The level is N (items per session). Knobs: `memo_density` (memos per item), `dependency_density` (share of
follow-up and repeat-customer items), `output_tokens` (tool-file size). Item kinds are shuffled, so position
is not confounded with case type; dependencies only constrain order where they must (a follow-up comes
after its original, a repeat after the customer's first case).

Gold per item is computed here (policy -> exception -> memos in announcement order -> quota). `solve_session`
replays the session independently and must agree (tests). `state_at(k)` is the ground-truth agent state
after k items; `render_oracle_state` is what the O-state arm sees.
"""

import json
import math
import random
from collections import Counter

from ..llm.tokens import count_tokens
from . import gen_f3, gen_f7
from .gen_f7 import DOCUMENTS, TIERS
from .spec import TaskItem, World

WINDOW = 32_000  # nominal window W (config/run_plan.yaml study_g.window; D-021)
QUOTA = 2  # approvals per customer per shift
CHECKPOINTS = (5, 15, 30, 45, 60)  # forked state probes after these items (those <= N)
KB_LEVELS = ("10", "5")  # (F7 level, F3 level): the capability anchor's cells
DEFAULT_KNOBS = {"memo_density": 0.125, "dependency_density": 0.3, "output_tokens": 1000}
TICKET_SHARE = 0.3  # of the non-dependency items
AWAIT_RATE = 0.35  # fresh policy cases that wait for a customer follow-up
WITHDRAW_RATE = 0.4
MEMO_TYPES = (("escalate", 0.3), ("deadline", 0.25), ("document", 0.25), ("procedure_code", 0.2))
PROBE_CATEGORIES = ("completed", "pending", "memos_in_force", "open_followups", "escalated")
BASE_SYSTEM = (
    "You are an operations assistant at a retail company, working one support shift. Company knowledge is provided "
    "below. Follow policies, procedures and the shift rules exactly as written, and use the tools to gather facts you "
    "are not given."
)
REPORT_DISPOSITIONS = ("approve", "deny", "escalate", "resolved")

SHIFT_RULES = f"""## Shift rules
- You work through a queue of cases, one at a time. Each case arrives as a message; finish it before the next one arrives.
- Policy cases: decide exactly as company policy requires and call submit_decision with the case ID.
- Tickets: make exactly the tool calls the standard operating procedure requires for the ticket's order, then call finish with the case ID.
- Memos announced during the shift override the handbook, any exception and earlier memos, from the case in which they are announced until they are withdrawn. When two memos set the same thing, the later one wins.
- Quota: at most {QUOTA} approvals per customer per shift. If a decision would be a further approval for a customer who already has {QUOTA} approvals this shift, escalate it to the team lead instead, keeping the deadline and document.
- Follow-ups: a follow-up re-decides an earlier case's original request under the policies and memos in force now. The quota does not apply to follow-ups and they do not count toward it.
- Some cases wait for a follow-up from the customer; they stay open until their follow-up arrives.
- At the end of the shift you submit a report with submit_shift_report."""

REPORT_REQUEST = (
    "End of shift. Call submit_shift_report with `report` set to a JSON object with three keys: "
    "\"dispositions\" (every case ID handled this shift mapped to approve, deny or escalate for decisions, "
    "or resolved for tickets), \"pending_recheck\" (every case you escalated this shift) and "
    "\"open_followups\" (cases still waiting for a follow-up)."
)

_FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hiro", "Ines", "Jamal", "Kira", "Luis", "Mona", "Nils", "Olu", "Pia", "Quinn", "Rosa", "Sven", "Tala"]
_LAST = ["Abara", "Berg", "Costa", "Dimitrov", "Evans", "Fujita", "Garcia", "Haddad", "Ivanova", "Jensen", "Kowalski", "Lindqvist", "Moreau", "Nakamura", "Okafor", "Petrov"]
_PRODUCTS = [
    ("SKU-1042", "wireless earbuds"), ("SKU-2210", "espresso grinder"), ("SKU-3307", "trail running shoes"), ("SKU-4471", "smart thermostat"),
    ("SKU-5128", "ceramic cookware set"), ("SKU-6093", "mechanical keyboard"), ("SKU-7316", "down winter jacket"), ("SKU-8840", "robot vacuum"),
    ("SKU-9025", "yoga mat"), ("SKU-1187", "4K monitor"), ("SKU-2264", "standing desk"), ("SKU-3391", "air purifier"),
]
_COUNTRIES = ["Canada", "Germany", "Japan", "Brazil", "Australia", "United Kingdom", "Mexico", "France", "India", "Kenya"]
_CITIES = ["Porto", "Leeds", "Osaka", "Denver", "Lyon", "Pune", "Quito", "Perth", "Gdansk", "Accra"]
_NOTES = [
    "Customer asked about gift wrapping options for the holidays.",
    "Called to update the phone number; verified by security question.",
    "Prefers e-mail contact; do not call before 10am local time.",
    "Previously reported a courier left a parcel at the wrong door; resolved.",
    "Asked whether loyalty points expire; explained the 24-month rule.",
    "Requested an invoice copy for an older order; sent by e-mail.",
    "Mentioned moving house next month; no address change requested yet.",
    "Complimented the support team on a quick resolution.",
]
_SCANS = ["label created", "picked up by carrier", "arrived at sort facility", "departed sort facility", "in transit", "customs processing", "out for delivery", "delivery attempted", "held at depot"]


def _money(x: int) -> str:
    return f"${x:,}"


def _phrase(world: World, domain: str) -> str:
    return dict(gen_f7.DOMAINS)[domain]


def _situation(domain: str) -> str:
    return dict(gen_f3.DOMAINS)[domain]


# --- Decisions: policy -> exception -> memos -> quota ------------------------------------------------------


def base_decision(world: World, customer: dict, domain: str, amount: int) -> dict:
    """The handbook outcome, with an exception for the customer's tier applied (the F7 rule)."""
    matches = [p for p in world.policies if p.domain == domain and p.region == customer["region"] and p.lo <= amount < p.hi]
    assert len(matches) == 1, f"{len(matches)} policies for {domain}/{customer['region']}/{amount}"
    p = matches[0]
    for x in world.exceptions:
        if x.policy_id == p.id and x.tier == customer["tier"]:
            return dict(x.outcome)
    return dict(p.outcome)


def memo_applies(memo: dict, *, domain: str | None = None, amount: int | None = None, tier: str | None = None, procedure: str | None = None) -> bool:
    t = memo["type"]
    if t == "procedure_code":
        return procedure == memo["procedure"]
    if domain != memo["domain"]:
        return False
    if t == "escalate":
        return amount is not None and amount >= memo["threshold"]
    if t == "document":
        return tier == memo["tier"]
    return t == "deadline"


def apply_memos(decision: dict, memos: list[dict], customer: dict, domain: str, amount: int) -> dict:
    """Memos in announcement order; a later memo setting the same field wins."""
    d = dict(decision)
    for m in memos:
        if not memo_applies(m, domain=domain, amount=amount, tier=customer["tier"]):
            continue
        if m["type"] == "escalate":
            d["action"], d["approver"] = "escalate", "compliance_officer"
        elif m["type"] == "deadline":
            d["deadline_days"] = m["days"]
        elif m["type"] == "document":
            d["document"] = m["document"]
    return d


def apply_quota(decision: dict, approvals_before: int) -> dict:
    if decision["action"] == "approve" and approvals_before >= QUOTA:
        return {**decision, "action": "escalate", "approver": "team_lead"}
    return dict(decision)


def ticket_calls(world: World, procedure_id: str, order_id: str, memos: list[dict]) -> list[dict]:
    proc = next(p for p in world.procedures if p.id == procedure_id)
    calls = [{"tool": s["tool"], "args": {k: (order_id if v == "{order_id}" else v) for k, v in s["args"].items()}} for s in proc.steps]
    for m in memos:  # announcement order: a later memo on the same code wins
        if memo_applies(m, procedure=procedure_id):
            for c in calls:
                if c["tool"] == m["tool"] and m["param"] in c["args"]:
                    c["args"][m["param"]] = m["new"]
    return calls


def memo_active(memo: dict, position: int) -> bool:
    """In force at a case's position: announced at or before it and not withdrawn at or before it."""
    return memo["position"] <= position and (memo["withdrawn_at"] is None or position < memo["withdrawn_at"])


def decisions_equal(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return (
            str(a["action"]) == b["action"]
            and str(a["approver"]) == b["approver"]
            and int(a["deadline_days"]) == int(b["deadline_days"])
            and str(a["document"]) == b["document"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def calls_key(calls: list[dict]) -> list[str]:
    return sorted(json.dumps({"tool": c["tool"], "args": {k: str(v) for k, v in c["args"].items()}}, sort_keys=True) for c in calls)


# --- Generation ----------------------------------------------------------------------------------------------


def variant_tag(knobs: dict | None) -> str:
    """The world-ID tag of a knob setting ("" at the defaults, e.g. "o2250"): `tasks.study_g` selects a run_plan
    session cell's worlds by it. Unknown knobs raise KeyError."""
    diff = {k: v for k, v in (knobs or {}).items() if v != DEFAULT_KNOBS[k]}
    short = {"memo_density": "m", "dependency_density": "d", "output_tokens": "o"}
    return "".join(f"{short[k]}{v:g}" for k, v in sorted(diff.items()))


def _merge_kb(world: World, f7: World, f3: World) -> None:
    for src in (f7, f3):
        for fid, f in src.facts.items():
            world.add_fact(fid, f.text, f.kind)
    world.policies, world.exceptions = list(f7.policies), list(f7.exceptions)
    world.tools, world.procedures = list(f3.tools), list(f3.procedures)
    world.documents = list(f7.documents) + list(f3.documents)


def _kinds(rng: random.Random, n: int, dep_density: float) -> tuple[list[str], int, int]:
    n_dep = min(n - 2, round(dep_density * n)) if dep_density > 0 else 0
    n_dep = max(n_dep, 1 if dep_density > 0 and n >= 4 else 0)
    n_follow = n_dep // 2 if n_dep >= 2 else n_dep
    n_repeat = n_dep - n_follow
    rest = n - n_dep
    n_ticket = round(TICKET_SHARE * rest)
    kinds = ["policy"] * (rest - n_ticket) + ["ticket"] * n_ticket + ["followup"] * n_follow + ["repeat"] * n_repeat
    rng.shuffle(kinds)
    return kinds, n_follow, n_repeat


def _plan_items(rng: random.Random, world: World, n: int, dep_density: float) -> list[dict]:
    kinds, _, n_repeat = _kinds(rng, n, dep_density)
    n_freq = math.ceil(n_repeat / 2) if n_repeat else 0
    region = world.policies[0].region  # F7-10 has one region; any F7 level works (customers get a policy region)
    regions = sorted({p.region for p in world.policies})
    domains = sorted({p.domain for p in world.policies})
    exc_by_policy = {x.policy_id: x for x in world.exceptions}
    procs = sorted(world.procedures, key=lambda p: p.id)
    used: set[str] = set()

    def fresh(prefix: str, digits: int) -> str:
        while True:
            x = f"{prefix}-{rng.randint(10 ** (digits - 1), 10 ** digits - 1)}"
            if x not in used:
                used.add(x)
                return x

    customers: dict[str, dict] = {}
    items: list[dict] = []
    frequent: list[str] = []
    open_await: list[int] = []  # indexes of awaiting items not yet followed up

    def pick_case(customer: dict, prefer_approve: bool) -> tuple[str, int]:
        cands = [p for p in world.policies if p.region == customer["region"]]
        if prefer_approve:
            approve = [p for p in cands if base_decision(world, customer, p.domain, p.lo)["action"] == "approve"]
            cands = approve or cands
        p = rng.choice(cands)
        return p.domain, rng.randint(p.lo, p.hi - 1)

    i = 0
    while i < len(kinds):
        kind = kinds[i]
        later_policy = next((j for j in range(i + 1, len(kinds)) if kinds[j] == "policy"), None)
        if kind == "followup" and not open_await or kind == "repeat" and not frequent:
            if later_policy is not None:
                kinds[i], kinds[later_policy] = kinds[later_policy], kinds[i]
            else:
                kinds[i] = "policy"
            kind = kinds[i]
        item = {"position": i + 1, "case_id": fresh("C", 6), "kind": kind, "awaiting": False}
        if kind == "policy":
            p_region = region if len(regions) == 1 else rng.choice(regions)
            pol = rng.choice([p for p in world.policies if p.region == p_region])
            exc = exc_by_policy.get(pol.id)
            tier = exc.tier if exc and rng.random() < 0.5 else rng.choice(TIERS)
            cid = fresh("CU", 5)
            customers[cid] = {"region": p_region, "tier": tier}
            is_freq = len(frequent) < n_freq
            if is_freq:
                frequent.append(cid)
                domain, amount = pick_case(customers[cid], prefer_approve=True)
            else:
                domain, amount = pol.domain, rng.randint(pol.lo, pol.hi - 1)
                item["awaiting"] = rng.random() < AWAIT_RATE or (sum(k == "followup" for k in kinds[i + 1 :]) > len(open_await))
            item.update(customer_id=cid, domain=domain, amount=amount)
            if item["awaiting"]:
                open_await.append(i)
        elif kind == "repeat":
            cid = rng.choice(frequent)
            domain, amount = pick_case(customers[cid], prefer_approve=True)
            item.update(customer_id=cid, domain=domain, amount=amount)
        elif kind == "followup":
            j = open_await.pop(rng.randrange(len(open_await)))
            items[j]["followed_up_at"] = i + 1
            item.update(original=items[j]["case_id"])
        else:
            proc = rng.choice(procs)
            oid = fresh("O", 6)
            item.update(order_id=oid, procedure=proc.id, order={"region": proc.situation.split("/")[1], "status": "open"})
        items.append(item)
        i += 1
    assert len(domains) >= 1
    world.entities["session_customers"] = customers
    return items


def _plan_memos(rng: random.Random, world: World, items: list[dict], memo_density: float) -> list[dict]:
    n = len(items)
    n_memos = max(1, round(memo_density * n)) if memo_density > 0 else 0
    positions = sorted(rng.sample(range(1, max(2, n)), min(n_memos, max(1, n - 1))))
    memos: list[dict] = []
    customers = world.entities["session_customers"]
    for p in positions:
        later = [it for it in items if it["position"] > p]
        has_ticket = any(it["kind"] == "ticket" for it in later)
        has_policy = any(it["kind"] in ("policy", "repeat") for it in later)
        types = [(t, w) for t, w in MEMO_TYPES if (t == "procedure_code" and has_ticket) or (t != "procedure_code" and has_policy)]
        if not types:
            continue
        t = rng.choices([t for t, _ in types], [w for _, w in types])[0]
        mid = f"M-{len(memos) + 1}"
        if t == "procedure_code":
            target = rng.choice([it for it in later if it["kind"] == "ticket"])
            proc = next(x for x in world.procedures if x.id == target["procedure"])
            step = rng.choice(proc.steps)
            param = next(k for k in step["args"] if k != "order_id")
            old = step["args"][param]
            new = f"{old.split('-')[0]}-{rng.randint(100, 999)}"
            while new == old:
                new = f"{old.split('-')[0]}-{rng.randint(100, 999)}"
            memo = {"type": t, "procedure": proc.id, "tool": step["tool"], "param": param, "old": old, "new": new}
            text = f"in procedure {proc.id}, call {step['tool']} with {param}={new} instead of {param}={old}."
        else:
            target = rng.choice([it for it in later if it["kind"] in ("policy", "repeat")])
            cust = customers[target["customer_id"]]
            phrase = _phrase(world, target["domain"])
            current = base_decision(world, cust, target["domain"], target["amount"])
            if t == "escalate":
                pol = next(x for x in world.policies if x.domain == target["domain"] and x.region == cust["region"] and x.lo <= target["amount"] < x.hi)
                memo = {"type": t, "domain": target["domain"], "threshold": max(pol.lo, 1)}
                text = f"every {phrase} request of {_money(memo['threshold'])} or more must be escalated to the compliance officer; the deadline and required document stay as policy says."
            elif t == "deadline":
                days = rng.choice([d for d in range(1, 11) if d != current["deadline_days"]])
                memo = {"type": t, "domain": target["domain"], "days": days}
                text = f"{phrase} requests must now be answered within {days} business day{'s' if days != 1 else ''}."
            else:
                doc = rng.choice([d for d in DOCUMENTS if d != current["document"]])
                memo = {"type": t, "domain": target["domain"], "tier": cust["tier"], "document": doc}
                need = "no longer require any supporting document" if doc == "none" else f"now require {gen_f7._DOC_TEXT[doc]}"
                text = f"{phrase} requests from {cust['tier']} customers {need}, whatever policy says."
        withdrawn_at = None
        if rng.random() < WITHDRAW_RATE and target["position"] < n:
            withdrawn_at = rng.randint(target["position"] + 1, n)
        memos.append({"id": mid, **memo, "position": p, "target": target["position"], "withdrawn_at": withdrawn_at, "text": f"MEMO {mid} (in force from this case until withdrawn): {text}"})
    return memos


def _gold(world: World, items: list[dict], memos: list[dict]) -> None:
    """Gold per item, sequentially, with the counterfactuals the failure taxonomy needs."""
    customers = world.entities["session_customers"]
    by_case = {it["case_id"]: it for it in items}
    approvals: Counter = Counter()

    def decide(it: dict, active: list[dict], quota_count: int | None) -> dict:
        src = by_case[it["original"]] if it["kind"] == "followup" else it
        cust = customers[src["customer_id"]]
        d = apply_memos(base_decision(world, cust, src["domain"], src["amount"]), active, cust, src["domain"], src["amount"])
        return d if quota_count is None else apply_quota(d, quota_count)

    for it in items:
        pos = it["position"]
        active = [m for m in memos if memo_active(m, pos)]
        withdrawn = [m for m in memos if m["withdrawn_at"] is not None and m["withdrawn_at"] <= pos]
        kinds: list[str] = []
        cf: dict = {"without_memo": {}, "with_withdrawn": {}}
        if it["kind"] == "ticket":
            gold = {"calls": ticket_calls(world, it["procedure"], it["order_id"], active)}
            cf["stateless"] = {"calls": ticket_calls(world, it["procedure"], it["order_id"], [])}
            for m in active:
                alt = ticket_calls(world, it["procedure"], it["order_id"], [x for x in active if x is not m])
                if calls_key(alt) != calls_key(gold["calls"]):
                    cf["without_memo"][m["id"]] = {"calls": alt}
            for m in withdrawn:
                alt = ticket_calls(world, it["procedure"], it["order_id"], sorted([*active, m], key=lambda x: x["position"]))
                if calls_key(alt) != calls_key(gold["calls"]):
                    cf["with_withdrawn"][m["id"]] = {"calls": alt}
        else:
            quota = None if it["kind"] == "followup" else approvals[it["customer_id"]]
            gold = decide(it, active, quota)
            pre_quota = decide(it, active, None)
            cf["stateless"] = decide(it, [], None)
            for m in active:
                alt = decide(it, [x for x in active if x is not m], quota)
                if not decisions_equal(alt, gold):
                    cf["without_memo"][m["id"]] = alt
            for m in withdrawn:
                alt = decide(it, sorted([*active, m], key=lambda x: x["position"]), quota)
                if not decisions_equal(alt, gold):
                    cf["with_withdrawn"][m["id"]] = alt
            if quota is not None and not decisions_equal(pre_quota, gold):
                cf["without_quota"] = pre_quota
            if it["kind"] == "repeat" and pre_quota["action"] == "approve":
                kinds.append("quota")
            if it["kind"] == "followup":
                kinds.append("followup")
                cf["original_gold"] = by_case[it["original"]]["gold"]
            if it["kind"] != "followup" and gold["action"] == "approve":
                approvals[it["customer_id"]] += 1
        if any(by_id["position"] < pos for by_id in memos if by_id["id"] in cf["without_memo"]):
            kinds.append("memo")
        if cf["with_withdrawn"]:
            kinds.append("withdrawn")
        it["gold"], it["counterfactuals"], it["dependency_kinds"] = gold, cf, kinds
        it["quota_triggered"] = "without_quota" in cf


def _ok(items: list[dict], memos: list[dict], n_repeat_planned: bool) -> bool:
    """Every memo changes at least one later case; at least one quota trigger when repeats exist."""
    for m in memos:
        if not any(m["id"] in it["counterfactuals"]["without_memo"] and it["position"] > m["position"] for it in items):
            return False
    repeats = sum(it["kind"] == "repeat" for it in items)
    return not (n_repeat_planned and repeats >= 2 and not any(it["quota_triggered"] for it in items))


def _message(world: World, it: dict, memos: list[dict]) -> str:
    lines = [m["text"] for m in memos if m["position"] == it["position"]]
    lines += [f"MEMO: {m['id']} is withdrawn, effective from this case." for m in memos if m["withdrawn_at"] == it["position"]]
    cid = it["case_id"]
    if it["kind"] in ("policy", "repeat"):
        text = (
            f"Case {cid}: customer {it['customer_id']} has submitted a {_phrase(world, it['domain'])} request for "
            f"{_money(it['amount'])}. Look up the customer, decide exactly as policy requires, and call submit_decision "
            f"with case_id {cid}."
        )
        if it["awaiting"]:
            text += " The customer has not yet sent all supporting documents: this case waits for a follow-up."
    elif it["kind"] == "followup":
        text = (
            f"Case {cid}: follow-up to case {it['original']}. The customer has now sent the supporting documents. "
            f"Re-decide case {it['original']}'s original request under the policies and memos in force now, and call "
            f"submit_decision with case_id {cid}."
        )
    else:
        text = (
            f"Ticket {cid}: {_situation(next(p.domain for p in world.procedures if p.id == it['procedure']))} (order {it['order_id']}). Resolve it exactly as the "
            f"standard operating procedure requires, then call finish with case_id {cid}."
        )
    return "\n".join([*lines, text])


def generate(level: str, split: str, seed: int, n_tasks: int = 0, *, memo_density: float | None = None, dependency_density: float | None = None, output_tokens: int | None = None, kb_levels: tuple[str, str] = KB_LEVELS) -> World:
    """One F8 session of N = int(level) cases. `n_tasks` is ignored: the level is the session length."""
    n = int(level)
    if n < 2:
        raise ValueError("an F8 session needs at least 2 cases")
    knobs = {
        "memo_density": DEFAULT_KNOBS["memo_density"] if memo_density is None else float(memo_density),
        "dependency_density": DEFAULT_KNOBS["dependency_density"] if dependency_density is None else float(dependency_density),
        "output_tokens": DEFAULT_KNOBS["output_tokens"] if output_tokens is None else int(output_tokens),
    }
    tag = variant_tag(knobs)
    world = World(id=f"F8-{n}{'-' + tag if tag else ''}-{split}-s{seed}", family="F8", level=str(n), seed=seed, split=split)
    _merge_kb(world, gen_f7.generate(kb_levels[0], True, split, seed, 0, "descriptive"), gen_f3.generate(kb_levels[1], split, seed, 0))
    for attempt in range(200):
        rng = random.Random(f"F8|{n}|{split}|{seed}|{sorted(knobs.items())}|{kb_levels}|{attempt}")
        items = _plan_items(rng, world, n, knobs["dependency_density"])
        memos = _plan_memos(rng, world, items, knobs["memo_density"])
        _gold(world, items, memos)
        if _ok(items, memos, knobs["dependency_density"] > 0):
            break
    else:
        raise RuntimeError(f"{world.id}: no valid session plan in 200 attempts")
    customers = world.entities.pop("session_customers")
    for m in memos:
        world.add_fact(f"f-{m['id']}", m["text"], "memo")
    world.facts = dict(sorted(world.facts.items()))
    world.entities = {
        "session": {
            "N": n,
            "knobs": knobs,
            "kb_levels": list(kb_levels),
            "quota": QUOTA,
            "window": WINDOW,
            "queue": [it["case_id"] for it in items],
            "memos": memos,
            "customers": customers,
            "orders": {it["order_id"]: it["order"] for it in items if it["kind"] == "ticket"},
        }
    }
    world.tasks = [_task(world, it, memos, items) for it in items]
    world.entities["session"]["report_gold"] = report_gold(world)
    world.entities["session"]["reference"] = reference_stats(world)
    return world


def _task(world: World, it: dict, memos: list[dict], items: list[dict]) -> TaskItem:
    by_case = {x["case_id"]: x for x in items}
    src = by_case[it["original"]] if it["kind"] == "followup" else it
    facts: list[str] = []
    if it["kind"] == "ticket":
        facts = [f"f-{it['procedure']}", *[f"f-tool-{c['tool']}" for c in it["gold"]["calls"]]]
    else:
        cust = session(world)["customers"][src["customer_id"]]
        pol = next(p for p in world.policies if p.domain == src["domain"] and p.region == cust["region"] and p.lo <= src["amount"] < p.hi)
        facts = [pol.fact_id] + [x.fact_id for x in world.exceptions if x.policy_id == pol.id]
    facts += [f"f-{m['id']}" for m in memos if m["id"] in it["counterfactuals"]["without_memo"]]
    tags = {
        "position": it["position"],
        "case_id": it["case_id"],
        "kind": it["kind"],
        "dependency": bool(it["dependency_kinds"]),
        "dependency_kinds": it["dependency_kinds"],
        "awaiting": it["awaiting"],
        "followed_up_at": it.get("followed_up_at"),
        "quota_triggered": it["quota_triggered"],
        "counterfactuals": it["counterfactuals"],
    }
    for k in ("customer_id", "domain", "amount", "original", "order_id", "procedure"):
        if k in it:
            tags[k] = it[k]
    if it["kind"] == "followup":
        tags.update(customer_id=src["customer_id"], domain=src["domain"], amount=src["amount"])
    return TaskItem(
        id=f"{world.id}-i{it['position']:03d}",
        world_id=world.id,
        family="F8",
        level=world.level,
        prompt=_message(world, it, memos),
        gold=it["gold"],
        gold_fact_ids=list(dict.fromkeys(facts)),
        answer_tool="finish" if it["kind"] == "ticket" else "submit_decision",
        tags=tags,
    )


# --- Session state, report and the O-state oracle -----------------------------------------------------------


def session(world: World) -> dict:
    return world.entities["session"]


def item(world: World, position: int) -> TaskItem:
    return world.tasks[position - 1]


def disposition(task: TaskItem) -> str:
    return "resolved" if task.tags["kind"] == "ticket" else task.gold["action"]


def report_gold(world: World) -> dict:
    tasks = world.tasks
    return {
        "dispositions": {t.tags["case_id"]: disposition(t) for t in tasks},
        "pending_recheck": sorted(t.tags["case_id"] for t in tasks if disposition(t) == "escalate"),
        "open_followups": sorted(t.tags["case_id"] for t in tasks if t.tags["awaiting"] and t.tags["followed_up_at"] is None),
    }


def state_at(world: World, k: int) -> dict:
    """Ground-truth agent state after k cases (k = 0 is the start of the shift)."""
    s = session(world)
    done = world.tasks[:k]
    return {
        "completed": [t.tags["case_id"] for t in done],
        "pending": s["queue"][k:],
        "memos_in_force": [m["id"] for m in s["memos"] if m["position"] <= k and (m["withdrawn_at"] is None or m["withdrawn_at"] > k)],
        "open_followups": [t.tags["case_id"] for t in done if t.tags["awaiting"] and (t.tags["followed_up_at"] is None or t.tags["followed_up_at"] > k)],
        "escalated": [t.tags["case_id"] for t in done if disposition(t) == "escalate"],
    }


def _decision_text(d: dict) -> str:
    approver = f", approver {d['approver']}" if d["approver"] != "none" else ""
    return f"{d['action']}{approver}, deadline {d['deadline_days']} days, document {d['document']}"


def render_oracle_state(world: World, k: int) -> str:
    """What the O-state arm sees before case k+1: the true shift state after k cases, with everything a later
    case can need (memo texts, approvals per customer, original requests of open follow-ups)."""
    s, st = session(world), state_at(world, k)
    memos = {m["id"]: m for m in s["memos"]}
    approvals = Counter(t.tags["customer_id"] for t in world.tasks[:k] if t.tags["kind"] in ("policy", "repeat") and t.gold["action"] == "approve")
    lines = [f"## Shift state (oracle) after {k} of {s['N']} cases", ""]
    lines.append("Pending cases, in order: " + (", ".join(st["pending"]) or "none"))
    lines.append("Memos in force:" + ("" if st["memos_in_force"] else " none"))
    lines += [f"- {memos[m]['text']}" for m in st["memos_in_force"]]
    gone = [m["id"] for m in s["memos"] if m["withdrawn_at"] is not None and m["withdrawn_at"] <= k]
    if gone:
        lines.append("Withdrawn memos (no longer apply): " + ", ".join(gone))
    lines.append("Approvals this shift per customer: " + (", ".join(f"{c}: {n}" for c, n in sorted(approvals.items())) or "none"))
    lines.append("Cases waiting for a follow-up: " + (", ".join(st["open_followups"]) or "none"))
    lines.append("Escalated cases (pending recheck): " + (", ".join(st["escalated"]) or "none"))
    lines.append("Completed cases:")
    for t in world.tasks[:k]:
        if t.tags["kind"] == "ticket":
            lines.append(f"- {t.tags['case_id']}: ticket for order {t.tags['order_id']}, resolved")
        else:
            what = f"{_phrase(world, t.tags['domain'])} request for {_money(t.tags['amount'])} from customer {t.tags['customer_id']}"
            follow = f" (follow-up to {t.tags['original']})" if t.tags["kind"] == "followup" else ""
            lines.append(f"- {t.tags['case_id']}{follow}: {what}; decided {_decision_text(t.gold)}")
    return "\n".join(lines)


# --- Prompts shared by the runner and the reference trajectory ----------------------------------------------


def system_prompt(world: World) -> str:
    """The session's system prompt: the base prompt, the whole (small) corpus and the shift rules. Every
    session arm gets the same one."""
    from .render import corpus_text

    return f"{BASE_SYSTEM}\n\n## Knowledge base\n{corpus_text(world)}\n\n{SHIFT_RULES}"


def start_message(world: World) -> str:
    q = session(world)["queue"]
    return f"Shift start. Your queue has {len(q)} cases, in this order: {', '.join(q)}. The first case follows."


# --- Bulky tool files (deterministic per world and ID) ------------------------------------------------------


def _target(world: World, key: str) -> int:
    base = session(world)["knobs"]["output_tokens"] if "session" in world.entities else DEFAULT_KNOBS["output_tokens"]
    return int(base * random.Random(f"F8-size|{world.seed}|{key}").uniform(0.6, 1.4))


def _grow(record: dict, key: str, make, target: int) -> str:
    """Append generated entries to record[key] until the JSON reaches about `target` tokens."""
    text = json.dumps(record, indent=1)
    tokens = count_tokens(text)
    while tokens < target:
        entry = make()
        record[key].append(entry)
        tokens += count_tokens(json.dumps(entry, indent=1)) + 2
    return json.dumps(record, indent=1)


def customer_file(world: World, customer_id: str) -> str:
    c = session(world)["customers"].get(customer_id)
    if c is None:
        return f"No customer with ID {customer_id}."
    rng = random.Random(f"F8-customer|{world.seed}|{customer_id}")
    past = [t for t in TIERS if t != c["tier"]]
    rec = {
        "customer_id": customer_id,
        "name": f"{rng.choice(_FIRST)} {rng.choice(_LAST)}",
        "region": c["region"],
        "tier": c["tier"],
        "tier_history": [{"tier": rng.choice(past), "from": f"20{rng.randint(15, 21)}-0{rng.randint(1, 9)}-01", "until": f"20{rng.randint(22, 24)}-0{rng.randint(1, 9)}-01"}],
        "billing_address": {"street": f"{rng.randint(1, 250)} {rng.choice(_LAST)} Street", "city": rng.choice(_CITIES), "country": rng.choice(_COUNTRIES)},
        "contact": {"email": f"{customer_id.lower()}@example.com", "phone": f"+{rng.randint(1, 99)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}"},
        "marketing": {"segment": rng.choice(["platinum-offers", "value-seekers", "new-arrivals", "lapsed"]), "newsletter": rng.random() < 0.5},
        "account_flags": rng.sample(["paperless", "two-factor", "price-alerts", "returns-frequent", "vip-events"], 2),
        "support_notes": rng.sample(_NOTES, 3),
        "orders": [],
    }

    def order() -> dict:
        lines = [{"sku": s, "name": nm, "qty": rng.randint(1, 3), "unit_price": round(rng.uniform(9, 900), 2)} for s, nm in rng.sample(_PRODUCTS, rng.randint(1, 3))]
        return {
            "order_id": f"O-{rng.randint(100000, 999999)}",
            "placed": f"202{rng.randint(2, 5)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            "status": rng.choice(["delivered", "delivered", "returned", "cancelled", "refunded"]),
            "lines": lines,
            "total": round(sum(x["qty"] * x["unit_price"] for x in lines), 2),
            "ship_to_country": rng.choice(_COUNTRIES),
        }

    return _grow(rec, "orders", order, _target(world, customer_id))


def order_file(world: World, order_id: str) -> str:
    o = session(world)["orders"].get(order_id)
    if o is None:
        return f"No order with ID {order_id}."
    rng = random.Random(f"F8-order|{world.seed}|{order_id}")
    others = [r for r in gen_f3.REGIONS if r != o["region"]]
    lines = [{"sku": s, "name": nm, "qty": rng.randint(1, 4), "unit_price": round(rng.uniform(9, 900), 2)} for s, nm in rng.sample(_PRODUCTS, rng.randint(1, 4))]
    rec = {
        "order_id": order_id,
        "destination_region": o["region"],
        "status": o["status"],
        "customer_id": f"CU-{rng.randint(10000, 99999)}",
        "placed": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "lines": lines,
        "payment": {"method": rng.choice(["card", "paypal", "gift card"]), "captured": True, "currency": rng.choice(["USD", "EUR", "GBP"])},
        "warehouse": {"origin_region": rng.choice(others), "site": f"WH-{rng.randint(10, 99)}", "picker": f"{rng.choice(_FIRST)} {rng.choice(_LAST)[0]}."},
        "notes": rng.sample(_NOTES, 2),
        "shipment_events": [],
    }

    def scan() -> dict:
        return {"ts": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}", "event": rng.choice(_SCANS), "facility": f"{rng.choice(_CITIES)} hub {rng.randint(1, 9)}"}

    return _grow(rec, "shipment_events", scan, _target(world, order_id))


# --- Reference trajectory (the solver's minimal calls) and its view sizes --------------------------------


def _call_tokens(tool: str, args: dict) -> int:
    return count_tokens(json.dumps({"function": tool, "arguments": args}))


def reference_steps(world: World) -> list[dict]:
    """Per item, the reference solver's generations: [{item, view_add (tokens appended before the call),
    call_tokens, result_tokens}], plus the report. Used for the W crossing, budget priors and tests."""
    steps = []
    for t in world.tasks:
        pos, tags = t.tags["position"], t.tags
        first = True
        calls: list[tuple[str, dict, str]] = []
        if tags["kind"] in ("policy", "repeat"):
            calls.append(("lookup_customer", {"customer_id": tags["customer_id"]}, customer_file(world, tags["customer_id"])))
        if tags["kind"] == "ticket":
            calls.append(("order_lookup", {"order_id": tags["order_id"]}, order_file(world, tags["order_id"])))
            calls += [(c["tool"], c["args"], f"{c['tool']} executed for {tags['order_id']}.") for c in t.gold["calls"]]
            calls.append(("finish", {"case_id": tags["case_id"]}, f"Ticket {tags['case_id']} closed."))
        else:
            calls.append(("submit_decision", {"case_id": tags["case_id"], **t.gold}, f"Decision recorded for {tags['case_id']}."))
        for tool, args, result in calls:
            steps.append({"item": pos, "view_add": count_tokens(t.prompt) if first else 0, "call_tokens": _call_tokens(tool, args), "result_tokens": count_tokens(result)})
            first = False
    rg = session(world)["report_gold"]
    steps.append({"item": len(world.tasks) + 1, "view_add": count_tokens(REPORT_REQUEST), "call_tokens": _call_tokens("submit_shift_report", {"report": json.dumps(rg)}), "result_tokens": 4})
    return steps


def reference_stats(world: World) -> dict:
    """View tokens per generation of the reference trajectory: full history (CM0) and oracle (O-state).
    A lower bound for a real agent, which adds reasoning, text and retries."""
    system_tokens = count_tokens(system_prompt(world))
    sys_tokens = system_tokens + count_tokens(start_message(world))
    steps = reference_steps(world)
    full, oracle, crossing = [], [], None
    view = sys_tokens
    item_view: dict[int, int] = {}
    for s in steps:
        view += s["view_add"]
        if s["view_add"]:
            item_view[s["item"]] = 0
        full.append(view)
        if crossing is None and view > WINDOW:
            crossing = s["item"]
        k = min(s["item"], len(world.tasks)) - (0 if s["item"] > len(world.tasks) else 1)
        oracle.append(system_tokens + count_tokens(render_oracle_state(world, k)) + s["view_add"] + item_view.get(s["item"], 0))
        item_view[s["item"]] = item_view.get(s["item"], 0) + s["view_add"] + s["call_tokens"] + s["result_tokens"]
        view += s["call_tokens"] + s["result_tokens"]
    before = [v for v in full if v <= WINDOW]
    files = [s["result_tokens"] for s in steps if s["result_tokens"] > 50]
    n = len(world.tasks)
    return {
        "generations": len(steps),
        "generations_per_item": round((len(steps) - 1) / n, 3),
        "w_crossing_item": crossing,
        "mean_view_full": round(sum(full) / len(full)),
        "mean_view_full_before_overflow": round(sum(before) / len(before)) if before else None,
        "mean_view_oracle": round(sum(oracle) / len(oracle)),
        "max_view_oracle": max(oracle),
        "tool_file_tokens_mean": round(sum(files) / len(files)) if files else 0,
        "system_tokens": sys_tokens,
    }


# --- Reference solver: an independent sequential replay ------------------------------------------------------


def solve_session(world: World) -> tuple[list[dict], dict]:
    """Replays the session from the case messages' structured fields and the KB, keeping its own state (memos
    announced so far, approvals per customer, original requests). Returns the answer per item and the report."""
    s = session(world)
    memos = {m["id"]: m for m in s["memos"]}
    announced: list[dict] = []
    approvals: Counter = Counter()
    requests: dict[str, tuple[str, str, int]] = {}
    awaiting: set[str] = set()
    answers, dispositions, escalated = [], {}, []
    for t in world.tasks:
        pos, tags = t.tags["position"], t.tags
        for m in s["memos"]:
            if m["position"] == pos:
                announced.append(m)
        announced = [m for m in announced if not (memos[m["id"]]["withdrawn_at"] is not None and memos[m["id"]]["withdrawn_at"] <= pos)]
        if tags["kind"] == "ticket":
            ans = {"calls": ticket_calls(world, tags["procedure"], tags["order_id"], announced)}
            dispositions[tags["case_id"]] = "resolved"
        else:
            if tags["kind"] == "followup":
                cid, domain, amount = requests[tags["original"]]
                awaiting.discard(tags["original"])
            else:
                cid, domain, amount = tags["customer_id"], tags["domain"], tags["amount"]
                requests[tags["case_id"]] = (cid, domain, amount)
            cust = s["customers"][cid]
            d = apply_memos(base_decision(world, cust, domain, amount), announced, cust, domain, amount)
            if tags["kind"] != "followup":
                d = apply_quota(d, approvals[cid])
                approvals[cid] += d["action"] == "approve"
            if tags["awaiting"]:
                awaiting.add(tags["case_id"])
            ans = d
            dispositions[tags["case_id"]] = d["action"]
            if d["action"] == "escalate":
                escalated.append(tags["case_id"])
        answers.append(ans)
    report = {"dispositions": dispositions, "pending_recheck": sorted(escalated), "open_followups": sorted(awaiting)}
    return answers, report
