"""Context-length sweep worlds (CONTEXT_SWEEP.md): an F8 shift replayed up to a target context size, then one probe.

Study G asks what context a model should be *given*; this asks how much context a model can *receive* and still
answer. Each sample is one model decision at the end of a replayed F8 shift whose transcript fills the context to a
target size (8K to 960K tokens). Only the amount of history between the start of the shift and the probe changes.

**One world per (kind, task, size).** A task is one seed (`task_seed`): its own handbook, customers, case 1 and probe.
Its cases, in queue order:

- **Case 1 (the head).** Where the probe's dependency lives, at the very start of the history.
- **The middle.** Neutral cases (policy cases and tickets, every customer and order fresh, no memos, no follow-ups,
  no repeats), drawn from one seeded sequence, so a smaller size's middle is a prefix of a larger size's. As many as
  fit the target (`fit`). No middle case is in the scope of a memo in force, so none of them shows the memo applied.
- **The probe (last case).** Its gold decision is the same at every size (checked when a world is built).

**Kinds** (what the probe needs besides the handbook, which is in the system prompt at the start of every context):

- `fresh`: a new policy case. Everything else it needs is in its own message and its customer file (its lookup is
  replayed just before the decision). It measures the cost of length alone.
- `memo`: a new policy case whose decision a memo changes. The memo is announced in case 1's message, at the start.
- `followup`: a follow-up to case 1 (an awaiting case at the start). The follow-up names only the case ID, so the
  original request (customer, request, amount) and the customer's tier must be found at the start; a memo announced
  in the follow-up's own message changes the decision, so copying case 1's earlier decision fails.

**The replayed history** is what Study G's CM0 view would hold after a perfect agent (the reference trajectory,
`gen_f8.reference_steps`): system prompt, a start message, then per case the case message, one assistant tool call per
generation (lookup, ticket calls, finish or submit_decision with the gold) and each tool's result (the bulky customer
and order files). The probe's case message follows, with its customer lookup replayed for `fresh` and `memo`
(`followup`, as in F8, makes no lookup). The start message names no queue, so it is the same at every size.

**Context size** = the decision call's input as the session meters it: o200k tokens of every message's text and tool
calls (`context_policy.view_tokens`) plus the session tools' schemas (`context_policy.tool_schema_tokens`). The fit
fills the middle with whole cases up to the target, so a world sits at most one case (about `output_tokens` + 150
tokens) under it; the world records the exact figure. The cache nonce a run puts at the head of the system prompt
adds a few tokens on top.

Worlds are F8 worlds in every way the session tools and scorer read (`entities["session"]`, tasks with F8 tags and
gold), so `env_f8.build_session_tools` and `scorers.session.item_success` work unchanged. `entities["sweep"]` holds
the design record.
"""

import random
from collections.abc import Iterator, Sequence

from . import gen_f3, gen_f7, gen_f8
from .spec import World

KINDS = ("fresh", "memo", "followup")
SIZES = (8_000, 64_000, 128_000, 256_000, 512_000, 960_000)
SPLIT = "sweep"
SEED_BASE = 40_000  # task j of a kind: SEED_BASE + len(KINDS) * j + KINDS.index(kind) (no other study's seeds are this high)
DEFAULT_OUTPUT_TOKENS = gen_f8.DEFAULT_KNOBS["output_tokens"]  # F8's tool-file size, as Study G's sessions
START = "Shift start. Your cases arrive one at a time, in queue order. The first case follows."
PROBE_MARK = 10**9  # the probe's position while planning (its real position is known only after the fit)


def world_id(kind: str, size: int, seed: int, output_tokens: int = DEFAULT_OUTPUT_TOKENS) -> str:
    tag = "" if output_tokens == DEFAULT_OUTPUT_TOKENS else f"-o{output_tokens}"
    return f"F8S-{kind}-{size // 1000}k{tag}-{SPLIT}-s{seed}"


def task_seed(kind: str, task: int = 0, seed_base: int = SEED_BASE) -> int:
    """The world seed of task `task` (0, 1, ...) of a kind: the kinds interleave, so task 0 of every kind comes first."""
    if kind not in KINDS:
        raise ValueError(f"unknown sweep kind {kind!r}; one of {KINDS}")
    return seed_base + len(KINDS) * int(task) + KINDS.index(kind)


def kind_seed(kind: str, seed_base: int = SEED_BASE) -> int:
    """Task 0's seed (`task_seed`)."""
    return task_seed(kind, 0, seed_base)


# --- Planning ------------------------------------------------------------------------------------------------


class _Plan:
    """The ID pool, customers and orders shared by the head, the middle and the probe of one kind and seed."""

    def __init__(self, world: World, rng: random.Random, output_tokens: int):
        self.world, self.rng = world, rng
        self.used: set[str] = set()
        self.customers: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.exc_by_policy = {x.policy_id: x for x in world.exceptions}
        # customer_file and order_file read the session's customers, orders and knobs: the dicts are shared.
        world.entities["session"] = {"knobs": {**gen_f8.DEFAULT_KNOBS, "output_tokens": output_tokens}, "customers": self.customers, "orders": self.orders}

    def fresh(self, prefix: str, digits: int) -> str:
        while True:
            x = f"{prefix}-{self.rng.randint(10 ** (digits - 1), 10**digits - 1)}"
            if x not in self.used:
                self.used.add(x)
                return x

    def policy_case(self, ok=lambda customer, domain, amount: True) -> dict:
        """A policy case for a fresh customer, drawn until `ok(customer, domain, amount)` holds."""
        while True:
            pol = self.rng.choice(self.world.policies)
            exc = self.exc_by_policy.get(pol.id)
            tier = exc.tier if exc and self.rng.random() < 0.5 else self.rng.choice(gen_f7.TIERS)
            customer = {"region": pol.region, "tier": tier}
            amount = self.rng.randint(pol.lo, pol.hi - 1)
            if ok(customer, pol.domain, amount):
                cid = self.fresh("CU", 5)
                self.customers[cid] = customer
                return {"case_id": self.fresh("C", 6), "kind": "policy", "awaiting": False, "customer_id": cid, "domain": pol.domain, "amount": amount}

    def ticket(self) -> dict:
        proc = self.rng.choice(sorted(self.world.procedures, key=lambda p: p.id))
        oid = self.fresh("O", 6)
        self.orders[oid] = {"region": proc.situation.split("/")[1], "status": "open"}
        return {"case_id": self.fresh("C", 6), "kind": "ticket", "awaiting": False, "order_id": oid, "procedure": proc.id}

    def memo_changing(self, it: dict, position: int) -> dict:
        """A memo (F8's escalate, deadline or document types and wording) that changes `it`'s handbook decision."""
        world, cust = self.world, self.customers[it["customer_id"]]
        base = gen_f8.base_decision(world, cust, it["domain"], it["amount"])
        phrase = gen_f8._phrase(world, it["domain"])
        options = []
        for t, w in gen_f8.MEMO_TYPES:
            if t == "escalate" and (base["action"], base["approver"]) != ("escalate", "compliance_officer"):
                options.append((t, w))
            elif t in ("deadline", "document"):
                options.append((t, w))
        t = self.rng.choices([t for t, _ in options], [w for _, w in options])[0]
        if t == "escalate":
            pol = next(p for p in world.policies if p.domain == it["domain"] and p.region == cust["region"] and p.lo <= it["amount"] < p.hi)
            memo = {"type": t, "domain": it["domain"], "threshold": max(pol.lo, 1)}
            text = f"every {phrase} request of {gen_f8._money(memo['threshold'])} or more must be escalated to the compliance officer; the deadline and required document stay as policy says."
        elif t == "deadline":
            days = self.rng.choice([d for d in range(1, 11) if d != base["deadline_days"]])
            memo = {"type": t, "domain": it["domain"], "days": days}
            text = f"{phrase} requests must now be answered within {days} business day{'s' if days != 1 else ''}."
        else:
            doc = self.rng.choice([d for d in gen_f7.DOCUMENTS if d != base["document"]])
            memo = {"type": t, "domain": it["domain"], "tier": cust["tier"], "document": doc}
            need = "no longer require any supporting document" if doc == "none" else f"now require {gen_f7._DOC_TEXT[doc]}"
            text = f"{phrase} requests from {cust['tier']} customers {need}, whatever policy says."
        changed = gen_f8.apply_memos(base, [memo], cust, it["domain"], it["amount"])
        assert not gen_f8.decisions_equal(changed, base), f"memo {memo} leaves {it['case_id']} unchanged"
        return {"id": "M-1", **memo, "position": position, "target": PROBE_MARK, "withdrawn_at": None, "text": f"MEMO M-1 (in force from this case until withdrawn): {text}"}


def _kb_world(wid: str, size: int, seed: int) -> World:
    world = World(id=wid, family="F8S", level=f"{size // 1000}k", seed=seed, split=SPLIT)
    gen_f8._merge_kb(world, gen_f7.generate(gen_f8.KB_LEVELS[0], True, SPLIT, seed, 0, "descriptive"), gen_f3.generate(gen_f8.KB_LEVELS[1], SPLIT, seed, 0))
    return world


def _design(kind: str, plan: _Plan) -> tuple[dict, dict, list[dict]]:
    """Case 1, the probe and the memos, at planning positions (case 1 at 1, the probe at PROBE_MARK)."""
    if kind == "fresh":
        head = plan.policy_case()
        probe = plan.policy_case()
        memos: list[dict] = []
    elif kind == "memo":
        probe = plan.policy_case()
        memo = plan.memo_changing(probe, position=1)
        head = plan.policy_case(ok=lambda c, d, a: not gen_f8.memo_applies(memo, domain=d, amount=a, tier=c["tier"]))
        memos = [memo]
    elif kind == "followup":
        head = {**plan.policy_case(), "awaiting": True, "followed_up_at": PROBE_MARK}
        probe = {"case_id": plan.fresh("C", 6), "kind": "followup", "awaiting": False, "original": head["case_id"]}
        memos = [plan.memo_changing(head, position=PROBE_MARK)]
    else:
        raise ValueError(f"unknown sweep kind {kind!r}; one of {KINDS}")
    return {**head, "position": 1}, {**probe, "position": PROBE_MARK}, memos


def _middle(kind: str, plan: _Plan, memos: Sequence[dict]) -> Iterator[dict]:
    """Neutral cases, endlessly: tickets at F8's share, otherwise policy cases outside every early memo's scope."""
    early = [m for m in memos if m["position"] == 1]

    def neutral(c: dict, d: str, a: int) -> bool:
        return not any(gen_f8.memo_applies(m, domain=d, amount=a, tier=c["tier"]) for m in early)

    while True:
        yield plan.ticket() if plan.rng.random() < gen_f8.TICKET_SHARE else plan.policy_case(ok=neutral)


def _provisional_gold(world: World, it: dict) -> dict:
    """A middle case's gold before the world is assembled: no memo applies and every customer is fresh."""
    if it["kind"] == "ticket":
        return {"calls": gen_f8.ticket_calls(world, it["procedure"], it["order_id"], [])}
    return gen_f8.base_decision(world, gen_f8.session(world)["customers"][it["customer_id"]], it["domain"], it["amount"])


# --- The replayed history --------------------------------------------------------------------------------


def _calls(world: World, it: dict, gold: dict, *, probe: bool) -> list[tuple[str, dict, str]]:
    """The reference trajectory's calls for one case (`gen_f8.reference_steps`): the probe keeps only its lookup."""
    calls: list[tuple[str, dict, str]] = []
    if it["kind"] in ("policy", "repeat"):
        calls.append(("lookup_customer", {"customer_id": it["customer_id"]}, gen_f8.customer_file(world, it["customer_id"])))
    if probe:
        return calls
    if it["kind"] == "ticket":
        calls.append(("order_lookup", {"order_id": it["order_id"]}, gen_f8.order_file(world, it["order_id"])))
        calls += [(c["tool"], c["args"], f"{c['tool']} executed for {it['order_id']}.") for c in gold["calls"]]
        calls.append(("finish", {"case_id": it["case_id"]}, f"Ticket {it['case_id']} closed."))
    else:
        calls.append(("submit_decision", {"case_id": it["case_id"], **gold}, f"Decision recorded for {it['case_id']}."))
    return calls


def case_messages(world: World, it: dict, prompt: str, gold: dict, *, probe: bool = False) -> list:
    """One case as the history holds it: the case message, then per call an assistant tool call and its result."""
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
    from inspect_ai.tool import ToolCall

    msgs: list = [ChatMessageUser(content=prompt)]
    digits = it["case_id"].split("-")[1]
    for i, (tool, args, result) in enumerate(_calls(world, it, gold, probe=probe), start=1):
        call_id = f"call_{digits}_{i}"
        msgs.append(ChatMessageAssistant(content="", tool_calls=[ToolCall(id=call_id, function=tool, arguments=dict(args))]))
        msgs.append(ChatMessageTool(content=result, tool_call_id=call_id, function=tool))
    return msgs


def _item(world: World, position: int) -> dict:
    """A task's fields back as the planner's item dict (what `case_messages` reads)."""
    t = world.tasks[position - 1]
    it = {"case_id": t.tags["case_id"], "kind": t.tags["kind"], "position": position}
    for k in ("customer_id", "domain", "amount", "order_id", "procedure", "original"):
        if k in t.tags:
            it[k] = t.tags[k]
    if it["kind"] == "followup":
        it.pop("customer_id", None)  # a follow-up makes no lookup (the tags carry the original's customer for analysis)
    return it


def history(world: World, nonce: str | None = None) -> list:
    """The probe's input: system prompt (with the run's cache nonce at its head), start message, every case before the
    probe with its replayed calls, then the probe's message and (fresh, memo) its replayed lookup."""
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    from ..agent.cache_nonce import with_nonce

    msgs: list = [ChatMessageSystem(content=with_nonce(gen_f8.system_prompt(world), nonce)), ChatMessageUser(content=START)]
    n = len(world.tasks)
    for pos in range(1, n + 1):
        t = world.tasks[pos - 1]
        msgs += case_messages(world, _item(world, pos), t.prompt, t.gold, probe=pos == n)
    return msgs


def tool_tokens(world: World) -> int:
    from ..agent.context_policy import tool_schema_tokens
    from .env_f8 import SessionRecorder, build_session_tools

    return tool_schema_tokens(list(build_session_tools(world, SessionRecorder(world)).values()))


def context_tokens(world: World) -> int:
    """The probe's context size: view tokens of `history(world)` plus the session tools' schemas (no nonce)."""
    from ..agent.context_policy import view_tokens

    return view_tokens(history(world)) + tool_tokens(world)


# --- Generation ------------------------------------------------------------------------------------------


def generate_set(kind: str, sizes: Sequence[int] = SIZES, seed: int | None = None, output_tokens: int = DEFAULT_OUTPUT_TOKENS) -> list[World]:
    """The worlds of one kind at each size, sharing case 1, the probe and the middle's sequence (module docstring)."""
    from ..agent.context_policy import view_tokens

    seed = kind_seed(kind) if seed is None else int(seed)
    sizes = sorted({int(s) for s in sizes})
    rng = random.Random(f"F8S|{kind}|{seed}|{output_tokens}")
    scratch = _kb_world(world_id(kind, sizes[-1], seed, output_tokens), sizes[-1], seed)
    plan = _Plan(scratch, rng, output_tokens)
    head, probe, memos = _design(kind, plan)

    def prompt(it: dict) -> str:
        return gen_f8._message(scratch, it, memos)

    head_gold = gen_f8.base_decision(scratch, plan.customers[head["customer_id"]], head["domain"], head["amount"])
    fixed = (
        view_tokens([_system(scratch), _start()])
        + tool_tokens_for(scratch)
        + view_tokens(case_messages(scratch, head, prompt(head), head_gold))
        + view_tokens(case_messages(scratch, probe, prompt(probe), {}, probe=True))
    )
    if fixed > sizes[0]:
        raise ValueError(f"{kind}: the system prompt, case 1 and the probe alone take {fixed} tokens, more than the smallest size {sizes[0]}")
    middle: list[dict] = []
    cum = [0]
    pool = _middle(kind, plan, memos)
    while fixed + cum[-1] <= sizes[-1]:
        it = next(pool)
        it["position"] = len(middle) + 2
        middle.append(it)
        cum.append(cum[-1] + view_tokens(case_messages(scratch, it, prompt(it), _provisional_gold(scratch, it))))
    worlds = []
    for size in sizes:
        k = max(i for i, c in enumerate(cum) if fixed + c <= size)
        worlds.append(_assemble(kind, size, seed, output_tokens, plan, head, middle[:k], probe, memos))
    golds = {w.id: w.tasks[-1].gold for w in worlds}
    if len({repr(sorted(g.items())) for g in golds.values()}) != 1:
        raise AssertionError(f"{kind}: the probe's gold differs across sizes: {golds}")
    return worlds


def _system(world: World):
    from inspect_ai.model import ChatMessageSystem

    return ChatMessageSystem(content=gen_f8.system_prompt(world))


def _start():
    from inspect_ai.model import ChatMessageUser

    return ChatMessageUser(content=START)


def tool_tokens_for(world: World) -> int:
    """`tool_tokens` before the world has a queue (the schemas do not depend on it)."""
    s = gen_f8.session(world)
    had = "queue" in s
    s.setdefault("queue", [])
    try:
        return tool_tokens(world)
    finally:
        if not had:
            s.pop("queue")


def _assemble(kind: str, size: int, seed: int, output_tokens: int, plan: _Plan, head: dict, middle: list[dict], probe: dict, memos: list[dict]) -> World:
    world = _kb_world(world_id(kind, size, seed, output_tokens), size, seed)
    n = len(middle) + 2
    items = [dict(head), *(dict(it) for it in middle), {**probe, "position": n}]
    for pos, it in enumerate(items, start=1):
        it["position"] = pos
    if kind == "followup":
        items[0]["followed_up_at"] = n
    ms = [{**m, "target": n, "position": n if m["position"] == PROBE_MARK else m["position"]} for m in memos]
    used_customers = {it["customer_id"] for it in items if "customer_id" in it}
    customers = {c: plan.customers[c] for c in plan.customers if c in used_customers}
    world.entities["session_customers"] = customers
    gen_f8._gold(world, items, ms)
    world.entities["session"] = {"knobs": {**gen_f8.DEFAULT_KNOBS, "output_tokens": output_tokens}, "customers": customers, "orders": plan.orders}
    for it in items[:-1]:  # case 1 and the middle replay exactly the gold the fit priced: no memo or quota reaches them
        if it["gold"] != _provisional_gold(world, it):
            raise AssertionError(f"{world.id}: case {it['case_id']} is not neutral (gold {it['gold']})")
    world.entities.pop("session_customers")
    for m in ms:
        world.add_fact(f"f-{m['id']}", m["text"], "memo")
    world.facts = dict(sorted(world.facts.items()))
    orders = {it["order_id"]: plan.orders[it["order_id"]] for it in items if it["kind"] == "ticket"}
    world.entities = {
        "session": {
            "N": n,
            "knobs": {**gen_f8.DEFAULT_KNOBS, "output_tokens": output_tokens},
            "kb_levels": list(gen_f8.KB_LEVELS),
            "quota": gen_f8.QUOTA,
            "window": None,
            "queue": [it["case_id"] for it in items],
            "memos": ms,
            "customers": customers,
            "orders": orders,
        }
    }
    world.tasks = [gen_f8._task(world, it, ms, items) for it in items]
    probe_task = world.tasks[-1]
    world.entities["sweep"] = {
        "kind": kind,
        "target_tokens": size,
        "context_tokens": context_tokens(world),
        "middle_cases": len(middle),
        "probe_position": n,
        "probe_case_id": probe_task.tags["case_id"],
        "dependency_position": None if kind == "fresh" else 1,
        "memo_position": ms[0]["position"] if ms else None,
        "output_tokens": output_tokens,
        "seed": seed,
        "probe_gold": probe_task.gold,
        "probe_stateless": probe_task.tags["counterfactuals"].get("stateless"),
    }
    return world


def generate(kind: str, size: int, seed: int | None = None, output_tokens: int = DEFAULT_OUTPUT_TOKENS) -> World:
    """One world (prefer `generate_set` for several sizes of a kind: it shares the planning)."""
    return generate_set(kind, [size], seed, output_tokens)[0]
