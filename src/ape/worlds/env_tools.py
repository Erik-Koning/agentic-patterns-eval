"""Executable tool simulators for each family, bound to per-sample state.

Every mutating call and the final answer are recorded in the sample store
(`env_calls`, `env_answer`), which is what the scorers read. Tools that are always
available regardless of tool exposure (lookups, the answer tool) are listed in
`always_on()`.
"""

import json
from collections.abc import Callable

from inspect_ai.tool import ToolDef, ToolParam, ToolParams
from inspect_ai.util import store

from .gen_f7 import APPROVERS, DOCUMENTS
from .gen_registry import RATINGS, render_record
from .spec import TaskItem, World

CALLS = "env_calls"
ANSWER = "env_answer"
LOOKUPS = "env_lookups"  # {key: times looked up}, for faults that fire on the first surfacing only
FAULTS_FIRED = "env_faults_fired"


def _tool(name: str, description: str, params: dict[str, ToolParam], fn: Callable) -> ToolDef:
    return ToolDef(fn, name=name, description=description, parameters=ToolParams(properties=params, required=list(params)))


def _record(name: str, args: dict) -> None:
    store().set(CALLS, [*store().get(CALLS, []), {"tool": name, "args": dict(args)}])


def _answer(value: dict) -> None:
    store().set(ANSWER, value)


def always_on(world: World) -> list[str]:
    return {
        "F7": ["lookup_customer", "submit_decision"],
        "F3": ["order_lookup", "finish"],
        "F5": ["submit_answer"],
        "F1": ["lookup_supplier", "submit_ratings"],
        "F2": ["lookup_supplier", "submit_chain"],
    }[world.family]


def build_tools(world: World, task: TaskItem) -> dict[str, ToolDef]:
    """All tools for this world, keyed by name. Tool exposure decides which are bound per step."""
    if world.family == "F7":
        return _f7_tools(task)
    if world.family == "F3":
        return _f3_tools(world, task)
    if world.family in ("F1", "F2"):
        return _registry_tools(world, task)
    return _f5_tools()


def _f7_tools(task: TaskItem) -> dict[str, ToolDef]:
    customers = task.setup["customers"]

    async def lookup_customer(customer_id: str) -> str:
        c = customers.get(customer_id)
        return json.dumps({"customer_id": customer_id, **c}) if c else f"No customer with ID {customer_id}."

    async def submit_decision(action: str, approver: str, deadline_days: int, document: str) -> str:
        _answer({"action": action, "approver": approver, "deadline_days": int(deadline_days), "document": document})
        return "Decision recorded."

    s = lambda d, **kw: ToolParam(type="string", description=d, **kw)  # noqa: E731
    return {
        "lookup_customer": _tool("lookup_customer", "Look up a customer's region and loyalty tier.", {"customer_id": s("The customer ID, e.g. CU-12345.")}, lookup_customer),
        "submit_decision": _tool(
            "submit_decision",
            "Submit the final handling decision required by policy for this case.",
            {
                "action": s("What policy requires.", enum=["approve", "deny", "escalate"]),
                "approver": s("Who the request is escalated to; 'none' unless escalating.", enum=["none", *APPROVERS]),
                "deadline_days": ToolParam(type="integer", description="Business days allowed to respond."),
                "document": s("Supporting document the customer must provide; 'none' if not required.", enum=DOCUMENTS),
            },
            submit_decision,
        ),
    }


def _f3_tools(world: World, task: TaskItem) -> dict[str, ToolDef]:
    orders = task.setup["orders"]

    async def order_lookup(order_id: str) -> str:
        o = orders.get(order_id)
        return json.dumps({"order_id": order_id, **o}) if o else f"No order with ID {order_id}."

    async def finish() -> str:
        _answer({"calls": store().get(CALLS, [])})
        return "Ticket closed."

    tools = {
        "order_lookup": _tool("order_lookup", "Look up an order's destination region and status.", {"order_id": ToolParam(type="string", description="The order ID.")}, order_lookup),
        "finish": ToolDef(finish, name="finish", description="Close the ticket once the procedure is complete.", parameters=ToolParams()),
    }
    for t in world.tools:
        def make(name: str) -> Callable:
            async def call(**kwargs) -> str:
                _record(name, kwargs)
                return f"{name} executed for {kwargs.get('order_id', '?')}."
            return call

        tools[t.name] = _tool(t.name, t.description, {k: ToolParam(type="string", description=v) for k, v in t.params.items()}, make(t.name))
    return tools


def _f5_tools() -> dict[str, ToolDef]:
    async def submit_answer(answer: str) -> str:
        _answer({"answer": answer})
        return "Answer recorded."

    return {"submit_answer": _tool("submit_answer", "Submit the final answer.", {"answer": ToolParam(type="string", description="The answer.")}, submit_answer)}


def _active_fault(task: TaskItem) -> dict | None:
    """The task's environment fault (gen_registry, brief §5.3), only when explicitly enabled for this task."""
    return task.tags.get("fault") if task.setup.get("faults_enabled") else None


def _registry_tools(world: World, task: TaskItem) -> dict[str, ToolDef]:
    suppliers = world.entities["suppliers"]
    fault = _active_fault(task)

    async def lookup_supplier(supplier_id: str) -> str:
        sid = supplier_id.strip().upper()
        rec = suppliers.get(sid)
        if rec is None:
            return f"No supplier with ID {supplier_id}."
        seen = {**store().get(LOOKUPS, {})}
        seen[sid] = seen.get(sid, 0) + 1
        store().set(LOOKUPS, seen)
        if fault and fault["tool"] == "lookup_supplier" and fault["key"] == sid and seen[sid] == fault["occurrence"]:
            rec = {**rec, fault["field"]: fault["value"]}
            store().set(FAULTS_FIRED, [*store().get(FAULTS_FIRED, []), {**fault, "lookup": seen[sid]}])
        return render_record(rec)

    async def submit_ratings(ratings: dict) -> str:
        if isinstance(ratings, str):
            try:
                ratings = json.loads(ratings)
            except ValueError:
                ratings = {}
        _answer({"ratings": {str(k): str(v) for k, v in (ratings or {}).items()}})
        return "Ratings recorded."

    async def submit_chain(final_supplier: str, chain: list[str]) -> str:
        if isinstance(chain, str):
            chain = [c.strip() for c in chain.replace(";", ",").split(",") if c.strip()]
        _answer({"final": str(final_supplier), "chain": [str(c) for c in chain or []]})
        return "Escalation chain recorded."

    s = lambda d, **kw: ToolParam(type="string", description=d, **kw)  # noqa: E731
    tools = {"lookup_supplier": _tool("lookup_supplier", "Look up a supplier's record in the supplier registry.", {"supplier_id": s("The supplier ID, e.g. SUP-12345.")}, lookup_supplier)}
    if world.family == "F1":
        tools["submit_ratings"] = _tool(
            "submit_ratings",
            "Submit the review rating of every supplier in the task, all at once.",
            {"ratings": ToolParam(type="object", description="Map from supplier ID to its rating.", additionalProperties=ToolParam(type="string", enum=list(RATINGS)))},
            submit_ratings,
        )
    else:
        tools["submit_chain"] = _tool(
            "submit_chain",
            "Submit where an escalated dispute ended up and the path it took.",
            {
                "final_supplier": s("The supplier ID holding the dispute after the last escalation."),
                "chain": ToolParam(type="array", description="Supplier IDs the dispute moved to, in order, excluding the original supplier; the last is the final supplier.", items=ToolParam(type="string")),
            },
            submit_chain,
        )
    return tools
