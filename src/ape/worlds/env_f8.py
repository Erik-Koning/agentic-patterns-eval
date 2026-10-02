"""Tools for an F8 session, bound to one session's recorder.

Every call is recorded as an event `{item, tool, args}`, where `item` is the queue position being worked on
when the call was made (0 before the first case, N + 1 during the report). Scoring reads only these events, so
it never depends on how a context-management arm shaped the view.
"""

import json
from collections.abc import Callable
from typing import Any

from inspect_ai.tool import ToolDef, ToolParam, ToolParams

from . import gen_f8
from .gen_f7 import APPROVERS, DOCUMENTS
from .spec import TaskItem, World

# Sample store keys the session runner writes and the scorer reads.
ITEMS, VIEWS, PROBES, EVENTS, REPORT, OVERFLOW = "f8_items", "f8_views", "f8_probes", "f8_events", "f8_report", "f8_overflow_at"


class SessionRecorder:
    """Per-session tool state: the current item and every call made."""

    def __init__(self, world: World):
        self.world = world
        self.item = 0
        self.events: list[dict] = []
        self.report: dict | None = None
        self.report_error: str | None = None

    def record(self, tool: str, args: dict) -> None:
        self.events.append({"item": self.item, "tool": tool, "args": dict(args)})

    def answered(self, task: TaskItem) -> bool:
        """Whether the current case got its answer call in its own window."""
        tool = "finish" if task.tags["kind"] == "ticket" else "submit_decision"
        cid = task.tags["case_id"]
        return any(e["item"] == task.tags["position"] and e["tool"] == tool and e["args"].get("case_id") == cid for e in self.events)


def _tool(name: str, description: str, params: dict[str, ToolParam], fn: Callable) -> ToolDef:
    return ToolDef(fn, name=name, description=description, parameters=ToolParams(properties=params, required=list(params)))


def build_session_tools(world: World, rec: SessionRecorder) -> dict[str, ToolDef]:
    s = lambda d, **kw: ToolParam(type="string", description=d, **kw)  # noqa: E731

    async def lookup_customer(customer_id: str) -> str:
        rec.record("lookup_customer", {"customer_id": customer_id})
        return gen_f8.customer_file(world, customer_id)

    async def order_lookup(order_id: str) -> str:
        rec.record("order_lookup", {"order_id": order_id})
        return gen_f8.order_file(world, order_id)

    async def submit_decision(case_id: str, action: str, approver: str, deadline_days: int, document: str) -> str:
        rec.record("submit_decision", {"case_id": case_id, "action": action, "approver": approver, "deadline_days": deadline_days, "document": document})
        return f"Decision recorded for {case_id}."

    async def finish(case_id: str) -> str:
        rec.record("finish", {"case_id": case_id})
        return f"Ticket {case_id} closed."

    async def submit_shift_report(report: str) -> str:
        rec.record("submit_shift_report", {"report": report})
        try:
            parsed = json.loads(report)
            if not isinstance(parsed, dict):
                raise TypeError("the report must be a JSON object")
        except (TypeError, ValueError) as e:
            rec.report_error = str(e)
            return f"Invalid report ({e}); send the JSON object again."
        rec.report, rec.report_error = parsed, None
        return "Report recorded. The shift is over."

    tools = {
        "lookup_customer": _tool("lookup_customer", "Open a customer's file: region, loyalty tier, history and notes.", {"customer_id": s("The customer ID, e.g. CU-12345.")}, lookup_customer),
        "order_lookup": _tool("order_lookup", "Open an order's file: destination region, status, lines and shipment events.", {"order_id": s("The order ID, e.g. O-123456.")}, order_lookup),
        "submit_decision": _tool(
            "submit_decision",
            "Submit the handling decision for a policy case or follow-up.",
            {
                "case_id": s("The case ID, e.g. C-123456."),
                "action": s("What policy requires.", enum=["approve", "deny", "escalate"]),
                "approver": s("Who the request is escalated to; 'none' unless escalating.", enum=["none", *APPROVERS]),
                "deadline_days": ToolParam(type="integer", description="Business days allowed to respond."),
                "document": s("Supporting document the customer must provide; 'none' if not required.", enum=DOCUMENTS),
            },
            submit_decision,
        ),
        "finish": _tool("finish", "Close a ticket once its procedure is complete.", {"case_id": s("The ticket's case ID.")}, finish),
        "submit_shift_report": _tool("submit_shift_report", "Submit the end-of-shift report (a JSON object as a string).", {"report": s("The report as a JSON object.")}, submit_shift_report),
    }
    for t in world.tools:
        def make(name: str) -> Callable:
            # `**kwargs: Any` makes Inspect pass the arguments through; an unannotated **kwargs fails every call
            # ("No type annotation available for parameter kwargs").
            async def call(**kwargs: Any) -> str:
                rec.record(name, kwargs)
                return f"{name} executed for {kwargs.get('order_id', '?')}."
            return call

        tools[t.name] = _tool(t.name, t.description, {k: ToolParam(type="string", description=v) for k, v in t.params.items()}, make(t.name))
    return tools
