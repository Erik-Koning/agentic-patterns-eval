"""Tools for an F8 session, bound to one session's recorder.

Every call is recorded as an event `{item, tool, args}`, where `item` is the queue position being worked on
when the call was made (0 before the first case, N + 1 during the report). Scoring reads only these events, so
it never depends on how a context-management arm shaped the view. Calls of a context policy's own tools (e.g.
`todo_write`) are recorded by the session loop with `policy: True` (and `error` when the tool failed); scoring
looks only at the tools below, so they never change an item's outcome.

The recorder's state (`state_dict` / `load_state`) is part of a mid-session checkpoint. The tools have no side
effects outside it, which is what makes re-running an interrupted item on resume valid.

Real-model robustness:
- Case IDs are normalized (case and surrounding space) before they are recorded; an ID that is not in the
  shift's queue is recorded with `rejected: True` (the taxonomy still sees it) and comes back to the model as
  a tool error instead of an acknowledgement.
- The report is accepted only in the report phase (after the last case) and only in its exact shape:
  {"dispositions": {case_id: str}, "pending_recheck": [str], "open_followups": [str]}. Anything else is a
  tool error the model can correct; the scorer never sees an ill-formed report.
"""

import json
from collections.abc import Callable
from typing import Any

from inspect_ai.tool import ToolDef, ToolError, ToolParam, ToolParams

from . import gen_f8
from .gen_f7 import APPROVERS, DOCUMENTS
from .spec import TaskItem, World

# Sample store keys the session runner writes and the scorer reads.
ITEMS, VIEWS, PROBES, EVENTS, REPORT, OVERFLOW = "f8_items", "f8_views", "f8_probes", "f8_events", "f8_report", "f8_overflow_at"
# Management events (policy logs, threshold reactions), usage totals by kind and model, and resumes
# (`ape.agent.session_checkpoint.resume_summary`; None for a session that ran in one attempt).
CM_EVENTS, USAGE, RESUME = "f8_cm_events", "f8_usage", "f8_resume"


class SessionRecorder:
    """Per-session tool state: the current item and every call made."""

    def __init__(self, world: World):
        self.world = world
        self.item = 0
        self.events: list[dict] = []
        self.report: dict | None = None
        self.report_error: str | None = None

    def record(self, tool: str, args: dict, rejected: bool = False, **extra: Any) -> None:
        self.events.append({"item": self.item, "tool": tool, "args": dict(args), **({"rejected": True} if rejected else {}), **extra})

    def state_dict(self) -> dict:
        return {"item": self.item, "events": self.events, "report": self.report, "report_error": self.report_error}

    def load_state(self, state: dict) -> None:
        self.item, self.events, self.report, self.report_error = state["item"], list(state["events"]), state["report"], state["report_error"]

    def answered(self, task: TaskItem) -> bool:
        """Whether the current case got its answer call in its own window."""
        tool = "finish" if task.tags["kind"] == "ticket" else "submit_decision"
        cid = task.tags["case_id"]
        return any(e["item"] == task.tags["position"] and e["tool"] == tool and e["args"].get("case_id") == cid for e in self.events)


def _tool(name: str, description: str, params: dict[str, ToolParam], fn: Callable) -> ToolDef:
    return ToolDef(fn, name=name, description=description, parameters=ToolParams(properties=params, required=list(params)))


def norm_case_id(case_id: Any) -> str:
    """A case ID as recorded and compared: case and surrounding space ignored ("c-123456 " -> "C-123456")."""
    return str(case_id).strip().upper()


REPORT_KEYS = ("dispositions", "pending_recheck", "open_followups")


def report_problem(parsed: Any) -> str | None:
    """Why a parsed report is not in the required shape, or None when it is."""
    if not isinstance(parsed, dict):
        return "the report must be a JSON object"
    if missing := [k for k in REPORT_KEYS if k not in parsed]:
        return f"missing key(s) {missing}"
    if extra := [k for k in parsed if k not in REPORT_KEYS]:
        return f"unexpected key(s) {extra}; use exactly {list(REPORT_KEYS)}"
    disp = parsed["dispositions"]
    if not isinstance(disp, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in disp.items()):
        return "dispositions must map each case ID (a string) to approve, deny, escalate or resolved (a string)"
    for key in ("pending_recheck", "open_followups"):
        v = parsed[key]
        if not isinstance(v, list) or not all(isinstance(i, str) for i in v):
            return f"{key} must be a list of case IDs (strings), e.g. [\"C-123456\"]"
    return None


def build_session_tools(world: World, rec: SessionRecorder) -> dict[str, ToolDef]:
    s = lambda d, **kw: ToolParam(type="string", description=d, **kw)  # noqa: E731
    queue = set(gen_f8.session(world)["queue"])
    n_cases = len(world.tasks)

    def case(tool: str, case_id: Any, args: dict) -> str:
        """Record an answer call under the normalized case ID; an ID outside the queue is recorded as rejected and
        raised as a tool error."""
        cid = norm_case_id(case_id)
        known = cid in queue
        rec.record(tool, {"case_id": cid, **args}, rejected=not known)
        if not known:
            raise ToolError(f"Unknown case ID {str(case_id)!r}: it is not in this shift's queue. Use the case ID from the case message, e.g. C-123456.")
        return cid

    async def lookup_customer(customer_id: str) -> str:
        rec.record("lookup_customer", {"customer_id": customer_id})
        return gen_f8.customer_file(world, customer_id)

    async def order_lookup(order_id: str) -> str:
        rec.record("order_lookup", {"order_id": order_id})
        return gen_f8.order_file(world, order_id)

    async def submit_decision(case_id: str, action: str, approver: str, deadline_days: int, document: str) -> str:
        cid = case("submit_decision", case_id, {"action": action, "approver": approver, "deadline_days": deadline_days, "document": document})
        return f"Decision recorded for {cid}."

    async def finish(case_id: str) -> str:
        cid = case("finish", case_id, {})
        return f"Ticket {cid} closed."

    async def submit_shift_report(report: str) -> str:
        rec.record("submit_shift_report", {"report": report})
        if rec.item <= n_cases:
            rec.report_error = "submitted before the end of the shift"
            raise ToolError("The shift is not over: submit the end-of-shift report when you are asked for it, after the last case.")
        try:
            parsed = json.loads(report)
        except (TypeError, ValueError) as e:
            parsed, problem = None, f"not valid JSON ({e})"
        else:
            problem = report_problem(parsed)
        if problem:
            rec.report_error = problem
            raise ToolError(f"Invalid report: {problem}. Send the JSON object again as a string.")
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
