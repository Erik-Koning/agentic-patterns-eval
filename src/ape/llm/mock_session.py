"""Scripted stand-ins for the agent in F8 session dry runs (`mockllm` `custom_outputs`).

`mock_session_agent` is naive: per case it makes the lookup, then the answer call; at the end it submits a report
from the decisions it made; probes get a JSON answer built from the tool results it can see. A context policy's
management calls (no tools, no probe schema) get, by purpose (`cm_prompts.purpose_of`): a todo list (JSON) for an
extraction, a short text for a summary, a handoff note or a mock compaction. Its answers are deliberately naive:
dry runs check plumbing, not accuracy.

`gold_session_agent(world)` knows the world's gold and plays every role perfectly: the reference trajectory's
lookup, then the gold decision or ticket calls for the case its view shows (it needs only the current case's
messages, which every arm keeps), the gold report, the true state for probes, the oracle state
(`render_oracle_state`) as summaries, handoff notes and mock compactions, the true todo list for extractions, and,
where the arm offers `todo_write`, one todo update at the start of each case. It counts the cases it answered (a
closure), so one instance serves one session.

For the topology arms (`ape.agent.multi.session_team`) it also plays the team, recognised by tool names:
- **orchestrator** (offered `delegate`): per case, one subtask (open the customer's file, or the ticket's order
  file and make its procedure calls), routed under M2 to the specialist covering the case's domain; then, once the
  worker's result names the customer (or reports the ticket's calls), the gold decision or `finish`. A follow-up is
  decided without a worker (the reference trajectory makes no lookup for it). The report is the gold report.
- **worker** (offered `report`): the lookup its subtask names, a ticket's gold calls for that order (only those its
  tools allow), then a report that carries what it read (the customer's region and tier, the calls made).
"""

import json
import re

from inspect_ai.model import ChatMessageTool, ModelOutput

from ..agent.cm_prompts import purpose_of
from ..worlds import gen_f8

MODEL = "mockllm/model"
CASE = re.compile(r"(?:Case|Ticket) (C-\d{6})")


def _current(messages) -> tuple[int, str]:
    """Index of the last user message that opens a case or asks for the report, and the text of that case (an
    O-state view prefixes it with the oracle state, which mentions other customers and orders)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "user" and "End of shift" in m.text:
            return i, m.text
        if m.role == "user" and (hits := list(CASE.finditer(m.text))):
            return i, m.text[hits[-1].start() :]
    return 0, ""


def _done(messages) -> list[str]:
    acks = " ".join(m.text for m in messages if isinstance(m, ChatMessageTool))
    return list(dict.fromkeys(re.findall(r"(?:Decision recorded for|Ticket) (C-\d{6})", acks)))


def _section(text: str, label: str) -> str:
    m = re.search(rf"## {re.escape(label)}\n(.*?)(?=\n## |\Z)", text, re.S)
    return m.group(1) if m else ""


def _naive_management(messages) -> str:
    text = messages[-1].text if messages else ""
    purpose = purpose_of(text)
    if purpose == "todo_extract":
        try:
            todos = json.loads(_section(text, "Current todo list"))
        except ValueError:
            todos = []
        todos = todos if isinstance(todos, list) else []
        known = " ".join(str(t.get("content", "")) for t in todos if isinstance(t, dict))
        message = _section(text, "Message")
        for cid in dict.fromkeys(re.findall(r"C-\d{6}", message)):
            if cid not in known:
                todos.append({"content": f"Case {cid}", "status": "pending"})
        for mid in dict.fromkeys(re.findall(r"\bM-\d+\b", message)):
            todos.append({"content": f"Memo {mid}", "status": "in_progress"})
        return json.dumps(todos)
    if purpose == "native_mock":
        return "compacted context"
    done = list(dict.fromkeys(re.findall(r"Decision recorded for (C-\d{6})", text))) or _done(messages)
    return "Shift notes: cases answered so far: " + (", ".join(done) or "none") + "."


def mock_session_agent(messages, tools, tool_choice, config) -> ModelOutput:
    if config.response_schema is not None and config.response_schema.name == "state_probe":
        start = next((m.text for m in messages if m.role == "user" and "Shift start" in m.text), "")
        queue, done = re.findall(r"C-\d{6}", start), _done(messages)
        answer = {"completed": done, "pending": [c for c in queue if c not in done], "memos_in_force": [], "open_followups": [], "escalated": []}
        return ModelOutput.from_content(MODEL, json.dumps(answer))
    if not tools:  # a management call on the cm role (agent calls always carry the session's tools)
        return ModelOutput.from_content(MODEL, _naive_management(messages))
    i, text = _current(messages)
    since = {m.function for m in messages[i + 1 :] if isinstance(m, ChatMessageTool)}
    if "End of shift" in text:
        report = {"dispositions": dict.fromkeys(_done(messages), "approve"), "pending_recheck": [], "open_followups": []}
        return ModelOutput.for_tool_call(MODEL, "submit_shift_report", {"report": json.dumps(report)})
    cid = CASE.search(text).group(1)
    decision = {"case_id": cid, "action": "approve", "approver": "none", "deadline_days": 1, "document": "none"}
    if "follow-up to case" in text:
        return ModelOutput.for_tool_call(MODEL, "submit_decision", decision)
    if (cust := re.search(r"customer (CU-\d+)", text)) and "lookup_customer" not in since:
        return ModelOutput.for_tool_call(MODEL, "lookup_customer", {"customer_id": cust.group(1)})
    if (order := re.search(r"order (O-\d+)", text)) and "order_lookup" not in since:
        return ModelOutput.for_tool_call(MODEL, "order_lookup", {"order_id": order.group(1)})
    if order:
        return ModelOutput.for_tool_call(MODEL, "finish", {"case_id": cid})
    return ModelOutput.for_tool_call(MODEL, "submit_decision", decision)


def gold_todos(world, k: int, current: str | None = None) -> list[dict]:
    """The true todo list after k cases (the current case in progress)."""
    st = gen_f8.state_at(world, k)
    todos = [{"content": f"Case {c}", "status": "completed"} for c in st["completed"]]
    todos += [{"content": f"Case {c}", "status": "in_progress" if c == current else "pending"} for c in st["pending"]]
    todos += [{"content": f"Memo {m} in force", "status": "in_progress"} for m in st["memos_in_force"]]
    todos += [{"content": f"Case {c} waits for a follow-up", "status": "pending"} for c in st["open_followups"]]
    return todos


TEAM_CUSTOMER = "Case {cid}: open customer {customer}'s file and report the customer's region and loyalty tier."
TEAM_TICKET = (
    "Ticket {cid}: open order {order}'s file, then make exactly the procedure calls its standard operating procedure "
    "requires for order {order} (memos in force: {memos}), and report the calls you made."
)


def _worker(world, messages, names: set[str]) -> ModelOutput:
    """The gold worker: its subtask's lookup, a ticket's gold calls, then a report of what it read."""
    sub = next(m.text for m in messages if m.role == "user").split("Subtask:", 1)[-1]
    called = [m for m in messages if isinstance(m, ChatMessageTool) and not m.error]
    customers = gen_f8.session(world)["customers"]
    if cust := re.search(r"customer (CU-\d+)", sub):
        cu = cust.group(1)
        if "lookup_customer" not in {m.function for m in called}:
            return ModelOutput.for_tool_call(MODEL, "lookup_customer", {"customer_id": cu})
        c = customers.get(cu, {})
        return ModelOutput.for_tool_call(MODEL, "report", {"result": f"Customer {cu}: region {c.get('region')}, tier {c.get('tier')}."})
    if order := re.search(r"order (O-\d+)", sub):
        oid = order.group(1)
        if "order_lookup" not in {m.function for m in called}:
            return ModelOutput.for_tool_call(MODEL, "order_lookup", {"order_id": oid})
        task = next(t for t in world.tasks if t.tags.get("order_id") == oid and t.tags["kind"] == "ticket")
        made = sum(m.function in {t.name for t in world.tools} for m in called)
        calls = task.gold["calls"]
        if made < len(calls):
            if calls[made]["tool"] not in names:
                return ModelOutput.for_tool_call(MODEL, "report", {"result": f"I do not have the tool {calls[made]['tool']}."})
            return ModelOutput.for_tool_call(MODEL, calls[made]["tool"], calls[made]["args"])
        return ModelOutput.for_tool_call(MODEL, "report", {"result": f"Order {oid}: made " + ", ".join(c["tool"] for c in calls) + "."})
    return ModelOutput.for_tool_call(MODEL, "report", {"result": "I could not read the subtask."})


def _team_subtask(world, task, specialized: bool):
    """The gold orchestrator's subtask for a case (routed to its specialist under M2)."""
    from ..agent.multi.specialists import specialists

    tags = task.tags
    if tags["kind"] == "ticket":
        memos = [m["id"] for m in gen_f8.session(world)["memos"] if gen_f8.memo_active(m, tags["position"])]
        text, domain = TEAM_TICKET.format(cid=tags["case_id"], order=tags["order_id"], memos=", ".join(memos) or "none"), next(
            p.domain for p in world.procedures if p.id == tags["procedure"])
    else:
        text, domain = TEAM_CUSTOMER.format(cid=tags["case_id"], customer=tags["customer_id"]), tags["domain"]
    if not specialized:
        return text
    spec = next(sp for sp in specialists(world) if domain in sp.covers)
    return {"specialist": spec.name, "task": text}


def gold_session_agent(world, *, todo_updates: bool = True):
    """A gold-knowing session model for every role (module docstring)."""
    by_case = {t.tags["case_id"]: t for t in world.tasks}
    mutating = {t.name for t in world.tools}
    answered: dict[str, None] = {}

    def agent(messages, tools, tool_choice, config) -> ModelOutput:
        k = len(answered)
        if config.response_schema is not None:
            return ModelOutput.from_content(MODEL, json.dumps(gen_f8.state_at(world, k)))
        if not tools:
            if purpose_of(messages[-1].text if messages else "") == "todo_extract":
                return ModelOutput.from_content(MODEL, json.dumps(gold_todos(world, k)))
            return ModelOutput.from_content(MODEL, gen_f8.render_oracle_state(world, k))
        names = {t.name for t in tools}
        if "report" in names:
            return _worker(world, messages, names)
        i, text = _current(messages)
        if "End of shift" in text:
            return ModelOutput.for_tool_call(MODEL, "submit_shift_report", {"report": json.dumps(world.entities["session"]["report_gold"])})
        cid = CASE.search(text).group(1)
        task, since = by_case[cid], messages[i + 1 :]
        called = {m.function for m in since if isinstance(m, ChatMessageTool) and not m.error}
        if "delegate" in names:  # the team's orchestrator
            results = " ".join(m.text for m in since if isinstance(m, ChatMessageTool) and m.function == "delegate" and not m.error)
            if task.tags["kind"] != "followup" and not results:
                specialized = next(t for t in tools if t.name == "delegate").parameters.properties["subtasks"].items.type == "object"
                return ModelOutput.for_tool_call(MODEL, "delegate", {"subtasks": [_team_subtask(world, task, specialized)]})
            needed = task.tags.get("order_id") if task.tags["kind"] == "ticket" else task.tags.get("customer_id")
            if task.tags["kind"] != "followup" and needed not in results:
                return ModelOutput.from_content(MODEL, "The worker's result does not cover this case.")
            answered[cid] = None
            if task.tags["kind"] == "ticket":
                return ModelOutput.for_tool_call(MODEL, "finish", {"case_id": cid})
            return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": cid, **task.gold})
        if todo_updates and "todo_write" in {t.name for t in tools} and "todo_write" not in called:
            return ModelOutput.for_tool_call(MODEL, "todo_write", {"todos": gold_todos(world, k, current=cid)})
        # The reference trajectory's lookups (gen_f8.reference_steps), so views carry the bulky tool files.
        if task.tags["kind"] in ("policy", "repeat") and "lookup_customer" not in called:
            return ModelOutput.for_tool_call(MODEL, "lookup_customer", {"customer_id": task.tags["customer_id"]})
        if task.tags["kind"] == "ticket" and "order_lookup" not in called:
            return ModelOutput.for_tool_call(MODEL, "order_lookup", {"order_id": task.tags["order_id"]})
        if task.tags["kind"] == "ticket":
            calls = task.gold["calls"]
            done = sum(isinstance(m, ChatMessageTool) and m.function in mutating and not m.error for m in since)
            if done < len(calls):
                return ModelOutput.for_tool_call(MODEL, calls[done]["tool"], calls[done]["args"])
            answered[cid] = None
            return ModelOutput.for_tool_call(MODEL, "finish", {"case_id": cid})
        answered[cid] = None
        return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": cid, **task.gold})

    return agent
