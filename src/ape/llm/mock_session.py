"""Scripted stand-in for the agent in F8 session dry runs (`mockllm` `custom_outputs`).

Per case it makes the lookup, then the answer call; at the end it submits a report from the decisions it
made; probes get a JSON answer built from the tool results it can see. Answers are deliberately naive: dry
runs check plumbing, not accuracy.
"""

import json
import re

from inspect_ai.model import ChatMessageTool, ModelOutput

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


def mock_session_agent(messages, tools, tool_choice, config) -> ModelOutput:
    if config.response_schema is not None and config.response_schema.name == "state_probe":
        start = next((m.text for m in messages if m.role == "user" and "Shift start" in m.text), "")
        queue, done = re.findall(r"C-\d{6}", start), _done(messages)
        answer = {"completed": done, "pending": [c for c in queue if c not in done], "memos_in_force": [], "open_followups": [], "escalated": []}
        return ModelOutput.from_content(MODEL, json.dumps(answer))
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
