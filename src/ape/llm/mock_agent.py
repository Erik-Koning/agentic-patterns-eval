"""Scripted stand-in for the agent model in dry runs (`mockllm` `custom_outputs`).

It exercises the full loop — a lookup call, then an answer-tool call — without any
model. Answers are deliberately naive; dry runs check plumbing, not accuracy.
"""

import json
import re

from inspect_ai.model import ChatMessageTool, ModelOutput

MODEL = "mockllm/model"


def mock_agent(messages, tools, tool_choice, config) -> ModelOutput:
    names = {t.name for t in tools}
    done = {m.function for m in messages if isinstance(m, ChatMessageTool)}
    task_text = next((m.text for m in messages if m.role == "user"), "")
    if "lookup_customer" in names and "lookup_customer" not in done:
        cid = re.search(r"CU-\d+", task_text)
        return ModelOutput.for_tool_call(MODEL, "lookup_customer", {"customer_id": cid.group(0) if cid else "CU-0"})
    if "submit_decision" in names:
        return ModelOutput.for_tool_call(MODEL, "submit_decision", {"action": "approve", "approver": "none", "deadline_days": 1, "document": "none"})
    if "order_lookup" in names and "order_lookup" not in done:
        oid = re.search(r"O-\d+", task_text)
        return ModelOutput.for_tool_call(MODEL, "order_lookup", {"order_id": oid.group(0) if oid else "O-0"})
    if "finish" in names:
        return ModelOutput.for_tool_call(MODEL, "finish", {})
    if "submit_answer" in names:
        return ModelOutput.for_tool_call(MODEL, "submit_answer", {"answer": "unknown"})
    return ModelOutput.from_content(MODEL, json.dumps({"note": "no tools"}))
