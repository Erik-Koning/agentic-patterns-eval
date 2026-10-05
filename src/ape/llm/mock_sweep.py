"""Scripted stand-ins for the agent in context-sweep dry runs (`mockllm` `custom_outputs`).

`gold_sweep_agent(worlds)` knows every given world's probe and submits its gold decision for the last case the
history opens (one generation). With `refuse_from` it plays the provider refusing every request whose world targets
that size or more, as Inspect renders OpenAI's `context_length_exceeded` (empty output, stop reason `model_length`), so
a dry run exercises the not-measured path. `naive_sweep_agent` decides every probe the same way (approve, no approver,
5 days, no document), so a dry run exercises the failure labels. Dry runs check plumbing, not accuracy.
"""

import re
from collections.abc import Iterable

from inspect_ai.model import ModelOutput

from ..worlds.spec import World

MODEL = "mockllm/model"
CASE = re.compile(r"Case (C-\d{6})")


def _probe_case(messages) -> str | None:
    for m in reversed(messages):
        if m.role == "user" and (hits := CASE.findall(m.text)):
            return hits[0]
    return None


REFUSAL = "This model's maximum context length is 1050000 tokens. However, your messages resulted in more tokens."


def gold_sweep_agent(worlds: Iterable[World], refuse_from: int | None = None):
    """The gold decision for the probe; with `refuse_from`, a size refusal for every world targeting that size or more.
    A probe case ID is shared by every size of its task, so the size comes from the conversation's length."""
    from ..worlds import gen_sweep

    worlds = list(worlds)
    gold = {w.tasks[-1].tags["case_id"]: w.tasks[-1].gold for w in worlds}
    # (probe case, history length) -> target size: the sizes of a task share the probe, not the history's length.
    target = {(w.tasks[-1].tags["case_id"], len(gen_sweep.history(w))): int(w.entities["sweep"]["target_tokens"]) for w in worlds} if refuse_from else {}

    def agent(messages, tools, tool_choice, config) -> ModelOutput:
        cid = _probe_case(messages)
        if cid is None or cid not in gold:
            return ModelOutput.from_content(MODEL, "No case to decide.")
        if refuse_from is not None and target.get((cid, len(messages)), 0) >= refuse_from:
            return ModelOutput.from_content(MODEL, REFUSAL, stop_reason="model_length")
        return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": cid, **gold[cid]})

    return agent


def naive_sweep_agent(messages, tools, tool_choice, config) -> ModelOutput:
    cid = _probe_case(messages)
    if cid is None:
        return ModelOutput.from_content(MODEL, "No case to decide.")
    return ModelOutput.for_tool_call(MODEL, "submit_decision", {"case_id": cid, "action": "approve", "approver": "none", "deadline_days": 5, "document": "none"})
