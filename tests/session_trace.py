"""A compact, deterministic trace of one F8 session run, for the CM0 / O-state regression test.

`trace_session` runs the real `f8_session` task on mockllm (`mock_session_agent`) and returns every model input
the session made (agent calls and probes, as hashes of role, text, tool calls and tool errors, with view tokens)
and the store records the scorer and the analysis read, restricted to the fields that existed before the
ContextPolicy layer (B7). The golden file `fixtures/f8_session_golden.json` was written by this module from the
pre-B7 code (commit 6f30eda); `test_session_policy` compares the current code against it, so the layer is proved
to change neither the views nor the records of CM0 and O-state.

Message and tool-call IDs are random per run and are left out; everything else in a view is hashed.
"""

import hashlib
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ModelOutput, get_model

from ape.agent.session import view_tokens
from ape.llm.mock_session import CASE, MODEL, mock_session_agent
from ape.tasks.study_g import f8_session

GOLDEN = Path(__file__).parent / "fixtures" / "f8_session_golden.json"
# (name, nudges, task kwargs): the two arms at the default window, with and without text answers that draw the
# loop's nudges, and CM0 past a small window (overflow).
CASES = (
    ("CM0", False, {"arm": "CM0"}),
    ("O-state", False, {"arm": "O-state"}),
    ("CM0-nudges", True, {"arm": "CM0"}),
    ("O-state-nudges", True, {"arm": "O-state"}),
    ("CM0-overflow", False, {"arm": "CM0", "window": 8000}),
)
# Fields of the pre-B7 records; fields added later are additive and not compared.
ITEM_FIELDS = ("position", "case_id", "kind", "dependency", "dependency_kinds", "generations", "view_tokens_first", "view_tokens_decision", "answered", "success")
VIEW_FIELDS = ("item", "view_tokens")
PROBE_FIELDS = ("k", "answer", "error", "view_tokens", "scores")
ARM_FIELDS = ("name", "window", "max_turns_per_item", "checkpoints")


def _canon(m) -> dict:
    d = {"role": m.role, "text": m.text}
    if isinstance(m, ChatMessageAssistant) and m.tool_calls:
        d["tool_calls"] = [{"function": c.function, "arguments": c.arguments} for c in m.tool_calls]
    if isinstance(m, ChatMessageTool):
        d["function"] = m.function
        d["error"] = m.error.message if m.error else None
    return d


def messages_hash(messages) -> str:
    return hashlib.sha256(json.dumps([_canon(m) for m in messages], sort_keys=True).encode()).hexdigest()[:20]


def _draws_nudge(messages) -> bool:
    """Answer in text (no tool call) when the view ends with the report request, or with the message opening a
    case whose ID ends in 0, 3 or 6: the loop then nudges once. Stateless, so it is the same under every view."""
    last = messages[-1]
    if last.role != "user":
        return False
    if "End of shift" in last.text:
        return True
    opened = CASE.findall(last.text)
    return bool(opened) and int(opened[-1][-1]) % 3 == 0


def trace_session(log_dir: Path, nudges: bool = False, **task_kwargs) -> dict:
    """Run one F8-12 session (worlds must be built) and return its trace."""
    calls: list[dict] = []

    def agent(messages, tools, tool_choice, config):
        kind = "probe" if config.response_schema is not None else "agent"
        calls.append({"kind": kind, "n": len(messages), "sha": messages_hash(messages), "tokens": view_tokens(messages)})
        if nudges and kind == "agent" and _draws_nudge(messages):
            return ModelOutput.from_content(MODEL, "Noted, working on it.")
        return mock_session_agent(messages, tools, tool_choice, config)

    task = f8_session(level="12", split="dev", checkpoints="5,10", **task_kwargs)
    log = inspect_eval(task, model=get_model("mockllm/model", custom_outputs=agent), log_dir=str(log_dir), display="none")[0]
    assert log.status == "success", log.error
    s = log.samples[0]
    st = s.store
    return {
        "calls": calls,
        "items": [{k: i[k] for k in ITEM_FIELDS} for i in st["f8_items"]],
        "views": [{k: v[k] for k in VIEW_FIELDS} for v in st["f8_views"]],
        "probes": [{k: p[k] for k in PROBE_FIELDS} for p in st["f8_probes"]],
        "events": st["f8_events"],
        "report": st["f8_report"],
        "overflow_at": st["f8_overflow_at"],
        "arm": {k: st["arm"][k] for k in ARM_FIELDS},
        "history": {"n": len(s.messages), "sha": messages_hash(s.messages)},
        "score": s.scores["f8_session_score"].value,
    }


def write_golden(log_dir: Path) -> dict:
    golden = {name: trace_session(log_dir, nudges, **kw) for name, nudges, kw in CASES}
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    return golden
