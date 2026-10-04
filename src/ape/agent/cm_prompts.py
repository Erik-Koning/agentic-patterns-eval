"""Every prompt and view header the context-management arms use (CONTEXT_MANAGEMENT_AUDIT §6), in one place.

Fairness (audit §6.3): each arm gets only a short, fixed text for its mechanism. The texts are neutral: they ask
for what the shift's own rules make later cases depend on (the system prompt states those rules to every arm) and
never for an answer. The summary, the handoff note and the todo extraction ask for the same content
(`WHAT_TO_KEEP`), so the arms differ in mechanism, not in instructions. Management calls see a plain-text
transcript (`render_transcript`), never tool-call messages, so they need no tool definitions on any provider.

The session mocks (`ape.llm.mock_session`) tell management calls apart by the first line of these prompts.
"""

import json
from collections.abc import Sequence

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool

WHAT_TO_KEEP = (
    "For every case handled so far: its ID, what was asked (the customer, the kind of request and the amount, or "
    "the ticket's order) and the decision submitted or the tool calls made. Every memo announced, word for word, "
    "and whether it has been withdrawn since. How many requests each customer has had approved this shift. The "
    "cases still waiting for a follow-up, each with its original request. The cases escalated. The cases still "
    "pending."
)

SUMMARY_PROMPTS = {
    # The default: the content the shift rules make later cases need.
    "structured": (
        "Write notes on this support shift so far.\n"
        "The notes replace the transcript below and any earlier notes: whoever continues the shift will see only "
        "the notes, the shift rules and the next case. Keep:\n"
        f"{WHAT_TO_KEEP}\n"
        "Be exact and complete; do not add anything the transcript does not say. Plain text."
    ),
    # A generic summary request (a tuning alternative): no list of what to keep.
    "plain": (
        "Write notes on this support shift so far.\n"
        "Summarise the transcript below and any earlier notes so that the work can continue from your notes "
        "alone. Plain text."
    ),
}

HANDOFF_PROMPT = (
    "Write a handoff note for this support shift.\n"
    "A colleague takes over the shift now and will see only your note, the shift rules, their todo list if they "
    "keep one, and the next case, not the transcript below. Keep:\n"
    f"{WHAT_TO_KEEP}\n"
    "Be exact and complete; do not add anything the transcript does not say. Plain text."
)

TODO_EXTRACT_PROMPT = (
    "Update the todo list for this support shift.\n"
    "Read the message below and return the complete updated todo list: keep every existing entry with its "
    "status, add an entry for each case, memo or follow-up the message introduces, and mark memos the message "
    "withdraws as completed. If the message opens a case, mark any case still in_progress as completed (a case "
    "ends with its answer) and the case it opens as in_progress. Reply with only a JSON array of objects "
    '{"content": "...", "status": "pending" | "in_progress" | "completed"}.'
)

# A case ends the moment its answer (submit_decision or finish) is recorded: the loop gives no later turn in it, so
# the list is updated at the start of the next case (review A-3). The same text for every arm with a todo list.
TODO_ADDENDUM = (
    "## Todo list\n"
    "Keep a todo list for this shift with the todo_write tool: an entry for each case in the queue, and entries for "
    "the memos in force and the cases waiting for a follow-up. A case ends as soon as you submit its answer, with no "
    "later turn in it, so update the list at the start of each case: first mark the previous case completed and this "
    "one in_progress, then work on the case. Your current todo list is shown to you with each case, also when earlier "
    "parts of the shift are no longer shown."
)

NATIVE_MOCK_PROMPT = (
    "Compact the conversation above.\n"
    "Return a compact representation that preserves everything the rest of the conversation may need."
)

SUMMARY_HEADER = "Notes on the shift so far (they replace the earlier messages, which are no longer shown):"
HANDOFF_HEADER = "Handoff note for the rest of the shift (the earlier messages are no longer shown):"
TODO_HEADER = "Your todo list (kept with todo_write):"
NO_NOTES = "(no notes were written)"

# First lines, by purpose: how a mock (or a log reader) tells management calls apart.
PURPOSES = {
    "summary": "Write notes on this support shift so far.",
    "handoff": "Write a handoff note for this support shift.",
    "todo_extract": "Update the todo list for this support shift.",
    "native_mock": "Compact the conversation above.",
}


def purpose_of(text: str) -> str | None:
    """Which management prompt a request carries (None for anything else)."""
    for purpose, first in PURPOSES.items():
        if first in text:
            return purpose
    return None


def render_transcript(messages: Sequence[ChatMessage]) -> str:
    """Messages as plain text for a management call: user messages, the assistant's text and tool calls, tool
    results (and tool errors)."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, ChatMessageAssistant):
            if m.text:
                lines.append(f"ASSISTANT: {m.text}")
            for c in m.tool_calls or []:
                lines.append(f"ASSISTANT called {c.function}({json.dumps(c.arguments, sort_keys=True)})")
        elif isinstance(m, ChatMessageTool):
            lines.append(f"TOOL {m.function} error: {m.error.message}" if m.error else f"TOOL {m.function} result: {m.text}")
        else:
            lines.append(f"{m.role.upper()}: {m.text}")
    return "\n".join(lines)


def management_request(prompt: str, sections: Sequence[tuple[str, str]]) -> str:
    """A management call's single user message: the prompt, then labelled sections."""
    parts = [prompt]
    for label, body in sections:
        parts.append(f"## {label}\n{body.strip() or '(none)'}")
    return "\n\n".join(parts)
