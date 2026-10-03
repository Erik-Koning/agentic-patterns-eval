"""Every prompt the session topology arms (M1, M2 in Study G; `session_team.py`) add, in one place, for review and
replacement as a set. Kept apart from `multi/prompts.py` (the main study's arms) so the two can change independently.

Rules a replacement must keep (as for the main study's prompts):
- **Every agent's system prompt is the session's own** (`gen_f8.system_prompt`: base prompt, the corpus, the shift
  rules). The orchestrator's team instructions are its policy addendum (appended to that system prompt, as any
  context policy's addendum is); a worker's role text goes in its first user message.
- **Neutral wording:** the protocol only (what each agent receives, what it may call, what happens to its output).
- M2's text differs from M1's only where specialization does (the roster, the specialist field, the specialty
  sentence), so M1 -> M2 is one switch.
- The gold mock (`llm/mock_session.py`) recognises roles by tool names, never by this prose.

Templates use `str.format`; every field is passed by `session_team.py`.
"""

ORCHESTRATOR_ADDENDUM = (
    "## Team\n"
    "You coordinate a team of {n_workers} workers for this shift. You cannot open customer or order files or run "
    "procedure tools yourself; workers do that.\n"
    "{team}\n"
    "For each case:\n"
    "1. Call delegate with up to {n_workers} subtasks. Each subtask runs in a fresh worker context that sees only the "
    "subtask's text, not this conversation, earlier cases, memos or the shift's state, so write each subtask so that it "
    "stands on its own, with every ID, memo and detail it needs. delegate returns each worker's result. Call it as often "
    "as you need.\n"
    "2. Answer the case yourself with submit_decision or finish and the case ID. A ticket's procedure calls must be made "
    "by a worker before you call finish.\n"
    "You keep the shift's state (memos, approvals, follow-ups) and submit the end-of-shift report yourself."
)
TEAM_IDENTICAL = "Every worker has the same knowledge and these tools: {tools}."
TEAM_SPECIALISTS = "Your workers are specialists. Name the specialist for each subtask; a specialist has only its own tools.\n{roster}"
ROSTER_LINE = "- {name}: covers {covers}. Tools: {tools}."

WORKER_NOTE = (
    "You are a worker on a support team. Carry out the subtask below with your tools, then call report with your "
    "result. The result is all the requester will see of your work, so include every value the subtask asks for. You do "
    "not submit decisions, close tickets or write the shift report."
)
WORKER_SPECIALTY = "You are the team's specialist for: {covers}."
WORKER_SUBTASK = "{note}\n\nSubtask:\n{subtask}"
NUDGE_WORKER = "Use your tools to complete the subtask, then call report with your result."
REPORT_DESCRIPTION = "Report the result of your subtask to the requester. Call it once, when the subtask is done."
REPORT_AGAIN = "A result was already reported; it stands and this one is ignored."

DELEGATE_DESCRIPTION = (
    "Have workers carry out up to {n_workers} subtasks for the current case. Each subtask runs in a fresh worker "
    "context that sees only the subtask's text. Returns each worker's result, in the order of the subtasks."
)
DELEGATE_SPECIALIST_DESCRIPTION = (
    "Have specialists carry out up to {n_workers} subtasks for the current case. Each subtask runs in a fresh context "
    "of the named specialist that sees only the subtask's text. Returns each result, in the order of the subtasks."
)
RESULT_LINE = "Subtask {n} ({worker}): {result}"
RESULT_TEXT = "[no report; the worker stopped with this message:] {text}"
RESULT_TURN_CAP = "[no report: the worker reached its turn limit; its last message:] {text}"
DELEGATE_COUNT = "Pass between 1 and {n_workers} subtasks per call; you passed {got}. Send the others in another call."
DELEGATE_EMPTY_TASK = "Subtask {n} is empty: write out what the worker should do."
DELEGATE_SPECIALIST = "Subtask {n}: name one of the specialists ({names}) and give the task text."
