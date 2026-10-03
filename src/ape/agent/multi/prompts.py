"""Every prompt the multi-agent and ensemble arms add, in one place, so they can be reviewed and swapped as a set (the
brief asks for an independent author of the M-arm prompts before the pilot).

Rules these drafts follow, which a replacement must keep:
- **The system prompt is the single-agent arms' own** (`kb_react.system_prompt`: base prompt, knowledge note and, for
  per-query delivery, the knowledge itself) for every agent. Role text goes in the agent's first user message, after
  the task or subtask, so knowledge delivery stays one switch (DEL) shared by every agent of an arm.
- **Nothing says whether workers run in parallel or in series.** M1 and M1s must give the model identical inputs.
- **Neutral wording:** no claim that a role or the team is better at anything; each prompt states only the protocol
  (what the agent receives, what it may call, what happens to its output).
- **The planning turn offers `plan` alone** (Inspect sends only the forced tool), so the S9 and orchestrator notes
  name the tools that come after it: S9's own, or the workers'.
- The role-detection in the gold mock (`llm/mock_multi.py`) keys on tool names, never on this prose.

Templates use `str.format` fields; every field a template uses is passed by the caller in `agent/multi`.
"""

# S9: plan-then-execute in one context (DEC=1, ISO=0, CTRL=dynamic).
S9_NOTE = (
    "Work in two phases. First call plan with the list of subtasks you will carry out. Then carry them out yourself, "
    "one after another, with your tools ({tools}); call plan again whenever the plan needs to change. Finish by "
    "calling {answer_tool}."
)

# The orchestrator of M1, M1s, M1k and M2 (DEC=1, ISO=1). {team} is TEAM_IDENTICAL or TEAM_SPECIALISTS.
ORCHESTRATOR_NOTE = (
    "You coordinate a team of {n_workers} workers. You cannot use the task's tools yourself, apart from "
    "{answer_tool}.\n"
    "{team}\n"
    "How to work:\n"
    "1. First call plan with the list of subtasks you intend to have carried out.\n"
    "2. Call delegate with up to {n_workers} subtasks at a time. Each subtask runs in a fresh worker context: the "
    "worker sees only the text of its subtask (not this conversation, the plan or the other subtasks), so write each "
    "subtask so that it stands on its own, with every ID and detail it needs. delegate returns each worker's result. "
    "Call delegate as often as you need, and call plan again whenever the plan needs to change.\n"
    "3. When you have what you need, call {answer_tool} with the final answer."
)
TEAM_IDENTICAL = "Every worker has the same knowledge and these tools: {tools}."
TEAM_SPECIALISTS = (
    "Your workers are specialists. Name the specialist for each subtask; a specialist has only its own tools.\n{roster}"
)
ROSTER_LINE = "- {name}: covers {covers}. Tools: {tools}."

# A worker's first user message (M1, M1s, M1k; M2 adds WORKER_SPECIALTY).
WORKER_NOTE = (
    "You are a worker on a team. Carry out the subtask below and then call report with your result. The result is "
    "all the requester will see of your work, so include every value the subtask asks for. You do not submit the "
    "team's final answer."
)
WORKER_SPECIALTY = "You are the team's specialist for: {covers}."
WORKER_SUBTASK = "{note}\n\nSubtask:\n{subtask}"
REPORT_DESCRIPTION = "Report the result of your subtask to the requester. Call it once, when the subtask is done."

# Tool descriptions for the planning and delegation tools.
PLAN_DESCRIPTION = "Record the plan: the list of subtasks, in the order they will be carried out. Call again to revise it."
DELEGATE_DESCRIPTION = (
    "Have workers carry out up to {n_workers} subtasks. Each subtask runs in a fresh worker context that sees only "
    "the subtask's text. Returns each worker's result, in the order of the subtasks."
)
DELEGATE_SPECIALIST_DESCRIPTION = (
    "Have specialists carry out up to {n_workers} subtasks. Each subtask runs in a fresh context of the named "
    "specialist that sees only the subtask's text. Returns each result, in the order of the subtasks."
)

# What delegate returns: one line per subtask, in subtask order.
RESULT_LINE = "Subtask {n} ({worker}): {result}"
RESULT_TEXT = "[no report; the worker stopped with this message:] {text}"
RESULT_TURN_CAP = "[no report: the worker reached its turn limit; its last message:] {text}"
RESULT_ERROR = "[no result: the worker failed with an error]"

# Tool errors the model sees (the call does nothing).
PLAN_EMPTY = "Pass the plan as a non-empty list of subtasks."
DELEGATE_COUNT = "Pass between 1 and {n_workers} subtasks per call; you passed {got}. Send the others in another call."
DELEGATE_EMPTY_TASK = "Subtask {n} is empty: write out what the worker should do."
DELEGATE_SPECIALIST = "Subtask {n}: name one of the specialists ({names}) and give the task text."
REPORT_AGAIN = "A result was already reported; it stands and this one is ignored."
PROPOSAL_AGAIN = "Your proposal for this round is already recorded; it stands and this one is ignored."
SELECT_AGAIN = "A candidate was already chosen; it stands and this one is ignored."
SELECT_RANGE = "Choose a candidate number from 1 to {n}."

# M7 council: k members, R critique rounds, a chair (COMM=real).
COUNCIL_MEMBER_NOTE = (
    "You are one of {k} council members working on this task independently. Instead of submitting a final answer, "
    "propose one: call {answer_tool} with your proposed answer and a short rationale (the evidence it rests on). The "
    "other members will review your proposal, and you will review theirs."
)
CRITIQUE_ROUND = (
    "Critique round {round} of {rounds}. The other members' current proposals:\n\n{proposals}\n\n"
    "Check them against the evidence and your own work (you may use the tools again), then call {answer_tool} with "
    "your proposal for this round, kept or revised, and its rationale."
)
CHAIR_NOTE = (
    "You chair a council of {k} members who each worked on this task. Their final proposals, after {rounds} rounds "
    "of critique:\n\n{proposals}\n\nDecide the final answer and submit it with {answer_tool}."
)
PROPOSAL_LINE = "Member {member}: proposed answer: {answer}\nRationale: {rationale}"
NO_PROPOSAL = "Member {member}: no proposal."
RATIONALE_PARAM = "Why you propose this answer: the evidence it rests on, in a few sentences."
PROPOSE_SUFFIX = " (Council: this records your proposal; the chair submits the final answer.)"

# S8k3: an LLM aggregator chooses among tied candidates (majority vote decides otherwise).
AGGREGATOR_NOTE = (
    "{k} independent attempts at this task produced these candidate answers, with no majority among them:\n\n"
    "{candidates}\n\nChoose the candidate most likely to be correct and call select_answer with its number."
)
CANDIDATE_LINE = "Candidate {n}: {answer}"
SELECT_DESCRIPTION = "Choose the final answer: the number of one of the candidates."

# Nudges after a text-only reply (one per text-only streak, as in kb_react).
NUDGE_TASK = "Use the tools to complete the task, then call {answer_tool}."
NUDGE_WORKER = "Use the tools to complete the subtask, then call report with your result."
NUDGE_PROPOSE = "Call {answer_tool} with your proposed answer and its rationale."
NUDGE_SUBMIT = "Call {tool} now."
