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

**Tuning variants (BUILD_PLAN B3).** The main study's dev tuning gives each arm the same budget of prompt candidates
(brief §7.3). A candidate picks one variant of the role notes (`VARIANTS`, `RoleNotes`) per arm with
`APE_MAS_<ARM>_PROMPT` (`knobs.py`); `default` is the notes above, unchanged, so an arm without the knob runs as B2
built it. Variants follow the rules above, and two more:
- **Protocol only.** A variant changes how the arm's own mechanism is used (how a plan or a subtask is written, what a
  report or a rationale holds, how a critique or the chair's decision proceeds), never generic advice on solving the
  task ("check the policy carefully"), which would help S1 as much and would make an M-arm contrast partly a contrast
  of instructions. S1 has no role text, so it has nothing to tune.
- **The data formats stay fixed:** `WORKER_SUBTASK`, `RESULT_LINE`, `PROPOSAL_LINE` and the tool descriptions are the
  same in every variant, and a variant keeps each note's fields (S9's and the orchestrator's name the tools that follow
  the planning turn).
M5's ledger prompts (`TASK_LEDGER_PROMPT`, `REPLAN_PROMPT`, `PROGRESS_LEDGER_PROMPT` and the block's labels) are one
fixed variant, the same under every variant of the role notes: M5 runs M1's notes plus the ledger, so M1 -> M5 changes
STATE only.
The variants below are drafts; the M-arm prompt author (an independent team member, brief §7.3) reviews or replaces
them, and the candidates in `config/tuning_grid_main.yaml`, before signing off. A text edited after a live tune changes
what that tune measured: `mas_params.prompt_sha` records the text each sample ran.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

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

# M5 and M5-spec: the Magentic-One-style text ledger (STATE=text-ledger; `ledger.py`). One variant: these texts are
# not tuning variants, so the M arms' `RoleNotes` and their `prompt_sha` do not include them (`mas_params.ledger_sha`
# records them). The ledger calls are side calls on the orchestrator's view; their answers are JSON (strict schemas in
# `ledger.py`), and the harness shows the current ledger to the orchestrator at the end of its latest message.
TASK_LEDGER_PROMPT = (
    "Before the work starts, write the task ledger for this task as JSON with these keys, each a list of short "
    "strings: given_facts (facts stated in the task), facts_to_look_up (facts that someone must look up with the "
    "tools), facts_to_derive (facts to work out from those), educated_guesses (guesses to be checked), plan (the steps, "
    "in order)."
)
REPLAN_PROMPT = (
    "The work has not progressed for {stalls} turns. Rewrite the task ledger as JSON with the same keys "
    "(given_facts, facts_to_look_up, facts_to_derive, educated_guesses, plan), using everything above: what is now "
    "known, what is still missing, and a new plan.\n\nThe current task ledger:\n{ledger}"
)
PROGRESS_LEDGER_PROMPT = (
    "Update the progress ledger as JSON with these keys: request_satisfied (true if the task's final answer can now be "
    "submitted), in_loop (true if the same steps are being repeated without new results), progress_being_made (true "
    "if the latest steps produced new results), next_action (what should happen next), instruction (the instruction "
    "for it, e.g. the subtasks to delegate), reason (why, in a sentence or two).\n\nThe task ledger:\n{ledger}"
)
LEDGER_HEADER = "## Ledger (kept by the harness, refreshed every turn)"
LEDGER_TASK = "### Task ledger (version {version})\n{body}"
LEDGER_PROGRESS = "### Progress ledger (turn {turn})\n{body}"
LEDGER_NONE = "(not available)"
LEDGER_FIELDS = {
    "given_facts": "Given facts",
    "facts_to_look_up": "Facts to look up",
    "facts_to_derive": "Facts to derive",
    "educated_guesses": "Educated guesses",
    "plan": "Plan",
    "request_satisfied": "Request satisfied",
    "in_loop": "In a loop",
    "progress_being_made": "Progress being made",
    "next_action": "Next action",
    "instruction": "Instruction",
    "reason": "Reason",
}

# Nudges after a text-only reply (one per text-only streak, as in kb_react).
NUDGE_TASK = "Use the tools to complete the task, then call {answer_tool}."
NUDGE_WORKER = "Use the tools to complete the subtask, then call report with your result."
NUDGE_PROPOSE = "Call {answer_tool} with your proposed answer and its rationale."
NUDGE_SUBMIT = "Call {tool} now."


# --- Tuning variants (BUILD_PLAN B3) ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoleNotes:
    """The role notes a tuning variant sets: S9's note; the orchestrator's and the worker's (M1, M1s, M1k, M2); the
    council member's first note, each critique round's and the chair's (M7). Fields as in the default templates."""

    s9: str
    orchestrator: str
    worker: str
    member: str
    critique: str
    chair: str

    def sha(self) -> str:
        """A short hash of the texts, recorded per sample (`mas_params.prompt_sha`)."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


VARIANTS: dict[str, RoleNotes] = {
    # B2's notes, unchanged.
    "default": RoleNotes(S9_NOTE, ORCHESTRATOR_NOTE, WORKER_NOTE, COUNCIL_MEMBER_NOTE, CRITIQUE_ROUND, CHAIR_NOTE),
    # The same protocol in as few words as it takes.
    "concise": RoleNotes(
        s9=(
            "First call plan with the subtasks you will carry out. Then carry them out yourself with your tools "
            "({tools}), calling plan again if the plan changes, and finish with {answer_tool}."
        ),
        orchestrator=(
            "You coordinate {n_workers} workers; of the task's tools you can use only {answer_tool}.\n"
            "{team}\n"
            "First call plan with the subtasks. Then call delegate with up to {n_workers} subtasks per call; a worker "
            "sees only its subtask's text, so make each one self-contained. Call plan again if the plan changes. "
            "Finish with {answer_tool}."
        ),
        worker=(
            "Carry out the subtask below, then call report with your result, including every value the subtask asks "
            "for. Only the result is passed on."
        ),
        member=(
            "You are one of {k} council members. Work on the task on your own, then propose an answer: call "
            "{answer_tool} with it and a short rationale. The members then review each other's proposals."
        ),
        critique=(
            "Critique round {round} of {rounds}. The other members' proposals:\n\n{proposals}\n\n"
            "Review them (you may use the tools again), then call {answer_tool} with your proposal for this round and "
            "its rationale."
        ),
        chair=(
            "You chair a council of {k} members. Their final proposals, after {rounds} rounds of critique:\n\n"
            "{proposals}\n\nSubmit the final answer with {answer_tool}."
        ),
    ),
    # Explicit formats (and the planning instruction's detail): a plan and subtasks that name their IDs and the values
    # they produce, reports and rationales that pair each value or fact with its ID.
    "structured": RoleNotes(
        s9=(
            "Work in two phases. First call plan with the list of subtasks you will carry out; write each subtask as "
            "what to find or do, the IDs it concerns and the value it should produce. Then carry them out yourself, "
            "one after another, with your tools ({tools}); call plan again whenever the plan needs to change. When "
            "every subtask has its value, call {answer_tool}."
        ),
        orchestrator=(
            "You coordinate a team of {n_workers} workers. You cannot use the task's tools yourself, apart from "
            "{answer_tool}.\n"
            "{team}\n"
            "How to work:\n"
            "1. First call plan with the list of subtasks you intend to have carried out; write each subtask as what to "
            "find or do, the IDs it concerns and the value it should produce.\n"
            "2. Call delegate with up to {n_workers} subtasks at a time. Each subtask runs in a fresh worker context: "
            "the worker sees only the text of its subtask (not this conversation, the plan or the other subtasks). "
            "Write each subtask in three parts: what to do; every ID and detail it needs; and what to report back, "
            "value by value, in the format you need. delegate returns each worker's result. Call delegate as often as "
            "you need, and call plan again whenever the plan needs to change.\n"
            "3. When every subtask has its value, call {answer_tool} with the final answer."
        ),
        worker=(
            "You are a worker on a team. Carry out the subtask below and then call report with your result. The "
            "result is all the requester will see of your work: give each value the subtask asks for together with "
            "the ID it belongs to, in the format the subtask asks for. You do not submit the team's final answer."
        ),
        member=(
            "You are one of {k} council members working on this task independently. Instead of submitting a final "
            "answer, propose one: call {answer_tool} with your proposed answer and a rationale that lists the facts it "
            "rests on, each with the ID of the record, policy or procedure it comes from. The other members will "
            "review your proposal, and you will review theirs."
        ),
        critique=(
            "Critique round {round} of {rounds}. The other members' current proposals:\n\n{proposals}\n\n"
            "For each point where a proposal differs from yours, check which answer the cited facts support (you may "
            "use the tools again). Then call {answer_tool} with your proposal for this round, kept or revised, and a "
            "rationale that lists the facts it rests on with their IDs."
        ),
        chair=(
            "You chair a council of {k} members who each worked on this task. Their final proposals, after {rounds} "
            "rounds of critique:\n\n{proposals}\n\nWhere the proposals differ, decide each point on the facts their "
            "rationales cite, checked against the company knowledge you have. Submit the final answer with "
            "{answer_tool}."
        ),
    ),
    # A check at each hand-off: the plan before answering; each worker result against its subtask, re-delegating what
    # is missing; a report that says what it could not find; a critique that revises only on evidence; a chair that
    # keeps a unanimous answer. The member's first note is the default's.
    "verify": RoleNotes(
        s9=(
            "Work in two phases. First call plan with the list of subtasks you will carry out. Then carry them out "
            "yourself, one after another, with your tools ({tools}); call plan again whenever the plan needs to "
            "change. Before calling {answer_tool}, go through the plan and make sure each subtask is done."
        ),
        orchestrator=(
            "You coordinate a team of {n_workers} workers. You cannot use the task's tools yourself, apart from "
            "{answer_tool}.\n"
            "{team}\n"
            "How to work:\n"
            "1. First call plan with the list of subtasks you intend to have carried out.\n"
            "2. Call delegate with up to {n_workers} subtasks at a time. Each subtask runs in a fresh worker context: "
            "the worker sees only the text of its subtask (not this conversation, the plan or the other subtasks), so "
            "write each subtask so that it stands on its own, with every ID and detail it needs. delegate returns each "
            "worker's result. Call delegate as often as you need, and call plan again whenever the plan needs to "
            "change.\n"
            "3. Check each result against its subtask. If a result is missing, incomplete or reports a problem, "
            "delegate that subtask again, with what was missing spelled out.\n"
            "4. When every subtask has a complete result, call {answer_tool} with the final answer."
        ),
        worker=(
            "You are a worker on a team. Carry out the subtask below and then call report with your result. The "
            "result is all the requester will see of your work, so include every value the subtask asks for. If "
            "something the subtask needs cannot be found or done, say so in the result instead of filling the gap. "
            "You do not submit the team's final answer."
        ),
        member=COUNCIL_MEMBER_NOTE,
        critique=(
            "Critique round {round} of {rounds}. The other members' current proposals:\n\n{proposals}\n\n"
            "Check them against the evidence and your own work (you may use the tools again). Change your proposal "
            "only where the evidence shows it is wrong, then call {answer_tool} with your proposal for this round and "
            "its rationale."
        ),
        chair=(
            "You chair a council of {k} members who each worked on this task. Their final proposals, after {rounds} "
            "rounds of critique:\n\n{proposals}\n\nIf every member proposes the same answer, submit that answer. If "
            "they differ, compare their rationales point by point against the company knowledge you have and submit "
            "the answer it supports. Submit with {answer_tool}."
        ),
    ),
}
DEFAULT_VARIANT = "default"
