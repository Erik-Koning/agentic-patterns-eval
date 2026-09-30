"""Programmatic primary outcome and delivered-evidence diagnostics."""

import json
import statistics

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, mean, scorer, stderr
from inspect_ai.solver import TaskState

from ..agent.kb_react import COMPILE_LOG
from ..kb.provenance import evidence_pr
from ..worlds.env_tools import ANSWER, CALLS
from ..worlds.spec import TaskItem


def _norm(s: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(s)).split())


def is_success(task: TaskItem, answer: dict | None, calls: list[dict]) -> bool:
    if task.family == "F3":
        # End state: exactly the procedure's mutating calls, in any order, nothing else.
        key = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
        return sorted(map(key, calls)) == sorted(map(key, task.gold["calls"]))
    if answer is None:
        return False
    if task.family == "F7":
        try:
            return {**answer, "deadline_days": int(answer["deadline_days"])} == task.gold
        except (KeyError, TypeError, ValueError):
            return False
    return _norm(answer.get("answer", "")) == _norm(task.gold["answer"])


@scorer(metrics=[accuracy(), stderr()])
def task_success():
    async def score(state: TaskState, target: Target) -> Score:
        task = TaskItem(**state.metadata["task"])
        answer = state.store.get(ANSWER)
        calls = state.store.get(CALLS, [])
        ok = is_success(task, answer, calls)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=json.dumps(answer if task.family != "F3" else {"calls": calls}, sort_keys=True),
            metadata={"turns_used": state.store.get("turns_used"), "answered": answer is not None},
        )

    return score


@scorer(
    metrics={
        "evidence_recall": [mean()],
        "evidence_recall_first": [mean()],
        "evidence_recall_step_mean": [mean()],
        "evidence_precision": [mean()],
        "ctx_tokens_mean": [mean()],
        "compiles": [mean()],
    }
)
def delivered_evidence():
    """Did the arm put the gold facts in front of the model?

    `evidence_recall` unions all compiles (push and pull), which favours arms that take more
    turns; `evidence_recall_first` (first compile) and `evidence_recall_step_mean` do not.
    """

    async def score(state: TaskState, target: Target) -> Score:
        task = TaskItem(**state.metadata["task"])
        log = state.store.get(COMPILE_LOG, [])
        delivered = list(dict.fromkeys(f for rec in log for f in rec["fact_ids"]))
        p, r = evidence_pr(delivered, task.gold_fact_ids)
        per_step = [evidence_pr(rec["fact_ids"], task.gold_fact_ids)[1] for rec in log]
        tokens = [rec["tokens"] for rec in log] or [0]
        return Score(
            value={
                "evidence_recall": r,
                "evidence_recall_first": per_step[0] if per_step else 0.0,
                "evidence_recall_step_mean": statistics.fmean(per_step) if per_step else 0.0,
                "evidence_precision": p,
                "ctx_tokens_mean": statistics.fmean(tokens),
                "compiles": len(log),
            },
            metadata={"prompt_hashes": [rec["prompt_hash"] for rec in log], "route": [rec["meta"].get("route") for rec in log if rec["meta"].get("route")]},
        )

    return score
