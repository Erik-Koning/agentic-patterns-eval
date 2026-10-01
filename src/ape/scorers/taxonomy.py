"""Partial credit and an automatic error taxonomy, computed from the world spec (no judgement).

The primary outcome stays all-or-nothing (`task_success`). These secondary measures say
*which part* went wrong, which (a) adds statistical sensitivity and (b) turns the gate's
NO-GO diagnosis into counts of labelled errors.
"""

import datetime as dt
import json

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..agent.arms import load_world
from ..worlds.env_tools import ANSWER, CALLS
from ..worlds.spec import TaskItem, World

F7_FIELDS = ("action", "approver", "deadline_days", "document")


def _norm(s) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in str(s)).split())


def f7_fields(answer: dict | None, gold: dict) -> dict[str, float]:
    a = answer or {}
    out = {}
    for f in F7_FIELDS:
        try:
            out[f] = float((int(a.get(f)) if f == "deadline_days" else a.get(f)) == gold[f])
        except (TypeError, ValueError):
            out[f] = 0.0
    return out


def f7_error(world: World, task: TaskItem, answer: dict | None) -> str:
    if answer is None:
        return "no_answer"
    try:
        ans = {**answer, "deadline_days": int(answer["deadline_days"])}
    except (KeyError, TypeError, ValueError):
        return "malformed_answer"
    if ans == task.gold:
        return "correct"
    policy = next(p for p in world.policies if p.id == task.tags["policy"])
    exc = next((x for x in world.exceptions if x.policy_id == policy.id), None)
    case = task.tags["case"]
    if case == "exception_applies" and ans == policy.outcome:
        return "missed_exception"
    if case == "exception_not_applicable" and exc is not None and ans == exc.outcome:
        return "spurious_exception"
    same_domain = [p for p in world.policies if p.domain == policy.domain and p.id != policy.id]
    for p in same_domain:
        if p.outcome == ans:
            if p.region == policy.region:
                return "wrong_band"
            if (p.lo, p.hi) == (policy.lo, policy.hi):
                return "wrong_region"
            return "wrong_policy"
    if any(p.outcome == ans for p in world.policies):
        return "wrong_domain"
    return "partial" if any(f7_fields(ans, task.gold).values()) else "other"


def f3_calls(calls: list[dict], gold: dict) -> dict[str, float]:
    key = lambda c: json.dumps(c, sort_keys=True)  # noqa: E731
    made, want = [key(c) for c in calls], [key(c) for c in gold["calls"]]
    hit = sum(min(made.count(k), want.count(k)) for k in set(want))
    p = hit / len(made) if made else 0.0
    r = hit / len(want) if want else 1.0
    return {"call_precision": p, "call_recall": r, "call_f1": (2 * p * r / (p + r)) if p + r else 0.0}


def f3_error(task: TaskItem, calls: list[dict]) -> str:
    want = task.gold["calls"]
    if sorted(map(json.dumps, calls)) == sorted(map(json.dumps, want)):
        return "correct"
    if not calls:
        return "no_calls"
    want_tools = {c["tool"] for c in want}
    made_tools = {c["tool"] for c in calls}
    if made_tools - want_tools:
        return "wrong_tool"
    if made_tools == want_tools and len(calls) == len(want):
        return "wrong_arguments"
    if len(calls) < len(want):
        return "missing_call"
    return "extra_call"


def f5_error(world: World, task: TaskItem, answer: dict | None) -> str:
    if answer is None:
        return "no_answer"
    got = _norm(answer.get("answer", ""))
    if got == _norm(task.gold["answer"]):
        return "correct"
    date = dt.date.fromisoformat(task.tags["date"])
    relation = "manager" if task.level == "1hop" else "office"
    timeline = [e for e in world.events if e.relation == relation and (e.subject == task.tags["team"] or task.level == "2hop")]
    if any(_norm(e.value) == got for e in timeline if dt.date.fromisoformat(e.date) != date):
        return "stale_or_wrong_time"
    known = {_norm(e.value) for e in world.events}
    return "wrong_entity" if got in known else "other"


@scorer(metrics={"partial_credit": [mean()]})
def error_analysis():
    """Secondary: partial credit (F7 per-field accuracy, F3 call F1, F5 exact) and an error label per sample."""

    async def score(state: TaskState, target: Target) -> Score:
        task = TaskItem(**state.metadata["task"])
        world = load_world(state.metadata["world_id"])
        answer = state.store.get(ANSWER)
        calls = state.store.get(CALLS, [])
        if task.family == "F7":
            fields = f7_fields(answer, task.gold)
            credit, meta = sum(fields.values()) / len(fields), {"error": f7_error(world, task, answer), "fields": fields}
        elif task.family == "F3":
            c = f3_calls(calls, task.gold)
            credit, meta = c["call_f1"], {"error": f3_error(task, calls), **c}
        else:
            label = f5_error(world, task, answer)
            credit, meta = float(label == "correct"), {"error": label}
        return Score(value={"partial_credit": credit}, metadata=meta)

    return score
