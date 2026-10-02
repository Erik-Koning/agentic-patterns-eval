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
from .success import norm_f5, norm_id, norm_ratings

F7_FIELDS = ("action", "approver", "deadline_days", "document")


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
    made, want = [_call_key(c) for c in calls], [_call_key(c) for c in gold["calls"]]
    hit = sum(min(made.count(k), want.count(k)) for k in set(want))
    p = hit / len(made) if made else 0.0
    r = hit / len(want) if want else 1.0
    return {"call_precision": p, "call_recall": r, "call_f1": (2 * p * r / (p + r)) if p + r else 0.0}


def _call_key(c: dict) -> str:
    """One call as compared: key order never matters (recorded calls are {tool, args}, golds {args, tool})."""
    return json.dumps(c, sort_keys=True)


def f3_error(task: TaskItem, calls: list[dict]) -> str:
    want = task.gold["calls"]
    if sorted(map(_call_key, calls)) == sorted(map(_call_key, want)):
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
    got = norm_f5(answer.get("answer", ""))
    if got == norm_f5(task.gold["answer"]):
        return "correct"
    date = dt.date.fromisoformat(task.tags["date"])
    relation = "manager" if task.level == "1hop" else "office"
    timeline = [e for e in world.events if e.relation == relation and (e.subject == task.tags["team"] or task.level == "2hop")]
    if any(norm_f5(e.value) == got for e in timeline if dt.date.fromisoformat(e.date) != date):
        return "stale_or_wrong_time"
    known = {norm_f5(e.value) for e in world.events}
    return "wrong_entity" if got in known else "other"


def f1_items(answer: dict | None, gold: dict) -> dict[str, float]:
    """Item-level credit for F1: a (supplier, rating) pair is an item; precision over submitted, recall over gold."""
    got, want = norm_ratings((answer or {}).get("ratings")), norm_ratings(gold["ratings"])
    hit = sum(got.get(k) == v for k, v in want.items())
    p = hit / len(got) if got else 0.0
    r = hit / len(want) if want else 1.0
    return {"item_precision": p, "item_recall": r, "item_f1": (2 * p * r / (p + r)) if p + r else 0.0, "items_correct": hit, "items": len(want)}


def f1_error(task: TaskItem, answer: dict | None) -> str:
    if answer is None:
        return "no_answer"
    if not isinstance(answer.get("ratings"), dict):
        return "malformed_answer"
    got, want = norm_ratings(answer["ratings"]), norm_ratings(task.gold["ratings"])
    if got == want:
        return "correct"
    missing, extra = set(want) - set(got), set(got) - set(want)
    wrong = any(got[k] != want[k] for k in set(got) & set(want))
    if missing and wrong:
        return "missing_and_wrong"
    if missing:
        return "missing_items"
    if wrong:
        return "wrong_ratings"
    return "extra_items" if extra else "other"


def f2_chain(answer: dict | None, start: str | None = None) -> list[str]:
    """The submitted F2 chain as compared: IDs normalized, and a leading start supplier dropped. The tool
    excludes the original supplier, but the prompt asks for "every supplier along the way", so a model may
    include it; both readings score the same."""
    raw = (answer or {}).get("chain")
    chain = [norm_id(c) for c in raw] if isinstance(raw, list) else []
    if start is not None and chain and chain[0] == norm_id(start):
        chain = chain[1:]
    return chain


def f2_prefix(answer: dict | None, gold: dict, start: str | None = None) -> int:
    """F2: how many leading hops of the submitted chain match the gold chain."""
    n = 0
    for got, want in zip(f2_chain(answer, start), gold["chain"]):
        if got != norm_id(want):
            break
        n += 1
    return n


def f2_error(task: TaskItem, answer: dict | None) -> str:
    if answer is None:
        return "no_answer"
    if not isinstance(answer.get("chain"), list):
        return "malformed_answer"
    start = task.tags.get("start")
    k, prefix, chain = len(task.gold["chain"]), f2_prefix(answer, task.gold, start), f2_chain(answer, start)
    final_ok = norm_id(answer.get("final", "")) == norm_id(task.gold["final"])
    if final_ok:
        return "correct" if prefix == k and len(chain) == k else "correct_final_wrong_path"
    if prefix == k:
        return "overshot" if len(chain) > k else "final_mismatch"
    if prefix == len(chain):
        return "stopped_early"
    return "wrong_hop"


@scorer(metrics={"partial_credit": [mean()]})
def error_analysis():
    """Secondary: partial credit (F7 per-field accuracy, F3 call F1, F5 exact, F1 item F1, F2 correct-prefix share)
    and an error label per sample."""

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
        elif task.family == "F1":
            c = f1_items(answer, task.gold)
            credit, meta = c["item_f1"], {"error": f1_error(task, answer), **c}
        elif task.family == "F2":
            k, prefix = len(task.gold["chain"]), f2_prefix(answer, task.gold, task.tags.get("start"))
            credit, meta = prefix / k, {"error": f2_error(task, answer), "prefix": prefix, "hops": k}
        else:
            label = f5_error(world, task, answer)
            credit, meta = float(label == "correct"), {"error": label}
        return Score(value={"partial_credit": credit}, metadata=meta)

    return score
