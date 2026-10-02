"""Programmatic scoring of an F8 session (CONTEXT_MANAGEMENT_AUDIT §5), from the session's tool events.

- Per item: a policy case or follow-up succeeds when the first decision submitted for it in its own window
  equals the gold; a ticket when its order's mutating calls in its window equal the gold calls (any order,
  nothing else) and it was closed with `finish`.
- Dependency items are those whose correctness needs earlier state (gen_f8: follow-up, quota, memo, withdrawn).
- Report: exact match of all three parts, plus per-part accuracy / F1.
- Session success: every item and the report exact, with no overflow.
- Probes and external state: F1 per category against `gen_f8.state_at(k)`.
- Failure taxonomy (§5.4): forgot constraint, stale state, resurrected done item, dropped item, hallucinated
  state, and overflow (CM0 past the window). `summary_loss` checks a compacted view for what a case needs.
"""

import json
import re

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..worlds.env_f8 import EVENTS, OVERFLOW, PROBES, REPORT
from ..worlds.gen_f8 import PROBE_CATEGORIES, calls_key, decisions_equal
from ..worlds.spec import TaskItem, World

CASE_ID = re.compile(r"\bC-\d{6}\b")
MEMO_ID = re.compile(r"\bM-\d+\b")
LABELS = ("forgot_constraint", "stale_state", "resurrected_done_item", "dropped_item", "hallucinated_state", "overflow")


def _mutating(world: World) -> set[str]:
    return {t.name for t in world.tools}


def submission(world: World, task: TaskItem, events: list[dict]):
    """What the agent submitted for this item in its own window: a decision (or None) for policy cases and
    follow-ups; {"finished", "calls"} for tickets."""
    pos, cid = task.tags["position"], task.tags["case_id"]
    window = [e for e in events if e["item"] == pos]
    if task.tags["kind"] == "ticket":
        mutating = _mutating(world)
        return {
            "finished": any(e["tool"] == "finish" and e["args"].get("case_id") == cid for e in window),
            "calls": [{"tool": e["tool"], "args": e["args"]} for e in window if e["tool"] in mutating and e["args"].get("order_id") == task.tags["order_id"]],
        }
    decisions = [e["args"] for e in window if e["tool"] == "submit_decision" and e["args"].get("case_id") == cid]
    return decisions[0] if decisions else None


def _matches(task: TaskItem, sub, gold) -> bool:
    if task.tags["kind"] == "ticket":
        return sub is not None and calls_key(sub["calls"]) == calls_key(gold["calls"])
    return decisions_equal(sub, gold)


def item_success(world: World, task: TaskItem, events: list[dict]) -> bool:
    sub = submission(world, task, events)
    if task.tags["kind"] == "ticket":
        return sub["finished"] and _matches(task, sub, task.gold)
    return _matches(task, sub, task.gold)


def answered(world: World, task: TaskItem, events: list[dict]) -> bool:
    sub = submission(world, task, events)
    return sub["finished"] if task.tags["kind"] == "ticket" else sub is not None


def set_f1(pred, gold) -> dict:
    p, g = set(pred or []), set(gold or [])
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def report_score(report: dict | None, gold: dict) -> dict:
    if not isinstance(report, dict):
        return {"exact": False, "dispositions_accuracy": 0.0, "pending_recheck_f1": 0.0, "open_followups_f1": 0.0, "submitted": False}
    disp = report.get("dispositions") if isinstance(report.get("dispositions"), dict) else {}
    gd = gold["dispositions"]
    acc = sum(str(disp.get(c, "")).strip().lower() == v for c, v in gd.items()) / len(gd)
    pending = set_f1(report.get("pending_recheck"), gold["pending_recheck"])["f1"]
    follow = set_f1(report.get("open_followups"), gold["open_followups"])["f1"]
    exact = acc == 1.0 and set(disp) == set(gd) and pending == 1.0 and follow == 1.0
    return {"exact": exact, "dispositions_accuracy": acc, "pending_recheck_f1": pending, "open_followups_f1": follow, "submitted": True}


def probe_score(answer: dict | None, state: dict) -> dict:
    """F1 per category of a probe's JSON answer against `state_at(k)`, plus their mean."""
    out = {}
    for cat in PROBE_CATEGORIES:
        pred = (answer or {}).get(cat) if isinstance(answer, dict) else None
        out[cat] = set_f1(pred if isinstance(pred, list) else [], state[cat])["f1"] if answer is not None else 0.0
    out["mean_f1"] = sum(out[c] for c in PROBE_CATEGORIES) / len(PROBE_CATEGORIES)
    return out


def external_state_score(state: dict, *, todos: list[dict] | None = None, notes: str | None = None) -> dict:
    """What the agent wrote down, scored against `state_at(k)`.

    - todos (Inspect `todo_write` items, {content, status}): completed = case IDs in completed todos; pending =
      case IDs in pending / in-progress todos; memos_in_force = memo IDs in open todos; open_followups = case IDs
      in open todos that mention a follow-up; escalated = case IDs in todos that mention an escalation. F1 each.
    - notes (free text): recall per category of the IDs mentioned anywhere (notes have no status field), plus
      hallucinated IDs (case IDs not in the queue, memo IDs never announced)."""
    if todos is not None:
        done = [t for t in todos if str(t.get("status")) == "completed"]
        open_ = [t for t in todos if str(t.get("status")) in ("pending", "in_progress")]
        ids = lambda ts, rx: [i for t in ts for i in rx.findall(str(t.get("content", "")))]  # noqa: E731
        pred = {
            "completed": ids(done, CASE_ID),
            "pending": ids(open_, CASE_ID),
            "memos_in_force": ids(open_, MEMO_ID),
            "open_followups": ids([t for t in open_ if "follow" in str(t.get("content", "")).lower()], CASE_ID),
            "escalated": ids([t for t in todos if "escalat" in str(t.get("content", "")).lower()], CASE_ID),
        }
        out = {c: set_f1(pred[c], state[c])["f1"] for c in PROBE_CATEGORIES}
        out["mean_f1"] = sum(out[c] for c in PROBE_CATEGORIES) / len(PROBE_CATEGORIES)
        return {"kind": "todos", **out}
    text = notes or ""
    cases, memos = set(CASE_ID.findall(text)), set(MEMO_ID.findall(text))
    out = {c: (len(set(state[c]) & (memos if c == "memos_in_force" else cases)) / len(state[c]) if state[c] else 1.0) for c in PROBE_CATEGORIES}
    out["mean_recall"] = sum(out[c] for c in PROBE_CATEGORIES) / len(PROBE_CATEGORIES)
    return {"kind": "notes", **out}


def _hallucinated_ids(world: World, ids) -> list[str]:
    s = world.entities["session"]
    known_cases, known_memos = set(s["queue"]), {m["id"] for m in s["memos"]}
    out = []
    for i in ids or []:
        i = str(i)
        if (CASE_ID.fullmatch(i) and i not in known_cases) or (MEMO_ID.fullmatch(i) and i not in known_memos):
            out.append(i)
    return out


def taxonomy(world: World, events: list[dict], report: dict | None = None, probes: list[dict] = (), overflow_at: int | None = None) -> dict:
    """Failure labels per item, from the events and the generator's counterfactual golds."""
    pos_of_case = {t.tags["case_id"]: t.tags["position"] for t in world.tasks}
    pos_of_order = {t.tags["order_id"]: t.tags["position"] for t in world.tasks if t.tags["kind"] == "ticket"}
    mutating = _mutating(world)
    per_item: dict[int, list[str]] = {t.tags["position"]: [] for t in world.tasks}
    session_level: list[str] = []
    for t in world.tasks:
        pos, labels = t.tags["position"], per_item[t.tags["position"]]
        if overflow_at is not None and pos >= overflow_at:
            labels.append("overflow")
            continue
        sub = submission(world, t, events)
        if not answered(world, t, events):
            labels.append("dropped_item")
            continue
        cf = t.tags["counterfactuals"]
        if not _matches(t, sub, t.gold):
            if any(_matches(t, sub, alt) for alt in cf["without_memo"].values()) or ("without_quota" in cf and _matches(t, sub, cf["without_quota"])):
                labels.append("forgot_constraint")
            stale = any(_matches(t, sub, alt) for alt in cf["with_withdrawn"].values())
            if t.tags["kind"] == "followup" and _matches(t, sub, cf["original_gold"]):
                stale = True
            if stale:
                labels.append("stale_state")
    for e in events:
        pos = e["item"]
        if pos not in per_item:
            continue
        case = e["args"].get("case_id") if e["tool"] in ("submit_decision", "finish") else None
        order = e["args"].get("order_id") if e["tool"] in mutating else None
        if case is not None:
            if case not in pos_of_case:
                per_item[pos].append("hallucinated_state")
            elif pos_of_case[case] < pos:
                per_item[pos].append("resurrected_done_item")
        if order is not None and order in pos_of_order and pos_of_order[order] < pos:
            per_item[pos].append("resurrected_done_item")
    bad: list[str] = []
    if isinstance(report, dict):
        disp = report.get("dispositions")
        bad = _hallucinated_ids(world, [*(disp if isinstance(disp, dict) else []), *(report.get("pending_recheck") or []), *(report.get("open_followups") or [])])
        if bad:
            session_level.append("hallucinated_state")
            per_item.setdefault(len(world.tasks) + 1, []).append("hallucinated_state")  # the report phase
    probe_bad = []
    for p in probes:
        ans = p.get("answer")
        if isinstance(ans, dict):
            probe_bad += _hallucinated_ids(world, [i for c in PROBE_CATEGORIES for i in (ans.get(c) or [])])
    if probe_bad:
        session_level.append("hallucinated_state")
    counts = {lab: sum(lab in v for v in per_item.values()) for lab in LABELS}
    counts["hallucinated_state"] += bool(probe_bad)
    return {
        "per_item": {k: sorted(set(v)) for k, v in per_item.items() if v},
        "counts": counts,
        "session_labels": sorted(set(session_level)),
        "report_hallucinated": bad,
        "probe_hallucinated": sorted(set(probe_bad)),
    }


def summary_loss(world: World, position: int, view_text: str) -> list[str]:
    """What case `position` needs from earlier in the session that is absent from a (compacted) view: the
    original request of a follow-up, the memos that change its answer, and a quota customer's earlier cases."""
    t = world.tasks[position - 1]
    s = world.entities["session"]
    memos = {m["id"]: m for m in s["memos"]}
    missing = []
    if t.tags["kind"] == "followup":
        orig = world.tasks[[x.tags["case_id"] for x in world.tasks].index(t.tags["original"])]
        if orig.tags["customer_id"] not in view_text or f"{orig.tags['amount']:,}" not in view_text:
            missing.append(f"original request of {t.tags['original']}")
    for mid in t.tags["counterfactuals"]["without_memo"]:
        m = memos[mid]
        if m["position"] < position and m["text"] not in view_text and mid not in view_text:
            missing.append(f"memo {mid}")
    if "quota" in t.tags["dependency_kinds"]:
        earlier = [x.tags["case_id"] for x in world.tasks[: position - 1] if x.tags.get("customer_id") == t.tags["customer_id"] and x.tags["kind"] in ("policy", "repeat")]
        if any(c not in view_text for c in earlier):
            missing.append(f"earlier cases of {t.tags['customer_id']}")
    return missing


def score_session(world: World, events: list[dict], report: dict | None, overflow_at: int | None = None) -> dict:
    """Items, dependency items, report and session success."""
    items = []
    for t in world.tasks:
        lost = overflow_at is not None and t.tags["position"] >= overflow_at
        ok = (not lost) and item_success(world, t, events)
        items.append({
            "position": t.tags["position"],
            "case_id": t.tags["case_id"],
            "kind": t.tags["kind"],
            "dependency": t.tags["dependency"],
            "dependency_kinds": t.tags["dependency_kinds"],
            "answered": (not lost) and answered(world, t, events),
            "success": ok,
            "overflow": lost,
        })
    dep = [i for i in items if i["dependency"]]
    rep = report_score(report if overflow_at is None else None, world.entities["session"]["report_gold"])
    all_items = all(i["success"] for i in items)
    return {
        "items": items,
        "item_success": sum(i["success"] for i in items) / len(items),
        "dependency_items": len(dep),
        "dependency_success": sum(i["success"] for i in dep) / len(dep) if dep else 0.0,
        "report": rep,
        "session_success": all_items and rep["exact"] and overflow_at is None,
        "overflow_at": overflow_at,
    }



@scorer(
    metrics={
        "item_success": [mean()],
        "dependency_success": [mean()],
        "report_exact": [mean()],
        "session_success": [mean()],
        "probe_f1": [mean()],
        "overflow": [mean()],
    }
)
def f8_session_score():
    """Session outcomes recomputed from the recorded tool events, with the failure taxonomy in the metadata."""

    async def score(state: TaskState, target: Target) -> Score:
        from ..agent.arms import load_world

        world = load_world(state.metadata["world_id"])
        events, report, overflow_at = state.store.get(EVENTS, []), state.store.get(REPORT), state.store.get(OVERFLOW)
        probes = state.store.get(PROBES, [])
        res = score_session(world, events, report, overflow_at)
        tax = taxonomy(world, events, report, probes, overflow_at)
        probe_f1 = [p["scores"]["mean_f1"] for p in probes]
        return Score(
            value={
                "item_success": res["item_success"],
                "dependency_success": res["dependency_success"],
                "report_exact": float(res["report"]["exact"]),
                "session_success": float(res["session_success"]),
                "probe_f1": sum(probe_f1) / len(probe_f1) if probe_f1 else 0.0,
                "overflow": float(overflow_at is not None),
            },
            answer=json.dumps(report, sort_keys=True) if report is not None else None,
            metadata={"items": res["items"], "dependency_items": res["dependency_items"], "report": res["report"], "overflow_at": overflow_at, "taxonomy": tax, "probes": [{"k": p["k"], "scores": p["scores"], "error": p["error"]} for p in probes]},
        )

    return score
