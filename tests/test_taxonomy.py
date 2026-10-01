"""Error taxonomy and partial credit: every label is reachable and matches its definition."""

from ape.scorers.taxonomy import f3_calls, f3_error, f5_error, f7_error, f7_fields
from ape.worlds.generate import make_world


def _f7():
    w = make_world("F7", "100", "dev", 0, 200)
    return w, {x.policy_id: x for x in w.exceptions}


def test_f7_labels():
    w, exc = _f7()
    applies = next(t for t in w.tasks if t.tags["case"] == "exception_applies")
    not_app = next(t for t in w.tasks if t.tags["case"] == "exception_not_applicable")
    policy = lambda t: next(p for p in w.policies if p.id == t.tags["policy"])  # noqa: E731
    assert f7_error(w, applies, dict(applies.gold)) == "correct"
    assert f7_error(w, applies, dict(policy(applies).outcome)) == "missed_exception"
    assert f7_error(w, not_app, dict(exc[not_app.tags["policy"]].outcome)) == "spurious_exception"
    t = next(t for t in w.tasks if t.tags["case"] == "no_exception")
    p = policy(t)
    other_band = next(q for q in w.policies if q.domain == p.domain and q.region == p.region and q.id != p.id and q.outcome != p.outcome)
    assert f7_error(w, t, dict(other_band.outcome)) == "wrong_band"
    assert f7_error(w, t, None) == "no_answer"
    assert f7_error(w, t, {"action": "approve"}) == "malformed_answer"


def test_f7_field_accuracy():
    gold = {"action": "escalate", "approver": "team_lead", "deadline_days": 3, "document": "receipt"}
    assert f7_fields({**gold, "deadline_days": "3"}, gold) == {k: 1.0 for k in gold}
    assert sum(f7_fields({**gold, "document": "none"}, gold).values()) == 3.0
    assert sum(f7_fields(None, gold).values()) == 0.0


def test_f3_labels_and_f1():
    w = make_world("F3", "20", "dev", 0, 5)
    t = w.tasks[0]
    want = t.gold["calls"]
    assert f3_error(t, want) == "correct" and f3_calls(want, t.gold)["call_f1"] == 1.0
    assert f3_error(t, []) == "no_calls"
    assert f3_error(t, want[:1]) == "missing_call"
    bad_arg = [dict(want[0], args={**want[0]["args"], "order_id": "O-0"}), *want[1:]]
    assert f3_error(t, bad_arg) == "wrong_arguments"
    other = next(tool.name for tool in w.tools if tool.name not in {c["tool"] for c in want})
    assert f3_error(t, [*want, {"tool": other, "args": {}}]) == "wrong_tool"
    assert 0 < f3_calls(want[:1], t.gold)["call_f1"] < 1


def test_f5_labels():
    w = make_world("F5", "1hop", "dev", 0, 30)
    t = w.tasks[0]
    assert f5_error(w, t, dict(t.gold)) == "correct"
    others = [e.value for e in w.events if e.subject == t.tags["team"] and e.relation == "manager" and e.value != t.gold["answer"]]
    assert f5_error(w, t, {"answer": others[0]}) == "stale_or_wrong_time"
    assert f5_error(w, t, {"answer": "Nobody Atall"}) == "other"
