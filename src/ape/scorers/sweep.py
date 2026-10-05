"""Scoring of a context-length sweep sample (CONTEXT_SWEEP.md), from the probe's tool events.

`success` is F8's item rule (`scorers.session.item_success`): the first decision submitted for the probe equals the
gold in all four fields. `answered` is whether any decision was submitted. The metadata carries the design (kind,
target and metered context size, middle cases), the submission and gold, and a failure label:

- `unanswered`: no decision for the probe within the generation cap;
- `copied_original` (followup): case 1's earlier decision repeated. The probe's memo is the only change since case 1,
  so this is also the decision without the memo: the original was found, the memo in the probe's own message not
  applied;
- `ignored_memo` (memo): the decision the probe would have without the memo announced at the start (the
  counterfactual `without_memo`);
- `wrong`: any other decision (for followup: the original request not found or misread).

`calls` gives per generation the metered view tokens, the tools called and the provider's usage, compact enough to
survive in Inspect's sample summaries.
"""

import json

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..agent.sweep import PROBE
from ..worlds.env_f8 import EVENTS
from ..worlds.gen_f8 import decisions_equal
from .session import USAGE_FIELDS, item_success, submission


def failure(task, sub: dict | None) -> str | None:
    """The failure label of a submission for the probe (None when it is correct)."""
    if decisions_equal(sub, task.gold):
        return None
    if sub is None:
        return "unanswered"
    cf = task.tags["counterfactuals"]
    if task.tags["kind"] == "followup" and decisions_equal(sub, cf.get("original_gold")):
        return "copied_original"
    if any(decisions_equal(sub, alt) for alt in (cf.get("without_memo") or {}).values()):
        return "ignored_memo"
    return "wrong"


@scorer(metrics={"success": [mean()], "answered": [mean()]})
def sweep_score():
    async def score(state: TaskState, target: Target) -> Score:
        from ..agent.arms import load_world

        world = load_world(state.metadata["world_id"])
        task = world.tasks[-1]
        events = state.store.get(EVENTS, [])
        probe = state.store.get(PROBE) or {}
        sub = submission(world, task, events)
        ok = item_success(world, task, events)
        design = world.entities["sweep"]
        return Score(
            value={"success": float(ok), "answered": float(sub is not None)},
            answer=json.dumps(sub, sort_keys=True) if sub is not None else None,
            metadata={
                "kind": design["kind"],
                "target_tokens": design["target_tokens"],
                "context_tokens": design["context_tokens"],
                "middle_cases": design["middle_cases"],
                "probe_case_id": design["probe_case_id"],
                "gold": task.gold,
                "submission": sub,
                "failure": failure(task, sub),
                "generations": len(probe.get("calls") or []),
                "calls": [
                    {
                        "view_tokens": c.get("view_tokens"),
                        "tools": c.get("tools"),
                        "stop_reason": c.get("stop_reason"),
                        "error": c.get("error"),
                        "usage": {f: c["usage"][f] for f in USAGE_FIELDS if f in (c.get("usage") or {})} if c.get("usage") else None,
                    }
                    for c in probe.get("calls") or []
                ],
            },
        )

    return score
