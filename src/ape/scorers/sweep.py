"""Scoring of a context-length sweep sample (CONTEXT_SWEEP.md), from the probe's tool events.

`success` is F8's item rule (`scorers.session.item_success`): the first decision submitted for the probe equals the
gold in all four fields. `answered` is whether any decision was submitted. `measured` is 0 when the provider refused
the request (`over_limit`, `rejected`: the model never saw it), so the analysis keeps those samples out of every rate.
The metadata carries the design (kind, target and metered context size, middle cases), the submission and gold, and a
failure label:

- `over_limit`: the provider refused the request's size (not measured);
- `rejected`: the provider refused the request for another reason (not measured; a harness problem to look at);
- `output_cap`: no decision, the last generation stopped at the output cap the solver sent;
- `unanswered`: no decision within the generation cap otherwise;
- `copied_original` (followup): case 1's earlier decision repeated. The probe's memo is the only change since case 1,
  so this is also the decision without the memo: the original was found, the memo in the probe's own message not
  applied;
- `ignored_memo` (memo): the decision the probe would have without the memo announced at the start (the
  counterfactual `without_memo`);
- `wrong`: any other decision (for followup: the original request not found or misread).

`calls` gives per generation the metered view tokens, the output cap sent, the tools called and the provider's usage,
compact enough to survive in Inspect's sample summaries.
"""

import json

from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import TaskState

from ..agent.sweep import PROBE
from ..worlds.env_f8 import EVENTS
from ..worlds.gen_f8 import decisions_equal
from .session import USAGE_FIELDS, item_success, submission

NOT_MEASURED = ("over_limit", "rejected")


def failure(task, sub: dict | None, probe: dict | None = None) -> str | None:
    """The failure label of a submission for the probe (None when it is correct)."""
    probe = probe or {}
    if probe.get("over_limit"):
        return "over_limit"
    if probe.get("rejected"):
        return "rejected"
    if decisions_equal(sub, task.gold):
        return None
    if sub is None:
        last = (probe.get("calls") or [{}])[-1]
        return "output_cap" if last.get("stop_reason") == "max_tokens" else "unanswered"
    cf = task.tags["counterfactuals"]
    if task.tags["kind"] == "followup" and decisions_equal(sub, cf.get("original_gold")):
        return "copied_original"
    if any(decisions_equal(sub, alt) for alt in (cf.get("without_memo") or {}).values()):
        return "ignored_memo"
    return "wrong"


@scorer(metrics={"success": [mean()], "answered": [mean()], "measured": [mean()]})
def sweep_score():
    async def score(state: TaskState, target: Target) -> Score:
        from ..agent.arms import load_world

        world = load_world(state.metadata["world_id"])
        task = world.tasks[-1]
        events = state.store.get(EVENTS, [])
        probe = state.store.get(PROBE) or {}
        sub = submission(world, task, events)
        label = failure(task, sub, probe)
        ok = label is None and item_success(world, task, events)
        design = world.entities["sweep"]
        return Score(
            value={"success": float(ok), "answered": float(sub is not None), "measured": float(label not in NOT_MEASURED)},
            answer=json.dumps(sub, sort_keys=True) if sub is not None else None,
            metadata={
                "kind": design["kind"],
                "target_tokens": design["target_tokens"],
                "context_tokens": design["context_tokens"],
                "middle_cases": design["middle_cases"],
                "probe_case_id": design["probe_case_id"],
                "gold": task.gold,
                "submission": sub,
                "failure": label,
                "generations": len(probe.get("calls") or []),
                "calls": [
                    {
                        "view_tokens": c.get("view_tokens"),
                        "max_output_tokens": c.get("max_output_tokens"),
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
