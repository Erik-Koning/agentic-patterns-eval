"""The F8 session loop for Study G (CONTEXT_MANAGEMENT_AUDIT §6.1).

The loop keeps the **full history** (logged as the sample's messages) and asks the arm's **context policy**
(`ape.agent.context_policy`) for what the model sees at each call. Every arm shares one system prompt (base
prompt, the whole corpus, shift rules; a policy may append a short fixed addendum; a study run's per-run cache nonce,
`ape.agent.cache_nonce`, leads it, one per arm), one tool set (a policy may add
its own tools), one turn cap per case and one nominal window W; only the view differs. Built so far:

- **CM0**: the full, append-only history.
- **O-state**: system prompt + the generator's true state after the previous case (`render_oracle_state`) + the
  current case's messages. The upper bound that defines Gap_T and R_x.
- **CM-prune, CM-trim, CM-sum, CM-todo, CM-reset, CM-native and S-CM***: the context-management arms
  (`ape.agent.cm_arms`, B8).
- **S1, M1 and M2**: the topology arms (`ape.agent.multi.session_team`, B9). S1 is CM0 under its topology name; in
  M1 and M2 the session's agent is an orchestrator (CM0's view, the answer and report tools only) that delegates
  each case's work to isolated workers through a policy tool.

**Window and threshold.** The harness enforces W for every policy: an agent call whose view, or a management call
whose input, would exceed W is an **overflow**, and that case and every later one fail (the report too). Before a
call whose view exceeds T_abs (run_plan.yaml `study_g.threshold`, the same token meter), the policy's
`on_threshold` hook may return a smaller view; each such reaction is logged as a management event.

Each case arrives as a user message; the loop moves on once the case's answer call is made (submit_decision or
finish with its case ID), after one nudge, or at `max_turns_per_item`. At each item boundary the policy's
`after_item` hook runs, then the probe (at a checkpoint), then the mid-session checkpoint is saved. After the last
case the loop asks for the end-of-shift report, with one nudge if the model answers that request in text. The
history is built in `state.messages` itself and the store is written in a `finally`, so a session cut short by a
sample limit keeps its records.

**Forked state probes** (§5.3): after each checkpoint case k <= N, the probe question goes to the policy's probe
view (what the model would see next) in a side call with role `probe` and a strict JSON schema. The answer is never
appended to the history, and the probe view must leave the policy's state unchanged (checked). Without a `probe`
role the agent's model answers, as the budget assumes ("at the agent's model and effort").

**Calls and cost.** Every call is recorded with its kind: `agent` (the arm's agent), `cm` (a policy's management
call, on the `cm` role, the agent's model when the run defines none) and `probe`, with its view tokens, model and
Inspect usage, so management and probe cost separate from agent cost. Per case the record adds the management
calls and the usage by kind (management calls at a boundary count toward the case that just ended; probes too). A
call that trips a token or cost limit raises inside `generate` after Inspect counted it; it is recorded from the
transcript, flagged `limit` (`SessionContext.record_limit_call`; a probe so cut is a checkpoint not taken), so the
records equal Inspect's usage also then (review A-5).

Recorded in the store (keys in `ape.worlds.env_f8`): per-case records (position, view tokens at the first call and
at the decision call, generations, answered, success, dependency, cm_calls, usage by kind), every agent and
management call, the probes, the management events, the tool events (a policy's tools flagged `policy`), the
report, the overflow position, the session's usage by kind and by model, its resumes (`f8_resume`), and the arm
(policy, knobs, window, threshold, turn cap, probe checkpoints, policy tools and their schema tokens).

**Mid-session checkpoints** (`ape.agent.session_checkpoint`): when APE_SESSION_CHECKPOINTS names a directory, the
session is saved when it starts and after every completed item, keyed by its whole configuration. A retried sample
(Inspect's `retry_on_error`, or a task retry) resumes after the last completed item instead of from item 1; a
finished session deletes its checkpoint. That module's docstring gives the semantics of a resumed session.
"""

import json
import logging
from functools import lru_cache
from typing import Any

import yaml
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageTool, ChatMessageUser, GenerateConfig, ResponseSchema, execute_tools, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import JSONSchema, LimitExceededError, store

from ..config import ROOT
from ..scorers.session import item_success, probe_score
from ..worlds import gen_f8
from ..worlds.env_f8 import CM_EVENTS, EVENTS, ITEMS, OVERFLOW, PROBES, REPORT, RESUME, USAGE, VIEWS, SessionRecorder, build_session_tools
from .cache_nonce import of as nonce_of
from .cache_nonce import with_nonce
from .cm_arms import CM_ARMS
from .context_policy import (
    ContextPolicy,
    SessionContext,
    SessionOverflow,
    SessionRecords,
    dump_messages,
    inspect_limit_hit,
    knob_env,
    limit_call_usage,
    load_messages,
    message_tokens,  # noqa: F401  (re-exported: the session's token meter)
    policy_class,
    resolve_knobs,
    tool_name,
    tool_schema_tokens,
    usage_by,
    usage_record,
    view_tokens,
)
from .multi.session_team import TOPOLOGY_ARMS
from .session_checkpoint import SessionCheckpoints, checkpoint_root, code_version, model_identity, resume_summary, usage_summary

log = logging.getLogger(__name__)

SESSION_ARMS = ("CM0", "O-state", *CM_ARMS, *TOPOLOGY_ARMS)  # the built arms (cm_arms and session_team register their own); tests register more
PROBE_ROLE = "probe"
CM_ROLE = "cm"
PROBE_PROMPT = (
    "Pause the shift for a quick state check. Without using tools, list as JSON: the case IDs you have completed "
    "(completed), the case IDs still pending (pending), the memo IDs in force now (memos_in_force), the case IDs "
    "waiting for a follow-up (open_followups) and the case IDs you escalated this shift (escalated)."
)
REPORT_NUDGE = "Submit the end-of-shift report now: call submit_shift_report with the report as a JSON object string."
PROBE_SCHEMA = JSONSchema.model_validate(
    {
        "type": "object",
        "additionalProperties": False,
        "required": list(gen_f8.PROBE_CATEGORIES),
        "properties": {c: {"type": "array", "items": {"type": "string"}} for c in gen_f8.PROBE_CATEGORIES},
    }
)


@lru_cache(maxsize=1)
def plan_threshold() -> int:
    """T_abs: config/run_plan.yaml `study_g.threshold` (absolute tokens, identical across tiers; D-021)."""
    return int(yaml.safe_load((ROOT / "config" / "run_plan.yaml").read_text())["study_g"]["threshold"])


def _state_json(policy: ContextPolicy) -> str:
    return json.dumps(policy.state_dict(), sort_keys=True, default=str)


@solver
def f8_session_agent(
    arm: str = "CM0",
    window: int = gen_f8.WINDOW,
    max_turns_per_item: int = 8,
    checkpoints: tuple[int, ...] = gen_f8.CHECKPOINTS,
    threshold: int | None = None,
) -> Solver:
    """One F8 session per sample under the arm's context policy. `threshold` (T_abs) defaults to run_plan.yaml's;
    the policy's knobs are read from APE_CM_* when the session starts (`context_policy.resolve_knobs`)."""
    cls = policy_class(arm)  # an unbuilt arm, or knobs it cannot run with, fail when the task is created
    cls.validate(resolve_knobs(cls))
    t_abs = plan_threshold() if threshold is None else int(threshold)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from .arms import load_world

        world = load_world(state.metadata["world_id"])
        n = len(world.tasks)
        rec = SessionRecorder(world)
        session_tools = build_session_tools(world, rec)
        base_tools = list(session_tools.values())
        model = get_model()
        cm_model = get_model(role=CM_ROLE, default=model)
        probe_model = get_model(role=PROBE_ROLE, default=model)
        knobs = resolve_knobs(cls)
        nonce = nonce_of(state.metadata)  # a study run's per-run cache nonce (`agent.cache_nonce`), or None

        ckpt = None
        if (root := checkpoint_root()) is not None:
            ckpt = SessionCheckpoints(
                root,
                state.sample_id,
                state.epoch,
                {
                    "arm": arm,
                    "policy": f"{cls.__module__}.{cls.__qualname__}",
                    "knobs": knobs,
                    "knob_env": knob_env(),
                    "models": {"agent": model_identity(model), "cm": model_identity(cm_model), "probe": model_identity(probe_model)},
                    "window": window,
                    "threshold": t_abs,
                    "max_turns_per_item": max_turns_per_item,
                    "checkpoints": list(checkpoints),
                    "world": {"id": world.id, "hash": world.content_hash()},
                    "code": code_version([cls.__module__, *cls.CODE_MODULES]),
                }
                | ({"cache_nonce": nonce} if nonce else {}),  # a session resumes only under the nonce its history carries
            )
        payload = ckpt.load() if ckpt else None
        restored: list[ChatMessage] | None = None
        records = SessionRecords()
        if payload is not None:
            try:
                restored, records = load_messages(payload["history"]), SessionRecords.from_state(payload["records"])
                done, segments, failures = int(payload["done"]), list(payload["segments"]), list(payload["failures"])
            except (KeyError, TypeError, ValueError) as e:  # a damaged file under a valid key: start afresh, never loop on it
                log.warning("session checkpoint %s does not restore (%s: %s); starting the session afresh", ckpt.path, type(e).__name__, e)
                payload, restored, records, ckpt.last = None, None, SessionRecords(), None
        if payload is None:
            done, segments, failures = 0, [], []
        ctx = SessionContext(world=world, window=window, threshold=t_abs, agent_model=model, cm_model=cm_model, records=records,
                             session_tools=session_tools, max_turns_per_item=max_turns_per_item, nonce=nonce)
        policy = cls(ctx, **knobs)
        extra = list(policy.tools())
        policy_tools = [tool_name(t) for t in extra]
        if clash := sorted(set(policy_tools) & {tool_name(t) for t in base_tools}):
            raise ValueError(f"{cls.__name__} tools {clash} clash with the session's own tools")
        agent_base = list(policy.agent_tools(base_tools))
        tools = [*agent_base, *extra]
        ctx.tools = tools
        arm_record = {
            "name": arm,
            "window": window,
            "max_turns_per_item": max_turns_per_item,
            "checkpoints": list(checkpoints),
            "threshold": t_abs,
            "policy": cls.__name__,
            "knobs": knobs,
            "policy_tools": policy_tools,
            "policy_tool_tokens": tool_schema_tokens(extra),
        }
        if len(agent_base) != len(base_tools):
            arm_record["agent_tools"] = [tool_name(t) for t in agent_base]
        saved = [len(records.views), len(records.probes)]  # records already in the last checkpoint
        segments.append({"attempt": len(segments) + 1, "sample_uuid": state.uuid, "after_item": done, "views_from": len(records.views), "probes_from": len(records.probes)})
        overflow_at = None
        finished = False
        history: list[ChatMessage] = []

        def save(done_now: int) -> None:
            if ckpt is None:
                return
            ckpt.save({
                "sample_id": state.sample_id,
                "epoch": state.epoch,
                "done": done_now,
                "history": dump_messages(history),
                "policy": policy.state_dict(),
                "recorder": rec.state_dict(),
                "records": records.state_dict(),
                "segments": segments,
                "failures": failures,
            })
            saved[:] = [len(records.views), len(records.probes)]

        def fail(error: BaseException) -> None:
            """An error ends this attempt: keep the last checkpoint for the retry, adding the failure (its error and the
            usage of the calls made since the save, which the retry makes again), this attempt's segment and the
            resumes so far."""
            if ckpt is None or ckpt.last is None:
                return
            lost = records.views[saved[0] :] + records.probes[saved[1] :]
            failures.append({"attempt": segments[-1]["attempt"], "sample_uuid": state.uuid, "after_item": ckpt.last["done"], "error": f"{type(error).__name__}: {error}"[:300], "usage": usage_summary(lost)})
            last = ckpt.last
            keep = ("sample_id", "epoch", "done", "history", "policy", "recorder")
            ckpt.save({**{k: last[k] for k in keep}, "records": {**last["records"], "resumes": records.resumes}, "segments": segments, "failures": failures})

        async def call(item_start: int, done_now: int, position: int) -> tuple[int, bool]:
            """One agent generation on the policy's view; returns (view tokens, made tool calls)."""
            v = await policy.view(history, item_start, done_now)
            vt = view_tokens(v)
            if vt > t_abs and (smaller := await policy.on_threshold(history, v, vt)) is not None:
                after = view_tokens(smaller)
                ctx.log("threshold", view_tokens=vt, view_tokens_after=after)
                v, vt = smaller, after
            if vt > window:
                raise SessionOverflow(position)
            try:
                output = await model.generate(v, tools=tools)
            except LimitExceededError:
                ctx.record_limit_call("agent", model, vt, v)  # Inspect counted it: so do the records
                raise
            ctx.record_call("agent", model, vt, output)  # before the append, which can trip the message limit
            history.append(output.message)
            appended: list[ChatMessage] = [output.message]
            acted = bool(output.message.tool_calls)
            if acted:
                result = await execute_tools(history, tools)
                history.extend(result.messages)
                appended += result.messages
                if policy_tools:
                    results = {m.tool_call_id: m for m in result.messages if isinstance(m, ChatMessageTool)}
                    for c in output.message.tool_calls:
                        if c.function in policy_tools:
                            r = results.get(c.id)
                            rec.record(c.function, c.arguments, policy=True, **({"error": r.error.message} if r is not None and r.error else {}))
            await policy.after_generate(history, v, output, appended)
            return vt, acted

        def tally(item: dict) -> None:
            """The item's management calls and its usage by kind (agent, cm, probe)."""
            mine = [c for c in records.calls() if c["item"] == item["position"]]
            item["cm_calls"] = sum(c["kind"] == "cm" for c in mine)
            item["usage"] = usage_by(mine, "kind")

        async def probe(k: int) -> None:
            before = _state_json(policy)
            view = await policy.probe_view(history, k)
            if _state_json(policy) != before:
                raise RuntimeError(f"{cls.__name__}.probe_view changed the policy's state: probes must not affect the session")
            msgs = [*view, ChatMessageUser(content=PROBE_PROMPT)]
            try:
                out = await probe_model.generate(msgs, config=GenerateConfig(response_schema=ResponseSchema(name="state_probe", json_schema=PROBE_SCHEMA, strict=True)))
            except LimitExceededError as e:
                # Counted by Inspect, so recorded (usage only); flagged `limit`, it is a checkpoint not taken.
                if (usage := limit_call_usage(msgs)) is not None:
                    records.probes.append({"k": k, "answer": None, "error": f"LimitExceededError: {e}"[:300], "view_tokens": view_tokens(msgs),
                                           "scores": probe_score(None, gen_f8.state_at(world, k)), "usage": usage_record(usage), "item": k,
                                           "kind": "probe", "model": str(probe_model), "limit": True})
                raise
            try:
                answer, error = json.loads(out.completion), None
            except (json.JSONDecodeError, TypeError) as e:
                answer, error = None, f"{type(e).__name__}: {e}"
            records.probes.append({
                "k": k,
                "answer": answer,
                "error": error,
                "view_tokens": view_tokens(msgs),
                "scores": probe_score(answer if isinstance(answer, dict) else None, gen_f8.state_at(world, k)),
                "usage": usage_record(out.usage),
                "item": k,
                "kind": "probe",
                "model": str(probe_model),
            })

        try:
            try:
                if payload is None:
                    addendum = policy.system_addendum()
                    system = ChatMessageSystem(content=with_nonce(gen_f8.system_prompt(world) + (f"\n\n{addendum}" if addendum else ""), nonce))
                    # The full history lives in state.messages itself, so Inspect's message limit applies to its appends
                    # and a session cut short logs exactly what ran.
                    state.messages = [system, ChatMessageUser(content=gen_f8.start_message(world))]
                    history = state.messages
                    ctx.system = system
                    await policy.start(history)
                    save(0)
                else:
                    state.messages = restored  # a restored history counts toward the message limit, as it did
                    history = state.messages
                    ctx.system = history[0]
                    rec.load_state(payload["recorder"])
                    policy.load_state(payload["policy"], history)
                    records.resumes.append({
                        "resume": len(records.resumes) + 1,
                        "after_item": done,
                        "sample_uuid": state.uuid,
                        "checkpoint": ckpt.path.name,
                        "saved_at": payload["saved_at"],
                        "restored": usage_summary(records.calls()),
                    })
                for task in world.tasks[done:]:
                    pos, cid = task.tags["position"], task.tags["case_id"]
                    rec.item = ctx.position = pos
                    ctx.stage = "item"
                    start = len(history)
                    history.append(ChatMessageUser(content=task.prompt))
                    first = decision = None
                    generations, nudged = 0, False
                    for _ in range(max_turns_per_item):
                        vt, acted = await call(start, pos - 1, pos)
                        generations += 1
                        first = vt if first is None else first
                        if rec.answered(task):
                            decision = vt
                            break
                        if not acted:
                            if nudged:
                                break
                            tool = "finish" if task.tags["kind"] == "ticket" else "submit_decision"
                            history.append(ChatMessageUser(content=f"Complete case {cid} with the tools, then call {tool} with case_id {cid}."))
                            nudged = True
                    item: dict[str, Any] = {
                        "position": pos,
                        "case_id": cid,
                        "kind": task.tags["kind"],
                        "dependency": task.tags["dependency"],
                        "dependency_kinds": task.tags["dependency_kinds"],
                        "generations": generations,
                        "view_tokens_first": first,
                        "view_tokens_decision": decision,
                        "answered": decision is not None,
                        "success": item_success(world, task, rec.events),
                    }
                    records.items.append(item)
                    tally(item)
                    ctx.stage = "boundary"
                    await policy.after_item(history, pos)
                    if pos in checkpoints:
                        await probe(pos)
                    tally(item)  # with the boundary's management calls and the probe
                    save(pos)
                rec.item = ctx.position = n + 1
                ctx.stage = "report"
                start = len(history)
                history.append(ChatMessageUser(content=gen_f8.REPORT_REQUEST))
                nudged = False
                for _ in range(max_turns_per_item):
                    _, acted = await call(start, n, n + 1)
                    if rec.report is not None:
                        break
                    if not acted:
                        # One nudge, as for cases: a report written as text (or a refusal) gets one more chance.
                        if nudged:
                            break
                        history.append(ChatMessageUser(content=REPORT_NUDGE))
                        nudged = True
            except SessionOverflow as e:
                overflow_at = e.position
            finished = True
        except LimitExceededError:
            finished = True  # a sample limit ends the session for good: Inspect scores what ran
            raise
        except Exception as e:
            fail(e)
            raise
        finally:
            # Written even when a sample limit cuts the session short, so its items and events are scored.
            calls = records.calls()
            store().set(ITEMS, records.items)
            store().set(VIEWS, records.views)
            store().set(PROBES, records.probes)
            store().set(EVENTS, rec.events)
            store().set(REPORT, rec.report)
            store().set(OVERFLOW, overflow_at)
            store().set(CM_EVENTS, records.cm_events)
            store().set(USAGE, {"by_kind": usage_by(calls, "kind"), "by_model": usage_by(calls, "model")})
            store().set(RESUME, resume_summary(records.views, records.probes, records.resumes, segments, failures))
            store().set("arm", arm_record)
            for key, value in policy.records().items():
                store().set(key, value)
            # A time or working limit ends the sample by cancelling the solver, not by raising LimitExceededError into it:
            # the session is over all the same, and its checkpoint must never be resumed.
            if not finished and inspect_limit_hit() is not None:
                finished = True
            if finished and ckpt is not None:
                ckpt.complete()
        return state

    return solve
