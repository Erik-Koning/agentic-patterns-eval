"""Topology arms inside Study G's F8 sessions (BUILD_PLAN B9; CONTEXT_MANAGEMENT_AUDIT §2.3, §6.2; G-H1, G-H3).

| Arm | What runs | Switches (brief §4.1) |
|---|---|---|
| S1 | CM0 itself (`FullHistory`), registered under its topology name and recorded as S1 | S1 |
| M1 | an orchestrator plus 3 identical workers | DEC=1, ISO=1, CONC=1 |
| M2 | M1 with 3 specialist workers (`specialists.py`, F8 domains) | M1 + SPEC=1 |

**Built on the ContextPolicy layer, not beside it.** The session loop (`ape.agent.session`) runs unchanged: the
session's agent is the **orchestrator**, so the loop's history, view, W check, per-case turn cap and nudge, the
report phase, the forked probes, the records and the mid-session checkpoints all apply to it as to CM0. The team adds
only what a policy may add:
- `agent_tools`: the orchestrator keeps only `submit_decision`, `finish` and `submit_shift_report`; it cannot open
  files or run procedure tools.
- `tools`: one policy tool, `delegate` (1 to 3 subtasks per call; M2's name a specialist), which runs workers.
- `system_addendum`: the team instructions (`session_prompts.ORCHESTRATOR_ADDENDUM`).
- `records`: the team's records and per-agent accounting, written when the session ends.

**What persists across items.** Only the orchestrator's history, under the CM0 rule: its view is the full
append-only history, with no context management, so the S1 / M1 / M2 contrast is a topology contrast and is not
confounded with a context-management strategy. It holds every case message, its own delegations and answers, and the
workers' results, never the bulky customer and order files, which stay in the workers' contexts: that is the isolation
G-H3 measures. Nothing persists for a worker: each subtask is a fresh context.

**What a worker sees per subtask:** the session's system prompt (the base prompt, the corpus and the shift rules, the
same as every agent; no team addendum), then one user message: the worker note (M2: plus its specialty) and the
subtask text the orchestrator wrote. Its tools are the session's tools minus the orchestrator's three (lookups and the
procedure tools; M2: its specialist's subset), plus `report`. It ends when it reports, on a second text-only reply in a
row (one nudge, as the loop's per-case rule), or at the loop's per-case turn cap; only its result (or a marked note)
returns to the orchestrator. Workers of one `delegate` call run concurrently.

**Who answers.** The orchestrator calls the case-answer tools (with the case IDs) and the report tool; workers make the
lookups and a ticket's procedure calls. Every tool call goes through the session's recorder, so the scorer reads
workers' and the orchestrator's calls alike, by the item being worked on.

**Window.** W applies to every agent's input on the same meter (`context_policy.view_tokens`): the loop checks the
orchestrator's views, and each worker call is checked before it is made. An overflow anywhere ends the session as for
CM0 (`SessionOverflow`: the current item and every later one fail). It propagates through the `delegate` tool call:
Inspect's `execute_tools` re-raises exceptions it does not map. T_abs triggers nothing (no agent manages context).

**Calls and accounting.** Every worker call is recorded like an agent call (`SessionContext.record_call`, kind
`agent`, item, view tokens, model, usage) plus its `agent` ID and `role`, so the loader (`g_load`) counts workers'
calls and tokens with the orchestrator's. `mas_accounting` sums the per-call records by agent and role (calls; input,
output, reasoning, cache-read, cache-write and total tokens; cost when priced): it is built from the session's own
records, so it covers restored calls after a resume and sums exactly to `f8_usage`'s agent usage (probes are separate,
kind `probe`). A call that trips a token or cost limit is recorded too (from the transcript, flagged `limit`:
`SessionContext.record_limit_call`), so both equal Inspect's own usage then. Each worker's wall-clock is in
`mas_agents`.

**Probes** go to the orchestrator's current view (it holds the session's state), as the loop does for every arm;
they are never appended.

**Failures.** A limit hit inside a worker (Inspect's `execute_tools` would turn it into a tool error) is held and
re-raised from `after_generate`, so the sample ends with its records. Any other exception in a worker (a provider
error) propagates like an error in the orchestrator's own call: the attempt fails and Inspect's retry resumes from the
last completed item (`session_checkpoint`), rather than the item failing silently. A worker that chats, never reports
or calls tools it lacks returns a marked result (tool errors for the latter), and `delegate` with no subtask, more
than 3 or an empty one is a tool error that runs nothing.

**Checkpoint/resume.** The team's state (its rounds and workers' records) is part of the policy state saved after
every item; the orchestrator's history, the recorder and the per-call records are the loop's. Workers keep nothing
between subtasks, so a resumed session continues exactly as an uninterrupted one from the next item (the interrupted
item reruns, as for every arm).
"""

import time
from copy import deepcopy
from typing import Any, ClassVar

import anyio
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, execute_tools
from inspect_ai.tool import ToolDef, ToolError, ToolParam, ToolParams
from inspect_ai.util import LimitExceededError

from ...llm.tokens import count_tokens, truncate_to_tokens
from ...worlds import gen_f8
from ..cache_nonce import with_nonce
from ..context_policy import FullHistory, SessionOverflow, register_policy, tool_name, view_tokens
from . import session_prompts as P
from .specialists import N_WORKERS, Specialist, specialists

TOPOLOGY_ARMS = ("S1", "M1", "M2")
ORCHESTRATOR_TOOLS = ("submit_decision", "finish", "submit_shift_report")
RESULT_MAX_TOKENS = 2000  # a worker's result as the orchestrator reads it (clipped with a marker), as in the main study
S1_SWITCHES = {"DEL": "MONO", "DEC": 0, "ISO": 0, "CONC": 0, "ENS": 1, "SPEC": 0, "STATE": "none", "CTRL": "dynamic", "COMM": "none", "HET": 0, "CMP": 0}
USAGE_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "input_tokens_cache_read", "input_tokens_cache_write", "total_cost")


def clip(text: str, max_tokens: int = RESULT_MAX_TOKENS) -> str:
    text = str(text or "")
    if count_tokens(text) <= max_tokens:
        return text
    return truncate_to_tokens(text, max_tokens) + f"\n[... cut to {max_tokens} tokens]"


def _params(**props: ToolParam) -> ToolParams:
    return ToolParams(properties=props, required=list(props))


async def _gather(fns: list) -> list:
    """Run worker coroutines as sibling tasks; results in order. A failure cancels the rest; a limit error wins over an
    overflow, which wins over anything else, and the winner is raised bare (the loop handles each)."""
    results: list[Any] = [None] * len(fns)

    def runner(i: int):
        async def run() -> None:
            results[i] = await fns[i]()

        return run

    try:
        async with anyio.create_task_group() as tg:
            for i in range(len(fns)):
                tg.start_soon(runner(i))
    except BaseExceptionGroup as group:
        leaves: list[BaseException] = []
        stack: list[BaseException] = [group]
        while stack:
            e = stack.pop(0)
            if isinstance(e, BaseExceptionGroup):
                stack[:0] = list(e.exceptions)
            else:
                leaves.append(e)
        for kind in (LimitExceededError, SessionOverflow, Exception):
            if first := next((x for x in leaves if isinstance(x, kind)), None):
                raise first from None
        raise
    return results


def accounting(views: list[dict], probes: list[dict]) -> dict:
    """Per agent and per role: calls and usage summed from the session's per-call records (agent calls carry the worker's
    `agent` ID; untagged agent calls are the orchestrator's), probes apart."""
    agents: dict[str, dict] = {}
    roles: dict[str, dict] = {}

    def add(bucket: dict, usage: dict | None) -> None:
        bucket["calls"] += 1
        for f in USAGE_FIELDS:
            if usage and usage.get(f) is not None:
                bucket[f] = bucket.get(f, 0) + usage[f]

    for v in views:
        if v.get("kind", "agent") != "agent":
            continue
        aid, role = v.get("agent", "orchestrator"), v.get("role", "orchestrator")
        a = agents.setdefault(aid, {"role": role, "items": [], "calls": 0})
        if v["item"] not in a["items"]:
            a["items"].append(v["item"])
        add(a, v.get("usage"))
        add(roles.setdefault(role, {"calls": 0}), v.get("usage"))
    probe = {"calls": 0}
    for p in probes:
        add(probe, p.get("usage"))
    totals = {"calls": 0}
    for r in roles.values():
        for k, n in r.items():
            totals[k] = totals.get(k, 0) + n
    return {"agents": agents, "roles": roles, "totals": totals, "probes": probe}


class SessionTeam(FullHistory):
    """M1: the session's agent is an orchestrator (CM0's view) that delegates each case's work to isolated workers."""

    name = "team"
    SPECIALIZED: ClassVar[bool] = False
    CODE_MODULES = ("ape.agent.multi.session_prompts", "ape.agent.multi.specialists")

    def __init__(self, session, **knobs: Any):
        super().__init__(session, **knobs)
        self.specs: list[Specialist] | None = specialists(session.world) if self.SPECIALIZED else None
        self.rounds: list[dict] = []
        self.workers: list[dict] = []
        self._pending: LimitExceededError | None = None

    # -- what the orchestrator gets ---------------------------------------------------------------------------------

    def agent_tools(self, tools: list[ToolDef]) -> list[ToolDef]:
        return [t for t in tools if tool_name(t) in ORCHESTRATOR_TOOLS]

    def tools(self) -> list[ToolDef]:
        if self.specs is None:
            item = ToolParam(type="string")
            description = P.DELEGATE_DESCRIPTION
        else:
            item = ToolParam(
                type="object",
                properties={
                    "specialist": ToolParam(type="string", description="The specialist who carries out the subtask.", enum=[s.name for s in self.specs]),
                    "task": ToolParam(type="string", description="The subtask, self-contained."),
                },
                required=["specialist", "task"],
                additionalProperties=False,
            )
            description = P.DELEGATE_SPECIALIST_DESCRIPTION
        params = _params(subtasks=ToolParam(type="array", description=f"1 to {N_WORKERS} self-contained subtasks.", items=item))
        # Never Inspect's 16 KiB default (a silent cut): each result is clipped already, with a marker.
        return [ToolDef(self._delegate, name="delegate", description=description.format(n_workers=N_WORKERS), parameters=params,
                        parallel=False, max_output=N_WORKERS * (RESULT_MAX_TOKENS * 8 + 512))]

    def system_addendum(self) -> str:
        if self.specs is None:
            team = P.TEAM_IDENTICAL.format(tools=", ".join(self.worker_tools(None)))
        else:
            team = P.TEAM_SPECIALISTS.format(roster="\n".join(
                P.ROSTER_LINE.format(name=s.name, covers=s.coverage, tools=", ".join(self.worker_tools(s))) for s in self.specs))
        return P.ORCHESTRATOR_ADDENDUM.format(n_workers=N_WORKERS, team=team)

    async def after_generate(self, history, view, output, appended) -> None:
        if self._pending is not None:  # a worker hit a sample limit inside delegate (execute_tools made it a tool error)
            raise self._pending

    # -- workers --------------------------------------------------------------------------------------------------

    def worker_tools(self, spec: Specialist | None) -> dict[str, ToolDef]:
        tools = {n: t for n, t in self.session.session_tools.items() if n not in ORCHESTRATOR_TOOLS}
        if spec is not None and spec.tools is not None:
            tools = {n: t for n, t in tools.items() if n in spec.tools}
        return tools

    def _subtasks(self, raw: Any) -> list[tuple[Specialist | None, str]]:
        items = raw if isinstance(raw, list) else []
        if not 1 <= len(items) <= N_WORKERS:
            raise ToolError(P.DELEGATE_COUNT.format(n_workers=N_WORKERS, got=len(items)))
        by_name = {s.name: s for s in self.specs or []}
        out: list[tuple[Specialist | None, str]] = []
        for n, item in enumerate(items, 1):
            if self.specs is None:
                spec, text = None, str(item).strip()
            else:
                spec = by_name.get(str(item.get("specialist", "")).strip()) if isinstance(item, dict) else None
                text = str(item.get("task", "")).strip() if isinstance(item, dict) else ""
                if spec is None:
                    raise ToolError(P.DELEGATE_SPECIALIST.format(n=n, names=", ".join(by_name)))
            if not text:
                raise ToolError(P.DELEGATE_EMPTY_TASK.format(n=n))
            out.append((spec, text))
        return out

    async def _delegate(self, subtasks: list) -> str:
        items = self._subtasks(subtasks)
        r = len(self.rounds) + 1
        record = {"round": r, "item": self.session.position, "subtasks": []}
        runs = []
        for i, (spec, text) in enumerate(items, 1):
            w = {"id": f"w{r}.{i}", "role": spec.name if spec else "worker", "item": self.session.position, "round": r, "slot": i,
                 "subtask": clip(text, 300), "turns": 0, "nudged": False, "tool_calls": [], "stop": None, "status": None, "result": None, "wall_s": 0.0}
            self.workers.append(w)
            record["subtasks"].append({"worker": w["id"], "role": w["role"], "task": clip(text, 500)})
            runs.append((w, text, spec))
        self.rounds.append(record)

        def unit(w, text, spec):
            return lambda: self._run_worker(w, text, spec)

        try:
            results = await _gather([unit(*run) for run in runs])
        except LimitExceededError as e:
            self._pending = e
            raise
        lines = []
        for n, ((w, _, spec), (status, text)) in enumerate(zip(runs, results, strict=True), 1):
            record["subtasks"][n - 1].update(status=status, result=clip(text, 500))
            result = {"reported": text, "text": P.RESULT_TEXT.format(text=text)}.get(status, P.RESULT_TURN_CAP.format(text=text))
            lines.append(P.RESULT_LINE.format(n=n, worker=spec.name if spec else f"worker {n}", result=result))
        return "\n\n".join(lines)

    async def _run_worker(self, w: dict, subtask: str, spec: Specialist | None) -> tuple[str, str]:
        """One subtask in a fresh context: (status, result text). Raises only an overflow, a limit or an error that
        should end the attempt (module docstring)."""
        ctx = self.session
        reported: list[str] = []

        async def report(result: str) -> str:
            if reported:
                return P.REPORT_AGAIN
            reported.append(str(result))
            return "Result reported."

        tools = [*self.worker_tools(spec).values(),
                 ToolDef(report, name="report", description=P.REPORT_DESCRIPTION, parameters=_params(result=ToolParam(type="string", description="Your result.")))]
        note = P.WORKER_NOTE if spec is None else f"{P.WORKER_NOTE} {P.WORKER_SPECIALTY.format(covers=spec.coverage)}"
        messages: list[ChatMessage] = [ChatMessageSystem(content=with_nonce(gen_f8.system_prompt(ctx.world), ctx.nonce)),
                                       ChatMessageUser(content=P.WORKER_SUBTASK.format(note=note, subtask=subtask))]
        status, t0 = "turn_cap", time.perf_counter()
        try:
            for _ in range(ctx.max_turns_per_item):
                tokens = view_tokens(messages)
                if tokens > ctx.window:
                    status = "overflow"
                    raise SessionOverflow(ctx.overflow_position())
                try:
                    output = await ctx.agent_model.generate(messages, tools=tools)
                except LimitExceededError:
                    # Inspect counted the call that tripped the limit: record it, so mas_accounting and f8_usage equal its usage.
                    if (entry := ctx.record_limit_call("agent", ctx.agent_model, tokens, messages)) is not None:
                        entry.update(agent=w["id"], role=w["role"])
                    raise
                ctx.record_call("agent", ctx.agent_model, tokens, output).update(agent=w["id"], role=w["role"])
                messages.append(output.message)
                w["turns"] += 1
                if output.message.tool_calls:
                    w["tool_calls"] += [c.function for c in output.message.tool_calls]
                    messages.extend((await execute_tools(messages, tools)).messages)
                    if reported:
                        status = "reported"
                        break
                elif w["nudged"]:
                    status = "text"
                    break
                else:
                    messages.append(ChatMessageUser(content=P.NUDGE_WORKER))
                    w["nudged"] = True
        except LimitExceededError:
            status = "limit"
            raise
        except BaseException:
            status = status if status == "overflow" else "interrupted"
            raise
        finally:
            w["wall_s"] = round(w["wall_s"] + time.perf_counter() - t0, 3)
            w["stop"] = status
        text = reported[0] if reported else next((m.text for m in reversed(messages) if isinstance(m, ChatMessageAssistant) and m.text), "")
        w["status"], w["result"] = status, clip(text, 500)
        return status, clip(text)

    # -- state and records ------------------------------------------------------------------------------------------

    def state_dict(self) -> dict:
        # A copy: the loop keeps the last saved state and saves it again when an attempt fails, after this item's
        # rounds have been appended to the live lists.
        return deepcopy({"rounds": self.rounds, "workers": self.workers})

    def load_state(self, state: dict, history: list[ChatMessage]) -> None:
        self.rounds, self.workers = deepcopy(list(state.get("rounds") or [])), deepcopy(list(state.get("workers") or []))

    def switches(self) -> dict:
        return {**S1_SWITCHES, "DEC": 1, "ISO": 1, "CONC": 1, "SPEC": int(self.SPECIALIZED), "delivery_arm": "session corpus (system prompt)"}

    def records(self) -> dict[str, Any]:
        recs = self.session.records
        orchestrator_calls = sum(1 for v in recs.views if v.get("kind", "agent") == "agent" and "agent" not in v)
        out = {
            "mas_switches": self.switches(),
            "mas_params": {"n_workers": N_WORKERS, "worker_turns": self.session.max_turns_per_item, "workers_scheduled": "concurrent",
                           "orchestrator_view": "full history (CM0 rule)", "orchestrator_tools": [*ORCHESTRATOR_TOOLS, "delegate"]},
            "mas_agents": [{"id": "orchestrator", "role": "orchestrator", "calls": orchestrator_calls}, *self.workers],
            "mas_rounds": self.rounds,
            "mas_accounting": accounting(recs.views, recs.probes),
        }
        if self.specs is not None:
            out["mas_specialists"] = [{"name": s.name, "covers": list(s.covers), "tools": list(s.tools or ())} for s in self.specs]
        return out


class SpecialistTeam(SessionTeam):
    """M2: M1 with specialist workers (SPEC=1)."""

    name = "team-specialists"
    SPECIALIZED = True


register_policy("S1", FullHistory)  # S1 in sessions is CM0, run and recorded under its topology name
register_policy("M1", SessionTeam)
register_policy("M2", SpecialistTeam)
