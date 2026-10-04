"""Reusable multi-agent primitives: a worker run, the orchestrator loop, a council, and an ensemble with its vote.

Plain async functions over a `core.Team` and a `TeamConfig`, independent of any task loop: the main study's solvers
(`multi/solvers.py`) call them once per sample, and Study G's session arms (BUILD_PLAN B9) can call `run_worker` and
`run_orchestrator` per session item.

**Orchestrator (M1, M1s, M1k, M2).** One orchestrator with three tools: `plan` (forced on its first turn: the planning
step, DEC=1), `delegate` (up to `n_workers` subtasks per call) and the task's answer tool (only the orchestrator can
answer). It can call `delegate` and `plan` as often as it likes (CTRL=dynamic), so F2's hop k can wait for hop k-1.
Each subtask runs in a **fresh worker context** (ISO=1): the worker sees the arm's system prompt and knowledge, the
worker note and its subtask text, nothing of the orchestrator's history, plan or other subtasks; it has the task's
non-terminal tools plus `report`, and only its result returns. One `delegate` call is one round:
- **M1 and M1s get identical inputs.** Workers of a round run concurrently (M1) or one after another (M1s), each in
  its own task with its own copy of the environment as it stood when the round began; after the round their
  environment changes are folded into the sample's store in subtask order (`core.merge_env`). So no worker can see a
  round-mate's effects in either arm, the results come back in subtask order, and no prompt says how workers are
  scheduled: concurrency changes only latency (and, at a limit hit, which calls were in flight).
- Two `delegate` calls in one turn run as two rounds, in the model's order (`delegate` is not parallel-safe).
- A worker that stops replying or reaches its turn cap returns a marked result; the round and the sample go on. A
  limit or an exception (a provider error) in a worker ends the sample, as in S1, so Inspect retries it
  (`Team.note_error`; BUILD_REVIEW A-2). The same holds for council members, the chair, ensemble attempts and the
  aggregator: an exception is the sample's, never a lost proposal or vote.

**Council (M7).** k members each attempt the task independently (DEL as S1; their answer tool records a proposal with a
rationale instead of answering), then `rounds` critique rounds: each member's own history continues with the others'
latest proposals and rationales, and it re-proposes (it may use the tools again). Members are synchronous per round
(all see the previous round's proposals; none sees a round-mate's current one) and each keeps its own environment
copy. Then a chair, in a fresh context with the arm's knowledge and only the answer tool, reads the final proposals and
submits. COMM=real.

**Ensemble (S8k3).** k full, independent S1 attempts: the gate's own `kb_agent` loop, unchanged, each with its own
TaskState and its own environment (an F3 attempt's mutations never reach another attempt). Majority vote over the
answer as the scorer compares it (`answer_key`); with no unique plurality an LLM aggregator chooses among the tied
candidates (falling back to the earliest tied attempt if it does not choose). The winning attempt's environment (its
calls and answer) becomes the sample's, so the scorers grade the chosen attempt's end state; faults fired and lookups
are unioned over attempts for the fault analyses. No main-study family has a free-form answer, so the aggregator is
used on ties only.
"""

import json
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState
from inspect_ai.tool import ToolDef, ToolError, ToolParam, ToolParams
from inspect_ai.util import LimitExceededError, Store

from ...scorers.success import norm_f5, norm_id, norm_ratings
from ...worlds.env_tools import ANSWER, CALLS, FAULTS_FIRED, LOOKUPS, always_on, build_tools
from ...worlds.spec import TaskItem
from ..kb_react import COMPILE_LOG, STEP_LOG
from . import prompts as P
from .core import TEXT_MAX_TOKENS, AgentRecord, Delivery, Team, as_json, clip, env_data, last_assistant_text, merge_env, react_loop, run_isolated
from .specialists import N_WORKERS, Specialist

CRITIQUE_ROUNDS = 2  # fixed a priori (brief §4.2)
COUNCIL_K = 3
ENSEMBLE_K = 3


@dataclass
class TeamConfig:
    """How a team runs: the arm's delivery, the tool-exposure policy and the turn caps (each worker and each council
    phase gets the task's `max_turns`, so a cap never binds a subtask before it would bind the whole task in S1).
    `tools` replaces `worker_tools` (the task's non-terminal tools) for teams whose environment is not
    `env_tools.build_tools`, e.g. Study G's sessions (B9). `notes` (the role notes of the arm's prompt variant) and
    `clip_tokens` (the cap on a worker's result or a member's rationale as another agent reads it) are the arm's tuning
    knobs (`knobs.py`); the defaults are B2's."""

    delivery: Delivery
    exposure: str
    max_turns: int
    worker_turns: int
    n_workers: int = N_WORKERS
    concurrent: bool = True
    specialists: list[Specialist] | None = None
    tools: Callable[[Team, Specialist | None], tuple[dict[str, ToolDef], list[str]]] | None = None
    notes: P.RoleNotes = P.VARIANTS[P.DEFAULT_VARIANT]
    clip_tokens: int = TEXT_MAX_TOKENS

    def worker_tools(self, team: Team, specialist: Specialist | None = None) -> tuple[dict[str, ToolDef], list[str]]:
        return (self.tools or worker_tools)(team, specialist)


@dataclass
class WorkerResult:
    worker: str
    label: str
    status: str  # reported | text | turn_cap | error
    text: str

    def line(self, n: int) -> str:
        result = {
            "reported": self.text,
            "text": P.RESULT_TEXT.format(text=self.text),
            "turn_cap": P.RESULT_TURN_CAP.format(text=self.text),
        }.get(self.status, P.RESULT_ERROR)
        return P.RESULT_LINE.format(n=n, worker=self.label, result=result)


def _params(**props: ToolParam) -> ToolParams:
    return ToolParams(properties=props, required=list(props))


def _turn(agent: AgentRecord) -> int:
    return agent.steps[-1]["turn"] if agent.steps else 0


# --- worker ---------------------------------------------------------------------------------------------------------


def worker_tools(team: Team, specialist: Specialist | None = None) -> tuple[dict[str, ToolDef], list[str]]:
    """The task's non-terminal tools (a specialist's subset on F3) and which of them are always on."""
    tools = build_tools(team.world, team.task)
    tools.pop(team.task.answer_tool, None)
    if specialist is not None and specialist.tools is not None:
        tools = {n: t for n, t in tools.items() if n in specialist.tools}
    return tools, [n for n in always_on(team.world) if n in tools]


async def run_worker(team: Team, agent: AgentRecord, subtask: str, cfg: TeamConfig, specialist: Specialist | None = None) -> WorkerResult:
    """One subtask in a fresh context; returns only its result. Raises only a limit or an error, which end the sample
    (recorded on the agent first)."""
    tools, always = cfg.worker_tools(team, specialist)
    reported: list[str] = []

    async def report(result: str) -> str:
        if reported:
            return P.REPORT_AGAIN
        reported.append(str(result))
        return "Result reported."

    tools["report"] = ToolDef(report, name="report", description=P.REPORT_DESCRIPTION, parameters=_params(result=ToolParam(type="string", description="Your result.")))
    always = [*always, "report"]
    note = cfg.notes.worker if specialist is None else f"{cfg.notes.worker} {P.WORKER_SPECIALTY.format(covers=specialist.coverage)}"
    query = subtask if specialist is None else f"{specialist.query_prefix}\n{subtask}"
    messages: list[ChatMessage] = []
    try:
        async with team.running(agent):
            ctx = await cfg.delivery.compile(agent, query, 0)
            messages = [cfg.delivery.system(ctx), ChatMessageUser(content=P.WORKER_SUBTASK.format(note=note, subtask=subtask))]
            await react_loop(team, agent, messages, delivery=cfg.delivery, ctx=ctx, query=query, tools=tools, always=always,
                             exposure=cfg.exposure, max_turns=cfg.worker_turns, done=lambda: bool(reported), nudge=P.NUDGE_WORKER)
    except Exception as e:  # a limit or a provider error ends the sample, as in S1 (BUILD_REVIEW A-2)
        if not isinstance(e, LimitExceededError):
            agent.error, agent.stop = f"{type(e).__name__}: {e}"[:500], "error"
        raise
    if reported:
        status, text = "reported", reported[0]
    else:
        status, text = agent.stop or "turn_cap", last_assistant_text(messages)
    agent.info.update(status=status, result=clip(text, 500))
    label = specialist.name if specialist is not None else f"worker {agent.info.get('slot', '?')}"
    return WorkerResult(agent.id, label, status, clip(text, cfg.clip_tokens))


# --- orchestrator ---------------------------------------------------------------------------------------------------


def plan_tool(agent: AgentRecord, sink: list[dict]) -> ToolDef:
    async def plan(subtasks: list) -> str:
        items = [str(s).strip() for s in subtasks or [] if str(s).strip()]
        if not items:
            raise ToolError(P.PLAN_EMPTY)
        sink.append({"agent": agent.id, "turn": _turn(agent), "version": len(sink) + 1, "subtasks": [clip(s, 300) for s in items]})
        return f"Plan recorded: {len(items)} subtasks."

    return ToolDef(plan, name="plan", description=P.PLAN_DESCRIPTION,
                   parameters=_params(subtasks=ToolParam(type="array", description="The subtasks, in order.", items=ToolParam(type="string"))))


def _subtasks(raw: Any, cfg: TeamConfig) -> list[tuple[Specialist | None, str]]:
    """The delegate call's subtasks, validated (a ToolError tells the model what to fix; nothing runs)."""
    items = raw if isinstance(raw, list) else []
    if not 1 <= len(items) <= cfg.n_workers:
        raise ToolError(P.DELEGATE_COUNT.format(n_workers=cfg.n_workers, got=len(items)))
    out: list[tuple[Specialist | None, str]] = []
    by_name = {s.name: s for s in cfg.specialists or []}
    for n, item in enumerate(items, 1):
        if cfg.specialists is None:
            spec, text = None, str(item if isinstance(item, str) else json.dumps(item)).strip()
        else:
            spec = by_name.get(str(item.get("specialist", "")).strip()) if isinstance(item, dict) else None
            text = str(item.get("task", "")).strip() if isinstance(item, dict) else ""
            if spec is None:
                raise ToolError(P.DELEGATE_SPECIALIST.format(n=n, names=", ".join(by_name)))
        if not text:
            raise ToolError(P.DELEGATE_EMPTY_TASK.format(n=n))
        out.append((spec, text))
    return out


def delegate_tool(team: Team, orch: AgentRecord, cfg: TeamConfig, rounds: list[dict]) -> ToolDef:
    async def delegate(subtasks: list) -> str:
        items = _subtasks(subtasks, cfg)
        r = len(rounds) + 1
        workers = [
            (team.agent(f"w{r}.{i}", spec.name if spec else "worker", parent=orch.id, round=r, slot=i, subtask=clip(text, 500)), text, spec)
            for i, (spec, text) in enumerate(items, 1)
        ]
        record = {"round": r, "turn": _turn(orch), "subtasks": [{"worker": a.id, "role": a.role, "task": clip(t, 500)} for a, t, _ in workers]}
        rounds.append(record)
        base = env_data(team.state.store)
        stores = [Store(base) for _ in workers]

        def unit(agent: AgentRecord, text: str, spec: Specialist | None):
            return lambda: run_worker(team, agent, text, cfg, spec)

        try:
            results = await run_isolated([unit(*w) for w in workers], stores, cfg.concurrent)
        except Exception as e:  # held: execute_tools would turn a limit (or a TimeoutError, ...) into a tool error
            team.note_error(e)
            raise
        finally:
            for s in stores:
                merge_env(team.state.store, base, s)
        for sub, res in zip(record["subtasks"], results, strict=True):
            sub.update(status=res.status, result=clip(res.text, 500))
        return "\n\n".join(res.line(n) for n, res in enumerate(results, 1))

    if cfg.specialists is None:
        item = ToolParam(type="string")
        description = P.DELEGATE_DESCRIPTION
    else:
        item = ToolParam(
            type="object",
            properties={
                "specialist": ToolParam(type="string", description="The specialist who carries out the subtask.", enum=[s.name for s in cfg.specialists]),
                "task": ToolParam(type="string", description="The subtask, self-contained."),
            },
            required=["specialist", "task"],
            additionalProperties=False,
        )
        description = P.DELEGATE_SPECIALIST_DESCRIPTION
    # Never Inspect's 16 KiB default (a silent cut): each result is clipped to the arm's clip already, with a marker.
    max_output = cfg.n_workers * (cfg.clip_tokens * 8 + 512)
    return ToolDef(delegate, name="delegate", description=description.format(n_workers=cfg.n_workers), parallel=False, max_output=max_output,
                   parameters=_params(subtasks=ToolParam(type="array", description=f"1 to {cfg.n_workers} self-contained subtasks.", items=item)))


def team_text(team: Team, cfg: TeamConfig) -> str:
    if cfg.specialists is None:
        tools, _ = cfg.worker_tools(team)
        return P.TEAM_IDENTICAL.format(tools=", ".join(tools))
    roster = "\n".join(P.ROSTER_LINE.format(name=s.name, covers=s.coverage, tools=", ".join(cfg.worker_tools(team, s)[0])) for s in cfg.specialists)
    return P.TEAM_SPECIALISTS.format(roster=roster)


async def run_orchestrator(team: Team, cfg: TeamConfig, *, prompt: str | None = None, answer: ToolDef | None = None,
                           done: Callable[[], bool] | None = None) -> None:
    """The orchestrator's loop over the sample's messages; returns when it has answered (`done`), stopped, or hit its
    cap. `prompt`, `answer` and `done` default to the task's prompt, its answer tool and "an answer is recorded"."""
    task = team.task
    prompt = prompt if prompt is not None else task.prompt
    answer = answer if answer is not None else build_tools(team.world, task)[task.answer_tool]
    orch = team.agent("orchestrator", "orchestrator")
    team.top = orch
    plans, rounds = team.records.setdefault("mas_plan", []), team.records.setdefault("mas_rounds", [])
    tools = {"plan": plan_tool(orch, plans), "delegate": delegate_tool(team, orch, cfg, rounds), answer.name: answer}
    note = cfg.notes.orchestrator.format(n_workers=cfg.n_workers, answer_tool=answer.name, team=team_text(team, cfg))
    async with team.running(orch):
        ctx = await cfg.delivery.compile(orch, prompt, 0)
        team.state.messages = [cfg.delivery.system(ctx), ChatMessageUser(content=f"{prompt}\n\n{note}")]
        await react_loop(team, orch, team.state.messages, delivery=cfg.delivery, ctx=ctx, query=prompt, tools=tools, always=list(tools),
                         exposure=cfg.exposure, max_turns=cfg.max_turns, done=done or team.answered, nudge=P.NUDGE_TASK.format(answer_tool=answer.name),
                         first_tool="plan")


# --- council --------------------------------------------------------------------------------------------------------


def proposal_tool(team: Team, answer: ToolDef, sink: list[dict], clip_tokens: int = TEXT_MAX_TOKENS) -> ToolDef:
    """The task's answer tool as a council member sees it: same name and parameters plus a rationale; it records a
    proposal (parsed exactly as the real tool parses an answer, in a scratch store; the rationale clipped to
    `clip_tokens`) and never answers the task. The first proposal of a round wins and every later one is told so: the
    slot is taken before the parse awaits, and the tool is not parallel-safe, so two proposals in one turn run in the
    model's order (BUILD_REVIEW A-8: both used to be told "recorded" while only the first counted)."""
    taken: list[bool] = []

    async def propose(**kwargs: Any) -> str:
        if sink or taken:
            return P.PROPOSAL_AGAIN
        taken.append(True)
        try:
            rationale = str(kwargs.pop("rationale", "") or "")
            scratch = Store()
            await run_isolated([lambda: answer.tool(**kwargs)], [scratch], concurrent=False)
        except BaseException:
            taken.clear()  # the call failed before a proposal was recorded: the slot is free again
            raise
        sink.append({"answer": scratch.get(ANSWER), "rationale": clip(rationale, clip_tokens)})
        return "Proposal recorded."

    params = ToolParams(properties={**answer.parameters.properties, "rationale": ToolParam(type="string", description=P.RATIONALE_PARAM)},
                        required=[*answer.parameters.required, "rationale"])
    return ToolDef(propose, name=answer.name, description=answer.description + P.PROPOSE_SUFFIX, parameters=params, parallel=False)


def _proposals(latest: list[dict | None], members: list[int]) -> str:
    return "\n\n".join(
        P.PROPOSAL_LINE.format(member=m + 1, answer=as_json(latest[m]["answer"]), rationale=latest[m]["rationale"] or "(none)")
        if latest[m] is not None else P.NO_PROPOSAL.format(member=m + 1)
        for m in members
    )


async def run_council(team: Team, cfg: TeamConfig, k: int = COUNCIL_K, rounds: int = CRITIQUE_ROUNDS) -> None:
    task = team.task
    answer = build_tools(team.world, task)[task.answer_tool]
    members = [team.agent(f"member_{i}", "member") for i in range(1, k + 1)]
    histories: list[list[ChatMessage]] = [[] for _ in members]
    first_ctx: list[Any] = [None] * k
    base = env_data(team.state.store)
    stores = [Store(base) for _ in members]
    latest: list[dict | None] = [None] * k
    log = team.records.setdefault("mas_council", {"k": k, "rounds": rounds, "phases": [], "chair": None})

    def unit(i: int, phase: int):
        async def run() -> dict | None:
            agent, sink = members[i], []
            tools, always = cfg.worker_tools(team)
            tools[answer.name] = proposal_tool(team, answer, sink, cfg.clip_tokens)
            always = [*always, answer.name]
            try:
                async with team.running(agent):
                    if phase == 0:
                        ctx = first_ctx[i] = await cfg.delivery.compile(agent, task.prompt, 0)
                        note = cfg.notes.member.format(k=k, answer_tool=answer.name)
                        histories[i][:] = [cfg.delivery.system(ctx), ChatMessageUser(content=f"{task.prompt}\n\n{note}")]
                    else:
                        others = _proposals(latest, [m for m in range(k) if m != i])
                        histories[i].append(ChatMessageUser(content=cfg.notes.critique.format(round=phase, rounds=rounds, proposals=others, answer_tool=answer.name)))
                        ctx = None if cfg.delivery.per_step else first_ctx[i]
                    await react_loop(team, agent, histories[i], delivery=cfg.delivery, ctx=ctx, query=task.prompt, tools=tools, always=always,
                                     exposure=cfg.exposure, max_turns=cfg.max_turns, done=lambda: bool(sink),
                                     nudge=P.NUDGE_PROPOSE.format(answer_tool=answer.name))
            except Exception as e:  # a limit or a provider error ends the sample, as in S1 (BUILD_REVIEW A-2)
                if not isinstance(e, LimitExceededError):
                    agent.error, agent.stop = f"{type(e).__name__}: {e}"[:500], "error"
                raise
            return sink[0] if sink else None

        return run

    try:
        for phase in range(rounds + 1):
            got = await run_isolated([unit(i, phase) for i in range(k)], stores, concurrent=True)
            latest = [g if g is not None else prev for g, prev in zip(got, latest, strict=True)]
            log["phases"].append([{"member": i + 1, **(g or {"answer": None, "rationale": None})} for i, g in enumerate(got)])
        chair = team.agent("chair", "chair")
        team.top = chair
        async with team.running(chair):
            ctx = await cfg.delivery.compile(chair, task.prompt, 0)
            note = cfg.notes.chair.format(k=k, rounds=rounds, proposals=_proposals(latest, list(range(k))), answer_tool=answer.name)
            team.state.messages = [cfg.delivery.system(ctx), ChatMessageUser(content=f"{task.prompt}\n\n{note}")]
            await react_loop(team, chair, team.state.messages, delivery=cfg.delivery, ctx=ctx, query=task.prompt, tools={answer.name: answer},
                             always=[answer.name], exposure=cfg.exposure, max_turns=cfg.max_turns, done=team.answered,
                             nudge=P.NUDGE_SUBMIT.format(tool=answer.name))
        log["chair"] = team.state.store.get(ANSWER)
    finally:
        for s in stores:  # lookups and faults fired, for the fault analyses; members never answer the task
            merge_env(team.state.store, base, s)


# --- ensemble -------------------------------------------------------------------------------------------------------


def answer_key(task: TaskItem, answer: dict | None, calls: list[dict]) -> str | None:
    """An attempt's vote: its answer as the scorer compares it (`scorers.success.is_success`), or None (no vote). F3's
    answer is its end state (the mutating calls, in any order); an F3 attempt votes once it made a call or finished."""
    try:
        if task.family == "F3":
            if answer is None and not calls:
                return None
            return as_json(sorted(as_json(c) for c in calls))
        if answer is None:
            return None
        if task.family == "F1":
            return as_json(norm_ratings(answer.get("ratings")))
        if task.family == "F2":
            return norm_id(answer.get("final", ""))
        if task.family == "F7":
            return as_json({**answer, "deadline_days": int(answer["deadline_days"])})
        return norm_f5(answer.get("answer", ""))
    except (KeyError, TypeError, ValueError, AttributeError):
        return as_json(answer)


def plurality(keys: list[str | None]) -> tuple[list[int], bool]:
    """Indices of the attempts holding the most common answer, each answer represented by its earliest attempt, and
    whether that answer is unique (a majority/plurality winner) or tied."""
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return [], False
    top = max(counts.values())
    tied = [k for k in dict.fromkeys(k for k in keys if k is not None) if counts[k] == top]
    return [keys.index(k) for k in tied], len(tied) == 1


async def aggregate(team: Team, cfg: TeamConfig, candidates: list[str]) -> int | None:
    """The LLM aggregator: picks one of the tied candidates (0-based), or None if it does not choose."""
    task = team.task
    agent = team.agent("aggregator", "aggregator")
    chosen: list[int] = []

    async def select_answer(candidate: int) -> str:
        if chosen:
            return P.SELECT_AGAIN
        if not isinstance(candidate, int) or not 1 <= candidate <= len(candidates):
            raise ToolError(P.SELECT_RANGE.format(n=len(candidates)))
        chosen.append(candidate - 1)
        return "Choice recorded."

    tool = ToolDef(select_answer, name="select_answer", description=P.SELECT_DESCRIPTION,
                   parameters=_params(candidate=ToolParam(type="integer", description="The candidate's number.")))
    lines = "\n".join(P.CANDIDATE_LINE.format(n=n, answer=c) for n, c in enumerate(candidates, 1))
    try:
        async with team.running(agent):
            ctx = await cfg.delivery.compile(agent, task.prompt, 0)
            messages: list[ChatMessage] = [cfg.delivery.system(ctx), ChatMessageUser(content=f"{task.prompt}\n\n{P.AGGREGATOR_NOTE.format(k=ENSEMBLE_K, candidates=lines)}")]
            await react_loop(team, agent, messages, delivery=cfg.delivery, ctx=ctx, query=task.prompt, tools={"select_answer": tool}, always=["select_answer"],
                             exposure=cfg.exposure, max_turns=cfg.max_turns, done=lambda: bool(chosen), nudge=P.NUDGE_SUBMIT.format(tool="select_answer"))
    except Exception as e:  # a limit or a provider error ends the sample, as in S1 (BUILD_REVIEW A-2)
        if not isinstance(e, LimitExceededError):
            agent.error, agent.stop = f"{type(e).__name__}: {e}"[:500], "error"
        raise
    return chosen[0] if chosen else None


def _adopt_attempt(agent: AgentRecord, st: Store, sub: TaskState, max_turns: int) -> None:
    """An attempt's own logs (written by kb_agent into its store and TaskState) onto its agent record."""
    agent.compile_log = [{**r, "agent": agent.id, "role": agent.role} for r in st.get(COMPILE_LOG, [])]
    agent.steps = list(st.get(STEP_LOG, []))
    agent.turns, agent.nudges = st.get("turns_used", 0) or 0, st.get("nudges", 0) or 0
    agent.tool_calls = [c.function for m in sub.messages if isinstance(m, ChatMessageAssistant) for c in (m.tool_calls or [])]
    agent.last_text = last_assistant_text(sub.messages)
    if agent.stop is None:
        answered = st.get(ANSWER) is not None
        agent.stop = "done" if answered else ("turn_cap" if agent.turns >= max_turns else "text")


async def run_ensemble(team: Team, cfg: TeamConfig, attempt: Solver, generate: Generate, k: int = ENSEMBLE_K) -> None:
    state, task = team.state, team.task
    agents = [team.agent(f"attempt_{i}", "attempt") for i in range(1, k + 1)]
    base = env_data(state.store)
    stores = [Store(base) for _ in agents]
    subs = [TaskState(model=state.model, sample_id=state.sample_id, epoch=state.epoch, input=state.input, messages=[], target=state.target, metadata=state.metadata)
            for _ in agents]

    def unit(i: int):
        async def run() -> None:
            try:
                async with team.running(agents[i]):
                    await attempt(subs[i], generate)
            except LimitExceededError:
                agents[i].stop = "limit"
                raise
            except Exception as e:  # a provider error ends the sample, as in S1, never a lost vote (BUILD_REVIEW A-2)
                agents[i].error, agents[i].stop = f"{type(e).__name__}: {e}"[:500], "error"
                raise
            except BaseException:  # cancelled (a sibling hit a limit, or the time limit)
                agents[i].stop = "interrupted"
                raise

        return run

    record: dict = {"k": k}
    team.records["mas_ensemble"] = record
    try:
        await run_isolated([unit(i) for i in range(k)], stores, concurrent=True)
    finally:
        for a, st, sub in zip(agents, stores, subs, strict=True):
            _adopt_attempt(a, st, sub, cfg.max_turns)
        for st in stores:  # faults fired and lookups over every attempt (read-only diagnostics)
            merge_env(state.store, base, Store({key: v for key, v in st.items() if key in (FAULTS_FIRED, LOOKUPS)}))
    answers = [st.get(ANSWER) for st in stores]
    keys = [answer_key(task, st.get(ANSWER), st.get(CALLS, [])) for st in stores]
    reps, unique = plurality(keys)
    record.update(answers=answers, votes=keys, counts=dict(Counter(k_ for k_ in keys if k_ is not None)))
    if not reps:
        winner, how = None, "no_votes"
    elif unique:
        winner, how = reps[0], "majority"
    else:
        views = [as_json({"calls": stores[i].get(CALLS, [])} if task.family == "F3" else answers[i]) for i in reps]
        pick = await aggregate(team, cfg, views)
        winner, how = (reps[pick], "aggregator") if pick is not None else (reps[0], "aggregator_fallback")
        record["tied"] = [i + 1 for i in reps]
    record.update(winner=winner + 1 if winner is not None else None, aggregation=how)
    if winner is not None:
        for key in (CALLS, ANSWER):
            if key in stores[winner]:
                state.store.set(key, deepcopy(stores[winner].get(key)))
    shown = winner if winner is not None else 0
    state.messages = subs[shown].messages
    state.output = subs[shown].output
    team.top = agents[shown]
