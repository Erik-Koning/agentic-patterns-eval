"""A gold-knowing stand-in for every role of the multi-agent and ensemble arms (`mockllm` `custom_outputs`).

`GoldMulti(worlds_dir)` knows every task's gold and plays whichever role the call is for, recognised by the tools it
is offered (never by prompt prose, which is meant to be rewritten):
- `delegate` -> orchestrator: plans, then delegates. F1: one subtask per supplier, three per round (from the task's
  suppliers, as `tags["subtasks"]` lists them); F2: one hop per round, each waiting for the previous hop's result;
  F3 and F7: one subtask. It answers from the workers' results, so a result lost or garbled on the way fails the task.
  Under M2 it routes each subtask to the specialist that covers it (`agent.multi.specialists`).
- `report` -> worker: makes the real tool calls its subtask needs and reports what it read (an F1 rating computed
  from the record it looked up, an F2 next hop from the record's escalation code, F3's procedure calls, an F7 decision).
- `plan` without `delegate` -> S9: plans, then works as a single agent.
- an answer tool with a `rationale` parameter -> council member: works as a single agent, proposes the gold.
- the answer tool alone -> chair: submits the majority of the proposals it is shown.
- `select_answer` -> S8k3 aggregator: picks the candidate equal to the gold.
- anything else -> a single agent (an S8k3 attempt): lookups, then the gold answer.

Every output is a deterministic function of the conversation it is given (tool-call IDs included), so two runs that
give the model the same inputs get the same outputs: the M1-vs-M1s identity test relies on it.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ChatMessageTool, ModelOutput
from inspect_ai.tool import ToolCall

from ..agent.multi.specialists import specialists
from ..worlds.gen_registry import rate
from ..worlds.spec import TaskItem, World

MODEL = "mockllm/model"
ANSWER_TOOLS = {"submit_ratings", "submit_chain", "finish", "submit_decision", "submit_answer"}
F1_SUBTASK = "Rate supplier {sid} for the {q} review under the supplier scorecard rules: look up its record and report '{sid}: <rating>'."
F2_SUBTASK = "Supplier {sid} holds a dispute. Look up {sid}'s record for its escalation code, find the supplier that code routes to in the Escalation Directory, and report 'next supplier: <ID>'."
F3_SUBTASK = "Ticket for order {oid}: {situation}. Look up the order's region and make the matching standard operating procedure's tool calls for order {oid}; report what you did."
F7_SUBTASK = "Customer {cid} submitted a request ({prompt}). Look up the customer, determine how policy requires it to be handled, and report the decision as JSON with action, approver, deadline_days and document."


def out(messages, *calls: tuple[str, dict], text: str = "") -> ModelOutput:
    """An assistant turn with `calls`; IDs derived from the conversation, so equal inputs give equal outputs."""
    seed = f"{len(messages)}|{messages[-1].text if messages else ''}"
    tcs = [ToolCall(id="call_" + hashlib.sha1(f"{seed}|{i}|{f}|{json.dumps(a, sort_keys=True)}".encode()).hexdigest()[:12], function=f, arguments=a)
           for i, (f, a) in enumerate(calls)]
    msg = ChatMessageAssistant(content=text, tool_calls=tcs or None, source="generate", model=MODEL)
    return ModelOutput(model=MODEL, choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls" if tcs else "stop")])


def tool_texts(messages, function: str) -> list[str]:
    return [m.text for m in messages if isinstance(m, ChatMessageTool) and m.function == function and not m.error]


def record(text: str) -> dict:
    """The JSON record at the start of a tool result (a per-step arm's view appends its knowledge after it)."""
    return json.JSONDecoder().raw_decode(text.strip())[0]


def calls_made(messages) -> list[tuple[str, dict]]:
    return [(c.function, c.arguments) for m in messages if isinstance(m, ChatMessageAssistant) for c in (m.tool_calls or [])]


def answer_args(task: TaskItem, value: dict) -> dict:
    """A recorded answer value (env_tools' ANSWER) as the answer tool's arguments."""
    if task.family == "F2":
        return {"final_supplier": value.get("final"), "chain": value.get("chain")}
    return dict(value)


def gold_args(task: TaskItem) -> dict:
    return answer_args(task, task.gold) if task.family != "F3" else {}


class GoldMulti:
    def __init__(self, worlds_dir):
        self.worlds = [World.load(p) for p in sorted(Path(worlds_dir).rglob("*.json"))]
        self.tasks = [(w, t) for w in self.worlds for t in w.tasks]
        self.suppliers = {sid: w for w in self.worlds if w.family in ("F1", "F2") for sid in w.entities["suppliers"]}
        self.orders = {t.tags["order_id"]: (w, t) for w, t in self.tasks if t.family == "F3"}
        self.customers = {t.tags["customer_id"]: (w, t) for w, t in self.tasks if t.family == "F7"}

    def __call__(self, messages, tools, tool_choice, config) -> ModelOutput:
        names = {t.name for t in tools}
        info = {t.name: t for t in tools}
        if "report" in names:
            return self.worker(messages, names)
        world, task = self.task_of(messages)
        if "select_answer" in names:
            return self.aggregator(messages, task)
        if "delegate" in names:
            return self.orchestrator(messages, world, task, info["delegate"])
        if "plan" in names and not any(f == "plan" for f, _ in calls_made(messages)):
            return out(messages, ("plan", {"subtasks": self.plan(world, task)}))
        answer = info.get(task.answer_tool)
        if answer is not None and "rationale" in answer.parameters.properties:
            step = self.single(messages, world, task, names)
            if step[0] == task.answer_tool:
                step = (step[0], {**step[1], "rationale": "Checked against the record and the rules."})
            return out(messages, step)
        if names == {task.answer_tool}:
            return self.chair(messages, task)
        return out(messages, self.single(messages, world, task, names))

    # -- finding the task ---------------------------------------------------------------------------------------------

    def task_of(self, messages) -> tuple[World, TaskItem]:
        first = next(m.text for m in messages if m.role == "user")
        for w, t in self.tasks:
            if t.prompt in first:
                return w, t
        raise KeyError(f"no task prompt in {first[:200]!r}")

    def plan(self, world: World, task: TaskItem) -> list[str]:
        if task.family == "F1":
            return [s["question"] for s in task.tags["subtasks"]]
        if task.family == "F2":
            return [f"Find escalation hop {j}" for j in range(1, task.tags["k"] + 1)]
        return ["Look up the case's record", "Work out what policy or procedure requires", "Act on it and answer"]

    # -- a single agent (S1 attempt, S9 after its plan, a council member) ---------------------------------------------

    def single(self, messages, world: World, task: TaskItem, names: set[str]) -> tuple[str, dict]:
        made = calls_made(messages)
        if task.family == "F1":
            done = {a.get("supplier_id") for f, a in made if f == "lookup_supplier"}
            todo = [s for s in task.tags["suppliers"] if s not in done]
            if todo:
                return ("lookup_supplier", {"supplier_id": todo[0]})
        elif task.family == "F2":
            chain = [world.entities["routes"][record(t)["escalation_code"]] for t in tool_texts(messages, "lookup_supplier")]
            if len(chain) < task.tags["k"]:
                return ("lookup_supplier", {"supplier_id": chain[-1] if chain else task.tags["start"]})
            return (task.answer_tool, {"final_supplier": chain[-1], "chain": chain})
        elif task.family == "F3":
            if not tool_texts(messages, "order_lookup"):
                return ("order_lookup", {"order_id": task.tags["order_id"]})
            pending = [c for c in task.gold["calls"] if (c["tool"], c["args"]) not in made]
            if pending:
                return (pending[0]["tool"], pending[0]["args"])
        elif task.family == "F7" and not tool_texts(messages, "lookup_customer"):
            return ("lookup_customer", {"customer_id": task.tags["customer_id"]})
        return (task.answer_tool, gold_args(task))

    # -- orchestrator ---------------------------------------------------------------------------------------------------

    def orchestrator(self, messages, world: World, task: TaskItem, delegate) -> ModelOutput:
        if not any(f == "plan" for f, _ in calls_made(messages)):
            return out(messages, ("plan", {"subtasks": self.plan(world, task)}))
        results = "\n".join(tool_texts(messages, "delegate"))
        ok = {m.tool_call_id for m in messages if isinstance(m, ChatMessageTool) and m.function == "delegate" and not m.error}
        sent = sum(len(c.arguments["subtasks"]) for m in messages if isinstance(m, ChatMessageAssistant) for c in (m.tool_calls or []) if c.id in ok)
        specialized = delegate.parameters.properties["subtasks"].items.type == "object"
        if task.family == "F1":
            todo = task.tags["suppliers"][sent:]
            if todo:
                return out(messages, ("delegate", {"subtasks": [self.route(world, task, sid, F1_SUBTASK.format(sid=sid, q=task.tags["quarter"]), specialized) for sid in todo[:3]]}))
            ratings = dict(re.findall(r"(SUP-\d+): (preferred|approved|probation)", results))
            return out(messages, ("submit_ratings", {"ratings": ratings}))
        if task.family == "F2":
            chain = re.findall(r"next supplier: (SUP-\d+)", results)
            if len(chain) < task.tags["k"]:
                cur = chain[-1] if chain else task.tags["start"]
                return out(messages, ("delegate", {"subtasks": [self.route(world, task, cur, F2_SUBTASK.format(sid=cur), specialized)]}))
            return out(messages, ("submit_chain", {"final_supplier": chain[-1], "chain": chain}))
        if not sent:
            if task.family == "F3":
                text = F3_SUBTASK.format(oid=task.tags["order_id"], situation=task.prompt.split(":", 1)[1].split("(order")[0].strip())
            else:
                text = F7_SUBTASK.format(cid=task.tags["customer_id"], prompt=task.prompt)
            return out(messages, ("delegate", {"subtasks": [self.route(world, task, None, text, specialized)]}))
        if task.family == "F3":
            return out(messages, ("finish", {}))
        found = re.search(r"\{.*\}", results)
        if found is None:
            return out(messages, text="The worker did not report a decision.")
        return out(messages, ("submit_decision", json.loads(found.group(0))))

    def route(self, world: World, task: TaskItem, sid: str | None, text: str, specialized: bool):
        if not specialized:
            return text
        specs = specialists(world)
        if task.family == "F1":
            spec = next(s for s in specs if world.entities["suppliers"][sid]["segment"] in s.covers)
        elif task.family == "F3":
            spec = next(s for s in specs if {c["tool"] for c in task.gold["calls"]} <= set(s.tools or ()))
        else:
            spec = next(s for s in specs if task.tags["domain"] in s.covers)
        return {"specialist": spec.name, "task": text}

    # -- worker ---------------------------------------------------------------------------------------------------------

    def worker(self, messages, names: set[str]) -> ModelOutput:
        sub = next(m.text for m in messages if m.role == "user").split("Subtask:", 1)[-1]
        if m := re.search(r"Rate supplier (SUP-\d+) for the (Q\d) review", sub):
            sid, q = m.groups()
            got = tool_texts(messages, "lookup_supplier")
            if not got:
                return out(messages, ("lookup_supplier", {"supplier_id": sid}))
            rec, world = record(got[-1]), self.suppliers[sid]
            rule = next(r for r in world.entities["rules"] if r["segment"] == rec["segment"])
            return out(messages, ("report", {"result": f"{sid}: {rate(rule, rec['scorecard'][q])}"}))
        if m := re.search(r"Supplier (SUP-\d+) holds a dispute", sub):
            sid = m.group(1)
            got = tool_texts(messages, "lookup_supplier")
            if not got:
                return out(messages, ("lookup_supplier", {"supplier_id": sid}))
            nxt = self.suppliers[sid].entities["routes"][record(got[-1])["escalation_code"]]
            return out(messages, ("report", {"result": f"next supplier: {nxt}"}))
        if m := re.search(r"order (O-\d+)", sub):
            world, task = self.orders[m.group(1)]
            if not tool_texts(messages, "order_lookup"):
                return out(messages, ("order_lookup", {"order_id": m.group(1)}))
            made = calls_made(messages)
            pending = [c for c in task.gold["calls"] if (c["tool"], c["args"]) not in made]
            if pending and pending[0]["tool"] not in names:
                return out(messages, ("report", {"result": f"I do not have the tool {pending[0]['tool']}."}))
            if pending:
                return out(messages, (pending[0]["tool"], pending[0]["args"]))
            return out(messages, ("report", {"result": "Procedure calls made: " + ", ".join(c["tool"] for c in task.gold["calls"])}))
        if m := re.search(r"Customer (CU-\d+)", sub):
            world, task = self.customers[m.group(1)]
            if not tool_texts(messages, "lookup_customer"):
                return out(messages, ("lookup_customer", {"customer_id": m.group(1)}))
            return out(messages, ("report", {"result": json.dumps(task.gold)}))
        return out(messages, ("report", {"result": "I could not parse the subtask."}))

    # -- the kg role ----------------------------------------------------------------------------------------------------

    def kg(self, messages, tools, tool_choice, config) -> ModelOutput:
        """A gold `kg` classifier for APG (pass as the `kg` role's `custom_outputs`): routes a query that names an F3
        order or an F7 customer to the outline node declaring that task's procedure or policy, so the KG arm's tool
        allowlist and context are the right ones; anything else goes to `mock_kg` (first node; LightRAG keywords)."""
        from .mock_agent import mock_kg

        if config.response_schema is None or config.response_schema.name != "classify":
            return mock_kg(messages, tools, tool_choice, config)
        user = next((m.text for m in messages if m.role == "user"), "")
        query, _, outline = user.partition("Category outline:")
        target = None
        if (m := re.search(r"O-\d{6}", query)) and m.group(0) in self.orders:
            target = f"Procedure {self.orders[m.group(0)][1].tags['procedure']}."
        elif (m := re.search(r"CU-\d{5}", query)) and m.group(0) in self.customers:
            target = f"Policy {self.customers[m.group(0)][1].tags['policy']} "
        node = next((line.split(":", 1)[0].strip() for line in outline.splitlines() if target and target in line), None)
        if node is None:
            return mock_kg(messages, tools, tool_choice, config)
        return ModelOutput.from_content(MODEL, json.dumps({"matches": [{"nodeId": node, "confidence": 0.95, "reason": "gold"}]}))

    # -- chair and aggregator -----------------------------------------------------------------------------------------

    def chair(self, messages, task: TaskItem) -> ModelOutput:
        shown = next(m.text for m in messages if m.role == "user")
        proposals = [line for line in re.findall(r"proposed answer: (.*)$", shown, flags=re.M) if line.strip() != "null"]
        if not proposals:
            return out(messages, text="No proposals.")
        top = Counter(proposals).most_common(1)[0][0]
        return out(messages, (task.answer_tool, answer_args(task, json.loads(top))))

    def aggregator(self, messages, task: TaskItem) -> ModelOutput:
        shown = next(m.text for m in messages if m.role == "user")
        cands = re.findall(r"^Candidate (\d+): (.*)$", shown, flags=re.M)
        want = task.gold if task.family != "F3" else {"calls": task.gold["calls"]}
        norm = lambda v: json.dumps(v, sort_keys=True)  # noqa: E731
        if task.family == "F3":
            norm = lambda v: sorted(json.dumps(c, sort_keys=True) for c in v["calls"])  # noqa: E731
        pick = next((int(n) for n, c in cands if norm(json.loads(c)) == norm(want)), 1)
        return out(messages, ("select_answer", {"candidate": pick}))
