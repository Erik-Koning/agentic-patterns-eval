"""Multi-agent and ensemble arms for the main study (BUILD_PLAN B2): S9, M1, M1s, M1k, M2, M7 and S8k3.

- `prompts.py`: every prompt these arms add, in one place.
- `core.py`: the team (agent registry, spans, per-agent accounting from the transcript), per-agent knowledge delivery,
  the shared turn loop (kb_agent's per-turn rules over any history), and environment isolation.
- `primitives.py`: a worker run, the orchestrator loop, a council, an ensemble and its vote; reusable outside the
  main study's solvers (Study G's B9).
- `specialists.py`: M2's specialization, from the world's domains.
- `solvers.py`: one Inspect solver per arm, with the switch vector each logs.

**Inspect 0.3.273 defaults that change information flow, and how the arms hold them constant** (brief §9):
- `react()`, `as_tool()`, `handoff()` and `run()` are not used for these loops. `react()` adds its own assistant
  prompt, a `submit()` tool and an "on continue" message after every text-only turn, where the single-agent arms have
  `kb_react`'s prompts, answer tools and one nudge per text-only streak; `as_tool()` hands the agent a single string
  and returns only its last message; `handoff()` (for M3, Tier B) filters the history it passes back
  (`output_filter=content_only` by default: system messages and reasoning removed, tool calls turned into text). The
  arms instead run `core.react_loop`, which applies `kb_agent`'s rules to every agent, so the M arms differ from the
  single-agent arms only in their switches.
- A forced tool choice (`ToolFunction`) makes Inspect send the provider that tool alone. The planning step (S9 and the
  orchestrators' first turn, DEC=1) forces `plan`, so that turn offers `plan` only; the S9 and orchestrator notes
  name the tools that follow (S9's own, the workers'), the same way in both, so S9 and M1s plan from equal inputs.
- `execute_tools` runs a turn's parallel-safe tool calls concurrently (`ToolDef.parallel` defaults to True): the same
  in every arm and every agent. `delegate` is not parallel-safe, so two `delegate` calls in one turn are two rounds
  in the model's order.
- `execute_tools` turns a `LimitExceededError` raised inside a tool into a tool error and continues. Workers run inside
  `delegate`, so a sample limit tripped by a worker is held on the team and re-raised after the tool call (`core`).
- The sample's `token_limit` (and cost, message, time and working limits) are ContextVar trees that child tasks
  inherit: every agent's calls, workers and the `kg` classify calls included, count against the one sample limit.
  The token check runs after a call's usage is recorded, so the tripping call is complete and logged.
- `Model.generate` checks the sample's `message_limit` against each call's input length, so that limit applies to every
  agent's own conversation; `TaskState.messages` (the top agent's history) is checked on append as well.
- A sample `turn_limit` would count every generation of every agent (and of the `kg` role); the arms set none. Their
  turn caps are the loops' own counters: each agent gets the task's `max_turns` per loop.
- `max_tool_output` (16 KiB) truncates tool results: `delegate` sets its own limit, and each worker result is clipped
  to `core.TEXT_MAX_TOKENS` with a visible marker.
- The model's connection limit (`max_connections`) is shared by every sample and agent, so concurrent workers can
  queue: M1's wall-clock gain over M1s depends on it (record it with the run).
"""
