# Context-length sweep (exploratory)

**Question.** How much context can a model receive and still answer one question correctly?

Study G asks a different question: what context a model should be *given*. It compares context-management policies under a 32K window that the harness enforces, so it never measures how much a model can actually take. This sweep measures that on the same F8 material, from 8K to 960K tokens.

**Status.** Built and tested offline (D-054). It is off by default and nothing has run live. It is exploratory and descriptive: there are no hypotheses and no freeze.

## Design

| | |
|---|---|
| Models | GPT-6 Luna, Sol and Astra, each at high effort (profiles `study_g_luna`, `study_g_sol` and `study_g_astra`). gpt-4o-mini, the anchor's model, is not swept. |
| Sizes | 8K, 64K, 128K, 256K, 512K and 960K tokens. The GPT-6 window of 1,050,000 tokens is shared by input and output, so each generation's output is capped at 64,000 tokens, lowered to about 43,000 at 960K so input and output fit together. |
| Tasks | 3 per size and model, one of each kind below. The same 3 tasks run at every size and on every model (paired). |
| Samples | 3 kinds × 6 sizes × 3 models = 54 per replicate. |

**Each sample** is one decision at the end of a replayed F8 shift (`ape.worlds.gen_sweep`). The replay is the reference trajectory, as a flawless agent would have run it:
- the system prompt (handbook and shift rules);
- a start message;
- each case's message;
- one assistant tool call per generation: a lookup, the ticket's procedure calls, `finish` or `submit_decision` with the gold decision;
- each tool's result, including the bulky customer and order files of about 1,000 tokens each, F8's default.

The shift has three parts:

- **Case 1.** This is where the probe's dependency sits, at the very start of the history.
- **The middle.** These are neutral cases: policy cases and tickets with fresh customers and orders, no memos, no follow-ups and no repeats. They are drawn from one seeded sequence, so the 8K middle is a prefix of the 960K middle. Cases are added until the next one would pass the target. No middle case falls under a memo, so none of them demonstrates it.
- **The probe (last case).** Its gold decision is identical at every size; the build checks this.

| Kind | The probe | What it needs from the start of the context |
|---|---|---|
| `fresh` | A new policy case. Its customer lookup is replayed just before the decision. | Only the handbook in the system prompt. Measures the cost of length itself. |
| `memo` | A new policy case whose decision a memo changes. | The memo, announced in case 1's message, and the handbook. |
| `followup` | A follow-up to case 1, which was an awaiting case. A memo announced in the follow-up's own message changes the decision. | Case 1's original request and its customer's tier. Copying case 1's earlier decision fails. |

**The model's turn.** The model acts on the probe with F8's session tools for at most 2 generations: one optional tool call, then the decision. Every generation re-sends the whole context. Nothing nudges the model.

**Scoring.** The probe is scored with F8's item rule: the first decision submitted must equal the gold in all four fields. A wrong answer gets a label:

| Label | Meaning |
|---|---|
| `ignored_memo` | (memo) The decision the probe would have without the memo. |
| `copied_original` | (followup) Case 1's earlier decision repeated. This is also the decision without the probe's memo. |
| `unanswered` | No decision within the 2 generations. |
| `output_cap` | No decision: the last generation hit its output cap. |
| `wrong` | Any other decision. |

**Not measured.** Some samples never reach the model, so they are kept out of every rate and out of the held size, and the report lists them on their own:

| Label | Meaning |
|---|---|
| `over_limit` | The provider refused the request's size: OpenAI's `context_length_exceeded`, or any 400 that names a size or limit, such as too many input items. |
| `rejected` | Any other 400: a request problem to look at. |
| `error` | A harness error. |

**Request sizing.** Each generation's output cap is lowered so that the input, as the provider may count it (the meter × 1.03 plus 4 tokens per message), plus the output fits the 1.05M window: about 43K at 960K. A refusal ends the sample without an Inspect error, so it is not retried and does not fail the model's eval set.

**Recovery.** After each model, the runner names the refused sizes and the fix: a new run at every size, with `--output-tokens 2000` (about half the messages, the most that still fits 8K) or without the size. It never re-renders one size on its own, because every size must share one rendering for the comparison to hold.

**Context size** is the decision call's input as the session meters it: o200k tokens of every message's text and tool calls, plus the session tools' schemas. Each world sits at most one case (about 1.6K tokens) under its target and records the exact figure. The provider's own input-token count is logged beside it.

**Worlds.** There is one world per kind and size, under `worlds/sweep/`. Seed 40000 is used for `fresh`, 40001 for `memo` and 40002 for `followup`. A new `--seed-base` gives new tasks. Building all 18 worlds takes seconds; the 960K world holds about 770 replayed cases.

## Outputs

`runs/context_sweep/<run id>/report/` holds:
- `report.md`: per model, success by size and kind (one mark per replicate), the largest size at which every task was right, failure labels, metered against provider tokens, and spend;
- `results.csv` and `results.json`: one row per sample.

## Running

The step-by-step runbook, metric definitions and guidance on reading results are in [`CONTEXT_SWEEP_README.md`](CONTEXT_SWEEP_README.md). It needs no other study's run. `--tasks-per-kind N` gives each kind N tasks (more tasks per size, for firmer conclusions; cost scales with N).

```bash
uv run python -m ape.run_sweep all --run-id rehearsal --offline            # mock agent, zero spend
# live: first set `enabled: true` on studies.context_sweep's cells in config/run_plan.yaml
uv run python -m ape.run_sweep all --run-id sweep-1                       # build, run every tier, analyze
uv run python -m ape.run_sweep all --run-id sweep-1                       # resumes: completed samples are kept
uv run python -m ape.run_sweep all --run-id sweep-1 --rerun               # replicate 2: the same tasks again
uv run python -m ape.run_sweep run --run-id sweep-1 --models astra --replicate 2   # one tier of one replicate
uv run python -m ape.run_sweep all --run-id sweep-2 --seed-base 40100     # new tasks: another run id
```

**Run design.** A run fixes its design at its first step: kinds, sizes, tiers, seeds and tool-file size. Replicates therefore stay comparable, and a later flag that changes the design is refused. `--models` picks a subset of the tiers for one invocation.

**Mechanics.** Each tier of each replicate is one Inspect eval set under `rep-<n>/<tier>/`, with the runner's retries, usage ledger, spend registry, per-sample cost guard and a 30-minute wall-clock guard. It runs under its own cache nonce, at 6 concurrent samples by default (`--max-samples`).

## Cost

Per replicate, with the cells enabled (`uv run python -m ape.budget`):

| Tier | Conservative | Expected |
|---|---|---|
| Luna | $1 | $1 |
| Sol | $25 | $18 |
| Astra | $212 | $157 |
| **Total** | **$238** | **$177** |

- **Why Astra dominates.** 89% of the cost is Astra at 512K and 960K. A request with more than 272K input tokens pays Astra's long-context rate on every token: $20 input and $75 output per 1M, as listed on 2026-10-05 (`budget_assumptions.yaml` `long_context_prices`). Confirm the rate on the day of the run.
- **How the projection is built.** It assumes 2 generations per sample (the cap) and 4,000 output tokens per generation. Most samples need 1 generation, so the realistic cost is about half.
- **Allocation.** $500 (`budget.allocations.context_sweep`), which leaves room for one `--rerun`.
- **Guard.** Before each tier, the runner checks that the conservative projection of the tier's remaining samples fits both limits:
  - what is left of the program budget, less the sweep's long-context surcharge, which Inspect's flat prices miss;
  - what is left of the allocation, with the sweep's own spend priced from its usage ledgers at the long-context rates.
- **Cells are off by default.** As with the M5 add-on, the program total ($2,996) excludes the sweep until its cells are enabled, and a live run refuses a tier whose cell is off.

## Caveats

- **Three tasks per size and model.** Read the results as patterns, not rates. A replicate adds sampling noise at the same tasks; only new seeds add task variety.
- **Every kind needs the handbook**, which sits in the system prompt at the very start of the context. `fresh` is free of session state, not of long-range reading.
- **The replayed history is a flawless agent's.** A real session also holds the agent's own text, reasoning and mistakes, so this measures reading a clean transcript.
- **Message counts are high.** At 960K the history has about 4,000 messages. If a provider refuses a request that size or that many items, start a new run with `--output-tokens 2000` (fewer, bulkier cases: about 2,200 messages at 960K). Larger files leave no room at 8K: with 3,000-token files, the fixed parts alone exceed 8K.
- **The token meter differs from the provider's count.** The provider count is recorded per sample; the offline mock's count means nothing.
