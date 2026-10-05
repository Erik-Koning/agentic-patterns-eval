# Context-length sweep: runbook

This experiment answers one question: **how much context can each GPT-6 model receive and still answer a question correctly?**

It needs no other study and has its own command. Each sample hands a model a replayed F8 support shift, 8K to 960K tokens long, and asks it to decide the last case. The same tasks run at every size and on every model, so the only thing that changes is how much history sits between the information the model needs and the question.

The design and its rationale are in [`CONTEXT_SWEEP.md`](CONTEXT_SWEEP.md). This file covers how to run the sweep, what it measures and how to read the results.

**What one run gives you**, per model (Luna, Sol, Astra):
- the largest context at which every task was answered correctly;
- the success rate at each size, with 95% intervals;
- success by kind of task, which separates *length itself* from *finding something far back*;
- why answers failed: a missed memo, a copied old decision, no answer, or something else;
- how much the model reasoned (output tokens) as context grows;
- what it cost, at real long-context prices.

---

## 1. Setup (once)

```bash
git checkout study-h-context-sweep        # or the branch this was merged into
uv sync --locked                          # Python 3.14
echo "OPENAI_API_KEY=<key>" > .env        # gitignored
uv run python readiness/probe_openai.py   # optional: confirms the GPT-6 models are served (snapshot check)
```

## 2. Rehearse offline (free, about a minute)

```bash
uv run python -m ape.run_sweep all --run-id rehearsal --offline
```

This builds all 18 worlds (3 kinds × 6 sizes) and runs every model with a mock that always answers correctly: 54 samples and about 23 MB of logs. It then writes `runs/context_sweep/rehearsal/report/report.md`, so the whole pipeline is exercised, 960K included, with zero spend. The report says "Offline dry run (mock agent)" at the top; its numbers mean nothing. Add `--mock naive` to see the failure labels appear, or `--mock refuse-largest` to see what happens when the provider refuses the largest size (section 9).

## 3. Switch it on

The sweep is off by default so the program budget doesn't count it. In `config/run_plan.yaml`, under `studies.context_sweep`, set `enabled: true` on the three cells you want to run:

```yaml
        - {id: sweep.luna, kind: sweep, enabled: true, ...}
        - {id: sweep.sol, kind: sweep, enabled: true, ...}
        - {id: sweep.astra, kind: sweep, enabled: true, ...}
```

`uv run python -m ape.budget` then shows the sweep in the program total. A live run refuses any model whose cell is still off.

## 4. Run

The recommended order is cheapest first, so problems surface for about $1:

```bash
uv run python -m ape.run_sweep all --run-id sweep-1 --models luna    # ~$1: proves 960K requests go through
uv run python -m ape.run_sweep all --run-id sweep-1 --models sol     # ~$18-25
uv run python -m ape.run_sweep all --run-id sweep-1 --models astra   # ~$157-212
# or everything at once:
uv run python -m ape.run_sweep all --run-id sweep-1
```

`all` runs three steps in order: `build` (the worlds), `run` (one Inspect eval set per model, saved under `runs/context_sweep/sweep-1/rep-1/<model>/`) and `analyze` (the report). Each step can also be run on its own: `build`, `run` or `analyze`.

| You want to | Command |
|---|---|
| Continue after a crash or Ctrl-C | the same command again: completed samples are kept |
| Repeat the same tasks (is a miss stable?) | `... all --run-id sweep-1 --rerun` → replicate 2, pooled in the report |
| Resume a specific replicate | `... run --run-id sweep-1 --replicate 2` |
| More tasks per size (firmer conclusions) | a new run: `... all --run-id sweep-big --tasks-per-kind 3` |
| Different tasks | a new run: `... all --run-id sweep-b --seed-base 40100` |
| Fewer sizes | a new run: `... all --run-id sweep-small --sizes 8000,64000,128000` |
| Rebuild only the report | `... analyze --run-id sweep-1` |

A run fixes its design (sizes, kinds, tasks, seeds) at its first step, so replicates stay comparable. Changing the design means a new run id. `--models` can always pick a subset of the run's models.

**Concurrency.** The default is 6 samples at once per model; `--max-samples 2` eases tokens-per-minute limits.

**Run time.** It hasn't been measured live. A 960K-token request at high effort can take minutes.

## 5. Outputs

Everything is under `runs/context_sweep/<run id>/`:

| File | What |
|---|---|
| `report/report.md` | The readable result: metrics, per-sample grid, failure labels, token check, spend |
| `report/results.csv` | One row per sample, for your own analysis (columns below) |
| `report/results.json` | The same rows, plus the run design and spend |
| `rep-<n>/<model>/*.eval` | Inspect logs, including full transcripts: `uv run inspect view --log-dir runs/context_sweep/<run id>/rep-1/luna` |
| `run.json` | The fixed design, the replicates and the world hashes |

`results.csv` has these columns:

| Column | Meaning |
|---|---|
| `replicate`, `tier`, `model`, `kind`, `seed` | Which sample (`seed` identifies the task) |
| `target_tokens`, `context_tokens` | Nominal size; the exact metered input of the decision call |
| `provider_input_tokens` | Input tokens as OpenAI billed the first generation |
| `success`, `answered` | 1 if the decision matched the gold; 1 if any decision was submitted |
| `measured` | 1 if the model actually got the request; 0 if the provider refused it or the harness errored (kept out of every rate) |
| `failure` | `ignored_memo`, `copied_original`, `unanswered`, `output_cap`, `wrong`; not measured: `over_limit`, `rejected`, `error`; empty when right |
| `generations`, `output_tokens`, `reasoning_tokens` | Effort spent on the decision |
| `cost_usd`, `cost_flat_usd` | Cost at long-context rates, and at Inspect's flat table |
| `error`, `limit` | A harness error or a limit the sample hit |
| `message` | The harness error, or the provider's message when it refused the request |

## 6. The metrics in `report.md`

| Metric | Definition | How to read it |
|---|---|---|
| **held** (usable context) | Largest size up to which *every measured* sample of the model was right, plus the first size with a miss (or the first size not measured) | The headline: "Sol held through 256K, first miss at 512K" |
| **Success by size** | right / samples at each size, all kinds pooled, with a 95% Wilson interval | Where accuracy starts to fall |
| **short vs long** | The same, pooled over the smaller half of the sizes (8K–128K) against the larger half (256K–960K) | Whether there is a drop at all, with more samples per number |
| **Success by kind** | right / samples per kind, all sizes pooled | What breaks: length itself or long-range retrieval |
| **Effort by size** | Median output tokens per sample (reasoning included) · mean generations | Whether the model works harder, or gives up, as context grows |
| **Not measured** | Samples the provider refused (`over_limit`: too large; `rejected`: another reason) or that errored, per model and size, with the first error message | A size here is a provider limit, *not* the model failing. They never count as right or wrong |
| **Per-sample grid** | One mark per task and replicate: ✓ right, ✗ wrong, – no answer, ⊘ refused for size, ! rejected or errored | Exactly which tasks failed where; consistent ✗ across replicates means a stable failure |
| **Failure labels** | Count per model, kind and label, over measured samples | Why answers were wrong |
| **Token check** | Provider input tokens / metered context | Should be close to 1; a big gap means "960K" is not what the model saw |
| **Spend** | Per eval set, at flat and long-context prices | What it actually cost |

## 7. Reading the results

**What each pattern means:**

| Pattern | Meaning |
|---|---|
| `fresh` fails as size grows | Length itself hurts: nothing had to be found far back except the handbook (which every kind needs). |
| `fresh` holds, `memo` fails (`ignored_memo`) | The model can work at that length but loses an instruction from the start of the context. This is long-range retrieval failing. |
| `followup` fails with `copied_original` | The model found case 1 but repeated its old decision instead of applying the memo in the final message. It retrieved but didn't reason. |
| `followup` fails with `wrong` | Usually the original request (customer, amount, tier) wasn't recovered from the start. |
| Larger models hold further | Capability buys usable context. Comparing the models' held sizes is the "does intelligence fix retrieval?" question. |
| Output tokens rise with size | The model spends more effort on long contexts. Falling output with falling accuracy suggests giving up or skimming. |

**How sure can you be?** With the default design there are 3 tasks per size per model, so one size's rate rests on 3 samples per replicate. 3/3 right has a 95% interval of 44–100%. So:

- **Say:** "Astra got all 18 samples right through 960K; Luna missed the memo task at 512K and 960K in both replicates." These are exact, descriptive statements.
- **Don't say:** "Luna is 33% accurate at 512K." The interval is far too wide for a rate.
- **Size of drop you can detect:**

  | Design | Samples per half (short vs long) | Example where the intervals separate |
  |---|---|---|
  | Default (3 tasks per size) | 9 | 9/9 (70–100%) vs 3/9 (12–65%): a drop of about 2/3 |
  | `--tasks-per-kind 3` | 27 | 27/27 (87–100%) vs 18/27 (48–81%): a drop of about 1/3 |

  Smaller drops need more tasks.
- **Replicates versus more tasks.** Replicates (`--rerun`) tell you whether a miss is stable on the same task: model randomness. More tasks (`--tasks-per-kind`, `--seed-base`) tell you whether a pattern holds beyond these particular tasks. Only more tasks support a general claim.
- **Cost of more tasks.** Cost scales with tasks. For a cheap firm-up, run more tasks on Luna and Sol, or drop 960K for Astra (`--sizes`).

**Limits to keep in mind:**
- The history is a perfect agent's transcript, so a real session (with the agent's own mistakes and chatter) may be harder.
- Every task needs the handbook at the very start of the context.
- These are F8 support-shift decisions, not general QA.

## 8. Cost

Per replicate, conservative (2 generations per sample) / expected:

| Model | Default (3 tasks per size) | `--tasks-per-kind 3` |
|---|---|---|
| Luna | $1 / $1 | $4 / $3 |
| Sol | $25 / $18 | $74 / $55 |
| Astra | $212 / $157 | $637 / $472 |
| **Total** | **$238 / $177** | **$715 / $530** |

- **Astra's sizes above 272K tokens** (512K and 960K) are charged its long-context rate of $20 input / $75 output per 1M tokens. That is $180 of Astra's $212. Confirm the rate on OpenAI's pricing page before running.
- **Typical cost is lower.** Most samples need one generation, so the actual cost is usually nearer half the conservative figure.
- **Allocation.** The sweep's allocation is $500 (`budget.allocations.context_sweep`). A design above it, such as 3 tasks per kind, needs the allocation raised first.
- **Guard.** Before each model, the runner refuses if the remaining samples' conservative cost exceeds what is left of the program budget or the allocation. The sweep's own spend is priced from its usage ledgers at long-context rates.

## 9. Troubleshooting

**If a request is too large.** The sweep protects the measurement in three ways:
1. **It sizes each request to fit.** Each generation's output cap is lowered so the input (as the provider may count it) plus the output fit the 1.05M window: 64K up to 512K, about 43K at 960K.
2. **A refusal is never scored as the model failing.** If the provider still refuses a request for its size, that sample is marked `over_limit` and "not measured". It is kept out of every rate and out of the held size.
3. **A refusal doesn't stop the run.** The sample ends without an error and isn't retried, so the model's other sizes still complete. The runner then prints which sizes were refused, records them in `run.json` (`over_limit_sizes`), and the report lists them under "Not measured".

Recovery is a new run at **every** size, never one size on its own: the comparison needs every size rendered the same way. Rehearse this path for free with `--mock refuse-largest`.

```bash
uv run python -m ape.run_sweep all --run-id sweep-dense --output-tokens 2000     # same sizes, ~1/2 the messages
uv run python -m ape.run_sweep all --run-id sweep-512 --sizes 8000,64000,128000,256000,512000
```

| Message or symptom | Fix |
|---|---|
| `refused: ... plan cell(s) ['sweep.luna'] are off` | Enable the cells (section 3) |
| `refused: the sweep's next tier: projected $X exceeds the remaining $Y` | Raise `budget.allocations.context_sweep` (or the program budget), or run fewer models, sizes or tasks |
| `refused: run X fixed its design` | You changed sizes, kinds, tasks or seeds: use a new `--run-id` |
| `refused: run X is offline` | Offline and live runs never share a run id |
| `warning: the provider refused <model>'s requests at 960K for size` (too long, or too many input items: about 4,000 messages) | Those samples show as `over_limit` / ⊘, "not measured"; every other size still ran and is valid. To measure that size, start a new run at every size with `--output-tokens 2000` (the same sizes in about half as many messages; 2000 is the most that still fits 8K), or leave the size out with `--sizes` (see above) |
| `refused: ... the system prompt, case 1 and the probe alone take N tokens, more than the smallest size` | `--output-tokens` is too large for the smallest size: use 2000 or less, or drop the smallest size |
| `rejected` samples in "Not measured" | The provider refused the request for a reason other than size: read the message in the report and fix the cause before re-running |
| Rate limits (429), slow progress | `--max-samples 2`; re-run the same command to resume |
| `run: ... incomplete` | Re-run the same command: completed samples are kept and only the rest run |

## 10. Files

| Path | Role |
|---|---|
| `src/ape/run_sweep.py` | The command: build, run, analyze, budget guard, replicates |
| `src/ape/worlds/gen_sweep.py` | Builds the worlds: head case, neutral middle fitted to the size, probe |
| `src/ape/tasks/sweep.py`, `src/ape/agent/sweep.py`, `src/ape/scorers/sweep.py` | Inspect task, solver (history replay and decision) and scorer |
| `src/ape/analyze_sweep.py` | The report and the metrics |
| `config/run_plan.yaml` (`studies.context_sweep`), `config/budget_assumptions.yaml` (`context_sweep`, `long_context_prices`) | Design defaults and prices |
| `tests/test_context_sweep.py` | Offline tests |
| `CONTEXT_SWEEP.md`, `DECISIONS.md` D-054 | Design and decision record |
