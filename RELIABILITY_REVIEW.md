# Reliability review (2026-10-02)

## Why this review

Every serious bug found so far sat on a path only a real model or real data exercises:
- the F3 tool crash;
- the pilot ignoring its budgets;
- LightRAG's lock errors;
- retries losing spend.

The offline mocks always produce well-formed calls and score about 0, so neither the agent loop nor the statistics had been tested against realistic behaviour.

## Method

Three independent reviewers, all read-only, each proved its findings with offline reproductions:
1. **Agent loop, tools and scoring**, driving the real loop with scripted misbehaviour.
2. **Knowledge-graph and embedding live paths**, using simulated imperfect LLM output.
3. **Statistics and orchestration**, using Monte Carlo on `power_sim`'s generative model and the real `gate_stats` code, at 10,000 bootstrap reps.

I re-checked the most consequential claims against the code; every one held. The repros are in the session scratchpad (`review-agent/`, `review-kg/`, `review-stats/`, `review-orch/`).

**Severity:**
- **A:** gives a wrong verdict, crashes, or invalidates the results.
- **B:** biases results or wastes significant money.
- **C:** minor.

**Bottom line:** nothing found sinks the design. But four kinds of defect would hurt the results if the gate ran today:
- the statistics are slightly anti-conservative and the pilot misjudges power;
- some measurements are silently wrong (provenance, error labels, truncation);
- the builds are fragile and waste money;
- the per-step arms lose their reasoning at every step.

All of them are fixable before the freeze, mostly at S–M effort.

## 1. Statistics and pre-registration (fix before the freeze)

| # | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| S1 | **A** | **The minimum-worlds rule counts the wrong worlds.** It counts worlds over every arm in the frame (S7, S3s), not the worlds where APG* and LGR* are paired. A cell with 2 paired worlds returns GO where §8 requires INCONCLUSIVE. | `gate_stats.py:274-277`; repro: 2 paired worlds → GO, CI [−4.0, +2.1] pp | Count paired worlds per cell from `(apg − lgr).dropna()` before the bootstrap. Also raise a clear error, not a `KeyError`, when a cell has no shared task. |
| S2 | B | **The percentile cluster bootstrap is anti-conservative.** The real GO rate at Δ = −5 pp is **3.3%** at 16 worlds and 3.4–3.9% at 12 worlds, against a nominal 2.5%. CI coverage is about 93.7%. | Simulation, N = 6,000 per scenario | Pre-register a small-sample interval: a t-interval with Satterthwaite df over cells gives 2.6% type I and 0.91 power at 16 worlds. Alternatively, a √(n/(n−1))-rescaled bootstrap. |
| S3 | B | **The pilot underestimates σ by about 40%, so it overstates power.** It converts with p̄(1−p̄), while the true mean slope is 0.67–0.71 of that. At σ_g = 0.8, the true power at 16 worlds is **0.58**, but the pilot reports **0.84**. | `pilot.py:78-82`; 300 simulated pilots | Moment-match by simulating `power_sim`'s model (or fit a GLMM) and report σ with CIs. Taking max(prior, estimate) must not hide an underestimate. |
| S4 | B | **Delivery-mode multiplicity is undisclosed.** At the margin, P(any GO-type label) = **5.6%**: GO 1.1%, GO_PUSH_ONLY 2.3%, GO_PULL_ONLY 2.3%. | Simulation, N = 4,000 | Either test each single-mode verdict at α/2 (or Holm), or state in §3/§8 that GO_X_ONLY carries about 2α familywise. |
| S5 | B | **PC2 and PC3 fail on noise at ties.** PC2 fails 18% at a true tie; PC3 fails 16% when S6 = S5o = APG* near the ceiling. Each spurious fail is a PRECONDITION_FAIL and a re-pilot. | Simulation, N = 1,000 | Gate on a one-sided CI bound against the tolerance, not the point estimate. |
| S6 | B | **The INCONCLUSIVE extension and the NO-GO fix cycle cannot run as pre-registered.** Test seeds are fixed at 3000+i, so a new run regenerates the already-inspected worlds. There is no extension mode, and stage 1 already spends the full α. | `generate.py:6,24`; `analyze_gate.py:86` | Freeze a per-run seed base, implement the extension analysis now, and restate the α split. |
| S7 | C | **PC4 passes by construction.** LightRAG's `max_total_tokens` is 4 × budget, which is PC4's bound. S7's budget is its own target. | `lgr/adapter.py:64` | Report PC4 as descriptive, or bound it against an independent reference. |
| S8 | C | **S7 is matched per compile, not per sample.** Against a per-step APG*, the placebo delivers a fraction of APG's tokens per sample, which weakens the GO rule's placebo test. | `baselines.py:122`; `run_gate.py:1321` | Match on delivered tokens per sample, or make S7 per-step. |
| S9 | C | Unpaired tasks are dropped silently. INCONCLUSIVE returns before the F7-1000 floor and the S7 test are reported. PC5 gates the placebo's cap hits. | `gate_stats.py:197,208,280` | Report dropped tasks and the secondary conditions in every verdict, and exempt S7 from PC5. |

## 2. Agent loop and scoring

| # | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| L1 | **A** (Study G) | **A plausible F8 report shape crashes the scorer.** Lists of objects in the report raise `TypeError` in `set()`. The whole session errors and is retried from scratch, up to twice. | `scorers/session.py:66`; repro | Validate the report shape in `submit_shift_report` and return any error to the model. Make the scorer tolerant. |
| L2 | B | **The per-step arms probably lose the model's reasoning at every step.** The refreshed knowledge-base context goes in as a new user message after the tool outputs. OpenAI only carries reasoning items forward since the last user message, so each step starts cold. This handicaps APG-s, LGR-s and S3s, and skews APG*/LGR* selection. Unproven live; documented OpenAI behaviour. | `kb_react.py:116`; message-order repro | Put the refreshed knowledge-base text into the last tool result's content, in the view only, with no new user turn. |
| L3 | B | **`search_kb` output is cut at Inspect's 16 KiB default, but its evidence is credited in full.** Larger S3s and LightRAG contexts are truncated in pull mode only. | `kb_react.py:92`; repro at a 5,000-token budget | Give `search_kb` a `max_output` sized to the largest budget, and log what the model actually saw. |
| L4 | B | **Exceptions inside `search_kb` kill the sample.** An empty or huge query makes embeddings return 400, which becomes a harness error (PC5) and a full retry. | `kb_react.py:89-92`; repro | Strip and cap the query, and return failures as tool errors. |
| L5 | B | **Every correct F3 sample is labelled `wrong_arguments`** because the comparison doesn't sort keys. The NO-GO diagnosis table would be wrong. | `taxonomy.py:76`; repro | Use `json.dumps(…, sort_keys=True)`. |
| L6 | B | **Only one nudge per sample, ever.** A second text-only turn ends the sample as `no_answer`, which is not a cap hit. | `kb_react.py:124-128`; repro | Reset the nudge after any turn that makes tool calls. |
| L7 | B | **The F8 report phase has no nudge.** A report written as text scores 0, so the session fails. | `session.py:214-217` | Nudge once, as for cases. |
| L8 | C | A sample that hits a limit loses its logged state: no compile log, and F8 items are dropped. | `kb_react.py:130`; `session.py:221` | Write the store and messages in `try/finally`. |
| L9 | C | F2 partial credit is 0 when the chain includes the start supplier, and the prompt invites that. F5 exact match rejects `Kraków`, `Krakow office` and `Krakow, Poland`. | repros | Strip the leading start ID; normalise accents and suffixes. |
| L10 | C | F8: case IDs must match exactly, unknown IDs are acknowledged, duplicate answers resolve differently from the gate, and probe scores average only the probes taken. | `env_f8.py:39,57`; `session.py:45,275` | Normalise IDs, reject unknown ones, use first-answer-wins everywhere, and score missed probes explicitly. |
| L11 | C | TE-retrieved tool matching is case-sensitive, while LightRAG title-cases entity names. In pull and per-step modes the system prompt still says "knowledge is provided below". | `step.py:36`; `kb_react.py:31` | Use `re.I`; reword the system prompt per mode. |
| L12 | C | **The F8 loop has never hit the live API:** no probe with a strict schema, no reasoning-only turns. | — | Add an N = 5 CM0 / O-state check to the smoke run. |

## 3. Knowledge-graph builds and embeddings

| # | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| K1 | B→A | **One bad chunk fails the whole APG world.** `gather` has no exception handling, and there is no per-chunk retry or cache. A refusal or truncation in any of F7-1000's ~550 calls loses the world, and a re-run pays for every chunk again. A chunk that always refuses makes the world unbuildable. | `author.py:211`; `build_client.py:57`; repro | Retry or repair per chunk (refusal, `finish_reason`, JSON), cache units by prompt hash, and cancel siblings on a fatal error. |
| K2 | B | **Degraded builds pass silently.** LightRAG's parser never raises, so prose or malformed extraction output builds an index with missing policies. That garbage is also cached and replayed on rebuild. An empty APG authoring reports `id_coverage = 1.0`. Pilot and test builds are never quality-checked. | `lgr/build.py:165`; repros at 0.77 and 0.38 coverage | Fail or flag worlds with empty documents, zero leaves or too many empty chunks. Run `build_quality` on pilot and test builds. Don't cache truncated or empty output. |
| K3 | B | **LightRAG provenance is wrong in both directions.** Every source file of every delivered entity is credited: a generic entity claims all 130 facts while 21 are actually in the context. Conversely, when keywords come back empty, real context is delivered with 0 facts credited. This corrupts evidence recall and the NO-GO diagnosis. | `lgr/adapter.py:89-92`; repro | Credit provenance only from the delivered text and chunks (text overlap per fact). Record keyword failures. |
| K4 | B | **`id_coverage` is sensitive to ID format.** An author writing "Policy P-1035" scores 0.00 even when the graph is perfectly linked, while LightRAG is measured by regex. This could wrongly trigger the $957 Sol fallback. | `author.py:177`; `build_quality.py:43`; repro | Tell the author to use bare IDs, normalise IDs before linking, and measure both systems the same way. |
| K5 | B | **APG classify failures silently deliver empty context** and are never counted. Fenced output and `"id: title"` matches fall back to the root node. | `classify.py:60`; repro | Count fallbacks per arm and cell, with a threshold. Strip fences and leading `id:` prefixes. |
| K6 | B | **Graphs are loaded with no freshness check.** Four stale oracle graphs in `indices/apg/` have 256-d fake vectors and the old "code code" text. Routing ran silently on 1536-d queries against 256-d vectors. | `apg/arm.py:51`; `ls indices/apg` | Write and check `meta` (world hash, embedding model) on every graph and index at query time, and delete the stale files. |
| K7 | C | Embeddings retry twice where builds retry 6 times; an embedding 429 after authoring throws the authoring away. Calls cancelled at the timeout are unmetered. LightRAG build embeddings lack world attribution. The D-017 label ignores the fallback builder. | `embeddings.py:157`; `lgr/common.py:36` | Use `max_retries=6`, make authoring resumable (K1), pass the embedding context, and fix the label. |

## 4. Orchestration

| # | Sev | Finding | Evidence | Fix |
|---|---|---|---|---|
| O1 | B | **A non-converging matched-budget calibration aborts the pilot.** It raises before `pilot.json` is written, so the freeze is impossible, contradicting D-022. A re-run fails the same way. | `run_gate.py:1362-1370`; repro on a step response | Write the closest caps with `converged: false`, warn, and mark the secondary "not matched". |
| O2 | B | **The freeze doesn't check the pilot is current.** Re-running tune after the pilot freezes a new APG*/LGR* with S7 targets and caps sized for the old selection. | `run_gate.py:2096`; repro | Recompute the upstream fingerprints in the freeze and refuse on a mismatch. |
| O3 | B | **The freeze guard misses code the verdict depends on:** scorers, generators (the test split is generated after the freeze), `kb_react`, the adapters and S7. Nothing compares HEAD with the freeze commit. The anchor can rewrite `pc1.json` after the freeze. | `run_gate.py:217`, `:1483`; no `refuse` on the anchor phase | Refuse build-test, test and anchor when `src/` differs from the freeze commit. |
| O4 | C | `APE_BACKUP_DIR` and `accept_pc1_failure` change phase fingerprints, so `all` stops after a freeze. On resume, the budget guard double-counts finished work. Live runs don't clear stray `APE_WORLDS` / `APE_CACHE`. | `run_gate.py:231`, `:2028` | Add them to MANAGED_ENV or exclude them from fingerprints; project only the remaining work. |

## 5. Do we have the eval set and the tests?

**Eval set:** yes, by design it is generated, not stored.
- Every world is deterministic from its seed, with split-disjoint seed ranges for dev, pilot and test.
- The test split is deliberately not generated yet; only the post-freeze `build-test` phase may create it.
- The real anchor, GraphRAG-Bench Medical, is downloaded to `cache/graphragbench`.
- The fix list includes per-run seed bases (S6) so an extension or a fix cycle never reuses inspected test worlds.

**Tests:** 337 offline tests, all passing.
- **Strong:** the orchestrator, generators, smoke, analysis, spend and budget.
- **Thin where real models press:**
  - the agent loop has no test file of its own;
  - LightRAG has 3 tests, APG authoring 3, scoring and error labels 4.
- Every finding above should land with a regression test built from its repro, so the suite also covers misbehaving models and realistic data.

**Not covered by any test or smoke check yet:**
- the F8 loop on the live API (L12);
- real LightRAG extraction at F7-1000 scale (the smoke run builds only small worlds);
- the main-study and Study G arms, which don't exist yet.

## 6. Fix plan

| Batch | Items | Effort | Before |
|---|---|---|---|
| **R1: statistics and pre-registration** | S1–S6, S8, S9; GATE_PREREG §2/§3/§4/§7/§8 updated to match | M | the freeze (pilot σ: before the pilot) |
| **R2: agent loop and scoring** | L2–L7, L8, L9, L11; L1 and L10 with the F8 code | M | the smoke run |
| **R3: builds and provenance** | K1–K7 | M | build-dev (K1, K2 and K4 waste money if skipped) |
| **R4: orchestration** | O1–O4, S7 | S–M | the pilot |
| **R5: smoke additions** | L12; a real-extraction check on one F7-1000 world, within the smoke cap | S | the first live gate run |

R1–R4 touch separate areas and can run in parallel, like the hardening round. Each fix gets a regression test made from its repro.
