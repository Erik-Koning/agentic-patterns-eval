# Audit: adding context management to the study

**Status:** audit and design, 2026-09-30. Nothing here is implemented yet.
**Scope:** how to add compaction, pruning, external state (notes/todos), task extraction, restarts, reflection, offline consolidation and subagent isolation to the Inspect harness. The aim is to measure them as performance changes and to test whether their value survives stronger models (the "bitter lesson" question).

Mechanism names are used throughout. The motivating analogies map to them as follows.

| Motivating term | Mechanism name (used in analysis) |
|---|---|
| compaction | **summary compaction** |
| forgetting | **pruning** (clearing stale tool results) and **trimming** (dropping the oldest turns) |
| todos | **external task state** (a structured todo list with states) |
| notes | **external notes** (a free-form persistent file) |
| task extraction | **structured goal/constraint extraction** into external state |
| restarting fresher / decompressing | **context reset with rehydration** from external state and/or a handoff summary |
| reflection | **error-triggered lesson writing** (environment feedback only) |
| dreaming | **offline consolidation** (distilling notes and transcripts into memory between sessions) |
| subagents | **subagent isolation** (exploration in separate contexts, condensed return) |

---

## 1. Summary

1. **The harness can host this with modest changes, and Inspect already implements most of the mechanisms.** Inspect 0.3.273 (installed, verified) provides:
   - `compaction(strategy, prefix, tools, model)` for custom loops. It returns a handler whose `compact_input(history)` produces the model's view and whose `record_output()` calibrates it.
   - Five strategies: `CompactionEdit` (pruning), `CompactionTrim`, `CompactionSummary`, `CompactionNative`, `CompactionAuto`.
   - A `memory()` tool (persistent notes within a sample) and a `todo_write()` tool (todo list with pending / in_progress / completed).
   - `deepagent()` subagents (`research`, `plan`, `general`) that run in isolated contexts and return only a report.

   What we must build ourselves: context reset with rehydration, lesson writing, offline consolidation, forked state probes, and the oracle-state arm.
2. **The current tasks are too short to show any context-management effect.** F7/F3/F5 tasks take 1–4 tool calls and never approach a context limit. The study needs a **long-horizon family (F8 "shift")** with controllable length, and a **multi-session family (F9)** for consolidation. Both can be built from the existing generators, so the programmatic ground truth for agent state comes for free.
3. **Four measurement traps to fix before running:**
   - **Scale:** raw deltas shrink near the ceiling even when the effect is constant. Pre-register the logit scale for interaction tests and use RER for reporting.
   - **Context windows differ by tier:** Inspect's default thresholds are *fractions of the model's window*, so they must be set in absolute tokens.
   - **Model-native context management is itself a bitter-lesson channel:** OpenAI's native compaction runs through Inspect's `CompactionNative`, so it must be its own arm.
   - **Three tiers give three points on the capability axis:** use measured capability, not tier labels, and add reasoning-effort levels as extra points.
4. **Recommended minimum viable study:** 3 tiers × 7 context arms on F8, plus 4 topology arms for H1/H3. The per-position analysis inside long sessions yields the degradation curve without running every length. Offline consolidation and reflection go in Tier B (costly and low prior).

---

## 2. Tightening the hypotheses and measurements

The questions are good. These changes make each hypothesis falsifiable and hard to misread.

### 2.1 H1 (topology convergence)

- **Scale.** Near the ceiling, a constant logit effect produces shrinking percentage-point gaps, so "the gap shrinks" in raw terms can be pure ceiling. Pre-register:
  - **Primary test:** the `topology × capability` interaction in a logistic mixed model, on the logit scale. H1 holds if it is negative (the effect shrinks), with a one-sided CI excluding 0.
  - **Reporting metric:** relative error reduction, RER = (s_P − s_ref) / (1 − s_ref). This is your "fraction of remaining errors closed".
  - **Guardrail:** RER is unstable when 1 − s_ref < 0.1, so report it with bootstrap CIs and flag near-ceiling cells.
- **Capability axis.** Use **measured capability**, not tier labels. Define it as the baseline arm's success on a short-horizon held-out anchor (e.g. F7-10 + F3-5) at that tier and effort.
  - Tier labels confound capability with context window, price, speed and default reasoning effort.
  - Three models give three points. Adding reasoning effort (low/high) doubles the points cheaply. Record the effort actually honoured (E3 probe), because effort also changes token use.
- **Claim strength.** Three to six capability points support a *trend with a CI*, not proof of future convergence. Frame the study as a living benchmark: same generator, same code, rerun on each new model release. Seeded generation makes this cheap.

### 2.2 H2 (context persistence)

- **"Gains do not shrink"** is a null claim, so it needs an **equivalence test**: TOST on the `strategy × capability` interaction, with the margin pre-registered in logit units. Non-significance alone proves nothing.
- **"Gains grow with task length"** is the `strategy × length` interaction. Length is measured at each item's decision point as **log(view tokens)**, and separately as item position within the session.
- **Oracle gap per tier** (headroom context management could recover): **Gap_T = s(O-state) − s(CM0)** at tier T.
- **Headroom recovered** by strategy x: **R_x = (s_x − s_CM0) / (s_O − s_CM0)**. This normalizes for tier and length.
- **The strongest form of H2:** Gap_T stays positive for long sessions at every tier, **and** R_x does not fall with capability.
- **Extra outcome to plan for:** the gap closes because the *baseline* improves. Better long-context retention in newer models flattens CM0's degradation curve. That is the bitter lesson operating through the model rather than the scaffold, and it is directly observable as CM0's degradation slope per tier (§5).

### 2.3 H3 (decomposition)

This needs a factorial, not a single contrast.

- **Arms:**
  - S1: single agent.
  - M1: orchestrator with identical workers, i.e. isolation only.
  - M2: M1 plus role specialization.
  - S-CM*: single agent with the best context-management stack, chosen on dev.
  - S-subiso: single agent that delegates only exploration to isolated subagents.
- **Decomposition:**
  - isolation share = (M1 − S1) / (M2 − S1)
  - specialization share = (M2 − M1) / (M2 − S1)
- **Pre-registered thresholds:**
  - H3a: isolation share ≥ 0.5.
  - H3b: S-CM* recovers ≥ 80% of (M2 − S1) at ≤ 60% of M2's cost per solved item.
- **Link to the gate study:** this reuses the brief's M1s/M1/M2 arms. S1+KG is the "single agent + knowledge graph" topology, using whichever KG the gate selects.

### 2.4 Outcome table (completed)

| H1 | H2 | Also observed | Interpretation |
|---|---|---|---|
| true | true | — | Invest in context management, not new topologies. |
| true | false | CM0 degradation slope flattens with capability | Models absorb both; the remaining value is cost and latency. |
| true | false | Native compaction ≈ best harness strategy at high tiers | Context management is absorbed *into the provider/model*, not made unnecessary. Harness-side work loses value; the capability persists. |
| true | true | Gains only beyond window W | Context management is capacity engineering (it enables longer work), not an accuracy technique. |
| false | true | — | Both matter. Report the per-task-type topology winners (the gate/main study) alongside persistent context-management gains. |
| false | false | — | Topology matters; context management is transient. Identify the topology winners per task type. |

### 2.5 Naming collision

The brief already defines H1–H6. Prefix this study's hypotheses **G-H1, G-H2, G-H3** (Study G), and add G to the brief's study table.

---

## 3. What exists vs what is needed

| Area | Current state | Gap | Action |
|---|---|---|---|
| Agent loop (`agent/kb_react.py`) | Custom loop. Full history equals the model's view (except the per-step KB block). Fixed `max_turns = 12`. | No separation between the full history and the view; no hooks. | Add a **ContextPolicy** layer (§6) that keeps the full history and the view separate. The loop stays identical across strategies. |
| Tasks | F7/F3/F5: 1–4 tool calls, a few thousand tokens. | No context pressure, no long horizon, no cross-item state, no sessions. | Build **F8 shift** and **F9 multi-session** generators (§4) on top of the existing ones. |
| Ground truth for agent state | The world spec knows every fact and gold answer. | No per-step state (done / pending / active constraints). | The F8 generator emits a state timeline; `state_at(step)` renders the O-state oracle and scores probes. |
| Compaction | Only the brief's S2 arm (planned, not built). | — | Reuse Inspect's `compaction()` with Edit / Trim / Summary / Native. **Thresholds must be absolute token counts** (the default `threshold=0.9` is a fraction of each model's window). |
| External state | None. | — | Reuse `memory()` (notes) and `todo_write()` (task state). Check that their state is per-sample and inspectable for probes. |
| Reset with rehydration | None. | Not built into Inspect. | Custom policy: at a trigger, the view becomes system + task spec + external state + optional handoff summary (§6). |
| Lesson writing (reflection) | None. | — | Custom policy: on an environment error (tool error, rejected action), a short lesson-writing call appends to notes. **No grader feedback.** |
| Offline consolidation | None. | Needs sessions. | Custom offline pass between F9 sessions, metered in the build ledger. |
| Subagent isolation | Planned M1/M1s arms (brief). | — | Reuse `deepagent` `research()` / `general()` as tools for S-subiso; M1/M2 from the brief. |
| Forked state probes | None. | — | A side-branch generate on the current **view** (not appended to history), role `probe`, JSON answer, scored against `state_at` (§5.3). |
| Model tiers | OpenAI only (D-000); IDs pending E3. | Need ≥ 3 tiers from one family, plus effort levels. | Extend `readiness/probe_openai.py`: tiers, efforts honoured, context window, whether `CompactionNative` works (Responses API models). |
| Cost metering | `role_usage` per sample; build ledger. | Management tokens not separated. | Summaries under role `cm`; probes under role `probe` (excluded from cost); consolidation in the ledger; memory and todo tool calls counted as agent tokens. |
| Statistics | Gate stats (world-clustered bootstrap, NI). | No mixed models, interaction tests, TOST, RER, degradation slopes. | New `analysis/cm_stats.py` (§7). Extend `power_sim.py` for session-clustered interaction tests. |

---

## 4. Task families for this study

### 4.1 F8 "shift": long horizon, one session

One session is a queue of **N cases** handled in order by one agent. Case types reuse F7 (policy decisions) and F3 (procedures) from the same world. Four features create real context-management demand, all with programmatic ground truth:

1. **Context pressure.** Tool outputs are realistic but bulky: order histories and customer files of 1–3k tokens with distractor fields. The session crosses a nominal window W at a known item index.
2. **Mid-session policy memos.** "From case 12 on, EU refunds above $500 need compliance review." Some memos expire. The generator records which memos are active at each item.
3. **Cross-item dependencies:**
   - quotas ("at most 3 approvals per customer per shift")
   - follow-ups ("case 31 is a follow-up to case 7; keep the same decision unless policy changed")
   - deferred work ("recheck all escalations at end of shift")
4. **End-of-shift report.** List every case, its disposition, and all pending follow-ups. This is scored exactly.

**Knobs:**
- N ∈ {10, 30, 60}
- memo density
- dependency density
- tool-output size

Item order is **randomized within a session**, so position is not confounded with case difficulty.

**Outcomes:**
- per-item success (programmatic, as in F7/F3)
- dependency-item success (items whose correctness requires earlier state)
- report accuracy
- session success (all items plus report)

### 4.2 F9 multi-session (for offline consolidation)

**S = 5 sessions** over the same world. Some knowledge is **only available through experience**:
- **Undocumented tool quirks**, revealed by error messages the first time. For example, "carrier codes must be upper-case" or "the refund tool rejects amounts with cents".
- **Corrections announced once** in session 1 ("handbook policy P-4411 is superseded").
- **Standing customer arrangements** discovered through tool lookups.

**Outcomes:**
- per-session success
- learning curve (success by session index)
- **transfer items:** later-session items that are only solvable with retained knowledge

---

## 5. Measurements

### 5.1 Performance and cost

- **Success:** per item, per dependency item, per session, and report accuracy.
- **RER** against CM0 and against S1. **Headroom recovered** R_x (§2.2).
- **Cost per solved item:** all agent tokens plus management tokens, divided by items solved.
  - Management tokens include summaries (`cm` role), lesson writing, memory/todo tool traffic and reset handoffs.
  - Consolidation is amortized over the sessions that follow.
  - Probe tokens are excluded and reported separately.
  - Cache-adjusted $ is primary, because compaction and pruning break prompt-cache prefixes. Fewer tokens can still cost more.
- **Latency:** wall-clock and working time, with compaction and reset time broken out.

### 5.2 Degradation curve

- **Model:** per-item success regressed on **log(view tokens at the decision step)** and on **item position**, by arm × tier. This is a logistic GLMM with a random intercept per session.
- **Key quantity:** the slope. H2 predicts context-management arms have flatter slopes than CM0 at every tier. The bitter-lesson channel predicts CM0's own slope flattens with capability.
- **One long session is enough:** N = 60 sessions already give the curve across positions 1–60. Shorter N serves as a control for session-level effects, such as the report burden.

### 5.3 Forked state probes

- **Timing:** at checkpoints after items k ∈ {5, 15, 30, 45, 60}.
- **Procedure:** take the policy's **current view** (exactly what the model would see next). Add one probe message: "Without using tools, list as JSON: completed case IDs; pending case IDs; policy memos in force; open follow-ups."
- **Mechanics:** call `generate` with role `probe` and a strict response schema. The result is never appended to the history, so the run is unaffected.
- **Scoring:** F1 per category against `state_at(k)`.
- **Two companion measures:**
  1. **External-state accuracy.** Parse the notes/todo contents directly and score them programmatically against `state_at(k)`. This is "what the agent wrote down".
  2. Optionally, a **tool-enabled probe** that may read memory: "what the agent can retrieve".
- **Consistency check:** probe F1 must predict next-item dependency success. If it doesn't, probes measure something the policy doesn't use; report that correlation.

### 5.4 Failure taxonomy (programmatic, from logs and ground truth)

- **Forgot constraint:** an active memo was violated.
- **Resurrected done item:** a completed case was acted on again.
- **Dropped item:** a pending case never handled.
- **Stale state:** a decision was based on an expired memo or a superseded fact.
- **Summary loss:** after compaction, a fact needed later is absent from the view.
- **Hallucinated state:** the probe lists a nonexistent case or memo.

---

## 6. Strategy catalogue and integration

### 6.1 ContextPolicy layer (the only change to the loop)

```text
policy.tools()            -> extra tools (memory, todo_write, subagent tools)
policy.system_addendum()  -> instructions the strategy needs (identical base prompt otherwise)
policy.view(history)      -> (model_input, messages_to_append)   # compaction / pruning / reset / oracle
policy.after_step(...)    -> triggers (lesson writing on env error, reset at boundary)
policy.on_item_boundary(i)
policy.on_session_end()   -> consolidation hook (F9)
```

The loop keeps the **full history** for logging and ground truth, and sends **model_input** to the model. It logs per step: view tokens, management events, and management tokens by role.

### 6.2 Arms

| ID | Mechanism | Implementation | Knobs (tuned on dev, equal budget) | Management cost | Bitter-lesson prior |
|---|---|---|---|---|---|
| CM0 | none | Full history. Exceeding W is recorded as an overflow failure for the remaining items. | — | 0 | Reference. Its degradation slope is the absorption signal. |
| CM-prune | pruning of stale tool results | `CompactionEdit(threshold=T_abs, keep_tool_uses=k, memory=False)` | T_abs, k | ~0 tokens; breaks cache prefixes | Cost gains persist; accuracy gains shrink except on long N. |
| CM-trim | trimming the oldest turns | `CompactionTrim(threshold=T_abs, preserve=p)` | T_abs, p | 0 | Naive control; expected to hurt dependency items. |
| CM-sum | summary compaction | `CompactionSummary(threshold=T_abs, model=get_model(role="cm"))` | T_abs, prompt | Summary calls | Stronger models summarize better, so gains may *grow* with tier on long N. |
| CM-native | provider-native compaction | `CompactionNative(threshold=T_abs)` (OpenAI Responses `responses.compact`; supported models UNVERIFIED until E3) | T_abs | Provider-side | The direct test of absorption *into the model*. |
| CM-notes | external notes | `memory()` tool + instruction to record state | — | Tool traffic | Persistent on long N with cross-item state. |
| CM-todo | structured extraction + task state | Extraction step at start and at each memo → `todo_write()`; agent updates states | — | Extraction + tool traffic | Persistent where pending/done bookkeeping dominates. |
| CM-reset | reset with rehydration | At trigger (T_abs or every m items): view = system + task spec + notes/todo + optional handoff summary | trigger, summary yes/no | Handoff call | Strong on very long N; risk of losing implicit context. |
| CM-lesson | error-triggered lesson writing | On an environment error, one short call writes a lesson to notes. **No grader signal.** | max lessons | Lesson calls | Weak prior. Expected to shrink with tier (Tier B). |
| CM-consol | offline consolidation | Between F9 sessions, an offline pass distills notes + transcripts into memory for the next session | distillation prompt | Offline, amortized | Persistent only where experience-only knowledge exists (Tier B). |
| CM-subiso | subagent isolation for exploration | `deepagent` `research()` / `general()` exposed as tools; only the report returns | delegation instruction | Subagent tokens | Bridges to H3. |
| CM* | best stack (e.g. prune + todo + reset) | Chosen on dev by the skeptic | — | Sum of parts | The single-agent contender for H3. |
| O-state | oracle context | View = system + task spec + generator-rendered `state_at(step)` + current item's messages | — | Not a real system | Upper bound; defines Gap_T and R_x. |

### 6.3 Fairness rules for this study

- **Absolute thresholds.** T_abs is in tokens, identical across tiers. W (nominal window) is enforced by the harness, identical across tiers.
- **Same base loop, prompts, tools and turn caps.** Strategies add only their own tools and a short, fixed addendum.
- **Summarizer choice.** Primary: **self-summarization** (the `cm` role uses the same model as the agent), which is the realistic setup and part of what improves with capability. Sensitivity: a fixed summarizer across tiers, to isolate the "better summaries" channel.
- **Equal tuning budget** per strategy on dev, logged.
- **No grader leakage.** Lesson writing and consolidation see only environment feedback. Probes are side-branch only.

---

## 7. Design and statistics

**Factors:**
- tier (3 OpenAI models from one family; plus 2 effort levels if the E3 probe shows effort is honoured)
- arm
- length (via N and position)
- for H1/H3, topology (S1, S1+KG, M1, M2, S-CM*)

**Primary models** (pre-registered):
- **G-H1:** `success ~ topology * capability + (1 | session) + (1 | item)` on the logit scale. Test the `topology × capability` slope (negative means convergence).
- **G-H2:**
  - **Equivalence:** TOST on the `strategy × capability` interaction.
  - **Length:** `strategy × log(view tokens)` interaction on item success (flatter slopes).
  - **Headroom:** Gap_T and R_x per tier, with session-clustered bootstrap CIs.
- **G-H3:** isolation and specialization shares, and S-CM* recovery at cost ratio. Bootstrap over sessions.

**Clustering and multiplicity:**
- Items are nested in sessions, and sessions in worlds. Bootstrap over sessions, the same approach as the gate's world-clustered bootstrap.
- Holm correction within each hypothesis family. Fixed-sequence gatekeeping for H3a → H3b.

**Epochs:** at least 3; 5 for N = 60, where long-horizon variance is highest. Report pass^k for session success.

**Power:** extend `power_sim.py` to (a) session-clustered item outcomes and (b) interaction effects. Calibrate its variances on the micro-pilot. A three-point capability axis has little power for slope differences, which is another reason to add effort levels.

---

## 8. Minimum viable study and cost drivers

**Core runs:**

| Block | Arms | Family | Cells |
|---|---|---|---|
| Context management | CM0, CM-prune, CM-sum, CM-todo, CM-reset, CM-native, O-state | F8, N = 60 (curve from positions); N = 10 control for CM0 and O-state only | 3 tiers × 12 sessions × 3–5 epochs |
| Topology (G-H1/G-H3) | S1, M1, M2, S-CM* | F8, N = 30 | 3 tiers × 12 sessions × 3 epochs |
| Tier B | CM-notes (if not in CM*), CM-lesson, CM-subiso, CM-consol (F9) | F8 / F9 | After the core |

**Session counts:**
- Context-management block: 3 × (5 × 1 + 2 × 2) × 12 × 3 ≈ **970 sessions**, most at N = 60.
- Topology block: 3 × 4 × 12 × 3 ≈ **430 sessions**.

**Main cost driver: cumulative input tokens.** Without management these grow roughly quadratically with steps. A 60-item session with ~6 calls per item and a view approaching W can reach tens of millions of input tokens, most of them cached.

The dollar figure depends on:
- tier prices
- OpenAI's cached-input discount
- how much each strategy breaks the cache

So it must be measured on a **micro-pilot** (≈ 3 sessions per arm at one tier) before committing. No dollar estimate is given here because the E5 price table does not exist yet.

---

## 9. Validity threats specific to this study

1. **Window confound.** Fractional thresholds or native windows would make strategies trigger at different points per tier. Use absolute T_abs and a harness-enforced W (§6.3).
2. **Provider-feature confound.** The Responses API is Inspect's default only for o-series, Codex and GPT-5 models. Every arm at a tier must use the same API mode, and CM-native is only possible where supported. Record per tier which models ran on which API.
3. **Cache economics.** Pruning, trimming and summary compaction rewrite the prefix. Report cache-adjusted $, not raw tokens.
4. **Summarizer capability** is part of the treatment under self-summarization. The fixed-summarizer sensitivity run separates it out.
5. **Position vs difficulty.** Randomize item order; include item random effects.
6. **Probe validity.** Probes use a different prompt from the task. Report the probe-to-behaviour correlation (§5.3) and don't treat probe F1 as outcome success.
7. **Reflection without ground truth.** Literature suggests self-correction without external feedback rarely helps (unchecked). Restrict lesson writing to environment feedback and keep it in Tier B.
8. **Consolidation contamination.** Distilled memory can carry forward mistakes. Measure "stale state" errors in later sessions and include a no-distillation carryover baseline (raw notes).
9. **Synthetic realism.** F8/F9 are structured. Add a public long-horizon anchor later (e.g. multi-issue τ³ sessions or Terminal-Bench long tasks) if the effect is found.
10. **Model churn.** Pin model IDs and dates. The living-benchmark design (§2.1) turns churn into data.

---

## 10. Implementation plan (estimates)

| Step | Work | Effort |
|---|---|---|
| 1 | Extend the E3 probe: tier list, context windows, effort honoured, `CompactionNative` support per model | 0.5 day |
| 2 | ContextPolicy layer in the loop (history / view split, event and role logging); CM0, prune, trim, sum, native via Inspect `compaction()` | 1–2 days |
| 3 | F8 generator: queue, bulky tool outputs, memos, dependencies, report, `state_at(step)`, O-state renderer; reference solver and tests | 3–4 days |
| 4 | CM-notes / CM-todo (Inspect `memory()` / `todo_write()`), CM-reset, extraction step; probe harness (side-branch, role `probe`) | 2 days |
| 5 | Scorers (item, dependency, report, session, probe F1, external-state accuracy, failure taxonomy) and `analysis/cm_stats.py` (GLMM, TOST, RER, R_x, slopes, clustered bootstrap) | 2–3 days |
| 6 | Power simulation for session-clustered interactions; micro-pilot plan | 1 day |
| 7 | Tier B: CM-lesson, CM-subiso (deepagent tools), F9 generator + CM-consol offline pass | 3–4 days |
| 8 | Add Study G to `ORCHESTRATOR_BRIEF_v2.md` (switches, hypotheses G-H1–G-H3, families F8/F9, arms) and a pre-registration | 1 day |

All offline and testable with `mockllm` plus fake embeddings, like the gate harness.

---

## 11. Open decisions

1. **Tiers:** which three OpenAI models, and whether to add effort levels as capability points (recommended).
2. **Nominal window W and threshold T_abs:** recommend W = 64k and T_abs = 40k. That way long F8 sessions exceed W at mid-session for every tier, whatever its native window.
3. **Primary summarizer:** self-summarization (recommended) or a fixed model.
4. **Tier B scope:** whether consolidation (F9) and lesson writing are in the first round.
5. **Ordering:** run Study G alongside the KG gate (it doesn't depend on the gate except for the S1+KG topology arm), or after.
