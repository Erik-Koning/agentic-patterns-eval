# Agentic Pattern Suitability: Eval Sets and Experiment Design

Status: design draft, 2026-09-29. Benchmark facts were checked against primary sources (papers, repos, HF cards, leaderboards) on 2026-09-29.

Tags used throughout:
- **[V]** verified at the linked source on 2026-09-29
- **[P]** partially verified
- **[U]** could not verify
- **[E]** my estimate, not a sourced figure

Power simulation: `power/power_sim.py` (run with `uv run --with numpy --with scipy python power/power_sim.py`).

---

## 0. Assumptions about your goals

1. **Weighting: big-data / analytical.** Your goal line says "big data analysis", which resolves the "[Pick one]" in the brief.
   - 5 of the 10 task types are analytical (T1–T5) and get about 65% of the budget.
   - T6 is analytical research over documents.
   - T7–T10 are **anchors**. They connect results to the literature (arXiv 2512.08296 used PlanCraft, BrowseComp-Plus, SWE-bench and others) and test whether conclusions generalize outside analytics.
2. **The deliverable is a decision rule, not a leaderboard.** The goal is to say "use pattern P for task type T", so cost and latency are first-class endpoints next to accuracy.
3. **"ASI level" means maximal rigor.** That means pre-registration, a power analysis, a compute-matched control, internal replication across benchmarks, and full trace release. It is not a claim about the systems under test.
4. **"Statistical proof" means error-controlled decisions.** Suitability claims are made under family-wise error control with effect sizes and CIs, plus replication across ≥2 benchmarks per task type. Nothing is "proven" beyond that.
5. **One model is held fixed per arm of the study, and two capability tiers are run.**
   - Primary: `claude-opus-5-5`. Secondary: `claude-haiku-4-5` or `claude-sonnet-5-5`.
   - Pattern rankings are known to depend on base-model capability (the capability-saturation effect in 2512.08296), so a single tier cannot support a general claim.
   - Swap in your own model choice if you have one.
6. **Infrastructure.**
   - You will implement all 9 patterns in one harness. Where possible this uses Microsoft Agent Framework (MAF) 1.0 orchestrations, and custom code for P0, P1, P2 and P6.
   - You have Docker and sandbox infrastructure.
   - I have **not** assumed you hold BigQuery or Snowflake accounts, so locally runnable sets are preferred.
7. **Use is research or internal evaluation.** Some recommended sets are non-commercial (QRData; KramaBench is unclear). They are flagged.
8. **What "tokens" covers.**
   - Included: every LLM call made by the pattern, including routers, selectors, reviewers and orchestrator ledgers.
   - Excluded but reported separately: grader/judge tokens and user-simulator tokens (τ³-bench).
9. **Budget.** The full factorial on a frontier tier is in the **tens of thousands of USD** (Section 2.7). If that is out of range, the staged and reduced designs in 2.7 apply.

---

## 1. Task taxonomy and eval sets

### 1.1 Patterns under test (with topology tags)

| ID | Pattern | Topology class | Control flow | Reference implementation (hold constant) |
|---|---|---|---|---|
| P0 | Single LLM call, no tools | single | none | One call. It gets the same task context every other pattern starts with (task text + file listing/schema summary). |
| P1 | Single agent + tool/code loop | single | model-driven | Tool loop with the full tool set, turn cap (e.g. 50) and token cap. **Primary baseline.** |
| P2 | Best-of-N at matched budget | independent | fixed | N i.i.d. P1 runs, then aggregation. Majority vote over normalized answers for exact-answer tasks; an LLM selector for artifact tasks (SQL, code, pipelines). N is set so mean tokens ≈ the costliest MAS pattern for that task type. The full BoN curve N=1..8 is also reported (Section 2.2). |
| P3 | Sequential (prompt chaining) | single *(ambiguous: pipelined, one locus at a time; see note)* | fixed | Planner → Executor (agent + tools) → Verifier/Reporter. MAF `Sequential`, with `chain_only_agent_responses=True` so each stage sees only the previous stage's output. |
| P4 | Concurrent fan-out + aggregate | independent | fixed | MAF `Concurrent`. k=3 agents with fixed lenses (e.g. data-quality / statistical / domain) solve the whole task without cross-talk. An LLM aggregator synthesizes. This is "sectioning" rather than "voting"; the voting variant is P2. |
| P5 | Handoff / routing | decentralized *(MAF Handoff is a mesh with "no central authority"; a pure classifier-router variant would be centralized)* | model-chosen edges, fixed agent set | Triage agent plus specialists (SQL/data-engineering, statistics, analysis/reporting). Each specialist owns a **partition** of the tools, since specialization is the intervention. Handoff cap. |
| P6 | Evaluator-optimizer | centralized *(ambiguous: the reviewer is a verification gate)* | fixed loop | The worker is P1. The reviewer has the same tools, can re-execute, and returns pass/fail + critique. ≤3 rounds. |
| P7 | Orchestrator-workers / Magentic | centralized | dynamic | MAF `Magentic`. Manager keeps a task/progress ledger; ≤5 workers per round; round, stall and reset caps. |
| P8 | Group chat / debate | decentralized | dynamic | 3 tool-enabled peers, 3 rounds, **round-robin speaker selection** (no LLM manager, which would make it centralized). Final answer by majority. |

**Topology class counts:** single {P0, P1, P3}, independent {P2, P4}, centralized {P6, P7}, decentralized {P5, P8}. Every multi-agent class has ≥2 patterns, which is needed to separate a *topology* effect from a *pattern-implementation* effect.

**Ambiguous tags.** P3, P5 and P6 are ambiguous. Pre-register a sensitivity analysis that re-tags them (P3→centralized, P5→centralized, P6→decentralized) and report whether the topology conclusions survive.

**Class definitions** follow 2512.08296 §3.1 [V]:
- *independent* = agents communicate only with an aggregator
- *centralized* = orchestrator ↔ agents only
- *decentralized* = all-to-all

**MAF status.**
- MAF orchestrations reached 1.0 in Python on 2026-06-18, with the announcement blog on 2026-07-08 [V] ([docs](https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/), [1.0 post](https://devblogs.microsoft.com/agent-framework/agent-frameworks-orchestration-patterns-reach-1-0/)).
- MAF's own docs warn that Magentic is "untested… outside of the original Magentic-One design", and that .NET Magentic is still marked experimental [V].

### 1.2 Taxonomy table

**Column meanings.**
- **Use n:** recommended sample for the main study.
- **Top reported:** the best published score, which may come from heavily engineered or multi-agent systems. The relevant saturation test is whether *your* P1 lands at 30–70%, which the pilot confirms.
- **Cost/pass:** one full pass over the "use n" items with **P1 on Opus 5.5 at list price** ($4 / $20 per MTok; cache reads $0.20).
  - [R] means the figure was reported by the source, with the model they used.
  - Multiply by about **21×** for all 9 patterns in one trial (Section 2.7).

#### Analytical core (big data)

| Task type | Eval set | Size (use n) | Grading | Top reported (saturation) | Contamination risk | License | Cost/pass (P1) |
|---|---|---|---|---|---|---|---|
| **T1. Large-scale data discovery & analysis (data lakes)** | [CoDA-Bench](https://arxiv.org/html/2606.15300) (ICML 2026) · [repo](https://github.com/ruc-datalab/CoDA-Bench) · [HF](https://huggingface.co/datasets/RUC-DataLab/CoDA-Bench) | 1,009 + 119 Hard; ~981 files per env, **20 MB – 45 GB** (use 100, stratified to include Hard) [V] | Normalized exact match, no LLM judge; deterministic [V] | 61.1% (Mini-SWE-Agent + GPT-5.5); Hard 49.6% → **unsaturated** [V] | **High unless sandboxed**: answers and reference code are public on HF. Block egress. [V] | MIT [V] | [R] $0.11–1.42/task, 6.8–39.4 turns → **~$50–140** [E] |
| | [KramaBench](https://arxiv.org/html/2506.06541) (ICLR 2026) · [repo](https://github.com/mitdbg/Kramabench) | 104 tasks / 633 subtasks; 1,764 files, **1.7 GB** (use all 104; subtasks as secondary) [V] | Exact 0/1; numeric 1/(1+RAE); list F1. Approximate strings use "ParaPluie", possibly model-based [U]; drop those items for a deterministic primary. | 55.8% (smolagents Deep-Research + Claude 3.7), 62.8% with oracle retrieval → unsaturated [V] | Medium-high: answers and reference pipelines public | Data **[U]** (paper CC BY-NC-SA; repo has no LICENSE) | **~$50–150** [E] |
| **T2. Data-science workflows (wrangling, EDA, ML, multi-step analytic QA)** | [DA-Code](https://arxiv.org/html/2410.07331v2) (EMNLP 2024) · [repo](https://github.com/yiyihum/da-code) | 500 (100 wrangling / 100 ML / 300 EDA) (use 200) [V] | Execution-based: table match, chart data/config, normalized ML metric; deterministic [V] | 38.5% (DS-STAR, a multi-agent system); GPT-4 30.5% → unsaturated [V] | Medium: gold public since 2024, likely in training data | MIT [V] | Avg 7.3 steps [R] → **~$60–120** [E] |
| | [DABstep](https://huggingface.co/datasets/adyen/DABstep) (Adyen/HF) — **hard split** | 378 hard (+72 easy); context ≈ **24 MB** (not big data) (use 378) [V] | `isclose` / fuzzy string; deterministic, **but answers hidden: scoring requires leaderboard submission** [V] | Validated: 89.95% hard (NVIDIA KGMON, a single agent + prebuilt library). Claude Code + Opus 4.5 as-is 66.9%; ReAct Sonnet 4 19.8%. Scaffold-sensitive, near-saturated only for engineered systems. [V] | Low leakage (hidden answers), but unlimited submissions allow probing; the unvalidated board is gamed. v2 announced, so pin v1. | CC BY 4.0 [V] | [R] $0.11/task (GPT-4o), $139/run (Claude 3.7) → **~$100–200** [E]. *Optional if leaderboard logistics (54 submissions) are infeasible.* |
| **T3. Enterprise text-to-SQL (warehouse-scale schemas)** | [LiveSQLBench](https://livesqlbench.ai/) Base-Full-v1 + **Large-v1** · [HF](https://huggingface.co/datasets/birdsql/livesqlbench-base-full-v1) | 600 + 480. Large: ~1,000 columns, 54 tables/DB, ~84K-token prompts (use 80 Large + 40 Base) [V] | Execution comparison (SELECT) + test cases; deterministic [V] | Base 48.0% (C3 AI DIA agent); Large 35.2% (GPT-5.5 xhigh, Codex) [P] → **unsaturated** | **Low**: hidden, rolling test set; each release's test becomes the next release's dev | CC BY 4.0 [V] | [R] Opus 4.6 ≈ $0.51/task on Large → **~$35–60** [E] |
| | [Spider 2.0-Lite](https://spider2-sql.github.io/) · [repo](https://github.com/xlang-ai/Spider2) | 547 (214 BigQuery / 198 Snowflake / 135 SQLite). **Snowflake eval account suspended since 2026-08-12**, so use the BigQuery + SQLite 349 (use 80) [V] | Execution accuracy on gold columns; deterministic [V] | 76.2% (proprietary agent, Jul 2026) vs 25.8% plain Spider-Agent + Claude 4 Sonnet on Snow → wide scaffold spread [V] | **High**: gold results public. The Snow variant has a 62.8% annotation-error audit; Lite is unaudited. [V] | MIT (BigQuery data terms [U]) | **~$40–80** API [E] + your BigQuery query charges [U] |
| **T4. Big-data engineering pipelines (Spark/Hive/Trino/Flink/dbt)** | [DataClawEval](https://arxiv.org/html/2607.28033) (Jul 2026) · [repo](https://github.com/Dicemy/DataClawEval) | 100 tasks across PySpark, MySQL, Hive, Trino, Flink in one Docker image (use 100) [V]; data volume [U], synthesized | Deterministic scripts, 70% outcome / 30% process [V] | 74.9 (GPT-5.5) [V] | **Low**: released Jul 2026 | MIT [V] | [R] 200k–1.53M tokens/task → **~$50–200** [E] |
| | [Spider 2.0-DBT](https://spider2-sql.github.io/) | 68 tasks, dbt on DuckDB (use 68) [V] | Execution-based; deterministic [V] | 65.6 (SignalPilot, May 2026) [V] | Medium: gold public | MIT [V] | **~$35–135** [E] |
| | *(optional)* [ELT-Bench-Verified](https://arxiv.org/html/2603.29399) | 100 pipelines, 835 source tables; Airbyte + dbt + Snowflake [V] | Deterministic data-model checks [P] | 32.5% (SWE-Agent + Sonnet 4.5) [V] | Low-medium | CC BY-SA 4.0 [V] | [R] **$343/run, ~2 d 7 h** |
| **T5. Statistical & causal inference from data** | [QRData](https://arxiv.org/abs/2402.17644) · [repo](https://github.com/xxxiaol/QRData) | 411 (248 multiple choice / 163 numeric; 269 causal / 142 statistical); 195 files, 313 MB (use 100, stratified) [V] | Multiple-choice prefix / numeric ±3%; deterministic [V] | GPT-4 58% (2024); frontier current **[U] → pilot must confirm headroom** | Medium-high: answers public since 2024 | **CC BY-NC(-SA) 4.0, non-commercial** [V] | **~$5–20** [E] |
| | [ScienceAgentBench](https://arxiv.org/abs/2410.05080) (verified release 2026-04-30) · [HAL](https://hal.cs.princeton.edu/scienceagentbench) | 102 tasks from 44 papers (use 102) [V] | Program executed + task-specific checks; **GPT-4o judge for figure tasks** (exclude those for the deterministic primary) [V] | 33.3% (o3, Self-Debug scaffold) vs **9.8% for the same o3 in the HAL generalist scaffold** → proven architecture sensitivity [V] | Low-medium: data in a password-protected zip | Code MIT; tasks mostly CC BY 4.0 [V] | [R] o3 $11.69/run (Self-Debug) → **~$15–60** [E] |
| | *(optional third)* [CausalReasoningBenchmark](https://arxiv.org/abs/2602.20571) (Feb 2026) | 174 queries / 139 datasets (use 100 if included) [V] | Identification spec + point estimate ± SE; determinism **[U]** | 81% strategy / 38% full spec [V] | Low-medium | CC BY 4.0 [V] | **~$10–35** [E] |

#### Analytical research + anchors

| Task type | Eval set | Size (use n) | Grading | Top reported (saturation) | Contamination risk | License | Cost/pass (P1) |
|---|---|---|---|---|---|---|---|
| **T6. Multi-hop retrieval & analytical research** | [BrowseComp-Plus](https://arxiv.org/abs/2508.06600) · [repo](https://github.com/texttron/BrowseComp-Plus) — **pin retriever (Qwen3-Embed-8B or BM25) + judge** | 830 queries, fixed 100,195-doc corpus (use 120) [V] | LLM judge (Qwen3-32B default; pin it); retrieval deterministic [V] | 95.2% with custom retrievers (8-agent MAS) → saturated. **With the pinned retriever: GPT-5 71.7% (Qwen3-Embed) / 57.6% (BM25)** → unsaturated [V] | **Low**: encrypted + canary [V] | MIT [V] | [R] ~$1,000 for 830 with o3 → **~$120–160** [E]; GPU for the embedding index |
| | [WideSearch](https://arxiv.org/html/2508.07999) (ICLR 2026) · [repo](https://github.com/ByteDance-Seed/WideSearch) | 200 (100 EN / 100 ZH) (use 100, EN) [V] | Per-column rules (exact/number_near/date_near/url), LLM only for free-text columns. **Item-F1 is continuous**, which gives more power per item. [V] | o3: success 5.1% MAS vs 4.5% SAS; Item-F1 57.3 vs 52.6 → unsaturated, and breadth-sensitive [V] | Medium; **live web → needs a record/replay cache** so all patterns see identical pages | Repo MIT; HF "other" [V] | **~$100–300** [E] |
| | *(replication anchor)* [Finance Agent](https://www.vals.ai/benchmarks/finance_agent) v1.1 (Vals AI) | 50 public + 150 validation on request (337 private) [V] | GPT-4o rubric judge; not deterministic [V] | 64.4% (Opus 4.7, single-agent harness) [V] | Live EDGAR/web | Public set CC BY 4.0; harness MIT [V] | [R] o3 $3.79/query → **~$190 (n=50)**. Underpowered; used only to replicate 2512.08296's +80.8% centralized result. |
| **T7. Stateful tool use / enterprise workflows** | [τ³-bench](https://github.com/sierra-research/tau2-bench) v1.0.1: airline + retail + banking_knowledge (skip telecom, which is saturated) | 50 + 114 + 97 = 261 (use 120, stratified by domain) [V] | DB end-state hash × required messages: deterministic grading, but the **LLM user simulator adds variance** (pin it). **pass^k built in.** [V] | pass^1/pass^4: airline 84.0/70.0, retail 84.4/59.7, **banking 55.2/35.1** [V] | Medium: public plaintext | MIT [V] | [R] $0.40/trajectory (Opus 4.5 airline) + simulator → **~$60–90** [E] |
| | [AppWorld](https://appworld.dev/) test-challenge | 417 (use 100) [V] | DB-state unit tests incl. collateral damage; deterministic, no simulator; TGC/SGC [V] | 85.6% TGC / 77.0% SGC (LARA **MAS**, Sep 2026); next best 73.4 [V] | **Low**: test truth encrypted | Apache-2.0 (+ keep-encrypted rule) [V] | **~$30–100** [E] |
| **T8. Long-horizon sequential planning** | [PlanCraft](https://arxiv.org/abs/2412.21033) · [repo](https://github.com/gautierdag/plancraft) | 580 test (includes *impossible* tasks) (use 120) [V] | Deterministic simulator (pure Python) [V] | No leaderboard. 2512.08296: SAS 0.568 vs MAS 0.170–0.346 (**all MAS −39% to −70%**) [V] | Medium: test data ships in the package; procedural | MIT [V] | [R] 17.5–25.5k tokens/episode → **~$12–25** [E] |
| | [TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner) | 1,000 test (leaderboard-graded) + 180 validation (local) (use 100 from validation) [V] | Deterministic constraint checker [V] | 52.65% (HiMAP-Travel, a trained MAS); LLM + SMT solver 93.9%; 2026 frontier **[U]** [V] | Low-medium: test answers go through the leaderboard | Code MIT, data CC BY 4.0 [V] | **~$10–40** [E] |
| **T9. Software engineering** | [SWE-rebench](https://swe-rebench.com/) (pooled monthly HF splits) | 860 pooled; current window 111 (use 200 most recent) [V] | Tests in Docker; deterministic [V] | 64.5±1.4% (Fable 5), Opus 5 63.4% → unsaturated [V] | **Low**: mined after model release dates | CC BY 4.0 [V] | [R] $0.10–4.40/problem → **~$200–400** [E] |
| | *(optional)* [Terminal-Bench](https://www.tbench.ai/) 2.1 (89) / 4.0 (66) | Use 4.0 with the frontier tier, 2.1 with the weaker tier [V] | Isolated verifier container; deterministic [V] | TB4 58.2%; TB2.1 83.8% [V] | Low: canary + versioned | Apache-2.0 [V] | [R] $5–22/trial → **~$330–1,450** |
| **T10. Hard reasoning (control: little decomposable structure)** | [HLE-Diamond](https://lastexam.ai/blog/hle-diamond) (1,000, text-only) / HLE text-only (~2,150) | Use 200, multiple-choice-heavy [V] | o3-mini judge; the **multiple-choice subset (~24%) is deterministic** [V] | HLE text-only 54.2% (no tools); Diamond 60.6% without tools / 82.9% with tools [V] | Low-medium: canary, gated | MIT [V] | **~$20–100** [E] |
| | *(exploratory)* [MathArena](https://matharena.ai/) ArXivLean + Apex | ~46/month + 12 [V] | Lean proof check, fully deterministic [V] | 87% specialized / 65.2% GPT-6 Astra (ArXivLean) [V] | **Very low**: fresh monthly | CC BY-SA 4.0 (arXiv-derived) [V] | [R] $0.01–12.77/problem-run |

#### Considered and excluded

- **Saturated:**
  - InfiAgent-DABench 94.9%; DataBench 95.0%; TQA-Bench 97.4%.
  - Spider 2.0-Snow 96.7% (plus a 62.8% annotation-error audit and the suspended account).
  - BIRD test 82.4% (52.8% annotation errors in mini-dev); DAB 95.4%.
  - GAIA 93.4%; BrowseComp ~92% (self-reported); WorkBench 97.7%; MCPMark 92.9%.
  - SWE-bench Pro V2 public 99.4%; OSWorld-Verified 90.2%; AIME/HMMT 2026 ~100%; GPQA 95.8%; FrontierMath (also private).
- **Contaminated or flawed:** SWE-bench Verified (flawed tests, verbatim gold-patch reproduction); HotpotQA and MuSiQue.
- **LLM-judged with a compressed spread:** DiscoveryBench (5 agents fall within 28.8–33.7); BLADE; DSBench analysis (also non-commercial data); DAComp analysis half.
- **Wrong construct:** StatQA (column metadata only, no execution); BFCL (evaluates models, not architectures); Natural Plan (single-call, answers public since 2024).
- **Infeasible:**
  - MLE-bench (24 h on an A10 per attempt; leaderboard closed).
  - TheAgentCompany (LLM coworkers + LLM judge, 30 GB of services).
  - Vending-Bench 2 (run in-house only).
  - LakeQA (9.5 TB). Top score 18.4%; it is the only true TB-scale set and worth watching.

---

## 2. Experiment design

### 2.1 Design

- **Fully crossed, paired, repeated-measures:** task × pattern × trial, within each model tier.
  - Every sampled task runs under all 9 patterns.
  - r = 3 trials per (task, pattern); r = 4 for τ³ to match its leaderboard convention.
- **Blocked randomization.**
  - A block is one task × all 9 patterns, executed in random order, interleaved across tasks.
  - This spreads API load and time-of-day drift evenly across patterns.
- **Pilot → main (staged).**
  - Pilot: 20 tasks per type × 9 patterns × 3 trials, on a **disjoint dev split**.
  - The pilot is used to (a) equalize prompt tuning, (b) calibrate P2's N and the budget caps, (c) estimate the variance components for the final power calculation, and (d) check that P1 lands at 30–70% (otherwise switch benchmark split or model tier).
  - Pilot tasks are never reused in the main study.

### 2.2 Metrics (per task i, pattern p, trial r)

**Accuracy**
- y_ipr ∈ {0,1} using each benchmark's official pass criterion. Partial-credit scores (Item-F1, RAE, TGC/SGC) are secondary endpoints.
- **pass@1** = mean over tasks of c_i/n, where c_i is the number of successes in n trials.
- **pass@k** (Chen et al. unbiased estimator) = mean over tasks of 1 − C(n−c_i, k)/C(n, k). Report it for context only; it is *not* a deployable metric.
- **pass^k** (reliability; τ-bench definition) = mean over tasks of C(c_i, k)/C(n, k), the probability that all k independent trials succeed ([τ² code](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/metrics/agent_metrics.py)) [V]. With r=3, report pass^1, pass^2 and pass^3.
- P2's pass@1 is the accuracy of its *single aggregated answer*. Don't confuse that with pass@N.

**Tokens (summed over every LLM call in the run: orchestrators, workers, routers, reviewers, selectors, aggregators)**
- Record per call: `input_tokens` (uncached), `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens` (thinking tokens are billed as output).
- **Cost_$:** each field × its price, using list prices for the tier.
  - Opus 5.5: $4 input / $20 output / $0.20 cache read / 1.25× input for 5-minute cache writes.
  - Haiku 4.5: $1 / $5 / $0.10.
  - Sonnet 5.5: $2 / $10 / $0.20.
  - **This is the primary cost endpoint.**
- **Cache-neutral tokens** = input + cache_creation + cache_read + output. Report as a robustness check, because patterns with long shared prefixes benefit disproportionately from caching.
- **Coordination metrics**, computed *per run*. (2512.08296's Table 5 metrics were architecture-level constants, which made them near-duplicates of the architecture labels [V].)
  - Number of LLM calls and tool calls.
  - Inter-agent message tokens, and their share of total tokens.
  - Overhead vs P1 on the same task = (T_p − T_P1)/T_P1.
  - Redundancy = mean pairwise cosine similarity of worker outputs.
  - Task-level error amplification = E_p/E_P1.
- **Excluded from pattern cost:** grader/judge tokens and user-simulator tokens. Log them separately.

**Latency**
- **Wall-clock:** task start → final answer, including tool execution.
- **Summed model time:** Σ over calls of API response duration (streaming start → end).
- **Critical-path model time:** the longest dependent chain of calls. The ratio summed/critical-path is the realized parallelism.
- **Tool time**, and **queue/backoff time** (429 retries). Queue/backoff is subtracted from the primary latency and reported separately.
- **Max concurrency** per run is fixed (e.g. ≤5 simultaneous sub-agents), because P4 and P7 speedups depend on it.

**Compute-matched control (P2)**
- For each task, run P1 N_max = 8 times. P1's 3 main trials are a subset of these 8.
- Compute BoN(N) accuracy and cost for N = 1..8 by exhaustive subsampling. This gives an **accuracy–cost curve** per task type.
- Each multi-agent pattern is compared with BoN *interpolated at that pattern's own measured mean cost*, so matching happens after the fact rather than by guessing N up front.
- This roughly triples P1 spend, but it makes P2 available at every budget level.

### 2.3 Controls

| Factor | Control |
|---|---|
| Model | One pinned model ID per tier. Effort set explicitly (Opus 5.5 defaults to `medium`; thinking cannot be disabled). Same `max_tokens`. Sampling parameters can't be set on current Claude models, so stochasticity is handled by trials, not temperature. |
| Tools | Identical implementations and descriptions (sandboxed Python/DuckDB, SQL executor, retriever, file tools). Every agent gets the full tool set, except P5, where the union of the specialists' partitions equals P1's set. |
| Prompts | One base system prompt. Role add-ons are templated, versioned and length-audited. **Equal prompt-tuning budget per pattern** (same number of dev iterations on the pilot split), then frozen. |
| Harness | One harness; MAF 1.0 builders for P3, P4, P5, P7, P8. Framework-injected prompts (e.g. Magentic ledgers) count as part of the pattern and are logged. |
| Budgets | Identical hard caps per task: tokens (e.g. 10× P1's pilot p90) and wall-clock. A cap hit counts as a failure and is reported separately. Also run a sensitivity check at 2× the caps. (2512.08296's repo enforces only iteration caps, not token caps [V].) |
| Tasks | Stratified random sample per benchmark (by difficulty tag or domain), frozen with a published seed and task-ID list. Identical across patterns. |
| Environment | Docker images pinned by digest. **Network egress blocked** except tool endpoints, because CoDA, DA-Code, KramaBench and QRData answers are on HF or GitHub. Fixed corpora (BrowseComp-Plus pinned retriever). Record/replay web cache for WideSearch and Finance Agent. Pinned τ³ user-simulator model. |
| Caching | Prompt caching on (realistic cost). A **per-run nonce at the start of the system prompt** stops cross-run and cross-pattern cache hits from making later patterns look cheaper, while caching inside each run still works. |
| Grading | Official graders. LLM judges pinned by model and prompt, blinded to pattern, and validated against ~100 human labels per judged benchmark (report κ). |
| Drift | All patterns run interleaved in the same calendar window. 10 canary tasks re-run daily; run date is a covariate. |

### 2.4 Sample size (power analysis)

**Method: simulation** (`power/power_sim.py`).
- Generative model: logit P(y) = μ + u_task + β_pattern + v_task×pattern + Bernoulli trial noise.
- σ_u = 1.5; σ_v ∈ {0.5, 1.0}; baseline 50%.
- Paired test on task-level means; two-sided α = 0.05/8 (Bonferroni over the 8 contrasts vs P1, which is conservative next to the Holm procedure used); 80% power.
- **Minimum tasks per task type:**

| Minimum detectable Δ (pp) | r=1, σ_v=0.5 | r=1, σ_v=1.0 | r=3, σ_v=0.5 | r=3, σ_v=1.0 | r=5, σ_v=0.5 | r=5, σ_v=1.0 |
|---|---|---|---|---|---|---|
| 5 | >1000 | >1000 | 800 | 1000 | 500 | 650 |
| 8 | 800 | 800 | 300 | 400 | 200 | 250 |
| **10** | 500 | 500 | **200** | **250** | 120 | 200 |
| 15 | 250 | 250 | 80 | 100 | 60 | 80 |

**Decision: n ≈ 200 tasks per task type, r = 3.**
- This gives a minimum detectable effect (MDE) of ≈10 pp for each pattern-vs-P1 contrast, per task type.
- For each task type this is *confirmatory* evidence, and the pooled model across task types has much more power.
- **Why r=3 rather than more trials:** total runs n·r for the same power are nearly flat across r (500×1 ≈ 200×3 ≈ 120×5). More tasks generalize better, and r ≥ 3 is needed for pass^3 and for estimating trial variance.
- **Small benchmarks** (KramaBench 104, ScienceAgentBench 102, DataClawEval 100, Spider2-DBT 68) are pooled within their task type to reach n ≈ 200. Benchmark is a random effect.
- **Report the achieved MDE** for every cell. An effect is only called "no difference" when the MDE is below the practical threshold δ.
- **Why a 10 pp MDE is adequate:** the published effects are large. Finance-Agent +80.8% relative; PlanCraft −39% to −70%; ScienceAgentBench 33% vs 10% for the same model in a different scaffold [V]. Smaller effects than 10 pp rarely change a deployment decision once cost is included.
- **Re-run the simulation with pilot estimates** of σ_u, σ_v and baseline accuracy before committing the main-study budget.

### 2.5 Analysis (pre-registered)

**A. Primary contrasts (per task type T), with serial gatekeeping to control family-wise error**

1. **Gate 1: P vs P1** for each P ∈ {P0, P2..P8}.
   - Statistic: paired difference of task-level mean success.
   - Test: sign-flip permutation (10k).
   - CIs: hierarchical cluster bootstrap. Resample benchmarks, then tasks within benchmarks, keeping all patterns and trials of a sampled task together. 10k reps, BCa.
   - Correction: **Holm across the 8 contrasts** within T.
2. **Gate 2: P vs compute-matched BoN**, only for multi-agent patterns that pass Gate 1 as superior. Uses the same test at full α. Fixed-sequence gatekeeping keeps family-wise error without splitting α again.
3. **Across task types:** Benjamini–Hochberg FDR q = 0.05 over the whole set of Gate 1/Gate 2 decisions, as a global safeguard.

**B. Mixed-effects model (the pattern × task-type interaction)**

Binomial GLMM (lme4 / glmmTMB; brms for the Bayesian version):

```
y_ipr ~ pattern * task_type
        + (1 | benchmark/task)            # difficulty, nested
        + (0 + pattern | benchmark)       # benchmark-specific pattern effects (heterogeneity within a type)
        + (1 | task:pattern)              # overdispersion across trials
        + run_date + (1 | tier)           # or fit per tier; tier × pattern as a fixed effect when both tiers are pooled
```

- **Omnibus test:** a likelihood-ratio test of `pattern:task_type` (df = 8 × 9 = 72). This is the formal test of "suitability depends on task type".
- **Cell estimates:** estimated marginal means (emmeans) per cell on the probability scale.
- **Topology model:** replace `pattern` with `topology + control` and add `(1 | pattern)` nested in topology. Test `topology:task_type`. Run the re-tagging sensitivity analysis from Section 1.1.
- **Capability-saturation test:** add P1-baseline accuracy (per benchmark × tier) and its interaction with multi-agent patterns. This is a direct replication of 2512.08296's β(P_SA × log(1+n)) = −0.236 [V], which the paper tested only with 6 clusters and no repeated seeds.

**C. Cost and latency**
- LMM on log(cost_$) and log(wall-clock), with the same random structure. Report **geometric-mean ratios vs P1** with 95% CIs.
- Also a gamma GLMM as a robustness check.

**D. Joint accuracy–cost (the core of "suitability")**
- **Bootstrap Pareto membership:** in each bootstrap replicate, compute (pass@1, mean cost) per pattern within T and record which patterns are non-dominated. Report P(Pareto) per pattern.
- **Incremental cost per additional solved task** (ICER, borrowed from health economics): ICER = ΔCost / ΔAcc vs P1. Report bootstrap CIs, and use the net-benefit form to avoid unstable ratios when ΔAcc ≈ 0.
- **Net benefit:** NB_T(λ, κ) = λ_T · Acc − Cost_$ − κ_T · Latency.
  - λ_T = dollar value of one correct answer for task type T. κ_T = dollar value of an analyst-second.
  - Plot **cost-effectiveness acceptability curves**: P(pattern has max NB) as λ varies from $0.01 to $100 per correct answer. This makes the recommendation explicit for any budget preference.
- **Accuracy gain per 1k tokens:** ΔAcc / Δ(cache-neutral kTokens) vs P1. Report it, but it is secondary to Cost_$, because a cached token costs 5–20× less than an uncached one.

**E. Reliability and heterogeneity**
- pass^3 differences vs P1 (same gatekeeping).
- Forest plots of per-benchmark effects within each task type, with I².
- **Discordant-item audit:** hand-check labels on tasks where patterns disagree. Label noise is concentrated there (Spider2-Snow had a 62.8% error rate, BIRD mini-dev 52.8% [V]), and those tasks drive the conclusions.

**F. Predictive model (does the rule generalize?)**
- Regress the pattern effect on task features: tool count, a decomposability rating, sequential-dependency depth, data size (files/GB), schema width, P1 baseline accuracy.
- Validate with **leave-one-benchmark-out cross-validation**, which 2512.08296 described as challenging and did not report [V].
- Target: select the best pattern (by NB at pre-set λ) for held-out benchmarks better than "always P1" and better than a capability-only rule.

### 2.6 Decision rule: "pattern P is suited to task type T"

Pre-register the practical thresholds: non-inferiority margin δ_NI = 3 pp, value per correct answer λ_T, latency SLA L_T.

| Verdict | Conditions (all required) |
|---|---|
| **Suited: dominant** | (1) Gate 1 superior: Holm-adjusted lower CI of ΔAcc(P − P1) > 0. (2) Gate 2 (multi-agent patterns only): ΔAcc(P − BoN at matched cost) lower CI > 0, meaning the gain is **not just compute**. (3) P(Pareto) ≥ 0.8 on (Acc, Cost_$). (4) pass^3 not worse than P1 by more than δ_NI (lower CI > −δ_NI). (5) **Replication:** the effect has the same sign, with point estimate > 0, on every benchmark in T. |
| **Suited: efficient** | Non-inferior accuracy (lower CI of ΔAcc > −δ_NI) **and** cost or latency significantly lower than P1 (the upper CI of the geometric-mean ratio is < 1) **and** P(Pareto) ≥ 0.8. |
| **Suited at value λ** | Dominant conditions (1), (2) and (4) hold, but not Pareto at list price. Report the λ* at which P becomes max-NB with ≥ 0.8 probability; "suited if a correct answer is worth ≥ $λ*". |
| **Not suited** | Gate 1 inferior (upper CI < −δ_NI), **or** it passes Gate 1 but fails Gate 2 (use BoN instead: cheaper, simpler), **or** P(Pareto) < 0.2 at every λ in range. |
| **Inconclusive** | Anything else. Report the achieved MDE and the n that would be needed. |

**Equivalent token-threshold form:**
- P is worth it iff ΔAcc/ΔCost_$ > 1/λ_T. In tokens: ΔAcc per extra 1k tokens > p̄/λ_T, where p̄ is the realized blended $ per 1k tokens.
- Example: λ = $5 per correct analytical answer and p̄ ≈ $0.002 per 1k tokens (mostly cache reads) gives a threshold of 0.04 pp per extra 1k tokens.
- A pattern spending 400k extra tokens must therefore buy ≥ 16 pp.

### 2.7 Budget model

**Cost per trial of the full sweep** = Σ_T n_T × c̄_P1,T × Σ_p m_p.
- c̄_P1,T is the per-task cost for P1 in task type T.
- m_p is the pattern's cost multiplier relative to P1. These are my priors, to be replaced by pilot measurements:
  - P0 0.1, P1 1, P2 4 (plus BoN pool), P3 1.5, P4 3, P5 1.5, P6 2.5, P7 4, P8 4.
  - Σ ≈ 21.6.
  - Sources for the priors: 2512.08296 reports multi-agent overhead of 1.6–6.2× tokens vs single agent [V]; Anthropic reports multi-agent ≈ 15× chat vs agents ≈ 4× chat [V].

**Estimates** (the "use n" column sums to ≈ 2,000 tasks; ≈ 200 per type):
- P1: ≈ **$1.4–1.8k per trial on Opus 5.5** [E].
- P1 + P2 via the shared pool of 8 P1 runs per task: ≈ $11–15k.
- P0 and P3–P8 (Σ m ≈ 16.6) × 3 trials: ≈ $70–90k.
- **Total ≈ $85–105k on Opus 5.5** [E].
- The same on Sonnet 5.5: roughly half. The weaker tier (Haiku 4.5): roughly a quarter.

**Ways to cut 50–70% without losing the confirmatory core:**
1. **Multi-arm multi-stage (MAMS) design.** Run an interim analysis at 1/3 of the tasks. Drop a pattern for a task type when P(Pareto) < 0.05 **and** its Gate 1 upper CI < 0. Futility-only stopping keeps α valid.
2. **Full 9-pattern grid on T1–T5 only.** On the anchors T6–T10, run a reduced set {P1, P2, P4, P6, P7, P8} with n = 100 (MDE ≈ 15 pp).
3. **Use Sonnet 5.5 as the primary tier** and Haiku 4.5 as the weaker tier (for the capability-saturation test) on T1–T5. Optionally add Opus 5.5 on T1–T3.
4. **No Batch API for the main runs:** agent loops need turn-by-turn responses, and batching would destroy the latency endpoint. Only P0 could be batched, as a separate accuracy-only run.

**Realistic reduced plan** [E]:
- Sonnet 5.5 as the primary tier, with (1) and (2) applied: ≈ $20k.
- Haiku 4.5 as the second tier on T1–T5: ≈ $7k.
- Pilot: ≈ $4–9k.
- **Total ≈ $30–40k.**

---

## 3. Expected hypotheses (pre-registered predictions)

### 3.1 Meta-hypotheses

- **H-M1: capability saturation.**
  - Prediction: multi-agent patterns' gain over P1 shrinks as P1 accuracy rises. The (P1 baseline × MAS) interaction is negative, and the multi-agent gains are larger on the weak tier than on Opus 5.5.
  - Basis: 2512.08296 v3 β = −0.236, p = 0.004, with a threshold of about 45% single-agent accuracy [V]; Gao et al. 2505.18286 [V].
- **H-M2: compute explains most gains.**
  - Prediction: in ≥ 7 of 10 task types, ≥ 50% of any multi-agent accuracy gain over P1 is recovered by compute-matched BoN (Gate 2 fails).
  - Basis:
    - Anthropic: token usage alone explains 80% of BrowseComp variance [V].
    - Choi et al. 2508.17536: majority voting accounts for most gains attributed to debate [V].
    - Tran & Kiela 2604.02460: a single agent is best or within CI at equal thinking budgets on FRAMES/MuSiQue [V].
    - Jwalapuram et al. 2606.13003: auto-generated MAS lose to CoT-SC at up to 10× the cost [V].
- **H-M3: error amplification by topology.**
  - Prediction: independent patterns (P4) show the highest task-level error amplification on sequential-dependency types (T2, T4, T7, T8); centralized patterns (P6, P7) the lowest among multi-agent patterns.
  - Basis: 2512.08296 trace-level amplification of 17.2× independent vs 4.4× centralized [V]. The task-level effect is expected to be much smaller; the paper reports ≈ 1.1–1.3.
- **H-M4: latency.**
  - Prediction: P4 and P7 achieve parallelism ≥ 2× (summed / critical path), but their wall-clock beats P1 only on breadth tasks (T1, T6-WideSearch). P3, P6 and P8 have the highest wall-clock because their rounds run serially.
  - Basis: Anthropic reports research time cut by up to 90% with parallel subagents [V].
- **H-M5: reliability.**
  - Prediction: P2 and P6 raise pass^3 more than they raise pass@1. P5 and P8 lower pass^3 relative to pass@1 (more stochastic control paths; MAST inter-agent misalignment accounts for 32.3% of multi-agent failures [V]).

### 3.2 Per task type

| Task type | Predicted accuracy leader | Predicted Pareto-suited | Predicted not suited | Why |
|---|---|---|---|---|
| T1 data-lake discovery (CoDA, Krama) | **P7** > P4 > P1 | P1, P7 | P3, P8 | Breadth: ~1k files per environment, up to 45 GB. Context isolation lets workers explore file groups in parallel. Closest analogue is Finance-Agent, where centralized gave +80.8% [V]. **The largest multi-agent effect in the study is expected here**, and it is the only analytical type where Gate 2 is expected to pass. |
| T2 DS workflows (DA-Code, DABstep) | P1 ≈ **P6** | P1, P6 | P4, P8, P5 | Notebook-style sequential dependency; execution feedback already gives P1 self-correction. DABstep shows scaffold effects are huge but come from *tooling/libraries*, not multi-agent topology (the #1 validated entry is a single agent [V]). |
| T3 text-to-SQL (LiveSQL, Spider2-Lite) | **P6** ≥ P2 > P1 | P1, P2, P6 | P8, P5 | Output is verifiable by execution. Schema-linking and dialect errors are catchable by a reviewer that re-runs queries. BIRD-Critic: agents 56.5% vs best single model 46.0% [V]. |
| T4 pipelines (DataClawEval, Spider2-DBT) | **P6** ≈ P3 | P1, P3 | P4, P8 | Stage structure (extract → transform → validate) matches chaining. Checkable artifacts favor a reviewer. Fan-out adds merge conflicts. |
| T5 stats/causal (QRData, SAB, CausalRB) | **P2** ≈ P6 | P1, P2 | P8, P4 | Discrete answers, so voting reduces variance. SAB shows scaffold sensitivity (33% vs 10% for o3 [V]) driven by a self-debug loop, i.e. P1/P6-like, not multi-agent. |
| T6 retrieval/research (BC-Plus, WideSearch) | **P7** (WideSearch: P4 ≈ P7) | P1, P4, P7 | P3 | Breadth favors fan-out: WideSearch Item-F1 57.3 MAS vs 52.6 SAS [V]. On narrow multi-hop (BrowseComp-Plus) the gap vs compute-matched BoN is expected to close (Tran & Kiela; 2512.08296 gave decentralized only +9.2% [V]). |
| T7 stateful tool use (τ³, AppWorld) | **P1** | P1 | P4, P7, P8 | Shared mutable state and policy adherence. Tool-coordination trade-off β = −0.096 [V]. Possible exception: **P5 on τ³ banking_knowledge** (698 policy docs) through domain routing. AppWorld's top entry is multi-agent [V], so expect a smaller P1 advantage there. |
| T8 planning (PlanCraft, TravelPlanner) | **P1** | P1 | all multi-agent | Replicates 2512.08296: every multi-agent variant −39% to −70% on PlanCraft [V]. Strict sequential dependency; fragmented context loses state. |
| T9 SWE (SWE-rebench) | **P1** (P6 +small) | P1, P6 | P4, P8 | 2512.08296 multi-agent −2% to −15% on SWE-bench [V]. Anthropic: coding has few truly parallel subtasks [V]. |
| T10 hard reasoning (HLE) | **P2** | P0, P2 | P8 at matched budget | Little decomposable structure; debate ≈ voting [V]. P0 is cheap and competitive where tools add little. |

**Falsification criteria.** The central thesis is "multi-agent helps on breadth, hurts on depth, and mostly buys compute". It is refuted if either of these holds:
- any multi-agent pattern passes Gate 2 on T7 or T8;
- P7 fails Gate 2 on T1 **and** on WideSearch.

---

## 4. Threats to validity

### Internal validity
1. **Implementation-quality confound.** A pattern's score reflects its prompts and plumbing, not only its topology.
   - Mitigation: equal tuning budgets; one harness; framework defaults; publish prompts.
   - On a 3-type subset, run a **second independent implementation** of P4, P7 and P8 (e.g. custom code vs MAF) and check that verdicts agree.
2. **Compute confound.** Multi-agent patterns spend 1.6–6.2× more tokens [V]. 2512.08296 claims token matching that its own Table 5 contradicts [V].
   - Mitigation: the BoN curve with post-hoc matching (Gate 2) and cost-adjusted endpoints.
3. **Budget caps bite unevenly.** Multi-agent patterns hit caps more often.
   - Report cap-hit rate per cell; run a 2× cap sensitivity analysis.
4. **Cache carry-over.** Shared prefixes across runs would make later-run patterns look cheaper.
   - Per-run nonce; report cache-neutral tokens.
5. **Latency confounded with API load.** Mitigation: blocked, interleaved randomization; subtract queue/backoff time; fixed concurrency limit.
6. **Model drift during a multi-week run.** Mitigation: pinned IDs, daily canary tasks, run-date covariate.
7. **Reward hacking and answer lookup.** CoDA, DA-Code, KramaBench and QRData answers are public; Opus 5 forged a `go.sum` checksum on SWE-Pro [V]. More-exploratory patterns get more opportunities, which would bias *toward* them.
   - Mitigation: egress blocks, and trajectory audits for access to answer sources and test files.
8. **User-simulator variance (τ³).** It adds noise equally to all patterns, but it can interact with P5's conversational handoffs. Pin the simulator and report simulator-failure rates.

### Construct validity
9. **LLM-judge bias.** Affects BrowseComp-Plus, Finance Agent, HLE, WideSearch free-text columns and SAB figure tasks. Judges can favor longer or more hedged outputs, which multi-agent aggregation produces.
   - Mitigation: blinding, pinned judges, human κ checks, and deterministic subsets as the primary.
10. **Label noise.** Documented 52.8–62.8% annotation-error rates in BIRD mini-dev and Spider2-Snow [V]; flawed SWE-bench Verified tests [V].
    - Noise biases toward the null and can reward error-matching.
    - Mitigation: verified or recent sets, plus the discordant-item audit.
11. **"Big data" coverage is MB–GB, not TB.** Only LakeQA (9.5 TB) reaches TB scale, and it is too hard today (18.4%). Conclusions about TB-scale production systems are extrapolations.
12. **Topology tags are partly subjective** (P3, P5, P6), and each class has only 2–3 patterns. Topology effects stay confounded with implementation; the re-tagging sensitivity analysis is mandatory.
13. **List-price cost ≠ your cost,** and benchmark pass@1 ≠ business value. The CEAC (λ sweep) makes value assumptions explicit.

### External validity
14. **Model- and time-specific.** H-M1 means verdicts shift as models improve.
    - Express verdicts as a function of P1 baseline accuracy ("P7 helps on T1 while P1 < X%"), not as permanent facts.
    - Two tiers are the minimum. Optionally add one cross-vendor model on a subset.
15. **Contamination interacts with pattern.** Memorized answers help all patterns, but sampling-heavy patterns (P2, P8) surface them more often.
    - Compare effects on post-cutoff items (LiveSQLBench hidden set, SWE-rebench recent windows, DataClawEval, MathArena monthly) with pre-cutoff items.
16. **Benchmark tasks are cleaner than production analytics:** less ambiguity, no stakeholders. The results bound suitability; they don't certify it.

### Statistical conclusion validity
17. **Small benchmarks** (≈ 100 items) are pooled within their type. Achieved MDE is reported per cell, and "no effect" is never claimed above δ.
18. **Clustering:** subtasks within tasks (KramaBench), tasks sharing a database or repository. Random effects handle it, and the cluster bootstrap resamples at the highest level.
19. **Forking paths:** pre-register endpoints, contrasts, thresholds (δ_NI, λ_T, L_T), exclusions and the analysis code before the main run. Release all traces.
20. **Ceiling and floor effects:** the pilot gates each benchmark into the 30–70% P1 band per tier. Swap split or tier if a benchmark falls outside it.

---

## Appendix A: Source notes that affect the design

- **arXiv 2512.08296 "Towards a Science of Scaling Agent Systems"** (Kim et al., Google Research / DeepMind / MIT) [V].
  - **v3 (2026-04-08) differs from v1:**
    - v1: 180 configurations, 4 benchmarks, R² = 0.513.
    - v3: 260 configurations, 6 benchmarks (BrowseComp-Plus 100, Finance-Agent 50, PlanCraft 100, WorkBench 100, SWE-bench Verified 20, Terminal-Bench 20), R²_CV = 0.373.
    - The [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/) (2026-01-28) quotes the v1 numbers.
  - Only 6 clusters. With cluster-robust SEs, only the single-agent baseline and error × baseline terms survive. No repeated seeds are reported.
  - Code: [ybkim95/agent-scaling](https://github.com/ybkim95/agent-scaling), MIT.
    - Loaders exist for all 6 benchmarks, and it can be reused as a harness.
    - No token-budget enforcement.
    - The README estimates multi-agent at ≈ 3–4× single-agent cost.
- **[Anthropic multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)** (2025-06-13) [V].
  - +90.2% vs single-agent Opus 4 on an internal eval.
  - Tokens explain 80% of BrowseComp variance; tokens + tool calls + model explain 95%.
  - Agents ≈ 4× chat tokens; multi-agent ≈ 15×.
  - Multi-agent is a poor fit for shared-context, dependency-heavy work and most coding.
- **[Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)** (2024-12-19) [V]: definitions of chaining, routing, parallelization (sectioning/voting), orchestrator-workers and evaluator-optimizer.
- **Related work** [V]:
  - [MAST failure taxonomy 2503.13657](https://arxiv.org/abs/2503.13657): system design 44.2%, inter-agent misalignment 32.3%, verification 23.5%.
  - [Debate or Vote 2508.17536](https://arxiv.org/abs/2508.17536).
  - [Single-agent at equal thinking budget 2604.02460](https://arxiv.org/abs/2604.02460).
  - [Illusion of Multi-Agent Advantage 2606.13003](https://arxiv.org/abs/2606.13003).
  - [Single-agent with skills 2601.04748](https://arxiv.org/abs/2601.04748): −2% to +4% accuracy, −53.7% tokens.
  - [SAS or MAS? Why not both 2505.18286](https://arxiv.org/abs/2505.18286).
- **Harnesses worth reusing:**
  - MAF 1.0 orchestrations.
  - [MASEval](https://github.com/maseval/MASEval) (ACL 2026, MIT; wraps τ²-bench, Gaia2, MultiAgentBench).
  - agent-scaling (above).

## Appendix B: Open items to resolve in the pilot

- Frontier P1 accuracy on QRData, TravelPlanner and CausalReasoningBenchmark [U]. This determines whether they stay in.
- KramaBench data license and whether ParaPluie grading is model-based [U].
- DABstep: whether 54+ leaderboard submissions are acceptable to Adyen, or whether to drop it.
- Spider 2.0-Lite BigQuery query costs [U]; whether the Snowflake account is restored.
- Calibrate the budget multipliers m_p and P2's N from pilot token logs.
