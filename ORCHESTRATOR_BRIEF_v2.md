# Orchestrator Brief v2: Mechanism Attribution in Agent Orchestration Patterns & KG-Compiled Context

> Paste this whole file as the task for a lead/orchestrator agent (e.g. Claude Code, or any agent runtime that can read/write files, browse docs and optionally spawn subagents).
> The file is self-contained: the agent does not need the conversation that produced it. The appendix lists what changed from v1.

---

## 0. Configuration

Humans fill this in before handing the brief to the agent. Every field is one of three kinds:

- **`TODO-BLOCKING`**: the design depends on it.
  - Agent: proceed using the stated fallback, but list the field under **"Blocking inputs"** at the top of `DECISIONS.md` and as item 3 of your final message.
  - Mark every section built on the fallback `PROVISIONAL(<field>)`.
- **`TODO`**: needed, but a stated fallback is acceptable.
  - Agent: use the fallback and log it in `DECISIONS.md`.
- **Suggested default** (a value is already set): the human may change it.
  - Agent: use it as given, and list it in `DECISIONS.md` under "Defaults the human should confirm".

A field still reading `TODO…` means the human has not filled it in. Never treat the example text in a comment as the value.

```yaml
# --- Required inputs ---
kg_ingestion_pipeline: "ape/apg/author.py"  # APG graphs are hand-authored in production (no document ingestion exists), so the realistic pipeline is
                                          # LLM authoring from the shared chunks with no world-spec access: taxonomy → leaf policy nodes → bring edges → leaf allowlists,
                                          # validated + human sample review. Oracle APG (S5o) is built from the world spec as an upper bound only. See the APG gate plan.
team: TODO                                # Roles fixed (D-018): APG owner, skeptic (not on the APG side), analyst. Names pending (O-3).
compute_budget_usd: 1500                  # Gate phase only (D-018). Study G / main study budgeted after their micro-pilots.
models_available:                         # Exact provider/model IDs per tier.
  provider: openai                        # Decided 2026-09-29: OpenAI only (chat + embeddings).
  frontier: gpt-6-astra                   # D-015; exact IDs confirmed by readiness/probe_openai.py
  mid_tier: gpt-6-sol
  small_or_cheap: gpt-6-luna
  embeddings: text-embedding-3-small      # One embedding model shared by APG shortlist, S3s and LightRAG.
  open_weight_self_hosted: TODO           # Needs a rented NVIDIA GPU (the dev machine is Apple M1); needed for Study E and determinism regime (a).
target_output: TODO                       # Conference talk date / workshop paper / blog + open release.
                                          # Fallback: workshop paper + open release; no fixed date.
pricing_source: TODO                      # OpenAI pricing page URL + retrieval date, recorded at readiness E5.
                                          # Fallback: all prices are placeholders.

# --- Suggested defaults (human confirms or changes) ---
project_name: agent-mechanism-attribution
repo_root: ./research-plan
kg_framework: "APG (Adaptive Prompt Graph), ~/Documents/Work/Code/EvolvingWisdomAgents, apg-core 0.1.0 pinned by tree hash (PROVENANCE.md)"
                                          # NOT ~/Documents/Work/WisdomGraph (a separate preference-hint store). Comparator/control KG: LightRAG 1.5.7.
micro_pilot_budget_usd: 250               # Spent by humans only, after sign-off on the Phase-5 package (§7.1).
primary_model_tier: mid_tier              # The MVS runs on one model; other tiers are replication.
nominal_context_window_tokens: 128000     # Harness-enforced window W for F4 and the F7 size cap; NOT the model's native window.
scaffold_code: false                      # true = also create a minimal Inspect package skeleton (no paid runs).

# --- Fixed (the agent must not change this) ---
max_spend_during_planning_usd: 0          # The orchestrator must NOT run paid evals.
```

---

## 1. Your role and mission

You are the **lead research orchestrator**. Produce a complete, execution-ready **documentation package** that tells a small team exactly how to:

1. **Build the eval set.** Build a structure-controlled, non-biased eval set: generators → rendered documents → KB artifacts → frozen splits.
2. **Build one Inspect AI project** (UK AISI `inspect_ai`, Python).
   - All arms are **configurations of a single mechanism-switchable agent** (§4), plus scorers, limits and the log pipeline.
3. **Attribute multi-agent gains to individual mechanisms.** Use single-switch contrasts against a compute-matched single-agent frontier.
4. **Test KG-compiled context.** Compare it against monolithic prompts, equally engineered flat retrieval, and multi-agent specialization.
   - All arms get the same source content, and the KG is built by the real ingestion pipeline, not copied from the generator.
5. **Measure the outcomes:** accuracy, cost (four meters), latency (two serving regimes), determinism and auditability. Every definition is pre-registered.
6. **Analyze, present and release the results.**

**MVP-first rule.** The package must define a **Minimum Viable Study (MVS)** that fits `compute_budget_usd` with ≥ 20% contingency.
- Everything outside the MVS is tiered (Tier B / Tier C) and runs only if budget remains after the MVS.
- A deliverable that assumes the full matrix is a defect.

You **document**; you do not execute the study. No paid model runs. Illustrative code snippets are allowed; a code skeleton only if `scaffold_code: true`.

---

## 2. Background (re-verify every number before citing)

Condensed from a prior review (Sept 2026). Status markers:
- **[checked]**: seen at the primary source in the prior review. Re-verify anyway; papers get revised.
- **[unchecked]**: never verified.

In the deliverables, every number needs a primary URL or the label `UNVERIFIED`.

**Kim et al., "Towards a Science of Scaling Agent Systems"** (arXiv 2512.08296) [checked]
- Versions: v1 Dec 2025; v3 Apr 2026 (latest).
- v3 scope: 260 configurations, 6 benchmarks, 5 architectures (Single, Independent, Centralized, Decentralized, Hybrid), 3 LLM families.
- Relative change vs single agent ranges from +80.8% (Finance-Agent, centralized) to −70.0% (PlanCraft, independent).
- Capability saturation sets in near 45% single-agent success.
- Cross-validated R²: 0.513 in v1 (4 benchmarks) vs 0.373 in v3 (6 benchmarks).
- **Caveats that matter for this study:**
  - The 17.2× (independent) vs 4.4× (centralized) error-amplification figures are **trace-level**. Task-level amplification is ≈ 1.1–1.3, and as a regression predictor it is not significant (p = 0.658).
  - The "matched compute" claim is contradicted by the paper's own Table 5: MAS used 1.6–6.2× the turns of SAS.
  - No repeated seeds are reported, and there are only 6 benchmark clusters.
  - The Google Research blog (Jan 2026) quotes the v1 numbers.

**Anthropic, "How we built our multi-agent research system"** (Jun 2025) [checked]
- Orchestrator-worker scored +90.2% over a single agent on an internal eval.
- Multi-agent used ~15× chat tokens (agents ~4×).
- Token usage alone explains ~80% of BrowseComp variance.
- Not budget-matched.

**Budget-matched audits**
- Tran & Kiela (arXiv 2604.02460) [checked]: at matched thinking tokens, single ≥ multi on multi-hop.
- Jwalapuram et al., "The Illusion of Multi-Agent Advantage" (2606.13003) [checked]: automated MAS lose to CoT-SC while costing up to 10× more.
- "At Equal Inference Cost…" (2609.04217) [unchecked].
- Wunderlich et al. (2605.01566) [unchecked].

**Debate / council**
- Choi et al., "Debate or Vote" (2508.17536) [checked]: majority voting explains most MAD gains.
- Zhang et al. (2502.08788) [unchecked].
- Li et al., Self-MoA (2502.00674) [unchecked].
- "The Cost of Consensus" (2605.00914) [unchecked].

**Harness / infrastructure**
- MAST failure taxonomy (2503.13657) [checked]: system design 44.2%, inter-agent misalignment 32.3%, verification 23.5%.
- [unchecked]: HAL (2510.11977), MASLab (2505.16988), AgentArch (2509.10769), Magentic-One ledger ablation (2411.04468).

**KG / RAG / memory** [all unchecked]
- GraphRAG-Bench (2506.05690).
- Han et al. RAG vs GraphRAG (2502.11371): token-matched tie overall.
- Mem0 (2504.19413); Zep (2501.13956).

**Personas** [unchecked]: Zheng et al. (arXiv 2311.10054).

**Tool overload** [unchecked]: LangChain, "Benchmarking Multi-Agent Architectures" (2025).

**Gaps this study targets**
1. No controlled mechanism attribution: named patterns are compared as bundles.
2. Wall-clock latency and cache-adjusted $ are essentially unreported.
3. Rankings depend on the budget meter, and nobody reports all meters.
4. KG as compiled context / shared state has not been studied against multi-agent specialization.
5. Determinism and auditability are never first-class outcomes.
6. There is no controlled fault injection.
7. Task-mix confounds and tuning asymmetry bias existing comparisons.
8. **KG evaluations typically use oracle graphs, which hides extraction error.**

---

## 3. Research questions and hypotheses (pre-registration candidates)

WS8 sharpens these drafts into pre-registered form: exact contrast, margin, test and falsification condition. Margins shown are defaults.

**RQ1 Mechanism attribution.** Which mechanisms account for multi-agent gains over a single ReAct agent, compared at matched *realized* cost against a compute-matched single-agent frontier? Candidate mechanisms: decomposition, context isolation, ensembling, concurrency, role specialization, shared state, peer communication, model heterogeneity.

**RQ2 KG-compiled context.** Does per-step KG-compiled context in a single agent recover the benefit of multi-agent specialization? Does graph structure beat equally engineered flat retrieval when the KG is built by the real pipeline?

**RQ3 Meter sensitivity.** Do arm rankings change across the four meters: tokens, LLM calls, cache-adjusted $, wall-clock?

**RQ4 Determinism & auditability.** At matched accuracy, do arms differ on *empirical* determinism and auditability metrics?

**RQ5 Predictability.** Do a-priori task features predict the best arm better than family identity? Does the answer shift with model tier?

### H1: mechanism decomposition (all single-switch contrasts, §4.3)

- **H1a Isolation.** M1s − S9 > 0 on F1-high; ≤ 0 on F2-high.
  - In the MVS, M1 stands in for M1s. This is valid only if the H1d equivalence (M1 ≈ M1s) holds; if it fails, H1a moves to Tier B with a full M1s run.
- **H1b Ensembling.** On F2, S8 at M1's realized cost ≥ M1. On F1-high, S8 < M1.
- **H1c Communication.** M7 − S8 at matched cost lies within ±3 pp (equivalence, TOST).
  - Tier B tests which part of communication matters: content (M7 vs M9a) and interactivity (M7 vs M9b).
- **H1d Concurrency is latency-only.** |M1 − M1s| accuracy < 2 pp (TOST), and the wall-clock ratio M1/M1s < 0.6 on F1-high.
- **H1e Specialization.** M2 − M1k lies within ±3 pp (TOST).

### H2: KG-compiled context

- **H2 (substitution).** In the 2×2 core {S1, S5, M1, M1k} (delivery × architecture), the interaction is negative: the multi-agent advantage shrinks under KG delivery.
  - Operational non-inferiority: on F3 and F7, S5 ≥ M2 − 3 pp (absolute margin), at a realized-cost ratio S5/M2 ≤ 0.5 (upper 95% CI ≤ 0.6).
  - Multi-agent keeps its advantage on F1-high and F4-high.
- **H2-struct (the product claim).** With the **extracted** KG, S5 > S3s on F7-high-relational and F3-high.
  - Secondary: the gap S5o − S5 (oracle KG minus extracted KG) is reported as the cost of extraction error.
- **H2b Persona.** S5 − S5-P0 lies within ±2 pp (TOST).
- **H2c KB scaling.** Over log(KB size) on F7:
  - S1's slope is negative.
  - S5's slope is within ±2 pp per decade (TOST).
  - S5 is flatter than S3s **only** under relational policies (precedence/exceptions), not under independent policies.

### H3–H6

- **H3 Meter rank flips.** Kendall τ between arm rankings under tokens vs cache-adjusted $ vs wall-clock is < 0.8 in ≥ 2 of 4 MVS families. WS8 pre-registers the direction of each predicted flip.
- **H4 Fault propagation (Tier B).**
  - Exposure-conditional error lift under **message faults**: M6 < M5 < M1.
  - Under **environment faults**, S8 ≈ S1: environment faults are correlated across ensemble members, so voting cannot absorb them (§6.5).
- **H5 Determinism/auditability.** At matched accuracy (|Δ| < 3 pp), KG single-agent arms beat MAS arms on **D1, D2, D5, D6, A2, A4, A6, A8**. The definitional metrics D4, A3 and A7 are descriptive only (§6.4).
- **H6 Predictability.**
  - A-priori task features beat family identity for choosing the best arm under leave-one-family-out validation.
  - The coordination payoff (M1 − S8 at matched cost) decreases with model tier.

---

## 4. Mechanism model and arm registry (single source of truth; do not rename IDs)

### 4.1 Mechanism switches

Every arm is one configuration of a single agent implementation, `mech_agent(switches)`. The switch vector is logged in every sample's metadata.

| Switch | Values | Meaning |
|---|---|---|
| `DEL` | MONO, FLATq, FLATs, KGq, KGs, KGs-oracle, ORACLE, RAND | How KB content reaches the model. q = once per query, s = recompiled/retrieved every step. |
| `DEC` | 0/1 | The task is decomposed into subtasks by a planning step. |
| `ISO` | 0/1 | Each subtask runs in a fresh context that returns only its result (vs executing inside the planner's own history). |
| `CONC` | 0/1 | Subtasks run concurrently. **Predicted to affect latency only.** |
| `ENS` | k ∈ {1,2,4,8} | Independent full attempts plus aggregation (majority vote; LLM aggregator for free-form outputs). |
| `SPEC` | 0/1 | Workers get role-specific context (persona, procedures, tool subset). |
| `STATE` | none / text-ledger / typed-KG | Shared state across agents. |
| `CTRL` | dynamic / static | Who decides the next subtask: the model (replanning) or a fixed template. |
| `COMM` | none / real / null-crosstask / null-indep | Peer-to-peer messages. The nulls replace peer messages with messages from a different task, or from an independent run on the same task. |
| `HET` | 0/1 | Agents use different models. |
| `CMP` | 0/1 | History compaction. |

**Content-control rule.** Every arm draws on the **same source corpus** (§5.1). Arms differ only in the switches.

### 4.2 Arm registry

Tier A = in the MVS. Families are the default coverage; WS7 may narrow them, never widen them, within the MVS.

| ID | Arm | Switch vector (differences from S1 only) | Tier | Families |
|---|---|---|---|---|
| S1 | ReAct baseline | DEL=MONO, DEC=0, ENS=1, SPEC=0, STATE=none, COMM=none | A | all |
| S2 | ReAct + compaction | CMP=1 | B | F2, F4 |
| S3 | Flat retrieval per query | DEL=FLATq | C | F3, F7 |
| S3s | Flat retrieval per step (hybrid BM25+dense + reranker, engineered by the skeptic) | DEL=FLATs | A | F3, F7 |
| S4 | KG per query | DEL=KGq | B | F3, F7 |
| S5 | KG per step (extracted KG) | DEL=KGs | A | F3, F7 (+F1, F2 in Tier B) |
| S5o | KG per step (oracle KG from world spec) | DEL=KGs-oracle | B | F3, F7 |
| S5-P0 | S5 without persona nodes | DEL=KGs, personas removed | A | F3, F7 |
| S6 | Oracle context (gold evidence only) | DEL=ORACLE | B | F3, F7 |
| S7 | Random KG nodes, token-matched to S5 | DEL=RAND | A | F3, F7 |
| S8 | Self-consistency (compute frontier) | ENS=k, built from the S1 pool (§4.5) | A | all |
| S9 | Plan-then-execute, single context | DEC=1, ISO=0, CTRL=dynamic | A | F1, F2 |
| M1s | Orchestrator, identical workers, serial | DEC=1, ISO=1, CONC=0 | B (accuracy-equivalence check on F1 in A, n=50) | F1, F2 |
| M1 | Orchestrator, identical workers | DEC=1, ISO=1, CONC=1 | A | all |
| M1k | M1 with KG-compiled worker context, no roles | M1 + DEL=KGs | A | F3, F7 |
| M2 | Orchestrator, specialized workers | M1k + SPEC=1 | A | F3, F7 |
| M2-P0 | M2 without persona nodes | M2, personas removed | B | F3, F7 |
| M3 | Specialized handoff (shared history) | SPEC=1, ISO=0, control transferred via `handoff` | B | F3, F7 |
| M4 | Static pipeline | M2 + CTRL=static | C | F2, F7 |
| M5 | Text-ledger orchestrator | M2 + STATE=text-ledger | B | F2, F5 |
| M6 | KG-state orchestrator | M2 + STATE=typed-KG | B | F2, F5 |
| M7 | Council, homogeneous | ENS=k members + 2 critique rounds + chair, COMM=real | A | F1, F2 |
| M7-R0 | Council, zero critique rounds | M7 with COMM=none | B | F1, F2 |
| M8 | Council, heterogeneous | M7 + HET=1 (matched on cache-adjusted $) | C | F1, F2 |
| M9a | Council, cross-task null messages | M7 + COMM=null-crosstask | B | F1, F2 |
| M9b | Council, independent-run null messages | M7 + COMM=null-indep | B | F1, F2 |

Suffix `-C` (prompt caching disabled vs enabled) is used only in the latency/cache sub-study (Study E).

**Fixed structural parameters.** Workers, rounds and k are **fixed a priori**: 3 workers, 2 critique rounds, council k = 3. They are *not* scaled to fill a budget. The budget sweep in Tier B varies them explicitly.

### 4.3 Mechanism contrasts

| Mechanism | Contrast | Single-switch? | Predicted |
|---|---|---|---|
| Decomposition | S9 − S1 | yes | small +, F1 only |
| Isolation | M1s − S9 | yes | + on F1-high; ≤ 0 on F2-high |
| Concurrency | M1 − M1s | yes | accuracy ≈ 0; latency ↓ |
| Ensembling | S8(k) vs S1 | yes | + everywhere at higher cost |
| KG delivery (vs monolith) | S5 − S1 | yes | + on F3-high and F7-high |
| Graph structure (vs flat) | S5 − S3s | yes | + only on relational F7 and on F3 |
| Extraction error | S5o − S5 | yes | ≥ 0 |
| Specialization | M2 − M1k | yes | ≈ 0 |
| Shared history vs isolation | M3 − M2 | **no** (ISO and control both change) | report as a bundle contrast |
| Static vs dynamic control | M4 − M2 | yes | − on F2 |
| Text ledger | M5 − M2 | yes | + on F2 |
| Typed state | M6 − M5 | yes | + under message faults |
| Communication | M7 − M7-R0 (and M7 vs S8 at matched cost) | yes | ≈ 0 at matched cost |
| Communication content / interactivity | M7 − M9a / M7 − M9b | yes | ≈ 0 / ≈ 0 |
| Heterogeneity | M8 − M7 | yes (at matched $) | small + |

Any claim about a mechanism must cite a single-switch row. Otherwise it is labelled a **bundle contrast** and must not be attributed to one mechanism.

### 4.4 Invariants (harness sanity checks; a violation blocks analysis until explained)

- **M1 ≈ M1s on accuracy.** Concurrency must not change what the model sees.
- **S5 ≈ M3.** The computation is nearly identical. A gap signals a harness artifact, e.g. handoff message filters.
- **M7-R0 ≈ S8(k=3) + chair.**
- **S6 ≥ S5o ≥ S5 ≥ S7**, and S5 > S7. If S5 ≈ S7, the KG effect is only a "shorter prompt" effect.
- **M1 with 1 worker and no decomposition ≈ S1**, within the orchestrator's overhead.

### 4.5 Budget policy (caps, not "matched budgets")

- **B0** = median realized total tokens of S1 per family-condition on the micro-pilot.
- **One generous cap for every arm in the MVS:** C_max = 8 × B0 tokens (and a wall-clock cap).
  - A cap hit counts as a failure. Cap-hit rate is reported per cell.
  - Caps are enforced on **total tokens**; the enforcement meter is stated in PREREGISTRATION.md. All four meters are *reported*.
- **Compute-matched frontier.**
  - S1 is run 8 times per task. S1's 3 main epochs are drawn from these 8 runs.
  - S8(k) for k = 1..8 is computed by exhaustive subsampling, giving an accuracy-vs-realized-cost curve per condition.
  - Every other arm is compared with **S8 interpolated at that arm's realized mean cost**, on each meter separately. This supersedes v1's "1× / 4× B0" matching, which a single agent that stops early cannot fill.
- **Tier B budget sweep:** caps at 2×, 4× and 16× B0, with structural parameters varied (workers 2/3/5, rounds 1/2/3), to trace each arm's own cost curve.

---

## 5. Eval set design (the main source of bias; treat with most care)

### 5.1 Pipeline: world spec → documents → KB artifacts

1. **World spec** (seeded): entities, relations, timestamps, procedures/policies (with an interdependence graph), tools, and **atomic facts with IDs**.
2. **Rendered source corpus.** Natural-language documents rendered from the world spec, with controlled redundancy (each fact appears in r ∈ {1, 2} documents), paraphrase and distractor text. Authored procedures, personas and tool specs are written as documents too.
   - **This corpus is the single source of truth every arm draws on.**
3. **KB artifacts**, all derived from the corpus and logged with content hashes:
   - **MONO:** the corpus (or its policy/procedure subset) as the system prompt.
   - **FLAT:** chunks. The chunking policy is set by the skeptic on `dev`.
   - **KG (extracted):** built by `kg_ingestion_pipeline` from the corpus exactly as in production. Report extraction precision and recall against the world spec.
   - **KG (oracle):** built directly from the world spec. Upper bound only (S5o).
4. **Provenance.** Every chunk and every KG node carries the list of world-spec fact IDs it contains. This makes evidence scoring granularity-neutral (A1).

### 5.2 Families

| ID | Family | Structural knob | Levels (MVS in **bold**) | Predicted winner at the high end | Tier |
|---|---|---|---|---|---|
| F1 | Breadth aggregation | N independent entity lookups | **2**, 8, **32** | Isolation arms (M1) | A |
| F2 | Dependency chains | Hop depth (hop k needs hop k−1) | **2**, 5, **10** | Single agent (S1, S9) | A |
| F3 | Tool load | Tools including distractor domains | **5**, 20, **60** | Contested (S5 vs M2) | A |
| F7 | Policy / procedure compliance | KB size × policy interdependence | size **10**, 100, **1000**; interdependence **relational** / independent | KG only if relational | A (relational) / B (independent) |
| F4 | Over-window synthesis | Evidence corpus ÷ nominal window W | 0.25×, 1×, 4× | Isolation or compaction | B |
| F5 | Relational / temporal QA | Single-hop → temporal multi-hop | 3 levels | KG arms. **Home-field risk:** report separately and never pool into H2 headline claims. | B |
| F6 | Verifiable reasoning / code | Difficulty tiers; hidden tests | 3 levels | Contested (S8 strong) | C |
| F0 | Anchors (real benchmarks) | — | see §5.4 | Comparability with the literature | A+ / B |

**Symmetry.** Within each family, the knob spans conditions predicted to favor opposite architectures. Always report per condition, never only pooled means.

**Knob as a slope.** The primary test is arm × log(knob), which uses all levels. MVS runs only the two endpoints; middle levels are Tier B.

**F7 size cap.** The largest F7 KB must fit in ≤ 0.8 W. Monolith degradation must reflect dilution, not truncation.

**F4 window.** Defined against the nominal W, not the model's native window. This keeps costs bounded with 1M-context models.

### 5.3 Generator requirements

- **Deterministic from seed.** Each task emits: the task, the gold answer, **gold fact IDs** (primary evidence key), gold tool calls where relevant, the structural parameters, and 2–3 semantically equivalent paraphrases.
- **Programmatic scoring only for primary outcomes:** exact match, unit tests, sandbox end-state checks.
- **Splits:**
  - `dev`: tuning, KB/chunking/KG authoring.
  - `pilot`: micro-pilot and power recalculation.
  - `test`: generated from fresh seeds **after** the corpus, KB artifacts, prompts and analysis code are frozen. Never inspected during development.
- **Fault hooks** (§6.5):
  - **Environment faults** at the content level: corrupt fact f_j wherever it first surfaces (tool result, document, chunk or KG node), keeping evidence redundancy equal across arms.
  - **Message faults:** replace one worker/peer return value with a plausible wrong result.
  - **Never implement faults by editing prior assistant turns.** Edited history is not a realistic fault and, on providers that bind reasoning to the conversation (e.g. OpenAI reasoning items, Claude preserved thinking), it can silently drop reasoning or be rejected. Inject only via tool results or sub-agent return values.
- **Validation:** the generator's own solver reaches 100% on gold evidence (S6 sanity), plus a human spot-check of 20 tasks per family.

### 5.4 F0 anchors (real benchmarks; programmatic scoring where available)

These were verified in the prior review on 2026-09-29; re-verify.

**MVS "A+" (first addition if budget allows): τ³-bench banking_knowledge**
- Source: https://github.com/sierra-research/tau2-bench, v1.0.1. 97 tasks plus 698 policy docs, graded by DB end-state, with built-in pass^k. Top pass^1 ≈ 55%. MIT.
- It is a real-world F7/F3 anchor for the KG claim, with the KG built by extraction from the 698 docs.
- Arms: S1, S3s, S5, M2. Trials: 4. Pin the user-simulator model.

**Tier B anchors**

| Anchor | For | Size | Grading | License | Caveat |
|---|---|---|---|---|---|
| AppWorld test-challenge (https://appworld.dev) | F3 (many-API tool load) | 417 | Deterministic DB-state tests | Apache-2.0 | — |
| BrowseComp-Plus (https://github.com/texttron/BrowseComp-Plus) | F1/F2 | 830 queries, fixed corpus | LLM judge; pin retriever and judge | MIT | Primary scoring is not programmatic, so treat as secondary |
| CoDA-Bench (https://github.com/ruc-datalab/CoDA-Bench) | F4 | environments up to 45 GB | Normalized exact match | MIT | Answers are public on HF: block network egress |

**Minimum anchor size:** 150 items per anchor used for inference. Smaller anchors are descriptive only.

---

## 6. Metrics specification

Pre-register the exact formulas and compute every metric from Inspect logs.

**Unit of analysis.** Every metric is computed per sample × arm × model × condition. Aggregate with task-clustered bootstrap CIs.

**Composites.** Any composite needs pre-registered weights, and its components must always be reported alongside it.

### 6.1 Outcome

- **Success:** binary, programmatic.
- **pass@k** and **pass^k** (all k epochs succeed), using the unbiased estimators: pass^k = mean over tasks of C(c,k)/C(n,k).

### 6.2 Cost (four meters, each reported for every arm)

- **Tokens:** input (uncached), cache-write, cache-read, output, and reasoning, each separately and as a total.
- **LLM calls:** model generations including subagent calls. Tool calls are counted separately.
- **Dollars:** list-price $ and **cache-adjusted $**, both **computed from token counts × the price table**, never from billed amounts.
  - Batch-mode runs have lower and unpredictable cache hit rates, so cache-hit rates and caching conclusions come only from non-batch runs (Study E).
- **Wall-clock:**
  - End-to-end time, and Inspect working time (excludes rate-limit/shared-resource waits).
  - Critical-path model time and summed model time. Realized parallelism = summed / critical-path.
- **Also recorded:**
  - Peak context tokens per agent.
  - A limit-hit flag, and which limit was hit.
  - A per-run cache nonce at the start of every system prompt. This prevents cross-run and cross-arm cache hits from making later arms look cheaper.

### 6.3 Determinism (k ≥ 5 epochs on the Study C subset)

Regimes:
- **(a) Minimum-variance settings where the provider honors them.** Record exactly which settings were honored.
  - The provider is OpenAI (decided 2026-09-29). Reasoning models may reject or ignore `temperature`; `seed` support varies by model. `readiness/probe_openai.py` records exactly which parameters each role's model honours (`cache/openai_probe.json`).
  - The controlled testbed is the self-hosted open-weight model. Verify whether your vLLM version offers a batch-invariant/deterministic mode.
- **(b) Production default settings.**

| ID | Metric | Definition | Class |
|---|---|---|---|
| D1 | Outcome stability | Fraction of samples where all k epochs agree on pass/fail | empirical |
| D2 | Answer agreement | Modal-answer share; normalized answer entropy H/log k | empirical |
| D3 | Trajectory similarity | Mean pairwise normalized edit similarity over (agent_id, tool_name, canonical-args hash) sequences; Jaccard of spawned subtask sets for orchestrators | empirical |
| D4 | Context stability | Mean pairwise Jaccard of context units per step given the same state; compiled-prompt hash equality | **descriptive** (the monolith is 1.0 by construction) |
| D5 | Perturbation robustness | Outcome flip rate across generator paraphrases | empirical |
| D6 | Resource predictability | CV of tokens, cache-adjusted $ and wall-clock across epochs (median and p90 over samples) | empirical |

### 6.4 Auditability / explainability

Every arm ends with the same **structured final output**: the answer plus cited evidence IDs (chunk IDs, KG node IDs, tool-call IDs, document IDs).

| ID | Metric | Definition | Class |
|---|---|---|---|
| A1 | Evidence precision / recall | Cited IDs are mapped to **world-spec fact IDs** via provenance and scored against the gold fact IDs. This makes the score granularity-neutral. | empirical |
| A2 | Citation validity | Fraction of cited IDs that actually appeared in that run's observed context or tool results | empirical |
| A3 | Versioned-source share | Fraction of each call's input tokens traceable to versioned, logged sources | **descriptive** (inter-agent text is unversioned by definition) |
| A4 | Fault localization accuracy | On failed fault-injected runs, a fixed automated auditor (same model and prompt for every arm) identifies the injected step/agent/node; report top-1 accuracy and auditor tokens. Human sub-study: ~30 traces per core arm. | empirical |
| A5 | Audit cost | Transcript tokens a reviewer must read to reconstruct the decision path (total and decision-relevant) | empirical |
| A6 | Rationale–action consistency | Scanner-labeled; validated against human labels (report Cohen's κ) | empirical |
| A7 | Replay reproducibility | Exact replay from logs with response caching; KG compile determinism | **descriptive** (infrastructure check; expect ≈ 100%) |
| A8 | Policy compliance | Violations per 100 actions against the permitted tool/permission set | empirical |

**H5 is tested only on empirical metrics.** Descriptive metrics are reported but carry no hypothesis test.

### 6.5 Error amplification (fault injection, Study D)

For each arm and each fault type (environment / message), report:
- **Unconditional lift:** P(wrong | fault) − P(wrong | clean).
- **Exposure-conditional lift:** the same quantity among runs that actually **read** the corrupted content. Arms that read less avoid faults by chance, so both numbers are needed.
- **Propagation depth:** the number of downstream calls whose inputs contain the fault.

Compare with Kim et al.'s **task-level** figures (≈ 1.1–1.3), not their trace-level 17.2× / 4.4×.

---

## 7. Protocol

### 7.1 Stages

1. **Micro-pilot** (humans, `micro_pilot_budget_usd`, after Phase-5 sign-off):
   - Scope: 2 families (F1, F7) × {S1, S5, M1} × 20 tasks × 2 epochs.
   - Purpose: measure B0, cost per rollout per arm, and variance components; check the generators; check the invariants.
2. **Pilot:** MVS arms on the `pilot` split, 20 tasks per condition. Used for power recalculation and for freezing the analysis code (blind analysis).
3. **MVS main** (Studies A, B, C on `test`).
4. **Tier B** studies, in the priority order set in DECISIONS.md.

### 7.2 Study modules

The Minimum Viable Study is Studies A + B + C on the primary model tier. Study D and the rest of Study E are Tier B.

| Study | Question | Arms | Families × levels | Tasks per condition | Epochs |
|---|---|---|---|---|---|
| **A** mechanism | RQ1, H1 | S1 (8-run pool → S8), S9, M1, M7 (+ M1s, n=50 on F1-high) | F1, F2 × 2 endpoints | 100 | 3 |
| **B** delivery/KG | RQ2, H2* | S1 (8-run pool → S8), S3s, S5, S5-P0, S7, M1, M1k, M2 | F3, F7-relational × 2 endpoints | 100 | 3 |
| **C** determinism | RQ4, H5 | S1, S5, S8(k=3), M1, M2, M7 | 30-task subset per MVS condition | — | +2 extra epochs (total 5) |
| **D** faults (Tier B) | H4 | S1, S5, S8, M1, M5, M6 | F2, F5 | 60 | 3 |
| **E** latency/cache | RQ3, H1d, H3 | S1, S5, M1s, M1, M2, M7, each ±C | F1-high, F7-high | 30 | 3 |
| **F** tiers (Tier B) | H6 | S1/S8, S5, M1, M2 | F1-high, F7-high | 100 | 3 |

**Study E latency regimes (both required):**
- **Capacity-limited:** self-hosted vLLM, fixed concurrency, prefix caching on/off.
- **Elastic:** a small API sample (API providers scale out, so parallel workers don't contend the way they do on one GPU), plus an analytical model: critical-path Σ(TTFT + output tokens / throughput) + tool time, calibrated against that API sample.
- Accuracy/cost runs may use batch mode. Latency and caching measurements never do.

**MVS rollout arithmetic** (WS7 must redo this with micro-pilot numbers):
- **Study A:** 400 tasks × (8 for the S1 pool + 3 × 3 arms) = 6,800, plus M1s 150. Total ≈ **6,950**.
- **Study B:** 400 tasks × (8 + 3 × 7) = **11,600**.
- **Study C:** ≈ 120 tasks × 6 arms × 2 extra epochs = **1,440**.
- **Total ≈ 20,000 rollouts.**
- Illustrative [estimate]: at $0.10–0.40 per rollout averaged over arms (short synthetic tasks, cheap/mid tier), the MVS costs **≈ $2–8k**.
- If the MVS does not fit `compute_budget_usd`, cut in this order, reporting the statistical power each cut costs:
  1. S5-P0 and S7 move to F7-high only.
  2. Drop M7 from Study A.
  3. Reduce to n = 70.

### 7.3 Controls

- **Tuning parity:** every arm gets the same automated prompt-optimization budget on `dev`.
  - An independent team member writes the M-arm prompts.
  - **The skeptic engineers S3s and the monolith, including caching, to be as strong as possible.**
- **One harness and one tool layer.** Arms differ only in switches.
- Provider, model version string, date and region are logged. The rerun policy is pre-defined.

### 7.4 Statistics

- **Primary analysis per family:** mixed-effects logistic regression
  `success ~ arm * log_knob + (1 | task) + (1 | task:arm)`.
  Each hypothesis maps to 1–2 **pre-registered contrasts** (§4.3). No omnibus three-way model is used for inference.
- **Superiority tests:** paired sign-flip permutation tests on task-level means, with Holm correction within each hypothesis family.
  - **Serial gatekeeping:** a multi-agent arm is tested against the S8 frontier only if it first beats S1.
- **Equivalence / non-inferiority** (H1c, H1d, H1e, H2b, H2c slope, H2): TOST or one-sided tests with the pre-registered margins. Non-significance is never reported as "no effect".
- **CIs:** task-clustered bootstrap (10k, BCa).
- **Power:** by simulation, at Holm-adjusted α.
  - v1's "155 tasks detects 10 pp" holds only at **unadjusted** α = 0.05. At α = 0.05/8 it needs about 250 single-epoch tasks.
  - With 3 epochs and the slope test pooling both endpoints (200 tasks per family), MDE ≈ 10 pp for arm main effects. Within-condition contrasts at n = 100 have MDE ≈ 13–15 pp.
  - WS8 recomputes power from pilot variance components and reports the achieved MDE per contrast.
- **Tier model (Study F):** `success ~ arm * tier + (1 | task)`.
- **Routing analysis (post hoc):** oracle-best arm per task; a cheap router trained on a-priori features, evaluated leave-one-family-out; report the fraction of the oracle-vs-best-fixed-arm gap it closes.
- **Blind analysis:** the analysis code is frozen on pilot data before test runs. Deviations are logged in DECISIONS.md.

---

## 8. Bias and validity controls

Each control must map to a concrete mechanism and a verification step in `10_BIAS_AND_VALIDITY_CONTROLS.md`.

**KG home-field advantage (new in v2)**
- The primary KG arm uses the **extracted** KG. The oracle KG is an upper bound only.
- Flat retrieval gets per-step refresh (S3s) and skeptic engineering.
- Evidence scoring works at fact-ID granularity.
- F5 is excluded from H2 headline claims.
- F7 includes an independent-policy control (Tier B).
- The definitional metrics are excluded from H5.

**Author bias**
- H2 and H5 thresholds are pre-registered before any test run.
- The skeptic role (above); an independent author for M-arm prompts.
- All logs and negative results are published.

**Design and scoring**
- Symmetric generators; per-condition reporting.
- Programmatic primary scoring. LLM judges only for secondary diagnostics, with human-agreement κ.
- Tuning parity; the test split is generated after the freeze.

**Contamination:** fresh seeded test split. Generators and the dev split are released only after results. Anchors that have public answers run with network egress blocked.

**Provider effects:** nondeterminism, outages and model-version drift are logged. 10 canary tasks are re-run daily.

**Harness artifacts**
- Document every Inspect default that changes information flow: handoff message filters, and subagent limit behavior (a subagent that hits a limit returns a message and the sample continues). Hold them constant or ablate them.
- The invariants in §4.4 are the detection mechanism.

**Budget artifacts:** caps rather than matched budgets, realized-cost comparisons, reported cap-hit rates, and the S8 frontier.

**Serving-regime artifacts:** two latency regimes (§7.2, Study E).

---

## 9. Inspect AI implementation requirements

**Verify every API against the current docs before documenting it.** Start from `https://inspect.aisi.org.uk/llms.txt` (full guide: `llms-full.txt`). Record the doc URL of each API in an "API verification table", or mark it `UNVERIFIED`.

**Architecture.** Implement `mech_agent(switches)` as one `@agent` factory that composes:
- `react()`
- `as_tool()` for isolated workers
- `handoff()` for M3
- `run()` — `asyncio.gather` gives concurrency (CONC=1) and sequential awaits give M1s
- compaction for S2

Every sample's metadata stores the switch vector, KB artifact hashes and the cache nonce.

**Capabilities to verify:**
- **Limits:** `token_limit` (including output-only and weighted types), `turn_limit`, `message_limit`, `time_limit`, `working_limit`, `cost_limit`; scoped limits (`token_limit()` context manager, `apply_limits`); per-agent `limits=` on `handoff`, `as_tool` and `run`.
- **Cost:** `set_model_cost(model, ModelCost(input, output, input_cache_write, input_cache_read))` or `--model-cost-config`.
- **Running:** `eval_set` (resumable sweeps), epochs and epoch reducers, caching, batch mode, early stopping, checkpointing, sandboxing, MCP tools.
- **Analysis:** log files, `samples_df()` and dataframes, events/transcripts for trajectory metrics, transcript scanners (check whether these live in Inspect itself or in the separate Inspect Scout package), Inspect Viz.
- **Epoch reducers vs post hoc:** decide which cross-epoch metrics (D1–D6) are epoch reducers and which are post-hoc log analysis, and document the choice.

**KG integration.** WS4 decides between two options and documents the trade-offs:
- In-process Python called before each `generate()`.
- The TypeScript implementation behind an MCP server.

Every compile call logs: graph version, input, state digest, selected node IDs, provenance fact IDs and prompt hash.

**Flat retrieval.** Every retrieval call logs: index version, query, chunk IDs and provenance fact IDs.

---

## 10. Workstreams

Dispatch these as subagents if your runtime supports it; otherwise run them sequentially.

**Before dispatching**, write `GLOSSARY.md` containing every ID exactly as defined here: arms, switches, families, levels, metrics, hypotheses, studies, splits. All subagents must use these IDs verbatim.

| WS | Title | Output file(s) | Depends on |
|---|---|---|---|
| WS1 | Literature verification & related work | `01_BACKGROUND_AND_RELATED_WORK.md` (verified-claims table with primary URLs; resolve every [unchecked]) | — |
| WS2 | Eval set & generator spec | `04_EVAL_SET_SPEC.md` (world spec schema, corpus rendering, redundancy/paraphrase controls, each family generator, gold fact IDs, fault hooks, splits, validation, F0 anchor adapters) | GLOSSARY |
| WS3 | Inspect project architecture | `06_INSPECT_PROJECT_ARCHITECTURE.md` (repo layout, `mech_agent` switch implementation, tasks, scorers, limits, cost config, eval_set sweep, logging schema, API verification table) | GLOSSARY |
| WS4 | KG integration & KB artifacts | `07_KG_INTEGRATION_SPEC.md` (extraction via `kg_ingestion_pipeline`, oracle-KG builder, provenance, extraction precision/recall report, compile API, versioning, logging, chunking for S3s, random-node sampler for S7) | WS2 |
| WS5 | Metrics & scorers spec | `05_METRICS_SPEC.md` (formula, log data source, implementation location, validation, empirical-vs-descriptive class) | WS3 |
| WS6 | Arm specifications | `03_ARM_REGISTRY.md` (per arm: switch vector, prompt policy, tools, control-flow pseudocode, fixed structural parameters, invariants) | WS3, WS4 |
| WS7 | Experiment matrix & budget | `08_EXPERIMENT_MATRIX_AND_BUDGET.md` (Studies A–F; MVS definition; rollout counts; cost model with price placeholders; ordered cut list with power cost per cut) | WS2, WS5, WS6 |
| WS8 | Hypotheses & analysis plan | `02_RESEARCH_QUESTIONS_AND_HYPOTHESES.md`, `09_ANALYSIS_PLAN.md`, `PREREGISTRATION.md` (contrasts, margins, tests, gatekeeping, power simulation, planned figures) | WS5, WS7 |
| WS9 | Validity controls | `10_BIAS_AND_VALIDITY_CONTROLS.md` (each control → mechanism → verification step) | all above |
| WS10 | Runbooks & timeline | `11_RUNBOOKS.md` (env setup, generator build, micro-pilot, pilot, MVS, latency sub-study, reruns, release), `12_TIMELINE_AND_MILESTONES.md`, `13_RISK_REGISTER.md` | all above |
| WS11 | Presentation & release plan | `14_PRESENTATION_AND_RELEASE_PLAN.md` (figures, talk outline, paper outline, open-release checklist, inspect_evals contribution path) | WS8 |

**Length discipline:** prefer tables; each doc should stay ≤ ~8 pages, with details in appendices.

**Subagent brief template** (use for every dispatch):
```
Objective: <one sentence>
Inputs: ORCHESTRATOR_BRIEF sections <x>, GLOSSARY.md, <prior outputs>
Output: <file path>, required sections: <list>
Acceptance criteria: <from §12>
Constraints: use glossary IDs verbatim; verify APIs/claims with URLs or mark UNVERIFIED;
  no invented numbers; placeholders for prices/models; flag open decisions;
  every mechanism claim cites a single-switch contrast (§4.3) or is labelled a bundle contrast.
Return: ≤200-word summary, list of open decisions, list of UNVERIFIED items.
```

---

## 11. Orchestration protocol

1. **Phase 0 (you):** read this brief; fetch the Inspect docs index; write `GLOSSARY.md` and a skeleton `00_INDEX.md`.
2. **Phase 1 (parallel):** WS1, WS2, WS3.
3. **Phase 2:** WS4, WS5, WS6 (in parallel where dependencies allow).
4. **Phase 3:** WS7, then WS8.
5. **Phase 4:** WS9, WS10, WS11.
6. **Phase 5 (you, consistency review):**
   - **Traceability matrix** in `00_INDEX.md`: hypothesis → contrast (§4.3) → arms → families/conditions → metrics → test → margin → planned figure. Any hypothesis without a complete row is a defect.
   - **ID and citation check:** every ID matches GLOSSARY; every Inspect API has a doc URL or `UNVERIFIED`; every literature number has a primary URL or `UNVERIFIED`.
   - **Budget check:** the MVS fits `compute_budget_usd` with ≥ 20% contingency. If not, apply the §7.2 cut order and state the MDE after each cut.
   - **Invariant check:** each invariant in §4.4 has a named check in the runbook.
   - **Open decisions:** consolidate them in `DECISIONS.md` (options, trade-offs, recommended default).
7. If `scaffold_code: true`: create a minimal package skeleton matching `06_INSPECT_PROJECT_ARCHITECTURE.md` (stub `mech_agent`, tasks and scorers; no paid runs; one dry run with a mock model if available).

---

## 12. Acceptance criteria (per deliverable)

- **Executable without this conversation:** a competent engineer can run every step. Each step has inputs, outputs, owner role, done-criteria and an effort estimate.
- **No vague metrics:** each has a formula, data source, implementation location, and a class (empirical or descriptive).
- **Named controls:** every design choice that could bias results names its control. KG home-field controls are explicit (§8).
- **Falsifiable hypotheses:** each has a falsification condition and margin written before data exists.
- **Attribution discipline:** every mechanism claim maps to a single-switch contrast or is labelled a bundle contrast.
- **Budget fit:** the MVS fits the budget. Tier B and C items are clearly separated.
- **Honest threats to validity** in each major doc: synthetic-task realism, extraction realism, provider nondeterminism, harness artifacts, serving-regime artifacts, author bias, model-generation staleness.
- **Style:** concise and technical, no marketing language.

---

## 13. Final output format (your last message)

1. File tree of everything written under `repo_root`.
2. A 10-line executive summary of the study design.
3. **Blocking inputs** (every §0 field still `TODO-BLOCKING`, and the sections marked `PROVISIONAL` because of it). Then the top 5 other open decisions from `DECISIONS.md` that need human input.
4. The list of `UNVERIFIED` items and the risk each poses.
5. Estimated micro-pilot, pilot, MVS and Tier B costs, with assumptions and the MDE at each budget level.

---

## Appendix: changes from v1 (for humans; the agent may ignore this)

1. **Arms → mechanism switches.**
   - All arms are now configurations of one `mech_agent(switches)`.
   - Added S9 (plan-then-execute in one context), M1s (serial M1), M1k (M1 with KG delivery, no roles) and M7-R0, so each mechanism has a single-switch contrast.
   - v1's adjacent arms changed several mechanisms at once: M1 vs S1 changed decomposition, isolation, concurrency and aggregation together, and M2 vs M1 changed delivery and specialization together.
2. **"Parallelism" split into ensembling (affects accuracy) and concurrency (affects latency only).** H1 is split into H1a–H1e.
3. **KG home-field controls.** The KG is built by the real ingestion pipeline from rendered documents (the oracle KG, S5o, is an upper bound only); flat retrieval per step (S3s) is engineered by a skeptic; evidence is scored at fact-ID granularity; F5 is excluded from headline claims; F7 gets a relational vs independent policy knob.
4. **Definitional metrics (D4, A3, A7) reclassified as descriptive** and removed from H5's tests.
5. **Budgets are caps with realized-cost comparison against the S8 frontier**, replacing 1× / 4× B0 matching, which a single agent that stops early cannot fill.
6. **Scale.** v1's full matrix was ≈ 2.9M rollouts (20 arms × 21 conditions × 155 tasks × 3 budgets × 5 models × 3 epochs), orders of magnitude over a $5k budget. v2 defines a ≈ 20k-rollout MVS (Studies A–C, one model, knob endpoints, slope tests) plus Tier B/C.
7. **Statistics.** Pre-registered contrasts instead of an omnibus three-way model; TOST for "≈ 0" claims; absolute margins instead of the "≥ 90% of accuracy" ratio; the power statement corrected for Holm.
8. **Fault injection.** Content-level (not step index), environment vs message faults, exposure-conditional lift, compared with Kim et al.'s *task-level* figures; faults never injected by editing assistant history.
9. **Latency** measured in two serving regimes (capacity-limited self-hosted and elastic API) plus an analytical critical-path model; batch-mode runs excluded from caching conclusions.
10. **F4 and F7 are defined against a nominal window W.** F7's largest KB fits in 0.8 W.
11. **F0 anchors named** (τ³-bench banking_knowledge first), with a 150-item minimum for inference.
12. **A micro-pilot budget** (human-run, after sign-off) grounds B0, costs and variances before the main budget is committed.
13. **Background corrections:** Kim et al.'s 17.2× / 4.4× are trace-level (task-level ≈ 1.1–1.3, not significant as a predictor); its matched-compute claim is internally inconsistent; the blog quotes v1 numbers.
