# Readiness audit (2026-10-03)

**Scope:** the whole program (the APG vs LightRAG gate, the main study and Study G) at commit `31ccef6`.

**Method:**
- Three rounds of independent, read-only review, each proving its findings with offline reproductions:
  - the program audit (2026-10-01);
  - `RELIABILITY_REVIEW.md` (2026-10-02);
  - a second review of the newest code (2026-10-03).
- Every finding was fixed, merged and given a regression test.

**Offline state:**
- 513 tests pass. Two opt-in tests also pass when enabled with `APE_TEST_BGE=1`; they use the real bge model.
- `run_gate all --offline` runs all nine phases, and `readiness/smoke.py --dry` passes.
- The cost model prints **$4,762 conservative / $3,239 expected** (D-028 added $6).
- **Nothing has run live.**

## 1. Bottom line

| Study | Readiness | What is true now |
|---|---|---|
| **Gate** | **Ready for the live smoke** | Engineering is complete and has been reviewed three times. What remains is your input (§3) and the first live checks, which the smoke run performs and the gate then requires. |
| **Main study** | **≈ 25%** | The families F1, F2, F3 and F7 exist, and so do the single-agent arms. Still missing: every multi-agent arm, the analysis, the runner and the pre-registration (§4). |
| **Study G** | **≈ 30%** | F8 exists, with CM0, O-state, probes and scoring. Still missing: the six context-management arms, the topology arms in sessions, the analysis, the runner and the pre-registration (§5). |

**Remaining risks to the gate:**
1. **PC1 (the LightRAG anchor).**
   - A faithful setup passes about 40% of the time with gpt-4o-mini, after the D-025 fidelity work (it was 10–15%). The remaining gap comes from LightRAG 1.5.7 vs 1.2.5.
   - If PC1 fails: diagnose it (judge parse rates are now recorded), then either accept it with a written diagnosis, which puts a caveat in the report, or stop.
2. **Budget headroom:** $238, or 4.8%. The live smoke re-projects the program from its real calls, and warns if the total passes $5,000 or the gate passes $700.
3. **Power:**
   - P(unqualified GO) at no true difference is 0.84 at 16 worlds per cell, 0.885 with the extension.
   - At 12 worlds it is only 0.68. That is the D-017 Sol-builder path, which costs +$1,074 and needs approval.
4. **Only a live run can answer** these questions:
   - Is reasoning carried across per-step calls?
   - How well does Luna extract at F7-1000?
   - Is gpt-4o-mini still available?
   - Are the snapshots dated IDs or echoed aliases?

   The smoke run measures each one before the gate is allowed to start.

## 2. What changed since the 2026-10-02 audit

| Round | Outcome |
|---|---|
| **Gate hardening** (A–D) | The probe now runs on the real Responses-API path, and model snapshots can be pinned. A program-wide spend registry and guard counts killed runs' flushed spend. Each sample has a cost cap, and error tolerance is counted, not a proportion. The embedding cache is float32, about 5× smaller. D-017 build quality is measured on every dev cell. The H5 spot-check was regenerated on the gate cells. Backups run after each phase. |
| **Reliability review** (R1–R5) | The statistics use a small-sample t-interval with Holm across delivery modes, an honest α split and a runnable extension. The agent loop was hardened against real-model misbehaviour, and reasoning now carries over in per-step arms. KG builds retry and repair, and evidence is credited only for delivered text. The freeze guards code and the current pilot. Each run gets its own test-seed block. The S7 placebo matches APG*'s schedule. New smoke checks cover F8 sessions, real F7-1000 extraction and reasoning carry-over. |
| **Pre-run mitigations** (M, P, Q) | Dry and live smoke runs are separated. The gate requires a fresh passing live smoke of the same code. The smoke recalibrates costs. Tuning requires owner sign-off. Preflight warns about backups and disk space. PC1 uses scorers matched to how each published number was produced, local bge, 1.2.5-like context caps and a restated rule (D-025). The pilot estimates σ from 8 worlds and gives a two-sided recommendation (D-026). |
| **Second review** (V1, V2, W) | **Four blockers fixed:** the first live smoke's orchestrator check was starved of budget; switching to the fallback builder mid-run broke the run; the bge embedder crashed LightRAG; an empty anchor index could be accepted. **Also:** the smoke is settled once per run on exact code; the anchor re-scores after a harness fix; frozen outputs live per run; design knobs are frozen; LightRAG keyword handling now matches stock LightRAG, with no extra retrieval attempt; provenance recall and precision were fixed (the false-credit rate on merged LightRAG descriptions dropped from about 11% to 0); the pilot uses the least-favourable σ corner; PC5 fails only on evidence (a true 1.5% error rate no longer fails half the time); a second extension is refused (D-027). |
| **Decision report** | Harness health is now reported: APG classify fallbacks, LightRAG keyword fallbacks and empty retrievals, search_kb errors and truncation, each with a warning above 5%. Matched-budget calibration status is reported too. |

## 3. Gate: from here to a verdict

**Needs you, before the first paid call:**
1. **O-1:** a new OpenAI key in `.env`, with a hard limit of about $300, ideally in a dedicated project so rate limits aren't shared.
2. **`APE_BACKUP_DIR`:** a folder outside the repo, such as an external drive or a synced folder.

**Needs you, before tuning and the freeze:**
1. **O-3:** names for the APG owner, the skeptic and the analyst. The skeptic reviews the LightRAG and S3s candidates; then fill in `owners` and `signed_off` in `config/tuning_grid.yaml`. Live `tune` refuses until that is done.
2. **H5:** review part A of `readiness/spotcheck.md` (30–45 min).
3. **D-017**, only if Luna fails the build check: Sol builds (+$1,074, with 12 worlds per cell giving power 0.68), or revise the authoring prompt on dev first.
4. **Freeze sign-off:** fill in the `[PILOT: …]` and `[USER: …]` items in GATE_PREREG.md from `pilot.json`, commit, then run the freeze.

**Live sequence:**
```
uv run --locked python readiness/probe_openai.py --profile gate   # about $0.02; review, then --pin; commit PROVENANCE.md
uv run --locked python readiness/smoke.py                         # cap $4 by default; projected about $2.95
uv run --locked python -m ape.run_gate anchor --run-id gate-1     # PC1 early, about $11, while the skeptic reviews
uv run --locked python -m ape.run_gate all --run-id gate-1        # build-dev, tune (once signed off), pilot; stops at the freeze
#   fill in GATE_PREREG.md, commit, then: run_gate freeze --run-id gate-1, then run_gate all --run-id gate-1 again
```

**What to read in the smoke report before going on:**
- The cost-model check: the re-projected total, with a warning if it passes $5,000.
- `extract_f7_1000`: Luna's coverage against 0.95, the first read on D-017.
- `perstep_reasoning`: reasoning items carried after a tool step.
- `f8_session`: the probe's schema validity.
- The anchor judge parse-failure rates.
- Effort verdicts and snapshot IDs from the probe.

## 4. Main study

**Ready:**
- the families F1, F2, F3 and F7, with programmatic scoring;
- the single-agent arms S1, S3s, S5 (APG), S6 and S7, plus LightRAG;
- the runner (`ape.runner`) with spend, retry and cost-limit safety;
- the cost model;
- per-run test-seed blocks, and `make_world(seed_base=…)` for study namespaces.

**Missing** (blocks the main study):

| Component | Effort | Notes |
|---|---|---|
| Multi-agent arms: S9, M1/M1s, M1k/M2, M7 and the S8 frontier | L | No `as_tool`, handoff or council code exists yet. F1 subtasks are ready for them. |
| A study runner with its own seed range, freeze and `PREREGISTRATION.md` | M | Generalize `run_gate`'s phases by study, and give the main study its own test-seed range. |
| Analysis: GLMM (lme4, pymer4 or bambi), Holm, TOST, BCa, S8 interpolation per meter, pass^k, Kendall τ; power simulation | L | None exists. |
| Tuning grids for S1, S5 and the multi-agent arms | M | The brief requires equal tuning budgets. |
| Build cells for F7-100 worlds and F1-2 KG builds | S | Planned cells run on them, but nothing builds them. |

**Design issues to decide** (§6):
- ~~H6 is confounded~~: fixed by D-028 (Luna F7-100 cell).
- ~~H1d underpowered, K3 untestable, K4's clause untestable~~: H1d and K4 are descriptive and K3 is dropped (D-028).
- **F1's context pressure is moderate:** about 13K tokens of records at N = 32 against W = 128K. The micro-pilot should confirm that it stresses a single context.

## 5. Study G

**Ready:**
- F8 generation and scoring;
- the CM0 and O-state runners;
- forked state probes;
- per-item records for the degradation curve;
- failure labels;
- error tolerance by count for small tasks.

**Missing** (blocks Study G):

| Component | Effort |
|---|---|
| The CM arms: prune, trim, sum, todo, reset and native, plus S-CM* | M |
| The topology arms (M1, M2) inside sessions | L |
| The `cm` role (summarizer) | S |
| Session checkpoint and resume (a retried session restarts all 240 calls, about $30 on Astra) | M |
| `cm_stats` and a session-clustered power simulation | M–L |
| Probing Sol, Astra and native compaction | S |
| Pre-registration and runner | M |

**Measured, and design issues** (§6):
- The reference solver needs 2.3 generations per item against a prior of 6, so Study G is probably over-priced.
- ~~Short sessions don't overflow~~: fixed by D-028 (larger tool files; median crossing at 0.70 of the session).
- The G-H1 single-agent reference is still open.
- CM-native runs at one tier only.
- S1+KG and S-subiso are in the hypotheses but not in the plan.
- The capability anchor is near the ceiling.

## 6. Decisions for you

| # | Decision | My recommendation |
|---|---|---|
| 1 | Key (O-1) and backup destination | A dedicated project key with about a $300 limit now, raised per study. `APE_BACKUP_DIR` on an external or synced disk. |
| 2 | Names (O-3) and the skeptic's sign-off | Needed before `tune`. |
| 3 | Contingency at 4.8% against the brief's 20% | Accept for now with staged spending. Recalibrate after the smoke and the pilot, and apply cuts C4/C5 then if needed. |
| 4 | D-017 Sol builder (+$1,074) | Decide only if Luna fails. First try revising the authoring prompt on dev; Sol would also leave the gate underpowered at 12 worlds. |
| 5 | If PC1 fails | Diagnose first. Accept only with a written diagnosis (the report carries a caveat); a high judge parse-failure rate makes PC1 "not evaluable", which can't be accepted. |
| 6 | H6 tier confound | ✅ Done (D-028): Luna F7-100 cell and the F7-100/F1-2 builds. |
| 7 | Main-study scope | ✅ K3 dropped; H1d and H2c descriptive (D-028). Still open: reduce H5 to D1, D2 and D6. |
| 8 | Study G overflow and G-H1 | ✅ Overflow fixed (D-028: 2,250- and 1,750-token tool files). Still open: G-H1's single-agent reference (BUILD_PLAN §4). |
| 9 | Anchors other than GraphRAG-Bench | ✅ Deferred to Tier B (D-028). |

## 7. Engineering plan

1. ~~Gate hardening, the reliability fixes and the second review~~: done. Next is the live sequence in §3.
2. **The build plan in `BUILD_PLAN.md`** (B1–B12), starting with the study runner (B1), the multi-agent arms (B2) and the session runner (B7).
3. **The multi-agent arms and their tuning grids** (L). Study G's topology block reuses them.
4. **The Study G harness:** the CM arms, S-CM*, the `cm` role and session checkpointing (M–L).
5. **Analysis and power for the main study and Study G, then the two pre-registrations** (L).
6. **A consistency pass on the briefs** once decisions 3–8 are made.

## 8. Live-run facts worth knowing

- **Wall clock:** about 270K agent and kg calls for the gate, so roughly 1–2 days at 16–32 concurrent calls. The whole program takes weeks, dominated by Study G on Sol and Astra.
- **Disk:**
  - logs, about 2 GB;
  - embedding cache, about 1–3 GB (float32);
  - indices, 1–2 GB;
  - the bge model, 1.34 GB, downloaded once on the first anchor run;
  - the backup copy, which mirrors all of the above.
- **Crash behaviour:** the gate resumes cleanly. That covers manifests, `eval_set` log reuse, idempotent builds, an atomic ledger and spend from killed runs. Freezes interrupted mid-write are recovered. The main study and Study G have no runner yet.
- **Reproducibility:**
  - **Pinned:** `uv.lock` (with `--locked`), the bge revision, model snapshots once `--pin` runs, and the per-run seed blocks.
  - **Gaps:**
    - APG is pinned through a local `file://` git source, which is not portable.
    - The GraphRAG-Bench data hash is recorded but never checked.
    - Reasoning models are non-deterministic, so epochs are the only control.
