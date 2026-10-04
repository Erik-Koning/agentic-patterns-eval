# Readiness audit (2026-10-04)

**Scope:** the whole program (the APG vs LightRAG gate, the main study and Study G) at commit `b41a518`, after the build in `BUILD_PLAN.md` (B1–B12) and its independent review (`BUILD_REVIEW.md`).

**Method:**
- Three rounds of independent, read-only review, each proving its findings with offline reproductions:
  - the program audit (2026-10-01);
  - `RELIABILITY_REVIEW.md` (2026-10-02);
  - a second review of the newest code (2026-10-03);
  - the independent review of the main-study and Study G build (2026-10-03, `BUILD_REVIEW.md`: 4 A and 12 B findings, all fixed).
- Every finding was fixed, merged and given a regression test.

**Offline state:**
- 987 tests pass (2 skipped). Two opt-in tests also pass when enabled with `APE_TEST_BGE=1`; they use the real bge model.
- `run_gate all --offline` runs all nine phases; `run_study all --offline` runs every phase and every arm of the main study (about 2 minutes) and of Study G (about 1 minute); `readiness/smoke.py --dry` passes, including the seven new arm-family checks.
- The cost model prints **$4,836 conservative / $3,249 expected** (gate $220, main $1,127, Study G $3,490; contingency $164).
- **Nothing has run live.**

## 1. Bottom line

| Study | Readiness | What is true now |
|---|---|---|
| **Gate** | **Ready for the live smoke** | Engineering is complete and has been reviewed four times. Since the build its freeze covers only its own slice of the shared config, and its analyze refuses post-freeze changes unless a deviation is recorded. What remains is your input (§3) and the first live checks. |
| **Main study** | **Built and reviewed; runnable once the gate has a verdict** | Every arm (S1, S3s, S5, S7, S9, M1, M1s, M1k, M2, M7, S8k3), the study runner, tuning grid, analysis, pre-registration and smoke checks exist and pass offline end to end. Live it needs the gate's verdict (its KG arm and S3s selection), the live smoke, role names and sign-offs, and its own pilot to fill the `[PILOT]` items (§4). |
| **Study G** | **Built and reviewed; runnable after the probe and smoke** | Every context-management and topology arm, mid-session resume, the runner, tuning grid, analysis, pre-registration and smoke checks exist and pass offline end to end. Live it needs the Sol/Astra probe (which writes the CM-native support records), the live smoke, role names and sign-offs, and your go-ahead after its micro-pilot (§5). |

**Remaining risks to the gate:**
1. **PC1 (the LightRAG anchor).**
   - A faithful setup passes about 40% of the time with gpt-4o-mini, after the D-025 fidelity work (it was 10–15%). The remaining gap comes from LightRAG 1.5.7 vs 1.2.5.
   - If PC1 fails: diagnose it (judge parse rates are now recorded), then either accept it with a written diagnosis, which puts a caveat in the report, or stop.
2. **Budget headroom:** $164, or 3.3%. The live smoke re-projects the program from its real calls; each study's guard stops a phase whose projection exceeds what is left of the program or its allocation (gate $300, main $1,225, Study G $3,600).
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

**Built and reviewed** (BUILD_PLAN B1–B6, B12; D-029–D-050):
- **Arms:** single-agent S1, S3s, S5 (the gate's KG arm), S7; multi-agent S9, M1, M1s, M1k, M2 (specialists on F3/F7 only), M7 (council), S8k3 (live self-consistency), all on one loop with the single agent's rules, exact per-agent accounting, and provider errors retried as for S1. S9 → M1s → M1 → M1k → M2 share one prompt variant, so each step changes one switch.
- **Runner:** `python -m ape.run_study <phase|all> --study main --run-id <id> --gate-run-id <gate run>`: its own seed namespaces and test lock, the KG arm and S3s selection copied from the gate run, token caps from the micro-pilot with a pilot gate on Clopper–Pearson bounds (D-045, D-050), per-arm selection env groups, a per-run cache nonce, a runaway guard, smoke currency per paid phase, a freeze of its own config slice, and a spend ledger that counts retried attempts.
- **Analysis and pre-registration:** world-clustered sign-flip, Holm and serial gatekeeping, TOST/NI with the user's margins (D-033), the S8 frontier with the matched cost's uncertainty, per-world influence; `PREREGISTRATION_MAIN.md` with its generated table. Confirmatory: M1, M2 (gated), M3 (with a documented residual, D-048), M5, K1, K2, K2-NI; the rest descriptive.

**Before a live run:** the gate's live verdict; the live smoke (gate + main checks); names and sign-offs (M-arm prompt author, skeptic, analyst; `config/tuning_grid_main.yaml`); then `run_study all`, which stops at the freeze for the pre-registration to be filled and frozen.

**Known limits** (stated in the pre-registration): superiority tests have an MDE of about 15 pp; K2-NI's power is 0.79; the sign-flip assumes symmetric world contributions (sensitivities reported); realised cost excludes errored attempts' usage (shown per arm from the ledger).

## 5. Study G

**Built and reviewed** (BUILD_PLAN B7–B11, B12; D-030–D-050):
- **Arms:** CM0 (= S1), O-state, CM-prune, CM-trim, CM-sum, CM-todo, CM-reset, CM-native (gated by the probe's support record), S-CM* (a tuned stack), and M1/M2 as an orchestrator with isolated workers; one ContextPolicy layer with management calls metered as `cm`, the window enforced on every input, probes never appended, and mid-session checkpoint/resume.
- **Runner:** `python -m ape.run_study <phase|all> --study study_g --run-id <id>`, with its own seeds and test lock, a micro-pilot that also pilots the topology arms and gates session errors, limits and overflow (D-050), and the safeguards of §4.
- **Analysis and pre-registration:** session-clustered t at a calibrated level (0.013) with a minimum number of sessions per planned point (D-049); `PREREGISTRATION_G.md` with its generated tables. Confirmatory: G-H2a (power 0.94), G-H3 (G-H3a 0.83–0.99; G-H3b 0.56); G-H1 and G-H2b are estimates with intervals (D-033), with tier order as a sensitivity (D-043).

**Before a live run:** the probe with `--study study_g` (honoured parameters and API mode per model; the CM-native support records for Luna and Sol); the live smoke (Study G checks); names and sign-offs (CM-arm owner, skeptic, analyst, independent prompt author; `config/tuning_grid_study_g.yaml`); your review of the micro-pilot and go-ahead at the planned sizes.

## 6. Decisions for you

| # | Decision | My recommendation |
|---|---|---|
| 1 | Key (O-1) and backup destination | A dedicated project key with about a $300 limit now, raised per study. `APE_BACKUP_DIR` on an external or synced disk. |
| 2 | Names and sign-offs | Gate: APG owner and the skeptic (O-3). Main: the M-arm prompt author (not the skeptic), the skeptic (signs off S1) and the analyst. Study G: the CM-arm owner, the skeptic (owns S-CM*), the analyst and an independent prompt author. They go in the three tuning grids and the pre-registrations' `[USER]` items. |
| 3 | H5 spot-check | A human reviews part A of `readiness/spotcheck.md` (30–45 min) before the pilot. |
| 4 | Contingency at 3.3% against the brief's 20% | Accept with staged spending; recalibrate after the gate pilot and each micro-pilot (`ape.budget calibrate`), which will likely lower Study G (its priors look conservative). |
| 5 | Study G sizing at the micro-pilot review | G-H3b has power 0.56; 24 Luna topology sessions per point would give about 0.67 for about +$51. Decide with the micro-pilot's numbers. |
| 6 | D-017 Sol builder (+$1,074) | Only if Luna fails the build-quality check; it would not fit the budget without cuts, so try revising the authoring prompt on dev first. |
| 7 | If PC1 fails | Diagnose first. Accept only with a written diagnosis (the report carries a caveat); a high judge parse-failure rate makes PC1 "not evaluable", which can't be accepted. |

Decided during the build (DECISIONS D-028–D-050): the power trade-offs (D-033), M2 off F1 (D-034), the Study G design questions (D-043), M3's residual (D-048), and the review's fixes (D-047, D-049, D-050).

## 7. Engineering plan

1. ~~Gate hardening, the reliability fixes and the second review~~: done.
2. ~~The build plan (B1–B12) and its independent review~~: done (BUILD_PLAN §6, BUILD_REVIEW.md).
3. **Next is the live sequence:** the key, the probe (`readiness/probe_openai.py`, then `--study study_g`), `--pin`, the live smoke, `run_gate all`; after the gate's verdict, `run_study all --study main`; Study G can start after the probe and smoke.
4. **Recalibrate** the cost model after the gate pilot and each micro-pilot, and revisit the contingency.

## 8. Live-run facts worth knowing

- **Wall clock:** about 270K agent and kg calls for the gate, so roughly 1–2 days at 16–32 concurrent calls. The whole program takes weeks, dominated by Study G on Sol and Astra.
- **Disk:**
  - logs, about 2 GB;
  - embedding cache, about 1–3 GB (float32);
  - indices, 1–2 GB;
  - the bge model, 1.34 GB, downloaded once on the first anchor run;
  - the backup copy, which mirrors all of the above.
- **Crash behaviour:** the gate resumes cleanly. That covers manifests, `eval_set` log reuse, idempotent builds, an atomic ledger and spend from killed runs. Freezes interrupted mid-write are recovered. The main study and Study G resume the same way, and a crashed F8 session resumes after its last completed case.
- **Reproducibility:**
  - **Pinned:** `uv.lock` (with `--locked`), the bge revision, model snapshots once `--pin` runs, and the per-run seed blocks.
  - **Gaps:**
    - APG is pinned through a local `file://` git source, which is not portable.
    - The GraphRAG-Bench data hash is recorded but never checked.
    - Reasoning models are non-deterministic, so epochs are the only control.
