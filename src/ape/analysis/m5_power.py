"""Power and type I error of the M5 add-on study's tests (D-053; `m5_hypotheses`), simulated through the main study's
model and analysis: `main_power.simulate` draws a study, `main_stats.evaluate_family` tests it (the function
`m5_report` runs).

    uv run python -m ape.analysis.m5_power [--reps 2000] [--null-reps 10000] [--families L1] [--design] [--sensitivity] [--out m5_power.json]

**Model.** `main_power`'s: logit P(success) = x + u_w (σ_w 0.5) + g_{w,arm} (σ_g 0.3) + v_t (σ_u 1.5) + e_{t,arm} (σ_v 0.5),
F1 and F2 worlds of one seed sharing u (one registry; the analysis clusters them as one), the gate's σ priors. The
ledger arm and its concurrent control share the tasks (v_t) and worlds (u_w), as they do: the m5 study runs both on the
main study's test worlds, paired on tasks.

**Sizes** (`m5_sizes`): the m5 study's test cells in config/run_plan.yaml (`studies.m5`), as the runner runs them:
100 tasks -> 9 worlds × 12 = 108 per cell, 3 epochs per arm; without a `studies.m5` block, D-053's design (the same).

**Scenarios.** Controls at the cell's S1 baseline (`main_power.BASELINES`) + 0.05 (M1) or + 0.03 (M2), as the main
study's scenarios put them; the ledger arm at control + d, capped at 0.98 (a scenario's Δ is the realised equal-weight
mean). Type I at d = 0 with `null_reps` studies; power with `reps`.

**Outputs.** `power_table` (the chosen families' nulls and effects; seed 20261053) gives the `power` strings of
m5_hypotheses' members, the table's column; `design_check` (seed 20261153) compares the candidate designs the
pre-registration chose between (which members are confirmatory, one family or two) and gives PREREGISTRATION_M5.md
§2.4's design table; `sensitivity` (seed 20261653) the chosen members at a larger σ_g or σ_w (§2.6). The pre-registered
numbers are 2,000 studies per effect and 10,000 per null.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .main_hypotheses import ALPHA, PRIMARY_TIER, STUDY_A_CELLS, STUDY_B_CELLS, Hypothesis, Member
from .main_load import tier_of
from .main_power import BASELINES, MODELS, RUN_PLAN, TASKS_PER_WORLD, TEST_WORLDS, Scenario, Settings, Sigmas, mc_se, simulate_family
from .m5_hypotheses import F1_CELLS, F2_CELLS, HYPOTHESES, L1_TERMS, L2_TERMS, confirmatory, get

CONTROL_OFFSET = {"M1": 0.05, "M2": 0.03}  # the controls above S1's baseline, as main_power's scenarios put them
CEILING = 0.98  # a ledger arm's success is capped here (a cell near the ceiling has little room)
DEFAULT_TEST = (  # D-053's design, used when config/run_plan.yaml has no `studies.m5`
    {"id": "m5.test.a", "arms": ["M5", "M1"], "cells": list(STUDY_A_CELLS), "n_tasks": 100, "epochs": 3},
    {"id": "m5.test.b", "arms": ["M5-spec", "M2"], "cells": list(STUDY_B_CELLS), "n_tasks": 100, "epochs": 3},
)
SEED = 20261053  # power_table
DESIGN_SEED = 20261153  # design_check


# --------------------------------------------------------------------------- planned sizes


def m5_sizes(path: Path = RUN_PLAN, models: Path = MODELS) -> dict[tuple[str, str, str], dict]:
    """(tier, cell, arm) -> {n_tasks, worlds, epochs, plan_cell} for the m5 study's test cells, as the runner runs
    them (⌈n / 12⌉ whole worlds of 12 tasks). Without a `studies.m5` block in the plan: D-053's design (DEFAULT_TEST)."""
    plan = yaml.safe_load(Path(path).read_text()) or {}
    profiles = (yaml.safe_load(Path(models).read_text()) or {}).get("profiles", {}) if Path(models).is_file() else {}
    study = (plan.get("studies") or {}).get("m5")
    cells = ((study or {}).get("phases") or {}).get("test") or list(DEFAULT_TEST)
    default = (study or {}).get("profile", "main_luna")
    out: dict = {}
    for cell in cells:
        prof = cell.get("profile", default)
        tier = tier_of(((profiles.get(prof) or {}).get("agent") or {}).get("model")) if profiles else PRIMARY_TIER
        worlds = math.ceil(int(cell["n_tasks"]) / TASKS_PER_WORLD)
        for arm in cell["arms"]:
            for c in cell["cells"]:
                out[(tier, c, arm)] = {"n_tasks": worlds * TASKS_PER_WORLD, "worlds": worlds, "epochs": int(cell["epochs"]), "plan_cell": cell["id"]}
    return out


# --------------------------------------------------------------------------- scenarios


def spec_for(arm: str, effects: dict[str, float], baselines: dict = BASELINES, tier: str = PRIMARY_TIER) -> dict:
    """{(tier, cell): {control: b + offset, arm: min(b + offset + d, CEILING)}} for each cell's effect d."""
    control = "M1" if arm == "M5" else "M2"
    out = {}
    for c, d in effects.items():
        base = baselines[c] + CONTROL_OFFSET[control]
        out[(tier, c)] = {control: base, arm: min(base + d, CEILING)}
    return out


def realised(spec: dict, cells: tuple[str, ...]) -> float:
    """The equal-weight mean of (ledger arm − control) over `cells` in a spec (after the ceiling cap)."""
    ds = []
    for (_, c), arms in spec.items():
        if c in cells:
            control = next(a for a in arms if a in ("M1", "M2"))
            ledger = next(a for a in arms if a != control)
            ds.append(arms[ledger] - arms[control])
    return sum(ds) / len(ds) if ds else float("nan")


def _with(*specs: dict) -> dict:
    out: dict = {}
    for s in specs:
        for k, v in s.items():
            out.setdefault(k, {}).update(v)
    return out


def scenarios(hyp_id: str, baselines: dict = BASELINES) -> list[tuple[Scenario, str]]:
    """Each confirmatory family's null (type I, kind "null") and effects (power): (scenario, kind of replicate count)."""
    if hyp_id == "L1":  # Holm over L1.F2 and L1.F1 (both on the M1 team, Study A's cells)

        def a(f1: float, f2: float | dict) -> dict:
            return spec_for("M5", dict.fromkeys(F1_CELLS, f1) | (f2 if isinstance(f2, dict) else dict.fromkeys(F2_CELLS, f2)), baselines)

        out = [
            (Scenario("M5 − M1 = 0 on F1 and F2", a(0.0, 0.0), ("L1.F2", "L1.F1")), "null"),
            (Scenario("M5 − M1 = +0.10 on F1, 0 on F2", a(0.10, 0.0), ("L1.F2",)), "null"),
            (Scenario("M5 − M1 = 0 on F1, +0.10 on F2", a(0.0, 0.10), ("L1.F1",)), "null"),
        ]
        for d in (0.08, 0.12):
            out.append((Scenario(f"M5 − M1 = 0 on F1, {d:+.2f} on F2", a(0.0, d), ("L1.F1",)), "power"))
        for d in (0.05, 0.08, 0.10, 0.12, 0.15):
            out.append((Scenario(f"M5 − M1 = {d:+.2f} on F1 and F2", a(d, d)), "power"))
        for d in (0.15, 0.20):
            out.append((Scenario(f"M5 − M1 = 0 on F1, {d:+.2f} on F2-10 only (F2 pooled {d / 2:+.3f})", a(0.0, {"F2-2": 0.0, "F2-10": d}), ("L1.F1",)), "power"))
        return out
    if hyp_id == "L2":
        out = [(Scenario("M5-spec − M2 = 0 on Study B", spec_for("M5-spec", dict.fromkeys(STUDY_B_CELLS, 0.0), baselines), ("L2.pooled",)), "null")]
        for d in (0.03, 0.04, 0.05, 0.06, 0.08):
            out.append((Scenario(f"M5-spec − M2 = {d:+.2f} in Study B's four cells", spec_for("M5-spec", dict.fromkeys(STUDY_B_CELLS, d), baselines)), "power"))
        return out
    raise KeyError(f"no scenarios for {hyp_id}")


def _row(hyp: Hypothesis, sc: Scenario, reps: int, seed: int, sigmas: Sigmas, settings: Settings, sizes: dict) -> dict:
    res = simulate_family(hyp, sc, reps, seed, sigmas, settings, sizes)
    row = {"family": hyp.id, "scenario": sc.name, "kind": sc.kind, "nulls": list(sc.nulls), "seed": seed} | res
    row["delta"] = {m.id: realised(sc.spec, m.cells) for m in hyp.members}
    if sc.nulls:
        row["exceeds_alpha"] = res["false_claims"] > hyp.alpha + 2 * mc_se(hyp.alpha, reps)
    return row


def power_table(reps: int = 1000, null_reps: int = 4000, seed: int = SEED, sigmas: Sigmas = Sigmas(), settings: Settings = Settings(), baselines: dict = BASELINES, sizes: dict | None = None, hypotheses=HYPOTHESES, families: tuple[str, ...] | None = None) -> dict:
    """Every confirmatory family's scenarios at the planned sizes (`families`: these only); scenario j of family i
    (its index among the confirmatory families) is seeded seed + 1000 i + j."""
    sizes = m5_sizes() if sizes is None else sizes
    out = {"reps": reps, "null_reps": null_reps, "seed": seed, "sigmas": asdict(sigmas), "settings": asdict(settings), "baselines": baselines, "families": {}}
    for i, hyp in enumerate(confirmatory(hypotheses)):
        if families and hyp.id not in families:
            continue
        rows = [_row(hyp, sc, null_reps if kind == "null" else reps, seed + 1000 * i + j, sigmas, settings, sizes) for j, (sc, kind) in enumerate(scenarios(hyp.id, baselines))]
        out["families"][hyp.id] = {"alpha": hyp.alpha, "procedure": hyp.procedure, "scenarios": rows}
    return out


# --------------------------------------------------------------------------- the design choice


def _family(fid: str, *members: Member) -> Hypothesis:
    return Hypothesis(fid, "design check", fid, "confirmatory", tuple(members), alpha=ALPHA)


L1_F2 = Member("L1.F2", L1_TERMS, F2_CELLS, "superiority")
L1_F1 = Member("L1.F1", L1_TERMS, F1_CELLS, "superiority")
L1_F2_10 = Member("L1.F2-10", L1_TERMS, ("F2-10",), "superiority")
L2_POOLED = Member("L2.pooled", L2_TERMS, STUDY_B_CELLS, "superiority")


def design_cases(baselines: dict = BASELINES) -> list[tuple[str, Hypothesis, Scenario, str]]:
    """(design, family, scenario, null|power): the candidate designs at their nulls and effects."""
    a = lambda eff: spec_for("M5", eff, baselines)  # noqa: E731
    b = lambda d: spec_for("M5-spec", dict.fromkeys(STUDY_B_CELLS, d), baselines)  # noqa: E731
    f2, f2f1, f2_10 = _family("L1", L1_F2), _family("L1", L1_F2, L1_F1), _family("L1", L1_F2_10)
    one = _family("L", L1_F2, L1_F1, L2_POOLED)
    l2 = _family("L2", L2_POOLED)
    out = []
    # A: L1 = F2 pooled alone
    out.append(("A: L1.F2 alone", f2, Scenario("Δ = 0 on F2", a(dict.fromkeys(F2_CELLS, 0.0)), ("L1.F2",)), "null"))
    for d in (0.06, 0.08, 0.10, 0.12):
        out.append(("A: L1.F2 alone", f2, Scenario(f"Δ = {d:+.2f} on F2-2 and F2-10", a(dict.fromkeys(F2_CELLS, d))), "power"))
    for d in (0.15, 0.20):
        out.append(("A: L1.F2 alone", f2, Scenario(f"Δ = {d:+.2f} on F2-10 only", a({"F2-2": 0.0, "F2-10": d})), "power"))
    # B: F2-10 alone (the chain-heavy cell)
    out.append(("B: L1.F2-10 alone", f2_10, Scenario("Δ = 0 on F2-10", a({"F2-10": 0.0}), ("L1.F2-10",)), "null"))
    for d in (0.08, 0.10, 0.12, 0.15, 0.20):
        out.append(("B: L1.F2-10 alone", f2_10, Scenario(f"Δ = {d:+.2f} on F2-10", a({"F2-10": d})), "power"))
    # C: L1 = F2 pooled + F1 pooled, Holm of 2
    out.append(("C: L1.F2 + L1.F1 (Holm of 2)", f2f1, Scenario("Δ = 0 on F1 and F2", a(dict.fromkeys(STUDY_A_CELLS, 0.0)), ("L1.F2", "L1.F1")), "null"))
    out.append(("C: L1.F2 + L1.F1 (Holm of 2)", f2f1, Scenario("Δ = +0.10 on F1, 0 on F2", a(dict.fromkeys(F1_CELLS, 0.10) | dict.fromkeys(F2_CELLS, 0.0)), ("L1.F2",)), "null"))
    for d in (0.08, 0.10, 0.12):
        out.append(("C: L1.F2 + L1.F1 (Holm of 2)", f2f1, Scenario(f"Δ = {d:+.2f} on F2, 0 on F1", a(dict.fromkeys(F1_CELLS, 0.0) | dict.fromkeys(F2_CELLS, d)), ("L1.F1",)), "power"))
    for d in (0.10, 0.15):
        out.append(("C: L1.F2 + L1.F1 (Holm of 2)", f2f1, Scenario(f"Δ = {d:+.2f} on F1 and F2", a(dict.fromkeys(STUDY_A_CELLS, d))), "power"))
    # D: L2 pooled alone
    out.append(("D: L2.pooled alone", l2, Scenario("Δ = 0 on Study B", b(0.0), ("L2.pooled",)), "null"))
    for d in (0.03, 0.04, 0.05, 0.06):
        out.append(("D: L2.pooled alone", l2, Scenario(f"Δ = {d:+.2f} on Study B", b(d)), "power"))
    # E: one family over L1.F2, L1.F1 and L2.pooled, Holm of 3
    out.append(("E: one family (Holm of 3)", one, Scenario("Δ = 0 everywhere", _with(a(dict.fromkeys(STUDY_A_CELLS, 0.0)), b(0.0)), ("L1.F2", "L1.F1", "L2.pooled")), "null"))
    out.append(("E: one family (Holm of 3)", one, Scenario("Δ = +0.10 on F2 only", _with(a(dict.fromkeys(F1_CELLS, 0.0) | dict.fromkeys(F2_CELLS, 0.10)), b(0.0)), ("L1.F1", "L2.pooled")), "power"))
    out.append(("E: one family (Holm of 3)", one, Scenario("Δ = +0.05 on Study B only", _with(a(dict.fromkeys(STUDY_A_CELLS, 0.0)), b(0.05)), ("L1.F2", "L1.F1")), "power"))
    return out


def design_check(reps: int = 1000, null_reps: int = 4000, seed: int = DESIGN_SEED, sigmas: Sigmas = Sigmas(), settings: Settings = Settings(), baselines: dict = BASELINES, only: tuple[str, ...] | None = None) -> list[dict]:
    """Every `design_cases` row, case i seeded seed + i (`only`: the designs whose name starts with one of these)."""
    sizes = m5_sizes()
    out = []
    for i, (design, hyp, sc, kind) in enumerate(design_cases(baselines)):
        if only and not any(design.startswith(o) for o in only):
            continue
        out.append({"design": design} | _row(hyp, sc, null_reps if kind == "null" else reps, seed + i, sigmas, settings, sizes))
    return out


def sensitivity(reps: int = 1000, seed: int = DESIGN_SEED + 500, baselines: dict = BASELINES) -> list[dict]:
    """The chosen members' power at their MDE when the ledger's effect varies more across worlds (σ_g 0.5, not 0.3)
    and when the worlds' own spread is larger (σ_w 0.8)."""
    sizes = m5_sizes()
    cases = [
        (get("L1"), Scenario("L1: Δ = +0.10 on F2, 0 on F1", spec_for("M5", dict.fromkeys(F2_CELLS, 0.10) | dict.fromkeys(F1_CELLS, 0.0), baselines), ("L1.F1",))),
        (get("L2"), Scenario("L2: Δ = +0.05 on Study B", spec_for("M5-spec", dict.fromkeys(STUDY_B_CELLS, 0.05), baselines))),
    ]
    out = []
    for j, (name, sig) in enumerate((("σ_g 0.5", Sigmas(g=0.5)), ("σ_w 0.8", Sigmas(w=0.8)))):
        for i, (hyp, sc) in enumerate(cases):
            out.append({"sigmas": name} | _row(hyp, sc, reps, seed + 10 * j + i, sig, Settings(), sizes))
    return out


# --------------------------------------------------------------------------- output


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def render(table: dict | None = None, design: list[dict] | None = None, sens: list[dict] | None = None) -> str:
    L = []
    if table:
        L += [f"# M5 add-on power ({table['reps']} studies per effect, {table['null_reps']} per null; seed {table['seed']})", ""]
        for fid, fam in table["families"].items():
            L += [f"## {fid} (α {fam['alpha']}, {fam['procedure']})", "", "| Scenario | Kind | Δ | Claim rate per member | False claims |", "|---|---|---|---|---|"]
            for r in fam["scenarios"]:
                L.append(f"| {r['scenario']} | {r['kind']} | {', '.join(f'{k} {v:+.3f}' for k, v in r['delta'].items())} | {', '.join(f'{k} {v:.3f}' for k, v in r['members'].items())} | {_fmt(r['false_claims'])}{' ⚠' if r.get('exceeds_alpha') else ''} |")
            L.append("")
    if design:
        L += ["# Design check", "", "| Design | Scenario | Kind | Reps | Seed | Claim rate per member | False claims |", "|---|---|---|---|---|---|---|"]
        for r in design:
            L.append(f"| {r['design']} | {r['scenario']} | {r['kind']} | {r['reps']} | {r['seed']} | {', '.join(f'{k} {v:.3f}' for k, v in r['members'].items())} | {_fmt(r['false_claims'])}{' ⚠' if r.get('exceeds_alpha') else ''} |")
        L.append("")
    if sens:
        L += ["# Sensitivity", "", "| σ | Scenario | Claim rate |", "|---|---|---|"]
        L += [f"| {r['sigmas']} | {r['scenario']} | {', '.join(f'{k} {v:.3f}' for k, v in r['members'].items())} |" for r in sens]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--null-reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--design", action="store_true", help="also run design_check (the candidate designs)")
    ap.add_argument("--only", nargs="*", default=None, help="design_check: designs whose name starts with these (e.g. A D)")
    ap.add_argument("--sensitivity", action="store_true", help="also run the σ sensitivity of the chosen members")
    ap.add_argument("--skip-table", action="store_true")
    ap.add_argument("--families", nargs="*", default=None, help="power table: these families only (e.g. L1)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    table = None if a.skip_table else power_table(a.reps, a.null_reps, a.seed, families=tuple(a.families) if a.families else None)
    design = design_check(a.reps, a.null_reps, only=tuple(a.only) if a.only else None) if a.design else None
    sens = sensitivity(a.reps) if a.sensitivity else None
    out = {"table": table, "design": design, "sensitivity": sens, "test_worlds": TEST_WORLDS}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1, default=str))
    sys.stdout.write(render(table, design, sens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CEILING", "CONTROL_OFFSET", "design_cases", "design_check", "m5_sizes", "power_table", "realised", "render", "scenarios", "sensitivity", "spec_for"]
