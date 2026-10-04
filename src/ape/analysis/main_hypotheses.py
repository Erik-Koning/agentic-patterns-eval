"""The main study's hypothesis table: the single source for the analysis (`main_stats`, `main_report`), the power
simulation (`main_power`) and PREREGISTRATION_MAIN.md (BUILD_PLAN B5 copies `hypothesis_table()` /
`render_hypothesis_table()` into it).

IDs are HYPOTHESES.md's (K1, K2, K4, M1-M5, T1, C1, A1); the brief's IDs are in `source` (T1 is the brief's H6 tier
clause). H2's operational non-inferiority is its own family, "K2-NI". Status follows
D-028 (M4 and K4 descriptive) and D-033, the owner's decision on B4's power findings (D-031, open item P-1):
- M3 and M5: the TOST margins widen from ±3 pp to ±6 pp (planned power 0.84 / 0.87 at Δ = 0, against 0.13 / 0.05);
- K2: the delivery × architecture interaction is its own family; the operational NI becomes one member pooled over
  Study B's four cells at a 5 pp margin in its own family, K2-NI (planned power 0.79, against 0.12-0.16 per family
  at 3 pp inside K2's Holm), keeping the S5/M2 cost-ratio condition;
- T1 (the brief's H6 tier clause): descriptive (planned power 0.44-0.54 even at a 15 pp drop), per cell and pooled;
- the superiority members (M1, M2, K1) are unchanged; their planned MDE (power 0.8) is ~15 pp (`Member.mde`).
Each member's `power` states its simulated operating characteristics at the planned sizes (`main_power`).

**Families.** Each hypothesis is one Holm family at its own one-sided α (brief §7.4: Holm within each hypothesis
family; no correction across families). `procedure="serial"` is the brief's serial gatekeeping: stage 2 is tested
(Holm at the full α) only after every stage-1 member is rejected; M2's stage 1 is "M1 beats S1", its stage 2 "M1
beats the S8 frontier".

**Tests** (each on the per-task contrast, pooled over `cells` with equal cell weights, world-clustered):
- `superiority`: H0 Δ ≤ 0 vs Δ > 0; `less`: H0 Δ ≥ 0 vs Δ < 0 (H2's interaction);
- `ni`: H0 Δ ≤ -margin; `tost`: two one-sided tests at ±margin, p = the larger (equivalence);
- `descriptive`: estimate and intervals only.

**Terms.** A contrast is Σ coef × term, per task. A term is an arm's task mean (epochs averaged) in one tier, or the
S8 frontier: `Term("S8", match="M1")` is S8 interpolated at M1's realised mean cost in the same tier and cell on the
family's meter; `pool=3` builds that frontier from the first 3 S1 runs only (T1: Sol has 3); `k=3` is S8(3) itself.

**Choices the brief left open** (D-031; also in `main_report.CHOICES`): the H1a and H1b F2 clauses ("≤ 0", "S8 ≥ M1")
have no margin, so they are descriptive; pooled members pool the study's four cells (H1c over Study A; H1e, H2's
interaction and H2's NI over Study B); α is one-sided 0.025 for superiority / NI / `less` families and 0.05 for TOST
families (each one-sided test at 0.05: the 90% interval inside ±margin, the convention D-028's numbers use); the
frontier meter is tokens (the cap-enforcement meter, §4.5), every other meter reported; T1 is the brief's H6
"coordination payoff M1 − S8 at matched cost decreases with tier", Luna → Sol on 3-run frontiers in both tiers so they
are built alike.
"""

from dataclasses import asdict, dataclass

STUDY_A_CELLS = ("F1-2", "F1-32", "F2-2", "F2-10")
STUDY_B_CELLS = ("F3-5", "F3-60", "F7-10", "F7-1000")
TIER_CELLS = ("F1-32", "F7-100")
KB_CELLS = ("F7-10", "F7-100", "F7-1000")
HIGH = {"F1": "F1-32", "F2": "F2-10", "F3": "F3-60", "F7": "F7-1000"}
PRIMARY_TIER = "luna"
PRIMARY_METER = "tokens"
ALPHA = 0.025  # one-sided: superiority, NI, `less`
ALPHA_TOST = 0.05  # each one-sided test of a TOST (the 90% interval)
TOST_MARGIN = 0.06  # H1c, H1e: ±6 pp (D-033; the brief's ±3 pp had power 0.13 / 0.05 at the planned sizes)
NI_MARGIN = 0.05  # H2's operational NI, pooled over Study B: S5 ≥ M2 − 5 pp (D-033; brief: 3 pp per family)
SUPERIORITY_MDE = 0.15  # the planned MDE (power 0.8) of the single-cell superiority members (main_power, D-031)


@dataclass(frozen=True)
class Term:
    arm: str
    coef: float = 1.0
    tier: str = PRIMARY_TIER
    match: str | None = None  # S8 only: interpolated at this arm's realised mean cost
    pool: int | None = None  # S8 only: the frontier from the first `pool` S1 runs
    k: int | None = None  # S8 only: S8(k) itself

    def label(self) -> str:
        if self.arm == "S8" and self.match:
            name = f"S8{'' if self.pool is None else f'[pool {self.pool}]'}@{self.match}"
        elif self.arm == "S8" and self.k:
            name = f"S8({self.k})"
        else:
            name = self.arm
        return name if self.tier == PRIMARY_TIER else f"{name}[{self.tier}]"


@dataclass(frozen=True)
class CostRatio:
    """A co-condition on a member's claim: mean cost(num) / mean cost(den) on its paired tasks, on `meter`, must be
    at most `max_point`, with the upper end of its two-sided 95% world-clustered interval at most `max_upper`."""

    num: str
    den: str
    max_point: float
    max_upper: float
    meter: str = PRIMARY_METER


@dataclass(frozen=True)
class Member:
    id: str
    terms: tuple[Term, ...]
    cells: tuple[str, ...]
    test: str  # superiority | less | ni | tost | descriptive
    margin: float = 0.0
    stage: int = 1
    cost_ratio: CostRatio | None = None
    note: str = ""
    mde: float | None = None  # planned minimum detectable effect at power 0.8 (main_power), for the pre-registration
    power: str = ""  # simulated operating characteristics at the planned sizes (main_power)

    def contrast(self) -> str:
        parts = []
        for t in self.terms:
            sign = "+" if t.coef > 0 else "−"
            mag = "" if abs(t.coef) == 1 else f"{abs(t.coef):g}·"
            parts.append(f"{sign} {mag}{t.label()}")
        s = " ".join(parts)
        return s[2:] if s.startswith("+ ") else s

    def arms(self) -> tuple[str, ...]:
        out = []
        for t in self.terms:
            for a in (t.arm, t.match) if t.arm == "S8" else (t.arm,):
                if a and a not in out:
                    out.append(a)
        if any(t.arm == "S8" for t in self.terms) and "S1" not in out:
            out.append("S1")
        return tuple(out)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    source: str
    statement: str
    status: str  # confirmatory | descriptive
    members: tuple[Member, ...] = ()
    alpha: float = ALPHA
    procedure: str = "holm"  # holm | serial
    meter: str = PRIMARY_METER
    estimator: str | None = None  # descriptive hypotheses with their own estimator (main_descriptive)
    note: str = ""


def _t(*spec) -> tuple[Term, ...]:
    """_t(("M1", 1), ("S9", -1)) -> Terms in the primary tier."""
    return tuple(Term(a, c) for a, c in spec)


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "M1", "brief H1a", "Context isolation drives multi-agent gains on breadth tasks: M1 − S9 > 0 on F1-high (M1 stands in for M1s in the MVS).",
        "confirmatory",
        (Member("M1.F1-32", _t(("M1", 1), ("S9", -1)), ("F1-32",), "superiority", mde=SUPERIORITY_MDE, power="type I 0.015; power 0.50 at +10 pp, 0.86 at +15 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),),
        note="Valid only if M1 ≈ M1s (M4, descriptive since D-028). M1s − S9 on M1s's 60 tasks (5 worlds × 12) is reported alongside (`contrasts`).",
    ),
    Hypothesis(
        "M2", "brief H1b", "Ensembling vs coordination: on F1-high M1 beats S8 at M1's realised cost, tested only after M1 beats S1 (serial gatekeeping).",
        "confirmatory",
        (
            Member("M2.gate", _t(("M1", 1), ("S1", -1)), ("F1-32",), "superiority", stage=1, mde=SUPERIORITY_MDE, power="type I 0.028 (M1 = S1); the same single-cell design as M1.F1-32 (1,000 simulated studies at 108 tasks per cell, gate σ priors)", note="gate: a multi-agent arm meets the S8 frontier only after beating S1 (brief §7.4)"),
            Member("M2.frontier", (Term("M1"), Term("S8", -1, match="M1")), ("F1-32",), "superiority", stage=2, mde=SUPERIORITY_MDE, power="false claims at the frontier boundary 0.019 / 0.029 / 0.023 with failed runs costing κ = 0 / 1 / 3 times more (κ 3, 3,000 studies: 0.027); power 0.56 at +10 pp, 0.92 at +15 pp (κ 3: 0.47 at +10 pp), after the gate (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),
        ),
        procedure="serial",
    ),
    Hypothesis(
        "M3", "brief H1c; D-033", "Communication adds ≈ 0 at matched cost: M7 − S8 (at M7's realised cost) within ±6 pp, averaged over Study A's four cells with equal weights (each cell's estimate is reported beside it).",
        "confirmatory",
        (Member("M3.pooled", (Term("M7"), Term("S8", -1, match="M7")), STUDY_A_CELLS, "tost", TOST_MARGIN, power="type I at nominal except up to 0.059 at the +6 pp boundary (κ 0) and 0.055 at −6 pp (κ 1), from the near-ceiling cells F1-2 and F2-2 (0.047 off the ceiling); power at Δ = 0: 0.89 / 0.79 / 0.69 at κ = 0 / 1 / 3 (4,000 studies per boundary, 1,000 for power, 108 tasks per cell, gate σ priors; main_power.m3_pool_check, seeds 20261040 / 20261050 / 20261060)"),),
        alpha=ALPHA_TOST,
        note="Margin ±6 pp (D-033; the brief's ±3 pp had power 0.13). Confirmatory on the four cells (D-048): the claim is an equal-weight average over four cells, two of them (F1-2, F2-2) near the ceiling, and is read with the per-cell estimates; those two cells push the type I to 0.059 at the +6 pp boundary (κ 0) and 0.055 at −6 pp (κ 1) against 0.05, and testing F1-32 and F2-10 alone (type I 0.030–0.041) left power 0.12–0.20. Not gated on M7 > S1: equivalence to the frontier is informative whether or not M7 beats S1. Cells where M7 costs more than S8(8) are left out and named.",
    ),
    Hypothesis(
        "M5", "brief H1e; D-033", "Role specialization adds ≈ 0: M2 − M1k within ±6 pp, averaged over Study B's four cells with equal weights (each cell's estimate is reported beside it).",
        "confirmatory",
        (Member("M5.pooled", _t(("M2", 1), ("M1k", -1)), STUDY_B_CELLS, "tost", TOST_MARGIN, power="type I 0.052 / 0.052 at +6 / −6 pp (4,000 studies); power 0.88 at Δ = 0, 0.72 at +2 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),),
        alpha=ALPHA_TOST,
        note="Margin ±6 pp (D-033; the brief's ±3 pp had power 0.05).",
    ),
    Hypothesis(
        "K1", "brief H2-struct", "Graph structure beats engineered flat retrieval: S5 > S3s on F7-high and F3-high.",
        "confirmatory",
        (
            Member("K1.F7-1000", _t(("S5", 1), ("S3s", -1)), ("F7-1000",), "superiority", mde=SUPERIORITY_MDE, power="family type I 0.010; power 0.41 at +10 pp, 0.83 at +15 pp (Holm of 2) (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),
            Member("K1.F3-60", _t(("S5", 1), ("S3s", -1)), ("F3-60",), "superiority", mde=SUPERIORITY_MDE, power="family type I 0.010; power 0.49 at +10 pp, 0.90 at +15 pp (Holm of 2) (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),
        ),
    ),
    Hypothesis(
        "K2", "brief H2; D-033", "KG delivery substitutes for multi-agent specialization: the delivery × architecture interaction (M1k − S5) − (M1 − S1) is negative, averaged over Study B's four cells with equal weights.",
        "confirmatory",
        (Member("K2.interaction", _t(("M1k", 1), ("S5", -1), ("M1", -1), ("S1", 1)), STUDY_B_CELLS, "less", mde=0.08, power="type I 0.023; power 0.47 at −5 pp, 0.87 at −8 pp, 0.96 at −10 pp (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),),
        note="Its own family since D-033 (the operational NI is K2-NI).",
    ),
    Hypothesis(
        "K2-NI", "brief H2 (operational clause); D-033", "Operationally, the single KG agent is within 5 pp of the specialists at no more than about half their cost: S5 ≥ M2 − 5 pp averaged over Study B's four cells with equal weights, with the realised token cost ratio S5/M2 ≤ 0.6 at 97.5% confidence and its point estimate ≤ 0.5.",
        "confirmatory",
        (Member("K2-NI.pooled", _t(("S5", 1), ("M2", -1)), STUDY_B_CELLS, "ni", NI_MARGIN, cost_ratio=CostRatio("S5", "M2", 0.5, 0.6), power="type I 0.022 at −5 pp; power 0.79 at Δ = 0, 0.38 at −2 pp, 0.98 at +3 pp, with the cost condition (1,000 simulated studies at 108 tasks per cell, gate σ priors)"),),
        note="D-033: one member pooled over F3 and F7 at 5 pp, its own family (the brief's 3 pp per family inside K2's Holm had power 0.12-0.16). The claim needs the NI rejection and the cost-ratio condition (intersection-union: no extra adjustment).",
    ),
    # ---- descriptive ----
    Hypothesis(
        "T1", "brief H6 (tier clause); D-028 #2; D-033", "The coordination payoff M1 − S8 (at M1's realised cost), Sol minus Luna on the same tasks, per cell (F1-32, F7-100) and pooled, with intervals (a decrease with tier is the brief's prediction).",
        "descriptive",
        tuple(
            Member(
                f"T1.{name}",
                (Term("M1", 1, "sol"), Term("S8", -1, "sol", match="M1", pool=3), Term("M1", -1), Term("S8", 1, match="M1", pool=3)),
                cells, "descriptive",
                note="3-run frontiers in both tiers (Sol's S1 has 3 epochs; Luna's pool is cut to its first 3 so both are built alike)",
            )
            for name, cells in (("F1-32", ("F1-32",)), ("F7-100", ("F7-100",)), ("pooled", TIER_CELLS))
        ),
        note="Descriptive since D-033 (planned power 0.44-0.54 at a 15 pp drop). HYPOTHESES.md: T1 (P1 is the brief H6's routing clause). F1-32 needs no M2 (D-034: Study F runs M2 on F7-100 only).",
    ),
    Hypothesis(
        "M1-F2", "brief H1a (F2 clause)", "Isolation does not help on F2-high: M1 − S9 on F2-10, reported with its interval.",
        "descriptive",
        (Member("M1-F2.F2-10", _t(("M1", 1), ("S9", -1)), ("F2-10",), "descriptive"),),
        note="The brief's '≤ 0' has no margin; pre-register one to make it a one-sided test.",
    ),
    Hypothesis(
        "M2-F2", "brief H1b (F2 clause)", "On F2, S8 at M1's realised cost ≥ M1: M1 − S8@M1 per F2 cell, reported with intervals.",
        "descriptive",
        tuple(Member(f"M2-F2.{c}", (Term("M1"), Term("S8", -1, match="M1")), (c,), "descriptive") for c in ("F2-2", "F2-10")),
        note="No margin in the brief.",
    ),
    Hypothesis(
        "M4", "brief H1d; D-028 #3", "Concurrency is latency-only: M1 − M1s accuracy and the wall-clock ratio M1/M1s on F1-32 (M1s: 60 tasks), with intervals.",
        "descriptive", (Member("M4.F1-32", _t(("M1", 1), ("M1s", -1)), ("F1-32",), "descriptive"),), estimator="m4",
    ),
    Hypothesis(
        "K4", "brief H2c; D-028 #3", "KB scaling: success slopes over log10 KB size (F7-10/100/1000) for S1, S5, S3s, M1, M2 and the paired slope differences S5 − S1 and S5 − S3s, with intervals.",
        "descriptive", estimator="k4",
        note="S3s has no F7-100 cell, so its slope rests on two levels. The relational-vs-independent clause is not tested (no independent cells).",
    ),
    Hypothesis(
        "C1", "brief H3", "Cost-meter rank flips: Kendall τ between arm rankings (cost per solved task) under tokens, cache-adjusted $ and wall-clock is < 0.8 in ≥ 2 of the 4 MVS families.",
        "descriptive", estimator="c1",
        note="A rule on point estimates with world-clustered bootstrap support; the brief pre-registers no flip directions, so there is no test.",
    ),
    Hypothesis(
        "A1", "brief H5", "Determinism: pass^k, outcome stability (D1), answer agreement (D2) and resource predictability (D6) per arm and cell on the Study C subset (5 epochs), and KG single-agent vs multi-agent contrasts at matched accuracy.",
        "descriptive", estimator="a1",
        note="Not confirmatory: D5 (paraphrases) and the auditability metrics (A2, A4, A6, A8) are not produced by the main study.",
    ),
)

# §4.3 single-switch mechanism contrasts, reported descriptively per cell (where both arms ran).
MECHANISM_CONTRASTS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("Decomposition", "S9", "S1", STUDY_A_CELLS),
    ("Isolation", "M1", "S9", STUDY_A_CELLS),
    ("Isolation (M1s)", "M1s", "S9", ("F1-32",)),
    ("Concurrency", "M1", "M1s", ("F1-32",)),
    ("Multi-agent vs single (monolith)", "M1", "S1", STUDY_A_CELLS + STUDY_B_CELLS),
    ("Council vs single", "M7", "S1", STUDY_A_CELLS),
    ("KG delivery", "S5", "S1", STUDY_B_CELLS + ("F7-100",)),
    ("Graph structure", "S5", "S3s", STUDY_B_CELLS),
    ("Flat retrieval", "S3s", "S1", STUDY_B_CELLS),
    ("KG vs placebo", "S5", "S7", STUDY_B_CELLS),
    ("KG workers", "M1k", "M1", STUDY_B_CELLS),
    ("Specialization", "M2", "M1k", STUDY_B_CELLS),
    ("Single KG vs specialists", "S5", "M2", STUDY_B_CELLS + ("F7-100",)),  # D-034: no M2 on F1-32
)


def get(hyp_id: str, hypotheses=HYPOTHESES) -> Hypothesis:
    for h in hypotheses:
        if h.id == hyp_id:
            return h
    raise KeyError(hyp_id)


def confirmatory(hypotheses=HYPOTHESES) -> tuple[Hypothesis, ...]:
    return tuple(h for h in hypotheses if h.status == "confirmatory")


def hypothesis_table(hypotheses=HYPOTHESES) -> list[dict]:
    """One row per member (descriptive hypotheses with their own estimator: one row), JSON-serialisable."""
    rows = []
    for h in hypotheses:
        base = {"id": h.id, "source": h.source, "status": h.status, "family": h.id, "alpha": h.alpha if h.status == "confirmatory" else None, "procedure": h.procedure if h.status == "confirmatory" else None, "statement": h.statement, "note": h.note}
        if not h.members:
            rows.append(base | {"member": h.id, "contrast": None, "arms": None, "cells": None, "test": h.estimator, "margin": None, "stage": None, "cost_ratio": None, "meter": h.meter, "mde": None, "planned_power": None})
        for m in h.members:
            rows.append(
                base
                | {
                    "member": m.id,
                    "contrast": m.contrast(),
                    "arms": list(m.arms()),
                    "cells": list(m.cells),
                    "test": m.test,
                    "margin": m.margin if m.test in ("ni", "tost") else None,
                    "stage": m.stage if h.procedure == "serial" else None,
                    "cost_ratio": asdict(m.cost_ratio) if m.cost_ratio else None,
                    "meter": h.meter if any(t.arm == "S8" for t in m.terms) else None,
                    "mde": m.mde,
                    "planned_power": m.power or None,
                    "member_note": m.note,
                }
            )
    return rows


TEST_TEXT = {
    "superiority": "one-sided superiority (H0 Δ ≤ 0)",
    "less": "one-sided (H0 Δ ≥ 0, H1 Δ < 0)",
    "ni": "one-sided non-inferiority (H0 Δ ≤ −margin)",
    "tost": "equivalence, TOST (H0 Δ ≤ −margin or Δ ≥ margin)",
    "descriptive": "estimate and intervals",
}


def render_hypothesis_table(hypotheses=HYPOTHESES) -> str:
    """The table as markdown, for PREREGISTRATION_MAIN.md."""
    out = ["| ID | Member | Source | Contrast | Cells | Test | Margin | Holm family (α, procedure) | Planned MDE / power | Status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in hypothesis_table(hypotheses):
        stage = "" if r["stage"] is None else f", stage {r['stage']}"
        fam = f"{r['family']} ({r['alpha']:g}, {r['procedure']}{stage})" if r["status"] == "confirmatory" else "–"
        margin = "–" if r["margin"] is None else f"±{100 * r['margin']:g} pp" if r["test"] == "tost" else f"{100 * r['margin']:g} pp"
        extra = f"; cost {r['cost_ratio']['num']}/{r['cost_ratio']['den']} ≤ {r['cost_ratio']['max_upper']} at 97.5% confidence, point estimate ≤ {r['cost_ratio']['max_point']}" if r.get("cost_ratio") else ""
        test = TEST_TEXT.get(r["test"], r["test"] or "–") + extra
        planned = "; ".join(x for x in ((f"MDE ≈ {100 * r['mde']:g} pp" if r.get("mde") else ""), r.get("planned_power") or "") if x) or "–"
        out.append(f"| {r['id']} | {r['member']} | {r['source']} | {r['contrast'] or r['statement']} | {', '.join(r['cells'] or []) or '–'} | {test} | {margin} | {fam} | {planned} | {r['status']} |")
    return "\n".join(out) + "\n"
