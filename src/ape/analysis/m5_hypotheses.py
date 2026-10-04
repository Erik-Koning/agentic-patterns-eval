"""The M5 add-on study's hypothesis table (D-053): the single source for its analysis (`m5_report`, `ape.analyze_m5`),
its power simulation (`m5_power`) and PREREGISTRATION_M5.md, which pastes `render_hypothesis_table()` verbatim.

**Arms** (each one switch, STATE = text ledger, from its team): `M5` = M1 + a Magentic-style task ledger and progress
ledger with stall-triggered replanning, on Study A's cells (F1, F2); `M5-spec` = M2 + the same ledger, on Study B's
cells (F3, F7). Each runs with a fresh concurrent control (M1, M2) on the main study's test worlds, paired on tasks.
The arm name `M5` is not the main study's hypothesis M5 (role specialization, brief H1e): this table's IDs are L1, L2.

**Families.** The add-on is its own study: its families are tested at their own α and add nothing to the main study's
families (PREREGISTRATION_MAIN.md's Holm families are unchanged). Each hypothesis is one Holm family at one-sided
α = 0.025, the main study's convention (brief §7.4: Holm within a family, none across families):
- L1, the ledger on the M1 team (M5 − M1): two members, Holm of 2: L1.F2 pooled over F2-2 and F2-10 (dependency
  chains, the brief's prediction, §4.3: "text ledger … + on F2") and L1.F1 pooled over F1-2 and F1-32 (breadth; D-053);
- L2, the ledger on the specialist team (M5-spec − M2): one member pooled over Study B's four cells.
The choice and the numbers behind it (which members, one family or two, the MDE) are `m5_power.design_check`'s, at the
planned sizes (`m5_power.m5_sizes`: 108 tasks per cell over 9 worlds, 3 epochs per arm), and the operating
characteristics in each member's `power` are `m5_power.power_table`'s (PREREGISTRATION_M5.md §2.4, §2.6).

**Clusters.** F1 and F2 worlds of one seed share one supplier registry: L1's members have 9 clusters however many
cells they pool (`main_stats.cluster_of`); L2's has 36.

**Descriptive.** Per-cell contrasts; the ledger's price (cost ratios M5/M1 and M5-spec/M2 on tokens and $, with
world-clustered intervals; `m5_report.cost_ratios`); its diagnostics (replans per task, the stall rate, the ledger's
share of the tokens; `m5_load`, `m5_report.ledger_diagnostics`).
"""

from . import main_hypotheses as mh
from .main_hypotheses import ALPHA, STUDY_A_CELLS, STUDY_B_CELLS, TEST_TEXT, Hypothesis, Member, Term

LEDGER_ARMS = {"M5": "M1", "M5-spec": "M2"}  # arm -> its concurrent control (the team it adds the ledger to)
F1_CELLS = ("F1-2", "F1-32")
F2_CELLS = ("F2-2", "F2-10")
PLAN_PREFIX = "m5.test."  # the m5 study's test plan cells; rows of any other plan cell are never in its analysis
L1_MDE = 0.10  # planned MDE (power 0.8) of each L1 member, Holm of 2, the other member null (m5_power)
L2_MDE = 0.055  # planned MDE (power 0.8) of L2.pooled over Study B's four cells (m5_power)


def _t(*spec) -> tuple[Term, ...]:
    return tuple(Term(a, c) for a, c in spec)


L1_TERMS = _t(("M5", 1), ("M1", -1))
L2_TERMS = _t(("M5-spec", 1), ("M2", -1))

HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "L1", "brief §4.3 (text ledger, + on F2); D-053",
        "The ledger helps the M1 team: M5 − M1 > 0 on dependency chains (F2-2 and F2-10, the brief's prediction) and on breadth tasks (F1-2 and F1-32), each averaged over its two cells with equal weights (each cell's estimate is reported beside it); Holm over the two.",
        "confirmatory",
        (
            Member("L1.F2", L1_TERMS, F2_CELLS, "superiority", mde=L1_MDE, power="family type I 0.019 (0.019 with F1 at +10 pp); power 0.85 at +10 pp with F1 null, 0.92 with both at +10 pp, 0.64 / 0.96 at +8 / +12 pp with F1 null, 0.84 if a +20 pp effect sits on F2-10 alone (Holm of 2; 10,000 studies per null, 2,000 per effect, 108 tasks per cell, gate σ priors; m5_power seed 20261053)"),
            Member("L1.F1", L1_TERMS, F1_CELLS, "superiority", mde=L1_MDE, power="family type I 0.019 (0.018 with F2 at +10 pp); power 0.84 at +10 pp with F2 null, 0.91 with both at +10 pp (Holm of 2; the same simulation)"),
        ),
        note="Holm of 2 over members that share L1's 9 registry clusters (F1 and F2 worlds of one seed share a registry). F1 is confirmatory because it is powered (0.84 at +10 pp); including it lowers L1.F2's power at +10 pp from 0.93 alone to 0.83-0.85. F2 is pooled over both cells rather than tested on F2-10 alone: 0.93 against 0.50 at a uniform +10 pp; F2-10 alone does better only if the whole effect sits there (0.87 against 0.71 at +15 pp on F2-10) (m5_power.design_check, seed 20261153).",
    ),
    Hypothesis(
        "L2", "D-053",
        "The ledger helps the specialist team: M5-spec − M2 > 0, averaged over Study B's four cells with equal weights (each cell's estimate is reported beside it).",
        "confirmatory",
        (Member("L2.pooled", L2_TERMS, STUDY_B_CELLS, "superiority", mde=L2_MDE, power="type I 0.023; power 0.36 / 0.57 / 0.77 / 0.90 / 0.995 at +3 / +4 / +5 / +6 / +8 pp (10,000 studies at the null, 2,000 per effect, 108 tasks per cell, gate σ priors; m5_power seed 20262053)"),),
        note="36 clusters (F3 and F7 worlds are their own). Its own family, as each of the main study's hypotheses is: one Holm family over L1's and L2's members would lower L2's power at +5 pp from 0.78 to 0.60 and L1.F2's at +10 pp from 0.83 to 0.78 (m5_power.design_check, seed 20261153).",
    ),
    # ---- descriptive ----
    Hypothesis(
        "L1-cells", "D-053", "M5 − M1 in each of Study A's four cells, with intervals.", "descriptive",
        tuple(Member(f"L1-cells.{c}", L1_TERMS, (c,), "descriptive") for c in STUDY_A_CELLS),
    ),
    Hypothesis(
        "L2-cells", "D-053", "M5-spec − M2 in each of Study B's four cells, with intervals.", "descriptive",
        tuple(Member(f"L2-cells.{c}", L2_TERMS, (c,), "descriptive") for c in STUDY_B_CELLS),
    ),
    Hypothesis(
        "LC", "D-053", "The ledger's price: the realised cost ratios M5/M1 (Study A's cells, F1, F2) and M5-spec/M2 (Study B) on tokens and $, per cell and pooled, with world-clustered 95% intervals.",
        "descriptive", estimator="cost",
    ),
    Hypothesis(
        "LD", "D-053", "Ledger diagnostics per arm and cell: replans per task, the share of samples that replanned, the stall rate (stalled rounds over rounds) and the ledger's share of the sample's tokens.",
        "descriptive", estimator="ledger",
        note="From each sample's `mas_ledger` record and `mas_accounting` (m5_load); reported as not available where a record lacks a field.",
    ),
)


def get(hyp_id: str, hypotheses=HYPOTHESES) -> Hypothesis:
    return mh.get(hyp_id, hypotheses)


def confirmatory(hypotheses=HYPOTHESES) -> tuple[Hypothesis, ...]:
    return mh.confirmatory(hypotheses)


def hypothesis_table(hypotheses=HYPOTHESES) -> list[dict]:
    return mh.hypothesis_table(hypotheses)


def render_hypothesis_table(hypotheses=HYPOTHESES) -> str:
    """The table as markdown, for PREREGISTRATION_M5.md (the main table's columns)."""
    return mh.render_hypothesis_table(hypotheses)


__all__ = ["ALPHA", "F1_CELLS", "F2_CELLS", "HYPOTHESES", "L1_MDE", "L2_MDE", "LEDGER_ARMS", "PLAN_PREFIX", "STUDY_A_CELLS", "STUDY_B_CELLS", "TEST_TEXT", "confirmatory", "get", "hypothesis_table", "render_hypothesis_table"]
