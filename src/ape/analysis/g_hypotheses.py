"""Study G's hypothesis table: the single source PREREGISTRATION_G.md (BUILD_PLAN B11) copies.

Each row names the arms and plan cells a hypothesis uses, its estimand, test, margin, Holm family, gatekeeping
step and role (confirmatory or descriptive). `ape.analysis.g_stats` computes every row, `g_report` reports each
row's decision, and `g_power` simulates every confirmatory row's type I error and power through the same functions.

Conventions (D-029; CONTEXT_MANAGEMENT_AUDIT §2, §7; HYPOTHESES G1–G3):
- **Sessions are the clusters.** A session is one F8 world; its epochs are pooled first (they share the world),
  and every interval, sign-flip and bootstrap works on sessions. A world that appears at several capability points
  (same seeds) is one cluster across them.
- **One-sided α = 0.025 per family** (the gate's convention: the reported intervals are two-sided 95%); Holm within a
  family, fixed-sequence gatekeeping where a row has `step`. Each TOST side is tested at α (a 95% interval inside
  the margin).
- **Primary test:** the session-clustered t on a linear combination of per-point session means (Welch–Satterthwaite
  df, capped at clusters − 1 when worlds are shared across points), the gate's validated interval (D-023). The exact
  session-level sign-flip (a wild sign-flip with null-restricted residuals for contrasts across points) is reported
  beside it, with its minimum attainable p; `g_power` shows where it cannot reach α.
- **Outcome:** a session's item success rate (overflowed items fail), pooled over epochs. The binary "every item and
  the report exact" session success is reported, not tested: on 20–40 items it is near 0 for real agents.
- **Scale:** G-H1's slope on the empirical-logit scale (the audit's §2.1: near the ceiling a constant logit effect
  shrinks in pp); gaps, shares and headroom on the probability scale, where they are defined.

Open for B11 (recorded in `OPEN_CHOICES`): G-H1's single-agent reference, the TOST estimand and margin, the
capability anchor near the ceiling, and whether Astra's Gap_T may rest on the t-test alone.
"""

from dataclasses import asdict, dataclass

ALPHA = 0.025

# G-H1's single-agent reference (BUILD_PLAN §4, open for B11). Recommended: S-CM*; see g_power and the report.
REFERENCES = ("S-CM*", "S1", "S1-pre")
DEFAULT_REFERENCE = "S-CM*"
# The TOST on strategy x capability: the change in headroom recovered R_x along the fitted line from the lowest to
# the highest measured capability point. The audit (§2.2) fixes the logit unit for a gain-based estimand but no
# value; R_x is scale-free and not driven by CM0's overflow rule (see g_stats.tost), so it is the default estimand.
TOST_ESTIMAND = "R"
TOST_MARGIN = {"R": 0.20, "gain": 0.50}  # R units; logit units
ISOLATION_SHARE = 0.5  # H3a
RECOVERY_SHARE = 0.8  # H3b
COST_RATIO = 0.6  # H3b: S-CM* cost per solved item at most this share of M2's
COST_METER = "cost_usd"  # cache-adjusted $ is primary (audit §5.1); tokens, calls and wall-clock are reported

CAP_CELLS = ("g.cap.luna-low", "g.cap.luna-high", "g.cap.sol-high", "g.cap.astra-high")
TOPO_CELLS = ("g.topo.luna", "g.topo.sol", "g.topo.astra")
CM_CELLS = ("g.cm.luna-low", "g.cm.luna-high", "g.cm.sol-high", "g.cm.astra-high")
SHORT_CELL = "g.cm.luna-n10"


@dataclass(frozen=True)
class GHypothesis:
    id: str
    statement: str
    arms: tuple[str, ...]
    cells: tuple[str, ...]
    estimand: str
    test: str
    alternative: str  # greater | less | equivalence | all_greater (intersection-union) | sequence
    margin: float | None
    margin_units: str
    family: str
    role: str  # confirmatory | descriptive
    step: int | None = None  # fixed-sequence position within the family
    alpha: float = ALPHA
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


HYPOTHESES: tuple[GHypothesis, ...] = (
    GHypothesis(
        "G-H1",
        "The multi-agent topology's advantage over the single-agent reference shrinks as measured capability rises.",
        ("M2", "S-CM*", "S1"),
        (*TOPO_CELLS, *CAP_CELLS[1:]),
        "Slope of the per-session gap Δ_s = logit(M2) − logit(ref) on measured capability (OLS over sessions; "
        "reported also as the change over the capability span). ref = S-CM* (Luna-high, Sol-high) by default; S1 and "
        "S1-pre (both arms restricted to items before S1's overflow) are sensitivity analyses.",
        "one-sided session-clustered t (Satterthwaite); wild sign-flip over sessions reported",
        "less",
        0.0,
        "logit per unit capability",
        "G-H1",
        "confirmatory",
        notes="The reference is open (B11). With S1 = CM0, overflow fails the rest of the session, so the gap carries a "
        "harness term that grows with capability; S-CM* is not run at Astra, so with it the slope rests on 2 points. "
        "g_power at the 2026-10-03 priors: type I 0.026 for every reference (10,000 studies); power 0.12 (S-CM*), 0.12 "
        "(S1-pre), 0.03 (S1) when M2's logit advantage falls 0.6 → 0.3 → 0 across the points; MDE80 ≈ 1.0–1.3 logit "
        "over the capability span. Under a constant mechanism S1's slope drifts +0.46 logit over the span (S-CM* +0.13, "
        "S1-pre 0).",
    ),
    GHypothesis(
        "G-H1-M1",
        "As G-H1 for M1 (isolation only).",
        ("M1", "S-CM*", "S1"),
        TOPO_CELLS[:2],
        "As G-H1 with M1 (Luna-high and Sol-high only).",
        "interval only",
        "less",
        None,
        "logit per unit capability",
        "G-H1",
        "descriptive",
    ),
    GHypothesis(
        "G-H2a",
        "Context management has headroom at every capability point on long sessions: Gap_T = s(O-state) − s(CM0) > 0.",
        ("O-state", "CM0"),
        CM_CELLS,
        "Per point: mean over sessions of s(O-state) − s(CM0), item success rate, N = 40 (Astra 24).",
        "intersection-union: one-sided session-clustered t at α at every point (no multiplicity adjustment); exact "
        "sign-flip reported",
        "all_greater",
        0.0,
        "pp",
        "G-H2-gap",
        "confirmatory",
        notes="Astra has 4 sessions: an exact sign-flip cannot go below 1/16 = 0.0625, so only the t-test can reach α there "
        "(type I 0.023 at Astra, 0.023–0.027 elsewhere; power ≈ 0.99 at the ~+25–30 pp gaps CM0's overflow implies).",
    ),
    GHypothesis(
        "G-H2b",
        "Context-management gains persist across capability: headroom recovered R_x does not change with capability by "
        "more than the margin (CM-sum and CM-todo, the strategies run at every point).",
        ("CM-sum", "CM-todo", "O-state", "CM0"),
        (*CM_CELLS, *CAP_CELLS),
        "θ_x = slope of R_x,c = (s_x − s_CM0) / (s_O − s_CM0) on measured capability × the capability span (per-point "
        "ratio estimators, linearised per session).",
        "TOST: two one-sided session-clustered t-tests at α against ±margin; Holm over the two strategies",
        "equivalence",
        TOST_MARGIN["R"],
        "R units over the capability span",
        "G-H2-tost",
        "confirmatory",
        notes="The audit's gain-on-logit estimand (margin in logit units) is reported as a sensitivity: CM0's overflow rule "
        "makes raw gains grow with capability even at a constant R_x (+0.16 to +0.22 logit over the span in g_power). "
        "Power at a constant R_x: 0.01 at ±0.20, 0.17 at ±0.30, 0.72 at ±0.50 (sd of θ 0.14 R; 80% needs ±0.46); type I "
        "at the margin 0.005. At the planned sizes this row cannot show a meaningful equivalence (B11).",
    ),
    GHypothesis(
        "G-H2c",
        "Headroom recovered R_x per strategy and point.",
        ("CM-prune", "CM-sum", "CM-todo", "CM-reset", "CM-native", "O-state", "CM0"),
        CM_CELLS,
        "R_x,c with delta-method, Fieller and session-bootstrap intervals; undefined when the headroom is ≤ 2 pp.",
        "intervals only",
        "greater",
        None,
        "R",
        "G-H2",
        "descriptive",
    ),
    GHypothesis(
        "G-H2d",
        "Gains grow with length: degradation slopes (success ~ log view tokens × position) by arm × point, the gain "
        "before vs after the window crossing, and the N = 10 short-session control.",
        ("CM0", "O-state", "CM-prune", "CM-sum", "CM-todo", "CM-reset", "CM-native"),
        (*CM_CELLS, SHORT_CELL),
        "Logistic slopes with session-clustered intervals; Gap(N = 40) − Gap(N = 10) at Luna-high (Welch over sessions).",
        "intervals only",
        "greater",
        None,
        "logit per log-token / pp",
        "G-H2",
        "descriptive",
        notes="CM0 fails every item after its overflow, so 'gains grow with length' holds for CM0 by construction; the "
        "slopes of the managed arms and O-state carry the information.",
    ),
    GHypothesis(
        "G-H3-pre",
        "The specialised topology beats the single agent: M2 − S1 > 0 (gatekeeper: the shares are undefined otherwise).",
        ("M2", "S1"),
        TOPO_CELLS[:2],
        "Mean over Luna-high and Sol-high (equal weights) of the per-session M2 − S1, item success rate.",
        "one-sided session-clustered t",
        "greater",
        0.0,
        "pp",
        "G-H3",
        "confirmatory",
        step=1,
    ),
    GHypothesis(
        "G-H3a",
        "Isolation carries at least half of the multi-agent gain: (M1 − S1) / (M2 − S1) ≥ 0.5.",
        ("M1", "M2", "S1"),
        TOPO_CELLS[:2],
        "Linear form (M1 − S1) − 0.5 (M2 − S1) > 0, pooled over Luna-high and Sol-high; the share itself with Fieller "
        "and bootstrap intervals.",
        "one-sided session-clustered t",
        "greater",
        ISOLATION_SHARE,
        "share",
        "G-H3",
        "confirmatory",
        step=2,
        notes="g_power: type I 0.026 at a share of 0.5; power 0.84 at 0.75 (0.59 at 0.7, 0.90 at 0.8; 0.61 with σ_arm 0.6).",
    ),
    GHypothesis(
        "G-H3b",
        "A single agent with context management recovers most of it more cheaply: S-CM* recovers ≥ 80% of (M2 − S1) at "
        "≤ 60% of M2's cost per solved item.",
        ("S-CM*", "M2", "S1"),
        TOPO_CELLS[:2],
        "(S-CM* − S1) − 0.8 (M2 − S1) > 0 and log(CPS_S-CM* / CPS_M2) < log 0.6 (cache-adjusted $, probes excluded), "
        "pooled over Luna-high and Sol-high.",
        "intersection-union of two one-sided session-clustered t-tests at α",
        "greater",
        RECOVERY_SHARE,
        "share; cost ratio 0.6",
        "G-H3",
        "confirmatory",
        step=3,
        notes="Fixed sequence G-H3-pre → G-H3a → G-H3b at α (audit §7). g_power: type I 0.024 (recovery at 0.8) and 0.026 "
        "(cost at 0.6); power 0.40 at recovery 0.95 and cost ratio 0.45 (0.20 at 0.9 / 0.5), the cost clause alone 0.78–0.99. "
        "A Luna-low topology cell plus 16 Luna sessions per point (+$73 conservative) gives 0.70; 24 sessions (+$124) 0.77.",
    ),
    GHypothesis(
        "G-H3-spec",
        "Specialisation share (M2 − M1) / (M2 − S1) = 1 − isolation share.",
        ("M1", "M2", "S1"),
        TOPO_CELLS[:2],
        "Ratio of pooled mean differences.",
        "intervals only",
        "greater",
        None,
        "share",
        "G-H3",
        "descriptive",
    ),
    GHypothesis(
        "G-D1",
        "Forked state probes: F1 per checkpoint per arm × point, and the probe-to-behaviour correlation (audit §5.3).",
        ("all session arms",),
        (*CM_CELLS, *TOPO_CELLS),
        "Mean probe F1 (a missed checkpoint scores 0) with session-clustered intervals.",
        "intervals only",
        "greater",
        None,
        "F1",
        "descriptive",
        "descriptive",
    ),
    GHypothesis(
        "G-D2",
        "Failure taxonomy per arm × point (audit §5.4).",
        ("all session arms",),
        (*CM_CELLS, *TOPO_CELLS),
        "Label counts and rates per item.",
        "counts only",
        "greater",
        None,
        "rate",
        "descriptive",
        "descriptive",
    ),
    GHypothesis(
        "G-D3",
        "Mixed-effects logistic models success ~ topology × capability and success ~ strategy × capability with session "
        "and item random intercepts (D-029: descriptive only).",
        ("all session arms",),
        (*CM_CELLS, *TOPO_CELLS),
        "Posterior means and SDs (statsmodels BinomialBayesMixedGLM, variational Bayes).",
        "none",
        "greater",
        None,
        "logit",
        "descriptive",
        "descriptive",
    ),
)

OPEN_CHOICES = (
    "G-H1's single-agent reference: S-CM* (recommended; 2 capability points unless S-CM* is added to g.topo.astra, "
    "+$284 conservative), S1 as scored (3 points; its gap grows with capability through the overflow rule, so it cannot "
    "show convergence), or S1-pre (3 points; both arms on items before S1's overflow; no harness drift).",
    "Whether G-H1 stays confirmatory: no affordable design reaches power 0.8 at the plausible effect (0.12 planned; "
    "0.44 with a Luna-low topology cell, 16 Luna sessions per point and S-CM* at Astra, +$357).",
    "The TOST estimand (R_x, recommended, or the audit's gain on the logit scale) and its margin (default ±0.20 R over "
    "the capability span, power 0.01; ±0.46 for 0.8). Recommended: report θ with its interval (descriptive), or a "
    "one-sided non-inferiority at −0.40.",
    "Whether G-H2a may rest on the t-test at Astra (4 sessions: an exact sign-flip cannot reach α = 0.025), or Astra "
    "CM gets ≥ 6 sessions.",
    "The capability anchor near the ceiling: measured capability spans about 0.72–0.90 with a standard error near 0.035 "
    "per point, so Sol and Astra may not separate; a harder anchor (F7-1000, F3-60) or the tier order would.",
    "One-sided α = 0.025 per family with each TOST side at α (a 95% interval), not the common 90% TOST interval.",
)


def table() -> list[dict]:
    return [h.to_dict() for h in HYPOTHESES]


def by_id(hid: str) -> GHypothesis:
    for h in HYPOTHESES:
        if h.id == hid:
            return h
    raise KeyError(hid)


def confirmatory() -> list[GHypothesis]:
    return [h for h in HYPOTHESES if h.role == "confirmatory"]


def markdown() -> str:
    """The table as PREREGISTRATION_G.md shows it."""
    rows = ["| ID | Role | Family (step) | Arms | Estimand | Test | Margin |", "|---|---|---|---|---|---|---|"]
    for h in HYPOTHESES:
        margin = "–" if h.margin is None else f"{h.margin:g} ({h.margin_units})"
        step = f" ({h.step})" if h.step else ""
        rows.append(f"| {h.id} | {h.role} | {h.family}{step} | {', '.join(h.arms)} | {h.estimand} | {h.test} | {margin} |")
    return "\n".join(rows) + "\n"
