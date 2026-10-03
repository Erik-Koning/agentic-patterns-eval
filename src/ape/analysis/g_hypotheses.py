"""Study G's hypothesis table: the single source PREREGISTRATION_G.md (BUILD_PLAN B11) copies.

Each row names the arms and plan cells a hypothesis uses, its estimand, test, margin, Holm family, gatekeeping
step and role (confirmatory or descriptive). `ape.analysis.g_stats` computes every row, `g_report` reports each
confirmatory row's decision and each descriptive row's estimate, and `g_power` simulates every confirmatory row's
type I error and power (and the descriptive rows' precision) through the same functions.

Conventions (D-029, D-032, D-033; CONTEXT_MANAGEMENT_AUDIT §2, §7; HYPOTHESES G1–G3):
- **Sessions are the clusters.** A session is one F8 world; its epochs are pooled first (they share the world),
  and every interval, sign-flip and bootstrap works on sessions. A world that appears at several capability points
  (same seeds) is one cluster across them.
- **One-sided α = 0.025 per family** (the gate's convention: the reported intervals are two-sided 95%); fixed-sequence
  gatekeeping where a row has `step`.
- **Decision test** (D-032): the session-clustered t on a linear combination of per-point session means
  (Welch–Satterthwaite df, capped at G_eff − 1, the effective number of clusters, when worlds are shared across
  points: D-038), the gate's validated interval (D-023). The exact session-level sign-flip (a wild sign-flip with null-restricted residuals for contrasts
  across points) is reported beside it, with its minimum attainable p.
- **Confirmatory rows (D-033):** G-H2a, and G-H3-pre → G-H3a → G-H3b. **G-H1 and G-H2b are descriptive**: no affordable
  design gives them useful power (G-H1 0.12–0.44, G-H2b's TOST 0.01 at ±0.20 R), so each is reported as an estimate
  with its 95% interval and nothing is decided on it.
- **Design (D-033):** topology cells at Luna-low and Luna-high with 16 sessions each, Sol-high 8, Astra-high 5 (S1 and
  M2 only), N = 20, 2 epochs. G-H3 pools the three points where all four topology arms run.
- **Outcome:** a session's item success rate (overflowed items fail), pooled over epochs. The binary "every item and
  the report exact" session success is reported, not tested: on 20–40 items it is near 0 for real agents.
- **Scale:** G-H1's slope on the empirical-logit scale (the audit's §2.1: near the ceiling a constant logit effect
  shrinks in pp); gaps, shares and headroom on the probability scale, where they are defined.

Still open for B11 (`OPEN_CHOICES`): the capability anchor near the ceiling, and the remaining priors the
micro-pilot calibrates.
"""

from dataclasses import asdict, dataclass

ALPHA = 0.025

# G-H1's single-agent reference (D-033): S-CM*; S1-pre (both arms on the items before S1's overflow) is the
# sensitivity analysis; S1 as scored is reported only descriptively (its gap carries CM0's overflow rule).
REFERENCES = ("S-CM*", "S1-pre", "S1")
DEFAULT_REFERENCE = "S-CM*"
SENSITIVITY_REFERENCE = "S1-pre"
# G-H2b's estimand: the change in headroom recovered R_x along the fitted line from the lowest to the highest measured
# capability point (descriptive, D-033). R_x is scale-free and not driven by CM0's overflow rule; the audit's gain on
# the logit scale is reported as a sensitivity. `TOST_MARGIN` stays the default of `g_stats.tost`, whose equivalence
# test is no longer decided.
TOST_ESTIMAND = "R"
TOST_MARGIN = {"R": 0.20, "gain": 0.50}  # R units; logit units
ISOLATION_SHARE = 0.5  # H3a
RECOVERY_SHARE = 0.8  # H3b
COST_RATIO = 0.6  # H3b: S-CM* cost per solved item at most this share of M2's
COST_METER = "cost_usd"  # cache-adjusted $ is primary (audit §5.1); tokens, calls and wall-clock are reported

CAP_CELLS = ("g.cap.luna-low", "g.cap.luna-high", "g.cap.sol-high", "g.cap.astra-high")
TOPO_CELLS = ("g.topo.luna-low", "g.topo.luna", "g.topo.sol", "g.topo.astra")  # D-033 adds g.topo.luna-low
H3_CELLS = TOPO_CELLS[:3]  # the cells where S1, M1, M2 and S-CM* all run
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
    alternative: str  # greater | less | equivalence | all_greater (intersection-union) | estimate (descriptive)
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
        "The multi-agent topology's advantage over a single agent with context management changes with measured "
        "capability (the bitter-lesson question: does it shrink?).",
        ("M2", "S-CM*", "S1"),
        (*TOPO_CELLS, *CAP_CELLS),
        "Slope of the per-session gap Δ_s = logit(M2) − logit(S-CM*) on measured capability (OLS over sessions), and "
        "the change along the fitted line over the capability span, at Luna-low, Luna-high and Sol-high (S-CM* does not "
        "run at Astra). Sensitivity: S1-pre (M2 − S1 on the items before S1's overflow; all four points). S1 as scored "
        "is reported only descriptively.",
        "estimate with 95% session-clustered t and session-bootstrap intervals; no decision (D-033)",
        "estimate",
        None,
        "logit over the capability span",
        "G-H1",
        "descriptive",
        notes="D-033 made this row descriptive. g_power at the D-033 design: its 95% interval covers the true 0 in 95.6% of null studies (S1-pre 94.8%); the change over the span has SD 0.29 logit (0.37 before D-033; S1-pre 0.36), so the interval excludes 0 in 26% of studies when M2's advantage falls 0.9 → 0 across the points (S1-pre 39%). Under a constant mechanism the S-CM* estimate drifts +0.24 and S1's +0.67 (CM0's overflow rule); S1-pre's does not.",
    ),
    GHypothesis(
        "G-H1-M1",
        "As G-H1 for M1 (isolation only).",
        ("M1", "S-CM*", "S1"),
        H3_CELLS,
        "As G-H1 with M1 (Luna-low, Luna-high and Sol-high).",
        "estimate with interval",
        "estimate",
        None,
        "logit over the capability span",
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
        notes="Decided over the plan's points: a planned point without paired sessions makes it INCOMPLETE, never "
        "SUPPORTED. Astra has 4 sessions: an exact sign-flip cannot go below 1/16 = 0.0625, so the decision rests on the t-test "
        "there (D-032). g_power: type I 0.020–0.024 per point (10,000 studies); power 0.99 by t at the ~+25–30 pp gaps CM0's overflow implies.",
    ),
    GHypothesis(
        "G-H2b",
        "Context-management gains persist across capability: the change in headroom recovered R_x across the capability "
        "points (CM-sum and CM-todo, the strategies run at every point).",
        ("CM-sum", "CM-todo", "O-state", "CM0"),
        (*CM_CELLS, *CAP_CELLS),
        "θ_x = slope of R_x,c = (s_x − s_CM0) / (s_O − s_CM0) on measured capability × the capability span (per-point "
        "ratio estimators, linearised per session). Sensitivity: the gain s_x − s_CM0 on the logit scale.",
        "estimate with 95% session-clustered interval; no decision (D-033)",
        "estimate",
        None,
        "R over the capability span",
        "G-H2",
        "descriptive",
        notes="D-033 made this row descriptive (the TOST at ±0.20 R had power 0.01; 80% would need ±0.46). The gain on "
        "the logit scale drifts +0.16 to +0.22 over the span under a constant R_x, because of CM0's overflow rule. g_power: θ has SD 0.14 R; its 95% interval covers 0 in 95% of studies at a constant R_x and excludes 0 in 20% when R_x falls by 0.2 over the span.",
    ),
    GHypothesis(
        "G-H2c",
        "Headroom recovered R_x per strategy and point.",
        ("CM-prune", "CM-sum", "CM-todo", "CM-reset", "CM-native", "O-state", "CM0"),
        CM_CELLS,
        "R_x,c with delta-method, Fieller and session-bootstrap intervals; undefined when the headroom is ≤ 2 pp.",
        "intervals only",
        "estimate",
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
        "estimate",
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
        H3_CELLS,
        "Mean over Luna-low, Luna-high and Sol-high (equal weights) of the per-session M2 − S1, item success rate.",
        "one-sided session-clustered t",
        "greater",
        0.0,
        "pp",
        "G-H3",
        "confirmatory",
        step=1,
        notes="Pools the plan's points; a planned point without ≥ 2 sessions of all four arms makes G-H3-pre, G-H3a and "
        "G-H3b INCOMPLETE, never SUPPORTED. g_power (D-033 design): power 1.0 (M2 − S1 ≈ +28–33 pp).",
    ),
    GHypothesis(
        "G-H3a",
        "Isolation carries at least half of the multi-agent gain: (M1 − S1) / (M2 − S1) ≥ 0.5.",
        ("M1", "M2", "S1"),
        H3_CELLS,
        "Linear form (M1 − S1) − 0.5 (M2 − S1) > 0, pooled over Luna-low, Luna-high and Sol-high (equal weights); the "
        "share itself with Fieller and bootstrap intervals.",
        "one-sided session-clustered t",
        "greater",
        ISOLATION_SHARE,
        "share",
        "G-H3",
        "confirmatory",
        step=2,
        notes="g_power (D-033 design): type I 0.026 at a share of 0.5 (10,000 studies); power 0.997 at 0.75, 0.996 at 0.80, 0.90 at 0.70, 0.66 at 0.65 (0.95 / 0.68 at 0.80 / 0.70 with σ_arm 0.6). Before D-033: 0.84 at 0.75.",
    ),
    GHypothesis(
        "G-H3b",
        "A single agent with context management recovers most of it more cheaply: S-CM* recovers ≥ 80% of (M2 − S1) at "
        "≤ 60% of M2's cost per solved item.",
        ("S-CM*", "M2", "S1"),
        H3_CELLS,
        "(S-CM* − S1) − 0.8 (M2 − S1) > 0 and log(CPS_S-CM* / CPS_M2) < log 0.6 (cache-adjusted $, probes excluded), "
        "pooled over Luna-low, Luna-high and Sol-high (equal weights).",
        "intersection-union of two one-sided session-clustered t-tests at α",
        "greater",
        RECOVERY_SHARE,
        "share; cost ratio 0.6",
        "G-H3",
        "confirmatory",
        step=3,
        notes="Fixed sequence G-H3-pre → G-H3a → G-H3b at α (audit §7). g_power (D-033 design): type I 0.027 (recovery at 0.8) and 0.026 (cost at 0.6) over 16,000 studies, 0.006 with both clauses at their margins; power 0.69 at recovery 0.95 and cost ratio 0.45, 0.37 at 0.90 / 0.50, 0.11 at 0.85 / 0.55 (the cost clause alone ≥ 0.99 at ≤ 0.5). With 24 Luna sessions per point 0.76; with 12 Sol sessions 0.74. Before D-033: 0.38.",
    ),
    GHypothesis(
        "G-H3-spec",
        "Specialisation share (M2 − M1) / (M2 − S1) = 1 − isolation share.",
        ("M1", "M2", "S1"),
        H3_CELLS,
        "Ratio of pooled mean differences.",
        "intervals only",
        "estimate",
        None,
        "share",
        "G-H3",
        "descriptive",
    ),
    GHypothesis(
        "G-D1",
        "Forked state probes: F1 per checkpoint per arm × point × session length, and the probe-to-behaviour correlation "
        "(audit §5.3).",
        ("all session arms",),
        (*CM_CELLS, SHORT_CELL, *TOPO_CELLS),
        "Mean probe F1 (a missed checkpoint scores 0) with session-clustered intervals.",
        "intervals only",
        "estimate",
        None,
        "F1",
        "descriptive",
        "descriptive",
    ),
    GHypothesis(
        "G-D2",
        "Failure taxonomy per arm × point × session length (audit §5.4).",
        ("all session arms",),
        (*CM_CELLS, SHORT_CELL, *TOPO_CELLS),
        "Label counts and rates per item.",
        "counts only",
        "estimate",
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
        "Posterior means and SDs (statsmodels BinomialBayesMixedGLM, variational Bayes); long sessions only (the N = 10 "
        "control is left out).",
        "none",
        "estimate",
        None,
        "logit",
        "descriptive",
        "descriptive",
    ),
)

OPEN_CHOICES = (
    "The capability anchor near the ceiling: measured capability spans about 0.72–0.90 with a standard error near 0.035 "
    "per point, so Sol and Astra may not separate; a harder anchor (F7-1000, F3-60) or the tier order would. It affects "
    "the descriptive G-H1 and G-H2b slopes only.",
    "The power priors (σ_world 0.5, σ_item 1.0, σ_arm 0.3, σ_epoch 0.2, σ_mix 0.5, crossing 0.70, base success per "
    "point) are assumptions until the micro-pilot (g.pilot.luna) calibrates them; G-H3a and G-H3b lose power if σ_arm "
    "is larger (see g_power's σ_arm 0.6 sensitivity).",
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


def design_markdown() -> str:
    """Each row's plan cells, level and planned power or precision (its `notes`, from g_power), as
    PREREGISTRATION_G.md shows them beside `markdown()`."""
    rows = ["| ID | Plan cells | One-sided α | Planned power and precision |", "|---|---|---|---|"]
    for h in HYPOTHESES:
        level = f"{h.alpha:g}" if h.role == "confirmatory" else "– (descriptive)"
        notes = (h.notes or "–").replace("|", "\\|")
        rows.append(f"| {h.id} | {', '.join(h.cells)} | {level} | {notes} |")
    return "\n".join(rows) + "\n"
