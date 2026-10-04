"""PREREGISTRATION_M5.md (D-053) against the code it pre-registers: the hypothesis table is `m5_hypotheses`' table
verbatim, every hypothesis and confirmatory member is described in the prose with its α and planned MDE, the markers
are well formed and the `[USER: …]` items are the declared decisions, and the document states it adds nothing to the
main study's families. Offline: reads files only."""

import re

from ape import run_gate
from ape.analysis.m5_hypotheses import HYPOTHESES, L1_MDE, L2_MDE, PLAN_PREFIX, confirmatory, render_hypothesis_table
from ape.config import ROOT

PREREG = ROOT / "PREREGISTRATION_M5.md"
BEGIN = "<!-- BEGIN GENERATED: m5_hypotheses.render_hypothesis_table() -->\n"
END = "<!-- END GENERATED: m5_hypotheses.render_hypothesis_table() -->"
# The decisions PREREGISTRATION_M5 leaves to the user. A new `[USER: …]` item is a new pending decision: add it here.
USER_DECISIONS = {"owner", "analyst"}


def _split(text: str) -> tuple[str, str, str]:
    assert text.count(BEGIN) == 1 and text.count(END) == 1, "exactly one generated hypothesis table"
    head, rest = text.split(BEGIN)
    table, tail = rest.split(END)
    return head, table, tail


def test_the_hypothesis_table_is_the_codes_table_verbatim():
    text = PREREG.read_text()
    head, table, _ = _split(text)
    assert table == render_hypothesis_table(), "paste m5_hypotheses.render_hypothesis_table()'s output between the BEGIN and END comments"
    assert len(head) > run_gate.prereg_body_start(text) > 0, "the table is in the body (from the first '## ' heading on)"


def test_every_hypothesis_and_confirmatory_member_is_in_the_prose_with_its_mde():
    head, _, tail = _split(PREREG.read_text())
    prose = head + tail

    def named(token: str) -> bool:
        return re.search(rf"(?<![\w-]){re.escape(token)}(?![\w.-])", prose) is not None

    missing = [h.id for h in HYPOTHESES if not named(h.id)] + [m.id for h in confirmatory() for m in h.members if not named(m.id)]
    assert not missing, f"not described outside the generated table: {missing}"
    for mde in (L1_MDE, L2_MDE):
        assert f"{100 * mde:g} pp" in prose, f"the prose states the planned MDE {100 * mde:g} pp"
    assert "0.025" in prose and PLAN_PREFIX in prose


def test_it_is_an_add_on_with_concurrent_controls():
    text = PREREG.read_text()
    assert "adds no hypothesis to the main study's families" in text
    assert "concurrent" in text and "drift" in text, "the controls run beside the ledger arms; the main study's runs are never used"
    assert "PREREGISTRATION_MAIN.md" in text and "## Deviations log" in text


def test_markers_are_well_formed_and_user_items_are_the_pending_decisions():
    text = PREREG.read_text()
    markers = run_gate.prereg_placeholders(text)
    assert not [p for p in markers if not p["label"]], "every marker has a label"
    kinds = {p["kind"] for p in markers}
    assert kinds <= {"USER"}, f"only [USER: …] items: no runner fills this study's pre-registration ({kinds})"
    labels = {p["label"] for p in markers}
    assert labels <= USER_DECISIONS, f"undeclared user decisions: {sorted(labels - USER_DECISIONS)}"
