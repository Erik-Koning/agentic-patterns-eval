"""H5 spot-check sheet (`readiness/spotcheck.py`): deterministic, committed up to date, and drawn from the gate's
own dev worlds."""

import importlib.util
import re

from ape.budget import load_plan
from ape.config import ROOT


def _load():
    spec = importlib.util.spec_from_file_location("readiness_spotcheck", ROOT / "readiness" / "spotcheck.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spotcheck = _load()


def test_the_sheet_is_deterministic_and_the_committed_file_is_current():
    text = spotcheck.render()
    assert text == spotcheck.render()
    assert (ROOT / "readiness" / "spotcheck.md").read_text() == text, "re-run `uv run python readiness/spotcheck.py` and commit it"


def test_part_a_covers_the_gate_cells_with_20_tasks_per_family_from_the_gate_dev_worlds():
    text = spotcheck.render()
    part_a = text.split("## Part A", 1)[1].split("## Part B", 1)[0]
    ids = re.findall(r"^##### A-(F\d-\d+)-\d\d · `([^`]+)`", part_a, flags=re.M)
    by_cell: dict[str, list[str]] = {}
    for cell, task_id in ids:
        by_cell.setdefault(cell, []).append(task_id)
    assert {c: len(t) for c, t in by_cell.items()} == {"F7-10": 10, "F7-1000": 10, "F3-5": 10, "F3-60": 10}
    worlds = load_plan().cell("gate.build.dev").spec["worlds"]
    for cell, task_ids in by_cell.items():
        # The exact dev worlds build-dev builds: seeds 1000.., relational with descriptive exceptions for F7.
        allowed = {f"{cell}{'-rel-desc' if cell.startswith('F7') else ''}-dev-s{1000 + i}" for i in range(worlds[cell])}
        assert all(t.rsplit("-t", 1)[0] in allowed for t in task_ids), (cell, task_ids)
        assert len(set(task_ids)) == len(task_ids)
    # F7 mixes every exception case in each cell; every task carries the four checks and a verdict line.
    for cell in ("F7-10", "F7-1000"):
        block = part_a.split(f"#### {cell} (", 1)[1].split("\n", 1)[0]
        assert all(case in block for case in ("exception applies", "exception not applicable", "no exception")), block
    assert part_a.count("- Verdict: [ ] OK  [ ] ISSUE") == 40 and part_a.count("**Gold facts:**") == 40


def test_part_b_has_f1_f2_and_one_f8_session():
    part_b = spotcheck.render().split("## Part B", 1)[1]
    assert len(re.findall(r"^#### B-F1-", part_b, flags=re.M)) == 10 and len(re.findall(r"^#### B-F2-", part_b, flags=re.M)) == 10
    assert part_b.count("### F8: one short session") == 1 and "**Report gold:**" in part_b and "(`state_at`" in part_b
