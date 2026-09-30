"""Readiness H5: write a spot-check sheet of 20 generated tasks per family for human review.

    uv run python readiness/spotcheck.py            # writes readiness/spotcheck.md

For each task the reviewer confirms: the prompt is unambiguous, the gold answer follows
from the gold facts alone, and the gold facts are findable in the rendered documents.
"""

import json
import random
from pathlib import Path

from ape.worlds.generate import make_world

OUT = Path(__file__).with_name("spotcheck.md")
FAMILIES = [("F7", "100"), ("F7", "1000"), ("F3", "20"), ("F3", "60"), ("F5", "1hop"), ("F5", "2hop")]


def main(per_family: int = 20) -> None:
    lines = ["# Task spot-check (H5)", "", "Mark each task OK / ISSUE. An ISSUE blocks the pilot until the generator is fixed.", ""]
    rng = random.Random(0)
    for family, level in FAMILIES:
        w = make_world(family, level, "dev", 0, 60)
        lines += [f"## {family}-{level} ({w.id})", ""]
        for t in rng.sample(w.tasks, min(per_family // 2, len(w.tasks))):
            lines += [f"### {t.id}  [ ] OK  [ ] ISSUE", "", f"**Prompt:** {t.prompt}", "", f"**Gold:** `{json.dumps(t.gold)}`", ""]
            lines += [f"**Setup:** `{json.dumps(t.setup)}`", ""] if t.setup else []
            lines += ["**Gold facts:**", *[f"- `{f}`: {w.facts[f].text}" for f in t.gold_fact_ids], ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
