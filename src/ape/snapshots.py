"""Model snapshot pins (readiness E3; READINESS_AUDIT §3).

An OpenAI alias such as `gpt-6-luna` can move to a new snapshot in the middle of a study. The readiness probe
(`readiness/probe_openai.py`) records, per alias, the snapshot that served it: the `model` field of the response.
After review, `probe_openai.py --pin` writes those into PROVENANCE.md's "Model snapshots" block (`write_pins`).
Live preflight (`ape.models.preflight`) then calls `snapshot_status`:
- It refuses a run when the latest probe resolved a pinned alias to a different snapshot.
- It warns when nothing is pinned yet, when the probe did not re-check a pinned alias, or when the probe is old.

Pins are keyed by bare alias: `openai/gpt-6-luna` (Inspect) and `gpt-6-luna` (the build client) are one alias.
"""

import datetime as dt
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from .config import ROOT

PROBE_PATH = ROOT / "cache" / "openai_probe.json"
PROVENANCE_PATH = ROOT / "PROVENANCE.md"
BEGIN = "<!-- model-snapshots:begin -->"
END = "<!-- model-snapshots:end -->"
PROBE_MAX_AGE_DAYS = 7  # an older probe still counts, with a warning to re-check the snapshots
_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


def alias(model: str) -> str:
    """`openai/gpt-6-luna` -> `gpt-6-luna`."""
    return model.split("/", 1)[-1]


def _block(text: str) -> tuple[int, int] | None:
    """Offsets of the text between the markers, or None when the file has no block."""
    b = text.find(BEGIN)
    e = text.find(END, b + len(BEGIN)) if b >= 0 else -1
    return (b + len(BEGIN), e) if b >= 0 and e >= 0 else None


def read_pins(provenance_path: Path = PROVENANCE_PATH) -> dict[str, str]:
    """Pinned {alias: snapshot} from PROVENANCE.md's block; {} when the file or block is missing or empty."""
    if not provenance_path.is_file():
        return {}
    text = provenance_path.read_text()
    span = _block(text)
    if span is None:
        return {}
    pins = {}
    for line in text[span[0] : span[1]].splitlines():
        if m := _ROW.match(line.strip()):
            pins[m.group(1)] = m.group(2)
    return pins


def probe_snapshots(probe: Mapping) -> dict[str, str | None]:
    """{alias: snapshot} as the probe resolved it (None when no call reported a served model)."""
    return {a: (v or {}).get("snapshot") for a, v in (probe.get("snapshots") or {}).items()}


def _probe_age_days(probe: Mapping, today: dt.date) -> int | None:
    try:
        return (today - dt.datetime.fromisoformat(str(probe["probed_at"])).date()).days
    except (KeyError, ValueError):
        return None


def snapshot_status(
    probe_path: Path = PROBE_PATH,
    provenance_path: Path = PROVENANCE_PATH,
    aliases: Iterable[str] | None = None,
    today: dt.date | None = None,
) -> dict:
    """Compare the pins with the latest probe for `aliases` (the models a run calls; default: every pinned alias).

    Returns {"pins", "resolved", "problems", "warnings"}. A problem is a pinned alias that the probe resolved to a
    different snapshot, or pins with no readable probe. Everything else is a warning.
    """
    pins = read_pins(provenance_path)
    out: dict = {"pins": pins, "resolved": {}, "problems": [], "warnings": []}
    wanted = sorted(set(aliases) if aliases is not None else set(pins))
    if not pins:
        out["warnings"].append(
            f"no model snapshots are pinned in {provenance_path.name} yet: review {probe_path.name}, then run "
            "`uv run python readiness/probe_openai.py --pin`"
        )
        return out
    try:
        probe = json.loads(probe_path.read_text())
    except (OSError, ValueError) as e:
        out["problems"].append(f"{provenance_path.name} pins model snapshots but the probe {probe_path} is unreadable ({type(e).__name__}): run readiness/probe_openai.py")
        return out
    resolved = out["resolved"] = probe_snapshots(probe)
    for a in wanted:
        if a not in pins:
            out["warnings"].append(f"{a}: called by this run but not pinned in {provenance_path.name}; probe it and run `probe_openai.py --pin`")
        elif resolved.get(a) is None:
            out["warnings"].append(f"{a}: pinned to {pins[a]} but the latest probe did not resolve it, so the pin was not re-checked")
        elif resolved[a] != pins[a]:
            out["problems"].append(
                f"{a}: the latest probe was served by snapshot {resolved[a]}, but {provenance_path.name} pins {pins[a]}. "
                "The model changed since it was pinned: record a deviation or re-pin (`probe_openai.py --pin`)"
            )
    age = _probe_age_days(probe, today or dt.date.today())
    if age is not None and age > PROBE_MAX_AGE_DAYS:
        out["warnings"].append(f"the probe is {age} days old: re-run readiness/probe_openai.py so the snapshot check is current")
    return out


def _parse_block(block: str) -> tuple[dict[str, str], list[str]]:
    """The block's table rows {alias: row} and its "Superseded pins" lines."""
    rows, superseded, in_superseded = {}, [], False
    for line in (x.strip() for x in block.splitlines()):
        if m := _ROW.match(line):
            rows[m.group(1)] = line
        elif line == "Superseded pins:":
            in_superseded = True
        elif in_superseded and line.startswith("- "):
            superseded.append(line)
    return rows, superseded


def pin_problems(probe: Mapping) -> list[str]:
    """Reasons the probe cannot be pinned: an alias with no served snapshot, or served by more than one."""
    problems = []
    for a, v in (probe.get("snapshots") or {}).items():
        seen = (v or {}).get("snapshots_seen") or []
        if not seen:
            problems.append(f"{a}: no call reported a served snapshot (was every call rejected?)")
        elif len(set(seen)) > 1:
            problems.append(f"{a}: served by several snapshots in one probe {sorted(set(seen))}; re-run the probe before pinning")
    if not probe.get("snapshots"):
        problems.append("the probe resolved no model snapshots")
    return problems


def write_pins(provenance_path: Path, probe: Mapping, *, probe_path: Path = PROBE_PATH, today: dt.date | None = None) -> dict:
    """Write the probe's snapshots into PROVENANCE.md's block, keeping pins of aliases this probe did not cover.
    A changed pin is kept under "Superseded pins". Returns {"pinned", "changed", "kept"}. Raises ValueError when
    `pin_problems` finds any."""
    if problems := pin_problems(probe):
        raise ValueError("cannot pin:\n" + "\n".join(f"  - {p}" for p in problems))
    today = today or dt.date.today()
    text = provenance_path.read_text() if provenance_path.is_file() else ""
    old = read_pins(provenance_path)
    span = _block(text)
    old_rows, superseded = _parse_block(text[span[0] : span[1]] if span else "")
    probed = str(probe.get("probed_at", ""))[:10]
    price = (probe.get("price_table") or {}).get("checked") or "unknown"
    rows, changed = {}, {}
    for a, v in sorted((probe.get("snapshots") or {}).items()):
        snap = v["snapshot"]
        roles = ", ".join(v.get("roles") or [])
        rows[a] = f"| `{a}` | `{snap}` | {roles} | {probed} | {price} |"
        if a in old and old[a] != snap:
            changed[a] = (old[a], snap)
            superseded.append(f"- {today.isoformat()}: `{a}` was `{old[a]}`, now `{snap}`")
    kept = {a: line for a, line in old_rows.items() if a not in rows}
    body = [
        "",
        f"Pinned {today.isoformat()} by `readiness/probe_openai.py --pin` from `{probe_path.name}` (probed {probe.get('probed_at')}, profile `{probe.get('profile')}`).",
        "Live preflight (`ape.snapshots`) refuses a run when the latest probe resolves a pinned alias to another snapshot.",
        "",
        "| Alias | Snapshot | Roles (call path) | Probed | Prices checked |",
        "|---|---|---|---|---|",
        *[rows.get(a) or kept[a] for a in sorted({*rows, *kept})],
        "",
    ]
    if superseded:
        body += ["Superseded pins:", *superseded, ""]
    block = "\n".join(body)
    if span:
        text = text[: span[0]] + block + text[span[1] :]
    else:
        text = text.rstrip("\n") + f"\n\n### Model snapshots\n\n{BEGIN}{block}{END}\n"
    provenance_path.write_text(text)
    return {"pinned": {a: v["snapshot"] for a, v in (probe.get("snapshots") or {}).items()}, "changed": changed, "kept": sorted(kept)}
