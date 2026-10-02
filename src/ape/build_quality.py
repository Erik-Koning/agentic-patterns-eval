"""Build-quality check for the builder model (DECISIONS D-017).

D-017 decides which model builds both knowledge systems for the gate:
- **Luna (high)** if, on the dev worlds with descriptive exceptions:
  - authored APG graphs declare ≥ 95% of the spec's IDs (`id_coverage`, `apg.author`);
  - LightRAG extraction recovers ≥ 95% of the policy, exception and procedure IDs as entities.
- **Otherwise the Sol (medium) fallback,** which costs more and needs approval.

This module measures both coverages per world from the built artifacts. It then aggregates per gate cell and turns
the result into a recommendation. `run_gate`'s build-dev phase writes it to `build-dev/build_quality.json`, the pilot
puts it in `pilot.json` for GATE_PREREG's `[USER: builder]` item, and `readiness/smoke.py`'s D017 check uses the same
code on its one world.

**Definitions** (unchanged from the smoke check they replace):
- **APG coverage:** the share of the spec's policy, exception, procedure and tool IDs that some authored leaf declares.
  This is the `id_coverage` in the graph's report.
- **LightRAG coverage:** the share of the spec's policy, exception and procedure IDs that appear in the index's
  entity store, matched by ID pattern.
- **Per cell:** both are pooled over the cell's worlds (covered IDs / spec IDs). The minimum per world is reported too.
- **Decisive worlds:** worlds with descriptive exceptions and at least one spec ID. id_only, messy and F5 worlds (F5
  has no such IDs) are reported as diagnostics only.
- **Decisive cells:** the gate's dev cells, F7-10, F7-1000, F3-5 and F3-60. A cell that was not measured, or has a
  world whose artifacts could not be read, makes the verdict `incomplete`, never a pass.

**Offline caveat:** offline builds use the scripted `perfect_author` and oracle LightRAG indices, so coverage is 1.0 by
construction. Such a report is marked `offline: true` and is not a quality measurement.
"""

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from .config import Config
from .worlds.spec import World

THRESHOLD = 0.95  # D-017
RULE = (
    "D-017: on the dev worlds with descriptive exceptions, authored APG graphs declare >= 95% of the spec's IDs and "
    "LightRAG extraction recovers >= 95% of the policy/exception/procedure IDs as entities, in every gate cell"
)
GATE_CELLS = ("F7-10", "F7-1000", "F3-5", "F3-60")
LIGHTRAG_ID = re.compile(r"\b(?:P-\d+|X-\d+|SOP-[\w-]+)")
OFFLINE_NOTE = "offline: scripted perfect_author and oracle LightRAG indices, so coverage is 1.0 by construction; not a quality measurement"
VERDICTS = ("builder_passes", "fallback_needed", "incomplete")


# --- per world ---------------------------------------------------------------------------------------


def apg_spec_ids(world: World) -> list[str]:
    """The IDs `apg.author` measures `id_coverage` against."""
    return [p.id for p in world.policies] + [x.id for x in world.exceptions] + [p.id for p in world.procedures] + [t.name for t in world.tools]


def lightrag_spec_ids(world: World) -> list[str]:
    """The IDs LightRAG's extraction must recover as entities (tools are not entities)."""
    return [p.id for p in world.policies] + [x.id for x in world.exceptions] + [p.id for p in world.procedures]


def lightrag_entity_file(cfg: Config, world_id: str, kind: str) -> Path:
    """LightRAG keeps a workspace's entities in a JSON KV file under the index directory; find it."""
    base = cfg.indices_dir / "lightrag" / f"{world_id}.{kind}"
    hits = sorted(base.rglob("*full_entities*.json")) or sorted(base.rglob("*vdb_entities*.json"))
    if not hits:
        raise FileNotFoundError(f"no LightRAG entity store under {base}")
    return hits[0]


def lightrag_entity_ids(store: dict | str) -> set[str]:
    """Every spec-style ID (P-…, X-…, SOP-…) anywhere in an entity store."""
    return set(LIGHTRAG_ID.findall(store if isinstance(store, str) else json.dumps(store)))


def lightrag_id_coverage(world: World, names: Iterable[str]) -> float:
    ids, have = lightrag_spec_ids(world), set(names)
    return sum(i in have for i in ids) / len(ids) if ids else 1.0


def apg_report_path(cfg: Config, world_id: str) -> Path:
    from .apg.arm import graph_path

    return graph_path(cfg, world_id, "authored").with_suffix(".report.json")


def exception_style(world: World) -> str:
    return world.entities.get("exception_style") or "descriptive"


def world_quality(world: World, cfg: Config, lightrag_kind: str) -> dict:
    """Both coverages for one built world, with ID counts so cells can pool them. A missing artifact is recorded in
    `errors`, never raised: one unreadable world must not hide the others."""
    row: dict = {
        "world_id": world.id,
        "cell": f"{world.family}-{world.level}",
        "exception_style": exception_style(world),
        "apg": None,
        "lightrag": None,
        "errors": [],
    }
    apg_ids, lgr_ids = apg_spec_ids(world), lightrag_spec_ids(world)
    row["decisive"] = row["exception_style"] == "descriptive" and bool(apg_ids or lgr_ids)
    if apg_ids:
        try:
            report = json.loads(apg_report_path(cfg, world.id).read_text())
            cov = float(report["id_coverage"])
            row["apg"] = {
                "ids": len(apg_ids),
                "covered": round(cov * len(apg_ids)),
                "coverage": cov,
                "declared_ids": report.get("declared_ids"),
                "unresolved_references": report.get("unresolved_references"),
                "author": report.get("author"),
            }
        except (OSError, ValueError, KeyError) as e:
            row["errors"].append(f"apg: no readable authoring report ({type(e).__name__}: {e})")
    if lgr_ids:
        try:
            names = lightrag_entity_ids(lightrag_entity_file(cfg, world.id, lightrag_kind).read_text())
            have = set(names)
            covered = sum(i in have for i in lgr_ids)
            row["lightrag"] = {"ids": len(lgr_ids), "covered": covered, "coverage": covered / len(lgr_ids), "entity_ids": len(names), "kind": lightrag_kind}
        except (OSError, ValueError) as e:
            row["errors"].append(f"lightrag: no readable entity store ({type(e).__name__}: {e})")
    return row


# --- per cell and overall ----------------------------------------------------------------------------


def _pool(rows: Sequence[dict], system: str) -> dict | None:
    parts = [r[system] for r in rows if r[system] is not None]
    if not parts:
        return None
    ids, covered = sum(p["ids"] for p in parts), sum(p["covered"] for p in parts)
    return {"coverage": covered / ids if ids else 1.0, "min_world": min(p["coverage"] for p in parts), "ids": ids, "covered": covered, "worlds": len(parts)}


def _cell_summary(rows: Sequence[dict], threshold: float) -> dict:
    apg, lgr = _pool(rows, "apg"), _pool(rows, "lightrag")
    errors = [f"{r['world_id']}: {e}" for r in rows for e in r["errors"]]
    passes = not errors and apg is not None and lgr is not None and apg["coverage"] >= threshold and lgr["coverage"] >= threshold
    reasons = list(errors)
    if apg is not None and apg["coverage"] < threshold:
        reasons.append(f"APG id_coverage {apg['coverage']:.3f} < {threshold}")
    if lgr is not None and lgr["coverage"] < threshold:
        reasons.append(f"LightRAG ID coverage {lgr['coverage']:.3f} < {threshold}")
    return {"worlds": len(rows), "apg": apg, "lightrag": lgr, "pass": passes, "reasons": reasons}


def aggregate(rows: Sequence[dict], expected_cells: Sequence[str] = GATE_CELLS, threshold: float = THRESHOLD) -> dict:
    """Decisive cells (pooled coverage, pass/fail), diagnostics (other styles and cells; reported only), and the
    verdict: `builder_passes` when every expected cell was measured cleanly and passes, `fallback_needed` when a
    cleanly measured expected cell fails, otherwise `incomplete`."""
    decisive = [r for r in rows if r["decisive"]]
    cells = {c: _cell_summary([r for r in decisive if r["cell"] == c], threshold) for c in dict.fromkeys(r["cell"] for r in decisive)}
    diagnostics: dict[str, dict] = {}
    for r in rows:
        if not r["decisive"] and (r["apg"] is not None or r["lightrag"] is not None or r["errors"]):
            diagnostics.setdefault(f"{r['cell']} ({r['exception_style']})", []).append(r)
    diag = {k: _cell_summary(v, threshold) for k, v in diagnostics.items()}
    missing = [c for c in expected_cells if c not in cells]
    unreadable = [c for c in expected_cells if c in cells and any(r["errors"] for r in decisive if r["cell"] == c)]
    failing = [c for c in expected_cells if c in cells and c not in unreadable and not cells[c]["pass"]]
    if failing:
        verdict = "fallback_needed"
    elif missing or unreadable:
        verdict = "incomplete"
    else:
        verdict = "builder_passes"
    return {
        "threshold": threshold,
        "expected_cells": list(expected_cells),
        "cells": cells,
        "diagnostics": diag,
        "missing_cells": missing,
        "unreadable_cells": unreadable,
        "failing_cells": failing,
        "verdict": verdict,
    }


def recommendation(agg: dict, builder: str, fallback_builder: str, fallback_usd: float | None, offline: bool) -> str:
    cost = f", +${fallback_usd:,.0f} conservative" if fallback_usd is not None else ""
    per_cell = "; ".join(
        f"{c} APG {s['apg']['coverage']:.2f} / LightRAG {s['lightrag']['coverage']:.2f}" for c, s in agg["cells"].items() if s["apg"] and s["lightrag"]
    )
    if agg["verdict"] == "builder_passes":
        text = f"{builder} passes the D-017 dev check ({per_cell})"
    elif agg["verdict"] == "fallback_needed":
        why = "; ".join(f"{c}: {', '.join(agg['cells'][c]['reasons'])}" for c in agg["failing_cells"])
        text = f"Sol fallback needed: {fallback_builder} (needs approval{cost}); {builder} failed the D-017 dev check ({why})"
    else:
        gaps = [f"not measured: {', '.join(agg['missing_cells'])}"] if agg["missing_cells"] else []
        gaps += [f"unreadable artifacts: {', '.join(agg['unreadable_cells'])}"] if agg["unreadable_cells"] else []
        text = f"undecided: the D-017 dev check is incomplete ({'; '.join(gaps)})"
    return f"{'OFFLINE, not a quality measurement: ' if offline else ''}{text}"


def assess(
    worlds: Iterable[World],
    cfg: Config,
    lightrag_kind: str,
    *,
    offline: bool,
    builder: str,
    fallback_builder: str,
    fallback_usd: float | None = None,
    expected_cells: Sequence[str] = GATE_CELLS,
    threshold: float = THRESHOLD,
) -> dict:
    """The full build-quality report for a set of built worlds (see the module docstring)."""
    rows = [world_quality(w, cfg, lightrag_kind) for w in worlds]
    agg = aggregate(rows, expected_cells, threshold)
    return {
        "rule": RULE,
        "offline": offline,
        "note": OFFLINE_NOTE if offline else "measured on the built dev artifacts",
        "builder": builder,
        "fallback_builder": fallback_builder,
        "fallback_extra_usd": fallback_usd,
        "lightrag_kind": lightrag_kind,
        **agg,
        "recommendation": recommendation(agg, builder, fallback_builder, fallback_usd, offline),
        "worlds": rows,
    }


def worlds_per_cell_note(report: dict | None) -> str:
    """D-017's sizing consequence, for the pilot's `test worlds per cell` item."""
    if report is None:
        return "D-017 build check not recorded (build-dev/build_quality.json missing)"
    prefix = "offline " if report.get("offline") else ""
    return {
        "builder_passes": f"{prefix}D-017 build check: the builder passes, so 16 per D-017",
        "fallback_needed": f"{prefix}D-017 build check: Sol fallback, so 12 per D-017",
        "incomplete": f"{prefix}D-017 build check incomplete: analyst decides between 16 and 12",
    }[report["verdict"]]
