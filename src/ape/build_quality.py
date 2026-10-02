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

**Definitions:**
- **APG coverage:** the share of the spec's policy, exception, procedure and tool IDs that some authored leaf declares.
  This is the `id_coverage` in the graph's report.
- **LightRAG coverage:** the share of the spec's policy, exception and procedure IDs that appear among the index's
  entity names (tools are not entities).
- **Same matching for both** (RELIABILITY_REVIEW K4): identifiers are normalised by `normalize_ident` (the ID inside
  "Policy P-1035" or "p-1035 (Refunds)" is `P-1035`; a tool name inside backticks or "Tool x" is `x`) and compared
  case-insensitively (`canon_id`). An author or extractor that names IDs with a prefix word is not penalised, and
  the two systems are measured against the same rule. Only IDs the documents render count (`rendered_spec_ids`):
  messy worlds render none, so their coverage is measured on facts (`fact_coverage`), not IDs.
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
# Spec identifiers: P-/X-/E-/EC-/SUP-/M-... numbers, and SOP-<domain>-<REGION> procedure IDs. Case-insensitive.
SPEC_ID = re.compile(r"(?<![\w-])(?:SOP-[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[A-Za-z]{1,4}-\d+)(?![\w-])", re.I)
LIGHTRAG_ID = SPEC_ID  # kept for callers of the earlier name
_TOOL_NAME = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_KIND_WORD = re.compile(r"^(?:policy|exception|procedure|tool|rule|memo|sop)\b[\s:#-]*", re.I)
OFFLINE_NOTE = "offline: scripted perfect_author and oracle LightRAG indices, so coverage is 1.0 by construction; not a quality measurement"
VERDICTS = ("builder_passes", "fallback_needed", "incomplete")


# --- identifiers -------------------------------------------------------------------------------------


def normalize_ident(raw: str) -> str:
    """The identifier inside an author's or extractor's spelling: "Policy P-1035" -> "P-1035", "`p-1035`" ->
    "P-1035", "Tool duplicate_charge_credit()" -> "duplicate_charge_credit". Anything else is returned trimmed,
    without a leading kind word. Number-style IDs get an upper-case prefix; SOP IDs and tool names keep their case
    (compare with `canon_id`)."""
    s = str(raw).strip().strip("`'\"").strip()
    m = SPEC_ID.search(s)
    if m:
        ident = m.group(0)
        head, _, tail = ident.partition("-")
        return f"{head.upper()}-{tail}" if tail.isdigit() else ("SOP-" + tail if head.upper() == "SOP" else ident)
    m = _TOOL_NAME.search(s)
    if m:
        return m.group(0)
    return _KIND_WORD.sub("", s).strip(" .;:,()[]")


def canon_id(raw: str) -> str:
    """The comparison key of an identifier: normalised, then case-folded."""
    return normalize_ident(raw).casefold()


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
    """Every spec-style ID (P-…, X-…, SOP-…, any case) anywhere in an entity store, normalised (`normalize_ident`)."""
    return {normalize_ident(m) for m in SPEC_ID.findall(store if isinstance(store, str) else json.dumps(store))}


def rendered_spec_ids(world: World, ids: Sequence[str]) -> list[str]:
    """The `ids` the knowledge base actually renders (messy worlds render none: their documents carry no IDs, so an ID
    coverage would measure nothing there; fact coverage does)."""
    text = "\n".join(p.text for d in world.documents for p in d.paragraphs)
    present = {canon_id(m) for m in SPEC_ID.findall(text)} | set(_TOOL_NAME.findall(text.casefold()))
    return [i for i in ids if canon_id(i) in present]


def id_coverage(spec_ids: Sequence[str], found: Iterable[str]) -> float | None:
    """Share of `spec_ids` among `found`, both compared by `canon_id`; None when there is nothing to cover."""
    if not spec_ids:
        return None
    have = {canon_id(i) for i in found}
    return sum(canon_id(i) in have for i in spec_ids) / len(spec_ids)


def lightrag_id_coverage(world: World, names: Iterable[str]) -> float:
    cov = id_coverage(rendered_spec_ids(world, lightrag_spec_ids(world)), names)
    return 1.0 if cov is None else cov


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
    apg_ids, lgr_ids = rendered_spec_ids(world, apg_spec_ids(world)), rendered_spec_ids(world, lightrag_spec_ids(world))
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
            have = {canon_id(n) for n in names}
            covered = sum(canon_id(i) in have for i in lgr_ids)
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


def builder_roles(profile) -> tuple[object, object | None, bool]:
    """(the role spec that builds, the D-017 fallback role spec or None, whether the fallback is the one building),
    as `ape.models.build_settings` chooses (APE_BUILD_FALLBACK=1)."""
    import os

    fallback = profile.roles.get("build_fallback")
    using = os.environ.get("APE_BUILD_FALLBACK") == "1" and fallback is not None
    return (fallback if using else profile.role("build")), fallback, using


# --- build health: every build phase (dev, pilot, test) ---------------------------------------------


def world_health(world: World, cfg: Config, lightrag_kind: str, threshold: float = THRESHOLD) -> dict:
    """Degradation signals of one built world (RELIABILITY_REVIEW K2): lost chunks in authoring or extraction, and
    coverage below `threshold`. Unlike D-017's verdict (dev only) this runs after every build phase, and it only
    warns: a degraded world is reported, never silently treated as fine."""
    from .lgr.common import index_dir, read_manifest

    row: dict = {"world_id": world.id, "cell": f"{world.family}-{world.level}", "exception_style": exception_style(world), "apg": None, "lightrag": None, "warnings": [], "errors": []}
    try:
        report = json.loads(apg_report_path(cfg, world.id).read_text())
        row["apg"] = {k: report.get(k) for k in ("leaves", "chunks", "chunks_without_units", "lost_chunks", "id_coverage", "fact_coverage", "author")}
        lost = report.get("lost_chunks") or {}
        if lost:
            row["warnings"].append(f"APG: {len(lost)} lost chunk(s) {sorted(lost)[:5]}")
        for key in ("id_coverage", "fact_coverage"):
            if report.get(key) is not None and report[key] < threshold:
                row["warnings"].append(f"APG {key} {report[key]:.3f} < {threshold}")
    except (OSError, ValueError) as e:
        row["errors"].append(f"apg: no readable authoring report ({type(e).__name__}: {e})")
    try:
        manifest = read_manifest(index_dir(cfg, world.id, lightrag_kind))
        extraction = manifest.get("extraction")
        lgr: dict = {"kind": lightrag_kind, "extraction": extraction, "id_coverage": None}
        if lgr_ids := rendered_spec_ids(world, lightrag_spec_ids(world)):
            names = lightrag_entity_ids(lightrag_entity_file(cfg, world.id, lightrag_kind).read_text())
            lgr["id_coverage"] = id_coverage(lgr_ids, names)
        row["lightrag"] = lgr
        if extraction and extraction.get("lost"):
            row["warnings"].append(f"LightRAG: {extraction['lost']} lost chunk(s) (FAILED {extraction['failed_docs'][:5]}, no entities {extraction['empty_docs'][:5]})")
        if lgr["id_coverage"] is not None and lgr["id_coverage"] < threshold:
            row["warnings"].append(f"LightRAG ID coverage {lgr['id_coverage']:.3f} < {threshold}")
    except (OSError, ValueError, KeyError) as e:
        row["errors"].append(f"lightrag: no readable index ({type(e).__name__}: {e})")
    return row


def build_health(worlds: Iterable[World], cfg: Config, lightrag_kind: str, *, offline: bool, threshold: float = THRESHOLD) -> dict:
    """`world_health` for every world of a build phase, with the flagged worlds and their warnings."""
    rows = [world_health(w, cfg, lightrag_kind, threshold) for w in worlds]
    flagged = [r for r in rows if r["warnings"] or r["errors"]]
    return {
        "threshold": threshold,
        "offline": offline,
        "note": OFFLINE_NOTE if offline else "measured on the built artifacts",
        "worlds_checked": len(rows),
        "flagged_worlds": [r["world_id"] for r in flagged],
        "warnings": [f"{r['world_id']}: {w}" for r in flagged for w in r["warnings"] + r["errors"]],
        "worlds": rows,
    }
