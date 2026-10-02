"""D-017 build-quality check (`ape.build_quality`): per-world coverage from built artifacts, pooled per gate cell,
the threshold, the verdicts and the recommendation text."""

import pytest

from ape import build_quality as bq
from ape.artifacts import build_artifacts
from ape.config import Config
from ape.worlds.generate import make_world


def _row(cell: str, apg: tuple[int, int] | None, lgr: tuple[int, int] | None, style: str = "descriptive", errors=(), world: str | None = None) -> dict:
    """A crafted world row: (covered, ids) per system."""
    part = lambda c: None if c is None else {"covered": c[0], "ids": c[1], "coverage": c[0] / c[1]}  # noqa: E731
    return {
        "world_id": world or f"{cell}-{style}",
        "cell": cell,
        "exception_style": style,
        "decisive": style == "descriptive",
        "apg": part(apg),
        "lightrag": part(lgr),
        "errors": list(errors),
    }


def _all_cells(apg=(100, 100), lgr=(100, 100)) -> list[dict]:
    return [_row(c, apg, lgr) for c in bq.GATE_CELLS]


def test_cells_pool_ids_over_worlds_and_the_threshold_is_inclusive():
    rows = [_row("F7-10", (19, 20), (20, 20), world="a"), _row("F7-10", (19, 20), (20, 20), world="b")] + _all_cells()[1:]
    agg = bq.aggregate(rows)
    cell = agg["cells"]["F7-10"]
    assert cell["apg"]["coverage"] == pytest.approx(0.95) and cell["apg"]["worlds"] == 2 and cell["apg"]["ids"] == 40
    assert cell["pass"] and agg["verdict"] == "builder_passes"
    # One ID short of 95% in the pooled cell fails it, and a failing gate cell means the fallback.
    rows[0] = _row("F7-10", (18, 20), (20, 20), world="a")
    agg = bq.aggregate(rows)
    assert not agg["cells"]["F7-10"]["pass"] and agg["failing_cells"] == ["F7-10"] and agg["verdict"] == "fallback_needed"
    assert "APG id_coverage 0.925 < 0.95" in agg["cells"]["F7-10"]["reasons"][0]
    assert agg["cells"]["F7-10"]["apg"]["min_world"] == pytest.approx(0.9)


def test_lightrag_coverage_fails_a_cell_on_its_own():
    rows = _all_cells()
    rows[1] = _row("F7-1000", (100, 100), (90, 100))
    agg = bq.aggregate(rows)
    assert agg["verdict"] == "fallback_needed" and agg["failing_cells"] == ["F7-1000"]
    assert agg["cells"]["F7-1000"]["reasons"] == ["LightRAG ID coverage 0.900 < 0.95"]


def test_missing_or_unreadable_gate_cells_are_incomplete_never_a_pass():
    agg = bq.aggregate(_all_cells()[:2])  # e.g. a smoke run: only some gate cells built
    assert agg["verdict"] == "incomplete" and agg["missing_cells"] == ["F3-5", "F3-60"]
    rows = _all_cells()
    rows[2] = _row("F3-5", None, (10, 10), errors=["apg: no readable authoring report"])
    agg = bq.aggregate(rows)
    assert agg["verdict"] == "incomplete" and agg["unreadable_cells"] == ["F3-5"] and not agg["cells"]["F3-5"]["pass"]


def test_diagnostic_styles_are_reported_but_never_decide():
    rows = _all_cells() + [_row("F7-1000", (50, 100), (40, 100), style="messy"), _row("F7-10", (60, 100), (60, 100), style="id_only")]
    agg = bq.aggregate(rows)
    assert agg["verdict"] == "builder_passes"
    assert set(agg["diagnostics"]) == {"F7-1000 (messy)", "F7-10 (id_only)"} and not agg["diagnostics"]["F7-1000 (messy)"]["pass"]


def test_recommendations_name_the_builder_the_fallback_and_its_cost():
    builder, fallback = "gpt-6-luna (high effort)", "gpt-6-sol (medium effort)"
    passing = bq.recommendation(bq.aggregate(_all_cells()), builder, fallback, 957.0, offline=False)
    assert passing.startswith("gpt-6-luna (high effort) passes the D-017 dev check (F7-10 APG 1.00 / LightRAG 1.00;")
    rows = _all_cells()
    rows[1] = _row("F7-1000", (80, 100), (100, 100))
    failing = bq.recommendation(bq.aggregate(rows), builder, fallback, 957.0, offline=False)
    assert failing.startswith("Sol fallback needed: gpt-6-sol (medium effort) (needs approval, +$957 conservative)") and "F7-1000" in failing
    assert "not measured: F3-5, F3-60" in bq.recommendation(bq.aggregate(_all_cells()[:2]), builder, fallback, None, offline=False)
    assert bq.recommendation(bq.aggregate(_all_cells()), builder, fallback, 957.0, offline=True).startswith("OFFLINE, not a quality measurement: ")
    # The sizing consequence for the pilot's `test worlds per cell` item (D-017: 16 if the builder passes, else 12).
    assert bq.worlds_per_cell_note({"verdict": "builder_passes", "offline": False}).endswith("so 16 per D-017")
    assert bq.worlds_per_cell_note({"verdict": "fallback_needed", "offline": False}).endswith("so 12 per D-017")
    assert "analyst decides" in bq.worlds_per_cell_note({"verdict": "incomplete", "offline": True})
    assert "missing" in bq.worlds_per_cell_note(None)


@pytest.fixture
def offline_env(tmp_path, monkeypatch):
    for k, v in {"APE_WORLDS": "worlds", "APE_CACHE": "cache", "APE_INDICES": "indices"}.items():
        monkeypatch.setenv(k, str(tmp_path / v))
    monkeypatch.setenv("APE_EMBEDDINGS", "fake")
    return tmp_path


def test_world_quality_reads_the_built_artifacts_and_records_what_is_missing(offline_env):
    cfg = Config()
    f7, f5 = make_world("F7", "10", "dev", 0, 2, True, "descriptive"), make_world("F5", "1hop", "dev", 0, 2)
    for w in (f7, f5):
        w.save(cfg.world_path(w.id))
    results = build_artifacts([cfg.world_path(f7.id), cfg.world_path(f5.id)], ("chunks", "apg", "lightrag"), "oracle", True, workers=1)
    assert all(r.status != "failed" for r in results), [r.summary() for r in results]
    row = bq.world_quality(f7, cfg, "oracle")
    assert row["decisive"] and row["cell"] == "F7-10" and not row["errors"]
    assert row["apg"]["ids"] == len(bq.apg_spec_ids(f7)) and row["apg"]["coverage"] == 1.0, "offline: perfect_author"
    assert row["lightrag"]["ids"] == len(bq.lightrag_spec_ids(f7)) and row["lightrag"]["coverage"] == 1.0, "offline: oracle index"
    # F5 has no policy, procedure or tool IDs: nothing to measure, so never decisive.
    assert not bq.world_quality(f5, cfg, "oracle")["decisive"]
    # A missing authoring report is recorded on the row, not raised.
    bq.apg_report_path(cfg, f7.id).unlink()
    row = bq.world_quality(f7, cfg, "oracle")
    assert row["apg"] is None and row["errors"] and row["errors"][0].startswith("apg: no readable authoring report")
    report = bq.assess([f7, f5], cfg, "oracle", offline=True, builder="b", fallback_builder="f", expected_cells=["F7-10"])
    assert report["verdict"] == "incomplete" and report["unreadable_cells"] == ["F7-10"] and report["offline"] is True
    assert "not a quality measurement" in report["note"]
