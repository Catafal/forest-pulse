"""Tests for compare_periods() with synthetic GeoDataFrames."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from forest_pulse.temporal import compare_periods


def _make_trees(coords: list[tuple[float, float]], crs: str = "EPSG:25831",
                areas: list[float] | None = None,
                health: list[str] | None = None) -> gpd.GeoDataFrame:
    """Build a test GeoDataFrame with tree_id, geometry, crown_area_m2."""
    n = len(coords)
    data = {
        "tree_id": list(range(n)),
        "crown_area_m2": areas if areas else [10.0] * n,
    }
    if health is not None:
        data["health_label"] = health
    return gpd.GeoDataFrame(
        data,
        geometry=[Point(x, y) for x, y in coords],
        crs=crs,
    )


def test_compare_periods_exact_match():
    """All trees in the same place → all matched, none missing/new."""
    before = _make_trees([(0, 0), (10, 0), (20, 0)])
    after = _make_trees([(0, 0), (10, 0), (20, 0)])

    report = compare_periods(before, after, match_tolerance_m=2.0)

    assert report.trees_before == 3
    assert report.trees_after == 3
    assert len(report.matched) == 3
    assert report.missing == []
    assert report.new == []
    # All distances should be ~0
    for m in report.matched:
        assert m.distance_m < 0.001


def test_compare_periods_missing_and_new():
    """Some trees vanish, some appear — build the full picture."""
    # Before: 4 trees
    before = _make_trees([(0, 0), (10, 0), (20, 0), (30, 0)])
    # After: trees at 0, 10, 50, 60, 70  (20 and 30 are missing; 50/60/70 are new)
    after = _make_trees([(0, 0), (10, 0), (50, 0), (60, 0), (70, 0)])

    report = compare_periods(before, after, match_tolerance_m=2.0)

    assert report.trees_before == 4
    assert report.trees_after == 5
    assert len(report.matched) == 2         # 2 found in both
    assert sorted(report.missing) == [2, 3] # tree_ids 2 and 3 disappeared
    assert sorted(report.new) == [2, 3, 4]  # tree_ids 2,3,4 in `after` are new


def test_compare_periods_tolerance_respected():
    """A tree 3m away with tolerance=2m should NOT match."""
    before = _make_trees([(0, 0)])
    after = _make_trees([(3, 0)])   # 3m east

    report = compare_periods(before, after, match_tolerance_m=2.0)
    assert len(report.matched) == 0
    assert report.missing == [0]
    assert report.new == [0]

    # With tolerance=5m, it should match
    report_wider = compare_periods(before, after, match_tolerance_m=5.0)
    assert len(report_wider.matched) == 1
    assert report_wider.matched[0].distance_m == pytest.approx(3.0, abs=0.01)


def test_compare_periods_crown_shrinkage_change():
    """crown_area_change reflects (after - before) / before."""
    before = _make_trees([(0, 0)], areas=[20.0])
    after = _make_trees([(0, 0)], areas=[10.0])

    report = compare_periods(before, after)
    assert len(report.matched) == 1
    # (10 - 20) / 20 = -0.5
    assert report.matched[0].crown_area_change == pytest.approx(-0.5, abs=0.001)
    # This is more than 20% shrinkage → declining
    assert len(report.declining_trees) == 1


def test_compare_periods_health_degradation():
    """Trees whose health worsens appear in health_degraded."""
    before = _make_trees([(0, 0), (10, 0)], health=["healthy", "healthy"])
    after = _make_trees([(0, 0), (10, 0)], health=["stressed", "dead"])

    report = compare_periods(before, after)
    assert len(report.matched) == 2
    # Both health transitions are worsening
    assert len(report.health_degraded) == 2


def test_compare_periods_empty_before():
    """No trees before → every tree after is new, none missing."""
    before = _make_trees([])
    after = _make_trees([(0, 0), (10, 0)])
    report = compare_periods(before, after)
    assert report.matched == []
    assert report.missing == []
    assert report.new == [0, 1]


def test_compare_periods_rejects_geographic_crs():
    """Matching by meters requires a projected CRS."""
    before = _make_trees([(0, 0)], crs="EPSG:4326")
    after = _make_trees([(0, 0)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="projected CRS"):
        compare_periods(before, after)
