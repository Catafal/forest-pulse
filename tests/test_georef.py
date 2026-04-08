"""Tests for georeference() — synthetic detections, no rasters needed."""

import numpy as np
import supervision as sv
from shapely.geometry import Point, Polygon

from forest_pulse.georef import georeference
from forest_pulse.health import HealthScore


def _make_dets(n: int) -> sv.Detections:
    """Build a synthetic sv.Detections with `n` 20x20 boxes along the x axis."""
    xyxy = np.array(
        [[i * 50, 10, i * 50 + 20, 30] for i in range(n)],
        dtype=np.float32,
    )
    confidences = np.array([0.9 - i * 0.05 for i in range(n)], dtype=np.float32)
    return sv.Detections(xyxy=xyxy, confidence=confidences)


def test_georef_basic():
    """5 detections → 5 rows with Point geometry + confidence + bbox cols."""
    dets = _make_dets(5)
    # Image: 640x640 pixels, covering 160m x 160m in EPSG:25831
    bounds = (450000.0, 4600000.0, 450160.0, 4600160.0)
    gdf = georeference(dets, bounds, (640, 640))

    assert len(gdf) == 5
    assert gdf.crs == "EPSG:25831"
    assert "tree_id" in gdf.columns
    assert "confidence" in gdf.columns
    assert "crown_width_m" in gdf.columns
    assert all(gdf.geometry.apply(lambda g: isinstance(g, Point)))
    # Tree IDs should be 0..4
    assert list(gdf["tree_id"]) == [0, 1, 2, 3, 4]


def test_georef_empty():
    """Empty detections → empty GeoDataFrame with the same schema."""
    empty = sv.Detections.empty()
    bounds = (450000.0, 4600000.0, 450160.0, 4600160.0)
    gdf = georeference(empty, bounds, (640, 640))
    assert len(gdf) == 0
    assert "tree_id" in gdf.columns
    assert "crown_width_m" in gdf.columns
    assert gdf.crs == "EPSG:25831"


def test_georef_point_is_bbox_center():
    """Geometry must be the geographic center of the bbox."""
    # Single detection spanning pixels (100, 100) → (300, 300)
    dets = sv.Detections(
        xyxy=np.array([[100.0, 100.0, 300.0, 300.0]], dtype=np.float32),
        confidence=np.array([0.5], dtype=np.float32),
    )
    # Image: 400x400 pixels covering 40m x 40m, so 0.1m/pixel
    bounds = (1000.0, 2000.0, 1040.0, 2040.0)
    gdf = georeference(dets, bounds, (400, 400))

    # Pixel bbox center is (200, 200) → 50% across both axes
    # x_center_geo = 1000 + 0.5 * 40 = 1020
    # y_center_geo = 2040 - 0.5 * 40 = 2020  (Y inverted)
    pt = gdf.geometry.iloc[0]
    assert abs(pt.x - 1020.0) < 0.01
    assert abs(pt.y - 2020.0) < 0.01


def test_georef_crown_area_in_meters():
    """Crown dimensions and area must be in meters (world units), not pixels."""
    dets = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 100.0, 50.0]], dtype=np.float32),
        confidence=np.array([1.0], dtype=np.float32),
    )
    # Image: 1000 px wide covering 100m → 0.1 m/pixel
    bounds = (0.0, 0.0, 100.0, 100.0)
    gdf = georeference(dets, bounds, (1000, 1000))

    # 100 px wide × 0.1 m/px = 10 m; 50 px tall × 0.1 m/px = 5 m
    assert abs(gdf["crown_width_m"].iloc[0] - 10.0) < 0.01
    assert abs(gdf["crown_height_m"].iloc[0] - 5.0) < 0.01
    assert abs(gdf["crown_area_m2"].iloc[0] - 50.0) < 0.01


def test_georef_with_health_scores():
    """When health_scores are supplied, health columns appear in the output."""
    dets = _make_dets(3)
    scores = [
        HealthScore(tree_id=0, grvi=0.25, exg=45.0, label="healthy", confidence=0.9),
        HealthScore(tree_id=1, grvi=0.05, exg=20.0, label="stressed", confidence=0.6),
        HealthScore(tree_id=2, grvi=-0.1, exg=5.0, label="dead", confidence=0.8),
    ]
    bounds = (0.0, 0.0, 160.0, 160.0)
    gdf = georeference(dets, bounds, (640, 640), health_scores=scores)

    assert "health_label" in gdf.columns
    assert "grvi" in gdf.columns
    assert list(gdf["health_label"]) == ["healthy", "stressed", "dead"]
    assert abs(gdf["grvi"].iloc[0] - 0.25) < 1e-4


# ============================================================
# Phase 11b — auto-detection of crown polygons in detections.data
# ============================================================


def test_georef_falls_back_to_points_without_crown_polygon():
    """No crown_polygon in detections.data → Point geometry (Phase 10d)."""
    dets = _make_dets(3)
    # No crown_polygon in data — backward-compat path
    bounds = (0.0, 0.0, 160.0, 160.0)
    gdf = georeference(dets, bounds, (640, 640))
    assert all(gdf.geometry.apply(lambda g: isinstance(g, Point)))


def test_georef_uses_crown_polygons_when_present():
    """When detections.data["crown_polygon"] is present and length-
    matches detections, the output geometries are POLYGONs and
    crown_area_m2 comes from polygon.area.
    """
    dets = _make_dets(2)
    # Build two known polygons in EPSG:25831 world coordinates that
    # the bounds (0, 0, 160, 160) cover.
    polygons = [
        Polygon([(10, 10), (20, 10), (20, 20), (10, 20)]),  # 10×10 = 100 m²
        Polygon([(50, 50), (55, 50), (55, 55), (50, 55)]),  # 5×5 = 25 m²
    ]
    dets.data["crown_polygon"] = polygons
    bounds = (0.0, 0.0, 160.0, 160.0)
    gdf = georeference(dets, bounds, (640, 640))

    # All output geometries are Polygons, not Points
    assert all(gdf.geometry.apply(lambda g: isinstance(g, Polygon)))
    assert len(gdf) == 2
    # crown_area_m2 should match the polygon areas
    assert abs(gdf["crown_area_m2"].iloc[0] - 100.0) < 0.01
    assert abs(gdf["crown_area_m2"].iloc[1] - 25.0) < 0.01
