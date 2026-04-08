"""Tests for forest_pulse.crowns — watershed crown segmentation.

Synthetic CHMs only. No real LAZ files. Tests cover:
  - Single peak → single polygon containing the marker
  - Two peaks → two non-overlapping polygons, larger gaussian wins
  - Out-of-bounds marker → fallback circle
  - Empty CHM → fallback circles
  - None transform → fallback circles
  - Empty input → empty output
  - Oversized basin → fallback circle
  - Polygon validity (every output is a valid shapely Polygon)
"""

from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from forest_pulse.crowns import (
    DEFAULT_FALLBACK_CROWN_RADIUS_M,
    _build_marker_raster,
    _chm_to_watershed_cost,
    _fallback_circle,
    segment_crowns_watershed,
)

# ============================================================
# Helpers
# ============================================================


def _make_gaussian_chm(
    shape: tuple[int, int],
    peaks: list[tuple[int, int, float, float]],
) -> np.ndarray:
    """Build a synthetic CHM with N gaussian bumps.

    Args:
        shape: (rows, cols)
        peaks: list of (row, col, height_m, sigma) tuples

    Returns:
        2D float32 CHM array.
    """
    chm = np.zeros(shape, dtype=np.float32)
    y, x = np.ogrid[:shape[0], :shape[1]]
    for row, col, height, sigma in peaks:
        chm += height * np.exp(
            -((x - col) ** 2 + (y - row) ** 2) / (2 * sigma ** 2)
        )
    return chm


def _identity_transform_05m():
    """Build a 0.5 m/px affine transform with origin at world (0, 20).

    For a 40x40 raster this gives world bounds (0, 0)-(20, 20).
    Note: from_origin(west, north, xres, yres) → row 0 maps to
    y_max=20, row 40 maps to y_min=0. Standard rasterio convention.
    """
    return from_origin(0.0, 20.0, 0.5, 0.5)


# ============================================================
# segment_crowns_watershed — happy paths
# ============================================================


def test_single_peak_produces_single_polygon():
    """A single gaussian bump → one polygon containing the peak."""
    chm = _make_gaussian_chm((40, 40), [(20, 20, 15.0, 5.0)])
    transform = _identity_transform_05m()
    # Pixel (20, 20) → world (10, 10) with this transform
    polys = segment_crowns_watershed(
        chm=chm,
        transform=transform,
        tree_tops_world=[(10.0, 10.0)],
    )
    assert len(polys) == 1
    p = polys[0]
    assert isinstance(p, Polygon)
    assert p.is_valid
    assert p.contains(_at(10.0, 10.0)) or p.distance(_at(10.0, 10.0)) < 0.5
    assert 5.0 < p.area < 200.0


def test_two_peaks_produce_non_overlapping_polygons():
    """Two well-separated gaussian bumps → two distinct polygons."""
    chm = _make_gaussian_chm(
        (40, 40),
        [
            (10, 10, 15.0, 4.0),  # at world (5, 15) — taller, wider
            (28, 28, 10.0, 3.0),  # at world (14, 6)  — shorter, narrower
        ],
    )
    transform = _identity_transform_05m()
    polys = segment_crowns_watershed(
        chm=chm,
        transform=transform,
        tree_tops_world=[(5.0, 15.0), (14.0, 6.0)],
    )
    assert len(polys) == 2
    assert all(isinstance(p, Polygon) for p in polys)
    assert all(p.is_valid for p in polys)
    # The polygons should not overlap (watershed basins are disjoint)
    intersection = polys[0].intersection(polys[1])
    assert intersection.area < 0.01  # negligible overlap
    # The taller, wider gaussian should have the bigger basin
    assert polys[0].area > polys[1].area


def test_each_polygon_contains_or_is_near_its_marker():
    """Each output polygon should contain (or be very close to) its
    input world position. Watershed basins are seeded by markers, so
    the marker pixel is inside its own basin by construction.
    """
    chm = _make_gaussian_chm(
        (40, 40),
        [(15, 15, 12.0, 3.0), (25, 25, 12.0, 3.0)],
    )
    transform = _identity_transform_05m()
    tops = [(7.5, 12.5), (12.5, 7.5)]
    polys = segment_crowns_watershed(
        chm=chm, transform=transform, tree_tops_world=tops,
    )
    for top, poly in zip(tops, polys):
        marker_pt = _at(*top)
        # Either contains the marker, or is within a half-pixel of it
        assert poly.contains(marker_pt) or poly.distance(marker_pt) < 1.0


# ============================================================
# Fallback paths
# ============================================================


def test_empty_input_returns_empty_list():
    chm = _make_gaussian_chm((20, 20), [(10, 10, 10.0, 3.0)])
    transform = _identity_transform_05m()
    polys = segment_crowns_watershed(
        chm=chm, transform=transform, tree_tops_world=[],
    )
    assert polys == []


def test_none_transform_returns_all_fallback_circles():
    """transform=None (unit-test scenario with stubbed CHMs) →
    every input gets a fallback circle.
    """
    chm = np.zeros((10, 10), dtype=np.float32)
    polys = segment_crowns_watershed(
        chm=chm,
        transform=None,
        tree_tops_world=[(5.0, 5.0), (3.0, 3.0)],
    )
    assert len(polys) == 2
    # Fallback circles should be ~ π * r² ≈ 19.6 m² for r=2.5 m
    expected_area = np.pi * DEFAULT_FALLBACK_CROWN_RADIUS_M ** 2
    for p in polys:
        assert isinstance(p, Polygon)
        assert p.is_valid
        # 16-segment buffer is slightly less than the true circle area
        assert abs(p.area - expected_area) / expected_area < 0.05


def test_marker_outside_chm_bounds_uses_fallback():
    """A tree-top whose pixel projection is outside the raster gets
    a fallback circle, while in-bounds tree-tops get real basins.
    """
    chm = _make_gaussian_chm((40, 40), [(20, 20, 15.0, 5.0)])
    transform = _identity_transform_05m()  # world bounds (0, 0)-(20, 20)
    tops = [
        (10.0, 10.0),    # in-bounds — real basin
        (50.0, 50.0),    # way outside — fallback
        (-5.0, -5.0),    # also outside — fallback
    ]
    polys = segment_crowns_watershed(
        chm=chm, transform=transform, tree_tops_world=tops,
    )
    assert len(polys) == 3
    # First one should be a real basin (much smaller than the
    # fallback would be? No — fallback area ≈ 19.6, basin could be
    # larger). Just check it exists and is valid.
    assert polys[0].is_valid

    # Out-of-bounds ones should be fallback circles centered on
    # their world positions
    fallback_area = np.pi * DEFAULT_FALLBACK_CROWN_RADIUS_M ** 2
    assert abs(polys[1].area - fallback_area) / fallback_area < 0.05
    assert abs(polys[2].area - fallback_area) / fallback_area < 0.05
    # And they should be centered on the input world positions
    # (use approximate comparison — buffer centroids drift by ~1e-15)
    assert abs(polys[1].centroid.x - 50.0) < 1e-9
    assert abs(polys[1].centroid.y - 50.0) < 1e-9


def test_chm_below_min_height_yields_fallbacks():
    """A CHM that's entirely below min_height → all basins get
    masked → all markers fall back to circles.
    """
    chm = np.full((40, 40), 2.0, dtype=np.float32)  # all below 5 m
    transform = _identity_transform_05m()
    polys = segment_crowns_watershed(
        chm=chm,
        transform=transform,
        tree_tops_world=[(10.0, 10.0), (5.0, 5.0)],
        min_height_m=5.0,
    )
    assert len(polys) == 2
    fallback_area = np.pi * DEFAULT_FALLBACK_CROWN_RADIUS_M ** 2
    for p in polys:
        assert abs(p.area - fallback_area) / fallback_area < 0.05


def test_oversized_basin_replaced_with_fallback():
    """A basin exceeding max_crown_area_m2 → fallback circle.

    Build a CHM with a single large flat plateau. Watershed will
    assign the entire plateau to the marker, producing a basin
    much bigger than max_crown_area.
    """
    chm = np.full((40, 40), 15.0, dtype=np.float32)  # uniform 15 m
    transform = _identity_transform_05m()  # 0.5 m/px → 20×20 m total
    # Total area = 400 m². With max_crown_area_m2=50, the watershed
    # basin (covering everything) is way too big → fallback.
    polys = segment_crowns_watershed(
        chm=chm,
        transform=transform,
        tree_tops_world=[(10.0, 10.0)],
        max_crown_area_m2=50.0,
    )
    assert len(polys) == 1
    fallback_area = np.pi * DEFAULT_FALLBACK_CROWN_RADIUS_M ** 2
    assert abs(polys[0].area - fallback_area) / fallback_area < 0.05


def test_empty_chm_yields_fallbacks():
    """A 0×0 CHM → all fallbacks."""
    polys = segment_crowns_watershed(
        chm=np.zeros((0, 0), dtype=np.float32),
        transform=_identity_transform_05m(),
        tree_tops_world=[(5.0, 5.0)],
    )
    assert len(polys) == 1
    assert polys[0].is_valid


# ============================================================
# Polygon validity
# ============================================================


def test_all_returned_polygons_are_valid():
    """Comprehensive validity check across multiple peaks."""
    chm = _make_gaussian_chm(
        (60, 60),
        [
            (10, 10, 15.0, 4.0),
            (10, 50, 12.0, 3.0),
            (50, 10, 18.0, 5.0),
            (50, 50, 10.0, 3.0),
            (30, 30, 14.0, 4.0),
        ],
    )
    transform = from_origin(0.0, 30.0, 0.5, 0.5)  # world bounds (0,0)-(30,30)
    tops = [
        (5.0, 25.0), (25.0, 25.0), (5.0, 5.0),
        (25.0, 5.0), (15.0, 15.0),
    ]
    polys = segment_crowns_watershed(
        chm=chm, transform=transform, tree_tops_world=tops,
    )
    assert len(polys) == 5
    for p in polys:
        assert isinstance(p, Polygon)
        assert p.is_valid
        assert p.area > 0


# ============================================================
# Helpers under test
# ============================================================


def test_fallback_circle_area_matches_radius():
    """Fallback circle area ≈ π · r² (within buffer approximation error)."""
    p = _fallback_circle((10.0, 20.0), radius_m=3.0)
    expected = np.pi * 9.0
    # 16-quad-segment buffer underestimates area slightly
    assert abs(p.area - expected) / expected < 0.05
    # Centroid drift in float math is ~1e-15; use approximate equality
    assert abs(p.centroid.x - 10.0) < 1e-9
    assert abs(p.centroid.y - 20.0) < 1e-9


def test_build_marker_raster_in_bounds_and_oob():
    """In-bounds positions populate the raster; OOB go to in_bounds=False."""
    transform = _identity_transform_05m()  # world (0, 0)-(20, 20)
    markers, in_bounds = _build_marker_raster(
        shape=(40, 40),
        transform=transform,
        tree_tops_world=[
            (10.0, 10.0),     # in-bounds → label 1
            (5.0, 5.0),       # in-bounds → label 2
            (-100.0, 0.0),    # OOB → in_bounds[2] = False
        ],
    )
    assert in_bounds == [True, True, False]
    # Two markers placed
    assert (markers > 0).sum() == 2
    assert markers.max() == 2


def test_chm_to_watershed_cost_inverts_correctly():
    """Tree-top pixels should have low cost; gap pixels max cost."""
    chm = np.array([
        [15.0, 0.0],
        [0.0, 10.0],
    ], dtype=np.float32)
    cost = _chm_to_watershed_cost(chm, min_height_m=5.0)
    # max_h = 15, so:
    #   chm=15 → cost = (1 - 15/15) * 255 = 0  (lowest)
    #   chm=10 → cost = (1 - 10/15) * 255 ≈ 85
    #   chm=0  → forced to 255 (below threshold)
    assert cost[0, 0] == 0
    assert cost[1, 1] == 85
    assert cost[0, 1] == 255  # below threshold
    assert cost[1, 0] == 255  # below threshold


# ============================================================
# Helpers
# ============================================================


def _at(x: float, y: float):
    """Build a shapely Point at the given coordinates."""
    from shapely.geometry import Point
    return Point(x, y)
