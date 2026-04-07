"""Tests for LiDAR feature extraction — synthetic point clouds, no real LAZ.

The production code uses `laspy.read` to load point clouds. For unit
tests we build an `_LAZPoints` struct directly and pass it into the
private helpers, so CI never needs network or laspy.
"""

import tempfile
from pathlib import Path

import numpy as np
import supervision as sv

from forest_pulse.lidar import (
    ASPRS_GROUND,
    ASPRS_HIGH_VEGETATION,
    LiDARFeatures,
    _bounds_key,
    _features_from_points,
    _filter_from_chm,
    _icgc_laz_url,
    _LAZPoints,
    _points_inside_bounds,
    _rasterize_cells,
    bbox_centers_to_world,
    compute_chm_from_laz,
    extract_lidar_features,
    find_tree_tops_from_chm,
)


def _make_points(
    xs, ys, zs, cls_codes, intensities=None, n_returns=None,
) -> _LAZPoints:
    """Build an _LAZPoints from raw Python arrays for tests."""
    n = len(xs)
    return _LAZPoints(
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        z=np.asarray(zs, dtype=np.float32),
        classification=np.asarray(cls_codes, dtype=np.uint8),
        intensity=np.asarray(
            intensities if intensities is not None else [100] * n,
            dtype=np.uint16,
        ),
        return_number=np.asarray([1] * n, dtype=np.uint8),
        number_of_returns=np.asarray(
            n_returns if n_returns is not None else [1] * n,
            dtype=np.uint8,
        ),
    )


# --- _icgc_laz_url ---

def test_icgc_laz_url_verified_montseny_point():
    """The exact URL verified live via curl during OBSERVE phase."""
    url = _icgc_laz_url(450000, 4625000)
    expected = (
        "https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/"
        "full10km4562/lidar-territorial-v3r1-full1km450625-2021-2023.laz"
    )
    assert url == expected


def test_icgc_laz_url_interior_point_same_tile():
    """A point inside the same 1 km tile produces the same URL."""
    url_a = _icgc_laz_url(450000, 4625000)
    url_b = _icgc_laz_url(450500, 4625500)
    assert url_a == url_b


def test_icgc_laz_url_adjacent_tile_differs():
    """Crossing a 1 km boundary produces a different URL."""
    url_a = _icgc_laz_url(450500, 4625500)
    url_b = _icgc_laz_url(451500, 4625500)
    assert url_a != url_b
    assert "full1km450625" in url_a
    assert "full1km451625" in url_b


# --- _points_inside_bounds ---

def test_points_inside_bounds_mask():
    points = _make_points(
        xs=[100, 200, 300, 350],
        ys=[100, 200, 300, 350],
        zs=[1, 2, 3, 4],
        cls_codes=[2, 2, 2, 2],
    )
    mask = _points_inside_bounds(points, (150, 150, 325, 325))
    assert mask.tolist() == [False, True, True, False]


# --- _rasterize_cells ---

def test_rasterize_cells_max_mode_picks_highest():
    """Two points in the same cell → max z wins for DSM."""
    # A 2x2 raster covering (0,0) to (10,10), resolution 5m
    x = np.array([1.0, 2.0, 7.0], dtype=np.float64)
    y = np.array([1.0, 1.0, 8.0], dtype=np.float64)
    z = np.array([10.0, 20.0, 5.0], dtype=np.float32)
    raster = _rasterize_cells(
        x, y, z, (0, 0, 10, 10), resolution_m=5.0,
        shape=(2, 2), mode="max",
    )
    # Points (1,1) and (2,1) are in the bottom-left cell (row 1, col 0)
    # → max = 20. Point (7, 8) is in top-right cell (row 0, col 1) → 5.
    assert raster.shape == (2, 2)
    assert raster[1, 0] == 20.0
    assert raster[0, 1] == 5.0
    # Empty cells → NaN
    assert np.isnan(raster[0, 0])
    assert np.isnan(raster[1, 1])


def test_rasterize_cells_min_mode_picks_lowest():
    """Two points in the same cell → min z wins for DTM."""
    x = np.array([1.0, 2.0], dtype=np.float64)
    y = np.array([1.0, 1.0], dtype=np.float64)
    z = np.array([10.0, 3.0], dtype=np.float32)
    raster = _rasterize_cells(
        x, y, z, (0, 0, 10, 10), resolution_m=5.0,
        shape=(2, 2), mode="min",
    )
    assert raster[1, 0] == 3.0


def test_rasterize_cells_empty_returns_nan_raster():
    raster = _rasterize_cells(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float32),
        (0, 0, 10, 10), resolution_m=5.0, shape=(2, 2), mode="max",
    )
    assert raster.shape == (2, 2)
    assert np.all(np.isnan(raster))


# --- _features_from_points ---

def test_features_extract_all_fields():
    """Synthetic points → LiDARFeatures with all 7 fields populated."""
    # A tree: 10 ground points at z=100m, 50 canopy points from 102-115m
    ground_xs = np.random.default_rng(1).uniform(50, 110, 10)
    ground_ys = np.random.default_rng(2).uniform(50, 110, 10)
    ground_zs = np.full(10, 100.0)

    canopy_xs = np.random.default_rng(3).uniform(50, 110, 50)
    canopy_ys = np.random.default_rng(4).uniform(50, 110, 50)
    canopy_zs = np.linspace(102.0, 115.0, 50)

    xs = np.concatenate([ground_xs, canopy_xs])
    ys = np.concatenate([ground_ys, canopy_ys])
    zs = np.concatenate([ground_zs, canopy_zs])
    cls = [ASPRS_GROUND] * 10 + [ASPRS_HIGH_VEGETATION] * 50
    intensities = [200] * 10 + [450] * 50
    n_returns = [1] * 10 + [2] * 50

    points = _make_points(xs, ys, zs, cls, intensities, n_returns)
    feat = _features_from_points(points, (50, 50, 110, 110), tree_id=7)

    assert feat.tree_id == 7
    assert feat.point_count == 60
    # Heights are above ground z=100 → canopy heights 2-15 m
    assert feat.height_p95_m > 10.0    # near top of canopy
    assert feat.height_p50_m > 1.0     # bulk of points are above 0
    assert feat.vertical_spread_m > 5  # p95 - p5 should be several meters
    assert feat.return_ratio > 0.5     # most points are multi-return
    assert feat.intensity_mean > 300   # blended intensity
    assert feat.intensity_std > 0      # variance between ground and canopy


def test_features_tall_tree_height_above_five_m():
    """A bbox over a tall tree must give height_p95 > 5 m."""
    # 20 canopy points at z=115, ground at z=100 → height above ground = 15 m
    xs = np.linspace(50, 100, 20)
    ys = np.linspace(50, 100, 20)
    zs = np.full(20, 115.0)
    cls = [ASPRS_HIGH_VEGETATION] * 20
    # Add a single ground point for the local reference
    xs = np.append(xs, 75.0)
    ys = np.append(ys, 75.0)
    zs = np.append(zs, 100.0)
    cls = cls + [ASPRS_GROUND]

    points = _make_points(xs, ys, zs, cls)
    feat = _features_from_points(points, (40, 40, 110, 110), tree_id=0)
    assert feat.height_p95_m >= 14.0
    assert feat.point_count == 21


def test_features_empty_bbox_returns_zeros():
    """No points in bbox → LiDARFeatures with all default (zero) fields."""
    points = _make_points(
        xs=[1000, 2000], ys=[1000, 2000], zs=[50, 60],
        cls_codes=[ASPRS_GROUND, ASPRS_HIGH_VEGETATION],
    )
    feat = _features_from_points(points, (0, 0, 100, 100), tree_id=3)
    assert feat.tree_id == 3
    assert feat.point_count == 0
    assert feat.height_p95_m == 0.0
    assert feat.intensity_mean == 0.0


# --- compute_chm_from_laz (integration with mocked laspy.read) ---

def test_compute_chm_produces_heights_with_mock_laz(monkeypatch):
    """compute_chm_from_laz → CHM raster with positive heights."""
    # Build synthetic points: ground at z=100, canopy at z=115 in a bbox
    ground_xs = np.linspace(0, 20, 30)
    ground_ys = np.linspace(0, 20, 30)
    ground_zs = np.full(30, 100.0)
    canopy_xs = np.linspace(0, 20, 50)
    canopy_ys = np.linspace(0, 20, 50)
    canopy_zs = np.full(50, 115.0)

    xs = np.concatenate([ground_xs, canopy_xs])
    ys = np.concatenate([ground_ys, canopy_ys])
    zs = np.concatenate([ground_zs, canopy_zs])
    cls = np.array(
        [ASPRS_GROUND] * 30 + [ASPRS_HIGH_VEGETATION] * 50, dtype=np.uint8,
    )

    fake_points = _LAZPoints(
        x=xs, y=ys, z=zs.astype(np.float32), classification=cls,
        intensity=np.full(80, 100, dtype=np.uint16),
        return_number=np.ones(80, dtype=np.uint8),
        number_of_returns=np.ones(80, dtype=np.uint8),
    )

    # Patch _read_laz_points to avoid needing a real LAZ file
    import forest_pulse.lidar as lidar_mod
    monkeypatch.setattr(lidar_mod, "_read_laz_points", lambda path: fake_points)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        laz_path = tmpdir / "fake.laz"
        laz_path.touch()

        chm_path = compute_chm_from_laz(
            laz_path, bounds=(0, 0, 20, 20), resolution_m=5.0,
            cache_dir=tmpdir,
        )
        assert chm_path.exists()

        import rasterio
        with rasterio.open(chm_path) as src:
            chm = src.read(1)
        # CHM should contain 15 m heights where canopy was placed
        assert chm.max() >= 14.0
        assert chm.min() >= 0.0  # no negatives after clipping


def test_extract_features_integration_with_mock(monkeypatch):
    """extract_lidar_features returns one feature per detection."""
    # Synthetic points in the LEFT HALF of the geographic area.
    # Image is 640x640 px covering (0,0)->(20,20). Left-half pixel bbox
    # (0, 0, 320, 320) maps to geo bbox (x in 0..10, y in 0..20).
    # Right-half pixel bbox (320, 0, 640, 320) maps to (x in 10..20, y in 0..20).
    xs = np.array([2, 4, 6, 8], dtype=np.float64)       # all x < 10
    ys = np.array([10, 12, 15, 18], dtype=np.float64)
    zs = np.array([100, 110, 112, 115], dtype=np.float32)
    cls = np.array(
        [ASPRS_GROUND, ASPRS_HIGH_VEGETATION, ASPRS_HIGH_VEGETATION, ASPRS_HIGH_VEGETATION],
        dtype=np.uint8,
    )
    fake_points = _LAZPoints(
        x=xs, y=ys, z=zs, classification=cls,
        intensity=np.array([100, 200, 250, 300], dtype=np.uint16),
        return_number=np.ones(4, dtype=np.uint8),
        number_of_returns=np.array([1, 2, 2, 1], dtype=np.uint8),
    )
    import forest_pulse.lidar as lidar_mod
    monkeypatch.setattr(lidar_mod, "_read_laz_points", lambda p: fake_points)

    dets = sv.Detections(
        xyxy=np.array([
            [0, 0, 320, 640],      # left half → gets all 4 points
            [320, 0, 640, 640],    # right half → no points
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
    )
    image_bounds = (0.0, 0.0, 20.0, 20.0)  # small test area in meters
    image_size = (640, 640)

    features = extract_lidar_features(
        dets, image_bounds, image_size, Path("fake.laz"),
    )

    assert len(features) == 2
    assert features[0].tree_id == 0
    assert features[0].point_count == 4        # all 4 points inside left half
    assert features[0].height_p95_m > 5        # canopy well above ground
    assert features[1].tree_id == 1
    assert features[1].point_count == 0        # right half has no points


def test_extract_features_empty_detections_returns_empty():
    empty = sv.Detections.empty()
    features = extract_lidar_features(
        empty, (0, 0, 10, 10), (100, 100), Path("fake.laz"),
    )
    assert features == []


# --- _bounds_key ---

def test_bounds_key_is_deterministic():
    k1 = _bounds_key((100, 200, 300, 400))
    k2 = _bounds_key((100, 200, 300, 400))
    assert k1 == k2


def test_bounds_key_differs_for_different_bounds():
    assert _bounds_key((0, 0, 10, 10)) != _bounds_key((0, 0, 20, 20))


# --- LiDARFeatures dataclass defaults ---

def test_lidar_features_defaults():
    f = LiDARFeatures(tree_id=5)
    assert f.tree_id == 5
    assert f.height_p95_m == 0.0
    assert f.point_count == 0


# ============================================================
# Phase 9.5a — tree-top detection + bbox_centers + deterministic filter
# ============================================================


def _identity_transform(resolution_m: float = 0.5):
    """Tiny rasterio.Affine with pixel (col,row) → world (col*res, row*res).

    Y is NOT inverted in this test transform so the assertions stay
    readable. `a` is the x-resolution which find_tree_tops uses to
    size the local-max window.
    """
    from rasterio.transform import Affine
    return Affine(resolution_m, 0.0, 0.0, 0.0, resolution_m, 0.0)


# --- find_tree_tops_from_chm with return_heights ---

def test_find_tree_tops_returns_heights_when_requested():
    """return_heights=True → parallel heights list; default preserves
    old single-return signature (back-compat guarantee).

    Uses a 5x5 plateau at 15 m so the Gaussian smoothing (sigma=1 px)
    preserves the peak value — a single-pixel peak would get smeared
    below the 5 m threshold and we'd find nothing.
    """
    chm = np.zeros((20, 20), dtype=np.float32)
    chm[8:13, 8:13] = 15.0  # 5x5 plateau, peak at center (10, 10)
    transform = _identity_transform(0.5)

    # Old signature: list of positions only
    positions_only = find_tree_tops_from_chm(chm, transform, min_height_m=5.0)
    assert isinstance(positions_only, list)
    assert len(positions_only) == 1

    # New signature with heights
    positions, heights = find_tree_tops_from_chm(
        chm, transform, min_height_m=5.0, return_heights=True,
    )
    assert len(positions) == 1
    assert len(heights) == 1
    # Smoothed value at the peak should be well above threshold
    assert heights[0] >= 5.0


def test_find_tree_tops_empty_chm_with_return_heights():
    """Empty CHM → ([], []) when return_heights=True (not a plain list)."""
    result = find_tree_tops_from_chm(
        np.zeros((0, 0), dtype=np.float32),
        _identity_transform(),
        return_heights=True,
    )
    assert result == ([], [])


# --- bbox_centers_to_world ---

def test_bbox_centers_to_world_known_bounds():
    """Centered bbox in a 640x640 image with bounds (0,0,160,160)."""
    dets = sv.Detections(
        xyxy=np.array([[0, 0, 640, 640]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
    )
    centers = bbox_centers_to_world(dets, (0.0, 0.0, 160.0, 160.0), (640, 640))
    assert centers.shape == (1, 2)
    # bbox covers whole image → center in world = (80, 80)
    assert abs(centers[0, 0] - 80.0) < 1e-6
    assert abs(centers[0, 1] - 80.0) < 1e-6


def test_bbox_centers_to_world_y_is_flipped():
    """Row 0 in image-space maps to y_max in world-space (Y inverted)."""
    # Pixel center at (320, 0) → world should be (80, 160) NOT (80, 0).
    dets = sv.Detections(
        xyxy=np.array([[310, 0, 330, 0]], dtype=np.float32),
        confidence=np.array([0.5], dtype=np.float32),
    )
    centers = bbox_centers_to_world(dets, (0.0, 0.0, 160.0, 160.0), (640, 640))
    assert abs(centers[0, 0] - 80.0) < 1e-6
    assert abs(centers[0, 1] - 160.0) < 1e-6  # row 0 → y_max


def test_bbox_centers_to_world_empty_detections():
    """Empty sv.Detections → zero-row ndarray, no crash."""
    centers = bbox_centers_to_world(
        sv.Detections.empty(), (0.0, 0.0, 100.0, 100.0), (100, 100),
    )
    assert centers.shape == (0, 2)


# --- _filter_from_chm (the test seam for lidar_tree_top_filter) ---

def test_filter_from_chm_empty_detections_passthrough():
    """Empty detections in → empty out, regardless of CHM."""
    chm = np.full((20, 20), 15.0, dtype=np.float32)
    out = _filter_from_chm(
        detections=sv.Detections.empty(),
        image_bounds=(0.0, 0.0, 10.0, 10.0),
        image_size_px=(100, 100),
        chm=chm,
        transform=_identity_transform(0.5),
        tolerance_m=2.0,
        min_height_m=5.0,
    )
    assert len(out) == 0


def test_filter_from_chm_no_tree_tops_drops_everything():
    """CHM below threshold everywhere → every detection dropped."""
    # All-zero CHM → no peaks above 5 m
    chm = np.zeros((20, 20), dtype=np.float32)
    dets = sv.Detections(
        xyxy=np.array([[0, 0, 50, 50], [50, 50, 100, 100]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
    )
    out = _filter_from_chm(
        detections=dets,
        image_bounds=(0.0, 0.0, 10.0, 10.0),
        image_size_px=(100, 100),
        chm=chm,
        transform=_identity_transform(0.5),
        tolerance_m=2.0,
        min_height_m=5.0,
    )
    assert len(out) == 0


def test_filter_from_chm_keeps_matching_drops_far():
    """Two detections: one over a CHM peak, one far from any peak.

    Setup: 20x20 CHM at 0.5 m/px → 10m x 10m patch. Single peak at
    (col=10, row=10) which maps to world (5, 5) with this non-
    Y-inverted identity transform. image_bounds match the patch so
    bbox-center→world also lives in [0, 10].
    """
    chm = np.zeros((20, 20), dtype=np.float32)
    chm[8:13, 8:13] = 15.0  # 5x5 plateau (survives sigma=1 smoothing)

    # Image is 100 px covering (0,0)-(10,10) → 0.1 m/px.
    # Detection A bbox center at pixel (50, 50) → world (5, 5) ≈ peak.
    # Detection B bbox center at pixel (90, 90) → world (9, 1) which is
    # far from the peak at (5, 5) (distance ≈ 5.66 m, well > tolerance).
    # Note: Y is inverted for bbox_centers_to_world (row 0 → y_max=10),
    # so pixel (50, 50) → world (5, 5) and pixel (90, 90) → world (9, 1).
    dets = sv.Detections(
        xyxy=np.array([
            [48, 48, 52, 52],    # center (50, 50)
            [88, 88, 92, 92],    # center (90, 90)
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
    )

    out = _filter_from_chm(
        detections=dets,
        image_bounds=(0.0, 0.0, 10.0, 10.0),
        image_size_px=(100, 100),
        chm=chm,
        transform=_identity_transform(0.5),
        tolerance_m=2.0,
        min_height_m=5.0,
    )
    # Only the matching detection should survive.
    assert len(out) == 1
    # The survivor should be the first detection (tightest match)
    assert out.confidence[0] == 0.9
