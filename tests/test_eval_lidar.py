"""Tests for autoresearch/eval_lidar.py — synthetic CHMs and detections.

No real LAZ or trained models required. The pure helpers (find_tree_tops,
match_predictions_to_truth, EvalResult.from_counts) are tested with
synthetic numpy arrays. The end-to-end `evaluate_patch_against_lidar`
test uses monkeypatch to mock CHM loading.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import supervision as sv

# Make the autoresearch directory importable as a top-level module path.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch"))

from eval_lidar import (  # noqa: E402
    EvalResult,
    _detection_centers_world,
    evaluate_patch_against_lidar,
    find_tree_tops_from_chm,
    match_predictions_to_truth,
)

# ============================================================
# Helpers for synthetic CHM construction
# ============================================================


def _identity_transform(resolution_m: float = 0.5):
    """Build a rasterio.Affine identity-like transform.

    Pixel (col, row) → world (col*res, row*res). Y is NOT inverted in
    this test transform — keeps the assertions easy to read.
    """
    from rasterio.transform import Affine
    return Affine(resolution_m, 0.0, 0.0, 0.0, resolution_m, 0.0)


def _chm_with_peaks(
    shape: tuple[int, int],
    peaks: list[tuple[int, int, float]],
    background: float = 1.0,
) -> np.ndarray:
    """Build a synthetic CHM with explicit peaks at given (row, col, height).

    Background is filled with `background` so the local-max filter has
    well-defined behavior. Each peak is set to its own value plus a small
    Gaussian halo so smoothing doesn't kill it.
    """
    chm = np.full(shape, background, dtype=np.float32)
    for row, col, h in peaks:
        # Set a 3x3 patch of values at this peak so smoothing doesn't
        # erase it. Center value is the max.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                    chm[rr, cc] = max(chm[rr, cc], h * 0.85)
        chm[row, col] = h
    return chm


# ============================================================
# EvalResult.from_counts
# ============================================================


def test_eval_result_perfect_match():
    r = EvalResult.from_counts(n_predictions=5, n_truth=5, n_true_positive=5)
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0
    assert r.n_false_positive == 0
    assert r.n_false_negative == 0


def test_eval_result_zero_predictions_no_division_error():
    r = EvalResult.from_counts(n_predictions=0, n_truth=10, n_true_positive=0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.n_false_negative == 10


def test_eval_result_zero_truth_no_division_error():
    r = EvalResult.from_counts(n_predictions=8, n_truth=0, n_true_positive=0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.n_false_positive == 8


def test_eval_result_partial_match_5_truth_3_match():
    """5 truth, 3 predictions, all 3 match → P=1.0 R=0.6 F1=0.75."""
    r = EvalResult.from_counts(n_predictions=3, n_truth=5, n_true_positive=3)
    assert r.precision == 1.0
    assert r.recall == 0.6
    assert r.f1 == 0.75
    assert r.n_false_positive == 0
    assert r.n_false_negative == 2


def test_eval_result_extra_predictions_become_fp():
    """3 truth, 5 predictions, all 3 truth matched → P=0.6 R=1.0 F1=0.75."""
    r = EvalResult.from_counts(n_predictions=5, n_truth=3, n_true_positive=3)
    assert r.precision == 0.6
    assert r.recall == 1.0
    assert r.f1 == 0.75


# ============================================================
# find_tree_tops_from_chm
# ============================================================


def test_find_tree_tops_three_known_peaks():
    """Synthetic CHM with 3 well-separated peaks → 3 detected."""
    chm = _chm_with_peaks(
        shape=(100, 100),
        peaks=[(20, 20, 12.0), (50, 60, 18.0), (80, 30, 8.0)],
    )
    transform = _identity_transform(0.5)
    peaks = find_tree_tops_from_chm(
        chm, transform, min_height_m=5.0, min_distance_m=2.0,
    )
    assert len(peaks) == 3
    # Peak coordinates correspond to (col, row) → (col*0.5, row*0.5)
    expected_xy = {
        (20 * 0.5 + 0.25, 20 * 0.5 + 0.25),
        (60 * 0.5 + 0.25, 50 * 0.5 + 0.25),
        (30 * 0.5 + 0.25, 80 * 0.5 + 0.25),
    }
    found_xy = {(round(x, 2), round(y, 2)) for (x, y) in peaks}
    assert found_xy == expected_xy


def test_find_tree_tops_below_threshold_dropped():
    """Peaks below the height threshold must be filtered out."""
    chm = _chm_with_peaks(
        shape=(60, 60),
        peaks=[(20, 20, 3.0), (40, 40, 12.0)],  # 3 m is a shrub
    )
    transform = _identity_transform(0.5)
    peaks = find_tree_tops_from_chm(chm, transform, min_height_m=5.0)
    assert len(peaks) == 1
    # The kept peak should be the 12 m one
    assert peaks[0][1] > 15.0  # row 40 → y = 20 m world


def test_find_tree_tops_min_distance_dedupes_close_peaks():
    """Two peaks within min_distance should collapse to ONE.

    The local-max filter window covers both, only the higher one survives.
    """
    chm = _chm_with_peaks(
        shape=(60, 60),
        peaks=[(30, 30, 10.0), (30, 32, 12.0)],  # 1 px apart at 0.5 m = 0.5 m
    )
    transform = _identity_transform(0.5)
    peaks = find_tree_tops_from_chm(
        chm, transform, min_height_m=5.0, min_distance_m=3.0,
    )
    assert len(peaks) == 1


def test_find_tree_tops_empty_chm():
    chm = np.zeros((10, 10), dtype=np.float32)
    transform = _identity_transform(0.5)
    peaks = find_tree_tops_from_chm(chm, transform)
    assert peaks == []


def test_find_tree_tops_handles_nan_values():
    """NaN cells must not be reported as peaks."""
    chm = np.full((20, 20), np.nan, dtype=np.float32)
    chm[10, 10] = 15.0
    chm[9:12, 9:12] = 12.0  # halo
    chm[10, 10] = 15.0
    transform = _identity_transform(0.5)
    peaks = find_tree_tops_from_chm(chm, transform, min_height_m=5.0)
    # Exactly one peak — at (10, 10)
    assert len(peaks) == 1


# ============================================================
# match_predictions_to_truth
# ============================================================


def test_match_perfect_overlap():
    """Predictions exactly match truth → P=R=F1=1.0."""
    truth = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    pred = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.f1 == 1.0


def test_match_no_overlap_far_apart():
    truth = [(10.0, 10.0), (20.0, 20.0)]
    pred = [(100.0, 100.0), (200.0, 200.0)]
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.precision == 0.0
    assert r.recall == 0.0
    assert r.f1 == 0.0
    assert r.n_false_positive == 2
    assert r.n_false_negative == 2


def test_match_partial_5_truth_3_match():
    """5 truth, 3 predictions, all 3 within tolerance → P=1, R=0.6."""
    truth = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]
    pred = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.precision == 1.0
    assert r.recall == 0.6
    assert r.f1 == 0.75


def test_match_extra_predictions_become_false_positives():
    truth = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)]
    pred = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (100.0, 100.0), (200.0, 200.0)]
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.precision == 0.6
    assert r.recall == 1.0
    assert r.f1 == 0.75


def test_match_empty_pred_with_truth():
    r = match_predictions_to_truth([], [(0.0, 0.0), (1.0, 1.0)], tolerance_m=2.0)
    assert r.n_predictions == 0
    assert r.n_truth == 2
    assert r.precision == 0.0
    assert r.recall == 0.0


def test_match_pred_with_empty_truth():
    r = match_predictions_to_truth([(0.0, 0.0)], [], tolerance_m=2.0)
    assert r.n_predictions == 1
    assert r.n_truth == 0
    assert r.precision == 0.0
    assert r.recall == 0.0


def test_match_tolerance_boundary_inclusive():
    """A prediction exactly at the tolerance distance should match."""
    truth = [(0.0, 0.0)]
    pred = [(2.0, 0.0)]  # exactly 2 m away
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.n_true_positive == 1


def test_match_just_outside_tolerance():
    """A prediction just outside tolerance should NOT match."""
    truth = [(0.0, 0.0)]
    pred = [(2.5, 0.0)]
    r = match_predictions_to_truth(pred, truth, tolerance_m=2.0)
    assert r.n_true_positive == 0
    assert r.n_false_positive == 1
    assert r.n_false_negative == 1


# ============================================================
# _detection_centers_world
# ============================================================


def test_detection_centers_y_inverted_correctly():
    """Image row 0 should map to y_max_geo (top of world bbox)."""
    dets = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
    )
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    size = (1000, 1000)
    centers = _detection_centers_world(dets, bounds, size)
    cx, cy = centers[0]
    # Pixel center (150, 150) → x=150, y=1000-150=850
    assert cx == 150.0
    assert cy == 850.0


def test_detection_centers_empty():
    centers = _detection_centers_world(sv.Detections.empty(), (0, 0, 10, 10), (10, 10))
    assert centers == []


# ============================================================
# evaluate_patch_against_lidar — integration with mocked CHM
# ============================================================


def test_evaluate_patch_with_mocked_chm(tmp_path):
    """End-to-end with mocked compute_chm_from_laz + raster IO."""
    # 1 detection at the center of a 100x100 px image covering 50x50 m world.
    dets = sv.Detections(
        xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
    )
    image_bounds = (0.0, 0.0, 50.0, 50.0)
    image_size = (100, 100)

    # Build a synthetic CHM with a tall peak at the same location.
    # Pixel center (50, 50) → world center (25, 25).
    # In the CHM raster of shape 100x100 covering (0,0)→(50,50) at 0.5m
    # the peak should be at row=50, col=50 in CHM grid.
    chm = _chm_with_peaks((100, 100), peaks=[(50, 50, 15.0)])

    fake_chm_path = tmp_path / "fake_chm.tif"
    fake_chm_path.write_bytes(b"placeholder")  # exists for the path check

    # Identity transform with 0.5 m resolution and origin at (0, 0)
    # but rasterio.open returns a transform whose Y axis points DOWN —
    # row 0 is at the TOP of the world (highest Y). Build accordingly.
    from rasterio.transform import from_origin
    transform = from_origin(0.0, 50.0, 0.5, 0.5)

    def fake_compute_chm(*args, **kwargs):
        return fake_chm_path

    def fake_read_chm(path):
        return chm, transform

    import eval_lidar as eval_mod
    with patch.object(eval_mod, "compute_chm_from_laz", fake_compute_chm), \
         patch.object(eval_mod, "_read_chm_array", fake_read_chm):
        result = evaluate_patch_against_lidar(
            detections=dets,
            image_bounds=image_bounds,
            image_size_px=image_size,
            laz_path=tmp_path / "fake.laz",
        )

    # The detection center is at world (25, 25). The CHM peak is at row=50,
    # col=50 in a top-down raster where row 0 = y_max (50). So pixel
    # (col=50, row=50) → world (25.25, 24.75) ≈ (25, 25). Within tolerance.
    assert result.n_predictions == 1
    assert result.n_truth >= 1
    assert result.n_true_positive == 1
    assert result.f1 > 0.0


# ============================================================
# evaluate_patches_against_lidar — aggregation
# ============================================================


def test_aggregate_micro_average():
    """Aggregate via pooled counts, not per-patch averaging."""
    from eval_lidar import EvalResult
    # Patch 1: 5 truth, 5 pred, 4 TP → P=0.8, R=0.8
    # Patch 2: 100 truth, 100 pred, 60 TP → P=0.6, R=0.6
    # Macro avg of F1 = (0.8 + 0.6)/2 = 0.7
    # Micro avg of F1 = pooled: 64 TP, 41 FP, 41 FN
    #   P = 64/(64+41) = 0.609..
    #   R = 64/(64+41) = 0.609..
    #   F1 = 0.609..
    pooled = EvalResult.from_counts(
        n_predictions=5 + 100,
        n_truth=5 + 100,
        n_true_positive=4 + 60,
    )
    assert abs(pooled.precision - 0.6095) < 0.01
    assert abs(pooled.recall - 0.6095) < 0.01
    assert abs(pooled.f1 - 0.6095) < 0.01
