"""Tests for segment.py — filter, IoU, dedup, stacking.

SAM2 itself is NOT loaded in these tests — they exercise the pure
helpers using synthetic masks. Integration tests that actually run
SAM2 go behind `@pytest.mark.slow`.
"""

import numpy as np
import pytest
import supervision as sv

from forest_pulse.segment import (
    CrownFilter,
    _dedup_auto_against_refined,
    _filter_crown_segments,
    _mask_bbox,
    _mask_centroid,
    _mask_circularity,
    _mask_iou,
    _stack_sv_detections,
)


def _disc(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    """Boolean disc mask centred at (cx, cy) with radius r."""
    yy, xx = np.mgrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r


def _rect(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


# --- _mask_bbox ---

def test_mask_bbox_rectangular():
    m = _rect(100, 100, 10, 20, 40, 50)
    bbox = _mask_bbox(m)
    assert tuple(bbox.tolist()) == (10.0, 20.0, 40.0, 50.0)


def test_mask_bbox_empty_mask():
    m = np.zeros((10, 10), dtype=bool)
    bbox = _mask_bbox(m)
    assert tuple(bbox.tolist()) == (0.0, 0.0, 0.0, 0.0)


# --- _mask_circularity ---

def test_circularity_disc_above_filter_threshold():
    """Discrete discs have circularity ~0.6 due to stair-stepped edges.

    The crown filter threshold is 0.45 — discs must score above that.
    """
    m = _disc(200, 200, 100, 100, 50)
    circ = _mask_circularity(m)
    # 4-neighbor perimeter over-counts diagonals → ~0.6 for a disc.
    # Thin rectangles score < 0.3 so 0.5 cleanly separates the two.
    assert circ > 0.5


def test_circularity_thin_rectangle_low():
    m = _rect(200, 200, 50, 95, 150, 105)  # 10px tall, 100px wide
    circ = _mask_circularity(m)
    assert circ < 0.3


def test_circularity_empty_mask_zero():
    m = np.zeros((50, 50), dtype=bool)
    assert _mask_circularity(m) == 0.0


# --- _mask_iou ---

def test_iou_identical_masks_is_one():
    a = _disc(100, 100, 50, 50, 20)
    assert _mask_iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint_masks_is_zero():
    a = _disc(100, 100, 20, 20, 10)
    b = _disc(100, 100, 80, 80, 10)
    assert _mask_iou(a, b) == 0.0


def test_iou_half_overlap():
    # Two 10x10 rectangles overlapping 5x10 → intersection 50, union 150
    a = _rect(50, 50, 0, 0, 10, 10)
    b = _rect(50, 50, 5, 0, 15, 10)
    iou = _mask_iou(a, b)
    assert iou == pytest.approx(50 / 150, abs=0.01)


# --- _mask_centroid ---

def test_centroid_symmetric_disc():
    m = _disc(100, 100, 50, 50, 20)
    cx, cy = _mask_centroid(m)
    assert abs(cx - 50) < 1.0
    assert abs(cy - 50) < 1.0


# --- _filter_crown_segments ---

def test_filter_keeps_round_medium_segments():
    disc_good = _disc(640, 640, 320, 320, 20)  # ~1250 px, circular
    masks = [disc_good]
    cf = CrownFilter()
    kept = _filter_crown_segments(masks, (640, 640), cf)
    assert kept == [0]


def test_filter_drops_tiny_segments():
    tiny = _disc(640, 640, 100, 100, 3)  # ~28 px, too small
    kept = _filter_crown_segments([tiny], (640, 640), CrownFilter())
    assert kept == []


def test_filter_drops_elongated_segments():
    thin = _rect(640, 640, 100, 300, 500, 310)  # 400x10 = elongated
    kept = _filter_crown_segments([thin], (640, 640), CrownFilter())
    assert kept == []


def test_filter_drops_huge_segments():
    huge = _rect(640, 640, 0, 0, 500, 500)  # 250000 px, too big
    kept = _filter_crown_segments([huge], (640, 640), CrownFilter())
    assert kept == []


def test_filter_drops_image_fraction_too_big():
    # 300x300 = 90000 px, <max_area_px(40k)? Actually > max_area.
    # Use smaller total image to hit the max_area_frac path specifically.
    half = _rect(100, 100, 0, 0, 50, 50)  # 2500 px = 25% of 10000
    cf = CrownFilter(min_area_px=100, max_area_px=100_000, max_area_frac=0.15)
    kept = _filter_crown_segments([half], (100, 100), cf)
    assert kept == []  # rejected because 25% > 15%


# --- _dedup_auto_against_refined ---

def test_dedup_drops_overlapping_auto_segments():
    m_refined = _disc(100, 100, 50, 50, 20)
    m_auto_overlap = _disc(100, 100, 52, 52, 20)     # large overlap
    m_auto_disjoint = _disc(100, 100, 85, 85, 10)    # disjoint

    refined = sv.Detections(
        xyxy=np.array([[30, 30, 70, 70]], dtype=np.float32),
        mask=m_refined[np.newaxis],
    )
    auto = sv.Detections(
        xyxy=np.array([[32, 32, 72, 72], [75, 75, 95, 95]], dtype=np.float32),
        mask=np.stack([m_auto_overlap, m_auto_disjoint]),
    )

    keep = _dedup_auto_against_refined(refined, auto, 0.30, 20.0)
    assert keep.tolist() == [False, True]  # overlap dropped, disjoint kept


def test_dedup_centroid_fallback_when_no_masks():
    refined = sv.Detections(
        xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
    )
    auto = sv.Detections(
        xyxy=np.array([
            [45, 45, 65, 65],   # center (55, 55), close to (50, 50) → drop
            [200, 200, 220, 220],  # far away → keep
        ], dtype=np.float32),
    )
    keep = _dedup_auto_against_refined(refined, auto, 0.30, 20.0)
    assert keep.tolist() == [False, True]


# --- _stack_sv_detections ---

def test_stack_detections_with_masks():
    m1 = np.ones((1, 100, 100), dtype=bool)
    m2 = np.ones((2, 100, 100), dtype=bool)
    left = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        mask=m1,
    )
    right = sv.Detections(
        xyxy=np.array([[20, 20, 30, 30], [40, 40, 50, 50]], dtype=np.float32),
        confidence=np.array([0.8, 0.7], dtype=np.float32),
        mask=m2,
    )
    stacked = _stack_sv_detections(left, right)
    assert len(stacked) == 3
    assert stacked.mask.shape == (3, 100, 100)
    assert stacked.confidence.tolist() == pytest.approx([0.9, 0.8, 0.7])
