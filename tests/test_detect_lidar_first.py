"""Tests for detect_trees_from_lidar (Phase 11a) + world_to_pixel helpers.

No real LAZ files, no real RF-DETR weights. Monkeypatch the LiDAR
primitives with fake tree-top lists and the image-space math is
verified directly. Matches the Phase 10c test pattern.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

import forest_pulse.detect as detect_mod
import forest_pulse.lidar as lidar_mod
from forest_pulse.detect import detect_trees_from_lidar
from forest_pulse.lidar import world_to_pixel, world_to_pixel_batch

# ============================================================
# world_to_pixel helpers
# ============================================================


def test_world_to_pixel_center():
    """Center of a 160m patch at 640x640 → pixel (320, 320)."""
    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    px, py = world_to_pixel(80.0, 80.0, bounds, size)
    assert abs(px - 320.0) < 1e-6
    assert abs(py - 320.0) < 1e-6


def test_world_to_pixel_y_axis_inverted():
    """World y_max maps to image row 0 (top), not row h_px (bottom).

    This is the recurring Y-inversion gotcha from Phase 7 onward.
    """
    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    # World (80, 160) is the top edge of the patch → image row 0
    px, py = world_to_pixel(80.0, 160.0, bounds, size)
    assert abs(px - 320.0) < 1e-6
    assert abs(py - 0.0) < 1e-6
    # World (80, 0) is the bottom edge → image row 640
    px, py = world_to_pixel(80.0, 0.0, bounds, size)
    assert abs(py - 640.0) < 1e-6


def test_world_to_pixel_out_of_bounds_not_clipped():
    """Out-of-frame world coords return out-of-frame pixel coords.

    Clipping is the caller's responsibility — detect_trees_from_lidar
    clips to the image frame + area filter.
    """
    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    # Way outside the patch to the right
    px, py = world_to_pixel(200.0, 80.0, bounds, size)
    assert px > 640.0
    # Negative x (left of the patch)
    px, py = world_to_pixel(-10.0, 80.0, bounds, size)
    assert px < 0.0


def test_world_to_pixel_batch_matches_scalar():
    """Batched version agrees with scalar calls row-by-row."""
    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    positions = np.array([
        [80.0, 80.0],     # center
        [80.0, 160.0],    # top
        [80.0, 0.0],      # bottom
        [0.0, 80.0],      # left
        [160.0, 80.0],    # right
    ], dtype=np.float64)

    batch_result = world_to_pixel_batch(positions, bounds, size)

    for i in range(len(positions)):
        scalar = world_to_pixel(
            float(positions[i, 0]), float(positions[i, 1]),
            bounds, size,
        )
        assert abs(batch_result[i, 0] - scalar[0]) < 1e-6
        assert abs(batch_result[i, 1] - scalar[1]) < 1e-6


def test_world_to_pixel_batch_empty_input():
    """Empty (0, 2) input returns empty (0, 2) output."""
    result = world_to_pixel_batch(
        np.zeros((0, 2), dtype=np.float64),
        (0.0, 0.0, 160.0, 160.0),
        (640, 640),
    )
    assert result.shape == (0, 2)


def test_world_to_pixel_round_trip_via_bbox_centers():
    """world_to_pixel(bbox_centers_to_world(dets)) should recover
    the original pixel centers within floating-point precision.
    """
    from forest_pulse.lidar import bbox_centers_to_world

    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    dets = sv.Detections(
        xyxy=np.array([
            [100, 100, 200, 200],   # center (150, 150)
            [300, 50, 400, 150],    # center (350, 100)
            [0, 0, 20, 20],         # center (10, 10)
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8, 0.7], dtype=np.float32),
    )

    world_centers = bbox_centers_to_world(dets, bounds, size)
    recovered = world_to_pixel_batch(world_centers, bounds, size)

    # Expected pixel centers
    expected = np.array([[150.0, 150.0], [350.0, 100.0], [10.0, 10.0]])
    np.testing.assert_allclose(recovered, expected, atol=1e-5)


# ============================================================
# detect_trees_from_lidar
# ============================================================


def _stub_lidar_pipeline(
    monkeypatch,
    positions: list[tuple[float, float]],
    heights: list[float],
):
    """Monkeypatch the LAZ/CHM/tree-top pipeline to return fixed data.

    Avoids needing real LAZ files in tests. `detect_trees_from_lidar`
    imports its helpers lazily from `forest_pulse.lidar`, so we
    patch on that module.
    """
    # compute_chm_from_laz: return a dummy path; _read_chm_raster is
    # also stubbed so the actual path is never dereferenced.
    from pathlib import Path as _Path
    monkeypatch.setattr(
        lidar_mod, "compute_chm_from_laz",
        lambda laz_path, bounds, **kwargs: _Path("/fake/chm.tif"),
    )
    monkeypatch.setattr(
        lidar_mod, "_read_chm_raster",
        lambda path: (np.zeros((10, 10), dtype=np.float32), None),
    )
    # find_tree_tops_from_chm: return the fixed positions+heights
    def _fake_find(chm, transform, **kwargs):
        if kwargs.get("return_heights", False):
            return (list(positions), list(heights))
        return list(positions)
    monkeypatch.setattr(lidar_mod, "find_tree_tops_from_chm", _fake_find)


def test_detect_from_lidar_single_peak_center(monkeypatch):
    """One peak at the image center → one detection with correct bbox."""
    bounds = (0.0, 0.0, 160.0, 160.0)
    size = (640, 640)
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[(80.0, 80.0)],  # center
        heights=[15.0],
    )

    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=bounds,
        image_size_px=size,
        crown_radius_m=2.5,
    )
    assert len(dets) == 1
    # 2.5m radius at 4px/m = 10px radius → bbox (310, 310, 330, 330)
    xyxy = dets.xyxy[0]
    assert abs(xyxy[0] - 310.0) < 1e-4
    assert abs(xyxy[1] - 310.0) < 1e-4
    assert abs(xyxy[2] - 330.0) < 1e-4
    assert abs(xyxy[3] - 330.0) < 1e-4


def test_detect_from_lidar_zero_peaks_empty_output(monkeypatch):
    """A patch with no LiDAR peaks → empty sv.Detections, no crash."""
    _stub_lidar_pipeline(monkeypatch, positions=[], heights=[])
    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=(0.0, 0.0, 160.0, 160.0),
        image_size_px=(640, 640),
    )
    assert len(dets) == 0
    assert isinstance(dets, sv.Detections)


def test_detect_from_lidar_confidence_from_height(monkeypatch):
    """Synthetic confidence is monotonic in peak height.

    Formula: conf = clip((h - 5) / 20, 0.01, 1.0)
      h = 5  → 0.01 (just above floor)
      h = 15 → 0.50
      h = 25 → 1.00
    """
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[(30.0, 30.0), (80.0, 80.0), (130.0, 130.0)],
        heights=[5.0, 15.0, 25.0],
    )
    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=(0.0, 0.0, 160.0, 160.0),
        image_size_px=(640, 640),
    )
    assert len(dets) == 3
    # Confidences in height order: 0.01, 0.50, 1.00
    confs = dets.confidence
    assert abs(confs[0] - 0.01) < 1e-5
    assert abs(confs[1] - 0.50) < 1e-5
    assert abs(confs[2] - 1.00) < 1e-5


def test_detect_from_lidar_edge_peak_clipped_to_image(monkeypatch):
    """A peak near the image edge gets its bbox clipped, not dropped."""
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[(2.0, 80.0)],  # 2m from the left edge → bbox starts at -0.5
        heights=[15.0],
    )
    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=(0.0, 0.0, 160.0, 160.0),
        image_size_px=(640, 640),
        crown_radius_m=2.5,
    )
    assert len(dets) == 1
    xyxy = dets.xyxy[0]
    # Left edge clipped to 0
    assert xyxy[0] == 0.0
    # Right edge = 2m + 2.5m = 4.5m × 4 px/m = 18 px
    assert abs(xyxy[2] - 18.0) < 1e-4


def test_detect_from_lidar_peak_outside_image_dropped(monkeypatch):
    """A peak entirely outside the image frame gets dropped."""
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[
            (80.0, 80.0),       # center — kept
            (200.0, 80.0),      # way right of the patch → dropped
            (-10.0, 80.0),      # way left of the patch → dropped
        ],
        heights=[15.0, 15.0, 15.0],
    )
    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=(0.0, 0.0, 160.0, 160.0),
        image_size_px=(640, 640),
        crown_radius_m=2.5,
    )
    # Only the center peak survives; the two out-of-frame peaks
    # have zero-area bboxes after clipping and get dropped.
    assert len(dets) == 1


def test_detect_from_lidar_rf_detr_verify_requires_image(monkeypatch):
    """rf_detr_verify=True without rf_detr_image raises ValueError."""
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[(80.0, 80.0)],
        heights=[15.0],
    )
    with pytest.raises(ValueError, match="rf_detr_verify"):
        detect_trees_from_lidar(
            laz_path="/fake.laz",
            image_bounds=(0.0, 0.0, 160.0, 160.0),
            image_size_px=(640, 640),
            rf_detr_verify=True,
            rf_detr_checkpoint="/fake.pt",
            # rf_detr_image NOT passed → should raise
        )


def test_detect_from_lidar_rf_detr_verify_drops_unmatched(monkeypatch):
    """rf_detr_verify drops LiDAR peaks RF-DETR doesn't see."""
    _stub_lidar_pipeline(
        monkeypatch,
        positions=[
            (80.0, 80.0),    # center — RF-DETR will see this
            (40.0, 40.0),    # bottom-left — RF-DETR will NOT see this
        ],
        heights=[15.0, 15.0],
    )

    # Mock detect_trees_sliced to return a detection only at
    # world (80, 80) → pixel (320, 320) → small bbox there.
    def _fake_sliced(image, model_name, confidence, **kwargs):
        return sv.Detections(
            xyxy=np.array([[318, 318, 322, 322]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=np.int64),
        )
    monkeypatch.setattr(detect_mod, "detect_trees_sliced", _fake_sliced)

    dets = detect_trees_from_lidar(
        laz_path="/fake.laz",
        image_bounds=(0.0, 0.0, 160.0, 160.0),
        image_size_px=(640, 640),
        rf_detr_verify=True,
        rf_detr_checkpoint="/fake.pt",
        rf_detr_image=np.zeros((640, 640, 3), dtype=np.uint8),
        rf_detr_verify_tolerance_m=2.0,
    )
    # Only the (80, 80) peak survives verification
    assert len(dets) == 1
    # Verify the kept detection is the center one (near (320, 320))
    kept_center_x = (dets.xyxy[0, 0] + dets.xyxy[0, 2]) / 2
    assert abs(kept_center_x - 320.0) < 1e-4
