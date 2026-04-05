"""Tests for health scoring — pure functions, no model loading."""

import numpy as np

from forest_pulse.health import classify_health, compute_exg, compute_grvi


def test_grvi_pure_green():
    """A pure green image should have GRVI close to 1.0."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 1] = 200  # green channel only
    assert compute_grvi(crop) > 0.9


def test_grvi_pure_red():
    """A pure red image should have GRVI close to -1.0."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 0] = 200  # red channel only
    assert compute_grvi(crop) < -0.9


def test_grvi_black():
    """A black image should return 0.0 (division guard)."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    assert compute_grvi(crop) == 0.0


def test_grvi_equal_red_green():
    """Equal R and G should give GRVI = 0."""
    crop = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert abs(compute_grvi(crop)) < 0.01


def test_exg_pure_green():
    """Pure green: ExG = 2*200 - 0 - 0 = 400."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 1] = 200
    assert compute_exg(crop) > 350


def test_exg_no_green():
    """No green, high R+B: ExG should be negative."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 0] = 200  # red
    crop[:, :, 2] = 200  # blue
    assert compute_exg(crop) < -300


def test_classify_healthy():
    """High GRVI + high ExG → healthy."""
    label, conf = classify_health(grvi=0.25, exg=50.0)
    assert label == "healthy"
    assert conf > 0.5


def test_classify_dead():
    """Negative GRVI → dead."""
    label, conf = classify_health(grvi=-0.1, exg=5.0)
    assert label == "dead"
    assert conf > 0.5


def test_classify_stressed():
    """Borderline indices → stressed."""
    label, conf = classify_health(grvi=0.05, exg=20.0)
    assert label == "stressed"
