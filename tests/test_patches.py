"""Tests for forest_pulse.patches — patch metadata helpers.

Synthetic CSVs in a temp directory, no dependency on the real
Montseny metadata file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from forest_pulse.patches import (
    get_patch_bounds,
    get_patch_center,
    iter_patch_names,
)

# ============================================================
# Helpers
# ============================================================


def _write_csv(tmp: Path, rows: list[dict]) -> Path:
    """Write a minimal patches_metadata.csv to a temp directory."""
    import csv as _csv

    path = tmp / "patches_metadata.csv"
    fields = list(rows[0].keys()) if rows else ["filename", "zone", "x_center", "y_center"]
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


SAMPLE_ROWS = [
    {"filename": "0001.jpg", "zone": "high", "x_center": "450000.0", "y_center": "4625000.0"},
    {"filename": "0002.jpg", "zone": "high", "x_center": "450160.0", "y_center": "4625000.0"},
    {"filename": "0003.jpg", "zone": "low",  "x_center": "449840.0", "y_center": "4624840.0"},
    {"filename": "0004.jpg", "zone": "mid",  "x_center": "451000.0", "y_center": "4626000.0"},
]


# ============================================================
# get_patch_center
# ============================================================


def test_get_patch_center_returns_known_coords():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        x, y = get_patch_center(csv_path, "0001.jpg")
        assert x == 450000.0
        assert y == 4625000.0


def test_get_patch_center_raises_on_missing_patch():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        with pytest.raises(ValueError, match="9999.jpg"):
            get_patch_center(csv_path, "9999.jpg")


def test_get_patch_center_raises_on_missing_csv():
    with pytest.raises(FileNotFoundError, match="does_not_exist.csv"):
        get_patch_center("does_not_exist.csv", "0001.jpg")


# ============================================================
# get_patch_bounds
# ============================================================


def test_get_patch_bounds_centers_on_known_point():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        bounds = get_patch_bounds(csv_path, "0001.jpg", 160.0)
        x_min, y_min, x_max, y_max = bounds
        assert x_min == 450000.0 - 80.0
        assert y_min == 4625000.0 - 80.0
        assert x_max == 450000.0 + 80.0
        assert y_max == 4625000.0 + 80.0


def test_get_patch_bounds_custom_size():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        bounds = get_patch_bounds(csv_path, "0001.jpg", 320.0)
        assert bounds[2] - bounds[0] == 320.0
        assert bounds[3] - bounds[1] == 320.0


# ============================================================
# iter_patch_names
# ============================================================


def test_iter_patch_names_returns_all_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        names = iter_patch_names(csv_path)
        assert names == ["0001.jpg", "0002.jpg", "0003.jpg", "0004.jpg"]


def test_iter_patch_names_filters_by_zone():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        names = iter_patch_names(csv_path, zone="high")
        assert names == ["0001.jpg", "0002.jpg"]


def test_iter_patch_names_limit():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        names = iter_patch_names(csv_path, limit=2)
        assert len(names) == 2
        assert names == ["0001.jpg", "0002.jpg"]


def test_iter_patch_names_zone_and_limit_combined():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = _write_csv(Path(tmp), SAMPLE_ROWS)
        names = iter_patch_names(csv_path, zone="high", limit=1)
        assert names == ["0001.jpg"]
