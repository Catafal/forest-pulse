"""Tests for NDVI and LiDAR detection filters.

Uses synthetic in-memory rasters — no network access, no large fixtures.
"""

import tempfile
from pathlib import Path

import numpy as np
import rasterio
import supervision as sv
from rasterio.transform import from_bounds

from forest_pulse.lidar import filter_by_height
from forest_pulse.ndvi import _pixel_bbox_to_geo, filter_by_ndvi


def _make_synthetic_raster(
    values: np.ndarray,
    bounds: tuple[float, float, float, float],
    out_path: Path,
) -> None:
    """Write a single-band float32 GeoTIFF with given values + bounds."""
    h, w = values.shape
    transform = from_bounds(*bounds, w, h)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:25831",
        "transform": transform,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(values.astype(np.float32), 1)


def test_ndvi_filter_drops_low_ndvi_boxes():
    """A box over low-NDVI pixels must be dropped; high-NDVI kept."""
    with tempfile.TemporaryDirectory() as tmp:
        raster_path = Path(tmp) / "ndvi.tif"
        # Left half = 0.05 (non-vegetation), right half = 0.5 (vegetation)
        values = np.zeros((100, 100), dtype=np.float32)
        values[:, :50] = 0.05
        values[:, 50:] = 0.50
        bounds = (0.0, 0.0, 100.0, 100.0)  # 1 unit per pixel
        _make_synthetic_raster(values, bounds, raster_path)

        # Two boxes: one over the low-NDVI half, one over the high-NDVI half
        # Image bounds match raster bounds; image size 100x100 pixels
        dets = sv.Detections(
            xyxy=np.array([
                [10, 40, 30, 60],   # left half → low NDVI → drop
                [60, 40, 90, 60],   # right half → high NDVI → keep
            ], dtype=np.float32),
        )

        filtered = filter_by_ndvi(
            dets, raster_path,
            image_bounds=bounds, image_size_px=(100, 100),
            threshold=0.15,
        )
        assert len(filtered) == 1
        # The kept box should be the right-half one (x_min >= 50)
        assert filtered.xyxy[0][0] >= 50


def test_height_filter_drops_short_boxes():
    """A box over short pixels must be dropped; tall pixels kept."""
    with tempfile.TemporaryDirectory() as tmp:
        raster_path = Path(tmp) / "chm.tif"
        # Top half = 2m (shrubs), bottom half = 10m (trees)
        values = np.zeros((100, 100), dtype=np.float32)
        values[:50, :] = 2.0
        values[50:, :] = 10.0
        bounds = (0.0, 0.0, 100.0, 100.0)
        _make_synthetic_raster(values, bounds, raster_path)

        # Remember: image y=0 is top → maps to y_max_geo=100
        # so pixel y=10 is geo y=90 (top half, shrubs)
        # and pixel y=90 is geo y=10 (bottom half, trees)
        dets = sv.Detections(
            xyxy=np.array([
                [40, 5, 60, 30],   # top (shrubs) → drop
                [40, 70, 60, 95],  # bottom (trees) → keep
            ], dtype=np.float32),
        )

        filtered = filter_by_height(
            dets, raster_path,
            image_bounds=bounds, image_size_px=(100, 100),
            threshold=5.0, aggregation="max",
        )
        assert len(filtered) == 1
        assert filtered.xyxy[0][1] >= 50  # the bottom-half box


def test_max_aggregation_preserves_tall_trees():
    """One tall pixel in a box of short pixels keeps the box under 'max'."""
    with tempfile.TemporaryDirectory() as tmp:
        raster_path = Path(tmp) / "chm.tif"
        # Mostly 2m (shrubs), single 15m pixel (lone tall tree)
        values = np.full((100, 100), 2.0, dtype=np.float32)
        values[50, 50] = 15.0
        bounds = (0.0, 0.0, 100.0, 100.0)
        _make_synthetic_raster(values, bounds, raster_path)

        # Box covering the tall pixel
        dets = sv.Detections(
            xyxy=np.array([[40, 40, 60, 60]], dtype=np.float32),
        )
        kept_max = filter_by_height(
            dets, raster_path,
            image_bounds=bounds, image_size_px=(100, 100),
            threshold=5.0, aggregation="max",
        )
        assert len(kept_max) == 1

        # Same box under 'mean' should be dropped (mean is ~2m)
        dropped_mean = filter_by_height(
            dets, raster_path,
            image_bounds=bounds, image_size_px=(100, 100),
            threshold=5.0, aggregation="mean",
        )
        assert len(dropped_mean) == 0


def test_empty_detections_pass_through():
    """Filters on empty detections must return empty without crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        raster_path = Path(tmp) / "r.tif"
        _make_synthetic_raster(
            np.ones((10, 10), dtype=np.float32),
            (0.0, 0.0, 10.0, 10.0), raster_path,
        )
        empty = sv.Detections.empty()
        assert len(filter_by_ndvi(
            empty, raster_path, (0.0, 0.0, 10.0, 10.0), (10, 10),
        )) == 0
        assert len(filter_by_height(
            empty, raster_path, (0.0, 0.0, 10.0, 10.0), (10, 10),
        )) == 0


def test_pixel_to_geo_bbox_transform():
    """Pixel→geo transform must invert Y correctly."""
    # Image is 640x640 px, geo bounds 1000-1160m (E), 2000-2160m (N)
    bounds = (1000.0, 2000.0, 1160.0, 2160.0)
    size = (640, 640)
    # Pixel (0, 0, 640, 640) = top-left corner to bottom-right
    geo = _pixel_bbox_to_geo(
        np.array([0.0, 0.0, 640.0, 640.0]), bounds, size,
    )
    # x_min/x_max span the whole width
    assert abs(geo[0] - 1000.0) < 0.01
    assert abs(geo[2] - 1160.0) < 0.01
    # Y inverted: pixel y=0 (top) = y_max_geo, pixel y=640 (bottom) = y_min_geo
    assert abs(geo[3] - 2160.0) < 0.01
    assert abs(geo[1] - 2000.0) < 0.01
