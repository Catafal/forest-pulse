"""Georeferencing — convert pixel coordinates to GPS coordinates.

Supports two input types:
1. Drone images with EXIF GPS metadata (automatic via piexif)
2. Orthophotos with CRS/transform metadata (ICGC, PNOA, GeoTIFF)
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import supervision as sv

from forest_pulse.health import HealthScore


def georeference(
    detections: sv.Detections,
    image_path: str | Path,
    health_scores: list[HealthScore] | None = None,
    crs: str | None = None,
) -> gpd.GeoDataFrame:
    """Convert pixel-space detections to GPS-referenced GeoDataFrame.

    Reads geospatial metadata from the image (EXIF GPS for drone images,
    or CRS/transform for orthophotos) and maps each bounding box center
    to real-world GPS coordinates.

    Args:
        detections: Supervision Detections with xyxy bounding boxes in pixel space.
        image_path: Path to the source image (for reading EXIF or GeoTIFF metadata).
        health_scores: Optional health scores to attach as attributes.
        crs: Override coordinate reference system (e.g., "EPSG:25831" for Catalunya).
            If None, auto-detected from image metadata.

    Returns:
        GeoDataFrame with columns: tree_id, geometry (Point), crown_area_m2,
        health_label, grvi, confidence, bbox_xyxy.
    """
    # TODO: Implement georeferencing pipeline
    # 1. Detect image type (EXIF drone vs GeoTIFF orthophoto)
    # 2. Extract geospatial metadata
    # 3. Compute ground sample distance (GSD) — meters per pixel
    # 4. Map each bbox center pixel → GPS coordinate
    # 5. Estimate crown area in m² from bbox area × GSD²
    # 6. Build GeoDataFrame
    raise NotImplementedError("georeference not yet implemented")


def _read_exif_gps(image_path: str | Path) -> dict:
    """Extract GPS coordinates, altitude, and camera params from EXIF."""
    # TODO: Implement EXIF GPS parsing via piexif
    raise NotImplementedError


def _read_geotiff_transform(image_path: str | Path) -> dict:
    """Extract CRS and affine transform from GeoTIFF metadata."""
    # TODO: Implement GeoTIFF metadata reading
    raise NotImplementedError
