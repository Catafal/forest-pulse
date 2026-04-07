"""Georeferencing — convert pixel-space detections into a GeoDataFrame.

Takes `sv.Detections` (pixel xyxy boxes) + the image's geographic bounds
and produces a GeoPandas GeoDataFrame with one row per tree. Each tree is
represented as a Point at the bbox center; bbox dimensions are stored as
columns so callers can derive polygons or compute crown area later.

## Why `image_bounds` instead of reading EXIF/GeoTIFF

The original stub assumed we'd read bounds from the image itself (EXIF for
drones, transform for GeoTIFFs). Our Montseny patches are plain JPEG with
no geo metadata — bounds come from `data/montseny/patches_metadata.csv`.
Taking bounds as an explicit argument matches the pattern in ndvi.py and
lidar.py and keeps this module usable for any image source.

## Output schema (contract for temporal.py and export.py)

Always present:
  tree_id, geometry, confidence,
  bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax (meters in the chosen CRS),
  crown_width_m, crown_height_m, crown_area_m2

If health_scores provided:
  health_label, grvi, exg, health_confidence
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import supervision as sv
from shapely.geometry import Point

from forest_pulse.health import HealthScore

logger = logging.getLogger(__name__)

# Columns always emitted by georeference(), in display order.
_BASE_COLUMNS = [
    "tree_id", "confidence",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    "crown_width_m", "crown_height_m", "crown_area_m2",
]

# Extra columns emitted when health_scores are provided.
_HEALTH_COLUMNS = ["health_label", "grvi", "exg", "health_confidence"]


def georeference(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    health_scores: list[HealthScore] | None = None,
    crs: str = "EPSG:25831",
) -> gpd.GeoDataFrame:
    """Convert pixel-space detections to a GeoDataFrame.

    Args:
        detections: Supervision Detections with xyxy boxes in pixel coords
            of the original image (not a resized raster).
        image_bounds: Geographic bounds of the image as
            (x_min, y_min, x_max, y_max) in the given CRS.
        image_size_px: (width, height) of the image in pixels.
        health_scores: Optional list of HealthScore objects, one per
            detection, in the same order. When supplied, health columns
            are added to the output.
        crs: Coordinate reference system of image_bounds. Default is
            EPSG:25831 (ETRS89 / UTM zone 31N, standard for Catalunya).

    Returns:
        GeoDataFrame with one row per detection. Point geometry at the
        bbox center, crown dimensions in meters, confidence, tree_id,
        and — if provided — health fields. Empty input produces an empty
        GeoDataFrame with the same schema (no crash).
    """
    # rfdetr adds source_shape/source_image to detections.data. These
    # don't align with the number of detections so any pandas/numpy
    # operation that iterates will trip over them. Strip before use.
    _strip_rfdetr_metadata(detections)

    n_dets = len(detections)
    columns = _BASE_COLUMNS + (_HEALTH_COLUMNS if health_scores is not None else [])

    if n_dets == 0:
        logger.info("georeference: empty detections → empty GeoDataFrame")
        return _empty_gdf(columns, crs)

    # If masks are present, we compute crown_area_m2 from the actual
    # mask pixel count × meters²/pixel — much more accurate than the
    # rectangular bbox area (which over-estimates by including gaps).
    has_masks = detections.mask is not None
    px_area_m2 = _compute_pixel_area_m2(image_bounds, image_size_px)

    rows = []
    geoms = []
    for i, xyxy in enumerate(detections.xyxy):
        # Convert pixel bbox → geographic bbox in CRS coordinates
        geo_box = _pixel_bbox_to_geo(xyxy, image_bounds, image_size_px)
        g_xmin, g_ymin, g_xmax, g_ymax = geo_box

        width_m = g_xmax - g_xmin
        height_m = g_ymax - g_ymin
        center = Point((g_xmin + g_xmax) / 2.0, (g_ymin + g_ymax) / 2.0)

        # Crown area: mask-derived when available, else rectangular bbox
        if has_masks:
            mask_px = int(detections.mask[i].sum())
            crown_area = mask_px * px_area_m2
        else:
            crown_area = width_m * height_m

        # Confidence may be None if the model didn't provide it; default to 0.
        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0

        row = {
            "tree_id": i,
            "confidence": round(conf, 4),
            "bbox_xmin": round(g_xmin, 2),
            "bbox_ymin": round(g_ymin, 2),
            "bbox_xmax": round(g_xmax, 2),
            "bbox_ymax": round(g_ymax, 2),
            "crown_width_m": round(width_m, 2),
            "crown_height_m": round(height_m, 2),
            "crown_area_m2": round(crown_area, 2),
        }

        if health_scores is not None:
            hs = health_scores[i] if i < len(health_scores) else None
            row["health_label"] = hs.label if hs else "unknown"
            row["grvi"] = round(hs.grvi, 4) if hs else 0.0
            row["exg"] = round(hs.exg, 4) if hs else 0.0
            row["health_confidence"] = round(hs.confidence, 4) if hs else 0.0

        rows.append(row)
        geoms.append(center)

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(rows, columns=columns),
        geometry=geoms,
        crs=crs,
    )
    logger.info("georeference: %d trees → GeoDataFrame in %s", n_dets, crs)
    return gdf


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Remove rfdetr's source_shape / source_image from detections.data.

    Same pattern as ndvi.py / lidar.py. rfdetr stores full-image arrays
    and shape tuples in `detections.data`, neither of which has length
    equal to `len(detections)`. Boolean indexing or slicing would crash.
    Stripping is safe — we don't need those fields downstream.
    """
    if not hasattr(detections, "data"):
        return
    for key in ("source_shape", "source_image"):
        if key in detections.data:
            del detections.data[key]


def _pixel_bbox_to_geo(
    xyxy_px,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Convert a pixel-space bbox to geographic coordinates.

    Image Y axis points DOWN (row 0 is top of image). Geographic Y axis
    points UP (y_max is the top of the world). The transform inverts Y
    accordingly. Duplicated in ndvi.py / lidar.py to keep each filter
    module self-contained — this project prefers small duplications over
    cross-module coupling.
    """
    x_min_geo, y_min_geo, x_max_geo, y_max_geo = image_bounds
    w_px, h_px = image_size_px
    x_scale = (x_max_geo - x_min_geo) / w_px
    y_scale = (y_max_geo - y_min_geo) / h_px
    x1_px, y1_px, x2_px, y2_px = xyxy_px.tolist()
    x_min = x_min_geo + x1_px * x_scale
    x_max = x_min_geo + x2_px * x_scale
    y_max = y_max_geo - y1_px * y_scale   # image top → world top
    y_min = y_max_geo - y2_px * y_scale   # image bottom → world bottom
    return (x_min, y_min, x_max, y_max)


def _compute_pixel_area_m2(
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
) -> float:
    """Compute the area of a single pixel in square meters.

    Used when we have pixel masks and want mask_area * px_area_m²
    instead of the rectangular bbox area. Only meaningful when the CRS
    is projected (e.g. EPSG:25831 — metric units).
    """
    x_min, y_min, x_max, y_max = image_bounds
    w_px, h_px = image_size_px
    x_m_per_px = (x_max - x_min) / w_px
    y_m_per_px = (y_max - y_min) / h_px
    return x_m_per_px * y_m_per_px


def _empty_gdf(columns: list[str], crs: str) -> gpd.GeoDataFrame:
    """Build an empty GeoDataFrame with the project schema.

    GeoPandas needs explicit column dtypes for an empty frame to round-trip
    cleanly. We default all columns to object dtype — that's fine for an
    empty frame because no computation will touch them.
    """
    empty_df = pd.DataFrame({col: [] for col in columns})
    return gpd.GeoDataFrame(empty_df, geometry=[], crs=crs)
