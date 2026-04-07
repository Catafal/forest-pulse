"""LiDAR Canopy Height Model filter for tree detections.

The Canopy Height Model (CHM) is computed as DSM - DTM:
  - DSM (Digital Surface Model): elevation of whatever is on top (tree canopy,
    buildings, ground where bare)
  - DTM (Digital Terrain Model): elevation of the bare ground underneath
  - CHM = DSM - DTM = height of everything above the ground

For tree detection, CHM lets us filter out short vegetation (bushes, heather,
saplings) that RF-DETR detects as "trees" because from above, a large shrub
and a small tree look identical. Trees taller than 5m are considered real
trees per Spanish Forest Inventory conventions.

Aggregation choice: we use MAX height inside the bbox, not MEAN. A bbox
containing one tall tree surrounded by shrubs should be kept as a tree.
Mean would dilute the tall tree with the surrounding ground and fail.

## Data source status (2026-04)

ICGC publishes a DTM via WCS at 5m resolution (coverage `icgc__met5` on
https://geoserveis.icgc.cat/geoserver/wcs) — verified, returns real float32
elevation in meters.

ICGC does NOT publish the DSM via WCS. The DSM is only served through WMS,
which returns a rendered grayscale visualization (uint8), not raw float
elevation. This makes automated `CHM = DSM - DTM` from ICGC alone infeasible
without LAZ point cloud processing.

## How to get a CHM raster (external step)

`filter_by_height()` works with ANY CHM GeoTIFF (EPSG:25831, heights in m).
To obtain one for Montseny, pick one:

  1. Download ICGC LiDAR LAZ and compute CHM with laspy/PDAL (best quality).
     LAZ bulk: https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
  2. Copernicus GLO-30 DSM - ICGC met5 DTM (coarse 30m but global + free).
  3. Any other CHM raster in EPSG:25831.

`fetch_chm_for_patch()` is a stub that raises NotImplementedError until one
of the above pipelines is wired in. The filter function itself is complete
and tested.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import supervision as sv
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

# Default threshold: 5m matches Iberian shrub research and Spanish Forest
# Inventory conventions. Shrubs/heather < 5m, trees > 5m.

# Default threshold: trees > 5m, shrubs < 5m (Iberian shrub research).
# Spanish Forest Inventory uses the same cutoff.
DEFAULT_HEIGHT_THRESHOLD = 5.0

RASTER_CACHE = (
    Path(__file__).parent.parent.parent / "data" / "montseny" / "rasters"
)


def fetch_chm_for_patch(
    x_center: float,
    y_center: float,
    patch_size_m: float = 160.0,
    cache_dir: Path = RASTER_CACHE,
) -> Path:
    """Fetch (or look up cached) CHM raster for a patch area.

    NOT IMPLEMENTED AUTOMATICALLY. ICGC does not publish a raw-float DSM
    via WCS — only a rendered WMS visualization that cannot be decoded
    into elevation meters. Automated CHM derivation from ICGC alone is
    therefore not possible without LAZ point cloud processing.

    This function checks for a pre-computed CHM at the cache location
    `{cache_dir}/chm_{x}_{y}.tif`. If found, returns it. Otherwise raises
    NotImplementedError with instructions for generating a CHM externally.

    Args:
        x_center: Patch center X in EPSG:25831.
        y_center: Patch center Y in EPSG:25831.
        patch_size_m: Side length of the square area, in meters.
        cache_dir: Where to look for / cache the CHM.

    Returns:
        Path to CHM GeoTIFF if it exists at the cache location.

    Raises:
        NotImplementedError: If no CHM is cached and automated fetch
            is unavailable. The error message explains how to generate
            one from LAZ or Copernicus GLO-30.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{int(round(x_center))}_{int(round(y_center))}"
    chm_path = cache_dir / f"chm_{key}.tif"

    if chm_path.exists():
        logger.info("CHM cached: %s", chm_path.name)
        return chm_path

    raise NotImplementedError(
        f"No CHM found at {chm_path}. Automated CHM fetch from ICGC is "
        "not possible (DSM not in WCS). To generate one:\n"
        "  1. Download LAZ from ICGC datacloud for this area, or\n"
        "  2. Use Copernicus GLO-30 DSM minus ICGC met5 DTM, or\n"
        "  3. Place any CHM GeoTIFF (EPSG:25831, meters) at the path above.\n"
        "See src/forest_pulse/lidar.py module docstring for details."
    )


def filter_by_height(
    detections: sv.Detections,
    chm_path: Path,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    threshold: float = DEFAULT_HEIGHT_THRESHOLD,
    aggregation: str = "max",
) -> sv.Detections:
    """Drop detections whose canopy height inside the bbox < threshold.

    Args:
        detections: sv.Detections with xyxy boxes in pixel coords of the
            ORIGINAL image.
        chm_path: Path to CHM raster (single band, heights in meters).
        image_bounds: (x_min, y_min, x_max, y_max) geographic bounds.
        image_size_px: (width, height) of the image in pixels.
        threshold: Minimum canopy height to keep a box. Default 5.0 m.
        aggregation: How to combine pixel heights inside a bbox.
            'max' (default): single tall pixel keeps the box — preserves
                tall trees near shrub patches.
            'p95': 95th percentile, more robust to DSM noise.
            'mean': only use for uniform canopy patches (rarely desired).

    Returns:
        Filtered sv.Detections.
    """
    if len(detections) == 0:
        return detections

    # rfdetr adds non-per-detection metadata that breaks boolean indexing.
    # Strip it before filtering (see ndvi.py for the same pattern).
    _strip_rfdetr_metadata(detections)

    keep_mask = np.zeros(len(detections), dtype=bool)

    with rasterio.open(chm_path) as src:
        for i, xyxy in enumerate(detections.xyxy):
            geo_box = _pixel_bbox_to_geo(xyxy, image_bounds, image_size_px)
            height = _sample_raster_agg(src, geo_box, aggregation)
            if height >= threshold:
                keep_mask[i] = True

    n_kept = int(keep_mask.sum())
    logger.info(
        "Height filter: kept %d/%d detections (threshold=%.1fm, agg=%s)",
        n_kept, len(detections), threshold, aggregation,
    )
    return detections[keep_mask]


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Remove rfdetr's non-per-detection fields from detections.data.

    Same helper as ndvi.py (duplicated to keep modules self-contained).
    See ndvi._strip_rfdetr_metadata for the rationale.
    """
    if not hasattr(detections, "data"):
        return
    for key in ("source_shape", "source_image"):
        if key in detections.data:
            del detections.data[key]


def _pixel_bbox_to_geo(
    xyxy_px: np.ndarray,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Convert a pixel-space bbox to geographic coordinates.

    Same transform as ndvi.py — image Y inverts vs world Y.
    Duplicated here instead of shared to avoid cross-module coupling
    (both modules are self-contained filters).
    """
    x_min_geo, y_min_geo, x_max_geo, y_max_geo = image_bounds
    w_px, h_px = image_size_px
    x_scale = (x_max_geo - x_min_geo) / w_px
    y_scale = (y_max_geo - y_min_geo) / h_px
    x1_px, y1_px, x2_px, y2_px = xyxy_px.tolist()
    x_min = x_min_geo + x1_px * x_scale
    x_max = x_min_geo + x2_px * x_scale
    y_max = y_max_geo - y1_px * y_scale
    y_min = y_max_geo - y2_px * y_scale
    return (x_min, y_min, x_max, y_max)


def _sample_raster_agg(
    src: rasterio.io.DatasetReader,
    geo_bounds: tuple[float, float, float, float],
    aggregation: str,
) -> float:
    """Read a raster window and aggregate valid pixels.

    Returns 0.0 if the window is empty or entirely nodata.
    """
    x_min, y_min, x_max, y_max = geo_bounds
    try:
        window = from_bounds(x_min, y_min, x_max, y_max, src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=np.nan)
    except (ValueError, rasterio.errors.WindowError):
        return 0.0

    if data.size == 0:
        return 0.0
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0

    if aggregation == "max":
        return float(valid.max())
    if aggregation == "p95":
        return float(np.percentile(valid, 95))
    if aggregation == "mean":
        return float(valid.mean())
    raise ValueError(f"Unknown aggregation: {aggregation}")
