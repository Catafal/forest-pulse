"""NDVI-based false positive filter for tree detections.

NDVI (Normalized Difference Vegetation Index) = (NIR - Red) / (NIR + Red).
Ranges from -1 (water/snow) to +1 (dense healthy vegetation). Values below
~0.15 indicate non-vegetation (rocks, roads, buildings, bare soil).

This filter drops detections where mean NDVI inside the bounding box is
below a threshold — killing false positives that no amount of RGB training
can distinguish (e.g., a brown rock that looks like a tree crown shadow).

NDVI alone CANNOT distinguish trees from bushes — both are chlorophyll-rich
vegetation with overlapping NDVI ranges (~0.3-0.6). For tree/shrub
discrimination, use the LiDAR height filter in lidar.py.

Data source: ICGC CIR orthophoto (3-band: NIR, Red, Green) via WMS.
Same endpoint as the RGB orthophoto we already use — just different layer.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
import supervision as sv
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

# ICGC CIR (Color Infrared) WMS endpoint — same as RGB, different layer.
# Band order in the returned TIFF: B1=NIR, B2=Red, B3=Green.
# This is the ICGC convention for CIR composites.
WMS_BASE = "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms"
WMS_LAYER_CIR = "ortofoto_infraroig_vigent"
RESOLUTION_M = 0.25  # 25 cm/pixel, matches existing RGB patches

# Default threshold: below 0.15 is almost certainly non-vegetation.
# Healthy forest is typically 0.6-0.9, stressed 0.3-0.5, bare soil <0.1.
DEFAULT_NDVI_THRESHOLD = 0.15

RASTER_CACHE = (
    Path(__file__).parent.parent.parent / "data" / "montseny" / "rasters"
)


def fetch_ndvi_for_patch(
    x_center: float,
    y_center: float,
    patch_size_m: float = 160.0,
    cache_dir: Path = RASTER_CACHE,
) -> Path:
    """Download ICGC CIR for a patch area and compute NDVI raster.

    Fetches a single CIR orthophoto tile covering the patch bounding box,
    computes per-pixel NDVI, saves as single-band float32 GeoTIFF.
    Caches the NDVI file — subsequent calls for the same area return cached.

    Args:
        x_center: Patch center X in EPSG:25831.
        y_center: Patch center Y in EPSG:25831.
        patch_size_m: Side length of the square area to fetch, in meters.
            Default 160m matches our 640px @ 0.25m patches.
        cache_dir: Directory for caching downloaded + computed rasters.

    Returns:
        Path to the NDVI GeoTIFF (single float32 band, EPSG:25831).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache key: rounded center coords (avoids float precision issues)
    key = f"{int(round(x_center))}_{int(round(y_center))}"
    ndvi_path = cache_dir / f"ndvi_{key}.tif"
    cir_path = cache_dir / f"cir_{key}.tif"

    if ndvi_path.exists():
        logger.debug("NDVI cached: %s", ndvi_path.name)
        return ndvi_path

    # Compute patch bounds in EPSG:25831
    half = patch_size_m / 2
    x_min = x_center - half
    y_min = y_center - half
    x_max = x_center + half
    y_max = y_center + half

    # Pixel size for WMS request (at native 0.25m resolution)
    width = int(patch_size_m / RESOLUTION_M)
    height = width

    # Download CIR tile if we don't have it
    if not cir_path.exists():
        url = (
            f"{WMS_BASE}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            f"&LAYERS={WMS_LAYER_CIR}"
            f"&STYLES=&SRS=EPSG:25831"
            f"&BBOX={x_min},{y_min},{x_max},{y_max}"
            f"&WIDTH={width}&HEIGHT={height}"
            f"&FORMAT=image/tiff"
        )
        logger.info("Downloading CIR tile: %s", cir_path.name)
        urllib.request.urlretrieve(url, cir_path)

    # Compute NDVI from CIR
    compute_ndvi_from_cir(cir_path, ndvi_path)
    return ndvi_path


def compute_ndvi_from_cir(cir_path: Path, output_path: Path) -> Path:
    """Compute NDVI from a 3-band CIR GeoTIFF and save as single-band.

    ICGC CIR band order: B1=NIR, B2=Red, B3=Green (NOT RGB).
    NDVI = (NIR - Red) / (NIR + Red).

    Uses float64 during computation to avoid uint8 division bugs,
    outputs float32 to keep file sizes reasonable.

    Args:
        cir_path: Path to 3-band CIR GeoTIFF.
        output_path: Where to write the NDVI raster.

    Returns:
        Path to the written NDVI file.
    """
    with rasterio.open(cir_path) as src:
        # Band 1 = NIR, Band 2 = Red (ICGC CIR standard)
        nir = src.read(1).astype(np.float64)
        red = src.read(2).astype(np.float64)
        profile = src.profile

    # NDVI computation with division-by-zero guard
    denominator = nir + red
    denominator[denominator == 0] = 1.0  # avoid NaN; pixel will be 0 anyway
    ndvi = (nir - red) / denominator
    # Clip to the valid NDVI range (numerical safety)
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)

    # Save as single-band float32 GeoTIFF preserving the CRS + transform
    profile.update(count=1, dtype=rasterio.float32, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    logger.debug("NDVI written: %s (mean=%.3f)", output_path.name, float(ndvi.mean()))
    return output_path


def filter_by_ndvi(
    detections: sv.Detections,
    ndvi_path: Path,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    threshold: float = DEFAULT_NDVI_THRESHOLD,
) -> sv.Detections:
    """Drop detections whose mean NDVI inside the bbox is below threshold.

    Converts each pixel-space bbox to EPSG:25831 coordinates using
    image_bounds, samples the NDVI raster inside that geographic window,
    and keeps only boxes where the mean value >= threshold.

    Args:
        detections: sv.Detections with xyxy boxes in pixel coords of the
            ORIGINAL image (not relative to the raster).
        ndvi_path: Path to NDVI raster (single band, EPSG:25831).
        image_bounds: (x_min, y_min, x_max, y_max) geographic bounds of the
            image whose pixel coordinates the detections refer to.
        image_size_px: (width, height) of that image in pixels.
        threshold: Minimum mean NDVI to keep a detection. Default 0.15.

    Returns:
        Filtered sv.Detections containing only boxes passing the threshold.
    """
    if len(detections) == 0:
        return detections

    # rfdetr adds `source_shape` (tuple) and `source_image` (full-res
    # ndarray) to detections.data. Both break supervision's boolean
    # indexing because their length doesn't match n_detections.
    # Strip them so downstream [mask] indexing works.
    _strip_rfdetr_metadata(detections)

    keep_mask = np.zeros(len(detections), dtype=bool)

    with rasterio.open(ndvi_path) as src:
        for i, xyxy in enumerate(detections.xyxy):
            # Convert pixel bbox → geographic bbox in raster CRS
            geo_box = _pixel_bbox_to_geo(
                xyxy, image_bounds, image_size_px,
            )
            mean_ndvi = _sample_raster_mean(src, geo_box)
            if mean_ndvi >= threshold:
                keep_mask[i] = True

    n_kept = int(keep_mask.sum())
    logger.info(
        "NDVI filter: kept %d/%d detections (threshold=%.2f)",
        n_kept, len(detections), threshold,
    )
    return detections[keep_mask]


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Remove rfdetr's non-per-detection fields from detections.data.

    rfdetr's predict() adds 'source_shape' (tuple) and 'source_image'
    (ndarray the size of the input image) to detections.data. Neither
    has length == n_detections, so supervision's boolean indexing
    (`detections[mask]`) crashes. Stripping them is safe — we don't
    need them downstream.
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
    """Convert a pixel-space bbox to geographic coordinates (EPSG:25831).

    Pixel (0,0) is top-left; Y axis points DOWN in image space but UP in
    geographic space — we invert Y accordingly.
    """
    x_min_geo, y_min_geo, x_max_geo, y_max_geo = image_bounds
    w_px, h_px = image_size_px

    # Meters per pixel in each axis
    x_scale = (x_max_geo - x_min_geo) / w_px
    y_scale = (y_max_geo - y_min_geo) / h_px

    x1_px, y1_px, x2_px, y2_px = xyxy_px.tolist()

    # Y inversion: image y=0 is top (= y_max_geo in world)
    x_min = x_min_geo + x1_px * x_scale
    x_max = x_min_geo + x2_px * x_scale
    y_max = y_max_geo - y1_px * y_scale
    y_min = y_max_geo - y2_px * y_scale

    return (x_min, y_min, x_max, y_max)


def _sample_raster_mean(
    src: rasterio.io.DatasetReader,
    geo_bounds: tuple[float, float, float, float],
) -> float:
    """Read a raster window and return the mean of finite pixel values.

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
    return float(valid.mean())
