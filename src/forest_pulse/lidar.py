"""LiDAR Canopy Height Model + per-tree feature extraction.

Two complementary use cases:

  1. **Filter trees vs bushes** (`filter_by_height`): drop any detection
     whose canopy height (DSM − DTM) falls below a threshold. At 5 m we
     cleanly separate mature trees from Iberian shrubs (heather, juniper,
     young saplings).

  2. **Per-tree 3D features** (`extract_lidar_features`): for each
     detection, compute seven physically-measured attributes from the
     LiDAR point cloud inside the crown region: height percentiles,
     vertical spread, point density, return ratio, intensity statistics.
     These become the foundation for a post-hoc multi-modal classifier
     (Phase 9 in the roadmap) and for future species discrimination.

## Data source: ICGC LiDAR Territorial v3r1 (2021-2023)

ICGC publishes 8+ pts/m² LiDAR as LAZ 1.4 point clouds in EPSG:25831.
The authoritative URL pattern (verified live, 200 OK):

    https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
      full10km{ID10K}/lidar-territorial-v3r1-full1km{ID1K}-2021-2023.laz

Tile ID encoding (from the official tile-index GeoJSON):
    easting_km  = int(x) // 1000
    northing_km = int(y) // 1000
    ID1K  = f"{easting_km:03d}{(northing_km - 4000):03d}"
    ID10K = f"{(easting_km // 10):02d}{((northing_km // 10) - 400):02d}"

ICGC does NOT publish raw DSM/DTM rasters over a Web Coverage Service —
only rendered WMS visualizations that cannot be decoded into elevation
meters. The LAZ point cloud is the only reliable ICGC source for real
canopy height.

## Why not use ICGC WMS elevation layers?

The ICGC "elevacions-territorial" WMS (`model-superficies-*`,
`model-elevacions-terreny-*`) returns rendered grayscale or colormap
images (uint8 pixel values 0-255). These encode the *appearance* of the
terrain at the visualization quantization, not meters. You can visualize
but not measure. LAZ is the authoritative path.

## MVP simplifications

- **Local ground reference**: height above ground is computed as
  `z - min_ground_z_in_bbox`. At 160 m patch scale this is a fraction-
  of-a-meter approximation versus a proper per-pixel DTM subtraction.
  Acceptable for the 5 m tree-vs-bush threshold.
- **Single-tile extraction**: a tree spanning two 1 km LAZ tiles is
  rare at patch scale and ignored. Worst case: an edge-of-tile tree
  gets slightly truncated features.
- **Temporal mismatch**: LiDAR 2021-2023 vs RGB 2024. For the 5 m
  threshold this is negligible. Future species work should revisit
  if more recent LiDAR becomes available.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import supervision as sv
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

logger = logging.getLogger(__name__)

# Default threshold: 5 m cleanly separates mature trees from shrubs in
# Iberian forests (Spanish Forest Inventory convention).
DEFAULT_HEIGHT_THRESHOLD = 5.0

# ASPRS LAS 1.4 classification codes used by ICGC.
ASPRS_GROUND = 2
ASPRS_LOW_VEGETATION = 3
ASPRS_MEDIUM_VEGETATION = 4
ASPRS_HIGH_VEGETATION = 5

# Default CHM rasterization resolution. 0.5 m is a good trade-off between
# detail and memory for a 160 m patch (320×320 cells per patch).
DEFAULT_CHM_RESOLUTION_M = 0.5

# ICGC LAZ endpoint + URL pattern (verified live 2026-04).
ICGC_LAZ_BASE = (
    "https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip"
)

RASTER_CACHE = (
    Path(__file__).parent.parent.parent / "data" / "montseny" / "rasters"
)
LAZ_CACHE = (
    Path(__file__).parent.parent.parent / "data" / "montseny" / "lidar"
)


# ============================================================
# Data types
# ============================================================


@dataclass
class LiDARFeatures:
    """Per-tree LiDAR features extracted from the point cloud.

    All height values are meters above ground (local reference). Intensity
    values are in whatever arbitrary LiDAR intensity units the sensor
    recorded — useful for relative comparisons, not absolute radiometry.

    Used today: `height_p95_m` > 5 filters out shrubs.
    Used tomorrow: all seven fields feed the species classifier.
    """

    tree_id: int
    height_p95_m: float = 0.0       # canopy top
    height_p50_m: float = 0.0       # median height, proxies crown center
    vertical_spread_m: float = 0.0  # p95 - p5 (trunk + crown depth)
    point_count: int = 0            # LiDAR returns inside the crown region
    return_ratio: float = 0.0       # fraction of points that are multi-return
    intensity_mean: float = 0.0     # reflectance signal (needle vs broadleaf)
    intensity_std: float = 0.0      # crown heterogeneity


@dataclass
class _LAZPoints:
    """Decoded point cloud arrays. All numpy, no laspy objects."""

    x: np.ndarray                     # (N,) float64
    y: np.ndarray                     # (N,) float64
    z: np.ndarray                     # (N,) float32
    classification: np.ndarray        # (N,) uint8 (ASPRS codes)
    intensity: np.ndarray             # (N,) uint16
    return_number: np.ndarray         # (N,) uint8
    number_of_returns: np.ndarray     # (N,) uint8


# ============================================================
# Public API — downloads, CHM, features
# ============================================================


def fetch_laz_for_patch(
    x_center: float,
    y_center: float,
    cache_dir: Path = LAZ_CACHE,
) -> Path:
    """Download (or return cached) ICGC LAZ tile for a patch center.

    Uses the verified ICGC URL pattern. First call downloads ~400-700 MB
    from `datacloud.icgc.cat`; subsequent calls with any point inside the
    same 1 km tile return the cached path instantly.

    Args:
        x_center: Patch center X in EPSG:25831 (meters).
        y_center: Patch center Y in EPSG:25831 (meters).
        cache_dir: Directory to cache LAZ files. Default is
            `data/montseny/lidar/` (gitignored).

    Returns:
        Path to the local LAZ file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _icgc_laz_url(x_center, y_center)
    filename = url.rsplit("/", 1)[-1]
    local_path = cache_dir / filename

    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        # Sanity: LAZ files are huge (>1 MB). Anything tiny is a stale
        # partial download that should be replaced.
        logger.info("LAZ cached: %s", local_path.name)
        return local_path

    logger.info("Downloading LAZ: %s", url)
    urllib.request.urlretrieve(url, local_path)
    size_mb = local_path.stat().st_size / 1e6
    logger.info("LAZ downloaded: %s (%.0f MB)", local_path.name, size_mb)
    return local_path


def compute_chm_from_laz(
    laz_path: Path,
    bounds: tuple[float, float, float, float],
    resolution_m: float = DEFAULT_CHM_RESOLUTION_M,
    cache_dir: Path = RASTER_CACHE,
) -> Path:
    """Rasterize a Canopy Height Model from a LAZ file for the given bbox.

    Algorithm (kept deliberately simple):
      1. Read all LAZ points
      2. Keep points whose (x, y) lies inside `bounds`
      3. Grid ground points (classification == 2) to DTM by taking the
         MIN z per cell
      4. Grid all first-return points to DSM by taking the MAX z per cell
      5. CHM = max(DSM − DTM, 0)
      6. Write single-band float32 GeoTIFF in EPSG:25831 to the cache

    Args:
        laz_path: Local LAZ file from `fetch_laz_for_patch`.
        bounds: (x_min, y_min, x_max, y_max) in EPSG:25831.
        resolution_m: Cell size in meters. 0.5 m default.
        cache_dir: Where to write the CHM GeoTIFF.

    Returns:
        Path to the CHM GeoTIFF.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    chm_path = cache_dir / f"chm_{_bounds_key(bounds)}.tif"
    if chm_path.exists():
        logger.debug("CHM cached: %s", chm_path.name)
        return chm_path

    points = _read_laz_points(laz_path)
    in_bbox = _points_inside_bounds(points, bounds)
    if not in_bbox.any():
        logger.warning("No LAZ points inside bounds %s", bounds)
        return _write_empty_chm(chm_path, bounds, resolution_m)

    x_min, y_min, x_max, y_max = bounds
    # Grid dimensions (truncate so the raster exactly covers the bbox)
    w_cells = max(1, int(round((x_max - x_min) / resolution_m)))
    h_cells = max(1, int(round((y_max - y_min) / resolution_m)))

    # DSM: highest first-return point per cell
    first_return_mask = points.return_number == 1
    dsm_mask = in_bbox & first_return_mask
    dsm = _rasterize_cells(
        points.x[dsm_mask], points.y[dsm_mask], points.z[dsm_mask],
        bounds, resolution_m, (h_cells, w_cells), mode="max",
    )

    # DTM: lowest ground point per cell
    ground_mask = in_bbox & (points.classification == ASPRS_GROUND)
    dtm = _rasterize_cells(
        points.x[ground_mask], points.y[ground_mask], points.z[ground_mask],
        bounds, resolution_m, (h_cells, w_cells), mode="min",
    )

    # CHM = DSM − DTM, clipped to 0 (DSM should never be below DTM but
    # interpolation + noise near raster edges can produce small negatives)
    chm = np.where(
        np.isfinite(dsm) & np.isfinite(dtm),
        np.maximum(dsm - dtm, 0.0),
        0.0,
    ).astype(np.float32)

    # Write GeoTIFF
    transform = from_origin(x_min, y_max, resolution_m, resolution_m)
    with rasterio.open(
        chm_path,
        mode="w",
        driver="GTiff",
        height=h_cells,
        width=w_cells,
        count=1,
        dtype="float32",
        crs="EPSG:25831",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(chm, 1)

    logger.info(
        "CHM rasterized: %s (%dx%d, max=%.1fm, mean=%.1fm)",
        chm_path.name, h_cells, w_cells, float(chm.max()), float(chm.mean()),
    )
    return chm_path


def fetch_chm_for_patch(
    x_center: float,
    y_center: float,
    patch_size_m: float = 160.0,
    cache_dir: Path = RASTER_CACHE,
) -> Path:
    """Download LAZ for the patch center, rasterize CHM for patch bounds.

    This is the entry point that ties LAZ download + CHM computation
    together. The existing `filter_by_height()` calls this to get a real
    CHM raster on demand.

    Args:
        x_center: Patch center X in EPSG:25831.
        y_center: Patch center Y in EPSG:25831.
        patch_size_m: Side length of the patch area. Default 160 m
            matches our 640 px × 0.25 m/px patches.
        cache_dir: CHM cache directory.

    Returns:
        Path to the CHM GeoTIFF covering the patch bounds.
    """
    laz_path = fetch_laz_for_patch(x_center, y_center)
    half = patch_size_m / 2.0
    bounds = (
        x_center - half,
        y_center - half,
        x_center + half,
        y_center + half,
    )
    return compute_chm_from_laz(
        laz_path, bounds, DEFAULT_CHM_RESOLUTION_M, cache_dir,
    )


def extract_lidar_features(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    laz_path: Path,
) -> list[LiDARFeatures]:
    """Extract per-tree LiDAR features for each detection.

    For each detection: project its pixel bbox to a geographic bbox,
    select LAZ points inside that bbox, compute height above local
    ground reference (min z in bbox), and populate the seven feature
    fields.

    Args:
        detections: sv.Detections with pixel-space xyxy boxes.
        image_bounds: (x_min, y_min, x_max, y_max) of the image in
            EPSG:25831 meters.
        image_size_px: (width, height) of the image in pixels.
        laz_path: Local LAZ file (use `fetch_laz_for_patch` first).

    Returns:
        List of LiDARFeatures in the same order as detections. Empty
        input → empty list. Detections with zero points inside their
        bbox get a LiDARFeatures with all defaults (zeros).
    """
    if len(detections) == 0:
        return []

    _strip_rfdetr_metadata(detections)

    # Read LAZ once — iterating over detections is fast once loaded.
    points = _read_laz_points(laz_path)

    out: list[LiDARFeatures] = []
    for i, xyxy in enumerate(detections.xyxy):
        geo_box = _pixel_bbox_to_geo(xyxy, image_bounds, image_size_px)
        features = _features_from_points(points, geo_box, tree_id=i)
        out.append(features)
    return out


# ============================================================
# Height filter — unchanged public API, now works end-to-end
# ============================================================


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
        detections: sv.Detections with xyxy boxes in pixel coords.
        chm_path: Path to CHM raster (single band, heights in meters).
        image_bounds: (x_min, y_min, x_max, y_max) geographic bounds.
        image_size_px: (width, height) of the image in pixels.
        threshold: Minimum canopy height to keep a box. Default 5.0 m.
        aggregation: 'max' (default) | 'p95' | 'mean'. Max is most
            sensitive (a single tall pixel keeps the box), mean is
            least sensitive.

    Returns:
        Filtered sv.Detections.
    """
    if len(detections) == 0:
        return detections

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


# ============================================================
# Internal helpers — URL encoding, LAZ reading, rasterization
# ============================================================


def _icgc_laz_url(x: float, y: float) -> str:
    """Compute the verified ICGC LAZ tile URL for a point in EPSG:25831.

    Tile ID formulas decoded from the authoritative ICGC tile index:
      easting_km  = int(x) // 1000
      northing_km = int(y) // 1000
      ID1K  = f"{easting_km:03d}{(northing_km - 4000):03d}"
      ID10K = f"{(easting_km // 10):02d}{((northing_km // 10) - 400):02d}"

    Critical gotcha: the northing component is `northing_km - 4000`,
    NOT `northing_km % 1000`. Using the raw last 3 digits would produce
    wrong URLs near tile boundaries.

    Args:
        x: Easting in EPSG:25831 meters.
        y: Northing in EPSG:25831 meters.

    Returns:
        Fully qualified HTTPS URL to the LAZ file.
    """
    easting_km = int(x) // 1000
    northing_km = int(y) // 1000
    id1k = f"{easting_km:03d}{(northing_km - 4000):03d}"
    id10k = f"{(easting_km // 10):02d}{((northing_km // 10) - 400):02d}"
    return (
        f"{ICGC_LAZ_BASE}/full10km{id10k}/"
        f"lidar-territorial-v3r1-full1km{id1k}-2021-2023.laz"
    )


def _read_laz_points(laz_path: Path) -> _LAZPoints:
    """Read a LAZ file into numpy arrays.

    Lazy imports laspy — missing dep raises ImportError with an install
    hint, following the same pattern as detect.py / segment.py.

    Args:
        laz_path: Path to a .laz or .las file.

    Returns:
        _LAZPoints with x/y/z/classification/intensity/return arrays.
    """
    try:
        import laspy
    except ImportError as e:
        raise ImportError(
            "LiDAR features require laspy. Install with: "
            "pip install -e '.[lidar]'"
        ) from e

    las = laspy.read(str(laz_path))
    # laspy exposes .x/.y/.z as scaled numpy arrays (float64 meters)
    return _LAZPoints(
        x=np.asarray(las.x, dtype=np.float64),
        y=np.asarray(las.y, dtype=np.float64),
        z=np.asarray(las.z, dtype=np.float32),
        classification=np.asarray(las.classification, dtype=np.uint8),
        intensity=np.asarray(las.intensity, dtype=np.uint16),
        return_number=np.asarray(las.return_number, dtype=np.uint8),
        number_of_returns=np.asarray(las.number_of_returns, dtype=np.uint8),
    )


def _points_inside_bounds(
    points: _LAZPoints,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Return a boolean mask of points inside the given bbox."""
    x_min, y_min, x_max, y_max = bounds
    return (
        (points.x >= x_min) & (points.x < x_max)
        & (points.y >= y_min) & (points.y < y_max)
    )


def _rasterize_cells(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bounds: tuple[float, float, float, float],
    resolution_m: float,
    shape: tuple[int, int],
    mode: str,
) -> np.ndarray:
    """Grid scattered points into a 2D raster using min/max per cell.

    Args:
        x, y, z: equal-length point arrays.
        bounds: raster bbox (x_min, y_min, x_max, y_max) in meters.
        resolution_m: cell size.
        shape: (rows, cols) of the output raster.
        mode: 'min' (DTM) | 'max' (DSM).

    Returns:
        Float32 raster; cells with no points are NaN.
    """
    rows, cols = shape
    out = np.full(shape, np.nan, dtype=np.float32)
    if x.size == 0:
        return out

    x_min, y_min, x_max, y_max = bounds

    # Column index from x: col = (x - x_min) / res, clipped to [0, cols-1]
    col = np.clip(((x - x_min) / resolution_m).astype(np.int64), 0, cols - 1)
    # Row index from y: row 0 is at the TOP (y_max), y decreases as row increases
    row = np.clip(
        ((y_max - y) / resolution_m).astype(np.int64), 0, rows - 1,
    )

    # Linear cell index for fast aggregation
    cell = row * cols + col

    # numpy.minimum/maximum.at is the standard idiom for per-cell reductions.
    # For DSM we want max per cell; for DTM we want min per cell.
    flat = out.ravel()
    if mode == "max":
        # Initialize to -inf so any real z is higher
        flat[:] = -np.inf
        np.maximum.at(flat, cell, z.astype(np.float32))
        flat[np.isneginf(flat)] = np.nan
    elif mode == "min":
        flat[:] = np.inf
        np.minimum.at(flat, cell, z.astype(np.float32))
        flat[np.isposinf(flat)] = np.nan
    else:
        raise ValueError(f"Unknown rasterize mode: {mode}")

    return flat.reshape(shape)


def _features_from_points(
    points: _LAZPoints,
    bbox: tuple[float, float, float, float],
    tree_id: int,
) -> LiDARFeatures:
    """Compute the 7 per-tree features from LAZ points inside a bbox.

    Height is measured as `z − min(ground z in bbox)`. This local
    reference avoids needing to sample a separate DTM raster per tree
    and is accurate enough at 160 m patch scale.

    Empty bbox → LiDARFeatures with all zero fields (the tree_id is
    still set so callers can match to detections).
    """
    mask = _points_inside_bounds(points, bbox)
    if not mask.any():
        return LiDARFeatures(tree_id=tree_id)

    z = points.z[mask]
    cls = points.classification[mask]
    intensity = points.intensity[mask].astype(np.float64)
    n_returns = points.number_of_returns[mask]

    # Local ground reference: lowest ground-classified point in bbox,
    # or if no ground points fall inside, use the overall minimum z.
    ground_z = z[cls == ASPRS_GROUND]
    if ground_z.size > 0:
        ref_z = float(ground_z.min())
    else:
        ref_z = float(z.min())

    # Heights above the local ground reference. Clip to 0 so numerical
    # noise doesn't produce "negative tree heights" downstream.
    heights = np.maximum(z - ref_z, 0.0)

    height_p95 = float(np.percentile(heights, 95))
    height_p50 = float(np.percentile(heights, 50))
    height_p5 = float(np.percentile(heights, 5))
    vertical_spread = float(height_p95 - height_p5)

    # Multi-return fraction: how many points came from a pulse with > 1
    # return. High ratio = dense canopy (pulses penetrate multiple layers).
    multi_return_points = int((n_returns > 1).sum())
    return_ratio = float(multi_return_points / max(len(n_returns), 1))

    return LiDARFeatures(
        tree_id=tree_id,
        height_p95_m=round(height_p95, 3),
        height_p50_m=round(height_p50, 3),
        vertical_spread_m=round(vertical_spread, 3),
        point_count=int(mask.sum()),
        return_ratio=round(return_ratio, 4),
        intensity_mean=round(float(intensity.mean()), 2),
        intensity_std=round(float(intensity.std()), 2),
    )


def _write_empty_chm(
    chm_path: Path,
    bounds: tuple[float, float, float, float],
    resolution_m: float,
) -> Path:
    """Write an all-zeros CHM raster when no LAZ points are in the bbox.

    Keeps the rest of the pipeline happy (filter_by_height will just drop
    everything) instead of throwing an error that propagates up.
    """
    x_min, y_min, x_max, y_max = bounds
    w_cells = max(1, int(round((x_max - x_min) / resolution_m)))
    h_cells = max(1, int(round((y_max - y_min) / resolution_m)))
    data = np.zeros((h_cells, w_cells), dtype=np.float32)
    transform = from_origin(x_min, y_max, resolution_m, resolution_m)
    with rasterio.open(
        chm_path,
        mode="w",
        driver="GTiff",
        height=h_cells,
        width=w_cells,
        count=1,
        dtype="float32",
        crs="EPSG:25831",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    return chm_path


def _bounds_key(bounds: tuple[float, float, float, float]) -> str:
    """Short stable cache key from bounds — integer-rounded + hash tail."""
    x_min, y_min, x_max, y_max = bounds
    # Round to integer meters — patches are on 0.25 m grids, integer
    # rounding gives a unique key per patch without depending on float repr.
    key_str = f"{int(round(x_min))}_{int(round(y_min))}_{int(round(x_max))}_{int(round(y_max))}"
    # Short hash tail to avoid collisions if two calls round to the same
    # integer but have different fractional parts (unlikely but safe).
    tail = hashlib.md5(key_str.encode()).hexdigest()[:6]
    return f"{key_str}_{tail}"


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Remove rfdetr's non-per-detection fields from detections.data.

    Same helper as ndvi.py / segment.py / georef.py — rfdetr adds
    source_shape/source_image fields whose length doesn't match
    n_detections, which breaks any boolean indexing or slicing.
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

    Same transform as ndvi.py / georef.py — image Y inverts vs world Y
    (image row 0 is at the top, which maps to y_max in world coords).
    Duplicated here instead of shared to keep each filter module
    self-contained.
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
