# Implementation Plan: lidar-ndvi-filtering

**Version:** 1
**Date:** 2026-04-07
**Based on:** SPEC.md + OBSERVE.md + research reports

## Summary

Two new modules (`ndvi.py` + `lidar.py`) each ~100 lines providing pure-function filters that take `sv.Detections` + image geo-bounds + threshold and return filtered `sv.Detections`. Both fetch/cache rasters lazily from ICGC WMS/WCS endpoints. One demo script (`apply_filters_demo.py`) wires everything together on a specific patch to show the pipeline working.

## Files to Create

| File | Purpose | Lines |
|---|---|---|
| `src/forest_pulse/ndvi.py` | NDVI-based non-vegetation filter | ~100 |
| `src/forest_pulse/lidar.py` | LiDAR CHM-based height filter | ~120 |
| `scripts/apply_filters_demo.py` | Demo: detect → filter → compare | ~80 |
| `tests/test_filters.py` | Unit tests with mock rasters | ~60 |

## Files to Modify

| File | Change | Rationale |
|---|---|---|
| `pyproject.toml` | Add `rioxarray>=0.15.0` optional dep | LiDAR raster windowed reads |
| `.gitignore` | Add `data/montseny/rasters/` | Cached CHM/NDVI rasters too large |

## Implementation: `ndvi.py`

**Public API:**
```python
def fetch_ndvi_for_patch(x_center: float, y_center: float, patch_size_m: float = 160) -> Path:
    """Download ICGC CIR orthophoto for patch bbox and compute NDVI.

    Caches result to data/montseny/rasters/ndvi_{x}_{y}.tif.
    Returns path to the NDVI GeoTIFF (single float32 band).
    """

def compute_ndvi_from_cir(cir_path: Path, output_path: Path) -> Path:
    """Compute NDVI from a 3-band CIR GeoTIFF.

    CIR band order (ICGC standard): B1=NIR, B2=Red, B3=Green.
    NDVI = (NIR - Red) / (NIR + Red), clipped to [-1, 1].
    """

def filter_by_ndvi(
    detections: sv.Detections,
    ndvi_path: Path,
    image_bounds: tuple[float, float, float, float],  # (x_min, y_min, x_max, y_max) in EPSG:25831
    image_size_px: tuple[int, int],  # (width, height)
    threshold: float = 0.15,
) -> sv.Detections:
    """Drop detections where mean NDVI inside bbox < threshold.

    Converts pixel-space bboxes → geographic coords using image_bounds.
    Samples NDVI raster inside each box, computes mean, filters.
    """
```

**Internal helpers:**
- `_pixel_bbox_to_geo(xyxy_px, image_bounds, image_size)` — coordinate transform
- `_sample_raster_mean(raster_path, geo_bounds)` — windowed read + mean

**Implementation notes:**
- Uses existing WMS pattern from `download_montseny.py` (swap layer name to `ortofoto_infraroig_vigent`)
- Uses `rasterio.windows.from_bounds()` for efficient sampling
- Lazy import of rasterio inside functions to keep module import fast

## Implementation: `lidar.py`

**Public API:**
```python
def fetch_chm_for_patch(x_center: float, y_center: float, patch_size_m: float = 160) -> Path:
    """Download DSM + DTM for patch bbox, compute CHM = DSM - DTM.

    Uses ICGC WCS at 2m resolution.
    Caches result to data/montseny/rasters/chm_{x}_{y}.tif.
    """

def filter_by_height(
    detections: sv.Detections,
    chm_path: Path,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    threshold: float = 5.0,
    aggregation: str = "max",  # or "p95", "mean"
) -> sv.Detections:
    """Drop detections where canopy height inside bbox < threshold.

    Default uses max height (preserves tall trees near bush patches).
    """
```

**Internal helpers:**
- `_fetch_raster_wcs(layer, bbox, output_path, resolution_m)` — WCS GetCoverage request
- `_sample_raster_max(raster_path, geo_bounds)` — windowed read + max

**Implementation notes:**
- Uses `rasterio.merge` if DSM and DTM come as separate tiles
- Uses `rioxarray` only if needed for tricky windowing — plain rasterio suffices for 2m CHM
- Height aggregation: **max** by default (not mean) — research report recommends this because mean dilutes tall trees with surrounding ground pixels

## Implementation: `apply_filters_demo.py`

```
For patch 0477 (the 157-detection dense canopy case):
  1. Load patch + run detect_trees
  2. Get patch geo-bounds from metadata CSV
  3. Fetch NDVI for this bbox
  4. Apply NDVI filter (threshold 0.15)
  5. Fetch CHM for this bbox
  6. Apply height filter (threshold 5.0)
  7. Print: original count | after NDVI | after LiDAR | what was dropped
  8. Save annotated images at each stage to outputs/filter_demo/
```

## Tests: `test_filters.py`

Unit tests with synthetic rasters (no network needed):

| Done Criterion # | Test | Type |
|---|---|---|
| 1 | `test_ndvi_filter_drops_low_ndvi_boxes` — synthetic NDVI raster with half pixels 0.5 (high) half 0.05 (low), verify boxes over low pixels are dropped | unit |
| 2 | `test_height_filter_drops_short_boxes` — synthetic CHM raster with half pixels at 8m half at 2m, verify boxes over short pixels dropped | unit |
| 1/2 | `test_empty_detections_pass_through` — edge case | unit |
| 1 | `test_pixel_to_geo_bbox_transform` — verify coord conversion correctness | unit |
| 2 | `test_max_aggregation_preserves_tall_trees` — single tall pixel in a box of short pixels should keep the box | unit |

## Approach Alternatives

| Alternative | Rejected Because |
|---|---|
| Modify detect.py to apply filters internally | Violates SRP — detect.py should only detect. Filters are a separate concern. |
| Apply filters inside eval.py | eval.py is LOCKED |
| Use mean NDVI instead of max CHM for LiDAR | Research explicitly says max/p95 > mean for height (mean dilutes) |
| Download full Montseny LiDAR upfront | 270GB for LAZ, wasteful. Fetch per-patch on demand, cache. |
| Integrate with georef.py | georef.py is still a stub, tangles feature with unrelated work |
| Make filters methods on a Filter class | Overkill. Pure functions are simpler and composable. |

## Risks and Side Effects

- **ICGC WCS endpoint reliability**: unauthenticated, might rate-limit. Mitigation: aggressive caching, patient retry logic.
- **LiDAR coverage gaps**: some Montseny tiles might not have LiDAR data. Mitigation: warn + skip filter for those patches (don't crash).
- **CHM temporal mismatch**: LiDAR is from 2021-2023, RGB orthophoto might be newer. Mitigation: note in docstring, acceptable for MVP.
- **Threshold tuning**: 0.15 and 5.0 are research defaults. Might need adjustment on Montseny. Mitigation: make thresholds CLI-configurable in demo.

## Estimated Scope

- 4 new files (~360 lines total)
- 2 modified files (1-2 lines each)
- New dependency: rioxarray (optional)
- Complexity: medium (new raster handling, but pure functions with clear contracts)
