# Implementation Plan: lidar-feature-extraction

**Version:** 1
**Date:** 2026-04-07
**Based on:** SPEC.md + OBSERVE.md + `.claude/research/icgc_laz_urls.md`

## Summary

Replace the `fetch_chm_for_patch()` stub in `lidar.py` with a working
LAZ-based pipeline: download ICGC LiDAR tiles on demand (verified URL
pattern), parse points with `laspy`, rasterize DSM − DTM = CHM, and
extract 7 per-tree features into a new `LiDARFeatures` dataclass. Add an
optional `lidar_features` parameter to `georef.py` that adds 7 new
columns to the GeoDataFrame when present. One smoke test script and one
downloader script. Unit tests use synthetic point clouds — no network or
real LAZ files required in CI.

## Files to Modify

| File | Change | Lines added |
|---|---|---|
| `src/forest_pulse/lidar.py` | Add `LiDARFeatures` dataclass + `_icgc_laz_url` + `fetch_laz_for_patch` + `_read_laz_points` + `compute_chm_from_laz` + `extract_lidar_features` + helpers. Rewrite `fetch_chm_for_patch` to use the new pipeline. | ~350 |
| `src/forest_pulse/georef.py` | Add optional `lidar_features` parameter. When provided, add 7 new columns (`lidar_height_p95`, `lidar_height_p50`, `lidar_vertical_spread`, `lidar_point_count`, `lidar_return_ratio`, `lidar_intensity_mean`, `lidar_intensity_std`). Backward compatible. | ~20 |
| `pyproject.toml` | Add `[lidar]` optional extra: `laspy[lazrs]>=2.5.0` | 3 |
| `.gitignore` | Add `data/montseny/lidar/` (LAZ cache) | 1 |

## Files to Create

| File | Purpose | Lines |
|---|---|---|
| `scripts/download_lidar.py` | Batch downloader for Montseny zones (CLI: `--zone low`, `--all-zones`, etc.) | ~80 |
| `scripts/lidar_smoke_test.py` | End-to-end verification: download → extract features → print table | ~90 |
| `tests/test_lidar_features.py` | Unit tests using synthetic point clouds (mock `laspy.read`) | ~150 |

## Module Contracts

### `LiDARFeatures` dataclass (new)

```python
@dataclass
class LiDARFeatures:
    tree_id: int
    height_p95_m: float        # canopy top — use for > 5m filter
    height_p50_m: float        # median, crown center
    vertical_spread_m: float   # p95 - p5, proxies trunk + crown depth
    point_count: int           # LiDAR returns inside crown region
    return_ratio: float        # n_returns > 1 / total (canopy density)
    intensity_mean: float      # needle vs broadleaf reflectance
    intensity_std: float       # crown heterogeneity
```

### New public API in `lidar.py`

```python
def fetch_laz_for_patch(
    x_center: float, y_center: float,
    cache_dir: Path | None = None,
) -> Path:
    """Download (or return cached) ICGC LAZ tile for the patch center.

    Uses the verified URL pattern:
      https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
        full10km{ID10K}/lidar-territorial-v3r1-full1km{ID1K}-2021-2023.laz

    Tile IDs computed from EPSG:25831 coordinates via _icgc_laz_url.
    Cache miss → download ~400-700 MB to cache_dir. Cache hit → instant.
    """

def compute_chm_from_laz(
    laz_path: Path,
    bounds: tuple[float, float, float, float],
    resolution_m: float = 0.5,
    cache_dir: Path | None = None,
) -> Path:
    """Rasterize a CHM (DSM − DTM) from a LAZ file for a given bbox.

    Grids ground points (class=2) to DTM and highest returns (first
    return, any class) to DSM, then CHM = max(DSM - DTM, 0). Writes
    the result as a single-band float32 GeoTIFF in EPSG:25831.
    Returns the path to the CHM file (cached by bounds hash).
    """

def extract_lidar_features(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    laz_path: Path,
) -> list[LiDARFeatures]:
    """Extract per-tree LiDAR features for each detection.

    For each detection: project its pixel bbox → geo bbox, select LAZ
    points inside the bbox, compute the 7 features.
    Returns a list of LiDARFeatures in the same order as detections.
    Empty detections → empty list.
    """
```

### Changes to existing `fetch_chm_for_patch`

Replace the NotImplementedError stub with:

```python
def fetch_chm_for_patch(
    x_center: float, y_center: float,
    patch_size_m: float = 160.0,
    cache_dir: Path = RASTER_CACHE,
) -> Path:
    """Download LAZ for the patch center, rasterize CHM for patch bounds."""
    laz_path = fetch_laz_for_patch(x_center, y_center)
    half = patch_size_m / 2.0
    bounds = (
        x_center - half, y_center - half,
        x_center + half, y_center + half,
    )
    return compute_chm_from_laz(laz_path, bounds, resolution_m=0.5, cache_dir=cache_dir)
```

This is a ~10-line rewrite. The existing `filter_by_height()` already
works; now `fetch_chm_for_patch()` actually returns a real CHM.

### `georef.py` change

```python
def georeference(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    health_scores: list[HealthScore] | None = None,
    lidar_features: list["LiDARFeatures"] | None = None,  # NEW
    crs: str = "EPSG:25831",
) -> gpd.GeoDataFrame:
```

When `lidar_features is not None`, the output schema gains:
- `lidar_height_p95`, `lidar_height_p50`, `lidar_vertical_spread`
- `lidar_point_count`, `lidar_return_ratio`
- `lidar_intensity_mean`, `lidar_intensity_std`

All rounded to 2 decimals. Identical behavior to before when
`lidar_features is None` (backward compat).

## Internal helpers (in lidar.py)

```python
def _icgc_laz_url(x: float, y: float) -> str:
    """Compute the verified ICGC LAZ tile URL for a point in EPSG:25831."""

def _read_laz_points(laz_path: Path) -> dict:
    """Read all points + classification + intensity + returns from a LAZ file.

    Returns dict with numpy arrays:
        x, y, z: (N,) float32
        classification: (N,) uint8
        intensity: (N,) uint16
        return_number: (N,) uint8
        number_of_returns: (N,) uint8
    Lazy imports laspy.
    """

def _rasterize_min_max(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    bounds: tuple, resolution_m: float, mode: str,
) -> np.ndarray:
    """Grid scattered points into a 2D raster using min or max per cell."""

def _features_from_points(
    points: dict,
    geo_bbox: tuple[float, float, float, float],
    tree_id: int,
) -> LiDARFeatures:
    """Compute the 7 per-tree features from points inside a bbox.

    Uses `z - min_ground_z_in_bbox` as height above ground (small bbox
    approximation — no full DTM subtraction per tree, which would be
    overkill at this scale).
    """
```

## Downloader script

```
scripts/download_lidar.py

Usage:
  python scripts/download_lidar.py --zone low
  python scripts/download_lidar.py --all-zones
  python scripts/download_lidar.py --x 450000 --y 4625000

Iterates ZONES from download_montseny.py, computes the unique LAZ
tile URL for each zone center, downloads if missing. Deduplicates
URLs (adjacent zones often share a LAZ tile). Prints total size.
```

## Smoke test script

```
scripts/lidar_smoke_test.py --patch 0250.jpg

Pipeline:
  1. Load patch 0250 (center coords from metadata CSV)
  2. Run RF-DETR detection → get 6 detections
  3. fetch_laz_for_patch(center) → local LAZ path
  4. extract_lidar_features(dets, bounds, size, laz_path)
  5. Print table of 5 features:
       tree_id | height_p95 | spread | point_count | intensity_mean
  6. Apply filter_by_height via the existing filter_by_height() call
     using the CHM produced by compute_chm_from_laz
  7. Print kept/dropped count
```

## Tests

All unit tests mock `laspy.read` to return synthetic point clouds. No
real LAZ files needed, CI stays fast.

| Done # | Test | Type |
|---|---|---|
| — | `test_icgc_laz_url_encoding` — verify (450000, 4625000) → known URL | unit |
| — | `test_icgc_laz_url_different_quadrant` — another coord sanity check | unit |
| 2 | `test_compute_chm_produces_heights` — synthetic tall + ground points → CHM values > 0 | unit |
| 2 | `test_compute_chm_empty_tile` — no points → zeros raster | unit |
| 3 | `test_extract_features_populates_all_fields` — synthetic points → LiDARFeatures with all 7 fields filled | unit |
| 4 | `test_extract_features_mature_tree_height_gt_5` — synth tall points → p95 > 5 | unit |
| — | `test_extract_features_empty_bbox` — no points in bbox → zero point_count, height_p95 = 0.0 | unit |
| 5 | `test_filter_by_height_still_works_with_real_chm_shape` — existing filter + our CHM output format | integration |
| 6 | `test_georef_with_lidar_columns` — georeference with lidar_features adds 7 columns | unit |
| 6 | `test_georef_without_lidar_unchanged` — no lidar_features → schema identical to current | unit |
| 8 | All 52 existing tests still pass | regression |

## Implementation order (strict)

Each step passes its tests before moving on — if something breaks, we
localize it immediately instead of debugging later.

1. **Install `laspy[lazrs]`** and verify import works
2. **`.gitignore`** for LAZ cache directory
3. **`_icgc_laz_url()` + tests** — pure function, no network
4. **`_read_laz_points()` + tests** with mocked laspy
5. **`_rasterize_min_max()` + tests** — pure numpy, synthetic points
6. **`compute_chm_from_laz()` + tests** — composes the above
7. **`_features_from_points()` + tests** — pure numpy
8. **`extract_lidar_features()` + tests** — composes above with bbox iteration
9. **`fetch_laz_for_patch()` with urllib** — real network, hit the
   verified Montseny URL, cache result
10. **Rewrite `fetch_chm_for_patch()`** — drop-in replacement
11. **`georef.py` add optional `lidar_features` param + tests**
12. **`scripts/download_lidar.py`** — standalone CLI
13. **`scripts/lidar_smoke_test.py`** — full end-to-end verification
14. **Full pytest run** — all 52+ new tests green
15. **Smoke test run** — actually downloads a LAZ (~645 MB), verifies
    features come out sensible (height > 5m for real trees)

## Approach alternatives

| Alternative | Rejected because |
|---|---|
| Use `pdal` instead of `laspy` | Heavier C++ dep, overkill for what we need |
| Parse LAZ via the tile-index GeoJSON first (to validate URL) | Extra 6.56 MB download per run, unnecessary given the URL formula is verified |
| Download the entire 12 GB of Montseny LAZ upfront | Violates MVP scope — download on demand |
| Use rasterio's `rasterio.merge` with multiple DSM/DTM rasters | ICGC doesn't publish rasters for DSM (that's why we're doing LAZ in the first place) |
| Build a full CHM pyramid per-zone | Overkill — one CHM per patch is enough |
| Trust ICGC's vegetation classification for DSM | Safer to use first-return max for DSM and classification==2 for DTM |
| Share the `_pixel_bbox_to_geo` helper via a utils module | Already duplicated across ndvi/lidar/georef/segment — 4 more lines is cheaper than a utils dependency |
| Cache CHM rasters by full bounds hash | Simpler: cache by integer rounded center coords (same as NDVI module) |

## Risks and mitigations

- **`laspy[lazrs]` wheel unavailable**: fall back to `laspy[laszip]`.
  Mitigation: the install command tries `[lazrs]` first and logs clearly
  on failure.
- **LAZ download times out**: use `urllib.request.urlretrieve` with a
  long timeout; if it fails, raise with the URL in the message so user
  can retry manually.
- **Memory spike when loading 645 MB LAZ**: read once, process, discard.
  ~2-3 GB peak on decompression, acceptable on both Macs.
- **LAZ points outside bbox** (because we load whole tile, process just
  a patch): filter by coordinate masking before feature computation.
  O(N) not O(tree × N).
- **Tile boundary trees**: small bbox = unlikely to be cut off;
  documented as known limitation.
- **Network flakiness during smoke test**: once cached, smoke test is
  instant — run once, iterate fast afterward.
- **Writing CHM rasters to cache directory**: use same pattern as
  `ndvi.py` — `data/montseny/rasters/` with filename keyed by integer
  rounded coordinates.

## Verification

1. `pip install -e ".[lidar]"` succeeds, `laspy` imports
2. `python -c "from forest_pulse.lidar import fetch_laz_for_patch, extract_lidar_features, LiDARFeatures"` imports cleanly
3. `pytest tests/` → all 52+ new tests pass
4. `ruff check` → clean
5. `python scripts/lidar_smoke_test.py --patch 0250.jpg` →
   - Downloads ~645 MB LAZ (one time)
   - Extracts features for 6 detections
   - Prints table showing height_p95 values > 5m for real trees
   - Runs filter_by_height on the CHM, reports kept/dropped
6. `python scripts/full_pipeline_demo.py --patch 0250.jpg` — unchanged
   (backward compat)

## Estimated scope

- 1 modified module (`lidar.py`): ~350 lines added
- 1 modified module (`georef.py`): ~20 lines added
- 3 new files: ~320 lines total
- 2 small config changes: `.gitignore` + `pyproject.toml`
- 1 new dependency: `laspy[lazrs]`
- Complexity: medium — lots of numpy but each piece is small
- Time: ~3-4 hours at the declared SOP quality bar
