# Codebase Observation: lidar-verified-eval-metric

**Date:** 2026-04-07
**GitNexus available:** no

## Existing eval.py — what we must NOT touch

`autoresearch/eval.py` (LOCKED):
  - `evaluate(model_path) → float` — returns mAP50, prints `val_map50: X.XXXX`
  - Reads ground truth from `data/rfdetr/valid/_annotations.coco.json`
    (the self-trained labels — biased)
  - Loads checkpoint via `rfdetr.RFDETRBase(pretrain_weights=...)`
  - Strips `source_shape` metadata from rfdetr output (we know that gotcha)
  - Uses `supervision.metrics.MeanAveragePrecision`

We add a NEW file `autoresearch/eval_lidar.py` that follows the same
overall structure but uses LiDAR-derived ground truth instead. We do
not touch eval.py at all.

## Existing lidar.py — what we reuse

Already shipped in Phase 7:
  - `fetch_laz_for_patch(x_center, y_center)` → cached LAZ path
  - `compute_chm_from_laz(laz_path, bounds, resolution_m)` → CHM GeoTIFF
  - `extract_lidar_features(...)` → list[LiDARFeatures]
  - `LiDARFeatures` dataclass

We use `compute_chm_from_laz` to get the CHM raster, then read it back
with rasterio to apply local-maximum filtering for tree-top detection.

## Patch metadata format

`data/montseny/patches_metadata.csv` columns:
  filename,zone,source_tile,x_center,y_center,exg

Example: `0001.jpg,high,tile_01_02.tif,453768.1,4626167.9,58.37`

This is the lookup table for patch geographic centers. The smoke test
script (`scripts/lidar_smoke_test.py`) already uses this CSV — we
follow the same lookup pattern.

## scipy availability

`scipy 1.17.1` is already installed (transitive dep via geopandas /
rasterio). `scipy.ndimage.gaussian_filter` and `scipy.ndimage.maximum_filter`
are exactly what we need for tree-top detection. **No new dependency.**

## Default eval set design

The natural reference set is the 10 patches we used in the SAM2 A/B test:
one per Montseny zone plus a couple of extras. They cover the full
diversity of forest types in the dataset.

Default eval set in the CLI:
  ['0043.jpg', '0158.jpg', '0250.jpg', '0357.jpg', '0477.jpg',
   '0547.jpg', '0642.jpg', '0756.jpg', '0092.jpg', '0278.jpg']

For each patch we already know the geographic center from the metadata
CSV. The LAZ tile for each patch will be downloaded once and cached
(some patches share LAZ tiles → fewer downloads than patches).

## RF-DETR detection contract

```python
from forest_pulse.detect import detect_trees
detections = detect_trees(image, model_name="checkpoints/current.pt", confidence=0.3)
```
Returns `sv.Detections` with `xyxy` (pixel coordinates) and `confidence`.
Our matching algorithm needs detection center in WORLD coordinates, so
we convert via the same `_pixel_bbox_to_geo` pattern used elsewhere.

## Pixel → world coordinate conversion (pattern)

Already duplicated in ndvi.py / lidar.py / segment.py / georef.py:
```python
def _pixel_bbox_to_geo(xyxy_px, image_bounds, image_size_px):
    x_min_geo, y_min_geo, x_max_geo, y_max_geo = image_bounds
    w_px, h_px = image_size_px
    x_scale = (x_max_geo - x_min_geo) / w_px
    y_scale = (y_max_geo - y_min_geo) / h_px
    x1, y1, x2, y2 = xyxy_px.tolist()
    return (
        x_min_geo + x1 * x_scale,
        y_max_geo - y2 * y_scale,
        x_min_geo + x2 * x_scale,
        y_max_geo - y1 * y_scale,
    )
```
We need just the center (x, y) from this. Add a small dedicated helper
in the new module to keep it self-contained.

## Tree-top detection algorithm (standard forestry)

Local-maximum filter on a smoothed CHM:
1. Optionally smooth CHM with a small Gaussian (σ = 1 px = 0.5 m).
   Reduces speckle without merging adjacent crown peaks.
2. Apply `maximum_filter(smoothed, size=window)` — each pixel gets the
   max value in a window around it.
3. A pixel is a local maximum if `smoothed[i,j] == max_filtered[i,j]`.
4. Drop maxima below the height threshold (5 m) — these are bushes.
5. Convert pixel positions → world coordinates via the raster transform.

Window size: 7 px = 3.5 m radius. A typical Mediterranean tree crown
is 4-8 m wide so the radius matches "half a typical crown".

## Greedy matching algorithm

```python
For each truth tree (sorted somehow — order doesn't matter for greedy):
    For each unmatched prediction within tolerance_m:
        Pick the closest one
    If found:
        Mark both as matched
    Else:
        Truth becomes a false negative
Remaining unmatched predictions become false positives.
```

Order-independence isn't quite true for greedy matching, but the
project's matching tolerance (2 m) is much smaller than typical tree
spacing (~5+ m) so collisions are rare. Acceptable simplification.

## EvalResult composition over patches

Aggregating across multiple patches:
- `n_truth_total = sum(n_truth_i)`
- `n_pred_total = sum(n_pred_i)`
- `n_tp_total = sum(n_tp_i)`
- precision = n_tp_total / (n_tp_total + n_fp_total)
- recall = n_tp_total / (n_tp_total + n_fn_total)
- F1 = 2 * P * R / (P + R)

This is the standard "micro-average" — counts pool across patches before
computing the ratio. Equivalent to weighting each tree equally (as
opposed to "macro-average" which would weight each patch equally).

Micro-average is the right choice here because we care about overall
detection quality, not per-patch averaging.

## Architectural constraints

- Max 200 lines per function, 1000 per file
- Imports at top
- Type hints + Google docstrings on public functions
- Pure functions where possible
- Strip rfdetr metadata before iterating detections (the recurring
  source_shape gotcha)

## Conditional skill routing

- [ ] /plan-eng-review — not applicable (focused scope, contracts clear)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable
