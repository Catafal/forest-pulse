# Feature Spec: LiDAR + NDVI Detection Filtering

**Slug:** lidar-ndvi-filtering
**Date:** 2026-04-07
**Status:** draft

## What

Two new filters applied to RF-DETR tree detections:

**1. NDVI filter (false positive killer):**
Drop any detection where the mean NDVI inside its bounding box < 0.15. Removes non-vegetation false positives (rocks, roads, buildings, bare soil). Uses ICGC CIR orthophoto (3-band: NIR, Red, Green) via the same WMS we already use for RGB.

**2. LiDAR height filter (tree vs shrub discriminator):**
Drop any detection where the max canopy height (CHM) inside its bounding box < 5m. Removes bushes, heather, and other short vegetation the model currently mislabels as trees. Uses ICGC Canopy Height Model computed as DSM − DTM, fetched via WCS endpoint at 2m resolution.

**Order of application:** `NDVI filter → LiDAR filter → verified tree list`

Filter outputs are drop-in: same `sv.Detections` type in, same type out. Existing visualize/health/export modules work unchanged.

## Why

Our self-trained RF-DETR reaches 0.904 mAP50 on its own labels, but visual review + tree density sanity check revealed the model detects "round green blobs from above" — including bushes, shrubs, and merged canopies. The metric is inflated by self-training feedback bias.

Research (see `.claude/research/`) confirms:
- NDVI cannot distinguish trees from shrubs (both are chlorophyll-rich) but excels at filtering non-vegetation
- LiDAR canopy height at 5m threshold is the industry standard (Spanish Forest Inventory uses exactly this)

For the project to move from "tree-like object detector" to "verified forest inventory tool" (which is what's useful for Jordi's father's forestry work), we need physical verification of each detection.

## Constraints

- Must reuse existing detect.py pipeline — no retraining
- Must NOT break the existing `detect_trees → health → visualize → export` flow
- `sv.Detections` stays as the interchange type between modules
- ICGC data access: no authentication required (open data, attribution)
- Caching: CHM and NDVI rasters must be cached locally (don't re-download on every inference)
- New dependency allowed: `rioxarray` for LiDAR raster handling (NDVI can use existing rasterio)
- All coordinates in EPSG:25831 (matches existing patches)
- Must work on MPS (no GPU-only code)
- Gold eval directory (`data/montseny/eval_gold/`) is still applicable — filters applied before eval

## Done Criteria

1. GIVEN a patch and its sv.Detections WHEN `filter_by_ndvi(detections, image_bounds, ndvi_raster, threshold=0.15)` runs THEN it returns sv.Detections with only high-NDVI boxes
2. GIVEN a patch and its sv.Detections WHEN `filter_by_height(detections, image_bounds, chm_raster, threshold=5.0)` runs THEN it returns sv.Detections with only tall boxes
3. GIVEN a geographic bbox WHEN `fetch_chm_for_aoi(bbox)` runs THEN it downloads/reads cached DSM and DTM at 2m, computes CHM = DSM - DTM, returns path
4. GIVEN a geographic bbox WHEN `fetch_ndvi_for_aoi(bbox)` runs THEN it downloads/reads cached CIR orthophoto, computes NDVI = (NIR-R)/(NIR+R), returns path
5. GIVEN a patch WHEN applied to our current 0.904 mAP model THEN the filter pipeline removes at least some low-NDVI (rocks/roads) and low-height (bushes) false positives visible in output
6. GIVEN patches 0477 (dense canopy, 157 detections) WHEN filters applied THEN filtered count < 157 and remaining detections are physically plausible

## Out of Scope

- Downloading LiDAR for all of Montseny upfront (fetch on-demand per patch bbox)
- Processing raw LAZ point clouds (use pre-computed DSM/DTM rasters only)
- Multi-temporal analysis (same year as existing RGB patches)
- Species classification (beech vs oak vs fir) — separate feature
- Retraining the model on filtered labels — separate feature
- Changes to the demo script CLI (filters are library functions, demo.py can be updated later)
