# Feature Spec: LiDAR Feature Extraction

**Slug:** lidar-feature-extraction
**Date:** 2026-04-07
**Status:** draft

## What

Connect ICGC LiDAR point cloud data to Forest Pulse so every detected tree
gets a set of physically-measured 3D features. Three new capabilities:

**1. LAZ tile download and caching** — given a patch center in EPSG:25831,
resolve the correct ICGC LAZ tile URL, download it (if not already cached),
and return the local path. LAZ files stay in `data/montseny/lidar/`
(gitignored; ~400-500 MB per 1km tile).

**2. LAZ → CHM raster** — given a LAZ file and a bounding box, rasterize
the Canopy Height Model (DSM − DTM) at a chosen resolution (default 0.5 m).
This replaces the current `fetch_chm_for_patch()` stub that raises
`NotImplementedError`. The existing `filter_by_height()` function starts
working end-to-end once this lands.

**3. Per-tree LiDAR features** — given detections, image bounds, and a
LAZ file, extract a `LiDARFeatures` dataclass for each detection containing
seven per-tree attributes: `height_p95_m`, `height_p50_m`,
`vertical_spread_m`, `point_count`, `return_ratio`, `intensity_mean`,
`intensity_std`. These become the foundation for the future post-hoc
classifier (Phase 9 in `progress.txt`) and are useful immediately as
richer columns in the GeoJSON export.

**4. Integration:**
- `georef.py` gains an optional `lidar_features` parameter. When provided,
  the GeoDataFrame carries the seven LiDAR columns alongside RGB and
  health columns.
- `scripts/download_lidar.py` — standalone downloader for Montseny zones.
- `scripts/lidar_smoke_test.py` — end-to-end verification on one patch,
  printing before/after counts and a sample of feature values.

## Why

RGB alone cannot distinguish trees from bushes (both are chlorophyll-rich)
and cannot measure 3D structure (crown volume, vertical profile, branching
density). LiDAR gives us both — canopy height, canopy density, and
intensity — and these are exactly the features the future species
classifier will need. This is the **single feature that unblocks the
largest part of the project's future roadmap**.

The `filter_by_height()` function already exists in `lidar.py` but has
been blocked on data access because ICGC only publishes DSM via WMS as a
rendered grayscale visualization. The one working path is LAZ point cloud
processing. This feature closes that gap.

Additionally, LiDAR features enable **auto-labeling for training the
future classifier**:
- `height_p95 > 5m` → auto-label "tree"
- `height_p95 < 2m` → auto-label "bush"

No human annotation needed. This is the biggest compound win from doing
the LAZ work.

## Constraints

- Must work on Apple MPS / CPU (no GPU required — laspy is pure Python numpy).
- Must cache downloads — never re-download LAZ files that are already local.
- Must handle missing `laspy` gracefully — raise `ImportError` with clear
  install instructions, following the SAM2 pattern.
- LAZ files stay gitignored. Feature artifacts (per-tree values) are small
  and can be committed later if useful.
- Must use ICGC's 2021-2023 LiDAR v3r1 (the 8+ pts/m² coverage).
- Must respect the existing `LiDARFeatures` contract — seven typed fields,
  nothing more, nothing less.
- EPSG:25831 internally (same as RGB). No reprojection.
- Must not break existing tests (52 passing). Backward-compatible changes
  only.
- Max 200 lines per function, 1000 per file, imports at top, type hints
  on public functions, Google-style docstrings.

## Done Criteria

1. GIVEN a patch center (EPSG:25831) WHEN `fetch_laz_for_patch(x, y)` runs
   for the first time THEN it downloads the correct ICGC LAZ tile and
   returns its local path. On a second call it returns the cached path
   without re-downloading.

2. GIVEN a LAZ file + a bounding box WHEN `compute_chm_from_laz(path, bounds)`
   runs THEN it returns a `(chm_array, transform)` tuple where `chm_array`
   is a 2D float32 numpy array of heights in meters and values are >= 0.

3. GIVEN detections + image bounds + LAZ path WHEN
   `extract_lidar_features(detections, bounds, size, laz_path)` runs THEN
   it returns a list of `LiDARFeatures` with the same length as detections,
   each having all seven fields filled.

4. GIVEN a detection over a real mature tree in Montseny WHEN extracted
   features are printed THEN `height_p95_m` > 5 and `point_count` > 20.

5. GIVEN the existing `filter_by_height()` WHEN called with a CHM produced
   by `compute_chm_from_laz()` THEN it filters detections correctly (tall
   kept, short dropped) without any API change.

6. GIVEN `georef.py::georeference()` is called with `lidar_features`
   THEN the output GeoDataFrame has seven new columns (`lidar_height_p95`,
   etc.) and the existing columns are unchanged. Without `lidar_features`,
   the output is identical to before (backward compat).

7. GIVEN `scripts/lidar_smoke_test.py --patch 0250.jpg` runs THEN it
   produces a filtered detection count and prints a table of the first
   five trees' LiDAR feature values.

8. GIVEN all existing tests WHEN the suite runs THEN 52 tests still pass
   (backward compatibility).

## Out of Scope

- Phase 8: Building the LiDAR-verified eval metric (this is a separate
  feature; we produce the data, that feature uses it).
- Phase 9: Training the XGBoost classifier on LiDAR features.
- Phase 10-12: Wiring classifier into pipeline, auto-research, species
  classification.
- Downloading LAZ for ALL 8 Montseny zones upfront (MVP downloads on demand).
- Handling trees that span two LAZ tiles (use single-tile extraction; a
  small fraction of edge cases is acceptable).
- Full Montseny inventory run with LiDAR (Phase 10 scope).
- Fine-tuning ICGC point classification (trust the auto-classification that
  ships with the LAZ files).
- GPU-accelerated point cloud processing (pure numpy is fast enough).
- Writing LAZ files back out (we only read).
- Multi-temporal LiDAR comparison (we have one epoch: 2021-2023).
