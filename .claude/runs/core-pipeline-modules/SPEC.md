# Feature Spec: Complete Core Pipeline Modules

**Slug:** core-pipeline-modules
**Date:** 2026-04-07
**Status:** draft

## What

Implement the three remaining stub modules so Forest Pulse has a complete
pixel-to-GIS pipeline:

**1. `src/forest_pulse/georef.py`** — convert detections to geographic
features. Replaces the current `NotImplementedError` stub. Given pixel-space
`sv.Detections` + image geographic bounds + (optional) health scores,
produces a `gpd.GeoDataFrame` with one row per tree: center point geometry
in EPSG:25831, tree_id, confidence, bbox coords in meters, and (if
provided) health label + GRVI + ExG.

**2. `src/forest_pulse/temporal.py`** — compare two tree inventories from
different time periods. Replaces current stub. Given two GeoDataFrames
produced by `georef.py`, performs nearest-neighbor spatial matching within
a tolerance (default 2m) to identify: matched trees, trees that disappeared
(in t₀ but not t₁), and new trees (in t₁ but not t₀). Returns a
`ChangeReport` dataclass with counts, matched pairs, and unmatched IDs.

**3. `src/forest_pulse/export.py`** — write tree inventories to standard
GIS + tabular formats. Replaces current stub. Implements:
- `to_geojson(gdf, path)` — reprojects to WGS84 (EPSG:4326) and writes
- `to_csv(gdf, path)` — flat CSV with lat, lon columns
- `to_shapefile(gdf, path)` — Shapefile in original CRS
- `to_change_report(change, path)` — JSON dump of ChangeReport

**4. `scripts/full_pipeline_demo.py`** — end-to-end demo:
patch → detect → health → NDVI filter → georef → export GeoJSON. Produces
a file that can be loaded directly in QGIS showing each tree as a point.

## Why

Phase 1 gave us detection + health + visualization. The project PRD
promised a full pipeline from aerial imagery to GIS-ready output. The
remaining modules are the "useful for a professional forester" layer —
without them, the model is just a research toy. With them, Jordi's father
can load a GeoJSON in QGIS and see every tree in his forest on a map.

This is also the simplest way to make the project a legitimate open-source
forestry tool. GeoJSON + CSV export means anyone can use the model output
in their own workflow (QGIS, ArcGIS, R, Python, web maps).

## Constraints

- Must preserve existing module contracts from `ARCHITECTURE.md` where
  they're reasonable. Where they're not (e.g., georef reading from image
  path), deviate and note why.
- Must use `geopandas` (already a project dependency) — no new dependencies.
- Output GeoDataFrames must have a consistent schema so `temporal.py` and
  `export.py` can work with any output of `georef.py`.
- Must handle empty detections gracefully (return empty GeoDataFrame).
- EPSG:25831 is the canonical CRS throughout the project (matches ICGC).
  Reproject to EPSG:4326 only at GeoJSON export time.
- Temporal matching must be deterministic (no random tie-breaks).
- Max 200 lines per function, 1000 per file, imports at top, type hints
  on public functions.

## Done Criteria

1. GIVEN `sv.Detections` with 5 trees in pixel coords + image bounds
   WHEN `georeference()` runs
   THEN it returns a GeoDataFrame with 5 rows, each with a Point geometry
   in EPSG:25831 at the bbox center, plus tree_id and confidence columns

2. GIVEN `sv.Detections` + matching `list[HealthScore]`
   WHEN `georeference()` runs with health_scores
   THEN the returned GeoDataFrame has health_label, grvi, exg columns

3. GIVEN two GeoDataFrames (t₀ with 10 trees, t₁ with 12 trees, 8 matching
   within 2m, 2 missing at t₁, 4 new at t₁)
   WHEN `compare_periods()` runs
   THEN the ChangeReport shows: trees_before=10, trees_after=12,
   matched has 8 pairs, missing has 2 ids, new has 4 ids

4. GIVEN a GeoDataFrame in EPSG:25831
   WHEN `to_geojson(gdf, path)` runs
   THEN the output file exists, is valid GeoJSON, has CRS EPSG:4326,
   and contains the same number of features as the input

5. GIVEN a GeoDataFrame
   WHEN `to_csv(gdf, path)` runs
   THEN the CSV has one row per tree with lon, lat columns (WGS84)

6. GIVEN a Montseny patch
   WHEN `scripts/full_pipeline_demo.py --patch 0150.jpg` runs
   THEN it produces `outputs/pipeline_demo/0150_trees.geojson` loadable
   in QGIS with tree points and attributes

## Out of Scope

- HTML reports with interactive maps (leaflet/folium) — add later if needed
- KML export (use GeoJSON, it's more modern)
- Tree-crown polygons (only center points for now; polygons are trivial
  to derive later from stored bbox coords)
- Multi-patch batch processing (single patch is enough for MVP)
- Temporal pipeline demo (compare_periods() is implemented, but the demo
  only shows single-time output; a two-time demo needs patches from two
  years which is a separate feature)
- Visualize changes on a map (temporal output is a dataclass, not a
  rendered visualization)
