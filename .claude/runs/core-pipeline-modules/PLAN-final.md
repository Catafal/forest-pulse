# Implementation Plan: core-pipeline-modules

**Version:** 1
**Date:** 2026-04-07
**Based on:** SPEC.md + OBSERVE.md

## Summary

Replace three stub modules (`georef.py`, `temporal.py`, `export.py`) with
working implementations. Use GeoPandas for all GIS operations (already
installed). Keep the schema consistent across modules so they compose
cleanly. Ship a full-pipeline demo that produces a QGIS-loadable GeoJSON.

## Files to Modify (rewrite stubs)

| File | Change | Lines |
|---|---|---|
| `src/forest_pulse/georef.py` | Rewrite: rip out EXIF/GeoTIFF path (our patches are JPEG). Take `image_bounds` + `image_size_px` explicitly. Output GeoDataFrame with tree_id, geometry, confidence, bbox_xyxy, plus health columns if provided. | ~120 |
| `src/forest_pulse/temporal.py` | Fill in `compare_periods()` using `gpd.sjoin_nearest`. Keep dataclasses as-is. Reject matches beyond tolerance. Build TreeMatch list, missing list (t₀ unmatched), new list (t₁ unmatched). | ~90 |
| `src/forest_pulse/export.py` | Implement `to_geojson`, `to_csv`, `to_shapefile`, `to_change_report`. All thin wrappers around GeoPandas. | ~80 |

## Files to Create

| File | Purpose | Lines |
|---|---|---|
| `scripts/full_pipeline_demo.py` | End-to-end: patch → detect → health → NDVI filter → georef → GeoJSON | ~80 |
| `tests/test_georef.py` | Unit tests for georeference() with synthetic detections | ~60 |
| `tests/test_temporal.py` | Unit tests for compare_periods() with synthetic GeoDataFrames | ~70 |
| `tests/test_export.py` | Unit tests for all export functions with tempfiles | ~60 |

## Module Contracts

### georef.py

```python
def georeference(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],  # EPSG:25831
    image_size_px: tuple[int, int],
    health_scores: list[HealthScore] | None = None,
    crs: str = "EPSG:25831",
) -> gpd.GeoDataFrame
```

**Output schema (always present):**
- `tree_id`: int (0-indexed, matches detection order)
- `geometry`: shapely Point (center of bbox) in the given CRS
- `confidence`: float — detection confidence from the model
- `bbox_xmin`, `bbox_ymin`, `bbox_xmax`, `bbox_ymax`: float (meters, in CRS)
- `crown_width_m`, `crown_height_m`: float — bbox dimensions in meters

**Output schema (if health_scores provided):**
- `health_label`: str — "healthy" | "stressed" | "dead" | "unknown"
- `grvi`: float — Green-Red Vegetation Index
- `exg`: float — Excess Green Index
- `health_confidence`: float

**Behavior for empty detections:** return empty GeoDataFrame with the same
columns (no crash).

**Strips rfdetr metadata** before extracting arrays (same trick as ndvi.py).

### temporal.py — compare_periods

Algorithm:
1. Strip any helper columns that might conflict with sjoin output
2. Use `gpd.sjoin_nearest(gdf_before, gdf_after, max_distance=tolerance)`
   with `distance_col="_dist_m"` — one row per matched t₀ tree
3. For each match row: build `TreeMatch(tree_id_before, tree_id_after,
   distance_m, crown_area_change, health_before, health_after)`
4. `missing` = t₀ tree_ids that appear in left-join with NaN right index
5. `new` = t₁ tree_ids not present in any match
6. Return ChangeReport with `date_before="unknown"`, `date_after="unknown"`
   by default — caller fills in real dates if needed (avoids adding params)

**Crown area change:** compute from bbox dimensions: before_area = w*h,
after_area = w'*h', change = (after - before) / before

**Health transition:** take `health_label` from each side (default
"unknown" if column missing)

### export.py

- `to_geojson(gdf, path)`: `gdf.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")`
- `to_csv(gdf, path)`: reproject to WGS84, extract `lon`/`lat` from geometry,
  drop geometry column, write CSV
- `to_shapefile(gdf, path)`: `gdf.to_file(path, driver="ESRI Shapefile")`
  (keeps original CRS, since Shapefile handles any CRS)
- `to_change_report(change, path)`: `json.dump()` with custom encoder for
  dataclasses (using `dataclasses.asdict`)

All functions create parent dirs if missing. All return the written Path.

## Tests

| Done Criterion # | Test | File | Type |
|---|---|---|---|
| 1 | `test_georef_basic` — 5 dets, verify 5 rows + Points | test_georef.py | unit |
| 1 | `test_georef_empty` — empty dets, empty gdf returned | test_georef.py | unit |
| 1 | `test_georef_point_is_bbox_center` — verify geom at bbox center in meters | test_georef.py | unit |
| 2 | `test_georef_with_health` — verify health columns present | test_georef.py | unit |
| 3 | `test_compare_periods_exact_match` — all trees in same place | test_temporal.py | unit |
| 3 | `test_compare_periods_missing_and_new` — partial overlap | test_temporal.py | unit |
| 3 | `test_compare_periods_tolerance` — match respects max_distance | test_temporal.py | unit |
| 4 | `test_to_geojson_valid_and_wgs84` — output file has CRS 4326 | test_export.py | unit |
| 4 | `test_to_geojson_feature_count` — same number of features | test_export.py | unit |
| 5 | `test_to_csv_lat_lon_columns` — csv has lon, lat cols | test_export.py | unit |
| 5 | `test_to_shapefile` — file + .shp/.shx/.dbf created | test_export.py | unit |

## Approach Alternatives

| Alternative | Rejected Because |
|---|---|
| Read bounds from image EXIF (original stub plan) | Our patches are JPEG with no geo metadata — the data comes from `patches_metadata.csv`. Explicit bounds is simpler and matches filter module pattern. |
| Output Polygons (bbox corners) instead of Points | Points are standard for forestry inventories. Polygons are easy to derive later from stored bbox fields. |
| Custom KD-tree for temporal matching | GeoPandas' `sjoin_nearest` uses a built-in spatial index (STRtree), same performance, one line of code. |
| HTML report with folium map | Adds folium dep, significant code. GeoJSON + QGIS covers the use case. |
| Tree-id across time periods | Hard problem (needs persistent identity). Use spatial matching as proxy. |
| Single-module approach (merge georef+export) | Violates SRP. Separate concerns: one makes gdfs, one writes them. |

## Risks and Side Effects

- **Schema drift:** if temporal.py/export.py expect columns that georef.py
  doesn't emit, we get KeyErrors. Mitigation: define the schema once in
  the georef docstring, assert invariants in tests.
- **CRS confusion:** GeoJSON requires WGS84, GeoPandas defaults to writing
  in the gdf's CRS. We explicitly `.to_crs("EPSG:4326")` before writing
  GeoJSON. CSV also reprojects.
- **Empty GeoDataFrames are weird in GeoPandas:** creating one with the
  right schema requires a bit of ceremony (can't just pass empty dict).
  Mitigation: test the empty case explicitly.
- **sjoin_nearest behavior:** returns one row per left record (joined or
  not). Unmatched rows have NaN on right side. We filter these out to
  build the matched list and capture them as "missing".

## Estimated Scope

- 3 modified files (~290 lines total replacing stubs)
- 4 new files (~270 lines)
- 0 new dependencies
- Complexity: medium — lots of small pieces but each is straightforward
  GeoPandas
