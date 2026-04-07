# Codebase Observation: core-pipeline-modules

**Date:** 2026-04-07
**GitNexus available:** no

## Relevant Files

### Existing stubs (to implement)
- `src/forest_pulse/georef.py` — stub with `georeference(detections, image_path, health_scores, crs)`. Current signature reads from image path (EXIF/GeoTIFF). **Deviation needed:** our patches are JPEG (no geo metadata), so we'll take `image_bounds` + `image_size_px` explicitly, matching ndvi.py/lidar.py pattern.
- `src/forest_pulse/temporal.py` — stub with full dataclasses already defined. `TreeMatch` has tree_id_before/after, distance_m, crown_area_change, health_before/after. `ChangeReport` has all expected fields + computed properties. `compare_periods(gdf_before, gdf_after, match_tolerance_m=2.0)` — already in place, just needs implementation. **No date_before/date_after in signature** (spec wants them) — the stub's ChangeReport has them as regular fields. I should keep the signature as-is and let the caller set dates on the returned ChangeReport (simpler than adding params).
- `src/forest_pulse/export.py` — stubs for `to_geojson`, `to_shapefile`, `to_csv`, `to_report`. All take gdf + output_path, return Path.

### Integration points
- `src/forest_pulse/health.py` — `HealthScore` dataclass has `tree_id, grvi, exg, label, confidence`. These are the per-tree fields we attach to the GeoDataFrame.
- `src/forest_pulse/detect.py` — produces `sv.Detections` with xyxy + confidence. rfdetr adds `source_shape`/`source_image` metadata that breaks boolean indexing — we need to strip these before any pandas operations.
- `src/forest_pulse/ndvi.py` / `lidar.py` — have `_pixel_bbox_to_geo()` helpers. Same pattern applies: bounds (EPSG:25831) + pixel size → geographic bbox. We can inline this logic in georef.py rather than importing (keeps modules decoupled).
- `data/montseny/patches_metadata.csv` — has `x_center`, `y_center` per patch filename. Useful for demo: lookup patch bounds from CSV.

### Existing demo for reference
- `scripts/apply_filters_demo.py` — template for the new full_pipeline_demo.py. Shows the pattern: resolve patch bounds from metadata, run detect, save annotated images.

## Key Patterns

- **Pure functions:** every public API function is `(inputs, options) → output`, no classes.
- **Type hints on public APIs:** Google-style docstrings.
- **Strip rfdetr metadata:** `_strip_rfdetr_metadata(dets)` helper (duplicated in ndvi.py and lidar.py). For georef.py we need the same trick before extracting arrays for pandas.
- **Coordinate conversion:** image Y inverts vs world Y (image y=0 is top, world y_max is top). Same transform as ndvi/lidar filter modules.
- **Default CRS:** EPSG:25831 throughout. Reproject to EPSG:4326 only at export time for GeoJSON.

## GeoPandas Capabilities (standard library)

- `gpd.GeoDataFrame(data={...}, geometry=Point_list, crs="EPSG:25831")` — create
- `gdf.to_crs("EPSG:4326")` — reproject
- `gdf.to_file(path, driver="GeoJSON")` — write GeoJSON
- `gdf.to_file(path, driver="ESRI Shapefile")` — write Shapefile
- `gdf.to_csv(path)` — write CSV (use shapely `.x`, `.y` to get lon/lat)
- `gpd.sjoin_nearest(left, right, max_distance=X, distance_col="dist")` — spatial matching (GeoPandas 0.10+, we have 1.1.3)

## Architecture Constraints

- Max 200 lines per function, 1000 per file
- All imports at top
- Type hints + Google docstrings on public functions
- Comments explain WHY not WHAT
- Dependencies: GeoPandas already installed (1.1.3 per pyproject.toml)

## Conditional Skill Routing

- [ ] /plan-eng-review — not applicable (3 small modules, well-defined contracts)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable (standard unit tests)
