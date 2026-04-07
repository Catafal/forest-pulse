# Codebase Observation: lidar-ndvi-filtering

**Date:** 2026-04-07
**GitNexus available:** no

## Relevant Files

- `src/forest_pulse/detect.py` — returns `sv.Detections(xyxy, confidence)`. Boxes are in pixel coords of the input image, not geographic coords. Filters need image geo-bounds to convert.
- `src/forest_pulse/health.py` — takes `image: np.ndarray` + `detections: sv.Detections` + returns `list[HealthScore]`. Filters slot BEFORE this.
- `src/forest_pulse/georef.py` — STUB. Has `georeference()` function that should convert pixel bboxes to GPS. Can we reuse or should filters be separate?
- `scripts/download_montseny.py` — WMS client with `build_wms_url(x_min, y_min, x_max, y_max, width, height)`. Patches' geographic bounds are in `patches_metadata.csv` (x_center, y_center per patch).
- `data/montseny/patches_metadata.csv` — has `x_center, y_center` in EPSG:25831 for every patch. We can derive patch bounds from center + known size (640px × 0.25m = 160m).

## Key Contracts

- **sv.Detections.xyxy**: pixel coords in float32, shape (N, 4)
- **Coordinate system for all geographic data**: EPSG:25831
- **Patch size**: 640×640 px at 0.25m/px = 160m × 160m real-world
- **Patch origin**: x_min = x_center - 80, x_max = x_center + 80 (same for y)

## Existing WMS Endpoints (reusable)

```
WMS base: https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms
RGB layer: ortofoto_color_vigent (existing)
IR layer:  ortofoto_infraroig_vigent (new for NDVI)
IR band order: B1=NIR, B2=Red, B3=Green (CIR composite)

WMS elevation: https://geoserveis.icgc.cat/icgc_mdt2m/wms/service
DSM layer: (mdt2m surface model)
DTM layer: (mdt2m terrain model)
```

## Existing Patterns

- WMS download via `urllib.request.urlretrieve()` with built URL — simple, works
- Caching pattern from previous scripts: check if file exists before downloading
- Lazy module loading (imports inside functions) used in detect.py for heavy deps
- Type hints + Google docstrings on public functions

## Architecture Constraints

- Max 200 lines/function, 1000 lines/file
- All imports at top of file
- `autoresearch/eval.py` is LOCKED
- CHM + NDVI rasters too large to commit — cache to `data/montseny/rasters/` (gitignored)
- Filters must be pure functions: (detections, raster, threshold) → filtered_detections

## New Dependencies

- `rioxarray` — for LiDAR raster windowed reads (recommended, not required)
- `rasterio` — already installed, sufficient for NDVI

## Conditional Skill Routing

- [ ] /plan-eng-review — not applicable (small scope, 2 new files)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable
