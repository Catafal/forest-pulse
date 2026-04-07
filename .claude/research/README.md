# Research: Tree vs Bush Discrimination

**Context:** Our self-trained RF-DETR model reaches 0.904 mAP50 on its own filtered labels, but visual + density sanity checks show it's detecting "round green blobs from above" — which includes bushes, shrubs, and merged crowns. The metric is inflated by self-training feedback bias.

**Goal:** Physically verify that a detected crown is a real tree (>5m tall, woody) vs a shrub/bush.

## Reports in this folder

| File | Topic | Status |
|---|---|---|
| `icgc_lidar.md` | ICGC LiDAR Territorial v3r1 for canopy height filtering | Recommended path |
| `ndvi_infrared.md` | NDVI from ICGC infrared orthophoto for vegetation filtering | Secondary use only |

## Key findings summary

### LiDAR — the answer to tree vs bush
- ICGC publishes LiDAR Territorial v3r1 (2021-2023), 8+ pts/m², LAZ 1.4 format
- Also provides **DSM and DTM rasters at 2m/5m/15m** via WCS — no point cloud processing needed
- **CHM = DSM − DTM** gives canopy height per pixel
- At 5m threshold: trees > 5m, shrubs/bushes < 5m (Iberian shrub-LiDAR research confirms)
- WCS endpoint: `https://geoserveis.icgc.cat/icgc_mdt2m/wms/service`
- Montseny full coverage at 2m CHM: few hundred MB (not 270GB — that's LAZ)
- Pipeline: RF-DETR box → sample CHM at box center → keep if height > 5m

### NDVI — NOT the answer
- ICGC infrared is 3-band CIR (NIR, Red, Green) — NDVI computable from single tile
- Same WMS endpoint, same grid, pixel-perfect alignment with RGB we already have
- **Critical finding: NDVI cannot distinguish trees from shrubs.** Multiple papers confirm this.
  - Both are healthy chlorophyll-rich vegetation
  - NDVI ranges overlap in the 0.3-0.6 zone
- What NDVI IS useful for:
  - Filtering non-vegetation false positives (rocks, roads, roofs): `ndvi < 0.15` → drop
  - Adding to health scoring (NDVI degrades before visible greenness)

## Decision

**Go with LiDAR for tree/shrub discrimination. Use NDVI only as a false-positive filter and health feature.**

## Proposed pipeline

```
Input image (RGB 25cm)
    → RF-DETR: detect crown boxes
        → NDVI filter: drop boxes where ndvi_mean < 0.15 (removes rocks/roads/roofs)
            → LiDAR height filter: keep boxes where chm_max > 5m (removes shrubs)
                → Health scoring: GRVI + ExG + ndvi_mean
                    → Final verified tree inventory
```

Trees surviving this pipeline are physically verified: vegetation (NDVI) + tall (LiDAR) + detected as crown (RF-DETR). Industry-standard approach used by Spanish Forest Inventory.

## Implementation plan (from research reports)

New modules (both reports include code):
1. `src/forest_pulse/lidar.py` — fetch_chm_for_aoi() via WCS + filter_detections_by_height()
2. `src/forest_pulse/ndvi.py` — compute_ndvi() + filter_detections_by_ndvi()

New dependencies:
- `rioxarray` (recommended for LiDAR, optional for NDVI)

Slot into pipeline between georef.py and health.py.
