# Forest Pulse

**Parameterized Mediterranean forest inventory from aerial imagery + LiDAR.**

Detects individual trees, segments crown shapes, classifies species groups, and estimates DBH + above-ground biomass across Parc Natural del Montseny (Catalunya) — entirely from publicly available ICGC data.

## What it produces

A single GeoJSON with **228,675 trees**, each carrying 28 attributes:

```
Per tree:
  geometry              Crown polygon (Polygon, EPSG:4326)
  crown_area_m2         Real area from watershed segmentation
  lidar_height_p95      Canopy top height from LiDAR CHM (meters)
  species_group         broadleaf | conifer (unsupervised classification)
  dbh_cm_estimate       Estimated DBH ± 30% CI (allometric)
  biomass_kg_estimate   Above-ground biomass ± 40% CI (allometric)
  health_label          healthy | stressed | dead | unknown (GRVI + ExG)
  + 21 supporting attributes (bbox, health indices, LiDAR features, etc.)
```

**Park-wide totals (calibrated):**

| Metric | Value |
|---|---:|
| Trees detected | 228,675 |
| Patches processed | 783 / 800 |
| Above-ground biomass | 96,131 tonnes |
| Average biomass density | 48.0 t/ha |
| Broadleaf fraction | 60% |
| Health: healthy / stressed / dead | 82.5% / 14.3% / 1.5% |

## Quick start

```bash
# Install
pip install -e ".[lidar]"

# Run inventory on 2 patches (smoke test, ~60s)
python scripts/inventory_montseny.py --detector lidar-first \
    --patch 0043.jpg --patch 0250.jpg

# Run full park inventory (~6 hours, needs ~30 GB disk for LAZ cache)
python scripts/inventory_montseny.py --detector lidar-first

# Output
open outputs/inventory/montseny_trees.geojson  # load in QGIS
```

## Architecture

```
ICGC LiDAR (LAZ tiles, 8+ pts/m²)
  |
  +-- compute_chm_from_laz()           -- CHM raster (DSM - DTM)
  |    |
  |    +-- find_tree_tops_from_chm()   -- tree-top world positions + heights
  |    |    |
  |    |    +-- detect_trees_from_lidar()  -- sv.Detections (one bbox per peak)
  |    |         |
  |    |         +-- segment_crowns_watershed()  -- per-tree crown polygons
  |    |         |
  |    |         +-- extract_lidar_features()   -- 7 per-tree LiDAR stats
  |    |
  |    +-- [optional] lidar_tree_top_filter()   -- precision filter for RF-DETR
  |
ICGC Orthophoto (25 cm/px RGB JPEG patches)
  |
  +-- score_health()                   -- GRVI + ExG -> health labels
  |
  +-- [fallback] detect_trees_sliced() -- RF-DETR for non-LiDAR regions

                    |
                    v

  georeference()        -- pixel -> world coords, GeoDataFrame
  classify_broadleaf_conifer()  -- species from LiDAR intensity + return_ratio
  estimate_tree_metrics_batch() -- DBH + biomass from published allometrics
  to_geojson() / to_csv()      -- export
```

**Two detection modes:**

| Mode | When to use | Input | Speed |
|---|---|---|---|
| `--detector lidar-first` | Catalunya (ICGC LiDAR available) | LAZ + RGB | 26 s/patch |
| `--detector sliced` | Drone / non-LiDAR regions | RGB only | 0.5 s/patch |

## Data sources

All data is publicly available from the [Institut Cartografic i Geologic de Catalunya (ICGC)](https://www.icgc.cat/):

- **Orthophotos**: 25 cm/px RGB, tiled to 640x640 patches (160m x 160m)
- **LiDAR**: Territorial v3r1 (2021-2023), 8+ pts/m2, EPSG:25831, LAZ 1.4 format
- **Coordinate system**: ETRS89 / UTM zone 31N (EPSG:25831), exported as WGS84 (EPSG:4326)

## Per-zone results

| Zone | Trees | Broadleaf % | AGB (t/ha) | Stressed % |
|---|---:|---:|---:|---:|
| high | 28,498 | 53.6% | 36.8 | 4.8% |
| low | 29,358 | 74.9% | 41.1 | 20.6% |
| mid | 33,809 | 54.6% | 52.4 | 1.6% |
| ne_slopes | 32,248 | 88.9% | 48.4 | 10.3% |
| nw_plateau | 25,184 | 63.6% | 74.8 | 23.4% |
| se_ridge | 27,840 | 46.0% | 45.6 | 32.1% |
| summit | 28,283 | 63.3% | 38.7 | 6.2% |
| sw_valley | 23,455 | 25.9% | 45.0 | 20.4% |
| **TOTAL** | **228,675** | **60.0%** | **48.0** | **14.3%** |

## Modules

| Module | Purpose |
|---|---|
| `detect.py` | Tree detection (RF-DETR, sliced inference, LiDAR-first) |
| `lidar.py` | LAZ download, CHM rasterization, tree-top detection, LiDAR features |
| `crowns.py` | Watershed crown segmentation on CHM |
| `health.py` | GRVI + ExG vegetation indices -> health labels |
| `species.py` | Unsupervised broadleaf/conifer from LiDAR intensity + return ratio |
| `allometry.py` | DBH + biomass from published allometric equations |
| `georef.py` | Pixel -> world coords, GeoDataFrame construction |
| `export.py` | GeoJSON, CSV, Shapefile output |
| `patches.py` | Patch metadata helpers |
| `ndvi.py` | NDVI filter (non-vegetation false positives) |
| `segment.py` | SAM2 hybrid segmentation (optional) |
| `temporal.py` | Tree change detection across time periods |
| `visualize.py` | Annotated image output via Supervision |
| `device.py` | Cross-platform GPU/MPS/CPU detection |

## Honest limitations

- **Health labels are relative indicators**, not absolute diagnoses. The per-zone gradient (1.6% to 32% stressed) maps the park's moisture/exposure conditions during summer aerial capture. Absolute percentages depend on imagery date.
- **DBH is +/-30% per tree, biomass +/-40% per tree.** Stand-level aggregates are +/-10-15%. Coefficients are representative Mediterranean values, not site-calibrated.
- **Species is binary** (broadleaf/conifer), not genus-level. Sufficient for allometric DBH but not for per-species management.
- **Crown polygons** use watershed segmentation on a 0.5 m/px CHM. 16% of trees fall back to circular approximations where basins were empty or oversized.
- **Catalunya only** for LiDAR-first detector. Non-LiDAR regions fall back to RF-DETR (lower recall, F1 ~0.49).
- **17 of 800 patches failed** (2.1%) due to LAZ download errors.

## Key insights

1. **LiDAR-first architecture**: inverting the conventional detector+filter pipeline (LiDAR detects, RF-DETR verifies) gives 1.57x more trees in 1/40th the runtime.

2. **Confidence sweep**: lowering RF-DETR's threshold from 0.30 to 0.02 + LiDAR filter = 2.34x F1. A 30-minute experiment that beat months of feature engineering.

3. **Unsupervised species**: two z-scored LiDAR features, thresholded at a published broadleaf fraction, reproduce the known Montseny ecological gradient.

## Tech stack

- Python 3.10+
- RF-DETR (DINOv2 backbone) -- detection model (optional in LiDAR-first mode)
- scipy -- watershed segmentation, LiDAR CHM processing
- rasterio -- GeoTIFF I/O
- geopandas + shapely -- GIS output
- supervision -- detection utilities
- laspy -- LAZ point cloud reading

## License

MIT

## Citation

```
Catafal, J. (2026). Forest Pulse: Parameterized Mediterranean forest
inventory from aerial imagery and LiDAR. Parc Natural del Montseny,
Catalunya. https://github.com/jordicatafal/forest-pulse
```
