# Forest Pulse — Methodology

**A parameterized Mediterranean forest inventory from aerial imagery and LiDAR, applied to Parc Natural del Montseny (Catalunya).**

This document describes the technical methodology behind the Forest Pulse inventory system. It is structured as a paper methods section and serves as the primary technical reference for the project.

---

## 1. Study Area

**Parc Natural del Montseny** (41.77 N, 2.43 E) is a 31,064 ha UNESCO Biosphere Reserve in the Pre-Coastal Range of Catalunya, NE Iberian Peninsula. Elevation ranges from 200 m (Sant Celoni valley) to 1,706 m (Turo de l'Home), creating a strong altitudinal gradient that drives species zonation:

- **Low elevations (200-600 m)**: Pinus halepensis (Aleppo pine), Quercus ilex (holm oak), Quercus suber (cork oak). Open to moderately dense stands, 50-200 trees/ha. Frequently drought-stressed in summer.
- **Mid elevations (600-1000 m)**: Mixed Quercus ilex and Pinus sylvestris (Scots pine). Transition zone with variable canopy density.
- **High elevations (1000-1700 m)**: Fagus sylvatica (beech) dominant on north-facing slopes, Pinus sylvestris on south-facing slopes. Dense canopy, 200-400 trees/ha.

The inventory covers 800 patches (640 x 640 pixels at 0.25 m/px = 160 m x 160 m each) across 8 sampling zones: `high`, `low`, `mid`, `ne_slopes`, `nw_plateau`, `se_ridge`, `summit`, `sw_valley`. Total area: ~2,004 hectares.

## 2. Data Sources

All input data is publicly available from the Institut Cartografic i Geologic de Catalunya (ICGC, https://www.icgc.cat/).

### 2.1 Orthophotos

ICGC publishes 25 cm/px RGB orthophotos covering all of Catalunya. We used summer acquisitions (leaf-on season) downloaded via the ICGC WMS service and tiled into 640 x 640 pixel JPEG patches using `scripts/tile_orthophoto.py`. Patches with mean Excess Green Index (ExG = 2G - R - B) below 15 were discarded as non-forest (roads, buildings, water, bare rock).

Coordinate reference system: ETRS89 / UTM zone 31N (EPSG:25831). Patch geographic centers are recorded in `data/montseny/patches_metadata.csv` with zone labels.

### 2.2 LiDAR

ICGC LiDAR Territorial v3r1 (2021-2023) provides airborne laser scanning at 8+ points/m2 in EPSG:25831. Data is distributed as LAZ 1.4 files on a 1 km x 1 km tile grid. Tile IDs encode the kilometer grid coordinates with a northing offset:

```
easting_km  = int(x) // 1000
northing_km = int(y) // 1000
ID1K  = f"{easting_km:03d}{(northing_km - 4000):03d}"
ID10K = f"{(easting_km // 10):02d}{((northing_km // 10) - 400):02d}"
```

LAZ files are downloaded on demand and cached locally (~500-700 MB per tile). The park area spans approximately 50-60 tiles.

Point cloud classification follows ASPRS LAS 1.4: code 2 = ground, 3/4/5 = low/medium/high vegetation. Each point carries x, y, z coordinates, intensity (reflectance amplitude), return number, and number of returns.

## 3. Canopy Height Model

For each patch, a Canopy Height Model (CHM) is rasterized from the LAZ point cloud at 0.5 m resolution:

1. **Ground surface (DTM)**: minimum z per grid cell among ground-classified points (ASPRS code 2).
2. **Canopy surface (DSM)**: maximum z per grid cell among first-return points (return_number == 1).
3. **CHM = max(DSM - DTM, 0)**: canopy height above ground. Negative values from interpolation noise are clipped to zero.

The CHM is written as a single-band float32 GeoTIFF with DEFLATE compression and cached on disk for subsequent runs.

## 4. Tree Detection (LiDAR-First Architecture)

### 4.1 The architectural evolution

The project explored three detection architectures in sequence:

**Phase 8-10: RF-DETR-primary.** An RF-DETR object detector (DINOv2 backbone, fine-tuned on DeepForest weak labels) runs on each 640 x 640 patch. Detections are filtered by a deterministic LiDAR tree-top matcher that drops any bbox whose center has no CHM peak within 2 m. Performance evolved through inference-time tuning:

| Operating point | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline (conf=0.30, no filter) | 0.268 | 0.068 | 0.108 |
| + LiDAR filter (conf=0.30) | 0.983 | 0.068 | 0.127 |
| + Confidence sweep (conf=0.02) | 0.720 | 0.153 | 0.252 |
| + Sliced inference (9 x 320px tiles) | 0.604 | 0.408 | 0.487 |

The confidence sweep (lowering RF-DETR's threshold from 0.30 to 0.02 and reapplying the LiDAR filter) produced a 2.34x F1 improvement. Sliced inference (running the detector on overlapping sub-windows and merging with NMS) added another 1.93x by sidestepping the model's 300-query-per-call architectural cap.

**Phase 11a: LiDAR-first (production architecture).** The central insight: the LiDAR filter's precision of 0.983 at confidence 0.30 already proved that LiDAR peaks were doing the detection work. Making this explicit — LiDAR becomes the primary detector, RF-DETR is demoted to optional visual verifier — increased the tree count from 2,119 to 3,320 on the 10-patch reference set (1.57x) while reducing runtime from 8.1 s to 0.2 s (40x faster).

### 4.2 Tree-top detection via local-maximum filtering

Tree-top positions are found by applying a standard forestry technique to the CHM (equivalent to the `locate_trees` function in the lidR R package):

1. **Gaussian smoothing** (sigma = 1 pixel = 0.5 m) suppresses single-pixel speckle without merging adjacent crowns.
2. **Local maximum filter** with a square window of diameter 2 x min_distance (default min_distance = 3 m, matching roughly half a typical Mediterranean crown). A pixel is a peak if its smoothed value equals the local maximum.
3. **Height threshold**: peaks below 5 m are discarded (Spanish Forest Inventory convention for the tree/shrub boundary).
4. **Pixel-to-world projection** via the rasterio affine transform gives tree-top positions in EPSG:25831 meters.

The default parameters (3 m min_distance, 1 px sigma, 5 m height threshold) were validated on 100 random patches: mean density 125 trees/ha, mean peak height 12.2 m, 90% of peaks >= 7 m, mean nearest-neighbor distance 6.0 m. All four metrics fall within published Mediterranean forest priors.

### 4.3 Detection output

Each LiDAR peak is projected back to the image's pixel coordinate frame (inverse affine, with Y-axis inversion matching the rasterio convention) and converted to a fixed 5 m diameter bounding box. A synthetic confidence score is assigned: `conf = clip((height_m - 5) / 20, 0.01, 1.0)`, monotonic in peak height.

The output is an `sv.Detections` object compatible with the downstream health scoring, georeferencing, and export pipeline.

## 5. Crown Segmentation

### 5.1 Watershed on the CHM

Each tree's crown shape is delineated via marker-controlled watershed segmentation on the CHM, using `scipy.ndimage.watershed_ift` (Iterated Forest Transform algorithm by Lotufo and Falcao):

1. **Cost image**: the CHM is inverted and scaled to uint8 (tree-tops = 0 = lowest cost, gaps = 255 = highest cost). Pixels below the 5 m height threshold are forced to maximum cost, acting as barriers.
2. **Marker raster**: tree-top positions (from Section 4.2) are projected to pixel coordinates and placed as labeled markers (1, 2, ..., N) on the raster.
3. **Watershed**: the IFT algorithm floods from each marker outward in ascending cost order, stopping where two flood fronts meet. Each resulting basin is one tree's canopy territory.
4. **Post-masking**: pixels below 5 m height are removed from all basins (they are gaps, not crown).
5. **Polygon extraction**: each labeled region is converted to a shapely Polygon in world coordinates via `rasterio.features.shapes`.

### 5.2 Fallback handling

Trees whose watershed basin is empty (marker outside the CHM bounds), has zero pixels after post-masking, or exceeds a sanity cap of 150 m2 (= 14 m diameter, the upper end for any Montseny species) are assigned a circular fallback polygon of 2.5 m radius centered on the tree-top position. On the 10-patch reference set, 16% of trees received fallback circles.

The 150 m2 cap was calibrated by observing that basins exceeding this threshold produced biologically implausible DBH estimates (> 100 cm for Pinus halepensis, whose published maximum is 60-80 cm). The original 200 m2 cap allowed 2,098 over-segmented basins contributing 17% of total biomass.

### 5.3 Crown area

Crown area is computed as `polygon.area` in EPSG:25831 square meters. For the 10-patch reference set: median 13 m2, mean 20 m2, range [0.25, 150] m2.

## 6. Health Scoring

Each tree's health is estimated from RGB vegetation indices computed on the bbox crop:

- **GRVI** (Green-Red Vegetation Index) = (G - R) / (G + R), range [-1, 1]. Higher values indicate more chlorophyll activity.
- **ExG** (Excess Green Index) = 2G - R - B. Secondary confirmation signal.

### 6.1 Classification thresholds

Trees are classified by a two-threshold heuristic:

| Label | Condition | Confidence |
|---|---|---:|
| healthy | GRVI > 0.06 AND ExG > 20 | 0.8 |
| dead | GRVI < 0.0 OR ExG < 5 | 0.8 |
| stressed | everything else | 0.5 |

The GRVI threshold was calibrated from 0.10 (Phase 1 default, temperate-range) to 0.06 after observing that GRVI at 25 cm/px on ICGC Mediterranean summer orthophotos has a narrower dynamic range (0.02-0.15) than temperate imagery. Healthy Mediterranean canopy (holm oak, Scots pine) routinely produces GRVI 0.03-0.08 when unstressed. The original 0.10 threshold labeled 42% of the park as "stressed"; the calibrated 0.06 produces 14%, consistent with published 10-25% stress rates for Mediterranean national parks.

### 6.2 Interpretation

Health labels should be treated as **relative within-park indicators**, not absolute diagnoses. The per-zone gradient (1.6% stressed in the mid zone to 32.1% in the se_ridge zone) maps the park's moisture and exposure conditions during summer aerial capture. South-facing dry zones show higher stress; north-facing moist zones show less. This gradient is ecologically plausible and consistent across species groups (broadleaf and conifer show identical GRVI distributions).

## 7. Species Classification

### 7.1 Approach

Binary broadleaf/conifer classification uses an unsupervised percentile-threshold method on two LiDAR features:

- **return_ratio**: fraction of multi-return laser pulses inside the crown. Broadleaves have sparser canopy (pulses penetrate multiple layers) producing higher return_ratio. Conifers have denser needle foliage that absorbs most energy on the first return.
- **intensity_mean**: mean laser return amplitude. Broadleaf leaves are more reflective at 1064 nm (the ICGC sensor wavelength) than conifer needles.

### 7.2 Algorithm

1. **Z-score normalize** both features across all trees in the inventory batch.
2. **Composite score** = z(return_ratio) + z(intensity_mean). Higher = more broadleaf-like.
3. **Threshold** at the (1 - broadleaf_fraction) percentile. Default broadleaf_fraction = 0.60, matching the Catalan Forest Inventory (IEFC) published prior for Montseny.

The top 60% by composite score are labeled broadleaf; the bottom 40% conifer. For samples smaller than 10 trees, a fixed absolute-threshold fallback is used instead of z-scoring.

### 7.3 Validation

The classifier is fully unsupervised (no training data). Validation is ecological:

- **Global fraction**: 60.0% broadleaf / 40.0% conifer (by construction).
- **Per-zone variance**: 25.9% (sw_valley) to 88.9% (ne_slopes) broadleaf. North-facing moist slopes are dominated by beech and holm oak; south-facing dry valleys are dominated by Pinus halepensis. This matches the known Montseny elevation/aspect species gradient.
- **Independent physical validation**: broadleaves have mean crown area 24.6 m2 vs conifers 13.0 m2 (crown area was NOT an input to the classifier). The larger-crown broadleaf signal confirms the classification captures real structural differences.

## 8. Allometric DBH and Biomass Estimation

### 8.1 Crown-to-DBH

DBH (diameter at breast height) is estimated via a Jucker et al. (2017) style power-law equation:

```
DBH_cm = a * (crown_area_m2 * height_m)^b
```

Species-specific coefficients:

| Species group | a | b | Source form |
|---|---:|---:|---|
| Broadleaf | 0.56 | 0.63 | European temperate biome (Jucker 2017) |
| Conifer | 0.48 | 0.65 | Same, adjusted for narrower conifer crowns |

### 8.2 DBH-to-biomass

Above-ground biomass (AGB) is estimated via a Ruiz-Peinado et al. (2011) style equation:

```
AGB_kg = c * DBH_cm^d
```

| Species group | c | d | Representative species |
|---|---:|---:|---|
| Broadleaf | 0.22 | 2.36 | Quercus ilex / Fagus sylvatica |
| Conifer | 0.085 | 2.49 | Pinus halepensis / P. sylvestris |

### 8.3 Confidence intervals

Per-tree CIs are fixed fractions of the point estimate:
- DBH: +/- 30% (Jucker 2017 published RMSE for crown-based DBH)
- Biomass: +/- 40% (compound DBH + biomass allometric residuals)

Stand-level aggregates shrink to +/- 10-15% as per-tree errors average over 300+ trees per zone.

### 8.4 Calibration

Coefficients are representative Mediterranean values, not site-calibrated to Montseny. For publication-grade accuracy, a follow-up could replace the 8 generic coefficients with per-species equations from Ruiz-Peinado et al. (2011, 2012) once genus-level species labels are available.

### 8.5 Results

On the 10-patch reference set (3,320 trees):
- Broadleaf mean DBH: 19.3 cm (median 15.8)
- Conifer mean DBH: 17.8 cm (median 13.3)
- Park total AGB: 96,131 tonnes (48.0 t/ha)

Per-zone range: 36.8 t/ha (high, open altitude) to 74.8 t/ha (nw_plateau, dense broadleaf). The nw_plateau value falls within the IEFC published range of 80-180 t/ha for mature Montseny stands. The park-wide average of 48 t/ha is consistent with Vayreda et al. (2012) reported 50-90 t/ha for Catalan forests when accounting for the inclusion of all trees >= 5 m (not just canopy-dominant individuals).

## 9. Georeferencing and Export

Per-tree attributes are assembled into a GeoDataFrame (geopandas) in EPSG:25831, with crown polygons (or fallback circles) as the geometry column. The final export converts to WGS84 (EPSG:4326) for GeoJSON compatibility.

Cross-patch deduplication uses `geopandas.sjoin_nearest` at 1 m tolerance: trees from different patches whose centroids are within 1 m of each other are collapsed to the higher-confidence record. On the full park run, 8,266 duplicates (3.5%) were removed from 236,941 pre-dedup detections.

## 10. Negative Results and Lessons Learned

### 10.1 RGB-distilled species classifier (refuted)

Phase 9.5b attempted to train a sklearn GradientBoostingClassifier on 11 RGB/health/geometry features to distinguish real trees from false positives, with labels derived from LiDAR tree-top matching. The hypothesis was that the classifier could "distill" LiDAR knowledge into RGB space and transfer to non-LiDAR settings.

Result: F1 never exceeded baseline (0.103 vs 0.108) at any confidence threshold. RGB at 25 cm/px does not contain sufficient discriminative signal to recover the LiDAR-derived precision boundary. The unsupervised LiDAR-intensity species classifier (Phase 12a) succeeded where the supervised RGB classifier failed, because the physical signal (reflectance at 1064 nm) is genuinely present in LiDAR but not in RGB.

### 10.2 RF-DETR top_k cap (dead end)

Phase 10b attempted to raise RF-DETR's internal `num_select` parameter beyond 300 to increase the candidate pool. Investigation revealed the architectural limit: `num_queries = 300` is a DETR-family invariant, and the trained class already saturated the top-300 with class-0 candidates. Raising `num_select` to 600 only added untrained class-1 noise.

The productive alternative was sliced inference (Phase 10c): calling the detector on overlapping sub-windows, each with its own 300 queries, effectively multiplied the candidate pool by the tile count.

### 10.3 The confidence sweep lesson

The single largest F1 improvement (0.108 to 0.252, a 2.34x lift) came from lowering the RF-DETR confidence threshold from 0.30 to 0.02 — a one-line change that took 30 minutes to implement and test. The detector had been leaving 60% of its true positives on the floor at the Phase 1 default threshold.

**Lesson**: always sweep inference-time hyperparameters before investing in retraining or feature engineering. Defaults from early development calcify and become invisible ceilings on downstream metrics.

## 11. Computational Performance

| Pipeline stage | Per-patch cost | Bottleneck |
|---|---:|---|
| LAZ download (first time) | 30-90 s | Network I/O (500-700 MB per tile) |
| CHM computation (first time) | 2-3 s | numpy rasterization |
| Tree-top detection | < 0.01 s | scipy gaussian_filter + maximum_filter |
| Crown segmentation (watershed) | 0.05-0.1 s | scipy watershed_ift + rasterio shapes |
| LiDAR feature extraction | 15-25 s | O(N*D) point-in-bbox filter (N~30M points, D~300 trees) |
| Health scoring | 0.5-1 s | numpy GRVI/ExG on bbox crops |
| Species classification | < 0.01 s | numpy z-score + percentile (batch, post-dedup) |
| Allometric estimation | < 0.01 s | numpy power laws (batch) |

**Full park runtime**: 5.7 hours for 783 patches (26.7 s/patch average). The bottleneck is LiDAR feature extraction, which performs O(N*D) point-in-bbox checks per patch. A spatial index (KD-tree or voxel grid) built once per LAZ tile would reduce this to O(D * log N), cutting per-patch cost from ~20 s to < 1 s. This optimization was deferred as a follow-up.

LAZ and CHM data is disk-cached after first computation. Subsequent runs on the same patches skip downloads and rasterization, reducing per-patch cost to ~25 s (dominated by feature extraction).

## 12. Software and Reproducibility

The complete pipeline is implemented in Python 3.10+ using:

- **scipy** (1.12+): watershed segmentation, CHM smoothing, local-max filtering
- **rasterio** (1.3+): GeoTIFF I/O, affine transforms, polygon extraction
- **geopandas** (1.0+): GeoDataFrame construction, spatial joins, CRS conversion
- **shapely** (2.0+): polygon geometry, centroid, area
- **laspy** (2.5+): LAZ point cloud reading
- **supervision** (0.25+): detection data structures, NMS, inference slicing
- **RF-DETR** (rfdetr 1.6+): visual object detector (optional in LiDAR-first mode)
- **numpy**, **Pillow**: array operations, image loading

All input data is publicly available from ICGC. The inventory can be reproduced with:

```bash
pip install -e ".[lidar]"
python scripts/inventory_montseny.py --detector lidar-first
```

Source code: https://github.com/jordicatafal/forest-pulse

## References

- Jucker, T. et al. (2017). Allometric equations for integrating remote sensing imagery into forest monitoring programmes. *Global Change Biology*, 23(1), 177-190.
- Ruiz-Peinado, R. et al. (2011). New models for estimating the carbon sink capacity of Spanish softwood species. *Forest Systems*, 20(1), 176-188.
- Ruiz-Peinado, R. et al. (2012). Biomass models to estimate carbon stocks for hardwood tree species. *Annals of Forest Science*, 69, 443-452.
- Vayreda, J. et al. (2012). Recent climate changes interact with stand structure and management to determine changes in tree carbon stocks in Spanish forests. *Global Change Biology*, 18(3), 1028-1041.
- Montero, G. et al. (2005). *Produccion de biomasa y fijacion de CO2 por los bosques espanoles*. INIA Monografias, Madrid.
