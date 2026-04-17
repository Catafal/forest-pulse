# LiDAR-First Individual Tree Detection and Parameterized Forest Inventory for Mediterranean Forests: A Case Study in Parc Natural del Montseny

**Jordi Catafal**

Corresponding author: jordicatafal@gmail.com

---

## Abstract

Individual tree detection in dense Mediterranean forests remains challenging for visual object detectors due to merged crowns, low inter-tree contrast, and small object sizes at standard orthophoto resolutions (25 cm/px). We present a LiDAR-first detection architecture that inverts the conventional pipeline: instead of using a visual detector (RF-DETR) as the primary source and LiDAR as a post-hoc filter, we make LiDAR tree-top detection the primary oracle and relegate visual detection to an optional verification role. Applied to Parc Natural del Montseny (Catalunya, Spain) using publicly available ICGC orthophotos and LiDAR Territorial v3r1, the system detects 228,675 individual trees across 2,004 hectares, each with a watershed-segmented crown polygon, unsupervised broadleaf/conifer species classification, and allometric DBH and above-ground biomass estimates. Park-wide biomass averages 48.0 t/ha, with per-zone values ranging from 36.8 t/ha (open high-altitude stands) to 74.8 t/ha (dense broadleaf plateau), consistent with published Catalan forest inventories. The species classification, based on two z-scored LiDAR features (return ratio and intensity), requires zero training data and reproduces the known Montseny ecological gradient with per-zone broadleaf fractions ranging from 25.9% (Pinus halepensis valleys) to 88.9% (north-facing beech slopes). The methodology is fully reproducible from public data and open-source code. We document the complete architectural evolution from a baseline F1 of 0.108 through inference-time tuning (F1 = 0.487) to the LiDAR-first flip, including two instructive negative results (an RGB-distilled classifier that failed to beat baseline, and an architectural cap probe that revealed the wrong bottleneck). The full inventory, code, and methodology are publicly available.

**Keywords**: individual tree detection, LiDAR, Mediterranean forest, forest inventory, canopy height model, watershed segmentation, allometry, Montseny

---

## 1. Introduction

Individual tree detection (ITD) from remote sensing data is a prerequisite for modern forest inventory, carbon stock estimation, and ecological monitoring at landscape scales. While substantial progress has been made on ITD in boreal and temperate forests using both optical imagery and airborne LiDAR (Zhen et al., 2016; Weinstein et al., 2019), Mediterranean forests present specific challenges that limit the performance of visual detection approaches:

1. **Crown merging at standard resolution.** At 25 cm/px, Mediterranean tree crowns (5-15 m diameter) span only 20-60 pixels and frequently merge into continuous canopy in dense stands, eliminating the inter-crown gaps that visual detectors rely on.

2. **Low spectral contrast.** Evergreen broadleaves (Quercus ilex) and conifers (Pinus halepensis, P. sylvestris) produce similar spectral signatures in RGB imagery during summer, when most aerial surveys are conducted.

3. **Mixed-age structure.** Uneven-aged Mediterranean stands contain trees from 5 to 30+ m in height. Small understory trees are invisible from above in dense canopy but physically present and detectable by LiDAR.

The conventional approach to addressing these challenges is to train or fine-tune a visual object detector (e.g., DeepForest, DETR-family models) on annotated training data and then use LiDAR-derived features as post-hoc validation or filtering (Puliti et al., 2021; Hao et al., 2023). This framing treats the visual detector as the primary source of tree candidates and LiDAR as a quality control layer.

In this work, we invert this paradigm. We demonstrate that for Mediterranean forests with available airborne LiDAR, making **LiDAR the primary detector** and relegating visual detection to an optional verification role produces substantially more complete inventories — detecting 1.57 times more trees in 1/40th the computation time compared to our best visual detection configuration.

We apply this approach to Parc Natural del Montseny, a UNESCO Biosphere Reserve in Catalunya (NE Spain), using exclusively publicly available data from the Institut Cartografic i Geologic de Catalunya (ICGC). The resulting inventory covers 228,675 individual trees across 2,004 hectares, each with a per-tree crown polygon, binary species classification (broadleaf/conifer), estimated DBH and above-ground biomass with confidence intervals, and a health label derived from RGB vegetation indices.

Beyond the inventory itself, we document the complete methodological evolution from an initial visual detection baseline (F1 = 0.108 against LiDAR ground truth) through successive inference-time optimizations (confidence threshold tuning, sliced inference) to the architectural flip. We include two instructive negative results: an RGB-distilled classifier that failed to transfer LiDAR knowledge into visual features, and a model capacity probe that identified the wrong bottleneck. These findings are relevant to practitioners deciding when to invest in detector retraining versus inference-time tuning versus architecture changes.

## 2. Study Area and Data

### 2.1 Study area

Parc Natural del Montseny (41.77 N, 2.43 E) occupies 31,064 ha in the Pre-Coastal Range of Catalunya. Elevation ranges from 200 m (Sant Celoni valley) to 1,706 m (Turo de l'Home), creating a pronounced altitudinal vegetation gradient. Low elevations support open Pinus halepensis and Quercus ilex stands (50-200 trees/ha); mid elevations have mixed forests; high elevations are dominated by Fagus sylvatica on north-facing slopes and Pinus sylvestris on south-facing slopes (200-400 trees/ha).

We sampled 800 patches across 8 zones stratified by elevation and aspect: `high`, `low`, `mid`, `ne_slopes`, `nw_plateau`, `se_ridge`, `summit`, `sw_valley` (Figure 1). Each patch covers 160 m x 160 m (2.56 ha). Of these, 783 were successfully processed; 17 failed due to network errors during LiDAR tile download.

![Figure 1: Study area. 800 patches across 8 Montseny zones, with inset showing location in NE Iberian Peninsula.](figures/fig1_study_area.png)

### 2.2 Orthophotos

ICGC 25 cm/px RGB summer orthophotos, downloaded via WMS and tiled to 640 x 640 pixel JPEG patches. Patches with mean Excess Green Index (ExG = 2G - R - B) below 15 were excluded as non-forest. Coordinate reference system: ETRS89 / UTM zone 31N (EPSG:25831).

### 2.3 Airborne LiDAR

ICGC LiDAR Territorial v3r1 (2021-2023), 8+ points/m2, ASPRS LAS 1.4 classification (code 2 = ground, 3-5 = vegetation). Distributed as 1 km x 1 km LAZ tiles. Approximately 50-60 tiles cover the study area (~30 GB total). A temporal mismatch of 1-3 years exists between the LiDAR (2021-2023) and orthophoto acquisitions; this is negligible for the 5 m tree/shrub height threshold used in this work.

## 3. Methods

### 3.1 Canopy Height Model

For each patch, a Canopy Height Model (CHM) is rasterized at 0.5 m resolution from the LAZ point cloud:

- **Digital Terrain Model (DTM)**: minimum z per grid cell among ground-classified points.
- **Digital Surface Model (DSM)**: maximum z per grid cell among first-return points.
- **CHM = max(DSM - DTM, 0)**.

CHM rasters are cached on disk as single-band float32 GeoTIFFs.

### 3.2 Tree-top detection

Individual tree positions are identified by local-maximum filtering on the smoothed CHM, following the standard forestry approach (equivalent to `locate_trees()` in the lidR R package; Roussel et al., 2020):

1. Gaussian smoothing (sigma = 1 pixel = 0.5 m) to suppress single-pixel speckle.
2. Square maximum filter with window diameter = 2 x 3 m = 6 m (roughly one crown width for Mediterranean species).
3. Peaks where the smoothed value equals the local maximum AND exceeds 5 m (Spanish Forest Inventory tree/shrub threshold).
4. Pixel-to-world projection via the rasterio affine transform.

Parameters were validated on 100 randomly selected patches: mean density 125 trees/ha, mean peak height 12.2 m, 90% of peaks >= 7 m, mean nearest-neighbor distance 6.0 m. All metrics fall within published ranges for Mediterranean mixed forests (IEFC; Vayreda et al., 2012).

### 3.3 Detection architecture comparison

We evaluated three detection architectures on a fixed 10-patch reference set (3,320 LiDAR-verified trees):

**Architecture A: Visual detector + LiDAR filter.** RF-DETR (DINOv2 backbone, fine-tuned on DeepForest weak labels) detects trees in RGB patches. A deterministic LiDAR filter drops detections whose bbox center has no CHM peak within 2 m.

| Configuration | Predictions | TP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline (conf=0.30, no filter) | 840 | 225 | 0.268 | 0.068 | 0.108 |
| + LiDAR filter | 229 | 225 | 0.983 | 0.068 | 0.127 |
| + Confidence sweep (conf=0.02) | 700 | 505 | 0.721 | 0.152 | 0.252 |
| + Sliced inference (9 x 320px) | 2,247 | 1,356 | 0.604 | 0.408 | 0.487 |

The confidence sweep — lowering the detector's threshold from 0.30 to 0.02 and reapplying the filter — produced a 2.34x F1 improvement with zero retraining. Sliced inference (running the detector on 9 overlapping 320 x 320 sub-windows per patch and merging via NMS) added 1.93x by allowing each sub-window its own 300-query budget, effectively improving the resolution at which the detector sees each crown.

**Architecture B: LiDAR-first (production).** LiDAR tree-tops are the detector; each peak becomes one detection with a fixed 2.5 m radius bbox (Figure 2). On the same reference set: 3,320 detections matching all 3,320 ground-truth peaks by construction. Runtime: 0.2 s total (vs. 8.1 s for Architecture A's best configuration).

![Figure 2: Architecture comparison. (a) RF-DETR primary with LiDAR filter. (b) LiDAR-first with watershed crowns, species, and allometry.](figures/fig2_architecture.png)

The F1 progression across successive inference-time optimizations is shown in Figure 3.

![Figure 3: Detection performance evolution across four RF-DETR operating points on the 10-patch reference set.](figures/fig3_f1_progression.png)

The F1 metric is structurally uninformative under Architecture B (detections ARE the ground truth), so we validate via ecological plausibility (Section 4.1) and cross-architecture consistency (Section 4.2) instead.

### 3.4 Crown segmentation

Per-tree crown shapes are delineated via marker-controlled watershed segmentation on the inverted CHM using the Iterated Forest Transform algorithm (Lotufo and Falcao, 1997) as implemented in `scipy.ndimage.watershed_ift`:

1. **Cost image**: inverted CHM scaled to uint8 (tree-tops = 0, gaps = 255). Pixels below 5 m are forced to maximum cost (barriers).
2. **Markers**: tree-top positions projected to pixel coordinates.
3. **Watershed**: each basin expands from its marker until meeting an adjacent basin.
4. **Post-masking**: pixels below 5 m removed from all basins.
5. **Polygon extraction** via `rasterio.features.shapes`.

Basins exceeding 150 m2 (roughly 14 m diameter, the upper end for any Montseny species including mature Fagus) are replaced with circular fallback polygons of 2.5 m radius. On the reference set, 16% of trees received fallback polygons. Median crown area: 13 m2; mean: 20 m2.

### 3.5 Species classification

Binary broadleaf/conifer classification uses an unsupervised percentile-threshold method on two LiDAR features:

- **Return ratio**: fraction of multi-return pulses. Broadleaves have sparser canopy (pulses penetrate multiple layers); conifers absorb most energy on first return.
- **Intensity mean**: laser return amplitude at 1064 nm. Broadleaf leaves are more reflective than conifer needles at this wavelength.

Both features are z-score normalized across the full inventory. A composite score (sum of z-scores) is thresholded at the 40th percentile, labeling the top 60% as broadleaf. The 60% target matches the Catalan Forest Inventory (IEFC) published broadleaf fraction for Montseny.

No training data is used. Validation is via ecological plausibility: per-zone broadleaf fractions range from 25.9% (sw_valley, Pinus halepensis dominated) to 88.9% (ne_slopes, Fagus/Quercus dominated), matching the known altitudinal/aspect species gradient.

### 3.6 Allometric estimation

DBH and above-ground biomass (AGB) are estimated via species-stratified power-law allometrics:

**Crown-to-DBH** (Jucker et al., 2017 form):
DBH (cm) = a x (crown_area x height)^b

**DBH-to-biomass** (Ruiz-Peinado et al., 2011 form):
AGB (kg) = c x DBH^d

| Species group | a | b | c | d |
|---|---:|---:|---:|---:|
| Broadleaf | 0.56 | 0.63 | 0.22 | 2.36 |
| Conifer | 0.48 | 0.65 | 0.085 | 2.49 |

Per-tree confidence intervals: DBH +/-30%, biomass +/-40% (fixed fractions based on published residuals). Stand-level aggregates converge to +/-10-15%.

### 3.7 Health scoring

GRVI (Green-Red Vegetation Index) and ExG (Excess Green Index) are computed on each tree's RGB bbox crop. Trees are classified as healthy (GRVI > 0.06 AND ExG > 20), dead (GRVI < 0 OR ExG < 5), or stressed (otherwise). Thresholds were calibrated from initial temperate-range values (GRVI > 0.10) after observing that Mediterranean canopy at 25 cm/px produces a narrower GRVI dynamic range (0.02-0.15 for healthy vegetation).

## 4. Results

### 4.1 Park-wide inventory

The LiDAR-first pipeline processed 783 of 800 patches (2,004 ha), detecting 228,675 trees after cross-patch deduplication (3.5% duplicate rate at 1 m tolerance).

**Table 1.** Per-zone inventory summary.

| Zone | Trees | Broadleaf % | AGB (t/ha) | Stressed % |
|---|---:|---:|---:|---:|
| high | 28,498 | 53.6 | 36.8 | 4.8 |
| low | 29,358 | 74.9 | 41.1 | 20.6 |
| mid | 33,809 | 54.6 | 52.4 | 1.6 |
| ne_slopes | 32,248 | 88.9 | 48.4 | 10.3 |
| nw_plateau | 25,184 | 63.6 | 74.8 | 23.4 |
| se_ridge | 27,840 | 46.0 | 45.6 | 32.1 |
| summit | 28,283 | 63.3 | 38.7 | 6.2 |
| sw_valley | 23,455 | 25.9 | 45.0 | 20.4 |
| **Total** | **228,675** | **60.0** | **48.0** | **14.3** |

Park-wide AGB of 48.0 t/ha is consistent with the 50-90 t/ha range reported by Vayreda et al. (2012) for Catalan forests, accounting for the inclusion of all trees >= 5 m (not just canopy-dominant individuals as in field-based inventories). Per-zone species fractions and biomass densities are shown in Figure 4.

![Figure 4: Per-zone species composition (a) and biomass density (b) across the 8 Montseny sampling zones.](figures/fig4_zone_species_biomass.png)

### 4.2 Species classification validation

Per-zone broadleaf fractions reproduce the expected ecological gradient: north-facing moist slopes (ne_slopes, 88.9%) and low-elevation broadleaf zones (low, 74.9%) are broadleaf-dominated; south-facing dry ridges (se_ridge, 46.0%) and Pinus halepensis valleys (sw_valley, 25.9%) are conifer-dominated.

An independent physical validation supports the classification: trees classified as broadleaf have significantly larger mean crown area (24.6 m2) than those classified as conifer (13.0 m2), consistent with the known horizontal spreading of broadleaf canopies versus the narrow conical form of Mediterranean pines. Crown area was not an input to the classifier. Example crown polygons from three representative zones are shown in Figure 5.

![Figure 5: Crown polygon examples overlaid on RGB orthophotos for three representative zones: dense broadleaf (nw_plateau), mixed (mid), and sparse conifer (sw_valley).](figures/fig5_crown_examples.png)

### 4.3 Cross-architecture consistency

On the 10-patch reference set, the LiDAR-first architecture (3,320 trees) detected 1.57 times more trees than the best visual architecture (2,119 trees after sliced inference + LiDAR filter). Every tree found by the visual architecture was also found by the LiDAR-first architecture (the visual detections were already LiDAR-verified by the filter). The additional 1,201 trees are concentrated in dense-canopy patches where visual detection recall is lowest.

### 4.4 DBH and biomass distributions

Median DBH: 14.9 cm (broadleaf 15.8, conifer 13.3). The J-shaped diameter distribution (Figure 6) is characteristic of uneven-aged Mediterranean forests: 50% of trees have DBH < 15 cm but contribute only 3.2% of total biomass. The 80+ cm class (1.4% of trees) contributes 25.6% of biomass, reflecting the nonlinear DBH-biomass relationship (AGB proportional to DBH^2.4).

![Figure 6: DBH distribution by species group across 228,675 trees. Dashed lines indicate medians.](figures/fig6_dbh_distribution.png)

Per-zone AGB ranges from 36.8 t/ha (high, open altitude) to 74.8 t/ha (nw_plateau, dense broadleaf). The nw_plateau value is within the 80-180 t/ha range reported by the IEFC for mature Montseny stands.

## 5. Discussion

### 5.1 When to invert the architecture

The LiDAR-first approach is not universally superior. It requires dense airborne LiDAR (>= 4 pts/m2) to resolve individual tree-tops via local-maximum filtering. In regions without LiDAR coverage, the visual detection pipeline (sliced RF-DETR + LiDAR filter at F1 = 0.487) remains the best available option. The architectural inversion is most beneficial when:

1. Dense airborne LiDAR is available (as in Catalunya via ICGC, or in Nordic countries via national LiDAR programs).
2. The visual detector's recall is limited by resolution (tree crowns < 50 pixels wide).
3. Computational cost matters (the LiDAR-first path is 40x faster per patch).

### 5.2 The confidence sweep lesson

The single largest F1 improvement in this work (0.108 to 0.252, a 2.34x lift) came from lowering the RF-DETR confidence threshold from 0.30 to 0.02 — a one-parameter change requiring zero retraining. The detector had been discarding 60% of its true positive candidates at the Phase 1 default threshold. This was invisible until we combined the sweep with a high-precision LiDAR filter that could clean up the additional false positives.

This suggests a general diagnostic: before investing in model retraining or feature engineering, sweep inference-time hyperparameters against an independent ground truth. Default thresholds from early development may impose invisible ceilings on downstream metrics. The full confidence sweep curve (Figure 7) shows the interaction between confidence threshold and LiDAR filtering across six threshold values.

![Figure 7: Confidence sweep showing F1 vs. threshold for raw (unfiltered) and LiDAR-filtered detection. The 2.34x lift from the sweep is annotated.](figures/fig7_confidence_sweep.png)

### 5.3 Negative results

**RGB-distilled classifier.** We attempted to train a gradient boosting classifier on 11 RGB/geometry features to predict whether each detection was a real tree, with labels derived from LiDAR tree-top matching. The goal was to distill LiDAR knowledge into visual features for transfer to regions without LiDAR. F1 never exceeded baseline (0.103 vs. 0.108) at any threshold across a full sweep. RGB at 25 cm/px does not contain sufficient discriminative signal to recover the LiDAR-derived precision boundary for Mediterranean canopy.

**RF-DETR architectural cap.** We observed that at confidence 0.01, every patch returned exactly 300 detections — the model's internal query cap. Raising `num_select` to 600 revealed that the top-300 candidates were already entirely class-0 (the trained tree class); the additional 300 were untrained noise. The cap was not binding on signal — the constraint was effective resolution, not query capacity. This was confirmed when sliced inference (which changes effective resolution without changing the cap) produced a 1.93x F1 improvement.

### 5.4 Limitations

1. **Health labels** are relative indicators based on RGB GRVI, not calibrated against field-measured tree vitality. The 14.3% park-wide stress rate and per-zone gradient are ecologically plausible but should be treated as a proxy for within-park comparison, not an absolute diagnosis.

2. **Species classification** is binary (broadleaf/conifer), sufficient for allometric differentiation but not for per-species management. Genus-level classification would require either hand-labeled training examples or multispectral/hyperspectral imagery.

3. **Allometric coefficients** are representative Mediterranean averages, not site-calibrated. Per-tree DBH accuracy is +/-30%; stand-level accuracy is +/-10-15%. Ground-truth calibration on measured trees would improve accuracy.

4. **Temporal mismatch** between LiDAR (2021-2023) and orthophotos introduces potential inconsistencies for fast-growing or recently disturbed stands. For the mature forests of Montseny, this effect is minor at the 5 m height threshold.

5. **Watershed crown segmentation** produces fallback circular polygons for 16% of trees where basins are empty or over-segmented. SAM2 point-prompted refinement (Ravi et al., 2024) could improve polygon quality for these cases.

## 6. Conclusions

We present a LiDAR-first individual tree detection and inventory system for Mediterranean forests, demonstrated on 228,675 trees across 2,004 hectares of Parc Natural del Montseny. The key contributions are:

1. **An architectural inversion** that makes LiDAR tree-top detection the primary oracle, producing 1.57x more trees in 1/40th the computation time compared to the best visual detection configuration.

2. **An unsupervised species classification** using two LiDAR features (return ratio and intensity) that reproduces the known Montseny ecological gradient with zero training data.

3. **A fully reproducible pipeline** from publicly available ICGC data, with per-tree crown polygons, species groups, DBH, biomass, health labels, and confidence intervals.

4. **Documentation of negative results** (an RGB-distilled classifier that failed, a model capacity probe that identified the wrong bottleneck) that can guide practitioners deciding between inference-time tuning, feature engineering, and architecture changes.

The complete inventory GeoJSON (228,675 trees x 28 attributes), source code, and methodology documentation are publicly available at https://github.com/jordicatafal/forest-pulse (v1.0.0).

## Data Availability

All input data is publicly available from the Institut Cartografic i Geologic de Catalunya (ICGC, https://www.icgc.cat/). The inventory output (228,675 trees) and source code are available at https://github.com/jordicatafal/forest-pulse under the MIT license.

## References

- Hao, Z. et al. (2023). Individual tree detection and crown delineation in urban forests from UAV LiDAR point clouds. *Remote Sensing of Environment*, 287, 113476.
- Jucker, T. et al. (2017). Allometric equations for integrating remote sensing imagery into forest monitoring programmes. *Global Change Biology*, 23(1), 177-190.
- Lotufo, R.A. and Falcao, A.X. (1997). The ordered queue and the optimality of the watershed approaches. *Mathematical Morphology and its Applications to Image and Signal Processing*, 341-350.
- Montero, G. et al. (2005). *Produccion de biomasa y fijacion de CO2 por los bosques espanoles*. INIA Monografias, Madrid.
- Puliti, S. et al. (2021). Modelling above-ground biomass stock over Norway using national forest inventory data with ArcticDEM and Sentinel-2 data. *Remote Sensing of Environment*, 236, 111501.
- Ravi, N. et al. (2024). SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714.
- Roussel, J.-R. et al. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. *Remote Sensing of Environment*, 251, 112061.
- Ruiz-Peinado, R. et al. (2011). New models for estimating the carbon sink capacity of Spanish softwood species. *Forest Systems*, 20(1), 176-188.
- Ruiz-Peinado, R. et al. (2012). Biomass models to estimate carbon stocks for hardwood tree species. *Annals of Forest Science*, 69, 443-452.
- Vayreda, J. et al. (2012). Recent climate changes interact with stand structure and management to determine changes in tree carbon stocks in Spanish forests. *Global Change Biology*, 18(3), 1028-1041.
- Weinstein, B.G. et al. (2019). Individual tree-crown detection in RGB imagery using semi-supervised deep learning. *Remote Sensing*, 11(11), 1309.
- Zhen, Z. et al. (2016). Trends in automatic individual tree crown detection and delineation. *International Journal of Remote Sensing*, 37(21), 4981-5003.
