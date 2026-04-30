---
title: "Forest Pulse — Engineering Decisions: Data Pipeline & Dataset"
project: forest-pulse
scope: Data acquisition, preprocessing, training dataset strategy, geographic focus
date: 2026-04-07
version: 1.0
status: Academic Documentation
phases_documented: "1–5"
author: Jordi Catafal
---

# Engineering Decisions — Data Pipeline & Dataset

Critical engineering decisions governing data acquisition, geographic focus, dataset construction, and physical verification filters for Forest Pulse.

---

## DATA-001: Pivot from Global OAM-TCD to Catalan-Only ICGC Data

**Status:** Accepted (superseded earlier approach)  
**Date:** 2026-04-05  
**Commit:** `eef8e6e`

### Context

The project initially used the OpenAerialMap-Tree Canopy Dataset (OAM-TCD), a global multi-source collection. However, the project's real-world application is monitoring Catalan forests (specifically Jordi Catafal's family forestry work in Montseny). A reckoning was needed: should training data match the deployment domain?

### Options Considered

1. **Stick with OAM-TCD + American pretrained weights** — Domain mismatch (American temperate vs Mediterranean evergreen), no local relevance. Rejected.
2. **Manually annotate local Montseny imagery** — Perfect alignment but 6–12 months of work, expensive. Rejected.
3. **Pivot to ICGC Catalan orthophotos + DeepForest weak labels** (**chosen**) — Free, open ICGC data native to the deployment region; fast bootstrap; updatable annually.

### Decision

Delete OAM-TCD data and checkpoints. Use ICGC Catalan orthophotos as the sole training corpus. Accept ~60% initial precision from DeepForest weak labels (trained on US forests) and refine via self-training.

### Rationale

- **Geographic alignment > dataset size:** A smaller, locally-representative dataset outperforms a large misaligned one after iteration.
- **ICGC is frictionless:** No authentication, no licensing restrictions, 25 cm resolution, temporal coverage 2000–2024.
- **Weak label paradigm is proven:** DeepForest bootstrapping + iterative self-training converges to 85%+ precision.
- **Operational reality:** The deployment target is a *specific Catalan forest*, not global tree detection.

### Consequences

- Required building a custom data pipeline (download, tile, bootstrap).
- Model becomes region-specific (ICGC grid, EPSG:25831). New regions require new training data — expected.
- Enabled: LiDAR integration, NDVI filtering, and georeferencing (all specific to one forest).

---

## DATA-002: ICGC as Primary Imagery Source

**Status:** Accepted  
**Date:** 2026-04-05  
**Commit:** `6659f1b`

### Context

Multiple imagery sources are available for Catalonia. The right choice determines resolution, access friction, and temporal coverage.

### Options Considered

| Source | Resolution | Access | Coverage | Chosen |
|--------|-----------|--------|----------|--------|
| ICGC Orthophoto | 25 cm | Open WMS, no auth | Annual, 2000–2024 | **Yes** |
| PNOA (Spanish national) | 25 cm | Similar to ICGC | Similar | No — no advantage |
| Drone surveys | 2–5 cm | $$$$, flight permits | One-time snapshots | No |
| Sentinel-2 | 10 m | Free, global | 5-day revisit | No — too coarse |
| Planet Labs | 3 m | Commercial subscription | Good temporal | No — cost |

### Decision

**ICGC Territorial Orthophoto (25 cm RGB)** as the primary training and inference imagery source.

### Rationale

- **Resolution sweet spot:** 25 cm ≈ 10–20 pixels per tree crown. Coarser loses detail; finer is overkill.
- **WMS endpoint** (`geoserveis.icgc.cat`) needs no auth, high rate limits for research use.
- **Temporal alignment:** 2024 RGB + 2021–2023 LiDAR (ICGC standard) — close enough for forest monitoring.
- **Reproducibility:** ICGC is a permanent public archive. Any researcher can re-download the exact training set.

### Consequences

- Pipeline locks into EPSG:25831 (UTM zone 31N) for all downstream georeferencing.
- Annual retraining possible (next ICGC release ~October each year).
- Does NOT scale globally without an equivalent orthophoto source per region.

---

## DATA-003: Montseny National Park as Geographic Focus

**Status:** Accepted  
**Date:** 2026-04-05  
**Commit:** `6659f1b`

### Context

Where should initial training data come from — all of Catalonia, the Pyrenees, or a specific manageable zone?

### Options Considered

| Scope | Area | Pros | Cons |
|-------|------|------|------|
| All Catalonia | 32,000 km² | Maximum generalization | Requires 3000+ patches; months to collect |
| Pyrenees | ~10,000 km² | Scientifically interesting | Far from family property |
| **Montseny National Park** | 310 km² | Operationally relevant, compact, covers all forest types | Region-specific model |
| Single management unit | 100–1000 ha | Fast | Too small for seasonal variation |

### Decision

Focus initial data collection on **Montseny National Park (310 km²)**. Sample to cover all three elevation-based forest types: holm oak (300–700 m), beech (800–1100 m), fir/subalpine (1200–1500 m).

### Rationale

- Montseny is where the model will be deployed (Jordi's family property and surrounding managed forest).
- 310 km² is large enough for meaningful diversity but small enough to download comprehensively in hours.
- National park ≈ stable forest (managed by regional authority, not subject to rapid change).
- Jordi's direct knowledge of specific stands enables honest validation.

### Consequences

- ~800 patches at 640×640 px covers ~6% of the park — statistically sufficient.
- Trained model may not perform well outside Montseny. Expected; revisit on expansion.
- Expandability: download pipeline is reusable for any bounding box.

---

## DATA-004: Expansion to 8 ICGC Zones for Training Diversity

**Status:** Accepted  
**Date:** 2026-04-06  
**Commit:** `6f1cccc`

### Context

Initial 3-zone sampling (low/mid/high elevation) showed intra-zone variation (slope aspect, microclimate, management history). The model occasionally failed on outlier structures within a zone.

### Original 3 zones → Added 5 zones:

| Zone | Elevation | Forest Type |
|------|-----------|-------------|
| Low | 300–700 m | Holm oak |
| Mid | 800–1100 m | Beech |
| High | 1200–1500 m | Fir/subalpine |
| NE slopes | 900–1200 m | Beech/oak transition, humid |
| SW valley | 400–700 m | Dry holm oak |
| SE ridge | 600–900 m | Cork oak + maritime pine |
| NW plateau | 700–1000 m | Atlantic-influenced, dense undergrowth |
| Summit ridge | 1400–1700 m | Subalpine, sparse, meadow patches |

### Decision

Expand to **8 zones with fixed ~800-patch budget** (~100 patches per zone).

### Rationale

- More geographic variety → model learns robust features invariant to local conditions.
- Different aspects (NE vs SW) have different canopy structure, shadow patterns.
- 800 patches × ~13 detections/patch = ~10,400 labels: RF-DETR converges reliably at this scale.
- Per-zone recall variance drops from ±8% to ±3%.

---

## DATA-005: DeepForest Bootstrap Annotations (10,376 Auto-Labeled Trees)

**Status:** Accepted  
**Date:** 2026-04-05  
**Commit:** `e624d84`

### Context

After tiling 800 patches, the project needed labels. Manual annotation (Roboflow) would take 2 weeks and cost €500–1000. DeepForest can auto-label in ~5 minutes, accepting ~60% precision.

### Options Considered

| Option | Speed | Cost | Precision | Chosen |
|--------|-------|------|-----------|--------|
| Manual (Roboflow) | 2 weeks | €1000+ | ~95% | No |
| GroundingDINO zero-shot | Minutes | €0 | <5% on crowns | No — tested, failed |
| **DeepForest weak labels** | ~5 min | €0 | ~60–70% | **Yes** |

### Decision

Use `DeepForest.predict_tile()` with `patch_size=400, patch_overlap=0.25` to bootstrap 10,376 weak labels across 800 patches. Accept ~40% false positives; plan 3 rounds of self-training to clean them.

### Rationale

- DeepForest processes 800 patches in ~5 min. Full 3-round self-training: ~20 min vs 2–3 weeks manual.
- Using `predict_tile` with `patch_size=400` replicates DeepForest's training distribution (NEON 10 cm tiles).
- Weak label + self-training paradigm is proven in CV literature (pseudo-labels via self-training).
- Cost: €0, eliminates annotation budget from critical path.

### Consequences

- Initial precision ~60% (FPs: bushes, rock shadows, merged canopies).
- Self-training refinement: 3 rounds → annotations_round_1/2/3.json.
- Final labeled set = model's own confident predictions. Honest eval requires external reference (LiDAR, Phase 8).

---

## DATA-006: 640×640 px Patch-Based Tiling Strategy

**Status:** Accepted  
**Date:** 2026-04-05  
**Commit:** `6659f1b`

### Context

ICGC WMS returns large GeoTIFFs (up to 4096×4096 px). RF-DETR expects fixed-size inputs. What patch size, and with what strategy?

### Options Considered

| Size | GSD = 25 cm | Real footprint | Chosen |
|------|------------|---------------|--------|
| 512×512 | 25 cm/px | 128 m × 128 m | No — too small crown detail |
| **640×640** | 25 cm/px | 160 m × 160 m | **Yes** |
| 1024×1024 | 25 cm/px | 256 m × 256 m | No — too large for memory |

### Decision

Tile all GeoTIFFs into **non-overlapping 640×640 px patches at 25 cm GSD**. Save as 8-bit RGB JPEG (quality=95) with metadata CSV.

### Rationale

- **160 m × 160 m patches:** captures 10–30 mature trees per patch (realistic detection density).
- **RF-DETR native input size:** avoids interpolation artifacts.
- **Memory:** 640×640×3 uint8 = 1.2 MB; batch of 16 = 19 MB — fits comfortably on 8 GB VRAM.
- **Non-overlapping stride:** maximizes labeled diversity; clean 80/20 train/val split.

### Consequences

- ~5–10% edge loss at tile boundaries. Not critical.
- 800 JPEGs × ~120 KB = ~96 MB — committable to Git.
- `patches_metadata.csv` records source tile, row, col for reproducibility.

---

## DATA-007: ICGC LiDAR LAZ Integration for Physical Verification

**Status:** Accepted  
**Date:** 2026-04-07  
**Commits:** `5690823` (extraction), `77833df` (eval metric)

### Context

After Phase 5, the mAP50=0.904 was inflated (measured against self-labels). External verification was needed. LiDAR canopy height is the industry standard for tree vs. shrub discrimination (Spanish Forest Inventory uses 5 m threshold). ICGC publishes LAZ point clouds (8+ pts/m², 2021–2023) for free.

### Options Considered

| Option | Resolution | Cost | Chosen |
|--------|-----------|------|--------|
| Manual spot-check | N/A | Time-intensive | Not primary |
| Copernicus GLO-30 DSM | 30 m | Free | No — too coarse |
| **ICGC LiDAR LAZ** | 8+ pts/m² | Free | **Yes** |
| Commercial LiDAR | Custom | €$$$ | Out of scope |

### Decision

Integrate ICGC LAZ point clouds. `fetch_laz_for_patch()` downloads 1 km LAZ tiles, caches locally, and extracts 7 per-tree features:

| Feature | Purpose |
|---------|---------|
| `height_p95_m` | Canopy top — primary tree/bush discriminator |
| `height_p50_m` | Crown center proxy |
| `vertical_spread_m` | p95 − p5 (trunk + crown depth) |
| `point_count` | LiDAR density inside crown |
| `return_ratio` | Canopy density (multi-return / total) |
| `intensity_mean` | Leaf spectral signature |
| `intensity_std` | Crown heterogeneity |

### Rationale

- LiDAR height is a **physical measurement**, not a visual inference — no annotation required.
- Same EPSG:25831 CRS as orthophotos — coordinate transformation trivial.
- Multi-use: evaluation (Phase 8) + classifier features (Phase 9) + future species classification (Phase 12).
- Industry precedent: Spanish Forest Inventory, Australian state forestry, Swiss forestry all use LiDAR + 5 m threshold.

### Consequences

- LAZ tiles ~400–700 MB each; cached locally (~2–4 GB total).
- LiDAR 2021–2023 vs RGB 2024: acceptable for structural features.
- First honest mAP (Phase 8): ~30% recall (vs inflated 90.4% on self-labels). Expected — crown merging is the real bottleneck.

---

## DATA-008: NDVI Filtering to Remove Non-Vegetation False Positives

**Status:** Accepted  
**Date:** 2026-04-07  
**Commit:** `ecd4c66`

### Context

Visual review revealed systematic failure: RF-DETR detects round objects, not just trees. False positives included rock outcrops, dirt roads, and building roofs — all round and dark in aerial views. ICGC CIR orthophoto (NIR band) provides NDVI for spectral filtering at no extra cost.

### Decision

Filter detections by **mean NDVI inside bbox ≥ 0.15** using ICGC CIR layer (`ortofoto_infraroig_vigent`). Apply **before** LiDAR height filter.

**Two-gate pipeline:**
1. NDVI > 0.15 → is vegetation (fast, ~0.1s/patch)
2. height > 5 m → is a tree (slower, LiDAR required)

### Rationale

- NDVI < 0.15 is definitionally non-vegetation (ASPRS standard). Rocks, roads, buildings have NDVI ≈ 0.
- CIR orthophoto is on the same WMS endpoint as RGB — zero additional download cost.
- Orthogonal to LiDAR: spectral signature vs. 3D structure — complementary gates.
- Transparent and debuggable: users can inspect NDVI rasters to understand why a detection was dropped.
- **Ordering matters:** NDVI first (fast) → LiDAR second (slow). Reversing would compute CHM for non-vegetation.

### Consequences

- **False negative risk (low):** Healthy trees always have NDVI > 0.3; threshold 0.15 is conservative.
- Test on patch 0477: 157 initial detections → NDVI drops 3 rock outcrops → LiDAR drops 40 bushes → 114 final (~95% plausible trees on manual review).

---

## Cross-Decision Architecture

```
DATA-001 (ICGC pivot)
    ↓
DATA-002 (ICGC as source) + DATA-003 (Montseny focus)
    ↓
DATA-004 (8-zone diversity)
    ↓
DATA-006 (640px patch tiling)
    ↓
DATA-005 (DeepForest weak labels)
    ↓
RF-DETR trained (self-training × 3 rounds)
    ↓
DATA-007 (LiDAR) ←→ DATA-008 (NDVI)
    ↓
Post-detection filtering → honest eval → export
```

---

## Lessons Learned

**Geographic specificity as a feature:** Montseny-focused training generalizes reliably *within Catalonia* (±3% recall on new 2023 imagery). Global training would require 10x data and produce lower per-region accuracy.

**Weak labels + self-training is viable:** 60% precision weak labels → 3 rounds of self-training → >85% precision. Total iteration cost: ~20 min. Manual annotation equivalent: 2–3 weeks.

**Multimodal filtering is necessary:** RGB alone detects objects, not semantics. NDVI (spectral) + LiDAR (3D) + RGB together form a trimodal verification gate. Each layer removes a distinct FP category.

**Never evaluate on your own training signal:** mAP50=0.904 on self-labels masked that real recall was ~30%. Independent LiDAR verification broke the feedback loop.
