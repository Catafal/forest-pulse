# NDVI from ICGC Infrared Orthophotos — Research

Date: 2026-04-05
Scope: Can ICGC's color-infrared (IRC) orthophoto give us NDVI good enough to
distinguish trees from shrubs/bushes inside RF-DETR bboxes over Catalan forest
(Montseny, 25 cm)?

---

## TL;DR (read this first)

1. ICGC publishes a color-infrared product at 25 cm called **OI-25C**
   (`ortofoto_infraroig_*`). It is **NOT a 4-band R/G/B/NIR GeoTIFF**. It is a
   **3-band false-color composite** where the channels are stored as
   **Band1 = NIR, Band2 = Red, Band3 = Green**. The blue band is not delivered.
   That's the standard CIR convention used by NAIP/ICGC/IGN.
2. Because Band 1 = NIR and Band 2 = Red are *both inside the IRC file*, you
   can compute NDVI from a **single IRC tile** — you do **not** need to
   download the RGB orthophoto and co-register two rasters. This is the most
   important practical takeaway.
3. The IRC file is delivered in the same grid, CRS (EPSG:25831), and 25 cm
   resolution as the RGB. Pixels are aligned exactly — they come from the same
   flight, the same orthorectification, the same tiling sheets.
4. **Honest assessment for our use case**: NDVI alone will *not* reliably tell
   a beech tree from a heather/box bush at 25 cm. It will reliably tell
   *vegetation* from *non-vegetation* (rock, road, building, bare soil, water).
   For the tree-vs-shrub question, NDVI is a weak feature; the literature is
   unanimous that you need **height** (CHM/LiDAR) or **temporal phenology** or
   **texture** to separate the two. Use NDVI as a **vegetation gate / false-
   positive killer** for RF-DETR detections, not as a tree-vs-shrub classifier.

---

## 1. ICGC Infrared Orthophoto product (OI-25C)

### 1.1 Product identity

| Field | Value |
|---|---|
| Short code | **OI-25C** ("Ortofoto Infraroja 25 cm") |
| Long name | Infrared orthophoto of Catalonia 25 cm v4r0 |
| Resolution | **25 cm GSD** (matches the RGB OF-25C exactly) |
| CRS | **EPSG:25831** (ETRS89 / UTM 31N) — same as RGB |
| Bands | **3 bands**, false-color composite |
| Band order | **B1 = NIR, B2 = Red, B3 = Green** (no blue) |
| Bit depth | 8-bit (per channel, same as ECW source) |
| File formats (per area) | TIFF, JP2, ECW, JPG, PDF |
| File formats (full Catalonia) | GeoPackage, ECW |
| Years available | Annual 1983–2024, plus 1977, 1975, 1970, 1956, 1945 (gaps in 1991, 1999) |
| License | CC-BY 4.0, no auth required |

The 3-band layout is the same convention as USDA NAIP CIR, IGN BD ORTHO IRC,
and every other "ortofoto infraroja" I checked. Vegetation appears bright red
because chlorophyll reflects strongly in NIR which is mapped to the red display
channel.

### 1.2 Why this matters: NDVI from a single file

NDVI = (NIR - Red) / (NIR + Red)

In an OI-25C tile opened with rasterio:

```python
nir = src.read(1).astype("float32")   # Band 1 of IRC = NIR
red = src.read(2).astype("float32")   # Band 2 of IRC = Red (real red wavelength)
ndvi = (nir - red) / (nir + red + 1e-6)
```

That's it. You do not need the RGB ortho. You do not need to co-register two
rasters. The "Red" inside the IRC file is the same red wavelength channel as
the "Red" inside the RGB ortho — they were simply demultiplexed from the same
4-band aerial sensor at flight time.

### 1.3 Download: same endpoints as the RGB we already use

**WMS endpoint** (identical to RGB):

```
https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms
```

**Layers** (verified from the WMS Territorial Orthophoto page):

| Layer name | Description |
|---|---|
| `ortofoto_infraroig_vigent` | Last definitive IRC orthophoto |
| `ortofoto_infraroig_provisional` | Provisional IRC (current production flight) |
| `ortofoto_infraroig_serie_anual` | Time series — use with `&TIME=YYYY` |
| `ortofoto_infraroig_2024` … `ortofoto_infraroig_2006-2007` | Year-pinned layers |

**Same WMS rules as RGB**: max 4096×4096 px per GetMap, supports
`image/tiff`, supports CRS EPSG:25831. So the existing WMS tiling code in
`scripts/tile_orthophoto.py` and `scripts/download_montseny.py` works
unchanged — just swap the layer name.

**FME bulk-clip endpoint** (the one OpenICGC plugin uses):

```
https://qgis:qgis@descarregues.icgc.cat/fmedatastreaming/orto-territorial/ICGC_orto-territorial_download.fmw
  ?x_min=…&y_min=…&x_max=…&y_max=…
  &poligon=
  &codi=openicgc
  &projecte=irc_vigent     # <-- IRC instead of rgb_vigent
  &gsd=0.25
```

Product codes for IRC via FME: `oi25c` (25 cm), `oi5m` (50 cm). These are
already documented in `.claude/tasks/ICGC_MONTSENY_DOWNLOAD_RESEARCH.md`.

**Bulk pre-zipped GeoPackages / ECW** (full Catalonia per year):

```
https://datacloud.icgc.cat/datacloud/orto-territorial/gpkg_unzip/
https://datacloud.icgc.cat/datacloud/orto-territorial/ecw_unzip/
```

### 1.4 Pixel-perfect registration with the RGB ortho

Yes — the IRC and RGB are produced from the same flight, the same camera, the
same orthorectification model, at the same 25 cm grid in EPSG:25831, on the
same sheet boundaries. If you request the same WMS bbox at the same WIDTH /
HEIGHT for both `ortofoto_color_vigent` and `ortofoto_infraroig_vigent`, the
two TIFFs will overlay pixel-for-pixel. No reprojection or warping needed.

### 1.5 Cloud cover

ICGC pre-screens flights for cloud cover before the "vigent" (definitive)
release; the published OI-25C is essentially cloud-free. Provisional layers
may contain occasional cloud-affected sheets but the QC is good. No need to
build a cloud mask.

---

## 2. NDVI computation — the basics applied to OI-25C

### 2.1 Formula and value range

NDVI = (NIR - Red) / (NIR + Red), range [-1, +1].

Because the OI-25C is 8-bit DN (0..255) without radiometric calibration, the
NDVI you compute is a **relative index**, not a calibrated reflectance NDVI.
Absolute thresholds from Sentinel-2 / Landsat literature do **not** transfer
1:1. You should treat any threshold as "tuned on Montseny imagery", not
"physically meaningful".

### 2.2 Reference thresholds (from calibrated sensors — use as priors only)

| Cover | Calibrated NDVI | Notes |
|---|---|---|
| Water | < 0 | NIR < Red |
| Bare soil / rock / road | 0.0 – 0.15 | Use as the "non-vegetation" floor |
| Sparse / dry grass | 0.15 – 0.30 | Mediterranean summer grassland |
| Shrubs / heath / maquis | 0.30 – 0.55 | Wide overlap with stressed trees |
| Healthy broadleaf canopy | 0.60 – 0.85 | Beech, oak in summer |
| Dense conifer canopy | 0.55 – 0.80 | Often slightly lower than broadleaf |

**There is no clean cut between "shrub" and "tree".** The 0.3–0.6 band is
shared. This is the central honest finding of the research below.

### 2.3 Practical thresholds for our pipeline (proposal)

Use NDVI as a **gate**, not a classifier:

- `ndvi_mean < 0.15` → reject the bbox (false positive: rock, road, building, water)
- `0.15 ≤ ndvi_mean < 0.35` → flag as "uncertain vegetation" (likely shrub or stressed)
- `ndvi_mean ≥ 0.35` → accept as vegetation; pass to RF-DETR + health scoring

These numbers should be re-tuned on a small Montseny calibration set. Don't
hardcode them — put them in `config.yaml`.

---

## 3. Honest answer: can NDVI distinguish trees from bushes at 25 cm?

**Short answer: No.**

**Long answer:** Every paper I reviewed says the same thing: NDVI is a measure
of *photosynthetic activity per pixel*. A healthy heather bush and a healthy
beech tree both have lots of chlorophyll per unit canopy area, and at 25 cm
they look spectrally identical. NDVI cannot tell them apart.

Key citations:

- **Wang et al.** ("Tree, Shrub, and Grass Classification Using Only RGB
  Images", *Remote Sensing*, OSTI 1668666): "NDVI cannot differentiate tree,
  shrub, and grass because of their similar spectral characteristics."
- **Stoffels et al.** ("An Object-Based Approach for Mapping Shrub and Tree
  Cover on Grassland Habitats by Use of LiDAR and CIR Orthoimages", *Remote
  Sensing* 5(2):558): explicitly combines CIR-derived NDVI **with LiDAR
  height** because NDVI alone could not separate the two classes. They use a
  height threshold (~5 m from CHM) as the actual tree-vs-shrub discriminator;
  NDVI only filters out non-vegetation.
- **Helman et al.** ("A Phenology-Based Method for Monitoring Woody and
  Herbaceous Vegetation in Mediterranean Forests from NDVI Time Series",
  *Remote Sensing* 7(9):12314): in Mediterranean forest specifically,
  separation only works using **multi-date NDVI** that exploits the
  green-up/senescence offset between woody and herbaceous components — i.e.
  you need a time series, not a single image.
- **ScienceDirect S2352938520302858** (forest cartography from NDVI series):
  "Mediterranean is rich of shrubland […] this approach finds it difficult to
  distinguish trees from other kind of vegetation forms."

What *does* work for tree-vs-shrub at 25 cm in the literature:

| Feature | Why it works | Cost for us |
|---|---|---|
| **Canopy height** (CHM from LiDAR) | Trees > ~3–5 m, shrubs < 2 m. Direct physical discriminator. | ICGC publishes LiDAR (LIDARCAT) — feasible, biggest single win. |
| **Texture** (GLCM entropy, local variance) | Tree crowns have more internal shadow + branch structure than shrubs. | Cheap (skimage), modest gain. |
| **Crown shape / size** | Trees have larger, rounder crowns | RF-DETR already gives bbox size. Free. |
| **Multi-date NDVI / phenology** | Deciduous trees vs evergreen shrubs differ in green-up curves | Requires N years of IRC. |
| **NIR alone (raw band)** | Near-IR reflectance scales with leaf area / LAI; tree canopies usually higher than shrub patches | One number, free from same IRC tile. |

### 3.1 Recommendation for Forest Pulse

NDVI from OI-25C should be added as **one feature** that does two specific
jobs in the pipeline:

1. **False-positive killer**: drop RF-DETR detections whose mean NDVI is below
   ~0.15. Rocks, scree, roads, bare soil, and roof shingles will not pass
   this. This alone will measurably reduce false positives in Montseny scree
   areas.
2. **Health input**: feed `ndvi_mean` and `ndvi_std` from the bbox crop into
   the existing `health.py` scorer alongside the RGB-derived greenness score.
   NDVI degrades faster than visible greenness when a tree is stressed, so it
   adds real signal to health scoring.

**Do not** use NDVI to classify tree vs shrub. To attack that question
properly, the next step is to pull ICGC's LIDARCAT and build a CHM. That is a
separate work item, not this one.

---

## 4. Implementation: NDVI for a bbox in EPSG:25831

### 4.1 Minimal function (drop into `health.py` or a new `ndvi.py`)

```python
# src/forest_pulse/ndvi.py
"""NDVI computation from ICGC OI-25C infrared orthophotos.

ICGC IRC tiles are 3-band CIR composites with band order:
    Band 1 = NIR (near-infrared, ~750-900 nm)
    Band 2 = Red (real red wavelength, ~600-700 nm)
    Band 3 = Green
There is NO blue band. The "Red" here is the actual red sensor band, not the
display-channel red of the false-color image. We can therefore compute NDVI
straight from a single IRC tile without needing the RGB orthophoto.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import from_bounds


def compute_ndvi_array(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - Red) / (NIR + Red).

    Inputs are 8-bit DN arrays from ICGC OI-25C; output is float32 in
    [-1, 1]. We add a small epsilon to avoid divide-by-zero on black pixels
    (image edges, no-data fill).
    """
    nir_f = nir.astype(np.float32)
    red_f = red.astype(np.float32)
    denom = nir_f + red_f
    # Mask pixels where both bands are zero (likely no-data) so NDVI is NaN
    # there and ignored downstream by np.nanmean / np.nanstd.
    valid = denom > 0
    ndvi = np.full_like(nir_f, np.nan, dtype=np.float32)
    ndvi[valid] = (nir_f[valid] - red_f[valid]) / denom[valid]
    return ndvi


def ndvi_for_bbox(
    irc_path: str,
    bbox_25831: tuple[float, float, float, float],
) -> dict[str, float]:
    """Return mean / std / min / max NDVI inside a bbox in EPSG:25831.

    Args:
        irc_path: path to an ICGC OI-25C GeoTIFF (3-band CIR, EPSG:25831).
        bbox_25831: (minx, miny, maxx, maxy) in metres, EPSG:25831.

    Returns:
        Dict with keys mean, std, min, max, n_valid_px.
    """
    # We open the IRC tile, slice the window matching the bbox, and read only
    # bands 1 (NIR) and 2 (Red). Window read avoids loading the whole tile.
    with rasterio.open(irc_path) as src:
        # Sanity: bail loudly if someone hands us the RGB by mistake
        if src.count < 2:
            raise ValueError(
                f"{irc_path} has {src.count} bands; expected ≥2 (NIR, Red)."
            )
        window = from_bounds(*bbox_25831, transform=src.transform)
        nir = src.read(1, window=window)
        red = src.read(2, window=window)

    ndvi = compute_ndvi_array(nir, red)
    valid = ~np.isnan(ndvi)
    if not valid.any():
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "n_valid_px": 0}
    return {
        "mean": float(np.nanmean(ndvi)),
        "std": float(np.nanstd(ndvi)),
        "min": float(np.nanmin(ndvi)),
        "max": float(np.nanmax(ndvi)),
        "n_valid_px": int(valid.sum()),
    }
```

Notes for the integration:

- The function takes a bbox in **EPSG:25831 metres**, not pixel coords. Our
  detection pipeline produces pixel bboxes; convert them via the existing
  `georef.py` (`pixel_bbox_to_crs`) before calling `ndvi_for_bbox`.
- Use `rasterio.windows.from_bounds` so we never load a full 4 GB tile.
- The function intentionally returns a dict — not a scalar — so `health.py`
  can use mean for thresholding and std as a texture proxy (high std inside a
  bbox means heterogeneous vegetation, often a tree edge).

### 4.2 Library choice

- `rasterio` (already in TECH_STACK) — sufficient, no new dep.
- `rioxarray` is nicer for big multi-tile mosaics but unnecessary for per-bbox
  reads. Skip it; KISS.
- `numpy` only for the math. No `gdal_calc.py` shellouts.

### 4.3 Where this plugs into the pipeline

```
detect.py (RF-DETR)              ──► sv.Detections (pixel bboxes)
        │
        └─► georef.py (pixel → EPSG:25831)
                │
                └─► ndvi.py (mean NDVI per bbox from OI-25C)
                        │
                        ├─► filter: drop where ndvi_mean < 0.15  (FP killer)
                        └─► health.py: use ndvi_mean + ndvi_std as features
```

The IRC tile lookup should mirror the existing RGB tile lookup — same sheet
index, same filename pattern, just `oi25c` instead of `of25c`.

---

## 5. ICGC WMS for IRC — concrete example matching our existing RGB code

The only change from our existing `download_montseny.py` is the layer name:

```python
from owslib.wms import WebMapService

wms = WebMapService(
    "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms",
    version="1.3.0",
)

img = wms.getmap(
    layers=["ortofoto_infraroig_vigent"],   # <-- the only change
    srs="EPSG:25831",
    bbox=(450000, 4620000, 451000, 4621000),
    size=(4000, 4000),                      # 25 cm at 1×1 km
    format="image/tiff",
)
with open("montseny_tile_irc.tif", "wb") as f:
    f.write(img.read())
```

The resulting GeoTIFF is 3-band, B1=NIR, B2=Red, B3=Green, EPSG:25831, 25 cm,
pixel-aligned with the RGB tile you'd get from `ortofoto_color_vigent` for the
same bbox/size. Confirmed against ICGC OI-25C metadata.

For temporal analysis use `ortofoto_infraroig_serie_anual` with `&TIME=2018`,
`&TIME=2024`, etc. Same pattern as the RGB time series we already use.

---

## 6. Real-world examples and tooling

- **OpenICGC QGIS plugin**: open source, uses the same FME / WMS endpoints
  documented above; handles both `rgb_vigent` and `irc_vigent`. Useful as a
  reference implementation. Source on GitHub under `ICGCat/qgis-plugins`.
- **earthpy** (Earth Lab, U. Colorado): MIT-licensed Python library with
  ready-made `earthpy.spatial.normalized_diff(b1, b2)` and CIR plotting
  helpers. Worth depending on **only** if we want quick visualisation; for the
  pipeline itself, the 10-line rasterio version above is enough.
- **DeepForest + NDVI**: there is no first-class fusion in DeepForest. The
  pattern people use (and what we should copy) is: detect with the deep model,
  then post-filter / re-rank detections with a vegetation index computed
  separately. That is exactly the gate-and-feature approach proposed above.
- **Spanish/Catalan forestry use of ICGC IRC**: CREAF (Centre de Recerca
  Ecològica i Aplicacions Forestals, Bellaterra) routinely uses ICGC IRC for
  Mediterranean forest mapping and feeds it into MiraMon and into PNOA-IR
  comparisons. Their published workflows always combine CIR + LiDAR (PNOA-LiDAR
  or LIDARCAT) — never CIR alone — which corroborates the honest assessment
  in §3.

---

## 7. Action items for Forest Pulse (concrete, MVP-sized)

1. Add `src/forest_pulse/ndvi.py` with `compute_ndvi_array` and
   `ndvi_for_bbox` (the two functions in §4.1, ~60 lines total).
2. Extend `scripts/download_montseny.py` with a `--product irc` flag that
   swaps the WMS layer to `ortofoto_infraroig_vigent` and writes alongside the
   RGB tile in `data/raw/montseny/irc/`.
3. In `detect.py`'s post-processing (or a new thin filter step), call
   `ndvi_for_bbox` for each detection and drop bboxes with `mean < 0.15`.
   Make the threshold a config value, default 0.15.
4. In `health.py`, add `ndvi_mean` and `ndvi_std` to the feature vector. Do
   not break the existing RGB-only health score — make NDVI features optional
   and skipped if no IRC tile is available.
5. **Out of scope for this task** but the right next step: download LIDARCAT
   for Montseny, build a CHM, and use height (not NDVI) as the actual
   tree-vs-shrub discriminator. NDVI will never solve that problem alone at
   25 cm.

---

## Sources

- [ICGC — Territorial Orthophoto product page](https://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Image/Territorial-Orthophoto)
- [ICGC — Infrared Orthophoto of Catalonia 25 cm (OI-25C) v4r0 2020 metadata](https://catalegs.ide.cat/geonetwork/srv/api/records/ortofoto-25cm-v4r0-infraroja-2020?language=eng)
- [ICGC — WMS Territorial Orthophoto (layer list)](https://www.icgc.cat/en/Geoinformation-and-Maps/Online-services-Geoservices/WMS-Orthoimages/WMS-Territorial-Orthophoto)
- [USDA FSA — Four Band Digital Imagery Information Sheet (CIR band order convention)](https://www.fsa.usda.gov/Internet/FSA_File/fourband_infosheet_2012.pdf)
- [EOS — Color Infrared (CIR) imagery in remote sensing](https://eos.com/make-an-analysis/color-infrared/)
- [Wang et al., "Tree, Shrub, and Grass Classification Using Only RGB Images" (OSTI 1668666)](https://www.osti.gov/servlets/purl/1668666)
- [Stoffels et al., "An Object-Based Approach for Mapping Shrub and Tree Cover on Grassland Habitats by Use of LiDAR and CIR Orthoimages", Remote Sensing 5(2):558](https://www.mdpi.com/2072-4292/5/2/558)
- [Helman et al., "A Phenology-Based Method for Monitoring Woody and Herbaceous Vegetation in Mediterranean Forests from NDVI Time Series", Remote Sensing 7(9):12314](https://www.mdpi.com/2072-4292/7/9/12314)
- [ScienceDirect — Analysis of NDVI multi-temporal series for forest cartography](https://www.sciencedirect.com/science/article/abs/pii/S2352938520302858)
- [ISPRS — Urban Vegetation Classification with NDVI Threshold Value](https://isprs-archives.copernicus.org/articles/XLII-4-W16/237/2019/isprs-archives-XLII-4-W16-237-2019.pdf)
- [Earth Lab — Calculate NDVI Using NAIP Remote Sensing Data in Python](https://earthdatascience.org/courses/use-data-open-source-python/multispectral-remote-sensing/vegetation-indices-in-python/calculate-NDVI-python/)
- [Wikipedia — Normalized Difference Vegetation Index](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index)
- [USGS — NDVI, the Foundation for Remote Sensing Phenology](https://www.usgs.gov/special-topics/remote-sensing-phenology/science/ndvi-foundation-remote-sensing-phenology)
