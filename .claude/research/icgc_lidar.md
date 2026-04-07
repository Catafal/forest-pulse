# ICGC LiDAR for Forest Pulse — RF-DETR + LiDAR Height Fusion

Research scope: filtering RF-DETR tree-crown bounding boxes (EPSG:25831) by
LiDAR-derived canopy height to drop non-tree detections (<5 m → bushes/shrubs).
Focus area: Montseny Natural Park, Catalonia.

---

## 1. ICGC LiDAR Product (what they actually publish)

ICGC maintains the **LiDAR Territorial** programme. As of 2024 the third
nationwide coverage is published and is the one to use:

| Item | Value |
|---|---|
| Product name | `lidar-territorial` v3r1 |
| Acquisition years | 2021–2023 (third coverage) |
| Format | **LAZ 1.4** (compressed LAS), ASPRS classification |
| Point density | **≥ 8 pts/m² minimum**, ~65% of tiles >10 pts/m² |
| Tile size | **1 km × 1 km** |
| CRS | **ETRS89 / UTM 31N — EPSG:25831** (the project's CRS) |
| Classification | Ground (2), vegetation (3/4/5), buildings (6), etc. ASPRS |
| License | Open data, free download |

Historical coverages also exist (1st: 2008–2011, 2nd: 2015–2016) → enables
multi-temporal analysis if you want change detection from LiDAR itself, not
just from RGB.

### Derived raster products published by ICGC

ICGC publishes several pre-rasterized elevation models from the same point
cloud (no need to compute them yourself if you don't need sub-metre):

- **DTM** — Digital Terrain Model (bare earth), available at **2 m, 5 m, 15 m**
- **DSM** — Digital Surface Model (top of everything), available at 2 m / 5 m
- **DTMv** — terrain + vegetation
- **DTMe** — terrain + buildings

> **CHM (Canopy Height Model) is NOT directly published as a separate product
> by ICGC.** You compute it yourself: `CHM = DSM − DTM`. This is a single
> `gdal_calc.py` step or one numpy subtraction.

---

## 2. Download Access

### Bulk download (LAZ point clouds)

Base directory:
```
https://datacloud.icgc.cat/datacloud/lidar-territorial/
```

Subdirectories:
- `laz_unzip/` — point clouds, organised by 10 km blocks
- `orto-ecw_unzip/` — 15 cm orthophotos (ECW)
- `orto-gpkg_unzip/` — 15 cm orthophotos (GeoPackage)
- `json/` — per-tile metadata
- `vigent/` — currently-valid index

LAZ tiles are grouped under 10 km parent directories:
```
https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/full10km<XXYY>/
```

File naming convention (1 km tile):
```
lidar-territorial-v3r1-full1km<TILEID>-2021-2023.laz
```
Example: `lidar-territorial-v3r1-full1km400620-2021-2023.laz`

The `TILEID` encodes the bottom-left ETRS89/UTM31N coordinate of the 1 km
square in km × 10 (so `400620` ≈ X=400000, Y=4620000 → easting/northing in
metres). The 10 km parent block follows the same scheme truncated.

**File sizes**: roughly **190 MB – 850 MB per 1 km LAZ tile**, average
~450 MB. A 10 km block (100 tiles) is therefore ~45 GB.

### Montseny coverage

Montseny Natural Park (~30 × 25 km, centred ~UTM31N X≈448000 Y≈4625000)
is fully covered. It spans roughly the 10 km blocks `4462`, `4562`, `4662`,
`4463`, `4563`, `4663` (~6 blocks → ~600 LAZ tiles → ~270 GB raw point
cloud worst case). For an MVP you only need the few 1 km tiles overlapping
your test patches.

### WMS / WCS — much lighter alternative for elevations

If you only need DSM/DTM rasters (which is *exactly* the case for height
filtering), skip the point clouds and use the WCS service. ICGC publishes:

- **WCS Digital Terrain Model of Catalonia** — 5×5 m and 15×15 m DTM
- **WMS Elevations / orientation / shadows** (visualisation only)
  - Endpoint: `https://geoserveis.icgc.cat/icgc_mdt2m/wms/service?`
- **WMS Territorial Elevations**, **WMS Coastal Elevations**

Geoservices hub: `https://www.icgc.cat/en/Geoinformation-and-Maps/Online-services-Geoservices/WMS-and-WCS-Elevations`

WCS supports `GetCoverage` with a bbox and CRS, returning a GeoTIFF clipped
to your area — perfect for fetching only what your detection patch needs.
Supported CRS include EPSG:25831 (native).

> **Practical recommendation**: for the Forest Pulse MVP, you most likely
> only need the **2 m DSM and 2 m DTM** for the patches you process. Pull
> them once via WCS for your AOI, store as a single GeoTIFF, compute
> `CHM = DSM − DTM`, and you're done. No LAZ processing required.

If you need sub-2-m precision (e.g. small saplings vs shrubs), then drop
to LAZ + PDAL.

---

## 3. Computing a Canopy Height Model (CHM)

### Option A — Cheapest: subtract published rasters

```python
# Once per AOI, with rasterio
import rasterio
import numpy as np

with rasterio.open("dsm_2m.tif") as dsm_src, rasterio.open("dtm_2m.tif") as dtm_src:
    dsm = dsm_src.read(1).astype("float32")
    dtm = dtm_src.read(1).astype("float32")
    profile = dsm_src.profile

chm = dsm - dtm
chm[chm < 0] = 0  # clean up small negative noise around buildings/edges

profile.update(dtype="float32", nodata=-9999)
with rasterio.open("chm_2m.tif", "w", **profile) as dst:
    dst.write(chm, 1)
```
Two seconds of work. Downside: 2 m resolution may smooth out small crowns.

### Option B — From raw LAZ via PDAL (if you need 0.5 m)

PDAL pipeline (`las_to_chm.json`):
```json
{
  "pipeline": [
    "tile.laz",
    { "type": "filters.range", "limits": "Classification[2:2]" },
    { "type": "writers.gdal", "filename": "dtm.tif",
      "resolution": 0.5, "output_type": "idw" },
    "tile.laz",
    { "type": "filters.range", "limits": "ReturnNumber[1:1]" },
    { "type": "writers.gdal", "filename": "dsm.tif",
      "resolution": 0.5, "output_type": "max" }
  ]
}
```
Run: `pdal pipeline las_to_chm.json`, then `gdal_calc.py -A dsm.tif -B dtm.tif --calc="A-B" --outfile=chm.tif`.

Python equivalents:
- `pdal` (Python bindings) — drives the pipeline
- `laspy` — pure-Python LAS/LAZ reader (with `lazrs` backend)
- `pyfor` — `tile.normalize(1); chm = tile.chm(0.5)` one-liner
- `lidR` (R) — most mature ecosystem, but adds an R dependency

For a Python-only stack, **PDAL pipeline + rasterio** is the industry
standard. `laspy` alone is sufficient if you want full control without
PDAL's C++ install footprint.

---

## 4. Querying CHM from a Detection Bounding Box

Detections are in pixel coords inside an EPSG:25831 patch. After
georeferencing (`georef.py`) each bbox is `(minx, miny, maxx, maxy)` in
metres, EPSG:25831 — same CRS as the CHM. No reprojection needed.

### Recommended approach: rioxarray + clip_box (vectorised, fast)

```python
# src/heights.py — height extraction from CHM for a list of bboxes
import rioxarray
import numpy as np
from typing import Sequence

def load_chm(path: str):
    """Load CHM raster lazily; CRS must be EPSG:25831 to match detections."""
    chm = rioxarray.open_rasterio(path, masked=True).squeeze()
    assert chm.rio.crs.to_epsg() == 25831, "CHM must be in EPSG:25831"
    return chm

def bbox_height_stats(chm, bbox: tuple[float, float, float, float]) -> dict:
    """
    Compute height statistics inside a single bbox.
    bbox: (minx, miny, maxx, maxy) in EPSG:25831 metres.
    Returns mean / max / p95 / pixel count.
    """
    minx, miny, maxx, maxy = bbox
    # clip_box is memory-efficient: only loads the windowed region
    clip = chm.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    arr = clip.values
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "max": np.nan, "p95": np.nan, "n": 0}
    return {
        "mean": float(arr.mean()),
        "max":  float(arr.max()),
        "p95":  float(np.percentile(arr, 95)),
        "n":    int(arr.size),
    }

def filter_by_height(detections, chm_path: str, min_height_m: float = 5.0):
    """
    Drop detections whose max canopy height is below min_height_m.
    Mirrors the >5 m real-tree threshold.
    """
    chm = load_chm(chm_path)
    keep = []
    for det in detections:
        stats = bbox_height_stats(chm, det.bbox_25831)
        det.height_max = stats["max"]
        det.height_mean = stats["mean"]
        if stats["max"] >= min_height_m:
            keep.append(det)
    return keep
```

Key library notes:
- **`rioxarray.clip_box`** is the right primitive: it does a windowed read so
  you never materialise the full raster. The bbox CRS must match the raster
  CRS (both EPSG:25831, so no `crs=` arg needed).
- For tens of thousands of bboxes, switch to `rasterio.windows.from_bounds`
  + a single full-array read (faster than per-bbox `clip_box`).
- Use `max` (or p95 to be robust to noise) — *not* mean — for the
  shrub-vs-tree decision. A 12 m tree's bbox can include a lot of ground
  pixels around the crown that pull the mean down.

### Why p95 over max?
LiDAR DSM has occasional spikes (birds, processing artifacts). p95 of the
clipped patch is a more robust "tree top height" metric. lidR uses the same
idea in `locate_trees()` (default min height = 2 m).

---

## 5. Prior Art — RF-DETR / DeepForest + LiDAR Fusion

Direct precedents:

1. **DeepForest** (Weinstein et al.) — the canonical RGB tree-crown detector.
   It does not ship a built-in LiDAR fusion module, but the documented
   workflow in the literature is exactly the one we're implementing:
   detect on RGB, validate/filter with CHM. See Weinstein 2019, Plant Methods.

2. **lidR** (R, r-lidar/lidR) — gold-standard for LiDAR forestry. The
   `locate_trees()` function uses Local Maxima Filtering on a CHM with a
   default `hmin = 2 m`. Workflow used in dozens of papers:
   `readLAS → classify_ground → normalize_height → grid_canopy → locate_trees`.
   Worth reading even if you stay in Python — algorithms transfer 1:1.

3. **Fusing aerial photographs and airborne LiDAR for individual tree
   detection in urban/peri-urban areas** (ScienceDirect, 2025) — RetinaNet
   on RGB combined with LiDAR-derived metrics; reports significant precision
   gains over RGB-only and LiDAR-only baselines. Same architecture family
   as RF-DETR. → strong validation that the fusion approach works.

4. **Single Tree Detection in Forest Areas with High-Density Lidar Data**
   (Eysn et al.) — benchmark of LiDAR-only ITD methods; uses 2 m as the
   minimum-tree threshold throughout.

5. **Shrub height estimation with UAV LiDAR** (Iberian Peninsula, 2024,
   T&F) — argues that filtering returns below 2 m removes shrubs/rocks but
   that shrubs *can* reach 2–5 m, supporting our **5 m threshold** for
   "real tree" rather than 2 m.

6. **OpenTopography 3DEP CHM workflow notebook**
   (`OpenTopography/OT_3DEP_Workflows`, notebook 05) — production-quality
   PDAL → CHM Python pipeline you can copy-paste with minimal edits.

7. **lidar2dems** (`applied-geosolutions/lidar2dems`) — Python wrapper around
   PDAL specifically for DTM/DSM/CHM generation. Useful if you want to skip
   writing the JSON pipeline yourself.

There is **no published RF-DETR + LiDAR fusion repo** as of April 2026 —
this is mildly novel territory, but the underlying technique (RGB-DL
detector + CHM filtering) is standard.

---

## 6. Practical Constraints & Sizing

**Storage** (Montseny full coverage):
- LAZ point cloud: ~270 GB worst case (~600 tiles × 450 MB) — *too much*
- 2 m DSM + DTM rasters via WCS, AOI-clipped: **a few hundred MB**
- 2 m CHM derived: similar

**Processing time** (rough):
- WCS fetch of ~30 km × 25 km AOI at 2 m: **seconds–minutes** (one HTTP call)
- DSM−DTM raster math: **<10 s** for the whole park
- Per-bbox `clip_box` height extraction: **~1 ms per bbox** → 10 k bboxes
  in ~10 s. For >100 k bboxes use a single windowed read + numpy indexing.
- LAZ→CHM with PDAL on a 1 km tile (8 pts/m²): **30–90 s per tile** on a
  laptop, dominated by IDW interpolation for the DTM.

**Memory**:
- 2 m raster of full Montseny (~30 km × 25 km = 15000 × 12500 px) at float32:
  **~720 MB**. Fits easily; no tiling needed for the rasters.
- A single 1 km LAZ tile loaded with `laspy` peaks at **2–4 GB RAM**
  (8–10 M points × XYZ + classification). Stream with PDAL or process
  tile-by-tile to keep below 8 GB.

---

## 7. Concrete Recommendation for Forest Pulse

**MVP path (1 day of work)**:

1. Add `src/forest_pulse/lidar.py` with two functions:
   - `fetch_chm_for_aoi(bbox_25831) -> Path` — calls ICGC WCS twice
     (DSM, DTM at 2 m), subtracts, writes one CHM GeoTIFF cached on disk.
   - `filter_detections_by_height(detections, chm_path, min_h=5.0)` — uses
     the rioxarray snippet above.
2. Wire it into the pipeline between `detect.py` and `health.py`:
   `detect → georef (already gives EPSG:25831 bboxes) → lidar filter → health`.
3. Add `--min-tree-height` CLI flag (default 5.0 m) and a `--no-lidar`
   bypass for environments without internet to ICGC.
4. Cache CHMs in `data/lidar/chm_<aoi_hash>.tif` — never re-download.
5. Tests: one fixture CHM (small synthetic GeoTIFF) + a handful of bboxes
   covering "tall tree", "shrub", "edge of raster", "bbox crossing nodata".

**Dependencies to add to `pyproject.toml`**:
- `rioxarray>=0.15`
- `rasterio>=1.3` (already present transitively)
- (optional) `owslib>=0.30` to drive WCS GetCoverage cleanly

**Skip for MVP**: PDAL, laspy, point-cloud download. Only revisit if the
2 m CHM is shown to be too coarse (e.g. losing real small trees in dense
canopy edges).

---

## 8. Sources

- ICGC Territorial LiDAR product page:
  http://www.icgc.cat/en/Geoinformation-and-Maps/Data-and-products/Elevations/Territorial-elevations/Territorial-LiDAR
- Geonetwork metadata, v3r1 2021–2023:
  https://catalegs.ide.cat/geonetwork/srv/api/records/lidar-territorial-v3r1-2021-2023?language=eng
- LAZ bulk download index:
  https://datacloud.icgc.cat/datacloud/lidar-territorial/laz_unzip/
- WMS/WCS Elevations hub:
  https://www.icgc.cat/en/Geoinformation-and-Maps/Online-services-Geoservices/WMS-and-WCS-Elevations
- WMS endpoint for elevation maps:
  https://geoserveis.icgc.cat/icgc_mdt2m/wms/service?
- Download viewer (interactive AOI selector):
  https://www.icgc.cat/en/Tools-and-viewers/Viewers/Download-viewer
- Visors AppDownloads:
  https://visors.icgc.cat/appdownloads/
- ICGC press release on 3rd LiDAR coverage:
  https://www.icgc.cat/en/ICGC/News/We-have-completed-third-LiDAR-coverage-Catalonia
- DeepForest GitHub: https://github.com/weecology/DeepForest
- lidR book — Individual Tree Detection: https://r-lidar.github.io/lidRbook/itd.html
- lidR `locate_trees`: https://rdrr.io/cran/lidR/man/locate_trees.html
- OpenTopography 3DEP CHM notebook:
  https://github.com/OpenTopography/OT_3DEP_Workflows/blob/main/notebooks/05_3DEP_Generate_Canopy_Height_Models_User_AOI.ipynb
- lidar2dems: https://applied-geosolutions.github.io/lidar2dems/
- PDAL tutorial (DTM creation): https://paulojraposo.github.io/pages/PDAL_tutorial.html
- pyfor CHM docs: https://pyfor-pdal-u.readthedocs.io/en/latest/topics/canopyheightmodel.html
- rioxarray clip_box example: https://corteva.github.io/rioxarray/stable/examples/clip_box.html
- Fusing aerial RGB + LiDAR for ITD (2025):
  https://www.sciencedirect.com/science/article/pii/S1618866725000305
- Shrub height with UAV LiDAR (Iberian Peninsula, 2024):
  https://www.tandfonline.com/doi/full/10.1080/22797254.2024.2438626
- LiDAR fusion review (2024):
  https://link.springer.com/article/10.1007/s40725-024-00223-7
- EPSG:25831 reference: https://epsg.org/crs_25831/ETRS89-UTM-zone-31N.html
