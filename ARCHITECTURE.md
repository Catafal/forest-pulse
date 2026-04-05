# Architecture — Forest Pulse

## System Overview

```
                    ┌─────────────────────────────────────────┐
                    │         AERIAL IMAGERY INPUT            │
                    │  (drone / ICGC / PNOA / OpenAerialMap)  │
                    └──────────────┬──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       detect.py           │
                    │  RF-DETR (DINOv2 backbone) │
                    │  Input: RGB image          │
                    │  Output: sv.Detections     │
                    └──────────┬───────────────┘
                               │
                               ▼
                    ┌──────────────────────────┐
                    │       health.py           │
                    │  GRVI + ExG per crop      │
                    │  Input: image + detections │
                    │  Output: list[HealthScore] │
                    └──────────┬───────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                                 ▼
    ┌──────────────────┐              ┌──────────────┐
    │  visualize.py    │              │  georef.py   │
    │  Supervision     │              │  pixel→GPS   │
    │  bbox + health   │              │  EXIF/CRS    │
    │  color coding    │              │  → GeoDataFrame
    └──────────────────┘              └──────┬───────┘
    (annotated image)                        │
                                             ▼
                                  ┌───────────────────┐
                                  │    export.py       │
                                  │  GeoJSON / SHP     │
                                  │  CSV / HTML report │
                                  └─────────┬─────────┘
                                            │
                               ┌────────────┴────────────┐
                               ▼                         ▼
                     ┌─────────────────┐      ┌──────────────────┐
                     │  Single-image   │      │   temporal.py     │
                     │  output (map)   │      │   Compare t₀, t₁  │
                     └─────────────────┘      │   Change report   │
                                              └──────────────────┘
```

**Dependency order:** detect → health → (visualize | georef). Visualize needs health
scores for color coding. Georef can run after health (attaches scores to GeoDataFrame).
Both visualize and georef depend on health completing first.

## Module Contracts

### detect.py

```python
def detect_trees(
    image: np.ndarray | str,        # RGB array or path to image
    model_name: str = "rfdetr-base", # model identifier
    confidence: float = 0.3,         # detection confidence threshold
) -> sv.Detections:
    """Detect individual tree crowns in aerial RGB imagery."""
```

### health.py

```python
def score_health(
    image: np.ndarray,               # full RGB image
    detections: sv.Detections,       # detected tree bboxes
) -> list[HealthScore]:
    """Compute RGB health indices for each detected tree crown."""

@dataclass
class HealthScore:
    tree_id: int
    grvi: float          # Green-Red Vegetation Index: (G-R)/(G+R)
    exg: float           # Excess Green Index: 2G - R - B
    label: str           # "healthy" | "stressed" | "dead"
    confidence: float    # 0.0 - 1.0
```

### georef.py

```python
def georeference(
    detections: sv.Detections,       # pixel-space detections
    image_path: str,                 # path to image (for EXIF)
    crs: str | None = None,         # override CRS (e.g., "EPSG:25831")
) -> gpd.GeoDataFrame:
    """Convert pixel bboxes to GPS-referenced GeoDataFrame."""
```

### temporal.py

```python
def compare_periods(
    gdf_before: gpd.GeoDataFrame,    # trees at t₀
    gdf_after: gpd.GeoDataFrame,     # trees at t₁
    match_tolerance_m: float = 2.0,  # GPS matching radius in meters
) -> ChangeReport:
    """Spatial-match trees across time periods, compute changes."""

@dataclass
class ChangeReport:
    date_before: str
    date_after: str
    trees_before: int
    trees_after: int
    matched: list[TreeMatch]
    missing: list[int]               # tree IDs in t₀ not found at t₁
    new: list[int]                   # tree IDs in t₁ not found at t₀
```

### export.py

```python
def to_geojson(gdf: gpd.GeoDataFrame, output_path: str) -> str:
    """Export georeferenced tree inventory as GeoJSON."""

def to_shapefile(gdf: gpd.GeoDataFrame, output_path: str) -> str:
    """Export as ESRI Shapefile for ArcGIS compatibility."""

def to_report(change: ChangeReport, output_path: str) -> str:
    """Generate HTML change report with summary statistics."""
```

### visualize.py

```python
def annotate_trees(
    image: np.ndarray,
    detections: sv.Detections,
    health_scores: list[HealthScore] | None = None,
) -> np.ndarray:
    """Draw bounding boxes with health color coding using Supervision."""
```

## Data Flow

```
1. User provides: image path (local file or URL)
2. detect.py loads model, runs inference → sv.Detections
3. health.py crops each bbox, computes GRVI/ExG → list[HealthScore]
4. FORK:
   a. visualize.py draws annotated image (needs detections + health) → saves PNG
   b. georef.py reads EXIF/CRS, attaches health scores → gpd.GeoDataFrame
5. export.py writes GeoJSON from GeoDataFrame → QGIS-ready output
6. (optional) temporal.py compares two GeoDataFrames → ChangeReport → HTML report
```

**Critical dependency:** Steps 4a and 4b both require health.py (step 3) to complete first.
Visualize needs health scores for color coding. Georef attaches health scores to the GeoDataFrame.

## Auto-Research Harness (Fine-Tuning Loop)

```
autoresearch/
├── eval.py      # LOCKED — mAP50 on validation set (NEVER edit during runs)
├── train.py     # EDITABLE — backbone, lr, augmentations, batch_size
├── program.md   # Agent instructions — defines the experiment loop
└── results.tsv  # UNTRACKED — full experiment log (survives git resets)

Loop: edit train.py → run → measure mAP50 → keep if improved → repeat
```

See `autoresearch/README.md` for full details.

## Integration Specs

### HuggingFace Datasets (OAM-TCD, SelvaBox)
- **SDK:** `datasets` >= 2.18.0 (HuggingFace Hub)
- **Auth:** None required (public datasets)
- **Access pattern:** `load_dataset("restor/tcd")` downloads Parquet + images automatically
- **Caching:** HuggingFace caches to `~/.cache/huggingface/datasets/` — first download is slow (~14 GB for OAM-TCD), subsequent loads are instant
- **Streaming:** `load_dataset(..., streaming=True)` for development (no full download)
- **Rate limits:** None for public datasets, but large downloads may time out on slow connections
- **Error handling:** Retry on `ConnectionError`; if `disk full` → clear cache or use streaming

### DeepForest Model Hub
- **SDK:** `deepforest` >= 2.0.0
- **Auth:** None required
- **Access pattern:** `model.load_model("weecology/deepforest-tree")` downloads from HuggingFace
- **Model size:** ~170 MB (RetinaNet + ResNet-50 weights)
- **Error handling:** If download fails, model falls back to local checkpoint if available

### ICGC Catalunya (Orthophotos)
- **Access:** Direct HTTP download from icgc.cat portal
- **Auth:** None required (open data, attribution required)
- **Format:** GeoTIFF with CRS metadata (usually EPSG:25831 — ETRS89 / UTM zone 31N)
- **Tile sizes:** 2-4 GB per standard tile — must be subdivided before inference
- **Error handling:** Large downloads may timeout; use `wget --continue` for resumable download
- **QGIS plugin:** `OpenICGC` for interactive tile selection

### Google Earth Engine (PNOA 10cm)
- **SDK:** `earthengine-api` (Python) or `ee` (JavaScript)
- **Auth:** Required — `ee.Authenticate()` + Google Cloud project
- **Dataset ID:** `Spain/PNOA/PNOA10`
- **Access pattern:** Define region of interest → export as GeoTIFF to Google Drive
- **Rate limits:** Free tier has export quotas; paid tier removes limits
- **Error handling:** Auth failures → re-run `ee.Authenticate()`; export quota → wait or use paid tier

---

## Architecture Decision Records (ADRs)

### ADR-1: RF-DETR over DeepForest as primary detection model

**Context:** DeepForest (RetinaNet, 2017) is the dominant library for tree crown detection in ecology. RF-DETR (DINOv2 backbone, ICLR 2026) is SOTA for object detection.

**Decision:** Use RF-DETR for production detection. Use DeepForest only as a one-time weak label bootstrapper for initial annotation.

**Alternatives rejected:**
- Fork/improve DeepForest: 80 open issues, RetinaNet backbone is fundamentally outdated (64% precision vs 90%+ for modern transformers). Swapping the backbone inside DeepForest = effectively rewriting it.
- YOLO11: Strong alternative, but RF-DETR's DINOv2 backbone has better contextual understanding for dense same-class objects (tree crowns from above).

**Consequences:** We lose DeepForest's ecology-specific utilities (tiling, NEON integration). We gain 26+ percentage points of precision and a cleaner fine-tuning API.

### ADR-2: Heuristic health scoring (Phase 1) before trained classifier (Phase 3)

**Context:** Health scoring can use simple RGB vegetation indices (GRVI, ExG) with heuristic thresholds, or a trained classifier on labeled data.

**Decision:** Start with heuristic thresholds, then replace with trained MobileNetV3 in Phase 3.

**Rationale:** Heuristics give immediate results with zero training data. The Swedish Forest Damages dataset (102K bboxes + damage labels) enables the trained classifier in Phase 3, but waiting for Phase 3 to have any health output would block the MVP demo.

**Consequences:** Phase 1 health scores will be less accurate (~60-70% estimated) than Phase 3 trained classifier (~80%+ target). Heuristic thresholds must be tunable per forest type.

### ADR-3: COCO format as intermediate annotation standard

**Context:** Multiple datasets use different annotation formats (COCO JSON, PascalVOC XML, custom CSV).

**Decision:** Use COCO bounding box format as the canonical intermediate representation.

**Rationale:** RF-DETR expects COCO format. OAM-TCD and BAMFORESTS already provide COCO annotations. Supervision can convert to/from COCO. One format reduces conversion bugs.

**Consequences:** Datasets in other formats (Swedish Forest Damages) need conversion scripts.
