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
                          │  Output: list[BBox]        │
                          └──────────┬───────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
          ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
          │  health.py   │  │ visualize.py │  │  georef.py   │
          │  GRVI + ExG  │  │  Supervision │  │  pixel→GPS   │
          │  per crop    │  │  bbox + color│  │  EXIF/CRS    │
          └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
                 │                │                  │
                 └────────────────┼──────────────────┘
                                  │
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
def compare_flights(
    gdf_before: gpd.GeoDataFrame,    # trees at t₀
    gdf_after: gpd.GeoDataFrame,     # trees at t₁
    match_tolerance_m: float = 2.0,  # GPS matching radius in meters
) -> ChangeReport:
    """Spatial-match trees across time periods, compute changes."""

@dataclass
class ChangeReport:
    trees_before: int
    trees_after: int
    matched: list[TreeMatch]
    missing: list[TreeRecord]        # in t₀ but not t₁
    new: list[TreeRecord]            # in t₁ but not t₀
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
4. visualize.py draws annotated image → saves PNG/shows in notebook
5. georef.py reads EXIF/CRS → gpd.GeoDataFrame with GPS per tree
6. export.py writes GeoJSON → QGIS-ready output
7. (optional) temporal.py compares two GeoDataFrames → ChangeReport
```

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
