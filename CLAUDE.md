# Project: Forest Pulse

Aerial imagery pipeline for individual tree detection, health scoring, and temporal change monitoring. Built on RF-DETR + Supervision.

## Tech Stack
- Python 3.10+
- RF-DETR (DINOv2 backbone, Apache 2.0) — primary detection model
- Supervision >= 0.25.0 — visualization, annotation, tracking
- DeepForest >= 2.0.0 — label bootstrapping only (not a runtime dependency)
- PyTorch >= 2.2.0
- GeoPandas >= 1.0.0 — GIS output

## Pipeline
```
Aerial imagery (ICGC/PNOA/drone) → detect.py → health.py → georef.py → export.py
                                       ↓
                                  visualize.py (Supervision bboxes + health colors)
                                       ↓
                                  temporal.py (compare two time periods)
```

## Before Any Task
1. Read ARCHITECTURE.md (pipeline diagram + data flow)
2. Read TECH_STACK.md (exact versions)
3. Read IMPLEMENTATION_PLAN.md (phased build sequence)
4. Check progress.txt for current state

## Coding Conventions
- Max 200 lines per function
- Max 1000 lines per file
- All imports at top of file
- Type hints on all public functions
- Docstrings on all public functions (Google style)
- Comments that explain WHY, not WHAT

## Commands
```bash
# Install (development)
pip install -e ".[dev,train,notebooks]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Download sample data
python scripts/download_data.py --dataset oam-tcd --sample

# Run demo
python scripts/demo.py --image path/to/aerial.tif
```

## Module Responsibilities
| Module | Does | Does NOT |
|--------|------|----------|
| `detect.py` | Run detection model, return bboxes | Train models, visualize |
| `health.py` | Score each bbox crop for health | Detect trees, export files |
| `visualize.py` | Draw annotated images via Supervision | Process data, save GIS |
| `georef.py` | Convert pixel coords → GPS coords | Detect, classify, visualize |
| `temporal.py` | Match trees across time, compute diff | Detect new trees, export |
| `export.py` | Write GeoJSON/Shapefile/CSV | Process, detect, score |

## File References
- See PRD.md for what we're building, who it's for, success criteria
- See ARCHITECTURE.md for pipeline diagram, module contracts, ADRs
- See TECH_STACK.md for exact versions and dataset sources
- See IMPLEMENTATION_PLAN.md for phased task breakdown
- Check progress.txt for current state before starting any work

## Non-Negotiables
- `autoresearch/eval.py` is LOCKED — never modify during harness runs
- Never commit data/ or outputs/ directories
- Never hardcode file paths — use relative paths or CLI args
- All dataset downloads go through scripts/download_data.py
- Module contracts in ARCHITECTURE.md are the source of truth for function signatures

## Known Gotchas
- DeepForest's `predict_image()` returns a pandas DataFrame, not sv.Detections — must convert
- OAM-TCD on HuggingFace uses Parquet + TIFF — needs `datasets` library, not raw download
- ICGC orthophotos are large GeoTIFFs (2-4 GB per tile) — must tile before inference
- piexif cannot read GeoTIFF CRS — use rasterio or pyproj for orthophoto georeferencing
- Apple MPS (M-series) may not support all torch ops — test with CPU fallback

## Current Focus
See progress.txt for state.
