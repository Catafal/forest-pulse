---
title: "Forest Pulse — Engineering Decisions: System Design & Pipeline Architecture"
project: forest-pulse
scope: Module architecture, tool selection, GIS output, training infrastructure, licensing
date: 2026-04-07
version: 1.0
status: Academic Documentation
phases_documented: "1–9"
author: Jordi Catafal
---

# Engineering Decisions — System Design & Pipeline Architecture

Decisions governing how Forest Pulse is structured as a software system: module boundaries, tool choices, output formats, and the training harness design.

---

## SYS-001: Module Separation Architecture

**Status:** Accepted (foundational)  
**Date:** Phase 1 design  
**Files:** `src/forest_pulse/{detect,health,georef,temporal,export,visualize}.py`

### Context

The pipeline has multiple stages: object detection, health assessment, georeferencing, temporal comparison, and export. A monolithic pipeline would create tight coupling, making it hard to:
- Swap detection models (DeepForest → RF-DETR)
- Replace health classifiers (heuristic → learned)
- Test individual components in isolation
- Parallelize processing in future

### Options Considered

| Approach | Coupling | Testability | Swappability | Chosen |
|----------|---------|-------------|-------------|--------|
| Monolithic single file | High | Low | Low | No |
| **Functional modules (chosen)** | Low | High | High | **Yes** |
| Microservices | Very low | High | High | Overkill for MVP |

### Decision

Six single-responsibility modules, each with a clear public API:

| Module | Input | Output | Does NOT do |
|--------|-------|--------|------------|
| `detect.py` | Image path or array | `sv.Detections` (bboxes + confidence) | Train, visualize |
| `health.py` | Image + `sv.Detections` | Health labels + scores per detection | Detect, export |
| `visualize.py` | Image + `sv.Detections` + health labels | Annotated PIL image | Process data, save GIS |
| `georef.py` | Pixel coords + GeoTIFF metadata | GPS coords per detection | Detect, classify |
| `temporal.py` | Two sets of `sv.Detections` + timestamps | Change diff DataFrame | Detect, export |
| `export.py` | `sv.Detections` + GPS coords | GeoJSON/Shapefile/CSV | Process, detect |

### Rationale

- **Single responsibility:** each module has one job, one failure mode, one test surface.
- **Stable interfaces:** `sv.Detections` as the canonical detection container — Supervision library manages this type, not us.
- **Swappability:** changing from DeepForest to RF-DETR required changing only `detect.py`, not health, georef, or export.
- **Progressive enhancement:** Phase 1 = detect + health + visualize. Phases 3–5 add georef + temporal + export without modifying the Phase 1 modules.

### Consequences

- ARCHITECTURE.md contracts are the source of truth for function signatures — modules cannot deviate.
- `detect.py` has no knowledge of GPS or health; it only returns bboxes. Downstream modules cannot assume geographic context.
- New functionality (LiDAR, SAM2, classifier) added as new modules (`lidar.py`, `segment.py`, `classifier.py`), not by growing existing modules.

---

## SYS-002: Supervision Library as Visualization/Annotation Backbone

**Status:** Accepted  
**Date:** Phase 1  
**Files:** All modules use `supervision` (imported as `sv`)

### Context

Visualization of object detection outputs (bounding boxes, health-colored overlays, masks) is a solved problem. The question was whether to use a dedicated library or build custom with OpenCV/matplotlib.

### Options Considered

| Library | Strengths | Weaknesses | Chosen |
|---------|-----------|-----------|--------|
| Raw OpenCV | Maximum control | Verbose; no tracking utilities | No |
| matplotlib | Familiar API | Slow; not designed for real-time annotation | No |
| **Supervision (Roboflow) ≥ 0.25.0** | sv.Detections container, built-in annotators, tracking, COCO tools | External dependency | **Yes** |

### Decision

Use **Supervision ≥ 0.25.0** (`roboflow/supervision`) throughout:
- `sv.Detections` as the canonical detection type (bbox array + confidence + class_id)
- `sv.BoxAnnotator` and `sv.LabelAnnotator` for visualization
- `supervision.metrics.MeanAveragePrecision` for evaluation
- `sv.COCO` utilities for dataset format conversion

### Rationale

- **`sv.Detections` as universal exchange format:** every module receives and returns `sv.Detections` — standardizes the pipeline without custom types.
- **Built-in annotators:** `BoxAnnotator` with health-color mapping replaces 50 lines of custom OpenCV.
- **mAP metric utilities:** no need to implement COCO evaluation from scratch.
- **Active maintenance by Roboflow:** well-documented, tested, compatible with RF-DETR and SAM2.

### Consequences

- **Critical dependency:** a Supervision API change could break the entire pipeline. Pinned to `>= 0.25.0` in pyproject.toml.
- **DeepForest integration wrinkle:** DeepForest's `predict_image()` returns a pandas DataFrame, not `sv.Detections` — must convert in detect.py. Documented as a known gotcha in CLAUDE.md.
- **Type safety:** functions typed as `-> sv.Detections` prevent returning wrong types.

---

## SYS-003: GeoJSON as Primary GIS Output Format

**Status:** Accepted  
**Date:** Phase 4–5  
**File:** `src/forest_pulse/export.py`

### Context

Forest Pulse exports detected trees with GPS coordinates for use in GIS applications. Multiple output formats exist: GeoJSON, Shapefile (.shp), KML, CSV, PostGIS SQL.

### Options Considered

| Format | Tooling Support | Complexity | Open | Chosen |
|--------|----------------|-----------|------|--------|
| **GeoJSON (primary)** | QGIS, Leaflet, Mapbox, PostGIS | Simple (JSON) | Yes | **Yes** |
| Shapefile | ArcGIS, QGIS (legacy) | Complex (4 files) | Partially | Secondary |
| KML | Google Earth | Limited GIS tools | Yes | No |
| CSV (lon/lat/confidence) | Any spreadsheet | No geometry | Yes | Supplementary |
| PostGIS SQL | PostgreSQL only | Infrastructure required | Yes | Deferred |

### Decision

**GeoJSON** as the primary output format. Also support Shapefile and CSV as secondary exports.

### Rationale

- **GeoJSON is the web GIS standard:** Leaflet, Mapbox, Maplibre, QGIS, and PostGIS all ingest it natively.
- **Single file:** unlike Shapefiles (4+ files: .shp, .dbf, .shx, .prj), GeoJSON is one JSON file — easy to version control, email, open in browser.
- **Properties flexibility:** each feature can carry arbitrary metadata (confidence, health label, LiDAR features) — not limited to fixed Shapefile column widths.
- **Human-readable:** engineers can inspect a GeoJSON in any text editor.
- **GeoPandas native:** `geopandas.GeoDataFrame.to_file("out.geojson", driver="GeoJSON")` is one line.

### Consequences

- Shapefile export still available for ArcGIS users (legacy workflows in Spanish forestry agencies).
- Large exports (1000+ trees) create large GeoJSON files. For production with millions of trees, PostGIS ingest is the next step.

---

## SYS-004: Patch-Based Inference Strategy

**Status:** Accepted  
**Date:** Phase 1  
**Files:** `detect.py` (inference loop), `scripts/tile_orthophoto.py` (preprocessing)

### Context

ICGC GeoTIFFs are 2–4 GB per tile (1 km²). Modern detection models expect fixed-size inputs (640×640 px). Two options: tile offline (during data preparation) or tile online (during inference).

### Decision

**Offline tiling during preprocessing** (via `scripts/tile_orthophoto.py`). Inference always receives pre-tiled 640×640 patches.

### Rationale

- **Reproducibility:** the same set of 800 tiles is used for every training run and every evaluation. No random crop variation.
- **Memory efficiency:** loading one 1.2 MB JPEG per inference call vs loading a 2 GB GeoTIFF.
- **GIS metadata:** each tile is saved with GPS coordinates in `patches_metadata.csv` — georeferencing is trivially computed from tile metadata.
- **Parallel inference:** tiles can be processed in parallel (independent files) without file handle contention.

### Consequences

- Edge tiles (smaller than 640×640 at zone boundaries) are dropped — ~5–10% edge loss. Acceptable.
- New areas require re-running tile preprocessing before inference.
- Production deployment would need online tiling for arbitrary new GeoTIFFs — `detect.py` inference loop already handles this with a sliding window option.

---

## SYS-005: autoresearch/ as Harness-Controlled Training Environment

**Status:** Accepted  
**Date:** Phase 2  
**Files:** `autoresearch/train.py`, `autoresearch/eval.py`

### Context

Phase 11 requires an agentic loop that modifies training hyperparameters overnight and measures results against a locked evaluation module. Where should this live? Options: Jupyter notebook, scripts/, autoresearch/.

### Decision

Create a dedicated `autoresearch/` directory with a strict contract:
- `train.py`: the **only file the agent may modify** (contains BACKBONE, LEARNING_RATE, BATCH_SIZE, FINE_TUNE_EPOCHS)
- `eval.py`: **LOCKED** — never modified during harness runs (see EVAL-002)
- `eval_gold.py`: human-verified gold set measurement (sanity check, not optimization target)
- `eval_lidar.py`: LiDAR-verified measurement (PRIMARY optimization target for Phase 11)

### Rationale

- **Clear separation of concerns:** optimization surface (`train.py`) vs. measurement surface (`eval.py`).
- **Harness safety:** agent cannot accidentally modify evaluation code while editing training code.
- **Explicit contracts:** `eval.py` always outputs `val_map50: X.XXXX` to stdout — machine-readable, grep-able by harness.
- **No interactive notebooks:** Jupyter notebooks cannot be safely automated (kernel state, cell ordering, random seeds). Pure Python scripts can be run headlessly.

### Consequences

- `autoresearch/` is the only directory where AI agents have write permission during harness runs.
- All module code in `src/forest_pulse/` is read-only during harness runs.
- Pattern: `python autoresearch/train.py && python autoresearch/eval.py | grep val_map50` — harness pipeline.

---

## SYS-006: Phase-Based MVP Approach (P1 → P9)

**Status:** Accepted  
**Date:** Project inception  
**Files:** `IMPLEMENTATION_PLAN.md`, `progress.txt`

### Context

The full vision (detect + health + georef + temporal + LiDAR + classifier + species) could be attempted all at once or built incrementally. Which approach?

### Decision

Build in phases, each phase a working MVP:

| Phase | Deliverable | Key Decision |
|-------|------------|-------------|
| 1 | detect + health + visualize | RF-DETR baseline; heuristic health |
| 2 | RF-DETR fine-tuning pipeline | training infrastructure |
| 3 | Montseny data + weak labels | ICGC pivot; DeepForest bootstrap |
| 4 | Self-training loop | TRAIN-001, TRAIN-002 |
| 5 | Full pipeline (georef + temporal + export) | GeoJSON output |
| 6 | SAM2 hybrid segmentation | ARCH-003 |
| 7 | LiDAR feature extraction | TRAIN-006, TRAIN-007 |
| 8 | LiDAR-verified eval metric | EVAL-001 |
| 9 | Auto-labeled tree classifier | ARCH-005, ARCH-006 |

### Rationale

- **Each phase ships something usable.** A stakeholder can use Phase 1 output (annotated image with health colors) even if Phase 9 isn't done.
- **Fail fast on each phase:** if a phase's core assumption is wrong, fail early (e.g., GroundingDINO teacher model failed in Phase 3 — discovered in hours, not weeks).
- **Technical debt is minimized:** each phase is a clean module addition, not a patch on a tangled codebase.
- **Academic documentation:** each phase corresponds to a commit or series of commits, creating a clear paper trail.

### Consequences

- Phase boundaries sometimes require refactoring (e.g., health.py heuristic → trained classifier in Phase 3 means health.py changes).
- Progress.txt is the authoritative state — always read it before starting any new work.

---

## SYS-007: Apache 2.0 License Constraint (RF-DETR)

**Status:** Accepted (constraint)  
**Date:** Phase 1  
**Files:** `LICENSE`, `pyproject.toml`

### Context

RF-DETR is licensed under Apache 2.0. This is a permissive open-source license, but it carries attribution and patent grant requirements that shape how the project can be distributed.

### Implications

| Aspect | Apache 2.0 Requirement | Impact on Forest Pulse |
|--------|----------------------|----------------------|
| Commercial use | Permitted | Forest Pulse can be used commercially |
| Modification | Permitted, with attribution | Can fine-tune RF-DETR; must note changes |
| Distribution | Permitted, must include NOTICE file | Any distribution must carry Apache 2.0 license |
| Patent grant | Contributors grant patent license to users | Protects users from RF-DETR patent claims |
| Warranty | Explicitly disclaimed | Forest Pulse carries same warranty disclaimer |

### Decision

Forest Pulse is licensed under **Apache 2.0** to match RF-DETR. All other dependencies (Supervision, PyTorch, GeoPandas, SAM2) are compatible with Apache 2.0.

### Rationale

- Matching the most restrictive dependency (RF-DETR Apache 2.0) avoids license conflicts.
- Apache 2.0 is compatible with commercial use (client forestry applications).
- NOTICE file requirement fulfilled by including RF-DETR attribution in pyproject.toml.

---

## Summary Table

| Decision | Phase | Rationale Core | Key File |
|----------|-------|---------------|----------|
| SYS-001 | 1 | Single-responsibility modules, stable sv.Detections interface | ARCHITECTURE.md |
| SYS-002 | 1 | Supervision as canonical detection container + annotator | all modules |
| SYS-003 | 4–5 | GeoJSON = open, single-file, web-native | export.py |
| SYS-004 | 1 | Offline tiling = reproducibility + memory efficiency | tile_orthophoto.py |
| SYS-005 | 2 | autoresearch/ = clear boundary: editable train.py, locked eval.py | autoresearch/ |
| SYS-006 | all | Phase-based MVP: usable output at every phase | IMPLEMENTATION_PLAN.md |
| SYS-007 | 1 | Apache 2.0 matches RF-DETR, enables commercial use | LICENSE |

---

## Architectural Contract Summary

From ARCHITECTURE.md (source of truth for function signatures):

```
detect_trees(image_path, model_name, confidence) -> sv.Detections
score_health(image, detections) -> list[HealthScore]
georef_detections(detections, src_path) -> gpd.GeoDataFrame
compare_time_periods(detections_t1, detections_t2, match_distance_m) -> ChangeReport
export_to_geojson(geodataframe, output_path) -> Path
visualize_detections(image, detections, health_scores) -> PIL.Image
```

These contracts are immutable once published — downstream modules depend on them. Any change requires bumping the module version and updating ARCHITECTURE.md.
