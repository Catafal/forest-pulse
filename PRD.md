# Product Requirements Document — Forest Pulse

## What We're Building

An open-source Python pipeline that takes aerial imagery (drone footage or public orthophotos) and produces:

1. **Individual tree detection** — bounding boxes around each tree crown
2. **Health score per tree** — healthy / stressed / dead classification from RGB color indices
3. **Temporal change detection** — what changed between two time periods (tree loss, crown decline, health progression)
4. **GIS-ready output** — GeoJSON with GPS coordinates per tree, directly usable in QGIS/ArcGIS

## Who It's For

- Forestry professionals who need tree inventories without expensive field surveys
- Researchers studying forest health at scale
- Conservation organizations monitoring deforestation/degradation
- Anyone with aerial imagery of forest who wants actionable tree-level data

## What Success Looks Like

### MVP (Phase 1)
- Run detection on any aerial RGB image → bounding boxes around trees
- Colorful visualization (Supervision) with health color coding
- Works in Google Colab with zero setup

### Full Pipeline (Phase 2-5)
- mAP50 >= 0.75 on NeonTreeEvaluation benchmark after auto-research fine-tuning
- Health classification accuracy >= 70% on Swedish Forest Damages validation set
- GeoJSON export with GPS coordinates from georeferenced imagery
- Temporal diff report showing tree-level changes between two time periods

## What We're NOT Building
- A web application or SaaS product
- Real-time drone processing
- Species identification (out of scope for v1)
- LiDAR or multispectral processing (RGB-only for v1)

## Novel Contributions
1. **Auto-research harness for ecological CV** — Karpathy-style overnight fine-tuning optimization applied to tree detection. Nobody has published this.
2. **RGB-only health scoring** — GRVI + ExG indices on consumer drone/aerial RGB. Published work uses multispectral.
3. **End-to-end temporal change detection** — no open-source tool chains detection + health + GIS output across time periods.
