---
title: "Forest Pulse — Engineering Decisions Index"
project: forest-pulse
date: 2026-04-07
version: 1.0
---

# Engineering Decisions — Index

Academic documentation of all engineering decisions made during the development of Forest Pulse. Each decision includes context, alternatives considered, rationale, and consequences. Suitable for academic review, technical due diligence, or onboarding new contributors.

## Document Structure

| Document | Scope | Decisions |
|----------|-------|-----------|
| [ARCH — Model & Architecture](ARCH-model-architecture.md) | Detection model, backbone, segmentation, classifier design | ARCH-001 to ARCH-009 |
| [DATA — Data Pipeline & Dataset](DATA-pipeline-dataset.md) | Data sources, geographic focus, tiling, annotations, filters | DATA-001 to DATA-008 |
| [EVAL — Evaluation Strategy](EVAL-evaluation-strategy.md) | Metrics, honest evaluation, label quality strategy | EVAL-001 to EVAL-006 |
| [TRAIN — Training Strategy](TRAIN-training-strategy.md) | Self-training, hyperparameters, LiDAR extraction, device support | TRAIN-001 to TRAIN-007 |
| [SYS — System Design](SYS-system-design.md) | Module architecture, tool selection, GIS output, harness design | SYS-001 to SYS-007 |

---

## All Decisions at a Glance

### Model & Architecture (ARCH)

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| ARCH-001 | RF-DETR as primary detection model | Accepted | 1–2 |
| ARCH-002 | DINOv2 backbone | Accepted | 2 |
| ARCH-003 | SAM2 hybrid segmentation (3 modes) | Accepted | 6 |
| ARCH-004 | Heuristic health scoring → trained classifier | Superseded | 1→3 |
| ARCH-005 | LiDAR-distilled RGB classifier design | Experimental | 9–9.5 |
| ARCH-006 | Two-stage "detect once, classify many times" | Experimental | 9 |
| ARCH-007 | DeepForest as bootstrapper only (not runtime) | Accepted | 1→3 |
| ARCH-008 | COCO format as annotation standard | Accepted | 2 |
| ARCH-009 | Module-level caching + auto device detection | Accepted | 1–6 |

### Data Pipeline & Dataset (DATA)

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| DATA-001 | Pivot from global OAM-TCD to Catalan ICGC data | Accepted | 2 |
| DATA-002 | ICGC as primary imagery source | Accepted | 2 |
| DATA-003 | Montseny National Park as geographic focus | Accepted | 3 |
| DATA-004 | Expansion to 8 ICGC zones | Accepted | 4 |
| DATA-005 | DeepForest bootstrap annotations (10,376 labels) | Accepted | 3 |
| DATA-006 | 640×640 px patch-based tiling strategy | Accepted | 3 |
| DATA-007 | ICGC LiDAR LAZ integration for physical verification | Accepted | 7 |
| DATA-008 | NDVI filtering for non-vegetation false positives | Accepted | 5 |

### Evaluation Strategy (EVAL)

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| EVAL-001 | LiDAR-verified metric as the "honest" evaluation standard | Accepted | 8 |
| EVAL-002 | eval.py locked during auto-research harness runs | Accepted | 2+ |
| EVAL-003 | Gold evaluation set (20 human-annotated patches) | Accepted (Superseded) | 4 |
| EVAL-004 | mAP50 as primary detection metric | Accepted | 2+ |
| EVAL-005 | Multi-round self-training annotation strategy (3 rounds) | Accepted | 3–4 |
| EVAL-006 | Bootstrap with weak labels before manual annotation | Accepted | 3 |

### Training Strategy & ML Pipeline (TRAIN)

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| TRAIN-001 | Self-training loop for iterative label refinement | Accepted | 3–4 |
| TRAIN-002 | Confidence threshold tuning (0.7 → 0.5) | Accepted | 4 |
| TRAIN-003 | RF-DETR fine-tuning hyperparameter choices | Accepted | 2 |
| TRAIN-004 | Cross-platform device detection (CUDA/MPS/CPU) | Accepted | 1+ |
| TRAIN-005 | Training on weak bootstrap labels first | Accepted | 3 |
| TRAIN-006 | LiDAR feature extraction — 7 per-tree features | Accepted | 7 |
| TRAIN-007 | CHM generation from LAZ files | Accepted | 7 |

### System Design & Pipeline Architecture (SYS)

| ID | Title | Status | Phase |
|----|-------|--------|-------|
| SYS-001 | Module separation (detect/health/georef/temporal/export/visualize) | Accepted | 1 |
| SYS-002 | Supervision library as visualization/annotation backbone | Accepted | 1 |
| SYS-003 | GeoJSON as primary GIS output format | Accepted | 4–5 |
| SYS-004 | Offline patch-based inference strategy | Accepted | 1 |
| SYS-005 | autoresearch/ as harness-controlled training environment | Accepted | 2 |
| SYS-006 | Phase-based MVP approach (P1 → P9) | Accepted | all |
| SYS-007 | Apache 2.0 license constraint (RF-DETR) | Accepted | 1 |

---

## Key Cross-Cutting Themes

### 1. Honesty in Evaluation
The most critical meta-decision: never evaluate against training signal. mAP50=0.904 on self-labels masked that real recall was ~30%. EVAL-001 (LiDAR-verified metric) broke the feedback loop. See `progress.txt` for the explicit acknowledgment.

### 2. Geographic Specialization Over Global Generalization
DATA-001 through DATA-004 document the deliberate pivot from global data to Catalan-specific ICGC imagery. A smaller, locally-representative dataset proved more useful than a large misaligned global dataset.

### 3. Weak Labels + Self-Training as a Viable Path
TRAIN-001, TRAIN-002, and EVAL-006 document that weak labels (60% precision) + 3 rounds of self-training + confidence filtering can achieve 85%+ precision without any manual annotation. Total cost: €0, ~4 hours compute.

### 4. Two-Stage "Detect Once, Classify Many Times"
ARCH-006 and TRAIN-003 establish the core architectural principle: freeze the detector (RF-DETR), iterate the classifier (scikit-learn GradientBoosting). Detector re-training = 20 min; classifier re-training = 5 min.

### 5. Multimodal Physical Verification
DATA-007, DATA-008, ARCH-005: RGB detection alone cannot distinguish trees from shrubs. LiDAR height (physical) + NDVI (spectral) + RGB (visual) form a trimodal verification gate. Each layer removes a distinct false positive category.

---

## Related Documentation

- `ARCHITECTURE.md` — pipeline diagram, module contracts, ADRs (source of truth for function signatures)
- `TECH_STACK.md` — exact dependency versions
- `IMPLEMENTATION_PLAN.md` — phased task breakdown
- `progress.txt` — current project state and key learnings
- `PRD.md` — product requirements and success criteria

---

## How to Use This Documentation

**For academic review:** Each decision includes "Options Considered" tables, explicit rationale, and quantified consequences where available. Start with ARCH-001 and follow the dependency chain.

**For new contributors:** Read SYS-001 (module architecture), then ARCH-001 (why RF-DETR), then EVAL-001 (why the inflated mAP50 is not the true metric).

**For reproducing results:** Key commits listed per decision. `progress.txt` tracks current state. `data/montseny/` contains committed patch files and annotation rounds.

**For extending the project:** Deferred decisions sections in each document list explicitly what was not done and why — these are the natural starting points for future work.
