---
title: "Forest Pulse — Engineering Decisions: Model & Architecture"
project: forest-pulse
scope: Model selection, backbone choices, segmentation integration, classifier design
date: 2026-04-07
version: 1.0
status: Academic Documentation
phases_documented: "1–9"
author: Jordi Catafal
---

# Engineering Decisions — Model & Architecture

A record of critical model and architecture choices made during the development of Forest Pulse. Each decision documents context, alternatives considered, rationale, and consequences.

---

## ARCH-001: RF-DETR as Primary Detection Model

**Status:** Accepted  
**Date:** Phase 1–2 (2026-Q1)  
**Commits:** `340597d`, `8326cca`

### Context

Tree crown detection from aerial RGB imagery had been dominated by DeepForest (RetinaNet backbone, 2017) for a decade. By project start, RF-DETR (DINOv2 backbone, ICLR 2026) was available and represented SOTA for dense same-class object detection — precisely the problem of tree crowns in dense forest canopy.

### Options Considered

| Option | Backbone | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| DeepForest (Phase 1 only) | RetinaNet (2017) | Ecology-specific utils, pretrained on forest data | Outdated, 64% precision, 80 open issues, no maintenance |
| **RF-DETR (chosen)** | DINOv2 (2024) | SOTA detection, 90%+ precision, dense-scene contextual understanding | Not ecology-specific, requires fine-tuning |
| YOLO11 | CSPDarknet (2023) | Fast inference | Weaker contextual understanding for merged crowns |
| Deformable DETR | ResNet-50 + deformable attn | Moderate improvement | Slower than RF-DETR+DINOv2, less proven on canopy |

### Decision

Use **RF-DETR with DINOv2 backbone** as the production detection model. Use DeepForest **only as a one-time weak label bootstrapper** (Phase 3), not for runtime detection after Phase 1.

### Rationale

- DINOv2's hierarchical vision transformer excels at dense scenes with many similar-sized objects — exactly tree canopy from above.
- RF-DETR achieves 90%+ precision on pretrained model without domain-specific training.
- DeepForest's 80+ open issues and slow maintenance make it a long-term liability.
- Separation of concerns: detector finds crown-like regions; classifier decides if they're real trees.

### Consequences

- **Lost:** DeepForest's built-in NEON dataset integration and tiling utilities.
- **Gained:** 90%+ baseline precision; DINOv2 contextual attention for merged crowns; clean two-stage architecture.
- **Trade-off:** 4–8x slower on CPU vs RetinaNet — acceptable since deployment uses GPU/MPS.

---

## ARCH-002: DINOv2 as RF-DETR Backbone

**Status:** Accepted  
**Date:** Phase 2  
**Commits:** `c2a1d46`

### Context

RF-DETR supports multiple backbones. The choice determines detection quality. Options ranged from traditional ResNets to Meta's DINOv2 (2024).

### Options Considered

| Backbone | Params | Strengths | Tree Crowns |
|----------|--------|-----------|-------------|
| ResNet-50 | 25M | Fast, proven | Adequate |
| ViT-base | 86M | Global attention | Good but slower |
| **DINOv2-small (chosen)** | 21M | Self-supervised, dense correspondence maps, ImageNet 22K pretraining | Excellent |
| DINOv2-large | 304M | More capacity | Overkill; memory pressure on MPS |

### Decision

Use **DINOv2-small** as the RF-DETR backbone. DINOv2-large available as opt-in for stronger recall.

### Rationale

- DINOv2 trained with DINO (self-supervised), which learns dense correspondence maps — naturally suited to object detection.
- Captures both local crown boundaries and global context (crown among neighbors).
- Small variant (21M params) fits in 24 GB MPS and standard GPU budgets.
- On dense object benchmarks, outperforms ResNet-50 by 15–20% AP.

### Consequences

- Phase 2 fine-tuning achieved 90.4% mAP50 on self-labeled validation.
- DINOv2 embeddings will transfer to species classification (Phase 12).
- Global context improves SAM2 hybrid quality (ARCH-003).

---

## ARCH-003: SAM2 Integration — Three Segmentation Modes

**Status:** Accepted  
**Date:** Phase 6  
**Commits:** `6b5442d`

### Context

RF-DETR's recall is limited by crown merging: when two crowns touch, the detector sees one object. Lowering confidence threshold below 0.3 increases false positives more than true positives. SAM2 (Meta, 2024) is a foundation segmentation model offering both box-prompted refinement and automatic segmentation.

### Options Considered

| Approach | Recall Gain | Speed Impact | Chosen |
|----------|-------------|--------------|--------|
| Lower RF-DETR confidence | Near zero (net) | None | No |
| SAM2 refinement only | 0% (no recall gain) | 2x slower | Partial |
| SAM2 automatic only | Recovers missed crowns | 7x slower | Partial |
| **SAM2 hybrid (chosen)** | +3.9% recall | 26x slower | Yes |

### Decision

Implement **three complementary modes**:
1. **Automatic refinement** (`refine_detections_with_sam2`): convert each RF-DETR bbox → precise crown mask.
2. **Automatic segmentation** (`segment_all_trees_sam2`): SAM2 "segment everything" to catch missed crowns.
3. **Hybrid pipeline** (`detect_trees_hybrid`): RF-DETR → SAM2 refine → SAM2 auto → deduplicate.

Automatic and hybrid modes are opt-in; refinement is default when masks are requested.

### Rationale

- SAM2 eliminates need for domain-specific segmentation training (zero-shot).
- Box prompting from RF-DETR gives SAM2 a strong prior; `sam2.1-hiera-small` (46M params) suffices.
- Deduplication via mask IoU prevents double-counting.
- Thresholds lowered (pred_iou 0.5, stability 0.75) to tolerate soft crown boundaries.
- MPS compatibility via `PYTORCH_ENABLE_MPS_FALLBACK=1` + fp32-only inference.

### Consequences

- Dense beech canopy test (patch 0250): 6 RF-DETR → 10 hybrid detections (+67% on that patch).
- A/B test (10 patches): 840 → 873 detections (+3.9% recall, controlled precision).
- Hybrid mode 26x slower (7s vs 0.3s per 640×640 patch) — opt-in for this reason.
- Precise crown masks enable better health scoring and area/perimeter features.

---

## ARCH-004: Heuristic Health Scoring Before Trained Classifier

**Status:** Superseded (by Phase 3 trained classifier)  
**Date:** Phase 1 (heuristic), Phase 3 (classifier planned)

### Context

Health assessment needed immediate output in Phase 1 MVP, before any labeled health data existed.

### Decision

**Phase 1:** GRVI (Green-Red Vegetation Index) and ExG (Excess Green Index) with tunable thresholds.  
**Phase 3 (supersedes):** MobileNetV3 trained on Swedish Forest Damages (102K crop images + damage labels).

### Rationale

- Phase 1: physics-based indices (chlorophyll reflectance theory), zero data cost, fast for MVP.
- Phase 3: Swedish Forest Damages covers European forests (directly relevant to Catalonia); MobileNetV3 is efficient (5M params, runs on CPU) and will exceed heuristic accuracy by 15–20%.

### Consequences

- Phase 1 heuristic: ~60–70% accuracy, cannot distinguish damage types, fails in shadow/winter.
- Phase 3 classifier: ~80%+ accuracy, nuanced categories, handles lighting variation.

---

## ARCH-005: LiDAR-Distilled RGB Classifier Design

**Status:** Experimental / In Progress  
**Date:** Phase 9–9.5  
**Commits:** `e49391a`

### Context

After Phase 5, real-world inspection revealed: RF-DETR mAP50=0.904 on self-labels, but actual recall ~20–30% in dense canopy; many detections are shrubs. NDVI cannot distinguish trees from bushes (both chlorophyll-rich). LiDAR height is the only reliable discriminator.

### Decision

Build a **two-stage pipeline**:

1. **Stage 1: LiDAR height filter** (deterministic) — Drop detections with canopy height < 5 m.
2. **Stage 2: Multi-modal classifier** (learned) — GradientBoostingClassifier on 18 features (RF-DETR confidence + bbox geometry + GRVI/ExG + RGB statistics + 7 LiDAR fields). Trained on LiDAR auto-labels (height ≥ 5 m = tree; ≤ 2 m = bush; 2–5 m excluded as ambiguous).

### Rationale

- LiDAR height is physical measurement — no manual annotation required.
- Two stages: LiDAR filter (~10 ms) cheap; classifier handles ambiguous cases.
- scikit-learn GradientBoosting beats neural networks on small tabular datasets (~500 examples).
- 2–5 m band excluded to reduce label noise (genuine ecotone ambiguity).

### Consequences

- First honest eval metric (Phase 8): LiDAR-verified mAP ≠ self-label inflation.
- Requires LAZ tile download (~400–700 MB per patch, cached).
- LiDAR 2021–2023 vs RGB 2024: minor temporal mismatch for 5 m threshold.

---

## ARCH-006: Two-Stage "Detect Once, Classify Many Times" Architecture

**Status:** Experimental  
**Date:** Phase 9  
**Commits:** `e49391a`

### Context

Every classification improvement requires retraining the detector (20+ min cycles). Solution: frozen detector, fast classifier iteration.

### Decision

**Stages:** RF-DETR candidates → (optional) LiDAR height filter → (optional) NDVI filter → GradientBoostingClassifier → GeoJSON export.

### Rationale

- RF-DETR frozen (proven 90%+ pretrained precision); classifier iterations take 5 min (scikit-learn).
- Each stage has clear contract and failure modes.
- 18-feature vector: RF-DETR confidence + bbox geometry + GRVI/ExG + RGB stats + 7 LiDAR fields.
- `feature_importances_` shows which features matter most — interpretable for ecologists.

### Consequences

- Phase 10: classifier becomes production pipeline stage.
- Phase 11: harness optimizes RF-DETR + classifier against LiDAR-verified mAP.
- Classifier cannot improve detection recall (misses by RF-DETR stay missed).
- Implementation: `src/forest_pulse/classifier.py` (526 lines), `src/forest_pulse/lidar.py` (702 lines).

---

## ARCH-007: DeepForest as Label Bootstrapper Only (Not Runtime)

**Status:** Accepted  
**Date:** Phase 1 (runtime) → Phase 2 (bootstrap only)  
**Commits:** `e624d84`, `6659f1b`

### Context

DeepForest (pretrained RetinaNet) could serve as weak label source for RF-DETR fine-tuning — a one-time annotation operation.

### Decision

Use DeepForest **only as a one-time weak label bootstrapper** (Phase 3). Do not use it for runtime detection after Phase 1.

**Workflow:** Phase 1 demo → Phase 3: run DeepForest on 800 patches (~10K weak labels) → train RF-DETR → Phase 4+: RF-DETR only.

### Rationale

- Zero annotation cost (weeks of manual work avoided).
- One-time cost: run DeepForest once (4h on 800 patches); RF-DETR trains 20+ times.
- DeepForest 70–75% precision is "good enough" for weak labels.

---

## ARCH-008: COCO Format as Intermediate Annotation Standard

**Status:** Accepted  
**Date:** Phase 2  

### Decision

All annotation data (OAM-TCD, Montseny weak labels, Swedish Forest Damages) converted to **COCO JSON format** before training.

### Rationale

- RF-DETR expects COCO natively.
- Supervision library has built-in COCO support.
- OAM-TCD already COCO, eliminates one conversion.
- Well-documented schema; portable across detectors.

---

## ARCH-009: Module-Level Model Caching + Auto Device Detection

**Status:** Accepted  
**Date:** Phase 1–2  
**Commits:** `09a4511`

### Decision

- **Caching:** Module-level `_MODEL_CACHE` dicts keyed by model identifier in `detect.py` and `segment.py`.
- **Device detection:** `device.py` auto-selects CUDA > MPS > CPU; logs selected device.
- **MPS fallback:** `PYTORCH_ENABLE_MPS_FALLBACK=1` set before any PyTorch import.

### Rationale

- RF-DETR load time ~3s, SAM2 ~4s; cache makes subsequent calls instant.
- Zero user configuration — same code on Colab (CUDA), Mac (MPS), laptop (CPU).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` silently handles ops not yet on MPS (bicubic upsample in SAM2).

---

## Summary Table

| Decision | Phase | Key Choice | Enabled |
|----------|-------|-----------|---------|
| ARCH-001 | 1–2 | RF-DETR primary detector | ARCH-003, ARCH-006 |
| ARCH-002 | 2 | DINOv2 backbone | SAM2 quality, Phase 12 species |
| ARCH-003 | 6 | SAM2 hybrid segmentation | +3.9% recall |
| ARCH-004 | 1–3 | Heuristic → trained classifier | Phase 5 change detection |
| ARCH-005 | 9–9.5 | LiDAR-distilled classifier | Honest eval metric |
| ARCH-006 | 9 | Two-stage frozen detector | Fast classifier iteration |
| ARCH-007 | 1–3 | DeepForest bootstrap only | Zero annotation cost |
| ARCH-008 | 2 | COCO format standard | Portable annotations |
| ARCH-009 | 1–6 | Cache + device auto-detect | Cross-platform, fast inference |

---

## Architectural Principle: Detect Once, Classify Many Times

The most important principle guiding Phases 9–12:

- **Detector stays frozen:** RF-DETR finds regions that *look* like tree crowns.
- **Classifier iterates:** GradientBoostingClassifier decides if they're real trees.
- **Fast improvement cycle:** Classification = 5 min (retrain scikit-learn); detection = 20 min (retrain RF-DETR).

This modularity enables incremental improvement, interpretable failure modes, and future extension (species classifier reuses same feature vector, different output head).

---

## Open Questions and Deferred Decisions

| Deferred | Reason |
|----------|--------|
| Copernicus GLO-30 DSM for LiDAR alternative | 30 m resolution too coarse for individual crowns |
| Fine-tuning SAM2 on tree crowns | Zero-shot already works; ROI too low given time cost |
| Roboflow manual annotation | Superseded by LiDAR auto-labels (Phase 9) |
| Interactive HTML reports with maps | GeoJSON covers the use case; Leaflet/Folium ingest as needed |

---

## Appendix: Honesty on Evaluation Metrics

A meta-decision cuts across all phases:

> "mAP 0.904 is inflated — measured against self-labels, not ground truth."  
> "Real recall ~20–30% in dense uniform canopy (crown merging problem, not model problem)."  
> — progress.txt

This prevents the "metrics squeezing" trap. Phase 6 (SAM2): +3.9% measured on visual inspection; Phase 8: first true mAP against LiDAR physical ground truth.
