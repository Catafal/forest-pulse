# Implementation Plan — Forest Pulse

## Phase 1: MVP Detection + Visualization (Weekend)

**Goal:** Run tree detection on aerial imagery, visualize with health colors, in Colab.

- [ ] 1.1 — Implement `detect.py`: wrap DeepForest pretrained model for quick demo
- [ ] 1.2 — Implement `health.py`: GRVI + ExG computation on cropped bboxes
- [ ] 1.3 — Implement `visualize.py`: Supervision annotator with health color mapping
- [ ] 1.4 — Create `scripts/demo.py`: CLI that runs detect → health → visualize on an image
- [ ] 1.5 — Create `notebooks/01_quickstart.ipynb`: Colab-ready demo notebook
- [ ] 1.6 — Write `scripts/download_data.py`: fetch sample images for demo
- [ ] 1.7 — Test on OAM-TCD sample image + any public aerial image

**Output:** Colorful annotated image showing detected trees with health color coding.

---

## Phase 2: RF-DETR Training + Auto-Research Harness

**Goal:** Fine-tune RF-DETR on OAM-TCD, optimize config overnight via harness.

- [ ] 2.1 — Implement data pipeline: OAM-TCD → COCO bbox format → RF-DETR dataloader
- [ ] 2.2 — Implement `autoresearch/eval.py`: locked mAP50 evaluation on validation shard
- [ ] 2.3 — Implement `autoresearch/train.py`: configurable RF-DETR fine-tuning script
- [ ] 2.4 — Write `autoresearch/program.md`: agent experiment loop instructions
- [ ] 2.5 — Run harness overnight → find best config → commit winning `train.py`
- [ ] 2.6 — Swap `detect.py` default model from DeepForest to fine-tuned RF-DETR
- [ ] 2.7 — Benchmark on NeonTreeEvaluation → report mAP50

**Output:** Fine-tuned RF-DETR model outperforming DeepForest baseline.

---

## Phase 3: Health Classification

**Goal:** Train proper health classifier using Swedish Forest Damages labels.

- [ ] 3.1 — Download Swedish Forest Damages dataset (102K bboxes + damage categories)
- [ ] 3.2 — Map damage categories → healthy/stressed/dead labels
- [ ] 3.3 — Train MobileNetV3 classifier on cropped tree crowns
- [ ] 3.4 — Integrate trained classifier into `health.py` (replace heuristic thresholds)
- [ ] 3.5 — Validate accuracy on held-out test split

**Output:** Learned health classifier replacing RGB heuristics.

---

## Phase 4: Georeferencing + GIS Export

**Goal:** Produce GPS-referenced GeoJSON from any aerial image.

- [ ] 4.1 — Implement `georef.py`: EXIF GPS parsing for drone images
- [ ] 4.2 — Implement `georef.py`: CRS-based georeferencing for ICGC/PNOA orthophotos
- [ ] 4.3 — Implement `export.py`: GeoJSON + Shapefile export
- [ ] 4.4 — Test full pipeline: image → detect → health → georef → GeoJSON
- [ ] 4.5 — Validate by loading GeoJSON in QGIS

**Output:** Drop a GeoJSON into QGIS → see every tree on a map with health color.

---

## Phase 5: Temporal Change Detection

**Goal:** Compare two time periods, report what changed.

- [ ] 5.1 — Implement `temporal.py`: nearest-neighbor spatial matching within GPS tolerance
- [ ] 5.2 — Implement change metrics: tree loss, crown area change, health progression
- [ ] 5.3 — Implement `export.py` HTML report: summary + change table + map
- [ ] 5.4 — Test on ICGC historical vs current orthophotos of same area

**Output:** Change report: "847 → 831 trees (-1.9%), 43 declining, 12 newly stressed."

---

## Phase 6: Polish + Release

**Goal:** Clean, documented, impressive open-source release.

- [ ] 6.1 — Write comprehensive README with demo images/GIFs
- [ ] 6.2 — Add tests for each module (pytest, minimum 70% coverage)
- [ ] 6.3 — Create GitHub Actions CI (lint + test)
- [ ] 6.4 — Publish to PyPI as `forest-pulse`
- [ ] 6.5 — Write blog post / X thread about the project
