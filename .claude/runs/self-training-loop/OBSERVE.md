# Codebase Observation: self-training-loop

**Date:** 2026-04-06
**GitNexus available:** no

## Relevant Files

- `autoresearch/train.py` — RF-DETR training loop. Calls `model.train(dataset_dir, epochs, ...)`. Config vars: BACKBONE, LR, BATCH_SIZE, FINE_TUNE_EPOCHS. Copies best checkpoint to `current.pt`.
- `autoresearch/eval.py` — LOCKED. Loads checkpoint via `rfdetr.RFDETRBase(pretrain_weights=...)`, runs inference on `data/rfdetr/valid/`, computes mAP50 via `supervision.metrics.MeanAveragePrecision`. Prints `val_map50: X.XXXX`.
- `scripts/prepare_rfdetr_dataset.py` — `prepare_from_single_json(annotations_path, images_dir, output_dir, split_ratio=0.8, seed=42)`. Splits COCO JSON → train/valid with symlinks. Idempotent (wipes output_dir first). Re-indexes annotation IDs per split.
- `src/forest_pulse/detect.py` — `detect_trees(image, model_name, confidence)` → `sv.Detections`. Loads RF-DETR from checkpoint via `rfdetr.RFDETRBase(pretrain_weights=str(path))`. Caches models in `_MODEL_CACHE`.
- `scripts/bootstrap_annotations.py` — Pattern for COCO annotation generation from model predictions. Uses `predict_tile(patch_size=400, patch_overlap=0.25)` for DeepForest.
- `data/montseny/patches/` — 300 JPEG patches (640x640, 72MB total)
- `data/montseny/annotations_raw.json` — 10,376 DeepForest weak labels
- `checkpoints/current.pt` — Current best model (41.5% mAP50, 10 epochs on weak labels)

## Existing Patterns

- **RF-DETR API:** `model = rfdetr.RFDETRBase(pretrain_weights=path)` for checkpoint loading. `model.predict(images=img, threshold=conf)` returns `sv.Detections` directly.
- **COCO format:** `{"images": [...], "annotations": [...], "categories": [{"id": 0, "name": "tree"}]}`. Bbox: `[x, y, w, h]`.
- **Dataset prep:** Single function call: `prepare_from_single_json(annot_path, img_dir, output_dir)` handles everything.
- **Training:** Single call: `model.train(dataset_dir=..., epochs=..., batch_size=..., lr=..., output_dir=...)`. RF-DETR handles device placement internally.
- **Checkpoint contract:** `checkpoints/current.pt` is the name eval.py expects.

## Architecture Constraints

- Max 200 lines per function, 1000 lines per file
- All imports at top, type hints on public functions, Google-style docstrings
- eval.py is LOCKED — never modify
- Module contracts in ARCHITECTURE.md are source of truth
- Use .venv Python for all commands

## GitNexus Insights

GitNexus not available — observation based on static analysis only.

## Conditional Skill Routing

- [ ] /plan-eng-review — not applicable (small scope, 1 new file)
- [ ] /cso — not applicable (no auth/PII/external APIs)
- [ ] /backend-docs-writer — not applicable (script, not backend service)
- [ ] /frontend-docs-writer — not applicable (no frontend)
- [ ] /testing-patterns — not applicable (simple integration test)
