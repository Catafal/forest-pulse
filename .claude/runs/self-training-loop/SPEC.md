# Feature Spec: Self-Training Loop for Label Refinement

**Slug:** self-training-loop
**Date:** 2026-04-06
**Status:** draft

## What

A `scripts/self_train.py` script that runs N rounds of self-training:
1. Load the current RF-DETR checkpoint
2. Re-label all 300 Montseny patches using the model at high confidence threshold
3. Keep only detections above the confidence threshold as new "clean" labels
4. Split into train/val, prepare RF-DETR dataset layout
5. Retrain RF-DETR on the cleaner labels for M epochs
6. Repeat for N rounds

Each round produces a new checkpoint and a new COCO annotation file. Metrics (mAP50) are logged per round so the user can see improvement (or decide to stop).

When invoked: `python scripts/self_train.py --rounds 3 --confidence 0.7 --epochs 10`
The script prints mAP50 per round and saves the best checkpoint.

## Why

Our current RF-DETR model (41.5% mAP50) was trained on DeepForest's weak labels (~60% precision). The model's own high-confidence predictions are cleaner than those original labels. By using the model as its own teacher — keeping only what it's confident about — each round produces progressively cleaner training data and a better model.

This avoids the need for manual annotation correction (1-2 hours in Roboflow) and the GroundingDINO teacher approach (which failed — it sees "forest" not individual crowns from aerial imagery).

## Constraints

- Must work on Apple MPS (Mac Mini 24GB / MacBook Pro 48GB)
- Must reuse existing `autoresearch/train.py` training config (backbone, LR, batch_size)
- Must reuse existing `autoresearch/eval.py` for mAP50 evaluation
- Must reuse existing `scripts/prepare_rfdetr_dataset.py` for dataset layout
- Confidence threshold for re-labeling must be configurable (default: 0.7)
- Must NOT modify `autoresearch/eval.py` (locked per auto-research contract)
- All annotations saved as COCO JSON (existing format)
- Each round's annotations saved separately (auditable trail)

## Done Criteria

1. GIVEN a trained checkpoint WHEN `self_train.py --rounds 1 --epochs 5` runs THEN it produces a new checkpoint and prints mAP50
2. GIVEN round N's checkpoint WHEN round N+1 runs THEN it re-labels patches using round N's model at the specified confidence threshold
3. GIVEN `--rounds 3` WHEN the script completes THEN 3 separate annotation files exist at `data/montseny/annotations_round_{1,2,3}.json`
4. GIVEN `--confidence 0.7` WHEN re-labeling THEN only detections with score >= 0.7 are kept as labels
5. GIVEN the final round completes WHEN the best checkpoint is compared to the initial THEN mAP50 is logged for both (improvement is observable)
6. GIVEN any round WHEN training completes THEN the script copies the best checkpoint to `checkpoints/current.pt` (eval.py contract)

## Out of Scope

- Manual annotation correction (Roboflow/CVAT) — that's a separate workflow
- Auto-research harness integration — this is a standalone script, harness comes later
- GroundingDINO or any external teacher model
- Health scoring or visualization — this is purely about detection quality
- Downloading additional data — works with existing 300 patches
