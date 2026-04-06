# Feature Spec: More ICGC Data + Clean Evaluation Set

**Slug:** more-data-clean-eval
**Date:** 2026-04-06
**Status:** draft

## What

Two changes to improve the tree detection model:

**1. More training data:** Expand `download_montseny.py` with 5 additional ICGC sampling zones spread across the full Montseny park (currently only 3 zones, ~9 km²). Then re-run tile + bootstrap + self-training on the larger dataset (~600-800 patches total).

**2. Clean evaluation set:** Create `data/montseny/eval_gold/` with 20 manually annotated patches (human-drawn bounding boxes) to measure real model performance. The current val set uses DeepForest's noisy labels as ground truth — the 70.6% mAP might be inaccurate. Update `autoresearch/eval.py` to use the gold eval set.

## Why

The model (70.6% mAP50 on noisy val) is bottlenecked by data quantity — 1,550 clean labels on 300 patches, with RF-DETR's 31.9M params massively underutilized. More diverse data (different slopes, altitudes, canopy densities) will improve generalization.

The eval set problem is equally important: we can't trust our metrics because ground truth is noisy DeepForest labels. A 20-image gold set gives us honest numbers and prevents us from optimizing against noise.

## Constraints

- New zones must use the existing WMS download pipeline (no new tools)
- Gold eval annotations must be in COCO format (same as everything else)
- eval.py changes must keep the same API: `evaluate(checkpoint_path) -> float`
- Gold eval patches must be committed to Git (small, ~5MB)
- Do NOT re-annotate all 300 patches — just 20 for eval
- Gold annotations are created by the user in Roboflow/CVAT (human action)

## Done Criteria

1. GIVEN `download_montseny.py` WHEN `--zones` is called THEN it supports at least 8 zones (3 existing + 5 new)
2. GIVEN the new zones WHEN tile + bootstrap runs THEN at least 500 total forest patches exist
3. GIVEN `--eval-gold` flag on eval.py WHEN run THEN it evaluates against `data/montseny/eval_gold/` instead of rfdetr val split
4. GIVEN 20 gold-annotated patches WHEN eval runs THEN mAP50 is computed against human-drawn boxes
5. GIVEN the expanded dataset WHEN self-training runs THEN it uses all patches (old + new)

## Out of Scope

- Training the model (user runs self_train.py themselves)
- Creating the gold annotations (user does this in Roboflow — we prepare the images + instructions)
- Changing model architecture (RF-DETR base stays)
- Multi-temporal data (different years — separate feature)
