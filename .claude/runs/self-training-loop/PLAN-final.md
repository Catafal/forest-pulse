# Implementation Plan: self-training-loop

**Version:** 1
**Date:** 2026-04-06
**Based on:** SPEC.md + OBSERVE.md

## Summary

Create `scripts/self_train.py` (~120 lines) that orchestrates N rounds of self-training by calling existing modules: RF-DETR inference for re-labeling, `prepare_rfdetr_dataset.py` for dataset layout, `train.py` config for training, and `eval.py` for mAP50. One new file, zero modifications to existing code.

## Files to Modify

| File | Change | Rationale |
|---|---|---|
| None | — | All existing files are reused as-is via imports/subprocess |

## Files to Create

| File | Purpose |
|---|---|
| `scripts/self_train.py` | Self-training loop orchestrator (~120 lines) |
| `tests/test_self_train.py` | Unit tests for re-labeling logic (~40 lines) |

## Implementation: `scripts/self_train.py`

**Core function: `self_train(rounds, confidence, epochs)`**

```
For round_num in 1..N:
    1. checkpoint = "checkpoints/current.pt"
       (round 1 uses whatever exists; round 2+ uses previous round's output)

    2. Re-label: load model from checkpoint → predict on all 300 patches
       at `confidence` threshold → save as COCO JSON
       → data/montseny/annotations_round_{round_num}.json

    3. Prepare dataset: call prepare_from_single_json(
           annotations_path=round_annots,
           images_dir=data/montseny/patches,
           output_dir=data/rfdetr
       )

    4. Train: import and call train() from autoresearch/train.py
       (temporarily override FINE_TUNE_EPOCHS to `epochs` param)

    5. Evaluate: import and call evaluate() from autoresearch/eval.py
       → get mAP50 for this round

    6. Log: print round summary (annotations count, mAP50)
```

**Re-labeling function: `relabel_patches(checkpoint_path, patch_dir, confidence, output_path)`**

Uses `rfdetr.RFDETRBase(pretrain_weights=checkpoint)` directly (not via detect.py, to avoid cache issues across rounds). Iterates patches, runs `model.predict(images=path, threshold=confidence)`, converts sv.Detections → COCO annotations.

**CLI:**
```
python scripts/self_train.py --rounds 3 --confidence 0.7 --epochs 10
```

## Tests to Write

| Done Criterion # | Test | Type |
|---|---|---|
| 1 | `test_relabel_produces_coco_json` — mock model, verify COCO structure | unit |
| 2 | `test_confidence_filtering` — verify only high-conf detections kept | unit |
| 3 | `test_round_annotations_saved_separately` — check file naming | unit |

## Documentation Updates

- `progress.txt` — update after implementation

## Approach Alternatives

| Alternative | Rejected Because |
|---|---|
| Modify train.py to accept checkpoint as starting weights | Would break auto-research harness contract. Self-train script handles this externally. |
| Use detect.py for re-labeling | detect.py caches models globally — round N's model would persist into round N+1. Direct rfdetr import is cleaner. |
| Store round checkpoints separately | Unnecessary complexity. eval.py expects `current.pt`. Each round overwrites it. Round annotations provide the audit trail. |
| Use DeepForest predict_tile for re-labeling | DeepForest is the weak teacher we're replacing. Our RF-DETR is the stronger model now. |

## Risks and Side Effects

- **Risk:** Self-training can collapse if confidence threshold is too high (no annotations survive) or too low (noisy labels persist). Mitigation: log annotation count per round, warn if count drops below 50% of previous round.
- **Risk:** Overwriting `data/rfdetr/` on each round destroys previous layout. Mitigation: annotations are saved per-round in separate files; layout is regenerated from those.
- **Side effect:** `checkpoints/current.pt` is overwritten each round. If training fails mid-round, the checkpoint from the previous round is lost. Mitigation: copy current.pt to `checkpoints/round_{N-1}.pt` before training.

## Estimated Scope

- 1 new file: `scripts/self_train.py` (~120 lines)
- 1 new test file: `tests/test_self_train.py` (~40 lines)
- Complexity: low — orchestration of existing modules
