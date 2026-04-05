"""LOCKED evaluation script — DO NOT MODIFY during harness runs.

This file defines the ground truth metric for the auto-research harness.
The agent must NEVER edit this file. Modifying it would allow the agent
to "improve" by changing the benchmark, not the model.

Metric: mAP50 on the fixed validation shard.
Output: prints "val_map50: X.XXXX" to stdout (machine-readable, grep-able).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# Fixed validation data — these paths must never change
VAL_DIR = Path(__file__).parent.parent / "data" / "rfdetr" / "valid"
VAL_ANNOT = VAL_DIR / "_annotations.coco.json"


def evaluate(model_path: str = "checkpoints/current.pt") -> float:
    """Run evaluation on the locked validation set.

    Loads the model checkpoint, runs inference on every validation image,
    computes mAP50 against ground truth annotations.

    Args:
        model_path: Path to the model checkpoint (.pt or .pth).

    Returns:
        mAP50 score (0.0 - 1.0).
    """
    import rfdetr

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not VAL_ANNOT.exists():
        raise FileNotFoundError(
            f"Validation annotations not found: {VAL_ANNOT}. "
            "Run: python scripts/prepare_rfdetr_dataset.py"
        )

    # Load ground truth
    with open(VAL_ANNOT) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}
    annots_by_image = defaultdict(list)
    for annot in coco["annotations"]:
        annots_by_image[annot["image_id"]].append(annot)

    # Load model from checkpoint
    logger.info("Loading checkpoint: %s", checkpoint)
    model = rfdetr.RFDETRBase.from_checkpoint(str(checkpoint))

    # Collect predictions and ground truths for each image
    all_predictions = []
    all_targets = []

    for img_id, img_info in images_by_id.items():
        img_path = str(VAL_DIR / img_info["file_name"])

        # Predict with very low confidence to capture full precision-recall curve
        preds = model.predict(image=img_path, threshold=0.01)

        # Ensure class_id is set (single-class: all trees = 0)
        if preds.class_id is None:
            preds.class_id = np.zeros(len(preds), dtype=int)

        all_predictions.append(preds)

        # Build ground truth sv.Detections from COCO annotations
        annots = annots_by_image.get(img_id, [])
        if annots:
            # COCO bbox [x, y, w, h] → xyxy [x1, y1, x2, y2]
            xyxy = np.array(
                [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                 for a in annots],
                dtype=np.float32,
            )
            targets = sv.Detections(
                xyxy=xyxy,
                class_id=np.zeros(len(annots), dtype=int),
            )
        else:
            targets = sv.Detections.empty()
            targets.class_id = np.array([], dtype=int)

        all_targets.append(targets)

    # Compute mAP50 using Supervision's metrics
    from supervision.metrics import MeanAveragePrecision

    metric = MeanAveragePrecision()
    result = metric.update(all_predictions, all_targets).compute()

    map50 = float(result.map50)
    logger.info("Evaluation complete: mAP50 = %.4f on %d images", map50, len(images_by_id))

    # Machine-readable output — the harness greps for this exact line
    print(f"val_map50: {map50:.4f}")
    return map50


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "checkpoints" / "current.pt"
    )
    evaluate(checkpoint)
