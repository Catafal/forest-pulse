"""Gold evaluation — mAP50 against human-annotated ground truth.

Separate from eval.py (which is LOCKED and uses noisy DeepForest labels).
This module evaluates against a small set of manually verified bounding boxes,
giving an honest measure of model quality.

The gold set lives at data/montseny/eval_gold/:
  images/       — 20 diverse patches (committed to Git)
  annotations.json  — human-drawn COCO bounding boxes

Usage:
    python autoresearch/eval_gold.py
    python autoresearch/eval_gold.py checkpoints/round_1.pt
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

GOLD_DIR = Path(__file__).parent.parent / "data" / "montseny" / "eval_gold"
GOLD_IMAGES = GOLD_DIR / "images"
GOLD_ANNOT = GOLD_DIR / "annotations.json"


def evaluate_gold(
    model_path: str = "checkpoints/current.pt",
) -> float:
    """Evaluate model against human-annotated gold set.

    Same metric as eval.py (mAP50 via supervision) but against clean
    human-drawn bounding boxes instead of noisy DeepForest labels.

    Args:
        model_path: Path to RF-DETR checkpoint.

    Returns:
        mAP50 score (0.0 - 1.0). Returns -1.0 if gold set not ready.
    """
    import rfdetr

    checkpoint = Path(model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if not GOLD_ANNOT.exists():
        logger.error("Gold annotations not found: %s", GOLD_ANNOT)
        logger.error("Run: python scripts/prepare_gold_eval.py")
        logger.error("Then annotate in Roboflow and replace annotations.json")
        return -1.0

    # Load ground truth
    with open(GOLD_ANNOT) as f:
        coco = json.load(f)

    n_annots = len(coco.get("annotations", []))
    if n_annots == 0:
        logger.warning(
            "Gold annotations.json has 0 annotations. "
            "Have you annotated in Roboflow yet?"
        )
        return -1.0

    images_by_id = {img["id"]: img for img in coco["images"]}
    annots_by_image = defaultdict(list)
    for annot in coco["annotations"]:
        annots_by_image[annot["image_id"]].append(annot)

    # Load model
    logger.info("Loading checkpoint: %s", checkpoint)
    model = rfdetr.RFDETRBase(pretrain_weights=str(checkpoint))

    all_predictions = []
    all_targets = []

    for img_id, img_info in images_by_id.items():
        img_path = str(GOLD_IMAGES / img_info["file_name"])

        # Predict at low threshold for full precision-recall curve
        preds = model.predict(images=img_path, threshold=0.01)
        if preds.class_id is None:
            preds.class_id = np.zeros(len(preds), dtype=int)
        # Strip metadata that supervision metrics can't handle
        if hasattr(preds, "data") and "source_shape" in preds.data:
            del preds.data["source_shape"]
        all_predictions.append(preds)

        # Build ground truth from COCO annotations
        annots = annots_by_image.get(img_id, [])
        if annots:
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

    # Compute mAP50
    from supervision.metrics import MeanAveragePrecision

    metric = MeanAveragePrecision()
    result = metric.update(all_predictions, all_targets).compute()
    map50 = float(result.map50)

    logger.info(
        "Gold evaluation: mAP50 = %.4f on %d images (%d annotations)",
        map50, len(images_by_id), n_annots,
    )

    # Machine-readable output
    print(f"gold_map50: {map50:.4f}")
    return map50


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).parent.parent / "checkpoints" / "current.pt"
    )
    evaluate_gold(checkpoint)
