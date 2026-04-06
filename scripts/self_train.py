"""Self-training loop for iterative label refinement.

Uses the current RF-DETR model as its own teacher: re-labels all patches
at a high confidence threshold, keeping only what the model is sure about.
Each round produces cleaner labels → better model → even cleaner labels.

This is semi-supervised learning: the model's confident predictions replace
the original noisy DeepForest weak labels, progressively filtering out
false positives and tightening bounding boxes.

Usage:
    python scripts/self_train.py --rounds 3 --confidence 0.7 --epochs 10
    python scripts/self_train.py --rounds 1 --confidence 0.5 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import rfdetr
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
MONTSENY_DIR = PROJECT_ROOT / "data" / "montseny"
RFDETR_DIR = PROJECT_ROOT / "data" / "rfdetr"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CURRENT_CHECKPOINT = CHECKPOINT_DIR / "current.pt"


def relabel_patches(
    checkpoint_path: Path,
    patch_dir: Path,
    confidence: float,
    output_path: Path,
) -> dict:
    """Re-label all patches using an RF-DETR checkpoint at high confidence.

    Loads the model fresh (no cache) to avoid stale weights between rounds.
    Only keeps detections above the confidence threshold — this is the core
    mechanism that filters noisy labels out of the training set.

    Args:
        checkpoint_path: Path to RF-DETR .pt checkpoint.
        patch_dir: Directory containing .jpg patches.
        confidence: Minimum confidence to keep a detection (e.g., 0.7).
        output_path: Where to save the COCO JSON.

    Returns:
        COCO annotation dict.
    """
    patches = sorted(patch_dir.glob("*.jpg"))
    if not patches:
        logger.error("No patches found in %s", patch_dir)
        return {}

    # Load model fresh each round — no caching across rounds
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = rfdetr.RFDETRBase(pretrain_weights=str(checkpoint_path))

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tree", "supercategory": "vegetation"}
        ],
    }

    annotation_id = 0
    total_trees = 0
    start = time.perf_counter()

    for img_id, patch_path in enumerate(patches):
        img = Image.open(patch_path).convert("RGB")
        w, h = img.size

        coco["images"].append({
            "id": img_id,
            "file_name": patch_path.name,
            "width": w,
            "height": h,
        })

        # Predict at the specified high confidence threshold
        detections = model.predict(images=img, threshold=confidence)

        if len(detections) == 0:
            continue

        # Convert sv.Detections → COCO annotations
        for xyxy in detections.xyxy:
            x1, y1, x2, y2 = xyxy.tolist()
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w < 5 or bbox_h < 5:
                continue

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [round(x1, 1), round(y1, 1),
                         round(bbox_w, 1), round(bbox_h, 1)],
                "area": round(bbox_w * bbox_h, 1),
                "iscrowd": 0,
            })
            annotation_id += 1

        total_trees += len(detections)

        if (img_id + 1) % 100 == 0:
            logger.info("  Relabeled %d/%d patches (%d trees)",
                        img_id + 1, len(patches), total_trees)

    elapsed = time.perf_counter() - start

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    n_annots = len(coco["annotations"])
    logger.info(
        "Relabeled %d patches → %d trees (%.1f/patch) in %.1fs",
        len(patches), n_annots, n_annots / max(len(patches), 1), elapsed,
    )
    return coco


def self_train(
    rounds: int = 3,
    confidence: float = 0.7,
    epochs: int = 10,
) -> list[dict]:
    """Run N rounds of self-training.

    Each round: relabel → prepare dataset → train → evaluate.

    Args:
        rounds: Number of self-training rounds.
        confidence: Min confidence for relabeling (higher = cleaner labels).
        epochs: Training epochs per round.

    Returns:
        List of round results with keys: round, n_annotations, map50.
    """
    # Import training and eval modules
    sys.path.insert(0, str(PROJECT_ROOT / "autoresearch"))
    import train as train_module
    from eval import evaluate

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from prepare_rfdetr_dataset import prepare_from_single_json

    if not CURRENT_CHECKPOINT.exists():
        logger.error("No checkpoint at %s. Train a model first.", CURRENT_CHECKPOINT)
        return []

    results = []
    prev_n_annots = None

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}/{rounds}")
        print(f"{'='*60}")

        # Backup checkpoint before training overwrites it.
        # Round 1 saves the original pre-self-training model as round_0.pt.
        backup_name = f"round_{round_num - 1}.pt"
        backup_path = CHECKPOINT_DIR / backup_name
        if CURRENT_CHECKPOINT.exists() and not backup_path.exists():
            shutil.copy2(CURRENT_CHECKPOINT, backup_path)
            logger.info("Backed up checkpoint → %s", backup_name)

        # Step 1: Relabel patches with current model at high confidence
        round_annots_path = MONTSENY_DIR / f"annotations_round_{round_num}.json"
        logger.info("Step 1: Relabeling at confidence >= %.2f", confidence)
        coco = relabel_patches(
            CURRENT_CHECKPOINT, PATCH_DIR, confidence, round_annots_path,
        )

        if not coco:
            logger.error("Relabeling failed. Stopping.")
            break

        n_annots = len(coco["annotations"])

        # Collapse warning: if annotations dropped drastically, warn user
        if prev_n_annots is not None and n_annots < prev_n_annots * 0.5:
            logger.warning(
                "Annotation count dropped >50%%: %d → %d. "
                "Consider lowering --confidence.",
                prev_n_annots, n_annots,
            )

        if n_annots < 50:
            logger.error(
                "Only %d annotations — too few to train. "
                "Lower --confidence or add more data.", n_annots,
            )
            break

        # Step 2: Prepare RF-DETR dataset layout
        logger.info("Step 2: Preparing dataset (80/20 split)")
        prepare_from_single_json(
            annotations_path=round_annots_path,
            images_dir=PATCH_DIR,
            output_dir=RFDETR_DIR,
        )

        # Step 3: Train for this round's epochs
        logger.info("Step 3: Training %d epochs", epochs)
        original_epochs = train_module.FINE_TUNE_EPOCHS
        train_module.FINE_TUNE_EPOCHS = epochs
        try:
            train_module.train()
        finally:
            train_module.FINE_TUNE_EPOCHS = original_epochs

        # Step 4: Evaluate
        logger.info("Step 4: Evaluating")
        map50 = evaluate(str(CURRENT_CHECKPOINT))

        results.append({
            "round": round_num,
            "n_annotations": n_annots,
            "map50": round(map50, 4),
        })
        prev_n_annots = n_annots

        print(f"\n  Round {round_num} result:")
        print(f"    Annotations: {n_annots}")
        print(f"    mAP50:       {map50:.4f}")

    # Final summary
    print(f"\n{'='*60}")
    print("  SELF-TRAINING COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  Round {r['round']}: {r['n_annotations']} annotations"
              f" → mAP50 = {r['map50']:.4f}")
    print(f"{'='*60}")

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Self-training loop for label refinement.",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Number of self-training rounds (default: 3).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Min confidence for relabeling (default: 0.5).",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Training epochs per round (default: 10).",
    )
    args = parser.parse_args()

    self_train(
        rounds=args.rounds,
        confidence=args.confidence,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
