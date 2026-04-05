"""Prepare RF-DETR dataset layout from COCO annotations + images.

RF-DETR requires images alongside the annotation JSON in each split folder:
    dataset/train/_annotations.coco.json + *.jpg
    dataset/valid/_annotations.coco.json + *.jpg

Supports two input modes:
  1. Pre-split: source has train.json + val.json + images/
  2. Single JSON: source has one COCO JSON + images folder → auto-splits 80/20

Usage:
    python scripts/prepare_rfdetr_dataset.py --source data/montseny
    python scripts/prepare_rfdetr_dataset.py --source data/sample/oam_tcd
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "rfdetr"
RFDETR_ANNOT_NAME = "_annotations.coco.json"


def prepare_from_single_json(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """Split a single COCO JSON into train/valid and create RF-DETR layout.

    Args:
        annotations_path: Path to COCO JSON with all annotations.
        images_dir: Path to directory containing image files.
        output_dir: Where to create the RF-DETR layout.
        split_ratio: Fraction of images for training (default: 0.8).
        seed: Random seed for reproducible splits.

    Returns:
        Path to output_dir.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Shuffle and split images
    random.seed(seed)
    shuffled = list(images)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * split_ratio)
    train_images = shuffled[:split_idx]
    val_images = shuffled[split_idx:]

    logger.info("Split: %d train, %d valid (%.0f%%/%.0f%%)",
                len(train_images), len(val_images),
                split_ratio * 100, (1 - split_ratio) * 100)

    # Build annotation lookup by image_id
    annots_by_image = {}
    for annot in annotations:
        annots_by_image.setdefault(annot["image_id"], []).append(annot)

    # Wipe and recreate for idempotency
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split_name, split_images in [("train", train_images), ("valid", val_images)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Collect annotations for this split and re-index
        split_annots = []
        annot_id = 0
        for img in split_images:
            for annot in annots_by_image.get(img["id"], []):
                new_annot = dict(annot)
                new_annot["id"] = annot_id
                split_annots.append(new_annot)
                annot_id += 1

        # Write COCO JSON for this split
        split_coco = {
            "images": split_images,
            "annotations": split_annots,
            "categories": categories,
        }
        with open(split_dir / RFDETR_ANNOT_NAME, "w") as f:
            json.dump(split_coco, f, indent=2)

        # Symlink images into split folder
        for img in split_images:
            src = (images_dir / img["file_name"]).resolve()
            dst = split_dir / img["file_name"]
            if src.exists():
                dst.symlink_to(src)

        logger.info("  %s: %d images, %d annotations",
                    split_name, len(split_images), len(split_annots))

    logger.info("RF-DETR dataset ready: %s", output_dir)
    return output_dir


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Prepare RF-DETR dataset layout.",
    )
    parser.add_argument(
        "--source", type=Path,
        default=Path(__file__).parent.parent / "data" / "montseny",
        help="Source directory.",
    )
    parser.add_argument(
        "--annotations", type=str,
        default="annotations_raw.json",
        help="COCO JSON filename within source.",
    )
    parser.add_argument(
        "--images", type=str, default="patches",
        help="Images subdirectory within source.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output RF-DETR layout directory.",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8).",
    )
    args = parser.parse_args()

    prepare_from_single_json(
        annotations_path=args.source / args.annotations,
        images_dir=args.source / args.images,
        output_dir=args.output,
        split_ratio=args.split_ratio,
    )


if __name__ == "__main__":
    main()
