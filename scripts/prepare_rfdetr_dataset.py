"""Restructure OAM-TCD sample into RF-DETR's expected folder layout.

RF-DETR requires images alongside the annotation JSON in each split folder:
    dataset/train/_annotations.coco.json + *.jpg
    dataset/valid/_annotations.coco.json + *.jpg

This script creates data/rfdetr/ from data/sample/oam_tcd/ using symlinks
for images (zero disk overhead). Idempotent — safe to re-run.

Usage:
    python scripts/prepare_rfdetr_dataset.py
    python scripts/prepare_rfdetr_dataset.py --source data/sample/oam_tcd --output data/rfdetr
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SOURCE = Path(__file__).parent.parent / "data" / "sample" / "oam_tcd"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "rfdetr"

# RF-DETR expects this exact filename for annotations
RFDETR_ANNOT_NAME = "_annotations.coco.json"

# Mapping from our naming convention to RF-DETR's split folders
SPLIT_MAP = {
    "train.json": "train",
    "val.json": "valid",  # RF-DETR uses "valid", not "val"
}


def prepare_rfdetr_dataset(
    source_dir: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT,
) -> Path:
    """Create RF-DETR compatible dataset layout from existing COCO data.

    Symlinks images into split folders and renames annotation JSONs to
    _annotations.coco.json. Wipes output_dir first for idempotency.

    Args:
        source_dir: Path to existing COCO dataset (with images/ + train.json + val.json).
        output_dir: Path to create RF-DETR layout.

    Returns:
        Path to output_dir.
    """
    images_dir = source_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Wipe and recreate for idempotency
    if output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info("Cleaned existing output: %s", output_dir)

    for annot_file, split_name in SPLIT_MAP.items():
        annot_path = source_dir / annot_file
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annot_path}")

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Load annotations to find which images belong to this split
        with open(annot_path) as f:
            coco = json.load(f)

        # Symlink each image into the split folder
        image_count = 0
        for img_info in coco["images"]:
            src = (images_dir / img_info["file_name"]).resolve()
            dst = split_dir / img_info["file_name"]
            if not src.exists():
                logger.warning("Image missing, skipping: %s", src)
                continue
            dst.symlink_to(src)
            image_count += 1

        # Copy annotation JSON with RF-DETR's expected filename
        dst_annot = split_dir / RFDETR_ANNOT_NAME
        shutil.copy2(annot_path, dst_annot)

        n_annots = len(coco.get("annotations", []))
        logger.info(
            "  %s: %d images, %d annotations → %s",
            split_name, image_count, n_annots, split_dir,
        )

    logger.info("RF-DETR dataset ready at: %s", output_dir)
    return output_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Prepare RF-DETR dataset layout.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source COCO dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dir.")
    args = parser.parse_args()

    prepare_rfdetr_dataset(source_dir=args.source, output_dir=args.output)
