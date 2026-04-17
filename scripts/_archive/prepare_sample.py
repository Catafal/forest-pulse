"""Download and prepare a small OAM-TCD sample for training demos.

Streams 25 images from HuggingFace, resizes to 640x640 (RF-DETR input size),
scales bounding box annotations, and saves in COCO format.

Commits to data/sample/oam_tcd/ — small enough for Git (~15-20MB).

Usage:
    python scripts/prepare_sample.py
    python scripts/prepare_sample.py --num-images 50 --split-ratio 0.8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample" / "oam_tcd"
TARGET_SIZE = 640  # RF-DETR default input size


def prepare_sample(num_images: int = 25, split_ratio: float = 0.8):
    """Download OAM-TCD sample, resize, convert annotations, save as COCO.

    Args:
        num_images: Total images to download (train + val).
        split_ratio: Fraction used for training (rest is validation).
    """
    from datasets import load_dataset

    images_dir = OUTPUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming %d images from OAM-TCD (restor/tcd)...", num_images)
    ds = load_dataset("restor/tcd", split="train", streaming=True)

    # Collect images and annotations
    all_images_meta = []
    all_annotations = []
    annotation_id = 1

    for idx, sample in enumerate(ds):
        if idx >= num_images:
            break

        image_id = idx + 1
        pil_image = sample["image"]
        orig_w, orig_h = pil_image.size

        # Resize to target size — scale factor for annotation adjustment
        scale_x = TARGET_SIZE / orig_w
        scale_y = TARGET_SIZE / orig_h
        resized = pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

        # Save as JPEG (much smaller than TIFF/PNG for aerial imagery)
        filename = f"{image_id:04d}.jpg"
        resized.convert("RGB").save(images_dir / filename, "JPEG", quality=90)

        all_images_meta.append({
            "id": image_id,
            "file_name": filename,
            "height": TARGET_SIZE,
            "width": TARGET_SIZE,
        })

        # Scale COCO bounding box annotations to match resized image
        # OAM-TCD returns annotations as JSON string — parse if needed
        coco_annots = sample.get("coco_annotations", [])
        if isinstance(coco_annots, str):
            coco_annots = json.loads(coco_annots)
        for annot in coco_annots:
            # COCO bbox format: [x_min, y_min, width, height]
            x, y, w, h = annot["bbox"]
            scaled_bbox = [
                round(x * scale_x, 2),
                round(y * scale_y, 2),
                round(w * scale_x, 2),
                round(h * scale_y, 2),
            ]

            # Skip tiny boxes after resize (< 4px in either dimension)
            if scaled_bbox[2] < 4 or scaled_bbox[3] < 4:
                continue

            all_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # all trees → single class
                "bbox": scaled_bbox,
                "area": round(scaled_bbox[2] * scaled_bbox[3], 2),
                "iscrowd": 0,
            })
            annotation_id += 1

        biome = sample.get("biome_name", "unknown")
        n_trees = len([a for a in coco_annots if a.get("category_id") == 1])
        logger.info(
            "  [%d/%d] %s — %d trees, biome: %s",
            idx + 1, num_images, filename, n_trees, biome,
        )

    # Split into train and val
    split_idx = int(len(all_images_meta) * split_ratio)
    train_images = all_images_meta[:split_idx]
    val_images = all_images_meta[split_idx:]

    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}

    train_annots = [a for a in all_annotations if a["image_id"] in train_ids]
    val_annots = [a for a in all_annotations if a["image_id"] in val_ids]

    categories = [{"id": 1, "name": "tree", "supercategory": "vegetation"}]

    # Save COCO JSON files
    _save_coco_json(OUTPUT_DIR / "train.json", train_images, train_annots, categories)
    _save_coco_json(OUTPUT_DIR / "val.json", val_images, val_annots, categories)

    # Summary
    total_trees = len(all_annotations)
    logger.info("="*50)
    logger.info("Sample prepared at: %s", OUTPUT_DIR)
    logger.info("  Train: %d images, %d annotations", len(train_images), len(train_annots))
    logger.info("  Val:   %d images, %d annotations", len(val_images), len(val_annots))
    logger.info("  Total: %d images, %d tree annotations", num_images, total_trees)
    logger.info("  Image size: %dx%d (resized from 2048x2048)", TARGET_SIZE, TARGET_SIZE)
    logger.info("="*50)


def _save_coco_json(path: Path, images: list, annotations: list, categories: list):
    """Write a COCO-format JSON annotation file."""
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    with open(path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info("Saved: %s (%d images, %d annotations)", path.name, len(images), len(annotations))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Prepare OAM-TCD sample for training.")
    parser.add_argument("--num-images", type=int, default=25, help="Number of images to download.")
    parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/val split ratio.")
    args = parser.parse_args()

    prepare_sample(num_images=args.num_images, split_ratio=args.split_ratio)
