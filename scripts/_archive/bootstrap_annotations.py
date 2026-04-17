"""Bootstrap tree crown annotations using DeepForest pretrained model.

Runs DeepForest on all patches in data/montseny/patches/, generates
weak bounding box labels in COCO format. These are "weak labels" —
the model was trained on American forests so precision on Catalan
forests is limited, but it's 5-10x faster than annotating from scratch.

Uses predict_tile() with sliding windows (patch_size=400, overlap=0.25)
because DeepForest was trained on 400x400 NEON tiles at 10cm — our
640x640 patches at 25cm need the windowing to match expected scale.

Output is meant to be manually corrected in Roboflow before training.

Usage:
    python scripts/bootstrap_annotations.py
    python scripts/bootstrap_annotations.py --confidence 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

PATCH_DIR = Path(__file__).parent.parent / "data" / "montseny" / "patches"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "montseny" / "annotations_raw.json"
)

# Lower default confidence because DeepForest is uncertain on non-American forests.
# predict_tile with NMS will handle duplicate removal.
DEFAULT_CONFIDENCE = 0.1


def bootstrap_annotations(
    patch_dir: Path = PATCH_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict:
    """Run DeepForest on all patches and export COCO annotations.

    Uses predict_tile() with sliding windows for better detection on
    640x640 patches (DeepForest was trained on 400x400 tiles).

    Args:
        patch_dir: Directory containing .jpg patches.
        output_path: Where to save the COCO JSON.
        confidence: Minimum detection confidence threshold.

    Returns:
        COCO annotation dict.
    """
    from deepforest import main as df_main

    patches = sorted(patch_dir.glob("*.jpg"))
    if not patches:
        logger.error("No patches found in %s", patch_dir)
        logger.error("Run: python scripts/tile_orthophoto.py")
        return {}

    # Load model once
    logger.info("Loading DeepForest model...")
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    logger.info("Bootstrapping annotations for %d patches...", len(patches))

    # COCO format structure
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
        image = np.array(Image.open(patch_path).convert("RGB"))
        h, w = image.shape[:2]

        # Register image in COCO
        coco["images"].append({
            "id": img_id,
            "file_name": patch_path.name,
            "width": w,
            "height": h,
        })

        # predict_tile does sliding window + NMS internally.
        # patch_size=400 matches DeepForest's training resolution.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = model.predict_tile(
                image=image,
                patch_size=400,
                patch_overlap=0.25,
            )

        if result is None or len(result) == 0:
            continue

        # Filter by confidence
        result = result[result["score"] >= confidence]
        if len(result) == 0:
            continue

        # Convert DataFrame → COCO annotations
        for _, row in result.iterrows():
            x1, y1 = row["xmin"], row["ymin"]
            bbox_w = row["xmax"] - x1
            bbox_h = row["ymax"] - y1

            # Skip tiny detections (likely noise)
            if bbox_w < 5 or bbox_h < 5:
                continue

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [
                    round(float(x1), 1), round(float(y1), 1),
                    round(float(bbox_w), 1), round(float(bbox_h), 1),
                ],
                "area": round(float(bbox_w * bbox_h), 1),
                "iscrowd": 0,
            })
            annotation_id += 1

        patch_trees = len(result)
        total_trees += patch_trees

        # Progress logging every 50 images
        if (img_id + 1) % 50 == 0:
            logger.info(
                "  Processed %d/%d patches (%d trees so far)",
                img_id + 1, len(patches), total_trees,
            )

    elapsed = time.perf_counter() - start

    # Save COCO JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info("Done in %.1fs", elapsed)
    logger.info(
        "Images: %d | Annotations: %d trees",
        len(coco["images"]), len(coco["annotations"]),
    )
    logger.info("Saved: %s", output_path)

    return coco


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Bootstrap tree annotations with DeepForest.",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="Min detection confidence (default: 0.1).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output COCO JSON path.",
    )
    args = parser.parse_args()

    coco = bootstrap_annotations(
        output_path=args.output,
        confidence=args.confidence,
    )

    if coco:
        n_images = len(coco["images"])
        n_annots = len(coco["annotations"])
        avg = n_annots / max(n_images, 1)
        print(f"\n{'='*50}")
        print("  Bootstrap Annotations Complete")
        print(f"{'='*50}")
        print(f"  Patches:     {n_images}")
        print(f"  Tree labels: {n_annots} ({avg:.1f} per patch)")
        print(f"  Output:      {args.output}")
        print(f"{'='*50}")
        print("\n  These are WEAK labels — correct in Roboflow,")
        print("  then export corrected COCO JSON for training.")


if __name__ == "__main__":
    main()
