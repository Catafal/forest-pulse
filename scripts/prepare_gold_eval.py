"""Prepare the gold evaluation set for human annotation.

Selects 20 diverse patches (spread across zones) and copies them to
data/montseny/eval_gold/images/. Creates an empty COCO annotations.json
with images registered but zero annotations — the user fills these in
by annotating in Roboflow/CVAT and exporting COCO JSON.

Usage:
    python scripts/prepare_gold_eval.py
    python scripts/prepare_gold_eval.py --n-patches 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

PATCH_DIR = Path(__file__).parent.parent / "data" / "montseny" / "patches"
METADATA_CSV = (
    Path(__file__).parent.parent / "data" / "montseny" / "patches_metadata.csv"
)
GOLD_DIR = Path(__file__).parent.parent / "data" / "montseny" / "eval_gold"


def prepare_gold_set(n_patches: int = 20, seed: int = 42) -> Path:
    """Select diverse patches and prepare gold directory.

    Picks patches spread across all available zones (round-robin),
    prioritizing high ExG (dense forest) for annotation quality.

    Args:
        n_patches: Number of patches to select for gold set.
        seed: Random seed for reproducible selection.

    Returns:
        Path to gold directory.
    """
    if not METADATA_CSV.exists():
        logger.error("Metadata not found: %s", METADATA_CSV)
        logger.error("Run tile_orthophoto.py first.")
        return GOLD_DIR

    # Read metadata to get zone information per patch
    with open(METADATA_CSV) as f:
        rows = list(csv.DictReader(f))

    # Group patches by zone for balanced selection
    by_zone = {}
    for row in rows:
        by_zone.setdefault(row["zone"], []).append(row)

    zones = sorted(by_zone.keys())
    logger.info("Found %d zones: %s", len(zones), ", ".join(zones))

    # Select patches: round-robin across zones, pick randomly within each
    random.seed(seed)
    selected = []
    per_zone = max(1, n_patches // len(zones))
    remainder = n_patches - per_zone * len(zones)

    for zone in zones:
        pool = by_zone[zone]
        random.shuffle(pool)
        count = per_zone + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        selected.extend(pool[:count])

    selected = selected[:n_patches]
    logger.info("Selected %d patches across %d zones", len(selected), len(zones))

    # Create gold directory
    images_dir = GOLD_DIR / "images"
    if GOLD_DIR.exists():
        shutil.rmtree(GOLD_DIR)
    images_dir.mkdir(parents=True)

    # Copy selected patches and build COCO structure
    coco = {
        "images": [],
        "annotations": [],  # Empty — user fills via Roboflow
        "categories": [
            {"id": 0, "name": "tree", "supercategory": "vegetation"}
        ],
    }

    for img_id, row in enumerate(selected):
        src = PATCH_DIR / row["filename"]
        dst = images_dir / row["filename"]
        shutil.copy2(src, dst)

        img = Image.open(src)
        w, h = img.size
        coco["images"].append({
            "id": img_id,
            "file_name": row["filename"],
            "width": w,
            "height": h,
        })

    # Save empty COCO annotations (user replaces after Roboflow export)
    annot_path = GOLD_DIR / "annotations.json"
    with open(annot_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info("Gold set ready: %s", GOLD_DIR)
    logger.info("  Images: %d", len(selected))
    logger.info("  Annotations: 0 (you need to annotate these)")

    return GOLD_DIR


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Prepare gold set for annotation.",
    )
    parser.add_argument(
        "--n-patches", type=int, default=20,
        help="Number of patches to select (default: 20).",
    )
    args = parser.parse_args()

    prepare_gold_set(n_patches=args.n_patches)

    print(f"\n{'='*55}")
    print("  Gold Set Prepared")
    print(f"{'='*55}")
    print(f"  Location: {GOLD_DIR}")
    print(f"  Patches:  {args.n_patches} (diverse zones)")
    print("  Annotations: EMPTY — you must annotate them")
    print(f"{'='*55}")
    print("\n  Next steps:")
    print("  1. Upload images/ folder to Roboflow (free)")
    print("  2. Draw bounding boxes around every tree crown")
    print("  3. Export as COCO JSON")
    print("  4. Replace annotations.json in the gold dir")
    print("  5. Run: python autoresearch/eval_gold.py")


if __name__ == "__main__":
    main()
