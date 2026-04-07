"""Demo: detect → NDVI filter → LiDAR filter on a Montseny patch.

Runs the full verification pipeline on a single patch to show how many
detections survive each filter stage. Saves annotated images at each
stage so you can visually compare.

Pipeline:
    RF-DETR detects crown boxes
        → NDVI filter drops non-vegetation (rocks, roads, roofs)
            → LiDAR filter drops short vegetation (bushes, shrubs)
                → verified tree list

Usage:
    python scripts/apply_filters_demo.py --patch 0477.jpg
    python scripts/apply_filters_demo.py --patch 0250.jpg --ndvi 0.2 --height 6
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees
from forest_pulse.lidar import fetch_chm_for_patch, filter_by_height
from forest_pulse.ndvi import fetch_ndvi_for_patch, filter_by_ndvi
from forest_pulse.patches import get_patch_center
from forest_pulse.visualize import annotate_trees

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "filter_demo"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"

# Patches are 640 px square at 0.25 m/px = 160 m real-world
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0


def _save_annotated(image, detections, name: str) -> None:
    """Save an annotated image to outputs/filter_demo/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vis = annotate_trees(image, detections, health_scores=None)
    Image.fromarray(vis).save(OUTPUT_DIR / name)


def run_demo(patch_name: str, ndvi_threshold: float, height_threshold: float) -> None:
    """Run the full filter pipeline on one patch and report counts."""
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")

    # Patch geographic bounds from metadata
    x_center, y_center = get_patch_center(METADATA_CSV, patch_name)
    half = PATCH_SIZE_M / 2
    image_bounds = (
        x_center - half,
        y_center - half,
        x_center + half,
        y_center + half,
    )
    image_size = (PATCH_SIZE_PX, PATCH_SIZE_PX)

    logger.info("Patch: %s at (%.0f, %.0f)", patch_name, x_center, y_center)

    # Stage 1: raw detection
    image = np.array(Image.open(patch_path).convert("RGB"))
    detections = detect_trees(
        image, model_name=str(CHECKPOINT), confidence=0.3,
    )
    n_raw = len(detections)
    logger.info("Stage 1 — raw detections: %d", n_raw)
    _save_annotated(image, detections, f"{patch_name}_1_raw.png")

    # Stage 2: NDVI filter (drops rocks, roads, buildings)
    logger.info("Stage 2 — fetching NDVI raster...")
    ndvi_path = fetch_ndvi_for_patch(x_center, y_center, PATCH_SIZE_M)
    detections_ndvi = filter_by_ndvi(
        detections, ndvi_path, image_bounds, image_size,
        threshold=ndvi_threshold,
    )
    n_ndvi = len(detections_ndvi)
    _save_annotated(image, detections_ndvi, f"{patch_name}_2_ndvi.png")

    # Stage 3: LiDAR height filter (drops shrubs, bushes)
    # Best-effort — skips gracefully if no CHM raster is available.
    # See src/forest_pulse/lidar.py for how to obtain one.
    logger.info("Stage 3 — looking up LiDAR CHM...")
    lidar_available = False
    try:
        chm_path = fetch_chm_for_patch(x_center, y_center, PATCH_SIZE_M)
        detections_final = filter_by_height(
            detections_ndvi, chm_path, image_bounds, image_size,
            threshold=height_threshold, aggregation="max",
        )
        lidar_available = True
    except NotImplementedError as e:
        logger.warning("LiDAR filter skipped: %s", str(e).split(chr(10))[0])
        detections_final = detections_ndvi

    n_final = len(detections_final)
    _save_annotated(image, detections_final, f"{patch_name}_3_final.png")

    # Summary
    print(f"\n{'='*55}")
    print(f"  Filter Pipeline: {patch_name}")
    print(f"{'='*55}")
    print(f"  Raw detections:       {n_raw}")
    print(f"  After NDVI filter:    {n_ndvi}  "
          f"(-{n_raw - n_ndvi} non-vegetation)")
    if lidar_available:
        print(f"  After LiDAR filter:   {n_final}  "
              f"(-{n_ndvi - n_final} short vegetation)")
    else:
        print("  After LiDAR filter:   SKIPPED (no CHM raster available)")
    print(f"  Final tree count:     {n_final}")
    print(f"{'='*55}")
    print(f"  Annotated images → {OUTPUT_DIR}")
    if not lidar_available:
        print("\n  Note: Install/generate a CHM raster to enable LiDAR")
        print("  filtering. See src/forest_pulse/lidar.py docstring.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run NDVI + LiDAR filter pipeline on a patch.",
    )
    parser.add_argument(
        "--patch", default="0477.jpg",
        help="Patch filename in data/montseny/patches/ (default: 0477.jpg).",
    )
    parser.add_argument(
        "--ndvi", type=float, default=0.15,
        help="Minimum NDVI to keep a detection (default: 0.15).",
    )
    parser.add_argument(
        "--height", type=float, default=5.0,
        help="Minimum canopy height in meters (default: 5.0).",
    )
    args = parser.parse_args()

    run_demo(args.patch, args.ndvi, args.height)


if __name__ == "__main__":
    main()
