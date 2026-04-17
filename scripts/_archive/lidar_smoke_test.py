"""LiDAR smoke test — end-to-end verification on one Montseny patch.

Pipeline:
  1. Look up patch center from metadata CSV
  2. Run RF-DETR detection on the patch
  3. Download the 1 km LAZ tile for the patch (first run only)
  4. Extract per-tree LiDAR features for each detection
  5. Print a table of the first N trees' features
  6. Compute a CHM for the patch bounds
  7. Apply filter_by_height() and report kept/dropped counts

This verifies the full LiDAR integration works on real data. First run
downloads ~400-700 MB — subsequent runs are cached and fast.

Usage:
    python scripts/lidar_smoke_test.py                     # patch 0250
    python scripts/lidar_smoke_test.py --patch 0477.jpg
    python scripts/lidar_smoke_test.py --patch 0250.jpg --top 10
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees
from forest_pulse.lidar import (
    DEFAULT_HEIGHT_THRESHOLD,
    compute_chm_from_laz,
    extract_lidar_features,
    fetch_laz_for_patch,
    filter_by_height,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"

PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Look up the geographic center of a patch from the metadata CSV."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def run_smoke_test(patch_name: str, top_n: int) -> None:
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")

    x_center, y_center = _get_patch_center(patch_name)
    half = PATCH_SIZE_M / 2
    bounds = (
        x_center - half, y_center - half,
        x_center + half, y_center + half,
    )
    size = (PATCH_SIZE_PX, PATCH_SIZE_PX)
    logger.info("Patch: %s at (%.0f, %.0f)", patch_name, x_center, y_center)

    # Stage 1: detect
    image = np.array(Image.open(patch_path).convert("RGB"))
    t0 = time.perf_counter()
    detections = detect_trees(
        image, model_name=str(CHECKPOINT), confidence=0.3,
    )
    t_detect = time.perf_counter() - t0
    logger.info("Detected %d trees in %.2fs", len(detections), t_detect)

    if len(detections) == 0:
        print(f"No trees detected in {patch_name}.")
        return

    # Stage 2: download LAZ (cached after first run)
    t0 = time.perf_counter()
    laz_path = fetch_laz_for_patch(x_center, y_center)
    t_laz = time.perf_counter() - t0
    laz_mb = laz_path.stat().st_size / 1e6
    logger.info("LAZ: %s (%.0f MB) in %.2fs", laz_path.name, laz_mb, t_laz)

    # Stage 3: extract features
    t0 = time.perf_counter()
    features = extract_lidar_features(detections, bounds, size, laz_path)
    t_feat = time.perf_counter() - t0
    logger.info(
        "Extracted LiDAR features for %d trees in %.2fs", len(features), t_feat,
    )

    # Stage 4: CHM + height filter
    t0 = time.perf_counter()
    chm_path = compute_chm_from_laz(laz_path, bounds, resolution_m=0.5)
    filtered = filter_by_height(
        detections, chm_path, bounds, size,
        threshold=DEFAULT_HEIGHT_THRESHOLD, aggregation="max",
    )
    t_filter = time.perf_counter() - t0
    logger.info(
        "CHM + filter took %.2fs, kept %d/%d trees",
        t_filter, len(filtered), len(detections),
    )

    # Summary table
    print(f"\n{'=' * 78}")
    print(f"  LiDAR Smoke Test: {patch_name}")
    print(f"{'=' * 78}")
    print(f"  Detections:         {len(detections)}")
    print(f"  LAZ tile:           {laz_path.name}  ({laz_mb:.0f} MB)")
    print(f"  Kept after >5m:     {len(filtered)}")
    print(f"  Dropped (< 5m):     {len(detections) - len(filtered)}")
    print()
    print(f"  Top {top_n} trees by LiDAR features:")
    print(f"  {'id':>3} {'h_p95':>8} {'h_p50':>8} {'spread':>8} "
          f"{'pts':>6} {'ret_ratio':>10} {'int_mean':>10} {'int_std':>10}")
    for f in features[:top_n]:
        print(
            f"  {f.tree_id:>3} "
            f"{f.height_p95_m:>8.2f} "
            f"{f.height_p50_m:>8.2f} "
            f"{f.vertical_spread_m:>8.2f} "
            f"{f.point_count:>6} "
            f"{f.return_ratio:>10.3f} "
            f"{f.intensity_mean:>10.1f} "
            f"{f.intensity_std:>10.1f}"
        )
    print(f"{'=' * 78}")
    print(f"  Timings — detect: {t_detect:.1f}s | laz: {t_laz:.1f}s "
          f"| features: {t_feat:.1f}s | chm+filter: {t_filter:.1f}s")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="LiDAR smoke test.")
    parser.add_argument(
        "--patch", default="0250.jpg",
        help="Patch filename in data/montseny/patches/.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="Print features for the top N trees.",
    )
    args = parser.parse_args()
    run_smoke_test(args.patch, args.top)


if __name__ == "__main__":
    main()
