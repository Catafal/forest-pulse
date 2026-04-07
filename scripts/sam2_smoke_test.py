"""SAM2 smoke test — verify the integration loads and runs on MPS.

This script is standalone: no RF-DETR, no project data required. It
loads SAM2, runs both box-prompted refinement and automatic mode on a
synthetic image, prints counts and timings, and exits.

Run this before using SAM2 in the full pipeline to confirm the model
downloads, loads, and infers without MPS errors.

Usage:
    python scripts/sam2_smoke_test.py
    python scripts/sam2_smoke_test.py --model facebook/sam2.1-hiera-tiny
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from forest_pulse.segment import (
    DEFAULT_MODEL_ID,
    refine_detections_with_sam2,
    segment_all_trees_sam2,
)


def _make_synthetic_image() -> np.ndarray:
    """Build a 400x400 test image with a few bright disc blobs on dark bg."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (30, 60, 30)  # dark green background

    # Three "crown-like" discs at different positions
    centers = [(100, 100, 40), (250, 150, 55), (200, 300, 35)]
    yy, xx = np.mgrid[:400, :400]
    for cx, cy, r in centers:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = (60, 160, 60)  # brighter green disc
    return img


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="SAM2 smoke test.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID,
        help=f"HuggingFace SAM2 model ID (default: {DEFAULT_MODEL_ID}).",
    )
    args = parser.parse_args()

    image = _make_synthetic_image()
    print(f"Synthetic image: {image.shape}, dtype={image.dtype}")

    # --- Test 1: box-prompted refinement ---
    print("\n[1/2] Box-prompted refinement...")
    import supervision as sv
    dets = sv.Detections(
        xyxy=np.array([
            [60, 60, 140, 140],
            [195, 95, 305, 205],
        ], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
    )
    start = time.perf_counter()
    refined = refine_detections_with_sam2(image, dets, model_id=args.model)
    t_refine = time.perf_counter() - start
    assert refined.mask is not None, "refined.mask is None!"
    print(f"  {len(refined)} detections refined")
    print(f"  mask shape: {refined.mask.shape}")
    print(f"  mask pixel sums: {[int(m.sum()) for m in refined.mask]}")
    print(f"  took {t_refine:.2f}s")

    # --- Test 2: automatic mode ---
    print("\n[2/2] Automatic mask generation...")
    start = time.perf_counter()
    auto = segment_all_trees_sam2(
        image, model_id=args.model, points_per_side=16,
    )
    t_auto = time.perf_counter() - start
    print(f"  {len(auto)} segments kept after crown filter")
    print(f"  took {t_auto:.2f}s")

    print("\n" + "=" * 45)
    print("  SAM2 smoke test passed")
    print("=" * 45)
    print(f"  Model:     {args.model}")
    print(f"  Refine:    {t_refine:.2f}s for 2 boxes")
    print(f"  Auto mode: {t_auto:.2f}s for 256 points")


if __name__ == "__main__":
    main()
