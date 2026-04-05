"""Quick demo — detect trees and visualize with health colors.

Usage:
    python scripts/demo.py --image path/to/aerial.tif
    python scripts/demo.py --image path/to/aerial.tif --output outputs/result.png
    python scripts/demo.py --sample  # uses DeepForest's bundled test image
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees
from forest_pulse.health import score_health
from forest_pulse.visualize import annotate_trees

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Forest Pulse — tree detection demo.")
    parser.add_argument("--image", help="Path to aerial RGB image.")
    parser.add_argument("--output", default="outputs/demo_result.png", help="Output path.")
    parser.add_argument("--model", default="deepforest", help="Detection model name.")
    parser.add_argument("--confidence", type=float, default=0.3, help="Min confidence (0-1).")
    parser.add_argument("--sample", action="store_true", help="Use DeepForest bundled sample.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    # Resolve image path
    image_path = _resolve_image_path(args)
    if image_path is None:
        sys.exit(1)

    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    logger.info("Image loaded: %s — shape %s", image_path.name, image.shape)

    # Run the full pipeline: detect → health → visualize
    pipeline_start = time.perf_counter()

    detections = detect_trees(image, model_name=args.model, confidence=args.confidence)
    if len(detections) == 0:
        print("No trees detected. Try lowering --confidence or using a different image.")
        sys.exit(0)

    health_scores = score_health(image, detections)
    annotated = annotate_trees(image, detections, health_scores)

    elapsed = time.perf_counter() - pipeline_start

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(output_path)

    # Print summary
    distribution = Counter(hs.label for hs in health_scores)
    print(f"\n{'='*50}")
    print("  Forest Pulse — Detection Complete")
    print(f"{'='*50}")
    print(f"  Trees detected:  {len(detections)}")
    for label in ["healthy", "stressed", "dead", "unknown"]:
        if label in distribution:
            print(f"    {label:>10}: {distribution[label]}")
    print(f"  Pipeline time:   {elapsed:.2f}s")
    print(f"  Output saved:    {output_path}")
    print(f"{'='*50}\n")


def _resolve_image_path(args) -> Path | None:
    """Determine which image to use based on CLI args."""
    if args.sample:
        # Use DeepForest's bundled test image — zero download needed
        try:
            from deepforest import get_data
            sample_path = Path(get_data("OSBS_029.png"))
            logger.info("Using DeepForest sample image: %s", sample_path)
            return sample_path
        except (ImportError, Exception) as e:
            logger.error("Could not load DeepForest sample: %s", e)
            print("Run: python scripts/download_data.py --sample-image")
            return None

    if args.image:
        path = Path(args.image)
        if not path.exists():
            logger.error("Image not found: %s", path)
            return None
        return path

    print("Provide --image <path> or --sample")
    return None


if __name__ == "__main__":
    main()
