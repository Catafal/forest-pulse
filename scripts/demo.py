"""Quick demo — detect trees and visualize with health colors.

Usage:
    python scripts/demo.py --image path/to/aerial.tif
    python scripts/demo.py --image path/to/aerial.tif --output outputs/annotated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Forest Pulse quick demo.")
    parser.add_argument("--image", required=True, help="Path to aerial RGB image.")
    parser.add_argument("--output", default="outputs/demo_result.png", help="Output path.")
    parser.add_argument("--model", default="deepforest", help="Detection model.")
    parser.add_argument("--confidence", type=float, default=0.3, help="Min confidence.")
    args = parser.parse_args()

    # TODO: Implement demo pipeline once modules are ready
    # 1. Load image
    # 2. detect_trees(image, model_name=args.model, confidence=args.confidence)
    # 3. score_health(image, detections)
    # 4. annotate_trees(image, detections, health_scores)
    # 5. Save annotated image

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("Demo not yet implemented — run notebooks/01_quickstart.ipynb for now.")


if __name__ == "__main__":
    main()
