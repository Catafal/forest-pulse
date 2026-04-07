"""Train the post-detector tree classifier on the 10-patch reference set.

End-to-end training driver. For each patch in the reference set:

  1. Look up the patch geographic center
  2. Run RF-DETR detection
  3. Score health (GRVI / ExG) for each detection
  4. Download (or reuse cached) LAZ tile
  5. Extract per-tree LiDAR features

Then auto-label every detection via LiDAR height (>5 m = tree, <2 m =
bush, in-between excluded), build the multi-modal feature matrix, run
a leakage-safe patch-level 8/2 split, train a sklearn
GradientBoostingClassifier, save the model + a CSV report.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --test-patch 0250.jpg --test-patch 0477.jpg
    python scripts/train_classifier.py --checkpoint checkpoints/round_1.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.classifier import (
    build_training_examples,
    save_classifier,
    train_tree_classifier_patch_split,
)
from forest_pulse.detect import detect_trees
from forest_pulse.health import score_health
from forest_pulse.lidar import extract_lidar_features, fetch_laz_for_patch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"
CLASSIFIER_OUTPUT_PATH = PROJECT_ROOT / "checkpoints" / "tree_classifier.joblib"
REPORT_DIR = PROJECT_ROOT / "outputs" / "classifier"

PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0

# Same 10-patch reference set as the SAM2 A/B test and Phase 8 eval
DEFAULT_PATCHES = [
    "0043.jpg", "0158.jpg", "0250.jpg", "0357.jpg", "0477.jpg",
    "0547.jpg", "0642.jpg", "0756.jpg", "0092.jpg", "0278.jpg",
]
# Patches reserved for the test split (leakage-safe)
DEFAULT_TEST_PATCHES = ["0092.jpg", "0278.jpg"]


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Look up the geographic center of a patch from the metadata CSV."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def _build_patch_record(patch_name: str, checkpoint: str) -> dict | None:
    """Run the full detect → health → LiDAR-features pipeline for one patch."""
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        logger.error("Patch missing: %s", patch_path)
        return None

    x_center, y_center = _get_patch_center(patch_name)
    half = PATCH_SIZE_M / 2.0
    bounds = (
        x_center - half, y_center - half,
        x_center + half, y_center + half,
    )
    image_size = (PATCH_SIZE_PX, PATCH_SIZE_PX)

    image = np.array(Image.open(patch_path).convert("RGB"))
    detections = detect_trees(image, model_name=checkpoint, confidence=0.3)
    if len(detections) == 0:
        return None

    health_scores = score_health(image, detections)
    laz_path = fetch_laz_for_patch(x_center, y_center)
    lidar_features = extract_lidar_features(
        detections, bounds, image_size, laz_path,
    )

    return {
        "name": patch_name,
        "image": image,
        "detections": detections,
        "health_scores": health_scores,
        "lidar_features": lidar_features,
        "bounds": bounds,
        "image_size": image_size,
    }


def _print_metrics(metrics: dict) -> None:
    """Pretty-print the training metrics + feature importance."""
    print(f"\n{'=' * 72}")
    print("  Tree Classifier Training Report")
    print(f"{'=' * 72}")
    print(f"  Train: {metrics['n_train']} examples"
          f" ({metrics['n_train_tree']} tree, {metrics['n_train_bush']} bush)")
    print(f"  Test:  {metrics['n_test']} examples"
          f" ({metrics['n_test_tree']} tree, {metrics['n_test_bush']} bush)")
    print()
    print("  Test set metrics:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print()
    print("  Feature importance (top 10):")
    for name, imp in metrics["feature_importance"][:10]:
        bar = "#" * int(imp * 50)
        print(f"    {name:<26} {imp:>7.4f}  {bar}")
    print(f"{'=' * 72}")


def _save_report_csv(metrics: dict, path: Path) -> None:
    """Save the metrics dict to a CSV report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in [
            "n_train", "n_test",
            "n_train_tree", "n_train_bush",
            "n_test_tree", "n_test_bush",
            "accuracy", "precision", "recall", "f1",
        ]:
            writer.writerow([key, metrics[key]])
        writer.writerow([])
        writer.writerow(["feature", "importance"])
        for name, imp in metrics["feature_importance"]:
            writer.writerow([name, imp])
    logger.info("Saved training report to %s", path)


def run_training(
    patches: list[str],
    test_patches: list[str],
    checkpoint: str,
) -> None:
    """Build patch records, train the classifier, save model + report."""
    print(f"Training tree classifier on {len(patches)} patches "
          f"({len(test_patches)} test)")
    print(f"Checkpoint: {checkpoint}")
    print()

    records: list[dict] = []
    for i, name in enumerate(patches, start=1):
        t0 = time.perf_counter()
        rec = _build_patch_record(name, checkpoint)
        elapsed = time.perf_counter() - t0
        if rec is None:
            print(f"  [{i}/{len(patches)}] {name} — skipped (no detections)")
            continue
        n_dets = len(rec["detections"])
        print(
            f"  [{i}/{len(patches)}] {name} — {n_dets} detections "
            f"({elapsed:.1f}s)"
        )
        records.append(rec)

    print()
    print("Building auto-labeled training examples...")
    examples = build_training_examples(records)
    if not examples:
        print("No labeled examples produced. Aborting.")
        return

    n_tree = sum(1 for e in examples if e.label == 1)
    n_bush = sum(1 for e in examples if e.label == 0)
    print(f"  Total examples: {len(examples)} ({n_tree} tree, {n_bush} bush)")

    print()
    print("Training GradientBoostingClassifier (patch-level split)...")
    classifier, metrics = train_tree_classifier_patch_split(
        examples, test_patch_names=set(test_patches),
    )

    _print_metrics(metrics)

    save_classifier(classifier, CLASSIFIER_OUTPUT_PATH)
    _save_report_csv(metrics, REPORT_DIR / "training_report.csv")
    print()
    print(f"  Model:  {CLASSIFIER_OUTPUT_PATH}")
    print(f"  Report: {REPORT_DIR / 'training_report.csv'}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train the post-detector tree classifier.",
    )
    parser.add_argument(
        "--patch", action="append", default=None,
        help=(
            "Patch filename(s) in data/montseny/patches/. "
            "Repeat for multiple. Default = 10-patch reference set."
        ),
    )
    parser.add_argument(
        "--test-patch", action="append", default=None,
        help=(
            "Patch filename(s) reserved for the test split. "
            "Default = ['0092.jpg', '0278.jpg']."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to RF-DETR checkpoint.",
    )
    args = parser.parse_args()

    patches = args.patch if args.patch else DEFAULT_PATCHES
    test_patches = args.test_patch if args.test_patch else DEFAULT_TEST_PATCHES
    run_training(patches, test_patches, args.checkpoint)


if __name__ == "__main__":
    main()
