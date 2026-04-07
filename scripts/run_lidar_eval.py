"""Run LiDAR-verified evaluation on a set of Montseny patches.

End-to-end runner that produces the project's first physically-grounded
evaluation metric. For each patch:

  1. Look up the patch geographic center from the metadata CSV
  2. Run RF-DETR detection
  3. Download the ICGC LAZ tile (cached after first run)
  4. Compute CHM, find tree-top truth via local-max filtering
  5. Match detections to truth, report precision / recall / F1

Aggregates across patches via micro-average. Saves a per-patch CSV.

Usage:
    # Default 10-patch reference set (one per Montseny zone)
    python scripts/run_lidar_eval.py

    # Custom patches
    python scripts/run_lidar_eval.py --patch 0250.jpg --patch 0477.jpg

    # Different checkpoint
    python scripts/run_lidar_eval.py --checkpoint checkpoints/round_1.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make autoresearch importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch"))

from eval_lidar import evaluate_patches_against_lidar  # noqa: E402

from forest_pulse.classifier import (  # noqa: E402
    TreeClassifier,
    extract_classifier_features,
    load_classifier,
    predict_tree_probabilities_batch,
)
from forest_pulse.detect import detect_trees  # noqa: E402
from forest_pulse.health import score_health  # noqa: E402
from forest_pulse.lidar import (  # noqa: E402
    _strip_rfdetr_metadata,
    fetch_laz_for_patch,
    lidar_tree_top_filter,
)

logger = logging.getLogger(__name__)

PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lidar_eval"

# Patch geometry — same as the rest of the project (640 px @ 0.25 m/px).
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0

# Default reference set: same 10 patches as the SAM2 A/B test.
# One representative patch per Montseny zone, plus extras for diversity.
DEFAULT_PATCHES = [
    "0043.jpg", "0158.jpg", "0250.jpg", "0357.jpg", "0477.jpg",
    "0547.jpg", "0642.jpg", "0756.jpg", "0092.jpg", "0278.jpg",
]


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Look up the geographic center of a patch from the metadata CSV."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def _build_patch_record(patch_name: str, checkpoint: str) -> dict | None:
    """Run detection + LAZ download for one patch.

    Returns a dict suitable for `evaluate_patches_against_lidar`, or
    None if the patch is missing.
    """
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
    detections = detect_trees(
        image, model_name=checkpoint, confidence=0.3,
    )

    # Download (or look up) the LAZ tile that contains this patch.
    laz_path = fetch_laz_for_patch(x_center, y_center)

    return {
        "name": patch_name,
        "image": image,
        "detections": detections,
        "image_bounds": bounds,
        "image_size_px": image_size,
        "laz_path": laz_path,
    }


def _apply_classifier(rec: dict, classifier: TreeClassifier) -> "sv.Detections":  # noqa: F821
    """Apply a trained classifier to a patch record's detections.

    Scores GRVI/ExG via score_health, extracts the 11-feature RGB
    vector per detection, batch-predicts tree probabilities, and
    keeps only those at or above `classifier.threshold`.

    Empty detections pass through unchanged.
    """
    detections = rec["detections"]
    if len(detections) == 0:
        return detections

    # Strip rfdetr's source_shape / source_image fields before any
    # boolean slicing — supervision indexes .data and those fields
    # have length 640 (image height), not n_detections.
    _strip_rfdetr_metadata(detections)

    image = rec["image"]
    health_scores = score_health(image, detections)

    feature_dicts = []
    for i in range(len(detections)):
        confidence = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        feature_dicts.append(extract_classifier_features(
            image=image,
            bbox_xyxy=detections.xyxy[i],
            bbox_confidence=confidence,
            health=health_scores[i],
        ))

    probs = predict_tree_probabilities_batch(classifier, feature_dicts)
    keep_mask = probs >= classifier.threshold
    return detections[keep_mask]


def run_evaluation(
    patches: list[str],
    checkpoint: str,
    mode: str = "baseline",
    classifier_path: Path | None = None,
) -> None:
    """Build records for each patch and run the LiDAR evaluator.

    Args:
        patches: Patch filenames to evaluate.
        checkpoint: RF-DETR checkpoint path.
        mode: One of "baseline" (no post-processing), "filter" (apply
            lidar_tree_top_filter before eval), or "classifier" (apply
            the trained RGB classifier before eval).
        classifier_path: Required when mode=="classifier". Path to the
            joblib file produced by `scripts/train_classifier.py`.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the classifier ONCE if needed — cheap joblib.load.
    classifier: TreeClassifier | None = None
    if mode == "classifier":
        if classifier_path is None:
            raise ValueError("classifier mode requires --classifier PATH")
        classifier = load_classifier(classifier_path)
        print(f"Loaded classifier from {classifier_path}")
        print(
            f"  Features: {len(classifier.feature_names)}, "
            f"threshold: {classifier.threshold:.2f}"
        )

    print(f"Running LiDAR-verified evaluation on {len(patches)} patches...")
    print(f"Checkpoint: {checkpoint}")
    print(f"Mode: {mode}")
    print()

    records = []
    for i, name in enumerate(patches, start=1):
        t0 = time.perf_counter()
        rec = _build_patch_record(name, checkpoint)
        elapsed = time.perf_counter() - t0
        if rec is None:
            continue

        if mode == "filter":
            # Phase 9.5a: deterministic LiDAR tree-top filter. Runs on
            # the same CHM the eval will compute a moment later, so the
            # cost is essentially a single extra numpy distance pass.
            n_before = len(rec["detections"])
            rec["detections"] = lidar_tree_top_filter(
                rec["detections"],
                rec["image_bounds"],
                rec["image_size_px"],
                rec["laz_path"],
            )
            n_after = len(rec["detections"])
            print(
                f"  [{i}/{len(patches)}] {name} — "
                f"{n_before} → {n_after} detections after filter "
                f"({elapsed:.1f}s)"
            )
        elif mode == "classifier":
            # Phase 9.5b: learned RGB classifier as a precision filter.
            n_before = len(rec["detections"])
            rec["detections"] = _apply_classifier(rec, classifier)
            n_after = len(rec["detections"])
            print(
                f"  [{i}/{len(patches)}] {name} — "
                f"{n_before} → {n_after} detections after classifier "
                f"({elapsed:.1f}s)"
            )
        else:
            n_dets = len(rec["detections"])
            print(
                f"  [{i}/{len(patches)}] {name} — {n_dets} detections "
                f"({elapsed:.1f}s)"
            )
        records.append(rec)

    if not records:
        print("\nNo patches to evaluate.")
        return

    print()
    print("Computing LiDAR ground truth + matching...")
    aggregate, per_patch = evaluate_patches_against_lidar(records)

    # Per-patch table to stdout
    print()
    print(f"{'=' * 88}")
    print(f"  LiDAR-Verified Evaluation — {len(records)} patches")
    print(f"{'=' * 88}")
    print(
        f"  {'patch':<14} {'pred':>5} {'truth':>6} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} "
        f"{'P':>7} {'R':>7} {'F1':>7}"
    )
    print(f"  {'-' * 84}")
    for r in per_patch:
        print(
            f"  {r['name']:<14} {r['n_predictions']:>5} {r['n_truth']:>6} "
            f"{r['n_tp']:>4} {r['n_fp']:>4} {r['n_fn']:>4} "
            f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}"
        )
    print(f"  {'-' * 84}")
    print(
        f"  {'AGGREGATE':<14} "
        f"{aggregate.n_predictions:>5} {aggregate.n_truth:>6} "
        f"{aggregate.n_true_positive:>4} {aggregate.n_false_positive:>4} "
        f"{aggregate.n_false_negative:>4} "
        f"{aggregate.precision:>7.3f} {aggregate.recall:>7.3f} "
        f"{aggregate.f1:>7.3f}"
    )
    print(f"{'=' * 88}")
    print()
    print(f"  Honest baseline F1: {aggregate.f1:.4f}")
    print("  Compare to inflated mAP50: 0.904 (self-trained labels)")

    # Save CSV — one file per mode so the three modes (baseline,
    # filter, classifier) can coexist without clobbering each other.
    csv_name = (
        "eval_summary.csv" if mode == "baseline"
        else f"eval_summary_{mode}.csv"
    )
    csv_path = OUTPUT_DIR / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name", "n_predictions", "n_truth",
                "n_tp", "n_fp", "n_fn",
                "precision", "recall", "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(per_patch)
        writer.writerow({
            "name": "AGGREGATE",
            "n_predictions": aggregate.n_predictions,
            "n_truth": aggregate.n_truth,
            "n_tp": aggregate.n_true_positive,
            "n_fp": aggregate.n_false_positive,
            "n_fn": aggregate.n_false_negative,
            "precision": aggregate.precision,
            "recall": aggregate.recall,
            "f1": aggregate.f1,
        })
    print(f"  CSV: {csv_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="LiDAR-verified evaluation runner.",
    )
    parser.add_argument(
        "--patch",
        action="append",
        default=None,
        help=(
            "Patch filename(s) in data/montseny/patches/. "
            "Repeat for multiple. Default = 10-patch reference set."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Path to RF-DETR checkpoint.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--filter",
        action="store_true",
        help=(
            "Apply the deterministic LiDAR tree-top filter "
            "(lidar_tree_top_filter) before eval. Produces the "
            "reference upper bound for any post-processor."
        ),
    )
    mode_group.add_argument(
        "--classifier",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a trained classifier joblib (produced by "
            "scripts/train_classifier.py). Applied as a precision "
            "filter before eval — measures the RGB distillation "
            "hypothesis against the filter upper bound."
        ),
    )
    args = parser.parse_args()

    patches = args.patch if args.patch else DEFAULT_PATCHES
    if args.filter:
        mode = "filter"
        classifier_path = None
    elif args.classifier:
        mode = "classifier"
        classifier_path = Path(args.classifier)
    else:
        mode = "baseline"
        classifier_path = None
    run_evaluation(
        patches, args.checkpoint,
        mode=mode, classifier_path=classifier_path,
    )


if __name__ == "__main__":
    main()
