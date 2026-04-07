"""Phase 10a: Confidence sweep × LiDAR filter — recall lever experiment.

For each of the 10 reference patches:
  1. Run RF-DETR ONCE at the minimum threshold (0.01) to get a superset
     of all candidate detections.
  2. For each threshold T in [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]:
       a. Sub-filter the detections by `confidence >= T` (Python masking,
          no extra detection cost).
       b. For each mode in {raw, filter}:
            - raw   : evaluate detections directly
            - filter: apply Phase 9.5a `lidar_tree_top_filter` first
       c. Aggregate the results across all 10 patches via the existing
          `evaluate_patches_against_lidar` infrastructure.
  3. Print a side-by-side table and save CSV + Markdown to
     outputs/lidar_eval/.

Why this script exists
----------------------
Phases 8 and 9.5 proved that recall (6.8%) is the bottleneck on
LiDAR-verified F1, and that any post-detector filter is mathematically
ceilinged at F1 = 0.127. The cheapest possible recall lever — testable
with zero new code in production modules and zero retraining — is
RF-DETR's confidence threshold itself, currently hardcoded to 0.30
across the project. Lowering it releases more candidates, and the
9.5a filter (precision = 0.983) can clean up the resulting FPs.

Detect-once-then-subset trick
-----------------------------
RF-DETR's `threshold` parameter is a post-scoring filter applied
inside `model.predict()`, not a configuration that changes the forward
pass. So `predict(threshold=0.01)` returns a superset of every result
`predict(threshold=0.30)` would have produced. This means we can run
detection ONCE per patch and obtain all 6 confidence levels for free
via Python boolean indexing on `detections.confidence`.

Usage
-----
    python scripts/sweep_confidence.py
    python scripts/sweep_confidence.py --patch 0250.jpg --patch 0477.jpg
    python scripts/sweep_confidence.py --checkpoint checkpoints/round_1.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

# Make autoresearch importable as a top-level module path — same idiom
# as run_lidar_eval.py and train_classifier.py.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "autoresearch"))

from eval_lidar import evaluate_patches_against_lidar  # noqa: E402

from forest_pulse.detect import detect_trees  # noqa: E402
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

# Patch geometry — Montseny convention (640 px @ 0.25 m/px = 160 m).
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0

# Same 10-patch reference set as Phases 8 / 9.5a / 9.5b — directly
# back-comparable.
DEFAULT_PATCHES = [
    "0043.jpg", "0158.jpg", "0250.jpg", "0357.jpg", "0477.jpg",
    "0547.jpg", "0642.jpg", "0756.jpg", "0092.jpg", "0278.jpg",
]

# The sweep itself. Sorted high → low so the printed table reads
# from "current default" downward.
CONFIDENCE_LEVELS: list[float] = [0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
# Lowest level used for the single detection call per patch. Must
# equal min(CONFIDENCE_LEVELS).
MIN_DETECTION_CONF: float = 0.01


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Look up the geographic center of a patch from the metadata CSV."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def _build_low_conf_record(patch_name: str, checkpoint: str) -> dict | None:
    """Run detect_trees ONCE at the minimum threshold per patch.

    Returns a record containing the patch geometry, the LAZ tile path,
    and the low-confidence detections superset. Subsequent threshold
    levels are derived from this record without re-detecting.
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
        image, model_name=checkpoint, confidence=MIN_DETECTION_CONF,
    )
    # Strip rfdetr's source_shape / source_image keys NOW so every
    # subsequent boolean slice is safe.
    _strip_rfdetr_metadata(detections)

    # LAZ tile that contains this patch — already cached from prior
    # phases, so this is effectively free.
    laz_path = fetch_laz_for_patch(x_center, y_center)

    return {
        "name": patch_name,
        "image": image,
        "detections_low_conf": detections,
        "image_bounds": bounds,
        "image_size_px": image_size,
        "laz_path": laz_path,
    }


def _subset_by_confidence(
    detections: sv.Detections,
    threshold: float,
) -> sv.Detections:
    """Return the subset of detections at or above the given confidence.

    Pre-stripped detections (rfdetr metadata removed at build time) so
    boolean slicing is safe.
    """
    if len(detections) == 0 or detections.confidence is None:
        return detections
    mask = detections.confidence >= threshold
    return detections[mask]


def _build_eval_records(
    low_conf_records: list[dict],
    threshold: float,
    mode: str,
) -> list[dict]:
    """Shape per-patch records for `evaluate_patches_against_lidar`.

    For each low-conf record:
      1. Sub-filter detections to `confidence >= threshold`.
      2. If mode == 'filter', apply `lidar_tree_top_filter`.
      3. Wrap into the dict shape the eval expects.
    """
    eval_records: list[dict] = []
    for rec in low_conf_records:
        subset = _subset_by_confidence(
            rec["detections_low_conf"], threshold,
        )
        if mode == "filter":
            subset = lidar_tree_top_filter(
                subset,
                rec["image_bounds"],
                rec["image_size_px"],
                rec["laz_path"],
            )
        eval_records.append({
            "name": rec["name"],
            "detections": subset,
            "image_bounds": rec["image_bounds"],
            "image_size_px": rec["image_size_px"],
            "laz_path": rec["laz_path"],
        })
    return eval_records


def run_sweep(patches: list[str], checkpoint: str) -> list[dict]:
    """Top-level orchestration. Returns the list of result rows."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Confidence sweep on {len(patches)} patches × "
        f"{len(CONFIDENCE_LEVELS)} thresholds × 2 modes"
    )
    print(f"Checkpoint: {checkpoint}")
    print(f"Detection threshold (single call per patch): {MIN_DETECTION_CONF}")
    print()

    # Step 1: detect once per patch at the minimum threshold.
    low_conf_records: list[dict] = []
    for i, name in enumerate(patches, start=1):
        t0 = time.perf_counter()
        rec = _build_low_conf_record(name, checkpoint)
        elapsed = time.perf_counter() - t0
        if rec is None:
            continue
        n_low = len(rec["detections_low_conf"])
        print(
            f"  [{i}/{len(patches)}] {name} — "
            f"{n_low} candidate detections at conf >= {MIN_DETECTION_CONF} "
            f"({elapsed:.1f}s)"
        )
        low_conf_records.append(rec)

    if not low_conf_records:
        print("\nNo patches to evaluate.")
        return []

    print()
    print("Sweeping (mode, confidence) → eval...")

    # Step 2: for each (mode, threshold) combo, build eval records
    # from the cached low-conf supersets and run the eval.
    rows: list[dict] = []
    for mode in ("raw", "filter"):
        for threshold in CONFIDENCE_LEVELS:
            eval_records = _build_eval_records(
                low_conf_records, threshold, mode,
            )
            aggregate, _ = evaluate_patches_against_lidar(eval_records)
            row = {
                "mode": mode,
                "confidence": threshold,
                "n_pred": aggregate.n_predictions,
                "n_truth": aggregate.n_truth,
                "n_tp": aggregate.n_true_positive,
                "n_fp": aggregate.n_false_positive,
                "n_fn": aggregate.n_false_negative,
                "precision": aggregate.precision,
                "recall": aggregate.recall,
                "f1": aggregate.f1,
            }
            rows.append(row)
            logger.info(
                "mode=%s conf=%.2f → pred=%d TP=%d P=%.3f R=%.3f F1=%.4f",
                mode, threshold, row["n_pred"], row["n_tp"],
                row["precision"], row["recall"], row["f1"],
            )

    return rows


def _print_table(rows: list[dict]) -> None:
    """Pretty-print the sweep results as an aligned ASCII table."""
    print()
    print("=" * 78)
    print(
        "  mode    conf   n_pred  TP   FP    FN     "
        "P       R       F1"
    )
    print("-" * 78)
    last_mode = None
    for row in rows:
        # Insert a blank line between modes for readability
        if last_mode is not None and row["mode"] != last_mode:
            print("-" * 78)
        print(
            f"  {row['mode']:<6}  {row['confidence']:.2f}  "
            f"{row['n_pred']:>5}  {row['n_tp']:>4} {row['n_fp']:>4} "
            f"{row['n_fn']:>5}  "
            f"{row['precision']:>6.3f}  {row['recall']:>6.3f}  "
            f"{row['f1']:>7.4f}"
        )
        last_mode = row["mode"]
    print("=" * 78)

    # Highlight the best row across the whole sweep.
    best = max(rows, key=lambda r: r["f1"])
    print(
        f"  BEST: mode={best['mode']}  conf={best['confidence']:.2f}  "
        f"F1={best['f1']:.4f}  (P={best['precision']:.3f}, "
        f"R={best['recall']:.3f})"
    )
    print()
    print("  Reference: Phase 8 baseline F1 = 0.108 | "
          "Phase 9.5a filter F1 = 0.127")


def _save_csv(rows: list[dict], path: Path) -> None:
    """Save the sweep results as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode", "confidence", "n_pred", "n_truth",
                "n_tp", "n_fp", "n_fn",
                "precision", "recall", "f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved sweep CSV to %s", path)


def _save_markdown(rows: list[dict], path: Path) -> None:
    """Save the sweep results as a Markdown table for progress.txt embedding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| mode | conf | n_pred | TP | FP | FN | P | R | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['confidence']:.2f} | "
            f"{row['n_pred']} | {row['n_tp']} | {row['n_fp']} | "
            f"{row['n_fn']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | **{row['f1']:.4f}** |"
        )
    path.write_text("\n".join(lines) + "\n")
    logger.info("Saved sweep Markdown to %s", path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Phase 10a: confidence × LiDAR-filter sweep.",
    )
    parser.add_argument(
        "--patch", action="append", default=None,
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
    args = parser.parse_args()

    patches = args.patch if args.patch else DEFAULT_PATCHES
    rows = run_sweep(patches, args.checkpoint)
    if not rows:
        return

    _print_table(rows)
    _save_csv(rows, OUTPUT_DIR / "confidence_sweep.csv")
    _save_markdown(rows, OUTPUT_DIR / "confidence_sweep.md")


if __name__ == "__main__":
    main()
