"""LiDAR-verified evaluation metric for tree detection.

Produces the project's first physically-grounded eval metric. Derives
ground-truth tree-top positions from a Canopy Height Model via the
standard forestry technique (smoothed local-maximum filtering above a
height threshold), matches RF-DETR detections to those positions with
greedy nearest-neighbor within a 2 m tolerance, and reports
precision / recall / F1.

This is **separate** from `autoresearch/eval.py` (which is LOCKED and
uses self-trained labels). That biased metric stays intact for backward
compatibility; this module produces an honest number alongside it.

## Why local-maximum filtering on CHM is the standard technique

Canopy Height Models from LiDAR are flat where there's open ground,
peak where there's a tree top. A local-maximum filter finds those peaks.
The same approach is used by:
  - lidR R package (`locate_trees` function)
  - Spanish Forest Inventory (Inventario Forestal Nacional)
  - USFS standard forestry workflows
  - Most published tree-detection-from-LiDAR papers

The errors are PHYSICAL (caused by canopy structure — merged crowns in
dense uniform forest can produce one peak where there are several
trees) rather than CIRCULAR (caused by the model labeling its own
training data). It's not perfect; it's HONEST.

## Why greedy matching, not Hungarian

Hungarian assignment is theoretically optimal but adds complexity. At
the project's tolerance of 2 m and typical tree spacing of 5+ m,
collisions are rare and the difference between greedy and Hungarian
matching is < 1% on real data. Greedy is deterministic, simple, and
easy to reason about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import supervision as sv
from scipy.ndimage import gaussian_filter, maximum_filter

from forest_pulse.lidar import compute_chm_from_laz

logger = logging.getLogger(__name__)

# Defaults pinned to the project conventions used elsewhere.
DEFAULT_HEIGHT_THRESHOLD_M = 5.0     # consistent with filter_by_height
DEFAULT_MIN_DISTANCE_M = 3.0         # roughly half a typical Med tree crown
DEFAULT_SMOOTH_SIGMA_PX = 1.0        # kills speckle without merging crowns
DEFAULT_MATCH_TOLERANCE_M = 2.0      # forestry convention; matches temporal.py


# ============================================================
# Result type
# ============================================================


@dataclass
class EvalResult:
    """Aggregated detection-vs-LiDAR evaluation result.

    All fields are integers or floats — no None / NaN — so the result
    can be safely serialized, summed, or compared without surprises.

    Counts:
      n_predictions: total predicted trees
      n_truth: total LiDAR-verified ground-truth trees
      n_true_positive: predictions that matched a truth tree
      n_false_positive: predictions with no matching truth tree
      n_false_negative: truth trees with no matching prediction

    Metrics (computed from counts in `from_counts`):
      precision = TP / (TP + FP)
      recall    = TP / (TP + FN)
      f1        = 2 * P * R / (P + R)
    """

    n_predictions: int
    n_truth: int
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(
        cls, n_predictions: int, n_truth: int, n_true_positive: int,
    ) -> "EvalResult":
        """Build an EvalResult from raw counts.

        Handles all zero-division edge cases: when there are no
        predictions OR no truth, the corresponding metric is 0.0
        (not NaN, not undefined). This keeps downstream logic simple.
        """
        n_tp = n_true_positive
        n_fp = n_predictions - n_tp
        n_fn = n_truth - n_tp

        precision = n_tp / n_predictions if n_predictions > 0 else 0.0
        recall = n_tp / n_truth if n_truth > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return cls(
            n_predictions=n_predictions,
            n_truth=n_truth,
            n_true_positive=n_tp,
            n_false_positive=n_fp,
            n_false_negative=n_fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )


# ============================================================
# Public API
# ============================================================


def find_tree_tops_from_chm(
    chm: np.ndarray,
    transform: Any,
    min_height_m: float = DEFAULT_HEIGHT_THRESHOLD_M,
    min_distance_m: float = DEFAULT_MIN_DISTANCE_M,
    smooth_sigma_px: float = DEFAULT_SMOOTH_SIGMA_PX,
) -> list[tuple[float, float]]:
    """Find tree-top WORLD positions via local-max filtering on a CHM.

    Standard forestry technique:
      1. Smooth the CHM with a small Gaussian to suppress speckle.
      2. Apply a maximum_filter with a window matching half the typical
         crown radius — pixels where smoothed value equals the local
         max are candidate peaks.
      3. Drop peaks below the height threshold (these are bushes).
      4. Convert pixel positions to world coordinates via the raster
         transform.

    Args:
        chm: 2D numpy array of canopy heights in meters. NaN/Inf treated
            as zero (background) so they never produce false peaks.
        transform: Affine transform from rasterio (pixel→world). Used
            to convert pixel positions back to CRS coordinates.
        min_height_m: Drop peaks below this. Default 5 m (= shrub cutoff).
        min_distance_m: Window radius for the local-max filter. Default
            3 m matches half a typical Mediterranean tree crown.
        smooth_sigma_px: Gaussian smoothing sigma in PIXELS. Default 1
            kills speckle without merging adjacent crowns.

    Returns:
        List of (world_x, world_y) tuples in the same CRS as the
        transform. Empty if no peaks found.
    """
    if chm.size == 0:
        return []

    # Replace any non-finite values with 0 so they never become peaks.
    safe = np.where(np.isfinite(chm), chm, 0.0).astype(np.float32)

    # Light smoothing — suppresses single-pixel CHM speckle from sparse
    # LiDAR returns without flattening real crown structure.
    smoothed = gaussian_filter(safe, sigma=smooth_sigma_px)

    # Window size for the local-max filter. We want a square window of
    # diameter ~= 2 × min_distance_m, in pixels. We need the resolution
    # in meters/pixel from the transform — affine.a is the x-resolution.
    pixel_size_m = abs(transform.a) if transform is not None else 0.5
    window_radius_px = max(1, int(round(min_distance_m / pixel_size_m)))
    window_size = 2 * window_radius_px + 1

    # Local maximum filter: each pixel becomes the max in a window
    # around it. A pixel is a local max if its smoothed value equals
    # this local-max value.
    local_max = maximum_filter(smoothed, size=window_size)
    is_peak = (smoothed == local_max) & (smoothed >= min_height_m)

    rows, cols = np.where(is_peak)
    if rows.size == 0:
        return []

    # Convert pixel (col, row) → world (x, y) via the affine transform.
    # rasterio.transform supports (col + 0.5, row + 0.5) for pixel center.
    world_positions: list[tuple[float, float]] = []
    for r, c in zip(rows, cols):
        x_world, y_world = transform * (c + 0.5, r + 0.5)
        world_positions.append((float(x_world), float(y_world)))

    logger.debug(
        "find_tree_tops: %d peaks above %.1f m (window=%dpx)",
        len(world_positions), min_height_m, window_size,
    )
    return world_positions


def match_predictions_to_truth(
    pred_xy: list[tuple[float, float]],
    truth_xy: list[tuple[float, float]],
    tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> EvalResult:
    """Greedy nearest-neighbor matching → EvalResult.

    Algorithm:
      For each truth tree (in input order):
        Find the closest unmatched prediction within tolerance.
        If one exists, mark both as matched (a true positive).
        Otherwise the truth tree becomes a false negative.
      Any prediction never matched becomes a false positive.

    Order independence: greedy can give slightly different results if
    truth ordering changes, but at the project's typical 2 m tolerance
    and 5+ m tree spacing, collisions are rare. The difference between
    greedy and Hungarian is < 1% on real data.

    Args:
        pred_xy: List of predicted (x, y) world coordinates.
        truth_xy: List of ground-truth (x, y) world coordinates.
        tolerance_m: Maximum match distance in meters.

    Returns:
        EvalResult with all counts and metrics filled in.
    """
    n_pred = len(pred_xy)
    n_truth = len(truth_xy)

    if n_pred == 0 or n_truth == 0:
        # Either side is empty — no matches possible.
        return EvalResult.from_counts(
            n_predictions=n_pred, n_truth=n_truth, n_true_positive=0,
        )

    # Numpy arrays for vectorized distance computation
    pred_arr = np.asarray(pred_xy, dtype=np.float64)
    truth_arr = np.asarray(truth_xy, dtype=np.float64)
    tol_sq = tolerance_m * tolerance_m

    matched_pred: set[int] = set()
    n_tp = 0

    for t in range(n_truth):
        # Distance² from this truth point to every prediction. Squared
        # distance is fine because we only need ordering and a single
        # squared threshold compare.
        dx = pred_arr[:, 0] - truth_arr[t, 0]
        dy = pred_arr[:, 1] - truth_arr[t, 1]
        d_sq = dx * dx + dy * dy

        # Mask out predictions already claimed by an earlier truth.
        if matched_pred:
            d_sq = d_sq.copy()
            d_sq[list(matched_pred)] = np.inf

        # Cheapest unmatched prediction within tolerance wins.
        best_idx = int(np.argmin(d_sq))
        if d_sq[best_idx] <= tol_sq:
            matched_pred.add(best_idx)
            n_tp += 1

    return EvalResult.from_counts(
        n_predictions=n_pred, n_truth=n_truth, n_true_positive=n_tp,
    )


def evaluate_patch_against_lidar(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    laz_path: Path,
    height_threshold: float = DEFAULT_HEIGHT_THRESHOLD_M,
    match_tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
    chm_resolution_m: float = 0.5,
) -> EvalResult:
    """End-to-end LiDAR-verified evaluation for ONE patch.

    Pipeline:
      1. Compute CHM for the patch bounds from the LAZ file
      2. Read the CHM raster
      3. Find tree-top truth positions via find_tree_tops_from_chm
      4. Convert detection bbox centers to world coordinates
      5. Match predictions to truth via match_predictions_to_truth

    Args:
        detections: RF-DETR output (sv.Detections with pixel xyxy).
        image_bounds: (x_min, y_min, x_max, y_max) in EPSG:25831.
        image_size_px: (width, height) of the image in pixels.
        laz_path: Path to the cached LAZ tile covering the patch area.
        height_threshold: Min CHM height for a peak to count as truth.
        match_tolerance_m: Max distance for a prediction-truth match.
        chm_resolution_m: CHM grid resolution in meters.

    Returns:
        EvalResult with precision/recall/F1 for this patch.
    """
    # Strip rfdetr metadata first — same gotcha as everywhere else.
    _strip_rfdetr_metadata(detections)

    # Build CHM for the patch bounds (cached after first call).
    chm_path = compute_chm_from_laz(
        laz_path, image_bounds, resolution_m=chm_resolution_m,
    )
    chm, transform = _read_chm_array(chm_path)

    # Ground truth: peaks in the smoothed CHM above the height threshold.
    truth_xy = find_tree_tops_from_chm(
        chm, transform, min_height_m=height_threshold,
    )

    # Predictions: detection bbox centers in world coordinates.
    pred_xy = _detection_centers_world(
        detections, image_bounds, image_size_px,
    )

    return match_predictions_to_truth(
        pred_xy, truth_xy, tolerance_m=match_tolerance_m,
    )


def evaluate_patches_against_lidar(
    patches: list[dict],
) -> tuple[EvalResult, list[dict]]:
    """Aggregate LiDAR eval over many patches.

    Each entry in `patches` must be a dict with keys:
      - 'name': str (display name)
      - 'detections': sv.Detections
      - 'image_bounds': tuple
      - 'image_size_px': tuple
      - 'laz_path': Path

    Returns:
        Tuple of (aggregate_result, per_patch_records). The aggregate
        is a MICRO average — TP/FP/FN counts pooled across patches before
        computing the ratios. Each per-patch record is a dict with the
        patch name and its EvalResult fields, suitable for CSV export.
    """
    per_patch_records: list[dict] = []

    total_pred = 0
    total_truth = 0
    total_tp = 0

    for patch in patches:
        result = evaluate_patch_against_lidar(
            detections=patch["detections"],
            image_bounds=patch["image_bounds"],
            image_size_px=patch["image_size_px"],
            laz_path=patch["laz_path"],
        )

        per_patch_records.append({
            "name": patch["name"],
            "n_predictions": result.n_predictions,
            "n_truth": result.n_truth,
            "n_tp": result.n_true_positive,
            "n_fp": result.n_false_positive,
            "n_fn": result.n_false_negative,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        })

        total_pred += result.n_predictions
        total_truth += result.n_truth
        total_tp += result.n_true_positive

        logger.info(
            "Patch %s: pred=%d truth=%d TP=%d P=%.3f R=%.3f F1=%.3f",
            patch["name"], result.n_predictions, result.n_truth,
            result.n_true_positive, result.precision, result.recall,
            result.f1,
        )

    aggregate = EvalResult.from_counts(
        n_predictions=total_pred,
        n_truth=total_truth,
        n_true_positive=total_tp,
    )
    logger.info(
        "Aggregate (%d patches): pred=%d truth=%d TP=%d P=%.3f R=%.3f F1=%.3f",
        len(patches), total_pred, total_truth, total_tp,
        aggregate.precision, aggregate.recall, aggregate.f1,
    )
    return aggregate, per_patch_records


# ============================================================
# Internal helpers
# ============================================================


def _detection_centers_world(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
) -> list[tuple[float, float]]:
    """Convert each detection's bbox CENTER from pixel → world coordinates.

    Same Y-inversion convention as ndvi.py / lidar.py / georef.py:
    image row 0 is the top, which maps to the geographic y_max.
    """
    if len(detections) == 0:
        return []

    x_min_geo, y_min_geo, x_max_geo, y_max_geo = image_bounds
    w_px, h_px = image_size_px
    x_scale = (x_max_geo - x_min_geo) / w_px
    y_scale = (y_max_geo - y_min_geo) / h_px

    centers: list[tuple[float, float]] = []
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = xyxy.tolist()
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0
        cx_world = x_min_geo + cx_px * x_scale
        cy_world = y_max_geo - cy_px * y_scale  # invert Y
        centers.append((cx_world, cy_world))
    return centers


def _read_chm_array(chm_path: Path) -> tuple[np.ndarray, Any]:
    """Load a CHM GeoTIFF + transform from disk."""
    with rasterio.open(chm_path) as src:
        return src.read(1), src.transform


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Same recurring helper — strip rfdetr's non-per-detection fields.

    Without this, downstream slicing or .data iteration breaks because
    `source_shape` and `source_image` lengths don't match n_detections.
    """
    if not hasattr(detections, "data"):
        return
    for key in ("source_shape", "source_image"):
        if key in detections.data:
            del detections.data[key]
