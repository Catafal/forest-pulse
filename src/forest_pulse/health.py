"""RGB-based tree health scoring.

Computes vegetation indices (GRVI, ExG) from cropped tree crown bounding boxes
to classify each tree as healthy, stressed, or dead — using only RGB imagery,
no multispectral or NIR required.

Phase 1: Heuristic thresholds (tunable constants below).
Phase 3: Replace with trained MobileNetV3 classifier on Swedish Forest Damages.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# --- Tunable thresholds for heuristic classification ---
# GRVI (Green-Red Vegetation Index): range [-1, 1].
# Healthy vegetation reflects more green than red → GRVI > 0.
#
# Phase 12b calibration (2026-04): GRVI at 25 cm/px on ICGC
# Mediterranean summer orthophotos has a NARROWER dynamic range
# than temperate imagery. Healthy holm oak and Scots pine routinely
# produce GRVI 0.03-0.08 — well below the original 0.10 threshold.
# The original 0.10 was calibrated for temperate-range GRVI and
# labeled 42% of a national park as "stressed" (published realistic
# rate for Mediterranean is 10-25%). Lowered to 0.06 to match the
# Mediterranean GRVI range and bring global stress to ~15%.
GRVI_HEALTHY_THRESHOLD = 0.06   # above this = likely healthy
GRVI_DEAD_THRESHOLD = 0.0      # below this = likely dead/bare

# ExG (Excess Green Index): range approx [-255, 510].
# Higher values = more green vegetation. Secondary confirmation signal.
# Lowered from 30 → 20 alongside the GRVI recalibration — Mediterranean
# canopy at 25 cm/px has naturally lower ExG than temperate, and the
# AND condition with GRVI was double-penalizing dark-crowned species.
EXG_HEALTHY_THRESHOLD = 20.0   # above this + GRVI healthy = confident healthy
EXG_DEAD_THRESHOLD = 5.0       # below this = likely dead/bare

# Minimum crop size for reliable index computation (pixels)
MIN_CROP_SIZE = 4


@dataclass
class HealthScore:
    """Health assessment for a single detected tree crown."""

    tree_id: int
    grvi: float          # Green-Red Vegetation Index: (G-R)/(G+R), range [-1, 1]
    exg: float           # Excess Green Index: 2G - R - B, range [-255, 510]
    label: str           # "healthy" | "stressed" | "dead" | "unknown"
    confidence: float    # 0.0 - 1.0


def score_health(
    image: np.ndarray,
    detections: sv.Detections,
    use_masks: bool = False,
) -> list[HealthScore]:
    """Compute RGB health indices for each detected tree crown.

    For each detection, computes GRVI and ExG vegetation indices on the
    crown pixels and classifies health status.

    Args:
        image: Full RGB image as numpy array (H, W, 3), uint8.
        detections: Supervision Detections with xyxy bounding boxes.
        use_masks: If True and detections have a .mask field (e.g. from
            SAM2 refinement), compute indices only on mask pixels — this
            excludes background gaps inside the bounding box and gives
            more accurate GRVI/ExG values. Default False to preserve
            backward compatibility with bbox-only detections.

    Returns:
        List of HealthScore, one per detected tree, in same order as detections.
        Empty list if no detections.
    """
    if len(detections) == 0:
        logger.info("No detections to score — returning empty health list")
        return []

    # Only use mask path if requested AND masks are actually present.
    # This lets callers pass use_masks=True unconditionally without
    # worrying about whether segmentation was run.
    masks_available = use_masks and detections.mask is not None

    scores = []
    for i, xyxy in enumerate(detections.xyxy):
        if masks_available:
            grvi, exg, too_small = _indices_from_mask(image, detections.mask[i])
        else:
            crop = _crop_detection(image, xyxy)
            too_small = (
                crop.shape[0] < MIN_CROP_SIZE or crop.shape[1] < MIN_CROP_SIZE
            )
            if too_small:
                grvi, exg = 0.0, 0.0
            else:
                grvi = compute_grvi(crop)
                exg = compute_exg(crop)

        if too_small:
            logger.debug("Tree %d: too few pixels, marking unknown", i)
            scores.append(HealthScore(
                tree_id=i, grvi=0.0, exg=0.0, label="unknown", confidence=0.0,
            ))
            continue

        label, conf = classify_health(grvi, exg)
        scores.append(HealthScore(tree_id=i, grvi=grvi, exg=exg, label=label, confidence=conf))

    # Log health distribution for traceability
    distribution = Counter(s.label for s in scores)
    logger.info(
        "Health scored %d trees: %s",
        len(scores),
        ", ".join(f"{k}={v}" for k, v in sorted(distribution.items())),
    )
    return scores


def compute_grvi(crop: np.ndarray) -> float:
    """Green-Red Vegetation Index: (G - R) / (G + R).

    Healthy vegetation has high green relative to red (GRVI > 0.1).
    Stressed vegetation shows lower GRVI (0.0 - 0.1).
    Dead/bare has negative or near-zero GRVI.

    Args:
        crop: RGB image crop as numpy array (H, W, 3), uint8.

    Returns:
        GRVI value, range [-1.0, 1.0]. Returns 0.0 for black/empty crops.
    """
    # Float64 to avoid uint8 overflow in subtraction
    r = crop[:, :, 0].astype(np.float64)
    g = crop[:, :, 1].astype(np.float64)

    mean_g = g.mean()
    mean_r = r.mean()
    denominator = mean_g + mean_r

    # Guard against division by zero (pure black crop)
    if denominator < 1e-6:
        return 0.0

    return float((mean_g - mean_r) / denominator)


def compute_exg(crop: np.ndarray) -> float:
    """Excess Green Index: 2*G - R - B.

    Higher values indicate more vegetation. Useful as a secondary signal
    alongside GRVI to reduce false positives on non-vegetation green objects.

    Args:
        crop: RGB image crop as numpy array (H, W, 3), uint8.

    Returns:
        Mean ExG value across all pixels.
    """
    r = crop[:, :, 0].astype(np.float64)
    g = crop[:, :, 1].astype(np.float64)
    b = crop[:, :, 2].astype(np.float64)

    return float(np.mean(2.0 * g - r - b))


def classify_health(grvi: float, exg: float) -> tuple[str, float]:
    """Classify tree health from vegetation indices.

    Uses heuristic thresholds (Phase 1). Both GRVI and ExG must agree
    for high confidence. Disagreement lowers confidence.

    Phase 3 replaces this with a trained MobileNetV3 classifier.

    Args:
        grvi: Green-Red Vegetation Index value.
        exg: Excess Green Index value.

    Returns:
        Tuple of (label, confidence) where label is one of:
        "healthy", "stressed", "dead".
    """
    # Both indices indicate healthy vegetation
    if grvi > GRVI_HEALTHY_THRESHOLD and exg > EXG_HEALTHY_THRESHOLD:
        return ("healthy", 0.8)

    # Either index indicates dead/bare — flag it
    if grvi < GRVI_DEAD_THRESHOLD or exg < EXG_DEAD_THRESHOLD:
        return ("dead", 0.7)

    # Middle ground — indices disagree or are borderline
    return ("stressed", 0.5)


def _indices_from_mask(
    image: np.ndarray, mask: np.ndarray,
) -> tuple[float, float, bool]:
    """Compute GRVI + ExG using only mask pixels (not the whole bbox).

    When a mask is available (e.g. from SAM2), this gives more accurate
    vegetation indices because we exclude background pixels (sky gaps,
    shadows between crowns) that would contaminate a simple bbox crop.

    Args:
        image: Full RGB image (H, W, 3), uint8.
        mask: Boolean mask (H, W) selecting the crown pixels.

    Returns:
        (grvi, exg, too_small). too_small is True when there are fewer
        than MIN_CROP_SIZE**2 True pixels (not enough signal).
    """
    # Extract only the pixels where the mask is True
    pixels = image[mask]  # shape (N, 3)
    if pixels.shape[0] < MIN_CROP_SIZE * MIN_CROP_SIZE:
        return 0.0, 0.0, True

    # float64 prevents uint8 overflow in subtractions
    r = pixels[:, 0].astype(np.float64)
    g = pixels[:, 1].astype(np.float64)
    b = pixels[:, 2].astype(np.float64)

    mean_g = g.mean()
    mean_r = r.mean()
    denom = mean_g + mean_r
    grvi = 0.0 if denom < 1e-6 else float((mean_g - mean_r) / denom)

    exg = float((2.0 * g - r - b).mean())

    return grvi, exg, False


def _crop_detection(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Extract a bounding box crop from the image, clamped to image bounds.

    Args:
        image: Full image (H, W, 3).
        xyxy: Bounding box as [x1, y1, x2, y2].

    Returns:
        Cropped region as numpy array.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2]))
    y2 = min(h, int(xyxy[3]))
    return image[y1:y2, x1:x2]
