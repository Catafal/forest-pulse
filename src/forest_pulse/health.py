"""RGB-based tree health scoring.

Computes vegetation indices (GRVI, ExG) from cropped tree crown bounding boxes
to classify each tree as healthy, stressed, or dead — using only RGB imagery,
no multispectral or NIR required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv


@dataclass
class HealthScore:
    """Health assessment for a single detected tree crown."""

    tree_id: int
    grvi: float          # Green-Red Vegetation Index: (G-R)/(G+R), range [-1, 1]
    exg: float           # Excess Green Index: 2G - R - B, range [-255, 510]
    label: str           # "healthy" | "stressed" | "dead"
    confidence: float    # 0.0 - 1.0


def score_health(
    image: np.ndarray,
    detections: sv.Detections,
) -> list[HealthScore]:
    """Compute RGB health indices for each detected tree crown.

    For each bounding box in detections, crops the tree crown from the image
    and computes GRVI and ExG vegetation indices to classify health status.

    Args:
        image: Full RGB image as numpy array (H, W, 3), uint8.
        detections: Supervision Detections with xyxy bounding boxes.

    Returns:
        List of HealthScore, one per detected tree, in same order as detections.
    """
    # TODO: Implement health scoring pipeline
    # 1. Crop each bbox from image
    # 2. Compute GRVI and ExG per crop
    # 3. Classify using thresholds (Phase 1) or trained classifier (Phase 3)
    raise NotImplementedError("score_health not yet implemented")


def compute_grvi(crop: np.ndarray) -> float:
    """Green-Red Vegetation Index: (G - R) / (G + R).

    Healthy vegetation has high green relative to red (GRVI > 0.2).
    Stressed vegetation shows lower GRVI (0.05 - 0.2).
    Dead/bare has negative or near-zero GRVI.
    """
    # TODO: Implement GRVI computation
    raise NotImplementedError


def compute_exg(crop: np.ndarray) -> float:
    """Excess Green Index: 2*G - R - B.

    Higher values indicate more vegetation. Useful as a secondary signal
    alongside GRVI to reduce false positives on non-vegetation green objects.
    """
    # TODO: Implement ExG computation
    raise NotImplementedError


def classify_health(grvi: float, exg: float) -> tuple[str, float]:
    """Classify tree health from vegetation indices.

    Returns (label, confidence) where label is one of:
    "healthy", "stressed", "dead".

    Phase 1: Heuristic thresholds (tunable).
    Phase 3: Replace with trained MobileNetV3 classifier.
    """
    # TODO: Implement classification logic
    raise NotImplementedError
