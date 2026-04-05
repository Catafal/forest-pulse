"""Visualization using Supervision library.

Draws annotated bounding boxes with health-based color coding on aerial imagery.
Maps health labels → class IDs → color palette for per-detection coloring.
"""

from __future__ import annotations

import logging

import numpy as np
import supervision as sv

from forest_pulse.health import HealthScore

logger = logging.getLogger(__name__)

# Health label → integer class ID for Supervision's color lookup
LABEL_TO_CLASS_ID = {"healthy": 0, "stressed": 1, "dead": 2, "unknown": 3}

# Color palette ordered to match class IDs above
# green (healthy) → orange (stressed) → crimson (dead) → gray (unknown)
HEALTH_PALETTE = sv.ColorPalette.from_hex(["#228B22", "#FFA500", "#DC143C", "#646464"])


def annotate_trees(
    image: np.ndarray,
    detections: sv.Detections,
    health_scores: list[HealthScore] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw bounding boxes with health color coding using Supervision.

    When health_scores are provided, each box is colored by health status:
    green (healthy), orange (stressed), red (dead), gray (unknown).
    Without health_scores, all boxes are drawn in a uniform default color.

    Args:
        image: RGB image as numpy array (H, W, 3).
        detections: Supervision Detections with xyxy bounding boxes.
        health_scores: Optional health scores to color-code boxes.
        show_labels: Whether to show tree ID and health label text.

    Returns:
        Annotated image as numpy array (H, W, 3). Original is not mutated.
    """
    scene = image.copy()

    if len(detections) == 0:
        logger.info("No detections to annotate — returning original image")
        return scene

    # Assign class IDs from health labels so Supervision colors by health status
    if health_scores is not None:
        class_ids = np.array(
            [LABEL_TO_CLASS_ID.get(hs.label, 3) for hs in health_scores],
            dtype=np.int32,
        )
        detections.class_id = class_ids

        box_annotator = sv.BoxAnnotator(
            color=HEALTH_PALETTE,
            thickness=2,
            color_lookup=sv.ColorLookup.CLASS,
        )
    else:
        box_annotator = sv.BoxAnnotator(thickness=2)

    scene = box_annotator.annotate(scene=scene, detections=detections)

    # Add text labels above each box
    if show_labels:
        labels = _build_labels(detections, health_scores)

        if health_scores is not None:
            label_annotator = sv.LabelAnnotator(
                color=HEALTH_PALETTE,
                color_lookup=sv.ColorLookup.CLASS,
                text_padding=4,
            )
        else:
            label_annotator = sv.LabelAnnotator(text_padding=4)

        scene = label_annotator.annotate(
            scene=scene, detections=detections, labels=labels,
        )

    logger.info("Annotated %d trees on image", len(detections))
    return scene


def _build_labels(
    detections: sv.Detections,
    health_scores: list[HealthScore] | None,
) -> list[str]:
    """Build label strings for each detection.

    With health: "#47 healthy (0.31)" — tree ID, status, GRVI value.
    Without health: "tree 0", "tree 1", etc.
    """
    if health_scores is not None:
        return [
            f"#{hs.tree_id} {hs.label} ({hs.grvi:.2f})"
            for hs in health_scores
        ]
    return [f"tree {i}" for i in range(len(detections))]
