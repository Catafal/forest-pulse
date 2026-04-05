"""Visualization using Supervision library.

Draws annotated bounding boxes with health-based color coding on aerial imagery.
Color scheme: green = healthy, yellow = stressed, red = dead.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from forest_pulse.health import HealthScore


# Health status → visualization color (BGR for OpenCV compatibility)
HEALTH_COLORS = {
    "healthy": sv.Color(34, 139, 34),     # forest green
    "stressed": sv.Color(255, 165, 0),    # orange
    "dead": sv.Color(220, 20, 60),        # crimson
    "unknown": sv.Color(100, 100, 100),   # gray
}


def annotate_trees(
    image: np.ndarray,
    detections: sv.Detections,
    health_scores: list[HealthScore] | None = None,
    show_labels: bool = True,
) -> np.ndarray:
    """Draw bounding boxes with health color coding using Supervision.

    Args:
        image: RGB image as numpy array (H, W, 3).
        detections: Supervision Detections with xyxy bounding boxes.
        health_scores: Optional health scores to color-code boxes.
            If None, all boxes are drawn in default color.
        show_labels: Whether to show tree ID and health label text.

    Returns:
        Annotated image as numpy array (H, W, 3).
    """
    # TODO: Implement visualization pipeline
    # 1. Create BoundingBoxAnnotator with health-based colors
    # 2. Optionally add LabelAnnotator with tree ID + health label
    # 3. Return annotated image
    raise NotImplementedError("annotate_trees not yet implemented")
