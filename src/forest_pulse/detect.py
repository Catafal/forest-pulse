"""Tree crown detection from aerial RGB imagery.

Wraps detection models (DeepForest for bootstrap, RF-DETR for production)
and returns Supervision Detections objects for downstream processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv


def detect_trees(
    image: np.ndarray | str | Path,
    model_name: str = "deepforest",
    confidence: float = 0.3,
) -> sv.Detections:
    """Detect individual tree crowns in aerial RGB imagery.

    Args:
        image: RGB image as numpy array (H, W, 3) or path to image file.
        model_name: Detection model to use. Options:
            - "deepforest": Pre-trained DeepForest RetinaNet (fast bootstrap).
            - "rfdetr-base": Fine-tuned RF-DETR with DINOv2 backbone.
            - Path to a local checkpoint.
        confidence: Minimum detection confidence threshold (0.0 - 1.0).

    Returns:
        sv.Detections with xyxy bounding boxes, confidence scores, and class IDs.
    """
    # TODO: Implement model loading and inference
    # Phase 1: DeepForest pretrained
    # Phase 2: RF-DETR fine-tuned (swap after auto-research harness completes)
    raise NotImplementedError("detect_trees not yet implemented")


def _load_deepforest():
    """Load DeepForest pretrained tree crown model from HuggingFace."""
    # TODO: Implement DeepForest model loading
    raise NotImplementedError


def _load_rfdetr(checkpoint_path: str | None = None):
    """Load RF-DETR model, optionally from a fine-tuned checkpoint."""
    # TODO: Implement RF-DETR model loading
    raise NotImplementedError


def _to_supervision_detections(raw_output, confidence_threshold: float) -> sv.Detections:
    """Convert model-specific output to Supervision Detections format."""
    # TODO: Normalize different model outputs into sv.Detections
    raise NotImplementedError
