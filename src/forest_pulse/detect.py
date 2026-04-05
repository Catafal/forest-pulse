"""Tree crown detection from aerial RGB imagery.

Wraps detection models (DeepForest for bootstrap, RF-DETR for production)
and returns Supervision Detections objects for downstream processing.

Phase 1: DeepForest pretrained (RetinaNet backbone).
Phase 2: Swap to fine-tuned RF-DETR (DINOv2 backbone) via auto-research harness.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy model cache — avoids reloading 170MB weights on every call.
# Keyed by model_name so multiple models can coexist if needed.
_MODEL_CACHE: dict = {}


def detect_trees(
    image: np.ndarray | str | Path,
    model_name: str = "deepforest",
    confidence: float = 0.3,
) -> sv.Detections:
    """Detect individual tree crowns in aerial RGB imagery.

    Args:
        image: RGB image as numpy array (H, W, 3) or path to image file.
        model_name: Detection model to use. Options:
            - "deepforest": Pre-trained DeepForest RetinaNet (Phase 1 default).
            - Path to a local RF-DETR checkpoint (Phase 2+).
        confidence: Minimum detection confidence threshold (0.0 - 1.0).

    Returns:
        sv.Detections with xyxy bounding boxes and confidence scores.
        Returns sv.Detections.empty() if no trees found.

    Raises:
        FileNotFoundError: If image path does not exist.
        ValueError: If model_name is not supported.
    """
    # Load image from path if needed
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        logger.info("Loaded image from %s — shape: %s", image_path, image.shape)

    if model_name != "deepforest":
        raise ValueError(
            f"Model '{model_name}' not supported yet. "
            "Phase 1 only supports 'deepforest'. RF-DETR comes in Phase 2."
        )

    model = _load_deepforest()

    # DeepForest expects RGB numpy array, warns about uint8→float32 (expected behavior)
    import warnings
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Image type is uint8")
        warnings.filterwarnings("ignore", message="An image was passed directly")
        raw_predictions = model.predict_image(image=image)
    elapsed = time.perf_counter() - start
    logger.info("Inference completed in %.2fs", elapsed)

    # DeepForest returns None when no trees detected
    if raw_predictions is None or len(raw_predictions) == 0:
        logger.warning("No trees detected in image")
        return sv.Detections.empty()

    detections = _to_supervision_detections(raw_predictions, confidence)
    logger.info("Detected %d trees (confidence >= %.2f)", len(detections), confidence)
    return detections


def _load_deepforest():
    """Load DeepForest pretrained model, cached after first call.

    Returns the cached model on subsequent calls to avoid re-downloading
    the 170MB weights from HuggingFace.
    """
    from deepforest import main as df_main

    if "deepforest" in _MODEL_CACHE:
        return _MODEL_CACHE["deepforest"]

    from forest_pulse.device import get_device

    logger.info("Loading DeepForest pretrained model (first call — downloads weights)...")
    # Log device for traceability — DeepForest (PyTorch Lightning) handles
    # its own device placement, so we don't call model.to(device) here.
    get_device()
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    _MODEL_CACHE["deepforest"] = model
    logger.info("DeepForest model loaded and cached")
    return model


def _to_supervision_detections(
    df,
    confidence_threshold: float,
) -> sv.Detections:
    """Convert DeepForest DataFrame output to Supervision Detections.

    DeepForest's predict_image() returns a pandas DataFrame with columns:
    xmin, ymin, xmax, ymax, label, score. We extract xyxy + score and
    filter by confidence threshold.
    """
    # Filter by confidence before conversion
    df = df[df["score"] >= confidence_threshold]
    if len(df) == 0:
        return sv.Detections.empty()

    xyxy = df[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)
    confidence = df["score"].values.astype(np.float32)

    return sv.Detections(xyxy=xyxy, confidence=confidence)
