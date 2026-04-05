"""Tree crown detection from aerial RGB imagery.

Wraps detection models and returns Supervision Detections for downstream processing.
Supports:
  - DeepForest pretrained (RetinaNet backbone) — quick demo, American forests
  - RF-DETR pretrained (DINOv2 backbone) — SOTA, no fine-tuning
  - RF-DETR from checkpoint — fine-tuned on your own data (the goal)
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy model cache — avoids reloading weights on every call.
# Keyed by model_name so multiple models can coexist.
_MODEL_CACHE: dict = {}


def detect_trees(
    image: np.ndarray | str | Path,
    model_name: str = "deepforest",
    confidence: float = 0.3,
) -> sv.Detections:
    """Detect individual tree crowns in aerial RGB imagery.

    Args:
        image: RGB image as numpy array (H, W, 3) or path to image file.
        model_name: Detection model to use:
            - "deepforest": Pre-trained DeepForest RetinaNet.
            - "rfdetr-base", "rfdetr-large": Pre-trained RF-DETR (no fine-tuning).
            - Path to .pt/.pth file: Fine-tuned RF-DETR checkpoint.
        confidence: Minimum detection confidence threshold (0.0 - 1.0).

    Returns:
        sv.Detections with xyxy bounding boxes and confidence scores.
        Returns sv.Detections.empty() if no trees found.
    """
    # Load image from path if needed, keep original path for RF-DETR
    image_path = None
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))
        logger.info("Loaded image from %s — shape: %s", image_path, image.shape)

    start = time.perf_counter()

    if model_name == "deepforest":
        detections = _predict_deepforest(image, confidence)

    elif model_name.startswith("rfdetr"):
        detections = _predict_rfdetr_pretrained(image, model_name, confidence)

    elif Path(model_name).suffix in (".pt", ".pth"):
        detections = _predict_rfdetr_checkpoint(image, model_name, confidence)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Options: 'deepforest', 'rfdetr-base', "
            "'rfdetr-large', or path to a .pt/.pth checkpoint."
        )

    elapsed = time.perf_counter() - start
    logger.info(
        "Detected %d trees in %.2fs (confidence >= %.2f)",
        len(detections), elapsed, confidence,
    )
    return detections


# --- DeepForest path (Phase 1) ---

def _predict_deepforest(image: np.ndarray, confidence: float) -> sv.Detections:
    """Run detection using DeepForest pretrained model."""
    model = _load_deepforest()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Image type is uint8")
        warnings.filterwarnings("ignore", message="An image was passed directly")
        raw = model.predict_image(image=image)

    if raw is None or len(raw) == 0:
        logger.warning("DeepForest: no trees detected")
        return sv.Detections.empty()

    # Convert DataFrame → sv.Detections, filter by confidence
    df = raw[raw["score"] >= confidence]
    if len(df) == 0:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=df[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32),
        confidence=df["score"].values.astype(np.float32),
    )


def _load_deepforest():
    """Load DeepForest pretrained model, cached after first call."""
    from deepforest import main as df_main

    if "deepforest" in _MODEL_CACHE:
        return _MODEL_CACHE["deepforest"]

    from forest_pulse.device import get_device
    get_device()  # log device for traceability

    logger.info("Loading DeepForest model (first call — downloads weights)...")
    model = df_main.deepforest()
    model.load_model(model_name="weecology/deepforest-tree", revision="main")
    _MODEL_CACHE["deepforest"] = model
    logger.info("DeepForest loaded and cached")
    return model


# --- RF-DETR paths (Phase 2) ---

def _predict_rfdetr_pretrained(
    image: np.ndarray, variant: str, confidence: float,
) -> sv.Detections:
    """Run detection using a pretrained RF-DETR model (no fine-tuning)."""
    model = _load_rfdetr_pretrained(variant)
    # RF-DETR predict() accepts PIL Image or numpy array, returns sv.Detections
    detections = model.predict(images=Image.fromarray(image), threshold=confidence)
    return detections if len(detections) > 0 else sv.Detections.empty()


def _predict_rfdetr_checkpoint(
    image: np.ndarray, checkpoint_path: str, confidence: float,
) -> sv.Detections:
    """Run detection using a fine-tuned RF-DETR checkpoint."""
    model = _load_rfdetr_checkpoint(checkpoint_path)
    detections = model.predict(images=Image.fromarray(image), threshold=confidence)
    return detections if len(detections) > 0 else sv.Detections.empty()


def _load_rfdetr_pretrained(variant: str):
    """Load pretrained RF-DETR by variant name, cached."""
    if variant in _MODEL_CACHE:
        return _MODEL_CACHE[variant]

    import rfdetr

    _VARIANT_MAP = {
        "rfdetr-base": rfdetr.RFDETRBase,
        "rfdetr-large": rfdetr.RFDETRLarge,
    }
    cls = _VARIANT_MAP.get(variant)
    if cls is None:
        raise ValueError(f"Unknown RF-DETR variant '{variant}'. Options: {list(_VARIANT_MAP)}")

    logger.info("Loading RF-DETR %s pretrained model...", variant)
    model = cls()
    _MODEL_CACHE[variant] = model
    logger.info("RF-DETR %s loaded and cached", variant)
    return model


def _load_rfdetr_checkpoint(checkpoint_path: str):
    """Load fine-tuned RF-DETR from checkpoint, cached by path.

    RF-DETR loads custom weights via the pretrain_weights kwarg at init,
    not a separate from_checkpoint method.
    """
    if checkpoint_path in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint_path]

    import rfdetr

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("Loading RF-DETR from checkpoint: %s", path)
    model = rfdetr.RFDETRBase(pretrain_weights=str(path))
    _MODEL_CACHE[checkpoint_path] = model
    logger.info("RF-DETR checkpoint loaded and cached")
    return model
