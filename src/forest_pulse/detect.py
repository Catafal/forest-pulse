"""Tree crown detection from aerial RGB imagery.

Wraps detection models and returns Supervision Detections for downstream processing.
Supports:
  - DeepForest pretrained (RetinaNet backbone) — quick demo, American forests
  - RF-DETR pretrained (DINOv2 backbone) — SOTA, no fine-tuning
  - RF-DETR from checkpoint — fine-tuned on your own data (the goal)
  - Sliced inference (`detect_trees_sliced`) — Phase 10c, sidesteps
    rfdetr's per-call 300-query cap by running on overlapping
    sub-windows and merging with NMS
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


# --- Sliced inference (Phase 10c) ---


def detect_trees_sliced(
    image: np.ndarray | str | Path,
    model_name: str,
    confidence: float = 0.3,
    slice_wh: int | tuple[int, int] = 320,
    overlap_wh: int | tuple[int, int] = 160,
    iou_threshold: float = 0.5,
) -> sv.Detections:
    """Run detect_trees on overlapping sub-windows and merge with NMS.

    Sidesteps rfdetr's per-call 300-query cap (Phase 10b finding) by
    invoking the detector once per tile. Each tile gets its own 300
    object queries, so an N-tile slice yields up to N * 300 effective
    candidates per parent image.

    Tiles overlap so that any tree straddling a tile boundary is
    fully visible in at least one tile; cross-tile duplicate
    detections (boundary trees seen by multiple tiles) are removed
    with non-max suppression at `iou_threshold`. The supervision
    library's InferenceSlicer handles the grid geometry and
    coordinate translation — returned detections are in the
    ORIGINAL image's pixel frame.

    Args:
        image: RGB image as a numpy array (H, W, 3) or path to an
            image file. Mirrors `detect_trees`.
        model_name: Detection model identifier — same values as
            `detect_trees`: 'deepforest', 'rfdetr-base',
            'rfdetr-large', or a path to a .pt/.pth checkpoint.
        confidence: Per-tile minimum confidence threshold. Applied
            BEFORE cross-tile NMS. Phase 10a's operating point is
            0.02 with the 9.5a filter; the sliced regime may prefer
            a different threshold.
        slice_wh: Tile dimensions in pixels. Accepts an int for
            square tiles or a (width, height) tuple. Default 320.
            Must be smaller than the input image in each dimension.
        overlap_wh: Overlap in pixels between adjacent tiles. Must
            be large enough to fully contain the largest object at
            any tile boundary — at 25 cm/px, a 10-15 m Mediterranean
            tree crown is ~40-60 pixels, so 160 px is a safe default
            for 320-px tiles (50% overlap).
        iou_threshold: NMS threshold for cross-tile duplicates.
            Default 0.5. Tune higher (e.g. 0.7) if precision in
            dense canopy suffers from over-merging.

    Returns:
        sv.Detections in the original image coordinate frame.
        Empty input produces an empty Detections (no crash).
    """
    # Load image from path if needed — same idiom as detect_trees.
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = np.array(Image.open(image_path).convert("RGB"))

    if image.size == 0:
        return sv.Detections.empty()

    # Normalize slice/overlap to the supervision InferenceSlicer
    # format (tuple[int, int]).
    if isinstance(slice_wh, int):
        slice_wh = (slice_wh, slice_wh)
    if isinstance(overlap_wh, int):
        overlap_wh = (overlap_wh, overlap_wh)

    def _slice_callback(image_slice: np.ndarray) -> sv.Detections:
        """Per-tile callback for InferenceSlicer.

        Resolves `detect_trees` via the module namespace at call
        time so unit tests can monkeypatch it cleanly. Strips
        rfdetr's non-per-detection metadata keys
        (`source_shape`, `source_image`) so supervision's
        cross-tile concatenation doesn't crash on mismatched
        lengths.
        """
        dets = detect_trees(
            image_slice, model_name=model_name, confidence=confidence,
        )
        # The rfdetr gotcha: these keys are per-image metadata, not
        # per-detection, and break sv.Detections.merge when their
        # length doesn't match n_detections across tiles.
        if hasattr(dets, "data"):
            for key in ("source_shape", "source_image"):
                if key in dets.data:
                    del dets.data[key]
        return dets

    slicer = sv.InferenceSlicer(
        callback=_slice_callback,
        slice_wh=slice_wh,
        overlap_wh=overlap_wh,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        iou_threshold=iou_threshold,
    )

    start = time.perf_counter()
    detections = slicer(image)
    elapsed = time.perf_counter() - start

    logger.info(
        "Sliced detect: %d detections in %.2fs "
        "(slice=%s, overlap=%s, conf>=%.2f, iou<=%.2f)",
        len(detections), elapsed, slice_wh, overlap_wh,
        confidence, iou_threshold,
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
