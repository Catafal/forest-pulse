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


# --- LiDAR-first detection (Phase 11a) ---


def detect_trees_from_lidar(
    laz_path: "Path",
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    crown_radius_m: float = 2.5,
    min_height_m: float = 5.0,
    rf_detr_verify: bool = False,
    rf_detr_checkpoint: str | None = None,
    rf_detr_image: np.ndarray | None = None,
    rf_detr_verify_tolerance_m: float = 2.0,
    chm_resolution_m: float = 0.5,
) -> sv.Detections:
    """LiDAR-first detector. Each LiDAR tree-top becomes one detection.

    This is the Phase 11a architectural flip: instead of treating
    LiDAR as a post-hoc precision filter on RF-DETR output, LiDAR
    becomes the primary detector and RF-DETR is demoted to optional
    visual verifier.

    Pipeline:
      1. Compute CHM from the LAZ tile for this patch's bounds
      2. Extract LiDAR tree-top positions via local-max filtering
      3. Project each world position back to pixel coordinates
      4. Construct a fixed-radius bbox around each pixel center
      5. Clip bboxes to the image frame; drop zero-area results
      6. Synthesize confidence from peak height (taller → higher)
      7. (Optional) verify each peak against a sliced RF-DETR run:
         drop peaks with no RF-DETR detection within the tolerance

    Args:
        laz_path: Path to the LAZ tile covering this patch.
        image_bounds: (x_min, y_min, x_max, y_max) in EPSG:25831.
        image_size_px: (width, height) in pixels.
        crown_radius_m: Half-width of each detection bbox. Default
            2.5 (= 5 m diameter, published Mediterranean average).
        min_height_m: Minimum CHM peak height for a LiDAR tree-top
            to count. Default 5 m (Spanish Forest Inventory cutoff).
        rf_detr_verify: If True, drop LiDAR peaks that have NO
            matching RF-DETR detection within `rf_detr_verify_tolerance_m`.
            Default False (pure LiDAR output).
        rf_detr_checkpoint: Path to the RF-DETR checkpoint. Required
            when rf_detr_verify is True.
        rf_detr_image: Full RGB image as numpy array (H, W, 3).
            Required when rf_detr_verify is True — detect_trees_sliced
            needs pixels.
        rf_detr_verify_tolerance_m: Max distance between a LiDAR peak
            and an RF-DETR detection for the peak to survive
            verification. Default 2 m.
        chm_resolution_m: CHM grid resolution. Default 0.5 m.

    Returns:
        sv.Detections with one bbox per surviving LiDAR peak. Empty
        if no peaks found or if all peaks were rejected. Never
        raises on empty input.
    """
    # Lazy imports to keep the top of detect.py clean and to avoid
    # a circular import if forest_pulse.lidar imports from detect.
    from forest_pulse.lidar import (
        _read_chm_raster,
        compute_chm_from_laz,
        find_tree_tops_from_chm,
        world_to_pixel_batch,
    )

    # ---- Stage 1+2: CHM → tree-tops ----
    chm_path = compute_chm_from_laz(
        laz_path, image_bounds, resolution_m=chm_resolution_m,
    )
    chm, transform = _read_chm_raster(chm_path)
    positions, heights = find_tree_tops_from_chm(
        chm, transform, min_height_m=min_height_m, return_heights=True,
    )
    if not positions:
        return sv.Detections.empty()

    # ---- Stage 3: project to pixel space ----
    pos_arr = np.asarray(positions, dtype=np.float64)
    pixel_centers = world_to_pixel_batch(
        pos_arr, image_bounds, image_size_px,
    )

    # ---- Stage 4: construct fixed-radius bboxes ----
    w_px, h_px = image_size_px
    x_min, y_min, x_max, y_max = image_bounds
    # Per-axis scale in case the patch isn't perfectly square.
    px_per_m_x = w_px / (x_max - x_min)
    px_per_m_y = h_px / (y_max - y_min)
    radius_px_x = crown_radius_m * px_per_m_x
    radius_px_y = crown_radius_m * px_per_m_y

    xyxy = np.stack([
        pixel_centers[:, 0] - radius_px_x,
        pixel_centers[:, 1] - radius_px_y,
        pixel_centers[:, 0] + radius_px_x,
        pixel_centers[:, 1] + radius_px_y,
    ], axis=1)

    # ---- Stage 5: clip to image frame, drop zero-area ----
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, float(w_px))
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, float(h_px))
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, float(w_px))
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, float(h_px))

    widths = xyxy[:, 2] - xyxy[:, 0]
    heights_px = xyxy[:, 3] - xyxy[:, 1]
    keep = (widths > 0) & (heights_px > 0)

    xyxy = xyxy[keep]
    heights_kept = np.asarray(heights, dtype=np.float64)[keep]
    if len(xyxy) == 0:
        return sv.Detections.empty()

    # ---- Stage 6: synthetic confidence from peak height ----
    # Deterministic and monotonic in height. A 5 m peak maps to
    # conf 0.01 (just above the floor), a 25 m peak to 1.0.
    # Downstream filters can rank detections by height proxy.
    conf = np.clip((heights_kept - 5.0) / 20.0, 0.01, 1.0)

    dets = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=conf.astype(np.float32),
        class_id=np.zeros(len(xyxy), dtype=np.int64),
    )

    logger.info(
        "LiDAR-first detect: %d LiDAR peaks → %d detections "
        "(crown_radius=%.1fm, min_height=%.1fm)",
        len(positions), len(dets), crown_radius_m, min_height_m,
    )

    # ---- Stage 7 (optional): RF-DETR visual verification ----
    if rf_detr_verify:
        if rf_detr_checkpoint is None or rf_detr_image is None:
            raise ValueError(
                "rf_detr_verify=True requires both rf_detr_checkpoint "
                "and rf_detr_image"
            )
        dets = _verify_with_rf_detr(
            dets=dets,
            image=rf_detr_image,
            image_bounds=image_bounds,
            image_size_px=image_size_px,
            checkpoint=rf_detr_checkpoint,
            tolerance_m=rf_detr_verify_tolerance_m,
        )

    return dets


def _verify_with_rf_detr(
    dets: sv.Detections,
    image: np.ndarray,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    checkpoint: str,
    tolerance_m: float,
) -> sv.Detections:
    """Drop LiDAR detections that no RF-DETR detection agrees with.

    Runs the existing sliced RF-DETR at the Phase 10c operating
    point and compares each LiDAR detection's bbox center (in world
    coords) to the set of RF-DETR detection centers. A LiDAR
    detection survives if at least one RF-DETR detection sits
    within `tolerance_m` of its center.

    This is the "optional visual verifier" layer: it trades some
    recall (trees RF-DETR can't see get dropped) for higher
    precision (phantom LiDAR peaks from boulders / buildings get
    dropped).

    Kept as a private helper to avoid cluttering
    detect_trees_from_lidar's signature with an inner pipeline.
    """
    from forest_pulse.lidar import bbox_centers_to_world

    # Run sliced RF-DETR at Phase 10c defaults.
    rf_dets = detect_trees_sliced(
        image=image,
        model_name=checkpoint,
        confidence=0.02,
        slice_wh=320,
        overlap_wh=160,
        iou_threshold=0.5,
    )
    if len(rf_dets) == 0:
        # If RF-DETR sees nothing, reject all LiDAR peaks.
        return sv.Detections.empty()

    lidar_centers = bbox_centers_to_world(dets, image_bounds, image_size_px)
    rf_centers = bbox_centers_to_world(rf_dets, image_bounds, image_size_px)

    # For each LiDAR detection, find squared distance to nearest
    # RF-DETR center. Keep if within tolerance.
    tol_sq = tolerance_m * tolerance_m
    dx = lidar_centers[:, 0:1] - rf_centers[:, 0]
    dy = lidar_centers[:, 1:2] - rf_centers[:, 1]
    d_sq = dx * dx + dy * dy
    nearest_sq = d_sq.min(axis=1)
    keep_mask = nearest_sq <= tol_sq

    n_before = len(dets)
    n_after = int(keep_mask.sum())
    logger.info(
        "RF-DETR verify: %d LiDAR peaks → %d after visual confirmation "
        "(tolerance=%.1fm)",
        n_before, n_after, tolerance_m,
    )
    return dets[keep_mask]


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
