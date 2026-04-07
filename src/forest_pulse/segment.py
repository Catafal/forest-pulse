"""SAM2 hybrid segmentation — refinement + automatic crown detection.

Meta's Segment Anything Model 2 (SAM2) is an instance segmentation foundation
model. We use it in two complementary modes to address RF-DETR's recall
problem in dense canopy:

  1. Refinement — for each RF-DETR bounding box, prompt SAM2 with the box
     and receive a precise pixel mask of the crown. Better health scoring
     (no background pixels in the crop), better crown area (actual shape).

  2. Automatic mode — segment every distinct object in the image, then
     filter the output by size/shape to keep only tree-like segments.
     Catches trees that RF-DETR missed because their crowns merged.

The `detect_trees_hybrid` function composes both modes and dedupes the
overlap, preserving the 0.904-mAP RF-DETR detections as the precision
backbone while using SAM2 only where it helps.

## Model choice

We default to `facebook/sam2.1-hiera-small` (46 M params, ~184 MB, ~1.5 GB
inference RAM on MPS). For box-prompted refinement, a strong prior (the
RF-DETR box) makes even the small variant produce tight masks.
`facebook/sam2.1-hiera-tiny` is the fallback if memory pressure shows up.

## MPS rules

SAM2 officially supports CUDA and CPU. MPS works with caveats:
  - Must set PYTORCH_ENABLE_MPS_FALLBACK=1 before torch imports
    (handled in forest_pulse/__init__.py and device.py).
  - Run in fp32 — do NOT use mixed precision autocast on MPS.
  - `bicubic_upsample` and one `grid_sampler_2d` path fall back to CPU
    silently (~50-100 ms overhead per call). Acceptable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# Module-level model cache. Loading SAM2 takes ~4s on MPS — we load once.
_SAM2_CACHE: dict[str, Any] = {}

# Default model. 'small' is the sweet spot for tree crowns given the
# RF-DETR box prior — larger variants add memory without visible gains.
DEFAULT_MODEL_ID = "facebook/sam2.1-hiera-small"

# Tree-crown filter thresholds, tuned for ICGC 25 cm/px aerial imagery.
# A single crown at that GSD is typically 80-4000 px2 (2-100 m2 area).
DEFAULT_MIN_AREA_PX = 400         # ~25 m2 — rejects noise
DEFAULT_MAX_AREA_PX = 40_000      # ~2500 m2 — rejects merged-forest blobs
DEFAULT_MAX_AREA_FRAC = 0.15      # rejects "whole image" segments
DEFAULT_MIN_CIRCULARITY = 0.45    # 4*pi*A/P^2, ~0.45 = roughly round
DEFAULT_MAX_ASPECT_RATIO = 2.5    # rejects elongated shapes (roads, fences)


@dataclass
class CrownFilter:
    """Tree-crown filter thresholds, bundled for easy overriding."""

    min_area_px: int = DEFAULT_MIN_AREA_PX
    max_area_px: int = DEFAULT_MAX_AREA_PX
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC
    min_circularity: float = DEFAULT_MIN_CIRCULARITY
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO


# ============================================================
# Public API
# ============================================================


def refine_detections_with_sam2(
    image: np.ndarray,
    detections: sv.Detections,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> sv.Detections:
    """Refine RF-DETR bboxes into precise SAM2 crown masks.

    Prompts SAM2 with each detection's bbox and returns a new
    `sv.Detections` with the same xyxy/confidence plus a `.mask` field
    (N, H, W) containing boolean crown masks. Empty input passes through.

    Args:
        image: RGB image as (H, W, 3) uint8 numpy array.
        detections: RF-DETR output with xyxy boxes in pixel coordinates.
        model_id: HuggingFace model identifier. Default is the small
            SAM2.1 variant which is memory-safe on Mac Mini 24 GB.
        device: Torch device string. None = auto-detect via
            `forest_pulse.device.get_device`.

    Returns:
        sv.Detections with xyxy + confidence + mask. Mask is a boolean
        ndarray of shape (N, H, W).
    """
    if len(detections) == 0:
        return detections

    _strip_rfdetr_metadata(detections)

    model, processor, torch_device = _load_sam2(model_id, device)

    # SAM2 wants boxes as list[list[list[float]]] — one outer list per
    # image, one middle list per prompt, each prompt is [x1,y1,x2,y2].
    # Getting this nesting wrong silently produces garbage masks.
    boxes_nested = [[xyxy.tolist() for xyxy in detections.xyxy]]

    import torch

    inputs = processor(
        images=image,
        input_boxes=boxes_nested,
        return_tensors="pt",
    ).to(torch_device)

    # fp32 everywhere on MPS — mixed precision breaks mask heads.
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    # post_process_masks upsamples to the original image size and
    # applies a 0.5 sigmoid threshold. Returns list[tensor]: one per
    # image in the batch.
    masks_list = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
    )
    # masks_list[0] shape: (n_prompts, n_multimask_output_or_1, H, W)
    # We called with multimask_output=False → squeeze dim 1.
    masks_t = masks_list[0]
    if masks_t.ndim == 4:
        masks_t = masks_t[:, 0]
    masks_np = masks_t.numpy().astype(bool)

    return sv.Detections(
        xyxy=detections.xyxy.copy(),
        confidence=(
            detections.confidence.copy()
            if detections.confidence is not None
            else None
        ),
        mask=masks_np,
    )


def segment_all_trees_sam2(
    image: np.ndarray,
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.5,
    stability_score_thresh: float = 0.75,
    crown_filter: CrownFilter | None = None,
) -> sv.Detections:
    """Automatic mask generation + tree-crown filtering.

    Runs SAM2 in 'segment everything' mode via the HuggingFace
    `mask-generation` pipeline, then filters the output to keep only
    segments that look like tree crowns (size, circularity, aspect ratio).

    Slower than RF-DETR (~several seconds per patch on MPS) but finds
    crowns RF-DETR missed in dense canopy.

    ## Why these defaults

    HF's pipeline defaults are `pred_iou_thresh=0.88` and
    `stability_score_thresh=0.95` — way too strict for dense aerial
    canopy where individual crowns have soft boundaries. Empirically
    those defaults produce ONE mask per patch (the whole canopy as a
    blob). Lowering to 0.5 / 0.75 produces 40-70 candidate crowns on
    the same patches; the downstream crown filter drops non-tree
    segments, so lower SAM2 thresholds = more recall without hurting
    precision.

    Args:
        image: RGB image as (H, W, 3) uint8 numpy array.
        model_id: HuggingFace model identifier.
        device: Torch device. None = auto-detect.
        points_per_side: Grid density for automatic prompts. 32 = 1024
            query points, good tradeoff. Lower = faster.
        pred_iou_thresh: SAM2 IoU prediction cutoff. Lower = more masks.
        stability_score_thresh: SAM2 mask stability cutoff. Lower = more.
        crown_filter: Filter thresholds. None = project defaults.

    Returns:
        sv.Detections with xyxy + mask for each kept segment.
    """
    from PIL import Image
    from transformers import pipeline

    if crown_filter is None:
        crown_filter = CrownFilter()

    device_str = _resolve_device(device)

    cache_key = f"auto::{model_id}::pps={points_per_side}"
    if cache_key in _SAM2_CACHE:
        generator = _SAM2_CACHE[cache_key]
    else:
        logger.info("Loading SAM2 auto pipeline %s on %s...", model_id, device_str)
        generator = pipeline(
            "mask-generation",
            model=model_id,
            device=device_str,
            points_per_batch=64,
        )
        _SAM2_CACHE[cache_key] = generator
        logger.info("SAM2 auto pipeline loaded")

    # HF pipeline needs a PIL Image (not a numpy array). Convert once.
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    # HF pipeline returns {"masks": List[ndarray], "scores": tensor}
    result = generator(
        pil_image,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    raw_masks = result.get("masks", [])
    raw_scores = result.get("scores", None)

    logger.info("SAM2 auto mode produced %d candidate segments", len(raw_masks))

    if len(raw_masks) == 0:
        return sv.Detections.empty()

    kept_indices = _filter_crown_segments(
        raw_masks, image.shape[:2], crown_filter,
    )

    if len(kept_indices) == 0:
        logger.info("All SAM2 segments filtered out as non-tree-like")
        return sv.Detections.empty()

    kept_masks = np.stack([np.asarray(raw_masks[i], dtype=bool) for i in kept_indices])
    xyxy = np.stack([_mask_bbox(m) for m in kept_masks]).astype(np.float32)

    if raw_scores is not None:
        scores = np.array([float(raw_scores[i]) for i in kept_indices], dtype=np.float32)
    else:
        scores = np.ones(len(kept_indices), dtype=np.float32)

    logger.info("SAM2 auto mode kept %d tree-like segments", len(kept_indices))

    return sv.Detections(
        xyxy=xyxy,
        confidence=scores,
        mask=kept_masks,
    )


def detect_trees_hybrid(
    image: np.ndarray,
    rfdetr_checkpoint: str,
    rfdetr_confidence: float = 0.3,
    sam2_model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    iou_dedup_threshold: float = 0.30,
    centroid_dedup_px: float = 20.0,
    crown_filter: CrownFilter | None = None,
    points_per_side: int = 32,
) -> sv.Detections:
    """Full hybrid pipeline: RF-DETR → SAM2 refine → SAM2 auto → merge.

    Main entry point when you want maximum tree recall. Preserves the
    high-precision RF-DETR detections and only adds SAM2 automatic
    segments that don't overlap.

    Args:
        image: RGB image as (H, W, 3) uint8 numpy array.
        rfdetr_checkpoint: Path to fine-tuned RF-DETR checkpoint.
        rfdetr_confidence: Min confidence for RF-DETR detections.
        sam2_model_id: SAM2 HuggingFace model ID.
        device: Torch device. None = auto-detect.
        iou_dedup_threshold: Mask IoU above this means "same tree".
        centroid_dedup_px: If masks are empty, use centroid distance.
        crown_filter: Tree-crown filter for SAM2 automatic output.
        points_per_side: SAM2 automatic prompt density.

    Returns:
        sv.Detections containing RF-DETR detections (refined with SAM2
        masks) plus new SAM2-only detections for trees RF-DETR missed.
    """
    from forest_pulse.detect import detect_trees

    # Stage 1 — RF-DETR detection
    rfdetr_dets = detect_trees(
        image, model_name=rfdetr_checkpoint, confidence=rfdetr_confidence,
    )
    _strip_rfdetr_metadata(rfdetr_dets)
    n_rfdetr = len(rfdetr_dets)
    logger.info("Hybrid stage 1: RF-DETR detected %d trees", n_rfdetr)

    # Stage 2 — refine each RF-DETR box into a SAM2 crown mask
    if n_rfdetr > 0:
        refined = refine_detections_with_sam2(
            image, rfdetr_dets, sam2_model_id, device,
        )
    else:
        refined = rfdetr_dets
    logger.info("Hybrid stage 2: refined %d detections", len(refined))

    # Stage 3 — automatic mask generation for crowns RF-DETR missed.
    # Pass permissive SAM2 thresholds — HF defaults are too strict for
    # dense canopy and produce only 1 whole-patch mask. The crown filter
    # drops non-tree segments downstream.
    auto = segment_all_trees_sam2(
        image,
        model_id=sam2_model_id,
        device=device,
        points_per_side=points_per_side,
        crown_filter=crown_filter,
    )
    logger.info("Hybrid stage 3: SAM2 auto found %d candidate trees", len(auto))

    # Stage 4 — dedupe: drop SAM2 segments that overlap refined detections
    if len(auto) == 0:
        return refined
    if len(refined) == 0:
        return auto

    kept_auto_mask = _dedup_auto_against_refined(
        refined, auto, iou_dedup_threshold, centroid_dedup_px,
    )
    auto_new = auto[kept_auto_mask]
    logger.info(
        "Hybrid stage 4: kept %d / %d SAM2 segments after dedupe",
        len(auto_new), len(auto),
    )

    # Stage 5 — merge
    merged = _stack_sv_detections(refined, auto_new)
    logger.info("Hybrid final: %d total detections", len(merged))
    return merged


# ============================================================
# Internal helpers
# ============================================================


def _load_sam2(
    model_id: str, device: str | None,
) -> tuple[Any, Any, Any]:
    """Load SAM2 model + processor, cached by model_id."""
    import torch

    device_str = _resolve_device(device)

    cache_key = f"box::{model_id}"
    if cache_key in _SAM2_CACHE:
        model, processor = _SAM2_CACHE[cache_key]
        return model, processor, torch.device(device_str)

    try:
        from transformers import Sam2Model, Sam2Processor
    except ImportError as e:
        raise ImportError(
            "SAM2 requires transformers >= 4.45. "
            "Install with: pip install -U 'transformers>=4.45'"
        ) from e

    logger.info("Loading SAM2 model %s on %s...", model_id, device_str)
    model = Sam2Model.from_pretrained(model_id).to(device_str)
    # Put model in inference mode (disables dropout etc.)
    model.train(False)
    processor = Sam2Processor.from_pretrained(model_id)
    _SAM2_CACHE[cache_key] = (model, processor)
    logger.info("SAM2 model loaded and cached")
    return model, processor, torch.device(device_str)


def _resolve_device(device: str | None = None) -> str:
    """Resolve 'auto' → concrete device string ('cuda' / 'mps' / 'cpu')."""
    if device is not None:
        return device
    from forest_pulse.device import get_device
    return str(get_device())


def _filter_crown_segments(
    masks: list,
    image_hw: tuple[int, int],
    cf: CrownFilter,
) -> list[int]:
    """Return indices of masks that pass the tree-crown filter.

    Tests applied in order (cheap first, expensive last):
      1. Area — reject tiny noise and huge merged blobs
      2. Area fraction — reject segments covering most of the image
      3. Aspect ratio — reject elongated shapes (roads, fences)
      4. Circularity — reject non-round shapes (4pi*A/P^2 > threshold)
    """
    kept = []
    total_px = image_hw[0] * image_hw[1]

    for idx, mask_like in enumerate(masks):
        mask = np.asarray(mask_like, dtype=bool)
        area = int(mask.sum())

        if area < cf.min_area_px or area > cf.max_area_px:
            continue
        if (area / total_px) > cf.max_area_frac:
            continue

        bbox = _mask_bbox(mask)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw <= 0 or bh <= 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > cf.max_aspect_ratio:
            continue

        circ = _mask_circularity(mask)
        if circ < cf.min_circularity:
            continue

        kept.append(idx)

    return kept


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    """Compute the tight xyxy bounding box of a boolean mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
        dtype=np.float32,
    )


def _mask_circularity(mask: np.ndarray) -> float:
    """Return 4*pi*A/P^2 where A is mask area and P its perimeter.

    A perfect circle has circularity 1.0. Elongated shapes approach 0.
    Perimeter is estimated by counting True pixels bordering False pixels.
    """
    area = float(mask.sum())
    if area <= 0:
        return 0.0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    edges = (
        (padded[1:-1, 1:-1] & ~padded[:-2, 1:-1]).astype(np.int32)
        + (padded[1:-1, 1:-1] & ~padded[2:, 1:-1]).astype(np.int32)
        + (padded[1:-1, 1:-1] & ~padded[1:-1, :-2]).astype(np.int32)
        + (padded[1:-1, 1:-1] & ~padded[1:-1, 2:]).astype(np.int32)
    )
    perimeter = float(edges.sum())
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over union for two boolean masks of the same shape."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (x, y) centroid of a boolean mask in pixel coordinates."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _dedup_auto_against_refined(
    refined: sv.Detections,
    auto: sv.Detections,
    iou_threshold: float,
    centroid_px: float,
) -> np.ndarray:
    """Return a bool mask selecting auto detections NOT overlapping refined.

    Primary test: mask IoU. Secondary (if masks missing or IoU is 0 for
    any reason): centroid distance under `centroid_px`.
    """
    keep = np.ones(len(auto), dtype=bool)
    refined_masks = refined.mask if refined.mask is not None else None

    for i in range(len(auto)):
        a_mask = auto.mask[i] if auto.mask is not None else None
        dropped = False

        if refined_masks is not None and a_mask is not None:
            for r_mask in refined_masks:
                if r_mask.shape != a_mask.shape:
                    continue
                if _mask_iou(a_mask, r_mask) >= iou_threshold:
                    dropped = True
                    break

        if not dropped:
            if a_mask is not None:
                a_cx, a_cy = _mask_centroid(a_mask)
            else:
                a_cx = float((auto.xyxy[i][0] + auto.xyxy[i][2]) / 2.0)
                a_cy = float((auto.xyxy[i][1] + auto.xyxy[i][3]) / 2.0)
            for r_xyxy in refined.xyxy:
                r_cx = float((r_xyxy[0] + r_xyxy[2]) / 2.0)
                r_cy = float((r_xyxy[1] + r_xyxy[3]) / 2.0)
                dist = ((a_cx - r_cx) ** 2 + (a_cy - r_cy) ** 2) ** 0.5
                if dist < centroid_px:
                    dropped = True
                    break

        if dropped:
            keep[i] = False

    return keep


def _stack_sv_detections(
    left: sv.Detections, right: sv.Detections,
) -> sv.Detections:
    """Concatenate two sv.Detections objects, handling mask stacking."""
    xyxy = np.concatenate([left.xyxy, right.xyxy])

    if left.confidence is not None and right.confidence is not None:
        confidence = np.concatenate([left.confidence, right.confidence])
    else:
        confidence = None

    if left.mask is not None and right.mask is not None:
        mask = np.concatenate([left.mask, right.mask])
    elif left.mask is not None:
        mask = left.mask
    elif right.mask is not None:
        mask = right.mask
    else:
        mask = None

    return sv.Detections(xyxy=xyxy, confidence=confidence, mask=mask)


def _strip_rfdetr_metadata(detections: sv.Detections) -> None:
    """Remove rfdetr's non-per-detection fields from detections.data.

    rfdetr adds source_shape/source_image fields whose length doesn't
    match n_detections, which breaks any boolean indexing or slicing.
    """
    if not hasattr(detections, "data"):
        return
    for key in ("source_shape", "source_image"):
        if key in detections.data:
            del detections.data[key]
