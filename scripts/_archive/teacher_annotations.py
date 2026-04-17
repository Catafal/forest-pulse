"""Generate tree crown annotations using GroundingDINO as teacher model.

GroundingDINO is a zero-shot object detector — given a text prompt ("tree crown")
it finds matching objects in images. Its Swin Transformer backbone has much stronger
visual understanding than DeepForest's RetinaNet, producing higher-quality labels
for fine-tuning RF-DETR.

This is a form of knowledge distillation: a large pretrained vision model (teacher)
generates training labels for a smaller task-specific model (student = RF-DETR).

Usage:
    python scripts/teacher_annotations.py
    python scripts/teacher_annotations.py --prompt "tree crown" --box-threshold 0.2
    python scripts/teacher_annotations.py --prompt "tree" --box-threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

PATCH_DIR = Path(__file__).parent.parent / "data" / "montseny" / "patches"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent / "data" / "montseny" / "annotations_gdino.json"
)

# GroundingDINO tiny — 172M params, runs on MPS
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Default prompt. GroundingDINO expects labels separated by periods.
# "tree crown." works better than "tree." for aerial imagery because
# it biases toward individual crown shapes rather than entire forests.
DEFAULT_PROMPT = "tree crown."

# Box threshold: how confident the model must be to keep a detection.
# Lower = more detections (higher recall, lower precision).
# 0.15-0.25 is typical for GroundingDINO zero-shot.
DEFAULT_BOX_THRESHOLD = 0.20
DEFAULT_TEXT_THRESHOLD = 0.20


def load_grounding_dino(device: str = "mps"):
    """Load GroundingDINO model and processor.

    Uses HuggingFace transformers implementation for easy MPS support.
    First call downloads ~340MB of weights from HuggingFace Hub.

    Args:
        device: Torch device ("mps", "cuda", or "cpu").

    Returns:
        Tuple of (model, processor).
    """
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
    )

    logger.info("Loading GroundingDINO from %s...", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("GroundingDINO loaded: %.1fM params on %s", n_params, device)
    return model, processor


def detect_with_gdino(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> list[dict]:
    """Run GroundingDINO on a single image.

    Args:
        model: GroundingDINO model.
        processor: GroundingDINO processor.
        image: PIL Image.
        prompt: Text prompt (e.g., "tree crown.").
        box_threshold: Minimum box confidence.
        text_threshold: Minimum text matching confidence.
        device: Torch device.

    Returns:
        List of dicts with keys: bbox (COCO format), score.
    """
    inputs = processor(
        images=image, text=prompt, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )

    detections = []
    if len(results) > 0 and len(results[0]["boxes"]) > 0:
        boxes = results[0]["boxes"].cpu()
        scores = results[0]["scores"].cpu()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.tolist()
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Skip tiny detections (noise)
            if bbox_w < 5 or bbox_h < 5:
                continue

            detections.append({
                "bbox": [
                    round(x1, 1), round(y1, 1),
                    round(bbox_w, 1), round(bbox_h, 1),
                ],
                "score": round(float(score), 4),
            })

    return detections


def generate_teacher_annotations(
    patch_dir: Path = PATCH_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    prompt: str = DEFAULT_PROMPT,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
) -> dict:
    """Run GroundingDINO on all patches and export COCO annotations.

    Args:
        patch_dir: Directory containing .jpg patches.
        output_path: Where to save the COCO JSON.
        prompt: Text prompt for detection.
        box_threshold: Minimum box confidence.
        text_threshold: Minimum text confidence.

    Returns:
        COCO annotation dict.
    """
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model, processor = load_grounding_dino(device)

    patches = sorted(patch_dir.glob("*.jpg"))
    if not patches:
        logger.error("No patches found in %s", patch_dir)
        return {}

    logger.info(
        "Generating teacher annotations: %d patches, prompt='%s', "
        "box_thresh=%.2f",
        len(patches), prompt, box_threshold,
    )

    # COCO format
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "tree", "supercategory": "vegetation"}
        ],
    }

    annotation_id = 0
    total_trees = 0
    start = time.perf_counter()

    for img_id, patch_path in enumerate(patches):
        image = Image.open(patch_path).convert("RGB")
        w, h = image.size

        coco["images"].append({
            "id": img_id,
            "file_name": patch_path.name,
            "width": w,
            "height": h,
        })

        detections = detect_with_gdino(
            model, processor, image, prompt,
            box_threshold, text_threshold, device,
        )

        for det in detections:
            bbox = det["bbox"]
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 1),
                "iscrowd": 0,
            })
            annotation_id += 1

        total_trees += len(detections)

        if (img_id + 1) % 50 == 0:
            elapsed_so_far = time.perf_counter() - start
            rate = (img_id + 1) / elapsed_so_far
            remaining = (len(patches) - img_id - 1) / rate
            logger.info(
                "  %d/%d patches | %d trees | %.1f img/s | ~%.0fs left",
                img_id + 1, len(patches), total_trees, rate, remaining,
            )

    elapsed = time.perf_counter() - start

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info("Done in %.1fs (%.1f img/s)", elapsed, len(patches) / elapsed)
    logger.info(
        "Images: %d | Annotations: %d trees (%.1f per image)",
        len(coco["images"]), len(coco["annotations"]),
        len(coco["annotations"]) / max(len(coco["images"]), 1),
    )
    logger.info("Saved: %s", output_path)

    return coco


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate teacher annotations with GroundingDINO.",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Detection prompt (default: 'tree crown.').",
    )
    parser.add_argument(
        "--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD,
        help="Min box confidence (default: 0.20).",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD,
        help="Min text confidence (default: 0.20).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output COCO JSON path.",
    )
    args = parser.parse_args()

    coco = generate_teacher_annotations(
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        output_path=args.output,
    )

    if coco:
        n_images = len(coco["images"])
        n_annots = len(coco["annotations"])
        avg = n_annots / max(n_images, 1)
        print(f"\n{'='*50}")
        print("  GroundingDINO Teacher Annotations")
        print(f"{'='*50}")
        print(f"  Model:       {MODEL_ID}")
        print(f"  Prompt:      {args.prompt}")
        print(f"  Patches:     {n_images}")
        print(f"  Tree labels: {n_annots} ({avg:.1f} per patch)")
        print(f"  Output:      {args.output}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
