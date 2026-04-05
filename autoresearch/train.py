"""EDITABLE training configuration — the auto-research agent modifies this file.

This is the search surface for the auto-research harness. The agent
experiments by changing the configuration values below, then running
the training loop to see if mAP50 improves.

Configuration dimensions the agent can explore:
- Model backbone (rfdetr-base, rfdetr-large)
- Learning rate (1e-3 to 1e-6)
- Image size (400, 512, 640, 800) — set by model variant, not a direct param
- Batch size (1, 2, 4, 8)
- Fine-tune epochs (5, 10, 20, 50)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — Agent edits these values
# ============================================================

BACKBONE = "rfdetr-base"          # rfdetr-base | rfdetr-large
LEARNING_RATE = 1e-4              # float, typically 1e-3 to 1e-6
BATCH_SIZE = 2                    # int, keep low for MPS (2-4)
FINE_TUNE_EPOCHS = 10             # int, 10 epochs for first real training

# ============================================================
# FIXED PATHS — Do not change (eval.py depends on these)
# ============================================================

DATASET_DIR = str(Path(__file__).parent.parent / "data" / "rfdetr")
CHECKPOINT_DIR = str(Path(__file__).parent.parent / "checkpoints")

# Maps config string → rfdetr class name
_BACKBONE_MAP = {
    "rfdetr-base": "RFDETRBase",
    "rfdetr-large": "RFDETRLarge",
}

# ============================================================
# TRAINING
# ============================================================


def train():
    """Run RF-DETR fine-tuning with current config.

    Uses the rfdetr library's built-in training loop which handles:
    - Data loading from COCO format
    - Augmentations (built-in)
    - Device placement (auto CUDA/MPS/CPU)
    - Checkpoint saving (best + last)

    The agent only needs to tweak the config values above.
    """
    import rfdetr

    # Validate dataset exists
    dataset_path = Path(DATASET_DIR)
    if not (dataset_path / "train" / "_annotations.coco.json").exists():
        raise FileNotFoundError(
            f"Training data not found at {DATASET_DIR}. "
            "Run: python scripts/prepare_rfdetr_dataset.py"
        )

    # Resolve model class from backbone config
    model_cls_name = _BACKBONE_MAP.get(BACKBONE)
    if model_cls_name is None:
        raise ValueError(f"Unknown backbone '{BACKBONE}'. Options: {list(_BACKBONE_MAP)}")

    model_cls = getattr(rfdetr, model_cls_name)
    model = model_cls()

    # Auto-compute gradient accumulation to approximate effective batch of 16.
    # Smaller BATCH_SIZE (needed for MPS memory) compensated by more accumulation steps.
    grad_accum = max(1, 16 // BATCH_SIZE)

    checkpoint_path = Path(CHECKPOINT_DIR)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training %s | lr=%s | bs=%d | grad_accum=%d | epochs=%d",
        BACKBONE, LEARNING_RATE, BATCH_SIZE, grad_accum, FINE_TUNE_EPOCHS,
    )

    model.train(
        dataset_dir=DATASET_DIR,
        epochs=FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=grad_accum,
        lr=LEARNING_RATE,
        output_dir=CHECKPOINT_DIR,
    )

    # Copy best checkpoint to current.pt — the filename eval.py expects
    best = checkpoint_path / "checkpoint_best_total.pth"
    current = checkpoint_path / "current.pt"
    if best.exists():
        shutil.copy2(best, current)
        logger.info("Best checkpoint saved as: %s", current)
    else:
        logger.warning("No best checkpoint found at %s", best)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")

    print(f"Config: backbone={BACKBONE} lr={LEARNING_RATE} "
          f"bs={BATCH_SIZE} epochs={FINE_TUNE_EPOCHS}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {CHECKPOINT_DIR}")
    train()
