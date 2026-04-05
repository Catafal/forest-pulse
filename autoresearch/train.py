"""EDITABLE training configuration — the agent modifies this file.

This is the search surface for the auto-research harness. The agent
experiments by changing the configuration values below, then running
the training loop to see if mAP50 improves.

Configuration dimensions the agent can explore:
- Model backbone (rfdetr-nano, rfdetr-base, rfdetr-large)
- Learning rate (1e-3 to 1e-6)
- Image size (400, 512, 640, 800, 1024)
- Batch size (2, 4, 8)
- Augmentations (flip, rotate, mosaic, color jitter, cutout)
- Freeze backbone (True/False)
- Fine-tune epochs (5, 10, 20, 50)
- Learning rate scheduler (cosine, step, constant)
- Warmup epochs (0, 1, 2, 5)
"""

from __future__ import annotations

# ============================================================
# CONFIGURATION — Agent edits these values
# ============================================================

BACKBONE = "rfdetr-base"          # rfdetr-nano | rfdetr-base | rfdetr-large
LEARNING_RATE = 1e-4              # float, typically 1e-3 to 1e-6
IMAGE_SIZE = 640                  # int, must be divisible by 32
BATCH_SIZE = 4                    # int, limited by GPU memory
FREEZE_BACKBONE = False           # True = only train detection head
FINE_TUNE_EPOCHS = 20             # int, capped by wall-clock budget
AUGMENTATIONS = [
    "horizontal_flip",
    "random_rotate_90",
]
LR_SCHEDULER = "cosine"           # cosine | step | constant
WARMUP_EPOCHS = 2                 # int, 0 = no warmup

# ============================================================
# DATA PATHS — Do not change (fixed by eval.py contract)
# ============================================================

TRAIN_DATA = "data/oam_tcd/train/"
VAL_DATA = "data/oam_tcd/val/"
CHECKPOINT_DIR = "checkpoints/"

# ============================================================
# TRAINING LOOP — Agent can edit logic below if needed
# ============================================================


def train():
    """Run training with current configuration. Saves checkpoint on completion."""
    # TODO: Implement training pipeline (Phase 2)
    # 1. Load dataset from TRAIN_DATA in COCO format
    # 2. Apply augmentations
    # 3. Initialize model with BACKBONE
    # 4. Set optimizer with LEARNING_RATE and LR_SCHEDULER
    # 5. Train for FINE_TUNE_EPOCHS (or until wall-clock budget expires)
    # 6. Save checkpoint to CHECKPOINT_DIR/current.pt
    raise NotImplementedError(
        "Training not yet implemented. "
        "Requires: rfdetr package + downloaded OAM-TCD dataset."
    )


if __name__ == "__main__":
    print(f"Config: backbone={BACKBONE} lr={LEARNING_RATE} img={IMAGE_SIZE} "
          f"bs={BATCH_SIZE} freeze={FREEZE_BACKBONE} epochs={FINE_TUNE_EPOCHS}")
    print(f"Augmentations: {AUGMENTATIONS}")
    print(f"Scheduler: {LR_SCHEDULER}, warmup: {WARMUP_EPOCHS}")
    train()
