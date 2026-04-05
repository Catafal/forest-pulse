"""LOCKED evaluation script — DO NOT MODIFY during harness runs.

This file defines the ground truth metric for the auto-research harness.
The agent must NEVER edit this file. Modifying it would allow the agent
to "improve" by changing the benchmark, not the model.

Metric: mAP50 on a fixed validation shard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# TODO: Implement evaluation pipeline
# 1. Load the fixed validation images + annotations (NEVER change these)
# 2. Load the model from checkpoints/current.pt
# 3. Run inference on all validation images
# 4. Compute mAP50 using supervision's evaluation utilities
# 5. Print result in machine-readable format: "val_map50: 0.XXXX"


def evaluate(model_path: str = "checkpoints/current.pt") -> float:
    """Run evaluation on the locked validation set.

    Returns:
        mAP50 score (0.0 - 1.0).
    """
    # TODO: Implement once training pipeline is ready (Phase 2)
    raise NotImplementedError(
        "eval.py not yet implemented. "
        "Requires: validation dataset downloaded + model checkpoint."
    )


if __name__ == "__main__":
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/current.pt"
    score = evaluate(checkpoint)
    # Machine-readable output — the agent greps for this exact line
    print(f"val_map50: {score:.4f}")
