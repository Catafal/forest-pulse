"""Device detection for cross-platform training and inference.

Auto-detects the best available compute backend:
  CUDA (NVIDIA GPU)  →  MPS (Apple Silicon)  →  CPU (fallback)

Handles platform-specific quirks:
  - MPS: no mixed precision (float16 unreliable), some ops fall back to CPU
  - CUDA: supports AMP (automatic mixed precision) for faster training
  - CPU: always works, just slower
"""

from __future__ import annotations

import logging
import os

# Enable silent CPU fallback for ops not yet supported on Apple MPS.
# SAM2 uses a couple of ops (bicubic upsample, one grid_sampler_2d path)
# that aren't on MPS — without this the ops error out instead of
# transparently running on CPU. Must be set BEFORE torch is imported
# anywhere in the process; setting it here (the only module that touches
# torch at import time) guarantees that.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402 — env var must be set before import

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA > MPS > CPU.
    Logs which device was selected for traceability.

    Returns:
        torch.device for model and tensor placement.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        logger.info("Using CUDA: %s (%.1f GB VRAM)", gpu_name, vram_gb)

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders)")

    else:
        device = torch.device("cpu")
        logger.warning("No GPU detected — using CPU (training will be slow)")

    return device


def supports_amp(device: torch.device) -> bool:
    """Check if automatic mixed precision (AMP) is supported on this device.

    AMP (float16) speeds up training ~2x on CUDA but is unreliable on MPS.
    Only enable AMP on CUDA — use float32 everywhere else.
    """
    return device.type == "cuda"
