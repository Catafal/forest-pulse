"""Forest Pulse — Aerial imagery tree detection, health scoring, and change monitoring."""

# Set MPS fallback env var BEFORE any torch-importing module loads.
# SAM2 and some detection ops need CPU fallback for a couple of Metal-
# unsupported kernels. Setting it here (the package entry point) makes it
# apply to any caller regardless of import order.
import os as _os

_os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

from forest_pulse.detect import detect_trees
from forest_pulse.health import HealthScore, score_health
from forest_pulse.visualize import annotate_trees

__all__ = ["detect_trees", "score_health", "HealthScore", "annotate_trees"]
