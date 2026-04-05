"""Forest Pulse — Aerial imagery tree detection, health scoring, and change monitoring."""

__version__ = "0.1.0"

from forest_pulse.detect import detect_trees
from forest_pulse.health import HealthScore, score_health
from forest_pulse.visualize import annotate_trees

__all__ = ["detect_trees", "score_health", "HealthScore", "annotate_trees"]
