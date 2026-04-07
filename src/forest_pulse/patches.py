"""Patch metadata helpers — shared across all scripts.

Consolidates the `_get_patch_center` / `_get_patch_bounds` helper that
was previously duplicated across 5 script files
(`full_pipeline_demo.py`, `apply_filters_demo.py`, `run_lidar_eval.py`,
`train_classifier.py`, `sweep_confidence.py`). Each of those scripts
had its own 5-line CSV lookup; Phase 10d consolidates them here.

The Montseny patches are 640×640 pixels at 0.25 m/px = 160 m × 160 m
in real-world meters, in EPSG:25831 (ETRS89 / UTM zone 31N). The
metadata CSV at `data/montseny/patches_metadata.csv` stores the
per-patch center coordinates produced by `scripts/tile_orthophoto.py`
during dataset preparation.

Usage:
    from forest_pulse.patches import get_patch_center, get_patch_bounds

    x, y = get_patch_center(METADATA_CSV, "0043.jpg")
    bounds = get_patch_bounds(METADATA_CSV, "0043.jpg", 160.0)
"""

from __future__ import annotations

import csv
from pathlib import Path


def get_patch_center(
    metadata_csv: Path | str,
    patch_name: str,
) -> tuple[float, float]:
    """Look up a patch's geographic center from the metadata CSV.

    Args:
        metadata_csv: Path to `patches_metadata.csv`. The CSV must
            contain at least `filename`, `x_center`, and `y_center`
            columns (produced by `scripts/tile_orthophoto.py`).
        patch_name: Filename of the patch (e.g. "0043.jpg"). Matched
            against the CSV's `filename` column exactly (no
            case-folding, no extension massaging).

    Returns:
        `(x_center, y_center)` tuple of floats in EPSG:25831 meters.

    Raises:
        FileNotFoundError: if `metadata_csv` does not exist.
        ValueError: if `patch_name` is not found in the CSV.
    """
    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])

    raise ValueError(f"Patch {patch_name!r} not found in {csv_path}")


def get_patch_bounds(
    metadata_csv: Path | str,
    patch_name: str,
    patch_size_m: float = 160.0,
) -> tuple[float, float, float, float]:
    """Look up a patch's geographic bounds (a bounding box).

    Centers the bounding box on the patch's recorded center and
    extends it by `patch_size_m / 2` in each direction. Assumes
    square patches — which matches the Montseny convention
    (640 px × 0.25 m/px = 160 m).

    Args:
        metadata_csv: Path to `patches_metadata.csv`.
        patch_name: Filename of the patch (e.g. "0043.jpg").
        patch_size_m: Side length of the patch in meters. Default 160.

    Returns:
        `(x_min, y_min, x_max, y_max)` tuple in EPSG:25831 meters.

    Raises:
        FileNotFoundError: if `metadata_csv` does not exist.
        ValueError: if `patch_name` is not found in the CSV.
    """
    x_center, y_center = get_patch_center(metadata_csv, patch_name)
    half = patch_size_m / 2.0
    return (
        x_center - half,
        y_center - half,
        x_center + half,
        y_center + half,
    )


def iter_patch_names(
    metadata_csv: Path | str,
    zone: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Return a list of patch filenames from the metadata CSV.

    Args:
        metadata_csv: Path to `patches_metadata.csv`.
        zone: If set, filter to patches whose `zone` column equals
            this value (e.g. "high", "low", "mid", "ne_slopes",
            "nw_plateau"). Default None returns all patches.
        limit: If set, cap the returned list to the first N patches.
            Useful for smoke-testing batch scripts.

    Returns:
        List of patch filenames in CSV order.

    Raises:
        FileNotFoundError: if `metadata_csv` does not exist.
    """
    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    names: list[str] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if zone is not None and row.get("zone") != zone:
                continue
            names.append(row["filename"])
            if limit is not None and len(names) >= limit:
                break
    return names
