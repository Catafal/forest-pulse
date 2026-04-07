"""Full pipeline demo: patch → detect → health → NDVI → georef → GeoJSON.

End-to-end demonstration of the Forest Pulse pipeline on a single Montseny
patch. Produces a GeoJSON file that can be opened directly in QGIS, showing
each detected tree as a point with its health label and attributes.

Pipeline stages:
  1. Load patch JPEG
  2. Run RF-DETR detection (checkpoints/current.pt)
  3. Score each crown's health from RGB indices (GRVI + ExG)
  4. NDVI filter — drops non-vegetation false positives
  5. Georeference — pixel bboxes → Point geometries in EPSG:25831
  6. Export — GeoJSON (WGS84) + CSV + Shapefile

Usage:
    python scripts/full_pipeline_demo.py --patch 0150.jpg
    python scripts/full_pipeline_demo.py --patch 0547.jpg --skip-ndvi
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees
from forest_pulse.export import to_csv, to_geojson, to_shapefile
from forest_pulse.georef import georeference
from forest_pulse.health import score_health
from forest_pulse.ndvi import fetch_ndvi_for_patch, filter_by_ndvi

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_demo"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"

# Montseny patches are 640x640 px at 0.25 m/px = 160m x 160m real-world.
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Look up the geographic center of a patch from the metadata CSV."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def run_pipeline(patch_name: str, skip_ndvi: bool) -> None:
    """Execute the full pipeline on one patch and write GIS outputs."""
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")

    # Look up bounds from metadata — patches are JPEG so no EXIF
    x_center, y_center = _get_patch_center(patch_name)
    half = PATCH_SIZE_M / 2
    image_bounds = (
        x_center - half, y_center - half,
        x_center + half, y_center + half,
    )
    image_size = (PATCH_SIZE_PX, PATCH_SIZE_PX)
    logger.info("Patch: %s at (%.0f, %.0f) in EPSG:25831",
                patch_name, x_center, y_center)

    # Stage 1-3: detect → health
    image = np.array(Image.open(patch_path).convert("RGB"))
    detections = detect_trees(
        image, model_name=str(CHECKPOINT), confidence=0.3,
    )
    if len(detections) == 0:
        print(f"No trees detected in {patch_name}.")
        return
    logger.info("Stage 1 — detected %d trees", len(detections))

    health = score_health(image, detections)
    logger.info("Stage 2 — scored %d trees", len(health))

    # Stage 4: NDVI filter (drops non-vegetation false positives)
    if not skip_ndvi:
        logger.info("Stage 3 — fetching NDVI raster...")
        ndvi_path = fetch_ndvi_for_patch(x_center, y_center, PATCH_SIZE_M)
        detections_filtered = filter_by_ndvi(
            detections, ndvi_path, image_bounds, image_size,
        )
        # Keep health scores aligned with filtered detections.
        # NDVI filter drops detections but doesn't update tree_ids, so
        # we rebuild the health list by iterating by position.
        n_kept = len(detections_filtered)
        health = health[:n_kept]
        detections = detections_filtered
        logger.info("Stage 3 — after NDVI filter: %d trees", len(detections))

    # Stage 5: georeference pixel detections → GeoDataFrame
    gdf = georeference(detections, image_bounds, image_size, health_scores=health)
    logger.info("Stage 4 — georeferenced %d trees", len(gdf))

    # Stage 6: export — GeoJSON (WGS84), CSV, Shapefile
    base = OUTPUT_DIR / patch_name.replace(".jpg", "_trees")
    geojson_path = to_geojson(gdf, base.with_suffix(".geojson"))
    csv_path = to_csv(gdf, base.with_suffix(".csv"))
    shp_path = to_shapefile(gdf, base.with_suffix(".shp"))

    # Summary
    from collections import Counter
    health_dist = Counter(h.label for h in health)
    print(f"\n{'='*55}")
    print(f"  Full Pipeline Demo: {patch_name}")
    print(f"{'='*55}")
    print(f"  Trees detected:  {len(gdf)}")
    for label, count in sorted(health_dist.items()):
        print(f"    {label:>10}: {count}")
    print("\n  Output files:")
    print(f"    GeoJSON:    {geojson_path}")
    print(f"    CSV:        {csv_path}")
    print(f"    Shapefile:  {shp_path}")
    print(f"{'='*55}")
    print("\n  Open the GeoJSON in QGIS to see the trees on a map.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Full Forest Pulse pipeline: image → tree GeoJSON.",
    )
    parser.add_argument(
        "--patch", default="0150.jpg",
        help="Patch filename in data/montseny/patches/.",
    )
    parser.add_argument(
        "--skip-ndvi", action="store_true",
        help="Skip the NDVI filter stage (useful for offline runs).",
    )
    args = parser.parse_args()
    run_pipeline(args.patch, args.skip_ndvi)


if __name__ == "__main__":
    main()
