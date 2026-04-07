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
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from forest_pulse.detect import detect_trees, detect_trees_sliced
from forest_pulse.export import to_csv, to_geojson, to_shapefile
from forest_pulse.georef import georeference
from forest_pulse.health import score_health
from forest_pulse.lidar import fetch_laz_for_patch, lidar_tree_top_filter
from forest_pulse.ndvi import fetch_ndvi_for_patch, filter_by_ndvi
from forest_pulse.patches import get_patch_bounds

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline_demo"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"

# Montseny patches are 640x640 px at 0.25 m/px = 160m x 160m real-world.
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0

# Production operating point from Phase 10c (F1 = 0.487 on reference set).
# Sliced inference at 9 tiles + low confidence + 9.5a LiDAR filter.
DEFAULT_CONFIDENCE = 0.02
SLICED_SLICE_WH = 320
SLICED_OVERLAP_WH = 160


def run_pipeline(
    patch_name: str,
    use_slice: bool = True,
    use_lidar_filter: bool = True,
    use_ndvi: bool = False,
    use_sam2: bool = False,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Execute the full pipeline on one patch and write GIS outputs.

    Production pipeline (Catalunya defaults):
      1. detect_trees_sliced (9× 320×320 tiles, NMS-merged)
      2. score_health
      3. lidar_tree_top_filter (Phase 9.5a)
      4. georeference
      5. export (GeoJSON / CSV / Shapefile)

    Fallback paths:
      --no-slice          → full-patch detect_trees (Phase 10a)
      --no-lidar-filter   → skip 9.5a (useful for non-Catalunya)
      --ndvi              → add NDVI filter before georef (layered defense)
      --use-sam2          → hybrid RF-DETR + SAM2 (Phase 6, slower)

    Args:
        patch_name: Filename of the patch (e.g. "0043.jpg").
        use_slice: Use sliced inference (Phase 10c). Default True.
        use_lidar_filter: Apply 9.5a filter after detection. Default True.
        use_ndvi: Apply NDVI filter as an additional layer. Default False.
        use_sam2: Use SAM2 hybrid detector. Default False. Takes
            precedence over use_slice when True.
        confidence: Detection confidence threshold. Default 0.02
            (Phase 10a sweet spot for the sliced + filter regime).
    """
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")

    # Look up bounds from metadata — patches are JPEG so no EXIF.
    image_bounds = get_patch_bounds(METADATA_CSV, patch_name, PATCH_SIZE_M)
    x_min, y_min, x_max, y_max = image_bounds
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    image_size = (PATCH_SIZE_PX, PATCH_SIZE_PX)
    logger.info(
        "Patch: %s at (%.0f, %.0f) in EPSG:25831",
        patch_name, x_center, y_center,
    )

    image = np.array(Image.open(patch_path).convert("RGB"))
    n_stages: list[tuple[str, int]] = []

    # ---- Stage 1: detect ----
    if use_sam2:
        from forest_pulse.segment import detect_trees_hybrid
        logger.info("Stage 1 — hybrid RF-DETR + SAM2 detector")
        detections = detect_trees_hybrid(
            image, str(CHECKPOINT), rfdetr_confidence=max(0.3, confidence),
        )
    elif use_slice:
        logger.info(
            "Stage 1 — sliced RF-DETR (Phase 10c, %dx tiles, conf=%.2f)",
            (PATCH_SIZE_PX // (SLICED_SLICE_WH - SLICED_OVERLAP_WH)) ** 2,
            confidence,
        )
        detections = detect_trees_sliced(
            image, model_name=str(CHECKPOINT), confidence=confidence,
            slice_wh=SLICED_SLICE_WH, overlap_wh=SLICED_OVERLAP_WH,
        )
    else:
        logger.info("Stage 1 — full-patch RF-DETR (conf=%.2f)", confidence)
        detections = detect_trees(
            image, model_name=str(CHECKPOINT), confidence=confidence,
        )

    n_stages.append(("detected", len(detections)))
    if len(detections) == 0:
        print(f"No trees detected in {patch_name}.")
        return

    # ---- Stage 2: health ----
    health = score_health(image, detections, use_masks=use_sam2)
    n_stages.append(("health_scored", len(health)))

    # ---- Stage 3: LiDAR tree-top filter (Phase 9.5a) ----
    if use_lidar_filter:
        logger.info("Stage 3 — LiDAR tree-top filter (9.5a)")
        laz_path = fetch_laz_for_patch(x_center, y_center)
        detections = lidar_tree_top_filter(
            detections, image_bounds, image_size, laz_path,
        )
        n_kept = len(detections)
        health = health[:n_kept]
        n_stages.append(("lidar_filter", n_kept))

    # ---- Stage 4: NDVI filter (optional layered defense) ----
    if use_ndvi and len(detections) > 0:
        logger.info("Stage 4 — NDVI filter (non-vegetation guard)")
        ndvi_path = fetch_ndvi_for_patch(x_center, y_center, PATCH_SIZE_M)
        detections = filter_by_ndvi(
            detections, ndvi_path, image_bounds, image_size,
        )
        n_kept = len(detections)
        health = health[:n_kept]
        n_stages.append(("ndvi_filter", n_kept))

    if len(detections) == 0:
        print(f"All detections dropped by filters in {patch_name}.")
        return

    # ---- Stage 5: georeference ----
    gdf = georeference(
        detections, image_bounds, image_size, health_scores=health,
    )
    n_stages.append(("georef", len(gdf)))

    # ---- Stage 6: export ----
    base = OUTPUT_DIR / patch_name.replace(".jpg", "_trees")
    geojson_path = to_geojson(gdf, base.with_suffix(".geojson"))
    csv_path = to_csv(gdf, base.with_suffix(".csv"))
    shp_path = to_shapefile(gdf, base.with_suffix(".shp"))
    n_stages.append(("exported", len(gdf)))

    # ---- Summary ----
    from collections import Counter
    health_dist = Counter(h.label for h in health)
    print(f"\n{'=' * 60}")
    print(f"  Full Pipeline Demo: {patch_name}")
    print(f"{'=' * 60}")
    print("  Per-stage counts:")
    for stage, n in n_stages:
        print(f"    {stage:<18} {n}")
    print()
    print("  Health distribution:")
    for label, count in sorted(health_dist.items()):
        print(f"    {label:<10} {count}")
    print("\n  Output files:")
    print(f"    GeoJSON:    {geojson_path}")
    print(f"    CSV:        {csv_path}")
    print(f"    Shapefile:  {shp_path}")
    print(f"{'=' * 60}")
    print("\n  Open the GeoJSON in QGIS to see the trees on a map.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Full Forest Pulse pipeline: image → filter-verified "
            "tree GeoJSON. Default is the Phase 10c Catalunya "
            "operating point (F1 = 0.487): sliced inference + "
            "LiDAR tree-top filter."
        ),
    )
    parser.add_argument(
        "--patch", default="0150.jpg",
        help="Patch filename in data/montseny/patches/.",
    )
    parser.add_argument(
        "--no-slice", action="store_true",
        help=(
            "Disable sliced inference. Use the full-patch "
            "detect_trees (Phase 10a regime). Useful for "
            "non-Catalunya / drone imagery."
        ),
    )
    parser.add_argument(
        "--no-lidar-filter", action="store_true",
        help=(
            "Disable the Phase 9.5a LiDAR tree-top filter. "
            "Useful when LAZ data is not available."
        ),
    )
    parser.add_argument(
        "--ndvi", action="store_true",
        help=(
            "Apply NDVI filter as an additional layered defense "
            "against non-vegetation false positives (rooftops, "
            "roads, bare rock). Off by default."
        ),
    )
    parser.add_argument(
        "--use-sam2", action="store_true",
        help="Use SAM2 hybrid detector (slower, higher recall on dense canopy).",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=(
            "Detection confidence threshold. Default 0.02 is the "
            "Phase 10a sweet spot for the sliced + filter regime."
        ),
    )
    args = parser.parse_args()
    run_pipeline(
        args.patch,
        use_slice=not args.no_slice,
        use_lidar_filter=not args.no_lidar_filter,
        use_ndvi=args.ndvi,
        use_sam2=args.use_sam2,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
