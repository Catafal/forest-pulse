"""Phase 10d: Park-wide tree inventory for Parc Natural del Montseny.

Batch driver that runs the Phase 10c production pipeline on every
patch in `data/montseny/patches_metadata.csv` (or a filtered subset)
and produces ONE merged GeoJSON containing every filter-verified
tree in the park, plus a per-zone summary CSV.

Pipeline per patch (matches `full_pipeline_demo.py` Catalunya
defaults):

    load JPEG
      → detect_trees_sliced (9× 320×320 tiles, Phase 10c)
      → score_health
      → lidar_tree_top_filter (Phase 9.5a)
      → georeference
      → append to combined GeoDataFrame

Cross-patch dedup (after accumulation):

    gpd.sjoin_nearest with max_distance=1m
      → trees straddling a patch boundary that get predicted twice
        are collapsed to the higher-confidence record

Final outputs:

    outputs/inventory/montseny_trees.geojson  — WGS84 point cloud
    outputs/inventory/montseny_trees.csv      — flat attribute table
    outputs/inventory/montseny_trees_summary.csv  — per-zone counts

Runtime estimate on MPS:
    - 0.5 s/patch sliced detection
    - 2-3 s/patch CHM computation (first time per unique bounds)
    - 800 patches × ~3 s = ~40 min cold-cache, ~7 min warm-cache
    - First run also triggers 20-40 new LAZ downloads (~10-20 GB)

Usage:
    # Smoke test on 2 pre-cached reference patches (~10 s):
    python scripts/inventory_montseny.py --patch 0043.jpg --patch 0250.jpg

    # Process first 20 patches (useful for partial smoke test):
    python scripts/inventory_montseny.py --limit 20

    # Process one zone:
    python scripts/inventory_montseny.py --zone high

    # Full park-wide run (slow on cold cache):
    python scripts/inventory_montseny.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image

from forest_pulse.detect import detect_trees_sliced
from forest_pulse.export import to_csv, to_geojson
from forest_pulse.georef import georeference
from forest_pulse.health import score_health
from forest_pulse.lidar import fetch_laz_for_patch, lidar_tree_top_filter
from forest_pulse.patches import (
    get_patch_bounds,
    iter_patch_names,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PATCH_DIR = PROJECT_ROOT / "data" / "montseny" / "patches"
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "current.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "inventory"

# Montseny patch geometry.
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0

# Phase 10c production operating point.
DEFAULT_CONFIDENCE = 0.02
DEFAULT_SLICE_WH = 320
DEFAULT_OVERLAP_WH = 160
DEFAULT_IOU_THRESHOLD = 0.5

# Cross-patch dedup tolerance. A tree on a patch boundary that gets
# detected by both adjacent patches will have ~0-1 m of jitter in
# world coords; collapsing within 1 m keeps the higher-confidence
# record and discards the duplicate.
DEDUP_TOLERANCE_M = 1.0


def _load_zone_lookup(metadata_csv: Path) -> dict[str, str]:
    """Build a {patch_name → zone} map from the metadata CSV.

    Used to annotate each row of the combined GeoDataFrame with its
    source zone for the per-zone summary. Reads the CSV once up-front
    so the per-patch loop doesn't re-parse it.
    """
    lookup: dict[str, str] = {}
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = row.get("zone", "unknown")
    return lookup


def _process_single_patch(
    patch_name: str,
    checkpoint: str,
    confidence: float,
    zone: str,
) -> gpd.GeoDataFrame | None:
    """Run the Phase 10c pipeline on one patch → GeoDataFrame.

    Returns None when no trees survive filtering (empty patch or
    detection failure).
    """
    patch_path = PATCH_DIR / patch_name
    if not patch_path.exists():
        logger.warning("Patch missing: %s", patch_path)
        return None

    image_bounds = get_patch_bounds(METADATA_CSV, patch_name, PATCH_SIZE_M)
    x_min, y_min, x_max, y_max = image_bounds
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    image_size = (PATCH_SIZE_PX, PATCH_SIZE_PX)

    image = np.array(Image.open(patch_path).convert("RGB"))

    # Stage 1: sliced detection (Phase 10c)
    detections = detect_trees_sliced(
        image, model_name=checkpoint, confidence=confidence,
        slice_wh=DEFAULT_SLICE_WH, overlap_wh=DEFAULT_OVERLAP_WH,
        iou_threshold=DEFAULT_IOU_THRESHOLD,
    )
    if len(detections) == 0:
        return None

    # Stage 2: health scoring
    health = score_health(image, detections)

    # Stage 3: Phase 9.5a LiDAR tree-top filter
    laz_path = fetch_laz_for_patch(x_center, y_center)
    detections = lidar_tree_top_filter(
        detections, image_bounds, image_size, laz_path,
    )
    if len(detections) == 0:
        return None
    # Align health scores with the filtered detections (filter
    # preserves per-detection ordering).
    health = health[:len(detections)]

    # Stage 4: georeference → GeoDataFrame with per-tree attributes.
    gdf = georeference(
        detections, image_bounds, image_size, health_scores=health,
    )
    if len(gdf) == 0:
        return None

    # Annotate each row with its source patch + zone so the summary
    # can aggregate by zone and so downstream debugging can trace a
    # tree back to its patch of origin.
    gdf["source_patch"] = patch_name
    gdf["source_zone"] = zone
    return gdf


def _dedup_cross_patch(
    gdf: gpd.GeoDataFrame,
    tolerance_m: float,
) -> gpd.GeoDataFrame:
    """Remove duplicate detections across adjacent patches.

    A tree that straddles a patch boundary can be detected twice —
    once in each adjacent patch. After accumulating all per-patch
    results into a single GeoDataFrame, we use geopandas'
    `sjoin_nearest` to find pairs of trees within `tolerance_m` of
    each other from DIFFERENT source patches, then keep the higher-
    confidence record.

    Trees from the SAME source patch are always kept (the within-
    patch NMS from sliced inference already handled those). Only
    cross-patch duplicates are collapsed here.

    Args:
        gdf: Combined GeoDataFrame with a `source_patch` column.
        tolerance_m: Max distance for a pair to be considered a
            duplicate. Default 1 m matches Phase 10d defaults.

    Returns:
        Deduplicated GeoDataFrame with the same schema.
    """
    if len(gdf) < 2:
        return gdf.reset_index(drop=True)

    # Work in a projected CRS so distances are meters. The incoming
    # gdf is already in EPSG:25831 from georeference().
    if gdf.crs is None:
        raise ValueError("Input gdf must have a CRS set for dedup.")

    work = gdf.reset_index(drop=True).copy()
    work["_row_id"] = work.index

    # sjoin_nearest with distance_col returns ALL near neighbors
    # within max_distance. Self-matches (same row) are always
    # distance=0; we exclude them.
    joined = gpd.sjoin_nearest(
        work, work,
        max_distance=tolerance_m,
        distance_col="_dedup_distance",
    )

    # Drop self-joins and pairs where both rows come from the same
    # patch (those are already unique within-patch).
    cross_patch_pairs = joined[
        (joined["_row_id_left"] != joined["_row_id_right"])
        & (joined["source_patch_left"] != joined["source_patch_right"])
    ]

    # For each cross-patch duplicate pair, mark the lower-confidence
    # row for removal. Keep the higher-confidence row.
    to_drop: set[int] = set()
    for _, row in cross_patch_pairs.iterrows():
        left_id = int(row["_row_id_left"])
        right_id = int(row["_row_id_right"])
        if left_id in to_drop or right_id in to_drop:
            continue
        left_conf = float(row["confidence_left"])
        right_conf = float(row["confidence_right"])
        if left_conf >= right_conf:
            to_drop.add(right_id)
        else:
            to_drop.add(left_id)

    kept = work[~work["_row_id"].isin(to_drop)].drop(columns=["_row_id"])
    logger.info(
        "Cross-patch dedup: removed %d / %d detections "
        "(tolerance=%.1fm)",
        len(to_drop), len(work), tolerance_m,
    )
    return kept.reset_index(drop=True)


def _save_summary_csv(
    gdf: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Write a per-zone + per-health-label summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Per-zone counts
    for zone, zone_df in gdf.groupby("source_zone"):
        labels = Counter(zone_df.get("health_label", pd.Series(dtype=str)).dropna())
        rows.append({
            "zone": zone,
            "n_trees": len(zone_df),
            "healthy": labels.get("healthy", 0),
            "stressed": labels.get("stressed", 0),
            "dead": labels.get("dead", 0),
            "unknown": labels.get("unknown", 0),
        })

    # Park-wide total row
    total_labels = Counter(gdf.get("health_label", pd.Series(dtype=str)).dropna())
    rows.append({
        "zone": "TOTAL",
        "n_trees": len(gdf),
        "healthy": total_labels.get("healthy", 0),
        "stressed": total_labels.get("stressed", 0),
        "dead": total_labels.get("dead", 0),
        "unknown": total_labels.get("unknown", 0),
    })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["zone", "n_trees", "healthy", "stressed", "dead", "unknown"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved summary CSV to %s", path)


def run_inventory(
    patches: list[str],
    checkpoint: str,
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Top-level driver. Processes every patch in `patches`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zone_lookup = _load_zone_lookup(METADATA_CSV)

    print(f"Inventory over {len(patches)} patches")
    print(f"Checkpoint: {checkpoint}")
    print(f"Operating point: sliced {DEFAULT_SLICE_WH}px × "
          f"{DEFAULT_OVERLAP_WH} overlap, conf={confidence}, +9.5a filter")
    print()

    per_patch_gdfs: list[gpd.GeoDataFrame] = []
    t_start = time.perf_counter()

    for i, name in enumerate(patches, start=1):
        t0 = time.perf_counter()
        zone = zone_lookup.get(name, "unknown")
        try:
            gdf = _process_single_patch(name, checkpoint, confidence, zone)
        except Exception as e:
            # Don't let one bad patch kill the whole inventory run.
            # Log and continue — the partial result is still useful.
            logger.error("Patch %s failed: %s", name, e)
            gdf = None

        elapsed = time.perf_counter() - t0
        n = len(gdf) if gdf is not None else 0
        print(f"  [{i}/{len(patches)}] {name} (zone={zone}) — "
              f"{n} trees ({elapsed:.1f}s)")
        if gdf is not None and len(gdf) > 0:
            per_patch_gdfs.append(gdf)

    if not per_patch_gdfs:
        print("\nNo trees found. Output files not written.")
        return

    # Concatenate all per-patch results into one park-wide GeoDataFrame.
    combined = gpd.GeoDataFrame(
        pd.concat(per_patch_gdfs, ignore_index=True),
        crs=per_patch_gdfs[0].crs,
    )

    # Cross-patch dedup.
    dedup = _dedup_cross_patch(combined, DEDUP_TOLERANCE_M)

    # Renumber tree_ids across the park-wide set so each tree has a
    # globally unique id (the per-patch georeference uses local ids).
    dedup = dedup.reset_index(drop=True)
    dedup["tree_id"] = dedup.index

    # Export.
    geojson_path = OUTPUT_DIR / "montseny_trees.geojson"
    csv_path = OUTPUT_DIR / "montseny_trees.csv"
    summary_path = OUTPUT_DIR / "montseny_trees_summary.csv"
    to_geojson(dedup, geojson_path)
    to_csv(dedup, csv_path)
    _save_summary_csv(dedup, summary_path)

    total_elapsed = time.perf_counter() - t_start

    # Final printed summary.
    print()
    print(f"{'=' * 64}")
    print("  Montseny Inventory Complete")
    print(f"{'=' * 64}")
    print(f"  Patches processed:  {len(per_patch_gdfs)} / {len(patches)}")
    print(f"  Trees (pre-dedup):  {len(combined)}")
    print(f"  Trees (post-dedup): {len(dedup)}")
    print()
    print("  Per-zone counts:")
    for zone, zone_df in dedup.groupby("source_zone"):
        print(f"    {str(zone):<14} {len(zone_df)}")
    print()
    health_labels = Counter(dedup.get("health_label", pd.Series(dtype=str)).dropna())
    print("  Health distribution:")
    for label in ("healthy", "stressed", "dead", "unknown"):
        print(f"    {label:<14} {health_labels.get(label, 0)}")
    print()
    print(f"  Runtime: {total_elapsed:.1f}s "
          f"({total_elapsed / max(len(patches), 1):.1f}s per patch)")
    print()
    print("  Output files:")
    print(f"    GeoJSON:  {geojson_path}")
    print(f"    CSV:      {csv_path}")
    print(f"    Summary:  {summary_path}")
    print(f"{'=' * 64}")
    print("\n  Open the GeoJSON in QGIS for a park-wide tree map.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Phase 10d: park-wide inventory driver. Runs the Phase 10c "
            "production pipeline on every patch and merges results into "
            "ONE GeoJSON of every filter-verified tree in Montseny."
        ),
    )
    parser.add_argument(
        "--patch", action="append", default=None,
        help=(
            "Patch filename(s) in data/montseny/patches/. Repeat for "
            "multiple. Default = all patches in patches_metadata.csv "
            "(subject to --zone and --limit)."
        ),
    )
    parser.add_argument(
        "--zone", default=None,
        help=(
            "Filter to patches from one zone "
            "(e.g. high / low / mid / ne_slopes / nw_plateau)."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of patches processed. Smoke-test friendly.",
    )
    parser.add_argument(
        "--checkpoint", default=str(DEFAULT_CHECKPOINT),
        help="Path to RF-DETR checkpoint.",
    )
    parser.add_argument(
        "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help=(
            "Detection confidence threshold. Default 0.02 is the "
            "Phase 10a sweet spot for the sliced + filter regime."
        ),
    )
    args = parser.parse_args()

    # Build the target patch list. Explicit --patch takes precedence
    # over --zone / --limit; otherwise read from metadata.
    if args.patch:
        patches = args.patch
    else:
        patches = iter_patch_names(
            METADATA_CSV, zone=args.zone, limit=args.limit,
        )

    if not patches:
        print("No patches selected.")
        return

    run_inventory(patches, args.checkpoint, confidence=args.confidence)


if __name__ == "__main__":
    main()
