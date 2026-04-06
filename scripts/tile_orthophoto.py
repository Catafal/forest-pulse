"""Tile large ICGC GeoTIFF orthophotos into 640x640 patches for training.

Reads raw tiles from data/montseny/raw/, cuts them into 640x640 px patches,
filters out non-forest patches (roads, buildings, water, bare rock), and
saves forest patches as JPEG with metadata CSV.

Filtering heuristic: Excess Green Index (ExG = 2G - R - B).
Patches with low mean ExG are likely non-forest — skip them.

Usage:
    python scripts/tile_orthophoto.py
    python scripts/tile_orthophoto.py --patch-size 640 --max-patches 300
    python scripts/tile_orthophoto.py --exg-threshold 15.0
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "montseny" / "raw"
PATCH_DIR = Path(__file__).parent.parent / "data" / "montseny" / "patches"

# Default patch size matches RF-DETR input
DEFAULT_PATCH_PX = 640

# Minimum mean Excess Green Index to keep a patch.
# ExG = 2*G - R - B. Forest typically has ExG > 20-30.
# Roads/buildings/water/bare rock have ExG < 10.
DEFAULT_EXG_THRESHOLD = 15.0

# Target: ~100 patches per zone, 3 zones = 300 total
DEFAULT_MAX_PATCHES = 800


def compute_exg(patch: np.ndarray) -> float:
    """Compute mean Excess Green Index for a patch.

    ExG = 2*G - R - B. Higher = more vegetation.
    Uses float64 to avoid uint8 overflow.
    """
    r = patch[:, :, 0].astype(np.float64)
    g = patch[:, :, 1].astype(np.float64)
    b = patch[:, :, 2].astype(np.float64)
    return float(np.mean(2.0 * g - r - b))


def tile_single_geotiff(
    tif_path: Path,
    patch_size: int,
    exg_threshold: float,
    zone_name: str,
) -> list[dict]:
    """Cut a GeoTIFF into patches and filter for forest content.

    Args:
        tif_path: Path to the GeoTIFF file.
        patch_size: Output patch dimensions (square).
        exg_threshold: Minimum ExG to keep a patch.
        zone_name: Elevation zone name (for metadata).

    Returns:
        List of patch metadata dicts with keys:
        filename, zone, source_tile, row, col, x_center, y_center, exg
    """
    patches = []

    with rasterio.open(tif_path) as src:
        # Read all bands as RGB (bands 1,2,3 = R,G,B)
        img = src.read([1, 2, 3])  # shape: (3, H, W)
        img = np.moveaxis(img, 0, -1)  # → (H, W, 3)
        transform = src.transform
        height, width = img.shape[:2]

    # Stride = patch_size (no overlap, maximizes diversity)
    for row_start in range(0, height - patch_size + 1, patch_size):
        for col_start in range(0, width - patch_size + 1, patch_size):
            patch = img[row_start:row_start + patch_size,
                        col_start:col_start + patch_size]

            # Skip if patch has any nodata (black edges from WMS)
            if patch.mean() < 5.0:
                continue

            # Filter by vegetation content
            exg = compute_exg(patch)
            if exg < exg_threshold:
                continue

            # Compute center GPS coordinates from the rasterio transform
            # transform maps pixel (col, row) → (x, y) in CRS
            x_center, y_center = rasterio.transform.xy(
                transform,
                row_start + patch_size // 2,
                col_start + patch_size // 2,
            )

            patches.append({
                "patch": patch,
                "zone": zone_name,
                "source_tile": tif_path.name,
                "row": row_start,
                "col": col_start,
                "x_center": round(x_center, 1),
                "y_center": round(y_center, 1),
                "exg": round(exg, 2),
            })

    return patches


def tile_all_zones(
    patch_size: int = DEFAULT_PATCH_PX,
    exg_threshold: float = DEFAULT_EXG_THRESHOLD,
    max_patches: int = DEFAULT_MAX_PATCHES,
) -> int:
    """Tile all downloaded zones into patches.

    Distributes the max_patches budget evenly across zones, then
    selects the patches with highest ExG (most vegetation) from each.

    Returns:
        Total number of patches saved.
    """
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Discover zone directories
    zone_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not zone_dirs:
        logger.error("No zone directories found in %s", RAW_DIR)
        logger.error("Run: python scripts/download_montseny.py")
        return 0

    patches_per_zone = max_patches // len(zone_dirs)
    logger.info("Budget: %d patches per zone (%d zones, %d total)",
                patches_per_zone, len(zone_dirs), max_patches)

    metadata_rows = []
    patch_counter = 0

    for zone_dir in zone_dirs:
        zone_name = zone_dir.name
        tif_files = sorted(zone_dir.glob("*.tif"))

        if not tif_files:
            logger.warning("No .tif files in %s — skipping", zone_dir)
            continue

        logger.info("Processing zone '%s': %d tiles", zone_name, len(tif_files))

        # Collect all candidate patches from this zone
        candidates = []
        for tif_path in tif_files:
            zone_patches = tile_single_geotiff(
                tif_path, patch_size, exg_threshold, zone_name,
            )
            candidates.extend(zone_patches)
            logger.info("  %s: %d forest patches found",
                        tif_path.name, len(zone_patches))

        # Select top patches by ExG (most vegetated)
        candidates.sort(key=lambda p: p["exg"], reverse=True)
        selected = candidates[:patches_per_zone]

        logger.info("  Zone '%s': %d candidates → %d selected",
                    zone_name, len(candidates), len(selected))

        # Save patches as JPEG
        for patch_data in selected:
            patch_counter += 1
            filename = f"{patch_counter:04d}.jpg"
            patch_path = PATCH_DIR / filename

            Image.fromarray(patch_data["patch"]).save(
                patch_path, quality=95,
            )

            metadata_rows.append({
                "filename": filename,
                "zone": patch_data["zone"],
                "source_tile": patch_data["source_tile"],
                "x_center": patch_data["x_center"],
                "y_center": patch_data["y_center"],
                "exg": patch_data["exg"],
            })

    # Write metadata CSV
    csv_path = PATCH_DIR.parent / "patches_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "zone", "source_tile",
                           "x_center", "y_center", "exg"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info("Saved %d patches to %s", patch_counter, PATCH_DIR)
    logger.info("Metadata: %s", csv_path)
    return patch_counter


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Tile ICGC orthophotos into training patches.",
    )
    parser.add_argument(
        "--patch-size", type=int, default=DEFAULT_PATCH_PX,
        help="Patch size in pixels (default: 640).",
    )
    parser.add_argument(
        "--exg-threshold", type=float, default=DEFAULT_EXG_THRESHOLD,
        help="Min ExG to keep a patch (default: 15.0).",
    )
    parser.add_argument(
        "--max-patches", type=int, default=DEFAULT_MAX_PATCHES,
        help="Max total patches across all zones (default: 300).",
    )
    args = parser.parse_args()

    count = tile_all_zones(
        patch_size=args.patch_size,
        exg_threshold=args.exg_threshold,
        max_patches=args.max_patches,
    )

    if count > 0:
        print(f"\n{count} forest patches ready at {PATCH_DIR}")
        print("Next step: python scripts/bootstrap_annotations.py")
    else:
        print("\nNo patches generated. Check download step.")


if __name__ == "__main__":
    main()
