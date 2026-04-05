"""Download ICGC orthophoto tiles for Parc Natural del Montseny.

Uses the ICGC WMS endpoint to download 25cm RGB orthophotos for 3 elevation
zones representing the full forest diversity of the park:
  - Low (300-700m): Holm oak, cork oak — evergreen Mediterranean
  - Mid (800-1100m): Beech forest — deciduous
  - High (1200-1500m): Fir + subalpine — mixed/sparse canopy

Each zone is downloaded as a grid of GeoTIFF tiles (~1km x 1km each).
Output: data/montseny/raw/{zone_name}/tile_{row}_{col}.tif

Usage:
    python scripts/download_montseny.py
    python scripts/download_montseny.py --zones low mid high
    python scripts/download_montseny.py --zones low --tile-size 2048
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "montseny"

# ICGC WMS endpoint — no authentication required (open data)
WMS_BASE = "https://geoserveis.icgc.cat/servei/catalunya/orto-territorial/wms"
WMS_LAYER = "ortofoto_color_vigent"  # most recent orthophoto (2024)

# Max pixels per WMS request. ICGC allows up to 4096x4096.
# At 25cm/px: 4096px = 1024m. We use 4096 for ~1km tiles.
DEFAULT_TILE_PX = 4096
RESOLUTION_M = 0.25  # 25cm per pixel

# --- Sampling zones in EPSG:25831 (ETRS89 / UTM zone 31N) ---
# Each zone is a 3km x 3km area centered on representative forest.
# Coordinates verified against ICGC viewer and OSM.
ZONES = {
    "low": {
        "name": "Low elevation — Holm oak (Arbúcies area, 300-700m)",
        # Dense evergreen Mediterranean forest, SE slope of Montseny
        "x_min": 458000,
        "y_min": 4623000,
        "x_max": 461000,
        "y_max": 4626000,
    },
    "mid": {
        "name": "Mid elevation — Beech forest (Viladrau area, 800-1100m)",
        # Deciduous beech forest, NW slope
        "x_min": 448000,
        "y_min": 4628000,
        "x_max": 451000,
        "y_max": 4631000,
    },
    "high": {
        "name": "High elevation — Fir/subalpine (Turó de l'Home, 1200-1500m)",
        # Mixed fir + beech, near summit ridge
        "x_min": 451000,
        "y_min": 4625000,
        "x_max": 454000,
        "y_max": 4628000,
    },
}


def build_wms_url(
    x_min: float, y_min: float, x_max: float, y_max: float,
    width: int, height: int,
) -> str:
    """Build a WMS GetMap URL for an ICGC orthophoto tile.

    Args:
        x_min, y_min, x_max, y_max: Bounding box in EPSG:25831.
        width, height: Output image size in pixels.

    Returns:
        Full WMS GetMap URL string.
    """
    return (
        f"{WMS_BASE}?"
        f"SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
        f"&LAYERS={WMS_LAYER}"
        f"&STYLES="
        f"&SRS=EPSG:25831"
        f"&BBOX={x_min},{y_min},{x_max},{y_max}"
        f"&WIDTH={width}&HEIGHT={height}"
        f"&FORMAT=image/tiff"
    )


def download_zone(zone_key: str, tile_px: int = DEFAULT_TILE_PX) -> list[Path]:
    """Download all tiles for a given elevation zone.

    Splits the zone bbox into a grid of tiles, each tile_px x tile_px pixels.
    At 25cm resolution, 4096px ≈ 1024m ≈ 1km per tile.

    Returns:
        List of paths to downloaded GeoTIFF files.
    """
    zone = ZONES[zone_key]
    output_dir = DATA_DIR / "raw" / zone_key
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Zone: %s", zone["name"])

    # Calculate tile size in meters
    tile_m = tile_px * RESOLUTION_M  # e.g., 4096 * 0.25 = 1024m

    # Build grid of tiles covering the zone bbox
    x_min, y_min = zone["x_min"], zone["y_min"]
    x_max, y_max = zone["x_max"], zone["y_max"]

    # How many tiles in each direction
    cols = int(np.ceil((x_max - x_min) / tile_m))
    rows = int(np.ceil((y_max - y_min) / tile_m))

    logger.info("  Grid: %d rows x %d cols = %d tiles (%.0fm x %.0fm each)",
                rows, cols, rows * cols, tile_m, tile_m)

    downloaded = []
    for row in range(rows):
        for col in range(cols):
            tile_path = output_dir / f"tile_{row:02d}_{col:02d}.tif"

            # Skip if already downloaded (idempotent)
            if tile_path.exists() and tile_path.stat().st_size > 1000:
                logger.debug("  Skipping %s (already exists)", tile_path.name)
                downloaded.append(tile_path)
                continue

            # Compute this tile's bbox
            tx_min = x_min + col * tile_m
            ty_min = y_min + row * tile_m
            tx_max = min(tx_min + tile_m, x_max)
            ty_max = min(ty_min + tile_m, y_max)

            # Pixel dimensions (may be smaller for edge tiles)
            w = int((tx_max - tx_min) / RESOLUTION_M)
            h = int((ty_max - ty_min) / RESOLUTION_M)

            url = build_wms_url(tx_min, ty_min, tx_max, ty_max, w, h)

            logger.info("  Downloading tile_%02d_%02d (%dx%d px)...",
                        row, col, w, h)

            try:
                urllib.request.urlretrieve(url, tile_path)
                time.sleep(0.5)  # be polite to ICGC servers
                downloaded.append(tile_path)
                logger.info("  Saved: %s (%.1f MB)",
                            tile_path.name, tile_path.stat().st_size / 1e6)
            except Exception as e:
                logger.error("  Failed to download tile_%02d_%02d: %s",
                             row, col, e)

    return downloaded


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download ICGC Montseny orthophotos."
    )
    parser.add_argument(
        "--zones", nargs="+", default=list(ZONES.keys()),
        choices=list(ZONES.keys()),
        help="Which elevation zones to download.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_PX,
        help="Tile size in pixels (default: 4096 ≈ 1km at 25cm).",
    )
    args = parser.parse_args()

    total_files = []
    for zone_key in args.zones:
        files = download_zone(zone_key, tile_px=args.tile_size)
        total_files.extend(files)

    total_mb = sum(f.stat().st_size for f in total_files) / 1e6
    print(f"\nDownloaded {len(total_files)} tiles ({total_mb:.1f} MB)")
    print(f"Location: {DATA_DIR / 'raw'}")
    print("\nNext step: python scripts/tile_orthophoto.py")


if __name__ == "__main__":
    main()
