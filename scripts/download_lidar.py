"""Download ICGC LiDAR LAZ tiles for Montseny sampling zones.

Iterates the ZONES dict from `download_montseny.py`, resolves each zone's
center to a 1 km LAZ tile URL, deduplicates (adjacent zones often share
a tile), and downloads anything missing from `data/montseny/lidar/`.

Each LAZ tile is ~400-700 MB. A typical Montseny 8-zone run downloads
between 2 and 6 tiles (= ~2-4 GB), depending on how the zones are laid
out. All tiles are cached, so re-runs are free.

Usage:
    python scripts/download_lidar.py                    # all 8 zones
    python scripts/download_lidar.py --zone low         # single zone
    python scripts/download_lidar.py --x 450000 --y 4625000   # single point
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from forest_pulse.lidar import LAZ_CACHE, fetch_laz_for_patch

logger = logging.getLogger(__name__)


def _load_zones() -> dict:
    """Lazy import ZONES dict to avoid pulling download_montseny at module load."""
    sys.path.insert(0, str(Path(__file__).parent))
    from download_montseny import ZONES
    return ZONES


def download_all_zones() -> list[Path]:
    """Download LAZ tiles covering every zone in download_montseny.ZONES.

    Deduplicates on URL — zones that fall in the same 1 km tile only
    trigger one download. Returns the list of local LAZ paths.
    """
    zones = _load_zones()
    seen_paths: dict[str, Path] = {}  # url → local path, deduped

    for zone_key, zone in zones.items():
        cx = (zone["x_min"] + zone["x_max"]) / 2.0
        cy = (zone["y_min"] + zone["y_max"]) / 2.0
        logger.info("Zone %s center: (%.0f, %.0f)", zone_key, cx, cy)
        try:
            path = fetch_laz_for_patch(cx, cy)
            seen_paths[path.name] = path
        except Exception as e:
            logger.error("Failed to download zone %s: %s", zone_key, e)

    return list(seen_paths.values())


def download_for_point(x: float, y: float) -> Path:
    """Download the LAZ tile containing a single point in EPSG:25831."""
    return fetch_laz_for_patch(x, y)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Download ICGC LiDAR LAZ tiles for Montseny.",
    )
    parser.add_argument(
        "--zone",
        help="Single zone name (e.g. 'low', 'summit'). Default = all zones.",
    )
    parser.add_argument(
        "--x", type=float,
        help="Manual EPSG:25831 easting (use with --y).",
    )
    parser.add_argument(
        "--y", type=float,
        help="Manual EPSG:25831 northing (use with --x).",
    )
    args = parser.parse_args()

    if args.x is not None and args.y is not None:
        path = download_for_point(args.x, args.y)
        print(f"\nDownloaded: {path}")
        return

    if args.zone is not None:
        zones = _load_zones()
        if args.zone not in zones:
            sys.exit(f"Unknown zone '{args.zone}'. Options: {list(zones)}")
        zone = zones[args.zone]
        cx = (zone["x_min"] + zone["x_max"]) / 2.0
        cy = (zone["y_min"] + zone["y_max"]) / 2.0
        path = fetch_laz_for_patch(cx, cy)
        print(f"\nDownloaded: {path}")
        return

    paths = download_all_zones()
    total_mb = sum(p.stat().st_size for p in paths) / 1e6
    print(f"\n{'='*55}")
    print("  ICGC LiDAR Download Complete")
    print(f"{'='*55}")
    print(f"  Unique tiles: {len(paths)}")
    print(f"  Total size:   {total_mb:.0f} MB")
    print(f"  Location:     {LAZ_CACHE}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
