"""Phase 11a validation probe: are LiDAR peaks in Montseny mostly real trees?

Before committing to the LiDAR-first detection architecture
(`detect_trees_from_lidar`), this probe runs statistical sanity
checks on LiDAR tree-top extraction across many Montseny patches
and prints a Go / No-Go verdict against published Mediterranean
forest priors.

Checks:
  1. Trees per hectare per patch. Realistic Mediterranean mixed
     forest ranges 50-800 trees/ha. Outside that range is a red
     flag (<50 = missing trees or LiDAR failure; >800 = probable
     over-detection of shrubs/artifacts).
  2. Peak height distribution. Mean should be 5-30 m; >80% of
     peaks should be >= 7 m (below 7 m are likely shrubs that
     crossed the 5 m CHM threshold).
  3. Spatial clustering. Mean nearest-neighbor distance should
     be 3-10 m. Too tight → over-detection; too spread → gaps.

Go / No-Go verdict:
  GO if density in [50, 800] AND mean height in [5, 30] AND
    >= 80% of peaks >= 7 m.
  NO-GO with a diagnostic line otherwise. In that case, STOP and
  reassess the Phase 11a hypothesis before building the detector.

Patch selection:
  By default, walks `patches_metadata.csv` and processes only
  patches whose LAZ tile is already cached in `data/montseny/lidar/`.
  This avoids triggering fresh downloads during the probe. Stops
  after `--n` successful patches (default 50) OR the end of the
  cached subset.

Usage:
    python scripts/probe_lidar_peaks.py           # default 50 patches
    python scripts/probe_lidar_peaks.py --n 100
    python scripts/probe_lidar_peaks.py --patch 0043.jpg  # single patch
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from forest_pulse.lidar import (
    LAZ_CACHE,
    _icgc_laz_url,
    _read_chm_raster,
    compute_chm_from_laz,
    find_tree_tops_from_chm,
)
from forest_pulse.patches import iter_patch_names

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
METADATA_CSV = PROJECT_ROOT / "data" / "montseny" / "patches_metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "lidar_probe"

# Montseny patch geometry.
PATCH_SIZE_PX = 640
PATCH_SIZE_M = 160.0
PATCH_AREA_HA = (PATCH_SIZE_M / 100.0) ** 2  # 160m x 160m = 2.56 ha

# Published Mediterranean forest density ranges (trees >= 5m tall).
# Sources: Spanish National Forest Inventory (IFN3), Catalan IEFC.
# - Open Pinus halepensis stands: 50-150 trees/ha
# - Mature Mediterranean mixed: 200-400 trees/ha
# - Dense holm oak (Q. ilex) stands: 400-800 trees/ha
# - Dense beech (F. sylvatica) stands: 300-500 trees/ha
MIN_REALISTIC_DENSITY = 50.0
MAX_REALISTIC_DENSITY = 800.0

# Peak height sanity bounds.
MIN_REALISTIC_MEAN_HEIGHT = 5.0
MAX_REALISTIC_MEAN_HEIGHT = 30.0

# Fraction of peaks that should be >= 7 m (below 7m are likely
# shrubs that barely crossed the 5m threshold — if >20% are in
# the 5-7m band, the CHM is probably picking up shrub noise).
MIN_FRACTION_PEAKS_ABOVE_7M = 0.80

# Spatial clustering — mean nearest-neighbor distance bounds.
# At 3m min_distance in the local-max filter, minimum NN distance
# is 3m by construction. Realistic Mediterranean canopies have
# mean NN in 3-10m.
MIN_REALISTIC_MEAN_NN = 3.0
MAX_REALISTIC_MEAN_NN = 15.0

# Default patch selection.
DEFAULT_N_PATCHES = 50


def _get_patch_center(patch_name: str) -> tuple[float, float]:
    """Small inline helper to avoid circular import via patches.py."""
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            if row["filename"] == patch_name:
                return float(row["x_center"]), float(row["y_center"])
    raise ValueError(f"Patch {patch_name} not found in {METADATA_CSV}")


def _laz_is_cached(x_center: float, y_center: float) -> bool:
    """Check if the LAZ tile for this patch center is already on disk.

    Reuses the same URL → filename mapping as `fetch_laz_for_patch`
    without triggering a download. Used to filter the probe's patch
    list to "cached-only" mode, avoiding any network calls during
    the gate check.
    """
    url = _icgc_laz_url(x_center, y_center)
    filename = url.rsplit("/", 1)[-1]
    path = LAZ_CACHE / filename
    return path.exists() and path.stat().st_size > 1_000_000


def _compute_patch_stats(patch_name: str) -> dict | None:
    """Run the CHM + tree-top pipeline on a single patch.

    Returns a dict of statistics for aggregation, or None if the
    patch can't be processed (missing LAZ, no peaks, etc.).
    """
    try:
        x_center, y_center = _get_patch_center(patch_name)
    except ValueError:
        return None

    if not _laz_is_cached(x_center, y_center):
        # Skip — don't download during a probe that's supposed to be fast.
        return None

    half = PATCH_SIZE_M / 2.0
    bounds = (
        x_center - half, y_center - half,
        x_center + half, y_center + half,
    )

    try:
        # LAZ already cached, so this is essentially free.
        from forest_pulse.lidar import fetch_laz_for_patch
        laz_path = fetch_laz_for_patch(x_center, y_center)
        chm_path = compute_chm_from_laz(laz_path, bounds)
        chm, transform = _read_chm_raster(chm_path)
        positions, heights = find_tree_tops_from_chm(
            chm, transform, return_heights=True,
        )
    except Exception as e:
        logger.warning("Patch %s failed: %s", patch_name, e)
        return None

    n_peaks = len(positions)
    if n_peaks == 0:
        return {
            "name": patch_name,
            "n_peaks": 0,
            "density_per_ha": 0.0,
            "mean_height_m": 0.0,
            "median_height_m": 0.0,
            "p5_height_m": 0.0,
            "p95_height_m": 0.0,
            "frac_above_7m": 0.0,
            "mean_nn_distance_m": float("nan"),
        }

    heights_arr = np.asarray(heights, dtype=np.float64)
    pos_arr = np.asarray(positions, dtype=np.float64)

    density = n_peaks / PATCH_AREA_HA
    frac_above_7m = float(np.mean(heights_arr >= 7.0))

    # Spatial clustering: mean nearest-neighbor distance.
    if n_peaks >= 2:
        tree = cKDTree(pos_arr)
        # k=2 because the nearest is the point itself at distance 0.
        nn_dists, _ = tree.query(pos_arr, k=2)
        mean_nn = float(nn_dists[:, 1].mean())
    else:
        mean_nn = float("nan")

    return {
        "name": patch_name,
        "n_peaks": n_peaks,
        "density_per_ha": round(density, 2),
        "mean_height_m": round(float(heights_arr.mean()), 2),
        "median_height_m": round(float(np.median(heights_arr)), 2),
        "p5_height_m": round(float(np.percentile(heights_arr, 5)), 2),
        "p95_height_m": round(float(np.percentile(heights_arr, 95)), 2),
        "frac_above_7m": round(frac_above_7m, 3),
        "mean_nn_distance_m": round(mean_nn, 2),
    }


def _aggregate(records: list[dict]) -> dict:
    """Roll per-patch stats into an aggregate verdict dict."""
    if not records:
        return {}

    densities = np.array([r["density_per_ha"] for r in records])
    mean_heights = np.array([r["mean_height_m"] for r in records])
    fracs_above_7 = np.array([r["frac_above_7m"] for r in records])
    mean_nns = np.array([
        r["mean_nn_distance_m"] for r in records
        if not np.isnan(r["mean_nn_distance_m"])
    ])

    total_peaks = sum(r["n_peaks"] for r in records)
    total_area_ha = len(records) * PATCH_AREA_HA

    return {
        "n_patches": len(records),
        "total_peaks": total_peaks,
        "total_area_ha": round(total_area_ha, 2),
        "global_density_per_ha": round(total_peaks / total_area_ha, 2),
        "mean_density": round(float(densities.mean()), 2),
        "median_density": round(float(np.median(densities)), 2),
        "min_density": round(float(densities.min()), 2),
        "max_density": round(float(densities.max()), 2),
        "mean_height_overall": round(float(mean_heights.mean()), 2),
        "median_frac_above_7m": round(float(np.median(fracs_above_7)), 3),
        "mean_nn_distance": round(float(mean_nns.mean()), 2) if len(mean_nns) else float("nan"),
    }


def _verdict(agg: dict) -> tuple[str, list[str]]:
    """Return 'GO' or 'NO-GO' + a list of issues found.

    Any single check failing → NO-GO. All pass → GO.
    """
    issues: list[str] = []

    density = agg.get("global_density_per_ha", 0)
    if density < MIN_REALISTIC_DENSITY:
        issues.append(
            f"Global density {density:.0f} trees/ha is BELOW the realistic "
            f"Mediterranean minimum of {MIN_REALISTIC_DENSITY}. LiDAR may "
            f"be missing too many trees (crown merging?)."
        )
    elif density > MAX_REALISTIC_DENSITY:
        issues.append(
            f"Global density {density:.0f} trees/ha is ABOVE the realistic "
            f"Mediterranean maximum of {MAX_REALISTIC_DENSITY}. LiDAR may "
            f"be over-detecting (shrub noise, CHM artifacts)."
        )

    mh = agg.get("mean_height_overall", 0)
    if mh < MIN_REALISTIC_MEAN_HEIGHT:
        issues.append(
            f"Mean peak height {mh:.1f} m is BELOW {MIN_REALISTIC_MEAN_HEIGHT} m "
            f"— peaks are too short to be trees."
        )
    elif mh > MAX_REALISTIC_MEAN_HEIGHT:
        issues.append(
            f"Mean peak height {mh:.1f} m is ABOVE {MAX_REALISTIC_MEAN_HEIGHT} m "
            f"— peaks are unrealistically tall."
        )

    frac7 = agg.get("median_frac_above_7m", 0)
    if frac7 < MIN_FRACTION_PEAKS_ABOVE_7M:
        issues.append(
            f"Median fraction of peaks >= 7 m is {frac7:.2f}, below the "
            f"{MIN_FRACTION_PEAKS_ABOVE_7M:.2f} threshold — too many "
            f"shrub-height peaks, CHM may be picking up sub-canopy noise."
        )

    nn = agg.get("mean_nn_distance", float("nan"))
    if not np.isnan(nn):
        if nn < MIN_REALISTIC_MEAN_NN:
            issues.append(
                f"Mean nearest-neighbor distance {nn:.1f} m is below the "
                f"{MIN_REALISTIC_MEAN_NN} m minimum — peaks are implausibly "
                f"close, possible over-detection."
            )
        elif nn > MAX_REALISTIC_MEAN_NN:
            issues.append(
                f"Mean nearest-neighbor distance {nn:.1f} m exceeds "
                f"{MAX_REALISTIC_MEAN_NN} m — peaks too sparse."
            )

    return ("GO" if not issues else "NO-GO", issues)


def _save_csv(records: list[dict], path: Path) -> None:
    """Save per-patch records as CSV."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved per-patch probe CSV to %s", path)


def run_probe(patches: list[str], limit: int) -> None:
    """Top-level probe driver."""
    print(f"LiDAR peak validation probe — target {limit} patches")
    print(f"Metadata: {METADATA_CSV}")
    print(f"LAZ cache dir: {LAZ_CACHE}")
    print()

    records: list[dict] = []
    skipped_no_laz = 0

    for i, name in enumerate(patches):
        if len(records) >= limit:
            break

        rec = _compute_patch_stats(name)
        if rec is None:
            skipped_no_laz += 1
            continue

        n = rec["n_peaks"]
        density = rec["density_per_ha"]
        mh = rec["mean_height_m"]
        frac7 = rec["frac_above_7m"]
        print(
            f"  [{len(records) + 1}] {name}: "
            f"{n} peaks, {density:.0f}/ha, mean={mh:.1f}m, "
            f">=7m={frac7:.2f}"
        )
        records.append(rec)

    if not records:
        print()
        print("NO CACHED LAZ TILES FOUND. Cannot run probe without network.")
        print("Either download some LAZ tiles first (via earlier phase runs)")
        print("or remove the cache-only gate in probe_lidar_peaks.py.")
        return

    agg = _aggregate(records)

    print()
    print("=" * 72)
    print("  LiDAR Peak Probe — Aggregate Statistics")
    print("=" * 72)
    print(f"  Patches processed:  {agg['n_patches']} / {limit}")
    print(f"  Skipped (no LAZ):   {skipped_no_laz}")
    print(f"  Total peaks:        {agg['total_peaks']}")
    print(f"  Total area (ha):    {agg['total_area_ha']}")
    print()
    print(f"  Global density:     {agg['global_density_per_ha']} trees/ha")
    print(f"  Per-patch density:  mean={agg['mean_density']}, "
          f"median={agg['median_density']}, "
          f"min={agg['min_density']}, max={agg['max_density']}")
    print(f"  Mean peak height:   {agg['mean_height_overall']} m")
    print(f"  Median frac >= 7m:  {agg['median_frac_above_7m']}")
    print(f"  Mean NN distance:   {agg['mean_nn_distance']} m")
    print()

    verdict, issues = _verdict(agg)
    print(f"  VERDICT: {verdict}")
    if issues:
        print()
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All checks passed. Proceed with Phase 11a implementation.")
    print("=" * 72)
    print()

    # Published priors for user reference.
    print("  Reference priors (Spanish NFI / Catalan IEFC):")
    print(f"    Realistic density:   {MIN_REALISTIC_DENSITY}-{MAX_REALISTIC_DENSITY} trees/ha")
    print(f"    Realistic mean ht:   {MIN_REALISTIC_MEAN_HEIGHT}-{MAX_REALISTIC_MEAN_HEIGHT} m")
    print(f"    Min frac >= 7m:      {MIN_FRACTION_PEAKS_ABOVE_7M}")
    print(f"    Realistic NN dist:   {MIN_REALISTIC_MEAN_NN}-{MAX_REALISTIC_MEAN_NN} m")
    print()

    _save_csv(records, OUTPUT_DIR / "probe_summary.csv")


def main():
    logging.basicConfig(
        level=logging.WARNING,  # quieter by default — probe is a user-facing tool
        format="%(name)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Phase 11a validation probe: are Montseny LiDAR peaks "
                    "mostly real trees?",
    )
    parser.add_argument(
        "--patch", action="append", default=None,
        help="Specific patch(es) to probe. Repeat for multiple. "
             "Default = walk patches_metadata.csv cached-only.",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_PATCHES,
        help=f"Number of patches to process (default {DEFAULT_N_PATCHES}).",
    )
    args = parser.parse_args()

    if args.patch:
        patches = args.patch
    else:
        patches = iter_patch_names(METADATA_CSV)

    run_probe(patches, limit=args.n)


if __name__ == "__main__":
    main()
