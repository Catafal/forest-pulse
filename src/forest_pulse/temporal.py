"""Temporal change detection between two georeferenced tree inventories.

Spatially matches trees across two time periods using nearest-neighbor
within a GPS tolerance. Produces a ChangeReport that answers the questions
foresters actually care about:
  - how many trees disappeared?
  - how many shrank significantly?
  - whose health degraded?

Matching strategy: `gpd.sjoin_nearest` with a `max_distance` filter.
Fast (uses a spatial index internally), deterministic, no random tie-breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import geopandas as gpd

logger = logging.getLogger(__name__)

# Health severity ordering — higher is worse. Used to detect degradation.
_HEALTH_SEVERITY = {"healthy": 0, "unknown": 0, "stressed": 1, "dead": 2}

# Declining crown threshold — a matched tree whose crown shrank by more
# than this fraction (20%) is flagged as "declining". Tuned for annual
# orthophoto cadence; shorter intervals may need a tighter threshold.
_DECLINING_THRESHOLD = -0.20


@dataclass
class TreeMatch:
    """A tree matched across two time periods."""

    tree_id_before: int
    tree_id_after: int
    distance_m: float           # straight-line distance between matched points
    crown_area_change: float    # (after - before) / before; negative = shrinking
    health_before: str
    health_after: str


@dataclass
class ChangeReport:
    """Summary of changes between two tree inventories."""

    date_before: str
    date_after: str
    trees_before: int
    trees_after: int
    matched: list[TreeMatch] = field(default_factory=list)
    missing: list[int] = field(default_factory=list)   # t₀ IDs with no t₁ match
    new: list[int] = field(default_factory=list)       # t₁ IDs with no t₀ match

    @property
    def tree_loss_count(self) -> int:
        return len(self.missing)

    @property
    def tree_loss_pct(self) -> float:
        if self.trees_before == 0:
            return 0.0
        return (len(self.missing) / self.trees_before) * 100.0

    @property
    def declining_trees(self) -> list[TreeMatch]:
        """Trees whose crown area shrank by more than 20%."""
        return [m for m in self.matched if m.crown_area_change < _DECLINING_THRESHOLD]

    @property
    def health_degraded(self) -> list[TreeMatch]:
        """Trees whose health class worsened between the two periods."""
        return [
            m for m in self.matched
            if _HEALTH_SEVERITY.get(m.health_after, 0)
            > _HEALTH_SEVERITY.get(m.health_before, 0)
        ]


def compare_periods(
    gdf_before: gpd.GeoDataFrame,
    gdf_after: gpd.GeoDataFrame,
    match_tolerance_m: float = 2.0,
    date_before: str = "unknown",
    date_after: str = "unknown",
) -> ChangeReport:
    """Spatial-match trees across two time periods and compute a change report.

    Uses GeoPandas `sjoin_nearest` with a max_distance filter. Both
    GeoDataFrames must be in the same projected CRS (meters — do NOT pass
    WGS84). The georeference() output in EPSG:25831 is the expected input.

    Args:
        gdf_before: Trees at time t₀. Must have `tree_id` and Point geometry.
        gdf_after: Trees at time t₁. Same schema.
        match_tolerance_m: Maximum distance (meters) for a valid match.
            Defaults to 2m — a typical forestry match tolerance that
            accommodates mild orthophoto misregistration.
        date_before: Human-readable date for t₀ (e.g., "2022-06").
        date_after: Human-readable date for t₁.

    Returns:
        ChangeReport with matched pairs, missing (t₀ not found), and
        new (t₁ not found in t₀) tree IDs.

    Raises:
        ValueError: If either GeoDataFrame has a non-projected CRS
            (geographic coordinates can't be matched by meters).
    """
    _validate_projected_crs(gdf_before, "gdf_before")
    _validate_projected_crs(gdf_after, "gdf_after")

    n_before = len(gdf_before)
    n_after = len(gdf_after)
    logger.info(
        "compare_periods: %d before, %d after, tolerance=%.1fm",
        n_before, n_after, match_tolerance_m,
    )

    # Edge case: either side empty → report everything as missing/new
    if n_before == 0 or n_after == 0:
        return ChangeReport(
            date_before=date_before,
            date_after=date_after,
            trees_before=n_before,
            trees_after=n_after,
            matched=[],
            missing=list(gdf_before["tree_id"]) if n_before else [],
            new=list(gdf_after["tree_id"]) if n_after else [],
        )

    # Spatial nearest-neighbor join. `how="left"` keeps every row in
    # gdf_before; unmatched rows have NaN in the right-side columns.
    # `max_distance` filters out matches beyond tolerance.
    joined = gpd.sjoin_nearest(
        gdf_before[["tree_id", "crown_area_m2", "geometry"]],
        gdf_after[["tree_id", "crown_area_m2", "geometry"]].rename(
            columns={"tree_id": "tree_id_after", "crown_area_m2": "crown_area_m2_after"},
        ),
        how="left",
        max_distance=match_tolerance_m,
        distance_col="_dist_m",
    )

    # Attach health_label columns (absent in pure georef output, present
    # if health_scores were supplied). Use 'unknown' as default.
    health_before = _get_health_map(gdf_before)
    health_after = _get_health_map(gdf_after)

    matched: list[TreeMatch] = []
    missing: list[int] = []
    matched_after_ids: set[int] = set()

    for _, row in joined.iterrows():
        tid_before = int(row["tree_id"])
        # NaN right-side means no match within tolerance.
        if _is_unmatched(row):
            missing.append(tid_before)
            continue

        tid_after = int(row["tree_id_after"])
        matched_after_ids.add(tid_after)

        area_before = float(row["crown_area_m2"])
        area_after = float(row["crown_area_m2_after"])
        area_change = (area_after - area_before) / area_before if area_before > 0 else 0.0

        matched.append(TreeMatch(
            tree_id_before=tid_before,
            tree_id_after=tid_after,
            distance_m=round(float(row["_dist_m"]), 3),
            crown_area_change=round(area_change, 4),
            health_before=health_before.get(tid_before, "unknown"),
            health_after=health_after.get(tid_after, "unknown"),
        ))

    # Trees in gdf_after that never appeared as a match are "new"
    new = [
        int(tid) for tid in gdf_after["tree_id"]
        if int(tid) not in matched_after_ids
    ]

    report = ChangeReport(
        date_before=date_before,
        date_after=date_after,
        trees_before=n_before,
        trees_after=n_after,
        matched=matched,
        missing=missing,
        new=new,
    )
    logger.info(
        "compare_periods: %d matched, %d missing, %d new",
        len(matched), len(missing), len(new),
    )
    return report


def _validate_projected_crs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Raise if the GeoDataFrame is not in a projected (metric) CRS.

    Matching by distance in meters requires a projected CRS. Geographic
    CRS (like EPSG:4326) would need conversion or spherical distance
    calculations, which aren't part of this MVP.
    """
    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS — set one before matching.")
    if gdf.crs.is_geographic:
        raise ValueError(
            f"{name} is in geographic CRS {gdf.crs}. "
            "Reproject to a projected CRS (e.g., EPSG:25831) before matching."
        )


def _get_health_map(gdf: gpd.GeoDataFrame) -> dict[int, str]:
    """Extract {tree_id: health_label} map from a GeoDataFrame.

    Returns an empty dict if the gdf has no health_label column (the
    output of georeference() without health_scores).
    """
    if "health_label" not in gdf.columns:
        return {}
    return {int(tid): str(lbl) for tid, lbl in zip(gdf["tree_id"], gdf["health_label"])}


def _is_unmatched(row) -> bool:
    """True if an sjoin_nearest row has no right-side match.

    sjoin_nearest marks unmatched rows with NaN in the right side's
    tree_id column. We check the renamed column `tree_id_after`.
    """
    val = row.get("tree_id_after")
    try:
        return val is None or (isinstance(val, float) and val != val)  # NaN check
    except (TypeError, ValueError):
        return True
