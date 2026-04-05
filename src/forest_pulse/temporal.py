"""Temporal change detection between two georeferenced tree inventories.

Spatially matches trees across time periods using GPS coordinates,
then computes change metrics: tree loss, crown decline, health progression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd


@dataclass
class TreeMatch:
    """A tree matched across two time periods."""

    tree_id_before: int
    tree_id_after: int
    distance_m: float           # GPS distance between matched positions
    crown_area_change: float    # relative change: (after - before) / before
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
    missing: list[int] = field(default_factory=list)   # tree IDs in t₀ not found at t₁
    new: list[int] = field(default_factory=list)        # tree IDs in t₁ not found at t₀

    @property
    def tree_loss_count(self) -> int:
        return len(self.missing)

    @property
    def tree_loss_pct(self) -> float:
        if self.trees_before == 0:
            return 0.0
        return (len(self.missing) / self.trees_before) * 100

    @property
    def declining_trees(self) -> list[TreeMatch]:
        """Trees whose crown area shrank by >20%."""
        return [m for m in self.matched if m.crown_area_change < -0.20]

    @property
    def health_degraded(self) -> list[TreeMatch]:
        """Trees whose health status worsened (e.g., healthy → stressed)."""
        severity = {"healthy": 0, "stressed": 1, "dead": 2}
        return [
            m for m in self.matched
            if severity.get(m.health_after, 0) > severity.get(m.health_before, 0)
        ]


def compare_periods(
    gdf_before: gpd.GeoDataFrame,
    gdf_after: gpd.GeoDataFrame,
    match_tolerance_m: float = 2.0,
) -> ChangeReport:
    """Spatial-match trees across two time periods and compute changes.

    Uses nearest-neighbor matching within a GPS tolerance radius.
    Trees in t₀ without a match in t₁ are classified as "missing" (possible death/removal).
    Trees in t₁ without a match in t₀ are classified as "new" (possible growth/detection improvement).

    Args:
        gdf_before: GeoDataFrame of trees at time t₀.
        gdf_after: GeoDataFrame of trees at time t₁.
        match_tolerance_m: Maximum GPS distance (meters) for a valid match.

    Returns:
        ChangeReport with matched trees, missing trees, and change metrics.
    """
    # TODO: Implement spatial matching + change computation
    # 1. Build spatial index on gdf_after
    # 2. For each tree in gdf_before, find nearest neighbor in gdf_after
    # 3. If distance < tolerance → match. Otherwise → missing.
    # 4. Trees in gdf_after not matched → new.
    # 5. For matched: compute crown area change, health transition
    raise NotImplementedError("compare_periods not yet implemented")
