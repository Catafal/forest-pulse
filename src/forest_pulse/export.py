"""Export tree inventories and change reports in GIS and human-readable formats.

Supports GeoJSON (QGIS), Shapefile (ArcGIS), CSV, and HTML reports.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from forest_pulse.temporal import ChangeReport


def to_geojson(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Export georeferenced tree inventory as GeoJSON.

    Output is directly loadable in QGIS, Leaflet, or any GIS tool.

    Args:
        gdf: GeoDataFrame with tree geometries and attributes.
        output_path: Path to write the GeoJSON file.

    Returns:
        Path to the written file.
    """
    # TODO: Implement GeoJSON export
    raise NotImplementedError


def to_shapefile(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Export as ESRI Shapefile for ArcGIS compatibility."""
    # TODO: Implement Shapefile export
    raise NotImplementedError


def to_csv(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Export tree inventory as flat CSV with lat/lon columns."""
    # TODO: Implement CSV export
    raise NotImplementedError


def to_report(change: ChangeReport, output_path: str | Path) -> Path:
    """Generate HTML change report with summary statistics and maps.

    Includes: tree count comparison, loss percentage, declining trees list,
    health transitions, and a map showing missing/declining trees.
    """
    # TODO: Implement HTML report generation
    raise NotImplementedError
