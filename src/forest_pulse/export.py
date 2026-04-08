"""Export tree inventories and change reports in GIS + tabular formats.

All functions operate on the GeoDataFrame schema produced by
`forest_pulse.georef.georeference()`. They are thin wrappers around
GeoPandas drivers and standard library writers, with two opinions:

  1. GeoJSON and CSV are written in EPSG:4326 (WGS84). Every GIS tool,
     web map, and data-science notebook expects lon/lat.
  2. Shapefile keeps the input CRS (usually EPSG:25831). ESRI Shapefile
     handles any CRS natively and professional QGIS users often prefer
     projected coordinates for accurate area measurements.

Output directories are created on demand. All writers return the Path
they wrote to so callers can chain or log.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd

from forest_pulse.temporal import ChangeReport

logger = logging.getLogger(__name__)

# CRS used when writing GeoJSON and CSV. WGS84 lon/lat is universally
# understood by downstream tools.
WGS84 = "EPSG:4326"


def to_geojson(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Write a tree inventory as GeoJSON, reprojected to WGS84.

    GeoJSON's RFC 7946 mandates WGS84. Any tool that loads GeoJSON will
    assume lon/lat, so we reproject before writing.

    Args:
        gdf: Tree inventory from `georeference()`. Any CRS is accepted —
            it will be reprojected.
        output_path: Destination file path. Parent dirs created if missing.

    Returns:
        The Path that was written.
    """
    path = _prepare_path(output_path)
    gdf_wgs84 = gdf.to_crs(WGS84) if gdf.crs != WGS84 else gdf
    gdf_wgs84.to_file(path, driver="GeoJSON")
    logger.info("Wrote GeoJSON (%d features): %s", len(gdf_wgs84), path)
    return path


def to_shapefile(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Write a tree inventory as ESRI Shapefile in its original CRS.

    Shapefile preserves whatever CRS the GeoDataFrame carries. We
    deliberately do NOT reproject — a metric CRS like EPSG:25831
    preserves area/distance accuracy for forestry work.

    Args:
        gdf: Tree inventory.
        output_path: Destination `.shp` path. Parent dirs created if
            missing. Writes companion files (.shx, .dbf, .prj) alongside.

    Returns:
        Path to the `.shp` file.
    """
    path = _prepare_path(output_path)
    gdf.to_file(path, driver="ESRI Shapefile")
    logger.info("Wrote Shapefile (%d features, CRS=%s): %s", len(gdf), gdf.crs, path)
    return path


def to_csv(gdf: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    """Write a tree inventory as CSV with lon + lat columns.

    Reprojects to WGS84, extracts lon/lat from the Point geometry, drops
    the geometry column (CSV is flat — no WKT blob). All other columns
    are preserved verbatim.

    Args:
        gdf: Tree inventory with Point geometries.
        output_path: Destination `.csv` path.

    Returns:
        The Path that was written.
    """
    path = _prepare_path(output_path)
    gdf_wgs84 = gdf.to_crs(WGS84) if gdf.crs != WGS84 else gdf

    # Extract lon/lat — Phase 11b adds Polygon geometries to the
    # output schema, so we use `.centroid.x / .centroid.y` which
    # works uniformly for both Points (centroid is the point itself)
    # and Polygons (centroid is the geometric center).
    flat = gdf_wgs84.drop(columns="geometry").copy()
    flat["lon"] = gdf_wgs84.geometry.apply(lambda g: round(g.centroid.x, 7))
    flat["lat"] = gdf_wgs84.geometry.apply(lambda g: round(g.centroid.y, 7))

    # Move lon/lat to the front so humans see them first when opening
    # the CSV in a spreadsheet.
    front = ["lon", "lat"]
    flat = flat[front + [c for c in flat.columns if c not in front]]

    flat.to_csv(path, index=False)
    logger.info("Wrote CSV (%d rows): %s", len(flat), path)
    return path


def to_change_report(change: ChangeReport, output_path: str | Path) -> Path:
    """Serialize a ChangeReport to JSON.

    The report contains dataclasses (ChangeReport, TreeMatch). We use
    `dataclasses.asdict` which walks nested dataclasses recursively and
    produces plain dicts / lists that json can handle.

    Args:
        change: The ChangeReport to serialize.
        output_path: Destination `.json` path.

    Returns:
        The Path that was written.
    """
    path = _prepare_path(output_path)
    payload = asdict(change)
    # Add derived summary so the JSON is self-describing without needing
    # the Python dataclass to compute them.
    payload["summary"] = {
        "tree_loss_count": change.tree_loss_count,
        "tree_loss_pct": round(change.tree_loss_pct, 2),
        "declining_trees_count": len(change.declining_trees),
        "health_degraded_count": len(change.health_degraded),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote change report: %s", path)
    return path


def _prepare_path(output_path: str | Path) -> Path:
    """Coerce to Path and create parent dirs if they don't exist."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
