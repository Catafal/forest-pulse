"""Tests for GeoJSON / CSV / Shapefile / change report exports."""

import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from forest_pulse.export import (
    to_change_report,
    to_csv,
    to_geojson,
    to_shapefile,
)
from forest_pulse.temporal import ChangeReport, TreeMatch


def _sample_gdf(n: int = 3) -> gpd.GeoDataFrame:
    """Build a small GeoDataFrame in EPSG:25831 for export tests."""
    return gpd.GeoDataFrame(
        pd.DataFrame({
            "tree_id": list(range(n)),
            "confidence": [0.9, 0.7, 0.5][:n],
            "crown_area_m2": [5.0, 10.0, 15.0][:n],
            "health_label": ["healthy", "stressed", "dead"][:n],
        }),
        geometry=[Point(450000 + i * 2, 4600000 + i * 2) for i in range(n)],
        crs="EPSG:25831",
    )


def test_to_geojson_valid_and_wgs84():
    """GeoJSON output is valid and reprojected to EPSG:4326."""
    gdf = _sample_gdf(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trees.geojson"
        result = to_geojson(gdf, path)

        assert result == path
        assert path.exists()

        # Load it back and verify CRS + count
        loaded = gpd.read_file(path)
        assert loaded.crs.to_string() == "EPSG:4326"
        assert len(loaded) == 3


def test_to_geojson_creates_parent_dirs():
    """Missing parent dirs are created automatically."""
    gdf = _sample_gdf(1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "deeper" / "trees.geojson"
        to_geojson(gdf, path)
        assert path.exists()


def test_to_csv_has_lon_lat_columns():
    """CSV output has lon and lat as the first columns."""
    gdf = _sample_gdf(3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trees.csv"
        to_csv(gdf, path)

        loaded = pd.read_csv(path)
        assert list(loaded.columns[:2]) == ["lon", "lat"]
        assert len(loaded) == 3
        # lon should be near the Catalunya range (~2 degrees east)
        assert 1.0 < loaded["lon"].iloc[0] < 3.0
        assert 41.0 < loaded["lat"].iloc[0] < 43.0


def test_to_csv_no_geometry_column():
    """CSV drops the binary geometry column."""
    gdf = _sample_gdf(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trees.csv"
        to_csv(gdf, path)
        loaded = pd.read_csv(path)
        assert "geometry" not in loaded.columns


def test_to_shapefile_writes_companion_files():
    """Shapefile writes .shp + .shx + .dbf + .prj together."""
    gdf = _sample_gdf(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trees.shp"
        to_shapefile(gdf, path)

        # All four companion files must exist
        for ext in (".shp", ".shx", ".dbf", ".prj"):
            assert path.with_suffix(ext).exists(), f"missing {ext}"

        loaded = gpd.read_file(path)
        assert len(loaded) == 2
        # Shapefile preserves original CRS (EPSG:25831)
        assert loaded.crs.to_string() == "EPSG:25831"


def test_to_change_report_json_structure():
    """ChangeReport JSON has all expected fields + a summary block."""
    change = ChangeReport(
        date_before="2022-06",
        date_after="2024-06",
        trees_before=10,
        trees_after=12,
        matched=[
            TreeMatch(
                tree_id_before=0, tree_id_after=0,
                distance_m=0.5, crown_area_change=-0.1,
                health_before="healthy", health_after="stressed",
            ),
        ],
        missing=[1, 2],
        new=[5, 6, 7],
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "change.json"
        to_change_report(change, path)

        with open(path) as f:
            data = json.load(f)

        assert data["date_before"] == "2022-06"
        assert data["trees_before"] == 10
        assert data["missing"] == [1, 2]
        assert len(data["matched"]) == 1
        assert "summary" in data
        assert data["summary"]["tree_loss_count"] == 2
