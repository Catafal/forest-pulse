"""Watershed crown segmentation — derives per-tree polygons from a CHM.

Phase 11b implementation. The classical forestry approach: each tree's
crown is the watershed basin around its peak on the (inverted) CHM.

Algorithm sketch:
  1. Project LiDAR tree-top world coordinates to pixel positions on
     the CHM raster, building a labeled marker raster.
  2. Convert the CHM into a uint8 "cost" image (tree-tops → low cost,
     gaps → high cost) so the watershed floods from each marker
     outward and stops at canopy gaps.
  3. Run scipy.ndimage.watershed_ift(cost, markers) — the Iterated
     Forest Transform watershed by Lotufo & Falcão. Real watershed,
     not Voronoi.
  4. Post-mask: pixels below `min_height_m` are excluded from all
     basins (they're gaps or ground, not crowns).
  5. Extract each labeled region as a shapely Polygon via
     rasterio.features.shapes — gives polygons in the CRS of the
     transform (EPSG:25831 for Catalunya).
  6. Replace empty / out-of-bounds / oversized basins with a fallback
     circle buffer of `fallback_radius_m`. This guarantees the output
     list always has one valid Polygon per input tree-top, in the
     same order.

## Why scipy.watershed_ift (not scikit-image)

scipy already has a real watershed implementation. scikit-image has a
slightly newer Vincent-Soille variant but adds a ~60 MB wheel and
doesn't change the result quality for sparse markers like ours
(LiDAR tree-tops have a 3 m min_distance enforced upstream by
find_tree_tops_from_chm). Pre-verified on synthetic data: scipy
correctly partitions adjacent gaussian bumps and the larger one gets
the bigger basin.

## Fallback behavior

The function ALWAYS returns a list of valid Polygons of length
`len(tree_tops_world)`, even when watershed cannot find a basin for
a particular marker. The fallback is a small circular buffer
centered on the tree-top in world coordinates. This means downstream
code can safely zip(detections, polygons) without None handling.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from rasterio.features import shapes as rio_shapes
from scipy import ndimage
from shapely.geometry import Point, Polygon
from shapely.geometry import shape as shapely_shape
from shapely.validation import make_valid

if TYPE_CHECKING:
    import rasterio

logger = logging.getLogger(__name__)

# ============================================================
# Constants — tuned for Montseny 25 cm/px patches with 0.5 m/px CHMs
# ============================================================

# Spanish Forest Inventory tree-vs-shrub cutoff. Same threshold used
# everywhere else in the project.
DEFAULT_MIN_CROWN_HEIGHT_M = 5.0

# Sanity cap on per-tree crown area. 150 m² ≈ 14 m diameter — the
# upper end of realistic individual crowns for any Montseny species
# including mature Fagus sylvatica (beech). Published max crown
# diameters: beech 12-16 m, holm oak 8-12 m, Pinus halepensis 6-10 m.
#
# Phase 12b calibration: the original 200 m² cap allowed 2,098
# over-segmented basins (0.92% of trees) that produced biologically
# implausible DBH (> 100 cm for conifers whose published max is
# 60-80 cm). Lowered to 150 m² to catch these while still allowing
# legitimate large beech / oak crowns.
DEFAULT_MAX_CROWN_AREA_M2 = 150.0

# Radius of the fallback circle when watershed can't produce a valid
# basin. Matches the Phase 11a fixed-radius default of 2.5 m.
DEFAULT_FALLBACK_CROWN_RADIUS_M = 2.5

# Buffer resolution for the fallback Point.buffer() — 16 segments
# gives a smooth circle without blowing up GeoJSON file size.
_FALLBACK_BUFFER_SEGMENTS = 16


# ============================================================
# Public API
# ============================================================


def segment_crowns_watershed(
    chm: np.ndarray,
    transform: "rasterio.Affine | None",
    tree_tops_world: list[tuple[float, float]],
    min_height_m: float = DEFAULT_MIN_CROWN_HEIGHT_M,
    max_crown_area_m2: float = DEFAULT_MAX_CROWN_AREA_M2,
    fallback_radius_m: float = DEFAULT_FALLBACK_CROWN_RADIUS_M,
) -> list[Polygon]:
    """Watershed-segment tree crowns from a CHM.

    For each input LiDAR tree-top world position, returns a shapely
    Polygon representing that tree's canopy footprint. The polygons
    are derived via marker-controlled watershed on the inverted CHM,
    with post-masking by `min_height_m` to exclude gaps.

    The function is GUARANTEED to return one valid Polygon per input
    tree-top, in the same order. Markers that produce no basin (or
    a basin that exceeds `max_crown_area_m2`) are replaced with a
    fallback circle of `fallback_radius_m`. Markers projecting
    outside the raster bounds also get fallbacks.

    Args:
        chm: 2D float array of canopy heights (meters). NaN/Inf
            treated as 0 (gap).
        transform: rasterio Affine pixel→world transform for the
            CHM. If None (e.g., unit-test scenarios with stubbed
            data), the function returns fallback circles for every
            input — graceful degradation.
        tree_tops_world: List of (x_world, y_world) tuples in the
            CRS of the transform. Order is preserved in the output.
        min_height_m: Pixels below this CHM value are masked from
            ALL watershed basins. Default 5 m.
        max_crown_area_m2: Maximum allowed basin area. Larger basins
            are replaced with fallback circles. Default 200 m².
        fallback_radius_m: Radius of the circular fallback buffer.
            Default 2.5 m (matches Phase 11a fixed-radius default).

    Returns:
        List of `shapely.geometry.Polygon`, one per input tree-top.
        Always the same length as `tree_tops_world`. Empty input →
        empty list.
    """
    n = len(tree_tops_world)
    if n == 0:
        return []

    # Graceful degradation: no transform → no projection possible →
    # all fallbacks. Used by unit tests with stubbed CHMs.
    if transform is None:
        return [_fallback_circle(p, fallback_radius_m) for p in tree_tops_world]

    # Empty / degenerate CHM → all fallbacks.
    if chm.size == 0:
        return [_fallback_circle(p, fallback_radius_m) for p in tree_tops_world]

    # Step 1: project markers from world coords → pixel coords.
    markers, in_bounds = _build_marker_raster(
        chm.shape, transform, tree_tops_world,
    )

    # Step 2-3: cost image + watershed. If NO markers landed inside
    # the raster, skip the watershed entirely and fall back.
    if markers.max() == 0:
        return [_fallback_circle(p, fallback_radius_m) for p in tree_tops_world]

    cost = _chm_to_watershed_cost(chm, min_height_m)
    labels = ndimage.watershed_ift(cost, markers)

    # Step 4: post-mask. Pixels below the height threshold are not
    # part of any crown — set them to background (label 0).
    safe_chm = np.where(np.isfinite(chm), chm, 0.0)
    labels[safe_chm < min_height_m] = 0

    # Step 5: extract polygons from the labeled raster.
    polygons_by_label = _polygons_from_labels(labels, transform)

    # Step 6: assemble per-input result with fallbacks.
    result: list[Polygon] = []
    n_fallback_oob = 0
    n_fallback_empty = 0
    n_fallback_oversize = 0
    for idx, world_pos in enumerate(tree_tops_world):
        label_id = idx + 1  # marker labels are 1-indexed
        if not in_bounds[idx]:
            n_fallback_oob += 1
            result.append(_fallback_circle(world_pos, fallback_radius_m))
            continue

        poly = polygons_by_label.get(label_id)
        if poly is None or poly.is_empty:
            n_fallback_empty += 1
            result.append(_fallback_circle(world_pos, fallback_radius_m))
            continue

        if poly.area > max_crown_area_m2:
            n_fallback_oversize += 1
            result.append(_fallback_circle(world_pos, fallback_radius_m))
            continue

        result.append(poly)

    n_real = n - (n_fallback_oob + n_fallback_empty + n_fallback_oversize)
    logger.info(
        "Watershed crown segmentation: %d trees → %d real basins, "
        "%d fallback (oob=%d, empty=%d, oversized=%d)",
        n, n_real,
        n_fallback_oob + n_fallback_empty + n_fallback_oversize,
        n_fallback_oob, n_fallback_empty, n_fallback_oversize,
    )
    return result


# ============================================================
# Internal helpers
# ============================================================


def _fallback_circle(
    world_pos: tuple[float, float],
    radius_m: float,
) -> Polygon:
    """Build a circular Polygon buffer around a tree-top world position.

    Used when watershed can't produce a valid basin (marker out of
    bounds, empty basin, basin too large). Returns a 16-quad-segment
    approximation of a circle (= 64 sides) — accurate enough for
    crown shapes while keeping the GeoJSON file compact.
    """
    pt = Point(world_pos[0], world_pos[1])
    return pt.buffer(radius_m, quad_segs=_FALLBACK_BUFFER_SEGMENTS)


def _build_marker_raster(
    shape: tuple[int, int],
    transform: "rasterio.Affine",
    tree_tops_world: list[tuple[float, float]],
) -> tuple[np.ndarray, list[bool]]:
    """Project tree-top world coords to a labeled marker raster.

    scipy.watershed_ift expects markers as int16 with labels 1..N
    at the seed pixel positions and 0 elsewhere. Out-of-bounds
    projections are dropped from the marker raster but recorded in
    `in_bounds` so the caller can substitute fallback circles.

    Args:
        shape: (rows, cols) of the CHM.
        transform: rasterio Affine. The inverse maps world → pixel.
        tree_tops_world: List of (x, y) world coordinates.

    Returns:
        Tuple of:
          - markers: int16 array of `shape`, with labels 1..N
          - in_bounds: list[bool] parallel to tree_tops_world,
            True when the marker landed inside the raster.
    """
    h, w = shape
    markers = np.zeros((h, w), dtype=np.int16)
    in_bounds: list[bool] = []
    inverse = ~transform

    for idx, (x_world, y_world) in enumerate(tree_tops_world):
        col_f, row_f = inverse * (x_world, y_world)
        col = int(round(col_f))
        row = int(round(row_f))
        if 0 <= row < h and 0 <= col < w:
            # Note: scipy.watershed_ift labels are limited to int16.
            # We use idx+1 as the label so 0 stays as background.
            markers[row, col] = idx + 1
            in_bounds.append(True)
        else:
            in_bounds.append(False)

    return markers, in_bounds


def _chm_to_watershed_cost(
    chm: np.ndarray,
    min_height_m: float,
) -> np.ndarray:
    """Convert a CHM into a uint8 cost image for watershed_ift.

    scipy.watershed_ift expects an integer (uint8 or uint16) "cost"
    image. The watershed algorithm flood-fills from each marker
    by ascending cost — pixels with LOW cost get flooded first,
    HIGH cost forms a barrier.

    For tree segmentation we want:
      - Tree-tops (high CHM): LOW cost (easy to flood from the marker)
      - Gaps (low CHM): HIGH cost (act as a flood barrier)

    So cost = inverted, scaled CHM. Pixels below `min_height_m` are
    forced to maximum cost (255) — they participate in the watershed
    propagation only as barriers, never as part of any basin (they
    will also be post-masked out of the labels).

    Args:
        chm: 2D float array of canopy heights in meters.
        min_height_m: Pixels below this become hard barriers.

    Returns:
        2D uint8 array of the same shape as `chm`.
    """
    safe = np.where(np.isfinite(chm), chm, 0.0)
    max_h = float(safe.max())
    if max_h <= 0.0:
        # Entirely empty CHM — return all-barriers; the watershed
        # won't propagate and the post-mask will zero everything.
        return np.full(chm.shape, 255, dtype=np.uint8)

    # Inverted, scaled to 0-255. Tree-tops → 0, mid-canopy → mid-grey,
    # near-zero canopy → 255. Round before cast to avoid float32
    # truncation surprises (e.g. 84.9999... → 84 instead of 85).
    inverted = (1.0 - safe / max_h) * 255.0
    cost = np.round(inverted).astype(np.uint8)

    # Anything below the height threshold is forced to maximum cost
    # so it can never be flooded into a basin.
    cost[safe < min_height_m] = 255
    return cost


def _polygons_from_labels(
    labels: np.ndarray,
    transform: "rasterio.Affine",
) -> dict[int, Polygon]:
    """Extract shapely Polygons from a labeled watershed raster.

    Uses rasterio.features.shapes which natively handles affine
    transforms and returns geometries in the raster's CRS. When a
    label appears in multiple disconnected regions (rare but
    possible if a basin is split by a thin gap), keeps the LARGEST
    by area.

    Returns:
        Dict mapping label_id → shapely Polygon. Background (label 0)
        is excluded. Invalid polygons are repaired with `make_valid`.
    """
    polygons: dict[int, Polygon] = {}
    label_mask = labels > 0
    for geom, val in rio_shapes(
        labels.astype(np.int32),
        transform=transform,
        mask=label_mask,
    ):
        label_id = int(val)
        if label_id == 0:
            continue
        try:
            poly = shapely_shape(geom)
        except Exception:
            continue
        if not poly.is_valid:
            poly = make_valid(poly)
            # make_valid can return a MultiPolygon or GeometryCollection.
            # We need a single Polygon.
            poly = _largest_polygon(poly)
            if poly is None:
                continue
        if not isinstance(poly, Polygon):
            poly = _largest_polygon(poly)
            if poly is None:
                continue
        existing = polygons.get(label_id)
        if existing is None or poly.area > existing.area:
            polygons[label_id] = poly
    return polygons


def _largest_polygon(geom) -> Polygon | None:
    """Return the largest Polygon component of a possibly multi-part geometry.

    Used to handle MultiPolygon results from `make_valid`. Returns
    None if the input has no Polygon components.
    """
    if isinstance(geom, Polygon):
        return geom
    if hasattr(geom, "geoms"):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if not polygons:
            return None
        return max(polygons, key=lambda p: p.area)
    return None
