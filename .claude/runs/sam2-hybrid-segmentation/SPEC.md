# Feature Spec: SAM2 Hybrid Segmentation

**Slug:** sam2-hybrid-segmentation
**Date:** 2026-04-07
**Status:** draft

## What

Add Meta's Segment Anything Model 2 (SAM2) to Forest Pulse as a hybrid
companion to the existing RF-DETR detector. Three new capabilities,
exposed as a single new module `src/forest_pulse/segment.py`:

**1. Bbox refinement** (`refine_detections_with_sam2`): Given RF-DETR
detections, use each bbox as a SAM2 prompt to produce a precise crown mask
for each tree. Returns the same `sv.Detections` with a new `.mask` field
attached. This makes health scoring and crown area measurements more
accurate because we crop pixels based on actual crown shape, not the
rectangular bbox which includes background gaps.

**2. Automatic segmentation** (`segment_all_trees_sam2`): Run SAM2 in
automatic mask-generation mode to segment every distinct object in the
image, then filter the output by size, shape, and NDVI to keep only
tree-like segments. Finds trees RF-DETR missed in dense uniform canopy.

**3. Hybrid pipeline** (`detect_trees_hybrid`): Combine (1) and (2) into
one call. Run RF-DETR → refine with SAM2 → run SAM2 automatic → filter →
dedupe (remove SAM2 segments that overlap RF-DETR detections) → merge.
Produces a tree inventory with higher recall than RF-DETR alone and
higher precision than SAM2 alone.

**4. Downstream integration:**
- `health.py` gains an optional `use_masks` parameter. When detections
  have masks AND use_masks=True, GRVI/ExG are computed on mask pixels
  only, not on the whole bbox. Default stays backward-compatible.
- `georef.py` gains optional mask-derived `crown_area_m2` — when masks
  are present, area is the count of mask pixels × pixel_area_m², not
  bbox width × height. More accurate for biomass estimates.
- `scripts/full_pipeline_demo.py` gains a `--use-sam2` flag that switches
  to `detect_trees_hybrid` instead of `detect_trees`.

## Why

The current RF-DETR model (mAP 0.904 on self-trained val, real recall
~20-30% against actual tree density) systematically undercounts trees in
dense uniform canopy (patch 0250: 6 detections in a beech forest that
should have 500-1000 trees). Visual review confirmed this is a crown
segmentation problem, not a detection problem — in dense canopy, crowns
touch and merge visually, so any bbox detector will fail.

SAM2 solves this because it's an instance segmentation foundation model.
It doesn't need the objects to be distinct in advance — it finds the
boundaries between touching crowns via learned foundation-model priors.
And because SAM2 outputs pixel masks (a strict superset of bboxes), every
existing downstream module keeps working — masks are convertible to
bboxes, centroids, and polygons.

The hybrid approach preserves our investment in the 0.904 mAP RF-DETR
model (high precision on distinct trees) while using SAM2 only where
needed (to find merged crowns and refine existing detections).

## Constraints

- Must work on Apple MPS (both M4 Pro 48GB and Mac Mini 24GB).
  SAM2-base model weights are ~160MB; SAM2-large is ~900MB. We pick the
  variant that fits on the Mac Mini RAM.
- Must keep the existing RF-DETR path fully functional. SAM2 is additive,
  not a replacement. Scripts that don't opt in must behave identically.
- Must not break existing tests. Backward-compatible changes to health.py
  and georef.py (new parameters default to current behavior).
- Model weights must be cached after first download (~1GB one-time).
- Must handle SAM2 unavailable gracefully (import error → clear message,
  fallback to RF-DETR-only).
- Must NOT auto-install SAM2 on every user. Keep it as an optional
  extra (`pip install -e ".[sam2]"`).
- Inference time per patch ≤ 10 seconds on MPS (SAM2 automatic mode is
  slower than RF-DETR; we tolerate the latency for the recall boost).
- Coordinate systems and contracts stay EPSG:25831 internal, WGS84 for
  GeoJSON export.

## Done Criteria

1. GIVEN a patch + RF-DETR detections WHEN `refine_detections_with_sam2()`
   runs THEN returned `sv.Detections` has `mask` attribute with the same
   number of masks as detections, each a boolean 2D array matching the
   image shape.

2. GIVEN a patch WHEN `segment_all_trees_sam2(image)` runs THEN it
   returns an `sv.Detections` with N > 0 segments on a dense-canopy
   patch like 0250 (the one where RF-DETR finds only 6 trees).

3. GIVEN a patch WHEN `detect_trees_hybrid(image)` runs THEN the returned
   detection count is >= `detect_trees(image)` for the same patch
   (hybrid finds at least as many trees as RF-DETR alone).

4. GIVEN detections with masks WHEN `score_health(use_masks=True)` runs
   THEN GRVI is computed only on pixels where mask is True, and the
   returned HealthScore values differ from the bbox-crop version on a
   patch with visible gaps between crowns.

5. GIVEN detections with masks WHEN `georeference()` runs THEN the
   `crown_area_m2` column is smaller than `crown_width_m * crown_height_m`
   (because mask area ≤ bbox area).

6. GIVEN `scripts/full_pipeline_demo.py --use-sam2 --patch 0250.jpg`
   WHEN run THEN it prints a detection count > 6 (the RF-DETR baseline
   for that patch) and produces a valid GeoJSON.

7. GIVEN existing tests WHEN the test suite runs THEN all 35 tests still
   pass (backward compatibility).

## Out of Scope

- Fine-tuning SAM2 on our data — use the pre-trained foundation model.
- Exporting tree-crown polygons in GeoJSON — we stick with Point geometry
  for consistency with existing schema. Mask-derived crown_area_m2 is
  exposed but the polygon itself is not written to GeoJSON yet.
- Updating the self-training loop to use SAM2 — keep that as a future
  feature. The hybrid detector is used only at inference time.
- Replacing RF-DETR entirely — the hybrid approach preserves RF-DETR.
- Retrained/reweighted model combining — we use simple IoU-based dedupe,
  not learned merging.
- Video inference (SAM2 can do video; we don't have video data).
- LiDAR height filter integration — that's a separate feature tracked in
  `.claude/runs/lidar-ndvi-filtering/`.
- Performance optimization beyond ensuring < 10s per patch on MPS.
