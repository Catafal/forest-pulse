# Implementation Plan: sam2-hybrid-segmentation

**Version:** 1
**Date:** 2026-04-07
**Based on:** SPEC.md + OBSERVE.md + .claude/research/sam2_integration.md

## Summary

Create `src/forest_pulse/segment.py` (~320 lines) as a self-contained
SAM2 wrapper with three public functions: `refine_detections_with_sam2`,
`segment_all_trees_sam2`, `detect_trees_hybrid`. Make health.py and
georef.py backward-compatibly aware of masks. Add `--use-sam2` to the
full pipeline demo. Use HuggingFace transformers (no new dependency),
`facebook/sam2.1-hiera-small` model, MPS-safe fp32 path, lazy loading.

## Files to Create

| File | Purpose | Lines |
|---|---|---|
| `src/forest_pulse/segment.py` | SAM2 wrapper — refinement, automatic, hybrid, filter, dedupe | ~320 |
| `scripts/sam2_smoke_test.py` | Smoke test — load model, box prompt, auto mode, print timings | ~60 |
| `tests/test_segment.py` | Tests — mocked SAM2 for filter + dedupe logic, real SAM2 behind `@pytest.mark.slow` | ~120 |

## Files to Modify

| File | Change | Rationale |
|---|---|---|
| `src/forest_pulse/health.py` | Add `use_masks: bool = False` parameter to `score_health`. When True + `detections.mask` present, compute indices only on mask pixels via `compute_grvi_masked` / `compute_exg_masked` helpers. Default False keeps all existing tests green. | Better health scoring when we have crown shapes instead of bounding rectangles. |
| `src/forest_pulse/georef.py` | When `detections.mask` is present, derive `crown_area_m2` from `mask.sum() × (pixel_size_m² )`. Schema unchanged, values more accurate. | Accurate biomass estimates. |
| `src/forest_pulse/device.py` | Add `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` at module top, before torch imports. | SAM2 needs this for unsupported ops to fall back to CPU silently. Setting globally is safe (only affects MPS paths). |
| `scripts/full_pipeline_demo.py` | Add `--use-sam2` flag. When set, replace `detect_trees(...)` with `detect_trees_hybrid(...)` from segment.py. Rest unchanged. | Users opt in to the heavier pipeline. |
| `pyproject.toml` | Add `[sam2]` optional extra. Pin `transformers>=4.45` (need Sam2Model). | Keeps SAM2 opt-in via `pip install -e ".[sam2]"`. |
| `tests/conftest.py` (create if absent) | Mark `@pytest.mark.slow` for real-model SAM2 tests; register the mark | Keep CI fast. |

## segment.py Public API

```python
def refine_detections_with_sam2(
    image: np.ndarray,
    detections: sv.Detections,
    model_id: str = "facebook/sam2.1-hiera-small",
    device: str | None = None,
) -> sv.Detections:
    """Refine each detection's bbox into a precise crown mask via SAM2.

    Returns sv.Detections with the same xyxy/confidence as input plus a
    .mask field (N, H, W) bool array. Empty input passes through.
    """

def segment_all_trees_sam2(
    image: np.ndarray,
    model_id: str = "facebook/sam2.1-hiera-small",
    device: str | None = None,
    points_per_side: int = 32,
    min_area_px: int = 400,
    max_area_px: int = 40_000,
    max_area_frac: float = 0.15,
    min_circularity: float = 0.45,
    max_aspect_ratio: float = 2.5,
) -> sv.Detections:
    """Automatic mask generation + tree-crown filtering.

    Runs SAM2 in 'segment everything' mode, filters the output to keep
    only tree-like segments using size/shape heuristics. NDVI filtering
    is applied separately by caller if desired.
    """

def detect_trees_hybrid(
    image: np.ndarray,
    rfdetr_checkpoint: str,
    rfdetr_confidence: float = 0.3,
    sam2_model_id: str = "facebook/sam2.1-hiera-small",
    iou_dedup_threshold: float = 0.30,
    centroid_dedup_px: float = 20.0,
) -> sv.Detections:
    """RF-DETR + SAM2 hybrid detection.

    1. Run RF-DETR → high-precision detections
    2. Refine each with SAM2 → precise crown masks
    3. Run SAM2 automatic on full image → all crown candidates
    4. Filter automatic output by size/shape
    5. Dedupe against RF-DETR detections (IoU or centroid)
    6. Merge — union of refined RF-DETR + new SAM2-only detections
    """
```

### Internal helpers

- `_load_sam2_model(model_id, device)` → cached `(model, processor)` tuple
- `_load_sam2_auto_generator(model_id, device, **kwargs)` → cached automatic mask generator
- `_filter_crown_segments(masks, img_shape, thresholds)` → filtered list
- `_mask_iou(mask_a, mask_b) -> float`
- `_centroid(mask) -> tuple[float, float]`
- `_dedup_masks(primary, candidates, iou_thresh, centroid_px)` → new-only list
- `_stack_sv_detections(left, right)` → concatenated sv.Detections
- `_strip_rfdetr_metadata(detections)` → reused pattern

## Tests (test_segment.py)

| Done # | Test | Mock? | Type |
|---|---|---|---|
| 1 | `test_refine_attaches_masks_to_detections` — mock Sam2Model, verify mask shape | Yes | unit |
| 4 | `test_crop_mask_excludes_background_pixels` — verify `_masked_indices_only` works | No mock | unit |
| 5 | `test_mask_area_smaller_than_bbox_area` — explicit mask → crown_area < bbox area | No | unit |
| — | `test_filter_rejects_tiny_segments` — size filter | No | unit |
| — | `test_filter_rejects_elongated_segments` — circularity | No | unit |
| — | `test_mask_iou_matches_known_overlap` — 50% overlap → IoU 0.33… | No | unit |
| — | `test_dedup_removes_overlapping_segments` | No | unit |
| — | `test_dedup_keeps_disjoint_segments` | No | unit |
| 3 | `test_hybrid_count_ge_rfdetr` — slow, real model | Real model | integration, `@pytest.mark.slow` |
| 7 | (existing 35 tests) still green after health.py / georef.py changes | — | existing |

## health.py Change (minimal diff)

```python
def score_health(
    image: np.ndarray,
    detections: sv.Detections,
    use_masks: bool = False,
) -> list[HealthScore]:
    ...
    for i, xyxy in enumerate(detections.xyxy):
        if use_masks and detections.mask is not None:
            grvi, exg = _compute_indices_from_mask(
                image, detections.mask[i]
            )
        else:
            crop = _crop_detection(image, xyxy)
            if crop.shape[0] < MIN_CROP_SIZE or ...:
                ... # existing small-crop path
            grvi = compute_grvi(crop)
            exg = compute_exg(crop)
        ...
```

Add a small `_compute_indices_from_mask(image, mask)` helper that:
- Slices `image[mask]` into a flat (N, 3) array
- Computes GRVI and ExG on those pixels only
- Returns 0.0 fallback for empty/tiny masks

## georef.py Change (minimal diff)

```python
def georeference(..., crs="EPSG:25831") -> gpd.GeoDataFrame:
    ...
    for i, xyxy in enumerate(detections.xyxy):
        geo_box = _pixel_bbox_to_geo(...)
        width_m, height_m = ...

        if detections.mask is not None:
            mask_area_px = int(detections.mask[i].sum())
            # Convert mask pixels to meters² using the image-to-world
            # scale derived from bounds / image_size
            px_area_m2 = (width_m / (xyxy[2] - xyxy[0])) * \
                         (height_m / (xyxy[3] - xyxy[1]))
            crown_area_m2 = mask_area_px * px_area_m2
        else:
            crown_area_m2 = width_m * height_m
```

This keeps the schema identical and existing tests pass.

## device.py Change (one line)

```python
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# THEN the torch import happens when get_device() is called
```

Actually we need this BEFORE torch is imported anywhere in the process.
Safest path: set it at the top of `src/forest_pulse/__init__.py` too so
it's set before any user-side torch import. Document why.

## scripts/full_pipeline_demo.py Change

```python
parser.add_argument("--use-sam2", action="store_true",
    help="Use SAM2 hybrid detector (slower, higher recall).")
...
if args.use_sam2:
    from forest_pulse.segment import detect_trees_hybrid
    detections = detect_trees_hybrid(image, str(CHECKPOINT), ...)
else:
    detections = detect_trees(image, model_name=str(CHECKPOINT), ...)
```

Also pass `use_masks=True` to score_health when SAM2 is on.

## Approach Alternatives

| Alternative | Rejected Because |
|---|---|
| Use Meta's `sam2` pip package | Extra dep. HF transformers path is simpler and already installed. |
| Use Ultralytics SAM wrapper | Adds Ultralytics as dep, another layer of abstraction to debug. |
| Replace RF-DETR entirely with SAM2 auto | Loses the 0.904 precision investment. Auto mode over-segments (rocks/shadows). Hybrid is the right tradeoff. |
| Fine-tune SAM2 on tree crowns | Out of scope. Foundation model works zero-shot here because RF-DETR box is a strong prior. |
| Write masks to GeoJSON as polygons | Schema change, downstream tooling impact. Can come in a future feature. |
| Put SAM2 loading in __init__.py | Slow cold import. Lazy inside functions + module-level cache is the standard pattern. |
| Share `_strip_rfdetr_metadata` helper in a utils module | Minor duplication across 4 files is cheaper than a utils module dependency. |

## Risks and Side Effects

- **MPS instability:** SAM2's bicubic upsample falls back to CPU. First-run can be slow (~8s model load + fallback). Mitigation: cache model at module level, document timing in demo output.
- **RAM spike during automatic mode:** `points_per_side=32` means 1024 queries per image. On Mac Mini 24 GB this may swap. Mitigation: expose `points_per_side` parameter, default to 32, let users lower.
- **Filter over-rejects:** Circularity threshold 0.45 might reject legitimate elongated crowns. Mitigation: expose threshold params, provide smoke test to tune.
- **Dedup too aggressive or too loose:** IoU 0.30 is a guess. Mitigation: parameterize, measure on smoke test.
- **Transformers version mismatch:** If installed transformers < 4.45, Sam2Model doesn't exist → ImportError. Mitigation: version-check + clear error message in `_load_sam2_model`.
- **Existing tests break:** health.py and georef.py changes could break tests. Mitigation: add `use_masks=False` default, run full test suite before commit, add new tests for the new code paths only.

## Estimated Scope

- 1 new module (~320 lines)
- 1 new smoke test script (~60 lines)
- 1 new test file (~120 lines)
- 4 files modified with small diffs (~60 lines total changes)
- 0 new dependencies (transformers already installed, add version pin)
- Complexity: medium-high — many pieces but each is small and well-understood

## Verification (after implementation)

1. `.venv/bin/pytest tests/ -q` → all existing 35 tests still green, + new ones pass
2. `.venv/bin/ruff check src/ scripts/ tests/` → clean
3. `python scripts/sam2_smoke_test.py` → loads model, refines a box, runs auto mode, prints counts + timings
4. `python scripts/full_pipeline_demo.py --use-sam2 --patch 0250.jpg` → count > 6 (the RF-DETR baseline for that dense beech patch) AND produces valid GeoJSON
5. `python scripts/full_pipeline_demo.py --use-sam2 --patch 0477.jpg` → compare count vs 157 RF-DETR baseline (expect similar or higher)
