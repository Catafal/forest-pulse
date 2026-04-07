# Codebase Observation: sam2-hybrid-segmentation

**Date:** 2026-04-07
**GitNexus available:** no

## External Research

Full report: `.claude/research/sam2_integration.md`. Key findings:

### SAM2 library choice
- **Use HuggingFace transformers** (already in deps). `Sam2Model` + `Sam2Processor` were merged into transformers main 2025-08-14, requires >= 4.45.
- Model ID: **`facebook/sam2.1-hiera-small`** (46 M params, ~184 MB disk, ~1.5 GB inference RAM, ~0.45s per box refinement on M4 Pro)
- Fallback if memory pressure: `facebook/sam2.1-hiera-tiny` (38.9M / 155 MB)
- `base-plus` / `large` are overkill — tree crowns are easy regime given an RF-DETR bbox prior

### MPS rules (critical to get right)
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` **before** `import torch`
- Run in **fp32** — do NOT use `torch.autocast("cuda", bfloat16)` on MPS (produces NaN masks)
- `bicubic_upsample` and `grid_sampler_2d` fall back to CPU silently (~50-100ms overhead — acceptable)
- Lazy-load model at module level (4s cold load is too slow per-call)

### API patterns
- **Box refinement:** `Sam2Processor(images=..., input_boxes=[[[x1,y1,x2,y2], ...]], return_tensors="pt").to(device)` → `model(**inputs, multimask_output=False)` → `processor.post_process_masks(outputs.pred_masks, ...)`. The three-level nesting on `input_boxes` is a silent-failure trap.
- **Automatic mode:** `pipeline("mask-generation", model="facebook/sam2.1-hiera-small")(image, points_per_batch=...)` returns list of `{"mask": np.array, "score": float, "bbox": [x,y,w,h]}` dicts.
- **sv.Detections from SAM:** for auto mode use `sv.Detections.from_sam(sam_result=result)`. For refinement output, construct manually with `sv.Detections(xyxy=..., mask=..., confidence=...)`.

### Tree-crown filtering thresholds (for ICGC 25 cm/px GSD)
- `min_area` = 400 px² (small tree crown ~5 m² → 80 px²; 400 is conservative)
- `max_area` = 40 000 px² (~ 156 m², a huge single tree, prevents merged-forest segments)
- `max_area_frac` = 0.15 (reject any segment > 15% of the full 640×640 patch — catches "whole image" segments)
- `min_circularity` = 0.45 (tree crowns are roughly circular; 4πA/P² > 0.45 rejects elongated shapes like roads/fences)
- `max_aspect_ratio` = 2.5 (rejects elongated shapes)
- `min_ndvi` = 0.30 (rejects non-vegetation; uses existing ndvi.py module if available)

### Dedup between RF-DETR and SAM2 auto
- Primary: mask-IoU ≥ 0.30 → consider "same tree"
- Fallback (if masks empty): centroid distance < 20 px (= 5 m at 25 cm GSD)

## Relevant Files in Codebase

### Files to create
- `src/forest_pulse/segment.py` — SAM2 wrapper module (lazy load, box refinement, auto mode, filter, dedupe, hybrid pipeline)
- `scripts/sam2_smoke_test.py` — smoke test before wiring into pipeline
- `tests/test_segment.py` — tests with synthetic data + mocked SAM2
- `.claude/research/sam2_integration.md` — already saved by research agent

### Files to modify
- `src/forest_pulse/health.py` — read above. Currently takes `image + detections`. Has `_crop_detection()` helper that does bbox crop. Add `use_masks: bool = False` parameter. When True + detections.mask is not None, crop by mask (bbox then zero non-mask pixels) and compute GRVI/ExG only on mask pixels. Default False preserves backward compatibility.
- `src/forest_pulse/georef.py` — when detections have masks, compute `crown_area_m2` from `mask.sum() × (pixel_width_m × pixel_height_m)` instead of `bbox_width × bbox_height`. Schema stays the same, values just get more accurate.
- `src/forest_pulse/device.py` — add `os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"` at import time (before torch imports). Currently does CUDA/MPS/CPU detection.
- `scripts/full_pipeline_demo.py` — add `--use-sam2` CLI flag. When set, replace `detect_trees(...)` with `detect_trees_hybrid(...)` from segment.py. Everything downstream works unchanged.
- `pyproject.toml` — add optional `[sam2]` extra with `transformers>=4.45.0` (may already be there), ensure Pillow is recent enough.
- `tests/test_filters.py` — no changes; existing tests don't touch segment.py.

## Key Patterns in Codebase (maintain these)

- **Lazy imports for heavy deps:** detect.py imports rfdetr inside functions. Apply same pattern: import transformers SAM2 classes inside segment.py functions.
- **Model cache dict:** detect.py uses `_MODEL_CACHE` module-global dict keyed by model name. Reuse this pattern for SAM2.
- **Pure functions with explicit inputs:** every filter/module takes inputs, returns outputs, no state. Segment module should follow the same style.
- **Graceful fallback:** if a dep isn't installed, raise a clear message (`raise ImportError("SAM2 requires ... — pip install -e '.[sam2]'")`).
- **Strip rfdetr metadata:** `source_shape`, `source_image` fields in detections.data break boolean indexing. Strip before use (same helper as ndvi.py / lidar.py / georef.py).

## Architecture Constraints

- Max 200 lines per function, 1000 per file
- All imports at top (except heavy lazy ones documented inline)
- Type hints + Google docstrings on public functions
- Comments explain WHY not WHAT
- MPS must work (set env var, use fp32, avoid autocast)

## Conditional Skill Routing

- [ ] /plan-eng-review — not applicable (scope is ~300 lines, well-understood)
- [ ] /cso — not applicable
- [ ] /backend-docs-writer — not applicable
- [ ] /frontend-docs-writer — not applicable
- [ ] /testing-patterns — not applicable
