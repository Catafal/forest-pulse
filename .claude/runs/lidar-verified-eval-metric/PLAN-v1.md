# Implementation Plan: lidar-verified-eval-metric

**Version:** 1
**Date:** 2026-04-07
**Based on:** SPEC.md + OBSERVE.md

## Summary

Add `autoresearch/eval_lidar.py` (~280 lines) — a new evaluation
module separate from the locked `eval.py`. It derives ground-truth
tree-top positions from LiDAR Canopy Height Models via local-maximum
filtering, then matches model detections against those positions using
greedy nearest-neighbor with a 2 m tolerance, and reports
precision/recall/F1. A CLI script (`scripts/run_lidar_eval.py`, ~120
lines) runs it on a default 10-patch reference set and saves a CSV
report. Unit tests (~150 lines) cover the pure functions with synthetic
CHMs and detections — no real LAZ files needed in CI.

This produces **the project's first physically-grounded evaluation
metric** without modifying any existing locked files and without adding
any new heavy dependencies.

## Files to Create

| File | Purpose | Lines |
|---|---|---|
| `autoresearch/eval_lidar.py` | LiDAR-verified eval module — find_tree_tops, match, evaluate | ~280 |
| `scripts/run_lidar_eval.py` | CLI runner — default 10 patches, prints table, saves CSV | ~120 |
| `tests/test_eval_lidar.py` | Unit tests with synthetic CHMs and detections | ~180 |

## Files to Modify

**None.** This is a strictly additive feature. `eval.py` stays locked,
`lidar.py` is reused as-is, `georef.py` and the rest of the pipeline
are untouched.

## Module Contracts

### `autoresearch/eval_lidar.py`

**Dataclasses:**
```python
@dataclass
class EvalResult:
    """Aggregated detection-vs-LiDAR evaluation result."""
    n_predictions: int
    n_truth: int
    n_true_positive: int
    n_false_positive: int
    n_false_negative: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def from_counts(cls, n_pred: int, n_truth: int, n_tp: int) -> "EvalResult":
        """Compute precision/recall/F1 from raw counts."""
```

**Public functions:**
```python
def find_tree_tops_from_chm(
    chm: np.ndarray,
    transform,                       # rasterio.Affine — pixel→world mapping
    min_height_m: float = 5.0,
    min_distance_m: float = 3.0,
    smooth_sigma_px: float = 1.0,
) -> list[tuple[float, float]]:
    """Find tree-top world coordinates via local-max filtering on a CHM.

    Standard forestry technique. Returns (x, y) coordinates in the same
    CRS as the raster transform.
    """

def match_predictions_to_truth(
    pred_xy: list[tuple[float, float]],
    truth_xy: list[tuple[float, float]],
    tolerance_m: float = 2.0,
) -> EvalResult:
    """Greedy nearest-neighbor matching → counts → EvalResult."""

def evaluate_patch_against_lidar(
    detections: sv.Detections,
    image_bounds: tuple[float, float, float, float],
    image_size_px: tuple[int, int],
    laz_path: Path,
    height_threshold: float = 5.0,
    match_tolerance_m: float = 2.0,
    chm_resolution_m: float = 0.5,
) -> EvalResult:
    """Run the full eval for ONE patch.

    Steps:
      1. Convert each detection bbox center → world (x, y)
      2. Compute CHM from LAZ for the image bounds
      3. Run find_tree_tops_from_chm → ground truth
      4. match_predictions_to_truth → EvalResult
    """

def evaluate_patches_against_lidar(
    patches: list[dict],   # each: {'name', 'image', 'detections', 'bounds', 'size', 'laz_path'}
) -> tuple[EvalResult, list[dict]]:
    """Run eval on many patches and return (aggregate, per-patch results).

    Aggregation is MICRO-average — pool TP/FP/FN counts across patches
    then compute the ratios. This weights each tree equally rather than
    each patch equally.
    """
```

**Internal helpers:**
```python
def _detection_centers_world(
    detections: sv.Detections,
    image_bounds, image_size_px,
) -> list[tuple[float, float]]:
    """Pixel bbox centers → world coordinates."""

def _read_chm_array(chm_path: Path) -> tuple[np.ndarray, Any]:
    """Load CHM raster + transform from disk."""
```

### `scripts/run_lidar_eval.py`

```bash
# Default 10-patch eval
python scripts/run_lidar_eval.py

# Custom patches
python scripts/run_lidar_eval.py --patch 0250.jpg --patch 0477.jpg

# Custom checkpoint
python scripts/run_lidar_eval.py --checkpoint checkpoints/round_1.pt
```

Output:
- Per-patch table to stdout
- Aggregate row at the bottom
- CSV saved to `outputs/lidar_eval/eval_summary.csv`
- Honest first F1 number for the model

## Tests

| Done # | Test | Type |
|---|---|---|
| 1 | `test_find_tree_tops_three_known_peaks` — synth CHM with 3 known peaks → 3 returned | unit |
| 1 | `test_find_tree_tops_below_threshold_dropped` — peaks at 3m are filtered out | unit |
| 1 | `test_find_tree_tops_min_distance_dedupes_close_peaks` — peaks 1m apart → 1 kept | unit |
| 2 | `test_match_perfect_overlap` — preds == truth → P=R=F1=1.0 | unit |
| 3 | `test_match_no_overlap_far_apart` — preds far from truth → P=R=F1=0.0 | unit |
| 4 | `test_match_partial_5_truth_3_pred_3_match` → P=1.0, R=0.6, F1=0.75 | unit |
| — | `test_match_extra_predictions_become_fp` — 3 truth, 5 pred, 3 match → P=0.6, R=1.0 | unit |
| — | `test_match_empty_pred_and_truth` — both empty → 0/0 not NaN, returns zeros | unit |
| — | `test_match_empty_pred_with_truth` — 0 pred, 5 truth → R=0 | unit |
| — | `test_match_pred_with_empty_truth` — 5 pred, 0 truth → P=0 | unit |
| — | `test_match_tolerance_boundary` — exactly at tolerance → matched | unit |
| 5 | `test_evaluate_patch_with_mocked_chm` — mocked find_tree_tops + matching | unit |
| — | `test_eval_result_from_counts_handles_zero_division` | unit |
| — | `test_aggregate_micro_average_correctness` — pool counts then compute ratio | unit |
| 7 | All 68 existing tests still pass | regression |

## Implementation Order

1. Install scipy if missing → already installed, skip
2. Create `autoresearch/eval_lidar.py` skeleton with imports + EvalResult dataclass
3. Implement `EvalResult.from_counts` + tests
4. Implement `find_tree_tops_from_chm` + tests
5. Implement `match_predictions_to_truth` + tests
6. Implement `_detection_centers_world` helper
7. Implement `evaluate_patch_against_lidar` (with mocked CHM in tests)
8. Implement `evaluate_patches_against_lidar` (aggregate + per-patch)
9. Write `scripts/run_lidar_eval.py`
10. Lint + run all tests
11. Run the CLI on real patch 0250 (LAZ already cached from Phase 7) — get the first honest number
12. Run the CLI on the default 10-patch set — get the aggregate baseline
13. Commit + merge

## Approach Alternatives

| Alternative | Rejected because |
|---|---|
| Modify `eval.py` to add LiDAR mode | Locked file. Adding to a separate module is cleaner. |
| Hungarian / optimal assignment matching (scipy.optimize.linear_sum_assignment) | Greedy is < 1% different in this regime; simpler and deterministic. |
| Compute proper precision-recall curve / mAP integral | Adds complexity; we want a single trustworthy number first. mAP can come later if needed. |
| Use `skimage.feature.peak_local_max` instead of `scipy.ndimage.maximum_filter` | Adds skimage dependency. scipy.ndimage is already transitive. |
| IoU-based matching against bounding boxes | Tree tops are POINTS in LiDAR — point-distance matching is the natural choice. |
| Per-class evaluation | Single-class problem (only "tree"). Multi-class will come with species (Phase 12). |
| Use CHM directly without smoothing | Speckle in CHM produces spurious peaks. 1px Gaussian removes most without merging real crowns. |
| Per-patch macro average instead of micro | Patches with 1 vs 100 trees would weigh equally, biasing toward sparse patches. Micro is the right call. |
| Allow multiple confidence thresholds | Out of scope. We report at the model's default threshold (0.3). |

## Risks and Mitigations

- **Tree-top detection may miss merged crowns in dense canopy.** This
  inflates the false-negative count. Mitigation: this is documented in
  the spec and known to be a fundamental limitation of CHM-based truth.
  The metric is still more honest than self-training.

- **Tree tops in dense canopy may produce more "trees" than RF-DETR
  finds**, making recall look terrible. Acceptable — this is the
  honest signal we want. Self-training was hiding it.

- **scipy.ndimage.maximum_filter** uses reflect mode by default —
  pixels at the raster edge produce slightly biased local-max results.
  Acceptable for 320×320 patches with central focus.

- **Empty CHM** (no LAZ points in bounds) → returns empty truth list
  → recall is undefined. We return 0/0 → 0.0 explicitly via
  `EvalResult.from_counts` handling.

- **LAZ download time on first run.** The CLI script downloads any
  missing LAZ tiles via the existing `fetch_laz_for_patch` cache. The
  10 default patches share at most ~6 LAZ tiles, so a fresh run takes
  ~2-4 GB download.

## Estimated Scope

- 1 new module: ~280 lines
- 1 new CLI script: ~120 lines
- 1 new test file: ~180 lines
- 0 modified files
- 0 new dependencies (scipy already installed)
- Complexity: medium — clear contracts, focused scope, well-known forestry algorithm
- Time: ~2 hours at SOP quality bar
