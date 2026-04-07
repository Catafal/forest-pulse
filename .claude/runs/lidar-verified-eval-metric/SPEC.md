# Feature Spec: LiDAR-Verified Evaluation Metric

**Slug:** lidar-verified-eval-metric
**Date:** 2026-04-07
**Status:** draft

## What

Produce the **first physically-grounded evaluation metric** for the
Forest Pulse tree detector. Use ICGC LiDAR (now wired up via Phase 7)
to derive ground-truth tree-top positions from the Canopy Height Model
via local-maximum filtering, then compute precision / recall / F1 of
the model's detections against those LiDAR-verified positions.

This replaces (in spirit, not in code — the old metric stays) the
biased self-trained `mAP50 = 0.904` with an honest number.

Three new pieces:

**1. `autoresearch/eval_lidar.py`** — new evaluation module separate
from the locked `eval.py`. Public API:
  - `find_tree_tops_from_chm(chm, transform, ...)` → list of (x, y)
    world-coordinate tree positions via local-maximum filtering on the
    smoothed CHM
  - `match_predictions_to_truth(pred_xy, truth_xy, tolerance_m)` →
    greedy nearest-neighbor matching with a distance tolerance
  - `evaluate_patch_against_lidar(detections, image_bounds, image_size,
    laz_path, ...)` → `EvalResult` for one patch
  - `evaluate_patches_against_lidar([...])` → aggregated `EvalResult`
    over many patches with per-patch breakdown

**2. `EvalResult` dataclass** — `n_predictions`, `n_truth`, `n_tp`,
`n_fp`, `n_fn`, `precision`, `recall`, `f1`. Returned from every
evaluation function. Composes cleanly across patches.

**3. `scripts/run_lidar_eval.py`** — CLI runner that takes a list of
patches (default = 10 representative patches, one per zone), runs the
detector, runs the LiDAR evaluation, and prints a per-patch + aggregate
table. Saves results to `outputs/lidar_eval/eval_summary.csv`.

## Why

Every metric in this project so far has been biased:

- **`autoresearch/eval.py`** computes mAP50 against the model's OWN
  self-trained validation labels. The model can't disagree with itself.
  It gets 0.904 — meaningless as a quality measure.
- **A/B test counts** (840 → 873 with SAM2 hybrid) measure detection
  density, not detection accuracy.
- **Visual review** is qualitative and limited to a handful of patches.

Without an honest metric:
- Auto-research (Phase 11) optimizes a biased target → makes the model
  "better" at agreeing with itself, not at finding real trees.
- The classifier (Phase 9) can be trained against LiDAR auto-labels,
  but we need an independent eval to know if the trained classifier
  actually generalizes.
- Every roadmap step downstream of Phase 8 needs this number as its
  target.

LiDAR gives us this for free. Tree-top detection from a Canopy Height
Model via local-maximum filtering is the standard forestry technique
(used by Spanish Forest Inventory, lidR R package, USFS). It's not
perfect — dense uniform canopy still produces ambiguity — but its
errors are PHYSICAL (caused by canopy structure) rather than CIRCULAR
(caused by the model labeling its own training data).

## Constraints

- **Do not modify `autoresearch/eval.py`** — it's LOCKED. New evaluator
  lives in a NEW file `autoresearch/eval_lidar.py`. This preserves the
  old metric for backward compat (anything depending on the existing
  contract still works).
- Must reuse the existing `forest_pulse.lidar` module for LAZ download
  and CHM computation. Zero duplication of LiDAR plumbing.
- Must work on Apple MPS / CPU (pure numpy/scipy for the eval logic;
  laspy + RF-DETR are the only heavy dependencies, both already wired).
- Must handle empty inputs gracefully (no detections, no truth, both).
- **All metric definitions must be transparent**: precision = TP/(TP+FP),
  recall = TP/(TP+FN), F1 = 2PR/(P+R). Document the matching algorithm
  clearly.
- **Greedy nearest-neighbor matching, not Hungarian** — simpler, deterministic,
  and the literature shows the difference is < 1% in this regime.
- Tolerance: 2 m default (Spanish Forest Inventory convention; matches
  the temporal.py default).
- Tree-top detection: 5 m height threshold (consistent with `filter_by_height`
  default), 3 m local-max window (typical Mediterranean crown radius).
- Smoothing: Gaussian sigma 1.0 px on the 0.5 m CHM (= 0.5 m sigma in real
  units; reduces speckle without merging adjacent crown peaks).
- Must NOT introduce a new heavy dependency. `scipy.ndimage` is the
  one new import — already a transitive dep via geopandas/rasterio.
- Max 200 lines per function, 1000 per file, imports at top, type hints
  on public functions, Google-style docstrings.

## Done Criteria

1. GIVEN a synthetic CHM with three known peaks at known positions
   WHEN `find_tree_tops_from_chm()` runs THEN it returns exactly three
   positions, each within one pixel of the truth.

2. GIVEN predictions that exactly match truth positions
   WHEN `match_predictions_to_truth()` runs THEN it returns a perfect
   match: `precision=1.0`, `recall=1.0`, `F1=1.0`.

3. GIVEN predictions far from truth (beyond tolerance)
   WHEN `match_predictions_to_truth()` runs THEN `precision=0.0`,
   `recall=0.0`, `F1=0.0`.

4. GIVEN partial overlap (e.g. 5 truth, 3 predictions, 3 matching, 0
   false positives, 2 false negatives) WHEN matched THEN
   `precision=1.0`, `recall=0.6`, `F1=0.75`.

5. GIVEN a real Montseny patch + cached LAZ + checkpoint
   WHEN `evaluate_patch_against_lidar()` runs THEN it returns an
   `EvalResult` with finite numbers (no NaN/Inf).

6. GIVEN the CLI `python scripts/run_lidar_eval.py --patches 0250.jpg`
   WHEN run THEN it prints a precision/recall/F1 table and saves a CSV.

7. GIVEN the existing 68 tests WHEN run after this feature lands THEN
   all 68 still pass (backward compat).

8. The eval CLI run on the default 10-patch set produces an
   **aggregate F1 number** that is the project's first honest measure
   of detection quality. Expected range: 0.3 - 0.6 (much lower than
   the biased 0.904).

## Out of Scope

- Modifying `autoresearch/eval.py` (the locked file) or replacing it
  in any pipeline.
- Wiring the new eval into auto-research / hyperparameter search
  (Phase 11).
- Computing a proper precision-recall curve / mAP integral. We report
  P/R/F1 at the model's default confidence threshold; mAP-style sweeps
  can come later if needed.
- Per-species evaluation (Phase 12).
- Hungarian / optimal assignment matching. Greedy is good enough.
- Training data cleaning based on the eval results.
- Visualization of TP/FP/FN per patch (could be a future feature; for
  now we report numbers only).
- Comparing the LiDAR ground truth to manual human annotations (the
  point of this feature is to NOT need humans).
- Multi-threshold evaluation (only run at the 0.3 default RF-DETR
  confidence used everywhere else in the project).
