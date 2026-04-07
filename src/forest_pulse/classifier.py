"""Post-detector tree classifier — RGB-distilled LiDAR knowledge.

Takes any RF-DETR detection plus its 2D context (bbox geometry +
GRVI/ExG vegetation indices + RGB color statistics from the bbox crop)
and returns a probability that the detection is a real tree rather
than a false positive.

## Labeling strategy — tree-top matching (Phase 9.5)

Training labels come from MATCHING detections to LiDAR tree-tops
found via local-max filtering on a Canopy Height Model:

  - bbox center within 2.0 m of a LiDAR tree-top (height >= 5 m) → 1
  - bbox center more than 4.0 m from any eligible tree-top          → 0
  - 2.0 m < distance <= 4.0 m                                       → None

The 2.0 m positive threshold matches the Phase 8 eval matching
tolerance EXACTLY — train on the metric you evaluate. The 4.0 m
negative threshold is generous because `find_tree_tops_from_chm`
uses a 3 m min_distance, so legitimate merged-crown partners can
live within 3 m of each other.

## Why RGB-only (no LiDAR features in the vector)

The critical insight from Phase 9: if labels come from LiDAR AND
features come from LiDAR, the classifier is a tautological
"LiDAR-in, LiDAR-out" filter. A 10-line deterministic function
(`lidar_tree_top_filter`) does the same thing with zero learning.

The VALUE of a learning-based classifier is distillation: train
the model to predict "would LiDAR say this is a tree?" from
RGB/bbox/health features ALONE, so the classifier carries LiDAR
knowledge into settings where LiDAR is not available (drone
imagery, non-Catalunya regions, future deployments).

That's why the feature vector here has NO `lidar_*` entries. The
18-feature Phase 9 schema was collapsed to 11 features deliberately.

## Class imbalance

With tree-top-match labels, the class split flips: most RF-DETR
detections are false positives, so class 0 dominates. We use
`sample_weight = compute_sample_weight('balanced', y_train)` in
the sklearn `.fit()` call to prevent the classifier from trivially
predicting "FP always". GradientBoostingClassifier does not accept
`class_weight=` directly, but sample_weight is equivalent.

## Why scikit-learn

`GradientBoostingClassifier` is the right tool for a few hundred
labeled examples on 11 tabular features. sklearn is small, already
installed via the `[classifier]` extra, works on macOS arm64, and
has no runtime GPU requirement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forest_pulse.health import HealthScore
from forest_pulse.lidar import bbox_centers_to_world

logger = logging.getLogger(__name__)

# ============================================================
# Constants — fixed by the SPEC, documented inline
# ============================================================

# Labeling thresholds for auto_label_from_tree_top_match.
# D_pos must equal the Phase 8 eval match tolerance (2 m) so training
# and evaluation optimize the same objective. D_neg is > 3 m because
# find_tree_tops uses min_distance_m=3, so two legitimate tree-tops
# can sit 3 m apart — detections in the 3-4 m zone could be valid
# matches to a merged-crown partner the local-max filter collapsed.
DEFAULT_POSITIVE_TOL_M = 2.0
DEFAULT_NEGATIVE_TOL_M = 4.0

# Minimum LiDAR peak height to count a tree-top as eligible. Spanish
# Forest Inventory tree cutoff. Consistent with Phase 7/8.
DEFAULT_MIN_TOP_HEIGHT_M = 5.0

# Default cutoff for converting probability → binary tree decision.
# Tunable later if precision/recall trade-off needs adjusting.
DEFAULT_PROB_THRESHOLD = 0.5

# Canonical feature order — 11 features, NO lidar_*. The trained model
# stores feature importances in this order, and `_features_to_vector`
# enforces it. Changing this list invalidates any saved model — bump
# model schema version if so.
FEATURE_NAMES: list[str] = [
    # Detection geometry — what RF-DETR thought of this region
    "bbox_confidence",
    "bbox_area_px",
    "bbox_aspect_ratio",
    # Vegetation indices — from health.py
    "grvi",
    "exg",
    # RGB color statistics — computed from the bbox crop
    "rgb_mean_r", "rgb_mean_g", "rgb_mean_b",
    "rgb_std_r",  "rgb_std_g",  "rgb_std_b",
]


# ============================================================
# Data types
# ============================================================


@dataclass
class TreeClassifier:
    """A trained binary classifier with metadata.

    Wraps the sklearn estimator with the feature schema and decision
    threshold so callers don't have to track those separately.
    """

    model: Any                                 # sklearn GradientBoostingClassifier
    feature_names: list[str]
    threshold: float = DEFAULT_PROB_THRESHOLD


@dataclass
class TrainingExample:
    """One labeled training example.

    `source_patch` and `detection_index` are not used by training
    itself — they're stored so the patch-level split can route examples
    to train vs test based on which patch they came from, and so we
    can debug a specific detection later.
    """

    features: dict[str, float]
    label: int                  # 1 = tree, 0 = bush
    source_patch: str
    detection_index: int


# ============================================================
# Public API — labeling
# ============================================================


def auto_label_from_tree_top_match(
    bbox_center_world: tuple[float, float],
    tree_tops_xy: list[tuple[float, float]],
    tree_top_heights: list[float],
    D_pos_m: float = DEFAULT_POSITIVE_TOL_M,
    D_neg_m: float = DEFAULT_NEGATIVE_TOL_M,
    min_height_m: float = DEFAULT_MIN_TOP_HEIGHT_M,
) -> int | None:
    """Label a detection by distance to the nearest eligible LiDAR tree-top.

    An "eligible" tree-top is one whose reported height is at least
    `min_height_m`. Short tops (shrubs, low vegetation) are ignored
    as if they didn't exist.

    Given the nearest eligible top at squared-distance d_min:

      - d_min <= D_pos_m  → return 1  (real tree)
      - d_min >  D_neg_m  → return 0  (false positive)
      - otherwise         → return None (ambiguous buffer, exclude)

    Edge case: if there are no eligible tree-tops at all, return 0.
    A detection in a patch with no LiDAR-verified trees is a false
    positive by construction — there is nothing for it to match.

    Args:
        bbox_center_world: Detection bbox center in world coords
            (EPSG:25831 meters, same CRS as tree_tops_xy).
        tree_tops_xy: List of (x, y) tree-top world coords for this
            patch. Produced by `find_tree_tops_from_chm`.
        tree_top_heights: List of peak heights parallel to
            tree_tops_xy. Produced by `find_tree_tops_from_chm` with
            `return_heights=True`.
        D_pos_m: Maximum distance for a positive label. Default 2 m
            matches the Phase 8 eval match tolerance.
        D_neg_m: Minimum distance for a negative label. Default 4 m
            leaves a 2-4 m ambiguous buffer.
        min_height_m: Ignore tops shorter than this. Default 5 m.

    Returns:
        1 (real tree), 0 (false positive), or None (ambiguous).
    """
    # Filter tree-tops by height. `find_tree_tops_from_chm` already
    # filters by the same threshold, so this is typically a no-op —
    # but keeping the re-filter here makes the function safe to call
    # in any context and keeps the semantics explicit.
    eligible = [
        xy for xy, h in zip(tree_tops_xy, tree_top_heights)
        if h >= min_height_m
    ]
    if not eligible:
        # No LiDAR-verified trees anywhere nearby → definitely a FP.
        return 0

    cx, cy = bbox_center_world
    # Squared distance is enough — we only need ordering + threshold.
    d_sq_min = float("inf")
    for tx, ty in eligible:
        dx = cx - tx
        dy = cy - ty
        d_sq = dx * dx + dy * dy
        if d_sq < d_sq_min:
            d_sq_min = d_sq

    d_min = float(np.sqrt(d_sq_min))
    if d_min <= D_pos_m:
        return 1
    if d_min > D_neg_m:
        return 0
    return None  # buffer zone — exclude from training


# ============================================================
# Public API — feature extraction
# ============================================================


def extract_classifier_features(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_confidence: float,
    health: HealthScore,
) -> dict[str, float]:
    """Build the canonical 11-feature vector for one detection.

    Combines:
      - RF-DETR's own confidence + bbox geometry (3 features)
      - GRVI / ExG from health scoring (2 features)
      - RGB color statistics computed from the bbox crop (6 features)

    NO LiDAR features — this classifier learns to distill LiDAR
    knowledge into RGB space so it transfers to non-LiDAR settings.
    See module docstring for rationale.

    Args:
        image: Full RGB image (H, W, 3) uint8.
        bbox_xyxy: Detection bbox in pixel coords [x1, y1, x2, y2].
        bbox_confidence: RF-DETR confidence for this detection.
        health: HealthScore for this detection.

    Returns:
        Dict with all 11 keys from FEATURE_NAMES, all finite floats.
    """
    geo = _bbox_geometry(bbox_xyxy)
    crop = _crop_image(image, bbox_xyxy)
    rgb = _rgb_statistics(crop)

    return {
        "bbox_confidence": float(bbox_confidence),
        "bbox_area_px": geo["area"],
        "bbox_aspect_ratio": geo["aspect_ratio"],
        "grvi": float(health.grvi),
        "exg": float(health.exg),
        "rgb_mean_r": rgb["mean_r"],
        "rgb_mean_g": rgb["mean_g"],
        "rgb_mean_b": rgb["mean_b"],
        "rgb_std_r": rgb["std_r"],
        "rgb_std_g": rgb["std_g"],
        "rgb_std_b": rgb["std_b"],
    }


def build_training_examples(
    patch_records: list[dict],
) -> list[TrainingExample]:
    """Convert per-patch detection data → labeled TrainingExamples.

    Each `patch_record` is a dict with keys:
        name, image, detections, health_scores,
        tree_tops_world, tree_top_heights,
        image_bounds, image_size_px

    The `tree_tops_world` + `tree_top_heights` lists come from
    `find_tree_tops_from_chm(..., return_heights=True)` — one per
    patch, shared across all detections in that patch.

    For each detection in each patch:
      1. Compute the detection's bbox center in world coords (batched
         per patch via `bbox_centers_to_world` for vectorization).
      2. Run `auto_label_from_tree_top_match` against the patch's
         tree-tops.
      3. If label is None (buffer zone), skip.
      4. Else extract the 11 RGB/bbox/health features and append
         a TrainingExample.

    Args:
        patch_records: List of dicts as described above.

    Returns:
        List of TrainingExample. Order is patch order then detection
        order. Empty input → empty list.
    """
    examples: list[TrainingExample] = []
    n_skipped_buffer = 0

    for record in patch_records:
        patch_name = record["name"]
        image = record["image"]
        detections = record["detections"]
        health_scores = record["health_scores"]
        tree_tops_world = record["tree_tops_world"]
        tree_top_heights = record["tree_top_heights"]
        image_bounds = record["image_bounds"]
        image_size_px = record["image_size_px"]

        if len(detections) == 0:
            continue

        # Vectorize pixel→world conversion once per patch — cheap and
        # avoids re-running the Y-flip math per detection.
        centers = bbox_centers_to_world(
            detections, image_bounds, image_size_px,
        )

        for i in range(len(detections)):
            center = (float(centers[i, 0]), float(centers[i, 1]))
            label = auto_label_from_tree_top_match(
                bbox_center_world=center,
                tree_tops_xy=tree_tops_world,
                tree_top_heights=tree_top_heights,
            )
            if label is None:
                n_skipped_buffer += 1
                continue

            xyxy = detections.xyxy[i]
            confidence = (
                float(detections.confidence[i])
                if detections.confidence is not None
                else 0.0
            )
            health = health_scores[i]

            features = extract_classifier_features(
                image=image,
                bbox_xyxy=xyxy,
                bbox_confidence=confidence,
                health=health,
            )
            examples.append(TrainingExample(
                features=features,
                label=label,
                source_patch=patch_name,
                detection_index=i,
            ))

    n_tree = sum(1 for e in examples if e.label == 1)
    n_fp = sum(1 for e in examples if e.label == 0)
    logger.info(
        "build_training_examples: %d total "
        "(%d tree, %d false-positive, %d skipped as buffer)",
        len(examples), n_tree, n_fp, n_skipped_buffer,
    )
    return examples


# ============================================================
# Public API — training
# ============================================================


def train_tree_classifier(
    examples: list[TrainingExample],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[TreeClassifier, dict]:
    """Train a GradientBoostingClassifier with a random train/test split.

    This is the simple version — for the leakage-safe version that
    splits by patch, see `train_tree_classifier_patch_split`. Use this
    one in tests with synthetic data where leakage isn't a concern.

    Args:
        examples: List of TrainingExample.
        test_size: Fraction of examples for the test split.
        random_state: For reproducibility.

    Returns:
        Tuple of (classifier, metrics_dict). Metrics keys:
        accuracy, precision, recall, f1, n_train, n_test,
        n_train_tree, n_train_bush, feature_importance.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_sample_weight

    if not examples:
        raise ValueError("Cannot train on an empty examples list.")

    X = np.array([_features_to_vector(e.features) for e in examples])
    y = np.array([e.label for e in examples], dtype=int)

    # Stratify only when both classes are present — otherwise sklearn
    # raises ValueError. Single-class is uninformative but shouldn't
    # crash the pipeline.
    stratify = y if len(set(y.tolist())) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )

    # Balanced sample_weight — tree-top-match labels are usually
    # dominated by the FP class. GradientBoostingClassifier doesn't
    # accept class_weight= directly, so we pass per-sample weights.
    sample_weight = compute_sample_weight("balanced", y_train)

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return _evaluate_and_wrap(
        model, X_train, y_train, X_test, y_test,
        precision_score, recall_score, f1_score, accuracy_score,
    )


def train_tree_classifier_patch_split(
    examples: list[TrainingExample],
    test_patch_names: set[str],
    random_state: int = 42,
) -> tuple[TreeClassifier, dict]:
    """Train with a PATCH-LEVEL split — leakage-safe version.

    Examples whose `source_patch` is in `test_patch_names` go to test;
    everything else goes to train. This prevents detections from the
    same patch leaking across the split.

    Args:
        examples: List of TrainingExample.
        test_patch_names: Patches whose examples form the test set.
        random_state: For reproducibility.

    Returns:
        Tuple of (classifier, metrics_dict). Same shape as
        `train_tree_classifier`.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.utils.class_weight import compute_sample_weight

    if not examples:
        raise ValueError("Cannot train on an empty examples list.")

    train_examples = [e for e in examples if e.source_patch not in test_patch_names]
    test_examples = [e for e in examples if e.source_patch in test_patch_names]

    if not train_examples:
        raise ValueError("Patch split produced an empty training set.")
    if not test_examples:
        raise ValueError("Patch split produced an empty test set.")

    X_train = np.array([_features_to_vector(e.features) for e in train_examples])
    y_train = np.array([e.label for e in train_examples], dtype=int)
    X_test = np.array([_features_to_vector(e.features) for e in test_examples])
    y_test = np.array([e.label for e in test_examples], dtype=int)

    # Balanced sample_weight — see train_tree_classifier for rationale.
    sample_weight = compute_sample_weight("balanced", y_train)

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    return _evaluate_and_wrap(
        model, X_train, y_train, X_test, y_test,
        precision_score, recall_score, f1_score, accuracy_score,
    )


# ============================================================
# Public API — prediction + persistence
# ============================================================


def predict_tree_probability(
    classifier: TreeClassifier,
    features: dict[str, float],
) -> float:
    """Score a single detection. Returns p(tree) ∈ [0, 1]."""
    X = _features_to_vector(features).reshape(1, -1)
    proba = classifier.model.predict_proba(X)[0]
    # GradientBoostingClassifier returns probabilities for each class.
    # Class 1 is "tree". When trained on a single class, predict_proba
    # may have only one column — handle that gracefully.
    if proba.size == 1:
        return float(classifier.model.classes_[0])
    return float(proba[1])


def predict_tree_probabilities_batch(
    classifier: TreeClassifier,
    feature_dicts: list[dict[str, float]],
) -> np.ndarray:
    """Vectorized prediction. Returns a 1D array of p(tree) values."""
    if not feature_dicts:
        return np.array([], dtype=np.float64)
    X = np.array([_features_to_vector(f) for f in feature_dicts])
    proba = classifier.model.predict_proba(X)
    if proba.shape[1] == 1:
        # Single-class model — return all zeros or ones based on which class
        only_class = int(classifier.model.classes_[0])
        return np.full(len(feature_dicts), float(only_class))
    return proba[:, 1].astype(np.float64)


def save_classifier(classifier: TreeClassifier, path: Path) -> Path:
    """Persist a TreeClassifier to disk via joblib.

    Saves a small dict that round-trips through `load_classifier`. We
    serialize the inner sklearn model PLUS the feature names + threshold
    so the loader doesn't need to know FEATURE_NAMES at load time.
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": classifier.model,
        "feature_names": classifier.feature_names,
        "threshold": classifier.threshold,
        "schema_version": 1,
    }
    joblib.dump(payload, path)
    logger.info("Saved classifier to %s", path)
    return path


def load_classifier(path: Path) -> TreeClassifier:
    """Load a previously saved classifier."""
    import joblib

    payload = joblib.load(path)
    return TreeClassifier(
        model=payload["model"],
        feature_names=payload["feature_names"],
        threshold=payload.get("threshold", DEFAULT_PROB_THRESHOLD),
    )


# ============================================================
# Internal helpers
# ============================================================


def _features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Convert a feature dict → 1D float numpy array in canonical order.

    Raises KeyError with a clear message if a required feature is
    missing — better than silently filling with zeros and producing
    a wrong prediction.
    """
    try:
        return np.array(
            [float(features[name]) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
    except KeyError as e:
        raise KeyError(
            f"Missing feature {e!s} — feature dict must contain all of: {FEATURE_NAMES}"
        ) from e


def _crop_image(image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    """Extract a bbox crop, clamped to image bounds.

    Returns the original image (uint8 H×W×3) restricted to the bbox
    region. If the bbox falls entirely outside the image, returns an
    empty array — callers handle that case.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(bbox_xyxy[0]))
    y1 = max(0, int(bbox_xyxy[1]))
    x2 = min(w, int(bbox_xyxy[2]))
    y2 = min(h, int(bbox_xyxy[3]))
    return image[y1:y2, x1:x2]


def _rgb_statistics(crop: np.ndarray) -> dict[str, float]:
    """Mean and std of R, G, B channels over the crop pixels.

    Returns zeros for empty/tiny crops so the feature vector is always
    fully populated. The downstream classifier sees these as a low-signal
    feature and weighs them accordingly.
    """
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return {
            "mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0,
            "std_r":  0.0, "std_g":  0.0, "std_b":  0.0,
        }
    # float64 to avoid uint8 overflow in mean/std arithmetic
    r = crop[:, :, 0].astype(np.float64)
    g = crop[:, :, 1].astype(np.float64)
    b = crop[:, :, 2].astype(np.float64)
    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r":  float(r.std()),
        "std_g":  float(g.std()),
        "std_b":  float(b.std()),
    }


def _bbox_geometry(bbox_xyxy: np.ndarray) -> dict[str, float]:
    """Bbox area + aspect ratio.

    Aspect ratio is `max(w, h) / min(w, h)` so it's always >= 1.
    A perfect square is 1.0, a thin rectangle is large.
    """
    x1, y1, x2, y2 = bbox_xyxy.tolist()
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    if w <= 0 or h <= 0:
        return {"area": 0.0, "aspect_ratio": 0.0}
    aspect = max(w, h) / min(w, h)
    return {"area": float(area), "aspect_ratio": float(aspect)}


def _evaluate_and_wrap(
    model,
    X_train, y_train,
    X_test, y_test,
    precision_score, recall_score, f1_score, accuracy_score,
) -> tuple[TreeClassifier, dict]:
    """Common evaluation + wrapping for both training entry points.

    Computes accuracy, precision, recall, F1 on the test set, plus
    feature importance ranking. Returns the wrapped classifier and a
    metrics dict.
    """
    y_pred = model.predict(X_test)

    # zero_division=0 stops sklearn from spamming warnings on
    # degenerate test sets (e.g. all-tree or all-bush splits).
    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_train_tree": int((y_train == 1).sum()),
        "n_train_bush": int((y_train == 0).sum()),
        "n_test_tree": int((y_test == 1).sum()),
        "n_test_bush": int((y_test == 0).sum()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # Feature importance — sorted high to low so the report shows the
    # most influential features first. Useful for sanity-checking that
    # the model isn't just learning the LiDAR height threshold.
    importances = list(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    importances.sort(key=lambda x: x[1], reverse=True)
    metrics["feature_importance"] = importances

    classifier = TreeClassifier(
        model=model,
        feature_names=list(FEATURE_NAMES),
        threshold=DEFAULT_PROB_THRESHOLD,
    )
    return classifier, metrics
