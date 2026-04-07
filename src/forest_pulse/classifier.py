"""Post-detector tree classifier — multi-modal binary tree-vs-not-tree.

Takes any RF-DETR detection plus its multi-modal context (RGB color
statistics + GRVI/ExG vegetation indices + LiDAR 3D features) and
returns a probability that the detection is a real tree (not a bush,
rock, or false positive).

Trained on **auto-labels derived from LiDAR**:
  - LiDAR `height_p95_m >= 5 m` → label = 1 (tree)
  - LiDAR `height_p95_m <= 2 m` → label = 0 (bush / not-tree)
  - 2 m < height < 5 m → ambiguous, excluded from training

Zero human annotation needed.

## Architecture rationale

This is a TWO-STAGE pipeline. RF-DETR's job is "find pixel regions
that look like tree crowns" (a 2D visual pattern matcher). The
classifier's job is "given this candidate region and all the data we
can attach to it, is it actually a real tree?" (a multi-modal decision).

Mixing the two in one model is an anti-pattern: every classification
improvement would require a 20-minute detector retrain. With the
two-stage design, the classifier iterates in seconds (sklearn
GradientBoostingClassifier on ~500 examples) while the detector stays
frozen.

## Why scikit-learn

`GradientBoostingClassifier` is the right tool for ~500 labeled examples
on 18 tabular features. Marginal accuracy gain from XGBoost / LightGBM
isn't worth the additional dependency at this scale. sklearn is small,
already required for nothing else, and clearly supported on macOS arm64.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forest_pulse.health import HealthScore
from forest_pulse.lidar import LiDARFeatures

logger = logging.getLogger(__name__)

# ============================================================
# Constants — fixed by the SPEC, documented inline
# ============================================================

# LiDAR auto-labeling thresholds. These match Phase 7 conventions
# (Spanish Forest Inventory: trees > 5 m, shrubs < 2 m).
TREE_HEIGHT_THRESHOLD_M = 5.0
BUSH_HEIGHT_THRESHOLD_M = 2.0

# Default cutoff for converting probability → binary tree decision.
# Tunable later if precision/recall trade-off needs adjusting.
DEFAULT_PROB_THRESHOLD = 0.5

# Canonical feature order. The trained model stores feature importances
# in this order, and `_features_to_vector` enforces it. Changing this
# list invalidates any saved model — bump model schema version if so.
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
    # LiDAR 3D features — from lidar.extract_lidar_features
    "lidar_height_p95_m",
    "lidar_height_p50_m",
    "lidar_vertical_spread_m",
    "lidar_point_count",
    "lidar_return_ratio",
    "lidar_intensity_mean",
    "lidar_intensity_std",
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


def auto_label_from_lidar(lidar: LiDARFeatures) -> int | None:
    """Apply the LiDAR height auto-labeling rule.

    The rule has THREE outcomes:
      - height_p95_m >= TREE_HEIGHT_THRESHOLD_M → 1 (definitely a tree)
      - height_p95_m <= BUSH_HEIGHT_THRESHOLD_M → 0 (definitely not)
      - 2 m < height < 5 m → None (ambiguous, exclude from training)

    Returning None for the middle band keeps the training set CLEAN —
    we'd rather have fewer high-confidence labels than many noisy ones.
    The classifier later learns to handle the ambiguous range itself
    (because all 18 features see the relationship).

    Args:
        lidar: LiDARFeatures from forest_pulse.lidar.

    Returns:
        1 (tree), 0 (bush), or None (ambiguous - skip).
    """
    if lidar.height_p95_m >= TREE_HEIGHT_THRESHOLD_M:
        return 1
    if lidar.height_p95_m <= BUSH_HEIGHT_THRESHOLD_M:
        return 0
    return None


# ============================================================
# Public API — feature extraction
# ============================================================


def extract_classifier_features(
    image: np.ndarray,
    bbox_xyxy: np.ndarray,
    bbox_confidence: float,
    health: HealthScore,
    lidar: LiDARFeatures,
) -> dict[str, float]:
    """Build the canonical 18-feature vector for one detection.

    Combines:
      - RF-DETR's own confidence + bbox geometry
      - GRVI / ExG from health scoring
      - RGB color statistics computed from the bbox crop
      - 7 LiDAR fields from extract_lidar_features

    Returns a flat dict keyed by FEATURE_NAMES (no nested structures).
    Callers turn this into a numpy array via `_features_to_vector`.

    Args:
        image: Full RGB image (H, W, 3) uint8.
        bbox_xyxy: Detection bbox in pixel coords [x1, y1, x2, y2].
        bbox_confidence: RF-DETR confidence for this detection.
        health: HealthScore for this detection.
        lidar: LiDARFeatures for this detection.

    Returns:
        Dict with all 18 keys from FEATURE_NAMES, all finite floats.
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
        "lidar_height_p95_m": float(lidar.height_p95_m),
        "lidar_height_p50_m": float(lidar.height_p50_m),
        "lidar_vertical_spread_m": float(lidar.vertical_spread_m),
        "lidar_point_count": float(lidar.point_count),
        "lidar_return_ratio": float(lidar.return_ratio),
        "lidar_intensity_mean": float(lidar.intensity_mean),
        "lidar_intensity_std": float(lidar.intensity_std),
    }


def build_training_examples(
    patch_records: list[dict],
) -> list[TrainingExample]:
    """Convert per-patch detection data → labeled TrainingExamples.

    Each `patch_record` is a dict with keys:
        name, image, detections, health_scores, lidar_features

    For each detection in each patch:
      1. Auto-label from LiDAR height
      2. If ambiguous (None), skip
      3. Else extract features and add a TrainingExample

    Args:
        patch_records: List of dicts as described above.

    Returns:
        List of TrainingExample. Order is patch order then detection
        order. Empty input → empty list.
    """
    examples: list[TrainingExample] = []
    for record in patch_records:
        patch_name = record["name"]
        image = record["image"]
        detections = record["detections"]
        health_scores = record["health_scores"]
        lidar_features = record["lidar_features"]

        # Iterate one detection at a time. Skip examples whose LiDAR
        # height falls in the ambiguous middle band.
        for i, lidar in enumerate(lidar_features):
            label = auto_label_from_lidar(lidar)
            if label is None:
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
                lidar=lidar,
            )
            examples.append(TrainingExample(
                features=features,
                label=label,
                source_patch=patch_name,
                detection_index=i,
            ))

    n_tree = sum(1 for e in examples if e.label == 1)
    n_bush = sum(1 for e in examples if e.label == 0)
    logger.info(
        "build_training_examples: %d total (%d tree, %d bush)",
        len(examples), n_tree, n_bush,
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

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

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

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

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
