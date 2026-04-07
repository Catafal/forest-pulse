"""Tests for forest_pulse.classifier — pure functions + synthetic training.

No real LAZ files, no real model checkpoints. Synthetic feature dicts
and synthetic image crops cover all the pure functions. The training
test uses a tiny separable dataset to verify end-to-end behavior.

Phase 9.5b: labels come from tree-top matching, not LiDAR height,
and the feature vector has no LiDAR fields.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from forest_pulse.classifier import (
    DEFAULT_MIN_TOP_HEIGHT_M,
    DEFAULT_NEGATIVE_TOL_M,
    DEFAULT_POSITIVE_TOL_M,
    DEFAULT_PROB_THRESHOLD,
    FEATURE_NAMES,
    TrainingExample,
    TreeClassifier,
    _bbox_geometry,
    _crop_image,
    _features_to_vector,
    _rgb_statistics,
    auto_label_from_tree_top_match,
    extract_classifier_features,
    load_classifier,
    predict_tree_probabilities_batch,
    predict_tree_probability,
    save_classifier,
    train_tree_classifier,
    train_tree_classifier_patch_split,
)
from forest_pulse.health import HealthScore

# ============================================================
# Helpers
# ============================================================


def _make_health(grvi: float = 0.2, exg: float = 30.0) -> HealthScore:
    return HealthScore(
        tree_id=0, grvi=grvi, exg=exg, label="healthy", confidence=0.8,
    )


def _make_features(label: int) -> dict[str, float]:
    """Build a complete 11-feature dict, biased high (tree) or low (FP).

    These separable profiles are used in the training tests to verify
    the classifier can learn a non-trivial decision boundary on pure
    RGB/bbox/health features — no LiDAR in the feature vector.
    """
    if label == 1:
        # Real-tree-like — confident bbox, bigger crown, healthy color
        return {
            "bbox_confidence": 0.9,
            "bbox_area_px": 1500.0,
            "bbox_aspect_ratio": 1.1,
            "grvi": 0.25,
            "exg": 45.0,
            "rgb_mean_r": 70.0,
            "rgb_mean_g": 130.0,
            "rgb_mean_b": 60.0,
            "rgb_std_r": 18.0,
            "rgb_std_g": 25.0,
            "rgb_std_b": 14.0,
        }
    # False-positive-like — low confidence, small bbox, dull color
    return {
        "bbox_confidence": 0.4,
        "bbox_area_px": 200.0,
        "bbox_aspect_ratio": 1.6,
        "grvi": 0.05,
        "exg": 12.0,
        "rgb_mean_r": 90.0,
        "rgb_mean_g": 95.0,
        "rgb_mean_b": 80.0,
        "rgb_std_r": 8.0,
        "rgb_std_g": 9.0,
        "rgb_std_b": 6.0,
    }


# ============================================================
# auto_label_from_tree_top_match
# ============================================================


def test_auto_label_tree_top_match_positive_within_D_pos():
    """Nearest eligible top at 1.5 m → positive label 1."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(1.5, 0.0)],
        tree_top_heights=[10.0],
    )
    assert label == 1


def test_auto_label_tree_top_match_negative_beyond_D_neg():
    """Nearest eligible top at 5 m → negative label 0."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(5.0, 0.0)],
        tree_top_heights=[10.0],
    )
    assert label == 0


def test_auto_label_tree_top_match_buffer_zone_returns_none():
    """Nearest eligible top at 3 m (2 < d <= 4) → None."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(3.0, 0.0)],
        tree_top_heights=[10.0],
    )
    assert label is None


def test_auto_label_tree_top_match_exact_D_pos_boundary_is_positive():
    """Distance exactly = D_pos (2.0) → label 1 (inclusive on lower)."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(2.0, 0.0)],
        tree_top_heights=[10.0],
    )
    assert label == 1


def test_auto_label_tree_top_match_exact_D_neg_boundary_is_buffer():
    """Distance exactly = D_neg (4.0) → None (4.0 still inside buffer,
    strictly greater-than rule applies)."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(4.0, 0.0)],
        tree_top_heights=[10.0],
    )
    assert label is None


def test_auto_label_tree_top_match_ignores_short_tops():
    """Only top nearby is 3 m tall (below min_height=5) → no eligible
    tops at all → label 0 by the no-tops rule (would be 1 if short
    tops were counted, so this test proves they aren't)."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(0.5, 0.0)],
        tree_top_heights=[3.0],
    )
    assert label == 0


def test_auto_label_tree_top_match_no_tops_at_all():
    """Empty tree-top lists → label 0 (nothing to match, definitely FP)."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[],
        tree_top_heights=[],
    )
    assert label == 0


def test_auto_label_tree_top_match_picks_nearest_eligible_over_short():
    """Short top (ineligible) closer than a tall top (eligible in
    buffer). If short tops were counted, label would be 1 (d=0.5).
    Because short tops are ignored, nearest ELIGIBLE is at d=3 →
    buffer zone → None. Proves the function uses the tall top only."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(0.5, 0.0), (3.0, 0.0)],
        tree_top_heights=[3.0, 10.0],  # first is short, second is tall
    )
    assert label is None  # nearest ELIGIBLE top is the tall one at d=3


def test_auto_label_tree_top_match_multiple_tops_picks_minimum_distance():
    """Multiple eligible tops → pick the nearest one for the decision."""
    label = auto_label_from_tree_top_match(
        bbox_center_world=(0.0, 0.0),
        tree_tops_xy=[(10.0, 0.0), (1.0, 0.0), (20.0, 0.0)],
        tree_top_heights=[10.0, 10.0, 10.0],
    )
    assert label == 1  # closest is at 1.0 → positive


# ============================================================
# Bbox geometry + RGB statistics + crop
# ============================================================


def test_bbox_geometry_known_values():
    bbox = np.array([10.0, 20.0, 30.0, 60.0])
    g = _bbox_geometry(bbox)
    assert g["area"] == 800.0       # 20 wide × 40 tall
    assert g["aspect_ratio"] == 2.0  # max(20, 40) / min(20, 40)


def test_bbox_geometry_square():
    bbox = np.array([0.0, 0.0, 10.0, 10.0])
    assert _bbox_geometry(bbox)["aspect_ratio"] == 1.0


def test_bbox_geometry_zero_area():
    bbox = np.array([5.0, 5.0, 5.0, 5.0])
    g = _bbox_geometry(bbox)
    assert g["area"] == 0.0
    assert g["aspect_ratio"] == 0.0


def test_crop_image_clamped_at_edges():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([-10.0, -10.0, 50.0, 50.0])  # extends past origin
    crop = _crop_image(image, bbox)
    # Should clamp to (0, 0, 50, 50) → 50x50x3
    assert crop.shape == (50, 50, 3)


def test_crop_image_completely_outside():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = np.array([200.0, 200.0, 300.0, 300.0])
    crop = _crop_image(image, bbox)
    # Crop is empty (0×0) — handled by callers
    assert crop.size == 0


def test_rgb_statistics_uniform_color():
    crop = np.full((10, 10, 3), 128, dtype=np.uint8)
    stats = _rgb_statistics(crop)
    assert stats["mean_r"] == 128.0
    assert stats["mean_g"] == 128.0
    assert stats["mean_b"] == 128.0
    assert stats["std_r"] == 0.0


def test_rgb_statistics_empty_crop_returns_zeros():
    crop = np.zeros((0, 0, 3), dtype=np.uint8)
    stats = _rgb_statistics(crop)
    assert stats["mean_r"] == 0.0
    assert stats["std_g"] == 0.0


# ============================================================
# extract_classifier_features
# ============================================================


def test_extract_features_returns_all_keys():
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    bbox = np.array([50.0, 50.0, 150.0, 150.0])
    health = _make_health(grvi=0.3, exg=40.0)
    features = extract_classifier_features(image, bbox, 0.85, health)
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_extract_features_finite_values():
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    bbox = np.array([50.0, 50.0, 150.0, 150.0])
    health = _make_health()
    features = extract_classifier_features(image, bbox, 0.7, health)
    for v in features.values():
        assert np.isfinite(v)


def test_extract_features_rgb_means_match_uniform_crop():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 200   # all-red
    image[:, :, 1] = 50
    image[:, :, 2] = 25
    bbox = np.array([10.0, 10.0, 100.0, 100.0])
    health = _make_health()
    features = extract_classifier_features(image, bbox, 0.5, health)
    assert features["rgb_mean_r"] == 200.0
    assert features["rgb_mean_g"] == 50.0
    assert features["rgb_mean_b"] == 25.0


# ============================================================
# _features_to_vector
# ============================================================


def test_features_to_vector_canonical_order():
    feat = _make_features(label=1)
    vector = _features_to_vector(feat)
    assert vector.shape == (len(FEATURE_NAMES),)
    # First element is bbox_confidence per FEATURE_NAMES order
    assert vector[0] == 0.9
    # Last element is rgb_std_b (the last non-lidar feature)
    assert FEATURE_NAMES[-1] == "rgb_std_b"
    assert vector[-1] == 14.0


def test_features_to_vector_missing_key_raises():
    bad = _make_features(label=1)
    del bad["grvi"]
    with pytest.raises(KeyError, match="grvi"):
        _features_to_vector(bad)


# ============================================================
# Training — synthetic separable dataset
# ============================================================


def test_train_synthetic_separable_dataset():
    """Tiny separable dataset → trained model gets >= 80% accuracy."""
    examples = []
    for i in range(8):
        examples.append(TrainingExample(
            features=_make_features(label=1),
            label=1,
            source_patch=f"tree_patch_{i}",
            detection_index=i,
        ))
    for i in range(8):
        examples.append(TrainingExample(
            features=_make_features(label=0),
            label=0,
            source_patch=f"bush_patch_{i}",
            detection_index=i,
        ))
    classifier, metrics = train_tree_classifier(examples, test_size=0.25)
    assert metrics["accuracy"] >= 0.8
    assert metrics["n_train"] >= 1
    assert metrics["n_test"] >= 1
    assert "feature_importance" in metrics


def test_train_handles_empty_examples():
    with pytest.raises(ValueError, match="empty"):
        train_tree_classifier([])


def test_patch_split_assigns_correctly():
    """Both classes in train, both in test — leakage-free split."""
    examples = []
    # Train patches (both classes present)
    for i in range(4):
        examples.append(TrainingExample(
            features=_make_features(label=1),
            label=1, source_patch="train_a", detection_index=i,
        ))
        examples.append(TrainingExample(
            features=_make_features(label=0),
            label=0, source_patch="train_a", detection_index=i + 100,
        ))
    # Test patches (both classes present)
    for i in range(2):
        examples.append(TrainingExample(
            features=_make_features(label=1),
            label=1, source_patch="test_p", detection_index=i,
        ))
        examples.append(TrainingExample(
            features=_make_features(label=0),
            label=0, source_patch="test_p", detection_index=i + 100,
        ))
    classifier, metrics = train_tree_classifier_patch_split(
        examples, test_patch_names={"test_p"},
    )
    assert metrics["n_train"] == 8
    assert metrics["n_test"] == 4
    # Should perfectly classify a separable synthetic set
    assert metrics["accuracy"] >= 0.8


def test_patch_split_empty_test_raises():
    examples = [
        TrainingExample(_make_features(1), 1, "p", 0),
    ]
    with pytest.raises(ValueError, match="test"):
        train_tree_classifier_patch_split(examples, test_patch_names=set())


# ============================================================
# Predict + save/load
# ============================================================


def _train_tiny_classifier() -> TreeClassifier:
    """Helper used by predict + save/load tests."""
    examples = (
        [TrainingExample(_make_features(1), 1, f"t{i}", i) for i in range(8)]
        + [TrainingExample(_make_features(0), 0, f"b{i}", i) for i in range(8)]
    )
    classifier, _ = train_tree_classifier(examples, test_size=0.25)
    return classifier


def test_predict_returns_float_in_unit_interval():
    classifier = _train_tiny_classifier()
    p = predict_tree_probability(classifier, _make_features(1))
    assert 0.0 <= p <= 1.0


def test_predict_high_for_tree_features():
    classifier = _train_tiny_classifier()
    p = predict_tree_probability(classifier, _make_features(1))
    assert p > 0.5  # tree features → tree probability


def test_predict_low_for_bush_features():
    classifier = _train_tiny_classifier()
    p = predict_tree_probability(classifier, _make_features(0))
    assert p < 0.5  # bush features → not-tree probability


def test_predict_batch_correct_length():
    classifier = _train_tiny_classifier()
    feats = [_make_features(1), _make_features(0), _make_features(1)]
    probs = predict_tree_probabilities_batch(classifier, feats)
    assert probs.shape == (3,)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_batch_empty():
    classifier = _train_tiny_classifier()
    probs = predict_tree_probabilities_batch(classifier, [])
    assert probs.shape == (0,)


def test_save_load_round_trip():
    classifier = _train_tiny_classifier()
    feats = _make_features(1)
    p_before = predict_tree_probability(classifier, feats)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "classifier.joblib"
        save_classifier(classifier, path)
        loaded = load_classifier(path)
        p_after = predict_tree_probability(loaded, feats)
        assert p_before == p_after
        assert loaded.feature_names == classifier.feature_names
        assert loaded.threshold == DEFAULT_PROB_THRESHOLD


# ============================================================
# Constants sanity
# ============================================================


def test_labeling_thresholds_match_spec():
    assert DEFAULT_POSITIVE_TOL_M == 2.0
    assert DEFAULT_NEGATIVE_TOL_M == 4.0
    assert DEFAULT_MIN_TOP_HEIGHT_M == 5.0


def test_feature_names_count_is_11():
    assert len(FEATURE_NAMES) == 11


def test_feature_names_contain_no_lidar_fields():
    """No lidar_* features — the entire Phase 9.5 point is RGB-only."""
    assert not any(name.startswith("lidar_") for name in FEATURE_NAMES)
