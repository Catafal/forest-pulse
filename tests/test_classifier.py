"""Tests for forest_pulse.classifier — pure functions + tiny synthetic training.

No real LAZ files, no real model checkpoints. Synthetic LiDARFeatures
and synthetic image crops cover all the pure functions. The training
test uses a tiny separable dataset to verify end-to-end behavior.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from forest_pulse.classifier import (
    BUSH_HEIGHT_THRESHOLD_M,
    DEFAULT_PROB_THRESHOLD,
    FEATURE_NAMES,
    TREE_HEIGHT_THRESHOLD_M,
    TrainingExample,
    TreeClassifier,
    _bbox_geometry,
    _crop_image,
    _features_to_vector,
    _rgb_statistics,
    auto_label_from_lidar,
    extract_classifier_features,
    load_classifier,
    predict_tree_probabilities_batch,
    predict_tree_probability,
    save_classifier,
    train_tree_classifier,
    train_tree_classifier_patch_split,
)
from forest_pulse.health import HealthScore
from forest_pulse.lidar import LiDARFeatures

# ============================================================
# Helpers
# ============================================================


def _make_lidar(height: float, **overrides) -> LiDARFeatures:
    defaults = {
        "tree_id": 0,
        "height_p95_m": height,
        "height_p50_m": height * 0.7,
        "vertical_spread_m": 8.0,
        "point_count": 100,
        "return_ratio": 0.5,
        "intensity_mean": 1000.0,
        "intensity_std": 200.0,
    }
    defaults.update(overrides)
    return LiDARFeatures(**defaults)


def _make_health(grvi: float = 0.2, exg: float = 30.0) -> HealthScore:
    return HealthScore(
        tree_id=0, grvi=grvi, exg=exg, label="healthy", confidence=0.8,
    )


def _make_features(label: int) -> dict[str, float]:
    """Build a complete feature dict, biased high (tree) or low (bush)."""
    if label == 1:
        # Tree-like — high LiDAR, big bbox, healthy color
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
            "lidar_height_p95_m": 18.0,
            "lidar_height_p50_m": 12.0,
            "lidar_vertical_spread_m": 14.0,
            "lidar_point_count": 800.0,
            "lidar_return_ratio": 0.85,
            "lidar_intensity_mean": 1400.0,
            "lidar_intensity_std": 300.0,
        }
    # Bush-like — low LiDAR, small bbox, dull color
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
        "lidar_height_p95_m": 1.2,
        "lidar_height_p50_m": 0.6,
        "lidar_vertical_spread_m": 1.0,
        "lidar_point_count": 80.0,
        "lidar_return_ratio": 0.2,
        "lidar_intensity_mean": 700.0,
        "lidar_intensity_std": 90.0,
    }


# ============================================================
# auto_label_from_lidar
# ============================================================


def test_auto_label_tree_above_5m():
    assert auto_label_from_lidar(_make_lidar(10.0)) == 1


def test_auto_label_bush_below_2m():
    assert auto_label_from_lidar(_make_lidar(1.0)) == 0


def test_auto_label_ambiguous_3m_returns_none():
    assert auto_label_from_lidar(_make_lidar(3.0)) is None


def test_auto_label_boundary_exactly_5m_is_tree():
    assert auto_label_from_lidar(_make_lidar(5.0)) == 1


def test_auto_label_boundary_exactly_2m_is_bush():
    assert auto_label_from_lidar(_make_lidar(2.0)) == 0


def test_auto_label_just_above_2m_is_ambiguous():
    assert auto_label_from_lidar(_make_lidar(2.5)) is None


def test_auto_label_just_below_5m_is_ambiguous():
    assert auto_label_from_lidar(_make_lidar(4.99)) is None


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
    lidar = _make_lidar(height=12.0)
    features = extract_classifier_features(image, bbox, 0.85, health, lidar)
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_extract_features_finite_values():
    image = np.full((200, 200, 3), 100, dtype=np.uint8)
    bbox = np.array([50.0, 50.0, 150.0, 150.0])
    health = _make_health()
    lidar = _make_lidar(height=10.0)
    features = extract_classifier_features(image, bbox, 0.7, health, lidar)
    for v in features.values():
        assert np.isfinite(v)


def test_extract_features_rgb_means_match_uniform_crop():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 200   # all-red
    image[:, :, 1] = 50
    image[:, :, 2] = 25
    bbox = np.array([10.0, 10.0, 100.0, 100.0])
    health = _make_health()
    lidar = _make_lidar(8.0)
    features = extract_classifier_features(image, bbox, 0.5, health, lidar)
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
    # Last element is lidar_intensity_std
    assert vector[-1] == 300.0


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


def test_height_thresholds_match_spec():
    assert TREE_HEIGHT_THRESHOLD_M == 5.0
    assert BUSH_HEIGHT_THRESHOLD_M == 2.0


def test_feature_names_count_is_18():
    assert len(FEATURE_NAMES) == 18
