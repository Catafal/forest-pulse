"""Tests for forest_pulse.species — broadleaf/conifer classifier.

All synthetic. No real LiDAR data. Covers:
  - Empty input
  - Small-sample fallback (< MIN_SAMPLES_FOR_ZSCORE)
  - Default 60% split
  - Custom target fractions
  - Monotonicity (higher score → more broadleaf-y label)
  - Zero-variance edge case
  - High/low feature extreme classification
  - Synthetic 100-tree dataset with 90%+ accuracy
"""

from __future__ import annotations

import numpy as np

from forest_pulse.species import (
    ABS_INTENSITY_THRESHOLD,
    ABS_RETURN_RATIO_THRESHOLD,
    DEFAULT_BROADLEAF_TARGET_FRACTION,
    SPECIES_GROUP_BROADLEAF,
    SPECIES_GROUP_CONIFER,
    SpeciesGroupPrediction,
    classify_broadleaf_conifer,
)

# ============================================================
# Edge cases
# ============================================================


def test_empty_input_returns_empty_list():
    """Empty arrays → empty output, no crash."""
    result = classify_broadleaf_conifer([], [])
    assert result == []


def test_mismatched_lengths_raise():
    """Mismatched-length inputs raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="same shape"):
        classify_broadleaf_conifer([0.3, 0.5], [1000.0])


def test_single_tree_uses_absolute_threshold():
    """With 1 tree, z-scoring is undefined → absolute fallback."""
    # A tree with HIGH intensity + HIGH return_ratio → broadleaf
    # under absolute thresholds (both above ABS_*_THRESHOLD)
    result = classify_broadleaf_conifer([0.5], [1500.0])
    assert len(result) == 1
    assert result[0].label == SPECIES_GROUP_BROADLEAF


def test_single_low_feature_tree_is_conifer():
    """Single tree with low features → conifer via absolute."""
    # Both features well below absolute thresholds
    result = classify_broadleaf_conifer([0.1], [500.0])
    assert len(result) == 1
    assert result[0].label == SPECIES_GROUP_CONIFER


def test_small_sample_uses_absolute_threshold():
    """5 trees is < MIN_SAMPLES_FOR_ZSCORE → absolute fallback."""
    # All trees have "high" features → all labeled broadleaf
    # via absolute thresholds
    rrs = [0.5, 0.6, 0.55, 0.48, 0.52]
    intys = [1200.0, 1400.0, 1300.0, 1100.0, 1250.0]
    result = classify_broadleaf_conifer(rrs, intys)
    assert len(result) == 5
    assert all(r.label == SPECIES_GROUP_BROADLEAF for r in result)


# ============================================================
# Default 60% split
# ============================================================


def test_default_fraction_splits_60_40():
    """20 trees with diverse features → exactly 60% broadleaf."""
    # Construct 20 trees with a clear gradient from low → high
    # features. At the 60% target, the top 12 should be broadleaf
    # and the bottom 8 should be conifer.
    n = 20
    rrs = np.linspace(0.1, 0.7, n)       # 0.1 → 0.7
    intys = np.linspace(500.0, 1500.0, n)  # 500 → 1500
    result = classify_broadleaf_conifer(rrs, intys)
    assert len(result) == n

    n_broadleaf = sum(1 for r in result if r.label == SPECIES_GROUP_BROADLEAF)
    # At default 60%, expect exactly 12 broadleaves (±1 for edge cases)
    assert abs(n_broadleaf - 12) <= 1


def test_custom_fraction_splits_50_50():
    """Custom 50% target → exactly 50/50 split."""
    n = 20
    rrs = np.linspace(0.1, 0.7, n)
    intys = np.linspace(500.0, 1500.0, n)
    result = classify_broadleaf_conifer(
        rrs, intys, broadleaf_target_fraction=0.5,
    )
    n_broadleaf = sum(1 for r in result if r.label == SPECIES_GROUP_BROADLEAF)
    assert abs(n_broadleaf - 10) <= 1


def test_custom_fraction_splits_80_20():
    """Custom 80% target → 80/20 split."""
    n = 20
    rrs = np.linspace(0.1, 0.7, n)
    intys = np.linspace(500.0, 1500.0, n)
    result = classify_broadleaf_conifer(
        rrs, intys, broadleaf_target_fraction=0.8,
    )
    n_broadleaf = sum(1 for r in result if r.label == SPECIES_GROUP_BROADLEAF)
    assert abs(n_broadleaf - 16) <= 1


# ============================================================
# Classification correctness
# ============================================================


def test_high_features_labeled_broadleaf():
    """Tree with HIGH return_ratio + HIGH intensity → broadleaf."""
    # 10 trees, with tree[0] having extreme high values and the
    # rest clustered at lower values. Tree[0] should always be
    # labeled broadleaf (it's the highest scorer).
    rrs = [0.9] + [0.2] * 9   # extreme high vs cluster of lows
    intys = [2000.0] + [500.0] * 9
    result = classify_broadleaf_conifer(rrs, intys)
    assert result[0].label == SPECIES_GROUP_BROADLEAF


def test_low_features_labeled_conifer():
    """Tree with LOW return_ratio + LOW intensity → conifer."""
    # 10 trees, tree[0] has extreme low values
    rrs = [0.05] + [0.5] * 9
    intys = [200.0] + [1400.0] * 9
    result = classify_broadleaf_conifer(rrs, intys)
    assert result[0].label == SPECIES_GROUP_CONIFER


def test_score_monotonic_with_labels():
    """All broadleaf labels should have scores >= all conifer labels.

    The percentile threshold is a strict cut on the score, so this
    monotonicity must hold by construction.
    """
    n = 20
    rng = np.random.default_rng(42)
    rrs = rng.uniform(0.1, 0.7, n)
    intys = rng.uniform(500, 1500, n)
    result = classify_broadleaf_conifer(rrs, intys)

    broadleaf_scores = [r.score for r in result if r.label == SPECIES_GROUP_BROADLEAF]
    conifer_scores = [r.score for r in result if r.label == SPECIES_GROUP_CONIFER]
    if broadleaf_scores and conifer_scores:
        assert min(broadleaf_scores) >= max(conifer_scores)


# ============================================================
# Zero-variance edge case
# ============================================================


def test_zero_variance_labels_all_broadleaf():
    """All trees with IDENTICAL features → all broadleaf (majority)."""
    n = 15
    rrs = [0.3] * n       # zero variance
    intys = [1000.0] * n  # zero variance
    result = classify_broadleaf_conifer(rrs, intys)
    assert len(result) == n
    assert all(r.label == SPECIES_GROUP_BROADLEAF for r in result)
    # Scores should all be 0 (no variation to score)
    assert all(r.score == 0.0 for r in result)


def test_single_varying_feature_still_classifies():
    """If only return_ratio varies (intensity is flat), use return_ratio alone."""
    n = 20
    rrs = np.linspace(0.1, 0.7, n)
    intys = np.full(n, 1000.0)  # flat
    result = classify_broadleaf_conifer(rrs, intys)
    # With default 60% target, bottom 40% should be conifer
    n_conifer = sum(1 for r in result if r.label == SPECIES_GROUP_CONIFER)
    assert abs(n_conifer - 8) <= 1
    # And the lowest-return-ratio tree should always be conifer
    assert result[0].label == SPECIES_GROUP_CONIFER


# ============================================================
# Synthetic ground-truth accuracy
# ============================================================


def test_synthetic_100_trees_90_percent_accuracy():
    """Construct 100 trees with KNOWN labels:
      - 60 broadleaves: HIGH return_ratio (0.4-0.7) + HIGH intensity (1100-1600)
      - 40 conifers: LOW return_ratio (0.1-0.3) + LOW intensity (500-1000)
    The classifier should recover ≥ 90% of the known labels.
    """
    rng = np.random.default_rng(123)
    n_broadleaf = 60
    n_conifer = 40

    # Broadleaf population
    bl_rr = rng.uniform(0.4, 0.7, n_broadleaf)
    bl_inty = rng.uniform(1100, 1600, n_broadleaf)

    # Conifer population (slight overlap at the edges is realistic)
    cf_rr = rng.uniform(0.1, 0.3, n_conifer)
    cf_inty = rng.uniform(500, 1000, n_conifer)

    all_rr = np.concatenate([bl_rr, cf_rr])
    all_inty = np.concatenate([bl_inty, cf_inty])
    # Ground truth: 1 for broadleaf, 0 for conifer
    truth = np.concatenate([
        np.ones(n_broadleaf, dtype=int),
        np.zeros(n_conifer, dtype=int),
    ])

    results = classify_broadleaf_conifer(
        all_rr, all_inty,
        broadleaf_target_fraction=0.60,  # matches the 60/40 split
    )

    predicted = np.array([
        1 if r.label == SPECIES_GROUP_BROADLEAF else 0
        for r in results
    ])
    accuracy = float((predicted == truth).mean())
    assert accuracy >= 0.90, (
        f"Expected ≥ 90% accuracy on well-separated synthetic "
        f"data, got {accuracy:.3f}"
    )


# ============================================================
# SpeciesGroupPrediction dataclass
# ============================================================


def test_prediction_dataclass_fields():
    """Every returned prediction has all four fields populated."""
    n = 15
    rrs = np.linspace(0.1, 0.7, n)
    intys = np.linspace(500, 1500, n)
    results = classify_broadleaf_conifer(rrs, intys)
    for r in results:
        assert isinstance(r, SpeciesGroupPrediction)
        assert r.label in (SPECIES_GROUP_BROADLEAF, SPECIES_GROUP_CONIFER)
        assert isinstance(r.score, float)
        assert isinstance(r.z_return_ratio, float)
        assert isinstance(r.z_intensity, float)


# ============================================================
# Constants sanity
# ============================================================


def test_default_target_fraction_is_60_pct():
    """Documented Montseny IEFC prior."""
    assert DEFAULT_BROADLEAF_TARGET_FRACTION == 0.60


def test_absolute_thresholds_published_values():
    """Absolute threshold constants match the documented defaults."""
    assert ABS_INTENSITY_THRESHOLD == 1000.0
    assert ABS_RETURN_RATIO_THRESHOLD == 0.30
