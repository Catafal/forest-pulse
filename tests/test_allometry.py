"""Tests for forest_pulse.allometry — DBH + biomass estimation.

Pure arithmetic checks. No synthetic LiDAR, no synthetic forests.
Just verifies the published-form power-law relationships behave
correctly for the edge cases and representative Mediterranean
trees.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from forest_pulse.allometry import (
    BIOMASS_CI_FRACTION,
    DBH_CI_FRACTION,
    SPECIES_GROUP_BROADLEAF,
    SPECIES_GROUP_CONIFER,
    TreeMetrics,
    estimate_tree_metrics,
    estimate_tree_metrics_batch,
)

# ============================================================
# Basic happy path
# ============================================================


def test_broadleaf_mature_holm_oak_reasonable_dbh():
    """A 15 m holm-oak-like tree with a 30 m² crown should have
    DBH in the reasonable 15-35 cm range and positive biomass.
    """
    m = estimate_tree_metrics(
        species_group=SPECIES_GROUP_BROADLEAF,
        height_m=15.0,
        crown_area_m2=30.0,
    )
    assert 15.0 <= m.dbh_cm <= 35.0, (
        f"Expected DBH in [15, 35] cm, got {m.dbh_cm:.1f}"
    )
    assert m.biomass_kg > 0
    assert m.dbh_cm_ci > 0
    assert m.biomass_kg_ci > 0


def test_conifer_mature_pine_reasonable_dbh():
    """A 15 m Pinus-like tree with a 15 m² crown should have DBH
    in the reasonable 10-30 cm range.
    """
    m = estimate_tree_metrics(
        species_group=SPECIES_GROUP_CONIFER,
        height_m=15.0,
        crown_area_m2=15.0,
    )
    assert 10.0 <= m.dbh_cm <= 30.0
    assert m.biomass_kg > 0


def test_broadleaf_and_conifer_differ_for_same_inputs():
    """Same height + crown area should give DIFFERENT DBH for
    broadleaf vs conifer — published coefficients differ.
    """
    bl = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, 15.0, 20.0)
    cf = estimate_tree_metrics(SPECIES_GROUP_CONIFER, 15.0, 20.0)
    assert abs(bl.dbh_cm - cf.dbh_cm) > 0.1
    assert abs(bl.biomass_kg - cf.biomass_kg) > 1.0


# ============================================================
# Edge cases
# ============================================================


def test_zero_crown_area_returns_zeros():
    """A tree with zero crown area → zero metrics, not NaN."""
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, 15.0, 0.0)
    assert m.dbh_cm == 0.0
    assert m.dbh_cm_ci == 0.0
    assert m.biomass_kg == 0.0
    assert m.biomass_kg_ci == 0.0


def test_zero_height_returns_zeros():
    """A tree with zero height → zero metrics."""
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, 0.0, 20.0)
    assert m.dbh_cm == 0.0
    assert m.biomass_kg == 0.0


def test_negative_height_returns_zeros():
    """Negative height shouldn't happen, but if it does → zeros."""
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, -1.0, 20.0)
    assert m.dbh_cm == 0.0


def test_nan_height_propagates_to_nan_output():
    """NaN inputs should NOT be silently coerced to 0."""
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, float("nan"), 20.0)
    assert math.isnan(m.dbh_cm)
    assert math.isnan(m.biomass_kg)


def test_nan_crown_area_propagates():
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, 15.0, float("nan"))
    assert math.isnan(m.dbh_cm)
    assert math.isnan(m.dbh_cm_ci)
    assert math.isnan(m.biomass_kg)
    assert math.isnan(m.biomass_kg_ci)


def test_unknown_species_falls_back_to_broadleaf(caplog):
    """Unknown species string → broadleaf coefficients with a warning."""
    with caplog.at_level("WARNING"):
        m_unknown = estimate_tree_metrics("mysterytree", 15.0, 20.0)
    m_broadleaf = estimate_tree_metrics(
        SPECIES_GROUP_BROADLEAF, 15.0, 20.0,
    )
    assert m_unknown.dbh_cm == m_broadleaf.dbh_cm
    assert "mysterytree" in caplog.text


# ============================================================
# Monotonicity + CI properties
# ============================================================


def test_dbh_monotonic_in_height():
    """Taller trees → larger DBH at fixed crown area."""
    heights = [5.0, 10.0, 15.0, 20.0, 25.0]
    dbhs = [
        estimate_tree_metrics(
            SPECIES_GROUP_BROADLEAF, h, 20.0,
        ).dbh_cm
        for h in heights
    ]
    for i in range(len(dbhs) - 1):
        assert dbhs[i + 1] > dbhs[i]


def test_dbh_monotonic_in_crown_area():
    """Larger crowns → larger DBH at fixed height."""
    crowns = [5.0, 10.0, 20.0, 40.0, 80.0]
    dbhs = [
        estimate_tree_metrics(
            SPECIES_GROUP_BROADLEAF, 15.0, ca,
        ).dbh_cm
        for ca in crowns
    ]
    for i in range(len(dbhs) - 1):
        assert dbhs[i + 1] > dbhs[i]


def test_biomass_monotonic_in_dbh_through_crown():
    """Larger crowns → larger biomass at fixed height (biomass
    monotonic in DBH, DBH monotonic in crown area).
    """
    crowns = [5.0, 10.0, 20.0, 40.0, 80.0]
    biomass = [
        estimate_tree_metrics(
            SPECIES_GROUP_CONIFER, 15.0, ca,
        ).biomass_kg
        for ca in crowns
    ]
    for i in range(len(biomass) - 1):
        assert biomass[i + 1] > biomass[i]


def test_ci_is_fixed_fraction_of_estimate():
    """CIs are fixed fractions of the point estimate."""
    m = estimate_tree_metrics(SPECIES_GROUP_BROADLEAF, 20.0, 40.0)
    assert abs(m.dbh_cm_ci - m.dbh_cm * DBH_CI_FRACTION) < 1e-9
    assert abs(
        m.biomass_kg_ci - m.biomass_kg * BIOMASS_CI_FRACTION
    ) < 1e-9


def test_ci_non_negative_for_valid_input():
    """CIs must never be negative for valid (positive) inputs."""
    for species in (SPECIES_GROUP_BROADLEAF, SPECIES_GROUP_CONIFER):
        for h in (5.0, 10.0, 20.0, 40.0):
            for ca in (1.0, 10.0, 50.0, 200.0):
                m = estimate_tree_metrics(species, h, ca)
                assert m.dbh_cm_ci >= 0
                assert m.biomass_kg_ci >= 0


# ============================================================
# Batch function
# ============================================================


def test_batch_matches_per_tree():
    """estimate_tree_metrics_batch returns the same values as
    calling estimate_tree_metrics per tree.
    """
    species = [SPECIES_GROUP_BROADLEAF, SPECIES_GROUP_CONIFER, SPECIES_GROUP_BROADLEAF]
    heights = [15.0, 18.0, 12.0]
    crowns = [25.0, 12.0, 18.0]

    batch_results = estimate_tree_metrics_batch(species, heights, crowns)
    per_tree_results = [
        estimate_tree_metrics(s, h, c)
        for s, h, c in zip(species, heights, crowns)
    ]

    assert len(batch_results) == 3
    for b, p in zip(batch_results, per_tree_results):
        assert b.dbh_cm == p.dbh_cm
        assert b.biomass_kg == p.biomass_kg


def test_batch_mismatched_length_raises():
    with pytest.raises(ValueError, match="same length"):
        estimate_tree_metrics_batch(
            species_groups=[SPECIES_GROUP_BROADLEAF, SPECIES_GROUP_CONIFER],
            heights_m=[15.0],
            crown_areas_m2=[20.0, 30.0],
        )


def test_batch_accepts_numpy_arrays():
    """Batch function works with numpy array inputs (the most
    common path from the inventory driver).
    """
    species = np.array(
        [SPECIES_GROUP_BROADLEAF, SPECIES_GROUP_CONIFER],
        dtype=object,
    )
    heights = np.array([15.0, 18.0])
    crowns = np.array([25.0, 12.0])
    results = estimate_tree_metrics_batch(species, heights, crowns)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, TreeMetrics)
        assert r.dbh_cm > 0


# ============================================================
# Constants sanity
# ============================================================


def test_default_ci_fractions():
    """The documented CI fractions match the published values."""
    assert DBH_CI_FRACTION == 0.30
    assert BIOMASS_CI_FRACTION == 0.40
