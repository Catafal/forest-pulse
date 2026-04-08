"""Binary broadleaf/conifer species classification from LiDAR features.

Phase 12a — the first species enrichment on the Forest Pulse
inventory. Uses two physically-motivated LiDAR signals that Phase 7
was specifically designed to surface:

  - `return_ratio`: fraction of multi-return LiDAR pulses. Broadleaves
    have SPARSER canopy (laser pulses penetrate multiple layers →
    more multi-returns) → HIGHER ratio. Conifers have DENSER needle
    foliage that absorbs most energy on the first return → LOWER
    ratio. Structural signal; robust to sensor drift.

  - `intensity_mean`: mean LiDAR reflectance (amplitude of the return
    pulse). Broadleaf leaves are MORE reflective at 1064 nm than
    conifer needles → HIGHER intensity for broadleaves. Published
    in multiple Mediterranean forestry LiDAR studies.

The comment on `LiDARFeatures.intensity_mean` already reads
"reflectance signal (needle vs broadleaf)" — this module finally
uses what Phase 7 set up.

## Approach: unsupervised percentile-threshold

1. Z-score normalize both features across all input trees.
2. Composite "broadleaf-ness" score = z_return_ratio + z_intensity.
   Higher → more broadleaf-like (sparser canopy AND more reflective).
3. Threshold at the (1 - broadleaf_target_fraction) percentile.
   The top N% become broadleaf, the rest conifer.
4. Default `broadleaf_target_fraction=0.60` matches the published
   Montseny prior from the Catalan Forest Inventory (IEFC).

### Why percentile instead of k-means / GMM

- Fully deterministic (no random_state, no sklearn dependency)
- Equally effective at this scale with two well-correlated features
- Trivially explainable to non-ML users: "top 60% by a composite
  score are labeled broadleaves"
- The RANKING is the physically meaningful part; the split is a
  policy choice (what fraction of the park IS broadleaf)

### Why global not per-patch

A conifer-dominated patch would still split 60/40 within itself if
classified locally — wrong. Global classification produces natural
per-zone variance: dense broadleaf zones naturally end up with
> 60% broadleaf labels, pine zones with < 60%.

## Validation without ground truth

No per-tree species labels exist for Montseny. Phase 12a is
validated indirectly:
  1. Global fraction ≈ 60% (by construction)
  2. Per-zone variance should be present (ecological check)
  3. Synthetic test: 100 trees with known "ground truth" by
     construction should score ≥ 90% accuracy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

SPECIES_GROUP_BROADLEAF = "broadleaf"
SPECIES_GROUP_CONIFER = "conifer"

# Catalan Forest Inventory (IEFC) and Spanish NFI priors for
# Parc Natural del Montseny: ~60% broadleaf-dominant stands
# (holm oak, beech, chestnut, cork oak) and ~40% conifer
# (Pinus halepensis / sylvestris / nigra, with some Abies alba).
DEFAULT_BROADLEAF_TARGET_FRACTION = 0.60

# Minimum sample size for z-score normalization to be meaningful.
# Below this threshold, the mean / std are too noisy to form a
# reliable composite score. Fall back to absolute thresholds.
MIN_SAMPLES_FOR_ZSCORE = 10

# Fallback absolute thresholds (for the small-sample case only).
# Rough Mediterranean Pinus halepensis values from ICGC LiDAR:
# broadleaves typically exceed both thresholds, conifers typically
# stay below both.
ABS_INTENSITY_THRESHOLD = 1000.0
ABS_RETURN_RATIO_THRESHOLD = 0.30


# ============================================================
# Data types
# ============================================================


@dataclass
class SpeciesGroupPrediction:
    """Per-tree species group prediction with provenance.

    `label` is the binary classification; `score` is the raw
    composite z-score (higher = more broadleaf-like); the
    z_* components are kept for downstream inspection / debugging.
    """

    label: str               # SPECIES_GROUP_BROADLEAF | SPECIES_GROUP_CONIFER
    score: float             # composite (z_return_ratio + z_intensity)
    z_return_ratio: float
    z_intensity: float


# ============================================================
# Public API
# ============================================================


def classify_broadleaf_conifer(
    return_ratios: np.ndarray | list[float],
    intensity_means: np.ndarray | list[float],
    broadleaf_target_fraction: float = DEFAULT_BROADLEAF_TARGET_FRACTION,
) -> list[SpeciesGroupPrediction]:
    """Binary broadleaf/conifer classification from LiDAR features.

    Fully unsupervised. Uses percentile-threshold on a composite
    z-score of `return_ratio` and `intensity_mean`. No training
    data, no external ML library, no random state.

    Args:
        return_ratios: Per-tree multi-return fractions from
            `LiDARFeatures.return_ratio`. Either a list or numpy
            array of floats, one per tree.
        intensity_means: Per-tree mean LiDAR intensities from
            `LiDARFeatures.intensity_mean`, in the same order and
            length as `return_ratios`.
        broadleaf_target_fraction: Target fraction of trees to
            label as broadleaf. Default 0.60 matches the Catalan
            Forest Inventory (IEFC) prior for Montseny. Must be
            in (0, 1).

    Returns:
        List of `SpeciesGroupPrediction`, one per input tree, in
        the same order as the inputs. Each prediction carries the
        binary label + raw composite score + z-scored components.

    Edge cases:
        - Empty input → empty list
        - `len(inputs) < MIN_SAMPLES_FOR_ZSCORE` → falls back to
          absolute-threshold classification (no percentile)
        - Zero variance in both features → logs a warning and
          labels all trees as broadleaf (majority class)
    """
    rrs = np.asarray(return_ratios, dtype=np.float64)
    intys = np.asarray(intensity_means, dtype=np.float64)

    if rrs.shape != intys.shape:
        raise ValueError(
            f"return_ratios and intensity_means must have the same "
            f"shape, got {rrs.shape} vs {intys.shape}"
        )
    if rrs.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {rrs.shape}")

    n = rrs.size
    if n == 0:
        return []

    # Small-sample fallback: z-scoring isn't reliable with fewer
    # than ~10 trees, so use absolute physical thresholds instead.
    if n < MIN_SAMPLES_FOR_ZSCORE:
        return [
            _classify_absolute(float(rrs[i]), float(intys[i]))
            for i in range(n)
        ]

    # Compute z-scores. Guard against zero-variance edge cases
    # (e.g., all trees with the same LiDAR features because of
    # a synthetic test or degenerate patch).
    rr_mean = float(rrs.mean())
    rr_std = float(rrs.std())
    inty_mean = float(intys.mean())
    inty_std = float(intys.std())

    zero_variance = (rr_std < 1e-9) and (inty_std < 1e-9)
    if zero_variance:
        logger.warning(
            "classify_broadleaf_conifer: both return_ratio and "
            "intensity_mean have zero variance; labeling all "
            "%d trees as broadleaf (majority class)", n,
        )
        return [
            SpeciesGroupPrediction(
                label=SPECIES_GROUP_BROADLEAF,
                score=0.0,
                z_return_ratio=0.0,
                z_intensity=0.0,
            )
            for _ in range(n)
        ]

    # Compute z-scores per feature. If only one has variance, the
    # other contributes 0 to the composite score.
    z_rr = (
        (rrs - rr_mean) / rr_std
        if rr_std > 1e-9 else np.zeros(n)
    )
    z_inty = (
        (intys - inty_mean) / inty_std
        if inty_std > 1e-9 else np.zeros(n)
    )

    # Composite broadleaf-ness score: higher = more broadleaf-like.
    scores = z_rr + z_inty

    # Threshold at the (1 - target_fraction) percentile. The top
    # N% of trees by score get labeled broadleaf; the bottom get
    # conifer. This enforces the global broadleaf fraction exactly
    # (modulo percentile discretization for small samples).
    threshold_pct = (1.0 - broadleaf_target_fraction) * 100.0
    threshold = float(np.percentile(scores, threshold_pct))

    results: list[SpeciesGroupPrediction] = []
    for i in range(n):
        if scores[i] >= threshold:
            label = SPECIES_GROUP_BROADLEAF
        else:
            label = SPECIES_GROUP_CONIFER
        results.append(SpeciesGroupPrediction(
            label=label,
            score=float(scores[i]),
            z_return_ratio=float(z_rr[i]),
            z_intensity=float(z_inty[i]),
        ))

    n_broadleaf = sum(1 for r in results if r.label == SPECIES_GROUP_BROADLEAF)
    logger.info(
        "classify_broadleaf_conifer: %d trees → %d broadleaf / "
        "%d conifer (target fraction %.2f, actual %.3f)",
        n, n_broadleaf, n - n_broadleaf,
        broadleaf_target_fraction, n_broadleaf / n,
    )
    return results


# ============================================================
# Internal helpers
# ============================================================


def _classify_absolute(
    return_ratio: float, intensity_mean: float,
) -> SpeciesGroupPrediction:
    """Small-sample fallback using absolute physical thresholds.

    When we have fewer than MIN_SAMPLES_FOR_ZSCORE trees, z-scoring
    is unreliable (the mean / std are dominated by noise from the
    few samples). We use absolute thresholds derived from published
    Mediterranean LiDAR studies:
      - return_ratio > 0.30 → broadleaf-like (sparse canopy)
      - intensity_mean > 1000 → broadleaf-like (bright reflectance)

    A simple composite score based on deviations from these
    thresholds decides the label.
    """
    rr_dev = return_ratio - ABS_RETURN_RATIO_THRESHOLD
    # Scale intensity deviation by 1000 so it's in the same
    # numerical range as the return_ratio deviation.
    inty_dev = (intensity_mean - ABS_INTENSITY_THRESHOLD) / 1000.0
    score = rr_dev + inty_dev

    label = (
        SPECIES_GROUP_BROADLEAF if score > 0
        else SPECIES_GROUP_CONIFER
    )
    return SpeciesGroupPrediction(
        label=label,
        score=score,
        z_return_ratio=0.0,  # not applicable in absolute mode
        z_intensity=0.0,
    )
