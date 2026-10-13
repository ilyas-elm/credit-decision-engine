"""Metrics against hand-computed examples.

Every expected value here is derived on paper in the comment above it. A metric
test that calls the implementation to compute its own expectation proves only
that the code is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from metrics import (
    brier_score,
    calibration_by_bucket,
    calibration_in_the_large,
    gini,
    psi,
    stability_metric,
    weekly_gini,
)

# --- Gini ----------------------------------------------------------------


def test_gini_perfect_ranking_is_one() -> None:
    # Scores order the two defaults above both non-defaults: AUC = 1, Gini = 1.
    assert gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_gini_reversed_ranking_is_minus_one() -> None:
    # AUC = 0, Gini = -1.
    assert gini([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(-1.0)


def test_gini_hand_computed_intermediate_case() -> None:
    # y = [0, 1, 0, 1], s = [0.1, 0.2, 0.3, 0.4].
    # Positives score {0.2, 0.4}; negatives {0.1, 0.3}. Concordant pairs:
    #   (0.2 vs 0.1) yes, (0.2 vs 0.3) no, (0.4 vs 0.1) yes, (0.4 vs 0.3) yes
    # AUC = 3/4 = 0.75  ->  Gini = 2(0.75) - 1 = 0.5.
    assert gini([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5)


def test_gini_is_invariant_to_monotone_rescaling() -> None:
    y = [0, 1, 0, 1, 1, 0]
    s = np.array([0.1, 0.9, 0.2, 0.7, 0.6, 0.05])
    assert gini(y, s) == pytest.approx(gini(y, s * 3.0 + 10.0))


def test_gini_is_nan_for_a_single_class_slice() -> None:
    # AUC is undefined; NaN keeps it visible rather than scoring it 0.
    assert math.isnan(gini([1, 1, 1], [0.1, 0.2, 0.3]))
    assert math.isnan(gini([], []))


def test_gini_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        gini([0, 1], [0.1, 0.2, 0.3])


# --- Weekly Gini ---------------------------------------------------------


def test_weekly_gini_scores_each_week_and_reports_unscoreable_ones() -> None:
    # Weeks 0 and 1 are perfectly ranked; week 2 has one class only.
    y = np.array([0, 1, 0, 1, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    weeks = np.array([0, 0, 1, 1, 2, 2])

    result = weekly_gini(y, s, weeks)
    assert result.weeks.tolist() == [0, 1]
    assert result.values == pytest.approx([1.0, 1.0])
    assert result.n_rows.tolist() == [2, 2]
    assert result.skipped_weeks.tolist() == [2]
    assert len(result) == 2


# --- Stability metric ----------------------------------------------------


def test_stability_metric_flat_curve_is_just_the_mean() -> None:
    # g = [0.5, 0.5, 0.5]: slope 0, residuals 0  ->  0.5 + 0 - 0 = 0.5.
    out = stability_metric(np.array([0.5, 0.5, 0.5]), np.array([0, 1, 2]))
    assert out.slope == pytest.approx(0.0, abs=1e-12)
    assert out.residual_std == pytest.approx(0.0, abs=1e-12)
    assert out.metric == pytest.approx(0.5)
    assert out.n_weeks == 3


def test_stability_metric_does_not_reward_a_rising_curve() -> None:
    # g = [0.4, 0.5, 0.6]: slope +0.1, min(0, slope) = 0, residuals 0 -> 0.5.
    out = stability_metric(np.array([0.4, 0.5, 0.6]), np.array([0, 1, 2]))
    assert out.slope == pytest.approx(0.1)
    assert out.metric == pytest.approx(0.5)


def test_stability_metric_penalises_a_falling_curve_by_88_times_the_slope() -> None:
    # g = [0.6, 0.5, 0.4]: slope -0.1, perfect fit so residual std 0.
    #   0.5 + 88(-0.1) - 0.5(0) = 0.5 - 8.8 = -8.3
    out = stability_metric(np.array([0.6, 0.5, 0.4]), np.array([0, 1, 2]))
    assert out.slope == pytest.approx(-0.1)
    assert out.metric == pytest.approx(-8.3)


def test_stability_metric_penalises_residual_noise() -> None:
    # g = [0.5, 0.6, 0.4] at weeks [0, 1, 2]. mean = 0.5.
    # Sxy = (-1)(0.0) + (0)(0.1) + (1)(-0.1) = -0.1 ; Sxx = 2 ; slope = -0.05
    # intercept = 0.5 - (-0.05)(1) = 0.55 ; fitted = [0.55, 0.50, 0.45]
    # residuals = [-0.05, +0.10, -0.05]
    # std(ddof=0) = sqrt((0.0025 + 0.01 + 0.0025)/3) = sqrt(0.005) = 0.070710678
    #   metric = 0.5 + 88(-0.05) - 0.5(0.070710678) = -3.935355339
    out = stability_metric(np.array([0.5, 0.6, 0.4]), np.array([0, 1, 2]))
    assert out.slope == pytest.approx(-0.05)
    assert out.residual_std == pytest.approx(0.070710678, abs=1e-8)
    assert out.metric == pytest.approx(-3.935355339, abs=1e-8)


def test_stability_metric_accepts_a_weekly_gini_object() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6])
    weeks = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    out = stability_metric(weekly_gini(y, s, weeks))
    assert out.n_weeks == 4
    assert out.metric == pytest.approx(1.0)


def test_stability_metric_refuses_one_week_or_nan() -> None:
    with pytest.raises(ValueError, match="at least 2 scoreable weeks"):
        stability_metric(np.array([0.5]), np.array([0]))
    with pytest.raises(ValueError, match="contains NaN"):
        stability_metric(np.array([0.5, np.nan]), np.array([0, 1]))


# --- PSI -----------------------------------------------------------------


def test_psi_of_a_distribution_against_itself_is_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5_000)
    assert psi(x, x, bins=10) == pytest.approx(0.0, abs=1e-9)


def test_psi_hand_computed_two_bin_shift() -> None:
    # Reference: 50 zeros, 50 ones -> the single interior quantile edge is 0.5,
    # giving reference shares 0.5 / 0.5.
    # Comparison: 90 zeros, 10 ones -> shares 0.9 / 0.1.
    #   PSI = (0.9 - 0.5) ln(0.9/0.5) + (0.1 - 0.5) ln(0.1/0.5)
    #       = 0.4(0.587786665) + (-0.4)(-1.609437912)
    #       = 0.235114666 + 0.643775165 = 0.878889831
    expected = np.array([0.0] * 50 + [1.0] * 50)
    actual = np.array([0.0] * 90 + [1.0] * 10)
    assert psi(expected, actual, bins=2) == pytest.approx(0.878889831, abs=1e-8)


def test_psi_grows_with_the_size_of_the_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 20_000)
    small = psi(ref, rng.normal(0.1, 1.0, 20_000), bins=10)
    large = psi(ref, rng.normal(1.0, 1.0, 20_000), bins=10)
    assert 0.0 <= small < large
    # A one-sigma mean shift is unambiguously "drop it" territory (> 0.25).
    assert large > 0.25


def test_psi_is_finite_when_a_bin_is_empty() -> None:
    # Without the epsilon floor this returns inf and the feature looks
    # maximally unstable purely because of an unpopulated bin.
    ref = np.linspace(0.0, 1.0, 1_000)
    actual = np.full(500, 0.99)
    value = psi(ref, actual, bins=10)
    assert np.isfinite(value) and value > 0.25


def test_psi_ignores_nulls_and_handles_empty_input() -> None:
    ref = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    assert np.isfinite(psi(ref, np.array([0.0, 1.0, 2.0, 3.0]), bins=2))
    assert math.isnan(psi(np.array([]), np.array([1.0])))


# --- Calibration ---------------------------------------------------------


def test_brier_score_hand_computed() -> None:
    # ((0.1-0)^2 + (0.9-1)^2 + (0.2-0)^2 + (0.6-1)^2)/4
    #   = (0.01 + 0.01 + 0.04 + 0.16)/4 = 0.22/4 = 0.055
    assert brier_score([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.6]) == pytest.approx(0.055)


def test_brier_score_is_zero_for_perfect_confident_predictions() -> None:
    assert brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]) == pytest.approx(0.0)


def test_calibration_by_bucket_recovers_a_well_calibrated_model() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.5, 40_000)
    y = rng.binomial(1, p)
    out = calibration_by_bucket(y, p, n_buckets=10)

    assert out["n"].sum() == 40_000
    # Mean PD increases across buckets by construction.
    assert np.all(np.diff(out["mean_pd"]) > 0)
    # A calibrated model tracks the observed rate within sampling noise.
    assert np.max(np.abs(out["mean_pd"] - out["observed_rate"])) < 0.02


def test_calibration_in_the_large_detects_a_level_shift() -> None:
    # Ranking is untouched week to week; only the level drifts. Gini would miss
    # this entirely, which is the reason this metric exists.
    rng = np.random.default_rng(3)
    weeks = np.repeat([0, 1], 10_000)
    p = np.where(weeks == 0, 0.05, 0.05)
    y = np.concatenate([rng.binomial(1, 0.05, 10_000), rng.binomial(1, 0.15, 10_000)])
    out = calibration_in_the_large(y, p, weeks)

    assert out["week"].tolist() == [0, 1]
    assert abs(out["gap"][0]) < 0.01
    assert out["gap"][1] < -0.08  # predicting 5% against a realised ~15%
