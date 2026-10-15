"""Isotonic probability calibration, shared by champion and challenger.

A model can rank well and still be wrong about levels. Gini is blind to that;
pricing, expected-loss and the profit cutoff are not, so probabilities get
calibrated before any of those are computed.

Isotonic rather than Platt scaling: it is non-parametric and so does not assume
the miscalibration is a sigmoid, which LightGBM's is generally not. The cost is
that it can only produce a step function and needs a reasonable amount of data,
which the val_a block supplies.

One caveat, stated because the split assigns val_a two jobs. The brief has val_a
do early stopping AND probability calibration. The tree count is therefore
chosen on the same rows the calibrator is then fitted to, so the calibrated
probabilities are mildly optimistic on val_a itself. It does not touch the test
weeks, which nothing is fitted on, and that is where every reported number comes
from. The alternative, a fifth block, would cost more out-of-time data than
the bias is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

from metrics import brier_score

__all__ = [
    "IsotonicCalibrator",
    "CalibrationAssessment",
    "assess_calibration",
    "needs_calibration",
]


@dataclass
class IsotonicCalibrator:
    """Monotone map from raw scores to calibrated probabilities.

    Monotone by construction, so calibration never INVERTS the ranking: if a
    raw score was lower, the calibrated one is lower or equal.

    Not quite "Gini is unchanged", though. Isotonic output is a step function,
    so it maps distinct raw scores onto equal calibrated ones, and those new
    ties shift AUC slightly, a tied pair counts 0.5 where it previously
    counted 0 or 1. Measured on the test fixtures the shift is around 3e-4 of
    Gini, and it can go either way. So the champion/challenger comparison
    reports which PD it used rather than claiming the two are interchangeable.
    """

    model: IsotonicRegression | None = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        if y_prob.size != y_true.size:
            raise ValueError(f"Length mismatch: y_prob={y_prob.size}, y_true={y_true.size}.")
        if y_prob.size == 0:
            raise ValueError("Cannot calibrate on an empty sample.")
        if np.unique(y_true).size < 2:
            raise ValueError(
                "Calibration needs both classes present; the calibration block has one."
            )
        # clip keeps test-time values outside the fitted range finite rather
        # than returning NaN, which would silently poison downstream metrics.
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model.fit(y_prob, y_true)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("IsotonicCalibrator.fit must be called before transform.")
        return np.clip(
            np.asarray(self.model.predict(np.asarray(y_prob, dtype=float).ravel())), 0.0, 1.0
        )

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(y_prob, y_true).transform(y_prob)


@dataclass(frozen=True)
class CalibrationAssessment:
    """Calibration in the large, plus the Brier score."""

    mean_pd: float
    observed_rate: float
    gap: float
    brier: float

    @property
    def abs_gap(self) -> float:
        return abs(self.gap)


def assess_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationAssessment:
    """Mean predicted PD against the observed default rate, overall."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size != y_prob.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_prob={y_prob.size}.")
    mean_pd = float(y_prob.mean())
    observed = float(y_true.mean())
    return CalibrationAssessment(
        mean_pd=mean_pd,
        observed_rate=observed,
        gap=mean_pd - observed,
        brier=brier_score(y_true, y_prob),
    )


def needs_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    max_abs_gap: float = 0.01,
) -> bool:
    """Whether a model is miscalibrated enough in the large to warrant fixing.

    The brief applies isotonic regression to the challenger unconditionally and
    to the scorecard "if it is miscalibrated". This is that test: a mean
    predicted PD more than ``max_abs_gap`` from the observed rate.

    Deliberately a level check, not a Brier comparison. Brier mixes
    discrimination and calibration, so a model can improve its Brier while its
    level drifts further out, and the level is what the cutoff and the
    expected-loss numbers depend on.
    """
    return assess_calibration(y_true, y_prob).abs_gap > max_abs_gap
