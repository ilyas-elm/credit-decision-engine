"""Mondrian (class-conditional) split-conformal prediction for the refer band.

Ported from the author's own tabpfn-conformal package (`calibration.py`,
`scores.py`), narrowed to the binary credit case and extended with the
per-class, per-week coverage reporting this project needs.

Two things carried over deliberately:

1. The finite-sample correction ``ceil((n + 1)(1 - alpha))`` rather than a plain
   empirical quantile. ``np.quantile(scores, 1 - alpha)`` is the most common
   conformal bug in the wild and it silently loses the guarantee at small n --
   which is exactly the regime of the default class here, at roughly 3% of a
   portfolio.
2. Returning ``+inf`` (the trivial all-labels set) with a warning when a class
   has too few calibration points to certify the requested alpha, instead of
   clipping the quantile and reporting a guarantee that does not hold.

Why class-conditional and not marginal: a marginal threshold guarantees coverage
*averaged over classes*. Under 3% defaults that average is dominated by the
non-defaulters, so the default class can be badly under-covered while the
headline coverage number looks fine. Mondrian gives one threshold per class, so
the promise is about defaulters specifically.

Calibration data: the val_b weeks, and nothing else. val_b is fitted on by
nothing -- not the model, not the probability calibrator -- because split
conformal's guarantee requires calibration scores to be exchangeable with test
scores and independent of the fitted model.

Drift breaks exchangeability, and this data drifts by construction (it is a
model-stability competition). So coverage is MEASURED per class per test week
rather than assumed; `coverage_report` is the honest check on whether the
guarantee survived.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

__all__ = [
    "InsufficientCalibrationWarning",
    "conformal_quantile",
    "one_minus_prob",
    "MondrianConformal",
    "marginal_thresholds",
    "Band",
    "coverage_report",
]

# Label convention, same as `target` throughout this repo.
NON_DEFAULT, DEFAULT = 0, 1
N_CLASSES = 2


class InsufficientCalibrationWarning(UserWarning):
    """A class has too few calibration points to certify the requested alpha.

    Warned rather than silently clipped. The threshold returned is ``+inf``,
    which yields the trivial all-labels set: still valid, merely uninformative,
    and the caller is told so.
    """


def conformal_quantile(scores: np.ndarray, alpha: float, *, group: object = None) -> float:
    """The ``ceil((n + 1)(1 - alpha))``-th smallest calibration score.

    Returns ``+inf`` when n is too small for the requested alpha, and ``-inf``
    when ``alpha >= 1``. Both are the conservative answers.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha!r}.")

    n = scores.size
    if n == 0:
        warnings.warn(
            f"No calibration points for group {group!r}; returning a trivial "
            "(all-labels) threshold.",
            InsufficientCalibrationWarning,
            stacklevel=2,
        )
        return math.inf

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k <= 0:
        return -math.inf
    if k > n:
        smallest = 1.0 / (n + 1)
        warnings.warn(
            f"Group {group!r} has n={n} calibration points, which cannot certify "
            f"alpha={alpha:g} (smallest achievable is {smallest:.4g}). Returning a "
            "trivial (all-labels) threshold.",
            InsufficientCalibrationWarning,
            stacklevel=2,
        )
        return math.inf

    return float(np.partition(scores, k - 1)[k - 1])


def one_minus_prob(proba: np.ndarray) -> np.ndarray:
    """Nonconformity score ``1 - p``: larger means stranger.

    Any strictly increasing transform of ``1 - p`` gives identical prediction
    sets, since conformal only compares a test score against a quantile of
    calibration scores and a monotone map preserves that order. The choice of
    score is therefore not a tuning knob here; which calibration points the
    quantile is taken over is what matters.
    """
    return 1.0 - np.asarray(proba, dtype=float)


def _as_two_column_proba(proba: np.ndarray) -> np.ndarray:
    """Accept either a 1-D PD vector or an (n, 2) probability matrix."""
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        pd = proba
        proba = np.column_stack([1.0 - pd, pd])
    if proba.ndim != 2 or proba.shape[1] != N_CLASSES:
        raise ValueError(
            f"Expected a 1-D PD vector or an (n, 2) probability matrix, got shape {proba.shape}."
        )
    return proba


def marginal_thresholds(cal_scores: np.ndarray, alpha: float) -> dict[int, float]:
    """One threshold shared by both classes.

    Provided for the comparison in the write-up: it is what *not* to use under
    heavy imbalance. Kept so the under-coverage of the default class can be
    demonstrated rather than asserted.
    """
    shared = conformal_quantile(cal_scores, alpha, group="marginal")
    return {NON_DEFAULT: shared, DEFAULT: shared}


@dataclass(frozen=True)
class MondrianConformal:
    """Per-class thresholds from a class-conditional split-conformal calibration."""

    alpha: float
    thresholds: dict[int, float]
    n_calibration: dict[int, int]

    @classmethod
    def fit(
        cls,
        proba_cal: np.ndarray,
        y_cal: np.ndarray,
        alpha: float,
    ) -> MondrianConformal:
        """Calibrate on held-out data that nothing else was fitted on (val_b)."""
        proba_cal = _as_two_column_proba(proba_cal)
        y_cal = np.asarray(y_cal).ravel().astype(int)
        if proba_cal.shape[0] != y_cal.size:
            raise ValueError(f"proba_cal has {proba_cal.shape[0]} rows but y_cal has {y_cal.size}.")
        unknown = set(np.unique(y_cal).tolist()) - {NON_DEFAULT, DEFAULT}
        if unknown:
            raise ValueError(f"y_cal must be 0/1 with 1 = default; found {sorted(unknown)}.")

        scores = one_minus_prob(proba_cal)
        # Each class's threshold uses only that class's own calibration points,
        # scored on the true label: score[i, y_i].
        thresholds, counts = {}, {}
        for k in (NON_DEFAULT, DEFAULT):
            mask = y_cal == k
            thresholds[k] = conformal_quantile(scores[mask, k], alpha, group=k)
            counts[k] = int(mask.sum())
        return cls(alpha=alpha, thresholds=thresholds, n_calibration=counts)

    def prediction_sets(self, proba: np.ndarray) -> np.ndarray:
        """Boolean ``(n, 2)`` membership matrix: label k is in the set for row i."""
        proba = _as_two_column_proba(proba)
        scores = one_minus_prob(proba)
        keep = np.empty(scores.shape, dtype=bool)
        for k in (NON_DEFAULT, DEFAULT):
            keep[:, k] = scores[:, k] <= self.thresholds[k]
        return keep

    def set_sizes(self, proba: np.ndarray) -> np.ndarray:
        return self.prediction_sets(proba).sum(axis=1)


class Band:
    """Decision bands. Strings, so they survive a parquet round-trip readably."""

    APPROVE = "approve"
    DECLINE = "decline"
    REFER = "refer"


def _bands_from_sets(sets: np.ndarray) -> np.ndarray:
    """Singleton sets decide; everything else is referred to a human.

    A two-label set {non-default, default} means the model cannot separate the
    two at this confidence level. An EMPTY set means the applicant looks strange
    under *both* labels -- rarer, but equally not a decision a model should make
    alone. Both go to manual review, which is what "not a singleton" means in
    the brief.
    """
    out = np.full(sets.shape[0], Band.REFER, dtype=object)
    only_good = sets[:, NON_DEFAULT] & ~sets[:, DEFAULT]
    only_bad = sets[:, DEFAULT] & ~sets[:, NON_DEFAULT]
    out[only_good] = Band.APPROVE
    out[only_bad] = Band.DECLINE
    return out


def bands(model: MondrianConformal, proba: np.ndarray) -> np.ndarray:
    """Convenience wrapper: probabilities straight to approve/decline/refer."""
    return _bands_from_sets(model.prediction_sets(proba))


def coverage_report(
    model: MondrianConformal,
    proba: np.ndarray,
    y_true: np.ndarray,
    weeks: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Realised coverage per class, optionally per week.

    Coverage for class k is the share of true-class-k rows whose prediction set
    contains k. The guarantee is ``>= 1 - alpha`` *under exchangeability*; this
    data drifts, so the point of this function is to find out where that fails
    rather than to confirm it holds.
    """
    proba = _as_two_column_proba(proba)
    y_true = np.asarray(y_true).ravel().astype(int)
    if proba.shape[0] != y_true.size:
        raise ValueError(f"proba has {proba.shape[0]} rows but y_true has {y_true.size}.")

    sets = model.prediction_sets(proba)
    covered = sets[np.arange(y_true.size), y_true]
    sizes = sets.sum(axis=1)

    if weeks is None:
        weeks = np.zeros(y_true.size, dtype=int)
        week_values: list = [0]
    else:
        weeks = np.asarray(weeks).ravel()
        if weeks.size != y_true.size:
            raise ValueError(f"weeks has {weeks.size} rows but y_true has {y_true.size}.")
        week_values = list(np.unique(weeks))

    rows = []
    for week in week_values:
        in_week = weeks == week
        for k in (NON_DEFAULT, DEFAULT):
            mask = in_week & (y_true == k)
            n = int(mask.sum())
            rows.append(
                (
                    week,
                    k,
                    n,
                    float(covered[mask].mean()) if n else math.nan,
                    float(sizes[mask].mean()) if n else math.nan,
                )
            )

    return {
        "week": np.array([r[0] for r in rows]),
        "label": np.array([r[1] for r in rows], dtype=int),
        "n": np.array([r[2] for r in rows], dtype=int),
        "coverage": np.array([r[3] for r in rows], dtype=float),
        "mean_set_size": np.array([r[4] for r in rows], dtype=float),
        "target_coverage": np.full(len(rows), 1.0 - model.alpha),
    }
