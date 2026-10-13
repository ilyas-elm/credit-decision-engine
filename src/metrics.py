"""Discrimination, stability and calibration metrics.

Shared by the champion, the challenger and every arm of the sampling-bias
experiment, so that no comparison is ever between two differently-computed
numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

__all__ = [
    "gini",
    "WeeklyGini",
    "weekly_gini",
    "StabilityResult",
    "stability_metric",
    "psi",
    "brier_score",
    "calibration_by_bucket",
    "calibration_in_the_large",
]


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """``2 * AUC - 1``.

    Returns NaN when the slice holds a single class, where AUC is undefined.
    Weekly slices of a 3%-default portfolio hit this regularly, and returning
    NaN keeps it visible instead of quietly scoring it 0.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.size != y_score.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_score={y_score.size}.")
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return math.nan
    return 2.0 * float(roc_auc_score(y_true, y_score)) - 1.0


@dataclass(frozen=True)
class WeeklyGini:
    """Per-week Gini, with the weeks that were too thin to score kept separate."""

    weeks: np.ndarray
    values: np.ndarray
    n_rows: np.ndarray
    skipped_weeks: np.ndarray

    def __len__(self) -> int:
        return int(self.weeks.size)


def weekly_gini(
    y_true: np.ndarray,
    y_score: np.ndarray,
    weeks: np.ndarray,
    *,
    min_rows: int = 1,
) -> WeeklyGini:
    """Gini per week, oldest week first.

    Weeks that are single-class or thinner than ``min_rows`` are reported in
    ``skipped_weeks`` rather than being dropped silently: a model whose later
    weeks are unscoreable is not a stable model, and that has to stay visible.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    weeks = np.asarray(weeks).ravel()
    if not (y_true.size == y_score.size == weeks.size):
        raise ValueError(
            f"Length mismatch: y_true={y_true.size}, y_score={y_score.size}, weeks={weeks.size}."
        )

    scored_weeks, scored_values, scored_counts, skipped = [], [], [], []
    for week in np.unique(weeks):
        mask = weeks == week
        value = gini(y_true[mask], y_score[mask]) if mask.sum() >= min_rows else math.nan
        if math.isnan(value):
            skipped.append(week)
            continue
        scored_weeks.append(week)
        scored_values.append(value)
        scored_counts.append(int(mask.sum()))

    return WeeklyGini(
        weeks=np.asarray(scored_weeks),
        values=np.asarray(scored_values, dtype=float),
        n_rows=np.asarray(scored_counts, dtype=int),
        skipped_weeks=np.asarray(skipped),
    )


@dataclass(frozen=True)
class StabilityResult:
    """The competition's stability metric, with its parts exposed.

    The parts are kept so that a change in the headline number can be attributed
    to the level, the trend or the noise rather than guessed at.
    """

    metric: float
    mean_gini: float
    slope: float
    residual_std: float
    n_weeks: int


def stability_metric(
    weekly: WeeklyGini | np.ndarray,
    weeks: np.ndarray | None = None,
    *,
    falling_penalty: float = 88.0,
    residual_penalty: float = 0.5,
) -> StabilityResult:
    """``mean(gini) + 88 * min(0, slope) - 0.5 * std(residuals)``.

    Computed honestly, on the weekly Gini curve as it comes out of the model.
    Scores are never post-processed per week to flatter this number: doing so
    would optimise the metric while making the model worse.

    ``std`` is the population standard deviation (``ddof=0``), matching the
    competition's reference implementation.
    """
    if isinstance(weekly, WeeklyGini):
        values, week_index = weekly.values, weekly.weeks
    else:
        values = np.asarray(weekly, dtype=float).ravel()
        if weeks is None:
            raise ValueError("Pass `weeks` when `weekly` is a bare array of Gini values.")
        week_index = np.asarray(weeks).ravel()

    if values.size != week_index.size:
        raise ValueError(f"Length mismatch: gini={values.size}, weeks={week_index.size}.")
    if values.size < 2:
        raise ValueError(
            f"Need at least 2 scoreable weeks to fit a trend, got {values.size}. "
            "A stability metric over one week is meaningless."
        )
    if np.isnan(values).any():
        raise ValueError("Weekly Gini contains NaN; drop or impute unscoreable weeks first.")

    x = week_index.astype(float)
    slope, intercept = np.polyfit(x, values, 1)
    residuals = values - (slope * x + intercept)

    mean_gini = float(values.mean())
    residual_std = float(residuals.std(ddof=0))
    metric = mean_gini + falling_penalty * min(0.0, float(slope)) - residual_penalty * residual_std
    return StabilityResult(
        metric=metric,
        mean_gini=mean_gini,
        slope=float(slope),
        residual_std=residual_std,
        n_weeks=int(values.size),
    )


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between a reference and a comparison sample.

    Bin edges are quantiles of ``expected`` -- the reference period, which in
    this project is always the train weeks. Empty bins are floored at
    ``epsilon`` so the logarithm stays finite; without that a single unpopulated
    bin returns infinity and the feature looks maximally unstable on an artefact.
    """
    expected = np.asarray(expected, dtype=float).ravel()
    actual = np.asarray(actual, dtype=float).ravel()
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return math.nan

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        # Constant reference: no spread to compare against.
        return 0.0 if np.array_equal(np.unique(actual), np.unique(expected)) else math.nan
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_share = np.maximum(exp_counts / expected.size, epsilon)
    act_share = np.maximum(act_counts / actual.size, epsilon)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error of the predicted probabilities."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size != y_prob.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_prob={y_prob.size}.")
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_by_bucket(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_buckets: int = 10,
) -> dict[str, np.ndarray]:
    """Mean predicted PD against observed default rate, by score bucket.

    Buckets are quantiles of the predicted probability, so each holds a similar
    number of applicants. Ties collapse buckets rather than splitting them
    arbitrarily, so fewer than ``n_buckets`` rows can come back.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.size != y_prob.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_prob={y_prob.size}.")

    edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_buckets + 1)))
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    which = np.digitize(y_prob, edges[1:-1], right=False)

    rows = []
    for bucket in np.unique(which):
        mask = which == bucket
        rows.append((bucket, mask.sum(), y_prob[mask].mean(), y_true[mask].mean()))
    rows.sort(key=lambda row: row[2])
    arr = np.asarray(rows, dtype=float)
    return {
        "bucket": arr[:, 0].astype(int),
        "n": arr[:, 1].astype(int),
        "mean_pd": arr[:, 2],
        "observed_rate": arr[:, 3],
    }


def calibration_in_the_large(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weeks: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mean predicted PD against observed default rate, per week.

    The quantity a monitoring dashboard alerts on: a model can keep its ranking
    power (Gini) while drifting badly in level, and only this catches that.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    weeks = np.asarray(weeks).ravel()
    if not (y_true.size == y_prob.size == weeks.size):
        raise ValueError(
            f"Length mismatch: y_true={y_true.size}, y_prob={y_prob.size}, weeks={weeks.size}."
        )

    unique = np.unique(weeks)
    mean_pd = np.empty(unique.size)
    observed = np.empty(unique.size)
    counts = np.empty(unique.size, dtype=int)
    for i, week in enumerate(unique):
        mask = weeks == week
        mean_pd[i] = y_prob[mask].mean()
        observed[i] = y_true[mask].mean()
        counts[i] = int(mask.sum())
    return {
        "week": unique,
        "n": counts,
        "mean_pd": mean_pd,
        "observed_rate": observed,
        "gap": mean_pd - observed,
    }
