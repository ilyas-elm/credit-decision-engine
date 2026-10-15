"""Conformal coverage on synthetic exchangeable data.

The guarantee is ``P(Y in C(X) | Y = k) >= 1 - alpha`` in expectation over the
draw of the calibration set, not for one particular calibration set. So the
coverage tests average over many independent draws rather than asserting on a
single fit, which would be a coin flip dressed up as a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from conformal import (
    DEFAULT,
    NON_DEFAULT,
    Band,
    InsufficientCalibrationWarning,
    MondrianConformal,
    bands,
    conformal_quantile,
    coverage_report,
    marginal_thresholds,
    one_minus_prob,
)

ALPHA = 0.10


def _exchangeable_draw(rng: np.random.Generator, n: int, *, base_rate: float = 0.03):
    """A well-specified, perfectly calibrated oracle.

    x -> true PD -> y. Returning the true PD as the prediction isolates the
    conformal machinery: any coverage shortfall is the calibration procedure's
    fault, not the model's.
    """
    x = rng.normal(0.0, 1.0, n)
    logit = math.log(base_rate / (1 - base_rate)) + 1.5 * x
    pd = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, pd)
    return pd, y


# --- The finite-sample correction ---------------------------------------


def test_conformal_quantile_is_the_ceil_n_plus_1_order_statistic() -> None:
    # n = 9, alpha = 0.1: k = ceil(10 * 0.9) = 9, so the 9th smallest of 9;
    # the maximum. A plain np.quantile(scores, 0.9) would return 8.2 here and
    # quietly under-cover.
    scores = np.arange(1.0, 10.0)  # 1..9
    assert conformal_quantile(scores, 0.10) == 9.0
    assert np.quantile(scores, 0.90) < 9.0

    # n = 19, alpha = 0.1: k = ceil(20 * 0.9) = 18 -> the 18th smallest.
    scores19 = np.arange(1.0, 20.0)
    assert conformal_quantile(scores19, 0.10) == 18.0


def test_conformal_quantile_order_is_what_is_returned_not_the_sorted_position() -> None:
    # Shuffled input must give the same answer as sorted input.
    rng = np.random.default_rng(0)
    scores = rng.permutation(np.arange(1.0, 20.0))
    assert conformal_quantile(scores, 0.10) == 18.0


def test_too_few_calibration_points_warns_and_returns_the_trivial_threshold() -> None:
    # n = 4, alpha = 0.1: k = ceil(5 * 0.9) = 5 > 4. Not certifiable.
    with pytest.warns(InsufficientCalibrationWarning, match="cannot certify"):
        assert conformal_quantile(np.arange(4.0), 0.10) == math.inf


def test_no_calibration_points_warns_and_returns_the_trivial_threshold() -> None:
    with pytest.warns(InsufficientCalibrationWarning, match="No calibration points"):
        assert conformal_quantile(np.array([]), 0.10) == math.inf


def test_alpha_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="alpha must lie"):
        conformal_quantile(np.arange(10.0), 1.5)


def test_score_is_one_minus_probability() -> None:
    assert one_minus_prob(np.array([0.2, 0.9])) == pytest.approx([0.8, 0.1])


# --- Coverage on exchangeable data --------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_mondrian_coverage_holds_per_class_on_exchangeable_data(alpha: float) -> None:
    """The headline guarantee, averaged over independent calibration draws."""
    rng = np.random.default_rng(12345)
    covered = {NON_DEFAULT: [], DEFAULT: []}

    for _ in range(200):
        # A high base rate keeps the default class populated enough to certify
        # alpha; the thin-class case is tested separately below.
        pd_cal, y_cal = _exchangeable_draw(rng, 1_500, base_rate=0.25)
        pd_test, y_test = _exchangeable_draw(rng, 1_000, base_rate=0.25)

        model = MondrianConformal.fit(pd_cal, y_cal, alpha)
        report = coverage_report(model, pd_test, y_test)
        for k in (NON_DEFAULT, DEFAULT):
            value = report["coverage"][report["label"] == k][0]
            if not math.isnan(value):
                covered[k].append(value)

    for k in (NON_DEFAULT, DEFAULT):
        mean_coverage = float(np.mean(covered[k]))
        assert mean_coverage >= 1.0 - alpha - 0.01, (
            f"class {k} covered {mean_coverage:.4f}, target {1 - alpha:.2f}"
        )
        # Conformal is conservative, not arbitrarily so: it should not be
        # covering essentially everything when alpha is loose.
        assert mean_coverage <= 1.0 - alpha + 0.06


def test_marginal_thresholds_undercover_the_minority_class() -> None:
    """Why this project uses Mondrian and not a marginal threshold.

    With ~3% defaults a shared threshold hits its 90% target on average while
    leaving the default class, the one the lender actually cares about,
    materially under-covered.
    """
    rng = np.random.default_rng(7)
    marginal_default, mondrian_default, marginal_overall = [], [], []

    for _ in range(150):
        pd_cal, y_cal = _exchangeable_draw(rng, 6_000, base_rate=0.03)
        pd_test, y_test = _exchangeable_draw(rng, 6_000, base_rate=0.03)

        # The marginal threshold pools the true-label scores of both classes.
        scores_cal = one_minus_prob(np.column_stack([1.0 - pd_cal, pd_cal]))
        shared = marginal_thresholds(scores_cal[np.arange(y_cal.size), y_cal], ALPHA)
        flat = MondrianConformal(alpha=ALPHA, thresholds=shared, n_calibration={})
        flat_report = coverage_report(flat, pd_test, y_test)
        value = flat_report["coverage"][flat_report["label"] == DEFAULT][0]
        if not math.isnan(value):
            marginal_default.append(value)
        marginal_overall.append(
            float(np.average(flat_report["coverage"], weights=flat_report["n"]))
        )

        mondrian = MondrianConformal.fit(pd_cal, y_cal, ALPHA)
        m_report = coverage_report(mondrian, pd_test, y_test)
        m_value = m_report["coverage"][m_report["label"] == DEFAULT][0]
        if not math.isnan(m_value):
            mondrian_default.append(m_value)

    marginal_default_mean = float(np.mean(marginal_default))
    mondrian_default_mean = float(np.mean(mondrian_default))

    # The marginal number looks fine overall...
    assert float(np.mean(marginal_overall)) >= 1.0 - ALPHA - 0.01
    # ...while failing the class that matters.
    assert marginal_default_mean < 1.0 - ALPHA - 0.02, (
        f"marginal covered defaults at {marginal_default_mean:.4f}; the "
        "under-coverage this project exists to avoid did not reproduce"
    )
    # Mondrian repairs it.
    assert mondrian_default_mean >= 1.0 - ALPHA - 0.01
    assert mondrian_default_mean > marginal_default_mean


def test_a_class_too_thin_to_certify_falls_back_to_the_trivial_set() -> None:
    # 10 defaults cannot certify alpha = 0.05 (needs ceil(11 * 0.95) = 11 > 10).
    pd = np.concatenate([np.full(200, 0.02), np.full(10, 0.9)])
    y = np.concatenate([np.zeros(200, dtype=int), np.ones(10, dtype=int)])
    with pytest.warns(InsufficientCalibrationWarning, match="cannot certify"):
        model = MondrianConformal.fit(pd, y, 0.05)
    assert model.thresholds[DEFAULT] == math.inf
    # The trivial threshold covers the default class completely, valid, if
    # uninformative, so every row now includes the default label.
    assert model.prediction_sets(pd)[:, DEFAULT].all()
    assert model.n_calibration == {NON_DEFAULT: 200, DEFAULT: 10}


# --- Bands ---------------------------------------------------------------


def test_singletons_decide_and_ambiguous_sets_are_referred() -> None:
    # Thresholds chosen by hand so each row lands in a known set:
    #   t[0] = 0.3 -> label 0 in the set when 1 - p0 <= 0.3, i.e. p0 >= 0.7
    #   t[1] = 0.3 -> label 1 in the set when p1 >= 0.7
    model = MondrianConformal(alpha=0.1, thresholds={0: 0.3, 1: 0.3}, n_calibration={})
    pd = np.array([0.05, 0.95, 0.50])
    #   pd 0.05 -> p0 0.95 in, p1 0.05 out -> {0}    -> approve
    #   pd 0.95 -> p0 0.05 out, p1 0.95 in -> {1}    -> decline
    #   pd 0.50 -> neither in                -> {}   -> refer
    out = bands(model, pd)
    assert out.tolist() == [Band.APPROVE, Band.DECLINE, Band.REFER]


def test_a_two_label_set_is_referred() -> None:
    # Loose thresholds put both labels in every set.
    model = MondrianConformal(alpha=0.1, thresholds={0: 1.0, 1: 1.0}, n_calibration={})
    out = bands(model, np.array([0.2, 0.5, 0.8]))
    assert (out == Band.REFER).all()
    assert model.set_sizes(np.array([0.2, 0.5, 0.8])).tolist() == [2, 2, 2]


def test_an_empty_set_is_referred_not_silently_approved() -> None:
    # Strict thresholds exclude both labels. The applicant looks strange under
    # either label, which is not a decision to make automatically.
    model = MondrianConformal(alpha=0.1, thresholds={0: -1.0, 1: -1.0}, n_calibration={})
    sets = model.prediction_sets(np.array([0.3]))
    assert sets.sum() == 0
    assert bands(model, np.array([0.3]))[0] == Band.REFER


# --- Coverage reporting --------------------------------------------------


def test_coverage_report_breaks_down_by_class_and_week() -> None:
    rng = np.random.default_rng(3)
    pd_cal, y_cal = _exchangeable_draw(rng, 4_000, base_rate=0.25)
    pd_test, y_test = _exchangeable_draw(rng, 4_000, base_rate=0.25)
    weeks = rng.integers(0, 4, size=y_test.size)

    model = MondrianConformal.fit(pd_cal, y_cal, ALPHA)
    report = coverage_report(model, pd_test, y_test, weeks)

    assert report["week"].size == 4 * 2  # 4 weeks x 2 classes
    assert report["n"].sum() == y_test.size
    assert np.all(report["target_coverage"] == pytest.approx(0.90))
    finite = ~np.isnan(report["coverage"])
    assert np.all(report["coverage"][finite] >= 0.0)
    assert np.all(report["mean_set_size"][finite] <= 2.0)


def test_coverage_report_accepts_a_two_column_probability_matrix() -> None:
    rng = np.random.default_rng(5)
    pd_cal, y_cal = _exchangeable_draw(rng, 3_000, base_rate=0.25)
    model = MondrianConformal.fit(pd_cal, y_cal, ALPHA)

    pd_test, y_test = _exchangeable_draw(rng, 1_000, base_rate=0.25)
    matrix = np.column_stack([1.0 - pd_test, pd_test])
    a = coverage_report(model, pd_test, y_test)
    b = coverage_report(model, matrix, y_test)
    assert a["coverage"] == pytest.approx(b["coverage"], nan_ok=True)


def test_shape_and_label_validation() -> None:
    rng = np.random.default_rng(1)
    pd_cal, y_cal = _exchangeable_draw(rng, 500, base_rate=0.25)
    with pytest.raises(ValueError, match="rows but y_cal has"):
        MondrianConformal.fit(pd_cal, y_cal[:-1], ALPHA)
    with pytest.raises(ValueError, match="1 = default"):
        MondrianConformal.fit(pd_cal, np.full(pd_cal.size, 2), ALPHA)
    with pytest.raises(ValueError, match="Expected a 1-D PD vector"):
        MondrianConformal.fit(np.zeros((10, 3)), np.zeros(10, dtype=int), ALPHA)
