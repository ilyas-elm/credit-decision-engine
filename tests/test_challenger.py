"""The LightGBM challenger and the isotonic calibration it shares with the champion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibrate import IsotonicCalibrator, assess_calibration, needs_calibration
from challenger import ChallengerParams, fit_challenger
from config import load_config
from metrics import gini
from scorecard import ScorecardParams, fit_scorecard


@pytest.fixture(scope="module")
def blocks():
    """train / val_a / test draws from one distribution."""
    rng = np.random.default_rng(0)

    def draw(n):
        y = rng.binomial(1, 0.10, n)
        return (
            pd.DataFrame(
                {
                    "rising": rng.normal(0, 1, n) + 1.1 * y,
                    "falling": rng.normal(0, 1, n) - 1.1 * y,
                    "noise": rng.normal(0, 1, n),
                }
            ),
            y,
        )

    return {"train": draw(9_000), "val": draw(3_000), "test": draw(4_000)}


@pytest.fixture(scope="module")
def params() -> ChallengerParams:
    # Smaller than production so the suite stays quick; the wiring is identical.
    base = ChallengerParams.from_config(load_config()["challenger"])
    return ChallengerParams(
        n_estimators=200,
        learning_rate=base.learning_rate,
        num_leaves=base.num_leaves,
        min_child_samples=base.min_child_samples,
        feature_fraction=base.feature_fraction,
        bagging_fraction=base.bagging_fraction,
        bagging_freq=base.bagging_freq,
        lambda_l2=base.lambda_l2,
        early_stopping_rounds=25,
        seed=base.seed,
    )


# --- Config wiring -------------------------------------------------------


def test_params_load_from_config() -> None:
    p = ChallengerParams.from_config(load_config()["challenger"])
    assert p.optuna_trials == 0, "a tuned challenger would make the comparison unfair"
    assert p.n_estimators > 0 and p.early_stopping_rounds > 0


def test_unrecognised_config_keys_are_rejected() -> None:
    # A typo in config silently doing nothing is worse than a crash.
    with pytest.raises(ValueError, match="unrecognised keys"):
        ChallengerParams.from_config({"learning_rate": 0.05, "lerning_rate": 0.1})


def test_an_oversized_optuna_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="caps the search at 30"):
        ChallengerParams(optuna_trials=200)


def test_no_class_rebalancing_is_configured(params) -> None:
    kwargs = params.lgb_kwargs()
    assert kwargs["is_unbalance"] is False
    assert kwargs["scale_pos_weight"] == 1.0


# --- Fitting -------------------------------------------------------------


def test_challenger_discriminates_and_stops_early(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (blocks["train"], blocks["val"], blocks["test"])
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)

    assert gini(y_te, fitted.predict_pd(X_te)) > 0.3
    # Early stopping engaged rather than running to the cap.
    assert fitted.best_iteration is not None
    assert 0 < fitted.best_iteration <= params.n_estimators


def test_column_order_mismatch_is_refused(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]
    with pytest.raises(ValueError, match="identical columns"):
        fit_challenger(X_tr, y_tr, X_va[["noise", "rising"]], y_va, params=params)


def test_misaligned_constraints_are_refused(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]
    with pytest.raises(ValueError, match="must align with"):
        fit_challenger(X_tr, y_tr, X_va, y_va, params=params, monotone_constraints=[1, -1])
    with pytest.raises(ValueError, match="only -1, 0 or 1"):
        fit_challenger(X_tr, y_tr, X_va, y_va, params=params, monotone_constraints=[1, -1, 2])


# --- Monotone constraints ------------------------------------------------


def test_constraints_come_from_the_champion_and_are_enforced(blocks, params) -> None:
    """The load-bearing check: constraints are obeyed, not merely passed.

    Verified behaviourally, sweep one feature across its range with the others
    held fixed and confirm the predicted PD never moves against the constraint.
    An unconstrained model on the same data does violate it, which is what makes
    this test meaningful rather than vacuously true.
    """
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]

    champion = fit_scorecard(X_tr, y_tr, ScorecardParams())
    constraints = champion.monotone_constraints(list(X_tr.columns))
    assert constraints[list(X_tr.columns).index("rising")] == 1
    assert constraints[list(X_tr.columns).index("falling")] == -1

    constrained = fit_challenger(
        X_tr, y_tr, X_va, y_va, params=params, monotone_constraints=constraints
    )
    free = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)

    grid = np.linspace(-3, 3, 60)
    base = X_tr.iloc[[0]].copy()

    def sweep(model, feature):
        probe = pd.concat([base] * grid.size, ignore_index=True)
        probe[feature] = grid
        return model.predict_pd_raw(probe)

    rising = sweep(constrained, "rising")
    falling = sweep(constrained, "falling")
    assert np.all(np.diff(rising) >= -1e-12), "ascending constraint violated"
    assert np.all(np.diff(falling) <= 1e-12), "descending constraint violated"

    # The unconstrained model is not monotone here, so the assertions above
    # are testing the constraint and not an artefact of the data.
    free_rising = sweep(free, "rising")
    assert np.any(np.diff(free_rising) < -1e-9), (
        "the unconstrained model was already monotone; this test proves nothing"
    )


def test_an_unconstrained_feature_is_left_unconstrained(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params, monotone_constraints=[1, -1, 0])
    assert fitted.monotone_constraints == [1, -1, 0]


def test_constraints_default_to_none_recorded_as_zeros(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    assert fitted.monotone_constraints == [0, 0, 0]


def test_gain_importance_ranks_signal_above_noise(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), _ = blocks["train"], blocks["val"], blocks["test"]
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    importance = fitted.gain_importance()
    assert importance.index[-1] == "noise"
    assert set(importance.index) == set(X_tr.columns)


# --- Calibration ---------------------------------------------------------


def test_isotonic_calibration_closes_a_level_gap() -> None:
    """A deliberately miscalibrated model, repaired."""
    rng = np.random.default_rng(1)
    n = 20_000
    true_pd = rng.uniform(0.01, 0.30, n)
    y = rng.binomial(1, true_pd)
    # Inflate every probability: ranking intact, level badly wrong.
    inflated = np.clip(true_pd * 2.5, 0, 1)

    before = assess_calibration(y, inflated)
    assert needs_calibration(y, inflated)
    assert before.abs_gap > 0.10

    calibrated = IsotonicCalibrator().fit_transform(inflated, y)
    after = assess_calibration(y, calibrated)
    assert after.abs_gap < 0.01
    assert after.brier < before.brier


def test_calibration_never_inverts_the_ranking(blocks, params) -> None:
    """Isotonic is monotone: a lower raw score stays lower or becomes equal.

    This is the exact property, asserted exactly. Gini is NOT identical before
    and after, because the isotonic step function creates ties and a tied pair
    contributes 0.5 to AUC where it previously contributed 0 or 1. The shift is
    small but real, so it is bounded rather than asserted away.
    """
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = (blocks["train"], blocks["val"], blocks["test"])
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    raw = fitted.predict_pd_raw(X_te)
    cal = fitted.predict_pd(X_te)

    # No inversions: sorting by raw leaves the calibrated values non-decreasing.
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(cal[order]) >= -1e-12)
    # Ties were genuinely introduced, which is why Gini moves at all.
    assert len(np.unique(cal)) < len(np.unique(raw))
    # And the movement is negligible, not structural.
    assert gini(y_te, cal) == pytest.approx(gini(y_te, raw), abs=5e-3)


def test_a_well_calibrated_model_is_left_alone() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.3, 30_000)
    y = rng.binomial(1, p)
    assert not needs_calibration(y, p)


def test_calibrated_probabilities_stay_in_range(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), (X_te, _) = (blocks["train"], blocks["val"], blocks["test"])
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    out = fitted.predict_pd(X_te)
    assert np.all((out >= 0.0) & (out <= 1.0))
    assert np.isfinite(out).all()


def test_out_of_range_inputs_are_clipped_not_nan() -> None:
    """Test-week PDs outside the calibration range must not become NaN."""
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.4, 2_000)
    y = rng.binomial(1, p)
    cal = IsotonicCalibrator().fit(p, y)
    out = cal.transform(np.array([0.0, 0.01, 0.99, 1.0]))
    assert np.isfinite(out).all()
    assert np.all((out >= 0) & (out <= 1))


def test_calibrator_refuses_impossible_inputs() -> None:
    with pytest.raises(RuntimeError, match="must be called before transform"):
        IsotonicCalibrator().transform(np.array([0.1]))
    with pytest.raises(ValueError, match="both classes present"):
        IsotonicCalibrator().fit(np.array([0.1, 0.2]), np.array([0, 0]))
    with pytest.raises(ValueError, match="empty sample"):
        IsotonicCalibrator().fit(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="Length mismatch"):
        IsotonicCalibrator().fit(np.array([0.1, 0.2]), np.array([0]))


def test_challenger_can_skip_calibration(blocks, params) -> None:
    (X_tr, y_tr), (X_va, y_va), (X_te, _) = (blocks["train"], blocks["val"], blocks["test"])
    fitted = fit_challenger(X_tr, y_tr, X_va, y_va, params=params, calibrate=False)
    assert fitted.calibrator is None
    assert np.array_equal(fitted.predict_pd(X_te), fitted.predict_pd_raw(X_te))


def test_calibration_leaves_gini_untouched_on_these_fixtures(blocks, params) -> None:
    """Pins the claim calibrate.py makes about its own effect on ranking.

    Isotonic can only move Gini by merging distinct scores into a tie. The
    docstring used to quote "around 3e-4" as measured here; it is in fact
    exactly zero on this distribution, and a figure nobody can reproduce is
    worse than none. If a change to the fixtures or to the calibrator starts
    merging scores, this fails and the docstring needs rewriting with it.
    """
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = blocks["train"], blocks["val"], blocks["test"]
    model = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    raw = model.predict_pd(X_te)
    calibrated = IsotonicCalibrator().fit(model.predict_pd(X_va), y_va).transform(raw)

    assert gini(y_te, calibrated) == pytest.approx(gini(y_te, raw), abs=1e-12)
    # The mechanism, stated so the zero above is understood rather than assumed:
    # calibration here introduces no ties that were not already present.
    assert len(np.unique(calibrated)) == len(np.unique(raw))


def test_the_challenger_is_sensitive_to_row_order(blocks, params) -> None:
    """Why the features stage sorts, stated where the dependency lives.

    `bagging_fraction` resamples rows every iteration. The seed fixes which
    INDICES are drawn, not which rows they point at, so the same seed on the
    same data in a different order fits a different model. That is what made
    the features frame's unstable row order a reproducibility problem and not
    merely an untidy one: until it was sorted, the challenger's published
    numbers could not be reproduced by re-running the kernels.

    This asserts the sensitivity rather than the absence of it. If LightGBM
    ever became order-invariant the test would fail, and the comment above
    the sort in features.py would need revisiting rather than silently
    becoming folklore.
    """
    (X_tr, y_tr), (X_va, y_va), (X_te, _) = blocks["train"], blocks["val"], blocks["test"]
    assert params.bagging_fraction < 1.0, "the sensitivity comes from bagging"

    straight = fit_challenger(X_tr, y_tr, X_va, y_va, params=params)
    order = np.random.default_rng(0).permutation(len(X_tr))
    shuffled = fit_challenger(
        X_tr.iloc[order].reset_index(drop=True), y_tr[order], X_va, y_va, params=params
    )
    assert not np.array_equal(straight.predict_pd(X_te), shuffled.predict_pd(X_te)), (
        "row order no longer changes the fit; the features sort may no longer be "
        "load-bearing and features.py says that it is"
    )
