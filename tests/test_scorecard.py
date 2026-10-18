"""The champion scorecard: binning, selection, scaling and the points table."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config import load_config
from metrics import gini
from scorecard import ScorecardParams, fit_scorecard, iv_table, select_features


@pytest.fixture(scope="module")
def frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Features with known roles: two informative, one copy, one useless."""
    rng = np.random.default_rng(0)
    n = 6_000
    y = rng.binomial(1, 0.08, n)
    rising = rng.normal(0.0, 1.0, n) + 1.0 * y  # event rate rises with value
    falling = rng.normal(0.0, 1.0, n) - 1.0 * y  # event rate falls with value
    X = pd.DataFrame(
        {
            "rising": rising,
            "falling": falling,
            # Near-duplicate of `rising` with slightly less signal: the
            # correlation screen must drop this one, not `rising`.
            "rising_copy": rising + rng.normal(0.0, 0.05, n),
            "useless": rng.normal(0.0, 1.0, n),
        }
    )
    return X, y


@pytest.fixture(scope="module")
def params() -> ScorecardParams:
    return ScorecardParams.from_config(load_config()["scorecard"])


# --- Config wiring -------------------------------------------------------


def test_params_come_from_config_not_hardcoded(params: ScorecardParams) -> None:
    assert params.monotonic_trend == "auto_asc_desc"
    assert params.min_iv == 0.02
    assert params.max_abs_corr == 0.80
    assert params.scaling_method == "pdo_odds"
    assert (params.points_at_odds, params.target_odds, params.pdo) == (600.0, 50.0, 20.0)


def test_scaling_params_match_the_exact_keys_optbinning_demands(params) -> None:
    # optbinning rejects the call unless the key set is exactly this.
    assert set(params.scaling_params()) == {"pdo", "odds", "scorecard_points"}


# --- IV and the leakage screen -------------------------------------------


def test_iv_table_ranks_informative_features_above_noise(frame, params) -> None:
    X, y = frame
    ivs = iv_table(X, y, params)

    assert list(ivs.columns) == ["feature", "dtype", "n_bins", "iv", "gini"]
    assert set(ivs["feature"]) == set(X.columns)
    # Sorted descending, and the useless column is last.
    assert ivs["iv"].is_monotonic_decreasing
    assert ivs.iloc[-1]["feature"] == "useless"
    assert ivs.set_index("feature").loc["useless", "iv"] < 0.02
    assert ivs.set_index("feature").loc["rising", "iv"] > 0.1


def test_univariate_gini_is_available_for_the_leakage_screen(frame, params) -> None:
    X, y = frame
    ivs = iv_table(X, y, params).set_index("feature")
    # optbinning's gini is the univariate discrimination the screen prints on.
    assert ivs.loc["rising", "gini"] > ivs.loc["useless", "gini"]
    assert 0.0 <= ivs.loc["useless", "gini"] < 0.2


# --- Selection -----------------------------------------------------------


def test_low_iv_features_are_dropped_with_a_stated_reason(frame, params) -> None:
    X, y = frame
    kept, decisions = select_features(X, y, params)

    assert "useless" not in kept
    reason = decisions.set_index("feature").loc["useless", "reason"]
    assert "iv below 0.02" in reason


def test_correlated_pair_keeps_the_higher_iv_member(frame, params) -> None:
    X, y = frame
    ivs = iv_table(X, y, params).set_index("feature")
    kept, decisions = select_features(X, y, params)

    # Exactly one of the near-duplicate pair survives...
    survivors = {"rising", "rising_copy"} & set(kept)
    assert len(survivors) == 1
    # ...and it is the one with the higher IV.
    winner = survivors.pop()
    loser = ({"rising", "rising_copy"} - {winner}).pop()
    assert ivs.loc[winner, "iv"] >= ivs.loc[loser, "iv"]

    reason = decisions.set_index("feature").loc[loser, "reason"]
    assert "corr" in reason and winner in reason


def test_uncorrelated_informative_features_both_survive(frame, params) -> None:
    X, y = frame
    kept, _ = select_features(X, y, params)
    assert "falling" in kept


def test_every_feature_gets_a_decision_row(frame, params) -> None:
    X, y = frame
    _, decisions = select_features(X, y, params)
    assert set(decisions["feature"]) == set(X.columns)
    assert (decisions["reason"].str.len() > 0).all()


def test_pure_noise_is_dropped_once_the_sample_is_large_enough() -> None:
    """The IV floor only screens noise when there is enough data behind it.

    IV is computed on the same bins it chose, so it is biased upward, and the
    bias shrinks with n. Measured on three pure-noise columns at an 8% event
    rate, max IV runs:

        n =   2,000  ->  0.0889   (4x ABOVE the 0.02 floor: noise survives)
        n =  10,000  ->  0.0117
        n =  50,000  ->  0.0034
        n = 200,000  ->  0.0006

    So min_iv = 0.02 is sound against the full ~900k train rows but NOT on the
    3% local sample. Screening decisions belong to the full-data run; the local
    sample is for making the code work, not for deciding which features live.
    """
    rng = np.random.default_rng(1)
    n = 50_000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = rng.binomial(1, 0.08, n)
    kept, decisions = select_features(X, y, ScorecardParams())

    assert kept == []
    assert not decisions["kept"].any()
    assert decisions["reason"].str.contains("iv below").all()


def test_the_iv_floor_does_not_screen_noise_on_a_small_sample() -> None:
    """The flip side, pinned so the limitation above cannot quietly regress.

    The draw order is part of the procedure, not an accident: y first, then the
    three noise columns. Drawing X first gives 0.0767 here rather than the
    0.0889 the docstring above and the README record, and this test only
    asserted an inequality, so the two disagreed without failing.
    ``test_docs_consistency`` recomputes all four sizes against the documents.
    """
    rng = np.random.default_rng(7)
    n = 2_000
    y = rng.binomial(1, 0.08, n)
    X = pd.DataFrame({f"noise{i}": rng.normal(size=n) for i in range(3)})
    ivs = iv_table(X, y, ScorecardParams())
    assert ivs["iv"].max() > 0.02, (
        "pure noise no longer clears the IV floor at n=2,000; the sample-size "
        "caveat in config/default.toml can be relaxed"
    )


# --- Fitting and scaling -------------------------------------------------


def test_fitted_scorecard_discriminates_and_produces_a_points_table(frame, params) -> None:
    X, y = frame
    kept, _ = select_features(X, y, params)
    fitted = fit_scorecard(X, y, params, features=kept)

    pd_hat = fitted.predict_pd(X)
    assert pd_hat.shape == (X.shape[0],)
    assert np.all((pd_hat >= 0) & (pd_hat <= 1))
    assert gini(y, pd_hat) > 0.3

    table = fitted.points_table()
    assert {"characteristic", "bin", "woe", "points"} <= set(table.columns)
    assert set(table["characteristic"]) == set(kept)
    assert table["points"].notna().all()


def test_points_fall_as_predicted_risk_rises(frame, params) -> None:
    """Under pdo_odds, more points must mean less risk, or the scale is inverted.

    Tested on RANKS. Points are linear in the log-odds, and PD is a logistic
    function of the log-odds, so points-against-PD is perfectly monotone but
    distinctly nonlinear, Pearson correlation on it sits around -0.78 and
    would make this assertion look like a real defect.
    """
    X, y = frame
    kept, _ = select_features(X, y, params)
    fitted = fit_scorecard(X, y, params, features=kept)

    # Sorting by points must put PD in non-increasing order, exactly.
    order = np.argsort(fitted.score(X), kind="stable")
    assert np.all(np.diff(fitted.predict_pd(X)[order]) <= 1e-12)


def test_pdo_is_honoured_twenty_points_doubles_the_odds(frame, params) -> None:
    """The defining property of the scale: PDO = 20 means +20 points doubles
    the good:bad odds. Checked by regressing log-odds on points.
    """
    X, y = frame
    kept, _ = select_features(X, y, params)
    fitted = fit_scorecard(X, y, params, features=kept)

    pd_hat = np.clip(fitted.predict_pd(X), 1e-9, 1 - 1e-9)
    good_bad_log_odds = np.log((1 - pd_hat) / pd_hat)
    points = fitted.score(X)

    slope = np.polyfit(points, good_bad_log_odds, 1)[0]
    points_to_double = math.log(2.0) / slope
    assert points_to_double == pytest.approx(params.pdo, rel=0.02), (
        f"{points_to_double:.2f} points double the odds, expected {params.pdo}"
    )


def test_score_at_the_anchor_odds_is_the_configured_points(frame, params) -> None:
    """600 points should correspond to 50:1 good:bad odds."""
    X, y = frame
    kept, _ = select_features(X, y, params)
    fitted = fit_scorecard(X, y, params, features=kept)

    pd_hat = np.clip(fitted.predict_pd(X), 1e-9, 1 - 1e-9)
    log_odds = np.log((1 - pd_hat) / pd_hat)
    slope, intercept = np.polyfit(log_odds, fitted.score(X), 1)
    points_at_anchor = slope * math.log(params.target_odds) + intercept
    assert points_at_anchor == pytest.approx(params.points_at_odds, rel=0.01)


def test_sample_weight_is_accepted_by_binning_and_regression(frame, params) -> None:
    """Needed by the weighted arms of the sampling-bias experiment (A3, A4, A6).

    The brief asks whether optbinning supports this; it does, so no fallback to
    weighting only the logistic regression is required.
    """
    X, y = frame
    kept, _ = select_features(X, y, params)
    weights = np.where(np.asarray(y) == 1, 3.0, 1.0)
    fitted = fit_scorecard(X, y, params, features=kept, sample_weight=weights)
    assert gini(y, fitted.predict_pd(X)) > 0.3


def test_fitting_without_features_is_refused(frame, params) -> None:
    X, y = frame
    with pytest.raises(ValueError, match="at least one feature"):
        fit_scorecard(X, y, params, features=[])
    with pytest.raises(ValueError, match="absent from X"):
        fit_scorecard(X, y, params, features=["nope"])


# --- Trend direction, for the challenger's monotone constraints ----------


def test_trend_direction_recovers_the_known_signs(frame, params) -> None:
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    directions = fitted.trend_direction()
    # `rising` was built so the event rate climbs with the value.
    assert directions["rising"] == 1
    assert directions["falling"] == -1


def test_monotone_constraints_follow_the_requested_column_order(frame, params) -> None:
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    assert fitted.monotone_constraints(["falling", "rising"]) == [-1, 1]
    # A feature the scorecard never binned gets 0, not a guess.
    assert fitted.monotone_constraints(["rising", "unseen"]) == [1, 0]


# --- Principal reasons ---------------------------------------------------


def test_points_below_max_is_non_negative_and_zero_at_the_best_bin(frame, params) -> None:
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    shortfall = fitted.points_below_max(X)

    assert list(shortfall.columns) == ["rising", "falling"]
    assert (shortfall.to_numpy() >= 0).all()
    # Someone sits in the best bin of each characteristic, costing nothing.
    assert shortfall["rising"].min() == pytest.approx(0.0)


def test_awarded_points_reconstruct_the_total_score(frame, params) -> None:
    """A real check on the bin->points mapping: the parts must sum to the whole.

    optbinning's intercept is spread across characteristics, so the per-feature
    points sum to the total score up to a constant offset. That offset must be
    the SAME for every row, which is what pins the mapping down.
    """
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    offset = fitted.score(X) - fitted.feature_points(X).sum(axis=1).to_numpy()
    assert offset.std() < 1e-6, "per-feature points do not reconstruct the score"


def test_principal_reasons_are_ranked_and_capped(frame, params) -> None:
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    reasons = fitted.principal_reasons(X.head(20), top_n=2)

    assert len(reasons) == 20
    for row in reasons:
        assert len(row) == 2
        names = [name for name, _ in row]
        assert set(names) <= {"rising", "falling"}
        shortfalls = [value for _, value in row]
        assert shortfalls == sorted(shortfalls, reverse=True)


def test_reasons_include_non_actionable_drivers(frame, params) -> None:
    """Reasons must be the true drivers, actionable or not.

    `falling` stands in for something like age or history length: if it drives a
    decline it has to appear, even though no applicant can act on it. Filtering
    that out is the recourse step's job, and would make the reasons dishonest.
    """
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    reasons = fitted.principal_reasons(X, top_n=1)
    top_names = {row[0][0] for row in reasons}
    assert "falling" in top_names


def test_a_weak_feature_gets_no_monotone_constraint(frame, params) -> None:
    """Noise that happens to bin monotonically must not constrain the challenger.

    Observed on a real run: a pure-noise column was binned into a monotone
    sequence and drew a confident -1. Constraining the challenger on that is
    constraining it on nothing, so the direction is gated on the same IV floor
    that governs selection.
    """
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling", "useless"])
    directions = fitted.trend_direction()

    ivs = iv_table(X, y, params).set_index("feature")
    assert ivs.loc["useless", "iv"] < params.min_iv
    assert directions["useless"] == 0, "a sub-threshold feature claimed a trend"
    # The informative ones keep their signs.
    assert directions["rising"] == 1
    assert directions["falling"] == -1


def test_sign_flips_are_detected_and_reported(frame, params) -> None:
    """A characteristic whose points rise with its own event rate is a defect.

    Found on the full-data scorecard: prev_amount_mean had an event rate rising
    from 2.92% to 3.95% across its bins while its points rose too, because the
    multivariate coefficient came out negative. The table then reads "the more
    you previously borrowed, the safer you are", which fails model review.
    """
    X, y = frame
    fitted = fit_scorecard(X, y, params, features=["rising", "falling"])
    flips = fitted.sign_flips()
    # These two are clean: more risk, fewer points.
    assert flips == {}, f"unexpected sign flip: {flips}"

    for feature in fitted.features:
        rows = fitted.points_table()
        rows = rows[rows["characteristic"] == feature]
        rows = rows[~rows["bin"].isin(["Special", "Missing"])]
        corr = np.corrcoef(rows["event_rate"], rows["points"])[0, 1]
        assert corr < 0, f"{feature} scores more points for more risk"


def test_sign_flip_detection_catches_an_inverted_characteristic() -> None:
    """Constructed inversion, so the detector is not merely returning empty."""
    from scorecard import FittedScorecard

    table = pd.DataFrame(
        {
            "Variable": ["x"] * 4,
            "Bin": ["(-inf, 1)", "[1, 2)", "[2, 3)", "[3, inf)"],
            "Event rate": [0.01, 0.02, 0.03, 0.04],
            "Points": [10.0, 20.0, 30.0, 40.0],  # rising with risk: wrong
            "WoE": [0.5, 0.1, -0.1, -0.5],
            "IV": [0.01] * 4,
        }
    )
    fitted = FittedScorecard(model=None, features=["x"], params=ScorecardParams(), table=table)
    flips = fitted.sign_flips()
    assert "x" in flips and flips["x"] == pytest.approx(1.0)


def test_excluded_characteristics_are_barred_before_any_other_rule(frame) -> None:
    """A sign-flipped characteristic is unusable however strong its IV.

    Measured on the full data: excluding one flip simply moves it to the next
    correlated characteristic, because the cause is joint collinearity in the
    previous-applications family rather than any single column.
    """
    X, y = frame
    params = ScorecardParams(excluded_characteristics=("rising",))
    kept, decisions = select_features(X, y, params)

    assert "rising" not in kept, "a barred characteristic survived selection"
    reason = decisions.set_index("feature").loc["rising", "reason"]
    assert "sign flip" in reason
    # It is barred despite having the strongest IV in the frame.
    ivs = iv_table(X, y, params).set_index("feature")["iv"]
    assert ivs["rising"] > params.min_iv
    # The others are unaffected.
    assert "falling" in kept


def test_the_configured_exclusions_are_the_measured_ones() -> None:
    params = ScorecardParams.from_config(load_config()["scorecard"])
    assert set(params.excluded_characteristics) == {
        "prev_amount_mean",
        "prev_amount_max",
        "prev_amount_total",
        "prev_n_applications",
    }


def test_selection_does_not_depend_on_the_row_order_of_the_iv_table() -> None:
    """The correlation rule walks candidates in order, so the order is the rule.

    `select_features` accepts a caller-supplied IV table. Given the same
    truthful table with its rows reversed, an unsorted implementation keeps the
    weaker member of every correlated pair while still appearing to apply the
    brief. The live caller passes `iv_table` output, which is already sorted,
    so this guards the assumption rather than a current defect.
    """
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.1, n)
    strong = y * 2.0 + rng.normal(0, 1.0, n)
    clone = strong + rng.normal(0, 0.05, n)  # all but identical, slightly different IV
    X = pd.DataFrame({"strong": strong, "clone": clone, "noise": rng.normal(0, 1, n)})
    params = ScorecardParams(
        monotonic_trend="auto_asc_desc",
        min_iv=0.02,
        max_abs_corr=0.8,
        scaling_method="pdo_odds",
        points_at_odds=600.0,
        target_odds=50.0,
        pdo=20.0,
    )

    ivs = iv_table(X, y, params)
    assert ivs["iv"].is_monotonic_decreasing, "iv_table must return descending IV"

    descending, _ = select_features(X, y, params, ivs=ivs)
    reversed_rows, _ = select_features(X, y, params, ivs=ivs.iloc[::-1].reset_index(drop=True))
    shuffled, _ = select_features(X, y, params, ivs=ivs.sample(frac=1.0, random_state=1))

    assert set(descending) == set(reversed_rows) == set(shuffled)
    # And it is the higher-IV member of the correlated pair that survives.
    best = ivs.loc[ivs["feature"].isin(["strong", "clone"]), "iv"].idxmax()
    assert ivs.loc[best, "feature"] in descending
