"""The decision layer: cutoff, bands, reasons, recourse, referral economics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal import Band, MondrianConformal
from decision import (
    Cutoff,
    DecisionEngine,
    Recourse,
    choose_cutoff,
    referral_economics,
)
from profit import ProfitAssumptions, expected_loss_fraction
from scorecard import ScorecardParams, fit_scorecard


@pytest.fixture(scope="module")
def world():
    """A fitted scorecard, a calibrated conformal band and a chosen cutoff.

    Calibration and decisioning use disjoint draws, mirroring the val_b / test
    separation the real pipeline enforces.
    """
    rng = np.random.default_rng(0)
    params = ScorecardParams()

    def draw(n):
        y = rng.binomial(1, 0.12, n)
        return (
            pd.DataFrame(
                {
                    "rising": rng.normal(0, 1, n) + 1.1 * y,
                    "falling": rng.normal(0, 1, n) - 1.1 * y,
                }
            ),
            y,
        )

    X_fit, y_fit = draw(8_000)
    fitted = fit_scorecard(X_fit, y_fit, params)

    X_cal, y_cal = draw(6_000)
    conformal = MondrianConformal.fit(fitted.predict_pd(X_cal), y_cal, 0.10)

    X_test, y_test = draw(6_000)
    cutoff = choose_cutoff(
        y_test, fitted.predict_pd(X_test), fitted.score(X_test), ProfitAssumptions()
    )
    return {
        "scorecard": fitted,
        "conformal": conformal,
        "cutoff": cutoff,
        "X": X_test,
        "y": y_test,
    }


# --- Cutoff --------------------------------------------------------------


def test_cutoff_maximises_profit_on_the_strategy_curve(world) -> None:
    from profit import strategy_curve

    cutoff = world["cutoff"]
    curve = strategy_curve(
        world["y"], world["scorecard"].predict_pd(world["X"]), cutoff.assumptions, n_points=201
    )
    assert cutoff.profit == pytest.approx(curve["profit"].max())
    assert 0.0 < cutoff.acceptance_rate <= 1.0


def test_pd_and_score_thresholds_select_the_same_applicants(world) -> None:
    """Two views of one line. Points are monotone decreasing in PD, so the
    lowest-PD fraction and the highest-scoring fraction must be the same set.
    """
    sc, X, cutoff = world["scorecard"], world["X"], world["cutoff"]
    by_pd = sc.predict_pd(X) <= cutoff.pd_threshold
    by_score = sc.score(X) >= cutoff.score_threshold
    disagreements = int(np.sum(by_pd != by_score))
    # Ties at the quantile can move a handful of rows; anything more means the
    # two thresholds have drifted apart.
    assert disagreements <= 2, f"{disagreements} applicants disagree"


def test_empcs_rate_is_treated_as_a_rejection_fraction(world) -> None:
    """empulse returns the predicted-positive rate, i.e. the fraction REJECTED.

    Reading it as an acceptance fraction would put the optimum near 6% here
    instead of ~96%, a factor of seventeen in the wrong direction.
    """
    cutoff = world["cutoff"]
    assert 0.0 <= cutoff.empcs_optimal_rejection_rate <= 1.0
    assert cutoff.empcs_implied_acceptance_rate == pytest.approx(
        1.0 - cutoff.empcs_optimal_rejection_rate
    )
    # The raw number is small while the real optimum is large: do not confuse them.
    assert cutoff.empcs_optimal_rejection_rate < 0.25
    assert cutoff.acceptance_rate > 0.75
    # Both agree the book should be mostly approved, within the gap expected
    # between two different objectives.
    assert abs(cutoff.empcs_acceptance_gap) < 0.15


def test_empcs_rejection_rate_rises_with_the_bad_rate() -> None:
    """The evidence for the direction, independent of the fixture.

    A rejection rate rises as the book gets worse; an acceptance rate falls.
    """
    rng = np.random.default_rng(1)
    rates = []
    for bad_rate in (0.05, 0.30):
        n = 8_000
        y = rng.binomial(1, bad_rate, n)
        pdv = np.clip(rng.normal(0.5, 0.15, n) + 0.5 * y, 1e-3, 1 - 1e-3)
        cut = choose_cutoff(y, pdv, -pdv, ProfitAssumptions(0.45, 0.10, 0.2644))
        rates.append(cut.empcs_optimal_rejection_rate)
    assert rates[1] > rates[0], f"rate fell as the book worsened: {rates}"


def test_cutoff_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        choose_cutoff(np.array([0, 1]), np.array([0.1, 0.2]), np.array([1.0]))


# --- Bands ---------------------------------------------------------------


def test_bands_are_only_ever_the_three_allowed_values(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    bands = engine.bands(world["X"])
    assert set(np.unique(bands)) <= {Band.APPROVE, Band.REFER, Band.DECLINE}


def test_referral_comes_from_conformal_and_the_cutoff_decides_the_rest(world) -> None:
    """The documented division of labour, asserted rather than assumed."""
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    X, sc, cutoff = world["X"], world["scorecard"], world["cutoff"]

    bands = engine.bands(X)
    sets = world["conformal"].prediction_sets(sc.predict_pd(X))
    singleton = sets.sum(axis=1) == 1

    # Every non-singleton is referred, and nothing else is.
    assert np.array_equal(bands == Band.REFER, ~singleton)
    # Among the non-referred, the cutoff alone decides.
    decided = singleton
    expected = np.where(sc.score(X)[decided] >= cutoff.score_threshold, Band.APPROVE, Band.DECLINE)
    assert np.array_equal(bands[decided], expected)


def test_a_confident_default_set_can_still_be_approved_on_economics(world) -> None:
    """The disagreement case, in the direction it actually occurs.

    At these assumptions (roi 0.2644 against E[loss] 0.275 on a ~12% bad book)
    approving almost everyone is profitable, so the cutoff lands near 96% and is
    looser than the conformal band is confident. The result is applicants whose
    prediction set is the singleton {default} -- the model is confident they
    will default -- being approved anyway because the economics say the portfolio
    still pays.

    That is the intended division of labour, not a bug: conformal measures
    confidence, the cutoff sets risk appetite. It is asserted here so the
    behaviour is a stated consequence rather than a surprise, and so a change in
    the assumptions that reverses it shows up as a test change.
    """
    sc, X, cutoff = world["scorecard"], world["X"], world["cutoff"]
    engine = DecisionEngine(sc, world["conformal"], cutoff)
    bands = engine.bands(X)
    sets = world["conformal"].prediction_sets(sc.predict_pd(X))

    confident_default = (sets.sum(axis=1) == 1) & sets[:, 1]
    approved_anyway = confident_default & (bands == Band.APPROVE)
    assert approved_anyway.any(), "no disagreement case in this sample to demonstrate"
    # Approved purely because the score clears the (loose) economic line.
    assert np.all(sc.score(X)[approved_anyway] >= cutoff.score_threshold)
    assert cutoff.acceptance_rate > 0.9


def test_auto_approve_ignores_the_refer_band(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    auto = engine.auto_approve(world["X"])
    expected = world["scorecard"].score(world["X"]) >= world["cutoff"].score_threshold
    assert np.array_equal(auto, expected)


# --- decide() ------------------------------------------------------------


def test_decide_returns_the_full_contract(world) -> None:
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising"],
    )
    decisions = engine.decide_batch(world["X"].head(200))
    assert len(decisions) == 200

    for d in decisions:
        assert 0.0 <= d.pd <= 1.0
        assert np.isfinite(d.score)
        assert d.band in {Band.APPROVE, Band.REFER, Band.DECLINE}
        if d.band == Band.DECLINE:
            # Reasons and recourse are attached only where a decline needs them.
            assert 1 <= len(d.principal_reasons) <= 4
            assert isinstance(d.recourse, Recourse)
        else:
            assert d.principal_reasons == []
            assert d.recourse is None


def test_decide_on_a_single_row_matches_the_batch(world) -> None:
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising"],
    )
    X = world["X"].head(25)
    batch = engine.decide_batch(X)
    for i in range(len(X)):
        single = engine.decide(X.iloc[[i]])
        assert single.band == batch[i].band
        assert single.pd == pytest.approx(batch[i].pd)
        assert single.score == pytest.approx(batch[i].score)


def test_decide_accepts_a_series(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    row = world["X"].iloc[0]
    from_series = engine.decide(row)
    from_frame = engine.decide(world["X"].iloc[[0]])
    assert from_series.band == from_frame.band
    assert from_series.score == pytest.approx(from_frame.score)


def test_decide_refuses_more_than_one_applicant(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    with pytest.raises(ValueError, match="takes one applicant"):
        engine.decide(world["X"].head(3))


def test_reasons_are_ranked_by_points_lost(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    for d in engine.decide_batch(world["X"].head(300)):
        if d.band == Band.DECLINE:
            values = [v for _, v in d.principal_reasons]
            assert values == sorted(values, reverse=True)
            assert all(v >= 0 for v in values)


def test_unknown_actionable_feature_is_rejected_at_construction(world) -> None:
    with pytest.raises(ValueError, match="does not use"):
        DecisionEngine(
            world["scorecard"],
            world["conformal"],
            world["cutoff"],
            actionable_features=["income_that_does_not_exist"],
        )


# --- Recourse ------------------------------------------------------------


def test_recourse_is_absent_when_nothing_is_configured_actionable(world) -> None:
    """The current config ships an empty actionable list, pending DATA_NOTES."""
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    declines = [d for d in engine.decide_batch(world["X"].head(400)) if d.is_decline]
    assert declines, "no declines in the sample to check"
    for d in declines:
        assert not d.recourse.feasible
        assert "no actionable characteristics" in d.recourse.note


def test_recourse_clears_the_cutoff_when_it_claims_to(world) -> None:
    """The load-bearing property: following the advice must actually work."""
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising", "falling"],
    )
    checked = 0
    for d in engine.decide_batch(world["X"].head(500)):
        if d.is_decline and d.recourse.feasible and d.recourse.steps:
            assert d.recourse.points_gained >= d.recourse.points_needed - 1e-9
            assert d.score + d.recourse.points_gained >= world["cutoff"].score_threshold - 1e-9
            checked += 1
    assert checked > 0, "no feasible recourse in the sample to verify"


def test_recourse_only_touches_actionable_characteristics(world) -> None:
    """`falling` stands in for age: a real driver that must never be advised on."""
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising"],
    )
    for d in engine.decide_batch(world["X"].head(500)):
        if d.is_decline and d.recourse.steps:
            assert {s.characteristic for s in d.recourse.steps} == {"rising"}


def test_reasons_can_name_a_characteristic_recourse_may_not(world) -> None:
    """Reasons stay honest even when recourse is restricted.

    An explanation that hid non-actionable drivers would be a different, worse
    product than one that reports them and declines to offer advice on them.
    """
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising"],
    )
    named_in_reasons = set()
    for d in engine.decide_batch(world["X"].head(500)):
        if d.is_decline:
            named_in_reasons.update(name for name, _ in d.principal_reasons)
    assert "falling" in named_in_reasons


def test_recourse_reports_infeasibility_instead_of_overpromising(world) -> None:
    """A cutoff nobody can reach must come back infeasible, not optimistic."""
    sc, cutoff = world["scorecard"], world["cutoff"]
    impossible = Cutoff(
        acceptance_rate=cutoff.acceptance_rate,
        pd_threshold=cutoff.pd_threshold,
        score_threshold=1e9,  # unreachable
        profit=cutoff.profit,
        profit_per_applicant=cutoff.profit_per_applicant,
        empcs_optimal_rejection_rate=cutoff.empcs_optimal_rejection_rate,
        assumptions=cutoff.assumptions,
    )
    engine = DecisionEngine(
        sc, world["conformal"], impossible, actionable_features=["rising", "falling"]
    )
    decisions = engine.decide_batch(world["X"].head(50))
    assert all(d.band == Band.DECLINE or d.band == Band.REFER for d in decisions)
    infeasible = [d for d in decisions if d.is_decline]
    assert infeasible
    for d in infeasible:
        assert not d.recourse.feasible
        assert "best bin" in d.recourse.note
        assert d.recourse.steps == []


def test_greedy_recourse_changes_the_fewest_characteristics(world) -> None:
    """The greedy order is optimal for 'fewest characteristics changed'.

    With one move available per characteristic, taking the largest gains first
    reaches any shortfall in the minimum number of moves. Checked against
    brute force over every subset.
    """
    from itertools import combinations

    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising", "falling"],
    )
    sc = world["scorecard"]
    X = world["X"].head(300)
    awarded = sc.feature_points(X)
    best = sc.max_points()

    for d, (_, current) in zip(engine.decide_batch(X), awarded.iterrows(), strict=True):
        if not (d.is_decline and d.recourse.feasible and d.recourse.steps):
            continue
        gains = {f: best[f] - float(current[f]) for f in ("rising", "falling")}
        needed = d.recourse.points_needed
        smallest = None
        for size in (1, 2):
            for combo in combinations(gains, size):
                if sum(gains[f] for f in combo) >= needed:
                    smallest = size
                    break
            if smallest:
                break
        assert len(d.recourse.steps) == smallest


def test_recourse_takes_the_cheapest_sufficient_bin(world) -> None:
    """Advice should be the smallest change that works, not the largest."""
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising", "falling"],
    )
    sc = world["scorecard"]
    for d in engine.decide_batch(world["X"].head(400)):
        if not (d.is_decline and d.recourse.feasible and len(d.recourse.steps) == 1):
            continue
        step = d.recourse.steps[0]
        bins = sc.points_by_bin(step.characteristic, intervals_only=True)
        current = bins[step.from_bin]
        # No cheaper bin would also have cleared the shortfall.
        cheaper = [
            pts
            for pts in bins.values()
            if current < pts < bins[step.to_bin] and pts - current >= d.recourse.points_needed
        ]
        assert not cheaper, f"a cheaper sufficient bin existed for {step.characteristic}"


def test_recourse_never_advises_moving_to_a_missing_bin(world) -> None:
    """ "Withhold this value" is not advice."""
    engine = DecisionEngine(
        world["scorecard"],
        world["conformal"],
        world["cutoff"],
        actionable_features=["rising", "falling"],
    )
    for d in engine.decide_batch(world["X"].head(400)):
        if d.is_decline and d.recourse.steps:
            for step in d.recourse.steps:
                assert step.to_bin not in {"Missing", "Special"}


# --- Referral economics --------------------------------------------------


def test_referral_benefit_depends_on_what_the_cutoff_would_have_done() -> None:
    """Hand-computed, four referred cases covering every quadrant.

    assumptions: roi 0.2644, success 0.55, default 0.10
      E[loss] = 0.10(1) + 0.35(0.5) = 0.275

      auto-approved + defaults -> avoided  0.275
      auto-approved + repays   -> avoided  0
      auto-declined + repays   -> recovered 0.2644
      auto-declined + defaults -> avoided  0
      total benefit = 0.275 + 0.2644 = 0.5394
    """
    assumptions = ProfitAssumptions(success_rate=0.55, default_rate=0.10, roi=0.2644)
    cutoff = Cutoff(0.5, 0.5, 0.0, 0.0, 0.0, 0.5, assumptions)  # rejection rate 0.5

    y = np.array([1, 0, 0, 1])
    auto = np.array([True, True, False, False])
    bands = np.array([Band.REFER] * 4)

    out = referral_economics(y, bands, auto, cutoff, review_costs=[0.0])
    assert expected_loss_fraction(assumptions) == pytest.approx(0.275)
    assert out["loss_avoided"][0] == pytest.approx(0.5394)
    assert out["share_referred"][0] == pytest.approx(1.0)
    assert out["n_referred"][0] == 4


def test_only_referred_cases_contribute_benefit() -> None:
    assumptions = ProfitAssumptions(0.55, 0.10, 0.2644)
    cutoff = Cutoff(0.5, 0.5, 0.0, 0.0, 0.0, 0.5, assumptions)  # rejection rate 0.5
    y = np.array([1, 1])
    auto = np.array([True, True])
    # Only the first is referred, so only its avoided loss counts.
    out = referral_economics(
        y, np.array([Band.REFER, Band.APPROVE]), auto, cutoff, review_costs=[0.0]
    )
    assert out["loss_avoided"][0] == pytest.approx(0.275)
    assert out["share_referred"][0] == pytest.approx(0.5)


def test_net_benefit_falls_as_review_cost_rises(world) -> None:
    engine = DecisionEngine(world["scorecard"], world["conformal"], world["cutoff"])
    X, y = world["X"], world["y"]
    out = referral_economics(
        y,
        engine.bands(X),
        engine.auto_approve(X),
        world["cutoff"],
        review_costs=[0.0, 0.005, 0.01, 0.02, 0.05],
    )
    assert np.all(np.diff(out["net_benefit"]) < 0)
    assert out["net_benefit"][0] == pytest.approx(out["loss_avoided"][0])
    # Somewhere in the configured sweep the band stops paying for itself.
    assert out["net_benefit"][0] > 0


def test_review_accuracy_scales_the_benefit() -> None:
    assumptions = ProfitAssumptions(0.55, 0.10, 0.2644)
    cutoff = Cutoff(0.5, 0.5, 0.0, 0.0, 0.0, 0.5, assumptions)  # rejection rate 0.5
    y, auto, bands = np.array([1]), np.array([True]), np.array([Band.REFER])
    full = referral_economics(y, bands, auto, cutoff, review_costs=[0.0])
    half = referral_economics(y, bands, auto, cutoff, review_costs=[0.0], review_accuracy=0.5)
    assert half["loss_avoided"][0] == pytest.approx(0.5 * full["loss_avoided"][0])


def test_referral_economics_validates_its_inputs() -> None:
    assumptions = ProfitAssumptions()
    cutoff = Cutoff(0.5, 0.5, 0.0, 0.0, 0.0, 0.5, assumptions)  # rejection rate 0.5
    with pytest.raises(ValueError, match="Length mismatch"):
        referral_economics(
            np.array([1, 0]),
            np.array([Band.REFER]),
            np.array([True]),
            cutoff,
            review_costs=[0.0],
        )
    with pytest.raises(ValueError, match="review_accuracy"):
        referral_economics(
            np.array([1]),
            np.array([Band.REFER]),
            np.array([True]),
            cutoff,
            review_costs=[0.0],
            review_accuracy=1.5,
        )
