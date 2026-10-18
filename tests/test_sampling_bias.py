"""The sampling-bias experiment: windows, the old policy, the arms, the measures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config import load_config
from profit import ProfitAssumptions, expected_loss_fraction
from sampling_bias import (
    ARMS,
    DRAWING_ARMS,
    SamplingBiasParams,
    arm_specs,
    build_arm,
    evaluation_bias,
    exploration_budget_for_detection,
    exploration_cost,
    exploration_economics,
    fit_plan,
    make_scenario,
    recovery_share,
    select_features,
    window_weeks,
)


@pytest.fixture
def scenario():
    """1,000 development cases with a 30% reject rate under a real policy."""
    rng = np.random.default_rng(0)
    n = 1_000
    y = rng.binomial(1, 0.15, n).astype(float)
    X = pd.DataFrame({"a": rng.normal(0, 1, n) + y, "b": rng.normal(0, 1, n)})
    # The old policy ranks defaulters higher, imperfectly.
    policy_pd = np.clip(rng.normal(0.3, 0.12, n) + 0.25 * y, 0.001, 0.999)
    return make_scenario(X, y, policy_pd, reject_rate=30)


@pytest.fixture
def arm_kwargs(scenario):
    """Inputs an arm may legitimately receive, all derived from A1, not truth."""
    rng = np.random.default_rng(1)
    n_rej = scenario.n_rejected
    return {
        "reject_pd": np.clip(scenario.policy_pd[scenario.rejected], 0.01, 0.99),
        "accepted_pd": np.clip(scenario.policy_pd[scenario.accepted], 0.01, 0.99),
        "uplift": 1.5,
        "rng": rng,
        "confidence": 0.05,
        "iterations": 5,
        "explore_pct": 5,
        "_n_rej": n_rej,
    }


# --- Config and windows --------------------------------------------------


def test_params_load_from_config() -> None:
    p = SamplingBiasParams.from_config(load_config()["sampling_bias"])
    assert p.reject_rates == (10, 20, 30, 50)
    assert p.n_features == 40
    assert p.seeds == (0, 1, 2)
    assert p.exploration_pct == (1, 2, 5)


def test_windows_are_contiguous_ordered_and_complete() -> None:
    windows = load_config().sampling_bias_windows
    blocks = window_weeks(range(100), windows)

    assert list(blocks) == ["policy", "dev", "test"]
    assigned = [w for name in blocks for w in blocks[name]]
    assert sorted(assigned) == list(range(100))
    assert blocks["policy"][-1] < blocks["dev"][0] < blocks["test"][0]
    assert abs(len(blocks["policy"]) / 100 - windows.policy_frac) < 0.02


def test_windows_are_independent_of_the_model_splits() -> None:
    """They answer different questions and must not drift together."""
    cfg = load_config()
    sb = window_weeks(range(100), cfg.sampling_bias_windows)
    assert len(sb["policy"]) != len(window_weeks(range(100), cfg.sampling_bias_windows)["dev"])
    # The main split's train block is a different size than the policy window.
    assert cfg.splits.train_frac != cfg.sampling_bias_windows.policy_frac


# --- The old policy ------------------------------------------------------


def test_the_policy_rejects_exactly_the_worst_share(scenario) -> None:
    assert scenario.n_rejected == 300
    # Every rejected case scores worse than every accepted one.
    assert (
        scenario.policy_pd[scenario.rejected].min() >= scenario.policy_pd[scenario.accepted].max()
    )


def test_rejected_labels_are_hidden(scenario) -> None:
    assert np.isnan(scenario.y_observed[scenario.rejected]).all()
    assert not np.isnan(scenario.y_observed[scenario.accepted]).any()


def test_the_accepted_population_is_cleaner_than_the_whole(scenario) -> None:
    """The bias the experiment exists to measure, present in the fixture."""
    accepted_rate = scenario.y_observed[scenario.accepted].mean()
    true_rate = scenario.oracle_labels().mean()
    assert accepted_rate < true_rate


def test_an_impossible_reject_rate_is_refused() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="reject_rate"):
        make_scenario(X, np.array([0.0, 1.0]), np.array([0.1, 0.2]), reject_rate=100)


# --- Leakage: the load-bearing property ----------------------------------


def test_arms_that_must_not_see_hidden_labels_do_not(scenario, arm_kwargs) -> None:
    """Scramble the rejects' true labels; A1-A5 must be bit-identical.

    This is the experiment's integrity check. If a reject-inference arm could
    see the outcomes it is supposed to be inferring, every recovery number would
    be meaningless, and it would look like a success, not a failure.
    """
    y_scrambled = scenario.oracle_labels().copy()
    rejected = scenario.rejected
    # Flip every hidden label. The accepted ones are untouched.
    y_scrambled[rejected] = 1.0 - y_scrambled[rejected]
    other = make_scenario(scenario.X, y_scrambled, scenario.policy_pd, reject_rate=30)
    assert not np.array_equal(scenario.oracle_labels()[rejected], other.oracle_labels()[rejected])

    blind_arms = [
        ("A1_accepts_only", {}),
        ("A2_hard_cutoff", {}),
        ("A3_fuzzy", {"reject_pd": arm_kwargs["reject_pd"]}),
        (
            "A4_parcelling",
            {
                "reject_pd": arm_kwargs["reject_pd"],
                "accepted_pd": arm_kwargs["accepted_pd"],
                "uplift": 1.5,
            },
        ),
        (
            "A5_self_learning",
            {"reject_pd": arm_kwargs["reject_pd"], "confidence": 0.05, "iterations": 5},
        ),
    ]
    for name, kwargs in blind_arms:
        a = build_arm(name, scenario, rng=np.random.default_rng(7), **kwargs)
        b = build_arm(name, other, rng=np.random.default_rng(7), **kwargs)
        assert np.array_equal(a.y, b.y), f"{name} changed when hidden labels changed"
        pd.testing.assert_frame_equal(a.X.reset_index(drop=True), b.X.reset_index(drop=True))
        if a.weight is None:
            assert b.weight is None
        else:
            assert np.allclose(a.weight, b.weight), f"{name} weights leaked"


def test_the_oracle_does_change_with_the_hidden_labels(scenario) -> None:
    """The converse: if A0 were also unchanged, the test above proves nothing."""
    y_scrambled = scenario.oracle_labels().copy()
    y_scrambled[scenario.rejected] = 1.0 - y_scrambled[scenario.rejected]
    other = make_scenario(scenario.X, y_scrambled, scenario.policy_pd, reject_rate=30)

    a = build_arm("A0_oracle", scenario)
    b = build_arm("A0_oracle", other)
    assert not np.array_equal(a.y, b.y)


def test_explore_returns_nothing_outside_the_mask(scenario) -> None:
    mask = np.zeros(len(scenario.X), dtype=bool)
    mask[np.flatnonzero(scenario.rejected)[:10]] = True
    observed = scenario.explore(mask)
    assert np.isnan(observed[~mask]).all()
    assert not np.isnan(observed[mask]).any()


# --- The arms ------------------------------------------------------------


def test_a0_uses_every_label(scenario) -> None:
    out = build_arm("A0_oracle", scenario)
    assert out.y.size == len(scenario.X)
    assert np.array_equal(out.y, scenario.oracle_labels())


def test_a1_uses_only_approved_cases(scenario) -> None:
    out = build_arm("A1_accepts_only", scenario)
    assert out.y.size == int(scenario.accepted.sum()) == 700
    assert not np.isnan(out.y).any()


def test_a2_labels_every_reject_bad(scenario) -> None:
    out = build_arm("A2_hard_cutoff", scenario)
    assert out.y.size == len(scenario.X)
    assert (out.y[scenario.rejected] == 1.0).all()
    # Accepted labels are untouched.
    assert np.array_equal(out.y[scenario.accepted], scenario.y_observed[scenario.accepted])


def test_a3_duplicates_rejects_with_complementary_weights(scenario, arm_kwargs) -> None:
    out = build_arm("A3_fuzzy", scenario, reject_pd=arm_kwargs["reject_pd"])
    n_acc, n_rej = int(scenario.accepted.sum()), scenario.n_rejected
    assert len(out.X) == n_acc + 2 * n_rej
    # Each reject contributes exactly 1.0 of total weight across its two copies.
    bad_w = out.weight[n_acc : n_acc + n_rej]
    good_w = out.weight[n_acc + n_rej :]
    assert np.allclose(bad_w + good_w, 1.0)
    assert np.allclose(out.weight[:n_acc], 1.0)


def test_a4_parcelling_uplift_increases_the_bad_rate(scenario, arm_kwargs) -> None:
    common = {
        "reject_pd": arm_kwargs["reject_pd"],
        "accepted_pd": arm_kwargs["accepted_pd"],
    }
    low = build_arm("A4_parcelling", scenario, uplift=1.0, rng=np.random.default_rng(3), **common)
    high = build_arm("A4_parcelling", scenario, uplift=2.0, rng=np.random.default_rng(3), **common)
    rejected = scenario.rejected
    assert high.y[rejected].mean() > low.y[rejected].mean()
    assert "k2" in high.arm


def test_a5_admits_only_confidently_scored_rejects(scenario, arm_kwargs) -> None:
    few = build_arm(
        "A5_self_learning",
        scenario,
        reject_pd=arm_kwargs["reject_pd"],
        confidence=0.01,
        iterations=1,
    )
    many = build_arm(
        "A5_self_learning",
        scenario,
        reject_pd=arm_kwargs["reject_pd"],
        confidence=0.20,
        iterations=5,
    )
    n_acc = int(scenario.accepted.sum())
    assert len(few.X) >= n_acc
    assert len(many.X) > len(few.X), "widening the band admitted no extra rejects"
    assert len(many.X) <= len(scenario.X)


def test_a6_explores_the_requested_share_with_the_right_weight(scenario) -> None:
    out = build_arm("A6_exploration", scenario, explore_pct=5, rng=np.random.default_rng(2))
    n_acc = int(scenario.accepted.sum())
    n_explored = int(round(scenario.n_rejected * 0.05))
    assert len(out.X) == n_acc + n_explored
    # Weight 100/p = 20, so the explored slice stands for the whole reject pool.
    assert np.allclose(out.weight[n_acc:], 20.0)
    assert np.allclose(out.weight[:n_acc], 1.0)
    assert np.isclose(out.weight[n_acc:].sum(), n_explored * 20.0)


def test_a6_rejects_an_impossible_budget(scenario) -> None:
    with pytest.raises(ValueError, match="explore_pct"):
        build_arm("A6_exploration", scenario, explore_pct=0, rng=np.random.default_rng(0))


def test_every_arm_produces_binary_labels(scenario, arm_kwargs) -> None:
    specs = [
        ("A0_oracle", {}),
        ("A1_accepts_only", {}),
        ("A2_hard_cutoff", {}),
        ("A3_fuzzy", {"reject_pd": arm_kwargs["reject_pd"]}),
        (
            "A4_parcelling",
            {
                "reject_pd": arm_kwargs["reject_pd"],
                "accepted_pd": arm_kwargs["accepted_pd"],
                "uplift": 1.5,
            },
        ),
        (
            "A5_self_learning",
            {"reject_pd": arm_kwargs["reject_pd"], "confidence": 0.05, "iterations": 5},
        ),
        ("A6_exploration", {"explore_pct": 5}),
    ]
    for name, kwargs in specs:
        out = build_arm(name, scenario, rng=np.random.default_rng(0), **kwargs)
        assert set(np.unique(out.y)) <= {0.0, 1.0}, name
        assert not np.isnan(out.y).any(), name
        assert len(out.X) == out.y.size, name


def test_an_unknown_arm_is_refused(scenario) -> None:
    with pytest.raises(ValueError, match="Unknown arm"):
        build_arm("A9_wishful", scenario)


# --- Measurement ---------------------------------------------------------


def test_recovery_share_anchors_at_the_two_endpoints() -> None:
    # Matching the oracle is full recovery; matching accepts-only is none.
    assert recovery_share(0.60, 0.40, 0.60) == pytest.approx(1.0)
    assert recovery_share(0.60, 0.40, 0.40) == pytest.approx(0.0)
    assert recovery_share(0.60, 0.40, 0.50) == pytest.approx(0.5)


def test_recovery_share_reports_harm_rather_than_clipping() -> None:
    """An arm worse than doing nothing is a real and common outcome."""
    assert recovery_share(0.60, 0.40, 0.30) == pytest.approx(-0.5)


def test_recovery_share_is_nan_when_there_is_no_gap() -> None:
    # A ratio against a near-zero denominator is amplified noise, not a result.
    assert math.isnan(recovery_share(0.5, 0.5, 0.7))
    assert math.isnan(recovery_share(0.505, 0.5, 0.7)), "gap below the floor"


def test_recovery_share_refuses_a_negative_gap() -> None:
    """Hiding the rejects' labels sometimes makes the model BETTER.

    Measured on the full data for the scorecard arms: A0 oracle 0.407 against
    A1 accepts-only 0.434. The denominator is then negative, so an arm far worse
    than both scores a large POSITIVE recovery, the catastrophic A5 arm came
    out at +2320%. There is no gap to recover and the honest answer is NaN.
    """
    oracle, accepts_only = 0.4071, 0.4338
    assert oracle - accepts_only < 0
    assert math.isnan(recovery_share(oracle, accepts_only, -0.1867))
    assert math.isnan(recovery_share(oracle, accepts_only, 0.45))


def test_a5_pseudo_labels_are_scale_free(scenario) -> None:
    """Confidence is a quantile, not an absolute probability.

    At a 3% base rate no predicted PD approaches 0.5, so an absolute rule admits
    rejects and labels every one of them GOOD, the opposite of A2. Measured
    before the fix at Gini 0.02 (LightGBM) and -0.19 (scorecard).
    """
    # PDs that never cross 0.5, as a real low-default-rate model produces.
    reject_pd = np.linspace(0.01, 0.30, scenario.n_rejected)
    out = build_arm(
        "A5_self_learning", scenario,
        reject_pd=reject_pd, confidence=0.10, iterations=2,
    )
    n_acc = int(scenario.accepted.sum())
    pseudo = out.y[n_acc:]
    assert pseudo.size > 0, "no rejects admitted at all"
    assert set(np.unique(pseudo)) == {0.0, 1.0}, (
        f"pseudo-labels are all one class ({np.unique(pseudo)}); the rule is "
        "not scale-free"
    )
    # Roughly balanced: the top and bottom quantiles are symmetric.
    assert 0.3 < pseudo.mean() < 0.7


def test_evaluation_bias_reports_the_direction_and_size() -> None:
    # A lender measuring only its approved book sees a FLATTERING number.
    out = evaluation_bias(metric_on_accepted=0.55, metric_on_full=0.70)
    assert out["bias"] == pytest.approx(-0.15)
    assert out["relative_bias"] == pytest.approx(-0.15 / 0.70)
    out_up = evaluation_bias(0.80, 0.70)
    assert out_up["bias"] == pytest.approx(0.10)


def test_exploration_cost_counts_losses_only() -> None:
    """ROI on explored loans that repaid is not netted off.

    Those applicants were rejected by policy and would never have been booked,
    so their return is not revenue the lender was otherwise forgoing. Counting
    it would make exploration look free or even profitable.
    """
    a = ProfitAssumptions(success_rate=0.55, default_rate=0.10, roi=0.2644)
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    # 2 defaults x E[loss] 0.275 = 0.55; the two repayments contribute nothing.
    assert expected_loss_fraction(a) == pytest.approx(0.275)
    assert exploration_cost(labels, a) == pytest.approx(0.55)
    assert exploration_cost(np.array([0.0, 0.0]), a) == pytest.approx(0.0)


def test_exploration_cost_scales_with_exposure() -> None:
    a = ProfitAssumptions()
    labels = np.array([1.0, 1.0])
    base = exploration_cost(labels, a)
    doubled = exploration_cost(labels, a, exposure=np.full(2, 2.0))
    assert doubled == pytest.approx(2 * base)


def test_exploration_cost_ignores_unobserved_entries() -> None:
    a = ProfitAssumptions()
    assert exploration_cost(np.array([np.nan, 1.0, np.nan]), a) == pytest.approx(0.275)


# --- Feature selection ---------------------------------------------------


def test_features_combine_iv_and_gain_rankings() -> None:
    iv = {"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.1}
    gain = {"a": 10.0, "b": 1.0, "c": 9.0, "d": 0.5}
    # a is top on both; c beats b on combined rank (0+1 vs 1+2... c: 2+1=3, b: 1+2=3)
    picked = select_features(iv, gain, n_features=2)
    assert picked[0] == "a"
    assert len(picked) == 2
    assert "d" not in picked


def test_feature_selection_needs_overlapping_rankings() -> None:
    with pytest.raises(ValueError, match="share no features"):
        select_features({"a": 1.0}, {"b": 1.0}, n_features=5)


def test_feature_selection_caps_at_the_requested_count() -> None:
    iv = {f"f{i}": float(100 - i) for i in range(60)}
    gain = {f"f{i}": float(i) for i in range(60)}
    assert len(select_features(iv, gain, n_features=40)) == 40


# --- Sizing an exploration budget for measurement -------------------------


def test_budget_scales_with_the_square_of_the_precision_wanted() -> None:
    """Binomial sampling: halving the tolerance costs four times the loans."""
    # tolerance is half the margin, so halving it needs half the margin:
    # (0.0176 - 0.010) / 2 = 0.0038 against (0.0138 - 0.010) / 2 = 0.0019.
    coarse = exploration_budget_for_detection(0.07, 0.3, 0.0176, 0.010)
    fine = exploration_budget_for_detection(0.07, 0.3, 0.0138, 0.010)
    assert fine["tolerance"] == pytest.approx(coarse["tolerance"] / 2, rel=1e-6)
    assert fine["n_explored"] == pytest.approx(4 * coarse["n_explored"], rel=1e-6)


def test_a_smaller_rejected_share_needs_a_smaller_budget() -> None:
    """The rejects enter the population mean weighted by their share.

    A policy that rejects little has a slice that barely moves the population
    number, so a looser estimate of it suffices and fewer loans are needed.
    Which is the right way round: the measurement problem is worst exactly
    where the policy rejects most, and that is where the gap is largest too.
    """
    big = exploration_budget_for_detection(0.07, 0.50, 0.0176, 0.010)
    small = exploration_budget_for_detection(0.07, 0.10, 0.0176, 0.010)
    assert small["n_explored"] < big["n_explored"]
    # Quadratic in the share, for the same reason it is quadratic in tolerance.
    assert big["n_explored"] == pytest.approx(25 * small["n_explored"], rel=1e-6)


def test_cost_is_the_realised_loss_on_the_explored_slice() -> None:
    a = ProfitAssumptions(success_rate=0.5, default_rate=0.2, roi=0.3)
    out = exploration_budget_for_detection(0.08, 0.4, 0.02, 0.01, a)
    assert out["expected_cost"] == pytest.approx(
        out["n_explored"] * 0.08 * expected_loss_fraction(a)
    )


def test_the_share_of_rejects_is_only_reported_when_the_count_is_known() -> None:
    assert "share_of_rejects" not in exploration_budget_for_detection(0.07, 0.3, 0.02, 0.01)
    sized = exploration_budget_for_detection(0.07, 0.3, 0.02, 0.01, n_rejected=10_000)
    assert sized["share_of_rejects"] == pytest.approx(sized["n_explored"] / 10_000)


def test_a_gap_inside_the_threshold_is_rejected() -> None:
    """Nothing to size for: the monitor would not fire on it either way."""
    with pytest.raises(ValueError, match="must exceed the threshold"):
        exploration_budget_for_detection(0.07, 0.3, 0.005, 0.010)


def test_budget_sizing_validates_its_rates() -> None:
    with pytest.raises(ValueError, match="bad_rate_rejected"):
        exploration_budget_for_detection(0.0, 0.3, 0.02, 0.01)
    with pytest.raises(ValueError, match="rejected_share"):
        exploration_budget_for_detection(0.07, 0.0, 0.02, 0.01)


# --- the fit plan ---------------------------------------------------------
#
# This lived inside the Kaggle stage script, where nothing ran it, and it was
# wrong for the whole project: the five arms that build their training set
# deterministically were fitted once each, as though that made their RESULT
# deterministic. It does not. They are then fitted with LightGBM, which bags
# rows by position, so their Ginis were single draws and every recovery share
# built on them carried unmeasured noise.


def _specs(params: SamplingBiasParams | None = None):
    params = params or SamplingBiasParams()
    return params, arm_specs(params, reject_pd=None, accepted_pd=None)


def test_every_arm_in_the_registry_is_actually_run() -> None:
    """An arm that exists and is never specced is an arm the write-up claims."""
    _, specs = _specs()
    assert {name for name, _, _ in specs} == set(ARMS)


def test_the_drawing_flag_matches_the_arms_that_draw() -> None:
    _, specs = _specs()
    assert {name for name, _, draws in specs if draws} == set(DRAWING_ARMS)
    # A4 and A6, and nothing else, take an rng-dependent decision.
    assert set(DRAWING_ARMS) == {"A4_parcelling", "A6_exploration"}


def test_every_arm_is_fitted_at_every_seed_with_lightgbm() -> None:
    """The correction. No arm may be a single LightGBM draw again."""
    params, specs = _specs()
    plan = fit_plan(params, specs)
    for name, _, _ in specs:
        seeds = {
            seed for arm, _, seed, models in plan if arm == name and "lightgbm" in models
        }
        assert seeds == set(params.seeds), (
            f"{name} is fitted with LightGBM at {sorted(seeds)}, not every seed; "
            "its Gini would be a single draw of a distribution nothing measures"
        )


def test_the_scorecard_is_not_refitted_where_it_cannot_change() -> None:
    """fit_scorecard takes no seed, so repeats would only pad the output."""
    from collections import Counter

    params, specs = _specs()
    plan = fit_plan(params, specs)
    # A4 and A6 contribute several specs each, one per uplift and per budget.
    variants = Counter(name for name, _, _ in specs)
    draws_by_name = {name: draws for name, _, draws in specs}
    for name, n_variants in variants.items():
        fits = sum(1 for arm, _, _, models in plan if arm == name and "scorecard" in models)
        per_variant = len(params.seeds) if draws_by_name[name] else 1
        assert fits == n_variants * per_variant, (
            f"{name} gets {fits} scorecard fits across {n_variants} variant(s), "
            f"expected {n_variants * per_variant}"
        )


def test_the_plan_is_the_size_the_readme_reports() -> None:
    """The run time and the row count in the documents follow from this."""
    params, specs = _specs()
    plan = fit_plan(params, specs)
    per_rate = sum(len(models) for *_, models in plan)
    assert per_rate == 50, f"{per_rate} fits per reject rate"
    assert per_rate * len(params.reject_rates) == 200


def test_every_spec_passes_kwargs_its_arm_accepts() -> None:
    """A typo in a spec's kwargs only shows up as a failed arm on Kaggle.

    The stage catches arm exceptions so one degenerate arm cannot kill the
    grid, which also means a misspelled keyword would be reported as a FAILED
    line in a log and the arm would silently vanish from the results.
    """
    import inspect

    _, specs = _specs()
    for name, kwargs, _ in specs:
        signature = inspect.signature(ARMS[name])
        takes_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        for key in kwargs:
            assert key in signature.parameters or takes_kwargs, (
                f"{name} is specced with {key!r}, which {ARMS[name].__name__} "
                "does not accept"
            )


def test_a_shorter_seed_tuple_still_plans_one_scorecard_fit() -> None:
    """Config may cut seeds first if a kernel runs long, per the brief."""
    params = SamplingBiasParams(seeds=(7,))
    plan = fit_plan(params, arm_specs(params, reject_pd=None, accepted_pd=None))
    assert {seed for _, _, seed, _ in plan} == {7}
    assert sum(len(models) for *_, models in plan) == 20


# --- exploration economics --------------------------------------------------


def _arms_frame(a1_lightgbm_profits: list[float]) -> pd.DataFrame:
    """A1 at one or several LightGBM seeds, the scorecard once, A6 at p=1."""
    rows = [
        {"reject_rate": 30, "arm": "A1_accepts_only", "base_arm": "A1_accepts_only",
         "model": "lightgbm", "seed": seed, "profit_full": profit}
        for seed, profit in enumerate(a1_lightgbm_profits)
    ]
    rows.append({"reject_rate": 30, "arm": "A1_accepts_only", "base_arm": "A1_accepts_only",
                 "model": "scorecard", "seed": 0, "profit_full": 50.0})
    for seed, profit in enumerate([90.0, 95.0, 100.0]):
        for model, offset in (("lightgbm", 0.0), ("scorecard", -40.0)):
            rows.append({"reject_rate": 30, "arm": "A6_exploration_p1",
                         "base_arm": "A6_exploration", "model": model, "seed": seed,
                         "profit_full": profit + offset})
    return pd.DataFrame(rows)


def _explore_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"reject_rate": [30, 30, 30], "explore_pct": [1, 1, 1], "seed": [0, 1, 2],
         "realised_loss_cost": [10.0, 10.0, 10.0]}
    )


def test_exploration_gain_survives_a_baseline_fitted_at_several_seeds() -> None:
    """The case that killed a full Kaggle run after the grid had finished.

    With A1 at three LightGBM seeds, a baseline indexed by reject rate alone
    returned a Series, every gain became one, and the next mean raised.
    """
    out = exploration_economics(_arms_frame([100.0, 110.0, 120.0]), _explore_frame())
    gains = out["profit_gain_vs_A1_lightgbm"]
    assert gains.dtype == float, f"gains are {gains.dtype}, not plain floats"
    # Against the MEAN baseline of 110: 90-110, 95-110, 100-110.
    assert gains.tolist() == [-20.0, -15.0, -10.0]
    assert out["net_vs_A1_lightgbm"].tolist() == [-30.0, -25.0, -20.0]
    out.groupby(["reject_rate", "explore_pct"])["net_vs_A1_lightgbm"].mean()


def test_a_single_seed_baseline_gives_the_same_answer_as_before() -> None:
    """The scorecard still runs A1 once; that path must be unchanged."""
    out = exploration_economics(_arms_frame([110.0]), _explore_frame())
    assert out["profit_gain_vs_A1_scorecard"].tolist() == [0.0, 5.0, 10.0]


def test_a_duplicated_a6_fit_is_an_error_not_a_silent_series() -> None:
    arms = _arms_frame([100.0])
    dup = arms[arms["base_arm"] == "A6_exploration"].iloc[[0]]
    with pytest.raises(ValueError, match="more than one fit"):
        exploration_economics(pd.concat([arms, dup]), _explore_frame())
