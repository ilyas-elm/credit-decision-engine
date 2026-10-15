"""The sampling-bias experiment: windows, the old policy, the arms, the measures."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from config import load_config
from profit import ProfitAssumptions, expected_loss_fraction
from sampling_bias import (
    SamplingBiasParams,
    build_arm,
    evaluation_bias,
    exploration_cost,
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
