"""The EMPCS label-encoding guard.

The brief warns that empulse needs the default class encoded as 0, the opposite
of `target`. For the pinned empulse (0.13.0) that is no longer true: the library
does the inversion internally and documents that it expects 'default': 1.

These tests pin the behaviour down empirically, so that the day a new empulse
flips the convention back, CI says so instead of the project silently reporting
a profit number that is pure artefact.
"""

from __future__ import annotations

import numpy as np
import pytest
from empulse.metrics import empcs_score

from profit import (
    ProfitAssumptions,
    empcs,
    empcs_sensitivity,
    expected_loss_fraction,
    strategy_curve,
)


@pytest.fixture(scope="module")
def portfolio() -> dict[str, np.ndarray]:
    """A portfolio with an informative PD model and a worthless one."""
    rng = np.random.default_rng(0)
    n = 4_000
    # 1 = default, exactly as the Home Credit base table encodes `target`.
    y = rng.binomial(1, 0.05, n)
    informative = np.clip(rng.normal(0.5, 0.15, n) + 0.30 * y, 0.001, 0.999)
    noise = rng.uniform(0.0, 1.0, n)
    return {"y": y, "informative": informative, "noise": noise}


def test_empcs_discriminates_with_target_passed_through_untouched(portfolio) -> None:
    """The load-bearing assertion: 1 = default, and better models score higher."""
    good = empcs(portfolio["y"], portfolio["informative"])
    bad = empcs(portfolio["y"], portfolio["noise"])
    assert good.emp > bad.emp, (
        "EMPCS failed to rank an informative model above noise with target passed "
        "through as-is. If this fails, re-check empulse's encoding convention "
        "before trusting any profit number in this repo."
    )


def test_inverting_the_target_makes_empcs_blind(portfolio) -> None:
    """Why the brief's 'invert the target' instruction must not be followed here.

    Inverted, EMPCS gives an informative model and pure noise the same value --
    and a value ~50x larger than the correct one, so the mistake flatters itself.
    """
    y_inverted = 1 - portfolio["y"]
    good = float(empcs_score(y_inverted, portfolio["informative"]))
    bad = float(empcs_score(y_inverted, portfolio["noise"]))

    assert good == pytest.approx(bad, abs=1e-4), (
        "Inverted labels now discriminate, so empulse's encoding convention has "
        "changed. Revisit profit.py and the note in config/default.toml."
    )
    correct = empcs(portfolio["y"], portfolio["informative"]).emp
    assert good > correct, "The inverted number is no longer the inflated one."


def test_empcs_is_invariant_to_monotone_rescaling_of_the_score(portfolio) -> None:
    # EMPCS depends on the ranking, not the scale, so a calibration change that
    # preserves order must not move it.
    a = empcs(portfolio["y"], portfolio["informative"]).emp
    b = empcs(portfolio["y"], portfolio["informative"] / 2.0).emp
    assert a == pytest.approx(b)


def test_empcs_returns_an_acceptance_fraction_in_the_unit_interval(portfolio) -> None:
    out = empcs(portfolio["y"], portfolio["informative"])
    assert 0.0 <= out.optimal_acceptance_rate <= 1.0


def test_non_binary_targets_are_rejected() -> None:
    with pytest.raises(ValueError, match="1 = default"):
        empcs(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


# --- Assumptions ---------------------------------------------------------


def test_impossible_assumptions_are_rejected() -> None:
    # success_rate + default_rate > 1 leaves negative mass for partial losses.
    with pytest.raises(ValueError, match="negative probability mass"):
        ProfitAssumptions(success_rate=0.7, default_rate=0.4)


def test_expected_loss_fraction_hand_computed() -> None:
    # default_rate 0.10 at a total loss, success_rate 0.55 at no loss,
    # the remaining 0.35 uniform on (0,1) with mean 0.5:
    #   0.10(1) + 0.35(0.5) = 0.10 + 0.175 = 0.275
    assert expected_loss_fraction(ProfitAssumptions(0.55, 0.10, 0.2644)) == pytest.approx(0.275)


def test_sensitivity_grid_skips_infeasible_corners(portfolio) -> None:
    grid = {
        "success_rate": [0.45, 0.65],
        "default_rate": [0.05, 0.50],  # 0.65 + 0.50 > 1 is infeasible
        "roi": [0.2644],
    }
    results = empcs_sensitivity(portfolio["y"], portfolio["informative"], grid)
    assert len(results) == 3
    for r in results:
        assert r.assumptions.success_rate + r.assumptions.default_rate <= 1.0


def test_sensitivity_grid_requires_every_key(portfolio) -> None:
    with pytest.raises(KeyError, match="missing"):
        empcs_sensitivity(portfolio["y"], portfolio["informative"], {"roi": [0.2]})


# --- Strategy curve ------------------------------------------------------


def test_strategy_curve_bad_rate_rises_as_approvals_loosen(portfolio) -> None:
    out = strategy_curve(portfolio["y"], portfolio["informative"])
    # Approving lowest-PD first, the realised bad rate among the approved must
    # trend upward as the cutoff loosens.
    tail = out["bad_rate"][out["n_approved"] >= 100]
    assert tail[-1] > tail[0]
    assert out["approval_rate"][0] == pytest.approx(0.0)
    assert out["approval_rate"][-1] == pytest.approx(1.0)


def test_strategy_curve_at_full_approval_matches_the_portfolio(portfolio) -> None:
    y = portfolio["y"]
    out = strategy_curve(y, portfolio["informative"])
    assert out["bad_rate"][-1] == pytest.approx(y.mean())
    # Total profit at 100% approval, computed directly from the assumptions.
    a = ProfitAssumptions()
    expected = (1 - y).sum() * a.roi - y.sum() * expected_loss_fraction(a)
    assert out["profit"][-1] == pytest.approx(expected)


def test_strategy_curve_honours_per_loan_exposure() -> None:
    # Two identical loans, one twice the size: doubling every exposure must
    # exactly double the profit.
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    base = strategy_curve(y, p)
    doubled = strategy_curve(y, p, exposure=np.full(4, 2.0))
    assert doubled["profit"][-1] == pytest.approx(2.0 * base["profit"][-1])


def test_strategy_curve_rejects_mismatched_exposure() -> None:
    with pytest.raises(ValueError, match="exposure has"):
        strategy_curve(np.array([0, 1]), np.array([0.1, 0.2]), exposure=np.ones(3))
