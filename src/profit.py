"""Profit metrics: EMPCS and the strategy curve.

LABEL ENCODING -- the one thing to get right here.

Verbraken et al.'s EMPCS formula is written with the default class as 0, and the
project brief warns accordingly that `target` must be inverted. For the pinned
empulse (0.13.0) that warning is out of date: `empcs_score` performs that
inversion internally and documents that it "assumes the standard notation
('default': 1, 'no default': 0)" -- exactly how the Home Credit base table
encodes `target`.

Verified empirically in tests/test_profit.py, on a model that is informative by
construction:

    empcs_score(target,     pd)  ->  0.004854   informative
    empcs_score(target,   rand)  ->  0.000000   noise, correctly scored worse
    empcs_score(1 - target, pd)  ->  0.255726   informative
    empcs_score(1 - target, rand) -> 0.255730   noise, scored THE SAME

Inverting makes the metric blind -- noise and signal land on the same value --
and returns a number ~50x larger, so the mistake looks like a triumph. Hence:
pass `target` through untouched, and let the test fail loudly if a future
empulse flips the convention back.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from empulse.metrics import empcs_score

__all__ = [
    "ProfitAssumptions",
    "EmpcsResult",
    "empcs",
    "empcs_sensitivity",
    "expected_loss_fraction",
    "strategy_curve",
]


@dataclass(frozen=True)
class ProfitAssumptions:
    """empulse's EMPCS parameters.

    success_rate : probability a defaulted loan is nonetheless repaid in full
    default_rate : probability the entire loan is lost
    roi          : return on investment on a repaid loan

    The remaining mass, ``1 - success_rate - default_rate``, is a uniform
    distribution over partial losses.
    """

    success_rate: float = 0.55
    default_rate: float = 0.10
    roi: float = 0.2644

    def __post_init__(self) -> None:
        for name in ("success_rate", "default_rate", "roi"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value!r}.")
        if self.success_rate + self.default_rate > 1.0:
            raise ValueError(
                f"success_rate + default_rate must not exceed 1, got "
                f"{self.success_rate} + {self.default_rate} = "
                f"{self.success_rate + self.default_rate}. There would be negative "
                "probability mass left for partial losses."
            )

    def as_kwargs(self) -> dict[str, float]:
        return {
            "success_rate": self.success_rate,
            "default_rate": self.default_rate,
            "roi": self.roi,
        }


@dataclass(frozen=True)
class EmpcsResult:
    """EMPCS and the acceptance fraction that achieves it."""

    emp: float
    optimal_acceptance_rate: float
    assumptions: ProfitAssumptions


def _check_target_encoding(y_true: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    unique = np.unique(y_true[~np.isnan(y_true.astype(float))])
    unknown = set(unique.tolist()) - {0, 1}
    if unknown:
        raise ValueError(
            f"y_true must be 0/1 with 1 = default (the Home Credit `target` convention); "
            f"found unexpected values {sorted(unknown)}."
        )
    return y_true


def empcs(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    assumptions: ProfitAssumptions | None = None,
) -> EmpcsResult:
    """EMPCS and the profit-maximising acceptance fraction.

    ``y_true`` is passed to empulse untouched: 1 = default. See the module
    docstring for why inverting it is wrong for this version.
    """
    assumptions = assumptions or ProfitAssumptions()
    y_true = _check_target_encoding(y_true)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    if y_true.size != y_pred_proba.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_score={y_pred_proba.size}.")

    kwargs = assumptions.as_kwargs()
    return EmpcsResult(
        emp=float(empcs_score(y_true, y_pred_proba, **kwargs)),
        optimal_acceptance_rate=float(empcs_score.optimal_rate(y_true, y_pred_proba, **kwargs)),
        assumptions=assumptions,
    )


def empcs_sensitivity(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    grid: dict[str, list[float]],
) -> list[EmpcsResult]:
    """EMPCS across the full cross-product of the assumption grid.

    success_rate, default_rate and roi are assumptions, not measurements, so a
    single EMPCS number is not reportable on its own. Infeasible combinations
    (success_rate + default_rate > 1) are skipped rather than clipped.
    """
    keys = ("success_rate", "default_rate", "roi")
    missing = [key for key in keys if key not in grid]
    if missing:
        raise KeyError(f"Sensitivity grid is missing {missing}; needs all of {list(keys)}.")

    results = []
    for success_rate, default_rate, roi in product(*(grid[key] for key in keys)):
        if success_rate + default_rate > 1.0:
            continue
        assumptions = ProfitAssumptions(
            success_rate=success_rate, default_rate=default_rate, roi=roi
        )
        results.append(empcs(y_true, y_pred_proba, assumptions))
    return results


def expected_loss_fraction(assumptions: ProfitAssumptions) -> float:
    """Expected fraction of principal lost on a defaulted loan.

    Point mass ``default_rate`` at a total loss, point mass ``success_rate`` at
    no loss, and the rest uniform on (0, 1) with mean 1/2 -- the same mixture
    EMPCS integrates over.
    """
    partial = 1.0 - assumptions.default_rate - assumptions.success_rate
    return assumptions.default_rate * 1.0 + partial * 0.5


def strategy_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    assumptions: ProfitAssumptions | None = None,
    *,
    n_points: int = 101,
    exposure: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Approval rate against bad rate and profit.

    Applicants are approved lowest-PD first. At each approval rate the curve
    reports the bad rate among those approved and the total profit, under

        profit per approved loan = roi * exposure          if it repays
                                 = -E[loss] * exposure     if it defaults

    ``exposure`` defaults to 1.0 per loan (unit exposure). The brief makes that
    the fallback when the data has no credit-amount column; whether a real one
    exists is settled in docs/DATA_NOTES.md, and passing it here switches the
    curve to monetary units without any other change.
    """
    assumptions = assumptions or ProfitAssumptions()
    y_true = _check_target_encoding(y_true).astype(float)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    if y_true.size != y_pred_proba.size:
        raise ValueError(f"Length mismatch: y_true={y_true.size}, y_score={y_pred_proba.size}.")
    n = y_true.size
    if n == 0:
        raise ValueError("strategy_curve needs at least one applicant.")

    exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=float).ravel()
    if exposure.size != n:
        raise ValueError(f"exposure has {exposure.size} rows, expected {n}.")

    order = np.argsort(y_pred_proba, kind="stable")
    bad, exp = y_true[order], exposure[order]
    loss = expected_loss_fraction(assumptions)
    per_loan = np.where(bad > 0.5, -loss * exp, assumptions.roi * exp)

    cum_bad = np.cumsum(bad)
    cum_profit = np.cumsum(per_loan)

    # Approval counts from 1..n on an even grid; rate 0 approves nobody.
    counts = np.unique(np.clip(np.round(np.linspace(0, n, n_points)).astype(int), 0, n))
    approval_rate = counts / n
    bad_rate = np.where(counts > 0, cum_bad[np.maximum(counts, 1) - 1] / np.maximum(counts, 1), 0.0)
    profit = np.where(counts > 0, cum_profit[np.maximum(counts, 1) - 1], 0.0)

    return {
        "approval_rate": approval_rate,
        "n_approved": counts,
        "bad_rate": bad_rate,
        "profit": profit,
        "profit_per_applicant": profit / n,
    }
