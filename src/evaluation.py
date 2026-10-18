"""The shared evaluation, computed on the test weeks.

Every model in this project, champion, challenger, and each arm of the
sampling-bias experiment, goes through this one function, so that no reported
comparison is ever between two differently-computed numbers.

What it answers, in the order the brief asks:

* discrimination over time (weekly Gini, and the competition stability metric)
* calibration (Brier, by bucket, and in-the-large per week)
* score drift (PSI of the test-week score distribution against train)
* profit (EMPCS, with a sensitivity grid over the assumptions, and the strategy
  curve)

The stability metric is computed on the weekly Gini curve exactly as the model
produces it. Scores are never post-processed per week to flatter it: that would
improve the metric while making the model worse, which is the one thing the
metric exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

# Agg before pyplot: CI has no display, and importing pyplot first would pick a
# GUI backend and fail there rather than here.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from metrics import (  # noqa: E402
    StabilityResult,
    WeeklyGini,
    brier_score,
    calibration_by_bucket,
    calibration_in_the_large,
    gini,
    psi,
    stability_metric,
    weekly_gini,
)
from profit import (  # noqa: E402
    EmpcsResult,
    ProfitAssumptions,
    empcs,
    empcs_sensitivity,
    strategy_curve,
)

__all__ = [
    "ModelEvaluation",
    "evaluate",
    "comparison_table",
    "plot_weekly_gini",
    "plot_calibration",
    "plot_strategy_curve",
    "plot_referral_economics",
    "write_figures",
]


@dataclass(frozen=True)
class ModelEvaluation:
    """Everything reported about one model on the test weeks."""

    name: str
    n: int
    n_weeks: int
    default_rate: float
    gini_overall: float
    weekly: WeeklyGini = field(repr=False)
    stability: StabilityResult = field(repr=False)
    brier: float
    calibration_buckets: dict[str, np.ndarray] = field(repr=False)
    citl: dict[str, np.ndarray] = field(repr=False)
    score_psi: float
    profit: EmpcsResult = field(repr=False)
    profit_sensitivity: list[EmpcsResult] = field(repr=False)
    strategy: dict[str, np.ndarray] = field(repr=False)

    @property
    def citl_max_abs_gap(self) -> float:
        """Worst weekly gap between mean predicted PD and the observed rate."""
        gaps = self.citl["gap"]
        finite = gaps[np.isfinite(gaps)]
        return float(np.max(np.abs(finite))) if finite.size else float("nan")

    @property
    def empcs_range(self) -> tuple[float, float]:
        """Min and max EMPCS across the assumption grid."""
        if not self.profit_sensitivity:
            return (float("nan"), float("nan"))
        values = [r.emp for r in self.profit_sensitivity]
        return (float(min(values)), float(max(values)))

    def summary_row(self) -> dict[str, float | str | int]:
        low, high = self.empcs_range
        return {
            "model": self.name,
            "n": self.n,
            "weeks": self.n_weeks,
            "gini": self.gini_overall,
            "mean_weekly_gini": self.stability.mean_gini,
            "gini_slope": self.stability.slope,
            "gini_resid_std": self.stability.residual_std,
            "stability_metric": self.stability.metric,
            "brier": self.brier,
            "citl_max_abs_gap": self.citl_max_abs_gap,
            "score_psi_vs_train": self.score_psi,
            "empcs": self.profit.emp,
            "empcs_min": low,
            "empcs_max": high,
            "empcs_rejection_rate": self.profit.optimal_acceptance_rate,
        }


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    weeks: np.ndarray,
    *,
    test_scores: np.ndarray | None = None,
    train_scores: np.ndarray | None = None,
    assumptions: ProfitAssumptions | None = None,
    sensitivity_grid: dict[str, list[float]] | None = None,
    exposure: np.ndarray | None = None,
    n_buckets: int = 10,
    psi_bins: int = 10,
) -> ModelEvaluation:
    """Evaluate one model on the test weeks.

    ``test_scores`` and ``train_scores`` drive the score-PSI only. They are the
    model's score or PD distribution on each block; when either is missing the
    PSI comes back NaN rather than silently reusing the PD, because a PSI
    computed against the wrong reference period looks like a real number and is
    not one.
    """
    assumptions = assumptions or ProfitAssumptions()
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    weeks = np.asarray(weeks).ravel()
    if not (y_true.size == y_pred_proba.size == weeks.size):
        raise ValueError(
            f"Length mismatch: y_true={y_true.size}, pd={y_pred_proba.size}, weeks={weeks.size}."
        )
    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    curve = weekly_gini(y_true, y_pred_proba, weeks)
    if len(curve) < 2:
        raise ValueError(
            f"Only {len(curve)} scoreable test week(s); the stability metric needs 2. "
            f"Unscoreable weeks: {curve.skipped_weeks.tolist()}."
        )

    if test_scores is None or train_scores is None:
        score_psi = float("nan")
    else:
        score_psi = psi(train_scores, test_scores, bins=psi_bins)

    return ModelEvaluation(
        name=name,
        n=int(y_true.size),
        n_weeks=len(curve),
        default_rate=float(np.mean(y_true)),
        gini_overall=gini(y_true, y_pred_proba),
        weekly=curve,
        stability=stability_metric(curve),
        brier=brier_score(y_true, y_pred_proba),
        calibration_buckets=calibration_by_bucket(y_true, y_pred_proba, n_buckets=n_buckets),
        citl=calibration_in_the_large(y_true, y_pred_proba, weeks),
        score_psi=score_psi,
        profit=empcs(y_true, y_pred_proba, assumptions),
        profit_sensitivity=(
            empcs_sensitivity(y_true, y_pred_proba, sensitivity_grid) if sensitivity_grid else []
        ),
        strategy=strategy_curve(y_true, y_pred_proba, assumptions, exposure=exposure),
    )


def comparison_table(evaluations: list[ModelEvaluation]) -> pd.DataFrame:
    """One row per model, champion first by convention."""
    if not evaluations:
        raise ValueError("comparison_table needs at least one evaluation.")
    return pd.DataFrame([e.summary_row() for e in evaluations])


# --- Figures -------------------------------------------------------------


def _finish(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weekly_gini(evaluations: list[ModelEvaluation], path: str | Path) -> Path:
    """Weekly Gini over time, with each model's fitted trend.

    The trend line is drawn because the stability metric penalises a falling
    slope at 88x, so the slope is the single most consequential thing in the
    figure and should be visible rather than inferred.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for e in evaluations:
        weeks = e.weekly.weeks.astype(float)
        ax.plot(weeks, e.weekly.values, marker="o", ms=3, lw=1.2, label=e.name)
        slope, intercept = np.polyfit(weeks, e.weekly.values, 1)
        ax.plot(
            weeks,
            slope * weeks + intercept,
            ls="--",
            lw=1.0,
            alpha=0.7,
            label=f"{e.name} trend ({slope:+.4f}/wk)",
        )
    ax.set_xlabel("WEEK_NUM (test weeks)")
    ax.set_ylabel("Gini (2·AUC − 1)")
    ax.set_title("Weekly Gini on the test weeks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_calibration(evaluations: list[ModelEvaluation], path: str | Path) -> Path:
    """Predicted against observed, by bucket, with the 45-degree reference."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    limit = 0.0
    for e in evaluations:
        buckets = e.calibration_buckets
        ax.plot(
            buckets["mean_pd"],
            buckets["observed_rate"],
            marker="o",
            ms=4,
            lw=1.2,
            label=f"{e.name} (Brier {e.brier:.4f})",
        )
        limit = max(limit, float(buckets["mean_pd"].max()), float(buckets["observed_rate"].max()))
    limit *= 1.05
    ax.plot([0, limit], [0, limit], ls=":", c="grey", lw=1, label="perfect calibration")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("mean predicted PD")
    ax.set_ylabel("observed default rate")
    ax.set_title("Calibration by bucket, test weeks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_strategy_curve(evaluations: list[ModelEvaluation], path: str | Path) -> Path:
    """Approval rate against bad rate and profit, with each profit optimum marked."""
    fig, (ax_bad, ax_profit) = plt.subplots(1, 2, figsize=(11, 4.5))
    for e in evaluations:
        s = e.strategy
        ax_bad.plot(s["approval_rate"], s["bad_rate"], lw=1.4, label=e.name)
        ax_profit.plot(s["approval_rate"], s["profit_per_applicant"], lw=1.4, label=e.name)
        best = int(np.argmax(s["profit"]))
        ax_profit.axvline(s["approval_rate"][best], ls="--", lw=0.9, alpha=0.6)
        ax_profit.annotate(
            f"{s['approval_rate'][best]:.0%}",
            (s["approval_rate"][best], s["profit_per_applicant"][best]),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
        )
    for ax, ylabel in ((ax_bad, "bad rate among approved"), (ax_profit, "profit per applicant")):
        ax.set_xlabel("approval rate")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax_bad.set_title("Risk as approvals loosen")
    ax_profit.set_title("Profit, with the optimum marked")
    return _finish(fig, path)


def plot_referral_economics(
    economics: dict[str, np.ndarray],
    path: str | Path,
    *,
    label: str = "refer band",
) -> Path:
    """Expected loss avoided against the cost of reviewing, by review price.

    The brief's "share referred vs expected loss avoided, with a review-cost
    parameter". Plotted as a net benefit so the break-even price is readable off
    the axis rather than inferred from two lines: that price is the number a
    review desk is actually deciding about.
    """
    costs = np.asarray(economics["review_cost"], dtype=float)
    net = np.asarray(economics["net_benefit"], dtype=float)
    avoided = float(np.asarray(economics["loss_avoided"], dtype=float)[0])
    share = float(np.asarray(economics["share_referred"], dtype=float)[0])

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(costs, net, marker="o", lw=1.8, label=f"net benefit ({label})")
    ax.axhline(0, color="k", lw=1)
    ax.fill_between(costs, 0, net, where=net >= 0, alpha=0.15, color="tab:green")
    ax.fill_between(costs, 0, net, where=net < 0, alpha=0.15, color="tab:red")

    # Break-even price, by linear interpolation between the bracketing points.
    sign = np.sign(net)
    crossing = np.flatnonzero(sign[:-1] * sign[1:] < 0)
    if crossing.size:
        i = int(crossing[0])
        frac = net[i] / (net[i] - net[i + 1])
        break_even = costs[i] + frac * (costs[i + 1] - costs[i])
        ax.axvline(break_even, ls="--", lw=1.2, color="grey")
        ax.annotate(
            f"break-even {break_even:.3f}/review",
            (break_even, 0), textcoords="offset points", xytext=(6, 14), fontsize=9,
        )
    ax.set_xlabel("cost per manual review")
    ax.set_ylabel("net benefit (loss avoided − review cost)")
    ax.set_title(
        f"The refer band stops paying for itself\n"
        f"{share:.1%} referred, {avoided:,.0f} of loss avoided (perfect reviewer)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return _finish(fig, path)


def plot_alpha_sweep(sweep: dict[str, np.ndarray], path: str | Path) -> Path:
    """What alpha buys: the size of the review band against the coverage.

    Two axes because the trade-off is between two different things, and one
    curve that is not monotone. Referrals are split into their two causes,
    stacked, because the referral rate alone cannot tell a desk which side of
    the curve it is sizing itself from: at small alpha the band refers cases
    whose label is ambiguous, at large alpha cases where NEITHER label
    qualifies, and those are opposite failures at similar rates.
    """
    alpha = np.asarray(sweep["alpha"], dtype=float)
    both = np.asarray(sweep["both_labels"], dtype=float)
    empty = np.asarray(sweep["empty_set"], dtype=float)
    cov_default = np.asarray(sweep["coverage_default"], dtype=float)
    target = 1.0 - alpha

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.stackplot(
        alpha, both, empty,
        labels=["referred: both labels plausible", "referred: neither label plausible"],
        colors=["tab:blue", "tab:red"], alpha=0.55,
    )
    ax.set_xlabel("α")
    ax.set_ylabel("share referred")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    right = ax.twinx()
    right.plot(alpha, cov_default, color="k", lw=1.8, label="default-class coverage")
    right.plot(alpha, target, color="k", lw=1.0, ls=":", label="target (1 − α)")
    right.set_ylabel("coverage of the default class")
    right.set_ylim(0, 1)

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, fontsize=8, loc="upper center")
    ax.set_title(
        "α is the only lever on the review band, and the curve is not monotone\n"
        "between the two regimes the band refers almost nobody"
    )
    return _finish(fig, path)


def write_figures(
    evaluations: list[ModelEvaluation],
    directory: str | Path,
) -> dict[str, Path]:
    """The three acceptance figures from the brief."""
    directory = Path(directory)
    return {
        "weekly_gini": plot_weekly_gini(evaluations, directory / "weekly_gini.png"),
        "calibration": plot_calibration(evaluations, directory / "calibration.png"),
        "strategy_curve": plot_strategy_curve(evaluations, directory / "strategy_curve.png"),
    }
