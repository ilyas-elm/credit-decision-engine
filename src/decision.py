"""The decision layer: cutoff, refer band, principal reasons, recourse.

`decide()` is the thing a lender would actually call. It returns the PD, the
score, a band, why a decline happened, and what (if anything) would reverse it.

How the two mechanisms combine
------------------------------
There are two independent instruments here and they answer different questions:

* The **conformal prediction set** answers "is the model confident enough to
  decide this alone?", a statistical question about this applicant.
* The **profit cutoff** answers "how much risk is worth taking?", an economic
  question about the lender's appetite.

So conformal decides refer-or-not, and among the non-referred the profit cutoff
decides approve-or-decline. The alternative, letting the conformal singleton
pick approve/decline directly, was rejected because it would hand the risk
appetite to a 1 - alpha coverage parameter, which has no economic meaning. The
two can disagree (a confident {non-default} set whose PD still sits above the
cutoff); when they do, the economics win and the case is declined.

Reasons and recourse are deliberately separate
----------------------------------------------
Principal reasons are the TRUE drivers of the decline, including ones the
applicant cannot change. That is what an explanation owes them. Recourse only
ever touches the characteristics listed as actionable in config. Merging the two
would either hide real drivers or promise changes nobody can make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conformal import Band, MondrianConformal
from profit import ProfitAssumptions, empcs, expected_loss_fraction, strategy_curve
from scorecard import FittedScorecard

__all__ = [
    "Cutoff",
    "choose_cutoff",
    "RecourseStep",
    "Recourse",
    "Decision",
    "DecisionEngine",
    "referral_economics",
]


# --- Cutoff --------------------------------------------------------------


@dataclass(frozen=True)
class Cutoff:
    """Where the approve/decline line sits, and what it is worth.

    ``pd_threshold`` and ``score_threshold`` are two views of the same line:
    points are monotone decreasing in PD, so approving the lowest-PD fraction
    and approving the highest-scoring fraction select the same applicants.
    """

    acceptance_rate: float
    pd_threshold: float
    score_threshold: float
    profit: float
    profit_per_applicant: float
    empcs_optimal_rejection_rate: float
    assumptions: ProfitAssumptions

    @property
    def empcs_implied_acceptance_rate(self) -> float:
        """EMPCS's optimum, converted to an acceptance fraction.

        ``empcs_score.optimal_rate`` returns the predicted-POSITIVE rate, and
        under this project's encoding the positive class is default, so the
        number it returns is the fraction REJECTED, not accepted. empulse's own
        docstring calls it "the fraction of loan applications that should be
        accepted", which contradicts its own encoding convention.

        Verified across nine bad-rate and loss-assumption combinations: the raw
        rate RISES with the bad rate (0.028 at 5% bad to 0.174 at 30% bad),
        which is what a rejection rate does, and ``1 - rate`` tracks the
        strategy curve's optimal acceptance rate throughout. Taking it at face
        value would mean accepting 2.8% of a low-risk portfolio.
        """
        return 1.0 - self.empcs_optimal_rejection_rate

    @property
    def empcs_acceptance_gap(self) -> float:
        """Curve optimum minus EMPCS's implied optimum.

        Reported rather than reconciled. The two optimise different
        objectives; EMPCS integrates over a distribution of loss fractions
        with its own
        weighting, while the strategy curve uses this project's stated linear
        profit model, so they are not expected to agree exactly. They do move
        together; a gap that grows unusually large is a signal to re-examine the
        assumptions, which is why it is surfaced.
        """
        return self.acceptance_rate - self.empcs_implied_acceptance_rate


def choose_cutoff(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    scores: np.ndarray,
    assumptions: ProfitAssumptions | None = None,
    *,
    exposure: np.ndarray | None = None,
    n_points: int = 201,
) -> Cutoff:
    """The profit-maximising acceptance fraction, read off the strategy curve.

    EMPCS's own ``optimal_rate`` is reported alongside for comparison but is not
    what sets the line: the strategy curve is computed under this project's
    stated profit model and exposure, so it is the number that can be audited.
    A large gap between the two is a signal that the assumptions need another
    look, which is why it is surfaced rather than averaged away.
    """
    assumptions = assumptions or ProfitAssumptions()
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not (y_true.size == y_pred_proba.size == scores.size):
        raise ValueError(
            f"Length mismatch: y_true={y_true.size}, pd={y_pred_proba.size}, scores={scores.size}."
        )

    curve = strategy_curve(y_true, y_pred_proba, assumptions, n_points=n_points, exposure=exposure)
    best = int(np.argmax(curve["profit"]))
    acceptance_rate = float(curve["approval_rate"][best])

    # The PD below which applicants are approved, and the equivalent score.
    if acceptance_rate <= 0.0:
        pd_threshold, score_threshold = -np.inf, np.inf
    elif acceptance_rate >= 1.0:
        pd_threshold, score_threshold = np.inf, -np.inf
    else:
        pd_threshold = float(np.quantile(y_pred_proba, acceptance_rate))
        score_threshold = float(np.quantile(scores, 1.0 - acceptance_rate))

    return Cutoff(
        acceptance_rate=acceptance_rate,
        pd_threshold=pd_threshold,
        score_threshold=score_threshold,
        profit=float(curve["profit"][best]),
        profit_per_applicant=float(curve["profit_per_applicant"][best]),
        # NB: empulse returns the REJECTION fraction here. See
        # Cutoff.empcs_implied_acceptance_rate for the evidence.
        empcs_optimal_rejection_rate=empcs(
            y_true, y_pred_proba, assumptions
        ).optimal_acceptance_rate,
        assumptions=assumptions,
    )


# --- Recourse ------------------------------------------------------------


@dataclass(frozen=True)
class RecourseStep:
    characteristic: str
    from_bin: str
    to_bin: str
    points_gained: float


@dataclass(frozen=True)
class Recourse:
    """The smallest set of actionable changes that clears the cutoff."""

    feasible: bool
    steps: list[RecourseStep]
    points_needed: float
    points_gained: float
    note: str = ""

    def __bool__(self) -> bool:
        return self.feasible


def _recourse_for_row(
    scorecard: FittedScorecard,
    current_points: pd.Series,
    shortfall: float,
    actionable: list[str],
) -> Recourse:
    """Exact greedy search over scorecard bins.

    For each actionable characteristic the best reachable gain is
    ``max(points over interval bins) - points now``. Taking those gains in
    descending order is *optimal* for the objective "fewest characteristics
    changed to cover a shortfall": with one move available per characteristic,
    the largest-first order reaches any target in the minimum number of moves.

    The last chosen characteristic is then eased back to the cheapest bin that
    still clears the line, so the advice is the smallest change rather than the
    largest one that happens to work.
    """
    if shortfall <= 0:
        return Recourse(True, [], 0.0, 0.0, "already above the cutoff")
    if not actionable:
        return Recourse(
            False,
            [],
            shortfall,
            0.0,
            "no actionable characteristics are configured, so no recourse can be offered",
        )

    # Best reachable gain per actionable characteristic.
    options: list[tuple[str, float, dict[str, float], float]] = []
    for feature in actionable:
        bins = scorecard.points_by_bin(feature, intervals_only=True)
        if not bins:
            continue
        now = float(current_points[feature])
        best = max(bins.values())
        if best > now:
            options.append((feature, best - now, bins, now))
    options.sort(key=lambda item: item[1], reverse=True)

    reachable = sum(gain for _, gain, _, _ in options)
    if reachable < shortfall:
        return Recourse(
            False,
            [],
            shortfall,
            reachable,
            f"even moving every actionable characteristic to its best bin gains only "
            f"{reachable:.1f} of the {shortfall:.1f} points needed",
        )

    chosen: list[RecourseStep] = []
    gained = 0.0
    for feature, _gain, bins, now in options:  # _gain only set the ordering
        if gained >= shortfall:
            break
        still_needed = shortfall - gained
        # The cheapest bin that covers what is still missing; otherwise the best.
        sufficient = {label: pts for label, pts in bins.items() if pts - now >= still_needed}
        if sufficient:
            label = min(sufficient, key=lambda k: sufficient[k])
            step_gain = sufficient[label] - now
        else:
            label = max(bins, key=lambda k: bins[k])
            step_gain = bins[label] - now
        # from_bin is filled in by _attach_from_bins, which has the row's labels.
        chosen.append(
            RecourseStep(
                characteristic=feature,
                from_bin="",
                to_bin=str(label),
                points_gained=float(step_gain),
            )
        )
        gained += step_gain

    return Recourse(True, chosen, shortfall, gained)


# --- decide() ------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """What the engine returns for one applicant."""

    pd: float
    score: float
    band: str
    principal_reasons: list[tuple[str, float]] = field(default_factory=list)
    recourse: Recourse | None = None

    @property
    def is_decline(self) -> bool:
        return self.band == Band.DECLINE


@dataclass(frozen=True)
class DecisionEngine:
    """Scorecard + conformal band + profit cutoff, wired into one call."""

    scorecard: FittedScorecard
    conformal: MondrianConformal
    cutoff: Cutoff
    actionable_features: list[str] = field(default_factory=list)
    top_reasons: int = 4

    def __post_init__(self) -> None:
        unknown = set(self.actionable_features) - set(self.scorecard.features)
        if unknown:
            raise ValueError(
                f"actionable_features names characteristics the scorecard does not use: "
                f"{sorted(unknown)}."
            )

    def bands(self, X: pd.DataFrame) -> np.ndarray:
        """Refer on a non-singleton conformal set; otherwise the cutoff decides."""
        pds = self.scorecard.predict_pd(X)
        scores = self.scorecard.score(X)
        sets = self.conformal.prediction_sets(pds)
        singleton = sets.sum(axis=1) == 1

        out = np.where(scores >= self.cutoff.score_threshold, Band.APPROVE, Band.DECLINE).astype(
            object
        )
        out[~singleton] = Band.REFER
        return out

    def auto_approve(self, X: pd.DataFrame) -> np.ndarray:
        """What the cutoff alone would decide, ignoring the refer band.

        The counterfactual the referral economics are measured against.
        """
        return self.scorecard.score(X) >= self.cutoff.score_threshold

    def decide_batch(self, X: pd.DataFrame) -> list[Decision]:
        pds = self.scorecard.predict_pd(X)
        scores = self.scorecard.score(X)
        bands = self.bands(X)
        shortfalls = self.scorecard.points_below_max(X)
        awarded = self.scorecard.feature_points(X)
        labels = self.scorecard.bin_labels(X)

        decisions = []
        for i, (_, row) in enumerate(shortfalls.iterrows()):
            band = bands[i]
            reasons: list[tuple[str, float]] = []
            recourse: Recourse | None = None
            if band == Band.DECLINE:
                ranked = row.sort_values(ascending=False).head(self.top_reasons)
                reasons = [(str(k), float(v)) for k, v in ranked.items()]
                current = awarded.iloc[i]
                recourse = _recourse_for_row(
                    self.scorecard,
                    current,
                    self.cutoff.score_threshold - float(scores[i]),
                    self.actionable_features,
                )
                recourse = _attach_from_bins(recourse, labels.iloc[i])
            decisions.append(
                Decision(
                    pd=float(pds[i]),
                    score=float(scores[i]),
                    band=str(band),
                    principal_reasons=reasons,
                    recourse=recourse,
                )
            )
        return decisions

    def decide(self, row: pd.Series | pd.DataFrame) -> Decision:
        """One applicant. Accepts a Series or a single-row DataFrame."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        if len(frame) != 1:
            raise ValueError(f"decide() takes one applicant, got {len(frame)} rows.")
        # optbinning needs numeric dtypes back after a Series round-trip.
        frame = frame.astype(
            {f: "float64" for f in self.scorecard.features if f in frame.columns},
            errors="ignore",
        )
        return self.decide_batch(frame)[0]


def _attach_from_bins(recourse: Recourse, row_labels: pd.Series) -> Recourse:
    """Fill in each step's origin bin, which the search does not carry."""
    if not recourse.steps:
        return recourse
    steps = [
        RecourseStep(
            characteristic=step.characteristic,
            from_bin=str(row_labels[step.characteristic]),
            to_bin=step.to_bin,
            points_gained=step.points_gained,
        )
        for step in recourse.steps
    ]
    return Recourse(
        feasible=recourse.feasible,
        steps=steps,
        points_needed=recourse.points_needed,
        points_gained=recourse.points_gained,
        note=recourse.note,
    )


# --- Referral economics --------------------------------------------------


def referral_economics(
    y_true: np.ndarray,
    bands: np.ndarray,
    auto_approve: np.ndarray,
    cutoff: Cutoff,
    *,
    review_costs: list[float],
    exposure: np.ndarray | None = None,
    review_accuracy: float = 1.0,
) -> dict[str, np.ndarray]:
    """Share referred against expected loss avoided, swept over review cost.

    The benefit of referring a case is whatever the automatic decision would
    have got wrong, so it depends on BOTH the outcome and the call the cutoff
    would have made (``auto_approve``, ignoring referral):

        auto-approved and it defaults -> benefit = the loss avoided
        auto-approved and it repays   -> benefit = 0, the call was already right
        auto-declined and it repays   -> benefit = the ROI recovered
        auto-declined and it defaults -> benefit = 0, already right

    Pricing every referred case as if it would have been approved, the obvious
    shortcut, overstates the band's value on the decline side and understates
    it on the approve side.

    ``review_accuracy`` is the share of referred cases a human resolves
    correctly. The default of 1.0 assumes a PERFECT reviewer: an optimistic
    upper bound, not an estimate, kept as a parameter so the optimism is visible
    and can be swept.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    bands = np.asarray(bands).ravel()
    auto_approve = np.asarray(auto_approve).ravel().astype(bool)
    if not (y_true.size == bands.size == auto_approve.size):
        raise ValueError(
            f"Length mismatch: y_true={y_true.size}, bands={bands.size}, "
            f"auto_approve={auto_approve.size}."
        )
    if not 0.0 <= review_accuracy <= 1.0:
        raise ValueError(f"review_accuracy must lie in [0, 1], got {review_accuracy!r}.")

    n = y_true.size
    exposure = np.ones(n) if exposure is None else np.asarray(exposure, dtype=float).ravel()
    if exposure.size != n:
        raise ValueError(f"exposure has {exposure.size} rows, expected {n}.")

    loss = expected_loss_fraction(cutoff.assumptions)
    roi = cutoff.assumptions.roi
    bad = y_true > 0.5

    auto = np.where(auto_approve, np.where(bad, -loss * exposure, roi * exposure), 0.0)
    # A correct human call books no bad loan and turns away no good one.
    perfect = np.where(bad, 0.0, roi * exposure)

    referred = bands == Band.REFER
    n_referred = int(referred.sum())
    benefit = float(np.sum((perfect - auto)[referred]) * review_accuracy)

    costs = np.asarray(review_costs, dtype=float)
    total_cost = costs * n_referred
    return {
        "review_cost": costs,
        "share_referred": np.full(costs.size, n_referred / n if n else 0.0),
        "n_referred": np.full(costs.size, n_referred, dtype=int),
        "loss_avoided": np.full(costs.size, benefit),
        "total_review_cost": total_cost,
        "net_benefit": benefit - total_cost,
    }
