"""Sampling bias and reject inference: the signature experiment.

A lender only learns the outcome of loans it approved. Models trained and, more
importantly, *validated* on that survivor population are measuring something
other than what they claim to.

LIMITATION, stated plainly: the Home Credit data already contains booked loans
only. The real rejects are gone and nothing here recovers them. This experiment
simulates a SECOND selection policy on top of an already-selected population, so
it measures how much each fix recovers under a known, synthetic selection, not
how much it would recover on the lender's original one. Every number here should
be read with that in mind.

Design
------
Three windows, independent of the main split so that changing one cannot
silently alter the other:

    policy weeks (~25%)  fit an "old policy" scorecard
    dev weeks    (~45%)  apply it, reject the worst r%, HIDE those labels
    test weeks   (~30%)  full population with all labels: the ground truth

Every arm receives the same :class:`Scenario` and returns a
:class:`TrainingSet`, so arms differ only in how they treat the rejects,
identical features, identical hyperparameters, identical subsample.

Why the hidden labels are hidden structurally
---------------------------------------------
`Scenario.y_observed` is NaN for every rejected applicant. The true labels live
behind :meth:`Scenario.oracle_labels` (A0 only) and :meth:`Scenario.explore`
(A6 only, which pays for what it sees). An arm that peeks has to call one of
those by name, so leakage is a visible line of code rather than an easy slip.
`tests/test_sampling_bias.py` scrambles the hidden labels and asserts A1-A5 are
bit-identical, which is the check that this actually holds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import checked_kwargs
from profit import ProfitAssumptions, expected_loss_fraction

__all__ = [
    "SamplingBiasParams",
    "window_weeks",
    "Scenario",
    "make_scenario",
    "TrainingSet",
    "ARMS",
    "build_arm",
    "select_features",
    "MIN_RECOVERY_GAP",
    "recovery_share",
    "evaluation_bias",
    "exploration_cost",
    "exploration_budget_for_detection",
]


@dataclass(frozen=True)
class SamplingBiasParams:
    """Experiment parameters, from [sampling_bias] in config."""

    reject_rates: tuple[int, ...] = (10, 20, 30, 50)
    n_features: int = 40
    dev_subsample: int = 150_000
    n_trees: int = 400
    seeds: tuple[int, ...] = (0, 1, 2)
    parcelling_uplift: tuple[float, ...] = (1.5, 2.0)
    exploration_pct: tuple[int, ...] = (1, 2, 5)
    self_learning_iters: int = 5
    self_learning_confidence: float = 0.05

    @classmethod
    def from_config(cls, section: dict) -> SamplingBiasParams:
        tuple_keys = {
            "reject_rates",
            "seeds",
            "parcelling_uplift",
            "exploration_pct",
        }
        checked = checked_kwargs(
            section,
            cls,
            "sampling_bias",
            # The window fractions in this section build
            # SamplingBiasWindows, not the experiment parameters.
            consumed_elsewhere={"policy_frac", "dev_frac", "test_frac"},
        )
        kwargs = {
            key: (tuple(value) if key in tuple_keys else value)
            for key, value in checked.items()
        }
        return cls(**kwargs)


def window_weeks(weeks, windows) -> dict[str, list[int]]:
    """Split distinct weeks into policy / dev / test blocks, oldest first.

    Shares the cumulative-rounding rule with :mod:`splits` so block sizes sum to
    exactly the number of weeks, but stays a separate function: these windows
    are not the model's splits and must not drift with them.
    """
    unique = sorted({int(w) for w in weeks})
    n = len(unique)
    ordered = windows.ordered_fractions
    if n < len(ordered):
        raise ValueError(f"Need at least {len(ordered)} distinct weeks, got {n}.")

    blocks: dict[str, list[int]] = {}
    prev, cumulative = 0, 0.0
    for name, frac in ordered[:-1]:
        cumulative += frac
        end = int(round(cumulative * n))
        remaining = len(ordered) - len(blocks) - 1
        end = min(max(end, prev + 1), n - remaining)
        blocks[name] = unique[prev:end]
        prev = end
    blocks[ordered[-1][0]] = unique[prev:]
    return blocks


@dataclass(frozen=True)
class Scenario:
    """One reject rate applied to the development weeks.

    ``y_observed`` is what the simulated lender knows: the true label where the
    old policy approved, NaN where it rejected.
    """

    X: pd.DataFrame
    accepted: np.ndarray
    y_observed: np.ndarray
    policy_pd: np.ndarray
    reject_rate: int
    _y_true: np.ndarray = field(repr=False)

    @property
    def rejected(self) -> np.ndarray:
        return ~self.accepted

    @property
    def n_rejected(self) -> int:
        return int(self.rejected.sum())

    def oracle_labels(self) -> np.ndarray:
        """Every true label. A0 only; this is the ground truth being hidden."""
        return self._y_true.copy()

    def explore(self, mask: np.ndarray) -> np.ndarray:
        """Labels for a subset the lender chose to approve anyway. A6 only.

        Approving a rejected applicant to learn their outcome is a real decision
        with a real cost; :func:`exploration_cost` prices it. Returns NaN outside
        the mask so the result cannot be used as a full label vector by accident.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self._y_true.shape:
            raise ValueError(f"mask has shape {mask.shape}, expected {self._y_true.shape}.")
        out = np.full(self._y_true.shape, np.nan)
        out[mask] = self._y_true[mask]
        return out


def make_scenario(
    X_dev: pd.DataFrame,
    y_dev: np.ndarray,
    policy_pd: np.ndarray,
    reject_rate: int,
) -> Scenario:
    """Apply the old policy: reject the worst ``reject_rate`` percent.

    "Worst" is highest predicted PD under the policy model, which is what a
    cutoff policy does.
    """
    y_dev = np.asarray(y_dev, dtype=float).ravel()
    policy_pd = np.asarray(policy_pd, dtype=float).ravel()
    if not (len(X_dev) == y_dev.size == policy_pd.size):
        raise ValueError(
            f"Length mismatch: X={len(X_dev)}, y={y_dev.size}, policy_pd={policy_pd.size}."
        )
    if not 0 <= reject_rate < 100:
        raise ValueError(f"reject_rate must lie in [0, 100), got {reject_rate}.")

    n_reject = int(round(len(X_dev) * reject_rate / 100.0))
    # Rank worst-first; stable so ties break deterministically.
    order = np.argsort(-policy_pd, kind="stable")
    accepted = np.ones(len(X_dev), dtype=bool)
    accepted[order[:n_reject]] = False

    y_observed = np.where(accepted, y_dev, np.nan)
    return Scenario(
        X=X_dev,
        accepted=accepted,
        y_observed=y_observed,
        policy_pd=policy_pd,
        reject_rate=reject_rate,
        _y_true=y_dev,
    )


@dataclass(frozen=True)
class TrainingSet:
    """What an arm hands to the model. Arms differ only in this."""

    arm: str
    X: pd.DataFrame
    y: np.ndarray
    weight: np.ndarray | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.X) != self.y.size:
            raise ValueError(f"X has {len(self.X)} rows, y has {self.y.size}.")
        if self.weight is not None and self.weight.size != self.y.size:
            raise ValueError(f"weight has {self.weight.size} entries, y has {self.y.size}.")
        if self.y.size and not np.isin(np.unique(self.y), (0.0, 1.0)).all():
            raise ValueError(f"y must be 0/1; found {np.unique(self.y)[:5]}.")


# --- The arms ------------------------------------------------------------


def arm_a0_oracle(scenario: Scenario, **_) -> TrainingSet:
    """Every label available. The ceiling nobody can reach in practice."""
    return TrainingSet(
        arm="A0_oracle",
        X=scenario.X,
        y=scenario.oracle_labels(),
        note="all development-week labels, including the rejects'",
    )


def arm_a1_accepts_only(scenario: Scenario, **_) -> TrainingSet:
    """Train on approved loans only. What a lender actually has."""
    mask = scenario.accepted
    return TrainingSet(
        arm="A1_accepts_only",
        X=scenario.X[mask],
        y=scenario.y_observed[mask],
        note=f"{int(mask.sum())} approved cases; {scenario.n_rejected} rejects discarded",
    )


def arm_a2_hard_cutoff(scenario: Scenario, **_) -> TrainingSet:
    """Label every reject bad.

    The crudest augmentation, and a deliberate overstatement: the old policy
    rejected the worst r%, but plenty of them would have repaid.
    """
    y = np.where(scenario.accepted, scenario.y_observed, 1.0)
    return TrainingSet(
        arm="A2_hard_cutoff",
        X=scenario.X,
        y=y,
        note=f"{scenario.n_rejected} rejects all labelled bad",
    )


def arm_a3_fuzzy(scenario: Scenario, *, reject_pd: np.ndarray, **_) -> TrainingSet:
    """Duplicate each reject as good AND bad, weighted by A1's predicted PD.

    The honest version of A2: instead of asserting a reject was bad, it carries
    the uncertainty as weight. ``reject_pd`` comes from the A1 model, which is
    the only estimate the simulated lender could legitimately have.
    """
    accepted_idx = np.flatnonzero(scenario.accepted)
    rejected_idx = np.flatnonzero(scenario.rejected)
    reject_pd = np.asarray(reject_pd, dtype=float).ravel()
    if reject_pd.size != rejected_idx.size:
        raise ValueError(f"reject_pd has {reject_pd.size} entries for {rejected_idx.size} rejects.")

    X = pd.concat(
        [
            scenario.X.iloc[accepted_idx],
            scenario.X.iloc[rejected_idx],  # as bad
            scenario.X.iloc[rejected_idx],  # as good
        ],
        ignore_index=True,
    )
    y = np.concatenate(
        [
            scenario.y_observed[accepted_idx],
            np.ones(rejected_idx.size),
            np.zeros(rejected_idx.size),
        ]
    )
    weight = np.concatenate([np.ones(accepted_idx.size), reject_pd, 1.0 - reject_pd])
    return TrainingSet(
        arm="A3_fuzzy",
        X=X,
        y=y,
        weight=weight,
        note="each reject duplicated as good and bad, weighted by A1's PD",
    )


def arm_a4_parcelling(
    scenario: Scenario,
    *,
    reject_pd: np.ndarray,
    accepted_pd: np.ndarray,
    uplift: float,
    rng: np.random.Generator,
    n_bands: int = 10,
    **_,
) -> TrainingSet:
    """Draw reject labels from the score-band bad rate, inflated by ``uplift``.

    Rejects are placed in score bands built from the ACCEPTED population, and
    assigned a bad label with probability ``band bad rate x uplift``. The uplift
    encodes the analyst's belief that a reject is worse than an accepted
    applicant with the same score, which is an assumption, so k is swept.
    """
    accepted_idx = np.flatnonzero(scenario.accepted)
    rejected_idx = np.flatnonzero(scenario.rejected)
    accepted_pd = np.asarray(accepted_pd, dtype=float).ravel()
    reject_pd = np.asarray(reject_pd, dtype=float).ravel()
    y_acc = scenario.y_observed[accepted_idx]

    edges = np.unique(np.quantile(accepted_pd, np.linspace(0, 1, n_bands + 1)))
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf

    band_acc = np.digitize(accepted_pd, edges[1:-1])
    band_rej = np.digitize(reject_pd, edges[1:-1])

    overall = float(np.mean(y_acc)) if y_acc.size else 0.0
    rates = {}
    for band in np.unique(band_acc):
        in_band = band_acc == band
        rates[band] = float(np.mean(y_acc[in_band])) if in_band.sum() else overall

    probs = np.array([min(rates.get(b, overall) * uplift, 1.0) for b in band_rej])
    drawn = rng.binomial(1, probs).astype(float)

    y = np.empty(len(scenario.X))
    y[accepted_idx] = y_acc
    y[rejected_idx] = drawn
    return TrainingSet(
        arm=f"A4_parcelling_k{uplift:g}",
        X=scenario.X,
        y=y,
        note=f"reject labels drawn from band bad rate x {uplift:g}",
    )


def arm_a5_self_learning(
    scenario: Scenario,
    *,
    reject_pd: np.ndarray,
    confidence: float,
    iterations: int,
    **_,
) -> TrainingSet:
    """Add confidently scored rejects with pseudo-labels, a band at a time.

    Inspired by the Bayesian-augmented self-learning of Kozodoi et al.; this is
    NOT a replication of BASL and should not be described as one. The simplified
    form here admits rejects whose predicted PD falls outside the middle of the
    distribution, widening the admitted band over ``iterations`` passes, and
    labels them by their own prediction.

    Single-pass by construction: refitting the model inside each iteration would
    need a validation set, and every validation set the simulated lender could
    build is itself drawn from the accepted population.
    """
    accepted_idx = np.flatnonzero(scenario.accepted)
    rejected_idx = np.flatnonzero(scenario.rejected)
    reject_pd = np.asarray(reject_pd, dtype=float).ravel()

    # Confidence is a QUANTILE of the reject score distribution, not an absolute
    # probability. An absolute rule is unusable at a 3% base rate: no predicted
    # PD approaches 0.5, so "confidently bad" never fires and "pd >= 0.5" labels
    # every admitted reject GOOD, the exact opposite of A2, and measured here
    # at Gini 0.02 (LightGBM) and -0.19 (scorecard) before the fix. Quantiles
    # make the rule scale-free, so the same configuration behaves sensibly
    # whatever the base rate.
    admitted = np.zeros(rejected_idx.size, dtype=bool)
    pseudo_all = np.zeros(rejected_idx.size, dtype=float)
    for step in range(1, iterations + 1):
        q = min(confidence * step, 0.5)
        low, high = np.quantile(reject_pd, [q, 1.0 - q])
        is_good = reject_pd <= low
        is_bad = reject_pd >= high
        admitted |= is_good | is_bad
        pseudo_all[is_bad] = 1.0

    keep = rejected_idx[admitted]
    pseudo = pseudo_all[admitted]

    X = pd.concat([scenario.X.iloc[accepted_idx], scenario.X.iloc[keep]], ignore_index=True)
    y = np.concatenate([scenario.y_observed[accepted_idx], pseudo])
    return TrainingSet(
        arm="A5_self_learning",
        X=X,
        y=y,
        note=(
            f"{int(admitted.sum())} of {rejected_idx.size} rejects admitted with "
            f"pseudo-labels after {iterations} widening passes"
        ),
    )


def arm_a6_exploration(
    scenario: Scenario,
    *,
    explore_pct: int,
    rng: np.random.Generator,
    **_,
) -> TrainingSet:
    """Approve a random p% of rejects and observe their outcomes.

    The only arm that buys new information rather than inventing it. The
    explored loans carry weight 100/p because each one stands for the 100/p
    rejects it was sampled from, without that the model would treat the
    explored slice as the whole reject population.
    """
    if not 0 < explore_pct <= 100:
        raise ValueError(f"explore_pct must lie in (0, 100], got {explore_pct}.")

    rejected_idx = np.flatnonzero(scenario.rejected)
    n_explore = int(round(rejected_idx.size * explore_pct / 100.0))
    chosen = (
        rng.choice(rejected_idx, size=n_explore, replace=False)
        if n_explore
        else np.array([], dtype=int)
    )

    mask = np.zeros(len(scenario.X), dtype=bool)
    mask[chosen] = True
    observed = scenario.explore(mask)  # the only legitimate look at hidden labels

    accepted_idx = np.flatnonzero(scenario.accepted)
    keep = np.concatenate([accepted_idx, chosen]).astype(int)
    y = np.concatenate([scenario.y_observed[accepted_idx], observed[chosen]])
    weight = np.concatenate([np.ones(accepted_idx.size), np.full(chosen.size, 100.0 / explore_pct)])
    return TrainingSet(
        arm=f"A6_exploration_p{explore_pct}",
        X=scenario.X.iloc[keep].reset_index(drop=True),
        y=y,
        weight=weight,
        note=f"{chosen.size} rejects approved at random, weighted {100.0 / explore_pct:g}",
    )


ARMS = {
    "A0_oracle": arm_a0_oracle,
    "A1_accepts_only": arm_a1_accepts_only,
    "A2_hard_cutoff": arm_a2_hard_cutoff,
    "A3_fuzzy": arm_a3_fuzzy,
    "A4_parcelling": arm_a4_parcelling,
    "A5_self_learning": arm_a5_self_learning,
    "A6_exploration": arm_a6_exploration,
}


def build_arm(name: str, scenario: Scenario, **kwargs) -> TrainingSet:
    """Dispatch to one arm by name."""
    try:
        arm = ARMS[name]
    except KeyError:
        raise ValueError(f"Unknown arm {name!r}. Available: {sorted(ARMS)}.") from None
    return arm(scenario, **kwargs)


# --- Measurement ---------------------------------------------------------


def select_features(
    iv_by_feature: dict[str, float],
    gain_by_feature: dict[str, float],
    n_features: int,
) -> list[str]:
    """Top ``n_features`` by average rank of IV and LightGBM gain.

    Computed on the POLICY weeks only. The main model's feature ranking must not
    be reused: it was fitted with development-week labels, including the ones
    this experiment hides, so importing it would leak the ground truth into
    every arm equally and quietly shrink the A0 - A1 gap.
    """
    shared = sorted(set(iv_by_feature) & set(gain_by_feature))
    if not shared:
        raise ValueError("IV and gain rankings share no features.")

    def ranks(scores: dict[str, float]) -> dict[str, int]:
        ordered = sorted(shared, key=lambda f: scores[f], reverse=True)
        return {feature: i for i, feature in enumerate(ordered)}

    iv_rank, gain_rank = ranks(iv_by_feature), ranks(gain_by_feature)
    combined = sorted(shared, key=lambda f: (iv_rank[f] + gain_rank[f], f))
    return combined[:n_features]


#: Smallest A0 - A1 Gini gap worth expressing a recovery against. Below this the
#: ratio is dominated by its denominator; 0.01 Gini is about the smallest
#: difference anyone would act on.
MIN_RECOVERY_GAP = 0.01


def recovery_share(
    oracle: float, accepts_only: float, arm: float, *, min_gap: float = MIN_RECOVERY_GAP
) -> float:
    """Share of the A0 - A1 gap an arm recovered.

    1.0 means the arm matched the oracle; 0.0 means it matched accepts-only;
    negative means it did worse than doing nothing, which is a real and
    frequently observed outcome and is reported rather than clipped.

    Returns NaN unless the oracle beats accepts-only by at least ``min_gap``.
    Two cases make the ratio meaningless and both occur in this data:

    * **A negligible gap.** Dividing by it amplifies noise into a percentage.
    * **A NEGATIVE gap**, where hiding the rejects' labels made the model
      BETTER. Measured here for the scorecard arms: A0 0.411 against A1 0.438
      at a 10 % reject rate.
      The ratio then has a negative denominator, so an arm that is far worse
      than both scores a large POSITIVE "recovery", the catastrophic A5 arm
      came out at +2320% before this guard. There is no gap to recover, and the
      honest answer is to say so rather than to print a number.

    Raw Gini for every arm is reported alongside, so a NaN here never hides a
    result; it only refuses to express one as a ratio it cannot support.
    """
    gap = oracle - accepts_only
    if not np.isfinite(gap) or gap < min_gap:
        return float("nan")
    return float((arm - accepts_only) / gap)


def evaluation_bias(
    metric_on_accepted: float,
    metric_on_full: float,
) -> dict[str, float]:
    """How wrong the lender's own validation would be.

    ``metric_on_accepted`` is what a real lender could measure: the metric over
    the test cases its policy would have approved. ``metric_on_full`` is the
    truth over everyone. The difference is the part of the brief's third
    question about validation rather than training.
    """
    return {
        "observed": metric_on_accepted,
        "true": metric_on_full,
        "bias": metric_on_accepted - metric_on_full,
        "relative_bias": (
            (metric_on_accepted - metric_on_full) / abs(metric_on_full)
            if abs(metric_on_full) > 1e-12
            else float("nan")
        ),
    }


def exploration_cost(
    explored_labels: np.ndarray,
    assumptions: ProfitAssumptions,
    *,
    exposure: np.ndarray | None = None,
) -> float:
    """Realised loss on the loans approved purely to learn from them.

    Only the losses count. The ROI on explored loans that repaid is not netted
    off: those applicants were rejected by policy and would not have been booked
    at all, so their return is not revenue the lender was otherwise going to
    forgo. Counting it would make exploration look free.
    """
    labels = np.asarray(explored_labels, dtype=float).ravel()
    # NaN marks a case that was never explored. Dropping those here means
    # `exposure`, if given, must already line up with the explored subset; the
    # size check below turns a mismatch into an error rather than a silent
    # pairing of the wrong rows.
    labels = labels[~np.isnan(labels)]
    if labels.size == 0:
        return 0.0
    exposure = (
        np.ones(labels.size) if exposure is None else np.asarray(exposure, dtype=float).ravel()
    )
    if exposure.size != labels.size:
        raise ValueError(f"exposure has {exposure.size} entries, expected {labels.size}.")
    return float(np.sum(labels * expected_loss_fraction(assumptions) * exposure))


def exploration_budget_for_detection(
    bad_rate_rejected: float,
    rejected_share: float,
    gap_to_detect: float,
    threshold: float,
    assumptions: ProfitAssumptions | None = None,
    *,
    n_rejected: int | None = None,
) -> dict[str, float]:
    """How many random approvals it takes to SEE a calibration gap, and its cost.

    A lender observes outcomes only for loans it approved, so a
    calibration-in-the-large check runs on the accepted book, where the gap is
    flattered. Approving a random slice of rejects restores the measurement,
    and this sizes that slice.

    The rejects' outcomes are a binomial sample, so their mean is estimated
    with standard error ``sqrt(p(1-p)/n)``, and it enters the population mean
    weighted by ``rejected_share``. Resolving the population gap to within
    half the margin between it and the alert threshold is enough to tell them
    apart, which is what ``n`` is solved for.

    This is a sizing calculation, not a simulation: it assumes the explored
    slice is drawn at random from the rejects, which is what arm A6 does, and
    it prices only the realised losses on those loans, which is what
    :func:`exploration_cost` prices. The point of it is that the same budget
    the experiment found harmful as a TRAINING intervention is roughly the
    budget that buys the measurement.
    """
    assumptions = assumptions or ProfitAssumptions()
    if not 0.0 < bad_rate_rejected < 1.0:
        raise ValueError(f"bad_rate_rejected must lie in (0, 1), got {bad_rate_rejected}.")
    if not 0.0 < rejected_share <= 1.0:
        raise ValueError(f"rejected_share must lie in (0, 1], got {rejected_share}.")
    margin = abs(gap_to_detect) - abs(threshold)
    if margin <= 0:
        raise ValueError(
            f"gap_to_detect ({gap_to_detect}) must exceed the threshold ({threshold}); "
            "there is nothing to tell apart otherwise."
        )

    tolerance = margin / 2.0
    standard_error = tolerance / rejected_share
    n_needed = bad_rate_rejected * (1.0 - bad_rate_rejected) / standard_error**2
    cost = n_needed * bad_rate_rejected * expected_loss_fraction(assumptions)
    out = {
        "n_explored": float(n_needed),
        "tolerance": float(tolerance),
        "expected_cost": float(cost),
    }
    if n_rejected:
        out["share_of_rejects"] = float(n_needed / n_rejected)
    return out
