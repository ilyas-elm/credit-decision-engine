"""Challenger model: LightGBM with monotone constraints.

The point of the challenger is not to win on AUC. It is to answer whether a
gradient-boosted model earns its added opacity once stability, calibration and
profit are measured -- so it is built to be a fair comparison, not a maximal one:

* Monotone constraints taken from the champion's binning trends, so the
  challenger cannot buy Gini with relationships the scorecard was forbidden to
  use. A characteristic the scorecard found non-monotone gets no constraint
  rather than a guessed one.
* Fixed, sensible hyperparameters with early stopping on val_a. No long search.
  Optuna is allowed by the brief at up to 30 trials but is NOT a dependency of
  this project and `optuna_trials` is 0 in config: a tuned challenger against an
  untuned champion would answer a different question than the one being asked.
* No class rebalancing anywhere. `is_unbalance` and `scale_pos_weight` are left
  at their neutral defaults, explicitly. Rebalancing distorts predicted
  probabilities, and this project needs calibrated PDs for the profit cutoff --
  it would trade the thing being measured for a metric nobody reports.
* Isotonic calibration on val_a, since raw LightGBM probabilities under a ~3%
  event rate are generally not calibrated in the large.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from calibrate import IsotonicCalibrator

__all__ = ["ChallengerParams", "FittedChallenger", "fit_challenger"]


@dataclass(frozen=True)
class ChallengerParams:
    """LightGBM parameters, from [challenger] in config."""

    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 200
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    early_stopping_rounds: int = 100
    seed: int = 20251013
    optuna_trials: int = 0

    @classmethod
    def from_config(cls, section: dict) -> ChallengerParams:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(section) - known
        if unknown:
            raise ValueError(
                f"[challenger] has unrecognised keys {sorted(unknown)}. Known: {sorted(known)}."
            )
        return cls(**{k: v for k, v in section.items() if k in known})

    def __post_init__(self) -> None:
        if self.optuna_trials > 30:
            raise ValueError(
                f"optuna_trials is {self.optuna_trials}; the brief caps the search at 30. "
                "A long search would make the champion/challenger comparison unfair."
            )

    def lgb_kwargs(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "colsample_bytree": self.feature_fraction,
            "subsample": self.bagging_fraction,
            "subsample_freq": self.bagging_freq,
            "reg_lambda": self.lambda_l2,
            "random_state": self.seed,
            "n_jobs": -1,
            "verbose": -1,
            # Stated rather than defaulted: no rebalancing, by design.
            "is_unbalance": False,
            "scale_pos_weight": 1.0,
        }


@dataclass
class FittedChallenger:
    """A fitted LightGBM plus its calibrator and the constraints it ran under."""

    model: lgb.LGBMClassifier
    features: list[str]
    monotone_constraints: list[int]
    calibrator: IsotonicCalibrator | None = None
    best_iteration: int | None = None
    params: ChallengerParams = field(default_factory=ChallengerParams)

    def predict_pd_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Uncalibrated PD, straight from LightGBM."""
        return np.asarray(self.model.predict_proba(X[self.features])[:, 1], dtype=float)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated PD, or the raw one when no calibrator was fitted.

        Calibration is monotone, so it never inverts the ranking. It does
        introduce ties, which move AUC by a small amount (~3e-4 of Gini on the
        fixtures); see calibrate.IsotonicCalibrator.
        """
        raw = self.predict_pd_raw(X)
        return raw if self.calibrator is None else self.calibrator.transform(raw)

    def gain_importance(self) -> pd.Series:
        """Feature importance by total split gain, descending."""
        return pd.Series(
            self.model.booster_.feature_importance(importance_type="gain"),
            index=self.features,
            dtype=float,
        ).sort_values(ascending=False)


def fit_challenger(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    params: ChallengerParams | None = None,
    monotone_constraints: list[int] | None = None,
    calibrate: bool = True,
    sample_weight: np.ndarray | None = None,
) -> FittedChallenger:
    """Fit with early stopping on val_a, then calibrate on the same block.

    ``monotone_constraints`` is aligned to ``X_train.columns`` and normally comes
    from ``FittedScorecard.monotone_constraints(list(X_train.columns))``. Passing
    None means no constraints, which is only appropriate when there is no
    champion to take trends from.
    """
    params = params or ChallengerParams()
    features = list(X_train.columns)
    if list(X_val.columns) != features:
        raise ValueError(
            "X_train and X_val must have identical columns in identical order; "
            f"train has {len(features)}, val has {len(X_val.columns)}."
        )
    if monotone_constraints is not None and len(monotone_constraints) != len(features):
        raise ValueError(
            f"monotone_constraints has {len(monotone_constraints)} entries for "
            f"{len(features)} features. It must align with X_train.columns."
        )
    if monotone_constraints is not None and not set(monotone_constraints) <= {-1, 0, 1}:
        raise ValueError(
            f"monotone_constraints must contain only -1, 0 or 1; got "
            f"{sorted(set(monotone_constraints))}."
        )

    kwargs = params.lgb_kwargs()
    if monotone_constraints is not None:
        kwargs["monotone_constraints"] = list(monotone_constraints)

    model = lgb.LGBMClassifier(**kwargs)
    model.fit(
        X_train[features],
        np.asarray(y_train),
        sample_weight=sample_weight,
        # eval_X / eval_y, not eval_set: LightGBM 4.7 deprecates the latter.
        eval_X=X_val[features],
        eval_y=np.asarray(y_val),
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(params.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    fitted = FittedChallenger(
        model=model,
        features=features,
        monotone_constraints=list(monotone_constraints or [0] * len(features)),
        best_iteration=getattr(model, "best_iteration_", None),
        params=params,
    )
    if calibrate:
        fitted.calibrator = IsotonicCalibrator().fit(
            fitted.predict_pd_raw(X_val), np.asarray(y_val)
        )
    return fitted
