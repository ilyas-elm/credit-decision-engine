"""Champion model: a binned, scaled logistic-regression scorecard.

This is the model a credit risk team would actually sign off on: monotone bins,
weight-of-evidence transforms, a linear model, and an integer-ish point scale
that a reviewer can read off a table. It is the baseline the LightGBM challenger
has to beat on more than AUC.

pandas appears here and nowhere upstream. optbinning requires a DataFrame, and
this is the model boundary the brief allows it at; the feature build stays in
polars.

Three details verified against the pinned optbinning (1.0.0) rather than assumed:

1. `monotonic_trend` is not a `BinningProcess` constructor argument. It is an
   `OptimalBinning` parameter, passed per variable through `binning_fit_params`.
2. `Scorecard` CLONES the `BinningProcess` it is given, so the instance handed in
   is never fitted. IV and univariate Gini come from `Scorecard.binning_process_`.
3. After fitting, `monotonic_trend` still reads `"auto_asc_desc"`, optbinning
   does not write back which direction it resolved to. So the trend direction
   used for the challenger's monotone constraints is derived from the event-rate
   sequence across bins, not read off the estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from optbinning import BinningProcess, Scorecard
from sklearn.linear_model import LogisticRegression

from config import checked_kwargs

__all__ = [
    "ScorecardParams",
    "iv_table",
    "select_features",
    "FittedScorecard",
    "fit_scorecard",
]

# Rows optbinning appends for out-of-range and null handling. They carry no
# ordering, so every trend or monotonicity judgement must exclude them.
NON_INTERVAL_BINS = ("Special", "Missing")


@dataclass(frozen=True)
class ScorecardParams:
    """Binning, selection and scaling parameters, from [scorecard] in config."""

    monotonic_trend: str = "auto_asc_desc"
    min_iv: float = 0.02
    max_abs_corr: float = 0.80
    scaling_method: str = "pdo_odds"
    points_at_odds: float = 600.0
    target_odds: float = 50.0
    pdo: float = 20.0
    excluded_characteristics: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, section: dict) -> ScorecardParams:
        checked_kwargs(section, cls, "scorecard")
        return cls(
            monotonic_trend=section["monotonic_trend"],
            min_iv=section["min_iv"],
            max_abs_corr=section["max_abs_corr"],
            scaling_method=section["scaling_method"],
            points_at_odds=section["points_at_odds"],
            target_odds=section["target_odds"],
            pdo=section["pdo"],
            excluded_characteristics=tuple(section.get("excluded_characteristics", ())),
        )

    def scaling_params(self) -> dict[str, float]:
        """optbinning requires exactly these three keys for ``pdo_odds``."""
        return {
            "pdo": self.pdo,
            "odds": self.target_odds,
            "scorecard_points": self.points_at_odds,
        }


def _binning_process(features: list[str], params: ScorecardParams) -> BinningProcess:
    return BinningProcess(
        variable_names=features,
        binning_fit_params={v: {"monotonic_trend": params.monotonic_trend} for v in features},
    )


def iv_table(
    X: pd.DataFrame,
    y: np.ndarray,
    params: ScorecardParams | None = None,
) -> pd.DataFrame:
    """Per-feature IV and univariate Gini, fitted on train weeks only.

    Returns the columns the feature report needs: feature, iv, gini, n_bins,
    dtype. Univariate Gini is the leakage screen, anything above
    [screening].leakage_gini is printed for manual review rather than dropped,
    because a legitimately strong feature and a leak look identical here.
    """
    params = params or ScorecardParams()
    features = list(X.columns)
    process = _binning_process(features, params)
    process.fit(X, np.asarray(y))

    summary = process.summary()
    out = summary.rename(columns={"name": "feature"})[
        ["feature", "dtype", "n_bins", "iv", "gini"]
    ].copy()
    out["iv"] = out["iv"].astype(float)
    out["gini"] = out["gini"].astype(float)
    return out.sort_values("iv", ascending=False).reset_index(drop=True)


def _woe_frame(X: pd.DataFrame, y: np.ndarray, params: ScorecardParams) -> pd.DataFrame:
    """WoE-transformed features, for the correlation screen."""
    features = list(X.columns)
    process = _binning_process(features, params)
    process.fit(X, np.asarray(y))
    transformed = process.transform(X, metric="woe")
    return pd.DataFrame(np.asarray(transformed), columns=features, index=X.index)


def select_features(
    X: pd.DataFrame,
    y: np.ndarray,
    params: ScorecardParams | None = None,
    *,
    ivs: pd.DataFrame | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Apply the brief's two selection rules, and say why each feature went.

    1. Drop IV < ``min_iv``.
    2. For pairs with ``|corr| > max_abs_corr``, keep the higher-IV feature.

    Correlation is measured on the WoE-transformed values, not the raw ones:
    WoE is what the logistic regression actually sees, and it is defined for
    categorical characteristics, which have no raw correlation at all.

    Features are considered in descending IV order, so a dropped feature is
    always the weaker member of the pair that evicted it.
    """
    params = params or ScorecardParams()
    ivs = iv_table(X, y, params) if ivs is None else ivs.copy()
    # Sorted here, not assumed. The correlation rule walks the candidates in
    # order and keeps the first of each clashing pair, so the order IS the
    # rule: given the same IV table with its rows reversed, the pruning would
    # keep the weaker feature of every pair and still look like it had applied
    # the brief. `iv_table` already returns descending order, so this only
    # matters for a caller that builds the table some other way.
    ivs = ivs.sort_values("iv", ascending=False).reset_index(drop=True)

    decisions = ivs.set_index("feature").copy()
    decisions["kept"] = False
    decisions["reason"] = ""

    # Barred outright, before any other rule: a characteristic whose fitted
    # coefficient inverts its univariate relationship makes the table unreadable
    # however strong its IV.
    barred = [f for f in decisions.index if f in set(params.excluded_characteristics)]
    decisions.loc[barred, "reason"] = "excluded: sign flip (see config)"

    weak = decisions.index[(decisions["iv"] < params.min_iv) & (~decisions.index.isin(barred))]
    decisions.loc[weak, "reason"] = f"iv below {params.min_iv}"

    candidates = [f for f in decisions.index if f not in set(weak) | set(barred)]
    if not candidates:
        return [], decisions.reset_index()

    woe = _woe_frame(X[candidates], y, params)
    corr = woe.corr().abs()

    kept: list[str] = []
    for feature in candidates:  # already in descending IV order
        clash = next(
            (k for k in kept if corr.loc[feature, k] > params.max_abs_corr),
            None,
        )
        if clash is None:
            kept.append(feature)
            decisions.loc[feature, "kept"] = True
            decisions.loc[feature, "reason"] = "kept"
        else:
            decisions.loc[feature, "reason"] = (
                f"|corr| {corr.loc[feature, clash]:.2f} > {params.max_abs_corr} "
                f"with {clash} (higher IV)"
            )
    return kept, decisions.reset_index()


def _interval_rows(table: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows = table[table["Variable"] == feature]
    return rows[~rows["Bin"].astype(str).isin(NON_INTERVAL_BINS)]


@dataclass
class FittedScorecard:
    """A fitted scorecard plus the artefacts the decision layer needs."""

    model: Scorecard
    features: list[str]
    params: ScorecardParams
    table: pd.DataFrame = field(repr=False)

    # --- predictions ---

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default, as the logistic regression gives it."""
        return np.asarray(self.model.predict_proba(X[self.features])[:, 1], dtype=float)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Total points. Higher points mean lower risk under ``pdo_odds``."""
        return np.asarray(self.model.score(X[self.features]), dtype=float)

    # --- artefacts ---

    def points_table(self) -> pd.DataFrame:
        """characteristic / bin / WoE / points, the signed-off scorecard."""
        out = self.table.rename(
            columns={
                "Variable": "characteristic",
                "Bin": "bin",
                "WoE": "woe",
                "Points": "points",
                "Event rate": "event_rate",
                "Count": "count",
                "IV": "iv",
            }
        )
        keep = ["characteristic", "bin", "count", "event_rate", "woe", "iv", "points"]
        return out[[c for c in keep if c in out.columns]].reset_index(drop=True)

    def max_points(self) -> dict[str, float]:
        """Best achievable points per characteristic, over its interval bins."""
        return {
            feature: float(_interval_rows(self.table, feature)["Points"].max())
            for feature in self.features
        }

    def trend_direction(self) -> dict[str, int]:
        """+1 if the event rate rises with the feature, -1 if it falls, else 0.

        Derived from the binning table because optbinning does not record which
        direction ``auto_asc_desc`` resolved to. A feature whose event rate is
        not monotone across bins gets 0 rather than a guessed sign: a wrong
        monotone constraint on the challenger is worse than none.

        A feature whose IV falls below ``params.min_iv`` also gets 0, however
        monotone it looks. Binned into a handful of bins, pure noise is
        frequently monotone by chance, on the test fixtures a noise column
        drew a confident -1, and constraining the challenger on that is
        constraining it on nothing. The same configured threshold that decides
        whether a feature is worth selecting decides whether its trend is worth
        believing.
        """
        directions: dict[str, int] = {}
        for feature in self.features:
            rows = _interval_rows(self.table, feature)
            rates = rows["Event rate"].to_numpy(dtype=float)
            rates = rates[~np.isnan(rates)]
            feature_iv = float(rows["IV"].astype(float).sum())
            if rates.size < 2 or feature_iv < self.params.min_iv:
                directions[feature] = 0
                continue
            diffs = np.diff(rates)
            if np.all(diffs >= 0) and rates[-1] > rates[0]:
                directions[feature] = 1
            elif np.all(diffs <= 0) and rates[-1] < rates[0]:
                directions[feature] = -1
            else:
                directions[feature] = 0
        return directions

    def sign_flips(self) -> dict[str, float]:
        """Characteristics whose points move the WRONG way against their own risk.

        Within a characteristic, a bin with a higher event rate must score FEWER
        points. When the logistic regression gives a characteristic a negative
        coefficient, usually because it shares its signal with correlated
        characteristics, the multivariate scorecard inverts the univariate
        relationship, and the table starts saying things like "the more you
        previously borrowed, the safer you are".

        Credit model review rejects those: the scorecard stops being readable,
        which is the whole reason for preferring one to a black box. Returns the
        offenders mapped to their event-rate/points correlation, which is ~-1 for
        a well-behaved characteristic and ~+1 for a flipped one.

        Reported rather than silently dropped. Removing a characteristic changes
        the model, and that is a decision for a person.
        """
        flips: dict[str, float] = {}
        for feature in self.features:
            rows = _interval_rows(self.table, feature)
            if len(rows) < 3:
                continue
            rates = rows["Event rate"].to_numpy(dtype=float)
            points = rows["Points"].to_numpy(dtype=float)
            if np.std(rates) < 1e-12 or np.std(points) < 1e-12:
                continue
            correlation = float(np.corrcoef(rates, points)[0, 1])
            # Correct behaviour is a NEGATIVE correlation: more risk, fewer points.
            if correlation > 0:
                flips[feature] = correlation
        return flips

    def monotone_constraints(self, feature_order: list[str]) -> list[int]:
        """LightGBM ``monotone_constraints``, aligned to a given column order.

        The challenger predicts PD, so a characteristic whose event rate rises
        with its value gets +1. Features the scorecard never binned get 0.
        """
        directions = self.trend_direction()
        return [int(directions.get(feature, 0)) for feature in feature_order]

    def bin_labels(self, X: pd.DataFrame) -> pd.DataFrame:
        """Which bin each row falls in, per characteristic, as the table names it.

        ``metric="bins"`` returns the bin label itself, which joins exactly to
        the ``Bin`` column. ``metric="indices"`` is not usable here: it returns 0
        for a Missing value, colliding with real bin 0.
        """
        labels = self.model.binning_process_.transform(X[self.features], metric="bins")
        return pd.DataFrame(np.asarray(labels, dtype=object), columns=self.features, index=X.index)

    def points_by_bin(self, feature: str, *, intervals_only: bool = False) -> dict[str, float]:
        """Bin label to points, for one characteristic.

        ``intervals_only`` drops the Special and Missing rows. Recourse needs
        that: "withhold this value" is not advice, so those bins are not
        destinations an applicant can be told to move to.
        """
        rows = (
            _interval_rows(self.table, feature)
            if intervals_only
            else (self.table[self.table["Variable"] == feature])
        )
        return dict(zip(rows["Bin"].astype(str), rows["Points"].astype(float), strict=True))

    def feature_points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points each characteristic awards each row."""
        labels = self.bin_labels(X)
        out = {}
        for feature in self.features:
            mapping = self.points_by_bin(feature)
            column = labels[feature].astype(str)
            unknown = set(column.unique()) - set(mapping)
            if unknown:
                raise ValueError(
                    f"Bin label(s) {sorted(unknown)} for {feature!r} are absent from the "
                    "scorecard table, so points cannot be assigned. optbinning's bin "
                    "labelling has changed."
                )
            out[feature] = column.map(mapping).to_numpy(dtype=float)
        return pd.DataFrame(out, index=X.index)

    def points_below_max(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points lost per characteristic, against the best bin available.

        This is the principal-reasons calculation: the drivers of a decline are
        the characteristics where the applicant gave away the most points. It
        reports the TRUE drivers, including ones the applicant cannot act on,
        keeping honest explanation separate from actionable advice is the job of
        the recourse step, not this one.

        The reference is the best INTERVAL bin, i.e. the best outcome available
        to an applicant who has a real value for the characteristic. A row
        sitting in Special or Missing can therefore score above that reference,
        so the shortfall is floored at zero rather than reported negative.
        """
        best = self.max_points()
        awarded = self.feature_points(X)
        shortfall = {
            feature: np.maximum(best[feature] - awarded[feature].to_numpy(), 0.0)
            for feature in self.features
        }
        return pd.DataFrame(shortfall, index=X.index)

    def principal_reasons(
        self, X: pd.DataFrame, *, top_n: int = 4
    ) -> list[list[tuple[str, float]]]:
        """The ``top_n`` characteristics costing each row the most points."""
        shortfall = self.points_below_max(X)
        reasons = []
        for _, row in shortfall.iterrows():
            ranked = row.sort_values(ascending=False)
            reasons.append([(str(k), float(v)) for k, v in ranked.head(top_n).items()])
        return reasons


def fit_scorecard(
    X: pd.DataFrame,
    y: np.ndarray,
    params: ScorecardParams | None = None,
    *,
    features: list[str] | None = None,
    sample_weight: np.ndarray | None = None,
) -> FittedScorecard:
    """Fit binning, logistic regression and point scaling in one pass.

    ``sample_weight`` is passed through to both the binning and the regression.
    Verified supported by this optbinning: ``BinningProcess.fit`` and
    ``Scorecard.fit`` both accept it, so the weighted arms of the sampling-bias
    experiment need no fallback.
    """
    params = params or ScorecardParams()
    features = list(X.columns) if features is None else list(features)
    if not features:
        raise ValueError("fit_scorecard needs at least one feature.")
    missing = [f for f in features if f not in X.columns]
    if missing:
        raise ValueError(f"Features absent from X: {missing}.")

    model = Scorecard(
        binning_process=_binning_process(features, params),
        estimator=LogisticRegression(max_iter=1000),
        scaling_method=params.scaling_method,
        scaling_method_params=params.scaling_params(),
    )
    model.fit(X[features], np.asarray(y), sample_weight=sample_weight)
    return FittedScorecard(
        model=model,
        features=features,
        params=params,
        table=model.table(style="detailed"),
    )
