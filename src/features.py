"""Feature construction in polars, lazy and streaming throughout.

Nothing here is written against the schema in the brief: every column name was
checked against the real data first, and the findings are recorded in
docs/DATA_NOTES.md. Two of them change how this module has to work:

* **Date columns are strings.** `date_decision` and every `D` column arrive as
  ISO strings, not dates, so they are parsed before any arithmetic.
* **The depth-2 bureau table is a monthly payment history.** Each row of
  `credit_bureau_a_2` is one contract-month, carrying `pmts_dpd_*` and
  `pmts_overdue_*` with the month and year it belongs to. That is what makes
  recency-weighted DPD and "DPD > 30 in the last N months" computable at all.

Post-decision records
---------------------
On the 3 % sample, 483,832 of the 5,744,322 depth-2 payment rows are dated
AFTER their own `date_decision`, and 132,759 of those carry a real DPD. Three
denominators are in play and they are easy to confuse: that is 8.4 % of all
rows, 24.1 % of the rows with a parseable payment month, and 10.7 % of the
rows carrying a DPD value, which is the figure MODEL.md quotes.

Taking the maximum post-decision DPD per case gives a univariate Gini of
+0.19 against the target over the 29,914 cases that have one: it is genuinely
predictive, because it is partly the outcome being predicted. Every record
dated after the decision is dropped, which is the brief's rule and is not a
formality here. `scripts/sample_recon.py` recomputes all of it.

Scope, stated because the brief's preparation step is broader
----------------------------------------------------------
The brief says to convert every date (D) column to days relative to
`date_decision` and drop the raw dates. Here dates are parsed and used to
EXCLUDE post-decision records, but they are not turned into features in their
own right: the six families below are built from named columns rather than by
aggregating everything the schema offers. That is a deliberate narrowing, a
blanket days-relative expansion would add roughly fifty largely redundant
columns to a set where the top seven already carry most of the signal, and it
is recorded here rather than left to be inferred from the absence.

Memory
------
`credit_bureau_a_2` is 188M rows on the full data. Nothing in this module calls
`collect()` on a depth table: everything is `scan_parquet` -> filter -> join ->
`group_by().agg()` with the streaming engine, so peak memory is bounded by the
aggregation state rather than the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from config import checked_kwargs

__all__ = [
    "FeatureParams",
    "DATE_FORMAT",
    "parse_date",
    "drop_post_decision",
    "safe_ratio",
    "recency_weight",
    "base_frame",
    "delinquency_features",
    "utilization_features",
    "inquiry_features",
    "previous_application_features",
    "affordability_features",
    "bureau_depth_features",
    "FAMILY_BUILDERS",
    "build_features",
    "table_paths",
    "find_data_dir",
    "ScreeningParams",
    "screen_features",
    "kept_features",
]

DATE_FORMAT = "%Y-%m-%d"
DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class FeatureParams:
    """From [features] in config."""

    recency_halflife_months: float = 12.0
    dpd_threshold: int = 30
    dpd_windows_months: tuple[int, ...] = (12, 24)
    inquiry_windows_days: tuple[int, ...] = (30, 90, 180, 360)
    thin_file_max_records: int = 3

    @classmethod
    def from_config(cls, section: dict) -> FeatureParams:
        tuples = {"dpd_windows_months", "inquiry_windows_days"}
        checked = checked_kwargs(section, cls, "features")
        return cls(**{k: (tuple(v) if k in tuples else v) for k, v in checked.items()})


# --- date handling -------------------------------------------------------


def parse_date(column: str) -> pl.Expr:
    """Parse an ISO date string column. Unparseable values become null."""
    return pl.col(column).str.to_date(DATE_FORMAT, strict=False)


def drop_post_decision(
    frame: pl.LazyFrame,
    date_columns: list[str],
    decision: str = "decision_date",
) -> pl.LazyFrame:
    """Drop rows holding any date later than their own decision date.

    A null date is kept: it carries no information about the future. Only a
    date that is present AND later than the decision is evidence the row could
    not have been seen at decision time.
    """
    if not date_columns:
        return frame
    condition = pl.lit(True)
    for column in date_columns:
        condition = condition & (pl.col(column).is_null() | (pl.col(column) <= pl.col(decision)))
    return frame.filter(condition)


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    """numerator / denominator, or NULL when the denominator is not positive.

    Never clip a denominator to a small epsilon to dodge a division by zero. It
    does avoid the error and replaces it with something worse: a finite number
    of enormous magnitude that looks like data. Measured on the sample, clipping
    the credit-limit denominator at 1e-9 produced a utilisation feature topping
    1.1e13, because 91 % of cases report a zero limit, a column that passes
    every screen and that the binning would happily split on.

    "Undefined" is the honest value, and the binning process has a Missing bin
    for exactly that.
    """
    return (
        pl.when(denominator.is_not_null() & (denominator > 0))
        .then(numerator / denominator)
        .otherwise(None)
    )


def recency_weight(months_ago: pl.Expr, halflife_months: float) -> pl.Expr:
    """Exponential decay with a half-life, clipped at zero months.

    Chosen over a linear ramp because it never goes negative for very old
    records and has one interpretable parameter.
    """
    if halflife_months <= 0:
        raise ValueError(f"halflife_months must be positive, got {halflife_months}.")
    decay = math.log(2.0) / halflife_months
    return (-decay * months_ago.clip(lower_bound=0.0)).exp()


def base_frame(sample_dir: str | Path) -> pl.LazyFrame:
    """The base table with `date_decision` parsed into `decision_date`."""
    path = Path(sample_dir) / "train_base.parquet"
    return pl.scan_parquet(path).with_columns(parse_date("date_decision").alias("decision_date"))


def table_paths(directory: str | Path, table: str) -> list[Path]:
    """Every parquet part belonging to one logical table, in part order.

    The local sample writes one file per table; the competition splits the
    larger ones across numbered parts (`train_static_0_0.parquet`,
    `train_credit_bureau_a_2_10.parquet`). Matching is exact-or-underscore-digits
    rather than a bare `table*` glob, which would let `train_static_0` swallow
    anything merely starting with that text.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    exact = directory / f"{table}.parquet"
    parts = sorted(
        (p for p in directory.glob(f"{table}_*.parquet") if p.stem[len(table) + 1 :].isdigit()),
        key=lambda p: int(p.stem[len(table) + 1 :]),
    )
    return ([exact] if exact.is_file() else []) + parts


def _scan(sample_dir: str | Path, table: str) -> pl.LazyFrame | None:
    paths = table_paths(sample_dir, table)
    if not paths:
        return None
    if len(paths) == 1:
        return pl.scan_parquet(paths[0])
    # Parts can disagree on a column's dtype (a column that is all-null in one
    # part reads as Null), so align schemas rather than letting concat fail.
    return pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")


def find_data_dir(root: str | Path, prefix: str = "train") -> Path:
    """Locate the directory holding `parquet_files/<prefix>` under a mount root.

    Competition data does not mount where the documentation implies: it appears
    at /kaggle/input/competitions/<slug>/, not /kaggle/input/<slug>/. Searching
    for the layout rather than assuming a path means a changed mount costs
    nothing. Falls back to `root` itself when it already holds the tables, which
    is how the local sample directory is passed in.
    """
    root = Path(root)
    if (root / "parquet_files" / prefix).is_dir():
        return root / "parquet_files" / prefix
    for candidate in sorted(root.rglob("parquet_files")):
        if (candidate / prefix).is_dir():
            return candidate / prefix
    return root


def _with_decision(frame: pl.LazyFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    return frame.join(base.select("case_id", "decision_date"), on="case_id", how="inner")


# --- families ------------------------------------------------------------


def delinquency_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """DPD statistics from the monthly bureau payment history.

    Aggregation runs depth 2 -> depth 1 -> case: payment months are first
    reduced per contract (`num_group1`), then contracts reduced per case, so a
    case with one badly delinquent contract is not averaged away by its many
    healthy ones.
    """
    frame = _scan(sample_dir, "train_credit_bureau_a_2")
    if frame is None:
        return None

    dpd_cols = [c for c in frame.collect_schema().names() if c.startswith("pmts_dpd_")]
    overdue_cols = [c for c in frame.collect_schema().names() if c.startswith("pmts_overdue_")]
    if not dpd_cols:
        return None

    month_col = "pmts_month_158T"
    year_col = "pmts_year_1139T"
    names = frame.collect_schema().names()
    if month_col not in names or year_col not in names:
        return None

    # One DPD and one overdue value per row: the table carries two variants of
    # each, and coalescing keeps whichever is populated.
    dpd = pl.coalesce([pl.col(c) for c in dpd_cols]).alias("dpd")
    overdue = (
        pl.coalesce([pl.col(c) for c in overdue_cols]).alias("overdue")
        if overdue_cols
        else pl.lit(None, dtype=pl.Float64).alias("overdue")
    )

    rows = (
        _with_decision(frame, base)
        .with_columns(dpd, overdue)
        .filter(pl.col("dpd").is_not_null())
        .with_columns(
            pl.date(
                pl.col(year_col).cast(pl.Int32, strict=False),
                pl.col(month_col).cast(pl.Int32, strict=False),
                1,
            ).alias("payment_date")
        )
        .filter(pl.col("payment_date").is_not_null())
        # The post-decision drop, applied to the payment month itself.
        .filter(pl.col("payment_date") <= pl.col("decision_date"))
        .with_columns(
            (
                (pl.col("decision_date") - pl.col("payment_date")).dt.total_days() / DAYS_PER_MONTH
            ).alias("months_ago")
        )
    )

    threshold = params.dpd_threshold
    window_exprs = [
        (
            ((pl.col("dpd") > threshold) & (pl.col("months_ago") <= window))
            .sum()
            .alias(f"dpd_gt{threshold}_last{window}m")
        )
        for window in params.dpd_windows_months
    ]

    weight = recency_weight(pl.col("months_ago"), params.recency_halflife_months)

    # depth 2 -> depth 1: one row per (case, contract).
    per_contract = rows.group_by("case_id", "num_group1").agg(
        pl.col("dpd").max().alias("dpd_max"),
        pl.col("dpd").mean().alias("dpd_mean"),
        pl.col("dpd").len().alias("n_payments"),
        (pl.col("dpd") * weight).sum().alias("_dpd_weighted_sum"),
        weight.sum().alias("_weight_sum"),
        pl.col("overdue").max().alias("overdue_max"),
        *window_exprs,
    )

    # depth 1 -> case.
    return per_contract.group_by("case_id").agg(
        pl.col("dpd_max").max().alias("delq_dpd_max"),
        pl.col("dpd_max").mean().alias("delq_dpd_max_mean_over_contracts"),
        pl.col("dpd_mean").mean().alias("delq_dpd_mean"),
        (safe_ratio(pl.col("_dpd_weighted_sum").sum(), pl.col("_weight_sum").sum())).alias(
            "delq_dpd_recency_weighted"
        ),
        pl.col("n_payments").sum().alias("delq_n_payment_months"),
        pl.col("overdue_max").max().alias("delq_overdue_max"),
        *[
            pl.col(f"dpd_gt{threshold}_last{w}m").sum().alias(f"delq_dpd_gt{threshold}_last{w}m")
            for w in params.dpd_windows_months
        ],
    )


def utilization_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """Balance divided by limit, from the depth-1 bureau table."""
    frame = _scan(sample_dir, "train_credit_bureau_a_1")
    if frame is None:
        return None
    names = frame.collect_schema().names()

    limits = [c for c in names if c.startswith("credlmt_")]
    balances = [c for c in names if c.startswith("outstandingamount_")]
    if not limits or not balances:
        return None

    date_cols = [c for c in names if c.endswith("D")]
    rows = (
        _with_decision(frame, base)
        .with_columns([parse_date(c).alias(c) for c in date_cols])
        .pipe(drop_post_decision, date_cols)
        .with_columns(
            pl.coalesce([pl.col(c) for c in limits]).alias("limit"),
            pl.coalesce([pl.col(c) for c in balances]).alias("balance"),
        )
        .with_columns(safe_ratio(pl.col("balance"), pl.col("limit")).alias("utilisation"))
    )
    return rows.group_by("case_id").agg(
        pl.col("utilisation").max().alias("util_max"),
        pl.col("utilisation").mean().alias("util_mean"),
        pl.col("balance").sum().alias("util_balance_total"),
        pl.col("limit").sum().alias("util_limit_total"),
        (safe_ratio(pl.col("balance").sum(), pl.col("limit").sum())).alias("util_overall"),
    )


def inquiry_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """Inquiry velocity.

    static_cb_0 already carries bureau-request counts over fixed windows
    (`days30_165L` … `days360_512L`), verified to be nested counts. They are
    taken as-is, plus the ratios that express acceleration: a file with 5 of its
    6 yearly inquiries in the last month is a different risk from one with them
    spread evenly.
    """
    frame = _scan(sample_dir, "train_static_cb_0")
    if frame is None:
        return None
    names = frame.collect_schema().names()

    windows: dict[int, str] = {}
    for window in params.inquiry_windows_days:
        match = [c for c in names if c.startswith(f"days{window}_")]
        if match:
            windows[window] = match[0]
    if not windows:
        return None

    selected = frame.select("case_id", *windows.values())
    exprs = [pl.col(col).alias(f"inq_last{w}d") for w, col in windows.items()]

    longest = max(windows)
    ratios = [
        safe_ratio(pl.col(col), pl.col(windows[longest])).alias(f"inq_share_last{w}d")
        for w, col in windows.items()
        if w != longest
    ]
    return selected.with_columns(*exprs, *ratios).select(
        "case_id", *[e.meta.output_name() for e in exprs + ratios]
    )


def previous_application_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """Count, share rejected, and approved amounts from prior applications."""
    frame = _scan(sample_dir, "train_applprev_1")
    if frame is None:
        return None
    names = frame.collect_schema().names()

    date_cols = [c for c in names if c.endswith("D")]
    rows = (
        _with_decision(frame, base)
        .with_columns([parse_date(c).alias(c) for c in date_cols])
        .pipe(drop_post_decision, date_cols)
    )

    status_col = next((c for c in names if c.startswith("status_")), None)
    reject_col = next((c for c in names if c.startswith("rejectreason_")), None)
    amount_cols = [c for c in names if c.startswith("credamount_")]

    aggs = [pl.len().alias("prev_n_applications")]
    if status_col:
        # Status is a masked category; "approved" cannot be assumed from its
        # value, so rejection is read from the reject-reason field instead.
        aggs.append(pl.col(status_col).n_unique().alias("prev_n_distinct_status"))
    if reject_col:
        rejected = pl.col(reject_col).is_not_null() & (pl.col(reject_col) != "a55475b1")
        aggs += [
            rejected.sum().alias("prev_n_rejected"),
            rejected.mean().alias("prev_share_rejected"),
        ]
    if amount_cols:
        amount = pl.coalesce([pl.col(c) for c in amount_cols])
        aggs += [
            amount.max().alias("prev_amount_max"),
            amount.mean().alias("prev_amount_mean"),
            amount.sum().alias("prev_amount_total"),
        ]
    return rows.group_by("case_id").agg(*aggs)


def affordability_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """annuity/income and credit/income.

    There is no income column in this data (docs/DATA_NOTES.md): the only direct
    match is a category. Income is therefore DERIVED as the total of tax-registry
    payment amounts per case, which is a proxy with its own assumptions, the
    feature names say `taxincome` rather than `income` so that nothing downstream
    mistakes it for declared income.
    """
    static = _scan(sample_dir, "train_static_0")
    if static is None:
        return None
    names = static.collect_schema().names()
    annuity_col = next((c for c in names if c.startswith("annuity_")), None)
    credit_col = next((c for c in names if c.startswith("credamount_")), None)

    income = None
    for table, prefix in (
        ("train_tax_registry_c_1", "pmtamount_"),
        ("train_tax_registry_a_1", "amount_"),
        ("train_tax_registry_b_1", "amount_"),
    ):
        frame = _scan(sample_dir, table)
        if frame is None:
            continue
        cols = [c for c in frame.collect_schema().names() if c.startswith(prefix)]
        if not cols:
            continue
        part = frame.group_by("case_id").agg(
            pl.coalesce([pl.col(c) for c in cols]).sum().alias(f"_inc_{table}")
        )
        income = (
            part if income is None else income.join(part, on="case_id", how="full", coalesce=True)
        )

    out = static.select(
        "case_id",
        *([pl.col(annuity_col).alias("afford_annuity")] if annuity_col else []),
        *([pl.col(credit_col).alias("afford_credit_amount")] if credit_col else []),
    )
    if income is not None:
        inc_cols = [c for c in income.collect_schema().names() if c.startswith("_inc_")]
        # A case absent from EVERY tax registry has unknown income, not income
        # of zero. fill_null(0) inside the sum is right; one missing registry
        # should not void the total, but applying it when all are missing
        # recorded 522,900 cases (34%) as earning exactly nothing, and the
        # binning then filed them in the lowest income bin beside genuinely low
        # earners rather than in its Missing bin. Those two groups are not the
        # same, and conflating them is what let "no employment record" stand in
        # for "no income", which, for the 92% of over-60s who are simply
        # retired, is false.
        present = pl.any_horizontal([pl.col(c).is_not_null() for c in inc_cols])
        out = out.join(income, on="case_id", how="left").with_columns(
            pl.when(present)
            .then(pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in inc_cols]))
            .otherwise(None)
            .alias("afford_taxincome_total")
        )
        denominator = pl.col("afford_taxincome_total")
        ratios = []
        if annuity_col:
            ratios.append(
                safe_ratio(pl.col("afford_annuity"), denominator).alias(
                    "afford_annuity_to_taxincome"
                )
            )
        if credit_col:
            ratios.append(
                safe_ratio(pl.col("afford_credit_amount"), denominator).alias(
                    "afford_credit_to_taxincome"
                )
            )
        out = out.with_columns(*ratios).drop(inc_cols)
    return out


def bureau_depth_features(
    sample_dir: str | Path, base: pl.LazyFrame, params: FeatureParams
) -> pl.LazyFrame | None:
    """Number of bureau records, and a thin-file flag."""
    parts = []
    for table, name in (
        ("train_credit_bureau_a_1", "bureau_a_records"),
        ("train_credit_bureau_b_1", "bureau_b_records"),
    ):
        frame = _scan(sample_dir, table)
        if frame is None:
            continue
        parts.append(frame.group_by("case_id").agg(pl.len().alias(name)))
    if not parts:
        return None

    out = parts[0]
    for extra in parts[1:]:
        out = out.join(extra, on="case_id", how="full", coalesce=True)

    counts = [c for c in out.collect_schema().names() if c.endswith("_records")]
    total = pl.sum_horizontal([pl.col(c).fill_null(0) for c in counts])
    return out.with_columns(
        *[pl.col(c).fill_null(0).alias(c) for c in counts],
        total.alias("bureau_total_records"),
        (total <= params.thin_file_max_records).cast(pl.Int8).alias("bureau_thin_file"),
    )


FAMILY_BUILDERS = {
    "delinquency": delinquency_features,
    "utilization": utilization_features,
    "inquiry": inquiry_features,
    "previous_applications": previous_application_features,
    "affordability": affordability_features,
    "bureau_depth": bureau_depth_features,
}


def build_features(
    sample_dir: str | Path,
    params: FeatureParams | None = None,
    *,
    out_path: str | Path | None = None,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Build every family and join them onto the base table.

    Returns the float32 feature frame and a map from feature name to family, so
    the feature report can attribute each column without re-deriving it.
    """
    params = params or FeatureParams()
    base = base_frame(sample_dir)
    out = base.select("case_id", "WEEK_NUM", "target")
    family_of: dict[str, str] = {}

    for family, builder in FAMILY_BUILDERS.items():
        frame = builder(sample_dir, base, params)
        if frame is None:
            continue
        columns = [c for c in frame.collect_schema().names() if c != "case_id"]
        if not columns:
            continue
        for column in columns:
            family_of[column] = family
        out = out.join(frame, on="case_id", how="left")

    collected = out.collect(engine="streaming")
    # float32 on every feature: halves the parquet and keeps /kaggle/working
    # inside its limit. Keys and target stay integral.
    feature_cols = [c for c in collected.columns if c not in ("case_id", "WEEK_NUM", "target")]
    # float32 on every feature, with non-finite values mapped to null. The
    # ratios are already guarded, but an infinity from any source would survive
    # screening (is_null() is false for inf) and poison the binning, so it is
    # caught once, here, rather than trusted not to arise.
    collected = collected.with_columns(
        [
            pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).cast(pl.Float32).alias(c)
            for c in feature_cols
        ]
    )
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        collected.write_parquet(out_path)
    return collected, family_of


def kept_features(report_path: str | Path) -> list[str]:
    """The features that survived screening, read from a feature report.

    build_features deliberately emits EVERY feature it constructs, so the report
    and the parquet can be inspected together. That makes it the caller's job to
    apply the screen, and a caller that forgets simply re-selects from the full
    set. Observed exactly that: the champion scorecard picked up
    bureau_a_records, which screening had dropped for PSI 0.614, because its IV
    was healthy and nothing enforced the earlier decision. In a project about
    model stability, that is the one feature the screen existed to exclude.
    """
    import pandas as pd

    report = pd.read_csv(report_path)
    missing = {"feature", "kept"} - set(report.columns)
    if missing:
        raise ValueError(f"{report_path} is missing {sorted(missing)}.")
    kept = report.loc[report["kept"].astype(bool), "feature"].tolist()
    if not kept:
        raise ValueError(f"{report_path} marks no feature as kept.")
    return kept


# --- screening -----------------------------------------------------------


@dataclass(frozen=True)
class ScreeningParams:
    """From [screening] in config."""

    max_missing_rate: float = 0.95
    psi_drop: float = 0.25
    psi_flag: float = 0.10
    leakage_gini: float = 0.50
    psi_bins: int = 10

    @classmethod
    def from_config(cls, section: dict) -> ScreeningParams:
        return cls(**checked_kwargs(section, cls, "screening"))


def screen_features(
    features: pl.DataFrame,
    family_of: dict[str, str],
    split_of_week: dict[int, str],
    params: ScreeningParams | None = None,
    *,
    min_iv: float = 0.02,
) -> pd.DataFrame:  # noqa: F821
    """Fit every screening decision on the TRAIN weeks only.

    The rules, from the brief:

      * drop when the train-week missing rate exceeds ``max_missing_rate``
      * PSI of train weeks against validation weeks (A and B together):
        above ``psi_drop`` drop, between ``psi_flag`` and ``psi_drop`` flag
      * report univariate Gini above ``leakage_gini`` for manual review

    Test weeks are never touched. Not once, not for a sanity check: a feature
    kept or dropped because of how it behaves on the test weeks turns the
    out-of-time evaluation into an in-sample one, and nothing downstream would
    reveal it. Test-week PSI belongs to evaluation and is computed there.

    A high univariate Gini is FLAGGED, never dropped automatically. A genuinely
    strong feature and a leak look identical to this test, and only a human who
    knows what the column means can tell them apart.
    """
    import pandas as pd

    from metrics import gini, psi

    params = params or ScreeningParams()
    feature_cols = [c for c in features.columns if c not in ("case_id", "WEEK_NUM", "target")]
    split = features["WEEK_NUM"].replace_strict(
        split_of_week, default="unknown", return_dtype=pl.Utf8
    )
    train_mask = (split == "train").to_numpy()
    val_mask = ((split == "val_a") | (split == "val_b")).to_numpy()

    if not train_mask.any():
        raise ValueError("No train weeks present; screening cannot be fitted.")

    y_train = features["target"].to_numpy()[train_mask]
    rows = []
    for name in feature_cols:
        values = features[name].to_numpy()
        train_values = values[train_mask]
        val_values = values[val_mask]

        missing = float(pl.Series(train_values).is_null().mean() or 0.0)
        missing = float(np.isnan(train_values).mean()) if missing == 0.0 else missing

        psi_value = (
            psi(train_values, val_values, bins=params.psi_bins) if val_mask.any() else float("nan")
        )
        finite = ~np.isnan(train_values)
        uni_gini = (
            gini(y_train[finite], train_values[finite]) if finite.sum() > 10 else float("nan")
        )

        kept, reason = True, "kept"
        if missing > params.max_missing_rate:
            kept, reason = False, f"missing {missing:.1%} > {params.max_missing_rate:.0%}"
        elif not np.isnan(psi_value) and psi_value > params.psi_drop:
            kept, reason = False, f"PSI {psi_value:.3f} > {params.psi_drop}"
        elif not np.isnan(psi_value) and psi_value > params.psi_flag:
            reason = f"kept, FLAGGED: PSI {psi_value:.3f} in [{params.psi_flag}, {params.psi_drop}]"

        leak = bool(not np.isnan(uni_gini) and abs(uni_gini) > params.leakage_gini)
        if leak:
            reason += f" | REVIEW: univariate Gini {uni_gini:+.3f}"

        rows.append(
            {
                "feature": name,
                "family": family_of.get(name, "unknown"),
                "missing_pct": round(100 * missing, 3),
                "iv": float("nan"),
                "psi_train_vs_val": psi_value,
                "univariate_gini": uni_gini,
                "kept": kept,
                "leakage_review": leak,
                "reason": reason,
            }
        )

    report = pd.DataFrame(rows)

    # IV, from optbinning, on the surviving features only: it is the expensive
    # step and a dropped feature's IV is not a decision input.
    survivors = report.loc[report["kept"], "feature"].tolist()
    if survivors:
        from scorecard import ScorecardParams, iv_table

        # Train rows only, matching y_train. Passing the whole frame with a
        # train-only target silently misaligns the two.
        frame = features.filter(pl.Series(train_mask)).select(survivors).to_pandas()
        ivs = iv_table(frame, y_train, ScorecardParams()).set_index("feature")["iv"]
        report["iv"] = report["feature"].map(ivs).astype(float)
        low_iv = report["kept"] & (report["iv"] < min_iv)
        report.loc[low_iv, "kept"] = False
        # Keep the leakage annotation when the IV rule drops a feature. A
        # PERFECTLY separating column comes back from optbinning as a single bin
        # with IV 0.0, so the IV screen reads a total leak as useless noise and
        # drops it, taking the one signal that would have surfaced it. The
        # univariate Gini is what catches that case, and it has to survive here.
        report.loc[low_iv, "reason"] = report.loc[low_iv].apply(
            lambda r: (
                f"IV {r['iv']:.4f} < {min_iv}"
                + (
                    f" | REVIEW: univariate Gini {r['univariate_gini']:+.3f}"
                    if r["leakage_review"]
                    else ""
                )
            ),
            axis=1,
        )
    return report.sort_values(["kept", "iv"], ascending=[False, False]).reset_index(drop=True)
