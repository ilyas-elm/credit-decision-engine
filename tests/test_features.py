"""The feature build: date handling, the families, and the screening rules."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import polars as pl
import pytest
from fixtures import make_dataset, write_dataset

from config import load_config
from features import (
    FeatureParams,
    ScreeningParams,
    build_features,
    drop_post_decision,
    parse_date,
    recency_weight,
    safe_ratio,
    screen_features,
)
from splits import split_weeks

# --- safe_ratio: the bug this exists to prevent -------------------------


def test_safe_ratio_returns_null_rather_than_a_huge_number() -> None:
    """Clipping a denominator to an epsilon is worse than dividing by zero.

    Measured on the real sample, a clipped credit-limit denominator produced a
    utilisation feature topping 1.1e13, because 91% of cases report a zero
    limit. That column passed every screen and the binning would have split on
    it. "Undefined" is the honest answer.
    """
    frame = pl.DataFrame({"n": [10.0, 10.0, 10.0, 10.0], "d": [2.0, 0.0, -1.0, None]})
    out = frame.select(safe_ratio(pl.col("n"), pl.col("d")).alias("r"))["r"].to_list()
    assert out[0] == pytest.approx(5.0)
    assert out[1] is None, "zero denominator must be null, not 1e9-scale"
    assert out[2] is None, "negative denominator must be null"
    assert out[3] is None, "null denominator must be null"


def test_safe_ratio_guards_zero_but_not_denormal_denominators() -> None:
    """The exact scope of the guard, stated rather than assumed.

    safe_ratio rules out zero, negative and null denominators -- the cases that
    occur in this data, where denominators are counts and currency amounts. It
    does NOT rule out an overflow from a denormal denominator; nothing here
    produces one, and build_features maps any non-finite value to null as a
    final catch, so the guarantee lives there rather than in an arbitrary
    magnitude threshold here.
    """
    frame = pl.DataFrame({"n": [1e30], "d": [1e-300]})
    value = frame.select(safe_ratio(pl.col("n"), pl.col("d")).alias("r"))["r"][0]
    assert math.isinf(value), "scope of the guard has changed; update the docstring"


# --- dates ---------------------------------------------------------------


def test_iso_date_strings_are_parsed_and_bad_values_become_null() -> None:
    frame = pl.DataFrame({"d": ["2020-03-01", None, "not-a-date"]})
    out = frame.select(parse_date("d").alias("p"))["p"].to_list()
    assert out[0] == dt.date(2020, 3, 1)
    assert out[1] is None and out[2] is None


def test_post_decision_rows_are_dropped_and_nulls_are_kept() -> None:
    """A null date carries no information about the future, so it stays.

    Only a date that is present AND later than the decision proves the row
    could not have been seen at decision time.
    """
    frame = pl.DataFrame(
        {
            "case_id": [1, 2, 3, 4],
            "decision_date": [dt.date(2020, 6, 1)] * 4,
            "event": [
                dt.date(2020, 1, 1),  # before: keep
                dt.date(2020, 6, 1),  # same day: keep
                dt.date(2020, 9, 1),  # after: drop
                None,  # unknown: keep
            ],
        }
    ).lazy()
    kept = drop_post_decision(frame, ["event"]).collect()["case_id"].to_list()
    assert kept == [1, 2, 4]


def test_drop_post_decision_checks_every_date_column() -> None:
    frame = pl.DataFrame(
        {
            "case_id": [1, 2],
            "decision_date": [dt.date(2020, 6, 1)] * 2,
            "a": [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            "b": [dt.date(2020, 1, 1), dt.date(2021, 1, 1)],
        }
    ).lazy()
    kept = drop_post_decision(frame, ["a", "b"]).collect()["case_id"].to_list()
    assert kept == [1]


def test_no_date_columns_is_a_no_op() -> None:
    frame = pl.DataFrame({"case_id": [1, 2]}).lazy()
    assert drop_post_decision(frame, []).collect().height == 2


# --- recency weighting ---------------------------------------------------


def test_recency_weight_halves_at_the_half_life() -> None:
    frame = pl.DataFrame({"m": [0.0, 12.0, 24.0, -5.0]})
    w = frame.select(recency_weight(pl.col("m"), 12.0).alias("w"))["w"].to_list()
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5)
    assert w[2] == pytest.approx(0.25)
    # Negative months (a record dated after the decision) clamp to the present
    # rather than being up-weighted above 1.
    assert w[3] == pytest.approx(1.0)


def test_recency_weight_rejects_a_non_positive_half_life() -> None:
    with pytest.raises(ValueError, match="halflife_months must be positive"):
        recency_weight(pl.col("m"), 0.0)


# --- build ---------------------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Fixtures written in the competition's layout, then built."""
    directory = tmp_path_factory.mktemp("sample")
    tables, _ = make_dataset(n_cases=900, n_weeks=24, seed=4)
    # The builders read real competition table names.
    renamed = {
        "train_base": tables["train_base"],
        "train_static_0": tables["train_static_0_0"],
        "train_applprev_1": tables["train_applprev_1_0"],
        "train_credit_bureau_a_2": tables["train_applprev_2"],
    }
    write_dataset(directory, renamed)
    feats, family = build_features(directory, FeatureParams())
    return feats, family, directory


def test_build_returns_one_row_per_case_in_float32(built) -> None:
    feats, family, _ = built
    assert feats.height == 900
    assert feats["case_id"].n_unique() == 900
    assert {"case_id", "WEEK_NUM", "target"} <= set(feats.columns)
    feature_cols = [c for c in feats.columns if c not in ("case_id", "WEEK_NUM", "target")]
    assert feature_cols, "no features were built"
    assert all(feats.schema[c] == pl.Float32 for c in feature_cols)


def test_every_feature_is_attributed_to_a_family(built) -> None:
    feats, family, _ = built
    feature_cols = [c for c in feats.columns if c not in ("case_id", "WEEK_NUM", "target")]
    assert set(feature_cols) == set(family)
    assert all(v != "unknown" for v in family.values())


def test_built_features_contain_no_infinities(built) -> None:
    feats, _, _ = built
    for column in feats.columns:
        if feats.schema[column] == pl.Float32:
            values = feats[column].to_numpy()
            assert not np.isinf(values[~np.isnan(values)]).any(), column


def test_build_writes_a_parquet_when_asked(built, tmp_path) -> None:
    _, _, directory = built
    out = tmp_path / "f.parquet"
    frame, _ = build_features(directory, FeatureParams(), out_path=out)
    assert out.is_file()
    assert pl.read_parquet(out).height == frame.height


def test_missing_tables_are_skipped_not_fatal(tmp_path) -> None:
    """Only the base table is required; a family with no source is omitted."""
    tables, _ = make_dataset(n_cases=200, n_weeks=8, seed=2)
    write_dataset(tmp_path, {"train_base": tables["train_base"]})
    feats, family = build_features(tmp_path, FeatureParams())
    assert feats.height == 200
    assert family == {} or all(v != "unknown" for v in family.values())


# --- screening -----------------------------------------------------------


@pytest.fixture
def screening_frame():
    """A frame with one feature per screening outcome, by construction."""
    rng = np.random.default_rng(0)
    n_weeks, per_week = 40, 60
    weeks = np.repeat(np.arange(n_weeks), per_week)
    n = weeks.size
    y = rng.binomial(1, 0.15, n)

    blocks = split_weeks(range(n_weeks), load_config().splits)
    split_of_week = {w: name for name, ws in blocks.items() for w in ws}
    is_test = np.array([split_of_week[w] == "test" for w in weeks])

    mostly_missing = np.where(rng.random(n) < 0.02, rng.normal(size=n), np.nan)
    drifting = rng.normal(size=n) + 8.0 * (weeks / n_weeks)
    good = rng.normal(size=n) + 1.2 * y
    leaky = y * 50.0 + rng.normal(size=n)
    useless = rng.normal(size=n)

    frame = pl.DataFrame(
        {
            "case_id": np.arange(n),
            "WEEK_NUM": weeks,
            "target": y,
            "f_mostly_missing": mostly_missing,
            "f_drifting": drifting,
            "f_good": good,
            "f_leaky": leaky,
            "f_useless": useless,
        }
    )
    family = {c: "synthetic" for c in frame.columns if c.startswith("f_")}
    return frame, family, split_of_week, is_test


def test_screening_applies_each_rule_with_a_stated_reason(screening_frame) -> None:
    frame, family, split_of_week, _ = screening_frame
    report = screen_features(frame, family, split_of_week, ScreeningParams()).set_index("feature")

    assert not report.loc["f_mostly_missing", "kept"]
    assert "missing" in report.loc["f_mostly_missing", "reason"]

    assert not report.loc["f_drifting", "kept"]
    assert "PSI" in report.loc["f_drifting", "reason"]

    assert not report.loc["f_useless", "kept"]
    assert "IV" in report.loc["f_useless", "reason"]

    assert report.loc["f_good", "kept"]


def test_a_leak_is_always_surfaced_for_review(screening_frame) -> None:
    """A leak must reach a human whatever else the screen decides about it.

    The subtle case, found on real data: optbinning collapses a PERFECTLY
    separating column into a single bin and reports IV 0.0, so the IV rule reads
    a total leak as useless noise and drops it. If the drop reason then
    overwrote the annotation, the one signal that would have surfaced the leak
    would disappear with it. The univariate Gini is what catches this, and it
    survives the drop.
    """
    frame, family, split_of_week, _ = screening_frame
    report = screen_features(frame, family, split_of_week, ScreeningParams()).set_index("feature")
    assert report.loc["f_leaky", "leakage_review"]
    assert abs(report.loc["f_leaky", "univariate_gini"]) > 0.5
    assert "REVIEW" in report.loc["f_leaky", "reason"]


def test_optbinning_reports_zero_iv_for_a_perfect_separator() -> None:
    """Documents why the IV rule cannot be the leakage screen.

    If this ever starts returning a large IV, the univariate-Gini screen is no
    longer load-bearing for this case and the comment above can be revised.
    """
    import pandas as pd

    from scorecard import ScorecardParams, iv_table

    rng = np.random.default_rng(0)
    n = 2_400
    y = rng.binomial(1, 0.15, n)
    frame = pd.DataFrame({"perfect": y * 50.0 + rng.normal(size=n)})
    row = iv_table(frame, y, ScorecardParams()).iloc[0]
    assert row["n_bins"] == 1
    assert row["iv"] == pytest.approx(0.0)


def test_screening_never_looks_at_the_test_weeks(screening_frame) -> None:
    """The load-bearing property of the whole screen.

    A feature kept or dropped because of how it behaves on the test weeks turns
    the out-of-time evaluation into an in-sample one, and nothing downstream
    would reveal it. Scrambling every test-week value must leave the report
    bit-identical.
    """
    frame, family, split_of_week, is_test = screening_frame
    rng = np.random.default_rng(99)

    scrambled = frame.clone()
    for column in [c for c in frame.columns if c.startswith("f_")]:
        values = frame[column].to_numpy().copy()
        values[is_test] = rng.normal(0, 100, int(is_test.sum()))
        scrambled = scrambled.with_columns(pl.Series(column, values))
    # Also corrupt the test-week targets.
    y = frame["target"].to_numpy().copy()
    y[is_test] = 1 - y[is_test]
    scrambled = scrambled.with_columns(pl.Series("target", y))

    a = screen_features(frame, family, split_of_week, ScreeningParams())
    b = screen_features(scrambled, family, split_of_week, ScreeningParams())

    assert a["feature"].tolist() == b["feature"].tolist()
    assert a["kept"].tolist() == b["kept"].tolist()
    assert a["reason"].tolist() == b["reason"].tolist()
    np.testing.assert_allclose(
        a["psi_train_vs_val"].to_numpy(), b["psi_train_vs_val"].to_numpy(), equal_nan=True
    )
    np.testing.assert_allclose(a["iv"].to_numpy(), b["iv"].to_numpy(), equal_nan=True)


def test_screening_needs_train_weeks(screening_frame) -> None:
    frame, family, _, _ = screening_frame
    all_test = {w: "test" for w in frame["WEEK_NUM"].unique().to_list()}
    with pytest.raises(ValueError, match="No train weeks"):
        screen_features(frame, family, all_test, ScreeningParams())


def test_report_has_the_columns_the_brief_requires(screening_frame) -> None:
    frame, family, split_of_week, _ = screening_frame
    report = screen_features(frame, family, split_of_week, ScreeningParams())
    for column in (
        "feature",
        "family",
        "missing_pct",
        "iv",
        "psi_train_vs_val",
        "univariate_gini",
        "kept",
        "reason",
    ):
        assert column in report.columns, column
    assert len(report) == len([c for c in frame.columns if c.startswith("f_")])


def test_params_load_from_config() -> None:
    cfg = load_config()
    fp = FeatureParams.from_config(cfg["features"])
    assert fp.dpd_threshold == 30
    assert fp.dpd_windows_months == (12, 24)
    assert fp.recency_halflife_months > 0
    sp = ScreeningParams.from_config(cfg["screening"])
    assert sp.max_missing_rate == 0.95
    assert sp.psi_drop == 0.25 and sp.psi_flag == 0.10
    assert math.isclose(sp.leakage_gini, 0.50)


# --- multi-part tables ---------------------------------------------------


def test_table_paths_collects_numbered_parts_in_order(tmp_path) -> None:
    """The full data splits big tables across numbered parts; the sample does not."""
    from features import table_paths

    for name in (
        "train_credit_bureau_a_2_0",
        "train_credit_bureau_a_2_2",
        "train_credit_bureau_a_2_10",  # sorts before _2 as a string, after it as a number
        "train_credit_bureau_a_1_0",
        "train_base",
    ):
        pl.DataFrame({"case_id": [1]}).write_parquet(tmp_path / f"{name}.parquet")

    parts = table_paths(tmp_path, "train_credit_bureau_a_2")
    assert [p.stem for p in parts] == [
        "train_credit_bureau_a_2_0",
        "train_credit_bureau_a_2_2",
        "train_credit_bureau_a_2_10",
    ], "parts must be ordered numerically, not lexicographically"
    assert [p.stem for p in table_paths(tmp_path, "train_base")] == ["train_base"]


def test_table_paths_does_not_over_match_a_longer_name(tmp_path) -> None:
    """`train_static_0` must not swallow `train_static_cb_0`, nor depth suffixes."""
    from features import table_paths

    for name in ("train_static_0_0", "train_static_0_1", "train_static_cb_0", "train_applprev_2"):
        pl.DataFrame({"case_id": [1]}).write_parquet(tmp_path / f"{name}.parquet")

    assert [p.stem for p in table_paths(tmp_path, "train_static_0")] == [
        "train_static_0_0",
        "train_static_0_1",
    ]
    assert [p.stem for p in table_paths(tmp_path, "train_static_cb_0")] == ["train_static_cb_0"]
    assert table_paths(tmp_path, "train_applprev_1") == []


def test_build_reads_a_table_split_across_parts(tmp_path) -> None:
    tables, _ = make_dataset(n_cases=400, n_weeks=10, seed=6)
    base = tables["train_base"]
    base.write_parquet(tmp_path / "train_base.parquet")
    # Split one table in two, as the competition does.
    static = tables["train_static_0_0"]
    half = static.height // 2
    static[:half].write_parquet(tmp_path / "train_static_0_0.parquet")
    static[half:].write_parquet(tmp_path / "train_static_0_1.parquet")

    feats, _ = build_features(tmp_path, FeatureParams())
    assert feats.height == 400
    # Affordability comes from static_0; it must see both halves.
    assert "afford_credit_amount" in feats.columns
    assert feats["afford_credit_amount"].null_count() == 0


def test_find_data_dir_handles_the_nested_competition_mount(tmp_path) -> None:
    """Data mounts at /kaggle/input/competitions/<slug>/, not /kaggle/input/<slug>/."""
    from features import find_data_dir

    nested = tmp_path / "competitions" / "home-credit" / "parquet_files" / "train"
    nested.mkdir(parents=True)
    assert find_data_dir(tmp_path) == nested
    # A directory that already holds the tables is returned unchanged.
    flat = tmp_path / "flat"
    flat.mkdir()
    assert find_data_dir(flat) == flat


def test_kept_features_reads_the_screening_decision(tmp_path) -> None:
    """Downstream stages must honour the screen, not re-select around it.

    features.parquet deliberately carries every feature that was built. The
    champion scorecard picked up bureau_a_records anyway -- dropped for
    PSI 0.614 -- because its IV was healthy and nothing enforced the earlier
    decision. In a project about stability that is the one feature the screen
    existed to exclude.
    """
    from features import kept_features

    report = tmp_path / "feature_report.csv"
    pd_mod = pytest.importorskip("pandas")
    pd_mod.DataFrame(
        {
            "feature": ["good_a", "psi_dropped", "good_b", "low_iv"],
            "kept": [True, False, True, False],
            "reason": ["kept", "PSI 0.614 > 0.25", "kept", "IV 0.001 < 0.02"],
        }
    ).to_csv(report, index=False)

    assert kept_features(report) == ["good_a", "good_b"]


def test_kept_features_rejects_a_malformed_or_empty_report(tmp_path) -> None:
    from features import kept_features

    pd_mod = pytest.importorskip("pandas")
    bad = tmp_path / "bad.csv"
    pd_mod.DataFrame({"feature": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        kept_features(bad)

    empty = tmp_path / "empty.csv"
    pd_mod.DataFrame({"feature": ["a"], "kept": [False]}).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="no feature as kept"):
        kept_features(empty)


# --- post-decision leakage, through the real pipeline --------------------


def _competition_shaped(tmp_path, *, post_decision_dpd: float = 999.0):
    """A base table plus a bureau payment history straddling the decision date.

    Pre-decision months carry DPD 10; post-decision months carry a value that
    cannot be mistaken for anything else. Any feature that reflects the latter
    is reading the future.
    """
    import datetime as dt

    decision = dt.date(2020, 6, 15)
    base = pl.DataFrame(
        {
            "case_id": [1, 2, 3],
            "date_decision": [decision.isoformat()] * 3,
            "MONTH": [202006] * 3,
            "WEEK_NUM": [10, 11, 12],
            "target": [0, 1, 0],
        }
    )
    rows = []
    for case_id in (1, 2, 3):
        for month in (1, 2, 3):  # well before the decision
            rows.append({"case_id": case_id, "num_group1": 0, "num_group2": month,
                         "pmts_dpd_1073P": 10.0, "pmts_overdue_1140A": 100.0,
                         "pmts_month_158T": float(month), "pmts_year_1139T": 2020.0})
        for month in (9, 10, 11):  # AFTER the decision
            rows.append({"case_id": case_id, "num_group1": 0, "num_group2": month,
                         "pmts_dpd_1073P": post_decision_dpd,
                         "pmts_overdue_1140A": 9_999.0,
                         "pmts_month_158T": float(month), "pmts_year_1139T": 2020.0})
    base.write_parquet(tmp_path / "train_base.parquet")
    pl.DataFrame(rows).write_parquet(tmp_path / "train_credit_bureau_a_2.parquet")
    return base


def test_the_feature_build_ignores_post_decision_payment_months(tmp_path) -> None:
    """The brief's "no post-decision dates" requirement, end to end.

    delinquency_features filters the payment month against date_decision inline
    rather than through drop_post_decision, so testing that helper in isolation
    leaves this path uncovered. On the real data 8.4 % of bureau payment rows
    post-date their own decision and the maximum post-decision DPD carries a
    univariate Gini of +0.097 against the target: it is partly the outcome being
    predicted, so this is the difference between a model and a time machine.
    """
    _competition_shaped(tmp_path)
    feats, _ = build_features(tmp_path, FeatureParams())

    assert feats.height == 3
    # Only the pre-decision months exist as far as the features are concerned.
    assert feats["delq_dpd_max"].to_list() == [10.0, 10.0, 10.0]
    assert feats["delq_n_payment_months"].to_list() == [3, 3, 3]
    assert feats["delq_overdue_max"].max() == pytest.approx(100.0)
    assert feats["delq_dpd_max"].max() < 999.0, "a post-decision DPD reached the features"


def test_that_test_would_fail_if_the_filter_were_removed(tmp_path) -> None:
    """Guards the guard: prove the fixture actually contains future rows.

    Without this, the assertions above would also pass on data that simply had
    no post-decision months, and would be protecting nothing.
    """
    import datetime as dt

    _competition_shaped(tmp_path)
    raw = pl.read_parquet(tmp_path / "train_credit_bureau_a_2.parquet")
    decision = dt.date(2020, 6, 15)
    future = raw.filter(
        pl.date(
            pl.col("pmts_year_1139T").cast(pl.Int32),
            pl.col("pmts_month_158T").cast(pl.Int32),
            1,
        )
        > pl.lit(decision)
    )
    assert future.height == 9, "the fixture plants no post-decision rows"
    assert future["pmts_dpd_1073P"].max() == 999.0


def test_absent_tax_registry_means_unknown_income_not_zero(tmp_path) -> None:
    """"No record" and "earned nothing" are different facts.

    Recording the first as the second put 522,900 cases (34 % of the book) into
    the lowest income bin beside genuinely low earners, instead of the binning's
    Missing bin. That is what let "no employment record" stand in for "no
    income" -- false for the 92 % of over-60s who are simply retired, and the
    mechanism behind a threefold over-prediction of their risk.
    """
    tables, _ = make_dataset(n_cases=300, n_weeks=10, seed=8)
    base = tables["train_base"]
    base.write_parquet(tmp_path / "train_base.parquet")
    tables["train_static_0_0"].write_parquet(tmp_path / "train_static_0.parquet")

    # Only the first 100 cases appear in a tax registry at all.
    ids = base["case_id"].to_list()
    pl.DataFrame(
        {
            "case_id": ids[:100],
            "num_group1": [0] * 100,
            "pmtamount_36A": [50_000.0] * 100,
        }
    ).write_parquet(tmp_path / "train_tax_registry_c_1.parquet")

    feats, _ = build_features(tmp_path, FeatureParams())
    income = feats.select("case_id", "afford_taxincome_total").sort("case_id")
    present = income.filter(pl.col("case_id").is_in(ids[:100]))
    absent = income.filter(~pl.col("case_id").is_in(ids[:100]))

    assert present["afford_taxincome_total"].null_count() == 0
    assert present["afford_taxincome_total"].min() > 0
    # The 200 with no registry row at all are UNKNOWN, not zero.
    assert absent["afford_taxincome_total"].null_count() == absent.height
    assert absent.height == 200
