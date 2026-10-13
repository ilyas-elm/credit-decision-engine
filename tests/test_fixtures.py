"""The fixture generator itself, including the defects it is supposed to plant.

If the generator stops planting them, the screening tests would pass on clean
data and prove nothing.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from fixtures import make_dataset, write_dataset

from metrics import gini, psi


@pytest.fixture(scope="module")
def dataset():
    return make_dataset(n_cases=1_200, n_weeks=30, seed=11)


def test_base_table_has_the_expected_schema(dataset) -> None:
    tables, _ = dataset
    base = tables["train_base"]
    assert set(base.columns) == {"case_id", "date_decision", "MONTH", "WEEK_NUM", "target"}
    assert base["case_id"].n_unique() == base.height
    assert base["target"].is_in([0, 1]).all()


def test_every_week_is_populated(dataset) -> None:
    # An empty week would make a split block empty and mask ordering bugs.
    tables, _ = dataset
    weeks = tables["train_base"]["WEEK_NUM"]
    assert sorted(weeks.unique().to_list()) == list(range(30))


def test_default_rate_is_in_a_believable_range(dataset) -> None:
    tables, _ = dataset
    rate = tables["train_base"]["target"].mean()
    assert 0.01 < rate < 0.10, f"synthetic default rate {rate:.3f} is not credible"


def test_depth_tables_are_keyed_as_the_schema_requires(dataset) -> None:
    tables, _ = dataset
    d1, d2 = tables["train_applprev_1_0"], tables["train_applprev_2"]
    assert "num_group1" in d1.columns and "num_group2" not in d1.columns
    assert {"num_group1", "num_group2"} <= set(d2.columns)
    # Depth 1 has many rows per case; depth 2 many per (case, num_group1).
    assert d1.height > tables["train_base"].height
    assert d1["case_id"].n_unique() == tables["train_base"].height
    assert d2.group_by("case_id", "num_group1").len()["len"].max() >= 1


def test_column_suffixes_cover_the_documented_transform_groups(dataset) -> None:
    tables, _ = dataset
    suffixes = {
        col[-1]
        for name, frame in tables.items()
        if name != "train_base"
        for col in frame.columns
        if col not in ("case_id", "num_group1", "num_group2")
    }
    # P = days past due, M = masked categories, A = amounts, D = dates, T/L = other.
    assert {"P", "M", "A", "D"} <= suffixes, suffixes


def test_post_decision_rows_are_actually_planted(dataset) -> None:
    tables, planted = dataset
    assert planted.n_post_decision_rows > 0

    joined = tables["train_applprev_1_0"].join(
        tables["train_base"].select("case_id", "date_decision"), on="case_id", how="left"
    )
    violations = joined.filter(pl.col(planted.post_decision_date) > pl.col("date_decision"))
    assert violations.height == planted.n_post_decision_rows


def test_leaky_column_is_detectable_by_the_univariate_gini_screen(dataset) -> None:
    tables, planted = dataset
    static = tables["train_static_0_0"].join(
        tables["train_base"].select("case_id", "target"), on="case_id", how="left"
    )
    value = gini(static["target"].to_numpy(), static[planted.leaky].to_numpy())
    assert abs(value) > 0.5, f"planted leak has Gini {value:.3f}, screen would miss it"


def test_mostly_missing_column_is_above_the_drop_threshold(dataset) -> None:
    tables, planted = dataset
    static = tables["train_static_0_0"]
    missing_rate = static[planted.mostly_missing].is_null().mean()
    nan_rate = float(np.isnan(static[planted.mostly_missing].to_numpy()).mean())
    assert max(missing_rate or 0.0, nan_rate) > 0.95


def test_drifting_column_is_above_the_psi_drop_threshold(dataset) -> None:
    tables, planted = dataset
    static = tables["train_static_0_0"].join(
        tables["train_base"].select("case_id", "WEEK_NUM"), on="case_id", how="left"
    )
    early = static.filter(pl.col("WEEK_NUM") < 18)[planted.drifting].to_numpy()
    late = static.filter(pl.col("WEEK_NUM") >= 18)[planted.drifting].to_numpy()
    assert psi(early, late, bins=10) > 0.25


def test_stable_noise_column_does_not_drift(dataset) -> None:
    tables, _ = dataset
    static = tables["train_static_0_0"].join(
        tables["train_base"].select("case_id", "WEEK_NUM"), on="case_id", how="left"
    )
    early = static.filter(pl.col("WEEK_NUM") < 18)["stablenoise_501L"].to_numpy()
    late = static.filter(pl.col("WEEK_NUM") >= 18)["stablenoise_501L"].to_numpy()
    assert psi(early, late, bins=10) < 0.1


def test_dataset_round_trips_through_parquet(dataset, tmp_path) -> None:
    tables, _ = dataset
    written = write_dataset(tmp_path, tables)
    assert set(written) == set(tables)
    for name, path in written.items():
        assert pl.read_parquet(path).height == tables[name].height
