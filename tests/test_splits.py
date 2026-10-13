"""Split integrity: the acceptance criteria from the brief, as tests."""

from __future__ import annotations

import polars as pl
import pytest
from fixtures import make_base

from config import SplitsConfig, load_config
from splits import SPLIT_ORDER, assert_split_integrity, assign_splits, split_weeks


@pytest.fixture(scope="module")
def splits_cfg() -> SplitsConfig:
    return load_config().splits


def test_config_fractions_sum_to_one(splits_cfg: SplitsConfig) -> None:
    assert abs(sum(splits_cfg.as_dict().values()) - 1.0) < 1e-9


def test_fractions_that_do_not_sum_to_one_are_rejected() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitsConfig(train_frac=0.6, val_a_frac=0.1, val_b_frac=0.1, test_frac=0.1)


def test_blocks_are_contiguous_ordered_and_cover_every_week(splits_cfg: SplitsConfig) -> None:
    weeks = list(range(92))
    blocks = split_weeks(weeks, splits_cfg)

    assert list(blocks) == list(SPLIT_ORDER)
    # Every week used exactly once.
    assigned = [week for name in SPLIT_ORDER for week in blocks[name]]
    assert sorted(assigned) == weeks
    assert len(assigned) == len(set(assigned))
    # Each block contiguous, and strictly earlier than the next.
    for name in SPLIT_ORDER:
        block = blocks[name]
        assert block == list(range(block[0], block[-1] + 1)), f"{name} is not contiguous"
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:], strict=False):
        assert blocks[earlier][-1] < blocks[later][0]


def test_block_sizes_track_the_configured_proportions(splits_cfg: SplitsConfig) -> None:
    n = 400
    blocks = split_weeks(range(n), splits_cfg)
    for name, frac in splits_cfg.ordered_fractions:
        assert abs(len(blocks[name]) / n - frac) < 0.01, f"{name} is off target"


def test_too_few_weeks_raises_rather_than_emptying_a_split(splits_cfg: SplitsConfig) -> None:
    with pytest.raises(ValueError, match="at least 4 distinct weeks"):
        split_weeks([0, 1, 2], splits_cfg)


def test_every_split_is_non_empty_at_the_minimum_week_count(splits_cfg: SplitsConfig) -> None:
    # 4 weeks and a 60/8/7/25 split: naive rounding would starve val_a and val_b.
    blocks = split_weeks(range(4), splits_cfg)
    assert all(len(blocks[name]) >= 1 for name in SPLIT_ORDER), blocks


def test_no_case_id_appears_in_two_splits(splits_cfg: SplitsConfig) -> None:
    base = make_base(n_cases=1_500, n_weeks=40, seed=7)
    assigned = assign_splits(base, splits_cfg)

    assert assigned.height == base.height
    assert assigned["case_id"].n_unique() == base.height
    per_case = assigned.group_by("case_id").agg(pl.col("split").n_unique().alias("k"))
    assert per_case["k"].max() == 1
    # And the integrity check agrees: it raises if anything is wrong.
    assert_split_integrity(assigned)


def test_integrity_check_catches_a_broken_ordering(splits_cfg: SplitsConfig) -> None:
    base = make_base(n_cases=1_500, n_weeks=40, seed=7)
    assigned = assign_splits(base, splits_cfg)
    # Relabel one late case as train: ordering is now violated.
    broken = assigned.with_columns(
        pl.when(pl.col("WEEK_NUM") == pl.col("WEEK_NUM").max())
        .then(pl.lit("train", dtype=assigned.schema["split"]))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    with pytest.raises(AssertionError, match="Out-of-time ordering is broken"):
        assert_split_integrity(broken)


def test_integrity_check_catches_a_duplicated_case(splits_cfg: SplitsConfig) -> None:
    base = make_base(n_cases=600, n_weeks=20, seed=3)
    assigned = assign_splits(base, splits_cfg)
    train_case = assigned.filter(pl.col("split") == "train").head(1)
    leaked = train_case.with_columns(pl.lit("test", dtype=assigned.schema["split"]).alias("split"))
    with pytest.raises(AssertionError, match="more than one split"):
        assert_split_integrity(pl.concat([assigned, leaked]))
