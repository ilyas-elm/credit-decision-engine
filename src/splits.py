"""Out-of-time splits by WEEK_NUM.

Random K-fold across time is never used: it lets a model see the future, and the
competition is explicitly about stability over weeks. Splits are contiguous
blocks of whole weeks, oldest first.

Each held-out block has exactly one job:

    train  -> fitting
    val_a  -> early stopping and probability calibration
    val_b  -> conformal calibration, and nothing else
    test   -> evaluation, touched by no selection decision

val_b is kept separate because split-conformal coverage is only guaranteed on
data that the underlying model and its calibrator never saw. Reusing val_a there
would void the guarantee while still producing plausible-looking coverage
numbers, which is the dangerous kind of wrong.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from config import SplitsConfig

SPLIT_ORDER = ("train", "val_a", "val_b", "test")

__all__ = [
    "SPLIT_ORDER",
    "split_weeks",
    "assign_splits",
    "assert_split_integrity",
]


def split_weeks(weeks: Sequence[int], splits: SplitsConfig) -> dict[str, list[int]]:
    """Partition sorted distinct week numbers into contiguous, ordered blocks.

    Boundaries come from cumulative rounding, so the block sizes sum to exactly
    ``len(weeks)`` with no drift and no week dropped.
    """
    unique_weeks = sorted(set(int(w) for w in weeks))
    n = len(unique_weeks)
    if n < len(SPLIT_ORDER):
        raise ValueError(
            f"Need at least {len(SPLIT_ORDER)} distinct weeks to form "
            f"{len(SPLIT_ORDER)} splits, got {n}."
        )

    ordered = splits.ordered_fractions
    blocks: dict[str, list[int]] = {}
    prev = 0
    cumulative = 0.0
    for name, frac in ordered[:-1]:
        cumulative += frac
        end = int(round(cumulative * n))
        # Leave room for every remaining block to get at least one week.
        remaining_blocks = len(ordered) - len(blocks) - 1
        end = min(max(end, prev + 1), n - remaining_blocks)
        blocks[name] = unique_weeks[prev:end]
        prev = end
    last_name = ordered[-1][0]
    blocks[last_name] = unique_weeks[prev:]

    for name, block in blocks.items():
        if not block:
            raise ValueError(
                f"Split {name!r} came out empty with {n} weeks and fractions "
                f"{splits.as_dict()}. Use more weeks or different proportions."
            )
    return blocks


def assign_splits(
    base: pl.DataFrame | pl.LazyFrame,
    splits: SplitsConfig,
    *,
    week_col: str = "WEEK_NUM",
) -> pl.DataFrame:
    """Add a ``split`` column to the base table.

    Returns an eager frame: the week boundaries have to be known to assign them,
    so there is nothing to defer.
    """
    frame = base.lazy()
    weeks = frame.select(pl.col(week_col).unique()).collect().to_series().to_list()
    blocks = split_weeks(weeks, splits)

    mapping = {week: name for name, block in blocks.items() for week in block}
    return frame.with_columns(
        pl.col(week_col)
        .replace_strict(mapping, return_dtype=pl.Enum(list(SPLIT_ORDER)))
        .alias("split")
    ).collect()


def assert_split_integrity(
    frame: pl.DataFrame,
    *,
    case_col: str = "case_id",
    week_col: str = "WEEK_NUM",
    split_col: str = "split",
) -> None:
    """The acceptance criteria from the brief, as assertions.

    1. No ``case_id`` appears in two splits.
    2. train weeks < val_a weeks < val_b weeks < test weeks, strictly.
    """
    # 1. Disjoint case ids.
    per_case = frame.group_by(case_col).agg(pl.col(split_col).n_unique().alias("n_splits"))
    offenders = per_case.filter(pl.col("n_splits") > 1)
    if offenders.height:
        examples = offenders.head(5)[case_col].to_list()
        raise AssertionError(
            f"{offenders.height} case_id(s) appear in more than one split, e.g. {examples}."
        )

    # 2. Strictly increasing week blocks.
    bounds = (
        frame.group_by(split_col)
        .agg(pl.col(week_col).min().alias("min_week"), pl.col(week_col).max().alias("max_week"))
        .to_dicts()
    )
    by_name = {row[split_col]: row for row in bounds}
    present = [name for name in SPLIT_ORDER if name in by_name]
    for earlier, later in zip(present, present[1:], strict=False):
        if by_name[earlier]["max_week"] >= by_name[later]["min_week"]:
            raise AssertionError(
                f"Split {earlier!r} (weeks {by_name[earlier]['min_week']}"
                f"-{by_name[earlier]['max_week']}) overlaps or post-dates {later!r} "
                f"(weeks {by_name[later]['min_week']}-{by_name[later]['max_week']}). "
                "Out-of-time ordering is broken."
            )
