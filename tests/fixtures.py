"""Synthetic fixtures matching the Home Credit schema.

Competition data is never committed, not even a sample, so every test and the CI
run against generated tables. They mimic the real thing where it matters to the
code under test: table names, the P/M/A/D/T/L column-suffix groups, depth 0/1/2
shapes with num_group1 / num_group2, and a base table keyed by case_id over
WEEK_NUM.

The generator deliberately plants the defects the pipeline is supposed to
catch: a leaky column, an all-but-missing column, a drifting column and rows
dated after date_decision, so that a screening test failing to flag them fails
the test rather than passing on clean data.

Shapes are asserted against the real tables once docs/DATA_NOTES.md exists; until
then these are the schema as the brief describes it, which is explicitly marked
unverified there.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "PlantedDefects",
    "make_base",
    "make_static_0",
    "make_depth1",
    "make_depth2",
    "make_dataset",
    "write_dataset",
]

EPOCH = dt.date(2019, 1, 1)

# The names the pipeline must treat as date columns (suffix D), amounts (A),
# days-past-due (P), masked categories (M) and the catch-all groups (T, L).
PLANTED = {
    "leaky": "leakyscore_42P",
    "mostly_missing": "sparsefield_77A",
    "drifting": "driftamount_13A",
    "post_decision_date": "lastupdate_99D",
}


@dataclass(frozen=True)
class PlantedDefects:
    """What the generator planted, so tests assert on names rather than guesses."""

    leaky: str = PLANTED["leaky"]
    mostly_missing: str = PLANTED["mostly_missing"]
    drifting: str = PLANTED["drifting"]
    post_decision_date: str = PLANTED["post_decision_date"]
    n_post_decision_rows: int = 0


def _weeks_to_dates(weeks: np.ndarray, rng: np.random.Generator) -> list[dt.date]:
    offsets = weeks * 7 + rng.integers(0, 7, size=weeks.size)
    return [EPOCH + dt.timedelta(days=int(offset)) for offset in offsets]


def _as_iso(dates: list) -> list[str | None]:
    """Dates as ISO strings.

    The real competition data stores date_decision and every D column as
    String, not Date (docs/DATA_NOTES.md). Fixtures that emit real date types
    would let the pipeline's date parsing go untested and then fail on the
    actual data, so the fixtures carry the same dtype the data does.
    """
    return [None if d is None else d.isoformat() for d in dates]


def make_base(
    n_cases: int = 2_000,
    n_weeks: int = 40,
    *,
    default_rate: float = 0.03,
    seed: int = 0,
) -> pl.DataFrame:
    """The base table: case_id, date_decision, MONTH, WEEK_NUM, target.

    Every week is populated, so splitting by week never produces an empty block.
    The default rate drifts mildly upward over weeks, which gives the stability
    and calibration-in-the-large tests something real to measure.
    """
    rng = np.random.default_rng(seed)
    if n_cases < n_weeks:
        raise ValueError(f"Need n_cases >= n_weeks, got {n_cases} < {n_weeks}.")

    # Cover every week once, then scatter the remainder.
    weeks = np.concatenate([np.arange(n_weeks), rng.integers(0, n_weeks, size=n_cases - n_weeks)])
    weeks.sort()
    case_id = np.arange(1, n_cases + 1)
    dates = _weeks_to_dates(weeks, rng)

    drift = 1.0 + 0.5 * (weeks / max(n_weeks - 1, 1))
    target = rng.binomial(1, np.clip(default_rate * drift, 0.0, 1.0))

    return pl.DataFrame(
        {
            "case_id": case_id.astype(np.int64),
            "date_decision": _as_iso(dates),
            "MONTH": [int(f"{d.year}{d.month:02d}") for d in dates],
            "WEEK_NUM": weeks.astype(np.int32),
            "target": target.astype(np.int32),
        }
    )


def make_static_0(base: pl.DataFrame, *, seed: int = 1) -> pl.DataFrame:
    """A depth-0 table: one row per case_id, with the planted defects."""
    rng = np.random.default_rng(seed)
    n = base.height
    target = base["target"].to_numpy()
    weeks = base["WEEK_NUM"].to_numpy()
    n_weeks = max(int(weeks.max()) + 1, 1)

    # A genuinely predictive feature, at a believable strength.
    signal = rng.normal(0.0, 1.0, n) + 0.8 * target
    # Leaky: near-perfect separation, which the screen must flag (Gini > 0.5).
    leaky = target * 100.0 + rng.normal(0.0, 1.0, n)
    # Mostly missing: ~98% null, above the 95% drop threshold.
    sparse = np.where(rng.random(n) < 0.02, rng.normal(50_000, 10_000, n), np.nan)
    # Drifting: mean shifts hard with week, so train-vs-validation PSI is large.
    drifting = rng.normal(0.0, 1.0, n) + 6.0 * (weeks / max(n_weeks - 1, 1))
    # A stable, uninformative column: PSI low, IV ~0, should be dropped on IV.
    noise = rng.normal(0.0, 1.0, n)

    return pl.DataFrame(
        {
            "case_id": base["case_id"],
            "annuity_780A": np.abs(rng.normal(20_000, 6_000, n)).astype(np.float32),
            "credamount_770A": np.abs(rng.normal(300_000, 90_000, n)).astype(np.float32),
            "income_523A": np.abs(rng.normal(150_000, 50_000, n)).astype(np.float32),
            "maxdpdlast12m_727P": np.clip(rng.gamma(1.2, 12.0, n) + 8 * target, 0, None).astype(
                np.float32
            ),
            "riskscore_900T": signal.astype(np.float32),
            "education_1103M": rng.choice(
                ["a55475b1", "P97_36_170", "P106_81_188", None], size=n
            ).tolist(),
            "birthdate_574D": _weeks_to_dates(weeks - rng.integers(1_000, 3_000, size=n), rng),
            PLANTED["leaky"]: leaky.astype(np.float32),
            PLANTED["mostly_missing"]: sparse.astype(np.float32),
            PLANTED["drifting"]: drifting.astype(np.float32),
            "stablenoise_501L": noise.astype(np.float32),
        }
    )


def make_depth1(
    base: pl.DataFrame,
    *,
    max_rows_per_case: int = 4,
    post_decision_rows: int = 25,
    seed: int = 2,
) -> tuple[pl.DataFrame, int]:
    """A depth-1 table (many rows per case_id, indexed by num_group1).

    ``post_decision_rows`` rows carry a date AFTER date_decision. Those are the
    rows the feature build must drop; a test asserts they are gone.
    """
    rng = np.random.default_rng(seed)
    case_ids = base["case_id"].to_numpy()
    # date_decision is an ISO string, as in the real data, so parse it back for
    # the arithmetic and write the result out as a string again.
    decision = {
        row["case_id"]: dt.date.fromisoformat(row["date_decision"])
        for row in base.iter_rows(named=True)
    }

    rows: list[dict] = []
    for case_id in case_ids:
        for group in range(int(rng.integers(1, max_rows_per_case + 1))):
            back = int(rng.integers(30, 1_500))
            rows.append(
                {
                    "case_id": int(case_id),
                    "num_group1": group,
                    "pmts_dpd_1073P": float(max(rng.gamma(1.1, 10.0), 0.0)),
                    "outstandingamount_354A": float(abs(rng.normal(80_000, 30_000))),
                    "credlmt_935A": float(abs(rng.normal(200_000, 60_000)) + 1_000),
                    "status_219L": str(rng.choice(["A", "D", "K", "T"])),
                    "classificationofcontr_13M": str(
                        rng.choice(["a55475b1", "P33_146_175", "P17_36_170"])
                    ),
                    PLANTED["post_decision_date"]: (
                        decision[int(case_id)] - dt.timedelta(days=back)
                    ).isoformat(),
                }
            )

    frame = pl.DataFrame(rows)

    # Plant the violations: move some rows to AFTER their own decision date.
    # Offsetting the existing date by a fixed amount does not work, a row
    # originally 1,500 days back stays in the past, so the planted dates are
    # computed from date_decision itself, which makes the count exact.
    n_planted = min(post_decision_rows, frame.height)
    if n_planted:
        idx = rng.choice(frame.height, size=n_planted, replace=False)
        flag = np.zeros(frame.height, dtype=bool)
        flag[idx] = True
        decision_dates = [decision[case_id] for case_id in frame["case_id"].to_list()]
        after = [
            (date + dt.timedelta(days=int(offset))).isoformat()
            for date, offset in zip(
                decision_dates, rng.integers(1, 400, size=frame.height), strict=True
            )
        ]
        frame = frame.with_columns(
            pl.when(pl.Series(flag))
            .then(pl.Series(after))
            .otherwise(pl.col(PLANTED["post_decision_date"]))
            .alias(PLANTED["post_decision_date"])
        )
    return frame, n_planted


def make_depth2(depth1: pl.DataFrame, *, seed: int = 3) -> pl.DataFrame:
    """A depth-2 table, indexed by (case_id, num_group1, num_group2)."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for row in depth1.select("case_id", "num_group1").iter_rows(named=True):
        for group2 in range(int(rng.integers(1, 3))):
            rows.append(
                {
                    "case_id": row["case_id"],
                    "num_group1": row["num_group1"],
                    "num_group2": group2,
                    "pmts_overdue_1140A": float(max(rng.normal(2_000, 1_500), 0.0)),
                    "pmts_month_158T": int(rng.integers(1, 13)),
                    "collater_typofvalofguarant_298M": str(rng.choice(["a55475b1", "P164_110_33"])),
                }
            )
    return pl.DataFrame(rows)


def make_dataset(
    n_cases: int = 2_000,
    n_weeks: int = 40,
    *,
    seed: int = 0,
) -> tuple[dict[str, pl.DataFrame], PlantedDefects]:
    """Every table plus a record of what was planted where."""
    base = make_base(n_cases, n_weeks, seed=seed)
    static = make_static_0(base, seed=seed + 1)
    depth1, n_planted = make_depth1(base, seed=seed + 2)
    depth2 = make_depth2(depth1, seed=seed + 3)
    tables = {
        "train_base": base,
        "train_static_0_0": static,
        "train_applprev_1_0": depth1,
        "train_applprev_2": depth2,
    }
    return tables, PlantedDefects(n_post_decision_rows=n_planted)


def write_dataset(directory, tables: dict[str, pl.DataFrame]) -> dict[str, str]:
    """Write the tables as parquet, named like the competition files."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, frame in tables.items():
        path = directory / f"{name}.parquet"
        frame.write_parquet(path)
        written[name] = str(path)
    return written
