#!/usr/bin/env python
"""Why a frozen Missing bin becomes a relative penalty when the book drifts.

The README explains the age disparity as drift: the Missing bin's points are
fixed when the scorecard is fitted, while the population with a value moves
into better bins. The figures behind that were worked out by hand, which left
a reader no way to re-derive them. This recomputes them.

    uv run python scripts/drift_recon.py <features.parquet>

`features.parquet` is the output of the `features` kernel, pulled with
`kaggle kernels output`. Pass the full-data one to reproduce the documented
numbers; the 3 % sample runs too and will differ. The scorecard bins and
points are read from docs/scorecard_table.csv, so this reports what the
committed champion does rather than refitting anything.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
SCORECARD = REPO / "docs" / "scorecard_table.csv"
#: Split boundaries, matching config/default.toml. Hardcoding them here would
#: let this drift away from the model, so they are read from the config.
CHARACTERISTICS = ("afford_annuity_to_taxincome", "afford_taxincome_total")


def _bins(characteristic: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Upper edges and points for the interval bins, plus the Missing points."""
    edges, points, missing = [], [], None
    with SCORECARD.open() as fh:
        for row in csv.DictReader(fh):
            if row["characteristic"] != characteristic:
                continue
            if row["bin"] == "Missing":
                missing = float(row["points"])
            elif row["bin"].startswith(("(", "[")):
                upper = row["bin"].split(",")[1].strip().rstrip(")")
                edges.append(float("inf") if upper == "inf" else float(upper))
                points.append(float(row["points"]))
    if missing is None:
        raise SystemExit(f"{characteristic} has no Missing bin in {SCORECARD}")
    return np.array(edges), np.array(points), missing


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    features = Path(argv[0])
    if not features.is_file():
        print(f"No features parquet at {features}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO / "src"))
    from config import load_config

    splits = load_config().splits
    frame = pl.scan_parquet(features)
    weeks = frame.select(pl.col("WEEK_NUM")).collect(engine="streaming")["WEEK_NUM"]
    blocks = splits_to_blocks(sorted(weeks.unique().to_list()), splits)

    for characteristic in CHARACTERISTICS:
        edges, points, missing = _bins(characteristic)
        print(f"\n{characteristic}")
        print(
            f"  Missing bin: {missing:.1f} points, in a range of "
            f"{points.min():.1f} to {points.max():.1f}"
        )
        for name in ("train", "test"):
            lo, hi = min(blocks[name]), max(blocks[name])
            values = (
                frame.filter((pl.col("WEEK_NUM") >= lo) & (pl.col("WEEK_NUM") <= hi))
                .select(characteristic)
                .collect(engine="streaming")[characteristic]
                .drop_nulls()
                .to_numpy()
            )
            index = np.clip(np.searchsorted(edges, values, side="left"), 0, points.size - 1)
            best_two = (index <= 1).mean()
            print(
                f"  {name:5} weeks {lo}-{hi}: n={values.size:>9,}  "
                f"median={np.median(values):>12,.3f}  mean points={points[index].mean():5.1f}  "
                f"in the two best bins={best_two:.1%}"
            )

    # The claim the drift explanation rests on: the absence is not riskier.
    print("\ndefault rate by whether a tax-registry income record exists")
    rates = (
        frame.with_columns(pl.col("afford_taxincome_total").is_null().alias("absent"))
        .group_by("absent")
        .agg(pl.len().alias("n"), pl.col("target").mean().alias("default_rate"))
        .sort("absent")
        .collect(engine="streaming")
    )
    for row in rates.iter_rows(named=True):
        label = "no record" if row["absent"] else "has record"
        print(f"  {label:11} n={row['n']:>9,}  default {row['default_rate']:.5f}")
    return 0


def splits_to_blocks(weeks: list[int], splits) -> dict[str, list[int]]:
    sys.path.insert(0, str(REPO / "src"))
    from splits import split_weeks

    return split_weeks(weeks, splits)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
