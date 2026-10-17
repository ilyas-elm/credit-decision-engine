#!/usr/bin/env python
"""Reconnaissance figures the documents cite as measured on the local sample.

Some claims in MODEL.md and docs/DATA_NOTES.md are not produced by any Kaggle
stage: the share of bureau payment rows that postdate their own application,
how much the outcome leaks through them, the coverage of each candidate birth
column, and the information value of a feature on the sample against the full
data. They were originally worked out ad hoc, which left a reader no way to
re-derive them.

    uv run python scripts/sample_recon.py

Needs data/sample, which kernel `make-sample` produces and `kaggle kernels
output` pulls. Nothing here touches the full competition data, and the numbers
describe the 3 % sample unless a line says otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from features import base_frame, find_data_dir, table_paths  # noqa: E402
from metrics import gini  # noqa: E402

SAMPLE = REPO / "data" / "sample"
BUREAU_PAYMENTS = "train_credit_bureau_a_2"
BIRTH_CANDIDATES = (
    ("person_1", "birth_259D"),
    ("static_cb_0", "dateofbirth_337D"),
    ("static_cb_0", "birthdate_574D"),
)


def _payment_rows(data_dir: Path) -> pl.LazyFrame:
    """Bureau payment rows carrying a DPD value and a parseable payment month."""
    base = base_frame(data_dir).select("case_id", "decision_date", "target")
    frame = pl.concat(
        [pl.scan_parquet(p) for p in table_paths(data_dir, BUREAU_PAYMENTS)],
        how="diagonal_relaxed",
    )
    dpd_columns = [c for c in frame.collect_schema().names() if c.startswith("pmts_dpd")]
    return (
        frame.select(["case_id", "pmts_month_158T", "pmts_year_1139T", *dpd_columns])
        .join(base, on="case_id", how="inner")
        .with_columns(pl.coalesce([pl.col(c) for c in dpd_columns]).alias("dpd"))
        .filter(pl.col("dpd").is_not_null())
        .with_columns(
            pl.date(
                pl.col("pmts_year_1139T").cast(pl.Int32, strict=False),
                pl.col("pmts_month_158T").cast(pl.Int32, strict=False),
                1,
            ).alias("payment_date")
        )
        .filter(pl.col("payment_date").is_not_null())
        .with_columns((pl.col("payment_date") > pl.col("decision_date")).alias("post_decision"))
    )


def post_decision_leakage(data_dir: Path) -> None:
    """Why the post-decision drop exists, in two numbers."""
    rows = _payment_rows(data_dir)
    share = rows.select(pl.col("post_decision").mean()).collect(engine="streaming").item()
    print(f"bureau payment rows dated after date_decision: {share:.2%}")

    # The leak itself: the worst delinquency recorded only AFTER the decision.
    # A feature built from these rows is reading the outcome.
    leaked = (
        rows.filter(pl.col("post_decision"))
        .group_by("case_id")
        .agg(pl.col("dpd").max().alias("post_dpd_max"), pl.col("target").first().alias("target"))
        .collect(engine="streaming")
    )
    g = gini(leaked["target"].to_numpy(), leaked["post_dpd_max"].to_numpy())
    print(f"  cases with any such row: {leaked.height:,}")
    print(f"  univariate Gini of their max DPD against the target: {g:+.4f}")


def birth_column_coverage(data_dir: Path) -> None:
    """Coverage of each candidate birth column, and whether two of them agree."""
    cases = base_frame(data_dir).select("case_id").collect()
    frames: dict[str, pl.DataFrame] = {}
    print(f"\nbirth columns over {cases.height:,} sampled cases:")
    for table, column in BIRTH_CANDIDATES:
        paths = table_paths(data_dir, f"train_{table}")
        if not paths:
            continue
        scan = pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")
        if column not in scan.collect_schema().names():
            continue
        if table == "person_1":
            scan = scan.filter(pl.col("num_group1") == 0)
        candidate = scan.select(
            "case_id", pl.col(column).cast(pl.Utf8, strict=False).alias("birth")
        ).collect(engine="streaming")
        frames[f"{table}.{column}"] = candidate
        coverage = cases.join(candidate, on="case_id", how="left")["birth"].is_not_null().mean()
        print(f"  {table}.{column:18} {coverage:.4f}")

    pair = ("person_1.birth_259D", "static_cb_0.dateofbirth_337D")
    if all(name in frames for name in pair):
        left, right = (frames[name] for name in pair)
        both = left.join(right, on="case_id", how="inner", suffix="_other").filter(
            pl.col("birth").is_not_null() & pl.col("birth_other").is_not_null()
        )
        if both.height:
            same = both.select(
                (pl.col("birth").str.slice(0, 4) == pl.col("birth_other").str.slice(0, 4)).mean()
            ).item()
            print(f"  {pair[0]} vs {pair[1]}: same birth year for {same:.4%} of {both.height:,}")


def main() -> int:
    if not SAMPLE.exists():
        print(
            f"No sample at {SAMPLE}. Run the make-sample kernel and pull it with\n"
            "  kaggle kernels output <user>/hc-make-sample -p data/sample",
            file=sys.stderr,
        )
        return 1
    data_dir = find_data_dir(SAMPLE)
    print(f"reading {data_dir}\n")
    post_decision_leakage(data_dir)
    birth_column_coverage(data_dir)
    print(
        "\nFull-data equivalents, where a stage records them, are in "
        "docs/feature_report.csv and docs/decision_summary.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
