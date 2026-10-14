"""Build the local development sample, and report what the data actually is.

Two jobs in one kernel, because both need a full pass over the data and a Kaggle
session is 12 h:

  1. SAMPLE. Pick ~3% of case_ids, stratified by (WEEK_NUM, target), and write
     every table filtered to those ids as parquet. Pulled locally into
     data/sample/ (gitignored). Target: under 1 GB on disk.

  2. RECONNAISSANCE. Emit schema_report.json: the real tables, their row counts,
     the WEEK_NUM range, the weekly default rate, and whether the columns later
     decisions depend on actually exist. The brief's table list is explicitly
     unverified, so docs/DATA_NOTES.md is written from this output, not from the
     brief.

Memory: nothing is ever fully materialised. Tables are scanned lazily and
filtered against the sampled ids, then streamed to parquet. credit_bureau_a_2
alone is many GB across many parts, so a collect() anywhere here would end the
session.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

INPUT = Path("/kaggle/input/home-credit-credit-risk-model-stability")
OUTPUT = Path("/kaggle/working")
SAMPLE_DIR = OUTPUT / "sample"

SAMPLE_FRACTION = 0.03
SAMPLE_SEED = 20251013

# Columns later choices depend on. If credit amount is absent, profit uses unit
# exposure; if age and sex are absent, the fairness audit is dropped entirely.
# Searched as case-insensitive substrings, since the real names are unverified.
COLUMNS_OF_INTEREST = {
    "credit_amount": ["credamount", "amtcredit", "credit_amount", "amount_credit"],
    "interest_rate": ["interestrate", "interest_rate", "nominalrate", "rate"],
    "age_or_birth": ["birth", "age", "dateofbirth", "days_birth"],
    "sex": ["sex", "gender", "code_gender"],
    "annuity": ["annuity"],
    "income": ["income", "incometotal"],
}


# The competition ships parquet_files/train/ and parquet_files/test/. The test
# tables are the unlabeled leaderboard submission set -- no `target` column, and
# train_base is ~1.5M rows against test_base's few hundred. Leaderboard
# submissions are out of scope, so only the train side is ever read.
#
# This is not cosmetic: the base table used to be picked with
# `endswith("_base")`, and "test" sorts before "train", so it selected
# test_base and the kernel would have sampled a few hundred unlabeled rows and
# then failed on the missing target.
TABLE_PREFIX = "train"


def discover_tables(prefix: str = TABLE_PREFIX) -> dict[str, list[Path]]:
    """Group the parquet files by logical table, collapsing `_0`, `_1`, ... parts.

    Verified against the real file list: 68 parquet files collapse to 17 train
    tables, including credit_bureau_a_2 with 11 parts (part numbers run past 10,
    so the depth suffix has to be distinguished from the part number).
    """
    tables: dict[str, list[Path]] = {}
    for path in sorted(INPUT.rglob("*.parquet")):
        stem = path.stem
        if not stem.startswith(f"{prefix}_"):
            continue
        # train_credit_bureau_a_2_10 -> train_credit_bureau_a_2
        parts = stem.split("_")
        while len(parts) > 1 and parts[-1].isdigit():
            candidate = "_".join(parts[:-1])
            # Depth suffixes _0/_1/_2 are part of the name, not a part number.
            if parts[-1] in {"0", "1", "2"} and not candidate.endswith(("_0", "_1", "_2")):
                break
            parts = parts[:-1]
        tables.setdefault("_".join(parts), []).append(path)
    return tables


def pick_sample_ids(base_files: list[Path]) -> pl.DataFrame:
    """~3% of case_ids, stratified by (WEEK_NUM, target).

    Stratifying on both keeps every week represented and preserves the default
    rate, so the local sample is usable for week-wise work rather than just a
    random slice that thins out the rare class.
    """
    base = pl.concat([pl.scan_parquet(path) for path in base_files])
    strata = (
        base.select("case_id", "WEEK_NUM", "target")
        .collect(engine="streaming")
        .with_columns(
            pl.int_range(pl.len()).shuffle(seed=SAMPLE_SEED).over("WEEK_NUM", "target").alias("_r"),
            pl.len().over("WEEK_NUM", "target").alias("_n"),
        )
    )
    # At least one case per stratum, so no (week, target) cell disappears.
    keep = strata.filter(
        pl.col("_r") < pl.max_horizontal(pl.lit(1), (pl.col("_n") * SAMPLE_FRACTION).ceil())
    )
    return keep.select("case_id", "WEEK_NUM", "target")


def base_report(base_files: list[Path]) -> dict:
    base = pl.concat([pl.scan_parquet(path) for path in base_files])
    summary = base.select(
        pl.len().alias("n_rows"),
        pl.col("case_id").n_unique().alias("n_cases"),
        pl.col("WEEK_NUM").min().alias("week_min"),
        pl.col("WEEK_NUM").max().alias("week_max"),
        pl.col("WEEK_NUM").n_unique().alias("n_weeks"),
        pl.col("target").mean().alias("default_rate"),
        pl.col("date_decision").min().alias("date_min"),
        pl.col("date_decision").max().alias("date_max"),
    ).collect(engine="streaming")

    weekly = (
        base.group_by("WEEK_NUM")
        .agg(pl.len().alias("n"), pl.col("target").mean().alias("default_rate"))
        .sort("WEEK_NUM")
        .collect(engine="streaming")
    )
    return {
        "summary": summary.to_dicts()[0],
        "weekly_default_rate": weekly.to_dicts(),
    }


def column_search(tables: dict[str, list[Path]]) -> dict:
    """Where the decision-relevant columns actually live, by exact name."""
    found: dict[str, list[dict]] = {key: [] for key in COLUMNS_OF_INTEREST}
    for table, files in tables.items():
        schema = pl.scan_parquet(files[0]).collect_schema()
        for column, dtype in schema.items():
            lowered = column.lower()
            for key, needles in COLUMNS_OF_INTEREST.items():
                if any(needle in lowered for needle in needles):
                    found[key].append({"table": table, "column": column, "dtype": str(dtype)})
    return found


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    tables = discover_tables()
    print(f"[make-sample] {len(tables)} logical tables", flush=True)

    base_key = f"{TABLE_PREFIX}_base"
    if base_key not in tables:
        raise SystemExit(
            f"No {base_key!r} among the discovered tables: {sorted(tables)}. "
            f"Expected {INPUT}/parquet_files/{TABLE_PREFIX}/{base_key}.parquet."
        )

    report: dict = {
        "tables": {},
        "base": base_report(tables[base_key]),
        "columns_of_interest": column_search(tables),
        "sample": {"fraction": SAMPLE_FRACTION, "seed": SAMPLE_SEED},
        "polars_version": pl.__version__,
    }
    print(json.dumps(report["base"]["summary"], indent=2, default=str), flush=True)

    sample_ids = pick_sample_ids(tables[base_key])
    ids_only = sample_ids.select("case_id")
    print(f"[make-sample] sampled {ids_only.height} case_ids", flush=True)
    report["sample"]["n_cases"] = ids_only.height

    total_bytes = 0
    for table, files in sorted(tables.items()):
        lazy = pl.concat([pl.scan_parquet(path) for path in files])
        schema = lazy.collect_schema()

        n_rows = lazy.select(pl.len()).collect(engine="streaming").item()
        destination = SAMPLE_DIR / f"{table}.parquet"
        # Streaming join against the id list; never materialise the table.
        lazy.join(ids_only.lazy(), on="case_id", how="inner").sink_parquet(destination)

        size = destination.stat().st_size
        total_bytes += size
        sampled_rows = pl.scan_parquet(destination).select(pl.len()).collect().item()
        report["tables"][table] = {
            "n_parts": len(files),
            "n_rows_full": int(n_rows),
            "n_rows_sample": int(sampled_rows),
            "n_columns": len(schema),
            "sample_bytes": int(size),
            "columns": {name: str(dtype) for name, dtype in schema.items()},
        }
        print(
            f"[make-sample] {table}: {n_rows:,} -> {sampled_rows:,} rows, {size / 1e6:.1f} MB",
            flush=True,
        )

    report["sample"]["total_bytes"] = total_bytes
    print(f"[make-sample] sample total: {total_bytes / 1e9:.2f} GB", flush=True)
    if total_bytes > 1e9:
        print(
            "[make-sample] WARNING: sample exceeds the 1 GB target; lower "
            "SAMPLE_FRACTION before pulling it locally.",
            flush=True,
        )

    (OUTPUT / "schema_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("[make-sample] wrote schema_report.json", flush=True)


if __name__ == "__main__":
    main()
