"""Build and screen features on the full competition data.

The expensive stage: 243M source rows, of which credit_bureau_a_2 alone is 188M
across 11 parts. Downstream kernels read this output through `kernel_sources`
so it runs once.

What it emits to /kaggle/working:

    features.parquet      one float32 row per case_id, plus WEEK_NUM and target
    feature_report.csv    the acceptance artefact: feature, family, missing %,
                          IV, PSI, univariate Gini, kept/dropped, reason
    screening_summary.json  counts and timings, for the run log

Screening is fitted on the TRAIN weeks only, exactly as locally. The test weeks
are never read for any selection decision -- that is what keeps the out-of-time
evaluation out-of-time.

Why this needs internet: optbinning computes the IV used by the screen and is
not on the Kaggle image. kaggle/env-probe confirms the install is clean and
leaves numpy, pandas and scikit-learn untouched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

OUTPUT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")


def ensure_optbinning() -> None:
    """Install the one package the screen needs that the image lacks."""
    try:
        import optbinning  # noqa: F401

        print("[features] optbinning already present", flush=True)
        return
    except ImportError:
        pass
    print("[features] installing optbinning", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "optbinning"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise SystemExit(f"pip install optbinning failed with {proc.returncode}")
    print("[features] optbinning installed", flush=True)


def main() -> None:
    started = time.time()
    ensure_optbinning()

    from config import load_config
    from features import (
        FeatureParams,
        ScreeningParams,
        build_features,
        find_data_dir,
        table_paths,
    )
    from splits import split_weeks

    cfg = load_config()
    data_dir = find_data_dir(INPUT_ROOT)
    print(f"[features] reading from {data_dir}", flush=True)

    found = {t: len(table_paths(data_dir, t)) for t in ("train_base", "train_credit_bureau_a_2")}
    print(f"[features] part counts: {found}", flush=True)
    if not table_paths(data_dir, "train_base"):
        listing = [str(p) for p in sorted(INPUT_ROOT.rglob("*"))[:40]]
        raise SystemExit(
            f"No train_base under {data_dir}. /kaggle/input holds:\n" + "\n".join(listing)
        )

    params = FeatureParams.from_config(cfg["features"])
    print(f"[features] params: {params}", flush=True)

    t0 = time.time()
    features, family_of = build_features(data_dir, params, out_path=OUTPUT / "features.parquet")
    build_seconds = time.time() - t0
    print(
        f"[features] built {features.height:,} rows x {features.width} cols "
        f"in {build_seconds / 60:.1f} min",
        flush=True,
    )

    # Splits come from the same config as everywhere else, so the train block
    # the screen is fitted on is the block the model will be fitted on.
    blocks = split_weeks(features["WEEK_NUM"].unique().to_list(), cfg.splits)
    split_of_week = {week: name for name, weeks in blocks.items() for week in weeks}
    print(
        "[features] week blocks: " + ", ".join(f"{n}={len(w)}w" for n, w in blocks.items()),
        flush=True,
    )

    from features import screen_features

    t1 = time.time()
    report = screen_features(
        features,
        family_of,
        split_of_week,
        ScreeningParams.from_config(cfg["screening"]),
        min_iv=cfg["scorecard"]["min_iv"],
    )
    screen_seconds = time.time() - t1
    report.insert(0, "source", "full_data")
    report.to_csv(OUTPUT / "feature_report.csv", index=False)

    kept = report.loc[report["kept"], "feature"].tolist()
    flagged = report.loc[report["leakage_review"], "feature"].tolist()
    print(f"\n[features] screened in {screen_seconds:.0f}s: {len(kept)} of {len(report)} kept")
    print(report.to_string(index=False), flush=True)
    if flagged:
        print(f"\n[features] FLAGGED for manual leakage review: {flagged}", flush=True)

    size_mb = (OUTPUT / "features.parquet").stat().st_size / 1e6
    summary = {
        "n_rows": int(features.height),
        "n_features_built": len(family_of),
        "n_features_kept": len(kept),
        "kept": kept,
        "flagged_for_review": flagged,
        "week_blocks": {name: [int(w[0]), int(w[-1])] for name, w in blocks.items()},
        "build_seconds": round(build_seconds, 1),
        "screen_seconds": round(screen_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
        "features_parquet_mb": round(size_mb, 1),
        "polars": pl.__version__,
        "python": ".".join(str(v) for v in sys.version_info[:3]),
    }
    (OUTPUT / "screening_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[features] features.parquet {size_mb:.1f} MB", flush=True)
    print(f"[features] total {summary['total_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
