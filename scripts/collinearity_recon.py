#!/usr/bin/env python
"""Why barring sign flips outright beats pruning correlated pairs.

Three previous-applications characteristics came out of the scorecard with
negative multivariate coefficients, scoring MORE points as their event rate
rose. The README attributes that to joint collinearity across the family
rather than to any one pair, and reports that pruning every pair above the
configured 0.80 still leaves a cluster behind. Those figures were worked out
by hand and one of them was wrong: the condition number was published as 15.1
and is 13.5.

    uv run python scripts/collinearity_recon.py <features.parquet>

`features.parquet` is the output of the `features` kernel, pulled with
`kaggle kernels output`. Pass the full-data one to reproduce the documented
numbers; the 3 % sample runs too and will differ.

Selection is run with `excluded_characteristics` emptied, which is the state
the diagnosis was made in: 16 characteristics rather than the committed 13.
Correlation is measured on the WoE values, because that is what the logistic
regression sees and what `select_features` prunes on -- the raw correlations
are a different matrix and give a different answer.
"""

from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent

#: The three characteristics whose fitted coefficients inverted. config's
#: excluded_characteristics also lists prev_n_applications, which the pair
#: prune removes anyway, so barring it changes nothing and 16 - 13 = 3.
FLIPPED = ("prev_amount_total", "prev_amount_max", "prev_amount_mean")


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
    from features import kept_features
    from scorecard import ScorecardParams, _woe_frame, iv_table, select_features
    from splits import split_weeks

    config = load_config()
    committed = ScorecardParams.from_config(config["scorecard"])
    unbarred = dataclasses.replace(committed, excluded_characteristics=())

    frame = pl.scan_parquet(features)
    built = [
        c for c in frame.collect_schema().names() if c not in ("case_id", "WEEK_NUM", "target")
    ]
    candidates = [c for c in kept_features(REPO / "docs" / "feature_report.csv") if c in built]
    weeks = sorted(
        frame.select("WEEK_NUM").unique().collect(engine="streaming")["WEEK_NUM"].to_list()
    )
    train = split_weeks(weeks, config.splits)["train"]
    rows = (
        frame.filter(pl.col("WEEK_NUM").is_in(train))
        .select(candidates + ["target"])
        .collect(engine="streaming")
    )
    X = rows.select(candidates).to_pandas()
    y = rows["target"].to_numpy().astype(float)
    print(f"train weeks {min(train)}-{max(train)}, {len(X):,} rows, {len(candidates)} candidates\n")

    ivs = iv_table(X, y, unbarred)
    kept_unbarred, _ = select_features(X, y, unbarred, ivs=ivs)
    kept_committed, _ = select_features(X, y, committed, ivs=ivs)
    print(f"  characteristics without the sign-flip bar: {len(kept_unbarred)}")
    print(f"  characteristics with it:                   {len(kept_committed)}")
    barred_out = len(kept_unbarred) - len(kept_committed)
    print(f"  difference:                                {barred_out}")

    survivors = [f for f in kept_unbarred if f.startswith("prev_")]
    print(f"\n  previous-applications survivors of the {committed.max_abs_corr} pair prune:")
    for f in survivors:
        print(f"    {f}")

    flipped = [f for f in FLIPPED if f in survivors]
    if len(flipped) < 2:
        print("\n  the flipping characteristics no longer survive selection; the")
        print("  README's diagnosis describes a state this run does not reach")
        return 0

    woe = _woe_frame(X[flipped], y, unbarred)
    corr = woe.corr().abs()
    pairs = sorted(
        (corr.loc[a, b], a, b) for a, b in itertools.combinations(flipped, 2)
    )
    print(f"\n  WoE |corr| among the {len(flipped)} that flipped:")
    for value, a, b in pairs:
        print(f"    {a:20} {b:20} {value:.2f}")
    print(f"    range {pairs[0][0]:.2f}-{pairs[-1][0]:.2f}, every pair under the "
          f"{committed.max_abs_corr} prune")
    print(f"\n  condition number of their WoE correlation matrix: "
          f"{np.linalg.cond(corr.values):.1f}")
    print(
        "\nNo pair breaches the threshold and the cluster is still there, which is"
        "\nwhy excluding one flip only moved it to the next characteristic. The"
        "\ncondition number is of the CORRELATION matrix; the standardised data"
        f"\nmatrix gives {np.linalg.cond(((woe - woe.mean()) / woe.std()).fillna(0).values):.1f}"
        " and the raw columns something else again, so the"
        "\ndefinition has to be stated with the number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
