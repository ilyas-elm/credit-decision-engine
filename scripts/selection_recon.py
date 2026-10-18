#!/usr/bin/env python
"""Why the accepts-only scorecard beats its own oracle at low reject rates.

Question 3 reports an anomaly: at 10 % and 20 % reject rates the scorecard
trained WITHOUT the rejects' labels scores higher out of time than the one
trained with every label. That is the wrong way round for a sampling-bias
experiment, and "binned logistic regression is more robust" is a description
rather than an explanation.

This reproduces it with the control that identifies the cause. Three fits on
the same development weeks:

    oracle            every case
    accepts-only      the worst 10 % by the old policy removed
    random control    a random 10 % removed

The control is run over several seeds, because one draw is not enough to say
anything: across six it spans 0.415 to 0.431 of test Gini, which is wider than
the effect being attributed. Reporting a single draw is how an earlier version
of this overstated the case.

    uv run python scripts/selection_recon.py <features.parquet>

`features.parquet` is the features kernel's output, pulled with
`kaggle kernels output`. The full-data one reproduces the documented numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
REJECT_RATE = 10
#: One random draw is noise; six give a spread to compare the effect against.
CONTROL_SEEDS = 6


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
    from metrics import gini
    from sampling_bias import SamplingBiasParams, make_scenario, window_weeks
    from scorecard import ScorecardParams, _interval_rows, fit_scorecard

    config = load_config()
    params = SamplingBiasParams.from_config(config["sampling_bias"])
    scorecard_params = ScorecardParams.from_config(config["scorecard"])
    frame = pl.scan_parquet(features)
    columns = [
        c for c in frame.collect_schema().names() if c not in ("case_id", "WEEK_NUM", "target")
    ]
    weeks = sorted(
        frame.select("WEEK_NUM").unique().collect(engine="streaming")["WEEK_NUM"].to_list()
    )
    windows = window_weeks(weeks, config.sampling_bias_windows)

    def block(names: list[int]):
        rows = frame.filter(pl.col("WEEK_NUM").is_in(names)).collect(engine="streaming")
        return rows.select(columns).to_pandas(), rows["target"].to_numpy().astype(float)

    X_policy, y_policy = block(windows["policy"])
    X_dev, y_dev = block(windows["dev"])
    X_test, y_test = block(windows["test"])

    policy = fit_scorecard(X_policy, y_policy, scorecard_params, features=columns)
    policy_pd = policy.predict_pd(X_dev)

    rng = np.random.default_rng(params.seeds[0])
    whole = make_scenario(X_dev, y_dev, policy_pd, REJECT_RATE)
    subsample = rng.choice(
        len(whole.X), size=min(params.dev_subsample, len(whole.X)), replace=False
    )
    scenario = make_scenario(
        whole.X.iloc[subsample].reset_index(drop=True),
        y_dev[subsample],
        policy_pd[subsample],
        REJECT_RATE,
    )
    truth = scenario.oracle_labels()
    accepted = scenario.accepted
    keep_n = int(accepted.mean() * len(scenario.X))
    arms = {
        "oracle, every case": (scenario.X, truth),
        f"accepts-only, worst {REJECT_RATE} % removed": (
            scenario.X[accepted].reset_index(drop=True),
            truth[accepted],
        ),
    }
    for seed in range(CONTROL_SEEDS):
        keep = np.random.default_rng(100 + seed).permutation(len(scenario.X))[:keep_n]
        arms[f"control {seed}, random {REJECT_RATE} % removed"] = (
            scenario.X.iloc[keep].reset_index(drop=True),
            truth[keep],
        )
    print(
        f"development weeks {min(windows['dev'])}-{max(windows['dev'])}, "
        f"{len(scenario.X):,} subsampled; test weeks "
        f"{min(windows['test'])}-{max(windows['test'])}\n"
    )
    print(
        f"  {'fit on':42} {'n':>8} {'in-sample':>10} {'test':>8} {'gap':>8} {'bins':>6} {'IV':>7}"
    )
    for name, (X, y) in arms.items():
        model = fit_scorecard(X, y, scorecard_params, features=columns)
        in_sample = gini(y, model.predict_pd(X))
        out = gini(y_test, model.predict_pd(X_test))
        bins = sum(len(_interval_rows(model.table, f)) for f in model.features)
        total_iv = sum(
            float(_interval_rows(model.table, f)["IV"].astype(float).sum()) for f in model.features
        )
        print(
            f"  {name:42} {len(X):>8,} {in_sample:>10.4f} {out:>8.4f} "
            f"{in_sample - out:>+8.4f} {bins:>6} {total_iv:>7.3f}"
        )
    print(
        "\nRead the GAP column, not the test column. Removing any 10 % lifts test"
        "\nGini a little, and the random controls spread widely enough to cover most"
        "\nof the difference. What only the policy removal does is collapse the"
        "\ngeneralisation gap and the bin count: the rejected tail is sparse and"
        "\nextreme, the binning carves it finely, and those bins do not survive out"
        "\nof time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
