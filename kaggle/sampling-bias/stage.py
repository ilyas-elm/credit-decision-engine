"""The sampling-bias experiment on the full data.

Answers question 3: if the lender only observed outcomes for loans it approved,
how wrong would its validation be, and which fix recovers the most per unit of
cost, reject inference, or a small random-approval budget?

LIMITATION, restated because it bounds every number here: the Home Credit data
already contains booked loans only. The real rejects are gone. This simulates a
SECOND selection policy on top of an already-selected population, so it measures
recovery under a known synthetic selection, not under the lender's original one.

Design, from the brief:

    policy weeks (~25%)  fit the "old policy" scorecard
    dev weeks    (~45%)  apply it, reject the worst r%, HIDE those labels
    test weeks   (~30%)  full population with all labels: ground truth

Every arm gets identical features, an identical subsample and identical fixed
hyperparameters, so the arms differ only in how they treat rejects. No early
stopping anywhere: every validation set the simulated lender could build is
itself drawn from the accepted population, and using the hidden labels to stop
would leak the ground truth this experiment is measuring against.

DEVIATION from the brief, stated: the brief asks for one kernel per reject rate,
as a safeguard against the 12 h session limit. Measured fit times here are about
10 s per model on the 150k subsample, so all four rates run in one kernel in
well under an hour. The per-rate loop is still the outer loop, so splitting it
remains a one-line change if the grid ever grows.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT = Path("/kaggle/working")

#: Acceptance rate at which arms are compared on profit. The profit-maximising
#: cutoff is degenerate on this portfolio (100% approval, where profit does not
#: depend on the model at all), so arms are compared where a cutoff actually
#: binds. Any binding rate serves, since the comparison is between arms at the
#: SAME rate; this one was the decision layer's operating point when the grid
#: was run. That operating point has since moved to 71.5% as the champion
#: changed, which does not invalidate the arm comparison but does mean the two
#: numbers are no longer the same thing.
PROFIT_ACCEPTANCE = 0.515
INPUT_ROOT = Path("/kaggle/input")


def ensure_packages() -> None:
    missing = []
    for module in ("optbinning", "empulse"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if not missing:
        return
    print(f"[sb] installing {missing}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise SystemExit(f"pip install failed: {proc.returncode}")
    print("[sb] installed", flush=True)


def find_features() -> Path:
    matches = sorted(INPUT_ROOT.rglob("features.parquet"))
    if matches:
        return matches[0]
    listing = "\n".join(f"  {p}" for p in sorted(INPUT_ROOT.rglob("*"))[:40])
    raise SystemExit(f"No features.parquet under {INPUT_ROOT}:\n{listing}")


def main() -> None:
    started = time.time()
    ensure_packages()

    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import polars as pl

    from calibrate import assess_calibration
    from config import load_config
    from metrics import gini
    from profit import ProfitAssumptions, strategy_curve
    from sampling_bias import (
        SamplingBiasParams,
        build_arm,
        evaluation_bias,
        exploration_cost,
        make_scenario,
        recovery_share,
        select_features,
        window_weeks,
    )
    from scorecard import ScorecardParams, fit_scorecard, iv_table

    cfg = load_config()
    params = SamplingBiasParams.from_config(cfg["sampling_bias"])
    sc_params = ScorecardParams.from_config(cfg["scorecard"])
    assumptions = ProfitAssumptions.from_config(cfg["profit"])
    print(f"[sb] params: {params}", flush=True)

    features_path = find_features()
    features = pl.read_parquet(features_path)
    feature_cols = [c for c in features.columns if c not in ("case_id", "WEEK_NUM", "target")]
    blocks = window_weeks(features["WEEK_NUM"].unique().to_list(), cfg.sampling_bias_windows)
    win = features["WEEK_NUM"].replace_strict(
        {w: n for n, ws in blocks.items() for w in ws}, default="?", return_dtype=pl.Utf8
    )
    features = features.with_columns(win.alias("window"))
    for name in ("policy", "dev", "test"):
        part = features.filter(pl.col("window") == name)
        print(
            f"[sb]   {name:6} weeks {blocks[name][0]:>3}-{blocks[name][-1]:<3} "
            f"n={part.height:>9,} default={part['target'].mean():.4f}",
            flush=True,
        )

    def block(name):
        part = features.filter(pl.col("window") == name)
        return part.select(feature_cols).to_pandas(), part["target"].to_numpy().astype(float)

    X_pol, y_pol = block("policy")
    X_dev, y_dev = block("dev")
    X_te, y_te = block("test")

    # --- features and old policy, fitted on POLICY weeks only ------------
    # The main model's ranking must not be reused: it was fitted with labels
    # from the development weeks, including the ones this experiment hides.
    t0 = time.time()
    ivs = iv_table(X_pol, y_pol, sc_params).set_index("feature")["iv"].to_dict()
    gain_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=200,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    ).fit(X_pol, y_pol)
    gains = dict(zip(feature_cols, gain_model.booster_.feature_importance("gain"), strict=True))
    chosen = select_features(ivs, gains, params.n_features)
    print(
        f"\n[sb] {len(chosen)} features chosen on policy weeks ({time.time() - t0:.0f}s)",
        flush=True,
    )

    policy = fit_scorecard(X_pol, y_pol, sc_params, features=chosen)
    policy_pd_dev = policy.predict_pd(X_dev)
    policy_pd_test = policy.predict_pd(X_te)
    print(
        f"[sb] old policy: policy-week Gini {gini(y_pol, policy.predict_pd(X_pol)):+.4f}, "
        f"test-week Gini {gini(y_te, policy_pd_test):+.4f}",
        flush=True,
    )

    X_pol = X_dev = None  # free

    lgb_kwargs = dict(
        n_estimators=params.n_trees,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=10.0,
        n_jobs=-1,
        verbose=-1,
        is_unbalance=False,
        scale_pos_weight=1.0,
    )

    def fit_and_score(training, model_type, seed):
        X = training.X[chosen]
        w = training.weight
        if model_type == "lightgbm":
            m = lgb.LGBMClassifier(random_state=seed, **lgb_kwargs)
            m.fit(X, training.y, sample_weight=w)
            return np.asarray(m.predict_proba(X_te[chosen])[:, 1], dtype=float)
        sc = fit_scorecard(X, training.y, sc_params, features=chosen, sample_weight=w)
        return sc.predict_pd(X_te[chosen])

    dev_full = features.filter(pl.col("window") == "dev")
    rows, exploration_rows = [], []

    for reject_rate in params.reject_rates:
        scen_all = make_scenario(
            dev_full.select(feature_cols).to_pandas(),
            y_dev_all := dev_full["target"].to_numpy().astype(float),
            policy_pd_dev,
            reject_rate,
        )
        # Fixed stratified subsample of the development weeks, identical for
        # every arm so differences are the arm and not the data.
        rng_sub = np.random.default_rng(params.seeds[0])
        n_sub = min(params.dev_subsample, len(scen_all.X))
        idx = rng_sub.choice(len(scen_all.X), size=n_sub, replace=False)
        scen = make_scenario(
            scen_all.X.iloc[idx].reset_index(drop=True),
            y_dev_all[idx],
            policy_pd_dev[idx],
            reject_rate,
        )
        accepted_pd = scen.policy_pd[scen.accepted]
        reject_pd = scen.policy_pd[scen.rejected]
        print(
            f"\n[sb] === reject rate {reject_rate}% : {n_sub:,} dev cases, "
            f"{scen.n_rejected:,} rejected ===",
            flush=True,
        )

        # What a lender could measure: the test cases its policy would approve.
        n_rej_te = int(round(len(X_te) * reject_rate / 100.0))
        order = np.argsort(-policy_pd_test, kind="stable")
        accepted_te = np.ones(len(X_te), dtype=bool)
        accepted_te[order[:n_rej_te]] = False

        specs = [
            ("A0_oracle", {}, [params.seeds[0]]),
            ("A1_accepts_only", {}, [params.seeds[0]]),
            ("A2_hard_cutoff", {}, [params.seeds[0]]),
        ]
        specs.append(("A3_fuzzy", {"reject_pd": reject_pd}, [params.seeds[0]]))
        for k in params.parcelling_uplift:
            specs.append(
                (
                    "A4_parcelling",
                    {"reject_pd": reject_pd, "accepted_pd": accepted_pd, "uplift": k},
                    list(params.seeds),
                )
            )
        specs.append(
            (
                "A5_self_learning",
                {
                    "reject_pd": reject_pd,
                    "confidence": params.self_learning_confidence,
                    "iterations": params.self_learning_iters,
                },
                [params.seeds[0]],
            )
        )
        for p in params.exploration_pct:
            specs.append(("A6_exploration", {"explore_pct": p}, list(params.seeds)))

        baseline = {}
        for arm_name, kwargs, seeds in specs:
            for seed in seeds:
                training = build_arm(arm_name, scen, rng=np.random.default_rng(seed), **kwargs)
                for model_type in ("lightgbm", "scorecard"):
                    t = time.time()
                    try:
                        pd_te = fit_and_score(training, model_type, seed)
                    except Exception as exc:  # a degenerate arm must not kill the grid
                        print(f"[sb]   {training.arm:24} {model_type:9} FAILED: {exc}", flush=True)
                        continue
                    g_full = gini(y_te, pd_te)
                    g_acc = gini(y_te[accepted_te], pd_te[accepted_te])
                    cal = assess_calibration(y_te, pd_te)
                    cal_acc = assess_calibration(y_te[accepted_te], pd_te[accepted_te])
                    # Profit on the FULL test population at a FIXED acceptance
                    # rate, not at each arm's own optimum.
                    #
                    # The optimum is useless here: at these assumptions the book
                    # is profitable at every cutoff, so every arm's optimum is
                    # 100% approval and the profit there is identical for every
                    # model (61,715 for the oracle and for pure noise alike).
                    # Comparing arms needs a cutoff that BINDS, where a better
                    # ranking approves fewer bads and earns more. PROFIT_ACCEPTANCE
                    # is the binding operating point the decision layer uses.
                    curve = strategy_curve(y_te, pd_te, assumptions, n_points=201)
                    at = int(np.argmin(np.abs(curve["approval_rate"] - PROFIT_ACCEPTANCE)))
                    profit_full = float(curve["profit"][at])
                    rows.append(
                        {
                            "reject_rate": reject_rate,
                            "arm": training.arm,
                            "base_arm": arm_name,
                            "model": model_type,
                            "seed": seed,
                            "n_train": int(training.y.size),
                            "gini_full": g_full,
                            "gini_accepted_only": g_acc,
                            "brier_full": cal.brier,
                            "citl_gap_full": cal.gap,
                            "citl_gap_accepted_only": cal_acc.gap,
                            "profit_full": profit_full,
                            "seconds": round(time.time() - t, 1),
                            "note": training.note,
                        }
                    )
                    if arm_name in ("A0_oracle", "A1_accepts_only"):
                        baseline[(arm_name, model_type)] = g_full
                    print(
                        f"[sb]   {training.arm:24} {model_type:9} seed={seed} "
                        f"gini_full={g_full:+.4f} gini_acc={g_acc:+.4f} "
                        f"({time.time() - t:.0f}s)",
                        flush=True,
                    )

                # A6 buys its information; price it.
                if arm_name == "A6_exploration":
                    explored = training.weight > 1.0
                    cost = exploration_cost(training.y[explored], assumptions)
                    exploration_rows.append(
                        {
                            "reject_rate": reject_rate,
                            "explore_pct": kwargs["explore_pct"],
                            "seed": seed,
                            "n_explored": int(explored.sum()),
                            "realised_loss_cost": cost,
                            "cost_per_dev_case": cost / max(n_sub, 1),
                        }
                    )

        for row in rows:
            if row["reject_rate"] != reject_rate:
                continue
            key = row["model"]
            o = baseline.get(("A0_oracle", key))
            a1 = baseline.get(("A1_accepts_only", key))
            row["recovery_share"] = (
                recovery_share(o, a1, row["gini_full"])
                if o is not None and a1 is not None
                else float("nan")
            )
            row["evaluation_bias"] = evaluation_bias(row["gini_accepted_only"], row["gini_full"])[
                "bias"
            ]

    arms = pd.DataFrame(rows)
    arms.to_csv(OUTPUT / "arms.csv", index=False)

    # Exploration economics: what the budget BOUGHT, against what it cost.
    explore = pd.DataFrame(exploration_rows)
    if not explore.empty:
        for model_type in ("lightgbm", "scorecard"):
            base_profit = arms[
                (arms.model == model_type) & (arms.base_arm == "A1_accepts_only")
            ].set_index("reject_rate")["profit_full"]
            arm_profit = arms[
                (arms.model == model_type) & (arms.base_arm == "A6_exploration")
            ].set_index(["reject_rate", "arm", "seed"])["profit_full"]
            # itertuples, not iterrows: iterrows coerces each row to a single
            # dtype, which turned explore_pct 1 into 1.0 and built the key
            # "A6_exploration_p1.0", matching nothing and yielding silent NaN.
            gains = [
                arm_profit.get(
                    (int(r.reject_rate), f"A6_exploration_p{int(r.explore_pct)}", int(r.seed)),
                    float("nan"),
                )
                - base_profit.get(int(r.reject_rate), float("nan"))
                for r in explore.itertuples()
            ]
            explore[f"profit_gain_vs_A1_{model_type}"] = gains
            explore[f"net_vs_A1_{model_type}"] = (
                explore[f"profit_gain_vs_A1_{model_type}"] - explore["realised_loss_cost"]
            )
        print("\n=== EXPLORATION ECONOMICS: gain vs cost (mean over seeds) ===", flush=True)
        print(
            explore.groupby(["reject_rate", "explore_pct"])[
                [
                    "realised_loss_cost",
                    "profit_gain_vs_A1_lightgbm",
                    "net_vs_A1_lightgbm",
                ]
            ]
            .mean()
            .round(2)
            .to_string(),
            flush=True,
        )
    explore.to_csv(OUTPUT / "exploration.csv", index=False)

    print("\n=== RAW Gini on the full test population (mean over seeds) ===", flush=True)
    raw = (
        arms.groupby(["model", "base_arm", "reject_rate"])["gini_full"]
        .mean()
        .unstack("reject_rate")
        .round(4)
    )
    print(raw.to_string(), flush=True)

    print("\n=== A0 - A1 gap, which recovery is expressed against ===", flush=True)
    for model in sorted(arms["model"].unique()):
        for rr in sorted(arms["reject_rate"].unique()):
            sel = arms[(arms.model == model) & (arms.reject_rate == rr)]
            o = sel[sel.base_arm == "A0_oracle"]["gini_full"].mean()
            a1 = sel[sel.base_arm == "A1_accepts_only"]["gini_full"].mean()
            usable = (o - a1) >= 0.01
            print(
                f"   {model:10} r={rr:>2}%  A0 {o:+.4f}  A1 {a1:+.4f}  gap {o - a1:+.4f}"
                f"   {'' if usable else '<- no gap to recover; recovery reported as NaN'}",
                flush=True,
            )

    print("\n=== RECOVERY: share of the A0-A1 Gini gap recovered (mean over seeds) ===", flush=True)
    piv = (
        arms.groupby(["model", "base_arm", "reject_rate"])["recovery_share"]
        .mean()
        .unstack("reject_rate")
        .round(3)
    )
    print(piv.to_string(), flush=True)
    print("\n=== EVALUATION BIAS: Gini on accepted-only minus Gini on everyone ===", flush=True)
    bias = arms.groupby(["model", "reject_rate"])["evaluation_bias"].mean().unstack().round(4)
    print(bias.to_string(), flush=True)

    summary = {
        "windows": {n: [int(w[0]), int(w[-1])] for n, w in blocks.items()},
        "n_features": len(chosen),
        "features": chosen,
        "dev_subsample": int(n_sub),
        "n_fits": len(arms),
        "reject_rates": list(params.reject_rates),
        "seeds": list(params.seeds),
        "total_seconds": round(time.time() - started, 1),
    }
    (OUTPUT / "sampling_bias_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    from provenance import write_provenance

    write_provenance(
        OUTPUT, "sampling-bias",
        inputs={"features": features_path},
        outputs={"arms": OUTPUT / "arms.csv"},
        extra={"n_fits": len(arms)},
    )
    print(f"\n[sb] {len(arms)} fits in {summary['total_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
