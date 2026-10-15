"""Champion vs challenger on the full data.

Reads `features.parquet` from the hc-features kernel through `kernel_sources`,
so the expensive feature build runs once and every model stage shares exactly
the same inputs.

What it answers: question 1 of the brief, what a gradient-boosted challenger
adds over a classic scorecard once stability, calibration and profit are
measured rather than only AUC.

Discipline carried over from the local modules:

* Feature selection and binning are fitted on the TRAIN weeks only.
* Early stopping and isotonic calibration use val_a.
* val_b is left completely untouched here. It is the conformal calibration
  block for the decision layer, and the guarantee is void if anything else is
  fitted on it.
* The test weeks are scored and never fitted.

Outputs to /kaggle/working:

    comparison.csv       champion vs challenger, the acceptance table
    predictions.parquet  per-case PD and score for both models, with the split
                         label, so downstream stages need not refit anything
    scorecard_table.csv  characteristic / bin / WoE / points
    weekly_gini.png, calibration.png, strategy_curve.png
    models_summary.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

OUTPUT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")


def ensure_packages() -> None:
    """optbinning and empulse are not on the Kaggle image."""
    missing = []
    for module, package in (("optbinning", "optbinning"), ("empulse", "empulse")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if not missing:
        print("[models] dependencies already present", flush=True)
        return
    print(f"[models] installing {missing}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise SystemExit(f"pip install {missing} failed with {proc.returncode}")
    print("[models] installed", flush=True)


def find_feature_report() -> Path:
    matches = sorted(INPUT_ROOT.rglob("feature_report.csv"))
    if not matches:
        raise SystemExit(f"No feature_report.csv under {INPUT_ROOT}; screening cannot be applied.")
    return matches[0]


def find_features() -> Path:
    """Locate features.parquet in whichever directory the upstream kernel mounts at.

    Not assumed: the competition data mounts a level deeper than documented, so
    the upstream kernel's output path is not worth guessing either. Failing with
    a listing costs one run and answers the question.
    """
    matches = sorted(INPUT_ROOT.rglob("features.parquet"))
    if matches:
        print(f"[models] features at {matches[0]}", flush=True)
        return matches[0]
    listing = "\n".join(f"  {p}" for p in sorted(INPUT_ROOT.rglob("*"))[:40])
    raise SystemExit(f"No features.parquet under {INPUT_ROOT}. It holds:\n{listing}")


def main() -> None:
    started = time.time()
    ensure_packages()

    import polars as pl

    from calibrate import IsotonicCalibrator, assess_calibration, needs_calibration
    from challenger import ChallengerParams, fit_challenger
    from config import load_config
    from evaluation import comparison_table, evaluate, write_figures
    from features import kept_features
    from profit import ProfitAssumptions
    from scorecard import ScorecardParams, fit_scorecard, iv_table, select_features
    from splits import SPLIT_ORDER, split_weeks

    cfg = load_config()
    features_path = find_features()
    features = pl.read_parquet(features_path)
    print(f"[models] {features.height:,} rows x {features.width} cols", flush=True)

    # Honour the screening decisions. features.parquet carries every feature
    # that was built, not just the survivors, so without this the scorecard's
    # own IV rule happily re-selects something the PSI screen dropped.
    built = [c for c in features.columns if c not in ("case_id", "WEEK_NUM", "target")]
    report_path = find_feature_report()
    feature_cols = [c for c in kept_features(report_path) if c in built]
    print(f"[models] screening kept {len(feature_cols)} of {len(built)} built; "
          f"excluded {sorted(set(built) - set(feature_cols))}", flush=True)
    blocks = split_weeks(features["WEEK_NUM"].unique().to_list(), cfg.splits)
    split_of_week = {w: name for name, ws in blocks.items() for w in ws}
    split = features["WEEK_NUM"].replace_strict(split_of_week, default="?", return_dtype=pl.Utf8)
    features = features.with_columns(split.alias("split"))
    for name in SPLIT_ORDER:
        block = features.filter(pl.col("split") == name)
        print(
            f"[models]   {name:6} weeks {blocks[name][0]:>3}-{blocks[name][-1]:<3} "
            f"n={block.height:>9,} default={block['target'].mean():.4f}",
            flush=True,
        )

    def block(name: str):
        part = features.filter(pl.col("split") == name)
        return (
            part.select(feature_cols).to_pandas(),
            part["target"].to_numpy(),
            part["WEEK_NUM"].to_numpy(),
            part["case_id"].to_numpy(),
        )

    X_tr, y_tr, _, _ = block("train")
    X_va, y_va, _, _ = block("val_a")
    X_te, y_te, w_te, id_te = block("test")

    # --- champion -------------------------------------------------------
    params_sc = ScorecardParams.from_config(cfg["scorecard"])
    t0 = time.time()
    ivs = iv_table(X_tr, y_tr, params_sc)
    kept, decisions = select_features(X_tr, y_tr, params_sc, ivs=ivs)
    print(f"\n[models] scorecard selection kept {len(kept)} of {len(feature_cols)}:", flush=True)
    print(decisions[["feature", "iv", "kept", "reason"]].to_string(index=False), flush=True)
    if not kept:
        raise SystemExit("Scorecard selection kept no features.")

    champion = fit_scorecard(X_tr, y_tr, params_sc, features=kept)
    print(f"[models] scorecard fitted in {time.time() - t0:.0f}s", flush=True)
    champion.points_table().to_csv(OUTPUT / "scorecard_table.csv", index=False)

    # --- challenger -----------------------------------------------------
    params_ch = ChallengerParams.from_config(cfg["challenger"])
    constraints = champion.monotone_constraints(list(X_tr.columns))
    print(
        "[models] monotone constraints: "
        f"{sum(c == 1 for c in constraints)} ascending, "
        f"{sum(c == -1 for c in constraints)} descending, "
        f"{sum(c == 0 for c in constraints)} free",
        flush=True,
    )
    t1 = time.time()
    challenger = fit_challenger(
        X_tr, y_tr, X_va, y_va, params=params_ch, monotone_constraints=constraints
    )
    print(
        f"[models] lightgbm fitted in {time.time() - t1:.0f}s, "
        f"best_iteration={challenger.best_iteration}",
        flush=True,
    )

    # --- calibration ----------------------------------------------------
    # The challenger is calibrated unconditionally; the champion only if it is
    # miscalibrated in the large, which is the brief's rule.
    champ_val_pd = champion.predict_pd(X_va)
    champ_assessment = assess_calibration(y_va, champ_val_pd)
    champ_needs = needs_calibration(y_va, champ_val_pd)
    print(
        f"\n[models] champion on val_a: mean PD {champ_assessment.mean_pd:.4f} vs "
        f"observed {champ_assessment.observed_rate:.4f} "
        f"(gap {champ_assessment.gap:+.4f}) -> "
        f"{'calibrating' if champ_needs else 'already calibrated'}",
        flush=True,
    )
    champ_calibrator = IsotonicCalibrator().fit(champ_val_pd, y_va) if champ_needs else None

    def champion_pd(frame):
        raw = champion.predict_pd(frame)
        return raw if champ_calibrator is None else champ_calibrator.transform(raw)

    # --- evaluation -----------------------------------------------------
    assumptions = ProfitAssumptions.from_config(cfg["profit"])
    grid = cfg["profit"]["sensitivity"]
    pd_champ_te, pd_chall_te = champion_pd(X_te), challenger.predict_pd(X_te)

    evaluations = [
        evaluate(
            "scorecard",
            y_te,
            pd_champ_te,
            w_te,
            test_scores=pd_champ_te,
            train_scores=champion_pd(X_tr),
            assumptions=assumptions,
            sensitivity_grid=grid,
        ),
        evaluate(
            "lightgbm",
            y_te,
            pd_chall_te,
            w_te,
            test_scores=pd_chall_te,
            train_scores=challenger.predict_pd(X_tr),
            assumptions=assumptions,
            sensitivity_grid=grid,
        ),
    ]
    table = comparison_table(evaluations)
    table.to_csv(OUTPUT / "comparison.csv", index=False)
    print("\n=== CHAMPION vs CHALLENGER, test weeks ===", flush=True)
    print(table.to_string(index=False), flush=True)

    figures = write_figures(evaluations, OUTPUT)
    print(f"\n[models] figures: {[p.name for p in figures.values()]}", flush=True)

    # --- predictions for downstream stages -------------------------------
    # Every split, so the decision layer can calibrate conformal on val_b
    # without refitting anything.
    all_X = features.select(feature_cols).to_pandas()
    predictions = features.select("case_id", "WEEK_NUM", "target", "split").with_columns(
        pl.Series("pd_scorecard", champion_pd(all_X)).cast(pl.Float32),
        pl.Series("score_scorecard", champion.score(all_X)).cast(pl.Float32),
        pl.Series("pd_lightgbm", challenger.predict_pd(all_X)).cast(pl.Float32),
        pl.Series("pd_lightgbm_raw", challenger.predict_pd_raw(all_X)).cast(pl.Float32),
    )
    predictions.write_parquet(OUTPUT / "predictions.parquet")

    gains = challenger.gain_importance()
    summary = {
        "n_rows": int(features.height),
        "features_available": len(feature_cols),
        "features_in_scorecard": kept,
        "monotone_constraints": dict(zip(list(X_tr.columns), constraints, strict=True)),
        "champion_calibrated": bool(champ_needs),
        "champion_val_a_gap": round(float(champ_assessment.gap), 5),
        "challenger_best_iteration": int(challenger.best_iteration or 0),
        "challenger_top_gain": {k: round(float(v), 1) for k, v in gains.head(10).items()},
        "week_blocks": {n: [int(w[0]), int(w[-1])] for n, w in blocks.items()},
        "comparison": json.loads(table.to_json(orient="records")),
        "total_seconds": round(time.time() - started, 1),
    }
    (OUTPUT / "models_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    from provenance import write_provenance

    write_provenance(
        OUTPUT, "models",
        inputs={"features": features_path, "feature_report": report_path},
        outputs={"predictions": OUTPUT / "predictions.parquet"},
        extra={"features_in_scorecard": kept},
    )
    print(f"\n[models] total {summary['total_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
