"""The decision layer on the full data.

Answers question 2: where the approval cutoff and the manual-review band should
sit, and what they cost.

Chains from hc-features (to refit the champion for recourse bins) and hc-models
(for the per-case PDs already produced), and reads the competition data for the
two columns the fairness audit needs, which are not in the engineered features.

val_b finally gets used, and only here. Nothing else in this project is fitted
on it, which is what makes the conformal coverage guarantee meaningful rather
than decorative.

The cutoff is reported across the whole profit sensitivity grid rather than as a
single number, because at the centre assumptions it is degenerate: the book is
profitable at every approval rate, so the profit-maximising cutoff approves
everyone and declines nobody. The decision layer is then exercised at the
tightest BINDING point in the grid, which is the only place where an approval
cutoff is a real decision on this portfolio.
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
    missing = [m for m in ("optbinning", "empulse") if not _have(m)]
    if not missing:
        return
    print(f"[dec] installing {missing}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:], flush=True)
        raise SystemExit(f"pip install failed: {proc.returncode}")


def _have(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _find(name: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name))
    if not matches:
        listing = "\n".join(f"  {p}" for p in sorted(INPUT_ROOT.rglob("*"))[:40])
        raise SystemExit(f"No {name} under {INPUT_ROOT}:\n{listing}")
    return matches[0]


def main() -> None:
    started = time.time()
    ensure_packages()

    import numpy as np
    import pandas as pd
    import polars as pl

    from config import load_config
    from conformal import (
        Band,
        MondrianConformal,
        coverage_report,
        marginal_thresholds,
        one_minus_prob,
    )
    from decision import DecisionEngine, choose_cutoff, referral_economics
    from features import find_data_dir, kept_features, table_paths
    from profit import ProfitAssumptions, expected_loss_fraction
    from scorecard import ScorecardParams, fit_scorecard, select_features

    cfg = load_config()
    dec_cfg = cfg["decision"]
    alpha = dec_cfg["conformal_alpha"]

    features_path = _find("features.parquet")
    predictions_path = _find("predictions.parquet")
    features = pl.read_parquet(features_path)
    preds = pl.read_parquet(predictions_path)
    # Honour the screening decisions, exactly as the models stage does. Without
    # this the refit champion re-selects features the PSI screen dropped, and
    # then explains decisions with a different model than the one that made
    # them, bureau_a_records turned up in the principal reasons that way.
    built = [c for c in features.columns if c not in ("case_id", "WEEK_NUM", "target")]
    feature_cols = [c for c in kept_features(_find("feature_report.csv")) if c in built]
    print(f"[dec] screening kept {len(feature_cols)} of {len(built)} built features", flush=True)
    joined = features.join(
        preds.select("case_id", "split", "pd_scorecard", "score_scorecard", "pd_lightgbm"),
        on="case_id",
        how="inner",
    )
    print(
        f"[dec] {joined.height:,} cases; splits {sorted(joined['split'].unique().to_list())}",
        flush=True,
    )

    def part(name):
        return joined.filter(pl.col("split") == name)

    tr, vb, te = part("train"), part("val_b"), part("test")
    print(
        f"[dec] val_b {vb.height:,} cases ({int(vb['target'].sum()):,} defaults), "
        f"test {te.height:,}",
        flush=True,
    )

    # --- refit the champion, and prove it is the same model ---------------
    sc_params = ScorecardParams.from_config(cfg["scorecard"])
    X_tr = tr.select(feature_cols).to_pandas()
    y_tr = tr["target"].to_numpy()
    kept, _ = select_features(X_tr, y_tr, sc_params)
    champion = fit_scorecard(X_tr, y_tr, sc_params, features=kept)
    refit = champion.predict_pd(te.select(feature_cols).to_pandas())
    stored = te["pd_scorecard"].to_numpy().astype(float)
    # The stored PD is calibrated; the refit is not, so compare the RANKING,
    # which calibration cannot change. A mismatch means the stages disagree.
    order_match = float(
        np.corrcoef(np.argsort(np.argsort(refit)), np.argsort(np.argsort(stored)))[0, 1]
    )
    print(f"[dec] refit champion vs hc-models ranking correlation: {order_match:.6f}", flush=True)
    flips = champion.sign_flips()
    print(f"[dec] scorecard sign flips: {flips or 'none'}", flush=True)

    # --- conformal band, calibrated on val_b ONLY -------------------------
    # The whole decision layer runs on the CHAMPION's probabilities. The band,
    # the cutoff, the reasons and the recourse then all describe one model.
    # Calibrating the band on the challenger while explaining the decision with
    # the scorecard would mean the reasons justify a decision a different model
    # made, which is both incoherent and, for a lender, a governance problem.
    # The challenger's coverage is reported below for comparison only.
    y_vb, pd_vb = vb["target"].to_numpy(), vb["pd_scorecard"].to_numpy().astype(float)
    y_te, pd_te = te["target"].to_numpy(), te["pd_scorecard"].to_numpy().astype(float)
    w_te = te["WEEK_NUM"].to_numpy()

    mondrian = MondrianConformal.fit(pd_vb, y_vb, alpha)
    print(
        f"\n[dec] Mondrian thresholds {mondrian.thresholds} from n={mondrian.n_calibration}",
        flush=True,
    )

    cov = coverage_report(mondrian, pd_te, y_te, w_te)
    cov_df = pd.DataFrame(cov)
    cov_df.to_csv(OUTPUT / "conformal_coverage_by_week.csv", index=False)
    overall = coverage_report(mondrian, pd_te, y_te)
    print(f"[dec] target coverage {1 - alpha:.2f}", flush=True)
    for label, name in ((0, "non-default"), (1, "DEFAULT")):
        row = overall["coverage"][overall["label"] == label][0]
        wk = cov_df[cov_df.label == label]["coverage"].dropna()
        print(
            f"[dec]   {name:12} overall {row:.4f} | weekly min {wk.min():.4f} "
            f"max {wk.max():.4f} | weeks below target: "
            f"{int((wk < 1 - alpha).sum())}/{len(wk)}",
            flush=True,
        )

    # The challenger, for comparison only: it does not drive any decision here.
    chal = MondrianConformal.fit(vb["pd_lightgbm"].to_numpy().astype(float), y_vb, alpha)
    chal_cov = coverage_report(chal, te["pd_lightgbm"].to_numpy().astype(float), y_te)
    print(
        f"[dec]   challenger band would give: non-default "
        f"{chal_cov['coverage'][0]:.4f}, DEFAULT {chal_cov['coverage'][1]:.4f}",
        flush=True,
    )

    # Marginal, for the comparison the write-up needs.
    sc_cal = one_minus_prob(np.column_stack([1 - pd_vb, pd_vb]))
    flat = MondrianConformal(
        alpha, marginal_thresholds(sc_cal[np.arange(y_vb.size), y_vb], alpha), {}
    )
    flat_cov = coverage_report(flat, pd_te, y_te)
    print(
        f"[dec]   marginal threshold would give: "
        f"non-default {flat_cov['coverage'][0]:.4f}, DEFAULT {flat_cov['coverage'][1]:.4f}",
        flush=True,
    )

    # --- cutoff across the profit grid ------------------------------------
    # Real scorecard points, not a negated PD: recourse works in points, so the
    # cutoff has to live on the same scale the recourse search moves along.
    scores_te = te["score_scorecard"].to_numpy().astype(float)
    grid, cut_rows = cfg["profit"]["sensitivity"], []
    for sr in grid["success_rate"]:
        for dr in grid["default_rate"]:
            for roi in grid["roi"]:
                if sr + dr > 1:
                    continue
                a = ProfitAssumptions(sr, dr, roi)
                c = choose_cutoff(y_te, pd_te, scores_te, a)
                cut_rows.append(
                    {
                        "success_rate": sr,
                        "default_rate": dr,
                        "roi": roi,
                        "break_even_bad_rate": roi / (roi + expected_loss_fraction(a)),
                        "acceptance_rate": c.acceptance_rate,
                        "profit_per_applicant": c.profit_per_applicant,
                        "empcs_implied_acceptance": c.empcs_implied_acceptance_rate,
                    }
                )
    cuts = pd.DataFrame(cut_rows).sort_values("acceptance_rate")
    cuts.to_csv(OUTPUT / "cutoff_sensitivity.csv", index=False)
    binding = cuts[cuts.acceptance_rate < 0.999]
    print(
        f"\n[dec] cutoff across {len(cuts)} assumption points: "
        f"acceptance {cuts.acceptance_rate.min():.1%} to {cuts.acceptance_rate.max():.1%}",
        flush=True,
    )
    print(f"[dec] points where a cutoff BINDS: {len(binding)}/{len(cuts)}", flush=True)
    if binding.empty:
        raise SystemExit("No binding cutoff anywhere in the grid; widen it before running this.")
    op = binding.iloc[0]
    print(
        f"[dec] operating point: roi={op.roi} default_rate={op.default_rate} "
        f"success_rate={op.success_rate} -> approve {op.acceptance_rate:.1%}",
        flush=True,
    )

    op_assumptions = ProfitAssumptions(op.success_rate, op.default_rate, op.roi)
    cutoff = choose_cutoff(y_te, pd_te, scores_te, op_assumptions)

    # --- bands and referral economics -------------------------------------
    engine = DecisionEngine(
        champion,
        mondrian,
        cutoff,
        actionable_features=[f for f in dec_cfg["actionable_features"] if f in champion.features],
        top_reasons=4,
    )
    print(
        f"[dec] actionable characteristics in the scorecard: {engine.actionable_features}",
        flush=True,
    )

    X_te = te.select(feature_cols).to_pandas()
    sets = mondrian.prediction_sets(pd_te)
    singleton = sets.sum(axis=1) == 1
    bands = np.where(scores_te >= cutoff.score_threshold, Band.APPROVE, Band.DECLINE).astype(object)
    bands[~singleton] = Band.REFER
    auto = scores_te >= cutoff.score_threshold

    counts = {b: int((bands == b).sum()) for b in (Band.APPROVE, Band.REFER, Band.DECLINE)}
    print(
        "\n[dec] bands on the test weeks: "
        + ", ".join(f"{k} {v:,} ({v / len(bands):.1%})" for k, v in counts.items()),
        flush=True,
    )
    for b in (Band.APPROVE, Band.REFER, Band.DECLINE):
        m = bands == b
        if m.any():
            print(f"[dec]   {b:8} bad rate {y_te[m].mean():.4f}", flush=True)

    econ = referral_economics(
        y_te,
        bands,
        auto,
        cutoff,
        review_costs=list(dec_cfg["review_cost_grid"]),
        review_accuracy=1.0,
    )
    econ_df = pd.DataFrame(econ)
    econ_df.to_csv(OUTPUT / "referral_economics.csv", index=False)
    print("\n[dec] referral economics (perfect reviewer, an upper bound):", flush=True)
    print(econ_df.to_string(index=False), flush=True)

    # --- decide() on real applicants --------------------------------------
    declined_idx = np.flatnonzero(bands == Band.DECLINE)[:200]
    examples = []
    if declined_idx.size:
        decisions = engine.decide_batch(X_te.iloc[declined_idx])
        for d in decisions:
            examples.append(
                {
                    "pd": round(d.pd, 5),
                    "score": round(d.score, 1),
                    "band": d.band,
                    "principal_reasons": [(n, round(v, 1)) for n, v in d.principal_reasons],
                    "recourse_feasible": bool(d.recourse and d.recourse.feasible),
                    "recourse": [
                        {
                            "characteristic": s.characteristic,
                            "from": s.from_bin,
                            "to": s.to_bin,
                            "points": round(s.points_gained, 1),
                        }
                        for s in (d.recourse.steps if d.recourse else [])
                    ],
                    "recourse_note": d.recourse.note if d.recourse else "",
                }
            )
        feasible = sum(e["recourse_feasible"] for e in examples)
        print(
            f"\n[dec] decide() on {len(examples)} declined applicants: "
            f"{feasible} have feasible recourse ({feasible / len(examples):.0%})",
            flush=True,
        )
        from collections import Counter

        reasons = Counter(n for e in examples for n, _ in e["principal_reasons"])
        print(f"[dec] most common principal reasons: {reasons.most_common(6)}", flush=True)
        (OUTPUT / "decide_examples.json").write_text(json.dumps(examples[:25], indent=2))

    # --- how far is recourse from working, per band ------------------------
    # decide() only attaches recourse to a DECLINE, which on its own makes the
    # result look like a flat "recourse never works". It does not: outright
    # declines are decisively bad, while the much larger REFERRED population
    # sits within reach of the actionable characteristics. Reporting the
    # shortfall against the available headroom separates those two.
    best_points = champion.max_points()
    awarded_points = champion.feature_points(X_te)
    headroom = np.zeros(len(X_te))
    for feature in engine.actionable_features:
        headroom += np.maximum(best_points[feature] - awarded_points[feature].to_numpy(), 0.0)
    shortfall = cutoff.score_threshold - scores_te

    reach = {}
    for label, mask in (
        ("below_cutoff_any_band", scores_te < cutoff.score_threshold),
        ("decline_band", bands == Band.DECLINE),
        ("refer_band", bands == Band.REFER),
    ):
        m = mask & (shortfall > 0)
        if not m.any():
            continue
        reach[label] = {
            "n": int(m.sum()),
            "mean_shortfall_points": round(float(shortfall[m].mean()), 1),
            "mean_actionable_headroom_points": round(float(headroom[m].mean()), 1),
            "share_reachable": round(float((headroom[m] >= shortfall[m]).mean()), 4),
        }
    print("\n[dec] how far recourse is from clearing the cutoff:", flush=True)
    for label, r in reach.items():
        print(
            f"[dec]   {label:22} n={r['n']:>7,}  short {r['mean_shortfall_points']:>5.1f} pts  "
            f"headroom {r['mean_actionable_headroom_points']:>5.1f} pts  "
            f"reachable {r['share_reachable']:.1%}",
            flush=True,
        )

    # --- fairness audit ----------------------------------------------------
    audit = {}
    try:
        data_dir = find_data_dir(INPUT_ROOT)
        person = table_paths(data_dir, "train_person_1")
        cb = table_paths(data_dir, "train_static_cb_0")
        demo = None
        if person:
            demo = (
                pl.concat([pl.scan_parquet(p) for p in person], how="diagonal_relaxed")
                .filter(pl.col("num_group1") == 0)
                .select("case_id", pl.col("sex_738L").alias("sex"))
                .collect(engine="streaming")
            )
        # Birth date: chosen by COVERAGE ON THE ROWS BEING AUDITED, not by
        # name. The first attempt used static_cb_0.birthdate_574D, which is 40 %
        # populated overall and looks perfectly serviceable, but collection of
        # it stops dead at about week 40, so it is 0 % across the test weeks and
        # the audit came back empty. person_1.birth_259D is 100 % populated
        # there. A column's overall fill rate says nothing about the period you
        # are measuring.
        birth_candidates: list[tuple[str, str, list]] = [
            ("person_1", "birth_259D", table_paths(data_dir, "train_person_1")),
            ("static_cb_0", "dateofbirth_337D", cb),
            ("static_cb_0", "birthdate_574D", cb),
        ]
        test_ids = te.select("case_id")
        birth, best = None, 0.0
        for table, column, paths in birth_candidates:
            if not paths:
                continue
            scan = pl.concat([pl.scan_parquet(q) for q in paths], how="diagonal_relaxed")
            if column not in scan.collect_schema().names():
                continue
            if table == "person_1":
                scan = scan.filter(pl.col("num_group1") == 0)
            candidate = scan.select(
                "case_id", pl.col(column).cast(pl.Utf8, strict=False).alias("birth")
            ).collect(engine="streaming")
            on_test = test_ids.join(candidate, on="case_id", how="left")
            coverage = float(on_test["birth"].is_not_null().mean() or 0.0)
            print(
                f"[dec] birth candidate {table}.{column}: test-week coverage {coverage:.4f}",
                flush=True,
            )
            if coverage > best:
                birth, best = candidate, coverage
        if birth is not None and best > 0.0:
            demo = (
                birth if demo is None else demo.join(birth, on="case_id", how="full", coalesce=True)
            )
            print(f"[dec] using birth column with {best:.1%} test-week coverage", flush=True)
        if demo is not None:
            t = (
                te.select("case_id", "target")
                .with_columns(pl.Series("band", bands.astype(str)), pl.Series("pd", pd_te))
                .join(demo, on="case_id", how="left")
            )
            if "birth" in t.columns:
                t = t.with_columns(
                    (
                        (
                            pl.lit(2020)
                            - pl.col("birth").str.slice(0, 4).cast(pl.Int32, strict=False)
                        )
                        // 10
                        * 10
                    ).alias("age_band")
                )
            for col in [c for c in ("sex", "age_band") if c in t.columns]:
                g = (
                    t.filter(pl.col(col).is_not_null())
                    .group_by(col)
                    .agg(
                        pl.len().alias("n"),
                        (pl.col("band") == "approve").mean().alias("approval_rate"),
                        pl.col("target").mean().alias("observed_default"),
                        pl.col("pd").mean().alias("mean_pd"),
                    )
                    .sort(col)
                )
                audit[col] = g.to_dicts()
                print(f"\n[dec] fairness audit by {col} (audit only, no mitigation):", flush=True)
                print(g.to_pandas().to_string(index=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[dec] fairness audit skipped: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "alpha": alpha,
        "mondrian_thresholds": {str(k): v for k, v in mondrian.thresholds.items()},
        "n_calibration": {str(k): v for k, v in mondrian.n_calibration.items()},
        "coverage_overall": {
            str(int(lab)): float(c)
            for lab, c in zip(overall["label"], overall["coverage"], strict=True)
        },
        "marginal_coverage": {
            str(int(lab)): float(c)
            for lab, c in zip(flat_cov["label"], flat_cov["coverage"], strict=True)
        },
        "band_counts": counts,
        "operating_point": {
            k: float(op[k]) for k in ("success_rate", "default_rate", "roi", "acceptance_rate")
        },
        "cutoff_acceptance_range": [
            float(cuts.acceptance_rate.min()),
            float(cuts.acceptance_rate.max()),
        ],
        "binding_points": int(len(binding)),
        "scorecard_sign_flips": flips,
        "refit_ranking_correlation": order_match,
        "recourse_reach": reach,
        "fairness": audit,
        "total_seconds": round(time.time() - started, 1),
    }
    (OUTPUT / "decision_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    from provenance import write_provenance

    write_provenance(
        OUTPUT, "decision",
        inputs={"features": features_path, "predictions": predictions_path},
        outputs={"decision_summary": OUTPUT / "decision_summary.json"},
    )
    print(f"\n[dec] total {summary['total_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
