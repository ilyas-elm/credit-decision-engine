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

#: Reporting choices for the age-gap attribution, not model parameters. The
#: audit already bands age by decade, and 60 is where approval collapses in
#: this book, so it is the split the attribution explains.
OLDER_FROM_AGE = 60
#: The characteristic whose absence is tested as an age proxy. It is derived
#: from the tax registry, which records employment payments, so retirees are
#: missing from it by construction rather than by chance.
INCOME_CHARACTERISTIC = "afford_taxincome_total"


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
        alpha_sweep,
        coverage_report,
        marginal_thresholds,
        one_minus_prob,
    )
    from decision import (
        DecisionEngine,
        age_gap_attribution,
        choose_cutoff,
        referral_economics,
    )
    from evaluation import plot_alpha_sweep
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

    # --- what alpha would buy ---------------------------------------------
    # The band refers 44 % of applicants at alpha = 0.10, which no review desk
    # could staff, and alpha is the only lever: recalibrating split conformal
    # is re-taking a quantile, not refitting. The curve is swept rather than
    # asserted because it is not monotone, and a desk sizing itself off the
    # referral rate alone could land on the wrong side of it.
    sweep = pd.DataFrame(
        alpha_sweep(pd_vb, y_vb, pd_te, y_te, [round(0.02 * i, 2) for i in range(1, 36)])
    )
    sweep.to_csv(OUTPUT / "conformal_alpha_sweep.csv", index=False)
    plot_alpha_sweep(
        {c: sweep[c].to_numpy() for c in sweep.columns},
        OUTPUT / "alpha_sweep.png",
    )
    # Not `singleton`: the band logic below binds that name to a mask.
    no_referrals = sweep.loc[sweep["refer_rate"] < 1e-9, "alpha"]
    workable = sweep.loc[sweep["refer_rate"] <= 0.10]
    print("\n[dec] alpha sweep: referral rate against coverage", flush=True)
    for _, r in sweep.iloc[::4].iterrows():
        print(
            f"[dec]   alpha {r['alpha']:.2f}  refer {r['refer_rate']:6.1%} "
            f"(both {r['both_labels']:5.1%}, empty {r['empty_set']:5.1%})  "
            f"cov {r['coverage_non_default']:.3f} / {r['coverage_default']:.3f}",
            flush=True,
        )
    if not workable.empty:
        first = workable.iloc[0]
        print(
            f"[dec]   a 10 % desk needs alpha >= {first['alpha']:.2f}, where the "
            f"default class is covered {first['coverage_default']:.3f}",
            flush=True,
        )
    if not no_referrals.empty:
        print(
            f"[dec]   the band refers nobody between alpha {no_referrals.min():.2f} "
            f"and {no_referrals.max():.2f}, then returns as empty sets",
            flush=True,
        )
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
        all_ids = joined.select("case_id")
        birth, best = None, 0.0
        # Recorded, not just printed. DATA_NOTES tabulates these and the
        # numbers were previously reconnaissance nobody could re-derive.
        birth_coverage: list[dict] = []
        candidate_frames: dict[str, pl.DataFrame] = {}
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
            # Not `overall`: that name holds the conformal coverage report
            # this function still needs further down.
            overall_coverage = float(
                all_ids.join(candidate, on="case_id", how="left")["birth"]
                .is_not_null()
                .mean()
                or 0.0
            )
            birth_coverage.append(
                {
                    "column": f"{table}.{column}",
                    "overall": round(overall_coverage, 4),
                    "test_weeks": round(coverage, 4),
                }
            )
            candidate_frames[f"{table}.{column}"] = candidate
            print(
                f"[dec] birth candidate {table}.{column}: overall {overall_coverage:.4f}, "
                f"test-week coverage {coverage:.4f}",
                flush=True,
            )
            if coverage > best:
                birth, best = candidate, coverage
        audit["birth_column_coverage"] = birth_coverage

        # Do two surviving columns actually describe the same person? DATA_NOTES
        # claims they agree; this measures it on the cases that carry both.
        pair = ("person_1.birth_259D", "static_cb_0.dateofbirth_337D")
        if all(name in candidate_frames for name in pair):
            left, right = (candidate_frames[name] for name in pair)
            both = left.join(right, on="case_id", how="inner", suffix="_other").filter(
                pl.col("birth").is_not_null() & pl.col("birth_other").is_not_null()
            )
            if both.height:
                same_year = (
                    both.select(
                        (
                            pl.col("birth").str.slice(0, 4)
                            == pl.col("birth_other").str.slice(0, 4)
                        ).mean()
                    )
                    .item()
                )
                audit["birth_column_agreement"] = {
                    "columns": list(pair),
                    "n_overlapping": int(both.height),
                    "same_birth_year": round(float(same_year), 4),
                }
                print(
                    f"[dec] {pair[0]} and {pair[1]} agree on birth year for "
                    f"{same_year:.4%} of {both.height:,} overlapping cases",
                    flush=True,
                )
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

            # --- the mechanism behind the age gap ---------------------------
            # The audit above is the finding; this is the cause. Age is not an
            # input, so the disparity arrives through characteristics that are,
            # and the model card quotes which ones. Joined on case_id: `t` came
            # out of a join, so its row order need not match X_te and the
            # points matrix.
            if "birth" in t.columns:
                points = pl.DataFrame({"case_id": te["case_id"]}).with_columns(
                    [pl.Series(f, awarded_points[f].to_numpy()) for f in champion.features]
                )
                age_years = t.select(
                    "case_id",
                    (
                        pl.lit(2020)
                        - pl.col("birth").str.slice(0, 4).cast(pl.Int32, strict=False)
                    ).alias("age"),
                    "target",
                ).filter(pl.col("age").is_not_null())
                if INCOME_CHARACTERISTIC in te.columns:
                    points = points.with_columns(
                        pl.Series(
                            "_no_income", te[INCOME_CHARACTERISTIC].is_null().to_numpy()
                        )
                    )
                aligned = points.join(age_years, on="case_id", how="inner")
                if aligned.height:
                    mechanism = age_gap_attribution(
                        aligned.select(champion.features).to_pandas(),
                        aligned["age"].to_numpy(),
                        aligned["target"].to_numpy(),
                        income_missing=(
                            aligned["_no_income"].to_numpy()
                            if "_no_income" in aligned.columns
                            else None
                        ),
                        older_from=OLDER_FROM_AGE,
                    )
                    audit["age_mechanism"] = mechanism
                    print(
                        f"\n[dec] age gap attribution ({OLDER_FROM_AGE}+ vs younger), "
                        f"net {mechanism['total_points_gap']:+.1f} pts, of which "
                        f"{mechanism['points_lost_to_negative_characteristics']:+.1f} "
                        "from characteristics that penalise them:",
                        flush=True,
                    )
                    for row in mechanism["by_characteristic"][:5]:
                        print(
                            f"[dec]   {row['characteristic']:34} "
                            f"{row['points_older']:>6.1f} vs {row['points_younger']:>6.1f}"
                            f"  gap {row['gap']:+.1f}",
                            flush=True,
                        )
                    proxy = mechanism.get("income_record_as_age_proxy")
                    if proxy:
                        print(
                            f"[dec]   an absent {INCOME_CHARACTERISTIC} is a "
                            f"{proxy['accuracy']:.1%}-accurate test for being "
                            f"{OLDER_FROM_AGE}+",
                            flush=True,
                        )
                        for row in mechanism["default_rate_by_age_and_income_record"]:
                            print(
                                f"[dec]   {row['group']:8} {row['income_record']:18} "
                                f"n={row['n']:>7,}  default {row['default_rate']:.4f}",
                                flush=True,
                            )
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
