"""The shared evaluation module, and a champion-vs-challenger run end to end."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from challenger import ChallengerParams, fit_challenger
from config import load_config
from evaluation import comparison_table, evaluate, write_figures
from profit import ProfitAssumptions
from scorecard import ScorecardParams, fit_scorecard


@pytest.fixture(scope="module")
def drifting_blocks():
    """train / val / test with the test weeks drifting, as the real data does.

    Discrimination decays slightly over the test weeks, so the stability
    metric has a real falling slope to penalise rather than a flat line.
    """
    rng = np.random.default_rng(0)

    def draw(n, week_start, n_weeks, signal=1.2):
        weeks = np.repeat(np.arange(week_start, week_start + n_weeks), n // n_weeks)
        n = weeks.size
        y = rng.binomial(1, 0.10, n)
        # Signal fades as weeks advance: a model trained early degrades later.
        decay = signal * (1.0 - 0.25 * (weeks - week_start) / max(n_weeks - 1, 1))
        return (
            pd.DataFrame(
                {
                    "rising": rng.normal(0, 1, n) + decay * y,
                    "falling": rng.normal(0, 1, n) - decay * y,
                    "noise": rng.normal(0, 1, n),
                }
            ),
            y,
            weeks,
        )

    return {
        "train": draw(9_000, 0, 30),
        "val": draw(3_000, 30, 8),
        "test": draw(6_000, 38, 20),
    }


@pytest.fixture(scope="module")
def evaluations(drifting_blocks):
    """Champion and challenger, both evaluated through the same function."""
    (X_tr, y_tr, _), (X_va, y_va, _), (X_te, y_te, w_te) = (
        drifting_blocks["train"],
        drifting_blocks["val"],
        drifting_blocks["test"],
    )
    cfg = load_config()
    assumptions = ProfitAssumptions.from_config(cfg["profit"])
    grid = cfg["profit"]["sensitivity"]

    champion = fit_scorecard(X_tr, y_tr, ScorecardParams())
    params = ChallengerParams(n_estimators=200, early_stopping_rounds=25)
    challenger = fit_challenger(
        X_tr,
        y_tr,
        X_va,
        y_va,
        params=params,
        monotone_constraints=champion.monotone_constraints(list(X_tr.columns)),
    )

    champ_eval = evaluate(
        "scorecard",
        y_te,
        champion.predict_pd(X_te),
        w_te,
        test_scores=champion.predict_pd(X_te),
        train_scores=champion.predict_pd(X_tr),
        assumptions=assumptions,
        sensitivity_grid=grid,
    )
    chall_eval = evaluate(
        "lightgbm",
        y_te,
        challenger.predict_pd(X_te),
        w_te,
        test_scores=challenger.predict_pd(X_te),
        train_scores=challenger.predict_pd(X_tr),
        assumptions=assumptions,
        sensitivity_grid=grid,
    )
    return [champ_eval, chall_eval]


# --- Structure -----------------------------------------------------------


def test_evaluation_reports_every_required_quantity(evaluations) -> None:
    for e in evaluations:
        assert e.n == 6_000
        assert e.n_weeks == 20
        assert 0.0 < e.default_rate < 0.3
        assert -1.0 <= e.gini_overall <= 1.0
        assert len(e.weekly) == 20
        assert e.stability.n_weeks == 20
        assert 0.0 <= e.brier <= 1.0
        assert e.calibration_buckets["n"].sum() == 6_000
        assert e.citl["week"].size == 20
        assert np.isfinite(e.score_psi)
        assert np.isfinite(e.profit.emp)
        assert len(e.profit_sensitivity) > 1
        assert e.strategy["approval_rate"][-1] == pytest.approx(1.0)


def test_stability_metric_matches_a_direct_computation(evaluations) -> None:
    """The headline number must equal the metric applied to the weekly curve."""
    from metrics import stability_metric

    for e in evaluations:
        direct = stability_metric(e.weekly)
        assert e.stability.metric == pytest.approx(direct.metric)
        assert e.stability.slope == pytest.approx(direct.slope)


def test_the_drift_shows_up_as_a_falling_slope(evaluations) -> None:
    # The fixture fades the signal over the test weeks, so at least one model
    # must register a negative trend; otherwise the fixture is not exercising
    # the thing the stability metric exists to measure.
    assert any(e.stability.slope < 0 for e in evaluations)


def test_stability_metric_penalises_a_falling_slope(evaluations) -> None:
    for e in evaluations:
        if e.stability.slope < 0:
            # metric = mean + 88*slope - 0.5*resid_std, so it must sit below mean.
            assert e.stability.metric < e.stability.mean_gini


def test_both_models_discriminate(evaluations) -> None:
    for e in evaluations:
        assert e.gini_overall > 0.25, f"{e.name} barely discriminates"


# --- Score PSI -----------------------------------------------------------


def test_score_psi_is_nan_without_both_blocks(drifting_blocks) -> None:
    """A PSI against the wrong reference looks like a real number and is not."""
    (X_tr, y_tr, _), _, (X_te, y_te, w_te) = (
        drifting_blocks["train"],
        drifting_blocks["val"],
        drifting_blocks["test"],
    )
    champion = fit_scorecard(X_tr, y_tr, ScorecardParams())
    out = evaluate("no-psi", y_te, champion.predict_pd(X_te), w_te)
    assert np.isnan(out.score_psi)


def test_score_psi_detects_a_shifted_score_distribution(drifting_blocks) -> None:
    (X_tr, y_tr, _), _, (X_te, y_te, w_te) = (
        drifting_blocks["train"],
        drifting_blocks["val"],
        drifting_blocks["test"],
    )
    champion = fit_scorecard(X_tr, y_tr, ScorecardParams())
    test_pd = champion.predict_pd(X_te)
    shifted = evaluate(
        "shifted",
        y_te,
        test_pd,
        w_te,
        test_scores=np.clip(test_pd * 3.0, 0, 1),
        train_scores=champion.predict_pd(X_tr),
    )
    assert shifted.score_psi > 0.25


# --- Validation ----------------------------------------------------------


def test_evaluation_refuses_too_few_scoreable_weeks() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    weeks = np.array([0, 0, 0, 0])
    with pytest.raises(ValueError, match="needs 2"):
        evaluate("one-week", y, p, weeks)


def test_evaluation_refuses_mismatched_or_empty_input() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate("bad", np.array([0, 1]), np.array([0.1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="empty test set"):
        evaluate("empty", np.array([]), np.array([]), np.array([]))


# --- Comparison table ----------------------------------------------------


def test_comparison_table_has_one_row_per_model(evaluations) -> None:
    table = comparison_table(evaluations)
    assert list(table["model"]) == ["scorecard", "lightgbm"]
    for column in (
        "gini",
        "mean_weekly_gini",
        "gini_slope",
        "stability_metric",
        "brier",
        "citl_max_abs_gap",
        "score_psi_vs_train",
        "empcs",
        "empcs_min",
        "empcs_max",
    ):
        assert column in table.columns, column
    assert table.notna().all().all()


def test_comparison_table_reports_a_profit_range_not_a_point(evaluations) -> None:
    """EMPCS assumptions are assumptions; a single number is not reportable."""
    table = comparison_table(evaluations).set_index("model")
    for model in table.index:
        assert table.loc[model, "empcs_min"] <= table.loc[model, "empcs"]
        assert table.loc[model, "empcs"] <= table.loc[model, "empcs_max"]
        assert table.loc[model, "empcs_min"] < table.loc[model, "empcs_max"]


def test_empty_comparison_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one evaluation"):
        comparison_table([])


# --- Figures -------------------------------------------------------------


def test_the_three_acceptance_figures_are_written(evaluations, tmp_path) -> None:
    paths = write_figures(evaluations, tmp_path)
    assert set(paths) == {"weekly_gini", "calibration", "strategy_curve"}
    for name, path in paths.items():
        assert path.is_file(), name
        # A real PNG, not an empty file.
        assert path.stat().st_size > 5_000
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_work_for_a_single_model(evaluations, tmp_path) -> None:
    paths = write_figures(evaluations[:1], tmp_path)
    assert all(p.is_file() for p in paths.values())


def test_referral_figure_marks_the_break_even_price(tmp_path) -> None:
    """The brief's section-6 figure: share referred vs loss avoided, by cost."""
    from evaluation import plot_referral_economics

    economics = {
        "review_cost": np.array([0.0, 0.005, 0.01, 0.02, 0.05]),
        "share_referred": np.full(5, 0.45),
        "n_referred": np.full(5, 102_054),
        "loss_avoided": np.full(5, 1981.6),
        "total_review_cost": np.array([0.0, 510.3, 1020.5, 2041.1, 5102.7]),
        "net_benefit": np.array([1981.6, 1471.3, 961.1, -59.5, -3121.1]),
    }
    path = plot_referral_economics(economics, tmp_path / "referral.png")
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 5_000


def test_referral_figure_handles_a_band_that_always_pays(tmp_path) -> None:
    """No break-even inside the swept range: the plot must still render."""
    from evaluation import plot_referral_economics

    economics = {
        "review_cost": np.array([0.0, 0.01]),
        "share_referred": np.full(2, 0.1),
        "n_referred": np.full(2, 100),
        "loss_avoided": np.full(2, 5000.0),
        "total_review_cost": np.array([0.0, 1.0]),
        "net_benefit": np.array([5000.0, 4999.0]),
    }
    assert plot_referral_economics(economics, tmp_path / "r.png").is_file()
