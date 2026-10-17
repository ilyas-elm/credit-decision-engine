"""The prose must quote the numbers the committed artefacts actually hold.

Every headline figure in README.md and MODEL.md comes from a file under docs/
that a Kaggle run produced. Nothing stopped the two from drifting apart, and
they did: both documents said 7 of 21 assumption points made a cutoff bind
when decision_summary.json said 8, and the sensitivity grid it was computed
from agreed with the JSON.

A reader cannot check a portfolio claim against a run they cannot execute, so
the artefacts are the record and the prose is a view of it. These tests fail
when the view stops matching.

Rerunning a stage legitimately changes these numbers. A failure here is not
automatically a bug in the code: it means the documents still describe the
previous run and need updating to the new one.
"""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"


def _text(name: str) -> str:
    return (REPO / name).read_text()


@pytest.fixture(scope="module")
def readme() -> str:
    return _text("README.md")


@pytest.fixture(scope="module")
def model_doc() -> str:
    return _text("MODEL.md")


@pytest.fixture(scope="module")
def comparison() -> dict[str, dict[str, str]]:
    with (DOCS / "comparison.csv").open() as fh:
        return {row["model"]: row for row in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((DOCS / "decision_summary.json").read_text())


@pytest.fixture(scope="module")
def sensitivity() -> list[dict[str, str]]:
    with (DOCS / "cutoff_sensitivity.csv").open() as fh:
        return list(csv.DictReader(fh))


# --- Headline comparison table -------------------------------------------

# (column in comparison.csv, decimal places as the documents quote it)
HEADLINE = [
    ("gini", 4),
    ("mean_weekly_gini", 4),
    ("stability_metric", 4),
    ("brier", 5),
    ("score_psi_vs_train", 4),
]


@pytest.mark.parametrize("column, places", HEADLINE)
@pytest.mark.parametrize("model", ["scorecard", "lightgbm"])
def test_headline_numbers_are_quoted_as_the_run_computed_them(
    comparison, readme, model_doc, model, column, places
) -> None:
    quoted = f"{float(comparison[model][column]):.{places}f}"
    assert quoted in readme, (
        f"README does not quote {model} {column} = {quoted} from comparison.csv"
    )
    assert quoted in model_doc, (
        f"MODEL.md does not quote {model} {column} = {quoted} from comparison.csv"
    )


# --- The claim that actually drifted -------------------------------------


def test_binding_points_agree_between_the_json_and_the_grid(summary, sensitivity) -> None:
    """decision_summary.json's count must match the grid it was computed from.

    A cutoff binds when the profit-maximising acceptance rate is below 1.0:
    anything at 1.0 means "approve everyone" and no cutoff is exercised.
    """
    from_grid = sum(1 for row in sensitivity if float(row["acceptance_rate"]) < 1.0)
    assert summary["binding_points"] == from_grid, (
        f"decision_summary.json says {summary['binding_points']} binding points, "
        f"cutoff_sensitivity.csv has {from_grid}"
    )


def test_both_documents_quote_the_binding_point_count(
    summary, sensitivity, readme, model_doc
) -> None:
    total = len(sensitivity)
    binding = summary["binding_points"]
    assert f"{binding} of {total}" in readme, (
        f"README must say '{binding} of {total}' points bind"
    )
    assert f"{binding} of {total}" in model_doc, (
        f"MODEL.md must say '{binding} of {total}' assumption points bind"
    )


# --- Decision layer ------------------------------------------------------


def test_conformal_coverage_is_quoted_to_three_places(summary, readme) -> None:
    """Both classes, since the default class missing its target is the finding."""
    for label in ("0", "1"):
        quoted = f"{summary['coverage_overall'][label]:.3f}"
        assert quoted in readme, (
            f"README does not quote class {label} coverage = {quoted}"
        )


def test_marginal_default_coverage_is_quoted(summary, readme) -> None:
    quoted = f"{summary['marginal_coverage']['1']:.3f}"
    assert quoted in readme, (
        f"README does not quote the marginal default coverage = {quoted}, the "
        "number that justifies making the band class-conditional"
    )


def test_operating_acceptance_rate_is_quoted(summary, readme, model_doc) -> None:
    rate = summary["operating_point"]["acceptance_rate"] * 100
    quoted = f"{rate:.1f} %"
    assert quoted in readme or quoted.replace(" ", "") in readme, (
        f"README does not quote the operating acceptance rate = {quoted}"
    )
    assert quoted in model_doc or quoted.replace(" ", "") in model_doc, (
        f"MODEL.md does not quote the operating acceptance rate = {quoted}"
    )


def test_band_counts_sum_to_the_scored_population(summary, comparison) -> None:
    scored = int(comparison["scorecard"]["n"])
    assert sum(summary["band_counts"].values()) == scored, (
        "band counts must partition the scored test population"
    )


def test_no_sign_flips_claim_matches_the_summary(summary, readme) -> None:
    """The README states the scorecard has no sign flips. It must be empty."""
    if summary["scorecard_sign_flips"]:
        pytest.fail(
            f"README claims no sign flips but the summary lists "
            f"{summary['scorecard_sign_flips']}"
        )
    assert "no sign flips" in readme


def test_recourse_reach_for_the_refer_band_is_quoted(summary, readme) -> None:
    share = summary["recourse_reach"]["refer_band"]["share_reachable"] * 100
    quoted = f"{share:.0f} %"
    assert quoted in readme or quoted.replace(" ", "") in readme, (
        f"README does not quote the refer-band recourse reach = {quoted}"
    )


# --- Sampling-bias experiment --------------------------------------------


@pytest.fixture(scope="module")
def arms() -> list[dict[str, str]]:
    with (DOCS / "sampling_bias_arms.csv").open() as fh:
        return list(csv.DictReader(fh))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _numeric(rows: list[dict[str, str]], column: str) -> list[float]:
    out = []
    for row in rows:
        # A blank cell means the arm reports no value here, which is not the
        # same as a zero: A0 and A1 have no recovery share by definition.
        with contextlib.suppress(ValueError):
            out.append(float(row[column]))
    return out


REJECT_RATES = ("10", "20", "30", "50")


@pytest.mark.parametrize("model", ["lightgbm", "scorecard"])
@pytest.mark.parametrize("reject_rate", REJECT_RATES)
def test_evaluation_bias_table_matches_the_arms_csv(arms, readme, model, reject_rate) -> None:
    """Every cell of the evaluation-bias table, to the 3 places it is quoted.

    One cell said -0.070 where the mean over the run's rows is -0.069476. No
    aggregation of that column reaches -0.070, so it was a transcription slip
    rather than a different definition.
    """
    selected = [r for r in arms if r["model"] == model and r["reject_rate"] == reject_rate]
    assert selected, f"no {model} rows at reject rate {reject_rate}"
    quoted = f"{_mean(_numeric(selected, 'evaluation_bias')):.3f}"
    assert quoted in readme, (
        f"README does not quote {model} evaluation bias at r={reject_rate} % = {quoted}"
    )


@pytest.mark.parametrize(
    "base_arm",
    ["A2_hard_cutoff", "A3_fuzzy", "A4_parcelling", "A5_self_learning", "A6_exploration"],
)
@pytest.mark.parametrize("reject_rate", REJECT_RATES)
def test_recovery_table_matches_the_arms_csv(arms, readme, base_arm, reject_rate) -> None:
    """The recovery table, averaged over seeds and over each arm's variants.

    A4 spans k in {1.5, 2} and A6 spans p in {1, 2, 5}, so a cell for those is
    a mean across the variants as well as the seeds.
    """
    selected = [
        r
        for r in arms
        if r["model"] == "lightgbm"
        and r["base_arm"] == base_arm
        and r["reject_rate"] == reject_rate
    ]
    values = _numeric(selected, "recovery_share")
    assert values, f"no recovery_share for {base_arm} at r={reject_rate}"
    quoted = f"{_mean(values):.2f}"
    assert quoted in readme, (
        f"README does not quote {base_arm} recovery at r={reject_rate} % = {quoted}"
    )


def test_the_fit_count_is_the_number_of_rows(arms, readme) -> None:
    assert f"{len(arms)} fits" in readme, (
        f"README must say '{len(arms)} fits' to match sampling_bias_arms.csv"
    )


# --- Fairness audit ------------------------------------------------------
#
# These numbers drifted the furthest. The tables in both documents were
# updated after a re-run and the prose underneath them was not, so the
# README's own table said 0.485 while the sentence below it said 47.6 %.


def test_fairness_rates_reconstruct_the_approve_count(summary) -> None:
    """The artefact has to be self-consistent before the prose is checked.

    Approvals implied by the per-sex rates must equal the approve band, which
    is what established that the JSON was right and the prose was stale.
    """
    by_sex = summary["fairness"]["sex"]
    implied = sum(row["n"] * row["approval_rate"] for row in by_sex)
    assert round(implied) == pytest.approx(summary["band_counts"]["approve"], abs=2)


@pytest.mark.parametrize("sex", ["F", "M"])
def test_sex_approval_and_default_rates_are_quoted(summary, readme, model_doc, sex) -> None:
    row = next(r for r in summary["fairness"]["sex"] if r["sex"] == sex)
    approval = f"{row['approval_rate'] * 100:.1f} %"
    default = f"{row['observed_default'] * 100:.2f} %"
    for name, doc in (("README.md", readme), ("MODEL.md", model_doc)):
        assert approval in doc, f"{name} does not quote {sex} approval = {approval}"
        assert default in doc, f"{name} does not quote {sex} observed default = {default}"


@pytest.mark.parametrize("sex", ["F", "M"])
def test_sex_calibration_gaps_are_quoted(summary, model_doc, sex) -> None:
    """mean PD minus observed default, the number that makes it miscalibration."""
    row = next(r for r in summary["fairness"]["sex"] if r["sex"] == sex)
    gap = f"{row['mean_pd'] - row['observed_default']:+.4f}"
    assert gap in model_doc, f"MODEL.md does not quote the {sex} calibration gap = {gap}"


def test_the_oldest_age_bands_approval_rates_are_quoted(summary, readme, model_doc) -> None:
    bands = {row["age_band"]: row for row in summary["fairness"]["age_band"]}
    for band in (60, 70):
        quoted = f"{bands[band]['approval_rate'] * 100:.1f} %"
        assert quoted in readme, f"README does not quote the {band}s approval = {quoted}"
        assert quoted in model_doc, f"MODEL.md does not quote the {band}s approval = {quoted}"


def test_the_younger_approval_range_is_quoted(summary, readme, model_doc) -> None:
    """The range over the bands below 60, which was quoted a point too wide."""
    younger = [r["approval_rate"] for r in summary["fairness"]["age_band"] if r["age_band"] < 60]
    quoted = f"{min(younger) * 100:.0f}-{max(younger) * 100:.0f} %"
    assert quoted in readme, f"README must give the younger approval range as {quoted}"
    assert quoted in model_doc, f"MODEL.md must give the younger approval range as {quoted}"


def test_every_doc_table_row_matches_the_fairness_artefact(summary, readme) -> None:
    """The tables themselves, so prose and table cannot drift apart again."""
    for row in summary["fairness"]["sex"]:
        cells = f"| {row['n']:,} | {row['approval_rate']:.3f} |"
        assert cells in readme, f"README sex table row for {row['sex']} does not match: {cells}"
    for row in summary["fairness"]["age_band"]:
        assert f"| {row['n']:,} |" in readme, (
            f"README age table is missing the {row['age_band']}s row (n={row['n']:,})"
        )


# --- The synthetic conformal comparison in the README --------------------
#
# This table is the one claim in either document that no Kaggle run produced:
# it is a synthetic experiment, so CI can reproduce it exactly. It was quoted
# as 300 draws while the only committed computation (test_conformal.py) used
# 150 and asserted inequalities, so nothing checked the printed values.


def test_the_synthetic_conformal_table_reproduces_exactly() -> None:
    """Recompute the README's marginal-vs-Mondrian table and match every cell.

    Uses the draw helper from test_conformal so there is one generator, not a
    copy that could drift away from the numbers the suite reasons about.
    """
    import math

    import numpy as np
    from test_conformal import _exchangeable_draw

    from conformal import (
        DEFAULT,
        NON_DEFAULT,
        MondrianConformal,
        coverage_report,
        marginal_thresholds,
        one_minus_prob,
    )

    alpha, draws, n = 0.10, 300, 6_000
    rng = np.random.default_rng(7)
    acc: dict[tuple[str, int], list[float]] = {
        (name, label): [] for name in ("marginal", "mondrian") for label in (NON_DEFAULT, DEFAULT)
    }
    for _ in range(draws):
        pd_cal, y_cal = _exchangeable_draw(rng, n, base_rate=0.03)
        pd_test, _y = _exchangeable_draw(rng, n, base_rate=0.03)
        y_test = _y
        scores = one_minus_prob(np.column_stack([1.0 - pd_cal, pd_cal]))
        shared = marginal_thresholds(scores[np.arange(y_cal.size), y_cal], alpha)
        models = {
            "marginal": MondrianConformal(alpha=alpha, thresholds=shared, n_calibration={}),
            "mondrian": MondrianConformal.fit(pd_cal, y_cal, alpha),
        }
        for name, model in models.items():
            report = coverage_report(model, pd_test, y_test)
            for label in (NON_DEFAULT, DEFAULT):
                value = report["coverage"][report["label"] == label]
                if value.size and not math.isnan(value[0]):
                    acc[(name, label)].append(float(value[0]))

    cells = {key: float(np.mean(values)) for key, values in acc.items()}
    expected = {
        ("marginal", NON_DEFAULT): "0.964",
        ("marginal", DEFAULT): "0.014",
        ("mondrian", NON_DEFAULT): "0.900",
        ("mondrian", DEFAULT): "0.899",
    }
    readme_text = _text("README.md")
    for key, quoted in expected.items():
        assert f"{cells[key]:.3f}" == quoted, (
            f"{key} recomputes to {cells[key]:.4f}, README quotes {quoted}"
        )
        assert quoted in readme_text, f"README no longer quotes {key} = {quoted}"

    # And the sentence under the table, which restates the marginal cell.
    pct = f"{cells[('marginal', DEFAULT)] * 100:.1f}%"
    assert pct in readme_text, f"README does not say the marginal band covers defaults {pct}"
    # Line breaks fall wherever the reflow put them, so collapse whitespace.
    collapsed = " ".join(readme_text.split())
    assert f"{draws} calibration draws" in collapsed, (
        f"README must state the draw count this test uses ({draws})"
    )


# --- Sampling-bias run metadata ------------------------------------------
#
# The runtime was quoted from a superseded run: the committed arms CSV comes
# from the run that reported 17.6 min, not the earlier 17.7 one.


@pytest.fixture(scope="module")
def sb_summary() -> dict:
    return json.loads((DOCS / "sampling_bias_summary.json").read_text())


def test_the_summary_describes_the_committed_arms_csv(sb_summary, arms) -> None:
    """Guards against quoting one run's metadata beside another run's numbers."""
    assert sb_summary["n_fits"] == len(arms)
    assert sorted({int(r["reject_rate"]) for r in arms}) == sorted(sb_summary["reject_rates"])
    assert sorted({int(r["seed"]) for r in arms}) == sorted(sb_summary["seeds"])
    # n_train varies by arm: A1 trains on the accepted subset and the
    # augmentation arms add rows back. Only the oracle sees the base subsample.
    oracle = {int(r["n_train"]) for r in arms if r["base_arm"] == "A0_oracle"}
    assert oracle == {sb_summary["dev_subsample"]}


def test_runtime_and_windows_are_quoted_from_that_summary(sb_summary, readme) -> None:
    collapsed = " ".join(readme.split())
    minutes = f"{sb_summary['total_seconds'] / 60:.1f} min"
    assert minutes in collapsed, f"README must quote the run's total as {minutes}"
    for name in ("policy", "dev", "test"):
        lo, hi = sb_summary["windows"][name]
        assert f"{lo}-{hi}" in collapsed, f"README must give the {name} window as {lo}-{hi}"


def test_the_feature_cap_claim_matches_the_config_and_the_run(sb_summary, readme) -> None:
    """The config asks for 40; the build yields fewer, so the cap never binds."""
    import tomllib

    configured = tomllib.loads((REPO / "config" / "default.toml").read_text())
    requested = configured["sampling_bias"]["n_features"]
    used = sb_summary["n_features"]
    assert used < requested, (
        f"the config requests {requested} features and the run used {used}; if the "
        "build ever yields more than the cap, the README sentence saying the "
        "selection never binds becomes false"
    )
    collapsed = " ".join(readme.split())
    assert f"top {requested} features" in collapsed
    assert f"produces {used}" in collapsed


# --- Referral economics --------------------------------------------------
#
# Four figures in one sentence were all from a superseded run: loss avoided,
# the review count, and the cost at each of the two review prices. Only the
# conclusion drawn from them was still right.


@pytest.fixture(scope="module")
def economics() -> list[dict[str, str]]:
    with (DOCS / "referral_economics.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_referral_economics_figures_are_quoted(economics, readme) -> None:
    base = economics[0]
    by_cost = {float(r["review_cost"]): r for r in economics}
    collapsed = " ".join(readme.split())
    for label, value in (
        ("loss avoided", f"{float(base['loss_avoided']):,.0f}"),
        ("reviews", f"{int(base['n_referred']):,}"),
        ("cost at 0.01", f"{float(by_cost[0.01]['total_review_cost']):,.0f}"),
        ("cost at 0.02", f"{float(by_cost[0.02]['total_review_cost']):,.0f}"),
    ):
        assert value in collapsed, f"README does not quote the {label} = {value}"


def test_the_break_even_claim_matches_the_sign_change(economics, readme) -> None:
    """The README says the band stops paying at about 0.02. Check it flips there."""
    by_cost = {float(r["review_cost"]): float(r["net_benefit"]) for r in economics}
    assert by_cost[0.01] > 0 > by_cost[0.02], (
        f"net benefit does not change sign between 0.01 and 0.02: {by_cost}"
    )
    assert "about 0.02 per review" in " ".join(readme.split())


def test_the_referral_rate_is_quoted_consistently(economics, summary, readme, model_doc) -> None:
    """One rate, three places: the CSV, the band counts and the prose."""
    from_csv = float(economics[0]["share_referred"])
    bands = summary["band_counts"]
    from_counts = bands["refer"] / sum(bands.values())
    assert from_csv == pytest.approx(from_counts, abs=1e-6)
    collapsed_readme = " ".join(readme.split())
    collapsed_model = " ".join(model_doc.split())
    assert f"{from_csv * 100:.0f} % referral rate" in collapsed_readme
    assert f"{from_csv * 100:.0f} % referral rate" in collapsed_model
    assert f"{from_csv * 100:.1f} %" in collapsed_model


# --- The age-gap attribution ---------------------------------------------
#
# This replaced a narrative that no code produced. Every figure the documents
# quote for it now comes from fairness.age_mechanism, and the previous text
# got three of them wrong (22 for 19.6, 24 for 21.1, 94 % for 93.2 %) while
# resting on a claim the data contradicts.


@pytest.fixture(scope="module")
def mechanism(summary) -> dict:
    assert "age_mechanism" in summary["fairness"], (
        "decision_summary.json has no fairness.age_mechanism; the documents' "
        "attribution would be unbacked again"
    )
    return summary["fairness"]["age_mechanism"]


def test_the_net_gap_and_the_losing_total_are_both_quoted(mechanism, readme) -> None:
    """Two numbers, because quoting one makes the other look like an error."""
    collapsed = " ".join(readme.split())
    net = abs(mechanism["total_points_gap"])
    losing = abs(mechanism["points_lost_to_negative_characteristics"])
    assert f"{net:.1f} scorecard points" in collapsed, f"README must quote the net gap {net:.1f}"
    assert f"{losing:.1f}" in collapsed, f"README must quote the losing total {losing:.1f}"
    assert losing > net, "the offsetting characteristics are the reason both are reported"


def test_the_two_income_characteristics_dominate_and_their_sum_is_quoted(
    mechanism, readme
) -> None:
    by_name = {r["characteristic"]: r for r in mechanism["by_characteristic"]}
    income = [c for c in by_name if c.startswith("afford_")]
    assert len(income) == 2, f"expected two income characteristics, found {income}"
    total = sum(by_name[c]["gap"] for c in income)
    assert f"{abs(total):.1f} of the loss" in " ".join(readme.split()), (
        f"README must attribute {abs(total):.1f} points to the income pair"
    )
    # They must be the two worst, or the sentence naming them is misleading.
    worst_two = {r["characteristic"] for r in mechanism["by_characteristic"][:2]}
    assert worst_two == set(income), f"the two worst are {worst_two}, not the income pair"


@pytest.mark.parametrize(
    "characteristic", ["afford_annuity_to_taxincome", "afford_taxincome_total"]
)
def test_each_income_characteristic_row_is_quoted(mechanism, readme, characteristic) -> None:
    row = next(r for r in mechanism["by_characteristic"] if r["characteristic"] == characteristic)
    cells = f"| {row['points_older']} | {row['points_younger']} | **{row['gap']}** |"
    assert cells in readme, f"README row for {characteristic} does not match: {cells}"


def test_the_proxy_figures_are_quoted(mechanism, readme, model_doc) -> None:
    proxy = mechanism["income_record_as_age_proxy"]
    for doc_name, doc in (("README", readme), ("MODEL", model_doc)):
        collapsed = " ".join(doc.split())
        assert f"{proxy['accuracy'] * 100:.1f}" in collapsed, (
            f"{doc_name} does not quote the proxy accuracy "
            f"{proxy['accuracy'] * 100:.1f} %"
        )
        for key in ("share_missing_among_older", "share_missing_among_younger"):
            assert f"{proxy[key] * 100:.1f} %" in collapsed, (
                f"{doc_name} does not quote {key} = {proxy[key] * 100:.1f} %"
            )


def test_the_absence_is_not_riskier_in_either_group(mechanism, readme) -> None:
    """The load-bearing fact. The previous narrative asserted the opposite.

    If a rerun ever makes the absence genuinely riskier for one group, the
    documents' explanation stops holding and this fails rather than going
    quietly stale.
    """
    rates = {
        (r["group"], r["income_record"]): r["default_rate"]
        for r in mechanism["default_rate_by_age_and_income_record"]
    }
    for group in ("older", "younger"):
        absent = rates[(group, "no_income_record")]
        present = rates[(group, "has_income_record")]
        assert absent <= present, (
            f"among the {group} group a missing income record now defaults at "
            f"{absent:.4f} against {present:.4f}; the documents say the absence "
            "carries no extra risk"
        )
        assert f"{absent * 100:.2f} %" in " ".join(readme.split()), (
            f"README does not quote the {group} absent-record rate {absent * 100:.2f} %"
        )
