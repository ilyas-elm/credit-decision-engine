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
import sys
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

    A cutoff binds when the profit-maximising acceptance rate is meaningfully
    below 1.0. The stage's threshold is 0.999, not 1.0, and mirroring it here
    matters now that the grid runs on tie-group boundaries: the largest
    achievable rate short of everyone is 0.9992, which is "approve everyone"
    in every sense except the arithmetic.
    """
    from_grid = sum(1 for row in sensitivity if float(row["acceptance_rate"]) < 0.999)
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
    if not values:
        # recovery_share is NaN by design when the oracle-to-accepts-only gap
        # is too small to divide by, which is the case at low reject rates.
        # The table says so rather than printing a ratio it cannot support.
        assert "n/a" in readme, (
            f"{base_arm} has no recovery at r={reject_rate} %, so the table must say n/a"
        )
        return
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


def test_the_income_absence_table_matches_the_artefact(mechanism, readme) -> None:
    """Every row, since the previous table was wrong in three of four cells."""
    for band in mechanism["income_record_missing_by_age_band"]:
        cells = f"| {band['age_band']}s | {band['n']:,} |"
        assert cells in readme, f"README has no row for the {band['age_band']}s: {cells}"
        share = f"{band['share_missing'] * 100:.1f} %"
        assert share in readme, (
            f"README does not quote the {band['age_band']}s missing share = {share}"
        )


def test_the_absence_steps_rather_than_drifting_up_with_age(mechanism, readme) -> None:
    """The README calls it a step. Check the data still shows one."""
    bands = sorted(
        mechanism["income_record_missing_by_age_band"], key=lambda b: b["age_band"]
    )
    shares = [b["share_missing"] for b in bands]
    biggest_jump = max(b - a for a, b in zip(shares, shares[1:], strict=False))
    assert biggest_jump > 0.5, (
        f"the largest jump between consecutive bands is {biggest_jump:.2f}; the "
        "README describes a step, not a gradient"
    )
    assert "it steps" in " ".join(readme.split())


# --- Birth column coverage -----------------------------------------------


def test_birth_column_coverage_table_matches_the_artefact(summary) -> None:
    coverage = summary["fairness"].get("birth_column_coverage")
    assert coverage, "decision_summary.json has no fairness.birth_column_coverage"
    notes = (REPO / "docs" / "DATA_NOTES.md").read_text()
    for row in coverage:
        cells = f"| `{row['column']}` | {row['overall']:.3f} |"
        assert cells in notes, f"DATA_NOTES row for {row['column']} does not match: {cells}"
        assert f"{row['test_weeks']:.4f}" in notes, (
            f"DATA_NOTES does not give {row['column']} test-week coverage "
            f"{row['test_weeks']:.4f}"
        )


def test_the_chosen_birth_column_is_the_best_covered_on_the_test_weeks(summary) -> None:
    """The point of the table: overall fill rate is not what matters."""
    coverage = summary["fairness"]["birth_column_coverage"]
    best = max(coverage, key=lambda r: r["test_weeks"])
    assert best["column"] == "person_1.birth_259D"
    # And the trap it documents: a column that looks usable overall yet is empty.
    trap = next(r for r in coverage if r["column"] == "static_cb_0.birthdate_574D")
    assert trap["overall"] > 0.3 and trap["test_weeks"] == 0.0


def test_the_birth_agreement_rate_is_quoted(summary) -> None:
    agreement = summary["fairness"].get("birth_column_agreement")
    assert agreement, "decision_summary.json has no fairness.birth_column_agreement"
    notes = " ".join((REPO / "docs" / "DATA_NOTES.md").read_text().split())
    assert f"{agreement['same_birth_year'] * 100:.2f} %" in notes
    assert f"{agreement['n_overlapping']:,}" in notes


# --- Claims the README makes about itself and the repo -------------------


def test_the_quoted_test_count_is_the_real_one(readme) -> None:
    """The README states how many tests there are, in two places.

    Both said 277 while the suite had grown past 350. Counting the collected
    tests here keeps the claim honest without anyone remembering to update it.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(REPO / "tests")],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    last = [ln for ln in result.stdout.strip().splitlines() if ln.strip()][-1]
    collected = int(last.split()[0])
    occurrences = readme.count(f"{collected} tests")
    assert occurrences == 2, (
        f"the suite collects {collected} tests; the README says "
        f"'{collected} tests' {occurrences} times, expected 2 "
        "(the status note and the build table)"
    )
    # MODEL.md quotes it too, in section 8, and was left at 277 while the
    # README was kept current. One check, both documents.
    model_doc = _text("MODEL.md")
    assert f"{collected} tests" in model_doc, (
        f"MODEL.md does not say '{collected} tests'; it quotes a count in "
        "section 8 and that count has to be this one"
    )


def test_the_residual_sd_in_prose_matches_the_table(comparison, readme) -> None:
    """Question 1 quoted 0.047 vs 0.051 above a table saying 0.0521 vs 0.0592."""
    champion = float(comparison["scorecard"]["gini_resid_std"])
    challenger = float(comparison["lightgbm"]["gini_resid_std"])
    assert champion < challenger, "the champion is said to win on smoothness"
    quoted = f"residual sd {champion:.3f} vs {challenger:.3f}"
    assert quoted in " ".join(readme.split()), f"README must say '{quoted}'"


def test_the_recourse_ratio_compares_shortfall_to_headroom(summary, readme) -> None:
    """The sentence claims declines sit further below the cutoff than recourse
    can lift them, so the ratio is their shortfall over their own headroom.
    An earlier fix used the refer band's shortfall instead, giving 3.9x."""
    band = summary["recourse_reach"]["decline_band"]
    ratio = band["mean_shortfall_points"] / band["mean_actionable_headroom_points"]
    assert f"about {ratio:.0f} times further below the cutoff" in " ".join(readme.split()), (
        f"README must give the decline shortfall-to-headroom ratio as {ratio:.0f}x"
    )


def test_the_grid_size_the_readme_explains_is_the_grid_that_ran(sensitivity, readme) -> None:
    """21 is 27 minus the infeasible corner, and the README now says so.

    Four places quote "21 assumption points" against a 3x3x3 grid in config,
    which reads as an error until the skipped combinations are named.
    """
    import tomllib
    from itertools import product

    grid = tomllib.loads((REPO / "config" / "default.toml").read_text())["profit"]["sensitivity"]
    combos = list(product(grid["success_rate"], grid["default_rate"], grid["roi"]))
    feasible = [c for c in combos if c[0] + c[1] <= 1.0]
    assert len(feasible) == len(sensitivity), (
        f"the grid yields {len(feasible)} feasible points but cutoff_sensitivity.csv "
        f"has {len(sensitivity)}"
    )
    collapsed = " ".join(readme.split())
    assert f"{len(combos)} combinations" in collapsed
    assert f"the {len(combos) - len(feasible)} where" in collapsed
    assert f"so {len(feasible)}" in collapsed


def test_the_monitoring_plan_carries_the_signal_the_audit_asked_for(model_doc, readme) -> None:
    """Section 6 concluded the drift remedy belonged in the monitoring plan.

    It was not there, and none of the other signals would catch it: the model
    ranks as well as ever and the bins are unchanged while the Missing cohort
    is marked down. If the row is dropped, the audit goes back to pointing at
    something that does not exist.
    """
    plan = model_doc[model_doc.index("## 7. Monitoring plan") :]
    assert "Missing-bin drift" in plan, (
        "MODEL.md section 6 says the remedy belongs in the monitoring plan; "
        "section 7 has no row for it"
    )
    # Both documents quote the same two figures for it.
    for figure in ("+1.3", "+16.5"):
        assert figure in model_doc, f"MODEL.md does not quote {figure}"
        assert figure in readme, f"README does not quote {figure}"
    # And the README no longer claims neither remedy is implemented.
    assert "Neither is implemented here" not in readme


# --- The alpha sweep -----------------------------------------------------


@pytest.fixture(scope="module")
def sweep() -> list[dict[str, float]]:
    with (DOCS / "conformal_alpha_sweep.csv").open() as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def _at(sweep: list[dict[str, float]], alpha: float) -> dict[str, float]:
    return next(r for r in sweep if abs(r["alpha"] - alpha) < 1e-9)


@pytest.mark.parametrize("alpha", [0.10, 0.22, 0.28, 0.50])
def test_each_quoted_alpha_row_matches_the_artefact(sweep, readme, alpha) -> None:
    row = _at(sweep, alpha)
    collapsed = " ".join(readme.split())
    assert f"{row['refer_rate'] * 100:.1f} %" in collapsed, (
        f"README does not quote the referral rate at alpha={alpha}"
    )
    assert f"{row['coverage_default']:.3f}" in collapsed, (
        f"README does not quote the default coverage at alpha={alpha}"
    )


def test_the_configured_alpha_agrees_with_the_decision_summary(sweep, summary) -> None:
    """The sweep's own row for the configured alpha must match the run."""
    row = _at(sweep, summary["alpha"])
    assert row["coverage_default"] == pytest.approx(summary["coverage_overall"]["1"], abs=5e-4)
    bands = summary["band_counts"]
    assert row["refer_rate"] == pytest.approx(bands["refer"] / sum(bands.values()), abs=5e-4)


def test_the_curve_really_is_non_monotone(sweep, readme) -> None:
    """The README calls this a trap, so the data has to still show it."""
    rates = [r["refer_rate"] for r in sweep]
    assert min(rates) == 0.0, "the README says the band refers nobody somewhere"
    assert rates[-1] > min(rates), "the referral rate must rise again after its minimum"
    # and the two regimes are the two causes, never mixed
    for row in sweep:
        assert not (row["both_labels"] > 0 and row["empty_set"] > 0), (
            f"alpha={row['alpha']} mixes both-label and empty referrals; the "
            "README describes them as separate regimes"
        )


def test_the_staffable_desk_claim_holds(sweep, readme) -> None:
    """A 10 % desk needs alpha >= 0.28 and gives up most of the guarantee."""
    affordable = [r for r in sweep if r["refer_rate"] <= 0.10]
    assert affordable, "no alpha reaches a 10 % referral rate"
    first = affordable[0]
    collapsed = " ".join(readme.split())
    assert f"needs α ≥ {first['alpha']:.2f}" in collapsed, (
        f"README must say a 10 % desk needs alpha >= {first['alpha']:.2f}"
    )
    assert first["coverage_default"] < 0.5, (
        "the README's point is that affordability costs the guarantee"
    )


def test_coverage_never_reaches_its_target_anywhere_on_the_curve(sweep) -> None:
    """The sharpest version of the section's claim, over every alpha tried."""
    missed = [r for r in sweep if r["coverage_default"] >= r["target_coverage"]]
    assert not missed, (
        f"the default class meets its target at alpha="
        f"{[r['alpha'] for r in missed]}, which the README says never happens"
    )


# --- What the challenger's ranking is worth ------------------------------


@pytest.fixture(scope="module")
def ranking() -> list[dict[str, float]]:
    with (DOCS / "ranking_value.csv").open() as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def test_the_comparison_holds_volume_equal(ranking) -> None:
    """The premise of the whole section: same book size, so it is ranking."""
    for row in ranking:
        assert row["defaults_avoided"] == pytest.approx(
            row["defaults_champion"] - row["defaults_challenger"]
        )
        assert row["bad_rate_champion"] == pytest.approx(
            row["defaults_champion"] / row["n_approved"], abs=1e-9
        )
        assert row["bad_rate_challenger"] == pytest.approx(
            row["defaults_challenger"] / row["n_approved"], abs=1e-9
        )


def test_the_tightest_binding_point_is_quoted(ranking, summary, readme) -> None:
    row = min(ranking, key=lambda r: r["acceptance_rate"])
    assert row["acceptance_rate"] == pytest.approx(
        summary["operating_point"]["acceptance_rate"], abs=1e-6
    ), "the quoted row must be the operating point the decision layer uses"
    collapsed = " ".join(readme.split())
    for value in (
        f"{int(row['defaults_champion']):,}",
        f"{int(row['defaults_challenger']):,}",
        f"{row['bad_rate_champion']:.3%}".replace("%", " %"),
        f"{row['bad_rate_challenger']:.3%}".replace("%", " %"),
        f"{int(row['n_approved']):,}",
    ):
        assert value in collapsed, f"README does not quote {value}"


def test_the_challenger_wins_at_every_binding_point(ranking, readme) -> None:
    """The README says every. If a rerun breaks it, the claim must break too."""
    losing = [r["acceptance_rate"] for r in ranking if r["defaults_avoided"] <= 0]
    assert not losing, f"the challenger approves no fewer defaults at acceptance {losing}"
    assert "every** binding point" in readme or "at **every** binding point" in readme


def test_the_advantage_shrinks_as_the_cutoff_loosens(ranking, readme) -> None:
    """The degeneracy, from the decision side. The README leans on this."""
    ordered = sorted(ranking, key=lambda r: r["acceptance_rate"])
    tightest, loosest = ordered[0], ordered[-1]
    assert tightest["defaults_avoided"] > loosest["defaults_avoided"], (
        "the README says the advantage shrinks as acceptance rises"
    )
    lift = [
        (r["profit_challenger"] / r["profit_champion"] - 1) * 100
        for r in (tightest, loosest)
        if r["profit_champion"]
    ]
    collapsed = " ".join(readme.split())
    assert f"{lift[0]:.1f} %" in collapsed or f"{lift[0]:.1f}%" in collapsed
    assert f"{lift[-1]:.1f} %" in collapsed or f"{lift[-1]:.1f}%" in collapsed


# --- What A2 actually buys -----------------------------------------------
#
# The write-up used to explain A2's win by its label being roughly right. It
# is wrong about most rejects, and what A2 really trades is calibration for
# ranking. These pin the corrected explanation, because it is the project's
# own question-1 standard applied to its own question-3 headline.


def _arm_mean(arms, arm, reject_rate, column, model="lightgbm"):
    values = [
        float(r[column])
        for r in arms
        if r["model"] == model and r["base_arm"] == arm and r["reject_rate"] == reject_rate
    ]
    assert values, f"no {model} rows for {arm} at r={reject_rate}"
    return sum(values) / len(values)


def test_a2_recovers_ranking_while_wrecking_calibration(arms, readme) -> None:
    oracle_brier = _arm_mean(arms, "A0_oracle", "30", "brier_full")
    a2_brier = _arm_mean(arms, "A2_hard_cutoff", "30", "brier_full")
    a2_gini = _arm_mean(arms, "A2_hard_cutoff", "30", "gini_full")
    oracle_gini = _arm_mean(arms, "A0_oracle", "30", "gini_full")

    assert a2_gini > 0.95 * oracle_gini, "A2 is supposed to nearly match the oracle's ranking"
    ratio = a2_brier / oracle_brier
    collapsed = " ".join(readme.split())
    assert f"{ratio:.1f}×" in collapsed or f"{ratio:.1f}x" in collapsed, (
        f"README must say A2's Brier is {ratio:.1f}x the oracle's"
    )
    assert f"{a2_brier:.3f}" in collapsed and f"{oracle_brier:.4f}" in collapsed


def test_a2_over_predicts_by_the_quoted_margin(arms, readme) -> None:
    """Both halves: the margin is what the artefact says, and the README says it."""
    collapsed = " ".join(readme.split())
    for reject_rate, phrase in (("30", "15 percentage points"), ("50", "29 points")):
        citl = _arm_mean(arms, "A2_hard_cutoff", reject_rate, "citl_gap_full")
        assert citl > 0, f"A2 should over-predict at r={reject_rate}, got {citl}"
        assert round(citl * 100) == int(phrase.split()[0]), (
            f"A2 over-predicts by {citl * 100:.0f} points at r={reject_rate}, "
            f"README says {phrase}"
        )
        assert phrase in collapsed, f"README does not say {phrase!r}"


def test_a3_is_the_calibrated_alternative_the_readme_recommends(arms, readme) -> None:
    """The README points a lender at A3 instead. It has to still be true."""
    oracle = _arm_mean(arms, "A0_oracle", "30", "brier_full")
    a3 = _arm_mean(arms, "A3_fuzzy", "30", "brier_full")
    a2 = _arm_mean(arms, "A2_hard_cutoff", "30", "brier_full")
    assert a3 < a2, "A3 must be better calibrated than A2 for the recommendation to hold"
    assert abs(a3 / oracle - 1) < 0.005, (
        f"README says A3 is within 0.5 % of the oracle's Brier; it is {a3 / oracle - 1:+.2%}"
    )
    assert f"{a3:.5f}" in " ".join(readme.split())


def test_accept_only_calibration_looks_better_than_it_is(arms, readme) -> None:
    """The mirror of the Gini bias, and the reason it is worth reporting.

    Gini on accepts is pessimistic, so a lender errs towards caution.
    Calibration on accepts is flattering, so it does not. If a rerun ever
    reverses that, the README's warning stops being true.
    """
    collapsed = " ".join(readme.split())
    for reject_rate in ("10", "20", "30", "50"):
        rows = [
            r for r in arms if r["model"] == "scorecard" and r["reject_rate"] == reject_rate
        ]
        assert rows
        full = sum(float(r["citl_gap_full"]) for r in rows) / len(rows)
        accepted = sum(float(r["citl_gap_accepted_only"]) for r in rows) / len(rows)
        assert abs(accepted) < abs(full), (
            f"at r={reject_rate} the accepts-only calibration gap ({accepted:+.4f}) is "
            f"not smaller than the true one ({full:+.4f}); the README says the "
            "lender's own book flatters it"
        )
        assert f"{full:+.4f}" in collapsed, f"README omits the true gap at r={reject_rate}"
        assert f"{accepted:+.4f}" in collapsed, f"README omits the accepts gap at r={reject_rate}"


def test_the_two_evaluation_biases_point_opposite_ways(arms, readme) -> None:
    """Gini understated, calibration overstated. The README leans on the contrast."""
    rows = [r for r in arms if r["model"] == "scorecard" and r["reject_rate"] == "50"]
    gini_bias = sum(float(r["evaluation_bias"]) for r in rows) / len(rows)
    full = sum(float(r["citl_gap_full"]) for r in rows) / len(rows)
    accepted = sum(float(r["citl_gap_accepted_only"]) for r in rows) / len(rows)
    assert gini_bias < 0, "accepts-only must understate Gini"
    assert full > 0 > accepted, (
        "the model must over-predict on everyone while looking unbiased on accepts"
    )
    assert "opposite direction" in " ".join(readme.split())


def test_the_monitoring_caveat_matches_the_threshold_it_qualifies(arms, model_doc) -> None:
    """The plan's recalibration trigger is measured where it is flattered.

    MODEL section 7 says the true gap fires its own threshold while the gap a
    lender can observe stays quiet. Both halves have to keep holding, since
    the caveat is the reason the trigger is not taken at face value.
    """
    rows = [r for r in arms if r["model"] == "scorecard" and r["reject_rate"] == "50"]
    assert rows
    full = sum(float(r["citl_gap_full"]) for r in rows) / len(rows)
    accepted = sum(float(r["citl_gap_accepted_only"]) for r in rows) / len(rows)

    # The threshold the plan sets for investigation.
    plan = model_doc[model_doc.index("## 7. Monitoring plan") :]
    assert "> 0.010" in plan, "the investigate threshold moved; the caveat quotes it"
    assert abs(full) > 0.010, "the true gap must breach the threshold for the point to stand"
    assert abs(accepted) < 0.010, "the observable gap must not breach it"
    assert full > 0 > accepted, "the two must differ in sign, which the caveat states"
    assert f"{full:+.4f}" in plan and f"{accepted:+.4f}" in plan
    ratio = abs(full / accepted)
    assert f"{ratio:.0f} times" in " ".join(plan.split()), (
        f"the caveat must say {ratio:.0f} times; the gap is {full:+.4f} against {accepted:+.4f}"
    )


@pytest.fixture(scope="module")
def reject_truth() -> list[dict[str, float]]:
    with (DOCS / "reject_truth.csv").open() as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def test_a2s_label_is_wrong_about_most_rejects(reject_truth, readme) -> None:
    """The measurement that replaced "close enough to true".

    If a rerun ever made the rejects mostly bad, A2's label would genuinely be
    roughly right and the explanation in the README would need reverting, not
    just rewording.
    """
    at_30 = next(r for r in reject_truth if r["reject_rate"] == 30)
    assert at_30["bad_rate_rejected"] < 0.5, (
        "the README's point is that A2 is wrong about most rejects; they are "
        f"now {at_30['bad_rate_rejected']:.1%} bad"
    )
    collapsed = " ".join(readme.split())
    assert f"{at_30['bad_rate_rejected']:.1%}".replace("%", " %") in collapsed
    assert f"{at_30['bad_rate_accepted']:.1%}".replace("%", " %") in collapsed
    assert f"{at_30['share_of_rejects_actually_good']:.0%}".replace("%", " %") in collapsed


def test_the_rejected_region_really_is_riskier(reject_truth, readme) -> None:
    """What A2 does get right, and the reason it works at all."""
    for row in reject_truth:
        assert row["lift"] > 1.0, (
            f"at r={row['reject_rate']} the rejects are not riskier, so the "
            "direction A2 asserts would be wrong too"
        )
    lifts = [r["lift"] for r in reject_truth]
    assert min(lifts) >= 3.5 and max(lifts) <= 4.5, (
        f"the README says about four times riskier; lifts are {lifts}"
    )
    assert "four times riskier" in " ".join(readme.split())


@pytest.mark.parametrize(
    "arm",
    [
        "A0_oracle",
        "A1_accepts_only",
        "A2_hard_cutoff",
        "A3_fuzzy",
        "A4_parcelling",
        "A5_self_learning",
        "A6_exploration",
    ],
)
def test_every_cell_of_the_calibration_table_matches_the_run(arms, readme, arm) -> None:
    collapsed = " ".join(readme.split())
    for column, spec in (("gini_full", ".4f"), ("brier_full", ".5f"), ("citl_gap_full", "+.4f")):
        value = format(_arm_mean(arms, arm, "30", column), spec)
        assert value in collapsed, f"README omits {arm} {column} = {value}"


def test_sampling_bias_costs_ranking_rather_than_calibration(arms, readme) -> None:
    """The claim the whole table is built around.

    If hiding the rejects' labels ever started costing calibration too, the
    README's framing, that there is almost nothing to recover there, stops
    being true and the arms would have to be read differently.
    """
    gini_cost = _arm_mean(arms, "A0_oracle", "30", "gini_full") - _arm_mean(
        arms, "A1_accepts_only", "30", "gini_full"
    )
    oracle_brier = _arm_mean(arms, "A0_oracle", "30", "brier_full")
    brier_cost = _arm_mean(arms, "A1_accepts_only", "30", "brier_full") - oracle_brier
    relative_gini = gini_cost / _arm_mean(arms, "A0_oracle", "30", "gini_full")
    relative_brier = brier_cost / oracle_brier
    # The README rounds these to "a tenth" and "a hundredth" of the oracle.
    # Asserted as bands rather than a ratio, because a ratio of the two is
    # brittle when the denominator is this small: it is 9.4 here, and a
    # threshold of 10 would fail on prose that is accurate.
    assert 0.05 <= relative_gini <= 0.15, (
        f"ranking costs {relative_gini:.1%}, which the README rounds to a tenth"
    )
    assert 0.005 <= relative_brier <= 0.015, (
        f"calibration costs {relative_brier:.1%}, which the README rounds to a hundredth"
    )
    assert "damages ranking, not calibration" in " ".join(readme.split())


def test_asserting_arms_wreck_calibration_and_weighting_arms_do_not(arms, readme) -> None:
    """The structural split the README draws, over the arms it names."""
    oracle = _arm_mean(arms, "A0_oracle", "30", "brier_full")
    for arm in ("A2_hard_cutoff", "A5_self_learning"):
        ratio = _arm_mean(arms, arm, "30", "brier_full") / oracle
        assert ratio > 2.0, f"{arm} asserts labels and should wreck Brier; it is {ratio:.1f}x"
    for arm in ("A3_fuzzy", "A4_parcelling"):
        ratio = _arm_mean(arms, arm, "30", "brier_full") / oracle
        # The README says within 1.2 %. The bound is on the weighting arms as
        # a class, and the gap to the asserting ones is two orders of
        # magnitude, so the exact figure matters less than the separation.
        assert abs(ratio - 1.0) < 0.015, (
            f"{arm} carries uncertainty as weight and should stay close to the "
            f"oracle's Brier; it is {ratio:.3f}x"
        )
    assert "within 1.2 % of the oracle" in " ".join(readme.split())
    # And A3 must be the best-ranking arm among those that keep calibration.
    keepers = {
        arm: _arm_mean(arms, arm, "30", "gini_full")
        for arm in ("A3_fuzzy", "A4_parcelling", "A6_exploration")
    }
    assert max(keepers, key=keepers.get) == "A3_fuzzy", (
        f"the README names A3 as the deployable arm that recovers most; got {keepers}"
    )


def _true_citl_gap_at_fifty() -> float:
    """The population calibration gap the sizing is meant to detect.

    Read from the run rather than written in, because it moves: it was 0.0176
    and is now 0.0183, and a hardcoded copy would silently size for the wrong
    target while still passing.
    """
    with (DOCS / "sampling_bias_arms.csv").open() as fh:
        gaps = [
            float(row["citl_gap_full"])
            for row in csv.DictReader(fh)
            if row["model"] == "scorecard" and row["reject_rate"] == "50"
        ]
    return sum(gaps) / len(gaps)


def test_the_exploration_sizing_in_the_documents_is_what_the_code_computes(
    reject_truth, readme, model_doc
) -> None:
    """Both documents quote this sizing, so neither may drift from the function.

    It is the one number in question 3 that is analytic rather than measured
    by an arm, which makes it the easiest to leave behind when the inputs
    change.
    """
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from config import load_config
    from profit import ProfitAssumptions
    from sampling_bias import exploration_budget_for_detection

    assumptions = ProfitAssumptions.from_config(load_config()["profit"])
    collapsed_readme = " ".join(readme.split())
    collapsed_model = " ".join(model_doc.split())
    for reject_rate in (30, 50):
        row = next(r for r in reject_truth if r["reject_rate"] == reject_rate)
        sized = exploration_budget_for_detection(
            row["bad_rate_rejected"],
            reject_rate / 100.0,
            _true_citl_gap_at_fifty(),
            0.010,
            assumptions,
            n_rejected=int(row["n_rejected"]),
        )
        loans = f"{sized['n_explored']:.0f}"
        assert loans in collapsed_readme, f"README omits {loans} explored loans"
        assert loans in collapsed_model, f"MODEL omits {loans} explored loans"
        assert f"{sized['share_of_rejects']:.2%}".replace("%", " %") in collapsed_readme
        assert f"{sized['expected_cost']:.1f}" in collapsed_readme


def test_the_smallest_arm_budget_is_the_right_order_for_measurement(
    reject_truth, readme
) -> None:
    """The point of the section: A6's p=1 % is about the size needed.

    If a rerun moved the reject bad rates enough that p=1 % stopped being
    close, the claim that the same budget does both jobs stops holding.
    """
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from sampling_bias import exploration_budget_for_detection

    row = next(r for r in reject_truth if r["reject_rate"] == 30)
    needed = exploration_budget_for_detection(
        row["bad_rate_rejected"], 0.30, _true_citl_gap_at_fifty(), 0.010
    )["n_explored"]
    one_percent = 0.01 * row["n_rejected"]
    assert 0.5 <= one_percent / needed <= 2.0, (
        f"p=1 % buys {one_percent:.0f} loans against {needed:.0f} needed; the README "
        "says it is roughly the right size"
    )
    assert "roughly the right size" in " ".join(readme.split())


def test_the_scorecard_anomaly_is_explained_with_its_control(arms, readme) -> None:
    """The accepts-only scorecard beating its oracle needs the control to mean
    anything. Without it the table is three numbers and no argument.

    And the control needs more than one draw. It had exactly one, once, and
    the conclusion published off it did not survive five more seeds: against
    its control the policy removal is worth +1.2 sd of test Gini at a 10 %
    reject rate, +0.1 sd at 20 % and -0.7 sd at 30 %, so it never clears the
    control's own noise. What survived was the in-sample half, where 18 of 18
    control draws land on the far side, so that is what this guards.
    """
    collapsed = " ".join(readme.split())
    a0 = _arm_mean(arms, "A0_oracle", "10", "gini_full", model="scorecard")
    a1 = _arm_mean(arms, "A1_accepts_only", "10", "gini_full", model="scorecard")
    assert a1 > a0, (
        f"the anomaly has gone: accepts-only {a1:.4f} no longer beats the oracle "
        f"{a0:.4f}, so the explanation in the README describes nothing"
    )
    # The control, reported as the several draws it takes to say anything.
    assert "random* 10 % removed, 6 seeds" in collapsed, (
        "a one-draw control puts the whole effect inside its own seed spread; "
        "the README must report the control as multiple seeds"
    )
    # The mechanism that did separate: the in-sample collapse, which every
    # random draw fails to reproduce and the policy removal always produces.
    assert "3.18" in collapsed and "1.43" in collapsed, (
        "the README explains this by where the in-sample IV lives; both totals "
        "must be quoted"
    )
    assert "+0.007" in collapsed and "+0.07" in collapsed, (
        "the generalisation gap carries the argument now that the test column "
        "cannot; both sides of it must be quoted"
    )
    # And the limit. The test column is not separable from the control, and
    # the one rate where it looks closest is also the only favourable one.
    assert "-0.7 sd" in collapsed and "never clears noise" in collapsed, (
        "the README must not re-acquire the claim that the test-Gini gain is "
        "attributable to which rows were removed; the sd against the control "
        "at each rate is what stops it"
    )


def test_the_anomaly_reverses_at_higher_reject_rates(arms, readme) -> None:
    """The README says the favour runs out by 30 %. It has to still run out."""
    pairs = {
        rate: (
            _arm_mean(arms, "A1_accepts_only", rate, "gini_full", model="scorecard")
            - _arm_mean(arms, "A0_oracle", rate, "gini_full", model="scorecard")
        )
        for rate in ("10", "20", "30", "50")
    }
    assert pairs["10"] > 0 and pairs["20"] > 0, "the favour should hold at low rates"
    assert pairs["30"] < 0 and pairs["50"] < 0, "and should have reversed by 30 %"
    # Not "the trimming ran out of overfit to remove" -- it ran out at 10 %,
    # where the gap hits zero, and the rates past that cut into signal. The
    # recon at 10/20/30 is what distinguishes those two readings.
    collapsed = " ".join(readme.split())
    assert "it ran out at 10 %, and everything after that is cutting into signal" in collapsed
    assert "1.43 \u2192 1.05 \u2192 0.84" in collapsed, (
        "the IV decay across the three rates is the evidence that the later "
        "trimming removes signal rather than overfit"
    )


#: What one row-order change does to the challenger, measured between two
#: full-data runs whose scorecard output was bit-identical: metric -> (size of
#: the perturbation, ratio the README quotes against the champion margin).
ROW_ORDER_PERTURBATION = {
    "gini": (0.00446056, 6.2),
    "brier": (0.00001880, 6.8),
    "citl_max_abs_gap": (0.00019759, 6.6),
}


def _quotes_ratios(text: str, ratios: list[float]) -> bool:
    """The ordered triple, in either multiplication glyph.

    Checked as one phrase rather than number by number: MODEL.md says "Brier
    6.6x the oracle's" about an unrelated arm, and a bare "6.6x" search passed
    on that sentence while the margin ratio it was meant to find was absent.
    """
    collapsed = " ".join(text.split())
    a, b, c = ratios
    return any(
        f"{a}{g}, {b}{g} and {c}{g}" in collapsed for g in ("\u00d7", "x")
    )


def test_the_thin_margins_are_quoted_against_the_noise_that_moves_them(
    comparison, readme, model_doc
) -> None:
    """Brier is awarded to the challenger on 0.00013 and worst-week calibration
    on 0.0013. Margins that small mean nothing without a scale, and the
    challenger has one: it bags by row position, so the frame's order moves its
    fit even on identical features.

    If a later run changes any of these margins, the published ratios stop
    being true and have to be re-derived rather than left standing.
    """
    collapsed = " ".join(readme.split())
    for metric, (perturbation, published) in ROW_ORDER_PERTURBATION.items():
        margin = abs(
            float(comparison["lightgbm"][metric]) - float(comparison["scorecard"][metric])
        )
        ratio = margin / perturbation
        assert round(ratio, 1) == published, (
            f"{metric}: margin {margin:.6f} is now {ratio:.1f}x the row-order "
            f"perturbation {perturbation}, not the {published}x the README quotes"
        )
    # Both published documents carry the table, so both carry the scale.
    # MODEL.md kept a superseded version of the exploration claim for one
    # commit because only the README was checked for it.
    ratios = [ROW_ORDER_PERTURBATION[m][1] for m in ("gini", "brier", "citl_max_abs_gap")]
    for name, text in (("README.md", readme), ("MODEL.md", model_doc)):
        assert _quotes_ratios(text, ratios), (
            f"{name} must quote the margins as {ratios[0]}x, {ratios[1]}x and "
            f"{ratios[2]}x of one row-order perturbation"
        )
    # And the caveat that keeps one perturbation from reading as a spread.
    assert "That is one perturbation and not a spread" in collapsed


def test_the_pooled_recovery_rows_disclose_what_they_pool(arms, readme, model_doc) -> None:
    """A4 and A6 are means over variants and seeds; the other rows are single
    fits. Pooling is only safe where the pooled fits agree, and these two do
    not agree equally: A6 is unanimous, A4 changes sign.

    The recovery cell for A4 at a 20 % reject rate reads +0.13 off six fits
    spanning -0.33 to +0.62. Quoting it without the spread is the same mistake
    as the one-draw control, so the README must carry the caveat.
    """
    collapsed = " ".join(readme.split())
    rows = [r for r in arms if r["model"] == "lightgbm" and r["recovery_share"]]

    def shares(prefix: str, rate: str) -> list[float]:
        return [
            float(r["recovery_share"])
            for r in rows
            if r["arm"].startswith(prefix) and r["reject_rate"] == rate
        ]

    # A6 is unanimous, which is what lets its row stand unqualified.
    a6 = [v for rate in ("20", "30", "50") for v in shares("A6_exploration", rate)]
    assert len(a6) == 27 and all(v < 0 for v in a6), (
        f"the README says all 27 A6 fits are negative; got {len(a6)} fits, "
        f"{sum(v >= 0 for v in a6)} of them non-negative"
    )
    for name, text in (("README.md", readme), ("MODEL.md", model_doc)):
        assert "negative in all 27 of its fits" in " ".join(text.split()), (
            f"{name} must report A6's unanimity rather than only its mean"
        )

    # A4 is not, and the README has to say so rather than quote the mean.
    for rate, positives in (("20", 4), ("30", 2)):
        a4 = shares("A4_parcelling", rate)
        assert len(a4) == 6, f"expected 2 uplifts x 3 seeds at r={rate} %, got {len(a4)}"
        assert sum(v > 0 for v in a4) == positives, (
            f"A4 at r={rate} % now has {sum(v > 0 for v in a4)} positive fits of 6, "
            f"not {positives}; the README quotes the count"
        )
    assert "A4's sign is not determined below r=50 %" in collapsed

    # The deployable-arm conclusion, only at the rates where A3 clears A4
    # outright rather than on the mean.
    a3 = {r["reject_rate"]: float(r["recovery_share"]) for r in rows if r["arm"] == "A3_fuzzy"}
    for rate in ("30", "50"):
        assert a3[rate] > max(shares("A4_parcelling", rate)), (
            f"A3 no longer clears A4's whole spread at r={rate} %, which is the "
            "claim the README makes for 30 % and 50 %"
        )
    assert a3["20"] < max(shares("A4_parcelling", "20")), (
        "A3 now clears A4's spread at 20 % too; the README calls that one "
        "unresolved and should be updated"
    )
    assert (
        "At 20 % it does not, so the ranking of the two weighting arms there is "
        "unresolved" in collapsed
    )


@pytest.fixture(scope="module")
def exploration() -> list[dict[str, str]]:
    with (DOCS / "sampling_bias_exploration.csv").open() as fh:
        return list(csv.DictReader(fh))


def test_the_exploration_verdict_is_scoped_to_the_model_that_survives_its_seeds(
    exploration, readme, model_doc
) -> None:
    """The A6 claims were published unscoped and two of them were wrong.

    "net is negative at every budget and reject rate" is true of the 3-seed
    means and false of one scorecard cell; "the damage falls monotonically,
    at every reject rate" holds for the GBM and fails for the scorecard at
    three of its four rates. Both are the single-draw mistake in another
    costume: a verdict quoted tighter than the spread underneath it.
    """
    collapsed = " ".join(readme.split())
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in exploration:
        key = (row["reject_rate"], row["explore_pct"])
        bucket = cells.setdefault(key, {"gbm": [], "sc": [], "cost": []})
        bucket["gbm"].append(float(row["net_vs_A1_lightgbm"]))
        bucket["sc"].append(float(row["net_vs_A1_scorecard"]))
        bucket["cost"].append(float(row["realised_loss_cost"]))
    assert len(cells) == 12, f"expected 4 rates x 3 budgets, got {len(cells)}"

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    # Negative on the mean everywhere, for both models. That part was right.
    for key, bucket in cells.items():
        assert mean(bucket["gbm"]) < 0 and mean(bucket["sc"]) < 0, (
            f"exploration turned a mean profit at {key}, so the README's "
            "'negative on the 3-seed mean of all twelve' no longer holds"
        )
    # The GBM holds per seed; the scorecard does not, and the README says so.
    assert all(v < 0 for b in cells.values() for v in b["gbm"]), (
        "the README claims every one of the GBM's 36 seed-level cells is "
        "negative"
    )
    positives = [(k, v) for k, b in cells.items() for v in b["sc"] if v > 0]
    assert len(positives) == 1 and positives[0][0] == ("50", "5"), (
        f"the README discloses exactly one positive scorecard cell at "
        f"r=50 %, p=5 %; found {positives}"
    )
    assert "one of its 36 cells comes out *positive*" in collapsed

    # Monotone damage in the budget: the GBM's, at every rate.
    def ratios(model: str, rate: str) -> list[float]:
        return [
            -mean(cells[(rate, p)][model]) / mean(cells[(rate, p)]["cost"])
            for p in ("1", "2", "5")
        ]

    for rate in ("10", "20", "30", "50"):
        r = ratios("gbm", rate)
        assert r[0] > r[1] > r[2], (
            f"the GBM's damage ratio is no longer monotone in the budget at "
            f"r={rate} %: {[round(x, 1) for x in r]}"
        )
    assert "the GBM's damage falls monotonically" in collapsed, (
        "the monotonicity claim must stay scoped to the GBM; the scorecard's "
        "ratio is not monotone"
    )
    # And the range quoted for it.
    everything = [x for rate in ("10", "20", "30", "50") for x in ratios("gbm", rate)]
    assert 3 <= min(everything) < 4 and 48 <= max(everything) < 50, (
        f"the README quotes 3-49x; the grid now spans "
        f"{min(everything):.1f}-{max(everything):.1f}x"
    )
    for name, text in (("README.md", readme), ("MODEL.md", model_doc)):
        flat = " ".join(text.split())
        assert "3-49× the cost" in flat or "3-49x the cost" in flat, (
            f"{name} must quote the GBM's damage range; the 2-48x it used to "
            "quote was neither model's"
        )


def test_the_gbm_never_gets_the_favour(arms, readme) -> None:
    """The contrast the explanation rests on: no binning, no regularisation."""
    for rate in ("10", "20", "30", "50"):
        gap = _arm_mean(arms, "A1_accepts_only", rate, "gini_full") - _arm_mean(
            arms, "A0_oracle", rate, "gini_full"
        )
        assert gap < 0, f"the GBM's accepts-only model beats its oracle at r={rate}"
    assert "degrades monotonically" in " ".join(readme.split())
