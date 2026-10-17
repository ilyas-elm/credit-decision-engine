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
