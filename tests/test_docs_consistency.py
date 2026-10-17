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
