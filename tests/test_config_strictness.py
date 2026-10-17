"""A config key that is misspelt must stop the run, not be ignored.

Keeping thresholds in TOML only helps if the file is read. A key that is
renamed or mistyped and then quietly dropped leaves the run on a default while
the diff claims otherwise, and the numbers still come out looking fine. That
is the failure the brief's rule about parameter changes going through config
is meant to prevent, so it has to fail loudly.

`ChallengerParams` enforced this from the start. The other five sections did
not, which these tests close and keep closed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from challenger import ChallengerParams
from config import checked_kwargs, load_config
from features import FeatureParams, ScreeningParams
from profit import ProfitAssumptions
from sampling_bias import SamplingBiasParams
from scorecard import ScorecardParams

REPO = Path(__file__).resolve().parent.parent

#: Every config section, the class that reads it, and the keys that class
#: requires to be present (only the scorecard reads keys positionally).
SECTIONS = [
    ("features", FeatureParams),
    ("screening", ScreeningParams),
    ("scorecard", ScorecardParams),
    ("challenger", ChallengerParams),
    ("profit", ProfitAssumptions),
    ("sampling_bias", SamplingBiasParams),
]


@pytest.mark.parametrize("name, cls", SECTIONS, ids=[s for s, _ in SECTIONS])
def test_the_committed_section_loads(name: str, cls: type) -> None:
    cls.from_config(load_config()[name])


@pytest.mark.parametrize("name, cls", SECTIONS, ids=[s for s, _ in SECTIONS])
def test_an_unrecognised_key_is_rejected(name: str, cls: type) -> None:
    section = dict(load_config()[name])
    section["a_key_that_does_not_exist"] = 1
    with pytest.raises(ValueError, match=f"\\[{name}\\] has unrecognised keys"):
        cls.from_config(section)


@pytest.mark.parametrize("name, cls", SECTIONS, ids=[s for s, _ in SECTIONS])
def test_the_error_names_the_offending_key_and_the_known_ones(name: str, cls: type) -> None:
    """A bare rejection is not much use at 2am in a kernel log."""
    section = dict(load_config()[name])
    section["psi_bims"] = 10  # a plausible fat-finger
    with pytest.raises(ValueError) as caught:
        cls.from_config(section)
    message = str(caught.value)
    assert "psi_bims" in message
    assert "Known:" in message


def test_a_nested_table_is_not_mistaken_for_a_key() -> None:
    """[profit] carries its sensitivity grid as a sub-table."""
    raw = tomllib.loads((REPO / "config" / "default.toml").read_text())
    assert isinstance(raw["profit"].get("sensitivity"), dict), (
        "this test is pointless if [profit.sensitivity] stops being a sub-table"
    )
    ProfitAssumptions.from_config(raw["profit"])


def test_keys_another_class_reads_are_allowed_through() -> None:
    """[sampling_bias] holds the window fractions and the arm parameters.

    They are read by two different classes, so neither may reject the other's
    keys, and neither may silently swallow a genuine typo either.
    """
    raw = tomllib.loads((REPO / "config" / "default.toml").read_text())["sampling_bias"]
    assert {"policy_frac", "dev_frac", "test_frac"} <= set(raw)
    SamplingBiasParams.from_config(raw)
    with pytest.raises(ValueError, match="unrecognised keys"):
        SamplingBiasParams.from_config({**raw, "polcy_frac": 0.25})


def test_the_helper_leaves_a_clean_section_untouched() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Example:
        alpha: int = 1
        beta: str = "b"

    assert checked_kwargs({"alpha": 2}, Example, "example") == {"alpha": 2}
