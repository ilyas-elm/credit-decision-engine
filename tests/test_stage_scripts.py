"""Static checks on the kernel stage scripts.

These scripts only ever run on Kaggle, against data that cannot be committed,
so a mistake in one is not caught by any other test here: it is caught by a
queued kernel failing some minutes later. Parsing them locally costs nothing
and rules out the syntax errors that would waste that wait.

A stronger check was tried and removed. The decision stage once broke because
a patch introduced `overall = float(...)` inside a loop, 160 lines above an
`overall["label"]` that read a coverage report assigned far earlier. Several
static rules were written to catch it and every one also flagged correct code,
because the bug has the same shape as an ordinary accumulator: assigned at
function level, reassigned inside a loop, read afterwards. Telling them apart
needs intent, which the syntax tree does not carry. The guard that works is
reading the surrounding function before adding a name to it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

KAGGLE = Path(__file__).resolve().parent.parent / "kaggle"
STAGES = sorted(KAGGLE.glob("*/stage.py"))


@pytest.mark.parametrize("path", STAGES, ids=[p.parent.name for p in STAGES])
def test_stage_script_parses(path: Path) -> None:
    ast.parse(path.read_text(), filename=str(path))


def test_every_stage_directory_has_a_script_and_metadata() -> None:
    """A stage directory missing either cannot be pushed."""
    for directory in sorted(
        d for d in KAGGLE.iterdir() if d.is_dir() and not d.name.startswith("_")
    ):
        assert (directory / "stage.py").is_file(), f"{directory.name} has no stage.py"
        assert (directory / "kernel-metadata.json").is_file(), (
            f"{directory.name} has no kernel-metadata.json"
        )


# --- reproduction scripts -------------------------------------------------

REPO = KAGGLE.parent
RECONS = sorted((REPO / "scripts").glob("*_recon.py"))


@pytest.mark.parametrize("path", RECONS, ids=[p.stem for p in RECONS])
def test_every_recon_script_parses(path: Path) -> None:
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", RECONS, ids=[p.stem for p in RECONS])
def test_every_recon_script_is_named_in_a_published_document(path: Path) -> None:
    """A reproduction nobody can find is not a reproduction.

    These scripts exist because a figure in the write-up was worked out by
    hand, and every one of them has since caught something: the single-draw
    control, the frozen Missing bin, and a condition number published as 15.1
    that is 13.5. An unreferenced one would leave the figure it backs looking
    like an assertion again.
    """
    published = (REPO / "README.md").read_text() + (REPO / "MODEL.md").read_text()
    assert f"scripts/{path.name}" in published, (
        f"{path.name} reproduces something, and neither document says so"
    )
