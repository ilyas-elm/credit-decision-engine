"""Run kaggle/make-sample/stage.py end to end against fixtures.

The stage gets one 12 h Kaggle session per attempt and a queue wait before it.
An API error or a mis-selected table is therefore expensive to discover there,
so the whole script is executed locally against synthetic tables laid out in the
competition's own directory structure (parquet_files/train, parquet_files/test).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest
from fixtures import make_dataset

STAGE_PATH = Path(__file__).resolve().parents[1] / "kaggle" / "make-sample" / "stage.py"


def _load_stage(input_dir: Path, output_dir: Path):
    """Import stage.py with its Kaggle paths redirected at the temp dirs."""
    spec = importlib.util.spec_from_file_location("hc_make_sample_stage", STAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.INPUT = input_dir
    module.OUTPUT = output_dir
    module.SAMPLE_DIR = output_dir / "sample"
    return module


@pytest.fixture
def competition_layout(tmp_path: Path):
    """Fixture tables written where the competition puts them."""
    tables, _ = make_dataset(n_cases=600, n_weeks=12, seed=5)
    train = tmp_path / "input" / "parquet_files" / "train"
    test = tmp_path / "input" / "parquet_files" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)

    for name, frame in tables.items():
        frame.write_parquet(train / f"{name}.parquet")
    # The unlabeled leaderboard set: present, and to be ignored. No `target`.
    tables["train_base"].drop("target").write_parquet(test / "test_base.parquet")
    tables["train_static_0_0"].write_parquet(test / "test_static_0_0.parquet")

    # File name -> logical table name. Part suffixes collapse, which is the
    # whole point of discover_tables: train_static_0_0.parquet is part 0 of the
    # depth-0 table train_static_0, not a table called train_static_0_0.
    logical = {
        "train_base": "train_base",
        "train_static_0_0": "train_static_0",
        "train_applprev_1_0": "train_applprev_1",
        "train_applprev_2": "train_applprev_2",
    }
    return tmp_path / "input", tmp_path / "working", tables, logical


def test_stage_runs_end_to_end_and_writes_the_sample(competition_layout) -> None:
    input_dir, output_dir, tables, logical = competition_layout
    stage = _load_stage(input_dir, output_dir)
    stage.main()

    sample_dir = output_dir / "sample"
    written = sorted(p.stem for p in sample_dir.glob("*.parquet"))
    assert written == sorted(logical.values()), written
    # The leaderboard tables were not sampled.
    assert not any(name.startswith("test_") for name in written)


def test_the_report_records_the_real_structure(competition_layout) -> None:
    input_dir, output_dir, tables, logical = competition_layout
    stage = _load_stage(input_dir, output_dir)
    stage.main()

    report = json.loads((output_dir / "schema_report.json").read_text())
    assert set(report["tables"]) == set(logical.values())
    base = report["base"]["summary"]
    assert base["n_rows"] == tables["train_base"].height
    assert base["n_cases"] == tables["train_base"]["case_id"].n_unique()
    assert base["week_min"] == 0 and base["week_max"] == 11
    assert 0.0 < base["default_rate"] < 0.5
    assert len(report["base"]["weekly_default_rate"]) == 12
    # Full row counts are recorded alongside the sampled ones.
    by_logical = {logical[file_name]: frame for file_name, frame in tables.items()}
    for name, info in report["tables"].items():
        assert info["n_rows_full"] == by_logical[name].height
        assert info["n_rows_sample"] <= info["n_rows_full"]
        assert info["columns"], f"{name} has no recorded schema"


def test_the_sample_is_one_consistent_set_of_case_ids(competition_layout) -> None:
    """Every table must be filtered to the SAME case_ids, or joins break."""
    input_dir, output_dir, _, _ = competition_layout
    stage = _load_stage(input_dir, output_dir)
    stage.main()

    sample_dir = output_dir / "sample"
    base_ids = set(pl.read_parquet(sample_dir / "train_base.parquet")["case_id"].to_list())
    assert base_ids
    for path in sample_dir.glob("*.parquet"):
        ids = set(pl.read_parquet(path)["case_id"].to_list())
        assert ids <= base_ids, f"{path.stem} has case_ids absent from the base sample"


def test_the_sample_keeps_every_week_and_both_classes(competition_layout) -> None:
    """Stratifying on (WEEK_NUM, target) is the point: a plain random sample
    would thin out the rare class and could empty a week.
    """
    input_dir, output_dir, _, _ = competition_layout
    stage = _load_stage(input_dir, output_dir)
    stage.main()

    sampled = pl.read_parquet(output_dir / "sample" / "train_base.parquet")
    assert sorted(sampled["WEEK_NUM"].unique().to_list()) == list(range(12))
    assert sampled["target"].n_unique() == 2


def test_column_search_finds_the_decision_relevant_columns(competition_layout) -> None:
    """Whether credit amount, age and sex exist decides later design choices."""
    input_dir, output_dir, _, _ = competition_layout
    stage = _load_stage(input_dir, output_dir)
    stage.main()

    found = json.loads((output_dir / "schema_report.json").read_text())["columns_of_interest"]
    # The fixtures carry credamount_770A and birthdate_574D.
    assert any(hit["column"] == "credamount_770A" for hit in found["credit_amount"])
    assert any(hit["column"] == "birthdate_574D" for hit in found["age_or_birth"])
    assert found["sex"] == [], "fixtures have no sex column; the search invented one"


def test_a_missing_base_table_fails_loudly(tmp_path: Path) -> None:
    """Rather than silently sampling whatever else is lying around."""
    train = tmp_path / "input" / "parquet_files" / "train"
    train.mkdir(parents=True)
    tables, _ = make_dataset(n_cases=120, n_weeks=4, seed=1)
    tables["train_static_0_0"].write_parquet(train / "train_static_0_0.parquet")

    stage = _load_stage(tmp_path / "input", tmp_path / "working")
    with pytest.raises(SystemExit, match="train_base"):
        stage.main()
