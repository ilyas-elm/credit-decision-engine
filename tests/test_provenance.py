"""Chain-staleness detection.

The bug this exists to catch: sampling-bias was computed on a feature set that a
later features run replaced, and nothing in the repo recorded the fact. It was
found by comparing directory timestamps by hand, which is not a control.
"""

from __future__ import annotations

import json

from provenance import PROVENANCE_FILE, check_chain, fingerprint, read_provenance, write_provenance


def _file(path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_fingerprint_is_content_based_not_timestamp_based(tmp_path) -> None:
    """Re-running a stage that produces identical bytes must not invalidate
    anything downstream, so the hash cannot depend on mtime or path.
    """
    import os
    import time

    a = _file(tmp_path / "a.bin", b"same content")
    b = _file(tmp_path / "b.bin", b"same content")
    time.sleep(0.01)
    os.utime(a, (0, 0))

    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(_file(tmp_path / "c.bin", b"different"))


def test_provenance_records_inputs_and_outputs(tmp_path) -> None:
    source = _file(tmp_path / "features.parquet", b"v1")
    out = tmp_path / "run"
    path = write_provenance(
        out, "models", inputs={"features": source}, outputs={"predictions": source},
        extra={"commit": "abc123"},
    )
    assert path.name == PROVENANCE_FILE
    record = json.loads(path.read_text())
    assert record["stage"] == "models"
    assert record["commit"] == "abc123"
    assert record["inputs"]["features"]["sha256"] == fingerprint(source)
    assert record["inputs"]["features"]["bytes"] == 2
    assert "written_at" in record


def test_a_missing_file_is_recorded_as_null_not_an_error(tmp_path) -> None:
    """Provenance is a diagnostic; it must never fail a completed run."""
    record = read_provenance(
        write_provenance(tmp_path / "r", "x", inputs={"gone": tmp_path / "nope"}).parent
    )
    assert record["inputs"]["gone"] is None


def test_a_consistent_chain_reports_nothing(tmp_path) -> None:
    features = _file(tmp_path / "f.parquet", b"v1")
    write_provenance(tmp_path / "runs" / "features" / "20260101T000000Z", "features",
                     outputs={"features": features})
    write_provenance(tmp_path / "runs" / "models" / "20260101T010000Z", "models",
                     inputs={"features": features})
    assert check_chain(tmp_path / "runs", {"models": ["features"]}) == []


def test_a_stale_downstream_run_is_caught(tmp_path) -> None:
    """The exact failure: downstream ran, THEN features was re-run."""
    runs = tmp_path / "runs"
    v1 = _file(tmp_path / "f1.parquet", b"version one")
    write_provenance(runs / "features" / "20260101T000000Z", "features", outputs={"features": v1})
    # Downstream consumes v1...
    write_provenance(runs / "sampling-bias" / "20260101T010000Z", "sampling-bias",
                     inputs={"features": v1})
    assert check_chain(runs, {"sampling-bias": ["features"]}) == []

    # ...then features is re-run with different content, and nothing downstream is.
    v2 = _file(tmp_path / "f2.parquet", b"version two")
    write_provenance(runs / "features" / "20260102T000000Z", "features", outputs={"features": v2})

    problems = check_chain(runs, {"sampling-bias": ["features"]})
    assert len(problems) == 1
    p = problems[0]
    assert p["stage"] == "sampling-bias"
    assert p["upstream"] == "features"
    assert "stale" in p["issue"]
    assert p["consumed_sha256"] == fingerprint(v1)
    assert p["current_sha256"] == fingerprint(v2)


def test_an_identical_rerun_does_not_invalidate_downstream(tmp_path) -> None:
    """Re-running a stage that produces the same bytes is a no-op downstream."""
    runs = tmp_path / "runs"
    v1 = _file(tmp_path / "f.parquet", b"stable")
    write_provenance(runs / "features" / "20260101T000000Z", "features", outputs={"features": v1})
    write_provenance(runs / "models" / "20260101T010000Z", "models", inputs={"features": v1})
    # Re-run features, same content.
    same = _file(tmp_path / "f_again.parquet", b"stable")
    write_provenance(runs / "features" / "20260102T000000Z", "features", outputs={"features": same})
    assert check_chain(runs, {"models": ["features"]}) == []


def test_a_run_without_provenance_is_reported(tmp_path) -> None:
    runs = tmp_path / "runs"
    (runs / "models" / "20260101T000000Z").mkdir(parents=True)
    problems = check_chain(runs, {"models": ["features"]})
    assert problems and problems[0]["issue"] == "no provenance recorded"


def test_staging_directories_are_ignored(tmp_path) -> None:
    """runs/<stage>/_push is scaffolding, not a run."""
    runs = tmp_path / "runs"
    v1 = _file(tmp_path / "f.parquet", b"v1")
    (runs / "features" / "_push").mkdir(parents=True)
    write_provenance(runs / "features" / "20260101T000000Z", "features", outputs={"features": v1})
    write_provenance(runs / "models" / "20260101T010000Z", "models", inputs={"features": v1})
    assert check_chain(runs, {"models": ["features"]}) == []


def test_check_chain_tolerates_absent_stages(tmp_path) -> None:
    assert check_chain(tmp_path / "nothing-here") == []


def test_unreadable_provenance_is_treated_as_absent(tmp_path) -> None:
    run = tmp_path / "runs" / "models" / "20260101T000000Z"
    run.mkdir(parents=True)
    (run / PROVENANCE_FILE).write_text("{ not json")
    assert read_provenance(run) is None
