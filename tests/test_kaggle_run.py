"""The Kaggle plumbing, exercised without a network or credentials.

None of this can be tested against Kaggle from CI, so the parts that would
silently waste a 12 h session -- a bundle missing a module, a kernel pushed
public, a kernel with no data attached, internet disabled so the pip install
dies at import -- are tested here instead.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import kaggle_run
import pytest
from kaggle_run import (
    STAGE_SCRIPT_NAME,
    StageError,
    build_bundled_script,
    bundle_src,
    load_stage,
    resolve_metadata,
    validate_metadata,
)

REPO_ROOT = Path(kaggle_run.__file__).resolve().parents[1]


def _valid_metadata(**overrides) -> dict:
    metadata = {
        "id": "KAGGLE_USERNAME/hc-test",
        "title": "hc-test",
        "code_file": "_bundled.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_internet": True,
        "competition_sources": ["home-credit-credit-risk-model-stability"],
    }
    metadata.update(overrides)
    return metadata


# --- Bundling ------------------------------------------------------------


def test_bundle_contains_every_src_module() -> None:
    expected = {
        str(p.relative_to(REPO_ROOT / "src"))
        for p in (REPO_ROOT / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    }
    with zipfile.ZipFile(io.BytesIO(bundle_src())) as archive:
        assert set(archive.namelist()) == expected
    assert "conformal.py" in expected and "metrics.py" in expected


def test_bundle_is_byte_identical_across_builds() -> None:
    # Fixed timestamps: an unchanged src/ must not look changed, or every push
    # would appear to be a new version.
    assert bundle_src() == bundle_src()


def test_bundled_modules_import_cleanly_once_unpacked(tmp_path: Path) -> None:
    """The check that matters: the kernel can actually import what it is sent.

    Run in a subprocess with -I so the repo's own src/ is not on the path and
    only the unpacked bundle can satisfy the imports.
    """
    destination = tmp_path / "_src"
    destination.mkdir()
    with zipfile.ZipFile(io.BytesIO(bundle_src())) as archive:
        archive.extractall(destination)

    program = (
        f"import sys; sys.path.insert(0, r'{destination}');"
        "import config, splits, metrics, profit, conformal;"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_bundled_script_is_valid_python_and_embeds_the_stage_body() -> None:
    stage = load_stage("make-sample")
    bundled = build_bundled_script(stage, commit="testcommit")

    compile(bundled, "_bundled.py", "exec")  # syntax, not execution
    assert "testcommit" in bundled
    assert "GENERATED FILE" in bundled
    # A distinctive line from the authored stage script survived.
    assert "schema_report.json" in bundled
    assert "_SRC_B64" in bundled


# --- Stage loading -------------------------------------------------------


def test_the_make_sample_stage_loads_and_is_configured_correctly() -> None:
    stage = load_stage("make-sample")
    assert stage.script.name == STAGE_SCRIPT_NAME
    assert stage.metadata["is_private"] is True
    assert stage.metadata["enable_internet"] is True
    assert not stage.metadata.get("enable_gpu")
    assert kaggle_run.COMPETITION in stage.metadata["competition_sources"]


def test_an_unknown_stage_lists_what_exists() -> None:
    with pytest.raises(StageError, match="No stage directory"):
        load_stage("does-not-exist")


# --- Metadata validation -------------------------------------------------


def test_a_public_kernel_is_refused() -> None:
    with pytest.raises(StageError, match="must stay private"):
        validate_metadata("s", _valid_metadata(is_private=False))


def test_internet_disabled_is_refused() -> None:
    # optbinning and empulse are not preinstalled; the kernel would die at
    # import after waiting in the queue.
    with pytest.raises(StageError, match="not preinstalled"):
        validate_metadata("s", _valid_metadata(enable_internet=False))


def test_requesting_a_gpu_is_refused() -> None:
    with pytest.raises(StageError, match="needs one"):
        validate_metadata("s", _valid_metadata(enable_gpu=True))


def test_a_kernel_with_no_data_attached_is_refused() -> None:
    with pytest.raises(StageError, match="run with no data"):
        validate_metadata("s", _valid_metadata(competition_sources=[], kernel_sources=[]))


def test_chaining_an_upstream_kernel_counts_as_data() -> None:
    # Downstream stages read features through kernel_sources, so they need no
    # competition_sources of their own.
    validate_metadata(
        "s",
        _valid_metadata(competition_sources=[], kernel_sources=["someone/hc-features"]),
    )


def test_code_file_may_not_be_the_authored_script() -> None:
    with pytest.raises(StageError, match="authored source"):
        validate_metadata("s", _valid_metadata(code_file=STAGE_SCRIPT_NAME))


def test_missing_required_fields_are_named() -> None:
    metadata = _valid_metadata()
    del metadata["title"]
    with pytest.raises(StageError, match=r"missing \['title'\]"):
        validate_metadata("s", metadata)


# --- Username resolution and credentials ---------------------------------


def test_username_placeholder_is_substituted_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "someuser")
    stage = load_stage("make-sample")
    assert resolve_metadata(stage)["id"] == "someuser/hc-make-sample"
    # The committed file still carries no personal identifier.
    committed = json.loads((stage.directory / "kernel-metadata.json").read_text())
    assert committed["id"].startswith("KAGGLE_USERNAME/")


def test_missing_credentials_name_both_manual_steps(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    with pytest.raises(StageError) as excinfo:
        kaggle_run.check_credentials()
    message = str(excinfo.value)
    assert "Accept the competition rules" in message
    assert "chmod 600" in message


def test_credentials_are_accepted_from_a_token_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "kaggle.json").write_text(json.dumps({"username": "u", "key": "k"}))
    kaggle_run.check_credentials()
    assert kaggle_run.kaggle_username() == "u"


# --- Busy-kernel guard ---------------------------------------------------


@pytest.mark.parametrize("status", ["running", "queued"])
def test_a_busy_kernel_is_not_pushed_over(monkeypatch, status: str) -> None:
    """Pushing over a live run would queue a second 12 h session and race outputs."""
    monkeypatch.setenv("KAGGLE_USERNAME", "someuser")
    monkeypatch.setenv("KAGGLE_KEY", "somekey")
    monkeypatch.setattr(kaggle_run, "kernel_status", lambda ref: status)

    def _fail(*args, **kwargs):
        raise AssertionError("push must not be attempted while the kernel is busy")

    monkeypatch.setattr(kaggle_run, "push", _fail)
    assert kaggle_run.main(["make-sample"]) == 2


def test_dry_run_needs_no_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    def _fail(*args, **kwargs):
        raise AssertionError("a dry run must not touch the Kaggle CLI")

    monkeypatch.setattr(kaggle_run, "_kaggle", _fail)
    assert kaggle_run.main(["make-sample", "--dry-run"]) == 0


def test_future_imports_are_hoisted_above_the_bootstrap() -> None:
    """A future statement must be the first statement, or the kernel will not start.

    This is the failure mode that costs a queue wait and a dead session, so it
    is pinned rather than left to the next person to rediscover.
    """
    from kaggle_run import _hoist_future_imports

    futures, body = _hoist_future_imports(
        '"""Doc."""\nfrom __future__ import annotations\nimport os\n'
    )
    assert futures == "from __future__ import annotations\n"
    assert "__future__" not in body

    bundled = build_bundled_script(load_stage("make-sample"), commit="c")
    assert bundled.startswith("from __future__ import annotations")
    assert bundled.count("from __future__ import annotations") == 1
