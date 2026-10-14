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


def test_missing_credentials_name_both_manual_steps(monkeypatch) -> None:
    """The check delegates to the CLI, so an unauthenticated CLI must surface
    the real instructions -- OAuth login, not the kaggle.json that CLI 2.x
    no longer uses.
    """

    class _Failed:
        returncode = 1

    monkeypatch.setattr(kaggle_run.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Failed())
    with pytest.raises(StageError) as excinfo:
        kaggle_run.check_credentials()
    message = str(excinfo.value)
    assert "Accept the competition rules" in message
    assert "kaggle auth login" in message
    assert "KAGGLE_API_TOKEN" in message
    # The old scheme must not be advertised any more.
    assert "kaggle.json" in message and "no kaggle.json" in message.lower()


def test_an_authenticated_cli_passes_the_check(monkeypatch) -> None:
    class _Ok:
        returncode = 0

    monkeypatch.setattr(kaggle_run.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Ok())
    kaggle_run.check_credentials()


def test_a_missing_cli_is_reported_clearly(monkeypatch) -> None:
    monkeypatch.setattr(kaggle_run.shutil, "which", lambda name: None)
    with pytest.raises(StageError, match="not on PATH"):
        kaggle_run.check_credentials()


def test_the_access_token_is_never_captured(monkeypatch) -> None:
    """The probe must discard stdout: it prints a live credential."""
    seen = {}

    class _Ok:
        returncode = 0

    def _spy(args, **kwargs):
        seen.update(kwargs)
        return _Ok()

    monkeypatch.setattr(kaggle_run.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(subprocess, "run", _spy)
    kaggle_run.check_credentials()
    assert seen.get("stdout") is subprocess.DEVNULL
    assert "capture_output" not in seen


def test_username_comes_from_config_when_the_environment_is_unset() -> None:
    """CLI 2.x stores no username, so it comes from config or the environment."""
    import os

    from config import load_config

    configured = load_config().raw.get("kaggle", {}).get("username", "")
    assert configured, "[kaggle] username is unset in config/default.toml"

    saved = os.environ.pop("KAGGLE_USERNAME", None)
    try:
        assert kaggle_run.kaggle_username() == configured
    finally:
        if saved is not None:
            os.environ["KAGGLE_USERNAME"] = saved


def test_the_environment_overrides_config(monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "someone-else")
    assert kaggle_run.kaggle_username() == "someone-else"


def test_username_error_names_both_ways_to_set_it(monkeypatch) -> None:
    """With neither source set, the error has to say how to fix it."""
    import config as config_module

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

    class _Empty:
        raw: dict = {"kaggle": {"username": ""}}

    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: _Empty())
    with pytest.raises(StageError) as excinfo:
        kaggle_run.kaggle_username()
    message = str(excinfo.value)
    assert "config/default.toml" in message
    assert "KAGGLE_USERNAME" in message


# --- Busy-kernel guard ---------------------------------------------------


@pytest.mark.parametrize("status", ["running", "queued"])
def test_a_busy_kernel_is_not_pushed_over(monkeypatch, status: str) -> None:
    """Pushing over a live run would queue a second 12 h session and race outputs."""
    monkeypatch.setenv("KAGGLE_USERNAME", "someuser")
    monkeypatch.setattr(kaggle_run, "check_credentials", lambda: None)
    monkeypatch.setattr(kaggle_run, "kernel_status", lambda ref: status)

    def _fail(*args, **kwargs):
        raise AssertionError("push must not be attempted while the kernel is busy")

    monkeypatch.setattr(kaggle_run, "push", _fail)
    assert kaggle_run.main(["make-sample"]) == 2


def test_dry_run_needs_no_credentials(monkeypatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)

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


# --- make-sample table discovery, against the real file list ----------------

# The actual parquet filenames in the competition, read from the Kaggle API.
# Names only, no data: the table inventory is something docs/DATA_NOTES.md is
# meant to publish anyway.
REAL_PARQUET_NAMES = [
    "parquet_files/train/train_base.parquet",
    "parquet_files/train/train_static_0_0.parquet",
    "parquet_files/train/train_static_0_1.parquet",
    "parquet_files/train/train_static_cb_0.parquet",
    "parquet_files/train/train_applprev_1_0.parquet",
    "parquet_files/train/train_applprev_1_1.parquet",
    "parquet_files/train/train_applprev_2.parquet",
    "parquet_files/train/train_credit_bureau_a_1_0.parquet",
    "parquet_files/train/train_credit_bureau_a_1_3.parquet",
    "parquet_files/train/train_credit_bureau_a_2_0.parquet",
    "parquet_files/train/train_credit_bureau_a_2_10.parquet",
    "parquet_files/train/train_credit_bureau_b_2.parquet",
    "parquet_files/train/train_tax_registry_a_1.parquet",
    "parquet_files/train/train_person_2.parquet",
    # The leaderboard submission set, which must be ignored entirely.
    "parquet_files/test/test_base.parquet",
    "parquet_files/test/test_applprev_1_0.parquet",
    "parquet_files/test/test_credit_bureau_a_2_11.parquet",
]


def _group(names: list[str], prefix: str = "train") -> dict[str, list[str]]:
    """The grouping rule from kaggle/make-sample/stage.py, applied to names."""
    from pathlib import PurePosixPath

    tables: dict[str, list[str]] = {}
    for path in sorted(names):
        stem = PurePosixPath(path).stem
        if not stem.startswith(f"{prefix}_"):
            continue
        parts = stem.split("_")
        while len(parts) > 1 and parts[-1].isdigit():
            candidate = "_".join(parts[:-1])
            if parts[-1] in {"0", "1", "2"} and not candidate.endswith(("_0", "_1", "_2")):
                break
            parts = parts[:-1]
        tables.setdefault("_".join(parts), []).append(path)
    return tables


def test_part_numbers_are_distinguished_from_depth_suffixes() -> None:
    """credit_bureau_a_2 has parts numbered past 10, so a trailing digit is
    ambiguous: _2 is the depth, _10 is the part.
    """
    tables = _group(REAL_PARQUET_NAMES)
    assert "train_credit_bureau_a_2" in tables
    assert len(tables["train_credit_bureau_a_2"]) == 2  # parts 0 and 10
    assert "train_applprev_1" in tables
    assert len(tables["train_applprev_1"]) == 2
    # A depth-2 table with no part suffix at all.
    assert "train_applprev_2" in tables
    # static_cb_0 must not collapse into static_0.
    assert "train_static_cb_0" in tables
    assert len(tables["train_static_0"]) == 2


def test_the_leaderboard_test_tables_are_ignored() -> None:
    """They are unlabeled and out of scope; reading them wastes a session."""
    tables = _group(REAL_PARQUET_NAMES)
    assert not any(name.startswith("test_") for name in tables)
    assert all("/test/" not in p for paths in tables.values() for p in paths)


def test_the_base_table_is_selected_by_exact_name_not_by_suffix() -> None:
    """Regression: `endswith("_base")` picked test_base, since "test" sorts
    before "train". That is the unlabeled submission set, a few hundred rows
    with no target column.
    """
    tables = _group(REAL_PARQUET_NAMES)
    assert "train_base" in tables
    # The old rule, over the unfiltered inventory, chose the wrong one.
    unfiltered = sorted({n.split("/")[-1].replace(".parquet", "") for n in REAL_PARQUET_NAMES})
    first_base_like = next(k for k in unfiltered if k.endswith("_base"))
    assert first_base_like == "test_base", "the hazard this guards against is gone"
