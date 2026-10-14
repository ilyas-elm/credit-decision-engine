#!/usr/bin/env python3
"""Push one pipeline stage to a Kaggle CPU kernel, wait for it, pull its output.

    uv run python scripts/kaggle_run.py <stage> [--dry-run]

Each stage is a directory `kaggle/<stage>/` holding a `kernel-metadata.json` and
a stage script. This tool:

  1. bundles `src/` into the stage script, so the kernel runs the committed code
  2. refuses to push when that kernel is already running or queued
  3. pushes, then polls `kaggle kernels status`
  4. pulls the output into `runs/<stage>/<timestamp>/`

Why bundle rather than ship a directory: `kaggle kernels push` uploads only the
single file named by `code_file`. Anything else in the folder is silently left
behind -- which would mean a kernel running whatever `src/` happened to be
baked into an older utility script, with no error to notice. So `src/` is zipped,
base64-encoded into the generated script, and unpacked to sys.path at startup.
One file, and it always carries the current source.

`--dry-run` builds the bundle and validates the metadata without touching the
network or needing credentials, which is also what the tests exercise.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

# Run as a script, scripts/ is on sys.path but src/ is not, so the project's own
# config loader would not import. pytest hides this by setting pythonpath, which
# is exactly why it has to be done here too.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
KAGGLE_DIR = REPO_ROOT / "kaggle"
RUNS_DIR = REPO_ROOT / "runs"

COMPETITION = "home-credit-credit-risk-model-stability"

# Each stage directory holds this hand-written script; `code_file` in the
# metadata names the GENERATED bundle, which is gitignored. Keeping the two
# separate means a push can never clobber the authored source.
STAGE_SCRIPT_NAME = "stage.py"

# Statuses that mean "do not push on top of this".
BUSY_STATUSES = {"running", "queued"}
TERMINAL_STATUSES = {"complete", "error", "cancelled", "cancelAcknowledged"}

BOOTSTRAP_TEMPLATE = """\
# ------------------------------------------------------------------------------
# GENERATED FILE -- do not edit.
# Built by scripts/kaggle_run.py from kaggle/{stage}/{script_name} and src/.
# Source commit: {commit}
# Built at:      {built_at}
# ------------------------------------------------------------------------------
import base64 as _b64, io as _io, pathlib as _pl, sys as _sys, zipfile as _zf

_SRC_B64 = "{payload}"

# The archive preserves the repo layout (src/ and config/ side by side), so
# config.load_config resolves exactly as it does locally.
_dest = _pl.Path("/kaggle/working/_bundle")
if not _dest.exists():
    _dest.mkdir(parents=True, exist_ok=True)
    with _zf.ZipFile(_io.BytesIO(_b64.b64decode(_SRC_B64))) as _archive:
        _archive.extractall(_dest)
_sys.path.insert(0, str(_dest / "src"))
print("[kaggle_run] project unpacked to", _dest, flush=True)
print("[kaggle_run] config:", sorted(p.name for p in (_dest / "config").glob("*.toml")), flush=True)
print("[kaggle_run] python", _sys.version, flush=True)

# ---------------------------- stage script ------------------------------------
"""


class StageError(RuntimeError):
    """A problem with the stage definition or the Kaggle CLI."""


@dataclass(frozen=True)
class Stage:
    name: str
    directory: Path
    metadata: dict
    script: Path

    @property
    def ref(self) -> str:
        return self.metadata["id"]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def bundle_src(repo_root: Path = REPO_ROOT) -> bytes:
    """Zip the project's runtime files, deterministically.

    Ships `src/**/*.py` AND `config/*.toml`, preserving the repo's relative
    layout inside the archive. The config is not optional: `config.load_config`
    resolves its path relative to the package directory, so a bundle carrying
    only the modules produces

        FileNotFoundError: /kaggle/working/config/default.toml

    the moment any stage reads a threshold -- after the queue wait and after
    pip-installing its dependencies. Keeping the two in one archive, at the same
    relative offsets as the repo, means the kernel resolves config exactly as a
    local run does.

    Fixed timestamps and sorted names keep the bundle byte-identical for
    identical input, so an unchanged stage does not look changed.
    """
    modules = sorted(
        p for p in (repo_root / "src").rglob("*.py") if "__pycache__" not in p.parts
    )
    if not modules:
        raise StageError(f"No Python modules found under {repo_root / 'src'}.")
    configs = sorted((repo_root / "config").glob("*.toml"))
    if not configs:
        raise StageError(
            f"No config files under {repo_root / 'config'}; stages read their "
            "thresholds from there."
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in modules + configs:
            info = zipfile.ZipInfo(str(path.relative_to(repo_root)))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def _hoist_future_imports(body: str) -> tuple[str, str]:
    """Split `from __future__ import ...` lines out of a stage script.

    A future statement must be the first statement in a file. Prepending the
    bootstrap to a stage script that opens with `from __future__ import
    annotations` produces a SyntaxError -- which on Kaggle means the kernel dies
    the instant it starts, after waiting out the queue. So the future imports are
    lifted to the very top of the generated file and removed from the body.
    """
    future_lines, remaining = [], []
    for line in body.splitlines(keepends=True):
        if line.startswith("from __future__ import"):
            future_lines.append(line)
        else:
            remaining.append(line)
    return "".join(future_lines), "".join(remaining)


def build_bundled_script(stage: Stage, *, commit: str | None = None) -> str:
    """The generated single-file kernel script: future imports, bootstrap, body."""
    payload = base64.b64encode(bundle_src()).decode("ascii")
    header = BOOTSTRAP_TEMPLATE.format(
        stage=stage.name,
        script_name=stage.script.name,
        commit=commit if commit is not None else _git_commit(),
        built_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        payload=payload,
    )
    futures, body = _hoist_future_imports(stage.script.read_text())
    return futures + header + body


def load_stage(name: str) -> Stage:
    """Read and validate `kaggle/<name>/`."""
    directory = KAGGLE_DIR / name
    if not directory.is_dir():
        available = sorted(p.name for p in KAGGLE_DIR.iterdir() if p.is_dir())
        raise StageError(f"No stage directory {directory}. Available: {available or 'none'}.")

    metadata_path = directory / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise StageError(f"Missing {metadata_path}.")
    metadata = json.loads(metadata_path.read_text())

    validate_metadata(name, metadata)

    script = directory / STAGE_SCRIPT_NAME
    if not script.is_file():
        raise StageError(f"Stage {name!r} has no {STAGE_SCRIPT_NAME} at {script}.")
    return Stage(name=name, directory=directory, metadata=metadata, script=script)


def validate_metadata(name: str, metadata: dict) -> None:
    """Fail before the network on the fields that silently ruin a run."""
    required = ("id", "title", "code_file", "language", "kernel_type")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise StageError(f"kaggle/{name}/kernel-metadata.json is missing {missing}.")

    if metadata.get("is_private") is not True:
        raise StageError(
            f"Stage {name!r} has is_private={metadata.get('is_private')!r}. Competition "
            "work must stay private; set it to true."
        )
    # optbinning and empulse are not preinstalled on Kaggle, so without internet
    # the kernel dies at import after burning its queue time.
    if metadata.get("enable_internet") is not True:
        raise StageError(
            f"Stage {name!r} has enable_internet={metadata.get('enable_internet')!r}. "
            "optbinning and empulse are not preinstalled on Kaggle and must be "
            "pip-installed in the kernel; set it to true."
        )
    if metadata.get("enable_gpu"):
        raise StageError(f"Stage {name!r} requests a GPU. No stage in this project needs one.")

    code_file = metadata["code_file"]
    if code_file == STAGE_SCRIPT_NAME:
        raise StageError(
            f"Stage {name!r} points code_file at {STAGE_SCRIPT_NAME}, which is the "
            "authored source. It must name the generated bundle instead, so a push "
            "cannot overwrite what you wrote."
        )
    if not code_file.endswith(".py"):
        raise StageError(f"Stage {name!r} has a non-Python code_file {code_file!r}.")

    sources = metadata.get("competition_sources", [])
    chained = metadata.get("kernel_sources", [])
    if COMPETITION not in sources and not chained:
        raise StageError(
            f"Stage {name!r} declares neither competition_sources containing "
            f"{COMPETITION!r} nor kernel_sources to chain an upstream stage. It would "
            "run with no data."
        )


USERNAME_PLACEHOLDER = "KAGGLE_USERNAME"


def kaggle_username() -> str:
    """The account's username, for building the kernel id.

    Kaggle CLI 2.x authenticates with a single OAuth/API token and no longer
    stores a username anywhere: `kaggle config` holds only competition, path and
    proxy, and there is no kaggle.json. So the username cannot be discovered
    from the credentials and has to be supplied.

    $KAGGLE_USERNAME wins when set; otherwise [kaggle] username in config. A
    Kaggle username is not a secret -- it is the public profile URL -- but it is
    personal, so the committed default is empty and the placeholder stays in the
    kernel metadata.
    """
    from_env = os.environ.get("KAGGLE_USERNAME")
    if from_env:
        return from_env

    # Deliberately not wrapped in a bare except: swallowing an ImportError here
    # once turned "src/ is not on sys.path" into "no username available", which
    # sent the search in entirely the wrong direction.
    from config import load_config

    configured = load_config().raw.get("kaggle", {}).get("username", "")
    if configured:
        return str(configured)

    raise StageError(
        "No Kaggle username available, and the CLI cannot supply one.\n"
        "  Set it in config/default.toml:   [kaggle]\n"
        '                                   username = "your-kaggle-username"\n'
        "  or export KAGGLE_USERNAME=your-kaggle-username\n"
        "It is the name in your profile URL, kaggle.com/<username>."
    )


def resolve_metadata(stage: Stage) -> dict:
    """Substitute the username placeholder in the kernel id.

    The committed metadata says `KAGGLE_USERNAME/hc-<stage>` so that the repo
    carries no personal identifier; the real id is assembled at push time.
    """
    metadata = dict(stage.metadata)
    ref = metadata["id"]
    if ref.startswith(f"{USERNAME_PLACEHOLDER}/"):
        metadata["id"] = ref.replace(USERNAME_PLACEHOLDER, kaggle_username(), 1)
    return metadata


# --- Kaggle CLI ----------------------------------------------------------


def kaggle_executable() -> str:
    """Locate the kaggle CLI.

    Looks beside the running interpreter first. `kaggle` is a dev dependency of
    this project, so when the script runs under .venv/bin/python the CLI is at
    .venv/bin/kaggle whether or not that directory is on PATH -- which it is not
    when the script is invoked by its interpreter directly.
    """
    beside = Path(sys.executable).parent / "kaggle"
    if beside.is_file():
        return str(beside)
    found = shutil.which("kaggle")
    if found is None:
        raise StageError(
            "The `kaggle` CLI was not found beside the interpreter "
            f"({beside}) or on PATH. It is a dev dependency of this repo: run "
            "`uv sync`, then invoke this script with .venv/bin/python."
        )
    return found


def _kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = kaggle_executable()
    result = subprocess.run([exe, *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise StageError(
            f"`kaggle {' '.join(args)}` failed with code {result.returncode}.\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )
    return result


def check_credentials() -> None:
    """Fail early if the CLI cannot authenticate.

    Delegated to the CLI rather than probing for a file. Kaggle CLI 2.x accepts
    OAuth credentials cached by `kaggle auth login`, $KAGGLE_API_TOKEN, or a
    token in ~/.kaggle/access_token -- and the old kaggle.json with a username
    and key is gone entirely. Asking the CLI covers all of them and cannot drift
    from whatever it supports next.

    The access token is never captured: stdout goes to DEVNULL and only the exit
    status is read.
    """
    exe = kaggle_executable()
    # Retried once. The OAuth access token is short-lived and the CLI refreshes
    # it lazily, so a probe can fail and the very next call succeed -- observed
    # mid-session, where it aborted a push against perfectly good credentials.
    # Only a genuine lack of credentials fails both attempts.
    for attempt in range(2):
        probe = subprocess.run(
            [exe, "auth", "print-access-token"],
            stdout=subprocess.DEVNULL,  # never capture the token
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return
        if attempt == 0:
            print("[kaggle_run] auth probe failed; refreshing and retrying", file=sys.stderr)
            time.sleep(5)
    raise StageError(
        "Kaggle CLI is not authenticated.\n"
        "  1. Accept the competition rules at\n"
        f"     https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
        "     The API refuses to serve the data until you do.\n"
        "  2. Log in:  uv run kaggle auth login\n"
        "     That opens a browser; credentials are cached locally.\n"
        "     Or generate a token at https://www.kaggle.com/settings/api and\n"
        "     export KAGGLE_API_TOKEN=... (there is no kaggle.json in CLI 2.x)."
    )


def kernel_status(ref: str) -> str:
    """Current status string, or 'absent' for a kernel that does not exist yet."""
    result = _kaggle("kernels", "status", ref, check=False)
    blob = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        lowered = blob.lower()
        # A kernel that has never been pushed reports as a PERMISSION error,
        # not a 404: the API will not confirm whether a kernel it will not show
        # you exists at all. Since `ref` is always built from this account's own
        # username, a denial here means "not created yet" rather than "someone
        # else's". Observed verbatim on the first push of a new stage:
        #   Cannot access kernel '<user>/<slug>' (Permission 'kernels.get' was denied)
        absent_markers = ("not found", "404", "permission", "denied", "cannot access")
        if any(marker in lowered for marker in absent_markers):
            return "absent"
        raise StageError(f"Could not read status of {ref}:\n{blob}")
    for status in (*BUSY_STATUSES, *TERMINAL_STATUSES):
        if status.lower() in blob.lower():
            return status
    return blob or "unknown"


def push(stage: Stage, bundled: str) -> str:
    """Assemble a staging directory and push that.

    Nothing committed is modified: the resolved metadata and the generated
    bundle are written under runs/<stage>/_push/ (gitignored), which is also
    what makes the exact uploaded code inspectable after the run.
    """
    staging = RUNS_DIR / stage.name / "_push"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    metadata = resolve_metadata(stage)
    (staging / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (staging / metadata["code_file"]).write_text(bundled)

    print(f"[kaggle_run] pushing {metadata['id']} from {staging}")
    _kaggle("kernels", "push", "-p", str(staging))
    return metadata["id"]


def wait_for(ref: str, *, poll_seconds: int = 60, timeout_hours: float = 12.0) -> str:
    """Poll until the kernel reaches a terminal status.

    The 12 h ceiling is Kaggle's own session limit, so a run that outlasts it
    has already been killed on their side.
    """
    deadline = time.monotonic() + timeout_hours * 3600
    while True:
        status = kernel_status(ref)
        print(f"[kaggle_run] {ref}: {status}", flush=True)
        if status in TERMINAL_STATUSES:
            return status
        if time.monotonic() > deadline:
            raise StageError(f"{ref} did not finish within {timeout_hours} h (last: {status}).")
        time.sleep(poll_seconds)


def pull_output(stage: Stage, ref: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = RUNS_DIR / stage.name / timestamp
    destination.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", ref, "-p", str(destination))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", help="a directory name under kaggle/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate without pushing; needs no credentials",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-wait", action="store_true", help="push and return without polling")
    args = parser.parse_args(argv)

    try:
        stage = load_stage(args.stage)
        bundled = build_bundled_script(stage)
        size_kb = len(bundled.encode()) / 1024
        print(f"[kaggle_run] stage {stage.name!r} -> {stage.ref}")
        print(f"[kaggle_run] bundled script: {size_kb:.1f} KB")

        if args.dry_run:
            out = stage.directory / stage.metadata["code_file"]
            out.write_text(bundled)
            print(f"[kaggle_run] dry run only; wrote {out}")
            return 0

        check_credentials()
        ref = resolve_metadata(stage)["id"]

        status = kernel_status(ref)
        if status in BUSY_STATUSES:
            raise StageError(
                f"{ref} is already {status}. Refusing to push on top of it: that "
                "would queue a second 12 h session and the outputs would race."
            )
        print(f"[kaggle_run] current status: {status}; pushing")
        push(stage, bundled)

        if args.no_wait:
            print("[kaggle_run] pushed; not waiting")
            return 0

        final = wait_for(ref, poll_seconds=args.poll_seconds)
        if final != "complete":
            print(f"[kaggle_run] finished as {final!r}; pulling whatever exists", file=sys.stderr)
        destination = pull_output(stage, ref)
        print(f"[kaggle_run] output -> {destination}")
        return 0 if final == "complete" else 1

    except StageError as exc:
        print(f"[kaggle_run] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
