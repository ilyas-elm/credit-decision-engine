"""Record what each pipeline stage consumed, so a stale chain is detectable.

The stages chain: features -> models -> decision, and features -> sampling-bias.
Re-running an upstream stage without re-running everything below it leaves the
downstream outputs describing a world that no longer exists. That happened here:
the sampling-bias results were computed on a feature set that a later features
run replaced, and nothing in the repo said so; it was caught by comparing
directory timestamps by hand, which is not a control.

The fix is to make each stage record a fingerprint of the files it read. A
downstream run whose recorded upstream fingerprint no longer matches the current
upstream output is stale, and `check_chain` says which one and why.

Fingerprints are content hashes, not timestamps: re-running a stage that
produces identical output does not invalidate anything downstream, which is the
behaviour you want when a re-run was a no-op.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "PROVENANCE_FILE",
    "fingerprint",
    "write_provenance",
    "read_provenance",
    "check_chain",
]

PROVENANCE_FILE = "provenance.json"
_CHUNK = 1 << 20


def fingerprint(path: str | Path, *, length: int = 16) -> str:
    """Short content hash of a file. Identical bytes give identical output."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def write_provenance(
    output_dir: str | Path,
    stage: str,
    *,
    inputs: dict[str, str | Path] | None = None,
    outputs: dict[str, str | Path] | None = None,
    extra: dict | None = None,
) -> Path:
    """Record the stage's inputs and outputs by content hash.

    ``inputs`` are the upstream files this stage read; ``outputs`` the files it
    produced that something downstream will read. A missing file is recorded as
    null rather than raising: provenance is a diagnostic and must never be the
    reason a completed run fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def digest(mapping):
        out = {}
        for name, path in (mapping or {}).items():
            path = Path(path)
            out[name] = (
                {"path": str(path), "bytes": path.stat().st_size, "sha256": fingerprint(path)}
                if path.is_file()
                else None
            )
        return out

    record = {
        "stage": stage,
        "written_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": digest(inputs),
        "outputs": digest(outputs),
        **(extra or {}),
    }
    destination = output_dir / PROVENANCE_FILE
    destination.write_text(json.dumps(record, indent=2))
    return destination


def read_provenance(run_dir: str | Path) -> dict | None:
    path = Path(run_dir) / PROVENANCE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _latest_run(runs_root: Path, stage: str) -> Path | None:
    directory = runs_root / stage
    if not directory.is_dir():
        return None
    runs = sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return runs[-1] if runs else None


def check_chain(
    runs_root: str | Path,
    chain: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Report downstream runs whose recorded upstream fingerprint is out of date.

    ``chain`` maps a stage to the stages it depends on. Returns one entry per
    problem found; an empty list means every downstream run was produced from
    the upstream output that is current.
    """
    runs_root = Path(runs_root)
    chain = chain or {
        "models": ["features"],
        "sampling-bias": ["features"],
        "decision": ["features", "models"],
    }

    problems: list[dict] = []
    for stage, upstreams in chain.items():
        run = _latest_run(runs_root, stage)
        if run is None:
            continue
        record = read_provenance(run)
        if record is None:
            problems.append(
                {"stage": stage, "run": run.name, "issue": "no provenance recorded"}
            )
            continue
        for upstream in upstreams:
            up_run = _latest_run(runs_root, upstream)
            if up_run is None:
                continue  # never run here: nothing to be stale against
            up_record = read_provenance(up_run)
            if up_record is None:
                # Reporting this is the whole point. With no fingerprint on the
                # upstream there is nothing to compare, so the comparison is
                # skipped, and a silent skip reads exactly like a pass: the
                # check says "no stale stage found" having checked nothing.
                problems.append(
                    {
                        "stage": stage,
                        "run": run.name,
                        "upstream": upstream,
                        "upstream_run": up_run.name,
                        "issue": "unverifiable: upstream recorded no provenance",
                    }
                )
                continue
            compared = False
            for name, current in (up_record.get("outputs") or {}).items():
                consumed = (record.get("inputs") or {}).get(name)
                if current is None or consumed is None:
                    continue
                compared = True
                if consumed["sha256"] != current["sha256"]:
                    problems.append(
                        {
                            "stage": stage,
                            "run": run.name,
                            "upstream": upstream,
                            "upstream_run": up_run.name,
                            "file": name,
                            "issue": "stale: consumed a superseded upstream output",
                            "consumed_sha256": consumed["sha256"],
                            "current_sha256": current["sha256"],
                        }
                    )
            if not compared:
                # Both sides recorded provenance and still shared no key, so
                # the dependency is declared but never actually checked.
                problems.append(
                    {
                        "stage": stage,
                        "run": run.name,
                        "upstream": upstream,
                        "upstream_run": up_run.name,
                        "issue": "unverifiable: no shared file between the two records",
                    }
                )
    return problems
