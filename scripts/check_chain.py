#!/usr/bin/env python3
"""Report pipeline stages whose results came from a superseded upstream run.

    uv run python scripts/check_chain.py

Exits non-zero when a downstream stage's recorded upstream fingerprint no longer
matches the current upstream output, which means re-running the kernels from
committed code would not reproduce that stage's committed numbers.

This is the control for a failure that happened: the sampling-bias results were
computed on a feature set that a later features run replaced, and the only thing
that caught it was comparing directory timestamps by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from provenance import check_chain  # noqa: E402


def main() -> int:
    problems = check_chain(REPO_ROOT / "runs")
    # A run that predates provenance is UNVERIFIABLE, not stale. Failing on it
    # would make the check cry wolf until every stage happens to be re-run, and
    # a check that always fails is a check nobody reads.
    stale = [p for p in problems if p["issue"] != "no provenance recorded"]
    unverifiable = [p for p in problems if p["issue"] == "no provenance recorded"]

    for p in unverifiable:
        print(f"[check_chain] note: {p['stage']} run {p['run']} predates "
              "provenance; it will be recorded the next time that stage runs")
    if not stale:
        print("[check_chain] no stale stage found")
        return 0

    print(f"[check_chain] {len(stale)} STALE stage(s):", file=sys.stderr)
    for p in stale:
        print(
            f"  {p['stage']:14} {p['run']} consumed {p['file']} "
            f"{p['consumed_sha256']} but {p['upstream']} now produces "
            f"{p['current_sha256']}, re-run {p['stage']}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
