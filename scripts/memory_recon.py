#!/usr/bin/env python
"""Peak memory of the feature build on the local sample.

The brief caps it at about 4 GB, because the local machine has 8 GB of unified
memory and the build must leave room for everything else. The README reports a
measured peak; this is the measurement, so the figure can be re-taken rather
than trusted.

    uv run python scripts/memory_recon.py [data/sample]

It runs `build_features` in this process and reads the peak resident set size
from `getrusage`, which counts everything the process touched, including
polars' native allocations that Python-level tracers miss. Run it in a fresh
process: the peak is a high-water mark, so anything loaded earlier in the same
interpreter would be counted against the build.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUDGET_GB = 4.0


def peak_rss_gb() -> float:
    """Peak resident set size so far, in GB.

    ru_maxrss is bytes on macOS and kilobytes on Linux; getting that wrong is
    a factor of 1,024 either way.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return peak * scale / 1024**3


def main(argv: list[str]) -> int:
    sample = Path(argv[0]) if argv else REPO / "data" / "sample"
    if not sample.is_dir():
        print(f"No sample at {sample}; the make-sample kernel produces it.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(REPO / "src"))
    from features import build_features

    before = peak_rss_gb()
    started = time.time()
    frame, families = build_features(sample)
    seconds = time.time() - started
    peak = peak_rss_gb()

    float_cols = [c for c, t in frame.schema.items() if t.is_float()]
    wide = [c for c in float_cols if str(frame.schema[c]) != "Float32"]
    print(f"sample          {sample}")
    print(f"rows x cols     {frame.height:,} x {frame.width}")
    print(f"features        {len(families)} in {len(set(families.values()))} families")
    print(f"float columns   {len(float_cols)}, of which not Float32: {len(wide)}")
    print(f"build time      {seconds:.1f} s")
    print(f"peak RSS        {peak:.2f} GB")
    print(f"  of which      {before:.2f} GB was the interpreter and imports, before the build")
    print(f"budget          {BUDGET_GB:.1f} GB -> {'within' if peak < BUDGET_GB else 'OVER'}")
    return 0 if peak < BUDGET_GB and not wide else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
