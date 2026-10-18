#!/usr/bin/env python
"""Draw question 3's two figures from the committed arms table.

    uv run python scripts/sampling_bias_figures.py

Reads docs/sampling_bias_arms.csv and writes docs/figures/evaluation_bias.png
and docs/figures/sampling_bias_recovery.png. Both figures were once drawn by
hand from a run two re-runs out of date, with nothing in the repo able to
redraw them; this is that something, and it needs no competition data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(REPO / "src"))
    from evaluation import plot_evaluation_bias, plot_recovery

    arms = pd.read_csv(REPO / "docs" / "sampling_bias_arms.csv")
    out = REPO / "docs" / "figures"
    for path in (
        plot_evaluation_bias(arms, out / "evaluation_bias.png"),
        plot_recovery(arms, out / "sampling_bias_recovery.png"),
    ):
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
