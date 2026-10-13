# Credit decision engine on Home Credit data

A credit decision system — not just a classifier — built on the Kaggle
*Home Credit: Credit Risk Model Stability* data, and evaluated the way a credit
risk team evaluates models: out-of-time, on calibration, stability and profit.

> **Status: in progress.** Nothing below is filled in with numbers yet. Sections
> marked _pending_ have no results, and this notice stays until they do. No
> figure or table in this README is a placeholder for a number that was measured
> and then rounded off — they are simply not measured yet.

## The three questions

1. What does a gradient-boosted challenger add over a classic scorecard once
   stability, calibration and profit are measured, not just AUC?
2. Where should the approval cutoff and the manual-review band sit, and what do
   they cost?
3. If the lender only observed outcomes for loans it approved — how wrong would
   its validation be, and which fix recovers the most per unit of cost: reject
   inference, or a small random-approval (exploration) budget?

## Headline numbers

_Pending._

## Figures

_Pending:_ weekly Gini (champion vs challenger), strategy curve,
sampling-bias recovery, evaluation bias.

## How to reproduce

```bash
uv sync
uv run pytest
```

Tests and CI run entirely on **generated fixtures** (`tests/fixtures.py`) that
match the competition schema. Competition data is never committed, so a fresh
clone needs no Kaggle credentials to go green.

Full-data results come from Kaggle CPU kernels run from this repo's committed
code; the kernel versions used will be linked here.

## What is built

| Area | Module | State |
|---|---|---|
| Config (all thresholds and parameters) | `src/config.py`, `config/default.toml` | done |
| Out-of-time splits by `WEEK_NUM` | `src/splits.py` | done |
| Gini, weekly Gini, stability metric, PSI, calibration | `src/metrics.py` | done |
| EMPCS profit + strategy curve | `src/profit.py` | done |
| Mondrian conformal refer band | `src/conformal.py` | done |
| Synthetic fixtures matching the schema | `tests/fixtures.py` | done |
| Kaggle push/poll/pull orchestration | `scripts/kaggle_run.py` | done, unexercised |
| `make-sample` + reconnaissance kernel | `kaggle/make-sample/` | done, unexercised |
| Data reconnaissance | `docs/DATA_NOTES.md` | blocked on data access |
| Features, scorecard, challenger, decision layer, sampling bias | — | not started |

92 tests, all on generated fixtures. The Kaggle plumbing is marked *unexercised*
because it has never been run against Kaggle — only against its own tests.

## Notes on three things worth knowing

**EMPCS label encoding.** The underlying formula (Verbraken et al., 2014) is
written with the default class as 0, but the pinned `empulse` (0.13.0) performs
that inversion internally and expects `default = 1` — the same encoding as the
Home Credit `target` column. `target` is therefore passed through untouched.
Inverting it makes the metric blind (an informative model and pure noise both
score ≈0.2557) while returning a number ~50× larger, so the error flatters
itself. `tests/test_profit.py` pins this down and will fail if a future
`empulse` flips the convention back.

**Stability metric.** `mean(gini) + 88·min(0, slope) − 0.5·std(residuals)`,
computed on the weekly Gini curve as the model produces it. Scores are never
post-processed per week to improve it.

**Why the refer band is class-conditional.** On synthetic exchangeable data at a
3% default rate and a 90% coverage target, averaged over 300 calibration draws:

| | non-default coverage | **default coverage** |
|---|---|---|
| marginal (one shared threshold) | 0.964 | **0.014** |
| Mondrian (per-class) | 0.900 | **0.899** |

A single shared threshold covers defaulters **1.4% of the time** while its
headline number looks healthy — under 3% defaults the pooled quantile is set
almost entirely by non-defaulters. Mondrian lands on target for both classes.
This is measured on exchangeable data, which is the favourable case; the real
data drifts, so coverage is reported per class *per test week* rather than
assumed.

## Stack

Python 3.11, polars (lazy/streaming) for the feature build, pandas only at the
model boundary, scikit-learn, LightGBM, optbinning, empulse, matplotlib, pytest.
