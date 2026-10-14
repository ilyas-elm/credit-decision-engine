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
| Champion scorecard (binning, selection, pdo_odds scaling) | `src/scorecard.py` | done |
| Mondrian conformal refer band | `src/conformal.py` | done |
| Decision layer: cutoff, bands, reasons, recourse | `src/decision.py` | done |
| Isotonic calibration (shared) | `src/calibrate.py` | done |
| Challenger: LightGBM with monotone constraints | `src/challenger.py` | done |
| Shared evaluation + the three figures | `src/evaluation.py` | done |
| Synthetic fixtures matching the schema | `tests/fixtures.py` | done |
| Kaggle push/poll/pull orchestration | `scripts/kaggle_run.py` | **done, exercised against the live API** |
| `make-sample` + reconnaissance kernel | `kaggle/make-sample/` | **done, run on full data** |
| Environment probe kernel | `kaggle/env-probe/` | **done, run** |
| Data reconnaissance | [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | **done, on full data** |
| Feature build (polars) | — | next |
| Sampling-bias experiment (arms A0-A6) | `src/sampling_bias.py` | done, not yet run |

226 tests, all on generated fixtures. Two kernels have been run end to end on the real
competition data.

## Environment

Pinned **to the Kaggle image**, not to the newest releases: Python 3.13.15,
polars 1.35.2, pandas 2.3.3, numpy 2.1.3, scikit-learn 1.6.1, LightGBM 4.6.0.
Full-data results come from Kaggle kernels, so local tests are only evidence
about those runs if the stack matches. The versions were measured by
`kaggle/env-probe/`, not assumed — see
[docs/DATA_NOTES.md](docs/DATA_NOTES.md#kaggle-environment-as-measured).

## Notes on four things worth knowing

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

**`empcs` returns a rejection rate, not an acceptance rate.** The brief
describes `empcs` as returning "the profit-maximizing acceptance fraction".
`empcs_score.optimal_rate` actually returns the predicted-*positive* rate, and
since the positive class here is default, that is the fraction **rejected**.
Across nine bad-rate and loss-assumption combinations the raw value *rises* with
the bad rate (0.028 at a 5% bad book to 0.174 at 30%) — the signature of a
rejection rate — while `1 - rate` tracks the strategy curve's acceptance optimum
throughout. Taken at face value it would have meant accepting 2.8% of a low-risk
portfolio. The cutoff is set from the strategy curve; EMPCS's implied optimum is
reported beside it, with the gap surfaced rather than reconciled, since the two
optimise different objectives.

**The IV floor is sample-size dependent.** IV is computed on the bins it
chose, so it is biased upward and the bias shrinks with n. Over three pure-noise
columns at an 8% event rate, max IV runs 0.0889 at n=2,000, 0.0117 at n=10,000
and 0.0006 at n=200,000. The configured floor of 0.02 therefore screens noise
against the full train set but **not** on the 3% local sample — so feature
selection is a full-data decision, and the local sample exists to make the code
run, not to decide which features live.

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
