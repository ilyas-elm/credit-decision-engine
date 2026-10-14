# Credit decision engine on Home Credit data

A credit decision system — not just a classifier — built on the Kaggle
*Home Credit: Credit Risk Model Stability* data, and evaluated the way a credit
risk team evaluates models: out-of-time, on calibration, stability and profit.

> **Status: in progress.** All three questions have full-data numbers below.
> The decision layer (cutoff, refer band, reasons, recourse) is built and tested
> but has not yet been run on the full data.

## The three questions

1. What does a gradient-boosted challenger add over a classic scorecard once
   stability, calibration and profit are measured, not just AUC?
2. Where should the approval cutoff and the manual-review band sit, and what do
   they cost?
3. If the lender only observed outcomes for loans it approved — how wrong would
   its validation be, and which fix recovers the most per unit of cost: reject
   inference, or a small random-approval (exploration) budget?

## Headline numbers

Full data, 1,526,659 cases, evaluated out-of-time on the last 23 weeks
(`WEEK_NUM` 69–91), which nothing was fitted on. Kernel:
`ilyaselmaazouzi/hc-models`.

| | scorecard | LightGBM | better |
|---|---|---|---|
| Gini | 0.4069 | **0.4675** | challenger |
| mean weekly Gini | 0.3953 | **0.4595** | challenger |
| Gini slope / week | +0.0032 | +0.0023 | — (both rising) |
| Gini residual sd | **0.0467** | 0.0511 | champion |
| **stability metric** | 0.3720 | **0.4339** | challenger |
| Brier | 0.02129 | **0.02104** | challenger |
| calibration-in-the-large, worst week | 0.0096 | **0.0088** | challenger |
| score PSI vs train | 0.3873 | **0.3162** | challenger |
| EMPCS | 0.000000 | 0.000024 | — (see below) |

**Question 1 — what the challenger adds.** About **6 Gini points** (0.407 →
0.468), and it keeps them once stability is measured: the stability metric
moves 0.372 → 0.434. Neither model's Gini decays over the test weeks — both
slopes are positive, so the metric's 88× falling-slope penalty never engages
and the gap is driven by level, not trend. The challenger is also *slightly*
better calibrated and noticeably more stable in its score distribution
(PSI 0.32 vs 0.39). The one thing the champion wins is week-to-week smoothness
(residual sd 0.047 vs 0.051).

It is a real but unspectacular gain, bought with 713 trees against a 17-line
scorecard, under monotone constraints taken from the champion's own binning so
the challenger could not exploit relationships the scorecard was forbidden.

**The profit comparison is degenerate, and that is the finding.** A loan is
profitable in expectation while `p(default) < roi / (roi + E[loss])`. At the
literature's assumptions that break-even is **49.0%**, against a book that
defaults at **3.1%** — so approving everyone is optimal, no cutoff binds, and
EMPCS is correctly ≈0 for both models. The original sensitivity grid never
crossed break-even either (lowest 34.8%). The grid now spans thin-margin,
high-loss-given-default credit (break-even 2.2%–49.0%) so it brackets the point
at which a cutoff starts to pay; the centre point is unchanged, because the
degeneracy there is a result about the assumptions rather than something to
tune away. Under this portfolio's economics, the model's value is not in
choosing an approval cutoff.

Score PSI above 0.25 for **both** models is worth flagging: in production that
is an alert threshold, and it reflects genuine drift over 92 weeks rather than
a modelling defect. Test-week PSI is reported here only — it is never used for
any selection decision.

## Figures

![Weekly Gini](docs/figures/weekly_gini.png)

[Calibration by bucket](docs/figures/calibration.png) ·
[Strategy curve](docs/figures/strategy_curve.png)

![Evaluation bias](docs/figures/evaluation_bias.png)

[Sampling-bias recovery](docs/figures/sampling_bias_recovery.png)

## Question 3 — sampling bias

Kernel `ilyaselmaazouzi/hc-sampling-bias`: 160 fits, 17.7 min. Policy weeks
0–22, development 23–63 (labels hidden for the rejected), test 64–91 with every
label. 150k development cases per arm, identical features and fixed
hyperparameters throughout, three seeds on the stochastic arms.

**Limitation, which bounds every number here:** the Home Credit data already
contains booked loans only. The real rejects are gone. This simulates a *second*
selection policy on top of an already-selected population.

### How wrong is the lender's own validation?

Measured on the test weeks its policy would have approved, against the truth
over everyone:

| reject rate | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|
| LightGBM | −0.068 | −0.075 | −0.075 | −0.069 |
| scorecard | −0.100 | −0.129 | −0.141 | −0.150 |

A lender measuring on its own approved book **understates its Gini by 7 to 15
points** — range restriction, not a modelling fault. The scorecard is hurt
roughly twice as badly as the challenger, and its bias grows steadily with the
reject rate while the challenger's is flat.

### Which fix recovers the most?

Share of the A0 − A1 gap recovered, LightGBM. Blank where the gap is below
0.01 Gini and a ratio against it would be noise:

| arm | r=20 % | r=30 % | r=50 % |
|---|---|---|---|
| A2 hard cutoff | −0.30 | 0.37 | 0.42 |
| A3 fuzzy | 0.11 | 0.15 | 0.45 |
| A4 parcelling | −0.01 | −0.04 | −0.41 |
| A5 self-learning | −0.38 | 0.30 | **0.62** |
| A6 exploration | −4.42 | −3.51 | −2.30 |

Three things come out of this:

1. **Reject inference only helps when the selection is severe.** At a 20 % reject
   rate nothing beats simply discarding the rejects. By 50 % the best arm
   recovers about 62 % of the lost signal — and the gap is only 0.09 Gini to
   begin with.
2. **The exploration budget is catastrophic here, and the reason is variance,
   not cost.** The money is trivial: 450 explored loans at r=30 % cost 8.5 units
   of realised loss. But the inverse-propensity weight of 100/p gives a few
   hundred cases the same total weight as 45,000, and the result swamps the
   training set — A6 has the largest seed-to-seed spread of any arm (sd 0.11
   against 0.05 for parcelling). Correct weighting, ruinous variance.
3. **The scorecard has almost no gap to recover at all.** At 10 % and 20 %
   reject rates its accepts-only model is *better* than its oracle (0.438 vs
   0.419), so the recovery ratio is undefined and is reported as such. Binned
   logistic regression is markedly more robust to this selection than the GBM.

**Against Kozodoi et al. (2024):** this agrees. Reject inference gave modest and
inconsistent gains, and fixing the *evaluation* mattered more — here the
evaluation bias (7–15 Gini points, present at every reject rate) is larger and
far more consistent than anything any inference arm recovered.

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
| Champion scorecard (binning, selection, pdo_odds scaling) | `src/scorecard.py` | **done, run on full data** |
| Mondrian conformal refer band | `src/conformal.py` | done |
| Decision layer: cutoff, bands, reasons, recourse | `src/decision.py` | done |
| Isotonic calibration (shared) | `src/calibrate.py` | done |
| Challenger: LightGBM with monotone constraints | `src/challenger.py` | **done, run on full data** |
| Shared evaluation + the three figures | `src/evaluation.py` | **done, run on full data** |
| Synthetic fixtures matching the schema | `tests/fixtures.py` | done |
| Kaggle push/poll/pull orchestration | `scripts/kaggle_run.py` | **done, exercised against the live API** |
| `make-sample` + reconnaissance kernel | `kaggle/make-sample/` | **done, run on full data** |
| Environment probe kernel | `kaggle/env-probe/` | **done, run** |
| Data reconnaissance | [`docs/DATA_NOTES.md`](docs/DATA_NOTES.md) | **done, on full data** |
| Feature build (polars, 6 families) | `src/features.py` | **done, run on full data** |
| Feature screening + report | [`docs/feature_report.csv`](docs/feature_report.csv) | **done, full data** |
| Sampling-bias experiment (arms A0-A6) | `src/sampling_bias.py` | **done, run on full data** |

258 tests, all on generated fixtures. Two kernels have been run end to end on the real
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

**The IV screen is blind to a perfect leak.** optbinning collapses a
perfectly separating column into a single bin and reports **IV 0.0** — so the IV
rule reads a total leak as useless noise and drops it. The univariate-Gini screen
is what catches that case, which is why the brief specifies the two separately,
and the review annotation now survives an IV-based drop rather than being
overwritten by it.

**Never clip a denominator to avoid dividing by zero.** It trades an error for
something worse: a finite number of enormous magnitude that looks like data. On
the real sample, clipping the credit-limit denominator at 1e-9 produced a
utilisation feature topping **1.1 × 10¹³**, because 91% of cases report a zero
limit. It would have passed every screen. `safe_ratio` returns null instead, and
the binning process has a Missing bin for exactly that.

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
