# MODEL.md, credit decision engine on Home Credit data

Every result here was measured on the full 1.5M-case dataset by a committed
Kaggle kernel, except the collinearity and sign-flip figures (computed locally
from those kernels' outputs), the post-decision figures in §3 (the 3 % sample)
and the test count in §8 (synthetic fixtures). Unflattering or unresolved
results are stated as such.

## 1. Purpose and decision

The system decides consumer credit applications, returning for each a
probability of default, a score, a band (approve / refer / decline), the
principal reasons for a decline and, separately, what would reverse it.

Two models are maintained:

- **Champion:** a monotone, binned, 13-characteristic logistic-regression
  scorecard on a `pdo_odds` points scale (600 points at 50:1 odds, PDO 20).
- **Challenger:** LightGBM under monotone constraints inherited from the
  champion's binning, so it cannot exploit relationships the scorecard was
  forbidden.

The **decision layer runs entirely on the champion**: banding with the
challenger while explaining with the scorecard would have the reasons justify
a decision another model made, a governance problem for any lender.

## 2. Data and target

| | |
|---|---|
| source | Kaggle *Home Credit, Credit Risk Model Stability* |
| rows | 1,526,659, one per `case_id` |
| period | 2019-01-01 to 2020-10-05, `WEEK_NUM` 0-91 |
| target | 1 = default, 3.144 % overall |
| tables | 17, 243M source rows (`credit_bureau_a_2` alone is 188M) |

**Splits are out-of-time, never random**, as contiguous blocks of whole weeks,
each with exactly one job:

| block | weeks | job |
|---|---|---|
| train | 0-54 | fitting, binning, feature screening |
| val_a | 55-62 | early stopping and probability calibration |
| val_b | 63-68 | **conformal calibration only** |
| test | 69-91 | evaluation; fitted on by nothing |

`val_b` stays separate because conformal coverage only means something on data
nothing else saw. Tests assert that no `case_id` crosses splits and that the
blocks are ordered in time.

## 3. Design choices

**Features.** 36 built in six families (delinquency, utilisation, inquiry
velocity, previous applications, affordability, bureau depth); 25 survive
screening; 13 enter the scorecard. Delinquency dominates, made possible by
`credit_bureau_a_2` being a *monthly payment history*. The strongest single
feature is `delq_dpd_mean` (IV 0.326).

**Screening is fitted on the train weeks only**, missing > 95 % drops, PSI
train-vs-validation > 0.25 drops, and univariate Gini > 0.5 is flagged for
human review, never dropped automatically; test weeks are never read for
selection.

**Post-decision records are dropped.** On the 3 % sample, 10.7 % of
`credit_bureau_a_2` rows with a DPD value postdate their own `date_decision`;
the maximum DPD over those rows scores a univariate Gini of +0.19, partly the
outcome itself. No stage records the full-data share;
`scripts/sample_recon.py` recomputes the sample's.

**No class rebalancing anywhere**: it would distort the probabilities the
profit cutoff depends on.

**Environment is pinned to the Kaggle image** (Python 3.13.15, polars 1.35.2,
pandas 2.3.3, LightGBM 4.6.0), so local tests say something about the kernel
runs. It caught a real defect: the challenger used LightGBM 4.7's
`eval_X`/`eval_y`, which would have failed every model kernel.

**Sign flips: found, measured and removed.** Three characteristics scored
*more* points as their event rate rose ("the more you previously borrowed, the
safer you are"), from joint collinearity across the previous-applications
family. They are barred in config at a cost of **0.0012 Gini and 0.0010
stability**, 16 characteristics down to 13; `FittedScorecard.sign_flips()`
guards against regression, and the README records the diagnosis.

## 4. Out-of-time performance

Test weeks 69-91, 225,514 cases, fitted on by nothing.

| | scorecard | LightGBM |
|---|---|---|
| Gini | 0.3944 | **0.4219** |
| mean weekly Gini | 0.3854 | **0.4162** |
| Gini slope / week | +0.0026 | +0.0020 |
| Gini residual sd | **0.0521** | 0.0595 |
| **stability metric** | 0.3594 | **0.3864** |
| Brier | 0.02131 | **0.02119** |
| worst-week calibration gap | 0.0095 | **0.0082** |
| PSI of predicted PD vs train | 0.5028 | **0.4365** |

The challenger adds about **2.7 Gini points** and keeps them under the
stability metric. Neither model decays, so its 88× falling-slope penalty never
engages and the gap is level, not trend; the champion wins only on
week-to-week smoothness.

The last three rows are won narrowly. The challenger bags by row position, so
row order alone moved it 0.00446 Gini, 0.0000188 Brier and 0.00020 of
worst-week gap between two runs on identical features. The margins are 6.2x,
6.8x and 6.6x that: safe for Gini and stability, a few times the noise for
calibration.

**Profit cannot separate them.** A loan is profitable while `p(default) < roi /
(roi + E[loss])`. At the literature's assumptions that break-even is **49.0 %**
against a book defaulting at **3.1 %**, so approving everyone is optimal, EMPCS
is ≈0 for both, and only 8 of 21 points in the widened grid make a cutoff
bind. **On this portfolio the model's value is not in
choosing an approval cutoff.**

## 5. Calibration and stability

The champion required recalibration (val_a gap -0.013) and received isotonic
regression; the challenger is calibrated unconditionally. Worst-week
calibration gaps are small (≤ 0.009) and the weekly default rate ranges 1.77
%-2.79 %.

**The referral band does not keep its promise.** Mondrian class-conditional
conformal at α = 0.10, calibrated on val_b:

| | coverage | target |
|---|---|---|
| non-default | 0.959 | 0.90 ✓ |
| **default** | **0.747** | 0.90 ✗, below target in **23 of 23 weeks** |

Conformal guarantees coverage *under exchangeability*; these weeks are not
exchangeable with val_b. It is still worth having: a marginal threshold covers
the default class **0.000** of the time, and Mondrian moves it to 75 %, though
the guarantee does not hold.

At the tightest binding cutoff (56.3 % acceptance) the bands separate risk
monotonically: approve 51.3 % at 1.1 % bad, refer 44.4 % at 2.9 %, decline 4.3
% at 7.8 %. A **44 % referral rate is beyond any real review desk**; it follows from
α = 0.10 on a 3 % default book, and α is a business decision. The band stops
paying for itself between **0.01 and 0.02 per review**, even with a perfect
reviewer.

## 6. Limitations

**Booked loans only.** The real rejects are absent and nothing recovers them;
the sampling-bias experiment simulates a *second* selection on an
already-selected population, so its recoveries are an upper bound.

**Sampling bias is large; reject inference helps only under severe selection.**
A lender measuring on its own approved book **understates its Gini by
0.08-0.14** at every reject rate, through range restriction rather than any
modelling fault. Reject inference matters less: at a 10 % reject rate the gap
is 0.010 Gini and nothing helps; at 50 % it is 0.108 and two arms recover most
of it. Best on Gini is **A2**, labelling every reject bad, at a Brier 6.6x the
oracle's, which **by this document's own standard disqualifies it**; A3
recovers less and stays calibrated. An exploration budget is net negative as
training data at every setting, the GBM's damage 3-46x the cost and negative
in all 27 of its fits. The direction agrees with Kozodoi et al. (2024,
arXiv:2407.13009): modest potential, and evaluation matters more.

**EMP assumptions are assumptions**, and at the literature's values the profit
analysis is degenerate (§4). Every profit number is reported across a grid.

**Drift is real and unaddressed.** PD PSI is 0.44-0.50 train-to-test, above
any conventional alert threshold, and it directly causes the conformal
coverage failure.

**Fairness.** Women are approved less often (48.5 % vs 56.4 %) while defaulting
less often (1.76 % vs 2.99 %); the model over-predicts their risk (+0.0102)
and under-predicts men's (-0.0052), miscalibration that runs against the
lower-risk group. **Audit only, no mitigation**, as scoped.

**By age the disparity is larger, and it is a drift artefact.** The 60s are
approved at 25.4 % and the 70+ at 11.8 %, against 61-67 % for everyone
younger, while defaulting less. Age is not an input, but 96.3 % of the
over-60s lack a tax-registry income record against 8.2 % of the younger, a
93.2 %-accurate proxy, and the income characteristics' Missing bin is frozen
at fit time while everyone with a value drifted into better bins. A static
Missing bin becomes a relative penalty whenever the population moves, so §7
monitors it; the durable fix is a pension and benefit income source. The
README derives it.

## 7. Monitoring plan

Thresholds come from the variation observed over the 23 test weeks, not from
convention.

| signal | observed | investigate | escalate |
|---|---|---|---|
| **weekly Gini** (challenger) | mean 0.416, sd 0.061, min 0.301 | any week < 0.29 (mean - 2 sd) | 2 weeks running < 0.29, or 4-week mean < 0.35 |
| **Gini trend** | slope +0.0020 /wk | 8-week slope < 0 | 8-week slope < -0.003 |
| **calibration in the large** | worst week 0.008, range -0.003 to +0.009 | any week \|gap\| > 0.010 | 2 weeks running \|gap\| > 0.015 |
| **PSI of predicted PD vs train** | median 0.42, range 0.30-0.71 | weekly PSI > 0.60 | weekly > 0.75, or 4-week median > 0.60 |
| **conformal coverage, default class** | 0.747, target 0.90 | any week < 0.70 | 4-week mean < 0.65 |
| **referral rate** | 44.4 % | outside 35-55 % | outside 30-60 % |
| **approved bad rate** | 1.1 % | > 1.5 % over 4 weeks | > 2.0 % over 4 weeks |
| **Missing-bin drift**, worst affordability characteristic | +16.5 pts, against +1.3 at fit | any week > +20 | 4-week mean > +25 |

PSI thresholds sit **above** the conventional 0.25, which this model already
exceeds against its own training period; a 0.25 alarm would fire permanently.
Coverage is set below target because the band already misses it. The
Missing-bin row is live non-missing mean points minus the Missing bin's fixed
points (`scripts/drift_recon.py`) and triggers a refit, since only a refit
moves a bin's points; recalibration follows calibration in the large, not
Gini, as ranking has proved more durable than levels.

**A caveat on the calibration trigger.** In production it is computed on
approved loans only, where it is flattered: at a 50 % rejection rate the model
under-predicts by 0.0049 across the population while its own book shows
0.0012, four times smaller, and at 20 % and 30 % the book has the opposite
sign. Every figure stays inside the 0.010 threshold, so on this data the
trigger is distorted rather than defeated, and the **approved bad rate** row,
defined on the observable population, is immune. A random-approval sample
would restore the measurement: for an assumed population gap of 0.020 it takes
245 explored loans at a 30 % rejection rate and 541 at 50 %, for 5 to 9 units
of realised loss: an illustration, not a measured result, as no gap here needs
it.

## 8. Reproducing

`uv sync && uv run pytest`: 492 tests, of which 488 run on synthetic fixtures
alone and four skip without the local sample. No competition data is
committed; full-data results come from re-running the `kaggle/` kernels.
