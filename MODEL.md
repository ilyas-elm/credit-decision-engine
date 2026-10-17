# MODEL.md, credit decision engine on Home Credit data

Every result here was measured on the full 1.5M-case dataset by a committed
Kaggle kernel, except where the line says otherwise: the collinearity and
sign-flip figures were computed locally from those kernels' outputs, the
post-decision figures in §3 come from the 3 % sample, and the test count in §8
from synthetic fixtures.

Where a result is unflattering or unresolved it is stated as such. A model
document that records only successes is not usable for review.

---

## 1. Purpose and decision

The system decides consumer credit applications. For each applicant it returns
a probability of default, a scorecard score, a band (approve / refer /
decline), the principal reasons for a decline, and, separately, what the
applicant could change to reverse it.

Two models are maintained:

- **Champion:** a monotone, binned, 13-characteristic logistic-regression
  scorecard on a `pdo_odds` points scale (600 points at 50:1 odds, PDO 20). It
  is the model whose decisions are explained.
- **Challenger:** LightGBM under monotone constraints inherited from the
  champion's binning, so it cannot exploit relationships the scorecard was
  forbidden.

The **decision layer runs entirely on the champion**. Calibrating the referral
band on the challenger while explaining decisions with the scorecard would have
the reasons justify a decision a different model made: incoherent, and for a
lender a governance problem.

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

`val_b` is kept separate because split-conformal coverage is only meaningful on
data the model and its calibrator never saw. It is asserted that no `case_id`
appears in two splits and that the blocks are strictly ordered in time.

## 3. Design choices

**Features.** 36 built in six families (delinquency, utilisation, inquiry
velocity, previous applications, affordability, bureau depth); 25 survive
screening; 13 enter the scorecard. Delinquency dominates, and only exists
because `credit_bureau_a_2` proved to be a *monthly payment history*, which is
what makes recency-weighted DPD computable. The strongest single feature is
`delq_dpd_mean` (IV 0.326).

**Screening is fitted on the train weeks only**, missing > 95 % drops, PSI
train-vs-validation > 0.25 drops, and univariate Gini > 0.5 is flagged for
human review, never dropped automatically. The test weeks are never read for
any selection decision.

**Post-decision records are dropped.** `credit_bureau_a_2` is a monthly payment
history and a large minority of its rows postdate the application they belong
to. On the 3 % sample, 10.7 % of rows carrying a DPD value are dated after
their own `date_decision`, and the maximum DPD over those rows scores a
univariate Gini of +0.19 against the target: it is partly the outcome being
predicted. No stage records the full-data share; `scripts/sample_recon.py`
recomputes the sample's.

**No class rebalancing anywhere.** `is_unbalance` and `scale_pos_weight` are
set explicitly to neutral. Rebalancing distorts the probabilities the profit
cutoff depends on, trading the thing being measured for a metric nobody
reports.

**Environment is pinned to the Kaggle image**, Python 3.13.15, polars 1.35.2,
pandas 2.3.3, LightGBM 4.6.0, because full-data results come from Kaggle
kernels, so local tests are evidence about those runs only if the stack
matches. This caught a real defect: the challenger used LightGBM's
`eval_X`/`eval_y`, which exist only in 4.7, so every model kernel would have
failed on the image.

**Sign flips: found, measured and removed.** Three characteristics scored
*more* points as their event rate rose, so the table read "the more you
previously borrowed, the safer you are". The cause was joint collinearity
across the previous-applications family rather than any one column, so they are
barred in config, at a cost measured locally on the full-data features of
**0.0012 Gini and 0.0010 stability**, about 0.3 % relative, 16 characteristics
down to 13. The scorecard now reads correctly end to end, which is the
champion's whole purpose, and `FittedScorecard.sign_flips()` guards against
regression. The README records the diagnosis.

## 4. Out-of-time performance

Test weeks 69-91, 225,514 cases, fitted on by nothing.

| | scorecard | LightGBM |
|---|---|---|
| Gini | 0.3944 | **0.4264** |
| mean weekly Gini | 0.3854 | **0.4205** |
| Gini slope / week | +0.0026 | +0.0019 |
| Gini residual sd | **0.0521** | 0.0592 |
| **stability metric** | 0.3594 | **0.3909** |
| Brier | 0.02131 | **0.02117** |
| worst-week calibration gap | 0.0095 | **0.0084** |
| PSI of predicted PD vs train | 0.5028 | **0.4202** |

The challenger adds about **3 Gini points** and keeps them once stability is
measured. Note *why*: neither model decays, both slopes are positive, so the
stability metric's 88× falling-slope penalty never engages and the gap is
level, not trend. The champion wins only on week-to-week smoothness.

**Profit cannot separate them.** A loan is profitable while `p(default) < roi /
(roi + E[loss])`. At the literature's assumptions that break-even is **49.0 %**
against a book defaulting at **3.1 %**, so approving everyone is optimal, EMPCS
is ≈0 for both, and only 8 of 21 assumption points in the widened sensitivity
grid make a cutoff bind at all. **On this portfolio the model's value is not in
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
exchangeable with val_b. It is still worth having, a marginal threshold covers
the default class **0.000** of the time, so Mondrian moves it from 0 % to 75 %,
but the guarantee should not be quoted as holding.

At the tightest binding cutoff (71.5 % acceptance) the bands separate risk
monotonically: approve 51.3 % at 1.1 % bad, refer 44.4 % at 2.9 %, decline 4.3
% at 7.8 %. A **44 % referral rate is beyond any real review desk** and follows
directly from α = 0.10 on a 3 % default book; α is the lever, and moving it is
a business decision. The band stops paying for itself between **0.01 and 0.02
per review**, against a *perfect-reviewer* upper bound.

## 6. Limitations

**Booked loans only.** The data contains applications Home Credit approved. The
real rejects are absent and nothing here recovers them. The sampling-bias
experiment simulates a *second* selection on an already-selected population, so
its recoveries are bounded by that.

**Sampling bias is large; reject inference helps only under severe selection.**
A lender measuring on its own approved book **understates its Gini by
0.05-0.16**, consistently at every reject rate, through range restriction
rather than any modelling fault. Reject inference is a different story: at a 10
% reject rate almost nothing helps and the gap is only 0.011 Gini, while at 50
% the gap is 0.135 and three arms recover most of it. The best performer is
**A2, the crudest arm**, label every reject bad with no model and no weighting,
which recovers 97 % at a 30 % reject rate and is not beaten by any more
elaborate method. A random-exploration budget was harmful everywhere, and on
variance rather than price: net is negative at every budget and the damage runs
2-53x the cost.

This agrees in direction with Kozodoi et al. (2024, arXiv:2407.13009; EJOR
324(2), 2025): reject inference has modest potential, fixing evaluation matters
more, and gains rise as acceptance falls. A5 reproduces that direction but
turns positive far earlier than their 20 % threshold, because the selection
here is simulated on an already-booked population. The hidden labels belong to
applicants the original lender approved, which is the easier problem, so **the
recoveries reported here are an upper bound**. The README gives the comparison.

**EMP assumptions are assumptions**, and at the literature's values the profit
analysis is degenerate (§4). Every profit number is reported across a grid.

**Drift is real and unaddressed.** The PD PSI is 0.42-0.50 train-to-test, above
any conventional alert threshold, and it is the direct cause of the conformal
coverage failure.

**Fairness.** Women are approved less often (48.5 % vs 56.4 %) while defaulting
less often (1.76 % vs 2.99 %), and the model over-predicts their risk (+0.0102)
while under-predicting men's (-0.0052). This is miscalibration by group, not a
threshold artefact, and it runs against the lower-risk group. **Audit only, no
mitigation**, as scoped.

**By age the disparity is larger.** Applicants in their 60s are approved at
25.4 % and those 70+ at 11.8 %, against 61-67 % for everyone younger, while
being the safest groups in the book (1.2-1.3 % default against 4.5 % for the
20s). The model predicts them as the riskiest, mean PD 3.7-3.8 % against a
realised 1.2 %, so a roughly threefold over-prediction becomes an approval gap
at the cutoff. Age is excluded from recourse, so those applicants are told why
and offered nothing to do about it.

**The mechanism is a proxy, and it is drift.** Age is not an input. The 60+
cohort gives up 19.6 scorecard points net, 21.1 of them on the two
income-derived characteristics. Income comes from the tax registry, which
records employment payments, so 96.3 % of the over-60s have no income record
against 8.2 % of everyone younger: the absence is a 93.2 %-accurate test for
being over 60. The scorecard does not treat it as risky. Its Missing bin scores
the neutral 45.1 points, matching a 3.05 % event rate against the train weeks'
3.13 %, and the absence is not riskier anywhere in the data. The gap is that
the bin is frozen at fit time while everyone else moved: applicants with an
income record averaged 46.4 points on `afford_annuity_to_taxincome` at fit time
and 61.7 by the test weeks. The over-60s were not marked down; the rest drifted
past them. The README works through it.

That failure generalises: a static Missing bin becomes a relative penalty
whenever the scored population moves, which belongs in the monitoring plan
below rather than in a threshold constraint. The durable fix is a pension and
benefit income source, so retirees leave the Missing bin altogether.

An earlier cause of the same disparity was a bug in the income column, fixed
before these numbers were produced and recorded in the README. Fixing it lifted
the champion from Gini 0.376 to 0.396 and stability 0.340 to 0.360, left the
challenger unmoved, and raised approval for the 70+ group from 8.6 % to 11.8 %.
Attributed and partly reduced, not removed.

## 7. Monitoring plan

Thresholds are set from the variation actually observed over the 23 test weeks,
not from convention.

| signal | observed | investigate | escalate |
|---|---|---|---|
| **weekly Gini** (challenger) | mean 0.421, sd 0.059, min ~0.31 | any week < 0.30 (mean - 2 sd) | 2 consecutive weeks < 0.30, or 4-week mean < 0.35 |
| **Gini trend** | slope +0.0017 /wk | 8-week slope < 0 | 8-week slope < -0.003 (the metric penalises this 88×) |
| **calibration in the large** | worst week 0.008, range -0.003 to +0.009 | any week \|gap\| > 0.010 | 2 consecutive weeks \|gap\| > 0.015 |
| **PSI of predicted PD vs train** | median 0.42, range 0.30-0.71 | weekly PSI > 0.60 | weekly PSI > 0.75, or 4-week median > 0.60 |
| **conformal coverage, default class** | 0.747, target 0.90 | any week < 0.70 | 4-week mean < 0.65 |
| **referral rate** | 44.4 % | outside 35-55 % | outside 30-60 % |
| **approved bad rate** | 1.1 % | > 1.5 % over 4 weeks | > 2.0 % over 4 weeks |
| **Missing-bin drift**, worst affordability characteristic | +16.5 pts, range +13.9 to +17.3, against +1.3 at fit | any week > +20 | 4-week mean > +25, or a refit that moves it by > 10 |

Two notes on these. The PSI thresholds are deliberately set **above** the
conventional 0.25: this model already exceeds that against its own training
period, so a 0.25 alarm would fire permanently and be ignored. Using it would
be worse than useless. And the default-class coverage threshold is set below
its nominal target because the band already misses it, the monitor's job is to
catch further deterioration, not to re-report a known shortfall every week.

The last row is there because of what the fairness audit turned up in §6. A
Missing bin's points are fixed when the scorecard is fitted, so as the
population that *has* a value drifts into better bins, the cohort that does not
is marked down relative to everyone else without anything about it changing. On
`afford_annuity_to_taxincome` that gap has gone from +1.3 points at fit time to
+16.5, which is most of the 19.6 points the over-60s give up. It is measured as
the live non-missing population's mean points minus the Missing bin's fixed
points, per characteristic, and `scripts/drift_recon.py` computes it. Its
thresholds sit above the current level for the same reason the PSI ones do.

Recalibration should be triggered by the calibration-in-the-large signal rather
than by Gini: the model's ranking is more durable than its levels, which is
what the drift evidence shows. This signal is different, and recalibration will
not fix it: only a refit moves a bin's points, so a sustained Missing-bin gap
is a refit trigger rather than a recalibration one.

## 8. Reproducing

`uv sync && uv run pytest`, 277 tests on synthetic fixtures; no competition
data is committed and none is needed. Full-data results come from re-running
the kernels in `kaggle/` from committed code: `make-sample`, `env-probe`,
`features`, `models`, `sampling-bias`, `decision`.
