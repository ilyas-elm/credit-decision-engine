# MODEL.md — credit decision engine on Home Credit data

Every number here was measured on the full data by a committed Kaggle kernel.
Nothing is illustrative. Where a result is unflattering or unresolved it is
stated as such, because a model document that only records successes is not
usable for review.

---

## 1. Purpose and decision

The system decides consumer credit applications. For each applicant it returns a
probability of default, a scorecard score, a band (**approve / refer / decline**),
the principal reasons for a decline, and — separately — what the applicant could
change to reverse it.

Two models are maintained:

- **Champion:** a monotone, binned, 13-characteristic logistic-regression
  scorecard on a `pdo_odds` points scale (600 points at 50:1 odds, PDO 20). It
  is the model whose decisions are explained.
- **Challenger:** LightGBM under monotone constraints inherited from the
  champion's binning, so it cannot exploit relationships the scorecard was
  forbidden.

The **decision layer runs entirely on the champion**. Calibrating the referral
band on the challenger while explaining decisions with the scorecard would mean
the reasons justify a decision a different model made — incoherent, and a
governance problem for a lender.

## 2. Data and target

| | |
|---|---|
| source | Kaggle *Home Credit — Credit Risk Model Stability* |
| rows | 1,526,659, one per `case_id` |
| period | 2019-01-01 to 2020-10-05, `WEEK_NUM` 0–91 |
| target | 1 = default, 3.144 % overall |
| tables | 17, 243M source rows (`credit_bureau_a_2` alone is 188M) |

**Splits are out-of-time, never random**, as contiguous blocks of whole weeks,
each with exactly one job:

| block | weeks | job |
|---|---|---|
| train | 0–54 | fitting, binning, feature screening |
| val_a | 55–62 | early stopping and probability calibration |
| val_b | 63–68 | **conformal calibration only** |
| test | 69–91 | evaluation; fitted on by nothing |

`val_b` is kept separate because split-conformal coverage is only meaningful on
data the model and its calibrator never saw. It is asserted that no `case_id`
appears in two splits and that the blocks are strictly ordered in time.

## 3. Design choices

**Features.** 36 built in six families (delinquency, utilisation, inquiry
velocity, previous applications, affordability, bureau depth); 25 survive
screening; 13 enter the scorecard. Delinquency dominates — and only exists
because `credit_bureau_a_2` proved to be a *monthly payment history*, which is
what makes recency-weighted DPD computable. The strongest single feature is
`delq_dpd_mean` (IV 0.326).

**Screening is fitted on the train weeks only** — missing > 95 % drops, PSI
train-vs-validation > 0.25 drops, and univariate Gini > 0.5 is flagged for human
review, never dropped automatically. The test weeks are never read for any
selection decision.

**Post-decision records are dropped.** 8.4 % of the bureau payment rows are
dated *after* their own `date_decision`, and the maximum post-decision DPD has a
univariate Gini of +0.097 against the target: it is partly the outcome being
predicted.

**No class rebalancing anywhere.** `is_unbalance` and `scale_pos_weight` are set
explicitly to neutral. Rebalancing distorts the probabilities the profit cutoff
depends on, trading the thing being measured for a metric nobody reports.

**Environment is pinned to the Kaggle image** — Python 3.13.15, polars 1.35.2,
pandas 2.3.3, LightGBM 4.6.0 — because full-data results come from Kaggle
kernels, so local tests are evidence about those runs only if the stack matches.
This caught a real defect: the challenger used LightGBM's `eval_X`/`eval_y`,
which exist only in 4.7, so every model kernel would have failed on the image.

## 4. Out-of-time performance

Test weeks 69–91, 225,514 cases, fitted on by nothing.

| | scorecard | LightGBM |
|---|---|---|
| Gini | 0.3944 | **0.4264** |
| mean weekly Gini | 0.3854 | **0.4205** |
| Gini slope / week | +0.0026 | +0.0019 |
| Gini residual sd | **0.0521** | 0.0592 |
| **stability metric** | 0.3594 | **0.3909** |
| Brier | 0.02131 | **0.02117** |
| worst-week calibration gap | 0.0095 | **0.0084** |
| score PSI vs train | 0.5028 | **0.4202** |

The challenger adds about **3 Gini points** and keeps them once stability is
measured. Note *why*: neither model decays — both slopes are positive — so the
stability metric's 88× falling-slope penalty never engages and the gap is level,
not trend. The champion wins only on week-to-week smoothness.

**Profit cannot separate them.** A loan is profitable while
`p(default) < roi / (roi + E[loss])`. At the literature's assumptions that
break-even is **49.0 %** against a book defaulting at **3.1 %**, so approving
everyone is optimal, EMPCS is ≈0 for both, and only 7 of 21 assumption points in
the widened sensitivity grid make a cutoff bind at all. **On this portfolio the
model's value is not in choosing an approval cutoff.**

## 5. Calibration and stability

The champion required recalibration (val_a gap −0.013) and received isotonic
regression; the challenger is calibrated unconditionally. Worst-week calibration
gaps are small (≤ 0.009) and the weekly default rate ranges 1.77 %–2.79 %.

**The referral band does not keep its promise.** Mondrian class-conditional
conformal at α = 0.10, calibrated on val_b:

| | coverage | target |
|---|---|---|
| non-default | 0.960 | 0.90 ✓ |
| **default** | **0.747** | 0.90 ✗ — below target in **23 of 23 weeks** |

Conformal guarantees coverage *under exchangeability*; these weeks are not
exchangeable with val_b. It is still worth having — a **marginal** threshold
covers the default class **0.000** of the time, so Mondrian moves it from 0 % to
75 % — but the guarantee should not be quoted as holding.

At the tightest binding cutoff (71.5 % acceptance) the bands separate risk
monotonically: approve 50.6 % at 1.1 % bad, refer 45.3 % at 2.9 %, decline 4.1 %
at 7.8 %. A **45 % referral rate is beyond any real review desk** and follows
directly from α = 0.10 on a 3 % default book; α is the lever, and moving it is a
business decision. The band stops paying for itself between **0.01 and 0.02 per
review**, against a *perfect-reviewer* upper bound.

## 6. Limitations

**Booked loans only.** The data contains applications Home Credit approved. The
real rejects are absent and nothing here recovers them. The sampling-bias
experiment simulates a *second* selection on an already-selected population, so
its recoveries are bounded by that.

**Sampling bias is large and reject inference barely helps.** A lender measuring
on its own approved book **understates its Gini by 0.07–0.15**. Reject inference
only helps under severe selection — at a 20 % reject rate nothing beats simply
discarding the rejects; at 50 % the best arm recovers 62 % of a 0.09 gap. A
random-exploration budget was actively harmful (−2.3 to −4.4), and the reason is
variance, not cost: 450 explored loans cost 8.5 units of realised loss, but the
100/p inverse-propensity weight gives a few hundred cases the same total weight
as 45,000. This agrees with Kozodoi et al. (2024): fixing the evaluation matters
more than inferring the labels.

**EMP assumptions are assumptions**, and at the literature's values the profit
analysis is degenerate (§4). Every profit number is reported across a grid.

**Drift is real and unaddressed.** Score PSI is 0.42–0.50 train-to-test, above
any conventional alert threshold, and it is the direct cause of the conformal
coverage failure.

**Sign flips: found, measured and removed.** Three characteristics scored *more*
points as their event rate rose, because their multivariate coefficients came out
negative — the table read "the more you previously borrowed, the safer you are".
The cause was joint collinearity in the previous-applications family rather than
any one column: pairwise pruning at 0.80 still left a cluster correlated
0.60–0.78 (condition number 15.1), so excluding one flip merely moved it to the
next. They are now barred in config, at a **measured cost of 0.0012 Gini and
0.0010 stability** — about 0.3 % relative, 16 characteristics down to 13. The
scorecard now reads correctly end to end, which is the champion's whole purpose.
`FittedScorecard.sign_flips()` guards against regression.

**Fairness.** Women are approved less often (47.6 % vs 56.1 %) while defaulting
less often (1.76 % vs 2.99 %), and the model over-predicts their risk (+0.0097)
while under-predicting men's (−0.0061). This is miscalibration by group, not a
threshold artefact, and it runs against the lower-risk group. **Audit only, no
mitigation**, as scoped.

**By age the disparity is larger.** Applicants in their 60s are approved at
25.4 % and those 70+ at 11.8 %, against 61–68 % for everyone younger, while being
the safest groups in the book (1.2–1.3 % default against 4.5 % for the 20s). The
model predicts them as the riskiest — mean PD 3.7–3.8 % against a realised
1.2 % — so a roughly threefold over-prediction becomes an approval gap at the
cutoff. Age is excluded from recourse, so those applicants are told why and
offered nothing to do about it.

**The mechanism is a proxy, and it is attributable.** Age is not an input. Of
the 22 scorecard points the 60+ cohort gives up, 24 come from the two
income-derived characteristics. Income is derived from the tax registry, which
records employment payments, so 92 % of applicants in their 60s and 100 % of
those 70+ have none — making "no income record" a 94 %-accurate proxy for being
over 60. Within that group the absence carries no risk signal whatsoever (1.78 %
default against 1.86 %), while for working-age applicants it is mildly
predictive (4.31 % against 3.47 %). The model learned the second relationship
and applied it to a population where it is meaningless. The remedy is in the
feature build — a pension and benefit income source, or restricting the proxy to
the population it describes — not a constraint at the threshold. One cause WAS a bug and was fixed as one: afford_taxincome_total recorded a case
absent from every tax registry as earning nothing rather than unknown, putting
34 % of the book in the lowest income bin beside genuinely low earners. Fixing
it lifted the champion (Gini 0.376 to 0.396, stability 0.340 to 0.360) while
leaving the challenger unmoved, and raised approval for the 70+ group from 8.6 %
to 11.8 %. The disparity survives: the Missing bin still pools retirees with
working-age applicants who have no employment record, for whom that absence
genuinely is risky. The remaining fix needs a pension and benefit income source,
which this dataset does not contain. Attributed and partly reduced, not
removed.

## 7. Monitoring plan

Thresholds are set from the variation actually observed over the 23 test weeks,
not from convention.

| signal | observed | investigate | escalate |
|---|---|---|---|
| **weekly Gini** (challenger) | mean 0.421, sd 0.059, min ~0.31 | any week < 0.30 (mean − 2 sd) | 2 consecutive weeks < 0.30, or 4-week mean < 0.35 |
| **Gini trend** | slope +0.0017 /wk | 8-week slope < 0 | 8-week slope < −0.003 (the metric penalises this 88×) |
| **calibration in the large** | worst week 0.008, range −0.003 to +0.009 | any week \|gap\| > 0.010 | 2 consecutive weeks \|gap\| > 0.015 |
| **score PSI vs train** | median 0.42, range 0.30–0.71 | weekly PSI > 0.60 | weekly PSI > 0.75, or 4-week median > 0.60 |
| **conformal coverage, default class** | 0.747, target 0.90 | any week < 0.70 | 4-week mean < 0.65 |
| **referral rate** | 45.3 % | outside 35–55 % | outside 30–60 % |
| **approved bad rate** | 1.1 % | > 1.5 % over 4 weeks | > 2.0 % over 4 weeks |

Two notes on these. The PSI thresholds are deliberately set **above** the
conventional 0.25: this model already exceeds that against its own training
period, so a 0.25 alarm would fire permanently and be ignored. Using it would be
worse than useless. And the default-class coverage threshold is set below its
nominal target because the band **already misses it** — the monitor's job is to
catch further deterioration, not to re-report a known shortfall every week.

Recalibration should be triggered by the calibration-in-the-large signal rather
than by Gini: the model's ranking is more durable than its levels, which is what
the drift evidence shows.

## 8. Reproducing

`uv sync && uv run pytest` — 267 tests on synthetic fixtures; no competition data
is committed and none is needed. Full-data results come from re-running the
kernels in `kaggle/` from committed code: `make-sample`, `env-probe`, `features`,
`models`, `sampling-bias`, `decision`.
