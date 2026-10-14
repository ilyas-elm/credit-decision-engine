# Data notes

Everything here was measured on the **full competition data** by the
`make-sample` Kaggle kernel (`kaggle/make-sample/`), not inferred from the
documentation. The kernel writes `schema_report.json`; this file is the reading
of it. Re-running the kernel regenerates every number below.

Run: `ilyaselmaazouzi/hc-make-sample`, 2026-10-14.

## Base table

| | |
|---|---|
| rows | **1,526,659** |
| distinct `case_id` | 1,526,659 (one row per case) |
| columns | `case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target` |
| `WEEK_NUM` range | **0 – 91** (92 weeks) |
| `date_decision` range | 2019-01-01 – 2020-10-05 |
| overall default rate | **3.144 %** |

This matches the brief's expectation of ~1.5M rows and ~3% defaults.

## Tables

17 logical tables on the train side, from 68 parquet files. The `_0`, `_1`, …
suffixes on filenames are **part numbers**, not table identities:
`train_credit_bureau_a_2` arrives in 11 parts numbered past 10, so a trailing
digit is ambiguous with the depth suffix and has to be disambiguated.

| table | depth | full rows | sampled | cols | parts |
|---|---|---|---|---|---|
| `credit_bureau_a_2` | 2 | 188,298,452 | 5,744,322 | 19 | 11 |
| `credit_bureau_a_1` | 1 | 15,940,537 | 483,555 | 79 | 4 |
| `applprev_2` | 2 | 14,075,487 | 432,960 | 6 | 1 |
| `applprev_1` | 1 | 6,525,979 | 200,802 | 41 | 2 |
| `tax_registry_c_1` | 1 | 3,343,800 | 99,925 | 5 | 1 |
| `tax_registry_a_1` | 1 | 3,275,770 | 98,375 | 5 | 1 |
| `person_1` | 1 | 2,973,991 | 87,775 | 37 | 1 |
| `person_2` | 2 | 1,643,410 | 49,313 | 11 | 1 |
| `base` | — | 1,526,659 | 45,896 | 5 | 1 |
| `static_0` | 0 | 1,526,659 | 45,896 | **168** | 2 |
| `static_cb_0` | 0 | 1,500,476 | 45,081 | 53 | 1 |
| `credit_bureau_b_2` | 2 | 1,286,755 | 39,344 | 6 | 1 |
| `tax_registry_b_1` | 1 | 1,107,933 | 32,653 | 5 | 1 |
| `debitcard_1` | 1 | 157,302 | 4,689 | 6 | 1 |
| `deposit_1` | 1 | 145,086 | 4,357 | 5 | 1 |
| `credit_bureau_b_1` | 1 | 85,791 | 2,646 | 45 | 1 |
| `other_1` | 1 | 51,109 | 1,572 | 7 | 1 |
| **total** | | **243,465,196** | **7,419,161** | | 68 |

The brief's expected structure is **confirmed**: `static_0` (in parts),
`static_cb_0`, ten depth-1 tables, four depth-2 tables, with
`credit_bureau_a_2` by far the largest at 188M rows.

## The four columns later choices depend on

All four exist, so no planned work is cut.

| needed | exists | column(s) to use |
|---|---|---|
| credit amount | **yes** | `credamount_770A` (`static_0`); also `credamount_590A` (`applprev_1`) |
| interest rate | **yes** | `interestrate_508L`, `nominalrate_281L`, `annualeffectiverate_199L` (`credit_bureau_a_1`); `interesteffectiverate_369L` (`credit_bureau_b_1`) |
| age / birth date | **yes** | `birthdate_574D`, `dateofbirth_337D` (`static_cb_0`); `birth_259D`, `birthdate_87D` (`person_1`) |
| sex | **yes** | `sex_738L`, `gender_992L` (`person_1`) |

Consequences, per the brief:

- **Profit uses monetary exposure, not unit exposure.** `credamount_770A` is
  present on every case through `static_0`.
- **The fairness audit is in scope.** Both sex and age are available.

The search that produced these is a case-insensitive substring match and has
false positives, which are listed here so nobody later mistakes them for hits:
`last180dayaveragebalance_704A` matched "age" inside *average*,
`language1_981M` matched "age" inside *language*, and
`lastotherlnsexpense_631A` matched "sex" inside *expense*. The columns named in
the table above are the real ones.

## Two things that change how the pipeline must be written

**1. Dates are stored as strings.** `date_decision` is `String`, and so is every
`D` column (`birthdate_574D`, `recorddate_4527225D`, …). They are ISO-formatted,
so lexicographic comparison happens to order them correctly, but the feature
build must parse them before computing "days relative to `date_decision`".
Nothing can rely on them already being a date type.

**2. There is no income amount column.** The only direct match is
`incometype_1044T` in `person_1`, which is a *category*, not an amount. Income
for the affordability ratios (`annuity/income`, `credit/income`) has to be
derived from the tax registry tables, which hold payment amounts per case:

- `tax_registry_a_1.amount_4527230A` (with `name_4527232M`)
- `tax_registry_b_1.amount_4917619A` (with `deductiondate_4917603D`)
- `tax_registry_c_1.pmtamount_36A` (with `employername_160M`)

`tax_registry_c_1` carries an employer name alongside the payment, so it is the
closest thing to a salary feed. These are depth-1 tables, so income is an
*aggregate over records*, not a field — the affordability features in the brief
are still buildable but are a derived quantity with its own assumptions, and
should be labelled as such rather than presented as declared income.

## The local sample

Built by the same kernel: ~3 % of `case_id`s, stratified by (`WEEK_NUM`,
`target`) with at least one case per stratum so no week or class empties.

| | |
|---|---|
| case_ids | 45,896 |
| rows across all tables | 7,419,161 |
| size on disk | **0.036 GB** |
| sample default rate | 3.242 % (vs 3.144 % full) |
| seed | `20251013` (config) |

Comfortably under the 1 GB budget — the 3 % fraction could be raised if a
larger local sample ever helps. It lives in `data/sample/` and is gitignored;
competition data is never committed.

> **Caution on the sample.** Feature *screening* decisions do not belong here.
> IV is biased upward at small n — measured at 0.089 for pure noise at n=2,000
> against a 0.02 selection floor — so selection is a full-data decision and the
> sample exists to make the code run.

## Kaggle environment, as measured

Measured by `kaggle/env-probe/`, which prints the image's versions, pip-installs
the two packages the image lacks, and re-checks the stack afterwards. Re-run it
before changing any pin.

| package | Kaggle image | this repo |
|---|---|---|
| Python | 3.13.15 | 3.13.15 |
| polars | 1.35.2 | 1.35.2 |
| pyarrow | 23.0.1 | 23.0.1 |
| numpy | 2.1.3 | 2.1.3 |
| pandas | 2.3.3 | 2.3.3 |
| scikit-learn | 1.6.1 | 1.6.1 |
| lightgbm | 4.6.0 | 4.6.0 |
| matplotlib | 3.10.0 | 3.10.0 |
| optbinning | absent → 1.0.0 | 1.0.0 |
| empulse | absent → 0.13.0 | 0.13.0 |

This repo is pinned **to the image**, not to the newest releases. Full-data
results come from Kaggle kernels, so the environment that matters is theirs, and
local tests are only evidence about those runs if the stack is the same one.

The probe confirms `pip install optbinning empulse` on the image is clean:
return code 0, and numpy, pandas and scikit-learn are byte-identical before and
after, so kernels run the image's versions whatever this repo pins. (pip reports
pre-existing conflicts in the image around protobuf, gym and bigframes; they are
unrelated to this project and present before the install.) A smoke test after
the install exercises optbinning, empulse, LightGBM and polars together.

Matching the image immediately paid for itself: the challenger was calling
LightGBM's `eval_X`/`eval_y`, which exist only in 4.7. On the image's 4.6.0 that
raises `TypeError`, so every model kernel would have failed. It was written that
way to silence a 4.7 deprecation warning — a change that was correct against the
newest release and wrong against the environment the results come from.

**Competition data does not mount where the documentation implies.** It appears
at `/kaggle/input/competitions/home-credit-credit-risk-model-stability/`, not
`/kaggle/input/<competition-slug>/`. The first run found zero tables because of
it. `make-sample` now locates the data by searching for `parquet_files/train`
and prints a listing of `/kaggle/input` when it cannot, so a wrong assumption
costs one run and reports the answer rather than needing a guess.
