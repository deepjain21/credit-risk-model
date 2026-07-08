# Decision Log

The most important deliverable. For every non-trivial choice: **what** was
decided, the **alternatives** considered, and **why**. Ordered by the lifecycle.
The parts I am least confident in are flagged **⚠ would revisit**.

---

## 1. Target definition

**Decision.** `default = 1` if `final_loan_status ∈ {Written-Off, Defaulted}`
**or** `max_dpd_ever ≥ 90`; `default = 0` if `Closed-Good`. `Active` loans are
**excluded** from the labelled set.

**Resulting label:** 5,313 labelled loans, **16.5% default rate**
(878 status-bad + additional 90+ DPD loans that cured but count as default).

**Alternatives considered**
- *repayments max-dpd* : Have calculated max-dpd from the repayments which comes around 647, all cases are already covered under the loan_servicing default/written-off label.
- *Written-Off / Defaulted only (ignore DPD).* Rejected: misses loans that were
  seriously delinquent (90+ DPD) but were later cured or restructured — these
  are genuine credit-risk events a risk model should rank highly. It also gives a
  very thin positive class (~8%).
- *Any late payment = default.* Rejected: a single missed installment is not
  default; it would label a huge, noisy positive class and dilute the signal.
- *Treat `Active` as good.* Rejected — this is the important one. Active loans
  have **not matured**; some will default later. Labelling them good would teach
  the model that future defaulters are safe (label leakage through censoring).
- *Survival / time-to-default model for the Active loans.* A more rigorous way to
  use the censored loans. Rejected as over-scoped for a one-week exercise, but
  it is the first thing I would add with more time (see §11).

**Why 90 DPD.** It is the Basel/industry-standard definition of default and is
available pre-outcome-close via `max_dpd_ever`. It aligns the label with how the
business and regulators already think about "bad".

**⚠ would revisit:** the exact DPD cutoff (60 vs 90) and whether restructured
loans should be forced to `default=1` regardless of DPD.

---

## 2. Which columns may be features (leakage discipline)

**Decision.** Features may come **only** from information available at the
lending decision: the application, the customer master, the **point-in-time**
bureau pull, and product/branch reference data.

**Excluded as leakage:** everything in `loan_servicing` (`max_dpd_ever`,
`num_late_payments_total`, `was_restructured`, `recovery_amount`,
`managing_unit`, `final_loan_status`) and all of `repayments`. These describe
loan *performance after* origination.

**Why.** At scoring time we have none of these. Including any of them would
inflate offline metrics and produce a model that cannot be reproduced in
production. A unit test enforces the exclusion.

---

## 3. Point-in-time bureau join

**Decision.** Attach the most recent bureau pull **on or before** the
application date (`merge_asof`, backward). A pull was found for 98.2% of loans.

**Alternatives.** Latest pull overall (leaks the future); average of all pulls
(same leak); nearest pull in either direction (leaks). All rejected on leakage
grounds. Loans with no prior pull get imputed bureau features + a
`has_bureau_pull=False` flag.

---

## 4. Cleaning calls

See the [Data-Quality Report](DATA_QUALITY_REPORT.md) for the full list. The
judgment calls:
- **Income winsorised, not dropped** — the applicant/label is real, only the
  magnitude is wrong; dropping would bias the sample.
- **Missing employer kept as a flag** — missingness may be informative; imputing
  a fake employer would not be.
- **Contaminated emirate flagged, not discarded** — the contamination marks a
  form/channel and may carry signal.

---

## 5. Feature construction

**Decision.** Raw decision-time fields + three engineered affordability ratios:
`loan_to_income`, `debt_to_income`, `installment_to_income`; plus `age` from DOB;
plus missingness/contamination flags.

Plus the **Debt Burden Ratio (DBR)** and a `dbr_above_threshold` flag — see
below.

**Why these ratios.** Affordability (can the borrower service the debt) is the
core of consumer-credit risk and is not captured by any single raw column. The
target analysis (nb 02) confirms all three rise monotonically with default.

**Not built:** repayment-history features (leakage); employer-level default
rates (thin, high-cardinality, and risks target leakage if computed naively).

**Dropped as redundant:** `product_name` is 1:1 with `product_id`, so one-hot
encoding both just duplicates the same four dummy columns. We keep `product_id`
and drop `product_name` as a feature (it is still carried for EDA labels only).
`interest_rate_band` is also product-derived but kept, because it additionally
feeds the DBR calculation. Removing the duplicate left the served model's metrics
essentially unchanged and tidied the feature importances.

### 5a. Debt Burden Ratio (DBR)

**Decision.** Add `dbr = (new-loan monthly installment + existing monthly debt
obligation) / monthly income`, plus a boolean `dbr_above_threshold` for
`dbr ≥ 0.50`. Logic lives in [`finance.py`](../src/loan_default/finance.py),
reused by training and serving.

**Why.** DBR is *the* affordability metric UAE lenders use, and the UAE Central
Bank caps retail DBR at **50%** of monthly income — so the flag encodes a real
regulatory risk boundary. In this data it separates strongly: loans with
DBR ≥ 50% default at **21%** vs **9.6%** below it (2.2×).

**How each piece is computed**
- *New-loan installment* — proper amortised annuity payment using the product's
  `interest_rate_band` midpoint and term (better than a flat principal/term).
- *Existing monthly obligation* — the bureau gives an outstanding **balance**,
  not a payment, so we proxy the monthly obligation by spreading that balance
  over an assumed tenor (`features.dbr.assumed_existing_debt_tenor_months`,
  default 48). This is an explicit assumption.

**⚠ would revisit:** the assumed existing-debt tenor is the weak point — with a
real bureau feed we would use the actual reported minimum monthly payments, or
at least condition the assumed tenor on `has_long_tenor_loan`. The 48-month
default makes DBR skew high (median ≈ 0.63) in this synthetic portfolio; the
*ranking* signal is what the model uses, and that is robust to the exact tenor.

---

## 6. Encoding & preprocessing

**Decision.** One `ColumnTransformer`: median-impute numerics (+ standardise for
the linear model only); one-hot categoricals with `handle_unknown='ignore'` and
`min_frequency=20`.

**Why.** `handle_unknown='ignore'` keeps the service robust to unseen categories
at scoring time. `min_frequency` folds rare levels into an "infrequent" bucket to
avoid overfitting to sparse categories. Wrapping preprocessing in the fitted
pipeline guarantees train/serve consistency.

---

## 7. Validation split

**Decision.** **Temporal** split — train on the oldest 80% of applications, test
on the most recent 20%.

**Alternatives.** Random k-fold (rejected: leaks time — the model would be
tested on loans contemporaneous with training, hiding drift and giving an
optimistic number). Grouped-by-customer split (customers rarely repeat here, so
low value).

**Why.** Production always scores the future from the past. A temporal split is
the honest estimate of that. It also surfaces the train/test default-rate shift
(15.9% → 19.0%), which a random split would hide.

**⚠ would revisit:** with a firm application-date range I would use a rolling
out-of-time validation rather than a single cut.

---

## 8. Metric choice

**Decision.** Lead with **ROC-AUC** and **PR-AUC**; report **KS** and **Brier**.

**Why.** The brief asks us to *rank* applicants → ROC-AUC (rank quality). Under
16.5% imbalance, **PR-AUC** is more discriminating between models. **KS** is the
metric a credit-risk team will expect. **Brier** checks calibration, which we
need because the decision threshold acts on the probability, not the rank.

We deliberately do **not** optimise headline accuracy — a model that predicts
"good" for everyone scores 83.5% accuracy and is useless.

---

## 9. Models & hyperparameters

**Decision.** Compare **Logistic Regression** (baseline, and — after fair tuning
— promoted to production via config), Random Forest, and XGBoost (the
gradient-boosted challenger). Imbalance handled with `class_weight='balanced'`
(linear/forest) and `scale_pos_weight` (XGBoost).

**Results (temporal test set, after tuning):**

All three models are Optuna-tuned (see §9a) for a fair comparison:

| Model | ROC-AUC | PR-AUC | KS |
|---|---|---|---|
| **Logistic Regression (tuned, served)** | **0.832** | **0.583** | **0.522** |
| Random Forest (tuned) | 0.823 | 0.533 | 0.511 |
| XGBoost (tuned) | 0.829 | 0.572 | 0.522 |

**Observation & decision.** Once *every* model is tuned (not just XGBoost), the
**logistic regression leads** on ROC-AUC and PR-AUC, tying XGBoost on KS. The
margins are small, but the direction is decisive for a regulated credit-scoring
use case: the most **transparent** model is also the best-ranking one. So we
**promote logistic to primary** (`model.primary: logistic_regression`) — you get
top performance *and* full interpretability, with no accuracy trade-off. The
`/predict` reason codes follow the served model: they now use the logistic
**coefficients** (`coef × encoded value`) instead of XGBoost SHAP.

### 9a. Hyperparameter tuning (Optuna)

**Decision.** Tune **all three** candidate models with Optuna (`uv run tune`,
40 TPE trials each), every model with its own search space, optimising
**PR-AUC** under **4-fold time-aware CV** (`TimeSeriesSplit`) on the *training*
portion only.

**Why tune all three (not just XGBoost).** Tuning only XGBoost while leaving the
others on defaults biases the comparison — you would be pitting a *tuned* model
against *untuned* ones. Tuning each (logistic's `C`/penalty, the forest's depth
and leaf size, XGBoost's full set) makes the head-to-head fair, so the model we
promote genuinely earned it. Search effort per model is proportionate: logistic
has essentially one knob, XGBoost has nine.

**Why the other choices**
- *Time-aware CV, not random k-fold* — random folds would train on loans booked
  after those they validate on, leaking time exactly as a random train/test
  split does. `TimeSeriesSplit` keeps every validation fold in the future of its
  training fold.
- *Tune on the training split only* — the most recent 20% hold-out is never seen
  during tuning, so the reported test metrics remain an honest out-of-time
  estimate.
- *Optimise PR-AUC* — more discriminating than ROC-AUC under 16.5% imbalance.
- *Conservative search ranges* to avoid overfitting the ~4.3k-row training set.

Best params per model are written to `artifacts/best_params.json` as
`{model_name: params}` and picked up automatically by the next `uv run train`;
each study (best params + CV score + trial history) is logged to MLflow.

**⚠ would revisit:** probability calibration (isotonic/Platt) before
threshold-setting; a larger trial budget with early-stopping/pruning.

---

## 10. Decision threshold

**Decision.** Default operating threshold **0.5**, exposed in config and analysed
in nb 03 (precision/recall vs threshold).

**Why it's a placeholder.** The right threshold is a **business** choice — it
trades defaults-approved against good-applicants-declined, and needs the loss
given default and approval-rate targets we don't have. The code makes it a config
knob so the business can set it from an expected-cost curve.

---

## 11. Things I'm least confident in / would do next
1. **Censored `Active` loans** — a survival model would use the ~5k loans we
   currently discard and likely improve ranking.

2. **Feature depth** — bureau trend features (score movement across pulls) and
   affordability vs product interest band are promising and unused.

3. **CI/CD pipelin** : automate the ci/cd pipeline on github using github actions functionality.