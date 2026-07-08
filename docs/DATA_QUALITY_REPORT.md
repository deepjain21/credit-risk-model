# Data-Quality Report

A short account of what was wrong with the data, how each issue was detected,
and what was done about it. 
Detection is reproduced in - [`notebooks/01_eda_and_data_quality.ipynb`](../notebooks/01_eda_and_data_quality.ipynb);
every fix lives in - [`src/loan_default/cleaning.py`](../src/loan_default/cleaning.py)
so it is applied identically in training and at scoring time.

## Summary of the extract

| Source | Rows | Grain |
|---|---|---|
| applications.csv | 10,734 | one funded loan |
| customers.json | 8,160 | one borrower |
| bureau_pulls.jsonl | 14,721 | one bureau pull (customers have several) |
| repayments.csv | 249,285 | one installment event |
| loan_servicing.csv | 10,524 | one loan outcome |
| products.csv / branches.csv | 4 / 6 | reference |

After cleaning and labelling, **5,313 loans** are model-ready with a **16.5%
default rate** (see the target definition in the decision log).

---

## Issues found and fixed

### 1. Three different date formats in the same column — **High severity**
- **Detected:** mix date format for `application_date`, `customers.date_of_birth`, `bureau_pulls.pull_date`
  mixes ISO different formats : `2021-04-15`, `31/05/2021`, `44547`(Excel serial numbers)
- **Why it matters:** naive parsing silently mis-reads day-first dates as
  month-first and drops the Excel serials to `NaT`, which corrupts both age and
  the point-in-time bureau join.
- **Fix:** `parse_mixed_date()` — decode 4–6 digit strings as Excel serials
  (origin 1899-12-30), parse the rest as ISO first then day-first.

### 2. `employment_type` — 13 spellings for 3 categories — **Medium**
- **Detected:** value counts show `SALARIED`, `salaried`, `Salaried`, `Salary`,
  `FT-Salaried`, `self employed`, `Self-Employed`, `SE`, `Self Emp`,
  `Unemployed`, `Not Employed`, … .
- **Fix:** `normalise_employment_type()` collapses to
  `{Salaried, Self-Employed, Unemployed}`.

### 3. `emirate` contaminated with channel values — **Medium**
- **Detected:** the emirate column contains `Web`, `Online`, `ONLINE`
  (~1,600 rows) — these are acquisition channels, not emirates. The real
  emirates are also in mixed case/abbreviations (`DXB`, `Abu-Dhabi`, `SHJ`).
- **Fix:** `normalise_emirate()` maps the real emirates to canonical names and
  sets contaminants to missing; `emirate_is_contaminated()`.

### 4. Implausible `declared_income` — **Medium**
- **Detected:** monthly income ranges 2,218 → **50,000,000** with a median near
  14,000. The tail is data-entry error or may be outlier (annual figures / trailing-zero typos).
- **Fix:** `clean_income()` winsorises to `[2,000, 200,000]`. We cap rather than
  drop — the applicant is real, only the magnitude is wrong — which keeps the
  row and its label usable.

### 5. Multiple bureau pulls per customer → leakage risk — **High**
- **Detected:** 4,443 customers have more than one bureau pull, at different
  dates. Some pulls happen *after* the application date.
- **Why it matters:** using a post-application pull leaks the future into the
  model and inflates offline metrics.
- **Fix:** `attach_point_in_time_bureau()` uses `merge_asof(direction=backward)`
  to attach only the most recent pull **on or before** the application date. A
  point-in-time pull was found for **98.2%** of loans; the rest get a
  `has_bureau_pull=False` flag.

### 6. Broken joins across systems — **Medium**
- **Detected:** 107 applications reference a `customer_id` absent from the
  customer master; **210 applications have no servicing outcome**.
- **Fix:** customer-master misses become missing demographic features (imputed).
  Loans with no outcome **cannot be labelled and are dropped** from the
  supervised set (config `cleaning.drop_unlabellable`).

### 7. Missing `employer_name` — **Low**
- **Detected:** 21.5% missing.
- **Fix:** kept as an `employer_missing` boolean feature rather than imputing a
  fake employer; Major of the employer are missing for the self-employed or unemployed application which makes sense.

### 8. Outcome / leakage columns present in the extract — **High (by design)**
- `loan_servicing` (`max_dpd_ever`, `num_late_payments_total`,
  `was_restructured`, `recovery_amount`, `managing_unit`, `final_loan_status`)
  and all of `repayments` describe how the loan *performed* after origination.
- **Fix:** these are used **only** to build the target. They are excluded from
  `FEATURE_COLUMNS`


### 9. Repayments due_date and paid_date have different date formats 
- `due_date` and `paid_date` have different format hence needs to standarized so that we can calculate the dpd
---

## Issues noted but deliberately left alone
- **`Active` loans (5,211)** are not a data error — they are censored outcomes
 which means we donot know wether these loans will be defaulted in future.
  Handled as a *modelling* decision (excluded), not a cleaning one.