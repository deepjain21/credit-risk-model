# Data Dictionary — Personal Loan Portfolio Extract (Synthetic)

This is a synthetic extract assembled from several operational systems of a fictional
UAE installment lender. Files come from different source systems and were exported at
different times; formats and conventions may vary between sources, as in any real
environment. All records are funded personal loans with a matured outcome history.

## applications.csv
One row per loan application that was approved and funded.
- application_id — identifier for the application/loan
- customer_id — identifier for the borrower
- application_date — date the application was booked
- requested_amount — loan principal (AED)
- loan_term_months — contractual term
- declared_income — borrower-declared income as captured at application
- employment_type — employment category captured at application
- employer_name — declared employer
- emirate — emirate captured for the application
- product_id — product (see products.csv)
- branch_id — originating branch (see branches.csv)
- channel — acquisition channel
- app_form_version — version of the application form used

## customers.json
Borrower master attributes (JSON array).
- customer_id, date_of_birth, gender, nationality_group, marital_status, dependents_count

## bureau_pulls.jsonl
Credit-bureau pulls (JSON Lines; one JSON object per line). A customer may have more
than one pull.
- bureau_pull_id, customer_id, pull_date, bureau_score, num_inquiries_last_3m,
  total_outstanding_debt, num_dpd_90_last_12m, has_long_tenor_loan

## repayments.csv
Installment-level repayment events across loans.
- repayment_id, application_id, due_date, paid_date, scheduled_amount, paid_amount,
  installment_number

## loan_servicing.csv
Servicing and collections summary per loan.
- application_id, final_loan_status, max_dpd_ever, num_late_payments_total,
  was_restructured, recovery_amount, managing_unit

## products.csv
- product_id, product_name, interest_rate_band, max_term_months

## branches.csv
- branch_id, emirate, branch_type
