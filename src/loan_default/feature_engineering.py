"""Row-wise engineered features (decision-time ratios + DBR).

This is the "feature engineering" layer: pure ``DataFrame -> DataFrame`` maths
that derives new columns from already-cleaned, already-joined raw columns. It is
deliberately **stateless** (nothing is fitted) and defined **once** here, then
called by BOTH:

  * the training dataset builder (``dataset.build_model_ready``), on the full
    joined table, and
  * the serving path (``predict.featurize_applicant``), on a one-row frame.

Because both paths call the same function, the engineered-feature formulas can
never drift between train and serve — the same class of skew the shared
``cleaning`` and ``finance`` modules already guard against.

Not here: table assembly (joins, the leakage-critical point-in-time bureau
merge, the target) stays in ``dataset.py`` because it needs other tables; the
fitted impute/encode/scale steps stay in ``features.py`` because they must be
learned on the training data. This module is only the stateless middle layer.

Expects these (cleaned) input columns to be present:
    declared_income, requested_amount, loan_term_months,
    total_outstanding_debt, interest_rate_band
and adds:
    loan_to_income, debt_to_income, installment_to_income, dbr, dbr_above_threshold
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import finance

# The columns this module adds. Handy for tests / documentation.
ENGINEERED_FEATURES = [
    "loan_to_income", "debt_to_income", "installment_to_income",
    "dbr", "dbr_above_threshold",
]


def engineer_features(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``df`` with the engineered decision-time features added.

    Numeric inputs are coerced defensively so the same code handles a clean
    training column and a sparse single-applicant payload (missing values become
    NaN and flow through as NaN, to be imputed by the fitted preprocessor).
    """
    df = df.copy()

    income = pd.to_numeric(df["declared_income"], errors="coerce").replace(0, np.nan)
    requested = pd.to_numeric(df["requested_amount"], errors="coerce")
    term = pd.to_numeric(df["loan_term_months"], errors="coerce")
    debt = pd.to_numeric(df["total_outstanding_debt"], errors="coerce")

    # Affordability ratios (annualised income where the numerator is a balance).
    df["loan_to_income"] = requested / (income * 12)
    df["debt_to_income"] = debt / (income * 12)
    # Crude flat installment = principal / term, as a share of monthly income.
    df["installment_to_income"] = (requested / term) / income

    # Debt Burden Ratio: a proper amortised new-loan payment (using the product's
    # interest band) + the existing monthly obligation, over monthly income.
    dbr_cfg = cfg["features"]["dbr"]
    assumed_tenor = dbr_cfg["assumed_existing_debt_tenor_months"]
    rate_mid = df["interest_rate_band"].map(finance.interest_band_midpoint)
    df["dbr"] = [
        finance.debt_burden_ratio(ra, tm, ar, ed, mi, assumed_tenor)
        for ra, tm, ar, ed, mi in zip(requested, term, rate_mid, debt, income)
    ]
    df["dbr_above_threshold"] = np.where(
        df["dbr"].isna(), np.nan, (df["dbr"] >= dbr_cfg["risky_threshold"]).astype(float)
    )
    return df
