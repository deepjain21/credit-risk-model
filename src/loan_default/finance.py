"""Financial calculations used to engineer decision-time features.

Kept in its own module (scalar, pure functions) so the DBR / installment logic
is written ONCE and reused by both the training builder (dataset.py) and the
serving path (predict.py) — no train/serve skew.

DBR (Debt Burden Ratio) is the share of a borrower's monthly income already
committed to debt repayments. In the UAE the Central Bank caps retail DBR at
50% of monthly income, so DBR >= 0.50 is the standard "risky" flag.

Data caveat: the bureau gives us `total_outstanding_debt` — a *balance*, not a
monthly payment. A true DBR needs the monthly obligation, so we proxy the
existing-debt payment by amortising/​spreading that balance over an assumed
tenor (config `features.dbr.assumed_existing_debt_tenor_months`). This is an
explicit modelling assumption, documented in the decision log.
"""
from __future__ import annotations

import re

import numpy as np


def interest_band_midpoint(band: object) -> float:
    """Parse an interest-rate band like '18-22%' to its midpoint as a decimal.

    '18-22%' -> 0.20, '12-16%' -> 0.14. Returns NaN if unparseable.
    """
    if not isinstance(band, str):
        return np.nan
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", band)]
    if not nums:
        return np.nan
    return (sum(nums) / len(nums)) / 100.0


def monthly_installment(principal: float, annual_rate: float, term_months: float) -> float:
    """Amortised monthly payment for a loan.

    Uses the standard annuity formula; falls back to principal/term when the
    rate is unknown or zero.
    """
    if not principal or not term_months or term_months <= 0:
        return np.nan
    if annual_rate is None or np.isnan(annual_rate) or annual_rate <= 0:
        return principal / term_months
    r = annual_rate / 12.0
    return principal * r / (1.0 - (1.0 + r) ** (-term_months))


def debt_burden_ratio(
    requested_amount: float,
    term_months: float,
    annual_rate: float,
    existing_debt: float,
    monthly_income: float,
    assumed_existing_tenor: float,
) -> float:
    """DBR = (new-loan installment + existing monthly obligation) / monthly income.

    The existing monthly obligation is proxied by spreading the bureau
    outstanding balance over ``assumed_existing_tenor`` months.
    """
    if not monthly_income or monthly_income <= 0:
        return np.nan
    new_inst = monthly_installment(requested_amount, annual_rate, term_months)
    if new_inst is None or np.isnan(new_inst):
        return np.nan
    existing_monthly = (existing_debt or 0.0) / assumed_existing_tenor if assumed_existing_tenor else 0.0
    return (new_inst + existing_monthly) / monthly_income
