"""Lightweight sanity tests. Run with:  uv run pytest -q

These are guardrails, not exhaustive coverage. The most important one is
``test_no_leakage_columns_in_features`` — it fails loudly if an outcome column
ever slips into the model's feature set.
"""
import numpy as np
import pandas as pd

from loan_default import cleaning, finance
from loan_default.dataset import (
    FEATURE_COLUMNS, LEAKAGE_COLUMNS, build_model_ready,
)
from loan_default.predict import predict_one


# --- cleaning ------------------------------------------------------------- #
def test_parse_mixed_date_handles_three_formats():
    s = pd.Series(["2021-04-15", "31/05/2021", "44547"])  # ISO, day-first, Excel
    out = cleaning.parse_mixed_date(s)
    assert out.iloc[0] == pd.Timestamp("2021-04-15")
    assert out.iloc[1] == pd.Timestamp("2021-05-31")
    assert out.iloc[2] == pd.Timestamp("2021-12-17")  # Excel serial 44547


def test_employment_type_collapses_to_three():
    s = pd.Series(["SALARIED", "FT-Salaried", "self employed", "SE", "Not Employed"])
    out = cleaning.normalise_employment_type(s)
    assert set(out) == {"Salaried", "Self-Employed", "Unemployed"}


def test_emirate_channel_contaminants_flagged_not_kept():
    s = pd.Series(["DXB", "Web", "Online"])
    assert cleaning.normalise_emirate(s).tolist()[0] == "Dubai"
    assert cleaning.normalise_emirate(s).isna().sum() == 2
    assert cleaning.emirate_is_contaminated(s).tolist() == [False, True, True]


def test_income_is_winsorised():
    s = pd.Series([1000, 14000, 50_000_000])
    out = cleaning.clean_income(s, 2000, 200000)
    assert out.tolist() == [2000, 14000, 200000]


# --- finance / DBR -------------------------------------------------------- #
def test_interest_band_midpoint():
    assert finance.interest_band_midpoint("18-22%") == 0.20
    assert finance.interest_band_midpoint("12-16%") == 0.14
    assert np.isnan(finance.interest_band_midpoint(None))


def test_dbr_is_share_of_income_and_flags_risky():
    # 300k over 60m at ~20% APR ~= 7.9k/m; plus 200k existing over 48m ~= 4.2k/m.
    # Against 10k monthly income that is well above the 50% cap.
    dbr = finance.debt_burden_ratio(
        requested_amount=300_000, term_months=60, annual_rate=0.20,
        existing_debt=200_000, monthly_income=10_000, assumed_existing_tenor=48,
    )
    assert dbr > 0.50                       # risky per UAE Central Bank cap
    # A small loan for a high earner is comfortably under the cap.
    low = finance.debt_burden_ratio(50_000, 36, 0.14, 0, 40_000, 48)
    assert low < 0.50


# --- dataset -------------------------------------------------------------- #
def test_no_leakage_columns_in_features():
    """No outcome/servicing column may appear among model features."""
    assert not (set(LEAKAGE_COLUMNS) & set(FEATURE_COLUMNS))


def test_build_model_ready_is_binary_and_labelled():
    df = build_model_ready()
    assert df["default"].isin([0, 1]).all()
    assert df["default"].nunique() == 2
    assert len(df) > 1000


# --- inference ------------------------------------------------------------ #
def test_predict_one_returns_probability():
    payload = {
        "application_date": "2023-06-15", "requested_amount": 150000,
        "loan_term_months": 36, "declared_income": 14000, "product_id": "PRD-01",
        "branch_id": "BR-DXB1", "bureau_score": 640,
    }
    out = predict_one(payload)
    assert 0.0 <= out["default_probability"] <= 1.0
    assert out["decision"] in {"approve", "decline"}
