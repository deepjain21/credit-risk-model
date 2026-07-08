"""Turn the seven raw sources into one clean, model-ready table.

Pipeline:
    1. Define the target from the servicing outcome (final_loan_status + DPD).
    2. Clean applications (mixed dates, employment, emirate, income).
    3. Attach customer master attributes (age, demographics).
    4. Attach the point-in-time bureau pull (latest pull STRICTLY on/before the
       application date — this is the leakage-critical step).
    5. Attach product and branch reference attributes.
    6. Engineer a few decision-time ratios (loan-to-income, debt-to-income).
    7. Keep ONLY features available at the lending decision + the target.

The columns that describe how the loan actually performed (max_dpd_ever,
repayments, recovery, restructuring, final status) never enter the feature set
— they are outcomes, not inputs. See LEAKAGE_COLUMNS below.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import cleaning
from .config import load_config
from .data_loading import load_all
from .feature_engineering import engineer_features

# Servicing/repayment columns that describe the OUTCOME. Never features.
LEAKAGE_COLUMNS = [
    "final_loan_status", "max_dpd_ever", "num_late_payments_total",
    "was_restructured", "recovery_amount", "managing_unit",
]

# The feature columns the model is allowed to see (present at decision time).
FEATURE_COLUMNS = [
    # application
    "requested_amount", "loan_term_months", "declared_income",
    "employment_type", "emirate", "product_id", "branch_id", "channel",
    "app_form_version", "employer_missing", "emirate_was_channel",
    # customer
    "age", "gender", "nationality_group", "marital_status", "dependents_count",
    # bureau (point-in-time)
    "bureau_score", "num_inquiries_last_3m", "total_outstanding_debt",
    "num_dpd_90_last_12m", "has_long_tenor_loan", "has_bureau_pull",
    # product / branch reference (product_name is 1:1 with product_id, so it is
    # a redundant duplicate and deliberately excluded; interest_rate_band stays
    # because it also feeds the DBR calculation)
    "interest_rate_band", "branch_type",
    # engineered ratios
    "loan_to_income", "debt_to_income", "installment_to_income",
    "dbr", "dbr_above_threshold",
]

CATEGORICAL_FEATURES = [
    "employment_type", "emirate", "product_id", "branch_id", "channel",
    "app_form_version", "gender", "nationality_group", "marital_status",
    "interest_rate_band", "branch_type",
]


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
def build_target(servicing: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per LABELLED application: application_id -> default (1/0).

    Active (not-yet-matured) loans are dropped first — their outcome is unknown
    (censored). Of the loans that remain, default = 1 if written off / defaulted
    or ever 90+ DPD, else 0. Loans still Active are simply absent from the result.
    """
    tcfg = cfg["target"]
    matured = servicing[~servicing["final_loan_status"].isin(tcfg["exclude_statuses"])]
    status = matured["final_loan_status"].astype("string")

    is_bad = status.isin(tcfg["bad_statuses"]) | (matured["max_dpd_ever"] >= tcfg["dpd_default_threshold"])
    return pd.DataFrame({
        "application_id": matured["application_id"],
        "default": is_bad.astype(int),
    })


# --------------------------------------------------------------------------- #
# Point-in-time bureau features (leakage-critical)
# --------------------------------------------------------------------------- #
def attach_point_in_time_bureau(apps: pd.DataFrame, bureau: pd.DataFrame) -> pd.DataFrame:
    """Attach each application's most recent bureau pull on/before its
    application date. A customer may have several pulls; using a later one would
    leak information from after the lending decision was made.

    merge_asof requires non-null, sorted join keys, so applications whose date
    didn't parse are set aside and rejoined afterwards — they simply have no
    point-in-time pull (NaN bureau columns, has_bureau_pull=False).
    """
    b = bureau.copy()
    b["pull_date"] = cleaning.parse_mixed_date(b["pull_date"])
    b = b.dropna(subset=["pull_date"]).sort_values("pull_date")
    b = b.drop(columns="bureau_pull_id", errors="ignore")   # an id, never a feature

    dated = apps["application_date"].notna()
    matched = pd.merge_asof(
        apps[dated].sort_values("application_date"), b,
        left_on="application_date", right_on="pull_date",
        by="customer_id",
        direction="backward",          # latest pull at or before the application
    )
    # concat unions the columns, so the undated rows get NaN bureau fields for free.
    out = pd.concat([matched, apps[~dated]], ignore_index=True)
    out["has_bureau_pull"] = out["bureau_score"].notna() if "bureau_score" in out else False
    return out


# --------------------------------------------------------------------------- #
# Full build
# --------------------------------------------------------------------------- #
def build_model_ready(cfg: dict[str, Any] | None = None, keep_diagnostics: bool = False) -> pd.DataFrame:
    """Build the model-ready dataset. Returns one row per labelled application.

    If ``keep_diagnostics`` is True, leakage/outcome columns and application_date
    are retained (for EDA / target analysis in notebooks) but they are never in
    FEATURE_COLUMNS, so they cannot reach the model.
    """
    cfg = cfg or load_config()
    raw = load_all(cfg)
    apps, customers = raw["applications"].copy(), raw["customers"].copy()

    # --- clean applications -------------------------------------------------
    apps["application_date"] = cleaning.parse_mixed_date(apps["application_date"])
    apps["employment_type"] = cleaning.normalise_employment_type(apps["employment_type"])
    apps["emirate_was_channel"] = cleaning.emirate_is_contaminated(apps["emirate"])
    apps["emirate"] = cleaning.normalise_emirate(apps["emirate"])
    apps["declared_income"] = cleaning.clean_income(
        apps["declared_income"], cfg["cleaning"]["income_min"], cfg["cleaning"]["income_max"]
    )
    apps["employer_missing"] = apps["employer_name"].isna()

    # --- customer master ----------------------------------------------------
    apps = apps.merge(customers, on="customer_id", how="left")
    apps["age"] = cleaning.age_from_dob(apps["date_of_birth"], apps["application_date"])

    # --- point-in-time bureau ----------------------------------------------
    apps = attach_point_in_time_bureau(apps, raw["bureau"])

    # --- product / branch reference ----------------------------------------
    apps = apps.merge(raw["products"][["product_id", "product_name", "interest_rate_band"]],
                      on="product_id", how="left")
    apps = apps.merge(raw["branches"][["branch_id", "branch_type"]], on="branch_id", how="left")

    # --- engineered decision-time features (ratios + DBR) ------------------
    # Row-wise derivation lives in feature_engineering.engineer_features, the
    # SAME function the serving path calls, so the formulas cannot drift.
    apps = engineer_features(apps, cfg)

    # --- target -------------------------------------------------------------
    target = build_target(raw["servicing"], cfg)
    apps = apps.merge(target, on="application_id", how="left")
    # Bring outcome columns along for diagnostics only (never in FEATURE_COLUMNS).
    apps = apps.merge(raw["servicing"], on="application_id", how="left", suffixes=("", "_srv"))

    # build_target only returns labelled loans, so after the left-merge the
    # unlabellable ones (Active / no servicing outcome) have a null default.
    if cfg["cleaning"]["drop_unlabellable"]:
        apps = apps[apps["default"].notna()].copy()
    apps["default"] = apps["default"].astype(int)

    # sklearn's imputers/encoders expect numpy NaN, not pandas <NA>. Cast the
    # nullable "string" categoricals back to plain object dtype.
    for col in CATEGORICAL_FEATURES:
        apps[col] = apps[col].astype(object).where(apps[col].notna(), np.nan)

    # application_date is always kept (needed for the temporal split) but it is
    # NOT in FEATURE_COLUMNS, so the model never trains on the raw date.
    keep = ["application_id", "customer_id", "application_date"] + FEATURE_COLUMNS + ["default"]
    if keep_diagnostics:
        # outcome columns + product_name (dropped as a feature, but a readable
        # label for EDA) — kept for notebooks, never inside FEATURE_COLUMNS.
        diagnostics = LEAKAGE_COLUMNS + ["product_name"]
        keep = keep + [c for c in diagnostics if c in apps.columns and c not in keep]
    return apps[keep].reset_index(drop=True)
