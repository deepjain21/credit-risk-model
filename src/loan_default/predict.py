"""Inference pipeline: featurize a raw applicant payload and score it.

The service receives the fields a loan officer has at the decision point:
application details, customer master attributes, and the bureau pull the lender
just fetched. We rebuild EXACTLY the model features using the same cleaning
functions as training (no train/serve skew), then let the saved sklearn
pipeline impute/encode and predict.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import cleaning
from .config import load_config
from .dataset import CATEGORICAL_FEATURES, FEATURE_COLUMNS
from .feature_engineering import engineer_features


@functools.lru_cache(maxsize=1)
def load_artifact(model_file: str | None = None) -> dict[str, Any]:
    """Load and cache the model artifact (fitted pipeline + metadata + reference).

    Source is chosen by ``serving.model_source`` in the config:
      * ``joblib``   — the self-contained artifacts/model.joblib (default).
      * ``registry`` — the MLflow Model Registry, resolved by champion alias.
    Both return the same dict shape, so the rest of serving is unaffected.
    """
    cfg = load_config()
    if cfg.get("serving", {}).get("model_source") == "registry":
        return _load_from_registry(cfg)
    path = Path(model_file or cfg["paths"]["model_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Train one first with `uv run train`."
        )
    return joblib.load(path)


def _load_from_registry(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the primary model (and its embedded metadata) from MLflow."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import Model

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    uri = f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['champion_alias']}"
    model = mlflow.sklearn.load_model(uri)
    metadata = Model.load(uri).metadata or {}
    return {"model": model, **metadata}


def featurize_applicant(payload: dict[str, Any], reference: dict[str, dict]) -> pd.DataFrame:
    """Build a one-row feature frame from a raw applicant payload.

    ``reference`` holds the small product/branch lookups saved with the model,
    so the caller only supplies product_id / branch_id, not derived attributes.
    """
    p = dict(payload)  # shallow copy; don't mutate caller's dict
    cfg = load_config()
    app_date = cleaning.parse_mixed_date(pd.Series([p.get("application_date")])).iloc[0]

    income = cleaning.clean_income(
        pd.Series([p.get("declared_income")]),
        cfg["cleaning"]["income_min"],
        cfg["cleaning"]["income_max"],
    ).iloc[0]

    prod = reference["products"].get(p.get("product_id"), {})
    branch = reference["branches"].get(p.get("branch_id"), {})

    # Assemble the cleaned raw columns. The engineered ratios/DBR are added by
    # engineer_features below — the SAME function training uses — so we never
    # re-implement the feature formulas here.
    row = {
        "requested_amount": p.get("requested_amount"),
        "loan_term_months": p.get("loan_term_months"),
        "declared_income": income,
        "employment_type": cleaning.normalise_employment_type(pd.Series([p.get("employment_type")])).iloc[0],
        "emirate": cleaning.normalise_emirate(pd.Series([p.get("emirate")])).iloc[0],
        "product_id": p.get("product_id"),
        "branch_id": p.get("branch_id"),
        "channel": p.get("channel"),
        "app_form_version": p.get("app_form_version"),
        "employer_missing": p.get("employer_name") in (None, "", np.nan),
        "emirate_was_channel": bool(cleaning.emirate_is_contaminated(pd.Series([p.get("emirate")])).iloc[0]),
        "age": cleaning.age_from_dob(pd.Series([p.get("date_of_birth")]), pd.Series([app_date])).iloc[0],
        "gender": p.get("gender"),
        "nationality_group": p.get("nationality_group"),
        "marital_status": p.get("marital_status"),
        "dependents_count": p.get("dependents_count"),
        "bureau_score": p.get("bureau_score"),
        "num_inquiries_last_3m": p.get("num_inquiries_last_3m"),
        "total_outstanding_debt": p.get("total_outstanding_debt"),
        "num_dpd_90_last_12m": p.get("num_dpd_90_last_12m"),
        "has_long_tenor_loan": p.get("has_long_tenor_loan"),
        "has_bureau_pull": p.get("bureau_score") is not None,
        "interest_rate_band": prod.get("interest_rate_band"),
        "branch_type": branch.get("branch_type"),
    }
    df = engineer_features(pd.DataFrame([row]), cfg)[FEATURE_COLUMNS]
    # Cleaning helpers can emit pandas <NA>; sklearn wants numpy NaN (matches training).
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
    return df


def predict_one(payload: dict[str, Any], model_file: str | None = None) -> dict[str, Any]:
    """Score a single applicant. Returns probability, decision, and reasons."""
    art = load_artifact(model_file)
    X = featurize_applicant(payload, art["reference"])
    model = art["model"]
    proba = float(model.predict_proba(X)[:, 1][0])
    threshold = art["decision_threshold"]
    return {
        "default_probability": round(proba, 4),
        "decision": "decline" if proba >= threshold else "approve",
        "risk_band": _risk_band(proba),
        "top_factors": explain_prediction(model, X),
        "threshold": threshold,
        "model_name": art["model_name"],
        "model_trained_at": art["trained_at"],
    }


def _risk_band(proba: float) -> str:
    """Coarse band for human consumption alongside the raw score."""
    if proba < 0.05:
        return "low"
    if proba < 0.15:
        return "medium"
    if proba < 0.30:
        return "high"
    return "very_high"


# Human-friendly labels for the reason codes shown to a loan officer.
_FEATURE_LABELS = {
    "bureau_score": "Credit bureau score",
    "dbr": "Debt burden ratio",
    "dbr_above_threshold": "Debt burden above 50% cap",
    "num_dpd_90_last_12m": "90+ day delinquencies (last 12m)",
    "num_inquiries_last_3m": "Credit inquiries (last 3m)",
    "debt_to_income": "Debt-to-income",
    "loan_to_income": "Loan-to-income",
    "installment_to_income": "Installment-to-income",
    "total_outstanding_debt": "Existing outstanding debt",
    "requested_amount": "Requested loan amount",
    "loan_term_months": "Loan term (months)",
    "declared_income": "Declared monthly income",
    "age": "Age",
    "employment_type": "Employment type",
    "emirate": "Emirate",
    "product_id": "Loan product",
    "interest_rate_band": "Interest-rate band",
    "branch_type": "Branch type",
    "channel": "Application channel",
    "nationality_group": "Nationality group",
    "marital_status": "Marital status",
    "dependents_count": "Number of dependents",
    "has_long_tenor_loan": "Has a long-tenor loan",
    "has_bureau_pull": "Bureau pull available",
    "employer_missing": "Employer not provided",
}


def _original_feature(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to its source feature.

    'num__bureau_score'  -> 'bureau_score'
    'cat__emirate_Dubai' -> 'emirate'  (matched against the categorical list)
    """
    name = transformed_name.split("__", 1)[-1]        # drop the num__ / cat__ prefix
    if name in FEATURE_COLUMNS:
        return name
    for cat in CATEGORICAL_FEATURES:                  # one-hot: <feature>_<category>
        if name == cat or name.startswith(cat + "_"):
            return cat
    return name


def _transformed_contributions(model, X: pd.DataFrame):
    """Per-transformed-column contribution to the default log-odds for this row.

    XGBoost -> SHAP contributions (pred_contribs). Linear/logistic -> coefficient
    x (scaled/encoded) value. Returns (names, contribs), or (None, None) if the
    model exposes neither — so the caller degrades gracefully.
    """
    prep = model.named_steps.get("prep")
    clf = model.named_steps.get("clf")
    if prep is None or clf is None:
        return None, None
    names = list(prep.get_feature_names_out())
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):            # one-hot output may be sparse
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)

    if hasattr(clf, "get_booster"):       # XGBoost / gradient boosting
        import xgboost as xgb
        dm = xgb.DMatrix(Xt, feature_names=names)
        return names, clf.get_booster().predict(dm, pred_contribs=True)[0][:-1]  # drop bias
    if hasattr(clf, "coef_"):             # logistic / linear
        return names, clf.coef_[0] * Xt[0]
    return None, None


def explain_prediction(model, X: pd.DataFrame, top_k: int = 5) -> list[dict[str, Any]]:
    """Top per-applicant factors behind the score, as log-odds contributions.

    Works for the tree model (SHAP) and the linear model (coef x value), so the
    reasons follow whichever model is promoted to primary. Returns
    [{feature, value, direction, impact}]; `direction` is 'increases'/'decreases'
    default risk. Empty list if the model exposes no contributions.
    """
    names, contribs = _transformed_contributions(model, X)
    if names is None:
        return []

    # Sum one-hot columns back onto their source feature.
    agg: dict[str, float] = {}
    for name, c in zip(names, contribs):
        feat = _original_feature(name)
        agg[feat] = agg.get(feat, 0.0) + float(c)

    row = X.iloc[0]
    factors = []
    for feat, contrib in sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True):
        impact = round(abs(float(contrib)), 3)
        if impact == 0 or len(factors) >= top_k:      # skip negligible factors
            continue
        val = row.get(feat)
        if pd.isna(val):
            shown = None
        elif isinstance(val, (int, float, np.number)):
            shown = round(float(val), 3)
        else:
            shown = str(val)
        factors.append({
            "feature": _FEATURE_LABELS.get(feat, feat),
            "value": shown,
            "direction": "increases" if contrib > 0 else "decreases",
            "impact": impact,
        })
    return factors
