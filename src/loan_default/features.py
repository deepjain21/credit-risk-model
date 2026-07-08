"""Feature-engineering pipeline.

A single sklearn ``ColumnTransformer`` that:
  * numeric  -> median-impute (+ optional standard-scale for linear models)
  * categorical -> impute a "missing" level + one-hot (unknown-safe)

Wrapping preprocessing in an sklearn transformer is deliberate: the exact same
fitted transformer is reused at scoring time, so a raw applicant payload is
processed identically in training and in the /predict service.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import CATEGORICAL_FEATURES, FEATURE_COLUMNS

# Numeric = every model feature that is not categorical (booleans count as 0/1).
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Return the preprocessing ColumnTransformer.

    scale_numeric=True is used for the logistic-regression baseline; tree models
    (RandomForest, XGBoost) are scale-invariant so they leave it False.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
