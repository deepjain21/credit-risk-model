"""Training pipeline: build data -> split -> fit candidates -> log -> save best.

Run it with:  ``uv run train``  (or ``python -m loan_default.train``)

Three models are compared on the same temporal split; all three are Optuna-tuned
(see tune.py) for a fair comparison:
  * logistic_regression — transparent, regulator-friendly; currently promoted
  * random_forest       — non-linear reference
  * xgboost             — gradient-boosted challenger

Every run is logged to MLflow (params, metrics, model). The model named in
``config.model.primary`` (not necessarily the top scorer) is saved to
``artifacts/model.joblib`` with the metadata the serving code needs (feature
list, threshold, training date).
"""
from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import load_config
from .dataset import FEATURE_COLUMNS, build_model_ready
from .features import build_preprocessor


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def temporal_split(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split so the test set is the most recent applications (production-like).

    Falls back to a random split if configured, but temporal is the default:
    we always score future applicants using a model fit on the past, and a
    temporal split is the honest way to estimate that.
    """
    frac = cfg["validation"]["test_fraction"]
    if cfg["validation"]["split_strategy"] == "random":
        test = df.sample(frac=frac, random_state=cfg["project"]["random_seed"])
        return df.drop(test.index), test

    ordered = df.sort_values("application_date")
    cut = int(len(ordered) * (1 - frac))
    return ordered.iloc[:cut], ordered.iloc[cut:]


# --------------------------------------------------------------------------- #
# Candidate models
# --------------------------------------------------------------------------- #
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")

# Sensible defaults per model, used for any params tuning hasn't overridden.
_MODEL_DEFAULTS = {
    "logistic_regression": {},   # C defaults to 1.0
    "random_forest": dict(n_estimators=400, max_depth=8, min_samples_leaf=25),
    "xgboost": dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0),
}


def _load_tuned(cfg: dict[str, Any]) -> dict[str, dict]:
    """Load the per-model Optuna-tuned params ({model_name: params}), or {}."""
    params_path = Path(cfg["paths"]["artifacts_dir"]) / Path(cfg["tuning"]["params_file"]).name
    return json.loads(params_path.read_text()) if params_path.exists() else {}


def _estimator(name: str, cfg: dict[str, Any], pos_weight: float, params: dict[str, Any]):
    """One classifier, with imbalance handled per model and `params` applied."""
    seed = cfg["project"]["random_seed"]
    if name == "logistic_regression":
        # liblinear supports both l1 and l2 penalties for a binary problem.
        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  solver="liblinear", random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "xgboost":
        fixed = dict(scale_pos_weight=pos_weight, eval_metric="aucpr", n_jobs=-1, random_state=seed)
        return XGBClassifier(**{**fixed, **params})
    raise ValueError(f"unknown model: {name}")


def build_pipeline(name: str, cfg: dict[str, Any], pos_weight: float,
                   params: dict[str, Any] | None = None) -> Pipeline:
    """Preprocessor + estimator for one model. Logistic gets scaled numerics;
    the tree models are scale-invariant so they don't. Shared by training and
    tuning so both fit the exact same pipeline shape."""
    scale = name == "logistic_regression"
    return Pipeline([
        ("prep", build_preprocessor(scale_numeric=scale)),
        ("clf", _estimator(name, cfg, pos_weight, params or {})),
    ])


def candidate_models(cfg: dict[str, Any], pos_weight: float) -> dict[str, Pipeline]:
    """Return {name: Pipeline} for all three models, each merging its defaults
    with any Optuna-tuned params found on disk (so training improves once
    `uv run tune` has been run). Class imbalance (~16%) is handled per model."""
    tuned = _load_tuned(cfg)
    if tuned:
        print(f"Using tuned params for: {', '.join(tuned)}")
    else:
        print("No tuned params found (run `uv run tune`); using defaults.")
    return {
        name: build_pipeline(name, cfg, pos_weight, {**_MODEL_DEFAULTS[name], **tuned.get(name, {})})
        for name in MODEL_NAMES
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    """Ranking + calibration + threshold metrics.

    ROC-AUC and PR-AUC measure ranking (the brief asks us to *rank* applicants).
    KS is the classic credit-risk separation metric. Brier checks calibration.
    """
    pred = (proba >= threshold).astype(int)
    # KS statistic: max separation between good/bad cumulative score distributions.
    order = np.argsort(proba)
    y_sorted = y_true[order]
    cum_bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    ks = float(np.max(np.abs(cum_bad - cum_good)))
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "ks": ks,
        "brier": float(brier_score_loss(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_training(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    cfg = cfg or load_config()
    seed = cfg["project"]["random_seed"]
    threshold = cfg["model"]["decision_threshold"]

    df = build_model_ready(cfg)
    Path(cfg["paths"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["paths"]["dataset_file"], index=False)

    train_df, test_df = temporal_split(df, cfg)
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["default"].to_numpy()
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["default"].to_numpy()
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    print(f"Rows: {len(df):,} | train {len(train_df):,} / test {len(test_df):,} "
          f"| default rate train {y_train.mean():.1%} test {y_test.mean():.1%}")

    # Quiet MLflow's env-capture / pickle-caution chatter; real errors still show.
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    # sklearn forward-compat notice for the l1 logistic penalty; harmless here.
    warnings.filterwarnings("ignore", message=".*penalty is deprecated.*")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    models = candidate_models(cfg, pos_weight)
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    primary = cfg["model"]["primary"]
    for name, pipe in models.items():
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, proba, threshold)
            results[name] = metrics
            fitted[name] = pipe

            mlflow.log_params({"model": name, "n_train": len(train_df),
                               "split": cfg["validation"]["split_strategy"],
                               "threshold": threshold})
            mlflow.log_metrics(metrics)
            _log_model(cfg, name, pipe, metrics, y_train, X_test, is_primary=name == primary)
            print(f"  {name:20s} ROC-AUC {metrics['roc_auc']:.3f} | "
                  f"PR-AUC {metrics['pr_auc']:.3f} | KS {metrics['ks']:.3f}")

    _save_primary(cfg, fitted, results, y_train)
    return results


def _reference_tables(cfg) -> dict[str, dict]:
    """Small product/branch lookups bundled with the model so the serving code
    can enrich a payload from product_id / branch_id alone."""
    from .data_loading import load_branches, load_products
    products = load_products(cfg).set_index("product_id")[["interest_rate_band"]]  # product_name isn't a feature
    branches = load_branches(cfg).set_index("branch_id")[["branch_type"]]
    return {"products": products.to_dict("index"), "branches": branches.to_dict("index")}


def _serving_metadata(cfg, name, metrics, y_train) -> dict[str, Any]:
    """Everything the serving path needs *besides* the fitted estimator.

    Shared by the joblib artifact and the MLflow model so both serving sources
    (disk or registry) carry the same threshold, target rule, and reference
    lookups — no drift between them.
    """
    return {
        "model_name": name,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": cfg["model"]["decision_threshold"],
        "target_definition": cfg["target"],
        "train_default_rate": float(np.mean(y_train)),
        "test_metrics": metrics,
        "reference": _reference_tables(cfg),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def _log_model(cfg, name, pipe, metrics, y_train, X_example, is_primary: bool) -> None:
    """Log the fitted pipeline to the active MLflow run.

    The serving metadata is embedded in the model itself so a run (or a
    registry version) is self-contained. Only the configured primary model is
    registered to the Model Registry; a moving alias points at the new version.
    """
    if not cfg["mlflow"].get("log_models", True):
        return
    registered = cfg["mlflow"].get("registered_model_name") if is_primary else None
    # Cast integer columns to float in the schema example: integers can't hold
    # NaN, so a missing value at inference would trip MLflow schema enforcement.
    # Doubles match how serving may legitimately receive a missing field.
    example = X_example.head(3).copy()
    example = example.astype({c: "float64" for c in example.select_dtypes("integer").columns})
    with warnings.catch_warnings():
        # the model's output is the integer class label (0/1); MLflow's generic
        # "integer column can't hold NaN" hint doesn't apply to a prediction.
        warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            input_example=example,
            metadata=_serving_metadata(cfg, name, metrics, y_train),
            registered_model_name=registered,
            # cloudpickle (same as joblib) — MLflow 3's default skops serializer
            # rejects the numpy.dtype objects inside the XGBoost/ColumnTransformer.
            serialization_format="cloudpickle",
        )
    if registered and cfg["mlflow"].get("champion_alias"):
        _set_champion_alias(cfg, info.registered_model_version)


def _set_champion_alias(cfg, version) -> None:
    """Point the champion alias at the just-registered primary version.

    Aliases replace the deprecated Staging/Production stages in MLflow 3, so
    serving can always resolve models:/<name>@champion to the current model.
    """
    client = mlflow.MlflowClient()
    name = cfg["mlflow"]["registered_model_name"]
    alias = cfg["mlflow"]["champion_alias"]
    client.set_registered_model_alias(name, alias, version)
    print(f"Registered '{name}' v{version} and set alias '{alias}'")


def _save_primary(cfg, fitted, results, y_train) -> None:
    """Persist the configured primary model + metadata for serving (joblib)."""
    name = cfg["model"]["primary"]
    artifact = {"model": fitted[name], **_serving_metadata(cfg, name, results[name], y_train)}
    joblib.dump(artifact, cfg["paths"]["model_file"])
    print(f"\nSaved primary model '{name}' -> {cfg['paths']['model_file']}")


def main() -> None:
    results = run_training()
    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\nBest by ROC-AUC: {best} ({results[best]['roc_auc']:.3f})")


if __name__ == "__main__":
    main()
