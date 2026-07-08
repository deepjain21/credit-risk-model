"""Hyperparameter tuning for every candidate model with Optuna.

Run it with:  ``uv run tune``

Design choices that matter:
  * We tune on the TRAINING portion of the temporal split only — the most recent
    20% stays untouched as a final hold-out, so tuning cannot peek at it.
  * Cross-validation inside tuning is **time-aware** (``TimeSeriesSplit`` on the
    time-ordered training rows), not random k-fold. Random folds would let a
    model train on loans booked *after* the ones it validates on, leaking time
    the same way a random train/test split would.
  * We optimise **PR-AUC** (average precision) by default — more discriminating
    than ROC-AUC under our ~16% class imbalance.

All three models are tuned, each with its own search space, so the comparison in
``train.py`` is fair (no tuned-vs-untuned bias). Best params per model are written
to ``config.tuning.params_file`` as ``{model_name: params}``; the next
``uv run train`` loads them automatically.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from .config import load_config
from .dataset import FEATURE_COLUMNS, build_model_ready
from .train import build_pipeline, temporal_split

_SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


# --------------------------------------------------------------------------- #
# Per-model search spaces. Ranges are deliberately conservative to avoid
# overfitting the ~4.3k-row training set.
# --------------------------------------------------------------------------- #
def _suggest_logistic(trial: optuna.Trial) -> dict[str, Any]:
    # solver is fixed to liblinear in train._estimator (it handles l1 and l2).
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),      # regularisation strength
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    }


def _suggest_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }


def _suggest_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


_SUGGEST = {
    "logistic_regression": _suggest_logistic,
    "random_forest": _suggest_random_forest,
    "xgboost": _suggest_xgboost,
}


def run_tuning(cfg: dict[str, Any] | None = None) -> dict[str, dict]:
    cfg = cfg or load_config()
    seed = cfg["project"]["random_seed"]
    tcfg = cfg["tuning"]
    scoring = _SCORING[tcfg["metric"]]

    # Build data and keep ONLY the training portion, time-ordered.
    df = build_model_ready(cfg)
    train_df, _ = temporal_split(df, cfg)
    train_df = train_df.sort_values("application_date")
    X = train_df[FEATURE_COLUMNS]
    y = train_df["default"].to_numpy()
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    cv = TimeSeriesSplit(n_splits=tcfg["cv_folds"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    best_all: dict[str, dict] = {}
    for name in tcfg["models"]:
        def objective(trial: optuna.Trial, name: str = name) -> float:
            pipe = build_pipeline(name, cfg, pos_weight, _SUGGEST[name](trial))
            scores = cross_val_score(pipe, X, y, scoring=scoring, cv=cv, n_jobs=1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        print(f"Tuning {name}: {tcfg['n_trials']} trials, {tcfg['cv_folds']}-fold "
              f"time-aware CV, optimising {tcfg['metric']} ...")
        study.optimize(objective, n_trials=tcfg["n_trials"], show_progress_bar=False)

        best_all[name] = study.best_params
        print(f"  best CV {tcfg['metric']}: {study.best_value:.4f}  params: {study.best_params}")
        _log_study(cfg, name, study)

    _save(cfg, best_all)
    return best_all


def _save(cfg, best_all: dict[str, dict]) -> None:
    """Write {model_name: best_params} to the params file for training to load."""
    params_path = Path(cfg["paths"]["artifacts_dir"]) / Path(cfg["tuning"]["params_file"]).name
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(best_all, indent=2))
    print(f"\nSaved tuned params for {', '.join(best_all)} -> {params_path}")


def _log_study(cfg, name: str, study) -> None:
    """Record one model's study (best params + CV score + trial history) in MLflow."""
    art = Path(cfg["paths"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    trials_csv = art / f"optuna_trials_{name}.csv"
    study.trials_dataframe().to_csv(trials_csv, index=False)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"{name}-optuna-tuning"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"cv_{cfg['tuning']['metric']}", study.best_value)
        mlflow.log_metric("n_trials", len(study.trials))
        mlflow.log_artifact(str(trials_csv))


def main() -> None:
    run_tuning()
    print("\nRun `uv run train` to train with these tuned parameters.")


if __name__ == "__main__":
    main()
