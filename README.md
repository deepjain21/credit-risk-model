# Personal-Loan Default-Risk Model

The task: from seven messy operational extracts of a fictional UAE installment
lender, build a model that **ranks new personal-loan applicants by default risk**
using only information available at the point of the lending decision — and
deliver it as a reproducible, production-shaped project.

> **Where to start reading:** the three documents in [`docs/`](docs/) carry the
> reasoning. [`DECISION_LOG.md`](docs/DECISION_LOG.md) is the primary deliverable.

---

## Results at a glance

- **5,313** model-ready loans after cleaning; **16.5%** default rate.
- **Target:** `default = 1` if written-off / defaulted **or** `max_dpd_ever ≥ 90`;
  `Active` (not-yet-matured) loans excluded. Full rationale in the decision log.
- **Temporal validation** (train on the past, test on the most recent 20%):

  All three models are Optuna-tuned (time-aware CV) for a fair comparison:

  | Model | ROC-AUC | PR-AUC | KS |
  |---|---|---|---|
  | **Logistic Regression** (tuned, served) | **0.832** | **0.583** | **0.522** |
  | Random Forest (tuned) | 0.823 | 0.533 | 0.511 |
  | XGBoost (tuned) | 0.829 | 0.572 | 0.522 |

  With every model tuned, the transparent **logistic regression wins** on ROC-AUC
  and PR-AUC — so it is the served model (`model.primary`). Best of both worlds
  for a regulated use case: top ranking *and* full interpretability. The
  `/predict` reason codes use its coefficients (`coef × value`).

---

## Project layout

```
config/config.yaml        # every knob: target rule, cleaning bounds, split, model
data/                     # the seven raw source files (given)
docs/
  DECISION_LOG.md         # * every non-trivial choice + alternatives + why
  DATA_QUALITY_REPORT.md  # what was wrong with the data and how it was fixed
  DESIGN_NOTE.md          # serving + how to monitor a slow-maturing model
notebooks/
  01_eda_and_data_quality.ipynb              # find the data problems
  02_dataset_construction_and_target.ipynb   # define target, target-vs-feature analysis
  03_model_experiments.ipynb                 # compare models, curves, importances
src/loan_default/
  config.py         # load config
  data_loading.py   # read each source
  cleaning.py       # reusable normalisers (dates, categories, income) — train == serve
  dataset.py        # join + point-in-time bureau + target -> model-ready table
  feature_engineering.py  # row-wise derived features (ratios, DBR) — shared train + serve
  features.py       # sklearn preprocessing ColumnTransformer (fitted impute/encode/scale)
  finance.py        # DBR / amortised-installment maths (shared train + serve)
  tune.py           # Optuna hyperparameter search (time-aware CV) for all 3 models
  train.py          # build -> temporal split -> fit 3 models -> MLflow -> save primary
  predict.py        # featurize one applicant + score + reason codes (reuses cleaning.py)
  serve.py          # FastAPI: form UI at /, plus /predict + /health
tests/              # sanity tests incl. a leakage guard
examples/           # sample /predict payload
```

The pipeline mirrors the lifecycle the brief asks for: **data quality ->
exploratory analysis -> feature construction -> model selection -> tuning ->
deployment stub**.

---

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (Python is pinned via `uv`).

```bash
uv sync                      # create env + install pinned deps

# 1. (optional) Tune all three models with Optuna (time-aware CV)
#    -> artifacts/best_params.json.  Skip it and step 2 uses sensible defaults.
uv run tune

# 2. Train: builds the dataset, compares 3 models, logs params + metrics + the
#    fitted model to MLflow, registers the primary to the Model Registry
#    (alias `champion`), and saves it to artifacts/model.joblib.
#    Picks up tuned params automatically if step 1 was run.
uv run train

# 3. Explore MLflow runs + registered models  (port 5001: on macOS the AirPlay
#    Receiver owns port 5000, so we avoid it) -> http://127.0.0.1:5001
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# 4. Serve the model
uv run uvicorn loan_default.serve:app --port 8000
#    -> open http://127.0.0.1:8000 for a simple form UI (score + reasons), or:
curl -s localhost:8000/predict -H 'content-type: application/json' \
     -d @examples/sample_applicant.json

# 5. Tests (includes the no-leakage guard)
uv run --with pytest pytest -q
```

## Reproducibility

- Dependencies are pinned in `uv.lock`; `random_seed` is set in `config.yaml`.
- `uv run train` regenerates `artifacts/model_ready.parquet` and
  `artifacts/model.joblib` deterministically.
- The notebooks import the **same** `src` code the pipeline uses, so they cannot
  drift from the production path.

## Design highlights

- **Leakage discipline is enforced, not assumed** — outcome columns are excluded
  from `FEATURE_COLUMNS` and a unit test fails the build if one slips in; the
  bureau join is strictly point-in-time.
- **No train/serve skew** — cleaning + the fitted pipeline are shared code
  between training and the `/predict` service.
- **Everything configurable** — target rule, cleaning bounds, split strategy,
  primary model, and threshold all live in `config.yaml`.
- **MLflow tracks *and* stores the models** — every run logs params, metrics, and
  the fitted pipeline; the primary model is registered to the MLflow Model
  Registry and tagged with a `champion` alias. The serving code loads from the
  self-contained `model.joblib` by default, or from the registry
  (`models:/…@champion`) by flipping `serving.model_source: registry` in the
  config — both carry the same threshold/reference metadata, so predictions match.
