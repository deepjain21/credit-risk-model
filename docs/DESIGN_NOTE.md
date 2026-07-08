# Design Note — Serving & Monitoring

One page on how the model is served today, and how I would run and monitor it in
production. The brief weights **design over infrastructure**, so this favours the
reasoning over a full deployment.

## What ships in this repo

A minimal, working prediction service:

```
POST /predict   -> score one applicant   (src/loan_default/serve.py, FastAPI)
GET  /health    -> liveness + which model/version is loaded
```

The endpoint accepts the fields a loan officer has **at the decision point**
(application + customer master + the bureau pull just fetched), rebuilds the
model features with the *same* cleaning code used in training, and returns:

```json
{ "default_probability": 0.53, "decision": "decline", "risk_band": "very_high",
  "top_factors": [
    { "feature": "Credit bureau score", "value": 640, "direction": "increases", "impact": 0.37 },
    { "feature": "Employment type", "value": "Salaried", "direction": "decreases", "impact": 0.05 }
  ],
  "threshold": 0.5, "model_name": "logistic_regression", "model_trained_at": "..." }
```

`top_factors` are the per-applicant reason codes — logistic coefficients (`coef ×
value`) for the served linear model, or XGBoost SHAP contributions if that model
is promoted. There is also a simple **form UI at `GET /`** for interactive scoring.

### The one design decision that matters most: no train/serve skew
Feature construction is **shared code**, not reimplemented for serving.
`cleaning.py` (date parsing, category normalisation, winsorising) and the fitted
`ColumnTransformer` are used in both `dataset.py` (training) and `predict.py`
(serving). The model artifact bundles the fitted pipeline **plus** the
product/branch reference tables and the decision threshold, so scoring needs only
the raw payload. This is what makes "the features are produced the same way at
scoring time" true rather than aspirational.

### Point-in-time correctness at serving
The service is handed the bureau pull that exists at decision time. It never
looks up "the latest pull", mirroring the point-in-time join used in training.

## How I would productionise it

| Concern | This stub | Production |
|---|---|---|
| Model registry | `joblib` file, **plus** MLflow Model Registry with a `champion` alias (already wired) | add approval/governance gates and CI promotion around the alias move |
| Feature parity | shared Python module | same module packaged; bureau/customer fetched via the same services used offline |
| Serving | single FastAPI process | containerised, autoscaled, behind the loan-origination system |
| Validation | Pydantic schema | + range/business-rule checks, reject-and-log on bad payloads |
| Auditability | response echoes model version | log every request/response + features + model version for regulatory review |
| Threshold | config value | governed by risk policy, versioned alongside the model |

## Monitoring — the hard part: outcomes take months to mature

A default label is only known 6–18 months after origination, so we **cannot wait
for ground truth** to know if the model is healthy. The monitoring strategy is
therefore layered from fast/proxy to slow/truth:

1. **Operational (real-time).** Latency, error rate, payload-schema failures,
   missing-feature rate (e.g. share of applicants with no bureau pull). Alert on
   spikes.

2. **Input drift (daily).** Population Stability Index (PSI) on each feature vs
   the training distribution — especially `bureau_score`, income, DTI, product
   mix. PSI > 0.2 on a key feature triggers investigation. This catches the
   world changing before any outcomes exist.

3. **Score/prediction drift (daily).** Distribution of `default_probability` and
   the approve/decline rate vs training. A sudden shift means either input drift
   or a broken feature pipeline.

4. **Early-outcome proxies (weekly/monthly).** Don't wait for final default —
   track early-delinquency signals (first-payment default, 30 DPD at month 3) by
   score band. If high-score loans start going 30 DPD, the ranking is decaying.

5. **Full performance (as loans mature).** On a rolling vintage basis, recompute
   ROC-AUC / KS / calibration on cohorts that have matured. Compare actual vs
   predicted default rate per score band (calibration curve).

6. **Fairness / stability.** Track approval and default rates across
   emirate / nationality_group / employment_type to catch disparate impact and
   segment-level decay.

### Retraining trigger
Retrain when: sustained input PSI breach on key features, early-outcome proxies
degrade by score band, or matured-vintage KS drops below an agreed floor.
Retraining reuses the exact `train.py` pipeline; the new model is registered,
shadow-scored against production, and promoted only if it wins on out-of-time
metrics — never auto-promoted.

## Known limitations of the stub
- No authentication, rate-limiting, or persistent request logging.
- Single-record scoring only (batch endpoint would be added for portfolio runs).
- Threshold is a static config value, not yet wired to an expected-cost policy.
