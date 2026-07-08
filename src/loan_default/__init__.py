"""Personal-loan default-risk package.

Modules mirror the training lifecycle:
    config        -> load YAML settings
    data_loading  -> read each raw source into a DataFrame
    cleaning      -> normalise messy formats (dates, categories, income)
    dataset       -> join sources, build point-in-time features, attach target
    feature_engineering -> row-wise derived features (ratios, DBR)
    features      -> sklearn preprocessing pipeline (fit at scoring time too)
    finance       -> DBR / amortised-installment maths (shared train + serve)
    tune          -> Optuna hyperparameter search for all three models
    train         -> fit/compare models, log to MLflow, save the configured primary
    predict       -> load the saved model and score a new applicant (with reasons)
    serve         -> FastAPI form UI + /predict + /health
"""

__version__ = "0.1.0"
