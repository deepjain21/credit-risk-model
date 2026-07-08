"""Load the project configuration from config.yaml.

One ``load_config()`` call returns the settings as a plain dict, with every path
resolved to absolute so the code runs the same from a notebook, a script, or the
serving process (no "works only from this folder" surprises).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root, found relative to this file (src/loan_default/config.py), so paths
# resolve correctly no matter which folder the process was launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read config.yaml and make every path under `paths:` absolute."""
    path = path or REPO_ROOT / "config" / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {name: REPO_ROOT / rel for name, rel in cfg["paths"].items()}
    # sources are repo-relative paths too (they carry their own "data/" prefix).
    cfg["sources"] = {name: REPO_ROOT / rel for name, rel in cfg["sources"].items()}

    # Anchor the MLflow sqlite DB to the repo root so `tune`/`train`/serving all
    # read & write the SAME mlflow.db regardless of the working directory.
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        cfg["mlflow"]["tracking_uri"] = f"sqlite:///{REPO_ROOT / uri[len('sqlite:///'):]}"
    return cfg
