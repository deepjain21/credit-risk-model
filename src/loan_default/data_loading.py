"""Read the raw source files into DataFrames.

I/O only — no cleaning (cleaning.py owns that, so the same cleaning can run at
scoring time). Each source has its own small loader so you can pull just one
file; ``load_all()`` composes the ones the model pipeline uses (repayments is
excluded as a leakage source). ``cfg`` is optional on every loader, so
``load_servicing()`` works on its own with no setup.

Each source path is resolved in load_config (relative to the repo root), so a
loader just reads cfg["sources"][name] directly.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import load_config


def load_applications(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(cfg["sources"]["applications"])


def load_customers(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    with open(cfg["sources"]["customers"]) as f:      # customers is a JSON array
        return pd.DataFrame(json.load(f))


def load_bureau(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_json(cfg["sources"]["bureau"], lines=True)     # JSON-lines


def load_repayments(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    # Post-origination performance — a leakage source, so the model pipeline
    # never loads it (excluded from load_all). Kept for EDA / target analysis.
    cfg = cfg or load_config()
    return pd.read_csv(cfg["sources"]["repayments"])


def load_servicing(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(cfg["sources"]["servicing"])


def load_products(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(cfg["sources"]["products"])


def load_branches(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    return pd.read_csv(cfg["sources"]["branches"])


def load_all(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Load the sources the model pipeline uses, keyed by name.

    `repayments` is intentionally excluded — it describes post-origination
    performance (a leakage source) and no feature is built from it; load it
    explicitly via load_repayments for EDA.
    """
    cfg = cfg or load_config()
    return {
        "applications": load_applications(cfg),
        "customers": load_customers(cfg),
        "bureau": load_bureau(cfg),
        "servicing": load_servicing(cfg),
        "products": load_products(cfg),
        "branches": load_branches(cfg),
    }
