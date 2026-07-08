"""Normalisers for the messy conventions across the source systems.

Each function is small, pure, and independently testable. They are used both
by the dataset builder (training) and by the serving code (scoring a single
applicant), so the SAME cleaning is applied in both places — no train/serve
skew.

Issues handled (see docs/DATA_QUALITY_REPORT.md for how each was detected):
  * dates in three formats: ISO (2021-04-15), day-first (31/05/2021) and
    Excel serial numbers (44547)
  * employment_type with 13 spellings collapsing to 3 real categories
  * emirate contaminated with acquisition-channel values (Web/Online) and
    written in mixed case / abbreviations
  * winsorising declared income to a plausible band (14k median, 50M outliers)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Excel's day-zero. Serial 44547 -> 2021-12-17. A plain Python datetime (not a
# pd.Timestamp) so the day offset below uses Python's timedelta and avoids a
# noisy NumPy-timedelta deprecation warning on every serial value.
_EXCEL_EPOCH = datetime(1899, 12, 30)


def parse_mixed_date(series: pd.Series) -> pd.Series:
    """Parse a column mixing ISO, day-first, and Excel-serial dates.

    One value at a time, which reads plainly: pure-digit strings are Excel
    serials (days since the Excel epoch); everything else goes to pandas with
    ``dayfirst=True``, which correctly reads ISO (YYYY-MM-DD, with or without a
    time), and UAE-style DD/MM/YYYY or DD-MM-YYYY. Unparseable values -> NaT.
    """
    def parse_one(val: object) -> object:
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        if s.isdigit():                                    # Excel serial (44547)
            return _EXCEL_EPOCH + timedelta(days=int(s))
        # format="mixed" lets each value use its own format (ISO / DMY) without
        # pandas re-inferring — and re-warning — on every single call.
        return pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")

    return pd.to_datetime(series.map(parse_one))


def normalise_employment_type(series: pd.Series) -> pd.Series:
    """Collapse the 13 raw spellings into {Salaried, Self-Employed, Unemployed}."""
    s = series.astype("string").str.lower().str.strip()

    def _map(v: object) -> object:
        if not isinstance(v, str):
            return pd.NA
        if "salar" in v or v == "ft-salaried":
            return "Salaried"
        if "self" in v or v in {"se", "self emp"}:
            return "Self-Employed"
        # match 'unemployed' / 'not employed' specifically — a bare "employ"
        # substring would also catch "employed"/"employee" and mislabel them.
        if "unemploy" in v or "not employ" in v:
            return "Unemployed"
        return pd.NA

    return s.map(_map).astype("string")


# Map the raw emirate strings to a canonical emirate. Values that are actually
# acquisition channels ("Web"/"Online") are NOT emirates -> flagged separately.
_EMIRATE_CANON = {
    "dubai": "Dubai", "dxb": "Dubai",
    "abu dhabi": "Abu Dhabi", "abu-dhabi": "Abu Dhabi",
    "abudhabi": "Abu Dhabi", "auh": "Abu Dhabi",
    "sharjah": "Sharjah", "shj": "Sharjah",
}
_CHANNEL_CONTAMINANTS = {"web", "online"}


def normalise_emirate(series: pd.Series) -> pd.Series:
    """Canonicalise emirate; channel-like contaminants become <NA> (unknown)."""
    s = series.astype("string").str.lower().str.strip()
    return s.map(lambda v: _EMIRATE_CANON.get(v, pd.NA) if isinstance(v, str) else pd.NA).astype("string")


def emirate_is_contaminated(series: pd.Series) -> pd.Series:
    """Boolean flag: the emirate field held a channel value, not a real emirate.

    Kept as a feature because *which channel leaked* can be informative and its
    presence marks a particular app-form/source combination.
    """
    s = series.astype("string").str.lower().str.strip()
    return s.isin(_CHANNEL_CONTAMINANTS).fillna(False)


def clean_income(series: pd.Series, lo: float, hi: float) -> pd.Series:
    """Winsorise declared monthly income to a plausible band [lo, hi].

    Income has entries up to 50,000,000 (annual figures, or trailing-zero typos)
    against a median near 14k. We cap rather than drop: the applicant is real,
    only the magnitude is implausible, and capping keeps the row usable.
    """
    x = pd.to_numeric(series, errors="coerce")
    return x.clip(lower=lo, upper=hi)


def age_from_dob(dob: pd.Series, as_of: pd.Series) -> pd.Series:
    """Age in years at the application date (both may be mixed-format dates)."""
    dob_parsed = parse_mixed_date(dob)
    as_of_parsed = as_of if pd.api.types.is_datetime64_any_dtype(as_of) else parse_mixed_date(as_of)
    age = (as_of_parsed - dob_parsed).dt.days / 365.25
    # Guard against nonsense ages from unparseable / swapped dates.
    return age.where((age >= 18) & (age <= 100))
