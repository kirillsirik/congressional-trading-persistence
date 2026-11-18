"""Fama-French factor loading.

Two sources are supported:

1. :func:`load_factors_daily` — the pre-merged ``factors_daily.csv``
   (``date, MktRF, SMB, HML, RMW, CMA, RF, MOM``, already in decimals).
2. :func:`build_factors_from_french` — raw Ken French library CSVs (FF5 daily
   + momentum daily). Preamble rows are skipped by locating the header line,
   dates are ``YYYYMMDD`` integers, and values are ALWAYS divided by 100
   (Ken French daily files are always in percent — no magnitude heuristics).

Canonical output columns: ``date, mkt_rf, smb, hml, rmw, cma, mom, rf``
(decimals), sorted by date.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.validation import require_columns

logger = logging.getLogger(__name__)

FACTOR_COLUMNS = ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]

# Pre-merged factors_daily.csv column names -> canonical names.
PREMERGED_RENAME = {
    "MktRF": "mkt_rf", "Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
    "RMW": "rmw", "CMA": "cma", "RF": "rf", "MOM": "mom", "Mom": "mom",
}

MAX_PLAUSIBLE_ABS_DAILY = 0.3   # |daily factor| above this fails validation
GAP_REPORT_BUSINESS_DAYS = 5    # coverage gaps longer than this are reported

_YYYYMMDD_RE = re.compile(r"^\s*\d{8}\s*$")


def load_factors_daily(path: Path) -> pd.DataFrame:
    """Load the pre-merged daily factor file (decimal units) canonically."""
    path = Path(path)
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in PREMERGED_RENAME.items() if k in df.columns})
    df.columns = [c.strip().lower() for c in df.columns]
    require_columns(df, FACTOR_COLUMNS, context=str(path))
    df["date"] = pd.to_datetime(df["date"])
    for col in FACTOR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[FACTOR_COLUMNS].sort_values("date").reset_index(drop=True)
    logger.info(
        "Factors: %d days, %s to %s", len(df), df["date"].min().date(), df["date"].max().date()
    )
    return df


def _read_french_csv(path: Path) -> pd.DataFrame:
    """Read a raw Ken French daily CSV: skip preamble, keep YYYYMMDD rows.

    The header row is located as the first line that names at least one factor
    column; data rows are those whose first field is an 8-digit date. Annual /
    monthly appendix sections (non-8-digit first field) are dropped. Values
    are converted from percent to decimal by dividing by 100 unconditionally.
    """
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    header_idx = None
    factor_tokens = {"mkt-rf", "smb", "hml", "rmw", "cma", "mom", "umd", "rf"}
    for i, line in enumerate(lines):
        fields = [t.strip().lower() for t in line.split(",")]
        if any(t in factor_tokens for t in fields):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not locate a factor header row in {path}")

    df = pd.read_csv(path, skiprows=header_idx, low_memory=False)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].astype(str)
    df = df[df["date"].str.match(_YYYYMMDD_RE.pattern)].copy()
    df["date"] = pd.to_datetime(df["date"].str.strip(), format="%Y%m%d")

    df.columns = [str(c).strip().lower().replace("-", "_") for c in df.columns]
    for col in df.columns:
        if col == "date":
            continue
        # Percent -> decimal, always. -99.99 / -999 are Ken French missing codes.
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals.where(vals > -99.0, np.nan)
        df[col] = vals / 100.0
    return df.sort_values("date").reset_index(drop=True)


def build_factors_from_french(ff5_path: Path, mom_path: Path) -> pd.DataFrame:
    """Merge raw FF5-daily and momentum-daily files into the canonical frame."""
    ff5 = _read_french_csv(ff5_path)
    require_columns(ff5, ["date", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"], context=str(ff5_path))
    mom = _read_french_csv(mom_path)
    if "mom" not in mom.columns and "umd" in mom.columns:
        mom = mom.rename(columns={"umd": "mom"})
    require_columns(mom, ["date", "mom"], context=str(mom_path))

    merged = ff5.merge(mom[["date", "mom"]], on="date", how="inner")
    merged = merged[FACTOR_COLUMNS].sort_values("date").reset_index(drop=True)
    logger.info(
        "Built factors from Ken French files: %d days, %s to %s",
        len(merged), merged["date"].min().date(), merged["date"].max().date(),
    )
    return merged


def validate_factors(
    factors: pd.DataFrame,
    max_abs_daily: float = MAX_PLAUSIBLE_ABS_DAILY,
) -> dict:
    """Validate plausibility and coverage; raise on implausible magnitudes.

    Returns a coverage report dict: date span, row count, and any gaps longer
    than :data:`GAP_REPORT_BUSINESS_DAYS` business days (list of
    ``(gap_start, gap_end, business_days)``).
    """
    require_columns(factors, FACTOR_COLUMNS, context="validate_factors")
    value_cols = FACTOR_COLUMNS[1:]

    extreme = factors[value_cols].abs().gt(max_abs_daily)
    if bool(extreme.any().any()):
        bad = factors.loc[extreme.any(axis=1), ["date", *value_cols]]
        raise ValueError(
            f"Implausible daily factor magnitudes (|x| > {max_abs_daily}); "
            f"likely a percent/decimal unit error. Offending rows:\n{bad.head()}"
        )

    dates = pd.to_datetime(factors["date"]).sort_values().reset_index(drop=True)
    d64 = dates.to_numpy().astype("datetime64[D]")
    gaps = []
    if len(d64) > 1:
        bdays = np.busday_count(d64[:-1], d64[1:])
        for i in np.nonzero(bdays > GAP_REPORT_BUSINESS_DAYS)[0]:
            gaps.append((str(dates[i].date()), str(dates[i + 1].date()), int(bdays[i])))

    n_missing = int(factors[value_cols].isna().sum().sum())
    report = {
        "n_rows": int(len(factors)),
        "start": str(dates.iloc[0].date()) if len(dates) else None,
        "end": str(dates.iloc[-1].date()) if len(dates) else None,
        "n_missing_values": n_missing,
        "n_gaps_over_threshold": len(gaps),
        "gaps": gaps,
    }
    if gaps:
        logger.warning("Factor coverage gaps > %d business days: %s", GAP_REPORT_BUSINESS_DAYS, gaps)
    if n_missing:
        logger.warning("Factor frame has %d missing values", n_missing)
    return report
