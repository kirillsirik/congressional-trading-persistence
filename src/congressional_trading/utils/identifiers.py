"""Stable record keys and content hashes for scraped trade records."""

from __future__ import annotations

import hashlib

import pandas as pd

KEY_FIELDS = [
    "politician_name",
    "ticker_clean",
    "issuer_name",
    "trade_date",
    "published_date",
    "transaction_type_clean",
    "owner_clean",
    "size_raw",
]


def _norm(val: object) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip().lower()


def stable_record_key(row: pd.Series) -> str:
    """Deterministic key identifying a disclosure record across scrape runs."""
    joined = "|".join(_norm(row.get(f)) for f in KEY_FIELDS)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()

def row_content_hash(row: pd.Series, fields: list[str]) -> str:
    """Hash of the full normalized content, used to detect amended records."""
    joined = "|".join(_norm(row.get(f)) for f in sorted(fields))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def add_record_keys(df: pd.DataFrame, content_fields: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    fields = content_fields or [c for c in df.columns if not c.startswith("_")]
    df["stable_record_key"] = df.apply(stable_record_key, axis=1)
    df["row_content_hash"] = df.apply(lambda r: row_content_hash(r, fields), axis=1)
    return df
