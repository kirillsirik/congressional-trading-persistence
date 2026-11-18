"""Outcome-free loaders for the canonical Act I frames in ``outputs/frames``.

Every loader drops ``ar_*``/``car_*`` columns at read time (``usecols``) and
passes the result through :func:`assert_no_outcome_columns`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMES_DIR = REPO_ROOT / "outputs" / "frames"
OUTCOME_PREFIXES = ("ar_", "car_")

TRANSACTION_DATE_COLS = (
    "official_public_date",
    "capitol_trades_publication_date_raw",
    "trade_date_official",
    "trade_date_capitol_trades_raw",
    "analysis_public_date",
    "trade_date_analysis",
    "transaction_date_reported",
)
TIMED_DATE_COLS = TRANSACTION_DATE_COLS + (
    "follower_entry_date",
    "transaction_benchmark_entry_date",
    "entry_raw_ct_date",
    "entry_plus1_date",
    "transaction_raw_entry_date",
)
SIGNAL_DATE_COLS = (
    "analysis_public_date",
    "txn_date_min",
    "txn_date_max",
    "follower_entry_date",
    "transaction_benchmark_entry_date",
    "entry_raw_ct_date",
    "entry_plus1_date",
    "transaction_raw_entry_date",
)
STRING_COLS = (
    "transaction_uid",
    "filer_uid",
    "document_uid",
    "signal_uid",
    "parent_transaction_uids",
    "amount_bracket",
    "amount_brackets",
    "ticker_text",
    "owner",
    "chamber",
)


class OutcomeColumnError(RuntimeError):
    """An outcome column reached a v4 data object before the tag."""


def is_outcome_column(name: str) -> bool:
    return name.startswith(OUTCOME_PREFIXES)


def assert_no_outcome_columns(frame: pd.DataFrame, where: str = "frame") -> pd.DataFrame:
    """Raise :class:`OutcomeColumnError` if any ``ar_``/``car_`` column is present."""
    bad = [c for c in frame.columns if is_outcome_column(str(c))]
    if bad:
        raise OutcomeColumnError(f"{where} carries outcome columns: {bad[:5]}")
    return frame


def normalize_chamber(values: pd.Series) -> pd.Series:
    """Lower-case chamber labels ('House' and 'house' both -> 'house')."""
    return values.astype("string").str.strip().str.lower()


def _header(path: Path) -> list[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh))


def _safe_usecols(path: Path, requested: tuple[str, ...] | None) -> list[str]:
    """Columns to read: the header minus outcome columns, or ``requested`` if it is clean.

    Raises :class:`OutcomeColumnError` BEFORE any read when an explicit request
    names an ``ar_``/``car_`` column, so no outcome value is ever loaded.
    """
    header = _header(path)
    if requested is None:
        return [c for c in header if not is_outcome_column(c)]
    bad = sorted(c for c in requested if is_outcome_column(c))
    if bad:
        raise OutcomeColumnError(f"{path.name}: requested outcome columns {bad[:5]}")
    missing = sorted(set(requested) - set(header))
    if missing:
        raise KeyError(f"{path.name} lacks columns {missing}")
    wanted = set(requested)
    return [c for c in header if c in wanted]


def read_frame(
    name: str,
    usecols: tuple[str, ...] | None = None,
    date_cols: tuple[str, ...] = (),
    frames_dir: Path = FRAMES_DIR,
) -> pd.DataFrame:
    """Read ``outputs/frames/<name>.csv`` without outcome columns."""
    path = frames_dir / f"{name}.csv"
    keep = _safe_usecols(path, usecols)
    dtype = {c: "string" for c in STRING_COLS if c in keep}
    frame = pd.read_csv(path, usecols=keep, dtype=dtype, low_memory=False)
    for col in date_cols:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    if "chamber" in frame.columns:
        frame["chamber"] = normalize_chamber(frame["chamber"])
    if "permno" in frame.columns:
        frame["permno"] = frame["permno"].astype("Int64")
    return assert_no_outcome_columns(frame, where=path.name)


SIGNAL_COLS = (
    "filer_uid",
    "document_uid",
    "permno",
    "direction",
    "analysis_public_date",
    "lot_count",
    "parent_transaction_uids",
    "txn_date_min",
    "txn_date_max",
    "amount_brackets",
    "analysis_public_date_source",
    "analysis_public_date_is_proxy",
    "signal_uid",
    "signal_direction",
    "mixed_direction",
    "direction_conflict",
    "follower_entry_date",
    "transaction_benchmark_entry_date",
    "record_source",
    "reporting_lag_days",
    "chamber",
)


def load_signals(frames_dir: Path = FRAMES_DIR) -> pd.DataFrame:
    """Public-clock signal frame (52,018 rows in the canonical release), no outcomes."""
    return read_frame("signals", SIGNAL_COLS, SIGNAL_DATE_COLS, frames_dir)


def load_timed_transactions(frames_dir: Path = FRAMES_DIR) -> pd.DataFrame:
    """PERMNO-matched, amendment-deduplicated, directional transactions with entry dates."""
    return read_frame("timed_transactions", None, TIMED_DATE_COLS, frames_dir)


def load_matched_transactions(frames_dir: Path = FRAMES_DIR) -> pd.DataFrame:
    return read_frame("matched_transactions", None, TRANSACTION_DATE_COLS, frames_dir)


def load_deduplicated_transactions(frames_dir: Path = FRAMES_DIR) -> pd.DataFrame:
    return read_frame("deduplicated_transactions", None, TRANSACTION_DATE_COLS, frames_dir)


def load_attrition_ledger(repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    return pd.read_csv(repo_root / "outputs" / "canonical" / "attrition_ledger.csv")
