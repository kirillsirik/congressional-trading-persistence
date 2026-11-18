"""The two units of the Pyun bridge: transaction LINES (Pyun's Table 3 unit) and our EVENTS.

Lines come from the canonical ``outputs/frames/timed_transactions.csv`` (read-only): one row per
reported transaction with the transaction date, the public date and the PERMNO. Vendor lines that
reproduce an official transaction carry ``dup_vendor`` from D's audit rows. Two anchors per clock:
Pyun's (close of the transaction day t / of the filing day d: the last session on or before the date,
returns from the next session) and ours (first session on or after t / strictly after d, returns from
the session after that). Events are D's tables (``events_T.parquet`` episodes, ``events_P.parquet``
non-mixed signals) joined to the released signals frame for the signal's own transaction anchor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data.calendar import load_calendar
from trading_on_disclosure_v4.estimators import event_frames as evf
from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import panel
from trading_on_disclosure_v4.pyun.returns import last_session_on_or_before, session_position

HOUSEHOLD = "filer_uid"
LINES_PATH = panel.FRAMES / "timed_transactions.csv"
SIGNALS_PATH = panel.FRAMES / "signals.csv"
AUDIT_ROWS = ef.EVENTS_DIR / "duplicate_audit_rows.parquet"
PERIOD = (pd.Timestamp("2015-01-01"), pd.Timestamp("2025-12-31"))
LAG_RANGE = (1, 365)
LINE_COLS = ["transaction_uid", HOUSEHOLD, "document_uid", "permno", "direction", "record_source",
             "trade_date_analysis", "analysis_public_date", "analysis_public_date_is_proxy",
             "follower_entry_date", "transaction_benchmark_entry_date", "chamber", "owner", "amount_bracket"]
DATE_COLS = ("trade_date_analysis", "analysis_public_date", "follower_entry_date", "transaction_benchmark_entry_date")


def load_lines(path: Path = LINES_PATH, audit_rows: Path = AUDIT_ROWS,
               calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Directional lines with a PERMNO, both dates, the lag, the vendor-duplicate flag and four anchors."""
    calendar = load_calendar() if calendar is None else calendar
    t = pd.read_csv(path, low_memory=False, usecols=LINE_COLS)
    t = t[t["direction"].isin(["purchase", "sale"]) & t["permno"].notna()].copy()
    t["permno"] = t["permno"].astype("int64")
    for c in DATE_COLS:
        t[c] = pd.to_datetime(t[c])
    dup = pd.read_parquet(audit_rows, columns=["transaction_uid", "dup_primary"])
    t = t.merge(dup.rename(columns={"dup_primary": "dup_vendor"}), on="transaction_uid", how="left")
    t["dup_vendor"] = t["dup_vendor"].eq(True)
    return add_anchors(t, calendar)


def add_anchors(t: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Lag in calendar days, filing month, and the four anchor positions (-1 = none)."""
    t = t.copy()
    t["lag_days"] = (t["analysis_public_date"] - t["trade_date_analysis"]).dt.days
    t["filing_month_pyun"] = t["analysis_public_date"].dt.to_period("M").astype(str)
    t["a_T_pyun"] = last_session_on_or_before(t["trade_date_analysis"], calendar)
    t["a_P_pyun"] = last_session_on_or_before(t["analysis_public_date"], calendar)
    t["a_T_ours"] = session_position(t["transaction_benchmark_entry_date"], calendar)
    t["a_P_ours"] = session_position(t["follower_entry_date"], calendar)
    return t.reset_index(drop=True)


def in_period(dates: pd.Series, period=PERIOD) -> pd.Series:
    return (dates >= period[0]) & (dates <= period[1])


def paired_lines(t: pd.DataFrame, period=PERIOD, lag_range=LAG_RANGE) -> pd.DataFrame:
    """Pyun's two-date sample: transaction and public date inside the period, lag 1-365 calendar days."""
    keep = in_period(t["trade_date_analysis"], period) & in_period(t["analysis_public_date"], period)
    keep &= t["lag_days"].between(lag_range[0], lag_range[1])
    return t.loc[keep].reset_index(drop=True)


def select_record(t: pd.DataFrame, record: str) -> pd.DataFrame:
    """all: every line; official: official-source lines; dedup: official + vendor lines that do not
    reproduce an official transaction (the line-level form of the plan's duplicate rule)."""
    if record == "all":
        return t
    if record == "official":
        return t.loc[t["record_source"] == "official"]
    if record == "dedup":
        return t.loc[~t["dup_vendor"]]
    raise ValueError(f"unknown record {record!r}")


def load_signal_events(signals_path: Path = SIGNALS_PATH, calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Non-mixed signals (the P unit) with both anchors, the plan's duplicate flag and the line list.
    Signals whose public anchor lies beyond the calendar end (no P event) are kept with ``a_P_ours`` = -1:
    the released transaction-clock rows keep them."""
    calendar = load_calendar() if calendar is None else calendar
    cols = ["signal_uid", HOUSEHOLD, "permno", "direction", "signal_direction", "analysis_public_date",
            "follower_entry_date", "transaction_benchmark_entry_date", "record_source", "parent_transaction_uids",
            "txn_date_max"]
    s = pd.read_csv(signals_path, low_memory=False, usecols=cols)
    s = s[s["signal_direction"].isin(["purchase", "sale"])].copy()
    for c in ("analysis_public_date", "follower_entry_date", "transaction_benchmark_entry_date", "txn_date_max"):
        s[c] = pd.to_datetime(s[c])
    flags = evf._dup_flags("P").assign(signal_uid=lambda f: f["event_id"].str.slice(2))
    s = s.merge(flags[["signal_uid", "dup_all_lines"]], on="signal_uid", how="left")
    s["dup_all_lines"] = s["dup_all_lines"].eq(True)
    s["permno"] = s["permno"].astype("int64")
    s["a_P_ours"] = session_position(s["follower_entry_date"], calendar)
    s["a_T_ours"] = session_position(s["transaction_benchmark_entry_date"], calendar)
    s["filing_month_pyun"] = s["analysis_public_date"].dt.to_period("M").astype(str)
    s["household_id"] = s[HOUSEHOLD].astype(str)
    return s.reset_index(drop=True)


def load_episode_events(calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """D's transaction-clock episodes (the T unit) with the anchor position and the line list."""
    calendar = load_calendar() if calendar is None else calendar
    ev = pd.read_parquet(ef.EVENTS_DIR / "events_T.parquet",
                         columns=["event_id", "household_id", "permno", "anchor_date", "direction", "period",
                                  "anchor_year", "record_source", "transaction_uids"])
    ev["a_T_ours"] = session_position(ev["anchor_date"], calendar)
    return ev.reset_index(drop=True)


def events_with_lines(events: pd.DataFrame, line_uids: set, uid_col: str) -> pd.Series:
    """Boolean: the event contains at least one of ``line_uids`` (its ``uid_col`` is ';'-joined)."""
    ex = events[[uid_col]].assign(_u=events[uid_col].astype(str).str.split(";")).explode("_u")
    hit = ex["_u"].isin(line_uids).groupby(level=0).any()
    return hit.reindex(events.index).fillna(False).astype(bool)


def signal_period_ab(s: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
    """The plan's period label of a P-unit row from its public anchor (A, B or X)."""
    from trading_on_disclosure_v4.data.calendar import window_end
    from trading_on_disclosure_v4.data.events import assign_period

    return assign_period(s["follower_entry_date"], window_end(s["follower_entry_date"], calendar))


def block_of(anchor_idx: np.ndarray, L: int = 20) -> np.ndarray:
    return np.asarray(anchor_idx, dtype=int) // L
