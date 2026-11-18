"""Event tables for both clocks (shared schema; D produces, N/E consume).

Transaction clock (T): one episode per (household, PERMNO, reported transaction
date, direction) built from the canonical ``timed_transactions`` frame, whose
amendment suppression and PERMNO matching are reused, not re-derived.
Public clock (P): the canonical non-MIXED signal, anchored at ``follower_entry_date``.

``lag_days`` has ONE definition on both clocks: the minimum over the event's
lines of (public date - transaction date) in calendar days, i.e. the lag of the
earliest-disclosed line, and only when the public date is officially observed
(``analysis_public_date_source == OFFICIAL_OBSERVED``). On T the lines share the
trade date, so this is the earliest official public date minus the trade date;
on P the lines share the public date, so this is the public date minus
``txn_date_max``. Values are stored raw (negative and > 365 kept; counted by
:func:`lag_days_report`); A8 filters to 1-365 downstream.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data.calendar import HORIZON, first_session_on_or_after, window_end
from trading_on_disclosure_v4.data.load import assert_no_outcome_columns, normalize_chamber

EVENT_COLUMNS = [
    "event_id",
    "household_id",
    "permno",
    "clock",
    "anchor_date",
    "direction",
    "period",
    "anchor_year",
    "anchor_quarter",
    "window_end_date",
    "complete_window_h20",
    "record_source",
    "chamber",
    "amount_bracket",
    "lag_days",
    "n_lines",
    "signal_uid",
    "transaction_uids",
]
PERIOD_A = (pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-31"))
PERIOD_B = (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))
OFFICIAL_OBSERVED = "OFFICIAL_OBSERVED"
T_KEY = ["filer_uid", "permno", "trade_date_analysis", "direction"]


def assign_period(anchor: pd.Series, window_end_date: pd.Series) -> pd.Series:
    """'A', 'B' or 'X' by the plan's rule (window must end inside the period)."""
    anchor = pd.to_datetime(anchor)
    end = pd.to_datetime(window_end_date)
    complete = end.notna()
    in_a = (anchor >= PERIOD_A[0]) & (anchor <= PERIOD_A[1]) & complete & (end <= PERIOD_A[1])
    in_b = (anchor >= PERIOD_B[0]) & (anchor <= PERIOD_B[1]) & complete
    return pd.Series(np.where(in_a, "A", np.where(in_b, "B", "X")), index=anchor.index, dtype="object")


def assign_parity_period(anchor: pd.Series, window_end_date: pd.Series) -> pd.Series:
    """Secondary split: 'odd'/'even' calendar year, 'purged' when the window crosses a year end."""
    anchor = pd.to_datetime(anchor)
    end = pd.to_datetime(window_end_date)
    crosses = end.isna() | (end.dt.year != anchor.dt.year)
    parity = np.where(anchor.dt.year % 2 == 1, "odd", "even")
    return pd.Series(np.where(crosses, "purged", parity), index=anchor.index, dtype="object")


def _event_id(prefix: str, parts: pd.DataFrame) -> pd.Series:
    joined = parts.astype(str).agg("|".join, axis=1)
    return prefix + joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:16])


def _finish(frame: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add window/period columns, enforce dtypes and column order."""
    frame = frame.copy()
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"])
    frame["window_end_date"] = window_end(frame["anchor_date"], calendar, HORIZON)
    frame["complete_window_h20"] = frame["window_end_date"].notna()
    frame["period"] = assign_period(frame["anchor_date"], frame["window_end_date"])
    frame["anchor_year"] = frame["anchor_date"].dt.year.astype("int16")
    frame["anchor_quarter"] = frame["anchor_date"].dt.to_period("Q").astype(str)
    frame["permno"] = frame["permno"].astype("int64")
    frame["chamber"] = normalize_chamber(frame["chamber"]).astype(object)
    frame["n_lines"] = frame["n_lines"].astype("int64")
    frame["lag_days"] = frame["lag_days"].astype("float64")
    for col in ("event_id", "household_id", "clock", "direction", "period", "record_source"):
        frame[col] = frame[col].astype(str)
    for col in ("amount_bracket", "signal_uid", "transaction_uids"):
        frame[col] = frame[col].astype(object).where(frame[col].notna(), None)
    out = frame[EVENT_COLUMNS].sort_values(["anchor_date", "household_id", "permno", "direction"])
    out = out.reset_index(drop=True)
    if not out["event_id"].is_unique:
        raise ValueError("event_id is not unique")
    return assert_no_outcome_columns(out, where="events")


def _observed_lag_days(frame: pd.DataFrame, txn_col: str) -> pd.Series:
    """Public date minus ``txn_col`` in calendar days; NaN unless the public date is observed."""
    observed = frame["analysis_public_date_source"].eq(OFFICIAL_OBSERVED)
    lag = (frame["analysis_public_date"] - frame[txn_col]).dt.days.astype("float64")
    return lag.where(observed, np.nan)


def _join_unique(values: pd.Series) -> str | None:
    vals = sorted({str(v) for v in values if pd.notna(v)})
    return ";".join(vals) if vals else None


def build_events_T(timed: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Transaction-clock episodes from the canonical timed frame; returns (events, drop counts)."""
    assert_no_outcome_columns(timed, where="timed_transactions")
    directional = timed["direction"].isin(["purchase", "sale"])
    has_permno = timed["permno"].notna()
    report = {
        "lines_in": int(len(timed)),
        "non_directional_dropped": int((~directional).sum()),
        "null_permno_dropped": int((directional & ~has_permno).sum()),
    }
    rows = timed.loc[directional & has_permno].copy()
    report["lines_kept"] = int(len(rows))
    rows["permno"] = rows["permno"].astype("int64")
    rows["trade_date_analysis"] = pd.to_datetime(rows["trade_date_analysis"])
    rows["lag_obs"] = _observed_lag_days(rows, "trade_date_analysis")
    rows["is_official"] = rows["record_source"].eq("official")
    rows["chamber"] = normalize_chamber(rows["chamber"])
    rows = rows.sort_values(T_KEY + ["analysis_public_date", "transaction_uid"])
    grouped = rows.groupby(T_KEY, sort=True, dropna=False)
    ep = grouped.agg(
        n_lines=("transaction_uid", "size"),
        transaction_uids=("transaction_uid", _join_unique),
        amount_bracket=("amount_bracket", _join_unique),
        chamber=("chamber", "first"),
        lag_days=("lag_obs", "min"),
        n_official=("is_official", "sum"),
    ).reset_index()
    # official takes precedence when lines from both sources fall in one episode
    ep["record_source"] = np.where(ep["n_official"] > 0, "official", "capitol_trades")
    ep = ep.rename(columns={"filer_uid": "household_id"})
    ep["anchor_date"] = first_session_on_or_after(ep["trade_date_analysis"], calendar)
    if ep["anchor_date"].isna().any():
        raise ValueError("transaction date beyond the calendar end")
    ep["clock"] = "T"
    ep["signal_uid"] = None
    ep["event_id"] = _event_id("T:", ep[["household_id", "permno", "trade_date_analysis", "direction"]])
    out = _finish(ep, calendar)
    report["episodes"] = int(len(out))
    report["multi_line_episodes"] = int((out["n_lines"] > 1).sum())
    return out, report


def build_events_P(signals: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Public-clock events = non-MIXED canonical signals; returns (events, drop counts)."""
    assert_no_outcome_columns(signals, where="signals")
    mixed = signals["mixed_direction"]
    if mixed.dtype != bool and not set(mixed.dropna().unique()) <= {True, False}:
        raise TypeError(f"mixed_direction must be boolean, got dtype {mixed.dtype}")
    keep = ~mixed.astype(bool)
    rows = signals.loc[keep].copy()
    no_anchor = rows["follower_entry_date"].isna()
    report = {
        "signals_in": int(len(signals)),
        "mixed_direction_excluded": int((~keep).sum()),
        "no_follower_entry_date_dropped": int(no_anchor.sum()),
    }
    rows = rows.loc[~no_anchor].copy()
    if not rows["direction"].eq(rows["signal_direction"]).all():
        raise ValueError("direction differs from signal_direction on a non-MIXED signal")
    # min lag over the signal's lots = public date minus the latest transaction date
    rows["lag_days"] = _observed_lag_days(rows, "txn_date_max")
    out = pd.DataFrame(
        {
            "event_id": "P:" + rows["signal_uid"].astype(str),
            "household_id": rows["filer_uid"].astype(str),
            "permno": rows["permno"].astype("int64"),
            "clock": "P",
            "anchor_date": rows["follower_entry_date"],
            "direction": rows["direction"].astype(str),
            "record_source": rows["record_source"].astype(str),
            "chamber": rows["chamber"],
            "amount_bracket": rows["amount_brackets"],
            "lag_days": rows["lag_days"],
            "n_lines": rows["lot_count"],
            "signal_uid": rows["signal_uid"].astype(object),
            "transaction_uids": None,
        }
    )
    return _finish(out, calendar), report


def lag_days_report(events: pd.DataFrame) -> dict:
    """Counts of stored ``lag_days`` values by clock: non-null, negative, zero, over 365."""
    out = {}
    for clock, sub in events.groupby("clock"):
        lag = sub["lag_days"]
        out[str(clock)] = {
            "non_null": int(lag.notna().sum()),
            "negative": int((lag < 0).sum()),
            "zero": int((lag == 0).sum()),
            "over_365": int((lag > 365).sum()),
            "in_1_to_365": int(lag.between(1, 365).sum()),
        }
    return out
