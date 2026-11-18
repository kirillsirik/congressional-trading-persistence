"""Two-clock timing derivation (TC-022; TIM-001..004, TIM-007).

Maps the two semantically distinct source dates into their entry dates on a
shared exchange calendar:

- public-action clock: ``follower_entry_date`` is the first trading day
  STRICTLY after ``official_public_date_proxy``;
- transaction clock: benchmark entry is the close on the reported transaction
  date, or the next trading day's close when that date is not a trading day.

The derived dates are used verbatim as cluster keys (TIM-007). Semantic
aliasing between the clock fields is rejected, never repaired (TIM-001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRANSACTION_DATE_COL = "transaction_date_reported"
PUBLIC_DATE_COL = "official_public_date_proxy"
FOLLOWER_ENTRY_COL = "follower_entry_date"
BENCHMARK_ENTRY_COL = "transaction_benchmark_entry_date"

_GENERIC_DATE_NAMES = frozenset({"event_date", "date"})


class TimingAliasError(RuntimeError):
    """A semantically distinct date field was aliased or is missing."""


def _day_reason(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    is_trading = dates.isin(calendar)
    is_weekend = dates.dt.dayofweek >= 5
    reason = np.where(
        is_trading, "trading_day", np.where(is_weekend, "weekend", "holiday_or_closure")
    )
    return pd.Series(reason, index=dates.index)


def _validate_calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(calendar, pd.DatetimeIndex) or len(calendar) == 0:
        raise ValueError("calendar must be a non-empty DatetimeIndex")
    if not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise ValueError("calendar must be sorted and duplicate-free")
    return calendar


def _next_trading_day(
    dates: pd.Series, calendar: pd.DatetimeIndex, *, strict: bool
) -> pd.Series:
    side = "right" if strict else "left"
    pos = calendar.searchsorted(pd.DatetimeIndex(dates), side=side)
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    in_range = pos < len(calendar)
    out.iloc[np.flatnonzero(in_range)] = calendar[pos[in_range]]
    out[dates.isna()] = pd.NaT
    return out


def derive_two_clock_entries(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    transaction_col: str = TRANSACTION_DATE_COL,
    public_col: str = PUBLIC_DATE_COL,
) -> pd.DataFrame:
    """Return ``frame`` plus entry dates and per-row mapping reasons."""
    calendar = _validate_calendar(calendar)
    if transaction_col == public_col:
        raise TimingAliasError(
            "transaction and public date columns must be distinct fields "
            f"(both {transaction_col!r})"
        )
    for role, col in (("transaction", transaction_col), ("public", public_col)):
        if col in _GENERIC_DATE_NAMES:
            raise TimingAliasError(
                f"{role} clock may not use the generic field name {col!r}"
            )
        if col not in frame.columns:
            raise TimingAliasError(f"required {role}-clock column {col!r} is missing")
    tx = pd.to_datetime(frame[transaction_col])
    pub = pd.to_datetime(frame[public_col])
    if np.shares_memory(tx.to_numpy(), pub.to_numpy()):
        raise TimingAliasError(
            "transaction and public date columns share memory; refusing aliased input"
        )

    out = frame.copy()
    out[FOLLOWER_ENTRY_COL] = _next_trading_day(pub, calendar, strict=True)
    out["follower_entry_reason"] = _day_reason(pub, calendar).radd(
        "strict_next_close_after_"
    )
    out[BENCHMARK_ENTRY_COL] = _next_trading_day(tx, calendar, strict=False)
    tx_reason = _day_reason(tx, calendar)
    out["transaction_benchmark_reason"] = np.where(
        tx_reason == "trading_day",
        "close_on_reported_transaction_date",
        "next_close_after_" + tx_reason,
    )
    for col, src in ((FOLLOWER_ENTRY_COL, pub), (BENCHMARK_ENTRY_COL, tx)):
        beyond = src.notna() & out[col].isna()
        reason_col = (
            "follower_entry_reason"
            if col == FOLLOWER_ENTRY_COL
            else "transaction_benchmark_reason"
        )
        out.loc[beyond, reason_col] = "beyond_calendar_end"
    return out


def entry_strictly_after(
    dates: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """First trading day STRICTLY after each date (public-action entry rule)."""
    return _next_trading_day(pd.to_datetime(dates), _validate_calendar(calendar), strict=True)


def close_on_or_next(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Close on the date itself when it trades, else the next trading close
    (transaction-clock benchmark anchor rule)."""
    return _next_trading_day(pd.to_datetime(dates), _validate_calendar(calendar), strict=False)


def return_window_dates(
    entry_date: pd.Timestamp, calendar: pd.DatetimeIndex, horizon: int
) -> pd.DatetimeIndex:
    """The ``horizon`` trading days after entry; the entry day itself is
    excluded (RET-001: first included return is entry close to next close)."""
    calendar = _validate_calendar(calendar)
    if pd.isna(entry_date):
        return pd.DatetimeIndex([])
    start = calendar.searchsorted(pd.Timestamp(entry_date), side="right")
    return calendar[start : start + horizon]
