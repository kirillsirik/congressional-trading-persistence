"""Trading-calendar date logic.

The trading calendar is derived from the benchmark return series (every date
with a benchmark return is a trading day), so no external calendar package is
needed and the calendar is exactly consistent with the return data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradingCalendar:
    """Sorted array of trading days with event-date and offset lookups."""

    def __init__(self, trading_days: pd.Series | pd.DatetimeIndex | np.ndarray):
        days = pd.DatetimeIndex(pd.to_datetime(pd.Series(trading_days).unique())).sort_values()
        if len(days) == 0:
            raise ValueError("Trading calendar is empty")
        self.days = days
        self._pos = {d: i for i, d in enumerate(days)}

    def first_trading_day_on_or_after(self, date: pd.Timestamp) -> pd.Timestamp | None:
        i = self.days.searchsorted(pd.Timestamp(date), side="left")
        return self.days[i] if i < len(self.days) else None

    def first_trading_day_after(self, date: pd.Timestamp) -> pd.Timestamp | None:
        i = self.days.searchsorted(pd.Timestamp(date), side="right")
        return self.days[i] if i < len(self.days) else None

    def offset(self, date: pd.Timestamp, k: int) -> pd.Timestamp | None:
        """Trading day k steps after ``date`` (which must be a trading day)."""
        i = self._pos.get(pd.Timestamp(date))
        if i is None:
            return None
        j = i + k
        return self.days[j] if 0 <= j < len(self.days) else None

    def window(self, date: pd.Timestamp, start: int, end: int) -> pd.DatetimeIndex | None:
        """Trading days from offset ``start`` to ``end`` inclusive, or None if truncated."""
        i = self._pos.get(pd.Timestamp(date))
        if i is None:
            return None
        lo, hi = i + start, i + end
        if lo < 0 or hi >= len(self.days):
            return None
        return self.days[lo : hi + 1]


def disclosure_lag_days(published: pd.Series, traded: pd.Series) -> pd.Series:
    """Calendar days between publication and trade execution."""
    return (pd.to_datetime(published) - pd.to_datetime(traded)).dt.days
