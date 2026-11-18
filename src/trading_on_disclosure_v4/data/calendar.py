"""The canonical v2 trading calendar plus session-arithmetic helpers.

The calendar is exactly ``pd.DatetimeIndex(sorted(benchmark.index.unique()))``
where ``benchmark`` is the ``sprtrn`` series of
``<DROPBOX_PROJECT_ROOT>/data/processed/benchmark_daily_2014plus.parquet``
(scripts/v2/21_canonical_top_journal_release.py, ``load_market_frames``).
The data and null-panel modules both import this module; nothing here reads returns.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.timing import (
    _validate_calendar,
    close_on_or_next,
    entry_strictly_after,
    return_window_dates,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RELPATH = Path("data") / "processed" / "benchmark_daily_2014plus.parquet"
HORIZON = 20

__all__ = [
    "HORIZON",
    "dropbox_project_root",
    "benchmark_path",
    "load_calendar",
    "first_session_on_or_after",
    "first_session_strictly_after",
    "session_index",
    "kth_session_after",
    "window_end",
    "return_window_dates",
    "close_on_or_next",
    "entry_strictly_after",
]


def dropbox_project_root() -> Path:
    """``DROPBOX_PROJECT_ROOT`` from the environment or the repo's ``.env``."""
    value = os.environ.get("DROPBOX_PROJECT_ROOT")
    if not value:
        env_file = REPO_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DROPBOX_PROJECT_ROOT="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not value:
        raise FileNotFoundError("DROPBOX_PROJECT_ROOT not set and not in .env")
    return Path(value)


def benchmark_path() -> Path:
    return dropbox_project_root() / BENCHMARK_RELPATH


@lru_cache(maxsize=4)
def _cached_calendar(path: str) -> pd.DatetimeIndex:
    bench = pd.read_parquet(path, columns=["date", "sprtrn"])
    bench["date"] = pd.to_datetime(bench["date"])
    benchmark = bench.set_index("date")["sprtrn"]
    return _validate_calendar(pd.DatetimeIndex(sorted(benchmark.index.unique())))


def load_calendar(path: Path | None = None) -> pd.DatetimeIndex:
    """The canonical v2 session calendar (3,518 sessions 2012-07-02..2026-06-30 in the release)."""
    return _cached_calendar(str(path or benchmark_path()))


def first_session_on_or_after(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Transaction-clock anchor: the date itself if it trades, else the next session."""
    return close_on_or_next(dates, calendar)


def first_session_strictly_after(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Public-clock anchor: the first session strictly after the public date."""
    return entry_strictly_after(dates, calendar)


def session_index(dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Position of each date in ``calendar`` (-1 when the date is not a session or NaT)."""
    calendar = _validate_calendar(calendar)
    idx = calendar.get_indexer(pd.DatetimeIndex(pd.to_datetime(dates)))
    return pd.Series(idx, index=dates.index, dtype="int64")


def kth_session_after(dates: pd.Series, calendar: pd.DatetimeIndex, k: int) -> pd.Series:
    """The k-th session after each session date; NaT when the calendar ends first.

    ``k=1`` is the next session (the first return date of a window anchored
    at ``dates``); ``k=HORIZON`` is the window's last return date.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    idx = session_index(dates, calendar)
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    target = idx.to_numpy() + k
    ok = (idx.to_numpy() >= 0) & (target < len(calendar))
    out.iloc[np.flatnonzero(ok)] = calendar[target[ok]]
    return out


def window_end(dates: pd.Series, calendar: pd.DatetimeIndex, horizon: int = HORIZON) -> pd.Series:
    """Last return date of the ``horizon``-session window after each anchor (NaT if truncated).

    Equals ``return_window_dates(anchor, calendar, horizon)[-1]`` when that window is full.
    """
    return kth_session_after(dates, calendar, horizon)
