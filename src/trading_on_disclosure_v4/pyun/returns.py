"""Forward buy-and-hold returns at arbitrary (PERMNO, session, horizon) cells, the v2 window rule.

The window of an anchor at calendar position ``a`` is the ``h`` sessions ``a + 1 .. a + h``; a window
with a missing security return, a missing benchmark return or one that runs past the calendar end is
NaN (never zero-filled), exactly as ``outcomes.forward_cache._window_products`` does. Benchmarks:
``vwretd`` (Pyun's value-weighted CRSP market), ``sprtrn`` (the existing paper's S&P 500), ``raw``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.data.calendar import load_calendar
from trading_on_disclosure_v4.outcomes import panel

BENCHMARKS = ("raw", "vw", "sp")
BENCH_COLS = {"vw": "vwretd", "sp": "sprtrn"}


def load_benchmarks(root: Path | None = None) -> pd.DataFrame:
    """Date-indexed ``sprtrn`` and ``vwretd`` (the benchmark file behind the calendar)."""
    frame = pd.read_parquet(panel.processed_dir(root) / "benchmark_daily_2014plus.parquet")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")[["sprtrn", "vwretd"]]


class ReturnGrid:
    """Total returns of a PERMNO universe on the session grid plus the two benchmark series."""

    def __init__(self, daily: pd.DataFrame, benchmarks: pd.DataFrame, calendar: pd.DatetimeIndex):
        self.calendar = calendar
        permnos = np.sort(daily["permno"].unique())
        self.permno_pos = pd.Series(np.arange(len(permnos)), index=permnos)
        pos = calendar.get_indexer(pd.DatetimeIndex(daily["date"]))
        keep = pos >= 0
        grid = np.full((len(permnos), len(calendar)), np.nan)
        grid[self.permno_pos.loc[daily["permno"].to_numpy()[keep]].to_numpy(), pos[keep]] = (
            daily["total_ret"].to_numpy(dtype=float)[keep])
        self.grid = grid
        self.bench = {k: benchmarks[c].reindex(calendar).to_numpy(dtype=float) for k, c in BENCH_COLS.items()}

    @classmethod
    def load(cls, permnos, root: Path | None = None) -> ReturnGrid:
        daily = pd.read_parquet(panel.daily_parquet_path(), columns=["permno", "date", "total_ret"])
        daily = daily[daily["permno"].isin(np.asarray(permnos, dtype=np.int64))]
        return cls(daily, load_benchmarks(root), load_calendar())

    def _products(self, rows: np.ndarray, anchors: np.ndarray, h: int) -> np.ndarray:
        T = self.grid.shape[1]
        ok = (anchors >= 0) & (anchors + h <= T - 1) & (rows >= 0)
        a = np.where(ok, anchors, 0)
        r = np.where(ok, rows, 0)
        prod = np.ones(len(anchors))
        for k in range(1, h + 1):
            prod = prod * (1.0 + self.grid[r, a + k])
        return np.where(ok, prod, np.nan)

    def _bench_products(self, anchors: np.ndarray, h: int, bench: str) -> np.ndarray:
        T = self.grid.shape[1]
        ok = (anchors >= 0) & (anchors + h <= T - 1)
        a = np.where(ok, anchors, 0)
        series = self.bench[bench]
        prod = np.ones(len(anchors))
        for k in range(1, h + 1):
            prod = prod * (1.0 + series[a + k])
        return np.where(ok, prod, np.nan)

    def bhar(self, permno: np.ndarray, anchor_idx: np.ndarray, h: int, bench: str = "vw") -> np.ndarray:
        """Buy-and-hold return over ``a + 1 .. a + h`` minus the benchmark's (decimal); NaN when incomplete."""
        if bench not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {bench!r}")
        rows = self.permno_pos.reindex(np.asarray(permno, dtype=np.int64)).fillna(-1).to_numpy(dtype=int)
        anchors = np.asarray(anchor_idx, dtype=int)
        stock = self._products(rows, anchors, h) - 1.0
        if bench == "raw":
            return stock
        return stock - (self._bench_products(anchors, h, bench) - 1.0)


def last_session_on_or_before(dates: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Calendar position of the last session <= date (Pyun's 'close of day t'); -1 when none or NaT."""
    d = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    pos = calendar.searchsorted(d, side="right") - 1
    return np.where(pd.isna(d), -1, pos)


def session_position(dates: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Calendar position of a session date; -1 when the date is NaT or not a session."""
    return calendar.get_indexer(pd.DatetimeIndex(pd.to_datetime(dates)))
