"""Buy-and-hold abnormal returns under the exact off-by-one contract.

Entry-day return is excluded; exactly ``h`` future trading-day returns are
included (RET-001); stock and benchmark share identical dates (RET-002);
CRSP delisting returns combine under the standard rule (RET-003); missing
returns propagate with reason codes, never zero-filled (RET-004); incomplete
windows are excluded with reasons (RET-006); the two clocks produce outcomes
from distinct frozen entry fields (RET-007).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.timing import (
    BENCHMARK_ENTRY_COL,
    FOLLOWER_ENTRY_COL,
    return_window_dates,
)

HORIZONS = (20, 5, 10, 40, 60, 127)

REASON_OK = "ok"
REASON_NO_ENTRY = "no_entry_date"
REASON_TRUNCATED = "end_of_sample_truncation"
REASON_MISSING_RETURN = "missing_return_days"
REASON_BENCHMARK_GAP = "benchmark_gap"
REASON_NO_SECURITY = "no_security_returns"


def combine_with_delisting(daily: pd.DataFrame) -> pd.DataFrame:
    """CRSP combination: (1+ret)(1+dlret)-1 when both exist; dlret alone when
    the ordinary return is missing on the delisting day (RET-003)."""
    out = daily.copy()
    ret = out["ret"]
    dlret = out.get("dlret")
    if dlret is None:
        out["total_ret"] = ret
        return out
    both = ret.notna() & dlret.notna()
    only_dl = ret.isna() & dlret.notna()
    total = ret.copy()
    total[both] = (1 + ret[both]) * (1 + dlret[both]) - 1
    total[only_dl] = dlret[only_dl]
    out["total_ret"] = total
    return out


def window_abnormal_return(
    permno: int,
    entry_date: pd.Timestamp,
    horizon: int,
    stock_returns: pd.Series,
    benchmark_returns: pd.Series,
    calendar: pd.DatetimeIndex,
) -> tuple[float, str]:
    """BHAR over exactly ``horizon`` post-entry trading days, or NaN + reason."""
    if pd.isna(entry_date):
        return np.nan, REASON_NO_ENTRY
    window = return_window_dates(pd.Timestamp(entry_date), calendar, horizon)
    if len(window) < horizon:
        return np.nan, REASON_TRUNCATED
    stock = stock_returns.reindex(window)
    bench = benchmark_returns.reindex(window)
    if stock.isna().all():
        return np.nan, REASON_NO_SECURITY
    if stock.isna().any():
        return np.nan, REASON_MISSING_RETURN
    if bench.isna().any():
        return np.nan, REASON_BENCHMARK_GAP
    bh_stock = float((1 + stock).prod() - 1)
    bh_bench = float((1 + bench).prod() - 1)
    return bh_stock - bh_bench, REASON_OK


#: default anchors: (label, entry column, horizons); variant anchors used by
#: the locked source/timing robustness family compute h20 only
DEFAULT_ANCHORS = (
    ("E", FOLLOWER_ENTRY_COL, HORIZONS),
    ("T", BENCHMARK_ENTRY_COL, HORIZONS),
)


def compute_signal_outcomes(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: pd.Series,
    calendar: pd.DatetimeIndex,
    horizons: tuple[int, ...] = HORIZONS,
    anchors: tuple[tuple[str, str, tuple[int, ...]], ...] | None = None,
) -> pd.DataFrame:
    """Anchor x horizon buy-and-hold abnormal returns. ``daily`` needs
    permno/date/ret[/dlret]; ``benchmark`` is a date-indexed return series."""
    if daily.duplicated(subset=["permno", "date"]).any():
        raise ValueError(
            "daily panel has duplicate (permno, date) rows; the return "
            "engine requires a unique panel — deduplicate upstream and "
            "record the drop in the attrition/engineering report"
        )
    combined = combine_with_delisting(daily)
    by_permno = {
        int(p): g.set_index("date")["total_ret"]
        for p, g in combined.groupby("permno")
    }
    out = signals.copy()
    if anchors is None:
        anchors = tuple(
            (label, col, tuple(horizons)) for label, col, _ in DEFAULT_ANCHORS
        )
    for label, entry_col, anchor_horizons in anchors:
        if entry_col not in out.columns:
            raise KeyError(f"signals missing entry column {entry_col!r}")
        for h in anchor_horizons:
            values, reasons = [], []
            for row in out.itertuples(index=False):
                stock = by_permno.get(int(row.permno), pd.Series(dtype=float))
                ar, reason = window_abnormal_return(
                    int(row.permno),
                    getattr(row, entry_col),
                    h,
                    stock,
                    benchmark,
                    calendar,
                )
                values.append(ar)
                reasons.append(reason)
            out[f"ar_{label}_h{h}"] = values
            out[f"ar_{label}_h{h}_reason"] = reasons
    return out
