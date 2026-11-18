"""Factor-model abnormal returns for the locked robustness family (§10).

CAPM and Fama-French five-factor plus momentum BHARs: loadings estimated per
(security, entry date) on a pre-entry window that ends before the disclosure
window, expected returns compounded over exactly the realized dates, missing
returns propagated with reasons and never zero-filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_on_disclosure_v2.outcomes.returns import combine_with_delisting
from trading_on_disclosure_v2.timing import return_window_dates

ESTIMATION_WINDOW = 252
ESTIMATION_GAP = 21  # trading days between estimation end and entry
MIN_ESTIMATION_OBS = 60

MODEL_FACTORS = {
    "capm": ("mkt_rf",),
    "ff5mom": ("mkt_rf", "smb", "hml", "rmw", "cma", "umd"),
}

REASON_OK = "ok"
REASON_NO_ENTRY = "no_entry_date"
REASON_TRUNCATED = "end_of_sample_truncation"
REASON_MISSING_RETURN = "missing_return_days"
REASON_FACTOR_GAP = "factor_gap"
REASON_SHORT_ESTIMATION = "insufficient_estimation_window"


def _estimate_loadings(
    stock: pd.Series,
    factors: pd.DataFrame,
    entry_date: pd.Timestamp,
    calendar: pd.DatetimeIndex,
    factor_cols: tuple[str, ...],
) -> np.ndarray | None:
    pos = calendar.searchsorted(entry_date, side="left")
    est_end = pos - ESTIMATION_GAP
    est_start = max(0, est_end - ESTIMATION_WINDOW)
    if est_end <= est_start:
        return None
    window = calendar[est_start:est_end]
    r = stock.reindex(window)
    f = factors.reindex(window)
    excess = r - f["rf"]
    X = f[list(factor_cols)]
    good = excess.notna() & X.notna().all(axis=1)
    if int(good.sum()) < MIN_ESTIMATION_OBS:
        return None
    Xg = np.column_stack(
        [np.ones(int(good.sum()))] + [X.loc[good, c].to_numpy(float) for c in factor_cols]
    )
    beta, *_ = np.linalg.lstsq(Xg, excess[good].to_numpy(float), rcond=None)
    return beta


def compute_factor_model_cars(
    signals: pd.DataFrame,
    daily: pd.DataFrame,
    factors: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    entry_col: str,
    horizon: int = 20,
    models: tuple[str, ...] = ("capm", "ff5mom"),
) -> pd.DataFrame:
    """Return ``signals`` plus ``car_<model>_h<horizon>`` and reason columns."""
    combined = combine_with_delisting(daily)
    by_permno = {
        int(p): g.set_index("date")["total_ret"]
        for p, g in combined.groupby("permno")
    }
    out = signals.copy()
    for model in models:
        factor_cols = MODEL_FACTORS[model]
        missing_cols = [c for c in (*factor_cols, "rf") if c not in factors.columns]
        col = f"car_{model}_h{horizon}"
        if missing_cols:
            out[col] = np.nan
            out[f"{col}_reason"] = f"factor_columns_missing:{missing_cols}"
            continue
        values, reasons = [], []
        for row in out.itertuples(index=False):
            entry = getattr(row, entry_col)
            if pd.isna(entry):
                values.append(np.nan)
                reasons.append(REASON_NO_ENTRY)
                continue
            entry = pd.Timestamp(entry)
            window = return_window_dates(entry, calendar, horizon)
            if len(window) < horizon:
                values.append(np.nan)
                reasons.append(REASON_TRUNCATED)
                continue
            stock = by_permno.get(int(row.permno), pd.Series(dtype=float))
            r = stock.reindex(window)
            f = factors.reindex(window)
            if r.isna().any():
                values.append(np.nan)
                reasons.append(REASON_MISSING_RETURN)
                continue
            if f[list(factor_cols) + ["rf"]].isna().any().any():
                values.append(np.nan)
                reasons.append(REASON_FACTOR_GAP)
                continue
            beta = _estimate_loadings(stock, factors, entry, calendar, factor_cols)
            if beta is None:
                values.append(np.nan)
                reasons.append(REASON_SHORT_ESTIMATION)
                continue
            expected = f["rf"].to_numpy(float) + beta[0] + np.column_stack(
                [f[c].to_numpy(float) for c in factor_cols]
            ) @ beta[1:]
            bh_stock = float((1 + r).prod() - 1)
            bh_expected = float(np.prod(1 + expected) - 1)
            values.append(bh_stock - bh_expected)
            reasons.append(REASON_OK)
        out[col] = values
        out[f"{col}_reason"] = reasons
    return out
