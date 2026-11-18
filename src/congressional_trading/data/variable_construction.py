"""Regression-variable construction (pre-event controls only).

Everything here is computable strictly before the event date, so these
columns are safe as predictive features:

* ``ln_size`` — log geometric bracket midpoint (paper's size measure);
* ``disclosure_lag_days_sq``;
* ``ln_mktcap`` — log market cap at the last month-end strictly BEFORE the
  event date (ported from legacy step4 ``find_month_end_mktcap``);
* ``pre_event_volatility`` — std of the 60 trading-day returns ending the day
  before the event date, annualized by sqrt(252);
* ``siccd`` / ``sector`` — 10-group SIC mapping (legacy step7);
* ``disclosure_month`` / ``trade_month`` period strings;
* ``ln_size_resid`` — see :func:`residualize_ln_size` for the fit-mask
  semantics (full-sample for descriptive use, train-only for prediction).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ANNUALIZATION_FACTOR, VOLATILITY_WINDOW
from ..utils.validation import require_columns

logger = logging.getLogger(__name__)

# Legacy step4 required >= 30 non-missing returns inside the 60-day window.
MIN_VOLATILITY_OBS = 30


# ── Simple transforms ────────────────────────────────────────────────────────


def add_size_and_lag_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``ln_size`` and ``disclosure_lag_days_sq``."""
    require_columns(df, ["size_midpoint_geometric", "disclosure_lag_days"], "add_size_and_lag")
    out = df.copy()
    mid = pd.to_numeric(out["size_midpoint_geometric"], errors="coerce")
    out["ln_size"] = np.log(mid.where(mid > 0))
    lag = pd.to_numeric(out["disclosure_lag_days"], errors="coerce").astype("float64")
    out["disclosure_lag_days_sq"] = lag**2
    return out


def add_month_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``disclosure_month`` and ``trade_month`` as 'YYYY-MM' strings."""
    require_columns(df, ["published_date", "trade_date"], "add_month_periods")
    out = df.copy()
    out["disclosure_month"] = (
        pd.to_datetime(out["published_date"]).dt.to_period("M").astype(str)
    )
    out["trade_month"] = pd.to_datetime(out["trade_date"]).dt.to_period("M").astype(str)
    return out


# ── SIC sector mapping (ported from legacy step7 sic_to_sector) ──────────────


def sic_to_sector(sic: object) -> str:
    """Map a SIC code to one of 10 broad sectors (legacy step7 grouping)."""
    try:
        s = int(str(int(float(sic)))[:2])
    except (ValueError, TypeError, OverflowError):
        return "Other"
    if s <= 9:
        return "Agriculture"
    if s <= 14:
        return "Mining"
    if s <= 17:
        return "Construction"
    if s <= 39:
        return "Manufacturing"
    if s <= 49:
        return "Utilities/Transport"
    if s <= 51:
        return "Wholesale"
    if s <= 59:
        return "Retail"
    if s <= 67:
        return "Finance"
    if s <= 89:
        return "Services"
    return "Other"


def add_sector(df: pd.DataFrame, siccd_col: str = "siccd") -> pd.DataFrame:
    out = df.copy()
    out["sector"] = out[siccd_col].map(sic_to_sector) if siccd_col in out.columns else "Other"
    return out


# ── Market-data controls (ln_mktcap, volatility, siccd) ─────────────────────


def _month_end_ln_mktcap(sub: pd.DataFrame, t0: pd.Timestamp) -> float:
    """ln(mktcap) at the most recent month-end strictly before ``t0``.

    Port of legacy step4 ``find_month_end_mktcap`` with its fallbacks: last
    completed calendar month's final observation; else the last (possibly
    partial-month) month-end value; else the last pre-event observation.
    """
    pre = sub.loc[sub.index < t0]
    if pre.empty:
        return np.nan
    monthly_last = pre["mktcap"].resample("ME").last()
    valid = monthly_last[monthly_last.index < t0.replace(day=1)]
    for candidates in (valid, monthly_last):
        if len(candidates):
            cap = candidates.iloc[-1]
            if pd.notna(cap) and cap > 0:
                return float(np.log(cap))
    last_cap = pre["mktcap"].dropna()
    if len(last_cap) and last_cap.iloc[-1] > 0:
        return float(np.log(last_cap.iloc[-1]))
    return np.nan


def _pre_event_volatility(
    sub: pd.DataFrame,
    t0: pd.Timestamp,
    window: int = VOLATILITY_WINDOW,
    min_obs: int = MIN_VOLATILITY_OBS,
    annualization: int = ANNUALIZATION_FACTOR,
) -> float:
    """Annualized std of the ``window`` trading-day returns ending before t0."""
    pre_idx = sub.index[sub.index < t0]
    if len(pre_idx) < window:
        return np.nan
    rets = sub.loc[pre_idx[-window:], "ret"].dropna().to_numpy()
    if len(rets) < min_obs:
        return np.nan
    return float(np.std(rets, ddof=1) * np.sqrt(annualization))


def _pre_event_siccd(sub: pd.DataFrame, t0: pd.Timestamp) -> float:
    """SIC code at the last pre-event observation (fallback: first non-null)."""
    sic = sub["siccd"].dropna()
    if sic.empty:
        return np.nan
    pre = sic.loc[sic.index < t0]
    return float(pre.iloc[-1]) if len(pre) else float(sic.iloc[0])


def add_market_variables(
    df: pd.DataFrame,
    daily_panel: pd.DataFrame,
    event_col: str = "event_date_disclosure",
) -> pd.DataFrame:
    """Attach ``ln_mktcap``, ``pre_event_volatility`` and ``siccd`` per trade.

    ``daily_panel`` is the CRSP panel from :mod:`crsp_loader`
    (``permno, date, ret, mktcap, siccd, ...``). All quantities use only data
    strictly before the event date.
    """
    require_columns(df, ["permno", event_col], "add_market_variables")
    require_columns(daily_panel, ["permno", "date", "ret", "mktcap", "siccd"], "daily_panel")

    out = df.copy()
    out["ln_mktcap"] = np.nan
    out["pre_event_volatility"] = np.nan
    out["siccd"] = np.nan

    events = out[[event_col]].assign(
        _event=pd.to_datetime(out[event_col]),
        _permno=pd.to_numeric(out["permno"], errors="coerce"),
    )
    needed_permnos = set(events["_permno"].dropna().astype(int))
    panel = daily_panel[daily_panel["permno"].isin(needed_permnos)]
    grouped = {
        permno: sub.set_index("date").sort_index()
        for permno, sub in panel.groupby("permno")
    }

    n_missing_panel = 0
    for idx, permno, t0 in zip(events.index, events["_permno"], events["_event"], strict=True):
        if pd.isna(permno) or pd.isna(t0):
            continue
        sub = grouped.get(int(permno))
        if sub is None:
            n_missing_panel += 1
            continue
        out.loc[idx, "ln_mktcap"] = _month_end_ln_mktcap(sub, t0)
        out.loc[idx, "pre_event_volatility"] = _pre_event_volatility(sub, t0)
        out.loc[idx, "siccd"] = _pre_event_siccd(sub, t0)

    if n_missing_panel:
        logger.warning("%d trades reference PERMNOs absent from the CRSP panel", n_missing_panel)
    logger.info(
        "Market variables: ln_mktcap %d/%d, volatility %d/%d non-missing",
        int(out["ln_mktcap"].notna().sum()), len(out),
        int(out["pre_event_volatility"].notna().sum()), len(out),
    )
    return out


# ── Size residualization ─────────────────────────────────────────────────────


def residualize_ln_size(
    df: pd.DataFrame,
    fit_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Attach ``ln_size_resid`` = residual of ``ln_size`` on ``ln_mktcap``.

    The OLS line (``np.polyfit`` degree 1) is fit ONLY on rows selected by
    ``fit_mask`` (intersected with non-missing ``ln_size``/``ln_mktcap``);
    residuals are then computed for every row with both inputs present using
    those fitted coefficients.

    Fit-mask semantics (document in any table using this column):
    * ``fit_mask=None`` (full sample) — descriptive regressions, matching the
      paper's step4 behavior; label the column as full-sample residualized.
    * train-only mask — REQUIRED when ``ln_size_resid`` feeds a predictive
      model, otherwise the fit leaks post-split information.
    """
    require_columns(df, ["ln_size", "ln_mktcap"], "residualize_ln_size")
    out = df.copy()
    both = out["ln_size"].notna() & out["ln_mktcap"].notna()
    fit_rows = both if fit_mask is None else (both & fit_mask.reindex(out.index).fillna(False))
    if int(fit_rows.sum()) < 3:
        raise ValueError("residualize_ln_size: fewer than 3 rows available for the fit")

    slope, intercept = np.polyfit(
        out.loc[fit_rows, "ln_mktcap"].to_numpy(dtype=float),
        out.loc[fit_rows, "ln_size"].to_numpy(dtype=float),
        deg=1,
    )
    out["ln_size_resid"] = np.nan
    out.loc[both, "ln_size_resid"] = out.loc[both, "ln_size"] - (
        slope * out.loc[both, "ln_mktcap"] + intercept
    )
    params = {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_fit": int(fit_rows.sum()),
        "fit_scope": "full_sample" if fit_mask is None else "masked",
    }
    logger.info(
        "Residualized ln_size on ln_mktcap (%s fit, n=%d, slope=%.4f)",
        params["fit_scope"], params["n_fit"], params["slope"],
    )
    return out, params


# ── Convenience wrapper used by the build stage ──────────────────────────────


def add_all_variables(
    df: pd.DataFrame,
    daily_panel: pd.DataFrame,
    config: dict,
    event_col: str = "event_date_disclosure",
) -> pd.DataFrame:
    """Attach all pre-event controls (sizes, lags, months, market vars, sector).

    ``ln_size_resid`` here uses the full-sample fit (descriptive convention);
    the predictive layer must refit with a train-only mask.
    """
    del config  # constants are methodology-level; kept for interface symmetry
    out = add_size_and_lag_variables(df)
    out = add_month_periods(out)
    out = add_market_variables(out, daily_panel, event_col=event_col)
    out = add_sector(out)
    out, _ = residualize_ln_size(out, fit_mask=None)
    return out
