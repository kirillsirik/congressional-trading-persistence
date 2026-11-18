"""Frozen public-date and transaction-date hierarchies (final directive §3-§4).

Source-native date fields are never overwritten; the analysis dates are new
columns derived under the frozen rules:

public-action clock (``analysis_public_date``):
  1. observed official filing/received date when valid;
  2. else Capitol Trades publication date MINUS ONE CALENDAR DAY, labeled
     CAPITOL_TRADES_MINUS_ONE_CALENDAR_DAY_PROXY (a proxy, never an observed
     official date);
  3. else the row is excluded from public-timing analyses with an explicit
     attrition reason.

transaction clock (``trade_date_analysis``):
  1. official transaction date when available;
  2. else Capitol Trades transaction date PLUS ONE CALENDAR DAY.

Both adjustments are calendar-day, not trading-day, arithmetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OFFICIAL_PUBLIC_COL = "official_public_date"
CT_PUBLIC_RAW_COL = "capitol_trades_publication_date_raw"
ANALYSIS_PUBLIC_COL = "analysis_public_date"
ANALYSIS_PUBLIC_SOURCE_COL = "analysis_public_date_source"
ANALYSIS_PUBLIC_PROXY_COL = "analysis_public_date_is_proxy"

TRADE_OFFICIAL_COL = "trade_date_official"
TRADE_CT_RAW_COL = "trade_date_capitol_trades_raw"
TRADE_ANALYSIS_COL = "trade_date_analysis"
TRADE_SOURCE_COL = "trade_date_source"
TRADE_CORRECTION_COL = "trade_date_correction_days"

SOURCE_OFFICIAL_OBSERVED = "OFFICIAL_OBSERVED"
SOURCE_CT_PROXY = "CAPITOL_TRADES_MINUS_ONE_CALENDAR_DAY_PROXY"
SOURCE_UNUSABLE = "NO_USABLE_PUBLIC_DATE"

TRADE_SOURCE_OFFICIAL = "OFFICIAL"
TRADE_SOURCE_CT_PLUS_ONE = "CAPITOL_TRADES_PLUS_ONE_CALENDAR_DAY"
TRADE_SOURCE_NONE = "NO_USABLE_TRANSACTION_DATE"

SOURCE_NATIVE_COLS = (
    OFFICIAL_PUBLIC_COL,
    CT_PUBLIC_RAW_COL,
    TRADE_OFFICIAL_COL,
    TRADE_CT_RAW_COL,
)


class DateHierarchyError(RuntimeError):
    pass


def _as_datetime(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise DateHierarchyError(f"required source-native column missing: {col!r}")
    return pd.to_datetime(frame[col], errors="raise")


def apply_public_date_hierarchy(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` plus the three analysis public-date columns.

    Rows with no usable date are RETAINED here with source
    NO_USABLE_PUBLIC_DATE and a NaT analysis date; the sample stage excludes
    them with an attrition code so the exclusion is ledgered, not silent.
    """
    for col in (
        ANALYSIS_PUBLIC_COL,
        ANALYSIS_PUBLIC_SOURCE_COL,
        ANALYSIS_PUBLIC_PROXY_COL,
    ):
        if col in frame.columns:
            raise DateHierarchyError(
                f"{col!r} already present; refusing to overwrite derived dates"
            )
    out = frame.copy()
    official = _as_datetime(out, OFFICIAL_PUBLIC_COL)
    ct_raw = _as_datetime(out, CT_PUBLIC_RAW_COL)

    use_official = official.notna()
    use_proxy = ~use_official & ct_raw.notna()

    analysis = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    analysis[use_official] = official[use_official]
    analysis[use_proxy] = ct_raw[use_proxy] - pd.Timedelta(days=1)

    source = pd.Series(SOURCE_UNUSABLE, index=out.index, dtype=object)
    source[use_official] = SOURCE_OFFICIAL_OBSERVED
    source[use_proxy] = SOURCE_CT_PROXY

    out[ANALYSIS_PUBLIC_COL] = analysis
    out[ANALYSIS_PUBLIC_SOURCE_COL] = source
    out[ANALYSIS_PUBLIC_PROXY_COL] = use_proxy.to_numpy()

    # source-native fields must survive untouched
    if not official.equals(pd.to_datetime(out[OFFICIAL_PUBLIC_COL])):
        raise DateHierarchyError("official_public_date was modified")
    if not ct_raw.equals(pd.to_datetime(out[CT_PUBLIC_RAW_COL])):
        raise DateHierarchyError("capitol_trades_publication_date_raw was modified")
    return out


def apply_trade_date_hierarchy(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` plus analysis transaction-date columns (§4)."""
    for col in (TRADE_ANALYSIS_COL, TRADE_SOURCE_COL, TRADE_CORRECTION_COL):
        if col in frame.columns:
            raise DateHierarchyError(
                f"{col!r} already present; refusing to overwrite derived dates"
            )
    out = frame.copy()
    official = _as_datetime(out, TRADE_OFFICIAL_COL)
    ct_raw = _as_datetime(out, TRADE_CT_RAW_COL)

    use_official = official.notna()
    use_ct = ~use_official & ct_raw.notna()

    analysis = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    analysis[use_official] = official[use_official]
    analysis[use_ct] = ct_raw[use_ct] + pd.Timedelta(days=1)

    source = pd.Series(TRADE_SOURCE_NONE, index=out.index, dtype=object)
    source[use_official] = TRADE_SOURCE_OFFICIAL
    source[use_ct] = TRADE_SOURCE_CT_PLUS_ONE

    correction = np.zeros(len(out))
    correction[use_ct.to_numpy()] = 1.0
    correction[(~use_official & ~use_ct).to_numpy()] = np.nan

    out[TRADE_ANALYSIS_COL] = analysis
    out[TRADE_SOURCE_COL] = source
    out[TRADE_CORRECTION_COL] = correction
    return out


def public_date_overlap_diagnostics(
    frame: pd.DataFrame, by: tuple[str, ...] = ()
) -> dict:
    """§3.2 diagnostics over rows carrying BOTH an official date and a raw
    Capitol Trades publication date. Pure date arithmetic — never consult
    return outcomes here (no tuning after viewing returns)."""
    official = pd.to_datetime(frame[OFFICIAL_PUBLIC_COL])
    ct_raw = pd.to_datetime(frame[CT_PUBLIC_RAW_COL])
    both = official.notna() & ct_raw.notna()
    matched = frame[both].copy()
    diff = (ct_raw[both] - official[both]).dt.days

    def _dist(d: pd.Series) -> dict:
        counts = d.value_counts().sort_index()
        window = {int(k): int(v) for k, v in counts.items() if -5 <= k <= 10}
        return window

    out: dict = {
        "n_matched": int(both.sum()),
        "mean_diff_days": float(diff.mean()) if len(diff) else np.nan,
        "median_diff_days": float(diff.median()) if len(diff) else np.nan,
        "modal_diff_days": int(diff.mode().iloc[0]) if len(diff) else None,
        "distribution_minus5_to_plus10": _dist(diff),
        "share_ct_exactly_one_day_later": float((diff == 1).mean()) if len(diff) else np.nan,
        "share_ct_same_day": float((diff == 0).mean()) if len(diff) else np.nan,
        "share_ct_more_than_one_day_later": float((diff > 1).mean()) if len(diff) else np.nan,
        "share_ct_earlier_than_official": float((diff < 0).mean()) if len(diff) else np.nan,
        "one_day_characterization_supported": bool(
            len(diff) and (diff == 1).mean() >= 0.5
        ),
    }
    for col in by:
        if col not in matched.columns:
            continue
        matched["_diff"] = diff.to_numpy()
        out[f"by_{col}"] = {
            str(k): {
                "n": int(len(g)),
                "mean_diff_days": float(g["_diff"].mean()),
                "share_exactly_plus_one": float((g["_diff"] == 1).mean()),
            }
            for k, g in matched.groupby(col)
        }
    return out
