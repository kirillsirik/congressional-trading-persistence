"""Main-sample inclusion criteria (paper §3.1.2) with full attrition trail.

Filters (each config-driven, applied sequentially so attrition rows are
interpretable):

1. published date present and inside the sample window (event basis);
2. buy/sell transactions only;
3. common-stock candidates only (ETF/fund/non-equity screen from cleaning);
4. valid PERMNO match;
5. disclosure lag within ``[min_disclosure_lag_days, max_disclosure_lag_days]``.

Horizon availability is handled by flags, not hard drops:
``in_main_sample_h{H} = passes filters AND has_return_h{H}``. The
``has_return_h{H}`` flags are attached later by the returns code (methods
layer); :func:`add_main_sample_flags` combines them. A hard common-sample
restriction across horizons (the legacy congress_trades behavior) is applied
only when ``sample.require_common_sample_across_horizons`` is true.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..utils.validation import require_columns
from .clean_capitol_trades import attrition_row

logger = logging.getLogger(__name__)


def _sample_cfg(config: dict) -> dict:
    return config.get("sample") or {}


def apply_main_sample_filters(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the main-sample filters; return ``(filtered_df, attrition_df)``.

    Rows failing any criterion are dropped (paper behavior); the attrition
    frame has one row per filter with ``stage, n_dropped, n_remaining, reason``.
    """
    cfg = _sample_cfg(config)
    require_columns(
        df,
        ["published_date", "transaction_type_clean", "is_common_stock_candidate",
         "disclosure_lag_days"],
        "apply_main_sample_filters",
    )
    out = df.copy()
    out["published_date"] = pd.to_datetime(out["published_date"])
    attrition: list[dict] = [attrition_row("matched_trades_input", 0, len(out), "input rows")]

    def apply(mask: pd.Series, stage: str, reason: str) -> None:
        nonlocal out
        mask = mask.fillna(False).astype(bool)
        n_dropped = int((~mask).sum())
        out = out.loc[mask].copy()
        attrition.append(attrition_row(stage, n_dropped, len(out), reason))
        if n_dropped:
            logger.info("Filter %s: dropped %d, %d remain", stage, n_dropped, len(out))

    # 1. Sample window on the disclosure (event) date.
    start, end = cfg.get("start"), cfg.get("end")
    window = out["published_date"].notna()
    reason = "published_date present"
    if start:
        window &= out["published_date"] >= pd.Timestamp(start)
        reason += f", >= {start}"
    if end:
        window &= out["published_date"] <= pd.Timestamp(end)
        reason += f", <= {end}"
    apply(window, "sample_window", reason)

    # 2. Buy/sell only.
    apply(
        out["transaction_type_clean"].isin(["buy", "sell"]),
        "buy_sell_only", "transaction_type_clean in {buy, sell}",
    )

    # 3. Common-stock candidates.
    if cfg.get("common_stocks_only", True):
        apply(
            out["is_common_stock_candidate"].astype(int) == 1,
            "common_stock_candidates", "passes ETF/fund/non-equity screens",
        )

    # 4. Valid PERMNO.
    if cfg.get("require_permno_match", True):
        require_columns(out, ["permno"], "apply_main_sample_filters (permno)")
        apply(out["permno"].notna(), "valid_permno", "matched to a CRSP PERMNO")

    # 5. Disclosure lag bounds.
    lo = cfg.get("min_disclosure_lag_days", 0)
    hi = cfg.get("max_disclosure_lag_days", 365)
    lag = pd.to_numeric(out["disclosure_lag_days"], errors="coerce")
    apply((lag >= lo) & (lag <= hi), "disclosure_lag_bounds", f"lag in [{lo}, {hi}] days")

    return out.reset_index(drop=True), pd.DataFrame(attrition)


def add_main_sample_flags(
    df: pd.DataFrame,
    horizons: list[int],
    config: dict | None = None,
    passes_col: str | None = None,
) -> pd.DataFrame:
    """Attach ``in_main_sample_h{H} = passes filters AND has_return_h{H}``.

    Expects ``has_return_h{H}`` columns (added by the returns code). If
    ``passes_col`` is given it marks rows passing the main filters; otherwise
    the frame is assumed to be already filtered (all rows pass). When
    ``sample.require_common_sample_across_horizons`` is true, every flag is
    additionally restricted to rows with ALL horizons available.
    """
    has_cols = [f"has_return_h{h}" for h in horizons]
    require_columns(df, has_cols, "add_main_sample_flags")
    out = df.copy()

    passes = (
        out[passes_col].fillna(0).astype(bool)
        if passes_col and passes_col in out.columns
        else pd.Series(True, index=out.index)
    )
    common = pd.Series(True, index=out.index)
    if config and _sample_cfg(config).get("require_common_sample_across_horizons", False):
        for col in has_cols:
            common &= out[col].fillna(0).astype(bool)
        logger.info(
            "Common-sample restriction active: %d of %d rows have all horizons",
            int(common.sum()), len(out),
        )

    for h in horizons:
        out[f"in_main_sample_h{h}"] = (
            passes & common & out[f"has_return_h{h}"].fillna(0).astype(bool)
        ).astype(int)
    return out
