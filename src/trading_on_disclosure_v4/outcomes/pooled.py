"""Pooled outcome distributions without household identity (pre-tag allowed).

Joins purchase events with period in {A, B} to the forward cache and writes,
per clock x period x horizon, counts, spread statistics, the share missing by
reason and the 1/99 winsorization cutoffs. Never groups by household.

Two files per duplicate treatment (lead directive 3): the COMMITTED table
(`data/v4/pooled/`) carries no location statistic (no mean, no p50; the
transaction-clock pooled purchase mean is the registered T1 point estimate);
the FULL table with mean and p50 stays under `data/v4/restricted/n/`.
Default duplicate treatment: drop `dup_primary` events on the P clock only
(see `event_filters`); `_keepdups` files keep them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from trading_on_disclosure_v4.outcomes import event_filters as ef
from trading_on_disclosure_v4.outcomes import forward_cache as fc
from trading_on_disclosure_v4.outcomes import panel

EVENTS_DIR = ef.EVENTS_DIR
POOLED_DIR = panel.REPO_ROOT / "data" / "v4" / "pooled"
OUT_PATH = POOLED_DIR / "pooled_outcome_distribution.csv"
FULL_PATH = panel.RESTRICTED_N / "pooled_full.csv"
HORIZON_COLS = {"ff5mom_h10": ("bhar_ff5mom_h10", "reason_h10"),
                "ff5mom_h20": ("bhar_ff5mom_h20", "reason_h20"),
                "ff5mom_h60": ("bhar_ff5mom_h60", "reason_h60"),
                "ff5mom_noalpha_h20": ("bhar_ff5mom_noalpha_h20", "reason_h20"),
                "mkt_h20": ("bhar_mkt_h20", "reason_mkt_h20")}
PCTS = (1, 5, 25, 50, 75, 95, 99)
LOCATION_COLS = ("mean", "p50")  # never in the committed file
REASONS = ("ok", "missing_return_days", "factor_gap", "insufficient_estimation_window",
           "end_of_sample_truncation", "no_security_returns", "benchmark_gap", "not_in_cache")


def load_events(
    clock: str, directions=("purchase",), periods=("A", "B"),
    drop_duplicates: bool = True, drop_duplicates_on_T: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """D's event table filtered on direction/period, then the duplicate filter (count report)."""
    ev = pd.read_parquet(EVENTS_DIR / f"events_{clock}.parquet",
                         columns=["event_id", "permno", "clock", "anchor_date", "direction", "period"])
    ev = ev[ev["direction"].isin(directions) & ev["period"].isin(periods)].reset_index(drop=True)
    return ef.drop_duplicate_events(ev, clock, drop_duplicates, drop_duplicates_on_T)


def join_cache(events: pd.DataFrame, cache: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Left-join events to cache cells on (permno, anchor_idx); unmatched -> 'not_in_cache'."""
    ev = events.copy()
    ev["anchor_idx"] = calendar.get_indexer(pd.DatetimeIndex(ev["anchor_date"]))
    out = ev.merge(cache, on=["permno", "anchor_idx"], how="left", suffixes=("", "_cache"))
    for _, reason_col in HORIZON_COLS.values():
        out[reason_col] = out[reason_col].fillna("not_in_cache")
    return out


def _stats(y: pd.Series, reasons: pd.Series) -> dict:
    ok = y.dropna()
    row = {"n_events": int(len(y)), "n_ok": int(len(ok)),
           "mean": float(ok.mean()), "sd": float(ok.std(ddof=1))}
    for p in PCTS:
        row[f"p{p}"] = float(np.percentile(ok, p))
    row["winsor_lo"], row["winsor_hi"] = row["p1"], row["p99"]
    counts = reasons.value_counts()
    for r in REASONS:
        row[f"share_{r}"] = float(counts.get(r, 0) / len(y))
    return row


def committed_view(table: pd.DataFrame) -> pd.DataFrame:
    """The full table without location statistics (SD, spread percentiles, counts, shares)."""
    return table.drop(columns=[c for c in LOCATION_COLS if c in table.columns])


def pooled_distribution(
    cache: pd.DataFrame | None = None, clocks=("T", "P"), out: Path = OUT_PATH,
    full_out: Path | None = FULL_PATH, drop_duplicates: bool = True,
    drop_duplicates_on_T: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Write the committed (no-location) and full tables; return (full table, filter reports)."""
    if cache is None:
        cols = ["permno", "anchor_idx"] + sorted({c for pair in HORIZON_COLS.values() for c in pair})
        cache = fc.load_cache(columns=cols)
    calendar = panel.v2_trading_calendar()
    rows, reports = [], {}
    for clock in clocks:
        events, reports[clock] = load_events(clock, drop_duplicates=drop_duplicates,
                                             drop_duplicates_on_T=drop_duplicates_on_T)
        joined = join_cache(events, cache, calendar)
        for (period,), grp in joined.groupby(["period"], sort=True):
            for horizon, (ycol, rcol) in HORIZON_COLS.items():
                row = {"clock": clock, "period": period, "direction": "purchase", "horizon": horizon,
                       "units": "decimal_return", "duplicates": ef.dup_suffix(
                           clock, drop_duplicates, drop_duplicates_on_T).lstrip("_")}
                row.update(_stats(grp[ycol], grp[rcol]))
                rows.append(row)
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    committed_view(table).to_csv(out, index=False, float_format="%.8g")
    if full_out is not None:
        full_out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(full_out, index=False, float_format="%.8g")
    return table, reports


def main() -> None:
    """Default (P duplicates dropped) and `_keepdups` variants; prints count-only columns."""
    import json

    show = ["clock", "period", "duplicates", "horizon", "n_events", "n_ok", "sd", "p1", "p99", "share_ok"]
    t, rep = pooled_distribution()
    print(committed_view(t)[show].to_string(index=False))
    print(json.dumps(rep, indent=1))
    t2, rep2 = pooled_distribution(out=POOLED_DIR / "pooled_outcome_distribution_keepdups.csv",
                                   full_out=panel.RESTRICTED_N / "pooled_full_keepdups.csv",
                                   drop_duplicates=False)
    print(committed_view(t2)[show].to_string(index=False))
    print(json.dumps(rep2, indent=1))


if __name__ == "__main__":
    main()
