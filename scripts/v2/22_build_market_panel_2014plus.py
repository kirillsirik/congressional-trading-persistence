#!/usr/bin/env python
"""Build the 2014+ processed market panel for the locked canonical release.

Reads the raw CRSP v2 (CIZ) trimmed extract `crsp_trimmed_2012plus.csv` in
chunks and writes to data/processed/:

- crsp_daily_2014plus.parquet   (permno, date, ret) from 2012-07-01 so the
  factor-model estimation window (252d + 21d gap) exists for 2014 entries;
- benchmark_daily_2014plus.parquet (date, sprtrn, vwretd) — sprtrn is the
  S&P 500 price return (excludes dividends), the primary market benchmark;
- crsp_names_2014plus.parquet   (permno, ticker, name windows, shrcd,
  exchcd) for date-aware security matching.

Delisting returns:
CIZ stores each security's delisting return as its OWN daily observation in
the return series, dated the trading day immediately after the last trading
date (StkDelists.DelDlyDt convention). On that row CRSP stamps
``PrimaryExch = 'X'`` and blanks SecurityType/ShareType, so the common-stock
trim screen drops exactly those rows. This builder therefore re-attaches,
for every security in the screened panel, its terminal delisting-return row
from the raw unscreened extract (flag ``is_delisting_return_row``). No
separate DelRet is ever compounded anywhere — the delisting effect enters
the series exactly once, as CRSP publishes it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from congressional_trading.paths import get_paths  # noqa: E402

START_DATE = "2012-07-01"
USECOLS = [
    "PERMNO", "DlyCalDt", "Ticker", "SecurityType", "ShareType",
    "PrimaryExch", "DlyRet", "vwretd", "sprtrn",
]

#: max calendar-day gap between a security's last screened trading row and
#: its delisting-return row (DelDlyDt is "the trading date immediately after
#: the delisting date"; the window absorbs weekends/holidays)
DELIST_ATTACH_WINDOW_DAYS = 14


def select_delisting_return_rows(
    x_rows: pd.DataFrame, last_panel_dates: pd.Series
) -> pd.DataFrame:
    """Pick, per panel security, the raw PrimaryExch='X' row that carries its
    delisting return: non-missing DlyRet, dated strictly after the security's
    last screened panel date and within DELIST_ATTACH_WINDOW_DAYS; earliest
    such row wins. Pure function so the rule is unit-testable."""
    cand = x_rows.dropna(subset=["ret"]).merge(
        last_panel_dates.rename("last_panel_date"),
        left_on="permno",
        right_index=True,
        how="inner",
    )
    gap = (cand["date"] - cand["last_panel_date"]).dt.days
    cand = cand[(gap > 0) & (gap <= DELIST_ATTACH_WINDOW_DAYS)]
    cand = cand.sort_values(["permno", "date"], kind="mergesort")
    picked = cand.drop_duplicates(subset=["permno"], keep="first")
    out = picked[["permno", "date", "ret"]].copy()
    out["is_delisting_return_row"] = True
    return out


def main() -> int:
    lock = REPO_ROOT / ".canonical_run.lock"
    if lock.exists():
        print(
            "REFUSED: canonical run in progress (lock present); the market "
            "panel may not be rebuilt concurrently (§3.3)",
            file=sys.stderr,
        )
        return 4
    t0 = time.time()
    project_root = get_paths(profile="ct_only", read_only=True).project_root
    raw = project_root / "data" / "raw" / "crsp" / "crsp_trimmed_2012plus.csv"
    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily_parts: list[pd.DataFrame] = []
    bench_parts: list[pd.DataFrame] = []
    name_agg: dict = {}
    rows_in = 0
    for chunk in pd.read_csv(
        raw,
        usecols=USECOLS,
        chunksize=2_000_000,
        dtype={
            "PERMNO": "int64",
            "Ticker": "string",
            "SecurityType": "category",
            "ShareType": "category",
            "PrimaryExch": "category",
        },
    ):
        rows_in += len(chunk)
        chunk["DlyCalDt"] = pd.to_datetime(chunk["DlyCalDt"])
        chunk = chunk[chunk["DlyCalDt"] >= START_DATE]
        if chunk.empty:
            continue
        chunk["DlyRet"] = pd.to_numeric(chunk["DlyRet"], errors="coerce")
        daily_parts.append(
            chunk[["PERMNO", "DlyCalDt", "DlyRet"]].rename(
                columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "ret"}
            )
        )
        bench_parts.append(
            chunk[["DlyCalDt", "sprtrn", "vwretd"]]
            .dropna(subset=["sprtrn"])
            .drop_duplicates(subset=["DlyCalDt"])
            .rename(columns={"DlyCalDt": "date"})
        )
        named = chunk.dropna(subset=["Ticker"])
        g = named.groupby(["PERMNO", "Ticker"], observed=True).agg(
            namedt=("DlyCalDt", "min"),
            nameendt=("DlyCalDt", "max"),
            sec=("SecurityType", "first"),
            shr=("ShareType", "first"),
            exch=("PrimaryExch", "first"),
        )
        for key, row in g.iterrows():
            if key in name_agg:
                prev = name_agg[key]
                prev["namedt"] = min(prev["namedt"], row["namedt"])
                prev["nameendt"] = max(prev["nameendt"], row["nameendt"])
            else:
                name_agg[key] = dict(row)

    daily = pd.concat(daily_parts, ignore_index=True)
    n_dup = int(daily.duplicated(subset=["permno", "date"]).sum())
    daily = daily.drop_duplicates(subset=["permno", "date"], keep="first")
    daily["is_delisting_return_row"] = False

    # re-attach delisting-return rows (PrimaryExch='X' convention; see module
    # docstring) from the raw unscreened extracts
    raw_dir = project_root / "data" / "raw" / "crsp"
    x_parts: list[pd.DataFrame] = []
    for gz in (
        raw_dir / "crsp_v2_daily_1980_present.csv.gz",
        raw_dir / "crsp_v2_daily_202604_202606.csv.gz",
    ):
        for chunk in pd.read_csv(
            gz,
            usecols=["PERMNO", "DlyCalDt", "PrimaryExch", "DlyRet"],
            chunksize=5_000_000,
            compression="gzip",
        ):
            x = chunk[chunk["PrimaryExch"] == "X"]
            if len(x):
                x = x.rename(
                    columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "ret"}
                )[["permno", "date", "ret"]].copy()
                x["date"] = pd.to_datetime(x["date"])
                x["ret"] = pd.to_numeric(x["ret"], errors="coerce")
                x_parts.append(x)
    x_rows = (
        pd.concat(x_parts, ignore_index=True)
        .drop_duplicates(subset=["permno", "date"])
        if x_parts
        else pd.DataFrame(columns=["permno", "date", "ret"])
    )
    last_dates = daily.groupby("permno")["date"].max()
    delist_rows = select_delisting_return_rows(x_rows, last_dates)
    daily = pd.concat([daily, delist_rows], ignore_index=True)
    daily = daily.drop_duplicates(subset=["permno", "date"], keep="first")
    daily = daily.sort_values(["permno", "date"], kind="mergesort")
    daily.to_parquet(out_dir / "crsp_daily_2014plus.parquet", index=False)

    bench = (
        pd.concat(bench_parts, ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )
    bench.to_parquet(out_dir / "benchmark_daily_2014plus.parquet", index=False)

    names = pd.DataFrame(
        [
            {
                "permno": permno,
                "ticker": ticker,
                "name_norm": str(ticker),
                "namedt": v["namedt"],
                "nameendt": v["nameendt"],
                # CIZ -> legacy convention (trim_crsp): common stock is
                # EQTY+NS; N/A/Q -> NYSE/AMEX/NASDAQ
                "shrcd": 10 if (v["sec"] == "EQTY" and v["shr"] == "NS") else 99,
                "exchcd": {"N": 1, "A": 2, "Q": 3}.get(str(v["exch"]), 0),
            }
            for (permno, ticker), v in name_agg.items()
        ]
    )
    names.to_parquet(out_dir / "crsp_names_2014plus.parquet", index=False)

    report = {
        "rows_read": rows_in,
        "panel_rows": int(len(daily)),
        "duplicate_rows_dropped": n_dup,
        "raw_x_rows_seen": int(len(x_rows)),
        "delisting_return_rows_attached": int(len(delist_rows)),
        "panel_date_min": str(daily["date"].min().date()),
        "panel_date_max": str(daily["date"].max().date()),
        "benchmark_days": int(len(bench)),
        "name_windows": int(len(names)),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "market_panel_2014plus_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
