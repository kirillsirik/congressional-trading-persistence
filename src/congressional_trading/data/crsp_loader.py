"""CRSP daily stock file loaders.

Primary input is the *trimmed* CRSP v2 (CIZ) file produced by
:func:`trim_crsp_v2_file` (14 columns; see legacy ``trim_crsp.py``). A v1
(legacy SIZ) loader is provided as a fallback for older extracts.

Delisting returns
-----------------
CRSP v2 stores each delisting return as its OWN daily ``DlyRet`` observation
dated the trading day after the last trade (StkDelists.DelDlyDt convention), so the v2
loader applies no adjustment. CAUTION: that row carries ``PrimaryExch='X'``
with blank SecurityType/ShareType, so the common-stock trim drops it —
panel builders must re-attach it (see
scripts/v2/22_build_market_panel_2014plus.py). The v1 loader applies the standard legacy adjustment
``(1+RET)(1+DLRET)-1`` (``DLRET`` alone when ``RET`` is missing, ``RET`` alone
when ``DLRET`` is missing).

Units: ``mktcap`` is in thousands of dollars in both formats (CRSP v2
``DlyCap``; v1 ``|PRC| * SHROUT`` with SHROUT in thousands).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import VALID_EXCHCD, VALID_SHRCD
from ..utils.dates import TradingCalendar

logger = logging.getLogger(__name__)

# ── v2 trim filters (paper §3.1.1; legacy trim_crsp.py) ─────────────────────
VALID_PRIMARY_EXCH = {"N", "A", "Q"}       # NYSE, AMEX, NASDAQ
VALID_SECURITY_TYPE = {"EQTY"}             # common equity
VALID_SHARE_TYPE = {"NS"}                  # normal shares (excludes ADRs etc.)

TRIM_KEEP_COLS = [
    "PERMNO", "DlyCalDt", "Ticker", "TradingSymbol", "HdrCUSIP", "PERMCO",
    "SICCD", "PrimaryExch", "SecurityType", "ShareType", "DlyCap", "DlyRet",
    "vwretd", "sprtrn",
]

V2_TRIMMED_COLUMNS = [
    "PERMNO", "HdrCUSIP", "PrimaryExch", "SecurityType", "ShareType", "Ticker",
    "TradingSymbol", "PERMCO", "SICCD", "DlyCalDt", "DlyCap", "DlyRet",
    "vwretd", "sprtrn",
]


@dataclass
class CrspData:
    """Daily panel + benchmark series + trading calendar."""

    daily: pd.DataFrame        # permno, date, ret, mktcap, siccd, ticker, trading_symbol
    benchmarks: pd.DataFrame   # date, sp500_ret, vwretd_ret (deduplicated by date)
    calendar: TradingCalendar  # derived from benchmark dates


def trim_crsp_v2_file(
    input_path: Path,
    output_path: Path,
    chunksize: int = 500_000,
    overwrite: bool = False,
) -> dict[str, int]:
    """Trim the full 94-column CRSP v2 file to the 14 analysis columns.

    Ported from legacy ``trim_crsp.py``: keeps ``PrimaryExch in {N,A,Q}``,
    ``SecurityType == EQTY``, ``ShareType == NS``; processes in chunks so the
    multi-GB input never has to fit in memory. Writes atomically (temp file +
    rename) and refuses to clobber unless ``overwrite=True``.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass overwrite=True to rebuild.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp"

    n_read = n_kept = 0
    try:
        reader = pd.read_csv(
            input_path, usecols=TRIM_KEEP_COLS, dtype=str,
            chunksize=chunksize, low_memory=False,
        )
        first = True
        for chunk in reader:
            n_read += len(chunk)
            mask = (
                chunk["PrimaryExch"].isin(VALID_PRIMARY_EXCH)
                & chunk["SecurityType"].isin(VALID_SECURITY_TYPE)
                & chunk["ShareType"].isin(VALID_SHARE_TYPE)
            )
            kept = chunk.loc[mask, TRIM_KEEP_COLS]
            n_kept += len(kept)
            kept.to_csv(tmp_path, mode="w" if first else "a", header=first, index=False)
            first = False
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info("Trimmed CRSP v2: read %d rows, kept %d -> %s", n_read, n_kept, output_path)
    return {"rows_read": n_read, "rows_kept": n_kept}


def _build_benchmarks(dates: pd.Series, sprtrn: pd.Series, vwretd: pd.Series) -> pd.DataFrame:
    bench = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "sp500_ret": pd.to_numeric(sprtrn, errors="coerce"),
            "vwretd_ret": pd.to_numeric(vwretd, errors="coerce"),
        }
    )
    bench = (
        bench.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return bench


def load_crsp_v2_trimmed(path: Path) -> CrspData:
    """Load the trimmed CRSP v2 file into the canonical daily panel.

    Returns a :class:`CrspData` with the panel columns
    ``permno, date, ret, mktcap, siccd, ticker, trading_symbol``, a benchmark
    frame ``date, sp500_ret, vwretd_ret`` (one row per trading day) and a
    :class:`TradingCalendar` derived from the benchmark dates.
    """
    path = Path(path)
    logger.info("Loading trimmed CRSP v2 panel from %s", path)
    df = pd.read_csv(
        path,
        usecols=["PERMNO", "DlyCalDt", "DlyRet", "DlyCap", "SICCD", "Ticker",
                 "TradingSymbol", "vwretd", "sprtrn"],
        dtype={"PERMNO": "int64", "Ticker": str, "TradingSymbol": str, "SICCD": str},
        low_memory=False,
    )
    benchmarks = _build_benchmarks(df["DlyCalDt"], df["sprtrn"], df["vwretd"])

    daily = pd.DataFrame(
        {
            "permno": df["PERMNO"].astype("int64"),
            "date": pd.to_datetime(df["DlyCalDt"]),
            "ret": pd.to_numeric(df["DlyRet"], errors="coerce"),
            "mktcap": pd.to_numeric(df["DlyCap"], errors="coerce"),
            "siccd": pd.to_numeric(df["SICCD"], errors="coerce"),
            "ticker": df["Ticker"].str.strip().str.upper(),
            "trading_symbol": df["TradingSymbol"].str.strip().str.upper(),
        }
    )
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    logger.info(
        "CRSP v2 panel: %d rows, %d PERMNOs, %s to %s",
        len(daily), daily["permno"].nunique(),
        daily["date"].min().date(), daily["date"].max().date(),
    )
    return CrspData(daily=daily, benchmarks=benchmarks, calendar=TradingCalendar(benchmarks["date"]))


def load_v2_ticker_ranges(path: Path) -> pd.DataFrame:
    """Light-weight load of only the columns needed for PERMNO matching.

    Returns ``permno, date, ticker, trading_symbol`` — avoids materializing
    returns/caps when only the ticker→PERMNO lookup is needed (script 03).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        usecols=["PERMNO", "DlyCalDt", "Ticker", "TradingSymbol"],
        dtype={"PERMNO": "int64", "Ticker": str, "TradingSymbol": str},
        low_memory=False,
    )
    return pd.DataFrame(
        {
            "permno": df["PERMNO"].astype("int64"),
            "date": pd.to_datetime(df["DlyCalDt"]),
            "ticker": df["Ticker"].str.strip().str.upper(),
            "trading_symbol": df["TradingSymbol"].str.strip().str.upper(),
        }
    )


def load_crsp_v1(path: Path) -> CrspData:
    """Load a CRSP v1 (SIZ) daily extract as a fallback.

    Expected columns (case-insensitive): ``PERMNO, date, RET, DLRET, SHRCD,
    EXCHCD`` plus optionally ``TICKER, SICCD, PRC, SHROUT, vwretd, sprtrn``.

    Applies the paper's universe filters (``SHRCD in {10,11}``, ``EXCHCD in
    {1,2,3}``) and the delisting adjustment ``(1+RET)(1+DLRET)-1`` (``RET``
    alone if ``DLRET`` missing; ``DLRET`` alone if ``RET`` missing). Benchmarks
    are only available when the extract includes index return columns.
    """
    path = Path(path)
    logger.info("Loading CRSP v1 panel from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["permno", "date", "ret", "shrcd", "exchcd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CRSP v1 file missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["shrcd"].isin(VALID_SHRCD) & df["exchcd"].isin(VALID_EXCHCD)].copy()

    def _num(col: str) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    ret = _num("ret")
    dlret = _num("dlret")
    ret_adj = ret.copy()
    both = ret.notna() & dlret.notna()
    ret_adj[both] = (1.0 + ret[both]) * (1.0 + dlret[both]) - 1.0
    only_dl = ret.isna() & dlret.notna()
    ret_adj[only_dl] = dlret[only_dl]

    mktcap = _num("prc").abs() * _num("shrout")  # thousands of dollars

    if "ticker" in df.columns:
        ticker = df["ticker"].astype(str).str.strip().str.upper()
    else:
        ticker = pd.Series(pd.NA, index=df.index, dtype=object)
    daily = pd.DataFrame(
        {
            "permno": df["permno"].astype("int64"),
            "date": df["date"],
            "ret": ret_adj,
            "mktcap": mktcap,
            "siccd": _num("siccd"),
            "ticker": ticker,
            "trading_symbol": ticker,
        }
    ).sort_values(["permno", "date"]).reset_index(drop=True)

    if "sprtrn" in df.columns or "vwretd" in df.columns:
        benchmarks = _build_benchmarks(
            df["date"],
            df.get("sprtrn", pd.Series(np.nan, index=df.index)),
            df.get("vwretd", pd.Series(np.nan, index=df.index)),
        )
    else:
        benchmarks = pd.DataFrame(
            {"date": np.sort(df["date"].unique()), "sp500_ret": np.nan, "vwretd_ret": np.nan}
        )
        logger.warning("CRSP v1 extract has no index return columns; benchmark returns are NaN")

    return CrspData(daily=daily, benchmarks=benchmarks, calendar=TradingCalendar(benchmarks["date"]))
