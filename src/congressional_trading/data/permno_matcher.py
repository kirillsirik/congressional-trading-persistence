"""Ticker → PERMNO matching against the CRSP daily panel.

Ports the legacy step2 approach: build ``ticker -> [(permno, first_date,
last_date)]`` activity windows from BOTH the ``ticker`` and ``trading_symbol``
columns of the daily panel, then match each trade's ``ticker_clean`` on the
date basis configured by ``matching.match_date_basis``:

* ``"trade_date"`` — paper baseline (the security identity at execution).
* ``"published_date"`` — sensitivity (identity at disclosure).

Ties (multiple PERMNOs active for the same ticker on the match date) are
broken by the longest active range, with ``match_ambiguous_flag = 1``.
A recovery pass applies the hand-curated :data:`TICKER_MAP` from legacy
step2b (corporate actions / ticker changes).

Intentionally NOT ported from step2b: the relaxed-ShareType recheck against
the full 94-column CRSP file (``US_STOCKS_RECHECK``). Those securities (REITs,
BDCs, closed-end funds, agency stocks) are excluded from the common-stock
universe by the paper's own §3.1.1 filters, so re-adding them through a side
channel was inconsistent; the ticker set is kept below for documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..utils.validation import require_columns

logger = logging.getLogger(__name__)

MATCH_DATE_BASES = ("trade_date", "published_date")

# ── Hand-curated recovery map (ported from legacy step2b_recover_unmatched) ──
# Capitol Trades ticker -> CRSP tickers to try, in order.
TICKER_MAP: dict[str, list[str]] = {
    "NCR":  ["NCR", "NCRV", "VYX"],        # NCR split Oct 2023
    "MRSH": ["MMC"],                        # Marsh & McLennan is MMC in CRSP
    "PSKY": ["PARA", "PARAA"],              # Paramount Global
    "B":    ["BRK.B", "BRK-B", "BRK B"],   # Berkshire Hathaway Class B
    "CDAY": ["DAY", "CDAY"],                # Ceridian -> Dayforce 2024
    "SQ":   ["SQ", "XYZ"],                  # Block Inc, SQ -> XYZ 2024
    "FLT":  ["FLT", "CPAY"],                # FleetCor -> Corpay 2023
    "PEAK": ["PEAK", "DOC"],                # Healthpeak -> DOC 2023
    "RE":   ["RE", "EG"],                   # Everest Re -> EG 2023
    "COR":  ["COR", "ABC"],                 # Cencora, was AmerisourceBergen
    "LTHM": ["LTHM", "ALTM"],               # Livent -> Arcadium Lithium 2024
    "PRMW": ["PRMW", "QDRN"],               # Primo Water -> Primo Brands
    "SGI":  ["TPX"],                        # Tempur Sealy is TPX in CRSP
    "WWE":  ["WWE", "TKO"],                 # WWE -> TKO Group 2023
    "BWIN": ["BRP", "BWIN"],                # BRP Group
    "JBGS": ["JBGS"],
    "DWAC": ["DWAC", "DJT"],                # Digital World -> Trump Media
    "GPS":  ["GPS", "GAP"],                 # Gap Inc -> GAP 2024
    "SIX":  ["SIX", "FUN"],                 # Six Flags / Cedar Fair merger
    "JBT":  ["JBT", "JBTX"],
    "CCL":  ["CCL", "CUK"],                 # Carnival dual listing
    "SMAR": ["SMAR"],
    "VMW":  ["VMW"],
    "AMED": ["AMED"],
    "SWTX": ["SWTX"],
    "BECN": ["BECN"],
    "RVTY": ["RVTY"],
    "SEAS": ["SEAS"],
    "ITCI": ["ITCI"],
    "X":    ["X"],
}

# Documented but intentionally unused (see module docstring).
US_STOCKS_RECHECK = {
    "EQR", "HTGC", "ARCC", "AMH", "CUBE", "NSA", "CBOE", "CG",
    "CSWI", "VAPO", "ADMR", "EPR", "RPT", "EQC", "FMCC", "FNMA",
    "SKX", "PNM", "ATH", "WTT",
}


@dataclass
class MatchResult:
    """Output bundle of :func:`match_trades_to_permnos`."""

    matched: pd.DataFrame    # trades with permno, match_method, match_ambiguous_flag
    unmatched: pd.DataFrame  # trades with match_method == "unmatched"
    report: pd.DataFrame     # counts by match method


def build_ticker_lookup(daily: pd.DataFrame) -> pd.DataFrame:
    """Build ticker activity windows from a CRSP daily panel.

    Uses BOTH ``ticker`` and ``trading_symbol``; per (ticker, permno) the
    windows from the two sources are merged to their union span
    (min first date, max last date). Returns columns
    ``ticker, permno, first_date, last_date``.
    """
    require_columns(daily, ["permno", "date", "ticker", "trading_symbol"], "build_ticker_lookup")
    parts = []
    for col in ("ticker", "trading_symbol"):
        sub = daily[[col, "permno", "date"]].dropna(subset=[col]).rename(columns={col: "ticker"})
        sub = sub[sub["ticker"].astype(str).str.strip() != ""]
        parts.append(sub)
    stacked = pd.concat(parts, ignore_index=True)
    stacked["ticker"] = stacked["ticker"].astype(str).str.strip().str.upper()
    lookup = (
        stacked.groupby(["ticker", "permno"], sort=True)["date"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    logger.info(
        "Ticker lookup: %d (ticker, permno) windows over %d tickers",
        len(lookup), lookup["ticker"].nunique(),
    )
    return lookup


def _match_on_window(
    trades: pd.DataFrame, lookup: pd.DataFrame, date_col: str
) -> pd.DataFrame:
    """Vectorized window match; one best row per input index.

    Returns a frame indexed like the matched subset of ``trades`` with columns
    ``permno, match_ambiguous_flag, matched_ticker``.
    """
    t = trades.loc[trades["ticker_clean"].notna() & trades[date_col].notna(),
                   ["ticker_clean", date_col]].copy()
    if t.empty:
        return pd.DataFrame(columns=["permno", "match_ambiguous_flag", "matched_ticker"])
    t = t.reset_index(names="_row")
    m = t.merge(lookup, left_on="ticker_clean", right_on="ticker", how="inner")
    m = m[(m[date_col] >= m["first_date"]) & (m[date_col] <= m["last_date"])]
    if m.empty:
        return pd.DataFrame(columns=["permno", "match_ambiguous_flag", "matched_ticker"])
    m["range_days"] = (m["last_date"] - m["first_date"]).dt.days
    # Longest active range wins; permno ascending as a deterministic tie-break.
    m = m.sort_values(["range_days", "permno"], ascending=[False, True])
    n_candidates = m.groupby("_row")["permno"].nunique()
    best = m.drop_duplicates(subset=["_row"], keep="first").set_index("_row")
    out = pd.DataFrame(
        {
            "permno": best["permno"].astype("int64"),
            "match_ambiguous_flag": (n_candidates.reindex(best.index) > 1).astype(int),
            "matched_ticker": best["ticker"],
        }
    )
    out.index.name = None
    return out


def _recover_with_ticker_map(
    trades: pd.DataFrame, lookup: pd.DataFrame, date_col: str
) -> pd.DataFrame:
    """Recovery pass over unmatched rows using :data:`TICKER_MAP`.

    Candidate tickers are tried in map order; the first with an active window
    covering the match date wins (window tie-break as in the main pass).
    """
    rows = trades[trades["ticker_clean"].isin(TICKER_MAP.keys()) & trades[date_col].notna()]
    results = {}
    for idx, row in rows.iterrows():
        for candidate in TICKER_MAP[row["ticker_clean"]]:
            cand = candidate.strip().upper()
            hits = lookup[
                (lookup["ticker"] == cand)
                & (lookup["first_date"] <= row[date_col])
                & (lookup["last_date"] >= row[date_col])
            ]
            if len(hits):
                hits = hits.assign(range_days=(hits["last_date"] - hits["first_date"]).dt.days)
                hits = hits.sort_values(["range_days", "permno"], ascending=[False, True])
                results[idx] = {
                    "permno": int(hits.iloc[0]["permno"]),
                    "match_ambiguous_flag": int(len(hits) > 1),
                    "matched_ticker": cand,
                }
                break
    return pd.DataFrame.from_dict(results, orient="index")


def match_trades_to_permnos(
    trades: pd.DataFrame,
    lookup: pd.DataFrame,
    config: dict,
) -> MatchResult:
    """Match trades to PERMNOs via ticker activity windows + recovery map.

    ``config["matching"]["match_date_basis"]`` selects the date basis
    (default ``"trade_date"``, the paper baseline).
    """
    basis = (config.get("matching") or {}).get("match_date_basis", "trade_date")
    if basis not in MATCH_DATE_BASES:
        raise ValueError(f"match_date_basis must be one of {MATCH_DATE_BASES}, got {basis!r}")
    require_columns(trades, ["ticker_clean", basis], "match_trades_to_permnos")

    df = trades.copy()
    df[basis] = pd.to_datetime(df[basis])

    primary = _match_on_window(df, lookup, basis)
    df["permno"] = primary["permno"].reindex(df.index)
    df["match_ambiguous_flag"] = primary["match_ambiguous_flag"].reindex(df.index).fillna(0).astype(int)
    df["matched_ticker"] = primary["matched_ticker"].reindex(df.index)
    df["match_method"] = "unmatched"
    df.loc[primary.index, "match_method"] = "ticker_window"

    recovery = _recover_with_ticker_map(df.loc[df["match_method"] == "unmatched"], lookup, basis)
    if len(recovery):
        df.loc[recovery.index, "permno"] = recovery["permno"]
        df.loc[recovery.index, "match_ambiguous_flag"] = recovery["match_ambiguous_flag"].astype(int)
        df.loc[recovery.index, "matched_ticker"] = recovery["matched_ticker"]
        df.loc[recovery.index, "match_method"] = "ticker_map_recovery"

    df["permno"] = df["permno"].astype("Int64")
    matched = df[df["match_method"] != "unmatched"].copy()
    unmatched = df[df["match_method"] == "unmatched"].copy()

    report = (
        df.groupby("match_method", sort=True)
        .agg(n=("match_method", "size"), n_ambiguous=("match_ambiguous_flag", "sum"))
        .reset_index()
    )
    report["match_date_basis"] = basis
    report["share"] = report["n"] / len(df) if len(df) else 0.0

    logger.info(
        "PERMNO matching (%s basis): %d matched (%d via recovery map), %d unmatched, %d ambiguous",
        basis, len(matched),
        int((df["match_method"] == "ticker_map_recovery").sum()),
        len(unmatched), int(df["match_ambiguous_flag"].sum()),
    )
    return MatchResult(matched=matched, unmatched=unmatched, report=report)


def compare_match_date_bases(trades: pd.DataFrame, lookup: pd.DataFrame, config: dict) -> dict:
    """Sensitivity: trade-date vs published-date matching.

    Returns ``{"n_total", "n_differing", "differing"}`` where ``differing``
    lists rows whose assigned PERMNO differs across bases (including matched
    under one basis only).
    """
    results = {}
    for basis in MATCH_DATE_BASES:
        cfg = {**config, "matching": {**(config.get("matching") or {}), "match_date_basis": basis}}
        res = match_trades_to_permnos(trades, lookup, cfg)
        full = pd.concat([res.matched, res.unmatched]).sort_index()
        results[basis] = full["permno"]

    a, b = results["trade_date"], results["published_date"]
    differs = a.fillna(-1).astype("int64") != b.fillna(-1).astype("int64")
    id_cols = [c for c in ("trade_id", "politician_name", "ticker_clean",
                           "trade_date", "published_date") if c in trades.columns]
    differing = trades.loc[differs, id_cols].copy()
    differing["permno_trade_date_basis"] = a[differs]
    differing["permno_published_date_basis"] = b[differs]
    out = {"n_total": int(len(trades)), "n_differing": int(differs.sum()), "differing": differing}
    logger.info(
        "Match-basis sensitivity: %d of %d trades differ between trade-date and "
        "published-date matching", out["n_differing"], out["n_total"],
    )
    return out
