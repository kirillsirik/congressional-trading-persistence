"""Normalized tables and the project-compatible export.

Builds, from the raw store + state DB:

- transactions_normalized: full disclosed universe, raw + normalized fields,
  provenance and confidence;
- transactions_current: post-amendment-reconciliation current view;
- transactions_project_compatible: the exact existing-project canonical
  trade-record contract, built ONLY from verified
  official rows. Additive official_* columns carry provenance.

The compat export reuses the existing project's own key functions
(utils.identifiers) so stable_record_key / row_content_hash behave
identically downstream.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

from ..config import SIZE_BRACKETS, TOP_CODED_CAP
from ..data.clean_capitol_trades import (
    COMMON_ETF_TICKERS,
    CONTENT_HASH_FIELDS,
    ETF_FUND_KEYWORDS,
    add_power_committee,
    normalize_person_name,
)
from ..data.clean_capitol_trades import clean_ticker as existing_clean_ticker
from ..utils.identifiers import add_record_keys
from . import NORMALIZATION_VERSION, SCHEMA_VERSION
from .amendments import build_amendment_groups, period_key_from_transactions, reconcile_transactions
from .config import OfficialConfig
from .db import StateStore
from .members import MemberResolver
from .normalize import (
    amount_midpoints,
    clean_ticker,
    existing_bracket_label,
    extract_asset_type_code,
    extract_house_ticker,
    normalize_amount,
    normalize_date,
    normalize_owner,
    normalize_transaction_type,
    plausible_transaction_date,
)

logger = logging.getLogger(__name__)


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[3],
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_raw_transactions(store: StateStore) -> pd.DataFrame:
    rows = store.conn.execute(
        "SELECT t.*, f.chamber, f.filer_name_raw, f.first_name_raw, f.last_name_raw, "
        "f.state_district_raw, f.office_raw, f.filing_date, f.received_date, f.report_year, "
        "f.is_amendment, f.source_url, f.source_doc_id, f.state as filing_state "
        "FROM filing_transactions_raw t JOIN filings f USING (filing_id) "
        # only filings whose parse actually stands: never ship rows left over
        # from an earlier parse of a filing now quarantined or excluded
        "WHERE f.state IN ('fetched_and_parsed','fetched_partial','manually_verified')"
    ).fetchall()
    records = []
    for r in rows:
        rec = dict(r)
        rec.update(json.loads(rec.pop("raw_json")))
        records.append(rec)
    return pd.DataFrame(records)


def normalize_transactions(config: OfficialConfig, store: StateStore,
                           resolver: MemberResolver | None = None) -> pd.DataFrame:
    df = load_raw_transactions(store)
    if df.empty:
        return df
    # dates
    df["transaction_date"] = df.get("transaction_date_raw").map(normalize_date)
    df["notification_date"] = df.get("notification_date_raw", pd.Series(index=df.index)).map(
        normalize_date
    )
    df["filing_date_best"] = df["filing_date"].where(
        df["filing_date"].notna(), df["received_date"]
    )
    df["transaction_date_plausible"] = [
        plausible_transaction_date(td, fd)
        for td, fd in zip(df["transaction_date"], df["filing_date_best"], strict=False)
    ]
    # owner / type
    df["owner"] = [normalize_owner(o, ch) for o, ch in zip(df.get("owner_raw"), df["chamber"], strict=False)]
    tt = df.get("transaction_type_raw").map(normalize_transaction_type)
    df["transaction_type"] = tt.map(lambda t: t[0])
    df["is_partial_sale"] = tt.map(lambda t: t[1]).astype(int)
    # amounts
    amounts = df.get("amount_raw").map(normalize_amount).apply(pd.Series)
    df = pd.concat([df, amounts], axis=1)
    # tickers + asset type codes: house = extracted from asset text; senate native
    is_house = df["chamber"] == "house"
    df["ticker_raw_field"] = df.get("ticker_raw")
    df.loc[is_house, "ticker_raw_field"] = df.loc[is_house, "asset_description_raw"].map(
        extract_house_ticker
    )
    df["ticker"] = df["ticker_raw_field"].map(clean_ticker)
    df["asset_type_code_raw"] = df.get("asset_type_raw")
    df.loc[is_house, "asset_type_code_raw"] = df.loc[is_house, "asset_description_raw"].map(
        extract_asset_type_code
    )
    # timeliness
    tdate = pd.to_datetime(df["transaction_date"], errors="coerce")
    fdate = pd.to_datetime(df["filing_date_best"], errors="coerce")
    df["days_transaction_to_filing"] = (fdate - tdate).dt.days
    df["late_by_transaction_rule"] = (df["days_transaction_to_filing"] > 45).astype("Int64")
    # identity (time-aware, on transaction date, fallback filing date)
    if resolver is not None:
        on_dates = df["transaction_date"].where(df["transaction_date"].notna(),
                                                df["filing_date_best"])
        matches = [
            resolver.resolve(
                ch,
                fn,
                ln,
                od,
                state=(sd or "")[:2] if ch == "house" else None,
            )
            for ch, fn, ln, od, sd in zip(
                df["chamber"], df["first_name_raw"], df["last_name_raw"],
                on_dates, df["state_district_raw"].fillna(""), strict=False,
            )
        ]
        df = pd.concat([df, pd.DataFrame(matches, index=df.index)], axis=1)
    df["normalization_version"] = NORMALIZATION_VERSION
    df["schema_version"] = SCHEMA_VERSION
    df["code_commit"] = _code_commit()
    return df


def reconcile(df: pd.DataFrame, store: StateStore) -> pd.DataFrame:
    filings = pd.read_sql_query("SELECT * FROM filings", store.conn)
    period_keys = period_key_from_transactions(df)
    filings["period_key"] = filings["filing_id"].map(period_keys)
    grouped = build_amendment_groups(filings)
    return reconcile_transactions(df, grouped)


# The existing contract as the cleaning stage actually emits it
# (clean_capitol_trades.clean_capitol_trades: canonical_cols + extra_cols).
COMPAT_CANONICAL_COLUMNS = [
    "trade_id", "stable_record_key", "politician_name", "legislator_id",
    "party", "chamber", "state", "issuer_name", "ticker_raw", "ticker_clean",
    "trade_date", "published_date", "filing_date", "disclosure_lag_days",
    "transaction_type_raw", "transaction_type_clean", "buy_indicator",
    "owner_raw", "owner_clean", "size_raw", "size_lower", "size_upper",
    "size_midpoint_geometric", "size_midpoint_arithmetic",
    "size_top_coded_flag", "senate_indicator", "republican_indicator",
    "power_committee", "is_common_stock_candidate",
]
COMPAT_EXTRA_COLUMNS = [
    "power_committee_names", "price_raw", "parser_warning",
    "row_content_hash", "conflicting_duplicate_flag",
]
OFFICIAL_ADDITIVE_COLUMNS = [
    "official_filing_id", "official_document_url", "official_source_system",
    "official_raw_transaction_id", "official_canonical_transaction_id",
    "official_parse_confidence", "official_review_status", "official_is_amendment",
    "official_filer_name_raw", "official_asset_description_raw",
    "official_transaction_date_after_filing",
]
COMPAT_COLUMNS = COMPAT_CANONICAL_COLUMNS + COMPAT_EXTRA_COLUMNS + OFFICIAL_ADDITIVE_COLUMNS

# Asset text on House forms ends with the printed ticker and asset-type code
# ("Apple Inc. - Common Stock (AAPL) [ST]"); the issuer display name is the
# part before them. Full text is preserved in official_asset_description_raw.
_ASSET_TAIL = re.compile(r"\s*\((?:[A-Za-z0-9.\-]{1,10})\)\s*(?:\[[A-Za-z]{2,4}\])?\s*$")
_ASSET_CODE_ONLY = re.compile(r"\s*\[[A-Za-z]{2,4}\]\s*$")
_HONORIFIC = re.compile(
    r"^\s*(?:hon\.?|honorable|mr\.?|mrs\.?|ms\.?|dr\.?|rep\.?|sen\.?)\s+", re.I
)

_OWNER_COMPAT = {
    "self": "self", "spouse": "spouse", "joint": "joint", "dependent_child": "child",
    "trust": "undisclosed", "multiple": "undisclosed", "not_reported": "undisclosed",
    "unknown": "undisclosed",
}
_TXTYPE_COMPAT = {
    "purchase": "buy", "sale_full": "sell", "sale_partial": "sell",
    "exchange": "exchange", "other": "other", "unknown": "other",
}


def build_project_compatible(df_current: pd.DataFrame) -> pd.DataFrame:
    """Existing-project canonical trade record from reconciled current rows.

    Includes only rows that are: current (post-amendment), plausible-dated,
    from parsed official filings. Excluded rows remain in the normalized and
    audit outputs with their exclusion reasons.
    """
    df = df_current[df_current["is_current"] == 1].copy()
    df = df[df["transaction_date"].notna() & df["transaction_date_plausible"]]

    df = df.reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    # Canonical roster name where identity resolved (matches the existing
    # project's common-name convention, e.g. "Tommy Tuberville"); otherwise the
    # filing name with honorifics stripped. Raw filing name is kept additively.
    filed_name = df["filer_name_raw"].fillna("").str.replace(_HONORIFIC, "", regex=True).str.strip()
    roster_name = df.get("official_full_name")
    if roster_name is None:
        roster_name = pd.Series([None] * len(df), index=df.index)
    out["politician_name"] = roster_name.where(roster_name.notna() & (roster_name != ""), filed_name)
    out["legislator_id"] = df.get("bioguide_id")
    fallback = filed_name.map(normalize_person_name).str.replace(" ", "_")
    out["legislator_id"] = out["legislator_id"].where(
        out["legislator_id"].notna() & (out["legislator_id"] != ""), fallback
    )
    out["party"] = df.get("party")
    out["chamber"] = df["chamber"].map({"house": "House", "senate": "Senate"})
    out["state"] = df.get("state")
    out.loc[out["state"].isna() & (df["chamber"] == "house"), "state"] = (
        df["state_district_raw"].fillna("").str[:2].replace("", None)
    )
    asset = df["asset_description_raw"].fillna("")
    out["issuer_name"] = (
        asset.str.replace(_ASSET_TAIL, "", regex=True)
        .str.replace(_ASSET_CODE_ONLY, "", regex=True)
        .str.strip()
    )
    out.loc[out["issuer_name"] == "", "issuer_name"] = asset[out["issuer_name"] == ""]
    out["ticker_raw"] = df["ticker_raw_field"]
    # the existing project's own cleaner: NaN-safe, CRSP share-class style
    # (BRK.B), rejects non-ticker text such as "-- DIS" or "AET CVS"
    out["ticker_clean"] = df["ticker_raw_field"].map(existing_clean_ticker)
    out["trade_date"] = df["transaction_date"]
    out["published_date"] = df["filing_date_best"]  # documented semantic shift
    out["filing_date"] = df["filing_date_best"]
    out["disclosure_lag_days"] = df["days_transaction_to_filing"]
    out["transaction_type_raw"] = df["transaction_type_raw"]
    out["transaction_type_clean"] = df["transaction_type"].map(_TXTYPE_COMPAT)
    out["buy_indicator"] = (out["transaction_type_clean"] == "buy").astype(int)
    out["owner_raw"] = df.get("owner_raw")
    out["owner_clean"] = df["owner"].map(_OWNER_COMPAT)
    out["size_raw"] = df["amount_min"].map(
        lambda v: existing_bracket_label(int(v)) if pd.notna(v) else None
    )
    out.loc[out["size_raw"].isna(), "size_raw"] = df.loc[out["size_raw"].isna(), "amount_raw"]
    # For labels the existing project defines, use ITS bounds verbatim so the
    # two sources are numerically identical (this also gives the open-ended
    # "50M+" bracket its documented finite upper bound).
    known = out["size_raw"].map(lambda lab: SIZE_BRACKETS.get(lab) if pd.notna(lab) else None)
    out["size_lower"] = [
        b[0] if b else (lo if pd.notna(lo) else None) for b, lo in zip(known, df["amount_min"], strict=False)
    ]
    out["size_upper"] = [
        b[1] if b else (hi if pd.notna(hi) else None) for b, hi in zip(known, df["amount_max"], strict=False)
    ]
    mids = pd.DataFrame(
        [
            amount_midpoints(lo, hi, TOP_CODED_CAP)
            for lo, hi in zip(out["size_lower"], out["size_upper"], strict=False)
        ],
        index=out.index,
    )
    out = pd.concat([out, mids], axis=1)
    out["price_raw"] = pd.NA  # not reported on official forms
    out["price_numeric"] = pd.NA
    late_filed = (
        pd.to_datetime(out["trade_date"], errors="coerce")
        > pd.to_datetime(out["published_date"], errors="coerce")
    )
    out["parser_warning"] = [
        ";".join(w for w in [
            "low_confidence" if (conf if pd.notna(conf) else 1) < 0.9 else None,
            "amendment" if amend else None,
            "amount_unparsed" if cat in (None, "unknown") or pd.isna(cat) else None,
            "trade_date_after_filing_date" if late else None,
        ] if w) or None
        for conf, amend, cat, late in zip(
            df["parse_confidence"], df["is_amendment"], df["amount_category"], late_filed,
            strict=False,
        )
    ]
    out["official_transaction_date_after_filing"] = late_filed.astype(int)
    out["official_filer_name_raw"] = df["filer_name_raw"]
    out["official_asset_description_raw"] = df["asset_description_raw"]
    out["official_filing_id"] = df["filing_id"]
    out["official_document_url"] = df["source_url"]
    out["official_source_system"] = df["chamber"].map(
        {"house": "house_clerk", "senate": "senate_efd"}
    )
    out["official_raw_transaction_id"] = df["raw_transaction_id"]
    out["official_canonical_transaction_id"] = df["canonical_transaction_id"]
    out["official_parse_confidence"] = df["parse_confidence"]
    out["official_review_status"] = "unreviewed"
    out["official_is_amendment"] = df["is_amendment"]

    # ── derived columns the downstream stages require ───────────────────────
    # Computed with the existing project's own logic so the official dataset is
    # a drop-in for `trades_clean` (re-running clean_capitol_trades on it would
    # instead destroy 89% of transaction types, whose vocabularies are
    # Capitol-Trades-specific).
    out["senate_indicator"] = (out["chamber"] == "Senate").astype(int)
    out["republican_indicator"] = (out["party"] == "Republican").astype(int)

    keyword_hit = out["issuer_name"].str.contains(
        "|".join(ETF_FUND_KEYWORDS), case=False, na=False, regex=True
    )
    etf_ticker_hit = out["ticker_clean"].isin(COMMON_ETF_TICKERS)
    out["is_common_stock_candidate"] = (
        out["ticker_clean"].notna() & ~keyword_hit & ~etf_ticker_hit
    ).astype(int)

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["published_date"] = pd.to_datetime(out["published_date"], errors="coerce")
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce")
    out = add_power_committee(out, date_col="trade_date")

    # keys + duplicate flags, same recipe as the cleaning stage
    out = add_record_keys(out, content_fields=[c for c in CONTENT_HASH_FIELDS if c in out.columns])
    hash_variety = out.groupby("stable_record_key")["row_content_hash"].transform("nunique")
    out["conflicting_duplicate_flag"] = (hash_variety > 1).astype(int)
    out["disclosure_lag_days"] = pd.to_numeric(
        out["disclosure_lag_days"], errors="coerce"
    ).astype("Int64")

    out = out.sort_values(["published_date", "politician_name", "ticker_clean"],
                          na_position="last").reset_index(drop=True)
    out["trade_id"] = range(1, len(out) + 1)

    missing = [c for c in COMPAT_CANONICAL_COLUMNS + COMPAT_EXTRA_COLUMNS
               if c not in out.columns]
    if missing:
        raise AssertionError(f"compat export missing contract columns: {missing}")
    ordered = COMPAT_CANONICAL_COLUMNS + COMPAT_EXTRA_COLUMNS + [
        c for c in out.columns if c not in COMPAT_CANONICAL_COLUMNS + COMPAT_EXTRA_COLUMNS
    ]
    return out[ordered]


def size_bracket_sanity(df_compat: pd.DataFrame) -> list[str]:
    problems = []
    known = set(SIZE_BRACKETS)
    labels = df_compat["size_raw"].dropna().unique()
    for label in labels:
        if label not in known and not str(label).startswith("$"):
            problems.append(f"unknown size_raw label: {label!r}")
    bad_bounds = df_compat[
        df_compat["size_lower"].notna() & df_compat["size_upper"].notna()
        & (df_compat["size_lower"] > df_compat["size_upper"])
    ]
    if len(bad_bounds):
        problems.append(f"{len(bad_bounds)} rows with size_lower > size_upper")
    return problems
