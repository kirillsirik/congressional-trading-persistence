"""Canonical source assembly: official parsed records + frozen Capitol Trades
records under the final source architecture (directive §3, §5, §6).

Produces the ``canonical_transactions`` frame consumed by the locked
pipeline, with source-native date fields preserved verbatim:

- every eligible parsed official transaction (all parser outputs, accepted
  under the author-review assumption);
- Capitol Trades rows uniquely linked to an official filing WITHOUT parsed
  transactions (coverage of paper/scan filings) carry the observed official
  date;
- CT rows linked to a filing WHOSE transactions are already parsed are
  vendor duplicates of official records and are excluded with a ledger
  reason;
- unlinked CT rows are vendor-only and later receive the CT-minus-one
  calendar-day public-date proxy.

Identity (§6): Bioguide ID when available; else the official identity
crosswalk household id; else a deterministic name+chamber key. Never surname
alone; never outcome-guided.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

from congressional_trading.official.hybrid import (
    identity_key,
    load_official_filing_identities,
    recover_official_filing_dates,
)

TERMINAL_STATES = (
    "fetched_and_parsed", "fetched_no_transactions", "ocr_extracted",
    "ocr_needs_manual_entry", "excluded_report_type", "excluded_filer_type",
    "duplicate", "superseded", "fetch_failed_terminal", "out_of_population",
)

_DIRECTION_MAP = {
    "purchase": "purchase",
    "sale": "sale",
    "sale_partial": "sale",
    "sale_full": "sale",
    "P": "purchase",
    "S": "sale",
    "S (partial)": "sale",
    "buy": "purchase",
    "sell": "sale",
}


def _fallback_uid(name: object, chamber: object) -> str:
    key = identity_key(name, chamber)
    if not key:
        return ""
    return "NAMEKEY-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def load_official_transactions(
    local_store: Path, identity_assignment: pd.DataFrame | None
) -> pd.DataFrame:
    normalized = pd.read_parquet(
        Path(local_store) / "normalized" / "transactions_current.parquet"
    )
    fuid_map = None
    if identity_assignment is not None:
        fuid_map = identity_assignment.drop_duplicates("filing_id").set_index(
            "filing_id"
        )["filer_uid"]

    bio = normalized["bioguide_id"].astype("string")
    filer_uid = bio.copy()
    if fuid_map is not None:
        filer_uid = filer_uid.fillna(
            normalized["filing_id"].map(fuid_map).astype("string")
        )
    fallback = pd.Series(
        [
            _fallback_uid(n, c)
            for n, c in zip(normalized["filer_name_raw"], normalized["chamber"], strict=True)
        ],
        index=normalized.index,
        dtype="string",
    )
    filer_uid = filer_uid.fillna(fallback)

    direction = normalized["transaction_type"].map(_DIRECTION_MAP).fillna("other")
    parent = normalized["group_root_filing_id"].astype("string")
    same = parent == normalized["filing_id"].astype("string")

    out = pd.DataFrame(
        {
            "transaction_uid": normalized["canonical_transaction_id"].astype(str),
            "filer_uid": filer_uid.astype(str),
            "source_native_household_id": normalized["member_id"]
            .astype("string")
            .fillna(bio)
            .fillna(fallback)
            .astype(str),
            "document_uid": normalized["filing_id"].astype(str),
            "record_source": "official",
            "direction": direction,
            "official_public_date": pd.to_datetime(
                normalized["filing_date_best"], errors="coerce"
            ),
            "capitol_trades_publication_date_raw": pd.NaT,
            "trade_date_official": pd.to_datetime(
                normalized["transaction_date"], errors="coerce"
            ),
            "trade_date_capitol_trades_raw": pd.NaT,
            "ticker_text": normalized["ticker"].where(
                normalized["ticker"].notna(), normalized["ticker_raw_field"]
            ),
            "security_name_text": normalized["asset_description_raw"],
            "owner": normalized["owner"],
            "amount_bracket": normalized["amount_category"],
            "is_amendment": normalized["is_amendment"].fillna(0).astype(int) == 1,
            "amendment_parent_uid": parent.where(~same, pd.NA),
            "chamber": normalized["chamber"],
            "report_year": normalized["report_year"],
            "document_format": normalized["extraction_method"].astype(str),
            "reporting_lag_days": normalized["days_transaction_to_filing"],
        }
    )
    return out


def load_ct_snapshot() -> pd.DataFrame:
    from congressional_trading.paths import get_paths
    from congressional_trading.scraping.snapshots import load_latest_canonical

    ct = load_latest_canonical(get_paths(profile="ct_only", read_only=True))
    if ct is None:
        raise FileNotFoundError("no finalized Capitol Trades snapshot found")
    return ct


def assemble_canonical_transactions(
    local_store: Path,
    identity_assignment: pd.DataFrame | None = None,
    ct: pd.DataFrame | None = None,
    compat_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (canonical_transactions, census, assembly_report)."""
    local_store = Path(local_store)
    report: dict = {}

    official = load_official_transactions(local_store, identity_assignment)
    report["official_rows"] = int(len(official))

    if ct is None:
        ct = load_ct_snapshot()
    report["ct_rows_raw"] = int(len(ct))

    parsed_filings = set(official["document_uid"].unique())

    state_db = local_store / "state.sqlite"
    filings = load_official_filing_identities(
        state_db,
        compat_csv
        or (local_store / "normalized" / "transactions_project_compatible.csv"),
    )
    ct = ct.copy()
    ct["published_date"] = pd.to_datetime(ct["published_date"], errors="coerce")
    recovery = recover_official_filing_dates(
        ct, filings, source_date_col="published_date"
    )
    ct = recovery.frame
    report["ct_link_stats"] = recovery.stats

    matched = ct["matched_filing_id"].notna()
    dup = matched & ct["matched_filing_id"].isin(parsed_filings)
    report["ct_vendor_duplicates_of_official"] = int(dup.sum())
    ct = ct[~dup]

    # repeat scrapes of one vendor record share a stable_record_key; keep the
    # first copy under a deterministic sort so transaction_uid stays unique
    n_before_dedup = len(ct)
    tiebreak = [
        c
        for c in ("scrape_timestamp_utc", "scrape_run_id", "row_content_hash")
        if c in ct.columns
    ]
    ct = (
        ct.sort_values(["stable_record_key", *tiebreak], kind="mergesort")
        .drop_duplicates("stable_record_key", keep="first")
        .sort_index()
    )
    report["ct_repeat_scrape_duplicates"] = int(n_before_dedup - len(ct))

    direction = ct["transaction_type_clean"].map(_DIRECTION_MAP).fillna("other")
    bio = ct["politician_slug"].astype("string")
    fallback = pd.Series(
        [
            _fallback_uid(n, c)
            for n, c in zip(ct["politician_name"], ct["chamber"], strict=True)
        ],
        index=ct.index,
        dtype="string",
    )
    filer_uid = bio.where(
        bio.str.match(r"^[A-Z]\d{6}$", na=False), fallback
    ).fillna(fallback)

    linked = ct["matched_filing_id"].notna()
    doc_uid = ct["matched_filing_id"].astype("string").where(
        linked,
        "ct-"
        + filer_uid.astype(str)
        + "-"
        + ct["published_date"].dt.strftime("%Y%m%d"),
    )
    ct_frame = pd.DataFrame(
        {
            "transaction_uid": "ct-" + ct["stable_record_key"].astype(str),
            "filer_uid": filer_uid.astype(str),
            "source_native_household_id": ct["politician_slug"].astype(str),
            "document_uid": doc_uid.astype(str),
            "record_source": "capitol_trades",
            "direction": direction,
            # observed official date ONLY when uniquely linked (§3.1 rule 1)
            "official_public_date": pd.to_datetime(
                ct["official_filing_date"], errors="coerce"
            ).where(linked, pd.NaT),
            "capitol_trades_publication_date_raw": ct["published_date"],
            "trade_date_official": pd.NaT,
            "trade_date_capitol_trades_raw": pd.to_datetime(
                ct["trade_date"], errors="coerce"
            ),
            "ticker_text": ct["ticker_clean"].where(
                ct["ticker_clean"].astype(str).str.len() > 0, ct["ticker_raw"]
            ),
            "security_name_text": ct["issuer_name"],
            "owner": ct["owner_clean"],
            "amount_bracket": ct["size_raw"],
            "is_amendment": False,
            "amendment_parent_uid": pd.NA,
            "chamber": ct["chamber"],
            "report_year": ct["published_date"].dt.year,
            "document_format": "vendor_feed",
            "reporting_lag_days": ct["disclosure_lag_days"],
        },
        index=ct.index,
    )
    report["ct_rows_kept"] = int(len(ct_frame))
    report["ct_rows_linked_to_unparsed_official"] = int(linked.sum())
    report["ct_rows_vendor_only"] = int((~linked).sum())

    txns = pd.concat([official, ct_frame], ignore_index=True)
    bad_uid = txns["filer_uid"].isin(["", "<NA>", "nan"])
    report["rows_dropped_no_identity"] = int(bad_uid.sum())
    txns = txns[~bad_uid]
    report["canonical_rows"] = int(len(txns))

    dup_uid = txns["transaction_uid"].duplicated(keep=False)
    if dup_uid.any():
        raise ValueError(
            "canonical_transactions transaction_uid is not unique: "
            f"{int(dup_uid.sum())} rows share "
            f"{txns.loc[dup_uid, 'transaction_uid'].nunique()} duplicated uids"
        )

    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    census = pd.read_sql_query(
        "SELECT filing_id AS document_uid, state FROM filings "
        "WHERE report_type='ptr'",
        con,
    )
    con.close()
    census["terminal"] = census["state"].isin(TERMINAL_STATES)
    report["census_rows"] = int(len(census))
    report["census_nonterminal"] = int((~census["terminal"]).sum())
    return txns, census, report
