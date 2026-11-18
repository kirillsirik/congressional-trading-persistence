"""Why the two largest vendor-only households have no parsed official filing.

The households are the two ``vendor_without_official_2023plus`` households with
the most vendor lines. For each one
this module asks the official store (state.sqlite, normalized parquet, identity
crosswalk; read-only) and the vendor snapshot: does the vendor politician map to
an official filer identity; how many official PTR filings exist for that filer
and in what parse state; how many vendor records recovered an official filing
date under the pipeline's own linkage; and would the records link under a
surname+chamber identity. Identities live only in memory; every output is a count.

Run: ``PYTHONPATH=src python -m trading_on_disclosure_v4.data.vendor_only_status``.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from congressional_trading.official.hybrid import (
    FILING_MATCH_WINDOW_DAYS,
    identity_key,
    load_official_filing_identities,
    recover_official_filing_dates,
)
from trading_on_disclosure_v4.data import coverage as cv
from trading_on_disclosure_v4.data import load

REPO_ROOT = load.REPO_ROOT
DIAG_DIR = REPO_ROOT / "data" / "v4" / "diagnostics"
OUTPUT_NAME = "vendor_only_households_official_status.csv"
VENDOR_SNAPSHOT = "data/raw/capitol_trades/20260721_221640/canonical_trades.parquet"
CROSSWALK_DIR = "outputs_v2/top_journal_two_clock_design/identity"
BIOGUIDE = re.compile(r"^[A-Z]\d{6}$")
PARSED_STATES = ("fetched_and_parsed", "fetched_partial", "manually_verified")
IMAGE_ONLY_STATES = ("ocr_needs_manual_entry", "quarantined")
OCR_REVIEW_STATES = ("ocr_extracted",)
FAILED_STATES = (
    "fetch_retry_pending", "official_not_found", "official_retention_unavailable",
    "invalid_document", "fetched_unsupported",
)
CLASS_IMAGE_ONLY = "image-only filings"
CLASS_FAILED_LINK = "failed link"
CLASS_NO_FILING = "no filing found"
# columns that may never appear in the committed CSV
FORBIDDEN_COLUMNS = {"household_id", "filer_uid", "politician_slug", "bioguide_id", "filing_id", "name", "politician_name"}


def dropbox_project_root(repo_root: Path = REPO_ROOT) -> Path:
    """DROPBOX_PROJECT_ROOT from .env (the path holds spaces, so no shell sourcing)."""
    for line in (repo_root / ".env").read_text().splitlines():
        if line.startswith("DROPBOX_PROJECT_ROOT="):
            return Path(line.split("=", 1)[1].strip().strip('"'))
    raise KeyError("DROPBOX_PROJECT_ROOT not in .env")


def select_vendor_only_households(matched: pd.DataFrame, n: int = 2) -> list[str]:
    """Ids of the n largest (by vendor lines) households without parsed official filings 2023+."""
    classes = cv.classify_households(matched)
    vendor = matched[matched["record_source"] == "capitol_trades"]
    lines = vendor.groupby("filer_uid").size()
    without = classes[classes == cv.CLASS_WITHOUT].index
    return lines.reindex(without).fillna(0).sort_values(ascending=False).index[:n].tolist()


def load_ptr_filings(state_db: Path) -> pd.DataFrame:
    """Every discovered PTR filing with its terminal state, identity key and filing date."""
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    try:
        filings = pd.read_sql_query(
            "SELECT filing_id, chamber, filer_name_raw, report_year, state, "
            "COALESCE(filing_date, received_date) AS fdate FROM filings WHERE report_type='ptr'",
            con,
        )
        docs = pd.read_sql_query(
            "SELECT filing_id, doc_format FROM filing_documents WHERE doc_role='primary_pdf'", con
        )
        failures = pd.read_sql_query("SELECT filing_id, failure_type FROM parse_failures WHERE resolved=0", con)
        raw = pd.read_sql_query("SELECT filing_id, COUNT(*) AS n_raw FROM filing_transactions_raw GROUP BY filing_id", con)
    finally:
        con.close()
    filings["fdate"] = pd.to_datetime(filings["fdate"], errors="coerce")
    filings["chamber_norm"] = filings["chamber"].str.lower()
    filings["identity_key"] = [identity_key(n, c) for n, c in zip(filings["filer_name_raw"], filings["chamber"], strict=True)]
    filings["surname"] = filings["identity_key"].str.split("|").str[0].str.split().str[-1].fillna("")
    filings["scanned_pdf"] = filings["filing_id"].isin(docs.loc[docs["doc_format"] == "scanned_pdf", "filing_id"])
    filings["ocr_disabled_failure"] = filings["filing_id"].isin(failures.loc[failures["failure_type"] == "ocr_disabled", "filing_id"])
    filings["n_raw"] = filings["filing_id"].map(raw.set_index("filing_id")["n_raw"]).fillna(0).astype(int)
    return filings.drop(columns=["filer_name_raw"])


def state_counts(filings: pd.DataFrame) -> dict:
    """Filing counts by parse-state group (parsed / image-only / OCR review-gated / failed / other)."""
    st = filings["state"]
    parsed = st.isin(PARSED_STATES)
    image = st.isin(IMAGE_ONLY_STATES)
    ocr = st.isin(OCR_REVIEW_STATES)
    failed = st.isin(FAILED_STATES)
    return {
        "filings": int(len(filings)),
        "parsed": int(parsed.sum()),
        "image_only": int(image.sum()),
        "ocr_extracted_review_gated": int(ocr.sum()),
        "failed": int(failed.sum()),
        "other": int((~(parsed | image | ocr | failed)).sum()),
        "scanned_pdf_documents": int(filings["scanned_pdf"].sum()),
        "ocr_disabled_parse_failures": int(filings["ocr_disabled_failure"].sum()),
        "raw_transaction_rows": int(filings["n_raw"].sum()),
    }


def repaired_identity_link(vendor: pd.DataFrame, candidates: pd.DataFrame, window_days: int = FILING_MATCH_WINDOW_DAYS) -> dict:
    """Would the vendor records link to the candidate filings under the pipeline's window rule
    (filing date within ``window_days`` before the vendor publish date, one distinct date)?"""
    src = vendor[["stable_record_key", "published_date"]].drop_duplicates("stable_record_key").copy()
    src["published_date"] = pd.to_datetime(src["published_date"], errors="coerce")
    cand = candidates.loc[candidates["fdate"].notna(), ["filing_id", "fdate", "state"]]
    if src.empty or cand.empty:
        return {"linkable_parsed": 0, "linkable_unparsed": 0, "ambiguous_dates": 0, "no_filing_in_window": int(len(src))}
    j = src.merge(cand, how="cross")
    j = j[(j["fdate"] <= j["published_date"]) & (j["fdate"] >= j["published_date"] - pd.Timedelta(days=window_days))]
    g = j.groupby("stable_record_key").agg(n_dates=("fdate", "nunique"), parsed=("state", lambda s: bool(s.isin(PARSED_STATES).any())))
    unique = g[g["n_dates"] == 1]
    return {
        "linkable_parsed": int(unique["parsed"].sum()),
        "linkable_unparsed": int((~unique["parsed"]).sum()),
        "ambiguous_dates": int((g["n_dates"] > 1).sum()),
        "no_filing_in_window": int(len(src) - len(g)),
    }


def classify(status: dict) -> str:
    """Plain-word reason the household has no parsed official transactions."""
    if status["official_ptr_filings_total"] == 0:
        return CLASS_NO_FILING
    if status["official_ptr_filings_2023plus_parsed"] == 0 and status["official_ptr_filings_2023plus_image_only"] + status["official_ptr_filings_2023plus_ocr_extracted_review_gated"] == status["official_ptr_filings_2023plus"]:
        return CLASS_IMAGE_ONLY
    return CLASS_FAILED_LINK


def household_status(
    household: str,
    matched: pd.DataFrame,
    snapshot: pd.DataFrame,
    filings: pd.DataFrame,
    normalized_bioguide: pd.Series,
    filing_assignment: pd.DataFrame,
    crosswalk: pd.DataFrame,
    link_frame: pd.DataFrame,
) -> dict:
    """All counts for one household. ``household`` is used for masks only and never returned."""
    vendor_lines = matched[(matched["filer_uid"] == household) & (matched["record_source"] == "capitol_trades")]
    vendor = snapshot[snapshot["politician_slug"] == household]
    linked = link_frame[link_frame["politician_slug"] == household].drop_duplicates("stable_record_key")
    chamber = str(vendor["chamber"].mode().iat[0]).lower() if len(vendor) else ""
    vkeys = {identity_key(n, c) for n, c in zip(vendor["politician_name"], vendor["chamber"], strict=True)}
    vtokens = {t for k in vkeys for t in k.split("|")[0].split()}
    surnames = {k.split("|")[0].split()[-1] for k in vkeys if k}

    exact = filings[filings["identity_key"].isin(vkeys) | (filings["bioguide_id"] == household)]
    by_surname = filings[filings["surname"].isin(surnames) & (filings["chamber_norm"] == chamber)]
    cand = pd.concat([exact, by_surname]).drop_duplicates("filing_id")
    ckeys = set(cand["identity_key"])
    otokens = {t for k in ckeys for t in k.split("|")[0].split()}
    cand23 = cand[cand["fdate"] >= cv.VENDOR_ERA_START]
    units = filing_assignment.loc[filing_assignment["filing_id"].isin(cand["filing_id"]), "filer_uid"]
    unit_rows = crosswalk[crosswalk["filer_uid"].isin(units)]

    recovered = linked["official_filing_date"].notna()
    fid = linked["matched_filing_id"]
    parsed_ids = set(filings.loc[filings["state"].isin(PARSED_STATES), "filing_id"])
    all_counts = state_counts(cand)
    c23 = state_counts(cand23)
    repaired = repaired_identity_link(vendor, cand)
    pub_years = pd.to_datetime(vendor["published_date"], errors="coerce").dt.year
    status = {
        "vendor_lines_matched_frame": int(len(vendor_lines)),
        "vendor_records_snapshot": int(vendor["stable_record_key"].nunique()),
        "vendor_publish_year_min": int(pub_years.min()) if len(vendor) else 0,
        "vendor_publish_year_max": int(pub_years.max()) if len(vendor) else 0,
        "vendor_id_is_bioguide_shaped": bool(BIOGUIDE.match(household)),
        "official_normalized_rows_with_vendor_bioguide": int((normalized_bioguide == household).sum()),
        "official_filings_with_vendor_bioguide": int((filings["bioguide_id"] == household).sum()),
        "official_filings_exact_name_chamber_key": int(filings["identity_key"].isin(vkeys).sum()),
        "vendor_name_tokens": int(len(vtokens)),
        "official_name_tokens": int(len(otokens)),
        "shared_name_tokens": int(len(vtokens & otokens)),
        "crosswalk_units_surname_chamber": int(units.nunique()),
        "crosswalk_units_unambiguous": int((unit_rows["identity_status"] == "unambiguous").sum()),
        "crosswalk_units_pending_review": int((unit_rows["human_decision"] == "PENDING").sum()),
        "official_ptr_filings_total": all_counts["filings"],
        "official_ptr_filings_total_parsed": all_counts["parsed"],
        "official_ptr_filings_total_image_only": all_counts["image_only"],
        "official_ptr_filings_total_ocr_extracted_review_gated": all_counts["ocr_extracted_review_gated"],
        "official_ptr_filings_total_failed": all_counts["failed"],
        "official_ptr_filings_total_other": all_counts["other"],
        "official_ptr_filings_pre2023": int((cand["fdate"] < cv.VENDOR_ERA_START).sum()),
        "official_ptr_filings_undated": int(cand["fdate"].isna().sum()),
        "official_ptr_filings_2023plus": c23["filings"],
        "official_ptr_filings_2023plus_parsed": c23["parsed"],
        "official_ptr_filings_2023plus_image_only": c23["image_only"],
        "official_ptr_filings_2023plus_ocr_extracted_review_gated": c23["ocr_extracted_review_gated"],
        "official_ptr_filings_2023plus_failed": c23["failed"],
        "official_ptr_filings_2023plus_other": c23["other"],
        "official_ptr_filings_2023plus_scanned_pdf": c23["scanned_pdf_documents"],
        "official_ptr_filings_2023plus_ocr_disabled_parse_failure": c23["ocr_disabled_parse_failures"],
        "official_raw_transaction_rows_all_filings": all_counts["raw_transaction_rows"],
        "vendor_records_with_recovered_official_date": int(recovered.sum()),
        "vendor_records_linked_to_parsed_filing": int((recovered & fid.isin(parsed_ids)).sum()),
        "vendor_records_linked_to_unparsed_filing": int((recovered & ~fid.isin(parsed_ids)).sum()),
        "vendor_records_no_filing_found": int((~recovered).sum()),
        "vendor_lines_matched_frame_with_official_date": int(vendor_lines["official_public_date"].notna().sum()),
        "repaired_link_records_to_parsed_filing": repaired["linkable_parsed"],
        "repaired_link_records_to_unparsed_filing": repaired["linkable_unparsed"],
        "repaired_link_records_ambiguous_dates": repaired["ambiguous_dates"],
        "repaired_link_records_no_filing_in_window": repaired["no_filing_in_window"],
    }
    status["classification"] = classify(status)
    status["name_link_failed"] = bool(
        status["official_filings_exact_name_chamber_key"] == 0 and status["official_filings_with_vendor_bioguide"] == 0
    )
    return status


def build_status_table(n: int = 2, dropbox_root: Path | None = None) -> pd.DataFrame:
    root = dropbox_root or dropbox_project_root()
    official = root / "data" / "official"
    matched = load.load_matched_transactions()
    households = select_vendor_only_households(matched, n)
    snapshot = pd.read_parquet(
        root / VENDOR_SNAPSHOT,
        columns=["politician_slug", "politician_name", "chamber", "published_date", "stable_record_key"],
    )
    filings = load_ptr_filings(official / "state.sqlite")
    identities = load_official_filing_identities(
        official / "state.sqlite", official / "normalized" / "transactions_project_compatible.csv"
    )
    filings["bioguide_id"] = filings["filing_id"].map(identities.drop_duplicates("filing_id").set_index("filing_id")["bioguide_id"])
    normalized = pd.read_parquet(official / "normalized" / "transactions_current.parquet", columns=["bioguide_id"])
    crosswalk = pd.read_parquet(root / CROSSWALK_DIR / "identity_crosswalk.parquet", columns=["filer_uid", "identity_status", "human_decision"])
    assignment = pd.read_parquet(root / CROSSWALK_DIR / "identity_filing_assignment.parquet", columns=["filing_id", "filer_uid"])
    # the pipeline's own linkage, re-run on the households' vendor records only
    sub = snapshot[snapshot["politician_slug"].isin(households)].copy()
    sub["published_date"] = pd.to_datetime(sub["published_date"], errors="coerce")
    link_frame = recover_official_filing_dates(sub, identities, source_date_col="published_date").frame
    rows = []
    for i, household in enumerate(households, start=1):
        status = household_status(household, matched, snapshot, filings, normalized["bioguide_id"], assignment, crosswalk, link_frame)
        rows.append({"household": f"household_{i}", **status})
    table = pd.DataFrame(rows)
    assert_no_identifier_columns(table)
    return table


def assert_no_identifier_columns(table: pd.DataFrame) -> None:
    bad = set(map(str, table.columns)) & FORBIDDEN_COLUMNS
    if bad:
        raise ValueError(f"identifier columns in status table: {sorted(bad)}")
    text = table.to_csv(index=False)
    if re.search(r"\b[A-Z][0-9]{6}\b", text) or "FUID-" in text or "NAMEKEY-" in text:
        raise ValueError("identifier-like token in status table")


def main(argv: list[str] | None = None) -> int:
    table = build_status_table()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out = DIAG_DIR / OUTPUT_NAME
    table.to_csv(out, index=False)
    print(table.set_index("household").T.to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
