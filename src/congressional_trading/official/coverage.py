"""Coverage accounting: every discovered filing must land in an accountable
terminal state, reported by chamber and year. Reports are written both to the
data root (full) and to the repo's reports/ directory (small summaries).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import OfficialConfig
from .db import StateStore

TERMINAL_STATES = {
    "fetched_and_parsed", "fetched_no_transactions", "fetched_partial",
    "fetched_unsupported", "official_not_found", "official_retention_unavailable",
    "invalid_document", "excluded_report_type", "excluded_filer_type",
    "quarantined", "manually_verified",
    "ocr_extracted", "ocr_needs_manual_entry",
}
# SEN-001: `fetch_partial_pending` is a paper filing whose page set is short of
# the expected count. Nonterminal by design, so it blocks release (freeze §9)
# instead of passing as a complete document.
NONTERMINAL_STATES = {
    "discovered", "fetched", "fetch_retry_pending", "fetch_partial_pending",
}


def coverage_by_year_chamber(store: StateStore) -> pd.DataFrame:
    filings = pd.read_sql_query(
        "SELECT chamber, report_year, report_type, state FROM filings", store.conn
    )
    if filings.empty:
        return pd.DataFrame()
    ptr = filings[filings["report_type"] == "ptr"]
    pivot = (
        ptr.groupby(["chamber", "report_year", "state"]).size().unstack(fill_value=0)
    )
    pivot["total_ptr_filings"] = pivot.sum(axis=1)
    tx_counts = pd.read_sql_query(
        "SELECT f.chamber, f.report_year, COUNT(*) AS transactions_extracted "
        "FROM filing_transactions_raw t JOIN filings f USING (filing_id) "
        "GROUP BY f.chamber, f.report_year",
        store.conn,
    ).set_index(["chamber", "report_year"])
    out = pivot.join(tx_counts, how="left").fillna({"transactions_extracted": 0})
    out["transactions_extracted"] = out["transactions_extracted"].astype(int)
    return out.reset_index()


def unaccounted_filings(store: StateStore) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT filing_id, chamber, report_year, state, state_detail FROM filings "
        f"WHERE report_type='ptr' AND state NOT IN ({','.join(repr(s) for s in TERMINAL_STATES)})",
        store.conn,
    )


def write_reports(config: OfficialConfig, store: StateStore, repo_reports_dir: Path) -> dict:
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    repo_reports_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    cov = coverage_by_year_chamber(store)
    cov.to_csv(config.reports_dir / "coverage_by_year_chamber.csv", index=False)
    cov.to_json(config.reports_dir / "coverage_by_year_chamber.json", orient="records", indent=2)
    cov.to_csv(repo_reports_dir / "coverage_by_year_chamber.csv", index=False)
    summary["coverage_rows"] = len(cov)

    fails = pd.read_sql_query(
        "SELECT url, error_category, error_detail, filing_id, fetched_at_utc "
        "FROM fetch_manifest WHERE result_status='failed'",
        store.conn,
    )
    fails.to_csv(config.reports_dir / "fetch_failures.csv", index=False)
    summary["fetch_failures"] = len(fails)

    pfails = pd.read_sql_query("SELECT * FROM parse_failures WHERE resolved=0", store.conn)
    pfails.to_csv(config.reports_dir / "parse_failures.csv", index=False)
    summary["parse_failures"] = len(pfails)

    zero = pd.read_sql_query(
        "SELECT filing_id, chamber, report_year, source_url, state_detail FROM filings "
        "WHERE state='fetched_no_transactions'",
        store.conn,
    )
    zero.to_csv(config.reports_dir / "zero_transaction_filings.csv", index=False)
    summary["zero_transaction_filings"] = len(zero)

    review = pd.read_sql_query(
        "SELECT * FROM review_queue WHERE review_status='pending'", store.conn
    )
    review.to_csv(config.reports_dir / "review_queue.csv", index=False)
    summary["review_pending"] = len(review)

    unavailable = pd.read_sql_query(
        "SELECT filing_id, chamber, report_year, source_url, state, state_detail FROM filings "
        "WHERE state IN ('official_not_found','official_retention_unavailable')",
        store.conn,
    )
    unavailable.to_csv(config.reports_dir / "unavailable_official_records.csv", index=False)
    summary["unavailable_official_records"] = len(unavailable)

    pending = unaccounted_filings(store)
    summary["nonterminal_ptr_filings"] = len(pending)
    pending.to_csv(config.reports_dir / "nonterminal_filings.csv", index=False)

    states = pd.read_sql_query(
        "SELECT chamber, state, COUNT(*) AS n FROM filings WHERE report_type='ptr' "
        "GROUP BY chamber, state ORDER BY chamber, n DESC",
        store.conn,
    )
    summary["state_counts"] = states.to_dict(orient="records")
    (config.reports_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2))
    (repo_reports_dir / "official_coverage_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary
