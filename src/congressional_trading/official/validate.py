"""Validation: invariants over the state DB and built datasets.

Failures return ok=False and the CLI exits nonzero; automation must treat
that as a failed run (no publishing).
"""

from __future__ import annotations

import json

import pandas as pd

from ..paths import REPO_ROOT
from .config import OfficialConfig
from .coverage import NONTERMINAL_STATES, TERMINAL_STATES
from .db import StateStore

#: Review reasons that do not block a release. Each waiver is a documented
#: coverage limitation whose rows cannot reach an analytical dataset, and each
#: needed evidence before it was granted (freeze §9; the audit permits a
#: waiver only with concrete repository evidence).
#:
#: * ``scanned_document_needs_ocr_or_manual_review`` / ``hand_filled_form_...``
#:   — filings whose state (``ocr_needs_manual_entry``) is excluded by the
#:   normalization filter in ``compat.py``.
#: * ``ocr_row_needs_verification`` — the 295 OCR-extracted rows sit on filings
#:   in state ``ocr_extracted``, which ``compat.py`` also excludes, so none of
#:   them reaches ``transactions_normalized``, ``transactions_verified`` or the
#:   project-compatible export. Verified 2026-07-26.
#:
#: ``unreconstructed_row`` is deliberately NOT waived: those rows sit on
#: ``fetched_partial`` filings whose successfully parsed rows DO ship, so the
#: dataset represents those filings with an incomplete row set.
WAIVED_REVIEW_REASONS = (
    "scanned_document_needs_ocr_or_manual_review",
    "hand_filled_form_needs_manual_entry",
    "ocr_row_needs_verification",
)


def _sql_list(values: tuple[str, ...]) -> str:
    return ",".join(repr(v) for v in values)


def run_validation(
    config: OfficialConfig, store: StateStore, run_id: str | None = None
) -> tuple[bool, dict]:
    results: dict = {"checks": []}
    ok = True

    # REL-003 / freeze §9: record the identity of the state the checks ran
    # against, so a release built from a *different* state can be refused.
    state_digest = store.content_digest()

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        results["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            ok = False

    # REL-001: nonterminal PTR filings must FAIL the run. Only explicitly
    # waived retention states are tolerated; everything else means the
    # collection is incomplete and must not be released.
    WAIVED = {"official_retention_unavailable"}
    nonterm = pd.read_sql_query(
        "SELECT state, COUNT(*) n FROM filings WHERE report_type='ptr' AND state IN "
        f"({','.join(repr(s) for s in NONTERMINAL_STATES)}) GROUP BY state",
        store.conn,
    )
    blocking = nonterm[~nonterm["state"].isin(WAIVED)]
    check("no_nonterminal_ptr_filings", blocking.empty,
          f"blocking nonterminal states: {blocking.to_dict(orient='records')}")

    # SEN-001 / freeze §9: a document is incomplete when its fetched page count
    # is below its expected page count. Read from the page ledger rather than by
    # pattern-matching state_detail text, and count filings in the partial state
    # even if no ledger row exists yet.
    incomplete_ledger = store.conn.execute(
        "SELECT COUNT(*) FROM (SELECT filing_id FROM filing_page_ledger "
        "GROUP BY filing_id HAVING SUM(status='ok') < COUNT(*))"
    ).fetchone()[0]
    incomplete_state = store.conn.execute(
        "SELECT COUNT(*) FROM filings WHERE state='fetch_partial_pending' "
        "OR (state='fetch_retry_pending' AND state_detail LIKE '%page images%')"
    ).fetchone()[0]
    check("no_incomplete_paper_documents",
          incomplete_ledger == 0 and incomplete_state == 0,
          f"{incomplete_ledger} filings short of their expected page count; "
          f"{incomplete_state} filings in an incomplete-page state")

    # unresolved failure/review thresholds
    unresolved_fetch = store.conn.execute(
        "SELECT COUNT(*) FROM fetch_manifest WHERE result_status='failed'").fetchone()[0]
    check("fetch_failures_within_threshold", unresolved_fetch == 0,
          f"{unresolved_fetch} unresolved fetch failures")
    high_review_rows = store.conn.execute(
        "SELECT reason, COUNT(*) FROM review_queue WHERE review_status='pending' "
        f"AND severity='high' AND reason NOT IN ({_sql_list(WAIVED_REVIEW_REASONS)}) "
        "GROUP BY reason ORDER BY 2 DESC"
    ).fetchall()
    high_review = sum(r[1] for r in high_review_rows)
    check("high_severity_review_within_threshold", high_review == 0,
          f"{high_review} unwaived high-severity review items: "
          + ", ".join(f"{r[0]}={r[1]}" for r in high_review_rows))

    # 1. every PTR filing is in a known state; report nonterminal count
    states = pd.read_sql_query(
        "SELECT state, COUNT(*) n FROM filings WHERE report_type='ptr' GROUP BY state",
        store.conn,
    )
    known = set(TERMINAL_STATES) | set(NONTERMINAL_STATES)
    unknown = states[~states["state"].isin(known)]
    check("all_states_known", unknown.empty, unknown.to_json())

    # 2. no duplicate raw transaction IDs (PK enforces, assert anyway)
    dupes = store.conn.execute(
        "SELECT COUNT(*) - COUNT(DISTINCT raw_transaction_id) FROM filing_transactions_raw"
    ).fetchone()[0]
    check("no_duplicate_raw_ids", dupes == 0, f"{dupes} duplicates")

    # 3. every raw transaction maps to an existing filing with a source doc
    orphans = store.conn.execute(
        "SELECT COUNT(*) FROM filing_transactions_raw t "
        "LEFT JOIN filings f USING (filing_id) WHERE f.filing_id IS NULL"
    ).fetchone()[0]
    check("no_orphan_transactions", orphans == 0, f"{orphans} orphans")

    # 4. parsed filings retain document provenance
    no_doc = store.conn.execute(
        "SELECT COUNT(*) FROM filings f WHERE f.state IN "
        "('fetched_and_parsed','fetched_partial','fetched_no_transactions') "
        "AND NOT EXISTS (SELECT 1 FROM filing_documents d WHERE d.filing_id=f.filing_id)"
    ).fetchone()[0]
    check("parsed_filings_have_documents", no_doc == 0, f"{no_doc} without documents")

    # 5. built datasets: bounds and traceability
    compat_path = config.normalized_dir / "transactions_project_compatible.csv"
    if compat_path.exists():
        compat = pd.read_csv(compat_path)
        bad = compat[
            compat["size_lower"].notna() & compat["size_upper"].notna()
            & (compat["size_lower"] > compat["size_upper"])
        ]
        check("amount_bounds_ordered", bad.empty, f"{len(bad)} rows lower>upper")
        check(
            "compat_rows_traceable",
            compat["official_raw_transaction_id"].notna().all()
            and compat["official_filing_id"].notna().all(),
            "all rows carry official filing + raw tx ids",
        )
        # schema contract against the committed schema JSON
        schema = json.loads(
            (REPO_ROOT / "configs" / "existing_project_schema.json").read_text()
        )
        missing = [
            c for c in schema["compatibility_columns"]
            if c not in compat.columns and c not in ("stable_record_key", "row_content_hash")
        ]
        check("compat_schema_columns_present", not missing, f"missing: {missing}")
        allowed_tx = set(
            schema["compatibility_columns"]["transaction_type_clean"]["allowed"]
        )
        bad_tx = set(compat["transaction_type_clean"].dropna()) - allowed_tx
        check("compat_txtype_categories", not bad_tx, f"unexpected: {bad_tx}")
    else:
        # REL-001: a missing required dataset is a failure, not a skip.
        check("compat_dataset_present", False,
              f"required dataset missing: {compat_path}")

    # 6. amended records do not erase originals: history keeps all versions
    norm_path = config.normalized_dir / "transactions_normalized.parquet"
    if norm_path.exists():
        norm = pd.read_parquet(norm_path)
        if "canonical_transaction_id" in norm.columns and len(norm):
            multi = norm.groupby("canonical_transaction_id")["version_number"].nunique()
            superseded_kept = norm[norm["is_current"] == 0]
            check(
                "amendments_preserve_originals",
                (multi > 1).sum() == 0 or len(superseded_kept) > 0,
                f"{int((multi > 1).sum())} multi-version groups, "
                f"{len(superseded_kept)} superseded rows retained",
            )

    results["ok"] = ok
    results["run_id"] = run_id
    results["state_digest"] = state_digest
    results["state_read_only"] = bool(getattr(store, "read_only", False))
    (config.reports_dir / "release_validation.json").write_text(
        json.dumps(results, indent=2)
    )
    return ok, results
