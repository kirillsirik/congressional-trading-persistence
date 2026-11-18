"""Document-parsing cascade: route each fetched filing to the right parser.

Order: structured HTML → text-native PDF → scanned PDF (OCR when enabled,
else review queue). No image-only filing is ever marked transaction-free —
scanned documents without OCR results are quarantined into the review queue
with an explicit terminal state.
"""

from __future__ import annotations

import json
import logging

from . import ids
from .config import OfficialConfig
from .db import StateStore
from .house import parse as house_parse
from .parse_failures import ParseFailure
from .senate import parse as senate_parse
from .storage import load_bytes

logger = logging.getLogger(__name__)


def parse_filing(config: OfficialConfig, store: StateStore, filing_row) -> str:
    fid = filing_row["filing_id"]
    chamber = filing_row["chamber"]
    docs = store.documents_for(fid)
    if not docs:
        store.set_filing_state(fid, "fetch_retry_pending", "no documents fetched")
        return "fetch_retry_pending"
    # re-parsing replaces this filing's rows, so its prior diagnostics go too
    store.clear_filing_diagnostics(fid)

    landing = [d for d in docs if d["doc_role"] == "landing_html"]
    pdfs = [d for d in docs if d["doc_role"] == "primary_pdf"]
    images = [d for d in docs if d["doc_role"] == "page_image"]

    try:
        if chamber == "senate" and landing and not images:
            return _parse_senate_html(config, store, filing_row, landing[-1])
        if chamber == "house" and pdfs:
            doc = pdfs[-1]
            if doc["doc_format"] == "text_pdf" or doc["doc_format"] == "mixed_pdf":
                return _parse_house_pdf(config, store, filing_row, doc)
            return _route_scanned(config, store, filing_row, doc["sha256"])
        if images:
            return _route_scanned(config, store, filing_row, images[0]["sha256"])
        store.set_filing_state(fid, "fetched_unsupported",
                               f"no parser for roles {[d['doc_role'] for d in docs]}")
        return "fetched_unsupported"
    except Exception as exc:
        logger.exception("parse failed for %s", fid)
        store.add_parse_failure(filing_id=fid, failure_type="error",
                               detail=f"{type(exc).__name__}: {exc}")
        store.set_filing_state(fid, "quarantined", f"parser error: {exc}")
        return "quarantined"


def _store_rows(
    config: OfficialConfig, store: StateStore, filing_row, doc_sha: str,
    parser_name: str, parser_version: str, rows: list[dict],
    failures: list[ParseFailure],
    filer_meta: dict,
) -> str:
    fid = filing_row["filing_id"]
    db_rows = []
    n_review = 0
    for i, row in enumerate(rows):
        page = row.pop("page", None)
        confidence = row.pop("confidence", None)
        extraction = row.pop("extraction", "unknown")
        raw_text = row.pop("raw_text", "")
        rid = ids.raw_transaction_id(fid, doc_sha, page, i, row)
        db_rows.append(
            dict(
                raw_transaction_id=rid, filing_id=fid, doc_sha256=doc_sha, page=page,
                row_index=i, parser_name=parser_name, parser_version=parser_version,
                extraction_method=extraction, parse_confidence=confidence,
                raw_json=json.dumps(row, sort_keys=True), source_text_evidence=raw_text,
            )
        )
        if confidence is not None and confidence < config.confidence_review_threshold:
            store.add_review(
                filing_id=fid, raw_transaction_id=rid, chamber=filing_row["chamber"],
                report_year=filing_row["report_year"], reason="low_parse_confidence",
                source_url=filing_row["source_url"], page=page, raw_text=raw_text,
                proposed_json=json.dumps(row, sort_keys=True), confidence=confidence,
            )
            n_review += 1
    store.replace_raw_transactions(fid, db_rows)

    if filer_meta:
        updates = {}
        if filer_meta.get("is_amendment_doc") == "true" and not filing_row["is_amendment"]:
            updates["is_amendment"] = 1
        if updates:
            store.upsert_filing(fid, **updates)

    # Parse failures are typed: only a blocking kind means a transaction is
    # actually missing. A wrapped asset-description fragment keeps its text for
    # review at low severity but leaves the filing complete, so it neither
    # blocks a release nor marks the filing partially parsed.
    for failure in failures:
        store.add_parse_failure(filing_id=fid, doc_sha256=doc_sha,
                               failure_type=failure.kind, detail=str(failure),
                               parser_name=parser_name, parser_version=parser_version)
        store.add_review(
            filing_id=fid, chamber=filing_row["chamber"], report_year=filing_row["report_year"],
            reason=failure.kind, source_url=filing_row["source_url"],
            raw_text=str(failure), severity=failure.severity,
        )
    blocking = [f for f in failures if f.blocking]

    if blocking and rows:
        store.set_filing_state(fid, "fetched_partial",
                               f"{len(rows)} rows, {len(blocking)} unreconstructed")
        return "fetched_partial"
    if blocking and not rows:
        store.set_filing_state(fid, "quarantined", f"{len(blocking)} rows unparsed")
        return "quarantined"
    if not rows:
        store.set_filing_state(fid, "fetched_no_transactions", "parsed; no transaction rows")
        return "fetched_no_transactions"
    detail = f"{len(rows)} rows" + (f", {n_review} to review" if n_review else "")
    store.set_filing_state(fid, "fetched_and_parsed", detail)
    return "fetched_and_parsed"


def _parse_senate_html(config, store, filing_row, doc) -> str:
    html = load_bytes(config.raw_store, doc["sha256"], doc["stored_suffix"] or ".html").decode(
        "utf-8", errors="replace"
    )
    result = senate_parse.parse_senate_ptr_html(html)
    return _store_rows(
        config, store, filing_row, doc["sha256"], senate_parse.PARSER_NAME,
        senate_parse.PARSER_VERSION, result.rows, result.failures, result.filer_meta,
    )


def _parse_house_pdf(config, store, filing_row, doc) -> str:
    content = load_bytes(config.raw_store, doc["sha256"], doc["stored_suffix"] or ".pdf")
    result = house_parse.parse_house_ptr_pdf(content)
    return _store_rows(
        config, store, filing_row, doc["sha256"], house_parse.PARSER_NAME,
        house_parse.PARSER_VERSION, result.rows, result.failures, result.filer_meta,
    )


#: Extraction methods whose rows a text-parser re-parse must not discard: they
#: were produced by a different pipeline stage, not by the parser being re-run.
NON_PARSER_EXTRACTIONS = ("ocr", "manual_entry")


def _route_scanned(config: OfficialConfig, store: StateStore, filing_row, doc_sha: str) -> str:
    """Scanned/paper filing: OCR when enabled and available, else review."""
    fid = filing_row["filing_id"]
    # A re-parse that routes here must not leave rows from an earlier *parser*
    # run — but it must also not destroy rows the OCR stage produced. Re-running
    # `parse --reparse` with OCR disabled used to delete them silently (this cost
    # 295 OCR rows on 2026-07-26; recovered from a state snapshot).
    existing = store.conn.execute(
        "SELECT extraction_method, COUNT(*) FROM filing_transactions_raw "
        "WHERE filing_id=? GROUP BY 1", (fid,)
    ).fetchall()
    preserved = {m: n for m, n in existing if m in NON_PARSER_EXTRACTIONS}
    if preserved:
        detail = ", ".join(f"{n} {m}" for m, n in preserved.items())
        store.add_parse_failure(
            filing_id=fid, doc_sha256=doc_sha, failure_type="reparse_preserved_rows",
            detail=f"scanned document; kept {detail} rows from a non-parser stage",
        )
        store.set_filing_state(
            fid, filing_row["state"],
            f"re-parse routed to scanned; preserved {detail} rows",
        )
        return filing_row["state"]
    store.replace_raw_transactions(fid, [])
    if config.ocr_enabled:
        try:
            from .ocr import ocr_available

            if not ocr_available():
                raise RuntimeError("OCR enabled but no OCR engine installed")
            # OCR row extraction for historical form families is tracked in the
            # review queue until template parsers are validated per family.
            store.add_parse_failure(filing_id=fid, doc_sha256=doc_sha,
                                   failure_type="ocr_template_pending",
                                   detail="scanned document awaiting validated OCR template")
        except Exception as exc:
            store.add_parse_failure(filing_id=fid, doc_sha256=doc_sha,
                                   failure_type="ocr_unavailable", detail=str(exc))
    else:
        store.add_parse_failure(filing_id=fid, doc_sha256=doc_sha,
                               failure_type="ocr_disabled",
                               detail="scanned document; OCR disabled in config")
    store.add_review(
        filing_id=fid, chamber=filing_row["chamber"], report_year=filing_row["report_year"],
        reason="scanned_document_needs_ocr_or_manual_review",
        source_url=filing_row["source_url"], severity="high",
    )
    store.set_filing_state(fid, "quarantined", "scanned/paper document pending OCR or review")
    return "quarantined"
