"""Run the OCR stage over quarantined scanned/paper filings.

For each filing: render + OCR every page, store the OCR text and the
word/confidence/bbox table as durable artifacts in the content-addressed raw
store, classify the filing, extract rows only when machine-printed, and put
everything else in the review queue with what a reviewer needs.

Terminal states set here:

- ``ocr_extracted`` — machine-printed; rows emitted (below the verification
  threshold, so still review-gated)
- ``ocr_needs_manual_entry`` — hand-filled form; artifacts queued for review
- ``invalid_document`` — could not be rendered
"""

from __future__ import annotations

import json
import logging

from . import ids
from .config import OfficialConfig
from .db import StateStore
from .ocr import OCR_PARSER_NAME, OCR_PARSER_VERSION, ocr_document
from .storage import load_bytes, store_bytes

logger = logging.getLogger(__name__)


def ocr_filing(config: OfficialConfig, store: StateStore, filing_row) -> str:
    fid = filing_row["filing_id"]
    docs = store.documents_for(fid)
    targets = [d for d in docs if d["doc_role"] in ("primary_pdf", "page_image")]
    if not targets:
        store.set_filing_state(fid, "invalid_document", "no document to OCR")
        return "invalid_document"

    store.clear_filing_diagnostics(fid)
    all_rows: list[dict] = []
    classifications: list[str] = []
    evidence: list[str] = []
    doc_sha = targets[0]["sha256"]

    for doc in targets:
        try:
            content = load_bytes(config.raw_store, doc["sha256"], doc["stored_suffix"] or "")
        except FileNotFoundError:
            evidence.append(f"{doc['sha256'][:12]}: object missing from raw store")
            continue
        result = ocr_document(content)
        classifications.append(result.classification)
        evidence.append(f"{doc['doc_role']}: {result.evidence}")

        if result.pages:
            text_sha, _ = store_bytes(config.raw_store, result.text.encode("utf-8"), ".ocr.txt")
            words_sha, _ = store_bytes(
                config.raw_store, result.word_table().encode("utf-8"), ".ocr.tsv"
            )
            store.add_document(
                filing_id=fid, doc_role="ocr_text", url=doc["url"], sha256=text_sha,
                stored_suffix=".ocr.txt", content_type="text/plain",
                doc_format="ocr_text", format_confidence=1.0,
                format_evidence=f"{OCR_PARSER_NAME}/{OCR_PARSER_VERSION}: {result.evidence}",
                page_count=len(result.pages),
            )
            store.add_document(
                filing_id=fid, doc_role="ocr_words", url=doc["url"], sha256=words_sha,
                stored_suffix=".ocr.tsv", content_type="text/tab-separated-values",
                doc_format="ocr_words", format_confidence=1.0,
                format_evidence="word-level confidence and bounding boxes",
                page_count=len(result.pages),
            )
        all_rows.extend(result.rows)

    detail = "; ".join(evidence)[:500]
    if not classifications:
        store.set_filing_state(fid, "invalid_document", detail or "no OCR output")
        return "invalid_document"

    if all_rows:
        db_rows = []
        for i, row in enumerate(all_rows):
            page = row.pop("page", None)
            conf = row.pop("confidence", None)
            row.pop("extraction", None)
            raw_text = row.pop("raw_text", "")
            rid = ids.raw_transaction_id(fid, doc_sha, page, i, row)
            db_rows.append(dict(
                raw_transaction_id=rid, filing_id=fid, doc_sha256=doc_sha, page=page,
                row_index=i, parser_name=OCR_PARSER_NAME, parser_version=OCR_PARSER_VERSION,
                extraction_method="ocr", parse_confidence=conf,
                raw_json=json.dumps(row, sort_keys=True), source_text_evidence=raw_text,
            ))
            store.add_review(
                filing_id=fid, raw_transaction_id=rid, chamber=filing_row["chamber"],
                report_year=filing_row["report_year"], reason="ocr_row_needs_verification",
                source_url=filing_row["source_url"], page=page, raw_text=raw_text,
                proposed_json=json.dumps(row, sort_keys=True), confidence=conf,
                severity="high",
            )
        store.replace_raw_transactions(fid, db_rows)
        store.set_filing_state(fid, "ocr_extracted", f"{len(db_rows)} OCR rows; {detail}")
        return "ocr_extracted"

    store.replace_raw_transactions(fid, [])
    store.add_review(
        filing_id=fid, chamber=filing_row["chamber"], report_year=filing_row["report_year"],
        reason="hand_filled_form_needs_manual_entry", source_url=filing_row["source_url"],
        raw_text=detail, severity="high",
    )
    store.add_parse_failure(
        filing_id=fid, doc_sha256=doc_sha, failure_type="ocr_hand_filled",
        detail=detail, parser_name=OCR_PARSER_NAME, parser_version=OCR_PARSER_VERSION,
    )
    store.set_filing_state(fid, "ocr_needs_manual_entry", detail)
    return "ocr_needs_manual_entry"
